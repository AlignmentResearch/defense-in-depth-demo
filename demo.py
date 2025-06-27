#!/usr/bin/env python3
"""
Interactive Defense-in-Depth Demo
Configurable pipeline with input/output filters and base model
"""
import os
from accelerate import Accelerator
from pathlib import Path
from robust_llm.config.layered_defense_configs import PromptedFilterTemplateConfig
from robust_llm.config.model_configs import GenerationConfig, ModelConfig
from robust_llm.file_utils import compute_repo_path
from robust_llm.layered_defense.components import (
    InputPipelineData,
    OutputPipelineData,
)
from robust_llm.layered_defense.prompted_filters.prompted_filters import (
    PromptedInputFilter,
    PromptedOutputFilter,
)
from robust_llm.message_utils import Message, MessageList
from robust_llm.models.wrapped_model import WrappedModel
import yaml

# Configuration - easily changeable
BASE_MODEL_CONFIG = {
    "name_or_path": "Qwen/Qwen3-14B",
    "family": "qwen3-chat",
    "max_new_tokens": 200,
    "gpu_memory_utilization": 0.5,
}

FILTER_MODELS = {
    "input_filter": "google/gemma-2-9b-it",
    "output_filter": "google/gemma-2-9b-it",
}
FILTER_THRESHOLDS = {
    "input_threshold": 0.04742,
    "output_threshold": 0.010987,
}

FILTER_TEMPLATES = {
    "input_filter_template": "prompt_templates/few_shot_input_filter.yaml",
    "output_filter_template": "prompt_templates/few_shot_output_filter.yaml",
}
DTYPE = "bfloat16"
ATTN_IMPL = "flash_attention_2"


def get_family(model_name: str) -> str:
    if "google/gemma" in model_name:
        return "gemma-chat"
    elif "Qwen3" in model_name:
        return "qwen3-chat"
    else:
        raise ValueError(f"Unsupported model: {model_name}")


def initialize_models():
    """Initialize base model and filter models"""
    print("Initializing models...")

    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    accelerator = Accelerator()

    # Base model
    model_config = ModelConfig(
        name_or_path=BASE_MODEL_CONFIG["name_or_path"],
        tokenizer_name=BASE_MODEL_CONFIG["name_or_path"],
        family=BASE_MODEL_CONFIG["family"],
        dtype=DTYPE,
        inference_type="generation",
        attention_implementation=ATTN_IMPL,
        env_minibatch_multiplier=1,
        gpu_memory_utilization=BASE_MODEL_CONFIG["gpu_memory_utilization"],
        enforce_eager=True,
        generation_config=GenerationConfig(
            max_new_tokens=BASE_MODEL_CONFIG["max_new_tokens"],
        ),
    )
    base_model = WrappedModel.from_config(model_config, accelerator)

    # Filter model config
    input_filter_config = ModelConfig(
        name_or_path=FILTER_MODELS["input_filter"],
        family=get_family(FILTER_MODELS["input_filter"]),
        dtype=DTYPE,
        inference_type="generation",
        attention_implementation=ATTN_IMPL,
        env_minibatch_multiplier=1,
    )
    output_filter_config = ModelConfig(
        name_or_path=FILTER_MODELS["output_filter"],
        family=get_family(FILTER_MODELS["output_filter"]),
        dtype=DTYPE,
        inference_type="generation",
        attention_implementation=ATTN_IMPL,
    )

    # Load filter templates
    ROOT = Path(compute_repo_path())

    with open(ROOT / FILTER_TEMPLATES["input_filter_template"], "r") as file:
        input_filter_prompt = yaml.safe_load(file)

    with open(ROOT / FILTER_TEMPLATES["output_filter_template"], "r") as file:
        output_filter_prompt = yaml.safe_load(file)

    # Input filter
    input_filter = PromptedInputFilter(
        victim=None,
        accelerator=accelerator,
        filter_config=input_filter_config,
        template_config=PromptedFilterTemplateConfig(
            template=input_filter_prompt["template"],
        ),
        threshold=FILTER_THRESHOLDS["input_threshold"],
    )

    # Output filter
    output_filter = PromptedOutputFilter(
        victim=None,
        accelerator=accelerator,
        filter_config=output_filter_config,
        template_config=PromptedFilterTemplateConfig(
            template=output_filter_prompt["template"],
        ),
        threshold=FILTER_THRESHOLDS["output_threshold"],
    )

    print("Models initialized successfully!")
    return base_model, input_filter, output_filter


def process_query(query, base_model, input_filter, output_filter):
    """Process a single query through the defense pipeline"""
    print("\n" + "=" * 80)
    print(f"Processing query: {query[:100]}{'...' if len(query) > 100 else ''}")
    print("=" * 80)

    # Create message list
    message_list = MessageList([Message(role="user", content=query)])

    # Step 1: Input filter
    print("\n1. Input Filter Check:")
    input_data = InputPipelineData(
        message_lists=[message_list],
        attention_masks=None,
        latents=None,
    )
    input_result = input_filter(input_data, None)
    input_flagged = input_result[1][0].flag
    input_score = input_result[1][0].score

    print(f"   Score: {input_score:.6f}")
    print(f"   Threshold: {FILTER_THRESHOLDS['input_threshold']}")
    print(f"   Status: {'🚨 FLAGGED' if input_flagged else '✅ PASSED'}")

    if input_flagged:
        print(
            "   Query flagged by input filter - continuing pipeline for demonstration"
        )

    # Step 2: Base model generation
    print("\n2. Base Model Generation:")
    print("   Generating response...")
    outputs = base_model.autoregressive_generation_from_messages(
        message_lists=[message_list],
    )

    generation = None
    for out in outputs:
        for gen in out:
            generation = gen[-1].content
            break
        break

    if not generation:
        print("   No generation produced")
        return

    print(f"   Generated: {generation}")

    # Step 3: Output filter
    print("\n3. Output Filter Check:")
    response_message_list = MessageList(
        [
            Message(role="user", content=query),
            Message(role="assistant", content=generation),
        ]
    )

    output_data = OutputPipelineData(
        message_lists=[response_message_list],
        attention_masks=None,
        latents=None,
    )
    output_result = output_filter(output_data, None)
    output_flagged = output_result[1][0].flag
    output_score = output_result[1][0].score

    print(f"   Score: {output_score:.6f}")
    print(f"   Threshold: {FILTER_THRESHOLDS['output_threshold']}")
    print(f"   Status: {'🚨 FLAGGED' if output_flagged else '✅ PASSED'}")


def main():
    """Main interactive loop"""
    print("Defense-in-Depth Interactive Demo")
    print("=================================")
    print(f"Base Model: {BASE_MODEL_CONFIG['name_or_path']}")
    print(f"Input Filter Model: {FILTER_MODELS['input_filter']}")
    print(f"Output Filter Model: {FILTER_MODELS['output_filter']}")
    print(f"Input Threshold: {FILTER_THRESHOLDS['input_threshold']}")
    print(f"Output Threshold: {FILTER_THRESHOLDS['output_threshold']}")
    print()

    # Initialize models
    base_model, input_filter, output_filter = initialize_models()

    print("\nReady! Enter queries to test the defense pipeline.")
    print("Type 'quit', 'exit', or Ctrl+C to stop.")
    print("-" * 50)

    try:
        while True:
            query = input("\nEnter your query: ").strip()

            if not query:
                continue

            if query.lower() in ["quit", "exit", "q"]:
                break

            try:
                process_query(query, base_model, input_filter, output_filter)
            except Exception as e:
                print(f"Error processing query: {e}")

    except KeyboardInterrupt:
        pass

    print("\nGoodbye!")


if __name__ == "__main__":
    main()
