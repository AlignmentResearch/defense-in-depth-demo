import contextlib
from functools import cache
from typing import Optional
from unittest.mock import patch

import torch
from accelerate import Accelerator
from transformers.modeling_utils import PreTrainedModel
from transformers.models.auto.configuration_auto import AutoConfig
from transformers.models.auto.modeling_auto import AutoModelForCausalLM

from robust_llm.models.model_utils import InferenceType, prepare_model_with_accelerate


@cache
def gemma_1p1_config():
    config = AutoConfig.from_pretrained("google/gemma-1.1-2b-it")
    config.intermediate_size = 10
    config.hidden_size = 128
    config.num_hidden_layers = 2
    config.torch_dtype = torch.float32
    return config


@cache
def gemma_2_config():
    config = AutoConfig.from_pretrained("google/gemma-2-9b-it")
    config.intermediate_size = 10
    config.hidden_size = 128
    config.num_hidden_layers = 2
    config.torch_dtype = torch.float32
    return config


@cache
def qwen_2p5_config():
    config = AutoConfig.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
    config.intermediate_size = 10
    config.hidden_size = 128
    config.num_hidden_layers = 2
    config.torch_dtype = torch.float32
    return config


@cache
def model_configs() -> dict[str, AutoConfig]:
    # We make this a cached function rather than a global variable because if it
    # were global, we'd call the gemma config functions and make slow network
    # requests during pytest collection even if no tests in this file run.
    return {
        "google/gemma-1.1-2b-it": gemma_1p1_config(),
        "google/gemma-2-9b-it": gemma_2_config(),
        "Qwen/Qwen2.5-7B-Instruct": qwen_2p5_config(),
    }


def mock_load_hf_model(
    accelerator: Accelerator | None,
    name_or_path: str,
    revision: str,
    inference_type: InferenceType,
    strict_load: bool,
    torch_dtype: torch.dtype,
    attention_implementation: Optional[str] = None,
    num_classes: Optional[int] = None,
    overwrite_cache: bool = False,
) -> PreTrainedModel:
    """Mock load_hf_model function.

    This function is used to test the WrappedModel class without actually
    loading a model from Hugging Face. Instead we just use a randomly
    initialized small model by getting a config.
    """
    config = model_configs()[name_or_path]
    print("USING MOCK LOAD HF MODEL")
    model = AutoModelForCausalLM.from_config(config)
    if accelerator is not None:
        model = prepare_model_with_accelerate(accelerator, model)
    return model


@contextlib.contextmanager
def patch_load_hf_model():
    """Context manager to patch load_hf_model with mock_load_hf_model."""
    with patch("robust_llm.models.wrapped_model.load_hf_model", mock_load_hf_model):
        yield
