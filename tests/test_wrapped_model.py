import contextlib
import dataclasses
from pathlib import Path

import pytest
import torch
from accelerate import Accelerator
from transformers.generation.utils import GenerationMixin
from transformers.models.llama.modeling_llama import LlamaForCausalLM
from transformers.models.llama.tokenization_llama_fast import LlamaTokenizerFast

from robust_llm.config.model_configs import GenerationConfig, ModelConfig
from robust_llm.dist_utils import dist_rmtree
from robust_llm.models import WrappedModel
from robust_llm.models.mock_models import patch_load_hf_model


def model_config_factory():
    return ModelConfig(
        # We use a finetuned model so that the classification head isn't
        # randomly initalized.
        name_or_path="AlignmentResearch/robust_llm_oskar-041f_clf_pm_pythia-14m_s-0",  # noqa: E501
        family="pythia",
        inference_type="classification",
        strict_load=True,
        revision="0d356c12e4f425b6221e1f951f302f91d2c4d563",
        dtype="float32",
        max_minibatch_size=4,
        eval_minibatch_multiplier=1,
        env_minibatch_multiplier=0.5,
    )


@pytest.fixture
def wrapped_model():
    config = model_config_factory()
    return WrappedModel.from_config(config, accelerator=None)


def test_strict_load():
    model_config = model_config_factory()
    strict_config = dataclasses.replace(
        model_config, name_or_path="EleutherAI/pythia-14m", revision="main"
    )

    # This should raise an error because the base model has no classification
    # head so it'll be randomly initialized.
    with pytest.raises(AssertionError):
        WrappedModel.from_config(strict_config, accelerator=None)


def test_device(wrapped_model: WrappedModel):
    original_device = wrapped_model.device

    wrapped_model.to(torch.device("cpu"))
    assert wrapped_model.device == torch.device("cpu")

    wrapped_model.to(original_device)
    assert wrapped_model.device == original_device


def test_eval_train(wrapped_model: WrappedModel):
    wrapped_model.train()
    assert wrapped_model.model.training

    wrapped_model.eval()
    assert not wrapped_model.model.training


def test_get_embeddings(wrapped_model: WrappedModel):
    input_ids = torch.tensor([[1, 2, 3, 4, 5]])
    wrapped_embeddings = wrapped_model.get_embeddings(input_ids)
    underlying_embeddings = wrapped_model.model.get_input_embeddings()(input_ids)
    assert torch.allclose(wrapped_embeddings, underlying_embeddings)


def test_forward(wrapped_model: WrappedModel):
    input_ids = torch.tensor([[1, 2, 3, 4, 5]])
    wrapped_output = wrapped_model(input_ids=input_ids)
    underlying_output = wrapped_model.model(input_ids=input_ids)
    assert torch.allclose(wrapped_output.logits, underlying_output.logits)


def test_tokenize(wrapped_model: WrappedModel):
    text = ["Hello, my dog is cute", "Hello, my dog is the cutest dog."]
    wrapped_input_ids = wrapped_model.tokenize(
        text, padding_side="right", return_tensors="pt"
    ).input_ids
    underlying_input_ids = wrapped_model.right_tokenizer(
        text, padding=True, return_tensors="pt"
    ).input_ids
    assert torch.allclose(wrapped_input_ids, underlying_input_ids)


def test_llama():
    cfg = ModelConfig(
        name_or_path="HuggingFaceM4/tiny-random-LlamaForCausalLM",
        family="llama2",
        revision="main",
        inference_type="generation",
        max_minibatch_size=4,
        eval_minibatch_multiplier=1,
        env_minibatch_multiplier=0.5,
    )
    wrapped_model = WrappedModel.from_config(cfg, accelerator=None)
    assert isinstance(wrapped_model.model, LlamaForCausalLM)
    assert isinstance(wrapped_model.right_tokenizer, LlamaTokenizerFast)


def test_determinism_single_batch():
    cfg = ModelConfig(
        name_or_path="EleutherAI/pythia-14m",
        family="pythia",
        revision="main",
        inference_type="generation",
        max_minibatch_size=4,
        eval_minibatch_multiplier=1,
        env_minibatch_multiplier=0.5,
        seed=42,
        generation_config=GenerationConfig(
            max_new_tokens=50, do_sample=True, temperature=10.0
        ),
    )
    assert cfg.generation_config is not None
    model = WrappedModel.from_config(cfg, accelerator=None)
    encoding = model.tokenize("Hello, my dog is cute", return_tensors="pt")
    inputs = {
        "input_ids": encoding.input_ids,
        "attention_mask": encoding.attention_mask,
        "generation_config": model.transformers_generation_config,
    }
    is_equal = model.generate(**inputs) == model.generate(**inputs)
    assert isinstance(is_equal, torch.Tensor)
    assert is_equal.all()

    underlying_model = model.model
    assert isinstance(underlying_model, GenerationMixin)
    not_equal = underlying_model.generate(**inputs) != underlying_model.generate(
        **inputs
    )
    assert isinstance(not_equal, torch.Tensor)
    assert not_equal.any()


def test_determinism_batched():
    cfg = ModelConfig(
        name_or_path="EleutherAI/pythia-14m",
        family="pythia",
        revision="main",
        inference_type="generation",
        max_minibatch_size=4,
        eval_minibatch_multiplier=1,
        env_minibatch_multiplier=0.5,
        seed=42,
        generation_config=GenerationConfig(
            max_new_tokens=50, do_sample=True, temperature=10.0, deterministic=True
        ),
    )
    assert cfg.generation_config is not None
    model = WrappedModel.from_config(cfg, accelerator=None)
    text = ["Hello, my dog is cute", "Greetings from the moon"]
    encoding1 = model.tokenize(text, return_tensors="pt")
    encoding2 = model.tokenize(text[::-1], return_tensors="pt")
    inputs1 = {
        "input_ids": encoding1.input_ids,
        "attention_mask": encoding1.attention_mask,
        "generation_config": model.transformers_generation_config,
    }
    inputs2 = {
        "input_ids": encoding2.input_ids,
        "attention_mask": encoding2.attention_mask,
        "generation_config": model.transformers_generation_config,
    }

    is_equal = model.generate(**inputs1) == model.generate(**inputs2)[[1, 0]]
    assert isinstance(is_equal, torch.Tensor)
    assert is_equal.all()

    model._set_seed()
    underlying_model = model.model
    assert isinstance(underlying_model, GenerationMixin)
    simple_out1 = underlying_model.generate(**inputs1)
    model._set_seed()
    simple_out2 = underlying_model.generate(**inputs2)
    not_equal = simple_out1 != simple_out2[[1, 0]]
    assert isinstance(not_equal, torch.Tensor)
    assert not_equal.any()


def test_generate_equivalence():
    cfg = ModelConfig(
        name_or_path="EleutherAI/pythia-14m",
        family="pythia",
        revision="main",
        inference_type="generation",
        max_minibatch_size=4,
        eval_minibatch_multiplier=1,
        env_minibatch_multiplier=0.5,
        seed=42,
        generation_config=GenerationConfig(
            max_new_tokens=50, do_sample=True, temperature=10.0
        ),
    )
    assert cfg.generation_config is not None
    model = WrappedModel.from_config(cfg, accelerator=None)
    text = "Hello, my dog is cute"
    encoding = model.tokenize(text, return_tensors="pt")
    inputs = {
        "input_ids": encoding.input_ids,
        "attention_mask": encoding.attention_mask,
        "generation_config": model.transformers_generation_config,
    }
    model._set_seed()
    underlying_model = model.model
    assert isinstance(underlying_model, GenerationMixin)
    simple_out = underlying_model.generate(**inputs)
    wrapped_out = model.generate(**inputs)
    is_equal = simple_out == wrapped_out
    assert isinstance(is_equal, torch.Tensor)
    assert is_equal.all()

    mb_text = ["Hello, my dog is cute", "Greetings from the moon"]
    mb_encoding = model.tokenize(mb_text, return_tensors="pt")
    mb_inputs = {
        "input_ids": mb_encoding.input_ids,
        "attention_mask": mb_encoding.attention_mask,
        "generation_config": model.transformers_generation_config,
    }
    assert model.generate(**mb_inputs).shape == model.generate(**mb_inputs).shape


@pytest.mark.multigpu
def test_save_load_model():
    cfg = ModelConfig(
        name_or_path="EleutherAI/pythia-14m",
        family="pythia",
        revision="main",
        inference_type="generation",
        max_minibatch_size=4,
        eval_minibatch_multiplier=1,
        env_minibatch_multiplier=0.5,
        seed=42,
        generation_config=GenerationConfig(
            max_new_tokens=50, do_sample=True, temperature=10.0
        ),
    )
    accelerator = Accelerator()
    model = WrappedModel.from_config(cfg, accelerator=accelerator)
    assert isinstance(model, WrappedModel)
    path = Path("artifacts/test_model")
    dist_rmtree(path)
    model.save_local(path, retries=3, cooldown_seconds=0.1, max_shard_size="7MB")

    cfg.name_or_path = str(path)
    loaded_model = WrappedModel.from_config(cfg, accelerator=accelerator)
    dist_rmtree(path)

    example_prompt = "Hello, my dog is cute"
    encoding = loaded_model.tokenize(example_prompt, return_tensors="pt")
    assert torch.allclose(
        model.forward(**encoding).logits, loaded_model.forward(**encoding).logits
    )


MODEL_CONFIGS_FOR_PADDING_TEST = [
    ModelConfig(
        # Use a standard generation Pythia model instead of classification
        name_or_path="EleutherAI/pythia-14m",
        tokenizer_name="EleutherAI/pythia-14m",
        family="pythia",
        inference_type="generation",
        revision="main",
        dtype="float32",
        max_minibatch_size=4,
        eval_minibatch_multiplier=1,
        env_minibatch_multiplier=0.5,
    ),
    ModelConfig(
        name_or_path="HuggingFaceM4/tiny-random-LlamaForCausalLM",
        tokenizer_name="HuggingFaceM4/tiny-random-LlamaForCausalLM",
        family="llama2",
        revision="main",
        inference_type="generation",
        max_minibatch_size=4,
        eval_minibatch_multiplier=1,
        env_minibatch_multiplier=0.5,
        dtype="float32",
    ),
    ModelConfig(
        name_or_path="HuggingFaceM4/tiny-random-LlamaForCausalLM",
        tokenizer_name="HuggingFaceM4/tiny-random-LlamaForCausalLM",
        family="llama3-chat",
        revision="main",
        inference_type="generation",
        max_minibatch_size=4,
        eval_minibatch_multiplier=1,
        env_minibatch_multiplier=0.5,
        dtype="float32",
    ),
]

# Define mock configs directly here or keep MOCK_MODEL_CONFIGS_FOR_PADDING_TEST
MOCK_MODEL_CONFIGS = [
    ModelConfig(
        name_or_path="google/gemma-1.1-2b-it",
        tokenizer_name="google/gemma-1.1-2b-it",
        family="gemma",
        inference_type="generation",
        revision="main",
        dtype="float32",
        max_minibatch_size=4,
        eval_minibatch_multiplier=1,
        env_minibatch_multiplier=0.5,
    ),
    ModelConfig(
        name_or_path="google/gemma-2-9b-it",
        tokenizer_name="google/gemma-2-9b-it",
        family="gemma",
        revision="main",
        inference_type="generation",
        max_minibatch_size=4,
        eval_minibatch_multiplier=1,
        env_minibatch_multiplier=0.5,
        dtype="float32",
    ),
    ModelConfig(
        name_or_path="Qwen/Qwen2.5-7B-Instruct",
        tokenizer_name="Qwen/Qwen2.5-7B-Instruct",
        family="qwen2.5",
        revision="main",
        inference_type="generation",
        max_minibatch_size=4,
        eval_minibatch_multiplier=1,
        env_minibatch_multiplier=0.5,
        dtype="float32",
    ),
]


# Create a list of (config, use_mock) tuples
PADDING_TEST_CFGS = [(config, False) for config in MODEL_CONFIGS_FOR_PADDING_TEST] + [
    (config, True) for config in MOCK_MODEL_CONFIGS
]


@pytest.mark.parametrize("model_config, use_mock_model", PADDING_TEST_CFGS)
def test_forward_with_padding(model_config: ModelConfig, use_mock_model: bool):
    """Tests forward pass consistency with and without padding."""
    context = patch_load_hf_model() if use_mock_model else contextlib.nullcontext()
    with context:
        wrapped_model = WrappedModel.from_config(model_config, accelerator=None)

    # Use a generic text input suitable for different tokenizers
    text = "Testing forward pass with padding."
    # Tokenize without padding first
    encoding = wrapped_model.tokenize(text, padding=False, return_tensors="pt")
    input_ids = encoding.input_ids.to(wrapped_model.device)
    attention_mask = encoding.attention_mask.to(wrapped_model.device)

    # 1. Compute output without padding
    out = wrapped_model(input_ids=input_ids, attention_mask=attention_mask)
    logits = out.logits

    # 2. Create padded inputs
    pad_len = 6
    pad_token_id = wrapped_model.right_tokenizer.pad_token_id
    assert isinstance(pad_token_id, int)

    padding_tensor = torch.full(
        (input_ids.shape[0], pad_len),
        pad_token_id,
        dtype=input_ids.dtype,
        device=wrapped_model.device,
    )
    input_ids_with_padding = torch.cat([input_ids, padding_tensor], dim=1)

    padding_mask = torch.zeros(
        (attention_mask.shape[0], pad_len),
        dtype=attention_mask.dtype,
        device=wrapped_model.device,
    )
    attention_mask_with_padding = torch.cat([attention_mask, padding_mask], dim=1)

    # 3. Compute output with padding
    out_with_padding = wrapped_model(
        input_ids=input_ids_with_padding, attention_mask=attention_mask_with_padding
    )
    # Slice the logits to match the original sequence length
    logits_with_padding = out_with_padding.logits[:, : input_ids.shape[1], :]

    # 4. Compare logits
    atol = 1e-6
    rtol = 1e-2
    abs_diff = torch.abs(logits - logits_with_padding)
    rel_diff = abs_diff / (torch.abs(logits) + 1e-8)
    assert torch.allclose(logits, logits_with_padding, atol=atol, rtol=rtol), (
        "Logits are not close to each other: "
        f"max abs diff: {torch.max(abs_diff)}\n"
        f"max rel diff: {torch.max(rel_diff)}\n"
        f"max abs diff within rtol: {torch.max(abs_diff[rel_diff < rtol])}\n"
        f"max abs diff outside rtol: {torch.max(abs_diff[rel_diff >= rtol])}\n"
        f"max rel diff within atol: {torch.max(rel_diff[abs_diff < atol])}\n"
        f"max rel diff outside atol: {torch.max(rel_diff[abs_diff >= atol])}\n"
        f"% outside of rtol: {torch.sum(rel_diff >= rtol) / rel_diff.numel()}\n"
        f"% outside of atol: {torch.sum(abs_diff >= atol) / abs_diff.numel()}"
    )
