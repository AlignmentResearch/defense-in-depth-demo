import hypothesis.strategies as st
import pytest
import torch
from hypothesis import given

from robust_llm.config.model_configs import ModelConfig
from robust_llm.models.caching_wrapped_model import (
    CachingWrappedModel,
    get_common_prefix_for_batch,
)
from robust_llm.models.wrapped_model import WrappedModel

# torch.allclose is causing tests to flake. Default atol (absolute tolerance) is
# 1e-8, which is lower than we care about.
ATOL = 1e-5


@pytest.fixture()
def models() -> tuple[WrappedModel, CachingWrappedModel]:
    model_config = ModelConfig(
        name_or_path="gpt2",
        family="gpt2",
        inference_type="classification",
        max_minibatch_size=4,
        eval_minibatch_multiplier=1,
        env_minibatch_multiplier=0.5,
        dtype="float32",
    )
    wrapped_model = WrappedModel.from_config(model_config, accelerator=None)
    return wrapped_model, CachingWrappedModel(wrapped_model)


def test_unbatched_kv_caching(models: tuple[WrappedModel, CachingWrappedModel]):
    wrapped_model, caching_model = models

    # Set up cache.
    cache_sequence = "Hello, my dog is cute."
    cache_inp = caching_model.tokenize(cache_sequence, return_tensors="pt")
    cache_ids = cache_inp["input_ids"]
    assert isinstance(cache_ids, torch.Tensor)
    caching_model.add_to_cache(cache_ids)

    # Evaluate cached and uncached on a sequence that is a superset of the cache.
    super_sequence = cache_sequence + " I like to play with him."
    super_inp = caching_model.tokenize(super_sequence, return_tensors="pt")

    uncached_super_logits = wrapped_model(**super_inp).logits
    cached_super_logits = caching_model(**super_inp).logits

    assert torch.allclose(uncached_super_logits, cached_super_logits, atol=ATOL)

    # Evaluate cached and uncached on a sequence that diverges from the cache.
    diverging_sequence = "Hello, my dog is fun. I like to play with him."
    diverging_inp = caching_model.tokenize(diverging_sequence, return_tensors="pt")

    uncached_diverging_logits = wrapped_model(**diverging_inp).logits
    cached_diverging_logits = caching_model(**diverging_inp).logits

    assert torch.allclose(uncached_diverging_logits, cached_diverging_logits, atol=ATOL)


def test_batched_kv_caching(models: tuple[WrappedModel, CachingWrappedModel]):
    wrapped_model, caching_model = models

    # Set up cache.
    cache_sequence = "Hello, my dog is cute."
    cache_inp = caching_model.tokenize(cache_sequence, return_tensors="pt")
    cache_ids = cache_inp["input_ids"]
    assert isinstance(cache_ids, torch.Tensor)
    caching_model.add_to_cache(cache_ids)

    # Evaluate cached and uncached on two sequences that diverge before the end
    # of the cache and have different lengths.
    super_sequence = cache_sequence + " I like to play with him in the park on Sundays."
    diverging_sequence = "Hello, my dog is fun. I like to play with him."
    batched_sequences = [super_sequence, diverging_sequence]
    batched_inp = caching_model.tokenize(
        # We use right-padding for non-autoregressive outputs.
        batched_sequences,
        return_tensors="pt",
        padding_side="right",
    )

    uncached_batched_logits = wrapped_model(**batched_inp).logits
    cached_batched_logits = caching_model(**batched_inp).logits
    assert torch.allclose(uncached_batched_logits, cached_batched_logits, atol=ATOL)


def test_wrapping_attributes(models: tuple[WrappedModel, CachingWrappedModel]):
    wrapped_model, caching_model = models

    # Testing access passes through to underlying model
    assert caching_model.device == wrapped_model.device

    # Testing assignment passes through to underlying model
    caching_model.new_attribute = "new_attribute"
    assert hasattr(wrapped_model, "new_attribute")
    assert wrapped_model.new_attribute == "new_attribute"  # type: ignore[attr-defined]


@given(st.lists(st.integers(min_value=0, max_value=100000), max_size=100))
def test_get_common_prefix_for_batch(common_prefix: list[int]):
    diverging_endings = torch.arange(100).reshape(10, 10)

    # Make a batch of sequences that are just the prefix.
    common_prefix_tensor = torch.tensor(common_prefix)
    prefix_only = torch.stack([common_prefix_tensor for _ in range(10)])
    out = get_common_prefix_for_batch(prefix_only)
    assert torch.allclose(out, common_prefix_tensor.unsqueeze(0), atol=ATOL)

    with_diverging_endings = torch.cat([prefix_only, diverging_endings], dim=1)
    out = get_common_prefix_for_batch(with_diverging_endings)
    assert torch.allclose(out, common_prefix_tensor, atol=ATOL)
    assert not out.shape[0] == with_diverging_endings.shape[1]
