import pytest
import torch

from robust_llm.config.model_configs import ModelConfig
from robust_llm.models.model_utils import build_dataloader
from robust_llm.models.wrapped_model import WrappedModel, deduped_flop_count_context

FORWARD_FLOP_COUNT = 28135424


@pytest.fixture
def model_config():
    return ModelConfig(
        name_or_path="EleutherAI/pythia-14m",
        family="pythia",
        revision="main",
        inference_type="generation",
        max_minibatch_size=4,
        eval_minibatch_multiplier=1,
        env_minibatch_multiplier=0.5,
    )


@pytest.fixture
def wrapped_model(model_config):
    return WrappedModel.from_config(
        model_config,
        accelerator=None,
    )


@pytest.fixture
def unique_wrapped_model(model_config):
    return WrappedModel.from_config(
        model_config,
        accelerator=None,
        # Avoid trying to run backward passes through the same
        # copy of the model in different tests.
        overwrite_cache=True,
    )


def test_flop_tracked_model_initialization(wrapped_model: WrappedModel):
    assert wrapped_model._flop_count == 0


def test_flop_tracked_model_compute_flops(wrapped_model: WrappedModel):
    input_dict = {"input_ids": torch.randint(0, 1024, (1, 1))}
    flops = wrapped_model.compute_flops(input_dict)
    assert flops == FORWARD_FLOP_COUNT


def test_flop_tracked_model_update_flop_count(wrapped_model: WrappedModel):
    input_dict = {"input_ids": torch.randint(0, 1024, (1, 1))}
    wrapped_model.update_flop_count(input_dict)
    assert wrapped_model._flop_count == FORWARD_FLOP_COUNT


def test_flop_count_context(wrapped_model: WrappedModel):
    input_dict = {"input_ids": torch.randint(0, 1024, (1, 1))}

    with wrapped_model.flop_count_context() as context:
        wrapped_model(**input_dict)

    assert context.flops > 0
    assert context.start_flops == 0
    assert context.end_flops == FORWARD_FLOP_COUNT
    assert context.flops == FORWARD_FLOP_COUNT


def test_nested_flop_count_contexts(wrapped_model: WrappedModel):
    input_dict = {"input_ids": torch.randint(0, 1024, (1, 1))}

    with wrapped_model.flop_count_context() as outer_context:
        wrapped_model(**input_dict)

        with wrapped_model.flop_count_context() as inner_context:
            wrapped_model(**input_dict)

    assert outer_context.flops > inner_context.flops
    assert inner_context.flops > 0


def test_backward_flop_count_increment(unique_wrapped_model: WrappedModel):
    unique_wrapped_model.train()
    input_dict = {"input_ids": torch.randint(0, 1024, (1, 1))}
    out = unique_wrapped_model(**input_dict)
    forward_flop_count = unique_wrapped_model._flop_count
    assert forward_flop_count == FORWARD_FLOP_COUNT
    loss = out.logits.sum()
    loss.backward()
    assert unique_wrapped_model._flop_count > forward_flop_count
    assert unique_wrapped_model._flop_count == 84406272


def test_call_count_increment(unique_wrapped_model: WrappedModel):
    unique_wrapped_model.train()
    input_dict = {"input_ids": torch.randint(0, 1024, (1, 1))}
    assert unique_wrapped_model._n_forward_calls == 0
    out = unique_wrapped_model(**input_dict)
    assert unique_wrapped_model._n_forward_calls == 1
    loss = out.logits.sum()
    assert unique_wrapped_model._n_backward_calls == 0
    loss.backward()
    assert unique_wrapped_model._n_backward_calls == 1


def test_input_shapes(wrapped_model: WrappedModel):
    wrapped_model.train()
    input_dict = {"input_ids": torch.randint(0, 1024, (1, 1))}
    assert wrapped_model._n_forward_calls == 0
    wrapped_model(**input_dict)
    assert wrapped_model._input_shapes == [(1, 1)]


@pytest.mark.parametrize("minibatch_size", [1, 2, 3, 4])
def test_flops_with_dataloader(minibatch_size: int, model_config: ModelConfig):
    wrapped_model = WrappedModel.from_config(
        model_config,
        accelerator=None,
        # Avoid trying to run backward passes through the same
        # copy of the model in different tests.
        overwrite_cache=True,
    )
    wrapped_model.train()
    text = ["Hello, how are you?", "What's the craic?"]
    tokenized = wrapped_model.tokenize(
        text,
        return_tensors="pt",
        padding_side="left",
    )
    dataloader = build_dataloader(minibatch_size=minibatch_size, **tokenized)
    for batch in dataloader:
        out = wrapped_model(**batch)
        loss = out.logits.sum()
        loss.backward()
    assert wrapped_model._flop_count == 506437632 * 2


def test_deduped_flop_count_context(
    wrapped_model: WrappedModel, model_config: ModelConfig
):
    # Test that deduped_flop_count_context counts the correct number of FLOPs
    # with two models (both [wrapped_model, wrapped_model] and [wrapped_model,
    # some_other_model]).
    same_model = wrapped_model
    different_wrap_and_hf = WrappedModel.from_config(
        model_config, accelerator=None, overwrite_cache=True
    )
    different_wrap_same_hf = WrappedModel.from_config(
        model_config, accelerator=None, overwrite_cache=False
    )

    input_dict = {"input_ids": torch.randint(0, 1024, (1, 1))}
    for other_model in [same_model, different_wrap_and_hf, different_wrap_same_hf]:
        with deduped_flop_count_context(wrapped_model, other_model) as context:
            wrapped_model(**input_dict)
            other_model(**input_dict)
        assert context.flops == 2 * FORWARD_FLOP_COUNT
