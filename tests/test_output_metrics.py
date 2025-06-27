from functools import partial
from unittest.mock import MagicMock, patch

import pytest
import torch

from robust_llm.models.metric_utils import loss_on_goal_from_logits, success_on_goal
from robust_llm.models.model_output_metrics import (
    ClassificationMetric,
    ConversationList,
    GenerationMetric,
    Metric,
    MetricFunctionFactory,
)
from robust_llm.models.wrapped_model import Message, MessageList, WrappedModel


def test_metric_function_factory_get():
    loss_logit_fn = MetricFunctionFactory.get(Metric.LOSS, "sum")
    success_fn = MetricFunctionFactory.get(
        Metric.SUCCESS,
        "mean",
    )  # Reduction ignored for success
    success_fn_logits = MetricFunctionFactory.get(
        Metric.SUCCESS,
        "sum",
    )  # Reduction ignored for success

    # Check if the correct partial functions are returned
    # We can't directly compare partial objects easily, so we check the func attribute
    assert isinstance(loss_logit_fn, partial)
    assert loss_logit_fn.func == loss_on_goal_from_logits
    assert loss_logit_fn.keywords == {"reduction": "sum"}

    assert success_fn == success_on_goal
    assert success_fn_logits == success_on_goal


def test_metric_function_factory_get_invalid():
    with pytest.raises(ValueError):
        # Let's try an invalid metric type first.
        MetricFunctionFactory.get("invalid_metric", "mean")  # type: ignore # noqa: E501


def test_classification_metric_should_not_instantiate():
    with pytest.raises(TypeError):
        ClassificationMetric()


def test_classification_metric_from_logits_loss(sample_logits, sample_goal_clf):
    losses = ClassificationMetric.from_logits(
        sample_logits, sample_goal_clf, metric=Metric.LOSS
    )
    # Expected: -log(softmax(logits)[correct_class])
    # Example 1:
    # Logits [1.0, -1.0], Goal 0. Softmax = [0.8808, 0.1192].
    # Loss = -log(0.8808) approx 0.1269
    # Example 2:
    # Logits [-0.5, 0.5], Goal 1. Softmax = [0.2689, 0.7311].
    # Loss = -log(0.7311) approx 0.3133
    expected_losses = torch.tensor([0.1269, 0.3133])
    assert torch.allclose(losses, expected_losses, atol=1e-4)


# probs: [[.88,.12],[.27,.73]], goals: [0,1]
@pytest.mark.parametrize(
    "clf_threshold, expected_successes",
    [
        (0.1, [False, True]),  # Pred:[1,1]. Succ:[F,T]
        (0.5, [True, True]),  # Pred:[0,1]. Succ:[T,T]
        (0.8, [True, False]),  # Pred:[0,0]. Succ:[T,F]
        (0.9, [True, False]),  # Pred:[0,0]. Succ:[T,F]
    ],
)
def test_classification_metric_from_logits_success(
    sample_logits,
    sample_goal_clf,
    clf_threshold,
    expected_successes,
):
    """Test classification success metric with different thresholds."""
    successes = ClassificationMetric.from_logits(
        sample_logits,
        sample_goal_clf,
        metric=Metric.SUCCESS,
        clf_threshold=clf_threshold,
    )
    assert successes == expected_successes


def test_classification_metric_from_logits_invalid_metric(
    sample_logits, sample_goal_clf
):
    with pytest.raises(ValueError, match="Metric invalid_metric not supported."):
        ClassificationMetric.from_logits(
            sample_logits, sample_goal_clf, metric="invalid_metric"  # type: ignore # noqa: E501
        )


@patch("robust_llm.models.model_output_metrics.ClassificationLogits")
def test_classification_metric_from_tokens(
    mock_classification_logits, mock_model, sample_goal_clf
):
    mock_logits_instance = MagicMock()
    # Simulate batches returned by the generator
    batch1_logits = torch.tensor([[1.0, -1.0]])
    batch2_logits = torch.tensor([[-0.5, 0.5]])
    mock_logits_instance.__iter__.return_value = iter([batch1_logits, batch2_logits])
    mock_classification_logits.from_tokens.return_value = mock_logits_instance

    input_ids = torch.tensor([[1, 2], [3, 4]])
    attention_mask = torch.tensor([[1, 1], [1, 1]])

    successes, all_logits_list = ClassificationMetric.from_tokens(
        mock_model,
        input_ids=input_ids,
        attention_mask=attention_mask,
        clf_label_data=sample_goal_clf,
        metric=Metric.SUCCESS,
        clf_threshold=0.5,
    )

    # Check if ClassificationLogits.from_tokens was called correctly
    mock_classification_logits.from_tokens.assert_called_once_with(
        mock_model, input_ids=input_ids, attention_mask=attention_mask
    )

    # Check results
    # (should be the same as test_classification_metric_from_logits_success)
    assert successes == [True, True]
    expected_logits = torch.cat([batch1_logits, batch2_logits]).tolist()
    assert all_logits_list == expected_logits

    # Test unsupported metric
    with pytest.raises(ValueError, match="Metric Metric.LOSS is not supported"):
        ClassificationMetric.from_tokens(
            mock_model,
            input_ids=input_ids,
            attention_mask=attention_mask,
            clf_label_data=sample_goal_clf,
            metric=Metric.LOSS,
        )


@patch("robust_llm.models.model_output_metrics.ClassificationLogits")
@pytest.mark.parametrize(
    "clf_threshold, expected_successes",
    [
        (0.1, [False, True]),  # preds: [1, 1], goals: [0, 1]
        (0.5, [True, True]),  # preds: [0, 1], goals: [0, 1]
        (0.8, [True, False]),  # preds: [0, 0], goals: [0, 1]
    ],
)
def test_classification_metric_from_tokens_clf_threshold(
    mock_classification_logits,
    mock_model,
    sample_goal_clf,
    clf_threshold,
    expected_successes,
):
    """Test the clf_threshold argument in from_tokens with various thresholds."""
    mock_logits_instance = MagicMock()
    # Logits:
    # [[1.0, -1.0]] -> Probs [0.8808, 0.1192], Goal 0. Correct prob: 0.8808
    # [[-0.5, 0.5]] -> Probs [0.2689, 0.7311], Goal 1. Correct prob: 0.7311
    batch1_logits = torch.tensor([[1.0, -1.0]])
    batch2_logits = torch.tensor([[-0.5, 0.5]])
    # Reset mock for each parameterization
    mock_logits_instance.__iter__.return_value = iter([batch1_logits, batch2_logits])
    mock_classification_logits.from_tokens.return_value = mock_logits_instance

    input_ids = torch.tensor([[1, 2], [3, 4]])
    attention_mask = torch.tensor([[1, 1], [1, 1]])

    successes, _ = ClassificationMetric.from_tokens(
        mock_model,
        input_ids=input_ids,
        attention_mask=attention_mask,
        clf_label_data=sample_goal_clf,
        metric=Metric.SUCCESS,
        clf_threshold=clf_threshold,
    )

    # Check if ClassificationLogits.from_tokens was called correctly
    mock_classification_logits.from_tokens.assert_called_once_with(
        mock_model, input_ids=input_ids, attention_mask=attention_mask
    )
    # Check the success results against the expected list for the given threshold
    assert successes == expected_successes


@patch("robust_llm.models.model_output_metrics.ClassificationMetric.from_tokens")
def test_classification_metric_from_messages(
    mock_from_tokens, mock_model, sample_input_data, sample_goal_clf
):
    # Mock the return value of the nested call
    mock_from_tokens.return_value = ([True, True], [[1.0, -1.0], [-0.5, 0.5]])

    successes, all_logits_list = ClassificationMetric.from_messages(
        mock_model,
        input_data=sample_input_data,
        clf_label_data=sample_goal_clf,
        metric=Metric.SUCCESS,
        clf_threshold=0.5,
    )

    # Check if message_lists_to_tokens was called
    mock_model.message_lists_to_tokens.assert_called_once_with(
        sample_input_data.message_lists,
        message_filter=sample_input_data.message_filter,
        add_generation_prompt=sample_input_data.add_generation_prompt,
        padding_side="right",
    )

    # Check if from_tokens was called correctly
    mock_from_tokens.assert_called_once()
    call_args = mock_from_tokens.call_args[0]
    call_kwargs = mock_from_tokens.call_args[1]
    assert call_args[0] == mock_model
    assert torch.equal(
        call_args[1], mock_model.message_lists_to_tokens.return_value.input_ids
    )
    assert torch.equal(
        call_args[2], mock_model.message_lists_to_tokens.return_value.attention_mask
    )
    assert call_args[3] == sample_goal_clf
    assert call_kwargs["metric"] == Metric.SUCCESS
    assert call_kwargs["clf_threshold"] == 0.5

    # Check results
    assert successes == [True, True]
    assert all_logits_list == [[1.0, -1.0], [-0.5, 0.5]]

    # Test unsupported metric
    with pytest.raises(ValueError, match="Metric Metric.LOSS is not supported"):
        ClassificationMetric.from_messages(
            mock_model,
            input_data=sample_input_data,
            clf_label_data=sample_goal_clf,
            metric=Metric.LOSS,
        )


def test_generation_metric_should_not_instantiate():
    with pytest.raises(TypeError):
        GenerationMetric()


@patch("robust_llm.models.model_output_metrics.get_target_slice_in_logits")
@patch("robust_llm.models.model_output_metrics.loss_on_goal_from_logits")
def test_generation_metric_from_logits_loss1(
    mock_loss_fn,
    mock_get_slice,
    sample_logits_gen,
    sample_input_ids_gen,
    sample_goal_gen,
):
    # Mock the helper functions
    mock_get_slice.side_effect = [slice(3, 5), slice(4, 5)]  # Example slices
    mock_loss_fn.side_effect = [torch.tensor(0.5), torch.tensor(0.8)]  # Example losses

    losses = GenerationMetric.from_logits(
        sample_logits_gen,
        sample_input_ids_gen,
        sample_goal_gen,
        metric=Metric.LOSS,
        reduction="mean",
    )

    assert mock_get_slice.call_count == 2
    assert mock_loss_fn.call_count == 2

    expected_losses = torch.tensor([0.5, 0.8])
    assert torch.equal(losses, expected_losses)


@patch("robust_llm.models.model_output_metrics.get_target_slice_in_logits")
@patch("robust_llm.models.model_output_metrics.success_on_goal")
def test_generation_metric_from_logits_success1(
    mock_success_fn,
    mock_get_slice,
    sample_logits_gen,
    sample_input_ids_gen,
    sample_goal_gen,
):
    mock_get_slice.side_effect = [slice(3, 5), slice(4, 5)]
    mock_success_fn.side_effect = [True, False]  # Example successes

    successes = GenerationMetric.from_logits(
        sample_logits_gen,
        sample_input_ids_gen,
        sample_goal_gen,
        metric=Metric.SUCCESS,
    )

    assert mock_get_slice.call_count == 2
    assert mock_success_fn.call_count == 2

    assert successes == [True, False]


@patch("robust_llm.models.model_output_metrics.get_target_slice_in_logits")
@patch("robust_llm.models.model_output_metrics.loss_on_goal_from_logits")
def test_generation_metric_from_logits_loss(
    mock_loss_fn,
    mock_get_slice,
    sample_logits_gen,
    sample_input_ids_gen,
    sample_goal_gen,
):
    mock_get_slice.side_effect = [
        slice(1, 3),
        slice(2, 3),
    ]  # Example slices for logits
    mock_loss_fn.side_effect = [torch.tensor(1.2), torch.tensor(0.9)]

    losses = GenerationMetric.from_logits(
        sample_logits_gen,
        sample_input_ids_gen,
        sample_goal_gen,
        metric=Metric.LOSS,
        reduction="sum",
    )

    assert mock_get_slice.call_count == 2
    assert mock_loss_fn.call_count == 2

    expected_losses = torch.tensor([1.2, 0.9])
    assert isinstance(losses, torch.Tensor)
    assert torch.equal(losses, expected_losses)


@patch("robust_llm.models.model_output_metrics.get_target_slice_in_logits")
@patch("robust_llm.models.model_output_metrics.success_on_goal")
def test_generation_metric_from_logits_success(
    mock_success_fn,
    mock_get_slice,
    sample_logits_gen,
    sample_input_ids_gen,
    sample_goal_gen,
):
    mock_get_slice.side_effect = [slice(1, 3), slice(2, 3)]
    mock_success_fn.side_effect = [False, True]

    successes = GenerationMetric.from_logits(
        sample_logits_gen,
        sample_input_ids_gen,
        sample_goal_gen,
        metric=Metric.SUCCESS,
    )

    assert mock_get_slice.call_count == 2
    assert mock_success_fn.call_count == 2

    assert successes == [False, True]


@patch("robust_llm.models.model_output_metrics.GenerationMetric.from_logits")
def test_generation_metric_from_messages_success(
    mock_from_logits,
    mock_model,
    sample_input_data,
    sample_gen_target_data,
):
    # Mock the nested call result for two batches
    mock_from_logits.side_effect = [[True], [False]]

    # Mock message_lists_to_tokens for this specific test context
    mock_tokenized_output = MagicMock(
        input_ids=torch.tensor([[1, 2, 3, 10, 20], [4, 5, 6, 10, 20]]),
        attention_mask=torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 1, 1]]),
    )
    mock_list_output = {
        "input_ids": [[40, 50], [55]],
        "attention_mask": [[1, 1], [1]],
    }
    mock_model.message_lists_to_tokens.return_value = mock_tokenized_output
    mock_model.tokenize.side_effect = lambda data, **kwargs: (
        mock_tokenized_output
        if kwargs.get("return_tensors") == "pt"
        else mock_list_output
    )

    # Mock logits generator to simulate batching (batch size 1 for simplicity)
    logits_batch1 = [torch.randn(5, 100)]  # Batch 1: seq 5, vocab 100
    logits_batch2 = [torch.randn(5, 100)]  # Batch 2: seq 5, vocab 100
    mock_model.generation_logits_from_tokens.return_value = iter(
        [logits_batch1, logits_batch2]
    )

    results = GenerationMetric.from_messages(
        mock_model,
        sample_input_data,
        sample_gen_target_data,
        metric=Metric.SUCCESS,
    )

    # Check model calls
    assert mock_model.tokenize.call_count == 1  # Called for gen_target_data
    mock_model.message_lists_to_tokens.assert_called_once()
    mock_model.generation_logits_from_tokens.assert_called_once_with(
        input_ids=mock_tokenized_output.input_ids,
        attention_mask=mock_tokenized_output.attention_mask,
    )

    # Check the nested call to from_logits (once per batch)
    assert mock_from_logits.call_count == 2
    first_call_args, first_call_kwargs = mock_from_logits.call_args_list[0]
    second_call_args, second_call_kwargs = mock_from_logits.call_args_list[1]

    assert first_call_args[0] == logits_batch1
    assert (
        first_call_kwargs["input_ids"] == mock_tokenized_output.input_ids[0:1].tolist()
    )
    assert first_call_kwargs["goal"] == mock_list_output["input_ids"][0:1]
    assert first_call_kwargs["metric"] == Metric.SUCCESS

    assert second_call_args[0] == logits_batch2
    assert (
        second_call_kwargs["input_ids"] == mock_tokenized_output.input_ids[1:2].tolist()
    )
    assert second_call_kwargs["goal"] == mock_list_output["input_ids"][1:2]
    assert second_call_kwargs["metric"] == Metric.SUCCESS

    # Check final result
    assert results == [True, False]


@patch("robust_llm.models.model_output_metrics.get_full_embeds")
@patch("robust_llm.models.model_output_metrics.get_full_encoded_prompts")
@patch("robust_llm.models.model_output_metrics.GenerationLogits")
@patch("robust_llm.models.model_output_metrics.GenerationMetric.from_logits")
def test_generation_metric_from_embeds_loss(
    mock_from_logits,
    mock_generation_logits,
    mock_get_full_ids,
    mock_get_full_embeds,
    mock_model,
):
    prompt_input_ids = torch.tensor([[1, 2]])
    prompt_input_embeds = torch.randn(1, 2, 10)
    gen_target_data = ["Target String"]  # Function expects a sequence
    target_input_ids = torch.tensor([[3, 4, 5]])  # Mocked tokenized target
    target_embeds = torch.randn(1, 3, 10)  # Mocked target embeds
    full_embeds = torch.randn(1, 5, 10)  # Mocked combined embeds
    full_ids = [[1, 2, 3, 4, 5]]  # Mocked combined ids
    batch_logits = torch.randn(1, 5, 100)  # Mocked logits output for the batch

    # Mock model functions
    mock_model.tokenize.side_effect = lambda data, **kwargs: {
        "input_ids": target_input_ids
    }
    mock_model.get_embeddings.return_value = target_embeds

    # Mock helper functions
    mock_get_full_embeds.return_value = full_embeds
    mock_get_full_ids.return_value = full_ids

    # Mock Logit generator
    mock_logit_generator_instance = MagicMock()
    mock_logit_generator_instance.__iter__.return_value = iter([batch_logits])
    mock_generation_logits.from_embeddings.return_value = mock_logit_generator_instance

    # Mock nested call to from_logits
    mock_from_logits.return_value = torch.tensor([1.5])  # Example loss

    loss = GenerationMetric.from_embeds(
        mock_model,
        prompt_input_ids=prompt_input_ids,
        prompt_input_embeds=prompt_input_embeds,
        gen_target_data=gen_target_data,
        use_no_grad=True,
        metric=Metric.LOSS,
        reduction="mean",
    )

    # Check model calls
    mock_model.tokenize.assert_called_once_with(
        list(gen_target_data), return_tensors="pt", padding_side=None
    )
    mock_model.get_embeddings.assert_called_once_with(target_input_ids)

    # Check helper calls
    mock_get_full_embeds.assert_called_once_with(prompt_input_embeds, target_embeds)
    mock_get_full_ids.assert_called_once_with(
        prompt_input_ids.tolist(), target_input_ids.tolist()
    )

    # Check GenerationLogits call
    mock_generation_logits.from_embeddings.assert_called_once_with(
        mock_model,
        input_ids=prompt_input_ids,
        embeddings=full_embeds,
        use_no_grad=True,
    )

    # Check from_logits call
    mock_from_logits.assert_called_once_with(
        logits=batch_logits,
        input_ids=full_ids,
        goal=target_input_ids.tolist(),
        metric=Metric.LOSS,
        reduction="mean",
    )

    # Check final result
    assert torch.equal(loss, torch.tensor([1.5]))


@pytest.fixture
def sample_logits():
    # Batch size 2, 2 classes
    return torch.tensor([[1.0, -1.0], [-0.5, 0.5]])


@pytest.fixture
def sample_goal_clf():
    # Batch size 2
    return [0, 1]


@pytest.fixture
def sample_input_ids_gen():
    # Batch size 2, sequence length 5
    # Example: Goal starts at index 3 for first, index 4 for second
    return [[10, 20, 30, 40, 50], [15, 25, 35, 45, 55]]


@pytest.fixture
def sample_goal_gen():
    # Batch size 2, goal length 2 and 1 respectively
    # Matches sample_input_ids_gen last elements
    return [[40, 50], [55]]


@pytest.fixture
def sample_logits_gen():
    # Batch size 2, seq length 5, vocab size 100 (arbitrary)
    return torch.randn(2, 5, 100)


@pytest.fixture
def mock_model():
    model = MagicMock(spec=WrappedModel)  # Use spec for better mocking
    model.accelerator = MagicMock()
    # Mock tokenize to return simple list of lists based on input length
    model.tokenize = MagicMock(
        side_effect=lambda data, **kwargs: {
            "input_ids": (
                [[1, 2, 3]] * len(data) if isinstance(data, list) else [[1, 2, 3]]
            )
        }
    )
    model.message_lists_to_tokens = MagicMock(
        return_value=MagicMock(
            input_ids=torch.tensor([[1, 2, 3], [4, 5, 6]]),
            attention_mask=torch.tensor([[1, 1, 1], [1, 1, 1]]),
        )
    )
    model.decode = MagicMock(return_value="decoded_text")
    # Mock generation_logits_from_tokens to yield batches of logits
    logit_gen_batch1 = [
        torch.tensor([[0.1, 0.2, 0.7], [0.4, 0.3, 0.3], [0.6, 0.1, 0.3]])
    ]
    logit_gen_batch2 = [
        torch.tensor([[0.2, 0.5, 0.3], [0.1, 0.8, 0.1], [0.3, 0.3, 0.4]])
    ]
    model.generation_logits_from_tokens = MagicMock(
        return_value=iter([logit_gen_batch1, logit_gen_batch2])
    )
    # Mock get_embeddings
    model.get_embeddings = MagicMock(return_value=torch.randn(1, 2, 10))
    # Batch 1, seq 2, embed_dim 10

    return model


@pytest.fixture
def sample_input_data():
    return ConversationList(
        message_lists=[
            MessageList([Message(role="user", content="Hello")]),
            MessageList([Message(role="user", content="Hi")]),
        ]
    )


@pytest.fixture
def sample_gen_target_data():
    return ["Okay", "Alright"]
