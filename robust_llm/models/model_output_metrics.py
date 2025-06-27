"""Methods for computing binary success/fail or tensor loss."""

from collections.abc import Sequence
from enum import Enum
from functools import partial
from typing import Any, Callable, Literal, Union, overload

import torch
import torch.nn.functional as F

from robust_llm.message_utils import ConversationList, Message
from robust_llm.models.metric_utils import (
    get_full_embeds,
    get_full_encoded_prompts,
    get_target_slice_in_logits,
    loss_on_goal_from_logits,
    success_on_goal,
)
from robust_llm.models.model_outputs import ClassificationLogits, GenerationLogits
from robust_llm.models.wrapped_model import WrappedModel


class Metric(Enum):
    LOSS = "loss"
    SUCCESS = "success"


class MetricFunctionFactory:
    @staticmethod
    def get(
        metric: Union[Literal[Metric.LOSS], Literal[Metric.SUCCESS]],
        reduction: str,
    ) -> Union[
        Callable[[torch.Tensor, Sequence[int]], torch.Tensor],
        Callable[[torch.Tensor, Sequence[int]], bool],
    ]:
        if metric == Metric.LOSS:
            return partial(loss_on_goal_from_logits, reduction=reduction)
        elif metric == Metric.SUCCESS:
            return success_on_goal
        else:
            raise ValueError(f"Metric {metric} not supported.")


class ClassificationMetric:
    """Utility class providing classification metrics computation."""

    def __new__(cls, *args, **kwargs):
        raise TypeError(
            f"{cls.__name__} should not be instantiated. Use class methods instead."
        )

    @staticmethod
    @overload
    def from_logits(
        logits: torch.Tensor,
        goal: Sequence[int],
        metric: Literal[Metric.LOSS],
        clf_threshold: float = 0.5,
    ) -> torch.Tensor: ...

    @staticmethod
    @overload
    def from_logits(
        logits: torch.Tensor,
        goal: Sequence[int],
        metric: Literal[Metric.SUCCESS],
        clf_threshold: float = 0.5,
    ) -> list[bool]: ...

    @staticmethod
    def from_logits(
        logits: torch.Tensor,
        goal: Sequence[int],
        metric: Union[Literal[Metric.LOSS], Literal[Metric.SUCCESS]],
        clf_threshold: float = 0.5,
    ) -> Union[torch.Tensor, list[bool]]:
        """Compute the classification loss from the logits.

        Args:
            logits: Shape (batch, n_classes). The logits from the model.
            goal: Len batch. The goal classes.
            metric: The metric to compute, i.e. loss or success.
            clf_threshold: The threshold for the positive class in the binary
                classification setting.

        Returns:
            The classification loss, shape (batch,).
        """
        assert logits.shape[0] == len(goal)
        if metric == Metric.LOSS:
            goal_tensor = torch.tensor(goal, device=logits.device)
            return F.cross_entropy(logits, goal_tensor, reduction="none")
        elif metric == Metric.SUCCESS:
            assert logits.shape[1] == 2, "clf_threshold assumes binary classification"
            probs = torch.softmax(logits, dim=1)
            preds = (probs[:, 1] >= clf_threshold).long()
            return [pred.item() == goal for pred, goal in zip(preds, goal)]
        else:
            raise ValueError(f"Metric {metric} not supported.")

    @classmethod
    def from_messages(
        cls,
        victim: WrappedModel,
        input_data: ConversationList,
        clf_label_data: list[int],
        metric: Metric = Metric.SUCCESS,
        clf_threshold: float = 0.5,
    ) -> tuple[list[bool], list[list[float]]]:
        if metric != Metric.SUCCESS:
            raise ValueError(
                f"Metric {metric} is not supported for classification from_messages."
            )
        # We use right-padding for non-autoregressive outputs.
        tokenized = victim.message_lists_to_tokens(
            input_data.message_lists,
            message_filter=input_data.message_filter,
            add_generation_prompt=input_data.add_generation_prompt,
            padding_side="right",
        )
        input_ids = tokenized.input_ids
        attention_mask = tokenized.attention_mask
        return cls.from_tokens(
            victim,
            input_ids,
            attention_mask,
            clf_label_data,
            metric=metric,
            clf_threshold=clf_threshold,
        )

    @classmethod
    def from_tokens(
        cls,
        victim: WrappedModel,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        clf_label_data: list[int],
        metric: Metric = Metric.SUCCESS,
        clf_threshold: float = 0.5,
    ) -> tuple[list[bool], list[list[float]]]:
        if metric != Metric.SUCCESS:
            raise ValueError(
                f"Metric {metric} is not supported for classification from_tokens."
            )

        all_successes: list[bool] = []
        logit_generator = ClassificationLogits.from_tokens(
            victim,
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        batch_start = 0
        all_logits = []
        for logits in logit_generator:
            assert victim.accelerator is not None
            assert isinstance(logits, torch.Tensor)
            batch_length = logits.shape[0]
            batch_label_data = clf_label_data[batch_start : batch_start + batch_length]
            successes = cls.from_logits(
                logits,
                batch_label_data,
                metric=metric,
                clf_threshold=clf_threshold,
            )
            all_successes.extend(successes)
            all_logits.append(logits)
            batch_start += batch_length
        logits = torch.cat(all_logits)
        # Convert to list so it can be JSON serialized.
        return all_successes, logits.tolist()


class GenerationMetric:
    """Utility class providing generation metrics computation."""

    def __new__(cls, *args, **kwargs):
        raise TypeError(
            f"{cls.__name__} should not be instantiated. Use class methods instead."
        )

    @staticmethod
    @overload
    def from_logits(
        logits: torch.Tensor,
        input_ids: Sequence[Sequence[int]],
        goal: Sequence[Sequence[int]],
        metric: Literal[Metric.LOSS],
        reduction: str = "mean",
    ) -> torch.Tensor: ...

    @staticmethod
    @overload
    def from_logits(
        logits: torch.Tensor,
        input_ids: Sequence[Sequence[int]],
        goal: Sequence[Sequence[int]],
        metric: Literal[Metric.SUCCESS],
        reduction: str = "mean",
    ) -> list[bool]: ...

    @staticmethod
    def from_logits(
        logits: torch.Tensor,
        input_ids: Sequence[Sequence[int]],
        goal: Sequence[Sequence[int]],
        metric: Union[Literal[Metric.LOSS], Literal[Metric.SUCCESS]],
        reduction: str = "mean",
    ) -> Union[torch.Tensor, list[bool]]:
        """Compute the generation metric from the logits.

        NOTE: This is only used for losses from embeds, which is only used for GCG.

        Args:
            logits: Shape (batch, l_seq_len, vocab_size). The logits from the model.
            input_ids: Shape (batch, a_seq_len). The input_ids.
            goal: List of length 'batch' of lists of token ids. The tokenized goals.
            reduction: The reduction to apply to the losses. Either "mean" or "sum",
                default "mean".
            metric: The metric to compute (loss or success).

        Returns:
            The generation metric, shape (batch,).
        """
        assert logits.ndim == 3, "Logits should be (batch, seq_len, vocab_size)."
        assert len(goal) == logits.shape[0]
        score_fn = MetricFunctionFactory.get(metric, reduction)

        # Create type-specific collection based on metric
        scores = []

        for example_logits, example_ids, example_goal in zip(logits, input_ids, goal):
            target_slice = get_target_slice_in_logits(example_ids, example_goal)
            goal_logits = example_logits[target_slice]
            scores.append(score_fn(goal_logits, example_goal))

        return torch.stack(scores) if metric == Metric.LOSS else scores  # type: ignore

    @classmethod
    def from_messages(
        cls,
        victim: WrappedModel,
        input_data: ConversationList,
        gen_target_data: list[str],
        metric: Union[Literal[Metric.LOSS], Literal[Metric.SUCCESS]],
    ) -> Union[list[torch.Tensor], list[bool]]:
        """Compute either success or loss metrics for generation.

        Args:
            victim: The model to evaluate
            input_data: List of input messages
            gen_target_data: List of target strings to compare against
            metric: Which metric to compute - either "loss" or "success"

        Returns:
            Either a list of tensors of losses or list of success booleans
        """
        message_lists = input_data.message_lists
        # Add the gen targets to the message lists
        gen_target_messages = [
            Message(role="assistant", content=target) for target in gen_target_data
        ]

        message_lists = [
            message_list + gen_target_message
            for message_list, gen_target_message in zip(
                input_data.message_lists, gen_target_messages, strict=True
            )
        ]
        # We don't add the generation prompt because we already added an assistant
        # message.
        tokenized = victim.message_lists_to_tokens(
            message_lists,
            message_filter=input_data.message_filter,
            add_generation_prompt=False,
            padding_side="right",
        )
        # Get the input ids for the gen targets as well
        gen_target_input_ids = victim.tokenize(gen_target_data)["input_ids"]
        assert isinstance(gen_target_input_ids, list)

        all_input_ids = tokenized.input_ids
        all_attention_masks = tokenized.attention_mask
        logits_generator = victim.generation_logits_from_tokens(
            input_ids=all_input_ids,
            attention_mask=all_attention_masks,
        )
        batch_start = 0
        all_metric_results: list[Any] = []
        for logits in logits_generator:
            assert victim.accelerator is not None
            batch_length = len(logits)
            batch_target_ids = gen_target_input_ids[
                batch_start : batch_start + batch_length
            ]
            batch_input_ids = all_input_ids[batch_start : batch_start + batch_length]
            metric_result = cls.from_logits(
                logits,
                input_ids=batch_input_ids.tolist(),
                goal=batch_target_ids,
                metric=metric,
            )
            all_metric_results.extend(metric_result)
            batch_start += batch_length
        return all_metric_results

    @classmethod
    def from_embeds(
        cls,
        victim: WrappedModel,
        prompt_input_ids: torch.Tensor,
        prompt_input_embeds: torch.Tensor,
        gen_target_data: Sequence[str],
        use_no_grad: bool,
        metric: Literal[Metric.LOSS],
        reduction: str = "mean",
    ) -> torch.Tensor:
        """Compute the losses from the embeddings input.

        Args:
            victim: The model to evaluate.
            prompt_input_ids: The tokenized prompt input.
            prompt_input_embeds: The embedded prompt input.
            gen_target_data: The target data.
            use_no_grad: Whether to use torch.no_grad() when computing the losses.
            metric: The metric to compute (loss or success).
            reduction: The reduction to apply to the losses. Either "mean" or "sum",
                default "mean".
        """
        assert len(gen_target_data) == 1
        target_input_ids = victim.tokenize(
            list(gen_target_data),
            return_tensors="pt",
            padding_side=None,
        )["input_ids"]
        assert isinstance(target_input_ids, torch.Tensor)
        target_embeds = victim.get_embeddings(target_input_ids)

        full_embeds = get_full_embeds(prompt_input_embeds, target_embeds)

        all_losses: list[torch.Tensor] = []
        logit_generator = GenerationLogits.from_embeddings(
            victim,
            # It's fine that input_ids doesn't have the target since it's just for
            # checking for cache hits with early parts of the prompt anyway.
            input_ids=prompt_input_ids,
            embeddings=full_embeds,
            use_no_grad=use_no_grad,
        )
        full_input_ids = get_full_encoded_prompts(
            prompt_input_ids.tolist(), target_input_ids.tolist()
        )

        batch_start = 0
        for logits in logit_generator:
            assert victim.accelerator is not None
            assert isinstance(logits, torch.Tensor)
            batch_length = logits.shape[0]
            batch_input_ids = full_input_ids[batch_start : batch_start + batch_length]
            batch_target_ids = target_input_ids[
                batch_start : batch_start + batch_length
            ]
            losses = cls.from_logits(
                logits=logits,
                input_ids=batch_input_ids,
                goal=batch_target_ids.tolist(),
                metric=metric,
                reduction=reduction,
            )
            all_losses.append(losses)
            batch_start += batch_length
        return torch.cat(all_losses)
