from collections.abc import Sequence
from typing import TypeVar

import torch

# Scoring function return type (Tensor for losses, bool for successes)
T = TypeVar("T")


def loss_on_goal_from_logits(
    logits: torch.Tensor, goal: Sequence[int], reduction: str = "mean"
):
    """Compute the generation losses from the logits.

    NOTE: This is only used for losses from embeds, which is only used for GCG.

    Args:
        logits:
            Shape (batch, target_len, vocab_size). The logits from the model on
            the target.
        goal:
            List of length 'target_len'. The tokenized goals.
        reduction:
            The reduction to apply to the losses. Either "mean" or "sum",
            default "mean".
    """
    all_logprobs = torch.log_softmax(logits, dim=-1)
    # Get logprobs just for the actual goal tokens
    goal_logprobs = all_logprobs[torch.arange(len(goal)), goal]
    return _loss_on_goal_from_goal_logprobs(goal_logprobs, reduction)


def _loss_on_goal_from_goal_logprobs(
    goal_logprobs: torch.Tensor, reduction: str = "mean"
):
    """Compute the generation losses from the goal logprobs.

    Args:
        goal_logprobs: Shape (batch, target_len). The logprobs for the goal tokens.
        reduction: The reduction to apply to the losses. Either "mean" or "sum",
            default "mean".
    """
    if reduction == "mean":
        return -goal_logprobs.mean()
    elif reduction == "sum":
        return -goal_logprobs.sum()
    else:
        raise ValueError(f"Invalid reduction: {reduction}")


def get_target_slice_in_tokens(
    input_ids: Sequence[int], target: Sequence[int]
) -> slice:
    """Get a slice corresponding to the last occurrence of 'target' in 'input_ids'."""
    assert len(input_ids) > 0 and isinstance(input_ids[0], int)

    # Get the start of the last occurrence of the target in the input_ids.
    start_indices = find_subsequence_start_indices(input_ids, target)
    if len(start_indices) == 0:
        raise ValueError(f"Could not find target {target} in input_ids {input_ids}.")
    start_index_in_tokens = start_indices[-1]
    end_index_in_logits = start_index_in_tokens + len(target)
    negative_start = start_index_in_tokens - len(input_ids)
    negative_end = end_index_in_logits - len(input_ids)
    # If the negative end index is 0, then we have to set it to None, because
    # otherwise 0 will be interpreted as the first index, and the slice will be empty.
    target_slice = slice(negative_start, negative_end or None)

    # We use 'or None' here to handle the case where target_slice.stop is 0,
    # which would mess up the slice.
    assert input_ids[target_slice.start : target_slice.stop] == target
    return target_slice


def get_target_slice_in_logits(
    input_ids: Sequence[int], target: Sequence[int]
) -> slice:
    """Get a slice corresponding to the last occurrence of 'target' using 'input_ids'.

    NOTE:
    - The slice will correspond to the target *in the logits*, which are offset
        by one. Thus when checking that the slice is correct, we have to add one.
    - The slice is *negative*, so it can be used in the logits even when caching
        means we get fewer logits than input_ids.

    """
    slice_in_tokens = get_target_slice_in_tokens(input_ids, target)
    if len(input_ids) + slice_in_tokens.start == 0:
        raise IndexError(
            f"Target {target} is at the beginning of input_ids {input_ids}, so"
            " we can't get logprobs for it."
        )
    slice_in_logits_start = slice_in_tokens.start - 1
    if slice_in_tokens.stop is None:
        slice_in_logits_stop = -1
    else:
        slice_in_logits_stop = slice_in_tokens.stop - 1

    return slice(slice_in_logits_start, slice_in_logits_stop)


def find_subsequence_start_indices(
    all_tokens: Sequence[T], subsequence: Sequence[T]
) -> list[int]:
    """Find all starting indices of a subsequence in a list of tokens.

    If the subsequence is not found, return an empty list.
    """
    indices = []
    for i in range(len(all_tokens) - len(subsequence) + 1):
        if all_tokens[i : i + len(subsequence)] == subsequence:
            indices.append(i)
    return indices


def success_on_goal(logits: torch.Tensor, goal: Sequence[int]):
    """Compute the generation successes from the logits.

    Args:
        logits:
            Shape (target_len, vocab_size). The logits from the model on
            the target.
        goal:
            List of length 'target_len'. The tokenized goals.
    """
    # For each position, get the token with highest logprob
    predicted_tokens = logits.argmax(dim=-1)
    assert len(predicted_tokens) == len(goal)
    return predicted_tokens.tolist() == goal


def get_full_encoded_prompts(
    prompt_ids: list[list[int]],
    target_ids: list[list[int]],
) -> list[list[int]]:
    """Get the full tokenized prompts by concatenating the prompt and target tokens.

    We can neglect the attention mask because the inputs should not have been
    padded yet. Padding will be done afterwards.

    Args:
        prompt_ids: The tokenized prompt input.
        target_ids: The tokenized target input.

    Returns:
        A list of the full input tokens, combining the prompt and target tokens.
    """

    return [prompt + target for prompt, target in zip(prompt_ids, target_ids)]


def get_full_embeds(
    prompt_embeds: torch.Tensor,
    target_embeds: torch.Tensor,
) -> torch.Tensor:
    """Get the full embedded prompts by concatenating prompt and target.

    We can neglect the attention mask because the inputs should not be padded
    when using embeds.

    Args:
        prompt_embeds: A tensor containing the embedded prompt inputs.
        target_embeds: A tensor containing the embedded target inputs.

    Returns:
        A tensor containing the full input embeddings, combining the prompt
        and target embeddings.
    """
    assert prompt_embeds.ndim == target_embeds.ndim == 3
    assert prompt_embeds.shape[0] == prompt_embeds.shape[0]

    if prompt_embeds.shape[0] != 1:
        raise NotImplementedError("Batch sizes greater than 1 not supported yet.")

    return torch.cat((prompt_embeds, target_embeds), dim=1)
