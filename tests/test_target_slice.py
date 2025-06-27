import pytest

from robust_llm.models.metric_utils import (
    get_target_slice_in_logits,
    get_target_slice_in_tokens,
)


def test_get_target_slice_in_tokens():
    input_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    target = [4, 5, 6]
    slice_in_tokens = get_target_slice_in_tokens(input_ids, target)
    assert slice_in_tokens.start == -7
    assert slice_in_tokens.stop == -4
    assert input_ids[slice_in_tokens.start : slice_in_tokens.stop] == target

    target = [1, 2, 3]
    slice_in_tokens = get_target_slice_in_tokens(input_ids, target)
    assert slice_in_tokens.start == -10
    assert slice_in_tokens.stop == -7
    assert input_ids[slice_in_tokens.start : slice_in_tokens.stop] == target

    target = [10]
    slice_in_tokens = get_target_slice_in_tokens(input_ids, target)
    assert slice_in_tokens.start == -1
    assert slice_in_tokens.stop is None
    assert input_ids[slice_in_tokens.start : slice_in_tokens.stop] == target


def test_get_target_slice_in_logits():
    input_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    target = [4, 5, 6]
    # If we were getting logits, tokens [3, 4, 5] would give the logits for [4, 5, 6].
    logit_target = [3, 4, 5]
    slice_in_logits = get_target_slice_in_logits(input_ids, target)
    assert slice_in_logits.start == -8
    assert slice_in_logits.stop == -5
    assert input_ids[slice_in_logits.start : slice_in_logits.stop] == logit_target

    target = [10]
    logit_target = [9]
    slice_in_logits = get_target_slice_in_logits(input_ids, target)
    assert slice_in_logits.start == -2
    assert slice_in_logits.stop == -1
    assert input_ids[slice_in_logits.start : slice_in_logits.stop] == logit_target


def test_get_target_slice_in_tokens_fail():
    input_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    # The target is not in the input_ids.
    target = [4, 4, 4]
    with pytest.raises(ValueError):
        get_target_slice_in_tokens(input_ids, target)


def test_get_target_slice_in_logits_fail():
    input_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    # The target is not in the input_ids.
    target = [4, 4, 4]
    with pytest.raises(ValueError):
        slice_in_logits = get_target_slice_in_logits(input_ids, target)

    # The target is at the beginning, so we can't get logits for it.
    target = [1, 2, 3]
    with pytest.raises(IndexError):
        slice_in_logits = get_target_slice_in_logits(input_ids, target)
        print(slice_in_logits)


def test_repetition():
    input_ids = [1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3]
    target = [1, 2, 3]
    logit_target = [3, 1, 2]
    slice_in_tokens = get_target_slice_in_tokens(input_ids, target)
    assert slice_in_tokens.start == -3
    assert slice_in_tokens.stop is None
    assert input_ids[slice_in_tokens.start : slice_in_tokens.stop] == target

    slice_in_logits = get_target_slice_in_logits(input_ids, target)
    assert slice_in_logits.start == -4
    assert slice_in_logits.stop == -1
    assert input_ids[slice_in_logits.start : slice_in_logits.stop] == logit_target
