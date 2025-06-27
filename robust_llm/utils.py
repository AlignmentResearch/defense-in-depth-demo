from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import random
import time
from collections.abc import Iterator, Sequence
from contextlib import ContextDecorator
from dataclasses import fields
from datetime import datetime
from functools import cached_property
from typing import TYPE_CHECKING, Any, Callable, Optional, Sized, TypeVar

import numpy as np
import torch
import torch.utils.data

from robust_llm import logger

# type checking
from robust_llm.dist_utils import DistributedRNG

T = TypeVar("T")
CallableT = TypeVar("CallableT", bound=Callable)


def to_numpy(tensor: torch.Tensor) -> np.ndarray:
    tensor = tensor.detach().cpu().clone()
    if tensor.dtype == torch.float16 or tensor.dtype == torch.bfloat16:
        tensor = tensor.float()
    return tensor.numpy()


def flatten_dict(d: dict, parent_key="", sep=".") -> dict:
    """
    Flatten a nested dictionary by concatenating nested keys with a separator.

    Args:
        d (dict): The dictionary to flatten
        parent_key (str): The parent key for nested dictionaries (used in recursion)
        sep (str): The separator to use between nested keys

    Returns:
        dict: A flattened dictionary
    """
    items: list = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k

        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep).items())
        else:
            items.append((new_key, v))

    return dict(items)


def list_of_dicts_to_dict_of_lists(
    list_of_dicts: list[dict[str, Any]],
) -> dict[str, list[Any]]:
    if len(list_of_dicts) == 0:
        return {}
    keys = list_of_dicts[0].keys()
    assert all(d.keys() == keys for d in list_of_dicts)
    return {key: [d[key] for d in list_of_dicts] for key in keys}


def deterministic_hash(obj: object) -> str:
    json_str = json.dumps(str(obj), sort_keys=True)
    encoded_string = json_str.encode()
    hash_object = hashlib.sha256(encoded_string)
    hash_hex = hash_object.hexdigest()
    return hash_hex


def nested_list_to_tuple(nested_list: list) -> tuple:
    return tuple(
        nested_list_to_tuple(sub_list) if isinstance(sub_list, list) else sub_list
        for sub_list in nested_list
    )


def maybe_make_deterministic(mode: bool, cublas_config: str) -> None:
    torch.use_deterministic_algorithms(mode)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = cublas_config


def write_lines_to_file(lines: list[str], file_path: str) -> None:
    # If the folder doesn't exist yet, make one
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    # Save the file
    with open(file_path, "w") as afile:
        afile.writelines([line + "\n" for line in lines])


def ask_for_confirmation(prompt: str) -> bool:
    while True:
        answer = input(prompt + " (y/n) ")
        if answer.lower() == "y":
            return True
        elif answer.lower() == "n":
            return False
        else:
            print("Please answer with 'y' or 'n'.")


def div_maybe_nan(a: int, b: int) -> float:
    if b == 0:
        return float("nan")
    return a / b


def get_readable_timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d_%H:%M:%S")


def get_randint_with_exclusions(
    high: int, exclusions: Sequence[int], rng: DistributedRNG
) -> int:
    """Get a random integer from [0, `high`), excluding the integers in `exclusions`."""
    assert len(exclusions) < high, "Too many excluded values!"
    MAX_NUM_ITERS = 1000

    value: Optional[int] = None

    # Replaced a previous implementation where we explicitly create a set of allowed
    # values. It was super slow when `high` was large and `exclusions` was small.
    iter = 0
    while value is None or value in exclusions:
        value = rng.randint(0, high - 1)
        iter += 1
        if iter > MAX_NUM_ITERS:
            raise ValueError("Too many iterations!")

    return value


class BalancedSampler(torch.utils.data.Sampler[int]):
    """A sampler that alternates between regular and adversarial data.

    Note that regardless of the relative sizes, regular and adversarial data will be
    sampled in equal proportions. Adversarial data points might be sampled many times
    during one loop (if there are more regular data than adversarial data).
    """

    def __init__(self, regular_data: Sized, adversarial_data: Sized) -> None:
        self.regular_data = regular_data
        self.adversarial_data = adversarial_data

        self.regular_data_sampler = torch.utils.data.RandomSampler(regular_data)
        self.adversarial_data_sampler = torch.utils.data.RandomSampler(
            adversarial_data, num_samples=len(regular_data)
        )

    def __iter__(self) -> Iterator[int]:
        n = len(self.regular_data)
        iter_regular = iter(self.regular_data_sampler)
        iter_adversarial = iter(self.adversarial_data_sampler)
        for _ in range(n):
            yield next(iter_regular)
            yield n + next(iter_adversarial)

    def __len__(self) -> int:
        return 2 * len(self.regular_data)


def auto_repr(cls):
    """Automatically generate __repr__ method for a class.

    This is useful for including `property`s and `cached_property`s in the repr.
    """

    def __repr__(self):
        cls_fields = fields(cls)
        parts = []
        for f in cls_fields:
            value = getattr(self, f.name)
            parts.append(f"{f.name}={value!r}")
        # Include properties
        for name, attr in cls.__dict__.items():
            if isinstance(attr, property) or isinstance(attr, cached_property):
                parts.append(f"{name}={getattr(self, name)!r}")
        return f"{cls.__name__}({', '.join(parts)})"

    cls.__repr__ = __repr__
    return cls


def is_correctly_padded(mask: torch.Tensor, padding_side: str) -> bool:
    """Check that the mask is for the correct padding side.

    For example, for right padding, the mask should be of the form:
    [1, 1, ..., 1, 0, 0, ..., 0].
    For left padding, the mask should be of the form:
    [0, 0, ..., 0, 1, 1, ..., 1].
    Args:
        mask (torch.Tensor): Must be 1D or 2D. Can contain True/False or 1/0.
        padding_side: Either "right" or "left".

    Returns:
        bool: Whether the mask is of the correct form.
    """
    assert mask.dim() <= 2, "The mask must be at most 2D."
    assert len(mask) > 0, "The mask should not be empty."

    is_boolean = set(mask.unique().tolist()) <= {0, 1}
    if not is_boolean:
        raise ValueError("The mask should contain only 0s and 1s.")

    if mask.dim() == 1:
        mask = mask.unsqueeze(0)

    for example_mask in mask:
        mask_sum = example_mask.sum().item()
        assert isinstance(mask_sum, int)

        if padding_side == "right":
            starts_ones = torch.all(example_mask[:mask_sum]).item()
            ends_zeros = torch.all(~example_mask[mask_sum:]).item()
            if not bool(starts_ones and ends_zeros):
                return False

        elif padding_side == "left":
            # For left padding, we look from the end of the sequence.
            switch_point = len(example_mask) - mask_sum

            starts_zeros = torch.all(~example_mask[:switch_point]).item()
            ends_ones = torch.all(example_mask[switch_point:]).item()
            if not bool(starts_zeros and ends_ones):
                return False
        else:
            raise ValueError(
                f"padding_side must be 'right' or 'left', not {padding_side}."
            )

    return True


class print_time(ContextDecorator):
    """Logs the time some code takes to execute.

    Usage:
        >>> with print_time("hello world"):
        ...    time.sleep(1)
        ...
        Time (hello world): 1.00s

        >>> @print_time()
        ... def my_func():
        ...    time.sleep(1)
        ...
        >>> my_func()
        Time (my_func): 1.00s
    """

    def __init__(self, label: str | None = None):
        self.label = label

    def __call__(self, func: CallableT) -> CallableT:
        # If this class is used as a decorator (not as a `with` block) then this
        # __call__ method is invoked. We can use the function name as the label.
        if self.label is None:
            self.label = func.__qualname__

        return super().__call__(func)

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, *exc):
        elapsed_time = time.perf_counter() - self.start_time
        label = f"({self.label})" if self.label else ""

        if elapsed_time >= 3600:
            time_str = (
                f"{int(elapsed_time//3600)}h{int(elapsed_time % 3600//60)}m"
                f"{elapsed_time % 60:.0f}s"
            )
        elif elapsed_time >= 60:
            time_str = f"{int(elapsed_time//60)}m{elapsed_time % 60:.0f}s"
        else:
            time_str = f"{elapsed_time:.2f}s"

        logger.info(f"Time {label}: {time_str}")
        return False


def restrict_to_last_nonpad_token(
    logits: torch.Tensor, attention_mask: torch.Tensor
) -> torch.Tensor:
    """Restrict logits to the last non-pad token in each right-padded sequence.

    Args:
        logits: The logits of shape (batch_size, seq_len, vocab_size).
        attention_mask: The attention mask of shape (batch_size, seq_len).
    """
    assert logits.shape[:2] == attention_mask.shape[:2]
    batch_size = logits.size(0)

    # Find last non-pad token position for each sequence
    last_token_pos = attention_mask.sum(dim=1) - 1

    final_token_logits = logits[torch.arange(batch_size), last_token_pos, ...]
    return final_token_logits


def deterministic_string(rng: random.Random) -> str:
    chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    return "".join(rng.choice(chars) for _ in range(8))


def is_ci() -> bool:
    """Returns true if the code is being executed in CircleCI or runpod.

    CI is a built-in CircleCI environment variable on CircleCI:
    https://circleci.com/docs/variables/#built-in-environment-variables

    We must set this manually in RunPod, since it doesn't set it by default.
    """
    return os.getenv("CI") == "true"
