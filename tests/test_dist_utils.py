"""Tests for dist_utils.py, which contains utilities for working with multiple GPUs."""

import math
from copy import deepcopy

import pytest
import torch
from accelerate import Accelerator
from hypothesis import given
from hypothesis import strategies as st

from robust_llm.dist_utils import (
    broadcast_list_of_bools,
    broadcast_list_of_floats,
    broadcast_list_of_str,
    broadcast_tensor,
)


@pytest.mark.multigpu
@given(
    tensor_dims=st.lists(st.integers(min_value=1, max_value=10), min_size=1, max_size=5)
)
def test_broadcast_tensor(tensor_dims: list[int]):
    torch.manual_seed(0)
    random_tensor = torch.randn(*tensor_dims)
    accelerator = Accelerator()
    inp = random_tensor if accelerator.is_main_process else None
    expected_output = random_tensor.clone()
    output = broadcast_tensor(inp, accelerator=accelerator)
    assert torch.allclose(output, expected_output.to(output.device))


@pytest.mark.multigpu
@given(bools=st.lists(st.booleans(), min_size=0, max_size=100))
def test_broadcast_list_of_bools(bools: list[bool]):
    accelerator = Accelerator()
    inp = bools if accelerator.is_main_process else None
    expected_output = deepcopy(bools)
    output = broadcast_list_of_bools(inp, accelerator=accelerator)
    assert output == expected_output


@pytest.mark.multigpu
@given(floats=st.lists(st.floats(width=32), min_size=0, max_size=100))
def test_broadcast_list_of_floats(floats: list[float]):
    accelerator = Accelerator()
    inp = floats if accelerator.is_main_process else None
    expected_output = deepcopy(floats)
    output = broadcast_list_of_floats(inp, accelerator=accelerator)
    assert len(output) == len(expected_output) and all(
        (math.isnan(a) and math.isnan(b)) or a == b
        for a, b in zip(output, expected_output)
    )


@pytest.mark.multigpu
@given(strings=st.lists(st.text(), min_size=0, max_size=100))
def test_broadcast_list_of_str(strings: list[str]):
    """Test broadcasting a list of strings."""
    accelerator = Accelerator()
    inp = strings if accelerator.is_main_process else None
    expected_output = deepcopy(strings)
    output = broadcast_list_of_str(inp, accelerator=accelerator)
    assert output == expected_output
