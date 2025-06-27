import dataclasses
import random
from functools import cache
from unittest.mock import patch

import pytest
import torch
from hypothesis import assume, given
from hypothesis import strategies as st
from transformers.models.auto.configuration_auto import AutoConfig
from transformers.models.auto.tokenization_auto import AutoTokenizer

from robust_llm.dist_utils import dist_rmtree
from robust_llm.utils import (
    deterministic_string,
    flatten_dict,
    is_correctly_padded,
    nested_list_to_tuple,
)


@pytest.fixture
def temp_directory(tmp_path):
    """Create a temporary directory for testing."""
    test_dir = tmp_path / "test_dir"
    test_dir.mkdir()
    (test_dir / "test_file.txt").write_text("Test content")
    yield test_dir


def test_remove_directory_success(temp_directory):
    """Test successful removal of a directory."""
    dist_rmtree(str(temp_directory))
    assert not temp_directory.exists()


@patch("shutil.rmtree")
@patch("time.sleep")
def test_remove_directory_retry(mock_sleep, mock_rmtree, temp_directory):
    """Test retrying removal on OSError."""
    mock_rmtree.side_effect = [OSError("Test error"), None]

    dist_rmtree(str(temp_directory), retries=2, cooldown_seconds=1)

    assert mock_rmtree.call_count == 2
    assert mock_sleep.call_count == 1
    mock_sleep.assert_called_with(1)


@patch("shutil.rmtree")
@patch("time.sleep")
def test_remove_directory_max_retries(mock_sleep, mock_rmtree, temp_directory):
    """Test maximum retries reached."""
    mock_rmtree.side_effect = OSError("Test error")

    with pytest.raises(OSError, match="Test error"):
        dist_rmtree(str(temp_directory), retries=4)

    assert mock_rmtree.call_count == 5
    assert mock_sleep.call_count == 4


@patch("shutil.rmtree")
@patch("time.sleep")
def test_remove_directory_exponential_backoff(mock_sleep, mock_rmtree, temp_directory):
    """Test exponential backoff in sleep times."""
    mock_rmtree.side_effect = [OSError("Test error")] * 4 + [None]

    dist_rmtree(str(temp_directory))

    assert mock_rmtree.call_count == 5
    assert mock_sleep.call_count == 4


def test_remove_directory_permission_error(temp_directory):
    """Test handling of PermissionError."""
    with patch("shutil.rmtree", side_effect=PermissionError("Permission denied")):
        with pytest.raises(PermissionError, match="Permission denied"):
            dist_rmtree(str(temp_directory), retries=0)


@pytest.fixture(scope="module")
def tokenizer():
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-14m")
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


@given(text1=st.text(), text2=st.text())
def test_is_correctly_padded_true(tokenizer, text1: str, text2: str):
    """Test the `is_correctly_padded` function.

    We do this by tokenizing some input texts and
    checking that the returned masks pass the test.
    """
    # If both texts are empty then the mask is empty.
    assume(text1 != "" and text2 != "")
    texts = [text1, text2]
    padding_side = "right"
    tokenizer.padding_side = padding_side
    tokenized = tokenizer(texts, padding=True, return_tensors="pt")
    masks = tokenized["attention_mask"]
    for mask in masks:
        assert is_correctly_padded(mask, padding_side)

    padding_side = "left"
    tokenizer.padding_side = padding_side
    tokenized = tokenizer(texts, padding=True, return_tensors="pt")
    masks = tokenized["attention_mask"]
    for mask in masks:
        assert is_correctly_padded(mask, padding_side)


def test_is_correctly_padded_false():
    left_mask = torch.tensor([0, 1, 1, 1, 1, 1])
    right_mask = torch.tensor([1, 1, 1, 1, 1, 0])
    bad_mask = torch.tensor([1, 1, 0, 0, 0, 1, 1])

    assert not is_correctly_padded(left_mask, "right")
    assert not is_correctly_padded(right_mask, "left")
    assert not is_correctly_padded(bad_mask, "right")
    assert not is_correctly_padded(bad_mask, "left")


def test_nested_list_to_tuple():
    nested = [[1, 2], [3, 4]]
    assert nested_list_to_tuple(nested) == ((1, 2), (3, 4))


def test_flatten_dict():
    d = {
        "a": 1,
        "b": {"c": 2, "d": {"e": 3}},
        "f": {"g": 4},
    }
    assert flatten_dict(d) == {"a": 1, "b.c": 2, "b.d.e": 3, "f.g": 4}


def test_deterministic_string_consistency():
    """Test that the same seed produces the same string"""
    first_rng = random.Random(123)
    second_rng = random.Random(123)
    first_result = deterministic_string(first_rng)
    second_result = deterministic_string(second_rng)
    assert first_result == second_result


def test_different_seeds_different_strings():
    """Test that different seeds produce different strings"""
    first_rng = random.Random(42)
    second_rng = random.Random(43)
    result1 = deterministic_string(first_rng)
    result2 = deterministic_string(second_rng)
    assert result1 != result2


def test_deterministic_strings_multiple_seeds():
    """Test the function with various seeds to ensure consistent behavior"""
    seeds = [0, 1, 2, 100, 1000000]
    results = [deterministic_string(random.Random(seed)) for seed in seeds]

    # Verify all results are unique
    assert len(set(results)) == len(seeds)


@cache
def gemma_1p1_config():
    config = AutoConfig.from_pretrained("google/gemma-1.1-2b-it")
    config.intermediate_size = 10
    config.hidden_size = 128
    config.num_hidden_layers = 2
    return config


@cache
def gemma_2_config():
    config = AutoConfig.from_pretrained("google/gemma-2-9b-it")
    config.intermediate_size = 10
    config.hidden_size = 128
    config.num_hidden_layers = 2
    return config


@cache
def model_configs() -> dict[str, AutoConfig]:
    # We make this a cached function rather than a global variable because if it
    # were global, we'd call the gemma config functions and make slow network
    # requests during pytest collection even if no tests in this file run.
    return {
        "google/gemma-1.1-2b-it": gemma_1p1_config(),
        "google/gemma-2-9b-it": gemma_2_config(),
    }
