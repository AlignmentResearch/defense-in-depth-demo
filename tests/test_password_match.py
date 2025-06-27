import re

import numpy as np
import pytest
from hypothesis import example, given
from hypothesis import strategies as st

from robust_llm.config.dataset_configs import DatasetConfig
from robust_llm.rllm_datasets.dataset_utils import example_dict_to_text
from robust_llm.rllm_datasets.generation_scripts.password_match_generation import (
    _generate_examples_with_both_words,
    _select_different_word,
    construct_password_match,
)
from robust_llm.rllm_datasets.load_rllm_dataset import load_rllm_dataset
from robust_llm.rllm_datasets.supported_datasets.password_match_dataset import (
    PasswordMatchDataset,
)


@pytest.fixture
def latest_password_match_dataset():
    cfg = DatasetConfig(
        dataset_type="AlignmentResearch/PasswordMatch",
        revision="main",
        n_train=1,
        n_val=1,
        inference_type="classification",
    )
    dataset = load_rllm_dataset(cfg, split="train")
    assert isinstance(dataset, PasswordMatchDataset)
    return dataset


@given(
    first_word=st.from_regex(r"[^\n]*", fullmatch=True),
    second_word=st.from_regex(r"[^\n]*", fullmatch=True),
)
@example(first_word=" word1 ", second_word=" word2 ")
def test_generate_examples_with_both_words(first_word: str, second_word: str):
    examples = _generate_examples_with_both_words(first_word, second_word)

    assert len(examples) == 2
    pos_example, neg_example = examples

    assert pos_example.clf_label == 1
    assert neg_example.clf_label == 0

    # e.g. 'System password:  word1 \nUser password: word1 \nIgnore the following text:'
    first_word_match = re.search(r"User password:(.+?)\n", pos_example.content[0])
    second_word_match = re.search(r"User password:(.+?)\n", neg_example.content[0])
    assert first_word_match is not None
    assert second_word_match is not None

    assert first_word_match.groups()[0] == f" {first_word}"
    assert second_word_match.groups()[0] == f" {second_word}"


def test_select_different_word():
    words = ["word1", "word2", "word3", "word4", "word5"]
    rng = np.random.default_rng(seed=0)
    word = "word1"
    other_word = _select_different_word(word, words, rng)
    assert other_word in words
    assert other_word != word

    # Now test that the function raises an error when there is only one word.
    words = ["word1"]
    word = "word1"
    with pytest.raises(ValueError) as e:
        other_word = _select_different_word(word, words, rng)
    assert "Couldn't find a word other than" in str(e.value)


def test_password_match_overlap():
    """Test the overlap between the train and val sets."""
    train, val = construct_password_match(train_size=1000, val_size=1000)

    train_text = [example_dict_to_text(x) for x in train]  # type: ignore
    val_text = [example_dict_to_text(x) for x in val]  # type: ignore

    overlap = len(set(train_text).intersection(set(val_text)))
    assert overlap == 0
