from robust_llm.rllm_datasets.dataset_utils import example_dict_to_text
from robust_llm.rllm_datasets.generation_scripts.word_length_generation import (
    _generate_example_for_words,
    construct_word_length,
)


def test_generate_example_for_words():
    """Check that the example is generated correctly."""
    first_word = "first_word"
    second_word = "second_word"
    clf_label = 1

    generated_example = _generate_example_for_words(
        first_word=first_word,
        second_word=second_word,
    )

    assert first_word in generated_example.content[0]
    assert second_word in generated_example.content[0]
    assert generated_example.content[1] == ""
    assert generated_example.clf_label == clf_label
    assert generated_example.gen_target == " SECOND"


def test_generate_example_for_words_same_length():
    """Check the correct label is generated when the words are the same length."""
    generated_example = _generate_example_for_words(
        "first_word",
        "first_word",
    )
    assert generated_example.clf_label == 0


def test_word_length_overlap():
    """Test the overlap between the train and val sets."""
    train, val = construct_word_length(train_size=1000, val_size=1000)
    train_text = [example_dict_to_text(x) for x in train]  # type: ignore
    val_text = [example_dict_to_text(x) for x in val]  # type: ignore

    overlap = len(set(train_text).intersection(set(val_text)))
    assert overlap == 0
