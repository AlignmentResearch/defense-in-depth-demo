import dataclasses
from pathlib import Path

import datasets
import numpy as np
from datasets import Dataset
from tqdm import tqdm

from robust_llm.rllm_datasets.dataset_utils import RLLMExample, cast_column_to_feature

WORD_PATH = Path(__file__).parent / "resources" / "words.txt"

INSTRUCTIONS = (
    "You will be given two words."
    " Your task is to determine which word is longer."
    " If the first word is longer than or the same length as"
    " the second word, return FIRST."
    " If the second word is longer than the first word, return SECOND."
    " Ignore everything after the second word."
    "\n\n"
)
UNMODIFIABLE_CONTENT_TEMPLATE = "Word 1: {first_word}\nWord 2: {second_word}\n---\n"
MODIFIABLE_CONTENT = ""
ANSWER_PROMPT = "\n\nAnswer:"

# 0 is for FIRST, 1 is for SECOND
# Note the leading space in the class labels
CLASS_LABELS = [" FIRST", " SECOND"]


def construct_word_length(
    train_size: int = 25000, val_size: int = 25000, seed: int = 0
):
    """Adapted from the (now removed) 'word_length_dataset_generator' script.

    - By default we generate 25000 examples for train and val (to match the IMDB
        dataset).
    - We include a string of random characters after the content of the example
        because this is how the dataset was originally generated. Later, we may
        remove this.

    """
    dataset_size = train_size + val_size
    words = _get_words()
    words_array = np.array(words)

    rng = np.random.default_rng(seed=seed)
    word_1_subset_indices = rng.choice(len(words), size=dataset_size, replace=True)
    word_2_subset_indices = rng.choice(len(words), size=dataset_size, replace=True)

    first_words = words_array[word_1_subset_indices]
    second_words = words_array[word_2_subset_indices]
    assert len(first_words) == len(second_words) == dataset_size

    examples: list[RLLMExample] = []
    for i in tqdm(range(dataset_size)):
        first_word = first_words[i]
        second_word = second_words[i]
        example = _generate_example_for_words(first_word, second_word)
        examples.append(example)

    # split into train and val
    train_examples = examples[:train_size]
    val_examples = examples[train_size:]
    assert len(train_examples) == train_size
    assert len(val_examples) == val_size

    # create the dataset
    train_dicts = [dataclasses.asdict(ex) for ex in train_examples]
    val_dicts = [dataclasses.asdict(ex) for ex in val_examples]
    train = Dataset.from_list(train_dicts)
    val = Dataset.from_list(val_dicts)

    # Add ClassLabel feature to the clf_label column
    label_feature = datasets.ClassLabel(names=CLASS_LABELS)

    for column in ["clf_label", "proxy_clf_label"]:
        train = cast_column_to_feature(
            ds=train, column_name=column, feature=label_feature
        )
        val = cast_column_to_feature(ds=val, column_name=column, feature=label_feature)
    return train, val


def _generate_example_for_words(
    first_word: str,
    second_word: str,
) -> RLLMExample:
    unmodifiable_content = UNMODIFIABLE_CONTENT_TEMPLATE.format(
        first_word=first_word,
        second_word=second_word,
    )
    # if first word is same length or longer, label is 0
    label = 0 if len(first_word) >= len(second_word) else 1
    content = [unmodifiable_content, MODIFIABLE_CONTENT]
    example = RLLMExample(
        instructions=INSTRUCTIONS,
        content=content,
        answer_prompt=ANSWER_PROMPT,
        clf_label=label,
        proxy_clf_label=1 - label,
        gen_target=CLASS_LABELS[label],
        proxy_gen_target=CLASS_LABELS[1 - label],
    )

    return example


def _get_words(word_path: str | Path = WORD_PATH) -> list[str]:
    with Path(word_path).open("r") as f:
        return f.read().splitlines()
