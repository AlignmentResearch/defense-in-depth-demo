"""Script to generate the JailbreakInputs dataset"""

from collections.abc import Sequence

import datasets
from datasets import Dataset

from robust_llm.rllm_datasets.dataset_utils import (
    cast_column_to_feature,
    concatenate_huggingface_datasets,
)
from robust_llm.rllm_datasets.generation_scripts.dataset_upload_handler import (
    DatasetUploadHandler,
)

DATASET_REPO_NAME = "AlignmentResearch/JailbreakInputs"
SPLIT_MAP: dict[str, dict[str, str | Sequence[str]]] = {
    "Mechanistic-Anomaly-Detection/llama3-jailbreaks": {
        "pos_train": "circuit_breakers_train",
        "neg_train": ["benign_instructions_train", "or_bench_train"],
        "pos_val": "circuit_breakers_test",
        "neg_val": ["benign_instructions_test", "or_bench_test"],
    },
    "AlignmentResearch/AugmentedJailbreaks": {
        "pos_train": [
            "semantic_jailbreak",
            "random_tokens",
            "cipher_suffix",
            "caesar_shift",
            "atbash",
            "base64",
            "binary",
            "braille",
            "hex",
            "morse",
            "pig_latin",
            "reverse",
            "rot13",
            "vigenere",
        ],
        "neg_train": [
            "make_harmless",
            "double_entendre",
        ],
    },
}
# 0 is for Benign, 1 is for Harmful
CLASS_LABELS = ["Benign", "Harmful"]
PREFIX = "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
SUFFIX = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
GEN_TARGETS = [
    " Benign",
    " Harmful",
]


def main(minor_version: int, patch_version: int):
    """Create and save the JailbreakInputs dataset.

    We use the generic prep_huggingface_dataset function to prepare the dataset,
    which assumes it's binary clf and that the columns text and label exist.
    """
    ds_dicts = concatenate_huggingface_datasets(
        SPLIT_MAP,
        process_jailbreak_ds,
    )
    # Upload section
    dataset_uploader = DatasetUploadHandler(
        ds_repo_name=DATASET_REPO_NAME,
        ds_dicts=ds_dicts,
        minor_version=minor_version,
        patch_version=patch_version,
    )
    dataset_uploader.push_to_hub_and_create_tag()


def process_jailbreak_ds(ds: Dataset, label_index: int) -> Dataset:
    """Add new columns to a single Llama3Jailbreak datasets.dataset."""
    label = CLASS_LABELS[label_index]
    proxy_label = CLASS_LABELS[1 - CLASS_LABELS.index(label)]

    # Make sure clf_label is a ClassLabel feature with labels that line up with
    # the gen_targets.
    label_feature = datasets.ClassLabel(names=CLASS_LABELS)

    columns_to_remove = [
        col
        for col in ds.column_names
        if col in ["prompt", "idx", "prompt_original", "dataset_source"]
    ]
    ds = ds.map(
        lambda x: {
            "instructions": "",
            "answer_prompt": "",
            "content": [x["prompt"].replace(PREFIX, "").replace(SUFFIX, "")],
            "clf_label": label,
            "proxy_clf_label": proxy_label,
            "gen_target": GEN_TARGETS[label_index],
            "proxy_gen_target": GEN_TARGETS[1 - label_index],
        },
        remove_columns=columns_to_remove,
    )
    for column in ["clf_label", "proxy_clf_label"]:
        ds = cast_column_to_feature(ds=ds, column_name=column, feature=label_feature)
    for chunks in ds["content"]:
        assert not any(
            [
                substr in chunks[0]
                for substr in [
                    "<|begin_of_text|>",
                    "<|start_header_id|>",
                    "<|end_header_id|>",
                    "<|eot_id|>",
                ]
            ]
        )
    return ds


if __name__ == "__main__":
    # bump the version here manually when you make changes
    # (see README for more info)
    MINOR_VERSION = 0
    PATCH_VERSION = 3
    main(minor_version=MINOR_VERSION, patch_version=PATCH_VERSION)
