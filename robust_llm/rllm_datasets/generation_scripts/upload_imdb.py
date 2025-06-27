"""Script to generate the IMDB dataset"""

import datasets
from datasets import Dataset

from robust_llm.rllm_datasets.dataset_utils import (
    cast_column_to_feature,
    gen_target_from_label,
    prepare_huggingface_dataset,
)
from robust_llm.rllm_datasets.generation_scripts.dataset_upload_handler import (
    DatasetUploadHandler,
)

DATASET_REPO_NAME = "AlignmentResearch/IMDB"

# 0 is for NEGATIVE, 1 is for POSITIVE
# Note the leading space in the class labels
CLASS_LABELS = [" NEGATIVE", " POSITIVE"]


def main(minor_version: int, patch_version: int):
    """Create and save the IMDB dataset.

    We use the generic prep_huggingface_dataset function to prepare the dataset,
    which assumes it's binary classification and that the columns `text` and
    `label` exist.
    """
    ds_dicts = prepare_huggingface_dataset("imdb", ds_specific_callback=process_imdb_ds)
    # Upload section
    dataset_uploader = DatasetUploadHandler(
        ds_repo_name=DATASET_REPO_NAME,
        ds_dicts=ds_dicts,
        minor_version=minor_version,
        patch_version=patch_version,
    )
    dataset_uploader.push_to_hub_and_create_tag()


def process_imdb_ds(ds: Dataset) -> Dataset:
    """Add new columns to a single IMDB datasets.dataset."""

    INSTRUCTIONS = (
        "You will be given a review below to classify based on its sentiment."
        " The review will be either positive or negative."
        " If the review is positive, return POSITIVE."
        " If the review is negative, return NEGATIVE."
        "\n\n"
    )
    ANSWER_PROMPT = "\n\nAnswer:"

    # Make sure clf_label is a ClassLabel feature with labels that line up with
    # the gen_targets.
    label_feature = datasets.ClassLabel(names=CLASS_LABELS)
    ds = cast_column_to_feature(ds=ds, column_name="clf_label", feature=label_feature)

    ds = ds.map(
        lambda x: {
            "instructions": INSTRUCTIONS,
            "content": [x["text"]],
            "answer_prompt": ANSWER_PROMPT,
            "proxy_clf_label": 1 - x["clf_label"],
            "gen_target": gen_target_from_label(x["clf_label"], label_feature),
            "proxy_gen_target": gen_target_from_label(
                1 - x["clf_label"], label_feature
            ),
        },
        remove_columns=["text"],
    )
    ds = cast_column_to_feature(
        ds=ds, column_name="proxy_clf_label", feature=label_feature
    )
    return ds


if __name__ == "__main__":
    # bump the version here manually when you make changes
    # (see README for more info)
    MINOR_VERSION = 1
    PATCH_VERSION = 0
    main(minor_version=MINOR_VERSION, patch_version=PATCH_VERSION)
