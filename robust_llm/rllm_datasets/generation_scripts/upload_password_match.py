"""Script to generate the PasswordMatch (formerly tensor_trust) dataset"""

from datasets import Dataset, DatasetDict

from robust_llm.rllm_datasets.dataset_utils import (
    DS_SHUFFLE_SEED,
    filter_dataset_length,
    make_pos_neg_versions,
)
from robust_llm.rllm_datasets.generation_scripts.dataset_upload_handler import (
    DatasetUploadHandler,
)
from robust_llm.rllm_datasets.generation_scripts.password_match_generation import (
    construct_password_match,
)

DATASET_REPO_NAME = "AlignmentResearch/PasswordMatch"


def main(minor_version: int, patch_version: int):
    """Create and save the PasswordMatch dataset.

    Process:
    - Generate a large PasswordMatch dataset using the same logic as the old
        tensor_trust code.
    - Apply our processing:
        - Filter out examples that are too long for our models.
        - Split the dataset into train and validation sets.
        - Shuffle both sets.
        - Create a 'chunked_text' column for each example.
    - Also save a couple of special versions:
        - Only positive examples.
        - Only negative examples.
    """

    train, val = construct_password_match()
    assert isinstance(train, Dataset)
    assert isinstance(val, Dataset)

    # Dataset creation section
    train = process_password_match(train)
    val = process_password_match(val)
    full_ds_dict = DatasetDict({"train": train, "validation": val})
    pos_ds_dict, neg_ds_dict = make_pos_neg_versions(full_ds_dict)

    # Upload section
    dataset_uploader = DatasetUploadHandler(
        ds_repo_name=DATASET_REPO_NAME,
        ds_dicts={"default": full_ds_dict, "pos": pos_ds_dict, "neg": neg_ds_dict},
        minor_version=minor_version,
        patch_version=patch_version,
    )
    dataset_uploader.push_to_hub_and_create_tag()


def process_password_match(ds: Dataset) -> Dataset:
    ds = filter_dataset_length(ds)
    # shuffle deterministically
    ds = ds.shuffle(seed=DS_SHUFFLE_SEED)
    return ds


if __name__ == "__main__":
    # bump the version here manually when you make changes
    # (see README for more info)
    MINOR_VERSION = 1
    PATCH_VERSION = 0
    main(minor_version=MINOR_VERSION, patch_version=PATCH_VERSION)
