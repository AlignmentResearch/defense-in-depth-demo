from typing import Callable

import datasets
from datasets import Dataset, DatasetDict

from robust_llm.rllm_datasets.dataset_utils import get_largest_version_below
from robust_llm.rllm_datasets.generation_scripts.dataset_upload_handler import (
    DatasetUploadHandler,
)


def get_old_ds(repo_name: str, split: str, config_name: str = "default") -> Dataset:
    """Get the largest version of the raw datasets.Dataset below 1.0.0."""
    version = get_largest_version_below(repo_name, "1.0.0")
    assert version is not None
    revision = str(version)
    dataset = datasets.load_dataset(
        path=repo_name,
        name=config_name,
        revision=revision,
        split=split,
    )
    assert isinstance(dataset, Dataset)
    return dataset


def convert_and_upload(
    repo_name: str,
    new_minor_version: int,
    new_patch_version: int,
    converter_func: Callable[[Dataset], Dataset],
):
    """Convert 0.x.x dataset to a new format and upload it."""
    # Built new ds_dicts
    ds_dicts: dict[str, DatasetDict] = {}
    for config_name in ["default", "pos", "neg"]:
        new_ds_dict = {}
        for split in ["train", "validation"]:
            old_dataset = get_old_ds(repo_name, split, config_name)
            new_dataset = converter_func(old_dataset)
            new_ds_dict[split] = new_dataset
        ds_dicts[config_name] = DatasetDict(new_ds_dict)

    # Upload section
    dataset_uploader = DatasetUploadHandler(
        ds_repo_name=repo_name,
        ds_dicts=ds_dicts,
        minor_version=new_minor_version,
        patch_version=new_patch_version,
    )
    dataset_uploader.push_to_hub_and_create_tag()
