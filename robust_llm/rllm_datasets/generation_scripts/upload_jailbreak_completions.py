"""Script to generate the Llama3Jailbreaks dataset"""

import json
from pathlib import Path

import datasets
import pandas as pd
import wandb
from datasets import Dataset, DatasetDict

from robust_llm.rllm_datasets.dataset_utils import make_pos_neg_versions
from robust_llm.rllm_datasets.generation_scripts.dataset_upload_handler import (
    DatasetUploadHandler,
)

WANDB_API = wandb.Api()
INPUT_REPO_NAME = "AlignmentResearch/JailbreakInputs"
INPUT_DATASET_REVISION = "2.0.2"
DATASET_REPO_NAME = "AlignmentResearch/JailbreakCompletions"
WANDB_PROJECT = "robust-llm"
WANDB_RUN_IDS = ["t4sb9wq0", "6kcje903", "j0m4f3s7", "alklbjai", "gnyv0jrh"]


def main(minor_version: int, patch_version: int):
    full_ds_dict = datasets.load_dataset(
        INPUT_REPO_NAME, revision=INPUT_DATASET_REVISION
    )
    assert isinstance(full_ds_dict, DatasetDict)

    # Process each run ID and get their completions
    processed_datasets = []
    for run_id in WANDB_RUN_IDS:
        ds_dict_copy = DatasetDict(full_ds_dict.copy())
        completions_df = download_completions_from_wandb(run_id)
        processed_ds = process_single_run(completions_df, ds_dict_copy)
        processed_datasets.append(processed_ds)

    # Concatenate all processed datasets
    concatenated_ds = datasets.concatenate_datasets(processed_datasets)
    full_ds_dict["train"] = concatenated_ds

    ds_dicts = create_dataset_versions(full_ds_dict)
    upload_dataset(ds_dicts, minor_version, patch_version)


def process_single_run(completions_df: pd.DataFrame, ds_dict: DatasetDict) -> Dataset:
    """Process a single run's completions and return the updated dataset."""
    train_ds = ds_dict["train"]
    indices_to_completions, indices_to_success = create_completion_mapping(
        completions_df
    )
    filtered_train_ds = update_and_filter_dataset(
        train_ds, indices_to_completions, indices_to_success
    )
    return filtered_train_ds


def create_completion_mapping(completions_df: pd.DataFrame) -> tuple[dict, dict]:
    indices_to_completions = {}
    indices_to_success = {}
    for row in completions_df.itertuples():
        index = row.filtered_indices  # type: ignore
        completion = row.generation_outputs  # type: ignore
        success = row.success  # type: ignore
        indices_to_completions[index] = completion
        indices_to_success[index] = success

    return indices_to_completions, indices_to_success


def update_and_filter_dataset(train_ds, indices_to_completions, indices_to_success):
    filtered_indices_set = set(indices_to_completions.keys())
    print(f"Number of examples to update: {len(filtered_indices_set)}")

    # Function to update completions in the dataset
    def update_completion(example, idx):
        if idx in indices_to_completions:
            example["completion"] = indices_to_completions[idx]

            # Store original values
            clf_label = example["clf_label"]
            proxy_clf_label = example["proxy_clf_label"]
            gen_target = example["gen_target"]
            proxy_gen_target = example["proxy_gen_target"]

            # If the model refused (`success=True`) and clf_label is 1,
            # then invert the label and gen_target.
            if idx in indices_to_success and indices_to_success[idx] and clf_label == 1:
                # Swap the values
                example["clf_label"] = proxy_clf_label
                example["proxy_clf_label"] = clf_label
                example["gen_target"] = proxy_gen_target
                example["proxy_gen_target"] = gen_target

        return example

    # Apply the update to the train split
    updated_train_ds = train_ds.map(
        update_completion,
        with_indices=True,
        desc="Updating completions from WandB data",
    )

    def filter_by_index(example, idx):
        return idx in filtered_indices_set

    filtered_train_ds = updated_train_ds.filter(
        filter_by_index,
        with_indices=True,
        desc="Filtering dataset to only include examples in WandB data",
    )

    return filtered_train_ds


def create_dataset_versions(full_ds_dict: DatasetDict) -> dict[str, DatasetDict]:
    pos_ds_dict, neg_ds_dict = make_pos_neg_versions(full_ds_dict)
    return {"default": full_ds_dict, "pos": pos_ds_dict, "neg": neg_ds_dict}


def upload_dataset(
    ds_dicts: dict[str, DatasetDict], minor_version: int, patch_version: int
):
    dataset_uploader = DatasetUploadHandler(
        ds_repo_name=DATASET_REPO_NAME,
        ds_dicts=ds_dicts,
        minor_version=minor_version,
        patch_version=patch_version,
    )
    dataset_uploader.push_to_hub_and_create_tag()


def download_completions_from_wandb(run_id: str) -> pd.DataFrame:
    run = WANDB_API.run(f"{WANDB_PROJECT}/{run_id}")
    tables = run.logged_artifacts()
    table = [
        table
        for table in tables
        if "pre_attack_callback" in table.name and "defense" not in table.name
    ][0]
    table_dir = table.download()
    table_path = Path(table_dir) / "pre_attack_callback.table.json"
    with open(table_path) as file:
        json_dict = json.load(file)
    return pd.DataFrame(json_dict["data"], columns=json_dict["columns"])


if __name__ == "__main__":
    # bump the version here manually when you make changes
    # (see README for more info)
    MINOR_VERSION = 0
    PATCH_VERSION = 4
    main(minor_version=MINOR_VERSION, patch_version=PATCH_VERSION)
