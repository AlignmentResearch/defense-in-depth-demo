"""Script to filter and reupload the ReNeLLM dataset.

Note the original dataset was uploaded by scripts/ian/make_renellm_data.py
"""

import argparse

import datasets
from semver import Version

from robust_llm.rllm_datasets.dataset_utils import (
    create_rllm_tag,
    filter_dataset_length,
)

# Remove hardcoded DATASET_REPO_NAME
# DATASET_REPO_NAME = "AlignmentResearch/ReNeLLMStrongREJECT"
# 0 is for Benign, 1 is for Harmful
CLASS_LABELS = ["Benign", "Harmful"]


def main(
    dataset_repo_name: str,
    base_config_name: str,
    major_version: int,
    minor_version: int,
    patch_version: int,
):
    ds = datasets.load_dataset(
        dataset_repo_name, name=base_config_name, split="validation"
    )
    assert isinstance(ds, datasets.Dataset)
    ds = filter_dataset_length(ds)
    # ReNeLLM has no train partition or negative (harmless) examples
    empty_ds = ds.select(range(0))
    ds_dict = datasets.DatasetDict({"train": empty_ds, "validation": ds})

    new_config_name = f"{base_config_name}-short"
    ds_dict.push_to_hub(
        dataset_repo_name,
        config_name=new_config_name,
    )
    print("Dataset pushed successfully.")
    version_tag = Version(major_version, minor_version, patch_version)
    try:
        print(f"Creating RLLM tag: {version_tag} for repo: {dataset_repo_name}")
        create_rllm_tag(dataset_repo_name, version_tag)
        print("RLLM tag created successfully.")
    except Exception as e:
        print(f"Error creating RLLM tag: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter and reupload ReNeLLM dataset.")
    parser.add_argument(
        "--dataset_repo_name",
        type=str,
        default="AlignmentResearch/ReNeLLMStrongREJECT",
        help="Dataset repository name on Hugging Face Hub.",
    )
    parser.add_argument(
        "--base_config_name",
        type=str,
        default="renellm-input-100x200",
        help="Base config name to load from the dataset repo.",
    )
    # bump the version here manually when you make changes
    # (see README for more info)
    # These can also be made into arguments if needed
    parser.add_argument("--major_version", type=int, default=0)
    parser.add_argument("--minor_version", type=int, default=0)
    parser.add_argument("--patch_version", type=int, default=5)

    args = parser.parse_args()

    main(
        dataset_repo_name=args.dataset_repo_name,
        base_config_name=args.base_config_name,
        major_version=args.major_version,
        minor_version=args.minor_version,
        patch_version=args.patch_version,
    )
