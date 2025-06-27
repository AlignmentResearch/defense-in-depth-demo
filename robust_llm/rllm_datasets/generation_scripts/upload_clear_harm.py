"""Script to generate the ClearHarm dataset"""

from pathlib import Path

import datasets
import yaml

from robust_llm.file_utils import compute_repo_path
from robust_llm.rllm_datasets.dataset_utils import cast_column_to_feature
from robust_llm.rllm_datasets.generation_scripts.dataset_upload_handler import (
    DatasetUploadHandler,
)

DATASET_REPO_NAME = "AlignmentResearch/ClearHarm"
# 0 is for Benign, 1 is for Harmful
CLASS_LABELS = ["Benign", "Harmful"]


def main(minor_version: int, patch_version: int):
    """Create and save the ClearHarm dataset."""
    ds_dicts = prepare_clear_harm_dataset()
    # Upload section
    dataset_uploader = DatasetUploadHandler(
        ds_repo_name=DATASET_REPO_NAME,
        ds_dicts=ds_dicts,
        minor_version=minor_version,
        patch_version=patch_version,
    )
    dataset_uploader.push_to_hub_and_create_tag()


def prepare_clear_harm_dataset() -> dict:
    with open(
        Path(compute_repo_path())
        / "robust_llm"
        / "rllm_datasets"
        / "generation_scripts"
        / "resources"
        / "clear_harm.yaml",
        "r",
    ) as f:
        data = yaml.safe_load(f)
    mapping = [
        {"prompt": prompt, "category": category}
        for category, prompts in data.items()
        for prompt in prompts
    ]
    ds = datasets.Dataset.from_list(
        mapping,
        features=datasets.Features(
            {"prompt": datasets.Value("string"), "category": datasets.Value("string")}
        ),
    )
    assert isinstance(ds, datasets.Dataset)
    ds = ds.shuffle(seed=42)
    INSTRUCTIONS = ""
    ANSWER_PROMPT = ""
    CLASS_LABELS = ["Benign", "Harmful"]

    ds = ds.add_column("clf_label", [1] * len(ds))  # type: ignore
    label_feature = datasets.ClassLabel(names=CLASS_LABELS)
    ds = cast_column_to_feature(ds=ds, column_name="clf_label", feature=label_feature)

    ds = ds.map(
        lambda x: {
            "instructions": INSTRUCTIONS,
            "content": [x["prompt"]],
            "answer_prompt": ANSWER_PROMPT,
            "proxy_clf_label": 1 - x["clf_label"],
            # We use a generic target that GCG can optimize for.
            # Note the leading space, which can be stripped from DatasetConfig.
            "gen_target": " I'm sorry, as an AI language model I can't help you with that.",  # noqa: E501
            "proxy_gen_target": " Sure, here",
        },
        remove_columns=[
            "prompt",
            "category",
        ],
    )
    # ClearHarm has no validation partition or negative (harmless) examples
    empty_ds = ds.select(range(0))
    ds_dict = datasets.DatasetDict({"train": ds, "validation": empty_ds})
    empty_ds_dict = datasets.DatasetDict({"train": empty_ds, "validation": empty_ds})
    return {"default": ds_dict, "pos": ds_dict, "neg": empty_ds_dict}


if __name__ == "__main__":
    # bump the version here manually when you make changes
    # (see README for more info)
    MINOR_VERSION = 0
    PATCH_VERSION = 2
    main(minor_version=MINOR_VERSION, patch_version=PATCH_VERSION)
