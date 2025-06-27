"""Script to generate the DoNotAnswer dataset"""

import datasets

from robust_llm.rllm_datasets.dataset_utils import cast_column_to_feature
from robust_llm.rllm_datasets.generation_scripts.dataset_upload_handler import (
    DatasetUploadHandler,
)

DATASET_REPO_NAME = "AlignmentResearch/DoNotAnswer"
# 0 is for Benign, 1 is for Harmful
CLASS_LABELS = ["Benign", "Harmful"]


def main(minor_version: int, patch_version: int):
    """Create and save the DoNotAnswer dataset."""
    ds_dicts = prepare_donotanswer_dataset()
    # Upload section
    dataset_uploader = DatasetUploadHandler(
        ds_repo_name=DATASET_REPO_NAME,
        ds_dicts=ds_dicts,
        minor_version=minor_version,
        patch_version=patch_version,
    )
    dataset_uploader.push_to_hub_and_create_tag()


def prepare_donotanswer_dataset() -> dict:
    ds = datasets.load_dataset(
        "walledai/DNA",
        split="train",
    )
    assert isinstance(ds, datasets.Dataset)
    ds = ds.filter(lambda x: x["harm_area"] == "Assisting illegal activities")
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
            "gen_target": " I'm sorry, as an AI language model I can't help you with that.",  # noqa: E501
            "proxy_gen_target": " Sure, here",
        },
        remove_columns=[
            "prompt",
            "risk_area",
            "harm_area",
            "harms",
        ],
    )
    # DoNotAnswer has no validation partition or negative (harmless) examples
    empty_ds = ds.select(range(0))
    ds_dict = datasets.DatasetDict({"train": ds, "validation": empty_ds})
    empty_ds_dict = datasets.DatasetDict({"train": empty_ds, "validation": empty_ds})
    return {"default": ds_dict, "pos": ds_dict, "neg": empty_ds_dict}


if __name__ == "__main__":
    # bump the version here manually when you make changes
    # (see README for more info)
    MINOR_VERSION = 0
    PATCH_VERSION = 0
    main(minor_version=MINOR_VERSION, patch_version=PATCH_VERSION)
