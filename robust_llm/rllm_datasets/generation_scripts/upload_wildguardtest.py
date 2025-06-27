"""Script to generate the WildGuardTest dataset"""

import datasets

from robust_llm.rllm_datasets.dataset_utils import (
    cast_column_to_feature,
    gen_target_from_label,
)
from robust_llm.rllm_datasets.generation_scripts.dataset_upload_handler import (
    DatasetUploadHandler,
)

DATASET_REPO_NAME = "AlignmentResearch/WildGuardTest"
# 0 is for Benign, 1 is for Harmful
CLASS_LABELS = ["Benign", "Harmful"]


def main(minor_version: int, patch_version: int):
    """Create and save the WildGuardTest dataset.

    We filter for single-turn benign conversations.
    """
    ds_dicts = prepare_wildguardtest_dataset()
    # Upload section
    dataset_uploader = DatasetUploadHandler(
        ds_repo_name=DATASET_REPO_NAME,
        ds_dicts=ds_dicts,
        minor_version=minor_version,
        patch_version=patch_version,
    )
    dataset_uploader.push_to_hub_and_create_tag()


def prepare_wildguardtest_dataset() -> dict:
    ds = datasets.load_dataset(
        path="allenai/wildguardmix", name="wildguardtest", split="test"
    )
    assert isinstance(ds, datasets.Dataset)
    INSTRUCTIONS = ""
    ANSWER_PROMPT = ""
    CLASS_LABELS = ["Benign", "Harmful"]

    # Filter out entries with None harmfulness labels (i.e., that lacked
    # annotator agreement on their harmfulness).
    ds = ds.filter(
        lambda x: all(
            x[column] is not None
            for column in ["prompt_harm_label", "response_harm_label"]
        )
    )
    assert all(x["prompt_harm_label"] in ["harmful", "unharmful"] for x in ds)  # type: ignore  # noqa: E501
    assert all(x["response_harm_label"] in ["harmful", "unharmful"] for x in ds)  # type: ignore  # noqa: E501

    ds = ds.map(
        lambda x: {
            "clf_label": 1 if x["prompt_harm_label"] == "harmful" else 0,
            "completion_clf_label": 1 if x["response_harm_label"] == "harmful" else 0,
        },
    )
    label_feature = datasets.ClassLabel(names=CLASS_LABELS)
    for column in ["clf_label", "completion_clf_label"]:
        ds = cast_column_to_feature(ds=ds, column_name=column, feature=label_feature)

    ds = ds.map(
        lambda x: {
            "instructions": INSTRUCTIONS,
            "content": [x["prompt"]],
            "completion": x["response"],
            "answer_prompt": ANSWER_PROMPT,
            "proxy_clf_label": 1 - x["clf_label"],
            "gen_target": gen_target_from_label(x["clf_label"], label_feature),
            "proxy_gen_target": gen_target_from_label(
                1 - x["clf_label"], label_feature
            ),
        },
        remove_columns=[
            "prompt",
            "response",
            "adversarial",
            "prompt_harm_label",
            "response_refusal_agreement",
            "response_refusal_label",
            "response_harm_label",
            "subcategory",
            "prompt_harm_agreement",
            "response_harm_agreement",
        ],
    )

    empty_ds = ds.select(range(0))
    return {
        "default": datasets.DatasetDict({"train": ds, "validation": empty_ds}),
        "pos": datasets.DatasetDict(
            {
                "train": ds.filter(
                    lambda x: x["clf_label"] == 1 and x["completion_clf_label"] == 1
                ),
                "validation": empty_ds,
            }
        ),
        "neg": datasets.DatasetDict(
            {
                "train": ds.filter(
                    lambda x: x["clf_label"] == 0 and x["completion_clf_label"] == 0
                ),
                "validation": empty_ds,
            }
        ),
        "prompt_pos": datasets.DatasetDict(
            {
                "train": ds.filter(lambda x: x["clf_label"] == 1),
                "validation": empty_ds,
            }
        ),
        "prompt_neg": datasets.DatasetDict(
            {
                "train": ds.filter(lambda x: x["clf_label"] == 0),
                "validation": empty_ds,
            }
        ),
        "completion_pos": datasets.DatasetDict(
            {
                "train": ds.filter(lambda x: x["completion_clf_label"] == 1),
                "validation": empty_ds,
            }
        ),
        "completion_neg": datasets.DatasetDict(
            {
                "train": ds.filter(lambda x: x["completion_clf_label"] == 0),
                "validation": empty_ds,
            }
        ),
    }


if __name__ == "__main__":
    # bump the version here manually when you make changes
    # (see README for more info)
    MINOR_VERSION = 0
    PATCH_VERSION = 0
    main(minor_version=MINOR_VERSION, patch_version=PATCH_VERSION)
