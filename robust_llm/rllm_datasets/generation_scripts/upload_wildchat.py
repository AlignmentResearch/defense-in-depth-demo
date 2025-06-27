"""Script to generate the WildChat dataset"""

import datasets

from robust_llm.rllm_datasets.dataset_utils import (
    cast_column_to_feature,
    filter_dataset_for_context_length,
    gen_target_from_label,
)
from robust_llm.rllm_datasets.generation_scripts.dataset_upload_handler import (
    DatasetUploadHandler,
)

DATASET_REPO_NAME = "AlignmentResearch/WildChat"
# 0 is for Benign, 1 is for Harmful
CLASS_LABELS = ["Benign", "Harmful"]


def main(minor_version: int, patch_version: int):
    """Create and save the WildChat dataset.

    We filter for single-turn benign conversations.
    """
    ds_dicts = prepare_wildchat_dataset()
    # Upload section
    dataset_uploader = DatasetUploadHandler(
        ds_repo_name=DATASET_REPO_NAME,
        ds_dicts=ds_dicts,
        minor_version=minor_version,
        patch_version=patch_version,
    )
    dataset_uploader.push_to_hub_and_create_tag()


def prepare_wildchat_dataset() -> dict:
    ds = datasets.load_dataset("allenai/WildChat", split="train[:50000]")
    assert isinstance(ds, datasets.Dataset)
    INSTRUCTIONS = ""
    ANSWER_PROMPT = ""
    CLASS_LABELS = ["Benign", "Harmful"]
    ds = ds.filter(lambda x: x["turn"] == 1)
    ds = ds.filter(lambda x: not x["toxic"])
    ds = ds.filter(lambda x: not x["redacted"])
    ds = ds.filter(lambda x: x["conversation"][0]["content"] != "")

    # Create a new column using add_column instead of direct assignment
    ds = ds.add_column("clf_label", [0] * len(ds))  # type: ignore
    label_feature = datasets.ClassLabel(names=CLASS_LABELS)
    ds = cast_column_to_feature(ds=ds, column_name="clf_label", feature=label_feature)

    ds = ds.map(
        lambda x: {
            "instructions": INSTRUCTIONS,
            "content": [x["conversation"][0]["content"]],
            "completion": x["conversation"][1]["content"],
            "answer_prompt": ANSWER_PROMPT,
            "proxy_clf_label": 1 - x["clf_label"],
            "gen_target": gen_target_from_label(x["clf_label"], label_feature),
            "proxy_gen_target": gen_target_from_label(
                1 - x["clf_label"], label_feature
            ),
        },
        remove_columns=[
            "conversation",
            "conversation_id",
            "model",
            "timestamp",
            "turn",
            "language",
            "openai_moderation",
            "detoxify_moderation",
            "toxic",
            "redacted",
        ],
    )
    ds = filter_dataset_for_context_length(ds)
    # WildChat has no validation partition or positive (harmful) examples
    empty_ds = ds.select(range(0))
    ds_dict = datasets.DatasetDict({"train": ds, "validation": empty_ds})
    empty_ds_dict = datasets.DatasetDict({"train": empty_ds, "validation": empty_ds})
    return {"default": ds_dict, "pos": empty_ds_dict, "neg": ds_dict}


if __name__ == "__main__":
    # bump the version here manually when you make changes
    # (see README for more info)
    MINOR_VERSION = 0
    PATCH_VERSION = 1
    main(minor_version=MINOR_VERSION, patch_version=PATCH_VERSION)
