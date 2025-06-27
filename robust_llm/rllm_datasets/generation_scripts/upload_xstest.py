"""Script to generate the Llama3Jailbreaks dataset"""

import datasets

from robust_llm.rllm_datasets.dataset_utils import (
    cast_column_to_feature,
    gen_target_from_label,
    make_pos_neg_versions,
)
from robust_llm.rllm_datasets.generation_scripts.dataset_upload_handler import (
    DatasetUploadHandler,
)

DATASET_REPO_NAME = "AlignmentResearch/XSTest"
# 0 is for Benign, 1 is for Harmful
CLASS_LABELS = ["Benign", "Harmful"]


def main(minor_version: int, patch_version: int):
    """Create and save the Llama3Jailbreaks dataset.

    We use the generic prep_huggingface_dataset function to prepare the dataset,
    which assumes it's binary clf and that the columns text and label exist.
    """
    ds_dicts = prepare_xstest_dataset()
    # Upload section
    dataset_uploader = DatasetUploadHandler(
        ds_repo_name=DATASET_REPO_NAME,
        ds_dicts=ds_dicts,
        minor_version=minor_version,
        patch_version=patch_version,
    )
    dataset_uploader.push_to_hub_and_create_tag()


def prepare_xstest_dataset() -> dict:
    ds = datasets.load_dataset("walledai/XSTest", split="test")
    assert isinstance(ds, datasets.Dataset)
    INSTRUCTIONS = ""
    ANSWER_PROMPT = ""
    CLASS_LABELS = ["Benign", "Harmful"]
    # Convert 'safe'/'unsafe' labels to 0/1
    ds = ds.map(lambda x: {"label": 0 if x["label"] == "safe" else 1})
    ds = ds.rename_column("label", "clf_label")
    label_feature = datasets.ClassLabel(names=CLASS_LABELS)
    ds = cast_column_to_feature(ds=ds, column_name="clf_label", feature=label_feature)

    ds = ds.map(
        lambda x: {
            "instructions": INSTRUCTIONS,
            "content": [x["prompt"]],
            "answer_prompt": ANSWER_PROMPT,
            "proxy_clf_label": 1 - x["clf_label"],
            "gen_target": gen_target_from_label(x["clf_label"], label_feature),
            "proxy_gen_target": gen_target_from_label(
                1 - x["clf_label"], label_feature
            ),
        },
        remove_columns=["prompt", "focus", "type", "note"],
    )
    # XSTest has no train partition, so we set this to an empty dataset
    train_ds = ds.select(range(0))
    ds_dict = datasets.DatasetDict({"validation": ds})
    pos_ds_dict, neg_ds_dict = make_pos_neg_versions(ds_dict)
    ds_dict["train"] = train_ds
    pos_ds_dict["train"] = train_ds
    neg_ds_dict["train"] = train_ds
    return {"default": ds_dict, "pos": pos_ds_dict, "neg": neg_ds_dict}


if __name__ == "__main__":
    # bump the version here manually when you make changes
    # (see README for more info)
    MINOR_VERSION = 0
    PATCH_VERSION = 0
    main(minor_version=MINOR_VERSION, patch_version=PATCH_VERSION)
