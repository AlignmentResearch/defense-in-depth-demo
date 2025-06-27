"""Converts IMDB dataset to use the new column format.

Takes the original IMDB dataset with 'text' and 'chunked_text'
columns and creates a new dataset with 'content', 'instructions', and
'answer_prompt' columns that contains the same exact data.
"""

import datasets
from datasets import Dataset

from robust_llm.rllm_datasets.dataset_utils import cast_column_to_feature
from robust_llm.rllm_datasets.generation_scripts.compatibility_versions.compatibility_utils import (  # noqa: E501
    convert_and_upload,
)


def imdb_old_to_new(old_dataset: Dataset) -> Dataset:
    """Convert the old IMDB dataset to the new format.

    IMDB was a single, PERTURBABLE chunk:
    - chunked_text[0] = the example data (PERTURBABLE).

    To make this fit the new format and ModifiableChunkSpec, we will have:
    - instructions = "" (IMMUTABLE).
    - content[0] = chunked_text[0] (PERTURBABLE).
    - answer_prompt = "" (IMMUTABLE).

    So the content is just the chunked_text, and instructions/answer_prompt are empty.

    And for the gen_target, we will just stringify the clf_label.
    """
    instructions = [""] * len(old_dataset)
    content = old_dataset["chunked_text"]
    answer_prompt = [""] * len(old_dataset)

    clf_label = old_dataset["clf_label"]
    proxy_clf_label = [1 - x for x in clf_label]
    gen_target = [str(x) for x in clf_label]
    proxy_gen_target = [str(x) for x in proxy_clf_label]

    new_dataset = Dataset.from_dict(
        {
            "instructions": instructions,
            "content": content,
            "answer_prompt": answer_prompt,
            "clf_label": clf_label,
            "proxy_clf_label": proxy_clf_label,
            "gen_target": gen_target,
            "proxy_gen_target": proxy_gen_target,
        }
    )
    # Add ClassLabel feature to the clf_label column
    label_feature = datasets.ClassLabel(names=["0", "1"])
    new_dataset = cast_column_to_feature(
        ds=new_dataset,
        column_name="clf_label",
        feature=label_feature,
    )
    return new_dataset


if __name__ == "__main__":
    MINOR_VERSION = 0
    PATCH_VERSION = 0
    repo_name = "AlignmentResearch/IMDB"
    convert_and_upload(
        repo_name=repo_name,
        new_minor_version=MINOR_VERSION,
        new_patch_version=PATCH_VERSION,
        converter_func=imdb_old_to_new,
    )
