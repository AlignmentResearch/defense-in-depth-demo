"""Utilities used in the generation of all datasets."""

import sys
from collections.abc import Sequence
from dataclasses import dataclass, fields
from functools import partial
from typing import Any, Callable, Literal, Optional

import datasets
import huggingface_hub as hf_hub
import semver
from datasets import Dataset, DatasetDict
from datasets.combine import interleave_datasets
from huggingface_hub.errors import RepositoryNotFoundError
from transformers.models.auto.configuration_auto import AutoConfig
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.tokenization_utils_base import BatchEncoding, PreTrainedTokenizerBase

from robust_llm import logger
from robust_llm.config.dataset_configs import MessageFilter
from robust_llm.message_utils import Message, MessageList
from robust_llm.utils import ask_for_confirmation

DS_SHUFFLE_SEED = 0


def gen_target_from_label(label: int, label_feature: datasets.ClassLabel) -> str:
    gen_target = label_feature.int2str(label)
    assert isinstance(gen_target, str)
    return gen_target


@dataclass
class RLLMExample:
    """Represents a single example in a RLLMDataset."""

    instructions: str
    content: list[str]
    answer_prompt: str
    # clf_label and gen_target are the correct responses we want to evaluate
    # against.
    # proxy_clf_label and proxy_gen_target are the bad responses we want to
    # optimize towards.
    clf_label: int
    proxy_clf_label: int
    gen_target: str
    proxy_gen_target: str
    completion: str | None = None
    completion_clf_label: int | None = None
    # NOTE(ian): This is a bit of a hack to allow the judge to see the original
    # prompt when we are pre-processing the dataset with BoN.
    original_text: str | None = None
    attack_index: int | None = None
    original_example_index: int | None = None


# Get the fields of a dataclass: https://stackoverflow.com/a/66499324
ALLOWED_COLUMNS = {f.name for f in fields(RLLMExample)}
REQUIRED_COLUMNS = {
    f.name
    for f in fields(RLLMExample)
    if f.default is None and f.default_factory is None
}


# Contains one model from each family of models we care (or might
# care) about for use in filtering to appropriate context lengths
SUPPORTED_MODELS = [
    "stanford-crfm/alias-gpt2-small-x21",
    "EleutherAI/pythia-14m",
    # To use llama you need to request access on hf and export HF_TOKEN
    "meta-llama/Llama-2-7b-hf",
    "Qwen/Qwen2.5-0.5B",
]


def filter_dataset_length(ds: Dataset) -> Dataset:
    """Filter dataset for rows with length zero or greater than the context length.

    Args:
        ds: The dataset to filter.

    Returns:
        The filtered dataset.
    """
    return filter_dataset_for_context_length(filter_empty_rows(ds))


def filter_empty_rows(ds: Dataset) -> Dataset:
    """Filter out rows with no data in the 'content' column."""
    prev_len = len(ds)
    new_ds = ds.filter(lambda x: len("".join(x["content"])) > 0)
    if len(new_ds) < prev_len:
        logger.warning(f"Filtered out {prev_len - len(new_ds)} rows with no content")
    else:
        logger.debug("No empty rows found")
    return new_ds


def filter_dataset_for_context_length(dataset: Dataset, buffer: int = 24) -> Dataset:
    """Filter out rows that are too long for all supported models.

    Args:
        dataset: The dataset to filter.
        buffer: The number of additional tokens to remove to leave space
            for tokens added by attacks. The default of 24 is not special,
            it's just gives a reasonable amount of space for most attacks.


    Returns:
        The filtered dataset, i.e. the provided dataset *minus* all examples
        that do not fit in the context length of all supported models.
    """
    for model_name in SUPPORTED_MODELS:
        dataset = filter_length_for_model(
            dataset=dataset, model_name=model_name, buffer=buffer
        )
    return dataset


def filter_length_for_model(
    dataset: Dataset,
    model_name: str,
    buffer: int,
) -> Dataset:
    """Filter for examples that fit in the context length of the given model.

    Args:
        dataset: The dataset to filter.
        model_name: The model whose tokenizer and context length we are
            filtering for.
        buffer: The number of additional tokens to remove to leave space
            for tokens added by attacks.

    Returns:
        The dataset filtered to fit in the context length of the model.
    """
    context_length = _get_context_length(model_name)
    token_target = context_length - buffer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize(x):
        return tokenizer(x["text"])["input_ids"]

    # Add a default text column to the dataset to use for filtering.
    dataset = dataset.map(
        lambda x: {"text": example_dict_to_text(x)},
    )
    tokenized_dataset = dataset.map(
        lambda x: {"input_ids": tokenize(x)},
        batched=True,
    )
    filtered_dataset = tokenized_dataset.filter(
        lambda x: len(x["input_ids"]) <= token_target
    )
    # Remove the columns that were added for filtering.
    return filtered_dataset.remove_columns(["input_ids", "text"])


def example_dict_to_text(example: dict) -> str:
    """Default way to convert a dataset example to a single string."""
    return "".join(example_dict_to_chunked_text(example))


def example_dict_to_chunked_text(example: dict) -> list[str]:
    """Default way to convert a dataset example to a 'chunked_text' list.

    Note that we might want more complicated ways to do this for chat models
    that have different roles for different messages.
    """
    chunks = []
    if "instructions" in example:
        chunks.append(example["instructions"])
    chunks.extend(example["content"])
    if "answer_prompt" in example:
        chunks.append(example["answer_prompt"])
    return chunks


def construct_text_and_chunked_text(ds: Dataset) -> Dataset:
    """Construct a dataset with both text and chunked_text columns."""
    ds = ds.map(
        lambda x: {
            "text": example_dict_to_text(x),
            "chunked_text": example_dict_to_chunked_text(x),
        },
    )
    return ds


def _get_context_length(model_name: str) -> int:
    """Get the context length of a model.

    It turns out that the easiest way to get the context length is
    the max_position_embeddings attribute of the model config.
    """
    config = AutoConfig.from_pretrained(model_name)
    return config.max_position_embeddings


def create_rllm_tag(repo_id: str, tag: semver.Version):
    """Create a tag for an rllm dataset on the huggingface hub."""

    hf_hub.create_tag(repo_id=repo_id, tag=str(tag), repo_type="dataset")


def valid_tag(tag: str):
    """Check if a tag is a valid semver tag."""
    try:
        _ = semver.Version.parse(tag)
    except ValueError:
        return False
    return True


def version_exists(repo_id: str, tag: semver.Version):
    """Check if a version exists on the hub repo as a tag.

    Dataset versions are stored as tags on the huggingface repo for the
    datasets. This checks if, for a given version number of the dataset, there
    is already a tag with that version number.
    """
    # If we can't load the gitref, then either we don't have access or the repo
    # doesn't exist but we should continue anyway, and let the push fail if it's
    # a permissions issue
    try:
        existing_tags = _get_versions(repo_id)
    except ValueError:
        return False

    for existing_tag in existing_tags:
        if existing_tag == tag:
            return True
    return False


def get_largest_version(repo_id: str) -> semver.Version | None:
    """Get the largest version of a dataset on the hub."""
    try:
        existing_tags = _get_versions(repo_id)
    except ValueError:
        return None

    if len(existing_tags) == 0:
        return None

    largest_version = semver.Version.parse(existing_tags[0])
    for existing_tag in existing_tags:
        parsed_tag = semver.Version.parse(existing_tag)
        if parsed_tag > largest_version:
            largest_version = parsed_tag
    return largest_version


def get_largest_version_below(repo_id: str, version: str) -> semver.Version | None:
    """Get the largest version of a dataset strictly less than a given version.

    Args:
        repo_id: The id of the dataset repo on the hf hub.
        version: A valid semver version number as a string.

    Returns:
        The largest version of the dataset that is strictly less than the given
        version. If no such version exists, returns None.

    """
    try:
        existing_tags = _get_versions(repo_id)
    except ValueError:
        return None

    if len(existing_tags) == 0:
        return None

    parsed_version_upper_bound = semver.Version.parse(version)

    largest_version = None
    for existing_tag in existing_tags:
        parsed_tag = semver.Version.parse(existing_tag)

        if parsed_tag >= parsed_version_upper_bound:
            continue
        if largest_version is None or parsed_tag > largest_version:
            largest_version = parsed_tag

    return largest_version


def _get_versions(repo_id: str) -> list[str]:
    """Get the versions of a dataset on the hub.

    Args:
        repo_id: The id of the dataset repo on the hf hub.

    Returns:
        A list of the version (tag) names of the dataset.

    Raises:
        ValueError: If the repo does not exist.
    """
    try:
        gitref = hf_hub.list_repo_refs(repo_id, repo_type="dataset")
    except RepositoryNotFoundError:
        raise ValueError(
            f"Repo {repo_id} does not exist or you do not have permission to view it."
        )
    return [tag.name for tag in gitref.tags]


def maybe_abort_for_larger_version(repo_name: str, version: semver.Version):
    """Check if a larger version of a dataset already exists on the hub.

    The reason we check this is in case it's unintential that we're uploading
    a version that's smaller than the largest version on the hub. However, if
    we're e.g. fixing bugs in a dataset for older formats, we might want to create
    the dataset anyway.
    """
    largest_version = get_largest_version(repo_name)
    if (largest_version is not None) and (largest_version > version):
        should_continue = ask_for_confirmation(
            f"Larger version {largest_version} of {repo_name} already exists. Continue?"
        )
        if not should_continue:
            print("Aborting")
            sys.exit(1)


def maybe_get_version(dataset_type: str, revision: str) -> str:
    """Maybe process the version given into an actual version to use.

    If a valid semver version was specified, use that. Otherwise if
    'main' was specified, use the latest version. If the string starts with
    '<', use the latest version that is strictly less than the specified
    version.

    NOTE: We don't simply use 'main' because we want to record the version
    used and avoid race conditions that could arise from separately loading
    'main' and looking up the most recent version.

    Args:
        dataset_type: The name of the dataset on huggingface hub.
        revision: The revision to use. (e.g. 'main', '1.0.0', '<1.0.0')
    """
    version: str | semver.Version | None
    if revision.startswith("<"):
        version = get_largest_version_below(dataset_type, revision[1:])
    elif revision == "main":
        version = get_largest_version(dataset_type)
    elif valid_tag(revision):
        version = revision
    else:
        raise ValueError(
            f"Invalid revision: {revision}."
            " Should be 'main' or a valid semver version."
        )
    if version is None:
        raise ValueError(f"No versions found for revision {revision}")
    return str(version)


def extract_single_label(ds: Dataset, label: int):
    return ds.filter(lambda x: x["clf_label"] == label)


def make_pos_neg_versions(ds_dict: DatasetDict) -> tuple[DatasetDict, DatasetDict]:
    """Make versions of the dataset with only positive and only negative examples.

    Assumes that the dataset has a "clf_label" column with 1 for positive
    examples and 0 for negative examples, and that each split has at least
    one positive and one negative example.

    Args:
        ds_dict: The dataset to split into positive and negative examples.
    """
    # preconditions
    splits = list(ds_dict.keys())
    for split in splits:
        assert "clf_label" in ds_dict[split].column_names
        assert set(ds_dict[split]["clf_label"]) == {0, 1}

    pos_splits = dict()
    neg_splits = dict()
    for split in splits:
        ds_split = ds_dict[split]
        pos_split = extract_single_label(ds=ds_split, label=1)
        neg_split = extract_single_label(ds=ds_split, label=0)
        pos_splits[split] = pos_split
        neg_splits[split] = neg_split

        assert len(pos_split) > 0
        assert len(neg_split) > 0
        assert len(pos_split) + len(neg_split) == len(ds_split)

    pos_dict = DatasetDict(pos_splits)
    neg_dict = DatasetDict(neg_splits)
    return pos_dict, neg_dict


def _prepare_huggingface_dataset(
    train: Dataset,
    val: Dataset,
) -> dict[str, DatasetDict]:
    filtered_train = filter_dataset_length(train)
    filtered_val = filter_dataset_length(val)

    full_ds_dict = DatasetDict({"train": filtered_train, "validation": filtered_val})
    pos_ds_dict, neg_ds_dict = make_pos_neg_versions(full_ds_dict)
    out_dict = {
        "default": full_ds_dict,
        "pos": pos_ds_dict,
        "neg": neg_ds_dict,
    }
    return out_dict


def prepare_huggingface_dataset(
    repo_id: str,
    ds_specific_callback: Callable[[Dataset], Dataset],
    split_map: Optional[dict[str, str]] = None,
    **kwargs,
) -> dict[str, DatasetDict]:
    """Prepare a huggingface dataset for use in RLLMDatasets.

    Most huggingface datasets have a similar structure. This function
    attempts to apply all the relevant processing to a binary classification
    dataset to make it ready for use in RLLMDatasets.

    Args:
        repo_id: The id of the dataset on the hub.
        ds_specific_callback: A function specific to the dataset that processes
            it to have the necessary columns and structure for RLLMDatasets.
            Currently, this means adding 'instructions', 'content', 'answer_prompt',
            and 'gen_target' columns.
        split_map: A mapping from the split names in the dataset to the
            names we want to use in RLLMDatasets. If None, uses a
            default split map.
        kwargs: Additional keyword arguments to pass to the huggingface
            datasets.load_dataset function.

    Returns:
        A dictionary of (config_name: DatasetDict) pairs.
    """
    if split_map is None:
        split_map = {
            "train": "train",
            "validation": "test",
        }
    train, val = datasets.load_dataset(
        repo_id,
        split=[split_map["train"], split_map["validation"]],  # type: ignore
        **kwargs,
    )
    assert isinstance(train, Dataset)
    assert isinstance(val, Dataset)
    # make sure it has the necessary columns
    prepped_train = prep_hf_split(train)
    prepped_val = prep_hf_split(val)

    processed_train = ds_specific_callback(prepped_train)
    processed_val = ds_specific_callback(prepped_val)

    return _prepare_huggingface_dataset(processed_train, processed_val)


def ensure_datasets(
    splits: list[list[Dataset]], seed: int = DS_SHUFFLE_SEED
) -> list[Dataset]:
    return [
        (
            ds
            if isinstance(ds, Dataset)
            else datasets.concatenate_datasets(ds).shuffle(seed=seed)
        )
        for ds in splits
    ]


def concatenate_huggingface_datasets(
    split_map: dict[str, dict[str, str | Sequence[str]]],
    ds_specific_callback: Callable[[Dataset, int], Dataset],
    stopping_strategy: Literal["first_exhausted", "all_exhausted"] = "all_exhausted",
    seed: int = DS_SHUFFLE_SEED,
    probabilities: list[float] = [0.5, 0.5],
    **kwargs,
) -> dict[str, DatasetDict]:
    """Prepare a huggingface dataset which is already split pos/neg.

    Interleaves the positive/negative examples and then does not shuffle,
    i.e. does not call prep_hf_split.

    Args:
        split_map: A nested dictionary mapping repository IDs to dictionaries that map
            from keys (pos_train, neg_train, pos_val, neg_val) to the corresponding
            split names in each HF dataset. For example:
            {
                "repo_id_1": {
                    "pos_train": "train_split_1",
                    "neg_train": "train_split_2"
                },
                "repo_id_2": {
                    "pos_val": "val_split_1",
                    "neg_val": "val_split_2"
                }
            }
        ds_specific_callback: A function specific to the dataset that processes
            it to have the necessary columns and structure for RLLMDatasets.
            Currently, this means adding 'instructions', 'content', 'answer_prompt',
            and 'gen_target' columns. Also takes an integer representing the
            class label of the dataset.
        stopping_strategy: The strategy to use for stopping the interleaving.
            "first_exhausted" will stop when the first dataset is exhausted.
            "all_exhausted" will stop when all datasets are exhausted.
        seed: The seed to use for shuffling the datasets.
        probabilities: The probabilities of the positive and negative examples.
        kwargs: Additional keyword arguments to pass to the huggingface
            datasets.load_dataset function.

    Returns:
        A dictionary of (config_name: DatasetDict) pairs.
    """
    # Validate split_map
    required_keys = ["pos_train", "neg_train", "pos_val", "neg_val"]
    all_provided_keys = [key for repo_keys in split_map.values() for key in repo_keys]
    for key in required_keys:
        if key not in all_provided_keys:
            raise ValueError(f"Missing required key in split_map: {key}")

    # Load datasets from each repository
    loaded_splits: dict[str, list[Dataset]] = {}
    for repo_id, repo_splits in split_map.items():
        for split_key, hf_split in repo_splits.items():
            dataset = datasets.load_dataset(
                repo_id,
                split=hf_split,  # type: ignore
                **kwargs,
            )
            if isinstance(dataset, Dataset):
                dataset = [dataset]
            assert isinstance(dataset, list)
            loaded_splits[split_key] = loaded_splits.get(split_key, []) + dataset

    # Ensure all required splits are loaded
    for key in required_keys:
        if key not in loaded_splits:
            raise ValueError(f"Failed to load required split: {key}")

    # Process each split - ensure_datasets handles both Dataset and list[Dataset]
    splits = [
        loaded_splits["neg_train"],
        loaded_splits["pos_train"],
        loaded_splits["neg_val"],
        loaded_splits["pos_val"],
    ]
    splits = ensure_datasets(splits)

    assert len(splits) == 4
    assert all(isinstance(ds, Dataset) for ds in splits)
    neg_train, pos_train, neg_val, pos_val = [
        ds_specific_callback(ds, i % 2) for i, ds in enumerate(splits)  # type: ignore # noqa: E501
    ]
    assert isinstance(pos_train, Dataset)
    assert isinstance(neg_train, Dataset)
    assert all([label == 1 for label in pos_train["clf_label"]])
    assert all([label == 0 for label in neg_train["clf_label"]])
    train = interleave_datasets(
        [pos_train, neg_train],
        stopping_strategy=stopping_strategy,
        seed=seed,
        probabilities=probabilities,
    )

    assert isinstance(pos_val, Dataset)
    assert isinstance(neg_val, Dataset)
    assert all([label == 1 for label in pos_val["clf_label"]])
    assert all([label == 0 for label in neg_val["clf_label"]])
    val = interleave_datasets(
        [pos_val, neg_val],
        stopping_strategy=stopping_strategy,
        seed=seed,
        probabilities=probabilities,
    )

    return _prepare_huggingface_dataset(train, val)


def prep_hf_split(ds: Dataset) -> Dataset:
    """Process a huggingface dataset split for use in RLLMDatasets."""
    if "text" in ds.column_names and "label" in ds.column_names:
        num_classes = _get_num_classes(ds)
        assert num_classes == 2, (
            "This class can't automatically create an `RLLMDataset`-compatible dataset "
            " from huggingface datasets with more than two classes. You can still do it"
            " manually if you want."
        )
        assert set(ds["label"]) == {0, 1}, "labels must be 0 and 1"
        # Classification target should be called clf_label.
        ds = ds.rename_column("label", "clf_label")
        # Drop all columns except the ones we need.
        HF_EXPECTED_COLUMNS = ["text", "clf_label"]
        ds = ds.remove_columns(
            [col for col in ds.column_names if col not in HF_EXPECTED_COLUMNS]
        )
    else:
        # The only other dataset format is used by Anthropic's Helpfulness and
        # Harmlessness dataset, which has 'chosen' and 'rejected' columns.
        assert ds.column_names == ["chosen", "rejected"]
    # Shuffle deterministically.
    ds = ds.shuffle(seed=DS_SHUFFLE_SEED)
    return ds


def _get_num_classes(ds: Dataset) -> int:
    """Get the number of classes in a dataset."""
    try:
        return ds.features["label"].num_classes
    except AttributeError:
        logger.warning(
            "'label' column does not have num_classes attribute."
            " Using `set()` to get a lower bound on the number of classes."
        )
        return len(set(ds["label"]))


def example_to_message_list(
    row: dict[str, Any],
    message_filter: MessageFilter,
    system_prompt: str | None,
    user_column="text",
    assistant_column="completion",
) -> MessageList:
    """Convert an example to a list of messages."""
    message_list = MessageList(
        ([Message(role="system", content=system_prompt)] if system_prompt else [])
        + (
            [Message(role="user", content=row[user_column])]
            if user_column in row
            else []
        )
        + (
            [Message(role="assistant", content=row[assistant_column])]
            if assistant_column in row
            else []
        )
    )
    return message_list.filter(message_filter)


def prepare_tokenizer_input(
    row: dict[str, Any],
    index: int,
    tokenizer: PreTrainedTokenizerBase,
    message_filter: MessageFilter,
    system_prompt: str | None,
) -> dict[str, Any]:
    message_list = example_to_message_list(row, message_filter, system_prompt)
    formatted = message_list.format(message_filter, tokenizer, seed=index)

    return {"tokenizer_input": formatted}


def tokenizer_function(
    x: dict[str, Any],
    tokenizer: PreTrainedTokenizerBase,
    padding: str,
    truncation: bool,
    return_tensors: Optional[str],
) -> BatchEncoding:
    return tokenizer(
        x["tokenizer_input"],
        padding=padding,
        truncation=truncation,
        return_tensors=return_tensors,
        # We add special tokens in `prepare_tokenizer_input` if
        # `tokenizer.chat_template` is not None.
        add_special_tokens=tokenizer.chat_template is None,
    )


def tokenize_dataset(
    dataset: Dataset,
    tokenizer: PreTrainedTokenizerBase,
    message_filter: MessageFilter,
    system_prompt: str | None,
    padding: str = "do_not_pad",
    truncation: bool = True,
    return_tensors: Optional[str] = None,
) -> Dataset:
    # Explicitly passing in padding argument seems necessary to avoid an error
    # (even if it's 'do_not_pad').
    prepare_fn = partial(
        prepare_tokenizer_input,
        tokenizer=tokenizer,
        message_filter=message_filter,
        system_prompt=system_prompt,
    )
    tokenizer_fn = partial(
        tokenizer_function,
        tokenizer=tokenizer,
        padding=padding,
        truncation=truncation,
        return_tensors=return_tensors,
    )

    pretokenized_dataset = dataset.map(prepare_fn, with_indices=True)
    tokenized_dataset = pretokenized_dataset.map(tokenizer_fn, batched=True)
    return tokenized_dataset


def cast_column_to_feature(
    ds: Dataset,
    column_name: str,
    feature: datasets.ClassLabel | datasets.Value,
) -> Dataset:
    """Cast a column in a dataset to a given feature."""
    new_features = ds.features.copy()
    new_features[column_name] = feature
    return ds.cast(new_features)


def cast_and_concatenate(
    ds: datasets.Dataset, other_ds: datasets.Dataset
) -> datasets.Dataset:
    """Cast the second dataset to the features of the first dataset and concatenate.

    Args:
        ds: The first dataset, whose features will be used.
        other_ds: The second dataset, whose features will be updated.
    """
    if ds.features != other_ds.features:
        other_ds = cast_features_like(ds, other_ds)
    new_ds = datasets.concatenate_datasets([ds, other_ds])
    return new_ds


def cast_features_like(
    ds: datasets.Dataset, other_ds: datasets.Dataset
) -> datasets.Dataset:
    """Cast the second dataset to the features of the first dataset.

    Args:
        ds: The dataset with features we want to copy.
        other_ds: The dataset to cast to those features.
    """
    new_ds = other_ds.cast(ds.features)
    return new_ds


def strip_leading_whitespace(ds: Dataset) -> Dataset:
    """Strip leading whitespace from the 'gen_target' column of a dataset.

    Also updates the Feature of the 'clf_label' column to remove leading whitespace.
    """
    ds = ds.map(
        lambda x: {
            "gen_target": x["gen_target"].lstrip(),
            "proxy_gen_target": x["proxy_gen_target"].lstrip(),
        }
    )
    # Also update the feature of clf_label
    stripped_feature = datasets.ClassLabel(
        names=[name.lstrip() for name in ds.features["clf_label"].names]
    )
    ds = cast_column_to_feature(
        ds=ds,
        column_name="clf_label",
        feature=stripped_feature,
    )
    return ds


DEPRECATED_VERSIONS = {
    "AlignmentResearch/PasswordMatch": [
        "2.0.0",
    ],
    "AlignmentResearch/WordLength": [
        "2.1.0",
        "2.0.0",
    ],
    "AlignmentResearch/IMDB": [
        "2.0.0",
    ],
    "AlignmentResearch/EnronSpam": [
        "2.0.0",
    ],
}


def check_revision_is_supported(dataset_type: str, revision: str) -> None:
    """Checks if the dataset version is still supported."""
    version = maybe_get_version(dataset_type, revision)
    CURRENT_MAJOR_VERSION = 2
    is_old_major_version = int(version[0]) < CURRENT_MAJOR_VERSION
    is_dep_version = version in DEPRECATED_VERSIONS.get(dataset_type, [])
    if is_old_major_version or is_dep_version:
        raise ValueError(
            f"Version {version} of dataset {dataset_type} is no longer supported."
        )


def partition_dataset(dataset: Dataset, n_parts: int) -> list[Dataset]:
    """Partition a dataset into n roughly equal consecutive parts."""
    if n_parts <= 0:
        raise ValueError("n_parts must be greater than 0")
    if n_parts > len(dataset):
        raise ValueError(
            "n_parts must be less than or equal to the size of the dataset"
        )

    total_size = len(dataset)
    part_size = total_size // n_parts
    remainder = total_size % n_parts

    parts = []
    start_idx = 0

    for i in range(n_parts):
        # Add one extra item to some partitions for the remainder
        current_part_size = part_size + (1 if i < remainder else 0)
        end_idx = start_idx + current_part_size

        parts.append(dataset.select(range(start_idx, end_idx)))
        start_idx = end_idx

    return parts
