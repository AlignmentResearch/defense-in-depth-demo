from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence
from typing import Any, Type, TypeVar, overload

import datasets
from accelerate import Accelerator
from datasets import Dataset
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from robust_llm import logger
from robust_llm.config.dataset_configs import DatasetConfig
from robust_llm.config.dataset_configs import MessageFilter
from robust_llm.dist_utils import DistributedRNG
from robust_llm.file_utils import get_shared_data_dir
from robust_llm.message_utils import MessageList
from robust_llm.models.model_utils import InferenceType
from robust_llm.rllm_datasets.dataset_utils import (
    ALLOWED_COLUMNS,
    REQUIRED_COLUMNS,
    cast_column_to_feature,
    check_revision_is_supported,
    construct_text_and_chunked_text,
    example_to_message_list,
    maybe_get_version,
    strip_leading_whitespace,
    tokenize_dataset,
)
from robust_llm.rllm_datasets.modifiable_chunk_spec import ModifiableChunkSpec

# TypeVar for the return type of methods that return a new RLLMDataset.
D = TypeVar("D", bound="RLLMDataset")


class RLLMDataset(ABC):
    """A class representing a dataset for robust LLM experiments.

    Mostly a wrapper around huggingface datasets, with some additional metadata.

    Attributes:
        ds: The underlying huggingface dataset object.
            Currently, we assume that a dataset has at least the following columns:
            - text: The input text.
            - chunked_text: The input text, chunked into modifiable and
            unmodifiable parts, matching 'modifiable_chunk_spec'.
            - clf_label: The classification label.
        num_classes: The number of classes in the dataset.
        modifiable_chunk_spec: Datasets consist of sections that can and cannot
            be modified in various ways. For example, we might want to leave the
            instructions intact while allowing part of the example content to be
            perturbed and the rest overwritten. modifiable_chunk_spec is a
            tuple of ChunkType, an enum that specifies whether each chunk is
            IMMUTABLE, PERTURBABLE, or OVERWRITABLE.
        is_tokenized: Whether the dataset has been tokenized, i.e., whether it
            has 'input_ids' and 'attention_mask' columns.
    """

    _registry: dict[str, Type[RLLMDataset]] = {}

    @classmethod
    def register_dataset(cls, name: str):
        """Register a subclass of RLLMDataset.

        Example usage:
        @RLLMDataset.register_dataset("MyDataset")
        class MyDataset(RLLMDataset):
            pass
        """

        def decorator(subclass):
            cls._registry[name] = subclass
            return subclass

        return decorator

    @classmethod
    def get_dataset(cls, name: str) -> Type[RLLMDataset]:
        if name not in cls._registry:
            raise ValueError(
                f"Dataset {name} not supported.\n"
                f"Supported datasets: {sorted(cls._registry.keys())}"
            )
        return cls._registry[name]

    def __init__(
        self,
        dataset_config: DatasetConfig,
        split: str,
        tokenizer: PreTrainedTokenizerBase | None = None,
        system_prompt: str | None = None,
    ) -> None:
        """Initialize an RLLMDataset.

        Args:
            dataset_config: Specifies most properties of the dataset.
            split: The split of the dataset to use (e.g., "train" or "validation").
            tokenizer: The tokenizer to use for the dataset. If None, the dataset will
                not be tokenized.
            system_prompt: The system prompt to use when tokenizing the dataset.
        """
        assert split in ("train", "validation")
        assert dataset_config.revision is not None
        if dataset_config.check_revision:
            check_revision_is_supported(
                dataset_config.dataset_type, dataset_config.revision
            )
        self.split = split
        self.tokenizer = tokenizer
        self.message_filter = dataset_config.message_filter
        self.dataset_type = dataset_config.dataset_type
        self.version = (
            maybe_get_version(dataset_config.dataset_type, dataset_config.revision)
            if dataset_config.check_revision
            else dataset_config.revision
        )
        self.inference_type = InferenceType(dataset_config.inference_type)
        self.classification_as_generation = dataset_config.classification_as_generation
        ds = self._load_dataset(
            cfg=dataset_config,
            revision=self.version,
            split=split,
            tokenizer=tokenizer,
            system_prompt=system_prompt,
        )
        # Make sure the dataset has the expected columns
        assert {"text", "chunked_text", "clf_label", "gen_target"}.issubset(
            set(ds.column_names)
        )
        # filter out rows where the text is empty
        # https://github.com/AlignmentResearch/robust-llm/issues/662
        original_len = len(ds)
        ds = ds.filter(lambda x: len(x["text"]) > 0)
        if len(ds) < original_len:
            logger.debug(f"Filtered out {original_len - len(ds)} empty rows")
        self.ds = ds

    @property
    @abstractmethod
    def num_classes(self) -> int:
        """Specifies how many classes there are in the dataset"""
        raise NotImplementedError("num_classes must be implemented")

    @property
    @abstractmethod
    def modifiable_chunk_spec(self) -> ModifiableChunkSpec:
        """Specifies which parts of the dataset can be modified, and how."""

    @property
    def is_tokenized(self) -> bool:
        if "input_ids" not in self.ds.column_names:
            return False
        assert "input_ids" in self.ds.column_names
        assert "attention_mask" in self.ds.column_names
        assert self.tokenizer is not None
        return True

    def _load_dataset(
        self,
        cfg: DatasetConfig,
        split: str,
        revision: str,
        tokenizer: PreTrainedTokenizerBase | None,
        system_prompt: str | None,
    ) -> Dataset:
        """Load the dataset and maybe tokenize it."""
        untokenized_ds = self._load_untokenized_dataset(cfg, split, revision)
        if tokenizer is None:
            return untokenized_ds
        else:
            tokenized_ds = tokenize_dataset(
                untokenized_ds,
                tokenizer,
                message_filter=self.message_filter,
                system_prompt=system_prompt,
            )
            return tokenized_ds

    def _load_untokenized_dataset(
        self, cfg: DatasetConfig, split: str, revision: str
    ) -> Dataset:
        """Load the untokenized dataset from huggingface.

        This first loads the raw dataset and then post-processes it.
        """

        dataset = self._load_raw_dataset(cfg, split, revision)
        actual_columns = set(dataset.column_names)

        assert (
            REQUIRED_COLUMNS <= actual_columns
        ), f"Required columns {REQUIRED_COLUMNS}, got {actual_columns}"
        extra_columns = list(actual_columns - ALLOWED_COLUMNS)
        if cfg.replace_content_with_message_filter:
            if self.message_filter == MessageFilter.OUTPUT:
                dataset = dataset.map(lambda x: {"content": [x["completion"]]})
            else:
                raise NotImplementedError(
                    f"{cfg.replace_content_with_message_filter = } with "
                    f"{cfg.message_filter = } is not supported yet."
                )
        if extra_columns:
            if cfg.leave_unused_columns:
                logger.warning(
                    f"Leaving extra columns {extra_columns} in dataset "
                    f"{cfg.dataset_type} revision {revision}"
                )
            else:
                # To be cautious, let's remove all extra columns.
                logger.warning(
                    f"Removing extra columns {extra_columns} from dataset "
                    f"{cfg.dataset_type} revision {revision}"
                )
                dataset = dataset.remove_columns(list(actual_columns - ALLOWED_COLUMNS))
        dataset = self._post_process_dataset(dataset, cfg)
        return dataset

    def _post_process_dataset(self, ds: Dataset, cfg: DatasetConfig) -> Dataset:
        """Post-process the dataset after loading it.

        Currently this involves
        - constructing 'text', and 'chunked_text' columns
            out of the 'instructions', 'content', and 'answer_prompt' columns.
        - Optionally stripping leading whitespace from the 'gen_target' column.
        - Optionally overriding the 'gen_target' column with a specified string.
        """
        assert isinstance(cfg.strip_leading_whitespace, bool)
        ds = construct_text_and_chunked_text(ds)

        # If there isn't already an 'original_text' column, add one based on the
        # 'text' column. This is used to track the original text for judging
        # even if we partially modify the 'text' column with e.g.
        # StaticTransformation.
        if "original_text" not in ds.column_names:
            ds = ds.add_column(
                "original_text",
                ds["text"],
                new_fingerprint=None,  # type: ignore  # (bug in datasets)
            )
        if cfg.gen_target_override is not None:
            assert cfg.inference_type == "generation"
            ds = ds.map(lambda x: {"gen_target": cfg.gen_target_override})
        if cfg.strip_leading_whitespace:
            ds = strip_leading_whitespace(ds)

        return ds

    def _load_raw_dataset(
        self, cfg: DatasetConfig, split: str, revision: str
    ) -> Dataset:
        """Load the raw dataset from huggingface.

        This is used to load the dataset without post-processing the columns.
        """
        if split == "train":
            if cfg.n_train == 0:
                raise ValueError(
                    "Cannot load train split when DatasetConfig.n_train is 0"
                )
            return self._load_split(
                cfg, split, n_examples=cfg.n_train, revision=revision
            )

        elif split == "validation":
            if cfg.n_val == 0:
                raise ValueError(
                    "Cannot load validation split when DatasetConfig.n_val is 0"
                )
            return self._load_split(
                cfg, cfg.validation_split, n_examples=cfg.n_val, revision=revision
            )
        else:
            raise ValueError(f"Unknown split {split}")

    def _load_split(
        self, cfg: DatasetConfig, split: str, n_examples: int, revision: str
    ) -> Dataset:
        """Load a split of the dataset with a given number of examples.

        Args:
            cfg: The DatasetConfig specifying the dataset to load.
            split: The split of the dataset to load.
            n_examples: The number of examples to load.
            revision: The revision of the dataset to load.

        Returns:
            The loaded dataset split.
        """
        try:
            ds = datasets.load_dataset(
                path=cfg.dataset_type,
                name=cfg.config_name,
                revision=revision,
                # We use slice splits to load a subset of the dataset.
                # https://huggingface.co/docs/datasets/en/loading#slice-splits
                split=f"{split}[:{n_examples}]",
                # We set 'reuse_cache_if_exists' to reuse the downloaded file from
                # HFHub for unit tests, but *not* reuse cached dataset operations.
                # Ideally we'd use 'force_redownload' to make sure every run happens
                # under the same conditions (no cache), but this is a compromise for
                # unit test speed.
                download_mode="reuse_cache_if_exists",
            )
        except Exception as e:
            logger.error(f"Error loading dataset {cfg.dataset_type}: {str(e)}")
            logger.info("Loading dataset from disk.")
            ds = datasets.load_from_disk(
                f"/{get_shared_data_dir()}/datasets/{cfg.dataset_type}/{revision}"
                f"/{cfg.config_name or 'default'}/{split}"
            )
            assert isinstance(ds, Dataset)
            ds = ds.select(range(n_examples))
        assert isinstance(ds, Dataset)
        if len(ds) == 0:
            raise ValueError(
                f"Split {split} of dataset {cfg.dataset_type} has no examples."
                " Are you sure that this dataset is supposed to have this split?"
            )
        return ds

    def get_random_subset(
        self: D,
        n: int,
        seed: int | None = None,
        accelerator: Accelerator | None = None,
        generator: DistributedRNG | None = None,
    ) -> D:
        """Return an RLLMDataset with a random subset of the original dataset."""
        assert (seed is None) != (
            generator is None
        ), "Exactly one of {seed, generator} must be provided"
        # When using multiple GPUs, we want to choose the same subset across
        # processes, so we use RNG from the main process.
        if seed is not None:
            generator = DistributedRNG(seed, accelerator=accelerator)
        assert generator is not None
        indices = generator.choice(len(self.ds), n, replace=False)
        return self.get_subset(indices)

    def get_subset(self: D, indices: Iterable[Any]) -> D:
        """Return an RLLMDataset with a subset of the original dataset.

        Iterable[Any] typing is to match Dataset.select
        """
        new_ds = self.ds.select(indices)
        return self.with_new_ds(new_ds)

    def with_attacked_text(self: D, attacked_text: Sequence[str]) -> D:
        """Returns a new RLLMDataset with the attacked text and attacked labels."""
        if "attacked_text" in self.ds.column_names:
            raise ValueError("Dataset already has attacked_text column")

        if len(attacked_text) != len(self.ds):
            raise ValueError("attacked_text must have the same length as the dataset")

        new_ds = self.ds.add_column(
            "attacked_text",
            attacked_text,
            new_fingerprint=None,  # type: ignore  # (bug in datasets?)
        )
        # Add initial 'attacked_' columns
        new_ds = new_ds.add_column(
            "attacked_clf_label",
            self.ds["clf_label"],
            new_fingerprint=None,  # type: ignore  # (bug in datasets?)
        )
        new_ds = new_ds.add_column(
            "attacked_gen_target",
            self.ds["gen_target"],
            new_fingerprint=None,  # type: ignore  # (bug in datasets?)
        )

        # Add proxy columns.
        new_ds = new_ds.add_column(
            "attacked_proxy_clf_label",
            self.ds["proxy_clf_label"],
            new_fingerprint=None,  # type: ignore  # (bug in datasets?)
        )
        new_ds = new_ds.add_column(
            "attacked_proxy_gen_target",
            self.ds["proxy_gen_target"],
            new_fingerprint=None,  # type: ignore  # (bug in datasets?)
        )

        # Make 'attacked_clf_label' have the same Feature as 'clf_label'
        new_ds = cast_column_to_feature(
            ds=new_ds,
            column_name="attacked_clf_label",
            feature=self.ds.features["clf_label"],
        )
        attacked_ds = self.with_new_ds(new_ds)
        return attacked_ds

    @overload
    def clf_label_to_gen_target(self, clf_label: int) -> str: ...

    @overload
    def clf_label_to_gen_target(self, clf_label: list[int]) -> list[str]: ...

    def clf_label_to_gen_target(
        self,
        clf_label: int | list[int],
    ) -> str | list[str]:
        """Convert one or more clf_label to gen_target.

        Uses the int2str method of the clf_label feature to convert the label(s).
        """
        feature = self.ds.features["clf_label"]
        assert isinstance(feature, datasets.ClassLabel)
        gen_target = feature.int2str(clf_label)

        assert isinstance(gen_target, str) or isinstance(gen_target, list)
        return gen_target

    @overload
    def gen_target_to_clf_label(self, gen_target: str) -> int: ...

    @overload
    def gen_target_to_clf_label(self, gen_target: list[str]) -> list[int]: ...

    def gen_target_to_clf_label(
        self,
        gen_target: str | list[str],
    ) -> int | list[int]:
        """Convert one or more clf_label to gen_target.

        Uses the str2int method of the clf_label feature to convert the label(s).
        """
        feature = self.ds.features["clf_label"]
        assert isinstance(feature, datasets.ClassLabel)
        clf_label = feature.str2int(gen_target)

        assert isinstance(clf_label, int) or isinstance(clf_label, list)
        return clf_label

    def with_new_ds(self: D, new_ds: Dataset) -> D:
        """Make a new RLLMDataset with the given huggingface dataset."""
        # We make a shallow copy of the RLLMDataset and replace the dataset.
        # This should be fine because the ds is the only attribute that might be
        # mutated and we're replacing it with a reference to a new object, but
        # if other mutable attributes are added to RLLMDataset then this may
        # need to be changed.
        new_dataset = copy.copy(self)
        new_dataset.ds = new_ds
        return new_dataset

    def tokenize(
        self: D,
        tokenizer: PreTrainedTokenizerBase,
        message_filter: MessageFilter,
        system_prompt: str | None,
    ) -> D:
        """Return a tokenized version of the dataset"""
        if self.is_tokenized:
            raise ValueError("Dataset is already tokenized")
        tokenized_ds = tokenize_dataset(
            self.ds,
            tokenizer,
            message_filter=message_filter,
            system_prompt=system_prompt,
        )
        new_dataset = self.with_new_ds(tokenized_ds)
        new_dataset.tokenizer = tokenizer
        return new_dataset

    def for_training(self) -> Dataset:
        """Returns a datasets.Dataset that is suitable for training.

        This involves:
        - Ensuring the dataset is tokenized
        - Renaming the `clf_label` column to `label`
        - Reducing to a minimal set of columns to avoid column mismatches
            when concatenating datasets (e.g. for adversarial training).

        Returns:
            A datasets.Dataset with columns 'text', 'input_ids', 'attention_mask',
            and 'label'.
        """
        assert self.is_tokenized
        assert self.tokenizer is not None
        if self.inference_type == InferenceType.CLASSIFICATION:
            ds_for_training = self.ds.rename_column("clf_label", "label")
            training_cols = ["text", "input_ids", "attention_mask", "label"]
            unused_cols = [
                c for c in ds_for_training.column_names if c not in training_cols
            ]
            ds_for_training = ds_for_training.remove_columns(unused_cols)
            return ds_for_training.with_format("torch")

        elif self.inference_type == InferenceType.GENERATION:
            # For generation, we tokenize the gen_target and stick it on the end
            # of the input_ids/attention_mask columns.
            ds_for_training = self.ds.remove_columns(["clf_label"])
            tokenized_gen_target = self.tokenizer(self.ds["gen_target"])
            all_target_ids = tokenized_gen_target["input_ids"]
            target_masks = tokenized_gen_target["attention_mask"]
            assert isinstance(all_target_ids, list)
            assert isinstance(target_masks, list)

            input_ids = [
                prompt_ids + target_ids
                for prompt_ids, target_ids in zip(self.ds["input_ids"], all_target_ids)
            ]

            attention_mask = [
                prompt_mask + target_mask
                for prompt_mask, target_mask in zip(
                    self.ds["attention_mask"],
                    target_masks,
                )
            ]

            # Remove and readd columns to update them
            ds_for_training = ds_for_training.remove_columns(
                ["input_ids", "attention_mask"]
            )
            ds_for_training = ds_for_training.add_column(
                "input_ids",
                input_ids,
                new_fingerprint=None,  # type: ignore  # (bug in datasets)
            )
            ds_for_training = ds_for_training.add_column(
                "attention_mask",
                attention_mask,
                new_fingerprint=None,  # type: ignore  # (bug in datasets)
            )

            training_cols = ["input_ids", "attention_mask"]
            unused_cols = [
                c for c in ds_for_training.column_names if c not in training_cols
            ]
            ds_for_training = ds_for_training.remove_columns(unused_cols)
            return ds_for_training

        else:
            raise ValueError(f"Unsupported inference type: {self.inference_type}")

    def as_adversarial_examples(self: D, system_prompt: str | None) -> D:
        """Formats attacked dataset as adversarial examples.

        This method is necessary because we need to take an attacked dataset
        (one with `attacked_text` and `attacked_clf_label` columns) and convert
        it into something that is compatible with the training data (i.e., with
        `text` and `clf_label` columns).

        To clarify:
        - Takes a tokenized RLLMDataset (self) that has `attacked_text` and
            `attacked_clf_label` columns.
        - Returns a tokenized RLLMDataset where `attacked_text` has been renamed
            to 'text' and `attacked_clf_label` has been renamed to `clf_label`.
        - NOTE: We don't need 'chunked_text' because we don't further attack
            these examples, we only use them for training.

        """
        if "attacked_text" not in self.ds.column_names:
            raise ValueError("Dataset does not have attacked_text column")
        if not self.is_tokenized:
            raise ValueError("Dataset is not tokenized")

        columns_to_keep = ["attacked_text", "attacked_clf_label", "attacked_gen_target"]
        columns_to_drop = [c for c in self.ds.column_names if c not in columns_to_keep]

        untokenized_new_ds = self.ds.map(remove_columns=columns_to_drop)
        untokenized_new_ds = (
            untokenized_new_ds.rename_column("attacked_text", "text")
            .rename_column("attacked_clf_label", "clf_label")
            .rename_column("attacked_gen_target", "gen_target")
        )

        assert self.tokenizer is not None
        new_ds = tokenize_dataset(
            untokenized_new_ds,
            self.tokenizer,
            message_filter=self.message_filter,
            system_prompt=system_prompt,
        )
        adversarial_dataset = self.with_new_ds(new_ds)

        return adversarial_dataset

    def __len__(self) -> int:
        return len(self.ds)

    def as_message_lists(
        self,
        system_prompt: str | None = None,
        user_column: str = "text",
        assistant_column: str = "completion",
    ) -> list[MessageList]:
        """Convert a column to a list of MessageLists."""
        return [
            example_to_message_list(
                row,  # type: ignore
                message_filter=self.message_filter,
                system_prompt=system_prompt,
                user_column=user_column,
                assistant_column=assistant_column,
            )
            for row in self.ds
        ]
