import warnings
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING


class MessageFilter(Enum):
    """Enum for filtering messages from the dataset

    Enum Values:
    -----------
    INPUT: Extracts just the user's prompt/input text.

    OUTPUT: Extracts just the assistant's response text with no chat formatting.

    TRANSCRIPT: Shows both user
        prompt and assistant response. Uses a standardized chat format with
        UUIDs to prevent prompt injection attacks that could confuse the
        model about the structure of the chat.

    IDENTITY: Passes all messages through, and applies the chat format of the
    victim tokenizer.

    When to Use Each Filter:
    ----------------------
    For input filters:
        Use INPUT to show the user prompt with no chat formatting

    For output filters:
        Use OUTPUT to show the assistant response with no chat formatting

    For transcript filters:
        Use TRANSCRIPT to show both prompt and response with universal chat
        formatting (uses UUIDs to prevent confusion attacks)

    For input probes:
        Use INPUT (with training.probe_enabled) to show the user prompt
        with chat formatting if needed

    For output probes:
        Use IDENTITY (with training.probe_enabled) to show prompt and
        response in the victim model's required chat format

    For generative models:
        Use INPUT (with model.inference_type=generation) to show the
        prompt with appropriate chat formatting and generation tokens
    """

    INPUT = "input"
    OUTPUT = "output"
    TRANSCRIPT = "transcript"
    IDENTITY = "identity"


@dataclass
class DatasetConfig:
    """Config used for dataset setup.

    Attributes:
        dataset_type: Type of dataset to use.
        n_train: Number of training examples.
        n_val: Number of validation examples.
        config_name: config_name from hf datasets (if applicable).
        revision: The huggingface revision to use. Use "<1.1.0" for
            compabitibility versions that match those used in the past, e.g.
            classification results in the workshop paper. Use "main" for the latest
            version.
        check_revision: Whether to check that the revision name is formatted
            correctly. Almost always should be true unless debugging.
        inference_type: The type of inference performed ("classification"
            or "generation")
        classification_as_generation: Whether we are doing classification
            using a generation model, in which case the 'gen_target' column
            represents the classification target.
        gen_target_override: A string to use as the gen_target
            everywhere, rather than the one given in the dataset.
        strip_leading_whitespace: Whether to strip leading whitespace
            from the gen_target.  This is necessary for some chat models where a
            leading space is undesirable because the chat template contains a
            newline. Defaults to True since most of our models are chat models.
        message_filter: Whether to use the user prompts, assistant responses or both
            from the dataset.
        replace_content_with_message_filter: Whether to replace the content of the
            dataset with the column specified by the message filter. This is a bit of
            a hack at the moment to make it possible to adversarially attack the
            output filter.
        validation_split: The split to use for validation.
        leave_unused_columns: Whether to leave the unused columns in the dataset.
            Defaults to False.
        proxy_gen_target_override: (Ignore; this is for backwards compatibility)
    """

    dataset_type: str = MISSING
    n_train: int = 0
    n_val: int = 0
    config_name: Optional[str] = None
    revision: str = MISSING
    check_revision: bool = True
    inference_type: str = "generation"
    classification_as_generation: bool = True
    gen_target_override: Optional[str] = None
    strip_leading_whitespace: bool = True
    message_filter: MessageFilter = MessageFilter.INPUT
    replace_content_with_message_filter: bool = False
    validation_split: str = "validation"
    leave_unused_columns: bool = False
    proxy_gen_target_override: str | None = None

    def __post_init__(self):
        if self.proxy_gen_target_override is not None:
            warnings.warn(
                "proxy_gen_target_override is deprecated."
                " Use a static_transformation instead."
            )


@dataclass
class ContactInfoDatasetConfig(DatasetConfig):
    """Config used for the ContactInfo dataset.

    Attributes:
        dataset_type: dataset type, always "ContactInfo".
        inference_type: The type of inference performed,
            always "generation"
        classification_as_generation: Whether we are doing classification
            using a generation model, which is never the case for this
            generative dataset.
        info_type:
            The type of information to generate, e.g. phone_number.
        revision: Unused for this dataset, set to a default value of "main".
    """

    dataset_type: str = "ContactInfo"
    inference_type: str = "generation"
    classification_as_generation: bool = False
    info_type: Optional[str] = None
    revision: str = "main"


cs = ConfigStore.instance()
cs.store(group="dataset", name="CONTACT_INFO", node=ContactInfoDatasetConfig)
