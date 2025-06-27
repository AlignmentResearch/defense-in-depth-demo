import semver
from datasets import Dataset, DatasetDict

from robust_llm.rllm_datasets.dataset_utils import (
    ALLOWED_COLUMNS,
    REQUIRED_COLUMNS,
    create_rllm_tag,
    maybe_abort_for_larger_version,
    valid_tag,
    version_exists,
)


class DatasetUploadHandler:
    """Defines the format we expect for the datasets we upload to the hub.

    This class is used to validate the format of the datasets we upload to the
    hub. When this class is changed, we bump the *major* version of the
    datasets we upload. The minor and patch versions are handled by the
    generation scripts for the specific datasets.

    We tie the major version to the version of the DatasetUploadHandler, since
    this defines the required fields, and so the overall format. We can still
    load different major versions of datasets without changing this, but in
    practice if we're changing the format we'll probably want to regenerate the
    datasets to match.

    Args:
        ds_repo_name: The name of the repo on the hub where the dataset is
            stored. This should start with 'AlignmentResearch'.
        config_name: The 'config_name' of the dataset. Used to designate a specific
            version of the dataset. Only set this if this is a special version
            of the dataset, otherwise 'default' is fine.
        ds_dicts: A mapping between 'config_name's and huggingface
            'DatasetDict's to push. The config_name is used to designate a specific
            version of the dataset. At least one 'config_name' must be 'default'.
            Each ds_dict should should have splits named "train" and "validation".
            The 'config_name's must be unique within the ds_dict.
        minor_version: The minor version of the dataset.
        patch_version: The patch version of the dataset.
    """

    MAJOR_VERSION = 2

    def __init__(
        self,
        ds_repo_name: str,
        ds_dicts: dict[str, DatasetDict],
        minor_version: int,
        patch_version: int,
    ):
        self.ds_repo_name = ds_repo_name
        self.ds_dicts = ds_dicts
        self.minor_version = minor_version
        self.patch_version = patch_version

        version_string = (
            f"{self.MAJOR_VERSION}.{self.minor_version}.{self.patch_version}"
        )
        if not valid_tag(version_string):
            raise ValueError(
                f"Tag must be in semver format, got {version_string}"
                " (see semver.org for more format information)."
            )
        self.version = semver.Version.parse(version_string)

        self._validate()

    def push_to_hub_and_create_tag(self):
        """Push the dataset to the hub and create a tag for the version."""
        for config_name, ds_dict in self.ds_dicts.items():
            ds_dict.push_to_hub(self.ds_repo_name, config_name=config_name)
        create_rllm_tag(self.ds_repo_name, self.version)

    def _validate(self):
        """Run checks against the dataset to be uploaded.

        Conditions:
        - Each 'config_name' must be unique within the ds_dict being uploaded.
        - At least one 'config_name' must be 'default'.
        - Dataset must have splits 'train' and 'validation'.
        - Dataset must have columns:
            - instructions
            - content
            - answer_prompt
            - clf_label
            - gen_target
        - 'content' must be a non-trivial list of strings.
        - Dataset must have at least one example.
        - Dataset repo name must start with 'AlignmentResearch'.
        - If the config_name is "pos", the dataset must only have examples with
            clf_label=1.
        - If the config_name is "neg", the dataset must only have examples with
            clf_label=0.
        - Minor version must be >= 0.
        - Patch version must be >= 0.
        - Version must not already exist on the hub.

        Non-conditions that should maybe be added in the future:
        - Dataset is filtered for context length.
        - Dataset is shuffled.
        """
        # Validate config names
        if len(set(self.ds_dicts.keys())) != len(self.ds_dicts):
            raise ValueError("Each 'config_name' must be unique.")
        if "default" not in self.ds_dicts:
            raise ValueError("At least one 'config_name' must be 'default'.")

        # validate dataset format
        for config_name, ds_dict in self.ds_dicts.items():
            self._validate_ds_dict(config_name, ds_dict)

        # validate version and naming
        if not self.ds_repo_name.startswith("AlignmentResearch"):
            raise ValueError("Dataset repo name must start with 'AlignmentResearch'.")
        if self.minor_version < 0:
            raise ValueError("Minor version must be >= 0.")
        if self.patch_version < 0:
            raise ValueError("Patch version must be >= 0.")
        if version_exists(self.ds_repo_name, self.version):
            raise ValueError(
                f"Version {self.version} of {self.ds_repo_name} already exists."
            )

        maybe_abort_for_larger_version(
            repo_name=self.ds_repo_name, version=self.version
        )

    def _validate_ds_dict(self, config_name: str, ds_dict: DatasetDict):
        """Run checks against the dataset format."""
        if not isinstance(ds_dict, DatasetDict):
            raise ValueError("All values in 'ds_dicts' must be 'DatasetDict's.")
        if set(ds_dict.keys()) != {"train", "validation"}:
            raise ValueError(
                "Each 'DatasetDict' must have splits 'train' and 'validation'."
            )
        for config_name, ds in ds_dict.items():
            self._validate_ds_split(config_name, ds)

    def _validate_ds_split(self, config_name: str, ds: Dataset):
        """Run checks on each individual dataset split.

        This is the part that's most liable to change between versions.
        """
        assert REQUIRED_COLUMNS <= set(ds.column_names) <= ALLOWED_COLUMNS, (
            f"Dataset must have columns: {REQUIRED_COLUMNS}"
            f", and no columns other than: {ALLOWED_COLUMNS}, got: {ds.column_names}."
        )
        content = ds["content"]
        # Check that each 'content' c in the content column is a list of strings.
        if not all(
            isinstance(c, list)
            and all(isinstance(s, str) for s in c)
            and any(len(s) > 0 for s in c)
            for c in content
        ):
            raise ValueError("'content' must be a list of strings.")
        if config_name == "pos" and not all(label == 1 for label in ds["clf_label"]):
            raise ValueError("'pos' dataset must only have examples with clf_label=1.")
        if config_name == "neg" and not all(label == 0 for label in ds["clf_label"]):
            raise ValueError("'neg' dataset must only have examples with clf_label=0.")
