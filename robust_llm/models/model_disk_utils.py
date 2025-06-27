"""Utility functions for saving and loading models from disk.

These functions are in their own file so that we can document the design
decisions behind model saving in one place.
"""

import datetime
import re
import uuid
from pathlib import Path
from typing import Generator

from robust_llm import logger
from robust_llm.file_utils import get_current_git_commit_hash


def get_revision_path(models_path: Path, model_name: str, revision: str) -> Path:
    return models_path / model_name / revision


def get_now_timestamp_str() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d-%H%M%S")


def timestamp_str_to_datetime(timestamp_string: str) -> datetime.datetime:
    return datetime.datetime.strptime(timestamp_string, "%Y%m%d-%H%M%S").replace(
        tzinfo=datetime.timezone.utc
    )


def generate_model_save_path(models_path: Path, model_name: str, revision: str) -> Path:
    """Generates a path at which to save a model.

    Models are saved in a subdirectory with a random string appended to avoid
    overwriting existing versions of a model revision. An "existing version"
    could be:
    - A complete model left by an old version of the run, in which case
      overwriting might be fine, but we'll play it safe and keep the old version
      to have a complete history.
    - A model being written by a concurrent run. This is not expected behavior
      but could happen due to misconfiguration, so again we play it safe and do
      not overwrite.
    - A partially written model from an old crashed run, in which overwriting
      would be fine.

    Args:
        models_path: Directory used for model storage.
        model_name: Name of the model being saved.
        revision: Revision of the model being saved.
    """
    timestamp = get_now_timestamp_str()
    random_string = str(uuid.uuid4())[-12:]
    subdirectory = f"{timestamp}-{random_string}"
    return (
        get_revision_path(
            models_path=models_path, model_name=model_name, revision=revision
        )
        / subdirectory
    )


def mark_model_save_as_finished(model_save_directory: Path) -> None:
    """Marks a model save as finished.

    This function should be called after a model revision has been saved to
    disk. The purpose of this is to distinguish models that are still being
    saved or got interrupted during saving from models that are fully saved and
    usable. The function also stores some useful metadata like the current
    commit hash.

    Args:
        model_save_directory: Save directory of the model revision version.
    """
    current_commit = get_current_git_commit_hash()
    with open(model_save_directory / "commit.txt", "w") as f:
        f.write(current_commit)
    with open(model_save_directory / "disk-write-timestamp.txt", "w") as f:
        # Having the disk write timestamp lets us avoid uploading disk models that
        # are older than what is already on HF Hub.
        f.write(get_now_timestamp_str())

    with open(model_save_directory / ".gitignore", "a") as f:
        # Indicates that the model is fully saved.
        f.write("\ndone-saving")
    with open(model_save_directory / "done-saving", "w"):
        pass


def is_model_save_finished(model_save_directory: Path) -> bool:
    """Returns whether a model save is finished."""
    return (model_save_directory / "done-saving").exists()


def get_model_versions_by_recency(
    model_revision_path: Path,
) -> Generator[Path, None, None]:
    """Yields versions of a model ordered by most recent."""
    if not model_revision_path.is_dir():
        return

    subdirectories = sorted(model_revision_path.iterdir(), reverse=True)
    for subdirectory in subdirectories:
        if not subdirectory.is_dir():
            continue
        if not re.match(r"\d{8}-\d{6}-[0-9a-f]{12}", subdirectory.name):
            # Subdirectory doesn't match the format we expect from
            # generate_model_save_path().
            logger.warning(
                "Directory name doesn't match expected format: %s", subdirectory
            )
            continue
        if not is_model_save_finished(subdirectory):
            continue
        yield subdirectory


def get_model_load_path(
    models_path: Path, model_name: str, revision: str
) -> Path | None:
    """Returns the path to the most recent version of the model.

    Args:
        models_path: Directory used for model storage.
        model_name: Name of the model being loaded.
        revision: Revision of the model being loaded.

    Returns:
        Path, or None if no model is found.
    """
    revision_path = get_revision_path(
        models_path=models_path, model_name=model_name, revision=revision
    )
    try:
        return next(get_model_versions_by_recency(revision_path))
    except StopIteration:
        logger.info(f"No model found at {revision_path}")
        return None
