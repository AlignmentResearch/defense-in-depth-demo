"""Downloads datasets from HF to /robust_llm_data/datasets."""

import logging
from pathlib import Path

import datasets
from huggingface_hub import HfApi
from tqdm import tqdm

from robust_llm.dist_utils import dist_rmtree
from robust_llm.file_utils import get_shared_data_dir

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

datasets.utils.logging.disable_progress_bar()


def download_one_dataset_revision(
    base_save_path: Path,
    dataset_name: str,
    revision: str,
    subset: str = "default",
) -> bool:
    """Downloads a specific revision of a dataset, returns False if download fails.

    Args:
        base_save_path: Base directory to save datasets
        dataset_name: Name of the dataset on HuggingFace
        revision: Git revision/tag
        subset: Subset of the dataset to download, e.g., "pos" or "default" (not
            the split).

    Raises:
        KeyboardInterrupt: If download was interrupted by keyboard interrupt.
    """
    dataset_dir = base_save_path / dataset_name / revision / subset
    dataset_str = f"{dataset_name}:{revision}:{subset}"
    if dataset_dir.exists():
        logger.info(f"Dataset {dataset_str} already exists, skipping")
        return True
    dataset_dir.mkdir(parents=True)

    try:
        splits = datasets.get_dataset_split_names(dataset_name, revision=revision)
        for split in splits:
            split_destination = dataset_dir / split
            logger.info(f"Downloading {dataset_str} to {split_destination}")
            try:
                dataset = datasets.load_dataset(
                    dataset_name,
                    split=split,
                    revision=revision,
                    name=subset,
                )
            except ValueError as e:
                # Some evaluation datasets like StrongREJECT have an empty train
                # split.
                if str(e) == 'Instruction "train" corresponds to no data!':
                    continue
                raise

            assert isinstance(dataset, datasets.Dataset)
            dataset.save_to_disk(split_destination)
        return True

    except Exception as e:
        logger.error(f"Failed to download {dataset_str}: {str(e)}")
        dist_rmtree(dataset_dir)
        return False

    except KeyboardInterrupt:
        dist_rmtree(dataset_dir)
        raise


def download_dataset_revisions(dataset_name: str, base_save_path: Path) -> bool:
    """Downloads all revisions of a dataset, returns False if any download fails.

    Args:
        dataset_name: Name of the dataset on HuggingFace
        base_save_path: Base directory to save datasets

    Raises:
        ValueError: If no revisions are found for the dataset.
    """
    try:
        api = HfApi()
        refs = api.list_repo_refs(dataset_name, repo_type="dataset")
    except Exception as e:
        logger.error(f"Error getting revisions for {dataset_name}: {str(e)}")
        return False

    revisions = [ref.name for ref in refs.tags]
    if not revisions:
        raise ValueError(f"No revisions found for dataset: {dataset_name}")

    logger.info(f"Found {len(revisions)} revisions for {dataset_name}: {revisions}")
    return all(
        download_one_dataset_revision(base_save_path, dataset_name, revision)
        for revision in revisions
    )


def main() -> None:
    SHARED_DATASET_DIR = Path(get_shared_data_dir()) / "datasets"
    DATASETS = [
        "AlignmentResearch/EnronSpam",
        "AlignmentResearch/Harmless",
        "AlignmentResearch/Helpful",
        "AlignmentResearch/IMDB",
        "AlignmentResearch/Llama3Jailbreaks",
        "AlignmentResearch/PasswordMatch",
        "AlignmentResearch/WordLength",
        "AlignmentResearch/XSTest",
    ]

    success = all(
        download_dataset_revisions(dataset_name, SHARED_DATASET_DIR)
        for dataset_name in tqdm(DATASETS, desc="Downloading datasets")
    )
    if not success:
        exit(1)


if __name__ == "__main__":
    main()
