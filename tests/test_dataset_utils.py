import pytest
import semver
from datasets import Dataset

from robust_llm.config import DatasetConfig
from robust_llm.rllm_datasets.dataset_utils import (
    filter_empty_rows,
    get_largest_version_below,
    partition_dataset,
)
from robust_llm.rllm_datasets.load_rllm_dataset import load_rllm_dataset


def test_get_largest_version_below():
    repo_id = "AlignmentResearch/PasswordMatch"
    # Test that we can get the largest version below a given version.
    # We can only test versions that already exist, because otherwise
    # a new larger version could be created and the test would fail.
    largest_below_0_0_4 = get_largest_version_below(repo_id, "0.0.4")
    assert isinstance(largest_below_0_0_4, semver.Version)
    assert largest_below_0_0_4 == "0.0.2"
    assert get_largest_version_below(repo_id, "0.0.3") == "0.0.2"
    assert get_largest_version_below(repo_id, "0.0.0") is None
    assert get_largest_version_below(repo_id, "0.1.0") == "0.0.4"
    assert get_largest_version_below(repo_id, "1.0.0") == "0.1.0"


def test_loading_largest_version_below():
    repo_id = "AlignmentResearch/PasswordMatch"
    cfg = DatasetConfig(
        dataset_type=repo_id,
        n_train=5,
        n_val=5,
        revision="<2.1.1",
        inference_type="classification",
        strip_leading_whitespace=False,
    )
    dataset = load_rllm_dataset(cfg, split="validation")
    assert dataset.version == "2.1.0"


def test_failing_largest_version_below():
    repo_id = "AlignmentResearch/PasswordMatch"
    cfg = DatasetConfig(
        dataset_type=repo_id,
        n_train=5,
        n_val=5,
        revision="<0.0.0",
        inference_type="classification",
        strip_leading_whitespace=False,
    )
    with pytest.raises(ValueError) as value_error:
        _ = load_rllm_dataset(cfg, split="validation")
    assert "No versions found" in str(value_error.value)


def test_filter_empty_rows():
    dataset = Dataset.from_dict(
        {
            "content": [["a"], [""], ["b"], ["c"], ["", ""]],
        }
    )
    filtered_dataset = filter_empty_rows(dataset)
    assert filtered_dataset["content"] == [["a"], ["b"], ["c"]]


def test_partition_dataset():
    """Test that partition_dataset correctly partitions datasets."""
    # Test case 1: Even division (no remainder)
    even_dataset = Dataset.from_dict({"content": list(range(10))})
    partitions = partition_dataset(even_dataset, n_parts=5)

    assert len(partitions) == 5
    for i, part in enumerate(partitions):
        assert len(part) == 2
        assert part["content"] == list(range(i * 2, (i + 1) * 2))

    # Test case 2: Uneven division (with remainder)
    uneven_dataset = Dataset.from_dict({"content": list(range(11))})
    partitions = partition_dataset(uneven_dataset, n_parts=5)

    assert len(partitions) == 5
    # First partition should have 3 items (2 + 1 from remainder)
    assert len(partitions[0]) == 3
    assert partitions[0]["content"] == [0, 1, 2]
    # Remaining partitions should have 2 items each
    for i in range(1, 5):
        assert len(partitions[i]) == 2
        assert partitions[i]["content"] == list(range(2 * i + 1, 2 * i + 3))

    # Test case 3: n_parts = 1
    single_part = partition_dataset(even_dataset, n_parts=1)
    assert len(single_part) == 1
    assert len(single_part[0]) == 10
    assert single_part[0]["content"] == list(range(10))

    # Test case 4: Remainder distribution
    dataset = Dataset.from_dict({"content": list(range(17))})
    partitions = partition_dataset(dataset, n_parts=5)

    assert len(partitions) == 5
    # Check sizes: should be [4, 4, 3, 3, 3]
    sizes = [len(part) for part in partitions]
    assert sizes == [4, 4, 3, 3, 3]
    # Check content
    assert partitions[0]["content"] == [0, 1, 2, 3]
    assert partitions[1]["content"] == [4, 5, 6, 7]
    assert partitions[2]["content"] == [8, 9, 10]
    assert partitions[3]["content"] == [11, 12, 13]
    assert partitions[4]["content"] == [14, 15, 16]


def test_partition_dataset_edge_cases():
    """Test edge cases for partition_dataset."""

    # n_parts > dataset size
    small_dataset = Dataset.from_dict({"content": [1, 2]})
    with pytest.raises(ValueError) as excinfo:
        partitions = partition_dataset(small_dataset, n_parts=5)
        assert "n_parts must be less than or equal to the size of the dataset" in str(
            excinfo.value
        )

    # Multiple columns
    multi_column = Dataset.from_dict(
        {"content": ["a", "b", "c", "d", "e"], "value": [1, 2, 3, 4, 5]}
    )
    partitions = partition_dataset(multi_column, n_parts=2)
    assert len(partitions) == 2
    assert len(partitions[0]) == 3
    assert len(partitions[1]) == 2
    assert partitions[0]["content"] == ["a", "b", "c"]
    assert partitions[0]["value"] == [1, 2, 3]
    assert partitions[1]["content"] == ["d", "e"]
    assert partitions[1]["value"] == [4, 5]


def test_partition_dataset_errors():
    """Test error cases for partition_dataset."""

    # n_parts = 0 should raise ValueError
    dataset = Dataset.from_dict({"content": list(range(10))})
    with pytest.raises(ValueError) as excinfo:
        partition_dataset(dataset, n_parts=0)
    assert "n_parts must be greater than 0" in str(excinfo.value)
