import numpy as np
import pytest
import torch

from robust_llm.stat_utils import (
    compute_auc,
    compute_roc_curve,
    dedup_by_fpr,
    sort_by_fpr_then_tpr,
)


@pytest.fixture
def perfect_predictions():
    """Perfect classifier predictions"""
    return torch.tensor([1.0, 1.0, 1.0, 0.0, 0.0]), torch.tensor([1, 1, 1, 0, 0])


@pytest.fixture
def random_predictions():
    """Random classifier predictions"""
    torch.manual_seed(42)  # for reproducibility
    probs = torch.rand(100)
    labels = torch.randint(0, 2, (100,))
    return probs, labels


@pytest.fixture
def worst_predictions():
    """Worst possible predictions (reversed labels)"""
    return torch.tensor([0.0, 0.0, 0.0, 1.0, 1.0]), torch.tensor([1, 1, 1, 0, 0])


def test_perfect_classifier(perfect_predictions):
    probs, labels = perfect_predictions
    fprs, tprs, _ = compute_roc_curve(probs, labels)
    auc = compute_auc(fprs, tprs)

    # Perfect classifier should have points at (0,1), and (1,1)
    assert len(fprs) == len(tprs) == 2
    assert fprs[0] == 0 and tprs[0] == 1  # 1 threshold is perfect
    assert fprs[1] == 1 and tprs[1] == 1  # 0 threshold always predicts positive
    assert abs(auc - 1.0) < 1e-6  # AUC should be 1.0


def test_random_classifier(random_predictions):
    probs, labels = random_predictions
    fprs, tprs, _ = compute_roc_curve(probs, labels)
    auc = compute_auc(fprs, tprs)

    # Random classifier should have AUC around 0.5
    assert 0.4 <= auc <= 0.6
    # Points should roughly follow y=x line
    assert all(abs(fpr - tpr) < 0.2 for fpr, tpr in zip(fprs, tprs))


def test_worst_classifier(worst_predictions):
    probs, labels = worst_predictions
    fprs, tprs, _ = compute_roc_curve(probs, labels)
    auc = compute_auc(fprs, tprs)

    # Worst classifier should have AUC close to 0
    assert auc < 0.1


def test_empty_input():
    probs = torch.tensor([])
    labels = torch.tensor([])
    with pytest.raises(AssertionError):
        compute_roc_curve(probs, labels)


def test_single_class():
    """Test handling of single-class edge cases"""
    # All positive
    probs_pos = torch.tensor([0.7, 0.3, 0.9])
    labels_pos = torch.tensor([1, 1, 1])
    fprs_pos, tprs_pos, _ = compute_roc_curve(probs_pos, labels_pos)

    # All negative
    probs_neg = torch.tensor([0.7, 0.3, 0.9])
    labels_neg = torch.tensor([0, 0, 0])
    fprs_neg, tprs_neg, _ = compute_roc_curve(probs_neg, labels_neg)

    # Should handle these cases without division by zero
    assert not np.isnan(fprs_pos).any()
    assert not np.isnan(tprs_pos).any()
    assert not np.isnan(fprs_neg).any()
    assert not np.isnan(tprs_neg).any()


def test_probability_bounds():
    """Test handling of probabilities outside [0,1]"""
    probs = torch.tensor([-0.5, 0.3, 1.5])
    labels = torch.tensor([0, 1, 1])

    with pytest.raises(AssertionError):
        compute_roc_curve(probs, labels)


def test_non_monotonic_example():
    """Test that ROC curve is monotonically increasing"""
    probs = torch.tensor([0.1, 0.4, 0.35, 0.8])
    labels = torch.tensor([0, 0, 1, 1])
    fprs, tprs, _ = compute_roc_curve(probs, labels)

    assert fprs.tolist() == [0.0, 0.5, 1.0]
    assert tprs.tolist() == [0.5, 1.0, 1.0]


def test_auc_bounds():
    """Test AUC values for various curves"""
    # Triangle above diagonal (AUC > 0.5)
    fprs_good = np.array([0.0, 0.0, 1.0])
    tprs_good = np.array([0.0, 1.0, 1.0])
    auc_good = compute_auc(fprs_good, tprs_good)
    assert auc_good == 1.0

    # Triangle below diagonal (AUC < 0.5)
    fprs_bad = np.array([0.0, 1.0, 1.0])
    tprs_bad = np.array([0.0, 0.0, 1.0])
    auc_bad = compute_auc(fprs_bad, tprs_bad)
    assert auc_bad == 0.0

    # Diagonal line (AUC = 0.5)
    fprs_mid = np.array([0.0, 1.0])
    tprs_mid = np.array([0.0, 1.0])
    auc_mid = compute_auc(fprs_mid, tprs_mid)
    assert abs(auc_mid - 0.5) < 1e-6


def test_sort_by_fpr_then_tpr():
    thresholds = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    fprs = np.array([0.1, 0.2, 0.2, 0.3, 0.4])
    tprs = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    random_order = np.random.permutation(len(thresholds))
    shuffled_fprs = fprs[random_order]
    shuffled_tprs = tprs[random_order]
    shuffled_thresholds = thresholds[random_order]
    sorted_fprs, sorted_tprs, sorted_thresholds = sort_by_fpr_then_tpr(
        shuffled_fprs, shuffled_tprs, shuffled_thresholds
    )
    assert np.all(sorted_fprs == fprs)
    assert np.all(sorted_tprs == tprs)
    assert np.all(sorted_thresholds == thresholds)


def test_dedup_by_fpr():
    thresholds = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    fprs = np.array([0.1, 0.2, 0.2, 0.3, 0.4])
    tprs = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    random_order = np.random.permutation(len(thresholds))
    shuffled_fprs = fprs[random_order]
    shuffled_tprs = tprs[random_order]
    shuffled_thresholds = thresholds[random_order]
    deduped_fprs, deduped_tprs, deduped_thresholds = dedup_by_fpr(
        shuffled_fprs, shuffled_tprs, shuffled_thresholds
    )
    assert deduped_fprs.tolist() == [0.1, 0.2, 0.3, 0.4]
    assert deduped_tprs.tolist() == [0.1, 0.3, 0.4, 0.5]
    assert deduped_thresholds.tolist() == [0.1, 0.3, 0.4, 0.5]
