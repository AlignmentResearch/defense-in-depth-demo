import numpy as np
import torch
from numpy.typing import NDArray

from robust_llm.utils import to_numpy


def sort_by_fpr_then_tpr(
    fprs: np.ndarray, tprs: np.ndarray, thresholds: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    fpr_tpr_sort_indices = np.lexsort((tprs, fprs))
    fprs = fprs[fpr_tpr_sort_indices]
    tprs = tprs[fpr_tpr_sort_indices]
    thresholds = thresholds[fpr_tpr_sort_indices]
    return fprs, tprs, thresholds


def dedup_by_fpr(
    fprs: np.ndarray, tprs: np.ndarray, thresholds: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    fprs, tprs, thresholds = sort_by_fpr_then_tpr(fprs, tprs, thresholds)

    # Walk backwards through the FPRs, dropping duplicates and
    # keeping the last entry among duplicates.
    fpr_drop_mask = np.concatenate((np.diff(fprs[::-1])[::-1] == 0, [False]))
    fprs = fprs[~fpr_drop_mask]
    tprs = tprs[~fpr_drop_mask]
    thresholds = thresholds[~fpr_drop_mask]
    return fprs, tprs, thresholds


def compute_roc_curve(
    probs: NDArray[np.float64] | torch.Tensor,
    labels: NDArray[np.int64] | torch.Tensor,
    max_thresholds: int = 100,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """
    Compute ROC curve points.

    Args:
        probs: Predicted probabilities [batch_size]
        labels: True binary labels [batch_size]
        max_thresholds: Maximum number of thresholds to compute

    Returns:
        tuple of (fprs, tprs, thresholds) tensors containing ROC curve points
    """
    if isinstance(probs, torch.Tensor):
        probs = to_numpy(probs)
    if isinstance(labels, torch.Tensor):
        labels = to_numpy(labels)

    assert len(probs) == len(labels), "Number of predictions and labels must match"
    assert len(probs) > 0, "Empty input"
    assert probs.min() >= 0 and probs.max() <= 1, "Probabilities must be in [0, 1]"

    # Sort by probability
    sorted_indices = np.argsort(probs)
    sorted_probs = probs[sorted_indices]

    # Get unique probabilities for thresholds
    unique_probs = np.unique(sorted_probs)

    # Add boundary thresholds if needed
    if unique_probs[0] > 0:
        unique_probs = np.concatenate([[0.0], unique_probs])
    if unique_probs[-1] < 1:
        unique_probs = np.concatenate([unique_probs, [1.0]])

    # Limit number of thresholds
    if len(unique_probs) > max_thresholds:
        unique_probs = unique_probs[:: len(unique_probs) // max_thresholds]

    n_thresholds = len(unique_probs)
    fprs = np.zeros(n_thresholds)
    tprs = np.zeros(n_thresholds)

    n_pos = (labels == 1).sum()
    n_neg = (labels == 0).sum()

    # Calculate TPR and FPR for each threshold
    for i, threshold in enumerate(unique_probs):
        predictions = (probs >= threshold).astype(int)
        tp = ((predictions == 1) & (labels == 1)).sum()
        fp = ((predictions == 1) & (labels == 0)).sum()

        tprs[i] = tp / n_pos if n_pos > 0 else 0
        fprs[i] = fp / n_neg if n_neg > 0 else 0

    fprs, tprs, thresholds = dedup_by_fpr(fprs, tprs, unique_probs)

    return fprs, tprs, thresholds


def compute_auc(fprs: np.ndarray, tprs: np.ndarray) -> float:
    """
    Compute Area Under Curve. Handles case where false positive rate is always 0.

    Args:
        fprs: False positive rates
        tprs: True positive rates

    Returns:
        AUC score as float
    """
    return np.trapz(tprs, fprs)
