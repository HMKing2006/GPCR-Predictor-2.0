"""Classification metrics for 50 nM binder / non-binder evaluation."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def _binary_labels(y_prob: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Threshold predicted probabilities into hard class labels.

    Args:
        y_prob: Predicted positive-class probabilities.
        threshold: Decision threshold applied to ``y_prob``.

    Returns:
        Integer array of predicted labels in ``{0, 1}``.
    """
    return (np.asarray(y_prob) >= threshold).astype(np.int32)


def accuracy(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Compute accuracy at a 0.5 probability threshold.

    Args:
        y_true: Ground-truth binary labels.
        y_prob: Predicted positive-class probabilities.

    Returns:
        Classification accuracy.
    """
    return float(accuracy_score(y_true, _binary_labels(y_prob)))


def precision(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Compute precision of the positive (binder) class.

    Args:
        y_true: Ground-truth binary labels.
        y_prob: Predicted positive-class probabilities.

    Returns:
        Precision, or ``0.0`` when there are no predicted positives.
    """
    return float(precision_score(y_true, _binary_labels(y_prob), zero_division=0))


def recall(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Compute recall of the positive (binder) class.

    Args:
        y_true: Ground-truth binary labels.
        y_prob: Predicted positive-class probabilities.

    Returns:
        Recall, or ``0.0`` when there are no true positives.
    """
    return float(recall_score(y_true, _binary_labels(y_prob), zero_division=0))


def f1(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Compute the F1 score of the positive (binder) class.

    Args:
        y_true: Ground-truth binary labels.
        y_prob: Predicted positive-class probabilities.

    Returns:
        F1 score, or ``0.0`` when undefined.
    """
    return float(f1_score(y_true, _binary_labels(y_prob), zero_division=0))


def auroc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Compute the area under the ROC curve.

    Args:
        y_true: Ground-truth binary labels.
        y_prob: Predicted positive-class probabilities.

    Returns:
        AUROC in ``[0, 1]``. Returns ``0.5`` when only one class is present.
    """
    if len(np.unique(y_true)) < 2:
        return 0.5
    return float(roc_auc_score(y_true, y_prob))


def auprc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Compute the area under the precision-recall curve.

    Args:
        y_true: Ground-truth binary labels.
        y_prob: Predicted positive-class probabilities.

    Returns:
        Average precision. Returns the positive-class prevalence when only one
        class is present.
    """
    if len(np.unique(y_true)) < 2:
        return float(np.mean(y_true))
    return float(average_precision_score(y_true, y_prob))


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    """Compute accuracy, precision, recall, F1, AUROC and AUPRC together.

    Args:
        y_true: Ground-truth binary labels.
        y_prob: Predicted positive-class probabilities.

    Returns:
        A mapping with keys ``"accuracy"``, ``"precision"``, ``"recall"``,
        ``"f1"``, ``"auroc"`` and ``"auprc"``.
    """
    return {
        "accuracy": accuracy(y_true, y_prob),
        "precision": precision(y_true, y_prob),
        "recall": recall(y_true, y_prob),
        "f1": f1(y_true, y_prob),
        "auroc": auroc(y_true, y_prob),
        "auprc": auprc(y_true, y_prob),
    }


def print_metrics(y_true: np.ndarray, y_prob: np.ndarray, label: str = "") -> dict[str, float]:
    """Compute metrics and print them in a compact one-line summary.

    Args:
        y_true: Ground-truth binary labels.
        y_prob: Predicted positive-class probabilities.
        label: Optional prefix identifying the evaluation (e.g. ``"test"``).

    Returns:
        The computed metric mapping (see :func:`compute_metrics`).
    """
    scores = compute_metrics(y_true, y_prob)
    prefix = f"{label} " if label else ""
    print(
        f"{prefix}Acc={scores['accuracy']:.4f}  Prec={scores['precision']:.4f}  "
        f"Rec={scores['recall']:.4f}  F1={scores['f1']:.4f}  "
        f"AUROC={scores['auroc']:.4f}  AUPRC={scores['auprc']:.4f}",
        flush=True,
    )
    return scores
