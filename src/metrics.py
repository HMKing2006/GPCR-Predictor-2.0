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


def _format_breakdown_line(
    name: str,
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> str:
    """Format one breakout metrics line, or ``n/a`` when undefined.

    Args:
        name: Bucket label.
        y_true: Labels for the bucket.
        y_prob: Probabilities for the bucket.

    Returns:
        A single printable summary line.
    """
    n = int(y_true.shape[0])
    if n == 0:
        return f"{name}: n=0  n/a"
    active_frac = float(np.mean(y_true))
    if len(np.unique(y_true)) < 2:
        return (
            f"{name}: n={n}  active_frac={active_frac:.3f}  "
            f"n/a (single class)"
        )
    scores = compute_metrics(y_true, y_prob)
    return (
        f"{name}: n={n}  active_frac={active_frac:.3f}  "
        f"AUROC={scores['auroc']:.4f}  AUPRC={scores['auprc']:.4f}  "
        f"F1={scores['f1']:.4f}"
    )


def print_breakdowns(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    protein_known: np.ndarray,
    scaffold_known: np.ndarray,
    label: str = "",
) -> dict[str, dict[str, float]]:
    """Print overall and known/unknown protein and scaffold breakout metrics.

    Args:
        y_true: Ground-truth binary labels for the eval set.
        y_prob: Predicted positive-class probabilities.
        protein_known: Boolean mask; ``True`` when the protein was in train.
        scaffold_known: Boolean mask; ``True`` when the scaffold was in train.
        label: Optional prefix identifying the evaluation set.

    Returns:
        Nested mapping of bucket name to metric dictionaries (empty dict when
        a bucket is ``n/a``).
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    protein_known = np.asarray(protein_known, dtype=bool)
    scaffold_known = np.asarray(scaffold_known, dtype=bool)
    prefix = f"{label} " if label else ""
    print(f"\n{prefix}novelty breakouts:", flush=True)

    buckets: dict[str, np.ndarray] = {
        "overall": np.ones(y_true.shape[0], dtype=bool),
        "protein_known": protein_known,
        "protein_unknown": ~protein_known,
        "scaffold_known": scaffold_known,
        "scaffold_unknown": ~scaffold_known,
        "prot_known_scaff_known": protein_known & scaffold_known,
        "prot_known_scaff_unknown": protein_known & ~scaffold_known,
        "prot_unknown_scaff_known": ~protein_known & scaffold_known,
        "prot_unknown_scaff_unknown": ~protein_known & ~scaffold_known,
    }

    results: dict[str, dict[str, float]] = {}
    for name, mask in buckets.items():
        line = _format_breakdown_line(f"  {name}", y_true[mask], y_prob[mask])
        print(line, flush=True)
        if mask.any() and len(np.unique(y_true[mask])) >= 2:
            results[name] = compute_metrics(y_true[mask], y_prob[mask])
        else:
            results[name] = {}
    return results
