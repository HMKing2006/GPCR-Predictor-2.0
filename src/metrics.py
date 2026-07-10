"""Regression metrics for binding-affinity evaluation."""

from __future__ import annotations

import numpy as np


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute the mean absolute error.

    Args:
        y_true: Ground-truth pActivity values.
        y_pred: Predicted pActivity values.

    Returns:
        The mean absolute error.
    """
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute the root-mean-squared error.

    Args:
        y_true: Ground-truth pActivity values.
        y_pred: Predicted pActivity values.

    Returns:
        The root-mean-squared error.
    """
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute the coefficient of determination (R^2).

    Args:
        y_true: Ground-truth pActivity values.
        y_pred: Predicted pActivity values.

    Returns:
        The R^2 score. Returns ``0.0`` when the target has zero variance.
    """
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if ss_tot == 0.0:
        return 0.0
    return 1.0 - ss_res / ss_tot


def pearson(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute the Pearson linear correlation coefficient.

    Args:
        y_true: Ground-truth pActivity values.
        y_pred: Predicted pActivity values.

    Returns:
        Pearson ``r`` in ``[-1, 1]``. Returns ``0.0`` when fewer than two
        samples are present or either input has zero variance.
    """
    if y_true.shape[0] < 2:
        return 0.0
    if np.std(y_true) == 0.0 or np.std(y_pred) == 0.0:
        return 0.0
    r = np.corrcoef(y_true, y_pred)[0, 1]
    return float(r) if np.isfinite(r) else 0.0


def _rank(values: np.ndarray) -> np.ndarray:
    """Rank values, assigning tied entries their average rank.

    Args:
        values: A one-dimensional array of values to rank.

    Returns:
        A ``float64`` array of ranks (average-tie convention).
    """
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.shape[0], dtype=np.float64)
    ranks[order] = np.arange(1, values.shape[0] + 1, dtype=np.float64)
    sorted_vals = values[order]
    # Average the ranks of equal-valued runs so ties do not bias the correlation.
    start = 0
    for i in range(1, sorted_vals.shape[0] + 1):
        if i == sorted_vals.shape[0] or sorted_vals[i] != sorted_vals[start]:
            if i - start > 1:
                ranks[order[start:i]] = ranks[order[start:i]].mean()
            start = i
    return ranks


def spearman(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute the Spearman rank correlation coefficient.

    Args:
        y_true: Ground-truth pActivity values.
        y_pred: Predicted pActivity values.

    Returns:
        Spearman ``rho`` in ``[-1, 1]``. Returns ``0.0`` when fewer than two
        samples are present or either ranking has zero variance.
    """
    if y_true.shape[0] < 2:
        return 0.0
    return pearson(_rank(y_true), _rank(y_pred))


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute MAE, RMSE, R^2 and rank/linear correlations together.

    Args:
        y_true: Ground-truth pActivity values.
        y_pred: Predicted pActivity values.

    Returns:
        A mapping with keys ``"mae"``, ``"rmse"``, ``"r2"``, ``"pearson"`` and
        ``"spearman"``.
    """
    return {
        "mae": mae(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "r2": r2(y_true, y_pred),
        "pearson": pearson(y_true, y_pred),
        "spearman": spearman(y_true, y_pred),
    }


def print_metrics(y_true: np.ndarray, y_pred: np.ndarray, label: str = "") -> dict[str, float]:
    """Compute metrics and print them in a compact one-line summary.

    Args:
        y_true: Ground-truth pActivity values.
        y_pred: Predicted pActivity values.
        label: Optional prefix identifying the evaluation (e.g. ``"test"``).

    Returns:
        The computed metric mapping (see :func:`compute_metrics`).
    """
    scores = compute_metrics(y_true, y_pred)
    prefix = f"{label} " if label else ""
    print(
        f"{prefix}MAE={scores['mae']:.4f}  RMSE={scores['rmse']:.4f}  R2={scores['r2']:.4f}  "
        f"Pearson={scores['pearson']:.4f}  Spearman={scores['spearman']:.4f}",
        flush=True,
    )
    return scores
