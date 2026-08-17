"""Metrics for multi-head potency-range classification."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from src.range_model.bins import BIN_LABELS, N_BINS
from src.range_model.config import HEAD_NAMES


def ordinal_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean absolute error on ordered bin indices.

    Args:
        y_true: Ground-truth bin indices.
        y_pred: Predicted bin indices.

    Returns:
        Mean ``|pred - true|``, or ``nan`` when empty.
    """
    if y_true.size == 0:
        return float("nan")
    return float(np.mean(np.abs(y_true.astype(np.float64) - y_pred.astype(np.float64))))


def compute_head_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    n_bins: int = N_BINS,
) -> dict[str, Any]:
    """Compute classification metrics for one assay head.

    Args:
        y_true: Ground-truth bins.
        y_pred: Predicted bins.
        n_bins: Number of ordered classes.

    Returns:
        Mapping with accuracy, macro_f1, ordinal_mae, support, and confusion.
    """
    true = np.asarray(y_true).reshape(-1)
    pred = np.asarray(y_pred).reshape(-1)
    if true.size == 0:
        return {
            "n": 0,
            "accuracy": float("nan"),
            "macro_f1": float("nan"),
            "ordinal_mae": float("nan"),
            "confusion": np.zeros((n_bins, n_bins), dtype=np.int64),
        }
    labels = list(range(n_bins))
    return {
        "n": int(true.size),
        "accuracy": float(accuracy_score(true, pred)),
        "macro_f1": float(
            f1_score(true, pred, labels=labels, average="macro", zero_division=0)
        ),
        "ordinal_mae": ordinal_mae(true, pred),
        "confusion": confusion_matrix(true, pred, labels=labels),
    }


def compute_range_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    head_idx: np.ndarray,
    *,
    n_bins: int = N_BINS,
    head_names: tuple[str, ...] = HEAD_NAMES,
) -> dict[str, Any]:
    """Compute per-head and mean metrics for a dual-head range model.

    Args:
        y_true: Ground-truth bins aligned with rows.
        y_pred: Predicted bins aligned with rows.
        head_idx: Per-row head id (``0``=IC50, ``1``=EC50).
        n_bins: Number of ordered classes.
        head_names: Display names for heads.

    Returns:
        Nested metrics mapping with ``per_head`` and ``mean_macro_f1``.
    """
    per_head: dict[str, Any] = {}
    f1_values: list[float] = []
    for hid, name in enumerate(head_names):
        mask = np.asarray(head_idx) == hid
        metrics = compute_head_metrics(y_true[mask], y_pred[mask], n_bins=n_bins)
        per_head[name] = metrics
        if metrics["n"] > 0 and np.isfinite(metrics["macro_f1"]):
            f1_values.append(float(metrics["macro_f1"]))
    mean_f1 = float(np.mean(f1_values)) if f1_values else float("nan")
    return {
        "per_head": per_head,
        "mean_macro_f1": mean_f1,
        "bin_labels": list(BIN_LABELS[:n_bins]),
    }


def print_range_metrics(metrics: dict[str, Any], label: str = "eval") -> None:
    """Pretty-print range metrics.

    Args:
        metrics: Output of :func:`compute_range_metrics`.
        label: Section label.

    Returns:
        None.
    """
    print(f"\n[{label}] mean_macro_f1={metrics['mean_macro_f1']:.4f}")
    for name, head in metrics["per_head"].items():
        print(
            f"  {name}: n={head['n']:,}  Acc={head['accuracy']:.4f}  "
            f"macroF1={head['macro_f1']:.4f}  ordMAE={head['ordinal_mae']:.4f}"
        )
