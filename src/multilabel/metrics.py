"""Multilabel classification metrics (micro / macro / per-label)."""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

def _safe_auroc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """AUROC with a defined fallback when only one class is present.

    Args:
        y_true: Binary labels.
        y_prob: Predicted probabilities.

    Returns:
        AUROC, or ``0.5`` when undefined.
    """
    if len(np.unique(y_true)) < 2:
        return 0.5
    return float(roc_auc_score(y_true, y_prob))


def _safe_auprc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Average precision with a prevalence fallback for single-class labels.

    Args:
        y_true: Binary labels.
        y_prob: Predicted probabilities.

    Returns:
        Average precision, or positive prevalence when undefined.
    """
    if len(np.unique(y_true)) < 2:
        return float(np.mean(y_true))
    return float(average_precision_score(y_true, y_prob))


def micro_auroc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Compute micro-averaged AUROC over flattened label predictions.

    Args:
        y_true: Binary label matrix ``(n, K)``.
        y_prob: Probability matrix ``(n, K)``.

    Returns:
        Micro AUROC.
    """
    return _safe_auroc(np.asarray(y_true).ravel(), np.asarray(y_prob).ravel())


def micro_auprc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Compute micro-averaged AUPRC over flattened label predictions.

    Args:
        y_true: Binary label matrix ``(n, K)``.
        y_prob: Probability matrix ``(n, K)``.

    Returns:
        Micro AUPRC.
    """
    return _safe_auprc(np.asarray(y_true).ravel(), np.asarray(y_prob).ravel())


def per_label_auprc(y_true: np.ndarray, y_prob: np.ndarray) -> np.ndarray:
    """Compute per-label AUPRC.

    Args:
        y_true: Binary label matrix ``(n, K)``.
        y_prob: Probability matrix ``(n, K)``.

    Returns:
        ``float64`` array of shape ``(K,)``.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    k = y_true.shape[1]
    scores = np.empty(k, dtype=np.float64)
    for j in range(k):
        scores[j] = _safe_auprc(y_true[:, j], y_prob[:, j])
    return scores


def per_label_auroc(y_true: np.ndarray, y_prob: np.ndarray) -> np.ndarray:
    """Compute per-label AUROC.

    Args:
        y_true: Binary label matrix ``(n, K)``.
        y_prob: Probability matrix ``(n, K)``.

    Returns:
        ``float64`` array of shape ``(K,)``.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    k = y_true.shape[1]
    scores = np.empty(k, dtype=np.float64)
    for j in range(k):
        scores[j] = _safe_auroc(y_true[:, j], y_prob[:, j])
    return scores


def macro_auprc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Mean per-label AUPRC over labels that contain both classes.

    Args:
        y_true: Binary label matrix ``(n, K)``.
        y_prob: Probability matrix ``(n, K)``.

    Returns:
        Macro AUPRC, or ``0.0`` when no label is evaluable.
    """
    y_true = np.asarray(y_true)
    scores = []
    for j in range(y_true.shape[1]):
        if len(np.unique(y_true[:, j])) < 2:
            continue
        scores.append(_safe_auprc(y_true[:, j], np.asarray(y_prob)[:, j]))
    return float(np.mean(scores)) if scores else 0.0


def macro_auroc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Mean per-label AUROC over labels that contain both classes.

    Args:
        y_true: Binary label matrix ``(n, K)``.
        y_prob: Probability matrix ``(n, K)``.

    Returns:
        Macro AUROC, or ``0.5`` when no label is evaluable.
    """
    y_true = np.asarray(y_true)
    scores = []
    for j in range(y_true.shape[1]):
        if len(np.unique(y_true[:, j])) < 2:
            continue
        scores.append(_safe_auroc(y_true[:, j], np.asarray(y_prob)[:, j]))
    return float(np.mean(scores)) if scores else 0.5


def compute_multilabel_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> dict[str, float]:
    """Compute micro/macro AUROC and AUPRC.

    Args:
        y_true: Binary label matrix ``(n, K)``.
        y_prob: Probability matrix ``(n, K)``.

    Returns:
        Mapping with micro/macro AUROC and AUPRC keys.
    """
    return {
        "micro_auroc": micro_auroc(y_true, y_prob),
        "micro_auprc": micro_auprc(y_true, y_prob),
        "macro_auroc": macro_auroc(y_true, y_prob),
        "macro_auprc": macro_auprc(y_true, y_prob),
    }


def per_label_metrics_frame(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    vocab: Optional[Sequence[str]] = None,
    *,
    decision_threshold: float = 0.5,
) -> pd.DataFrame:
    """Build a per-label metric table for every vocabulary column.

    Args:
        y_true: Binary label matrix ``(n, K)``.
        y_prob: Probability matrix ``(n, K)``.
        vocab: Optional label names aligned with columns.
        decision_threshold: Cutoff used for Precision / Recall.

    Returns:
        DataFrame with one row per label and columns ``label``, ``n``,
        ``n_pos``, ``active_frac``, ``AUROC``, ``AUPRC``, ``Precision``,
        ``Recall``, and ``evaluable``.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    if y_true.ndim != 2 or y_prob.shape != y_true.shape:
        raise ValueError("y_true and y_prob must be 2-D arrays of the same shape.")
    n, k = y_true.shape
    names = list(vocab) if vocab is not None else [str(i) for i in range(k)]
    if len(names) != k:
        raise ValueError(
            f"vocab length {len(names)} does not match label width {k}."
        )

    rows: list[dict[str, object]] = []
    for j in range(k):
        yt = y_true[:, j].astype(int)
        yp = y_prob[:, j].astype(float)
        n_pos = int(yt.sum())
        evaluable = len(np.unique(yt)) >= 2
        pred = (yp >= decision_threshold).astype(int)
        rows.append(
            {
                "label": names[j],
                "n": int(n),
                "n_pos": n_pos,
                "active_frac": float(n_pos / n) if n else 0.0,
                "AUROC": _safe_auroc(yt, yp),
                "AUPRC": _safe_auprc(yt, yp),
                "Precision": float(
                    precision_score(yt, pred, zero_division=0)
                ),
                "Recall": float(recall_score(yt, pred, zero_division=0)),
                "evaluable": bool(evaluable),
            }
        )
    return pd.DataFrame(rows)


def write_per_label_metrics(
    path: str,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    vocab: Optional[Sequence[str]] = None,
    *,
    decision_threshold: float = 0.5,
) -> pd.DataFrame:
    """Write per-label metrics to a spreadsheet (csv/tsv/xlsx by extension).

    Args:
        path: Output path; extension selects the writer.
        y_true: Binary label matrix ``(n, K)``.
        y_prob: Probability matrix ``(n, K)``.
        vocab: Optional label names aligned with columns.
        decision_threshold: Cutoff used for Precision / Recall.

    Returns:
        The DataFrame that was written.
    """
    from src.io_utils import write_table

    frame = per_label_metrics_frame(
        y_true,
        y_prob,
        vocab,
        decision_threshold=decision_threshold,
    )
    write_table(frame, path)
    return frame


def print_multilabel_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    label: str = "",
    vocab: Optional[Sequence[str]] = None,
    top_k: int = 5,
) -> dict[str, float]:
    """Print micro/macro metrics and top/bottom per-label AUROC.

    Args:
        y_true: Binary label matrix ``(n, K)``.
        y_prob: Probability matrix ``(n, K)``.
        label: Optional prefix identifying the evaluation set.
        vocab: Optional label names aligned with columns.
        top_k: Number of best/worst labels to print.

    Returns:
        The micro/macro metric mapping.
    """
    scores = compute_multilabel_metrics(y_true, y_prob)
    prefix = f"{label} " if label else ""
    print(
        f"{prefix}micro_AUROC={scores['micro_auroc']:.4f}  "
        f"micro_AUPRC={scores['micro_auprc']:.4f}  "
        f"macro_AUROC={scores['macro_auroc']:.4f}  "
        f"macro_AUPRC={scores['macro_auprc']:.4f}",
        flush=True,
    )

    y_true_arr = np.asarray(y_true)
    per_label = per_label_auroc(y_true_arr, y_prob)
    evaluable = [
        j for j in range(y_true_arr.shape[1]) if len(np.unique(y_true_arr[:, j])) >= 2
    ]
    if not evaluable:
        return scores

    ranked = sorted(evaluable, key=lambda j: per_label[j], reverse=True)
    names = list(vocab) if vocab is not None else [str(i) for i in range(y_true_arr.shape[1])]
    show = min(top_k, len(ranked))
    print(f"{prefix}top-{show} label AUROC:", flush=True)
    for j in ranked[:show]:
        n_pos = int(y_true_arr[:, j].sum())
        print(f"  {names[j]}: AUROC={per_label[j]:.4f}  n_pos={n_pos}", flush=True)
    print(f"{prefix}bottom-{show} label AUROC:", flush=True)
    for j in ranked[-show:]:
        n_pos = int(y_true_arr[:, j].sum())
        print(f"  {names[j]}: AUROC={per_label[j]:.4f}  n_pos={n_pos}", flush=True)
    return scores
