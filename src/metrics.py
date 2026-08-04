"""Classification metrics for binder / non-binder evaluation."""

from __future__ import annotations

from typing import Optional

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


_MACRO_METRIC_KEYS: tuple[str, ...] = (
    "auroc",
    "auprc",
    "precision",
    "recall",
    "f1",
)


def per_target_binary_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    protein_ids: np.ndarray,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Compute ranking and threshold metrics for every two-class protein.

    Args:
        y_true: Ground-truth binary labels.
        y_prob: Predicted positive-class probabilities.
        protein_ids: Per-row protein / target ids aligned with ``y_true``.

    Returns:
        A tuple ``(target_ids, scores)`` where ``scores`` maps metric name to a
        float array aligned with ``target_ids``. Targets with fewer than two
        classes are omitted. Precision / recall / F1 use a 0.5 probability
        threshold.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    protein_ids = np.asarray(protein_ids)
    empty = np.empty(0, dtype=np.float64)
    if y_true.shape[0] == 0:
        return (
            np.empty(0, dtype=np.int64),
            {key: empty.copy() for key in _MACRO_METRIC_KEYS},
        )

    order = np.argsort(protein_ids, kind="mergesort")
    y_sorted = y_true[order]
    p_sorted = y_prob[order]
    ids_sorted = protein_ids[order]
    unique_ids, start_idx = np.unique(ids_sorted, return_index=True)
    starts = start_idx.tolist()
    ends = starts[1:] + [int(ids_sorted.shape[0])]

    kept_ids: list[int] = []
    kept: dict[str, list[float]] = {key: [] for key in _MACRO_METRIC_KEYS}
    for target_id, start, end in zip(unique_ids, starts, ends):
        y_t = y_sorted[start:end]
        if len(np.unique(y_t)) < 2:
            continue
        p_t = p_sorted[start:end]
        kept_ids.append(int(target_id))
        kept["auroc"].append(float(roc_auc_score(y_t, p_t)))
        kept["auprc"].append(float(average_precision_score(y_t, p_t)))
        kept["precision"].append(precision(y_t, p_t))
        kept["recall"].append(recall(y_t, p_t))
        kept["f1"].append(f1(y_t, p_t))
    return (
        np.asarray(kept_ids, dtype=np.int64),
        {
            key: np.asarray(values, dtype=np.float64)
            for key, values in kept.items()
        },
    )


def per_target_auroc(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    protein_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute AUROC independently for every protein with both classes.

    Args:
        y_true: Ground-truth binary labels.
        y_prob: Predicted positive-class probabilities.
        protein_ids: Per-row protein / target ids aligned with ``y_true``.

    Returns:
        A tuple ``(target_ids, scores)`` of evaluable targets and their AUROCs.
        Targets with fewer than two classes are omitted.
    """
    target_ids, scores = per_target_binary_metrics(y_true, y_prob, protein_ids)
    return target_ids, scores["auroc"]


def macro_target_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    protein_ids: np.ndarray,
) -> tuple[dict[str, float], dict[str, int]]:
    """Average per-target AUROC, AUPRC, precision, recall, and F1.

    Args:
        y_true: Ground-truth binary labels.
        y_prob: Predicted positive-class probabilities.
        protein_ids: Per-row protein / target ids aligned with ``y_true``.

    Returns:
        A tuple ``(macros, info)`` where ``macros`` has keys ``auroc``,
        ``auprc``, ``precision``, ``recall``, and ``f1``, and ``info`` has
        ``n_targets``, ``n_evaluable``, and ``n_skipped``. When no target is
        evaluable, ``auroc`` is ``0.5`` and the other macros are ``0.0``.
    """
    protein_ids = np.asarray(protein_ids)
    n_targets = int(np.unique(protein_ids).shape[0]) if protein_ids.size else 0
    _, scores = per_target_binary_metrics(y_true, y_prob, protein_ids)
    n_evaluable = int(scores["auroc"].shape[0])
    info = {
        "n_targets": n_targets,
        "n_evaluable": n_evaluable,
        "n_skipped": max(0, n_targets - n_evaluable),
    }
    if n_evaluable == 0:
        return {
            "auroc": 0.5,
            "auprc": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
        }, info
    return (
        {key: float(np.mean(values)) for key, values in scores.items()},
        info,
    )


def macro_target_auroc(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    protein_ids: np.ndarray,
) -> tuple[float, dict[str, int]]:
    """Average per-target AUROC, skipping single-class proteins.

    Args:
        y_true: Ground-truth binary labels.
        y_prob: Predicted positive-class probabilities.
        protein_ids: Per-row protein / target ids aligned with ``y_true``.

    Returns:
        A tuple ``(macro_auroc, info)`` where ``info`` has ``n_targets``,
        ``n_evaluable``, and ``n_skipped``. When no target is evaluable the
        score is ``0.5``.
    """
    macros, info = macro_target_metrics(y_true, y_prob, protein_ids)
    return macros["auroc"], info


def compute_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    protein_ids: Optional[np.ndarray] = None,
) -> dict[str, float]:
    """Compute accuracy, precision, recall, F1, AUROC and AUPRC together.

    Args:
        y_true: Ground-truth binary labels.
        y_prob: Predicted positive-class probabilities.
        protein_ids: Optional per-row protein ids. When provided, adds macro
            per-target ``auroc`` / ``auprc`` / ``precision`` / ``recall`` /
            ``f1`` plus evaluable/skipped counts.

    Returns:
        A mapping with keys ``"accuracy"``, ``"precision"``, ``"recall"``,
        ``"f1"``, ``"auroc"`` and ``"auprc"``, plus optional macro fields.
    """
    scores: dict[str, float] = {
        "accuracy": accuracy(y_true, y_prob),
        "precision": precision(y_true, y_prob),
        "recall": recall(y_true, y_prob),
        "f1": f1(y_true, y_prob),
        "auroc": auroc(y_true, y_prob),
        "auprc": auprc(y_true, y_prob),
    }
    if protein_ids is not None:
        macros, info = macro_target_metrics(y_true, y_prob, protein_ids)
        scores["macro_auroc"] = macros["auroc"]
        scores["macro_auprc"] = macros["auprc"]
        scores["macro_precision"] = macros["precision"]
        scores["macro_recall"] = macros["recall"]
        scores["macro_f1"] = macros["f1"]
        scores["macro_auroc_n_evaluable"] = float(info["n_evaluable"])
        scores["macro_auroc_n_skipped"] = float(info["n_skipped"])
    return scores


def print_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    label: str = "",
    protein_ids: Optional[np.ndarray] = None,
) -> dict[str, float]:
    """Compute metrics and print them in a compact one-line summary.

    Args:
        y_true: Ground-truth binary labels.
        y_prob: Predicted positive-class probabilities.
        label: Optional prefix identifying the evaluation (e.g. ``"test"``).
        protein_ids: Optional per-row protein ids for macro per-target metrics.

    Returns:
        The computed metric mapping (see :func:`compute_metrics`).
    """
    scores = compute_metrics(y_true, y_prob, protein_ids=protein_ids)
    prefix = f"{label} " if label else ""
    line = (
        f"{prefix}Acc={scores['accuracy']:.4f}  Prec={scores['precision']:.4f}  "
        f"Rec={scores['recall']:.4f}  F1={scores['f1']:.4f}  "
        f"AUROC={scores['auroc']:.4f}  AUPRC={scores['auprc']:.4f}"
    )
    if protein_ids is not None:
        line += (
            f"  macroAUROC={scores['macro_auroc']:.4f}  "
            f"macroAUPRC={scores['macro_auprc']:.4f}  "
            f"macroPrec={scores['macro_precision']:.4f}  "
            f"macroRec={scores['macro_recall']:.4f}  "
            f"macroF1={scores['macro_f1']:.4f} "
            f"(targets={int(scores['macro_auroc_n_evaluable'])}/"
            f"{int(scores['macro_auroc_n_evaluable'] + scores['macro_auroc_n_skipped'])}, "
            f"skipped={int(scores['macro_auroc_n_skipped'])})"
        )
    print(line, flush=True)
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


def print_macro_target_breakouts(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    protein_ids: np.ndarray,
    protein_known: np.ndarray,
    label: str = "",
) -> dict[str, dict[str, float]]:
    """Print macro per-target metrics overall and for known/unknown proteins.

    Proteins are classified known/unknown by membership in the training protein
    set (passed as a per-row ``protein_known`` mask on the eval set).

    Args:
        y_true: Ground-truth binary labels for the eval set.
        y_prob: Predicted positive-class probabilities.
        protein_ids: Per-row protein ids for the eval set.
        protein_known: Boolean mask; ``True`` when the protein was in train.
        label: Optional prefix identifying the evaluation set.

    Returns:
        Nested mapping of bucket name to macro metric / count fields.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    protein_ids = np.asarray(protein_ids)
    protein_known = np.asarray(protein_known, dtype=bool)
    prefix = f"{label} " if label else ""
    print(f"\n{prefix}macro per-target metrics:", flush=True)

    buckets = {
        "overall": np.ones(y_true.shape[0], dtype=bool),
        "protein_known": protein_known,
        "protein_unknown": ~protein_known,
    }
    results: dict[str, dict[str, float]] = {}
    for name, mask in buckets.items():
        if not mask.any():
            print(f"  {name}: n=0  targets=0  n/a", flush=True)
            results[name] = {}
            continue
        macros, info = macro_target_metrics(
            y_true[mask], y_prob[mask], protein_ids[mask]
        )
        if info["n_evaluable"] == 0:
            print(
                f"  {name}: n={int(mask.sum())}  targets={info['n_targets']}  "
                f"n/a (no two-class targets; skipped={info['n_skipped']})",
                flush=True,
            )
            results[name] = {}
            continue
        print(
            f"  {name}: n={int(mask.sum())}  "
            f"macroAUROC={macros['auroc']:.4f}  "
            f"macroAUPRC={macros['auprc']:.4f}  "
            f"macroPrec={macros['precision']:.4f}  "
            f"macroRec={macros['recall']:.4f}  "
            f"macroF1={macros['f1']:.4f}  "
            f"targets={info['n_evaluable']}/{info['n_targets']}  "
            f"skipped={info['n_skipped']}",
            flush=True,
        )
        results[name] = {
            "macro_auroc": macros["auroc"],
            "macro_auprc": macros["auprc"],
            "macro_precision": macros["precision"],
            "macro_recall": macros["recall"],
            "macro_f1": macros["f1"],
            "macro_auroc_n_evaluable": float(info["n_evaluable"]),
            "macro_auroc_n_skipped": float(info["n_skipped"]),
        }
    return results


def print_breakdowns(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    protein_known: np.ndarray,
    scaffold_known: np.ndarray,
    label: str = "",
    protein_ids: Optional[np.ndarray] = None,
) -> dict[str, dict[str, float]]:
    """Print overall and known/unknown protein and scaffold breakout metrics.

    Args:
        y_true: Ground-truth binary labels for the eval set.
        y_prob: Predicted positive-class probabilities.
        protein_known: Boolean mask; ``True`` when the protein was in train.
        scaffold_known: Boolean mask; ``True`` when the scaffold was in train.
        label: Optional prefix identifying the evaluation set.
        protein_ids: Optional per-row protein ids; when provided, also prints
            macro per-target AUROC breakouts.

    Returns:
        Nested mapping of bucket name to metric dictionaries (empty dict when
        a bucket is ``n/a``). Macro buckets are keyed
        ``macro_overall`` / ``macro_protein_known`` / ``macro_protein_unknown``
        and include AUROC / AUPRC / precision / recall / F1.
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

    if protein_ids is not None:
        macro_results = print_macro_target_breakouts(
            y_true, y_prob, protein_ids, protein_known, label=label
        )
        for name, payload in macro_results.items():
            results[f"macro_{name}"] = payload
    return results
