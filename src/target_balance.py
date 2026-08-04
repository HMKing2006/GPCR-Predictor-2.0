"""Per-target class balancing helpers for pair MLP training.

Modes remove target-level active-rate priors by forcing every kept protein
toward a shared active fraction. Targets that have only one class in the fit
set are excluded from balanced training.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Union

import numpy as np

TARGET_BALANCE_MODES: tuple[str, ...] = ("none", "weights", "downsample", "upsample")
TARGET_BALANCE_RATIOS: tuple[str, ...] = ("equal", "dataset")
TARGET_BCE_REDUCTIONS: tuple[str, ...] = ("pooled", "mean")


@dataclass(frozen=True)
class TargetBalanceResult:
    """Result of applying a per-target balancing policy to fit rows.

    Attributes:
        train_rows: Fit-row indices after exclusion and optional resampling.
        sample_weights: Optional per-row weights indexed by original row id
            (length ``n_all``). ``None`` when the mode does not use weighting.
        summary: Counts describing kept/excluded targets and rows.
    """

    train_rows: np.ndarray
    sample_weights: Optional[np.ndarray]
    summary: dict[str, Any]


def resolve_target_active_fraction(
    ratio: Union[str, float],
    y_fit: np.ndarray,
) -> tuple[str, float]:
    """Resolve a ratio policy name or float into an active fraction in ``(0, 1)``.

    Args:
        ratio: ``"equal"`` (0.5), ``"dataset"`` (mean of ``y_fit``), or a float
            in ``(0, 1)``.
        y_fit: Binary labels on the candidate fit rows (before exclusion).

    Returns:
        A tuple ``(ratio_name, active_fraction)``.

    Raises:
        ValueError: If ``ratio`` is invalid or the resolved fraction is not in
            ``(0, 1)``.
    """
    if isinstance(ratio, str):
        key = ratio.strip().lower()
        if key == "equal":
            return "equal", 0.5
        if key == "dataset":
            y_arr = np.asarray(y_fit)
            if y_arr.size == 0:
                raise ValueError("Cannot resolve dataset active fraction from empty y.")
            pi = float(np.mean(y_arr > 0.5))
            if not (0.0 < pi < 1.0):
                raise ValueError(
                    f"Dataset active fraction must be in (0, 1); got {pi:.6f}."
                )
            return "dataset", pi
        raise ValueError(
            f"Unknown target_balance_ratio {ratio!r}; "
            f"expected one of {TARGET_BALANCE_RATIOS} or a float in (0, 1)."
        )
    pi = float(ratio)
    if not (0.0 < pi < 1.0):
        raise ValueError(
            f"target_balance_ratio float must be in (0, 1); got {pi!r}."
        )
    return "custom", pi


def _format_count(n: int) -> str:
    """Format an integer count for human-readable logs.

    Args:
        n: Non-negative count.

    Returns:
        Compact string (e.g. ``8.1M`` / ``12.3k`` / ``4120``).
    """
    value = int(n)
    if value >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    if value >= 10_000:
        return f"{value / 1_000:.1f}k"
    return str(value)


def format_balance_summary(mode: str, summary: dict[str, Any]) -> str:
    """Build the ``[balance]`` log line from a summary dict.

    Args:
        mode: Balancing mode name.
        summary: Mapping produced by :func:`apply_target_balance`.

    Returns:
        A single printable summary line.
    """
    ratio = summary.get("ratio", "equal")
    pi = summary.get("active_fraction", 0.5)
    return (
        f"[balance] mode={mode} ratio={ratio}(π={pi:.4f}) "
        f"kept={summary['n_targets_kept']} "
        f"excluded={summary['n_targets_excluded']} "
        f"(all_active={summary['n_targets_all_active']}, "
        f"all_inactive={summary['n_targets_all_inactive']}) "
        f"rows={_format_count(summary['n_rows_before'])}"
        f"→{_format_count(summary['n_rows_after'])}"
    )


def _target_counts_for_fraction(
    n_pos: int,
    n_neg: int,
    pi: float,
    *,
    upsample: bool,
) -> tuple[int, int]:
    """Choose per-class counts that realize active fraction ``pi``.

    Downsample-only never increases either class. Upsample-only never decreases
    either class. When the local rate already matches ``pi``, both counts are
    unchanged.

    Args:
        n_pos: Current positive count (must be > 0).
        n_neg: Current negative count (must be > 0).
        pi: Target active fraction in ``(0, 1)``.
        upsample: If ``True``, grow the scarce class; else shrink the abundant
            class.

    Returns:
        ``(n_pos_keep, n_neg_keep)``.
    """
    local = n_pos / float(n_pos + n_neg)
    if abs(local - pi) < 1e-12:
        return n_pos, n_neg

    if upsample:
        if local > pi:
            # Too many actives: grow negatives.
            n_neg_target = int(round(n_pos * (1.0 - pi) / pi))
            return n_pos, max(n_neg, n_neg_target)
        # Too few actives: grow positives.
        n_pos_target = int(round(n_neg * pi / (1.0 - pi)))
        return max(n_pos, n_pos_target), n_neg

    if local > pi:
        # Too many actives: shrink positives.
        n_pos_keep = int(round(n_neg * pi / (1.0 - pi)))
        n_pos_keep = max(1, min(n_pos, n_pos_keep))
        return n_pos_keep, n_neg
    # Too few actives: shrink negatives.
    n_neg_keep = int(round(n_pos * (1.0 - pi) / pi))
    n_neg_keep = max(1, min(n_neg, n_neg_keep))
    return n_pos, n_neg_keep


def apply_target_balance(
    train_rows: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    mode: str,
    rng: np.random.Generator,
    ratio: Union[str, float] = "equal",
) -> TargetBalanceResult:
    """Apply a per-target balancing policy to MLP fit rows.

    Single-class (monomorphic) targets are excluded in every non-``none`` mode.

    With ``ratio='equal'`` (active fraction 0.5):

    * ``weights``: positive weight ``n_neg / n_pos`` (negative weight ``1``).
    * ``downsample``: majority class reduced to the minority count.
    * ``upsample``: minority class duplicated up to the majority count.

    With ``ratio='dataset'`` (or a custom float ``π``), every kept target is
    driven toward active fraction ``π``:

    * ``weights``: ``w_pos = (π/(1-π)) * (n_neg/n_pos)``, ``w_neg = 1``.
    * ``downsample`` / ``upsample``: resample so empirical rate ≈ ``π``.

    Args:
        train_rows: Candidate fit-row indices into ``y`` / ``groups``.
        y: Full binary label vector aligned with ``groups``.
        groups: Full protein-id vector aligned with ``y``.
        mode: One of :data:`TARGET_BALANCE_MODES`.
        rng: NumPy Generator for deterministic sampling.
        ratio: ``equal``, ``dataset``, or a float active fraction in ``(0, 1)``.

    Returns:
        A :class:`TargetBalanceResult` with updated rows, optional weights, and
        a summary dict.

    Raises:
        ValueError: If ``mode`` / ``ratio`` is unknown or arrays mismatch.
    """
    mode_key = str(mode).strip().lower()
    if mode_key not in TARGET_BALANCE_MODES:
        raise ValueError(
            f"Unknown target_balance mode {mode!r}; "
            f"expected one of {TARGET_BALANCE_MODES}."
        )
    rows = np.asarray(train_rows, dtype=np.int64)
    y_arr = np.asarray(y)
    groups_arr = np.asarray(groups)
    if y_arr.shape[0] != groups_arr.shape[0]:
        raise ValueError(
            f"y length {y_arr.shape[0]} does not match groups length "
            f"{groups_arr.shape[0]}."
        )
    n_all = int(y_arr.shape[0])
    n_before = int(rows.shape[0])
    y_fit = (y_arr[rows] > 0.5).astype(np.int8) if n_before else np.empty(0, dtype=np.int8)
    ratio_name, pi = (
        ("equal", 0.5)
        if mode_key == "none" or n_before == 0
        else resolve_target_active_fraction(ratio, y_fit)
    )

    empty_summary = {
        "mode": mode_key,
        "ratio": ratio_name,
        "active_fraction": pi,
        "n_targets_total": 0,
        "n_targets_kept": 0,
        "n_targets_excluded": 0,
        "n_targets_all_active": 0,
        "n_targets_all_inactive": 0,
        "n_rows_before": n_before,
        "n_rows_after": n_before,
        "n_rows_dropped": 0,
    }
    if mode_key == "none" or n_before == 0:
        return TargetBalanceResult(
            train_rows=rows.copy(),
            sample_weights=None,
            summary=empty_summary,
        )

    g_fit = groups_arr[rows]
    unique_targets = np.unique(g_fit)
    kept_parts: list[np.ndarray] = []
    n_all_active = 0
    n_all_inactive = 0
    sample_weights: Optional[np.ndarray] = None
    if mode_key == "weights":
        sample_weights = np.ones(n_all, dtype=np.float32)
    odds = pi / (1.0 - pi)

    for target in unique_targets:
        local = np.flatnonzero(g_fit == target)
        labels = y_fit[local]
        n_pos = int(labels.sum())
        n_neg = int(labels.shape[0]) - n_pos
        if n_pos == 0:
            n_all_inactive += 1
            continue
        if n_neg == 0:
            n_all_active += 1
            continue
        target_rows = rows[local]
        pos_local = local[labels == 1]
        neg_local = local[labels == 0]
        pos_rows = rows[pos_local]
        neg_rows = rows[neg_local]

        if mode_key == "weights":
            assert sample_weights is not None
            # Weighted positive mass / total mass = pi when w_neg = 1.
            sample_weights[pos_rows] = float(odds * n_neg / n_pos)
            sample_weights[neg_rows] = 1.0
            kept_parts.append(target_rows)
            continue

        n_pos_keep, n_neg_keep = _target_counts_for_fraction(
            n_pos, n_neg, pi, upsample=(mode_key == "upsample")
        )
        if mode_key == "downsample":
            if n_pos_keep < n_pos:
                pos_rows = rng.choice(pos_rows, size=n_pos_keep, replace=False)
            if n_neg_keep < n_neg:
                neg_rows = rng.choice(neg_rows, size=n_neg_keep, replace=False)
        else:  # upsample
            if n_pos_keep > n_pos:
                extra = rng.choice(pos_rows, size=n_pos_keep - n_pos, replace=True)
                pos_rows = np.concatenate([pos_rows, extra])
            if n_neg_keep > n_neg:
                extra = rng.choice(neg_rows, size=n_neg_keep - n_neg, replace=True)
                neg_rows = np.concatenate([neg_rows, extra])
        kept_parts.append(np.concatenate([pos_rows, neg_rows]))

    n_excluded = n_all_active + n_all_inactive
    n_kept = int(unique_targets.shape[0]) - n_excluded
    if kept_parts:
        out_rows = np.concatenate(kept_parts).astype(np.int64, copy=False)
    else:
        out_rows = np.empty(0, dtype=np.int64)
    n_after = int(out_rows.shape[0])
    summary = {
        "mode": mode_key,
        "ratio": ratio_name,
        "active_fraction": pi,
        "n_targets_total": int(unique_targets.shape[0]),
        "n_targets_kept": n_kept,
        "n_targets_excluded": n_excluded,
        "n_targets_all_active": n_all_active,
        "n_targets_all_inactive": n_all_inactive,
        "n_rows_before": n_before,
        "n_rows_after": n_after,
        "n_rows_dropped": max(0, n_before - n_after) if mode_key != "upsample" else 0,
    }
    return TargetBalanceResult(
        train_rows=out_rows,
        sample_weights=sample_weights,
        summary=summary,
    )


def apply_target_size_exponent(
    train_rows: np.ndarray,
    groups: np.ndarray,
    sample_weights: Optional[np.ndarray],
    alpha: float,
    n_all: int,
) -> Optional[np.ndarray]:
    """Scale row weights by ``1 / n_t**alpha`` so large targets contribute less.

    Applied after :func:`apply_target_balance`. Within-target class-weight
    ratios are preserved; only the total mass of each target pool changes.
    When ``alpha == 0`` the input weights are returned unchanged (``None``
    stays ``None``).

    Args:
        train_rows: Fit-row indices after balancing.
        groups: Full protein-id vector aligned with the dataset.
        sample_weights: Optional per-row weights of length ``n_all``. When
            ``None`` and ``alpha > 0``, ones are allocated.
        alpha: Size exponent. Must be ``>= 0``. ``0`` is a no-op.
        n_all: Length of the full label / weight vector.

    Returns:
        Updated sample-weight vector, or ``None`` when ``alpha == 0`` and no
        weights were provided.

    Raises:
        ValueError: If ``alpha < 0`` or array lengths are inconsistent.
    """
    alpha_f = float(alpha)
    if alpha_f < 0.0:
        raise ValueError(f"target_size_exponent must be >= 0; got {alpha!r}.")
    rows = np.asarray(train_rows, dtype=np.int64)
    groups_arr = np.asarray(groups)
    if int(n_all) != int(groups_arr.shape[0]):
        raise ValueError(
            f"n_all={n_all} does not match groups length {groups_arr.shape[0]}."
        )
    if alpha_f == 0.0:
        return sample_weights
    if sample_weights is None:
        weights = np.ones(int(n_all), dtype=np.float32)
    else:
        weights = np.asarray(sample_weights, dtype=np.float32).copy()
        if weights.shape[0] != int(n_all):
            raise ValueError(
                f"sample_weights length {weights.shape[0]} does not match "
                f"n_all={n_all}."
            )
    if rows.shape[0] == 0:
        return weights
    g_fit = groups_arr[rows]
    unique_targets, inverse, counts = np.unique(
        g_fit, return_inverse=True, return_counts=True
    )
    del unique_targets
    scales = (counts.astype(np.float64) ** (-alpha_f)).astype(np.float32)
    weights[rows] = weights[rows] * scales[inverse]
    return weights
