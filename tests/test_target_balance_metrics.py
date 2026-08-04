"""Tests for per-target balancing and macro per-target AUROC."""

from __future__ import annotations

import numpy as np
import pytest

from src.metrics import auroc, compute_metrics, macro_target_auroc, per_target_auroc
from src.target_balance import (
    apply_target_balance,
    apply_target_size_exponent,
    format_balance_summary,
)


def test_macro_differs_from_pooled_with_unequal_target_sizes() -> None:
    """Macro AUROC equal-weights targets; pooled overweight large ones."""
    # Target 0: 100 rows, perfect ranking. Target 1: 4 rows, chance ranking.
    y = np.concatenate(
        [
            np.array([1, 0] * 50, dtype=np.float32),
            np.array([1, 1, 0, 0], dtype=np.float32),
        ]
    )
    p = np.concatenate(
        [
            np.linspace(1.0, 0.0, 100, dtype=np.float32),
            np.array([0.4, 0.6, 0.5, 0.55], dtype=np.float32),
        ]
    )
    groups = np.concatenate(
        [np.zeros(100, dtype=np.int64), np.ones(4, dtype=np.int64)]
    )
    pooled = auroc(y, p)
    macro, info = macro_target_auroc(y, p, groups)
    assert info["n_evaluable"] == 2
    assert info["n_skipped"] == 0
    assert macro != pytest.approx(pooled, abs=1e-6)
    # Target 0 AUROC ~1.0; target 1 worse → macro pulled down relative to pooled.
    assert macro < pooled


def test_macro_skips_single_class_targets() -> None:
    """Single-class proteins are omitted from the macro average."""
    y = np.array([1, 0, 1, 0, 1, 1, 1], dtype=np.float32)
    p = np.array([0.9, 0.1, 0.8, 0.2, 0.7, 0.6, 0.5], dtype=np.float32)
    groups = np.array([0, 0, 0, 0, 1, 1, 1], dtype=np.int64)
    ids, scores = per_target_auroc(y, p, groups)
    assert list(ids) == [0]
    assert scores.shape == (1,)
    macro, info = macro_target_auroc(y, p, groups)
    assert info["n_targets"] == 2
    assert info["n_evaluable"] == 1
    assert info["n_skipped"] == 1
    assert macro == pytest.approx(float(scores[0]))


def test_compute_metrics_includes_macro_fields() -> None:
    """Optional protein_ids add macro per-target fields to compute_metrics."""
    y = np.array([1, 0, 1, 0], dtype=np.float32)
    p = np.array([0.9, 0.1, 0.8, 0.2], dtype=np.float32)
    groups = np.array([0, 0, 1, 1], dtype=np.int64)
    scores = compute_metrics(y, p, protein_ids=groups)
    assert scores["macro_auroc"] == pytest.approx(1.0)
    assert scores["macro_auprc"] == pytest.approx(1.0)
    assert scores["macro_precision"] == pytest.approx(1.0)
    assert scores["macro_recall"] == pytest.approx(1.0)
    assert scores["macro_f1"] == pytest.approx(1.0)
    assert scores["macro_auroc_n_evaluable"] == 2.0
    assert scores["macro_auroc_n_skipped"] == 0.0


def _balanced_counts(rows: np.ndarray, y: np.ndarray, groups: np.ndarray) -> dict[int, tuple[int, int]]:
    """Return per-target (n_pos, n_neg) counts after resampling."""
    out: dict[int, tuple[int, int]] = {}
    for target in np.unique(groups[rows]):
        mask = groups[rows] == target
        labels = y[rows][mask] > 0.5
        out[int(target)] = (int(labels.sum()), int((~labels).sum()))
    return out


def test_downsample_balances_and_excludes_monomorphic() -> None:
    """Downsample equalizes counts and drops single-class targets."""
    # Target 0: 3 pos / 9 neg. Target 1: all active. Target 2: all inactive.
    y = np.array(
        [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0] + [1, 1, 1] + [0, 0, 0],
        dtype=np.float32,
    )
    groups = np.array(
        [0] * 12 + [1] * 3 + [2] * 3,
        dtype=np.int64,
    )
    rows = np.arange(y.shape[0], dtype=np.int64)
    rng = np.random.default_rng(0)
    result = apply_target_balance(rows, y, groups, "downsample", rng)
    assert result.summary["n_targets_kept"] == 1
    assert result.summary["n_targets_excluded"] == 2
    assert result.summary["n_targets_all_active"] == 1
    assert result.summary["n_targets_all_inactive"] == 1
    counts = _balanced_counts(result.train_rows, y, groups)
    assert counts == {0: (3, 3)}
    assert result.sample_weights is None
    line = format_balance_summary("downsample", result.summary)
    assert "excluded=2" in line
    assert "all_active=1" in line
    assert "all_inactive=1" in line


def test_upsample_balances_and_is_deterministic() -> None:
    """Upsample reaches majority size and is seed-deterministic."""
    y = np.array([1, 1, 0, 0, 0, 0, 1, 0], dtype=np.float32)
    groups = np.array([0, 0, 0, 0, 0, 0, 1, 1], dtype=np.int64)
    rows = np.arange(y.shape[0], dtype=np.int64)
    a = apply_target_balance(rows, y, groups, "upsample", np.random.default_rng(7))
    b = apply_target_balance(rows, y, groups, "upsample", np.random.default_rng(7))
    assert np.array_equal(a.train_rows, b.train_rows)
    counts = _balanced_counts(a.train_rows, y, groups)
    assert counts[0] == (4, 4)
    assert counts[1] == (1, 1)


def test_weights_mode_sets_per_target_pos_mass() -> None:
    """Weights mode keeps rows and equalizes positive/negative loss mass."""
    y = np.array([1, 1, 0, 0, 0, 0, 1, 0, 0], dtype=np.float32)
    groups = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1], dtype=np.int64)
    rows = np.arange(y.shape[0], dtype=np.int64)
    result = apply_target_balance(
        rows, y, groups, "weights", np.random.default_rng(0)
    )
    assert result.summary["n_targets_kept"] == 2
    assert result.summary["n_targets_excluded"] == 0
    assert result.sample_weights is not None
    # Target 0: 2 pos / 4 neg → pos weight 2.0
    assert result.sample_weights[0] == pytest.approx(2.0)
    assert result.sample_weights[2] == pytest.approx(1.0)
    # Target 1: 1 pos / 2 neg → pos weight 2.0
    assert result.sample_weights[6] == pytest.approx(2.0)
    pos_mass = float(result.sample_weights[y > 0.5].sum())
    neg_mass = float(result.sample_weights[y < 0.5].sum())
    assert pos_mass == pytest.approx(neg_mass)


def test_weights_excludes_monomorphic_targets() -> None:
    """Weights mode drops all-active / all-inactive targets from fit rows."""
    y = np.array([1, 0, 1, 1, 0, 0], dtype=np.float32)
    groups = np.array([0, 0, 1, 1, 2, 2], dtype=np.int64)
    result = apply_target_balance(
        np.arange(6, dtype=np.int64),
        y,
        groups,
        "weights",
        np.random.default_rng(0),
    )
    assert result.summary["n_targets_kept"] == 1
    assert result.summary["n_targets_excluded"] == 2
    assert set(np.unique(groups[result.train_rows]).tolist()) == {0}


def test_downsample_dataset_ratio_matches_fit_prevalence() -> None:
    """Dataset-ratio downsample drives each target toward fit-set π."""
    # Overall π = 4/16 = 0.25. Target 0 is inactive-heavy; target 1 active-heavy.
    y = np.array(
        [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] + [1, 1, 0, 0],
        dtype=np.float32,
    )
    groups = np.array([0] * 12 + [1] * 4, dtype=np.int64)
    rows = np.arange(y.shape[0], dtype=np.int64)
    result = apply_target_balance(
        rows, y, groups, "downsample", np.random.default_rng(0), ratio="dataset"
    )
    assert result.summary["ratio"] == "dataset"
    assert result.summary["active_fraction"] == pytest.approx(0.25)
    counts = _balanced_counts(result.train_rows, y, groups)
    # Target 0: 2 pos / 10 neg → keep 2 pos + 6 neg (π=0.25)
    assert counts[0] == (2, 6)
    # Target 1: 2 pos / 2 neg → keep 1 pos + 2 neg
    assert counts[1] == (1, 2)
    line = format_balance_summary("downsample", result.summary)
    assert "ratio=dataset" in line
    assert "π=0.2500" in line


def test_upsample_dataset_ratio_grows_scarce_class() -> None:
    """Dataset-ratio upsample grows the scarce class toward π."""
    # Already at dataset π → unchanged.
    y = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)  # π=0.1
    groups = np.zeros(10, dtype=np.int64)
    result = apply_target_balance(
        np.arange(10, dtype=np.int64),
        y,
        groups,
        "upsample",
        np.random.default_rng(1),
        ratio="dataset",
    )
    assert _balanced_counts(result.train_rows, y, groups)[0] == (1, 9)

    # π=4/12; target 0 active-heavy (3/1), target 1 inactive-heavy (1/7).
    y2 = np.array([1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)
    groups2 = np.array([0, 0, 0, 0] + [1] * 8, dtype=np.int64)
    result2 = apply_target_balance(
        np.arange(12, dtype=np.int64),
        y2,
        groups2,
        "upsample",
        np.random.default_rng(0),
        ratio="dataset",
    )
    pi = result2.summary["active_fraction"]
    assert pi == pytest.approx(4 / 12)
    counts = _balanced_counts(result2.train_rows, y2, groups2)
    n_neg_t0 = max(1, int(round(3 * (1.0 - pi) / pi)))
    n_pos_t1 = max(1, int(round(7 * pi / (1.0 - pi))))
    assert counts[0] == (3, n_neg_t0)
    assert counts[1] == (n_pos_t1, 7)


def test_weights_dataset_ratio_sets_loss_mass_to_pi() -> None:
    """Dataset-ratio weights make weighted positive mass fraction equal π."""
    y = np.array([1, 1, 0, 0, 0, 0, 1, 0, 0, 0], dtype=np.float32)  # π=0.3
    groups = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int64)
    result = apply_target_balance(
        np.arange(10, dtype=np.int64),
        y,
        groups,
        "weights",
        np.random.default_rng(0),
        ratio="dataset",
    )
    assert result.summary["active_fraction"] == pytest.approx(0.3)
    assert result.sample_weights is not None
    kept = result.train_rows
    w = result.sample_weights[kept]
    yk = y[kept]
    pos_mass = float(w[yk > 0.5].sum())
    total = float(w.sum())
    assert pos_mass / total == pytest.approx(0.3)


def test_size_exponent_zero_is_noop() -> None:
    """Alpha 0 leaves None weights unchanged."""
    rows = np.arange(4, dtype=np.int64)
    groups = np.array([0, 0, 1, 1], dtype=np.int64)
    out = apply_target_size_exponent(rows, groups, None, alpha=0.0, n_all=4)
    assert out is None


def test_size_exponent_equalizes_mass_and_preserves_ratios() -> None:
    """Alpha 1 equalizes mass under uniform weights; ratios survive class weights."""
    # Uniform row weights (downsample): unequal sizes → equal mass after alpha=1.
    y = np.array([1, 1, 0, 0, 0, 0, 1, 0], dtype=np.float32)
    groups = np.array([0, 0, 0, 0, 0, 0, 1, 1], dtype=np.int64)
    rows = np.arange(8, dtype=np.int64)
    down = apply_target_balance(
        rows, y, groups, "downsample", np.random.default_rng(0)
    )
    # After equal downsample: target 0 has 4 rows, target 1 has 2.
    scaled = apply_target_size_exponent(
        down.train_rows, groups, None, alpha=1.0, n_all=8
    )
    assert scaled is not None
    mass0 = float(scaled[down.train_rows[groups[down.train_rows] == 0]].sum())
    mass1 = float(scaled[down.train_rows[groups[down.train_rows] == 1]].sum())
    assert mass0 == pytest.approx(mass1, rel=1e-5)

    # Class-weight ratios are preserved under size scaling.
    weighted = apply_target_balance(
        rows, y, groups, "weights", np.random.default_rng(0)
    )
    assert weighted.sample_weights is not None
    t0 = rows[groups == 0]
    w_before = weighted.sample_weights[t0]
    y0 = y[t0]
    ratio_before = float(w_before[y0 > 0.5].mean() / w_before[y0 < 0.5].mean())
    scaled_w = apply_target_size_exponent(
        weighted.train_rows,
        groups,
        weighted.sample_weights,
        alpha=1.0,
        n_all=8,
    )
    assert scaled_w is not None
    w_after = scaled_w[t0]
    ratio_after = float(w_after[y0 > 0.5].mean() / w_after[y0 < 0.5].mean())
    assert ratio_after == pytest.approx(ratio_before)

