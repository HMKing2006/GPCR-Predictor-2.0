"""Tests for stratified batches, mean-target BCE, and RankNet loss."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.models import MLPModel
from src.rank_batches import (
    TargetStratifiedBatchSampler,
    mean_target_bce_loss,
    within_target_rank_loss,
)


def _toy_fit_arrays() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build a tiny two-target labeled index set.

    Returns:
        ``(train_rows, y, groups)`` with two two-class proteins.
    """
    # Target 0: 4 pos / 4 neg. Target 1: 2 pos / 2 neg. Target 2: all active (skip).
    y = np.array(
        [1, 1, 1, 1, 0, 0, 0, 0] + [1, 1, 0, 0] + [1, 1],
        dtype=np.float32,
    )
    groups = np.array(
        [0] * 8 + [1] * 4 + [2] * 2,
        dtype=np.int64,
    )
    rows = np.arange(y.shape[0], dtype=np.int64)
    return rows, y, groups


def test_stratified_sampler_layout_and_two_class_only() -> None:
    """Each batch has T blocks of k pos + k neg from two-class targets only."""
    rows, y, groups = _toy_fit_arrays()
    sampler = TargetStratifiedBatchSampler(
        rows,
        y,
        groups,
        targets_per_batch=2,
        samples_per_class=2,
        rng=np.random.default_rng(0),
    )
    assert sampler.n_eligible_targets == 2
    assert sampler.batch_row_count == 8
    batch = next(iter(sampler))
    assert batch.shape == (8,)
    # Layout: t0_pos(2), t0_neg(2), t1_pos(2), t1_neg(2)
    assert set(y[batch[:2]].tolist()) == {1.0}
    assert set(y[batch[2:4]].tolist()) == {0.0}
    assert set(y[batch[4:6]].tolist()) == {1.0}
    assert set(y[batch[6:8]].tolist()) == {0.0}
    # Monomorphic target 2 never appears.
    assert 2 not in set(groups[batch].tolist())


def test_mean_target_bce_equalizes_unequal_targets() -> None:
    """Mean-target BCE gives equal loss mass to unequal target blocks."""
    # Perfect predictions for both targets, but different block sizes would
    # matter for pooled BCE; here each target block is 4 rows after reshape.
    logits = torch.tensor(
        [10.0, 10.0, -10.0, -10.0, 10.0, 10.0, -10.0, -10.0],
        dtype=torch.float32,
    )
    labels = torch.tensor(
        [1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0],
        dtype=torch.float32,
    )
    loss = mean_target_bce_loss(logits, labels, None, n_targets=2, samples_per_class=2)
    assert float(loss) == pytest.approx(0.0, abs=1e-4)

    # Inflate loss on target 1 only; mean should be halfway between 0 and that.
    logits_bad = logits.clone()
    logits_bad[4:] = 0.0  # chance on target 1
    loss_bad = mean_target_bce_loss(
        logits_bad, labels, None, n_targets=2, samples_per_class=2
    )
    # Target 0 near 0; target 1 ≈ ln(2) ≈ 0.693 → mean ≈ 0.346
    assert float(loss_bad) == pytest.approx(0.5 * float(torch.nn.functional.binary_cross_entropy_with_logits(logits_bad[4:], labels[4:])), rel=1e-4)


def test_rank_loss_perfect_vs_inverted() -> None:
    """Rank loss is near zero for perfect order and higher when inverted."""
    # Layout: 2 targets, k=2 → [pos, pos, neg, neg] × 2
    perfect = torch.tensor(
        [5.0, 4.0, -1.0, -2.0, 3.0, 2.0, -3.0, -4.0],
        dtype=torch.float32,
    )
    inverted = torch.tensor(
        [-5.0, -4.0, 1.0, 2.0, -3.0, -2.0, 3.0, 4.0],
        dtype=torch.float32,
    )
    good = float(within_target_rank_loss(perfect, 2, 2))
    bad = float(within_target_rank_loss(inverted, 2, 2))
    assert good < 0.01
    assert bad > good + 1.0


def test_listwise_loss_perfect_vs_inverted() -> None:
    """Listwise CE is lower for perfect order than inverted order."""
    from src.rank_batches import within_target_listwise_loss

    perfect = torch.tensor(
        [5.0, 4.0, -1.0, -2.0, 3.0, 2.0, -3.0, -4.0],
        dtype=torch.float32,
    )
    inverted = torch.tensor(
        [-5.0, -4.0, 1.0, 2.0, -3.0, -2.0, 3.0, 4.0],
        dtype=torch.float32,
    )
    good = float(within_target_listwise_loss(perfect, 2, 2))
    bad = float(within_target_listwise_loss(inverted, 2, 2))
    assert good < bad


def test_mlp_fit_film_and_listwise_smoke() -> None:
    """FiLM architecture with listwise loss completes one epoch."""
    rng = np.random.default_rng(2)
    n = 48
    d = 8
    X = rng.normal(size=(n, d)).astype(np.float32)
    groups = np.repeat(np.arange(3), n // 3).astype(np.int64)
    y = np.zeros(n, dtype=np.float32)
    for t in range(3):
        mask = groups == t
        y[mask] = np.tile([1.0, 0.0], int(mask.sum()) // 2)
    model = MLPModel(
        hidden_dim=8,
        num_layers=2,
        dropout=0.0,
        epochs=1,
        patience=0,
        class_weights=False,
        target_balance="weights",
        target_bce_reduction="mean",
        rank_loss_weight=0.5,
        listwise_loss_weight=0.5,
        rank_targets_per_batch=2,
        rank_samples_per_class=2,
        use_film=True,
        protein_dim=4,
        ligand_dim=4,
        seed=2,
        device=torch.device("cpu"),
    )
    model.fit(X, y, verbose=False, groups=groups)
    probs = model.predict(X[:4])
    assert probs.shape == (4,)
    assert np.all(np.isfinite(probs))


def test_mlp_fit_mean_reduction_smoke() -> None:
    """MLP with mean-target BCE completes one epoch on tiny data."""
    rng = np.random.default_rng(0)
    n = 64
    d = 8
    X = rng.normal(size=(n, d)).astype(np.float32)
    # Four proteins, balanced labels.
    groups = np.repeat(np.arange(4), n // 4).astype(np.int64)
    y = np.tile([1, 1, 0, 0], n // 4).astype(np.float32)
    model = MLPModel(
        hidden_dim=8,
        num_layers=1,
        dropout=0.0,
        batch_size=16,
        epochs=1,
        patience=0,
        class_weights=False,
        target_balance="none",
        target_bce_reduction="mean",
        rank_loss_weight=0.0,
        rank_targets_per_batch=2,
        rank_samples_per_class=2,
        target_size_exponent=0.0,
        protein_dim=4,
        ligand_dim=4,
        seed=0,
        device=torch.device("cpu"),
    )
    model.fit(X, y, verbose=False, groups=groups)
    probs = model.predict(X)
    assert probs.shape == (n,)
    assert np.all(np.isfinite(probs))


def test_mlp_fit_mean_rank_and_size_exp_smoke() -> None:
    """MLP with mean BCE, RankNet, and size exponent completes one epoch."""
    rng = np.random.default_rng(1)
    n = 48
    d = 6
    X = rng.normal(size=(n, d)).astype(np.float32)
    groups = np.repeat(np.arange(3), n // 3).astype(np.int64)
    y = np.tile([1, 0, 1, 0], n // 4).astype(np.float32)[:n]
    # Ensure each target has both classes.
    for t in range(3):
        mask = groups == t
        y[mask] = np.tile([1.0, 0.0], int(mask.sum()) // 2)
    model = MLPModel(
        hidden_dim=8,
        num_layers=1,
        dropout=0.0,
        epochs=1,
        patience=0,
        class_weights=False,
        target_balance="weights",
        target_bce_reduction="mean",
        rank_loss_weight=1.0,
        rank_targets_per_batch=2,
        rank_samples_per_class=2,
        target_size_exponent=1.0,
        protein_dim=3,
        ligand_dim=3,
        seed=1,
        device=torch.device("cpu"),
    )
    model.fit(X, y, verbose=False, groups=groups)
    assert model.predict(X[:4]).shape == (4,)
