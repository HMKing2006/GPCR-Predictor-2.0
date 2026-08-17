"""Tests for IC50/EC50 potency-range helpers and masked CE."""

from __future__ import annotations

import numpy as np
import torch

from src.data_prep import PreparedRow, activity_to_pactivity
from src.range_model.bins import (
    BIN_LABELS,
    N_BINS,
    activity_nm_to_bin,
    pactivity_to_bin,
)
from src.range_model.config import BINARY_INACTIVE_RANGE_BIN, HEAD_EC50, HEAD_IC50
from src.range_model.labels import include_prepared_row
from src.range_model.models import RangeMLP, masked_range_ce_loss


def test_activity_nm_to_bin_edges() -> None:
    """Decade edges map into the five configured bins."""
    assert activity_nm_to_bin(1.0) == 0
    assert activity_nm_to_bin(9.99) == 0
    assert activity_nm_to_bin(10.0) == 1
    assert activity_nm_to_bin(99.9) == 1
    assert activity_nm_to_bin(100.0) == 2
    assert activity_nm_to_bin(999.0) == 2
    assert activity_nm_to_bin(1000.0) == 3
    assert activity_nm_to_bin(9999.0) == 3
    assert activity_nm_to_bin(10_000.0) == 4
    assert activity_nm_to_bin(1_000_000.0) == 4
    assert N_BINS == 5
    assert len(BIN_LABELS) == 5


def test_pactivity_roundtrip_bin() -> None:
    """pActivity conversion preserves the nM bin."""
    for nm in (5.0, 50.0, 500.0, 5000.0, 50_000.0):
        assert pactivity_to_bin(activity_to_pactivity(nm)) == activity_nm_to_bin(nm)


def test_filter_excludes_binary_by_default() -> None:
    """Binary Activity Label rows are dropped when include_binary is False."""
    row = PreparedRow(
        smiles="CCO",
        sequence="MKT",
        assay_type="Other",
        ph=7.4,
        temp=25.0,
        pactivity=activity_to_pactivity(1_000_000.0),
        activity_label=0.0,
        year=2020,
    )
    assert include_prepared_row(row, include_binary=False) == []


def test_filter_maps_binary_inactive_when_enabled() -> None:
    """Inactive binary Other rows map into both heads at the weak bin."""
    row = PreparedRow(
        smiles="CCO",
        sequence="MKT",
        assay_type="Other",
        ph=7.4,
        temp=25.0,
        pactivity=activity_to_pactivity(1_000_000.0),
        activity_label=0.0,
        year=2020,
    )
    out = include_prepared_row(row, include_binary=True)
    assert len(out) == 2
    assert {r.head_idx for r in out} == {HEAD_IC50, HEAD_EC50}
    assert all(r.y_bin == BINARY_INACTIVE_RANGE_BIN for r in out)
    assert all(r.from_binary for r in out)


def test_filter_keeps_quant_ic50_ec50() -> None:
    """Quantitative IC50/EC50 rows become single-head range examples."""
    ic = PreparedRow(
        smiles="CCO",
        sequence="MKT",
        assay_type="IC50",
        ph=7.4,
        temp=25.0,
        pactivity=activity_to_pactivity(25.0),
        activity_label=None,
        year=2019,
    )
    ec = PreparedRow(
        smiles="CCO",
        sequence="MKT",
        assay_type="EC50",
        ph=7.4,
        temp=25.0,
        pactivity=activity_to_pactivity(2500.0),
        activity_label=None,
        year=2019,
    )
    ic_out = include_prepared_row(ic)
    ec_out = include_prepared_row(ec)
    assert len(ic_out) == 1 and ic_out[0].head_idx == HEAD_IC50
    assert ic_out[0].y_bin == activity_nm_to_bin(25.0)
    assert len(ec_out) == 1 and ec_out[0].head_idx == HEAD_EC50
    assert ec_out[0].y_bin == activity_nm_to_bin(2500.0)


def test_filter_skips_ki() -> None:
    """Non IC50/EC50 quantitative assays are ignored."""
    row = PreparedRow(
        smiles="CCO",
        sequence="MKT",
        assay_type="Ki",
        ph=7.4,
        temp=25.0,
        pactivity=activity_to_pactivity(10.0),
        activity_label=None,
        year=None,
    )
    assert include_prepared_row(row) == []


def test_masked_ce_ignores_other_head() -> None:
    """Loss on an IC50-only batch does not depend on EC50 logits."""
    torch.manual_seed(0)
    batch = 8
    n_bins = 5
    ic_logits = torch.randn(batch, n_bins, requires_grad=True)
    ec_a = torch.randn(batch, n_bins, requires_grad=True)
    ec_b = torch.randn(batch, n_bins, requires_grad=True)
    y = torch.randint(0, n_bins, (batch,))
    head = torch.zeros(batch, dtype=torch.long)  # all IC50
    loss_a = masked_range_ce_loss(ic_logits, ec_a, y, head)
    loss_b = masked_range_ce_loss(ic_logits, ec_b, y, head)
    assert torch.allclose(loss_a, loss_b)
    loss_a.backward()
    assert ec_a.grad is None or torch.all(ec_a.grad == 0)


def test_range_mlp_smoke_train() -> None:
    """Tiny synthetic dual-head fit/predict cycle completes."""
    rng = np.random.default_rng(0)
    n = 120
    d = 16
    X = rng.normal(size=(n, d)).astype(np.float32)
    head_idx = np.array([0] * (n // 2) + [1] * (n - n // 2), dtype=np.int64)
    y_bin = rng.integers(0, N_BINS, size=n)
    groups = np.arange(n, dtype=np.int64) % 20
    years = np.array([2010] * 80 + [2020] * 40, dtype=np.int32)
    model = RangeMLP(
        hidden_dim=16,
        num_layers=1,
        epochs=2,
        patience=1,
        es_val_fraction=0.2,
        batch_size=32,
        class_weights=False,
        seed=0,
    )
    model.fit(X, y_bin, head_idx, groups=groups, years=years, verbose=False)
    pred = model.predict_bins(X, head_idx=head_idx)
    assert pred.shape == (n,)
    assert pred.min() >= 0 and pred.max() < N_BINS
    probs = model.predict_proba(X, head_idx)
    assert probs.shape == (n, N_BINS)
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-5)
