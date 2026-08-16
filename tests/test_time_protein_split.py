"""Tests for time-protein splits and MLP early-stop preference."""

from __future__ import annotations

import numpy as np

from src.models import MLPModel
from src.splits import time_protein_split


def test_time_protein_holds_out_new_proteins_in_latest_year() -> None:
    """Latest-year new proteins go to val; known proteins stay in train."""
    years = np.array([2010, 2010, 2010, 2020, 2020, 2020], dtype=np.int32)
    # Protein 0 is seen in 2010 and also appears in 2020; 1 is new in 2020.
    groups = np.array([0, 0, 0, 0, 1, 1], dtype=np.int64)
    split = time_protein_split(years, groups, n_years=1, min_val_rows=1)
    val = set(int(i) for i in split["val"])
    train = set(int(i) for i in split["train"])
    assert val == {4, 5}
    assert train == {0, 1, 2, 3}
    assert not val.intersection(train)


def test_time_protein_undated_rows_count_as_seen() -> None:
    """Proteins with undated rows are treated as known even in the latest year."""
    years = np.array([-1, -1, 2020, 2020], dtype=np.int32)
    groups = np.array([0, 1, 0, 2], dtype=np.int64)
    split = time_protein_split(years, groups, n_years=1, min_val_rows=1)
    assert set(split["val"].tolist()) == {3}
    assert set(split["train"].tolist()) == {0, 1, 2}


def test_time_protein_expands_window_when_one_year_empty() -> None:
    """Window expands when the latest year has no unseen proteins."""
    # 2020 only repeats protein 0 from 2010; 2019 introduces protein 1.
    years = np.array([2010, 2010, 2019, 2019, 2020, 2020], dtype=np.int32)
    groups = np.array([0, 0, 1, 1, 0, 0], dtype=np.int64)
    split = time_protein_split(years, groups, n_years=1, min_val_rows=1)
    assert set(split["val"].tolist()) == {2, 3}
    assert set(split["train"].tolist()) == {0, 1, 4, 5}


def test_mlp_fit_prefers_time_protein_es_holdout(capsys) -> None:
    """With dated rows, early stopping carves a time-protein holdout."""
    rng = np.random.default_rng(0)
    n_old, n_new = 80, 40
    n = n_old + n_new
    X = rng.normal(size=(n, 8)).astype(np.float32)
    y = (rng.random(n) > 0.5).astype(np.float32)
    groups = np.concatenate(
        [np.arange(n_old, dtype=np.int64) % 8, 100 + np.arange(n_new, dtype=np.int64) % 8]
    )
    years = np.array([2010] * n_old + [2020] * n_new, dtype=np.int32)

    model = MLPModel(
        hidden_dim=8,
        num_layers=1,
        epochs=1,
        patience=1,
        es_val_fraction=0.1,
        batch_size=32,
        seed=0,
    )
    model.fit(X, y, verbose=True, groups=groups, years=years)
    captured = capsys.readouterr().out
    assert "time-protein early-stop holdout" in captured
    assert "cold-protein early-stop holdout" not in captured
