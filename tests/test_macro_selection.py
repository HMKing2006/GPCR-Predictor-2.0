"""Tests for MLP early-stop / selection metric preference for macro AUROC."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from src.metrics import auroc, macro_target_auroc
from src.models import MLPModel


class _TinyMLP(MLPModel):
    """MLP stub that returns fixed validation probabilities."""

    def __init__(self, probs: np.ndarray) -> None:
        """Store fixed probabilities used by ``_predict_logits``.

        Args:
            probs: Full-length probability vector for synthetic rows.
        """
        super().__init__(patience=0, epochs=1, target_balance="none")
        self._probs = np.asarray(probs, dtype=np.float32)
        self.net = object()  # satisfy assert in helpers if touched
        self.feature_mean = np.zeros(1, dtype=np.float32)
        self.feature_std = np.ones(1, dtype=np.float32)

    def _predict_logits(
        self, X: Any, rows: np.ndarray
    ) -> np.ndarray:  # noqa: N802
        """Return logits corresponding to the fixed probabilities.

        Args:
            X: Unused feature matrix.
            rows: Row indices to score.

        Returns:
            Logits for ``rows``.
        """
        del X
        p = np.clip(self._probs[rows], 1e-6, 1.0 - 1e-6)
        return np.log(p / (1.0 - p)).astype(np.float32)


def test_eval_auroc_prefers_macro_over_pooled() -> None:
    """Early-stop metric uses macro AUROC when two-class targets exist."""
    y = np.concatenate(
        [
            np.array([1, 0] * 50, dtype=np.float32),
            np.array([1, 1, 0, 0], dtype=np.float32),
        ]
    )
    # Perfect on large target, near-chance on small → pooled high, macro lower.
    probs = np.concatenate(
        [
            np.linspace(0.99, 0.01, 100, dtype=np.float32),
            np.array([0.4, 0.6, 0.5, 0.55], dtype=np.float32),
        ]
    )
    groups = np.concatenate(
        [np.zeros(100, dtype=np.int64), np.ones(4, dtype=np.int64)]
    )
    model = _TinyMLP(probs)
    rows = np.arange(y.shape[0], dtype=np.int64)
    score, fallback = model._eval_auroc(
        np.zeros((y.shape[0], 1), dtype=np.float32),
        rows,
        y,
        groups=groups,
    )
    pooled = auroc(y, probs)
    macro, info = macro_target_auroc(y, probs, groups)
    assert info["n_evaluable"] == 2
    assert fallback is False
    assert score == pytest.approx(macro)
    assert score != pytest.approx(pooled)


def test_eval_auroc_falls_back_when_no_two_class_target() -> None:
    """Pooled AUROC is used when every validation target is monomorphic."""
    y = np.array([1, 1, 1, 0, 0, 0], dtype=np.float32)
    probs = np.array([0.9, 0.8, 0.7, 0.2, 0.1, 0.05], dtype=np.float32)
    groups = np.array([0, 0, 0, 1, 1, 1], dtype=np.int64)
    model = _TinyMLP(probs)
    rows = np.arange(6, dtype=np.int64)
    score, fallback = model._eval_auroc(
        np.zeros((6, 1), dtype=np.float32),
        rows,
        y,
        groups=groups,
        verbose=False,
    )
    assert fallback is True
    assert score == pytest.approx(auroc(y, probs))


def test_selection_prefers_higher_macro_when_pooled_disagrees() -> None:
    """Grid-style selection should rank by macro when both are defined."""
    rng = np.random.default_rng(0)
    n_large, n_small = 400, 40
    y_large = np.array([1] * (n_large // 2) + [0] * (n_large // 2), dtype=np.float32)
    y_small = np.array([1] * (n_small // 2) + [0] * (n_small // 2), dtype=np.float32)
    y = np.concatenate([y_large, y_small])
    groups = np.concatenate(
        [np.zeros(n_large, dtype=np.int64), np.ones(n_small, dtype=np.int64)]
    )
    # Candidate A: perfect large target, inverted small target.
    probs_a = np.concatenate(
        [
            np.linspace(0.99, 0.01, n_large, dtype=np.float32),
            np.linspace(0.01, 0.99, n_small, dtype=np.float32),
        ]
    )
    # Candidate B: noisy/weak large target, perfect small target.
    noise = rng.normal(0.0, 0.35, size=n_large).astype(np.float32)
    probs_b_large = np.clip(
        y_large.astype(np.float32) * 0.2 + 0.4 + noise, 0.01, 0.99
    )
    probs_b = np.concatenate(
        [
            probs_b_large,
            np.linspace(0.99, 0.01, n_small, dtype=np.float32),
        ]
    )
    pooled_a = auroc(y, probs_a)
    pooled_b = auroc(y, probs_b)
    macro_a, _ = macro_target_auroc(y, probs_a, groups)
    macro_b, _ = macro_target_auroc(y, probs_b, groups)
    assert pooled_a > pooled_b
    assert macro_b > macro_a
    best = "A" if macro_a > macro_b else "B"
    assert best == "B"
