"""Target-stratified minibatches and within-target ranking losses for MLP training.

Stratified batches draw a fixed number of positives and negatives from each of
``T`` proteins so the training objective can average per-target BCE (and
optionally RankNet) with equal target weight.
"""

from __future__ import annotations

import math
from typing import Iterator, Optional

import numpy as np
import torch
import torch.nn.functional as F


class TargetStratifiedBatchSampler:
    """Yield row-index batches with ``T`` targets × ``k`` pos + ``k`` neg each.

    Only proteins with both classes in the fit set are eligible. When a class
    has fewer than ``k`` rows, sampling uses replacement.

    Each batch is a flat ``int64`` array laid out as
    ``[t0_pos…, t0_neg…, t1_pos…, t1_neg…, …]`` so losses can ``view(T, 2k)``.
    """

    def __init__(
        self,
        train_rows: np.ndarray,
        y: np.ndarray,
        groups: np.ndarray,
        targets_per_batch: int,
        samples_per_class: int,
        rng: np.random.Generator,
    ) -> None:
        """Build per-target positive/negative row pools.

        Args:
            train_rows: Fit-row indices into ``y`` / ``groups``.
            y: Full binary label vector.
            groups: Full protein-id vector aligned with ``y``.
            targets_per_batch: Number of proteins ``T`` per batch.
            samples_per_class: Positives and negatives ``k`` drawn per protein.
            rng: NumPy Generator for deterministic sampling.

        Raises:
            ValueError: If no two-class targets remain or batch sizes are
                invalid.
        """
        self.targets_per_batch = int(targets_per_batch)
        self.samples_per_class = int(samples_per_class)
        self.rng = rng
        if self.targets_per_batch < 1:
            raise ValueError(
                f"targets_per_batch must be >= 1; got {targets_per_batch}."
            )
        if self.samples_per_class < 1:
            raise ValueError(
                f"samples_per_class must be >= 1; got {samples_per_class}."
            )

        rows = np.asarray(train_rows, dtype=np.int64)
        y_arr = np.asarray(y)
        groups_arr = np.asarray(groups)
        if rows.shape[0] == 0:
            raise ValueError("Stratified sampler requires a non-empty fit set.")

        y_fit = y_arr[rows] > 0.5
        g_fit = groups_arr[rows]
        pos_by_target: dict[int, np.ndarray] = {}
        neg_by_target: dict[int, np.ndarray] = {}
        for target in np.unique(g_fit):
            local = np.flatnonzero(g_fit == target)
            labels = y_fit[local]
            pos_rows = rows[local[labels]]
            neg_rows = rows[local[~labels]]
            if pos_rows.shape[0] == 0 or neg_rows.shape[0] == 0:
                continue
            tid = int(target)
            pos_by_target[tid] = pos_rows
            neg_by_target[tid] = neg_rows

        if not pos_by_target:
            raise ValueError(
                "Stratified sampler found no two-class targets in the fit set."
            )
        self._target_ids = np.asarray(sorted(pos_by_target.keys()), dtype=np.int64)
        self._pos_by_target = pos_by_target
        self._neg_by_target = neg_by_target

    @property
    def n_eligible_targets(self) -> int:
        """Number of two-class proteins available for sampling.

        Returns:
            Eligible target count.
        """
        return int(self._target_ids.shape[0])

    @property
    def batch_row_count(self) -> int:
        """Number of rows in each stratified batch.

        Returns:
            ``T * 2 * k``.
        """
        return self.targets_per_batch * 2 * self.samples_per_class

    def __len__(self) -> int:
        """Number of batches per epoch.

        Returns:
            ``ceil(n_eligible_targets / T)``.
        """
        return int(math.ceil(self.n_eligible_targets / float(self.targets_per_batch)))

    def __iter__(self) -> Iterator[np.ndarray]:
        """Yield one epoch of stratified row-index batches.

        Yields:
            Flat ``int64`` index arrays of length ``T * 2 * k``.
        """
        order = self.rng.permutation(self._target_ids)
        t = self.targets_per_batch
        k = self.samples_per_class
        for start in range(0, order.shape[0], t):
            chunk = order[start : start + t]
            if chunk.shape[0] < t:
                # Pad the last incomplete batch by resampling targets.
                extra = self.rng.choice(
                    self._target_ids, size=t - chunk.shape[0], replace=True
                )
                chunk = np.concatenate([chunk, extra])
            parts: list[np.ndarray] = []
            for tid in chunk:
                tid_i = int(tid)
                pos_pool = self._pos_by_target[tid_i]
                neg_pool = self._neg_by_target[tid_i]
                replace_pos = pos_pool.shape[0] < k
                replace_neg = neg_pool.shape[0] < k
                pos = self.rng.choice(pos_pool, size=k, replace=replace_pos)
                neg = self.rng.choice(neg_pool, size=k, replace=replace_neg)
                parts.append(pos)
                parts.append(neg)
            yield np.concatenate(parts).astype(np.int64, copy=False)


def mean_target_bce_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    weights: Optional[torch.Tensor],
    n_targets: int,
    samples_per_class: int,
) -> torch.Tensor:
    """Average per-target BCE over a stratified batch.

    Args:
        logits: Model logits shaped ``(T * 2 * k,)``.
        labels: Binary labels aligned with ``logits``.
        weights: Optional per-row weights aligned with ``logits``. When
            provided, each target's BCE is weight-normalized.
        n_targets: Number of proteins ``T`` in the batch.
        samples_per_class: Positives/negatives ``k`` per protein.

    Returns:
        Scalar mean-over-targets BCE loss.
    """
    t = int(n_targets)
    k = int(samples_per_class)
    rows_per = 2 * k
    per_row = F.binary_cross_entropy_with_logits(
        logits, labels, reduction="none"
    )
    per_row = per_row.view(t, rows_per)
    if weights is None:
        return per_row.mean()
    w = weights.view(t, rows_per)
    weighted = (per_row * w).sum(dim=1) / w.sum(dim=1).clamp_min(1e-8)
    return weighted.mean()


def within_target_rank_loss(
    logits: torch.Tensor,
    n_targets: int,
    samples_per_class: int,
) -> torch.Tensor:
    """RankNet softplus loss over within-target active/inactive pairs.

    Args:
        logits: Model logits shaped ``(T * 2 * k,)`` with layout
            ``[pos…, neg…]`` per target.
        n_targets: Number of proteins ``T`` in the batch.
        samples_per_class: Positives/negatives ``k`` per protein.

    Returns:
        Scalar mean-over-targets pairwise ranking loss.
    """
    t = int(n_targets)
    k = int(samples_per_class)
    scores = logits.view(t, 2 * k)
    pos = scores[:, :k]
    neg = scores[:, k:]
    diff = pos.unsqueeze(2) - neg.unsqueeze(1)
    return F.softplus(-diff).mean()
