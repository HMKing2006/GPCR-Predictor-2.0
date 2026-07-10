"""Cold-protein train/test/validation splitting with on-disk reuse.

A *cold protein* split guarantees that no protein appearing in the training set
also appears in the evaluation sets, giving an honest estimate of generalization
to unseen targets. Proteins (identified by the per-row group id) are shuffled and
greedily assigned to splits until each split reaches its target *row* fraction.

Splits are cached under ``data/splits/`` keyed by the dataset signature, the
split type and the random seed, so an identical configuration reuses the exact
same partition on subsequent runs.
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np

import config


def cold_protein_split(
    groups: np.ndarray,
    fractions: dict[str, float],
    seed: int = config.RANDOM_SEED,
) -> dict[str, np.ndarray]:
    """Partition row indices into protein-disjoint splits.

    Args:
        groups: Integer array ``(n_rows,)`` giving each row's protein group id.
        fractions: Mapping from split name to target fraction of rows. The
            fractions should sum to 1.0; the first-listed split absorbs any
            rounding remainder.
        seed: Random seed controlling the protein shuffle.

    Returns:
        A mapping from split name to a sorted ``int64`` array of row indices.
        Every protein's rows land entirely within a single split.
    """
    rng = np.random.default_rng(seed)
    unique_groups = np.unique(groups)
    rng.shuffle(unique_groups)

    n_rows = groups.shape[0]
    counts = {g: int(np.count_nonzero(groups == g)) for g in unique_groups}

    names = list(fractions)
    targets = {name: fractions[name] * n_rows for name in names}
    assigned_rows = {name: 0 for name in names}
    group_to_split: dict[int, str] = {}

    # Reserve the first split (usually train) as the overflow bucket.
    primary = names[0]
    for g in unique_groups:
        # Choose the split furthest below its row target (excluding the primary
        # unless everything else is already satisfied).
        best_name = primary
        best_deficit = -np.inf
        for name in names[1:]:
            deficit = targets[name] - assigned_rows[name]
            if deficit > best_deficit and deficit > 0:
                best_deficit = deficit
                best_name = name
        group_to_split[int(g)] = best_name
        assigned_rows[best_name] += counts[g]

    indices: dict[str, list[int]] = {name: [] for name in names}
    for row_idx, g in enumerate(groups):
        indices[group_to_split[int(g)]].append(row_idx)

    return {name: np.asarray(sorted(idx), dtype=np.int64) for name, idx in indices.items()}


def _split_path(signature: str, split_type: str, seed: int, splits_dir: str) -> str:
    """Build the cache path for a saved split.

    Args:
        signature: Dataset signature.
        split_type: Human-readable split label (e.g. ``"train_test"``).
        seed: Random seed used.
        splits_dir: Directory holding split files.

    Returns:
        The ``.npz`` path for this split configuration.
    """
    return os.path.join(splits_dir, f"{signature}__{split_type}__seed{seed}.npz")


def save_split(
    split: dict[str, np.ndarray],
    signature: str,
    split_type: str,
    seed: int = config.RANDOM_SEED,
    splits_dir: str = config.SPLITS_DIR,
) -> str:
    """Persist a split to disk.

    Args:
        split: Mapping from split name to row-index array.
        signature: Dataset signature.
        split_type: Human-readable split label.
        seed: Random seed used.
        splits_dir: Directory to write into.

    Returns:
        The path the split was written to.
    """
    os.makedirs(splits_dir, exist_ok=True)
    path = _split_path(signature, split_type, seed, splits_dir)
    np.savez(path, **split)
    return path


def load_split(
    signature: str,
    split_type: str,
    seed: int = config.RANDOM_SEED,
    splits_dir: str = config.SPLITS_DIR,
) -> Optional[dict[str, np.ndarray]]:
    """Load a previously saved split if present.

    Args:
        signature: Dataset signature.
        split_type: Human-readable split label.
        seed: Random seed used.
        splits_dir: Directory to read from.

    Returns:
        The split mapping, or ``None`` if no cached split exists.
    """
    path = _split_path(signature, split_type, seed, splits_dir)
    if not os.path.exists(path):
        return None
    with np.load(path) as data:
        return {name: data[name] for name in data.files}


def get_or_create_split(
    groups: np.ndarray,
    fractions: dict[str, float],
    signature: str,
    split_type: str,
    seed: int = config.RANDOM_SEED,
    splits_dir: str = config.SPLITS_DIR,
    verbose: bool = True,
) -> dict[str, np.ndarray]:
    """Return a saved split or create, save and return a new one.

    Args:
        groups: Per-row protein group ids.
        fractions: Target row fractions per split.
        signature: Dataset signature.
        split_type: Human-readable split label.
        seed: Random seed.
        splits_dir: Directory for split files.
        verbose: If ``True``, print whether the split was reused or created.

    Returns:
        The split mapping from name to row-index array.
    """
    existing = load_split(signature, split_type, seed, splits_dir)
    if existing is not None:
        if verbose:
            sizes = ", ".join(f"{k}={len(v)}" for k, v in existing.items())
            print(f"[split] reusing {split_type} ({sizes})")
        return existing
    split = cold_protein_split(groups, fractions, seed)
    save_split(split, signature, split_type, seed, splits_dir)
    if verbose:
        sizes = ", ".join(f"{k}={len(v)}" for k, v in split.items())
        print(f"[split] created {split_type} ({sizes})")
    return split


def train_test_split(
    groups: np.ndarray,
    signature: str,
    test_fraction: float = config.TEST_FRACTION,
    seed: int = config.RANDOM_SEED,
    splits_dir: str = config.SPLITS_DIR,
    verbose: bool = True,
) -> dict[str, np.ndarray]:
    """Create/reuse an 80-20 cold-protein train/test split.

    Args:
        groups: Per-row protein group ids.
        signature: Dataset signature.
        test_fraction: Fraction of rows for the test set.
        seed: Random seed.
        splits_dir: Directory for split files.
        verbose: If ``True``, print split info.

    Returns:
        A mapping with keys ``"train"`` and ``"test"``.
    """
    fractions = {"train": 1.0 - test_fraction, "test": test_fraction}
    return get_or_create_split(
        groups, fractions, signature, "train_test", seed, splits_dir, verbose
    )


def train_val_test_split(
    groups: np.ndarray,
    signature: str,
    val_fraction: float = config.GRID_VAL_FRACTION,
    test_fraction: float = config.GRID_TEST_FRACTION,
    seed: int = config.RANDOM_SEED,
    splits_dir: str = config.SPLITS_DIR,
    verbose: bool = True,
) -> dict[str, np.ndarray]:
    """Create/reuse an 80-10-10 cold-protein train/val/test split.

    Args:
        groups: Per-row protein group ids.
        signature: Dataset signature.
        val_fraction: Fraction of rows for validation.
        test_fraction: Fraction of rows for testing.
        seed: Random seed.
        splits_dir: Directory for split files.
        verbose: If ``True``, print split info.

    Returns:
        A mapping with keys ``"train"``, ``"val"`` and ``"test"``.
    """
    fractions = {
        "train": 1.0 - val_fraction - test_fraction,
        "val": val_fraction,
        "test": test_fraction,
    }
    return get_or_create_split(
        groups, fractions, signature, "train_val_test", seed, splits_dir, verbose
    )
