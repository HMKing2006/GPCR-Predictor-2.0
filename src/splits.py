"""Double-cold and cold-protein train/test/validation splitting with on-disk reuse.

A *double-cold* split guarantees that train/val/test share neither proteins nor
Murcko scaffolds: rows are partitioned by connected components of the
protein–scaffold co-occurrence graph. A *cold protein* split (legacy helper)
only enforces protein disjointness.

Splits are cached under ``data/splits/`` keyed by the dataset signature, the
split type and the random seed, so an identical configuration reuses the exact
same partition on subsequent runs.
"""

from __future__ import annotations

import os
from collections import defaultdict, deque
from typing import Optional

import numpy as np

import config


def _assign_blocks_to_splits(
    block_ids: np.ndarray,
    block_sizes: dict[int, int],
    fractions: dict[str, float],
    seed: int,
) -> dict[str, np.ndarray]:
    """Greedily assign atomic blocks to splits by target row fraction.

    Args:
        block_ids: Integer array ``(n_rows,)`` giving each row's block id.
        block_sizes: Mapping from block id to number of rows in that block.
        fractions: Mapping from split name to target fraction of rows. The
            fractions should sum to 1.0; the first-listed split absorbs any
            rounding remainder.
        seed: Random seed controlling the block shuffle.

    Returns:
        A mapping from split name to a sorted ``int64`` array of row indices.
    """
    rng = np.random.default_rng(seed)
    unique_blocks = np.asarray(sorted(block_sizes), dtype=np.int64)
    rng.shuffle(unique_blocks)

    n_rows = block_ids.shape[0]
    names = list(fractions)
    targets = {name: fractions[name] * n_rows for name in names}
    assigned_rows = {name: 0 for name in names}
    block_to_split: dict[int, str] = {}

    # Reserve the first split (usually train) as the overflow bucket.
    primary = names[0]
    for block in unique_blocks:
        best_name = primary
        best_deficit = -np.inf
        for name in names[1:]:
            deficit = targets[name] - assigned_rows[name]
            if deficit > best_deficit and deficit > 0:
                best_deficit = deficit
                best_name = name
        bid = int(block)
        block_to_split[bid] = best_name
        assigned_rows[best_name] += block_sizes[bid]

    indices: dict[str, list[int]] = {name: [] for name in names}
    for row_idx, block in enumerate(block_ids):
        indices[block_to_split[int(block)]].append(row_idx)

    return {name: np.asarray(sorted(idx), dtype=np.int64) for name, idx in indices.items()}


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
    unique_groups, group_counts = np.unique(groups, return_counts=True)
    count_by_group = {int(g): int(c) for g, c in zip(unique_groups, group_counts)}
    return _assign_blocks_to_splits(
        np.asarray(groups, dtype=np.int64),
        count_by_group,
        fractions,
        seed,
    )


def double_cold_split(
    protein_groups: np.ndarray,
    scaffold_groups: np.ndarray,
    fractions: dict[str, float],
    seed: int = config.RANDOM_SEED,
) -> dict[str, np.ndarray]:
    """Partition rows so splits share neither proteins nor scaffolds.

    Builds the undirected bipartite co-occurrence graph of proteins and
    scaffolds, takes connected components as atomic blocks, then greedily
    assigns blocks to splits by row count.

    Args:
        protein_groups: Integer array ``(n_rows,)`` of protein group ids.
        scaffold_groups: Integer array ``(n_rows,)`` of Murcko scaffold ids.
        fractions: Mapping from split name to target fraction of rows.
        seed: Random seed controlling the component shuffle.

    Returns:
        A mapping from split name to a sorted ``int64`` array of row indices.

    Raises:
        ValueError: If the group arrays differ in length.
    """
    protein_groups = np.asarray(protein_groups)
    scaffold_groups = np.asarray(scaffold_groups)
    if protein_groups.shape[0] != scaffold_groups.shape[0]:
        raise ValueError(
            f"protein_groups length {protein_groups.shape[0]} does not match "
            f"scaffold_groups length {scaffold_groups.shape[0]}."
        )

    # Bipartite adjacency: protein nodes keyed as ("p", id), scaffolds as ("s", id).
    neighbors: dict[tuple[str, int], set[tuple[str, int]]] = defaultdict(set)
    for prot, scaff in zip(protein_groups.tolist(), scaffold_groups.tolist()):
        p_node = ("p", int(prot))
        s_node = ("s", int(scaff))
        neighbors[p_node].add(s_node)
        neighbors[s_node].add(p_node)

    # Connected components over the bipartite graph.
    visited: set[tuple[str, int]] = set()
    node_to_component: dict[tuple[str, int], int] = {}
    component_id = 0
    for start in list(neighbors):
        if start in visited:
            continue
        queue: deque[tuple[str, int]] = deque([start])
        visited.add(start)
        while queue:
            node = queue.popleft()
            node_to_component[node] = component_id
            for nxt in neighbors[node]:
                if nxt not in visited:
                    visited.add(nxt)
                    queue.append(nxt)
        component_id += 1

    # Map each row to its component; isolated nodes still get a component via
    # the protein/scaffold id even if somehow missing from neighbors (should not
    # happen for observed pairs).
    n_rows = protein_groups.shape[0]
    block_ids = np.empty(n_rows, dtype=np.int64)
    block_sizes: dict[int, int] = defaultdict(int)
    for i, (prot, scaff) in enumerate(
        zip(protein_groups.tolist(), scaffold_groups.tolist())
    ):
        cid = node_to_component[("p", int(prot))]
        block_ids[i] = cid
        block_sizes[cid] += 1

    return _assign_blocks_to_splits(block_ids, dict(block_sizes), fractions, seed)


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


def get_or_create_double_cold_split(
    protein_groups: np.ndarray,
    scaffold_groups: np.ndarray,
    fractions: dict[str, float],
    signature: str,
    split_type: str,
    seed: int = config.RANDOM_SEED,
    splits_dir: str = config.SPLITS_DIR,
    verbose: bool = True,
) -> dict[str, np.ndarray]:
    """Return a saved double-cold split or create, save and return a new one.

    Args:
        protein_groups: Per-row protein group ids.
        scaffold_groups: Per-row Murcko scaffold group ids.
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
    split = double_cold_split(protein_groups, scaffold_groups, fractions, seed)
    save_split(split, signature, split_type, seed, splits_dir)
    if verbose:
        sizes = ", ".join(f"{k}={len(v)}" for k, v in split.items())
        print(f"[split] created {split_type} ({sizes})")
    return split


def get_or_create_split(
    groups: np.ndarray,
    fractions: dict[str, float],
    signature: str,
    split_type: str,
    seed: int = config.RANDOM_SEED,
    splits_dir: str = config.SPLITS_DIR,
    verbose: bool = True,
) -> dict[str, np.ndarray]:
    """Return a saved cold-protein split or create, save and return a new one.

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
    protein_groups: np.ndarray,
    scaffold_groups: np.ndarray,
    signature: str,
    test_fraction: float = config.TEST_FRACTION,
    seed: int = config.RANDOM_SEED,
    splits_dir: str = config.SPLITS_DIR,
    verbose: bool = True,
) -> dict[str, np.ndarray]:
    """Create/reuse an 80-20 double-cold train/test split.

    Args:
        protein_groups: Per-row protein group ids.
        scaffold_groups: Per-row Murcko scaffold group ids.
        signature: Dataset signature.
        test_fraction: Fraction of rows for the test set.
        seed: Random seed.
        splits_dir: Directory for split files.
        verbose: If ``True``, print split info.

    Returns:
        A mapping with keys ``"train"`` and ``"test"``.
    """
    fractions = {"train": 1.0 - test_fraction, "test": test_fraction}
    return get_or_create_double_cold_split(
        protein_groups,
        scaffold_groups,
        fractions,
        signature,
        "double_cold_train_test",
        seed,
        splits_dir,
        verbose,
    )


def train_val_test_split(
    protein_groups: np.ndarray,
    scaffold_groups: np.ndarray,
    signature: str,
    val_fraction: float = config.GRID_VAL_FRACTION,
    test_fraction: float = config.GRID_TEST_FRACTION,
    seed: int = config.RANDOM_SEED,
    splits_dir: str = config.SPLITS_DIR,
    verbose: bool = True,
) -> dict[str, np.ndarray]:
    """Create/reuse an 80-10-10 double-cold train/val/test split.

    Args:
        protein_groups: Per-row protein group ids.
        scaffold_groups: Per-row Murcko scaffold group ids.
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
    return get_or_create_double_cold_split(
        protein_groups,
        scaffold_groups,
        fractions,
        signature,
        "double_cold_train_val_test",
        seed,
        splits_dir,
        verbose,
    )
