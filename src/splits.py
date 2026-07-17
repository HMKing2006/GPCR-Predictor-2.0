"""Dataset-scoped train/test/validation splitting with validated on-disk reuse.

A *double-cold* split guarantees that train/val/test share neither proteins nor
Murcko scaffolds: rows are partitioned by connected components of the
protein–scaffold co-occurrence graph. A *cold protein* split only enforces
protein disjointness.

Splits are cached under the dataset's ``cache/datasets/<stem>/splits`` folder.
Filenames describe the strategy while embedded metadata binds each file to the
exact row layout that it indexes.
"""

from __future__ import annotations

import os
from typing import Callable, Optional

import numpy as np

import config


def _assign_blocks_to_splits(
    block_ids: np.ndarray,
    block_sizes: dict[int, int],
    fractions: dict[str, float],
    seed: int,
) -> dict[str, np.ndarray]:
    """Greedily assign atomic blocks to splits by target row fraction.

    Each block is given to the split with the largest remaining row deficit.
    Blocks are considered largest-first (ties broken by ``seed``) so a giant
    connected component lands in the largest target split instead of overflowing
    a small holdout.

    Args:
        block_ids: Integer array ``(n_rows,)`` giving each row's block id.
        block_sizes: Mapping from block id to number of rows in that block.
        fractions: Mapping from split name to target fraction of rows. Fractions
            should sum to 1.0.
        seed: Random seed controlling tie-breaking among equal-sized blocks.

    Returns:
        A mapping from split name to a sorted ``int64`` array of row indices.
    """
    if not block_sizes:
        return {name: np.empty(0, dtype=np.int64) for name in fractions}

    rng = np.random.default_rng(seed)
    blocks = np.fromiter(block_sizes, dtype=np.int64, count=len(block_sizes))
    rng.shuffle(blocks)
    sizes = np.fromiter(
        (block_sizes[int(b)] for b in blocks), dtype=np.int64, count=blocks.shape[0]
    )
    blocks = blocks[np.argsort(-sizes, kind="stable")]

    n_rows = int(block_ids.shape[0])
    names = list(fractions)
    targets = {name: fractions[name] * n_rows for name in names}
    assigned = {name: 0.0 for name in names}
    block_to_code = {}
    name_to_code = {name: i for i, name in enumerate(names)}

    for block in blocks:
        bid = int(block)
        best = max(names, key=lambda name: targets[name] - assigned[name])
        block_to_code[bid] = name_to_code[best]
        assigned[best] += block_sizes[bid]

    max_id = int(blocks.max())
    remap = np.full(max_id + 1, -1, dtype=np.int8)
    for bid, code in block_to_code.items():
        remap[bid] = code
    labels = remap[block_ids]
    return {
        name: np.flatnonzero(labels == code).astype(np.int64)
        for name, code in name_to_code.items()
    }


def _bipartite_component_ids(
    protein_groups: np.ndarray,
    scaffold_groups: np.ndarray,
) -> np.ndarray:
    """Label each row by connected component in the protein–scaffold graph.

    Args:
        protein_groups: Integer array ``(n_rows,)`` of protein group ids.
        scaffold_groups: Integer array ``(n_rows,)`` of Murcko scaffold ids.

    Returns:
        Integer array ``(n_rows,)`` of component ids.

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
    if protein_groups.shape[0] == 0:
        return np.empty(0, dtype=np.int64)

    _, p_inv = np.unique(protein_groups, return_inverse=True)
    _, s_inv = np.unique(scaffold_groups, return_inverse=True)
    n_p = int(p_inv.max()) + 1
    n_s = int(s_inv.max()) + 1
    parent = np.arange(n_p + n_s, dtype=np.int64)

    def find(x: int) -> int:
        """Return the union-find root of node ``x`` with path compression."""
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = int(parent[x])
        return x

    pairs = np.unique(np.stack((p_inv, s_inv), axis=1), axis=0)
    for pi, si in pairs:
        ra, rb = find(int(pi)), find(int(si) + n_p)
        if ra != rb:
            parent[rb] = ra

    for i in range(parent.shape[0]):
        parent[i] = find(i)
    return parent[p_inv]


def cold_protein_split(
    groups: np.ndarray,
    fractions: dict[str, float],
    seed: int = config.RANDOM_SEED,
) -> dict[str, np.ndarray]:
    """Partition row indices into protein-disjoint splits.

    Args:
        groups: Integer array ``(n_rows,)`` giving each row's protein group id.
        fractions: Mapping from split name to target fraction of rows.
        seed: Random seed controlling block tie-breaking.

    Returns:
        A mapping from split name to a sorted ``int64`` array of row indices.
        Every protein's rows land entirely within a single split.
    """
    groups = np.asarray(groups, dtype=np.int64)
    unique_groups, group_counts = np.unique(groups, return_counts=True)
    count_by_group = {int(g): int(c) for g, c in zip(unique_groups, group_counts)}
    return _assign_blocks_to_splits(groups, count_by_group, fractions, seed)


def double_cold_split(
    protein_groups: np.ndarray,
    scaffold_groups: np.ndarray,
    fractions: dict[str, float],
    seed: int = config.RANDOM_SEED,
) -> dict[str, np.ndarray]:
    """Partition rows so splits share neither proteins nor scaffolds.

    Builds connected components of the protein–scaffold co-occurrence graph,
    then greedily assigns components to splits by row count.

    Args:
        protein_groups: Integer array ``(n_rows,)`` of protein group ids.
        scaffold_groups: Integer array ``(n_rows,)`` of Murcko scaffold ids.
        fractions: Mapping from split name to target fraction of rows.
        seed: Random seed controlling component tie-breaking.

    Returns:
        A mapping from split name to a sorted ``int64`` array of row indices.
    """
    block_ids = _bipartite_component_ids(protein_groups, scaffold_groups)
    unique_blocks, counts = np.unique(block_ids, return_counts=True)
    block_sizes = {int(b): int(c) for b, c in zip(unique_blocks, counts)}
    return _assign_blocks_to_splits(block_ids, block_sizes, fractions, seed)


def _split_path(signature: str, split_type: str, seed: int, splits_dir: str) -> str:
    """Build the cache path for a saved split.

    Args:
        signature: Dataset signature (stored inside the file, not its name).
        split_type: Human-readable split label (e.g. ``"train_test"``).
        seed: Random seed used.
        splits_dir: Directory holding split files.

    Returns:
        The ``.npz`` path for this split configuration.
    """
    del signature
    safe_type = split_type.replace("/", "_").replace(" ", "_")
    return os.path.join(splits_dir, f"{safe_type}__seed{seed}.npz")


def _validate_split(split: dict[str, np.ndarray], n_rows: int) -> dict[str, np.ndarray]:
    """Validate and normalize a complete row partition.

    Args:
        split: Mapping from split names to index arrays.
        n_rows: Number of rows in the indexed dataset.

    Returns:
        Mapping with sorted ``int64`` arrays.

    Raises:
        ValueError: If indices are malformed, duplicated, overlapping,
            out-of-range, or do not cover every row exactly once.
    """
    normalized: dict[str, np.ndarray] = {}
    seen = np.zeros(n_rows, dtype=np.uint8)
    for name, values in split.items():
        array = np.asarray(values)
        if array.ndim != 1 or not np.issubdtype(array.dtype, np.integer):
            raise ValueError(f"Split {name!r} must be a one-dimensional integer array.")
        array = array.astype(np.int64, copy=False)
        if array.size and (int(array.min()) < 0 or int(array.max()) >= n_rows):
            raise ValueError(f"Split {name!r} contains indices outside 0..{n_rows - 1}.")
        if np.unique(array).size != array.size:
            raise ValueError(f"Split {name!r} contains duplicate indices.")
        if array.size and np.any(seen[array]):
            raise ValueError(f"Split {name!r} overlaps another split.")
        seen[array] = 1
        normalized[name] = np.sort(array)
    if int(seen.sum()) != n_rows:
        raise ValueError(
            f"Split covers {int(seen.sum())} of {n_rows} dataset rows."
        )
    return normalized


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
    n_rows = sum(int(np.asarray(values).size) for values in split.values())
    normalized = _validate_split(split, n_rows)
    os.makedirs(splits_dir, exist_ok=True)
    path = _split_path(signature, split_type, seed, splits_dir)
    temporary = f"{path}.tmp-{os.getpid()}.npz"
    np.savez(
        temporary,
        **normalized,
        __signature__=np.asarray(signature),
        __split_type__=np.asarray(split_type),
        __seed__=np.asarray(seed, dtype=np.int64),
        __n_rows__=np.asarray(n_rows, dtype=np.int64),
    )
    os.replace(temporary, path)
    return path


def load_split(
    signature: str,
    split_type: str,
    seed: int = config.RANDOM_SEED,
    splits_dir: str = config.SPLITS_DIR,
    n_rows: Optional[int] = None,
) -> Optional[dict[str, np.ndarray]]:
    """Load a previously saved split if present.

    Args:
        signature: Dataset signature.
        split_type: Human-readable split label.
        seed: Random seed used.
        splits_dir: Directory to read from.
        n_rows: Expected dataset row count. When omitted, use file metadata.

    Returns:
        The split mapping, or ``None`` if no cached split exists.
    """
    path = _split_path(signature, split_type, seed, splits_dir)
    if not os.path.exists(path):
        return None
    with np.load(path) as data:
        if "__signature__" not in data.files:
            return None
        if str(data["__signature__"].item()) != signature:
            return None
        if str(data["__split_type__"].item()) != split_type:
            return None
        if int(data["__seed__"].item()) != seed:
            return None
        stored_n_rows = int(data["__n_rows__"].item())
        expected_n_rows = stored_n_rows if n_rows is None else int(n_rows)
        if stored_n_rows != expected_n_rows:
            return None
        split = {
            name: data[name]
            for name in data.files
            if not name.startswith("__")
        }
    return _validate_split(split, expected_n_rows)


def _get_or_create_split(
    *,
    n_rows: int,
    signature: str,
    split_type: str,
    seed: int,
    splits_dir: str,
    verbose: bool,
    builder: Callable[[], dict[str, np.ndarray]],
    created_detail: str = "",
) -> dict[str, np.ndarray]:
    """Return a cached split or build, save, and return a new one.

    Args:
        n_rows: Expected dataset row count.
        signature: Dataset signature.
        split_type: Human-readable split label.
        seed: Random seed used for the cache key.
        splits_dir: Directory for split files.
        verbose: If ``True``, print reuse/create messages.
        builder: Zero-arg callable that builds the split mapping.
        created_detail: Optional extra text appended to the create log line.

    Returns:
        The split mapping from name to row-index array.
    """
    existing = load_split(signature, split_type, seed, splits_dir, n_rows=n_rows)
    if existing is not None:
        if verbose:
            sizes = ", ".join(f"{k}={len(v)}" for k, v in existing.items())
            print(f"[split] reusing {split_type} ({sizes})")
        return existing

    split = builder()
    save_split(split, signature, split_type, seed, splits_dir)
    if verbose:
        sizes = ", ".join(f"{k}={len(v)}" for k, v in split.items())
        suffix = f": {created_detail} | {sizes}" if created_detail else f" ({sizes})"
        print(f"[split] created {split_type}{suffix}")
    return split


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
    return _get_or_create_split(
        n_rows=int(np.asarray(protein_groups).shape[0]),
        signature=signature,
        split_type=split_type,
        seed=seed,
        splits_dir=splits_dir,
        verbose=verbose,
        builder=lambda: double_cold_split(
            protein_groups, scaffold_groups, fractions, seed
        ),
    )


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
        f"double_cold__test{100.0 * test_fraction:g}pct",
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
        (
            f"double_cold__val{100.0 * val_fraction:g}pct"
            f"__test{100.0 * test_fraction:g}pct"
        ),
        seed,
        splits_dir,
        verbose,
    )


def year_cutoffs_from_fractions(
    years: np.ndarray,
    val_fraction: float,
    test_fraction: float,
) -> tuple[int, int]:
    """Derive publication-year cutoffs from target row fractions.

    Only rows with ``year >= 0`` participate in the fraction calculation.
    Cutoffs snap to discrete year boundaries so actual split sizes may differ
    slightly from the requested fractions.

    Args:
        years: Per-row years; missing values should be ``-1``.
        val_fraction: Target fraction of *dated* rows for validation.
        test_fraction: Target fraction of *dated* rows for testing.

    Returns:
        ``(val_year, test_year)`` such that train is ``year <= val_year``,
        val is ``val_year < year <= test_year``, and test is ``year > test_year``.

    Raises:
        ValueError: If there are no dated rows, fractions are invalid, or
            cutoffs cannot be resolved.
    """
    if val_fraction < 0.0 or test_fraction < 0.0:
        raise ValueError("val_fraction and test_fraction must be non-negative.")
    if val_fraction + test_fraction >= 1.0:
        raise ValueError("val_fraction + test_fraction must be < 1.0.")

    years = np.asarray(years)
    dated = years[years >= 0]
    if dated.size == 0:
        raise ValueError(
            "No dated rows available for a time split. Rebuild features from a "
            "Papyrus Parquet file that includes Year."
        )

    unique_years, counts = np.unique(dated, return_counts=True)
    unique_years = unique_years.astype(np.int64)
    total = int(counts.sum())
    cum = np.cumsum(counts)
    train_target = (1.0 - val_fraction - test_fraction) * total
    pre_test_target = (1.0 - test_fraction) * total

    val_idx = int(np.searchsorted(cum, train_target, side="left"))
    val_idx = min(max(val_idx, 0), unique_years.size - 1)
    test_idx = int(np.searchsorted(cum, pre_test_target, side="left"))
    test_idx = min(max(test_idx, val_idx), unique_years.size - 1)

    val_year = int(unique_years[val_idx])
    test_year = int(unique_years[test_idx])
    if test_year < val_year:
        test_year = val_year
    return val_year, test_year


def time_split(
    years: np.ndarray,
    val_year: int,
    test_year: int,
    *,
    include_val: bool = True,
) -> dict[str, np.ndarray]:
    """Partition rows by publication year cutoffs.

    Missing years (``year < 0``) are assigned to train.

    Args:
        years: Per-row years; missing values should be ``-1``.
        val_year: Inclusive upper bound for the train window among dated rows.
        test_year: Inclusive upper bound for the validation window.
        include_val: If ``False``, merge the val window into train and return
            only ``train`` / ``test`` (for ``train.py`` two-way splits).

    Returns:
        Mapping with ``train`` / ``test``, and ``val`` when ``include_val``.
    """
    years = np.asarray(years)
    missing = years < 0
    train_mask = missing | (years <= val_year)
    val_mask = (~missing) & (years > val_year) & (years <= test_year)
    test_mask = (~missing) & (years > test_year)

    if not include_val:
        train_mask = train_mask | val_mask
        return {
            "train": np.flatnonzero(train_mask).astype(np.int64),
            "test": np.flatnonzero(test_mask).astype(np.int64),
        }

    return {
        "train": np.flatnonzero(train_mask).astype(np.int64),
        "val": np.flatnonzero(val_mask).astype(np.int64),
        "test": np.flatnonzero(test_mask).astype(np.int64),
    }


def get_or_create_time_split(
    years: np.ndarray,
    signature: str,
    val_fraction: float,
    test_fraction: float,
    *,
    include_val: bool = True,
    seed: int = config.RANDOM_SEED,
    splits_dir: str = config.SPLITS_DIR,
    verbose: bool = True,
) -> dict[str, np.ndarray]:
    """Return a cached percentage-based time split or create a new one.

    Args:
        years: Per-row years (``-1`` when missing).
        signature: Dataset signature.
        val_fraction: Target dated-row fraction for validation.
        test_fraction: Target dated-row fraction for testing.
        include_val: If ``False``, return a train/test split only.
        seed: Unused for assignment (deterministic from years) but included in
            the cache key for consistency with other split helpers.
        splits_dir: Directory for split files.
        verbose: If ``True``, print cutoff years and split sizes.

    Returns:
        The split mapping.
    """
    years = np.asarray(years)
    n = int(years.shape[0])
    split_type = (
        f"time__val{100.0 * val_fraction:g}pct"
        f"__test{100.0 * test_fraction:g}pct"
        + ("" if include_val else "__val-merged")
    )
    existing = load_split(signature, split_type, seed, splits_dir, n_rows=n)
    if existing is not None:
        if verbose:
            sizes = ", ".join(f"{k}={len(v)}" for k, v in existing.items())
            print(f"[split] reusing {split_type} ({sizes})")
        return existing

    val_year, test_year = year_cutoffs_from_fractions(years, val_fraction, test_fraction)
    split = time_split(years, val_year, test_year, include_val=include_val)
    save_split(split, signature, split_type, seed, splits_dir)
    if verbose:
        sizes = ", ".join(
            f"{k}={len(v)} ({100.0 * len(v) / max(n, 1):.1f}%)" for k, v in split.items()
        )
        print(
            f"[split] created {split_type}: val_year={val_year} test_year={test_year} "
            f"missing_year={int(np.sum(years < 0))} | {sizes}"
        )
    return split
