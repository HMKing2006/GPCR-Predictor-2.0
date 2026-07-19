"""Scaffold-cold and percentage time splits for ligand multilabel datasets.

Outer folds can use independent ``scaffold`` / ``time`` strategies: test is
carved first, then validation from the remainder (when a validation fold is
requested). Pure helpers are kept local so this package can be deleted without
depending on pair split APIs.
"""

from __future__ import annotations

import os
from typing import Callable, Optional

import numpy as np

from src.multilabel import config as ml_config

# Supported outer-fold strategies for multilabel ``--test-split`` /
# ``--validation-split``.
SPLIT_STRATEGIES: tuple[str, ...] = ml_config.SPLIT_STRATEGIES


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
        fractions: Mapping from split name to target fraction of rows.
        seed: Random seed controlling tie-breaking among equal-sized blocks.

    Returns:
        Mapping from split name to a sorted ``int64`` array of row indices.
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


def scaffold_cold_split(
    scaffold_groups: np.ndarray,
    fractions: dict[str, float],
    seed: int = ml_config.RANDOM_SEED,
) -> dict[str, np.ndarray]:
    """Partition ligands so Murcko scaffolds do not cross splits.

    Args:
        scaffold_groups: Integer array ``(n_rows,)`` of scaffold ids.
        fractions: Mapping from split name to target fraction of rows.
        seed: Random seed controlling block tie-breaking.

    Returns:
        Mapping from split name to sorted ``int64`` row indices.
    """
    groups = np.asarray(scaffold_groups, dtype=np.int64)
    unique_groups, group_counts = np.unique(groups, return_counts=True)
    count_by_group = {int(g): int(c) for g, c in zip(unique_groups, group_counts)}
    return _assign_blocks_to_splits(groups, count_by_group, fractions, seed)


def year_cutoffs_from_fractions(
    years: np.ndarray,
    val_fraction: float,
    test_fraction: float,
) -> tuple[int, int]:
    """Derive publication-year cutoffs from target dated-row fractions.

    Args:
        years: Per-row years; missing values should be ``-1``.
        val_fraction: Target fraction of dated rows for validation.
        test_fraction: Target fraction of dated rows for testing.

    Returns:
        ``(val_year, test_year)`` cutoffs.

    Raises:
        ValueError: If there are no dated rows or fractions are invalid.
    """
    if val_fraction < 0.0 or test_fraction < 0.0:
        raise ValueError("val_fraction and test_fraction must be non-negative.")
    if val_fraction + test_fraction >= 1.0:
        raise ValueError("val_fraction + test_fraction must be < 1.0.")

    years = np.asarray(years)
    dated = years[years >= 0]
    if dated.size == 0:
        raise ValueError(
            "No dated ligands available for a time split. Rebuild multilabel "
            "prepared tables from a Papyrus source that includes Year."
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
    """Partition ligands by publication year cutoffs.

    Missing years (``year < 0``) are assigned to train.

    Args:
        years: Per-row years; missing values should be ``-1``.
        val_year: Inclusive upper bound for the train window among dated rows.
        test_year: Inclusive upper bound for the validation window.
        include_val: If ``False``, merge the val window into train.

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


def _split_path(split_type: str, seed: int, splits_dir: str) -> str:
    """Build the cache path for a saved split.

    Args:
        split_type: Human-readable split label.
        seed: Random seed used.
        splits_dir: Directory holding split files.

    Returns:
        The ``.npz`` path for this split configuration.
    """
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
        ValueError: If indices are malformed or do not cover every row once.
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
        raise ValueError(f"Split covers {int(seen.sum())} of {n_rows} dataset rows.")
    return normalized


def save_split(
    split: dict[str, np.ndarray],
    signature: str,
    split_type: str,
    seed: int,
    splits_dir: str,
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
    path = _split_path(split_type, seed, splits_dir)
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
    seed: int,
    splits_dir: str,
    n_rows: Optional[int] = None,
) -> Optional[dict[str, np.ndarray]]:
    """Load a previously saved split if present and valid.

    Args:
        signature: Dataset signature.
        split_type: Human-readable split label.
        seed: Random seed used.
        splits_dir: Directory to read from.
        n_rows: Expected dataset row count.

    Returns:
        The split mapping, or ``None`` if no cached split exists.
    """
    path = _split_path(split_type, seed, splits_dir)
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
        split = {name: data[name] for name in data.files if not name.startswith("__")}
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


def get_or_create_scaffold_cold_split(
    scaffold_groups: np.ndarray,
    fractions: dict[str, float],
    signature: str,
    split_type: str,
    seed: int = ml_config.RANDOM_SEED,
    splits_dir: str = "",
    verbose: bool = True,
) -> dict[str, np.ndarray]:
    """Return a saved scaffold-cold split or create a new one.

    Args:
        scaffold_groups: Per-row scaffold group ids.
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
        n_rows=int(np.asarray(scaffold_groups).shape[0]),
        signature=signature,
        split_type=split_type,
        seed=seed,
        splits_dir=splits_dir,
        verbose=verbose,
        builder=lambda: scaffold_cold_split(scaffold_groups, fractions, seed),
    )


def train_test_scaffold_split(
    scaffold_groups: np.ndarray,
    signature: str,
    test_fraction: float = ml_config.TEST_FRACTION,
    seed: int = ml_config.RANDOM_SEED,
    splits_dir: str = "",
    verbose: bool = True,
) -> dict[str, np.ndarray]:
    """Create/reuse an 80/20 scaffold-cold train/test split.

    Args:
        scaffold_groups: Per-row scaffold group ids.
        signature: Dataset signature.
        test_fraction: Fraction of rows for the test set.
        seed: Random seed.
        splits_dir: Directory for split files.
        verbose: If ``True``, print split info.

    Returns:
        Mapping with keys ``"train"`` and ``"test"``.
    """
    fractions = {"train": 1.0 - test_fraction, "test": test_fraction}
    return get_or_create_scaffold_cold_split(
        scaffold_groups,
        fractions,
        signature,
        f"scaffold_cold__test{100.0 * test_fraction:g}pct",
        seed,
        splits_dir,
        verbose,
    )


def get_or_create_time_split(
    years: np.ndarray,
    signature: str,
    val_fraction: float,
    test_fraction: float,
    *,
    include_val: bool = False,
    seed: int = ml_config.RANDOM_SEED,
    splits_dir: str = "",
    verbose: bool = True,
) -> dict[str, np.ndarray]:
    """Return a cached percentage-based time split or create a new one.

    Args:
        years: Per-row years (``-1`` when missing).
        signature: Dataset signature.
        val_fraction: Target dated-row fraction for validation.
        test_fraction: Target dated-row fraction for testing.
        include_val: If ``False``, return a train/test split only.
        seed: Included in the cache key for consistency with other helpers.
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


def _validate_strategy(name: str, strategy: str) -> str:
    """Return ``strategy`` if it is a known multilabel split strategy.

    Args:
        name: Flag or argument name for error messages.
        strategy: Requested strategy token.

    Returns:
        The validated strategy string.

    Raises:
        ValueError: If ``strategy`` is not in :data:`SPLIT_STRATEGIES`.
    """
    strategy = str(strategy).strip()
    if strategy not in SPLIT_STRATEGIES:
        raise ValueError(
            f"Unknown {name}={strategy!r}; choose one of {SPLIT_STRATEGIES}."
        )
    return strategy


def _partition_pool(
    pool: np.ndarray,
    strategy: str,
    fractions: dict[str, float],
    *,
    scaffold_groups: np.ndarray,
    years: Optional[np.ndarray],
    seed: int,
) -> dict[str, np.ndarray]:
    """Partition a pool of global ligand indices with one strategy.

    Args:
        pool: Global row indices to partition.
        strategy: One of :data:`SPLIT_STRATEGIES`.
        fractions: Mapping from fold name to target fraction within ``pool``.
        scaffold_groups: Full-dataset scaffold ids (required for scaffold).
        years: Full-dataset years (required for time); ``-1`` when missing.
        seed: Random seed for scaffold assignment.

    Returns:
        Mapping from fold name to sorted global row-index arrays.

    Raises:
        ValueError: If a required array is missing for ``strategy``.
    """
    pool = np.asarray(pool, dtype=np.int64)
    if pool.size == 0:
        return {name: np.empty(0, dtype=np.int64) for name in fractions}

    strategy = _validate_strategy("strategy", strategy)
    scaffold_groups = np.asarray(scaffold_groups)

    if strategy == "scaffold":
        local = scaffold_cold_split(scaffold_groups[pool], fractions, seed=seed)
    else:  # time
        if years is None:
            raise ValueError("time splits require years.")
        years_arr = np.asarray(years)
        names = set(fractions)
        if names == {"train", "test"}:
            val_year, test_year = year_cutoffs_from_fractions(
                years_arr[pool], 0.0, float(fractions["test"])
            )
            local = time_split(
                years_arr[pool], val_year, test_year, include_val=False
            )
        elif names == {"train", "val"}:
            val_year, test_year = year_cutoffs_from_fractions(
                years_arr[pool], 0.0, float(fractions["val"])
            )
            two_way = time_split(
                years_arr[pool], val_year, test_year, include_val=False
            )
            local = {"train": two_way["train"], "val": two_way["test"]}
        elif names == {"train", "val", "test"}:
            val_year, test_year = year_cutoffs_from_fractions(
                years_arr[pool],
                float(fractions["val"]),
                float(fractions["test"]),
            )
            local = time_split(
                years_arr[pool], val_year, test_year, include_val=True
            )
        else:
            raise ValueError(
                f"Unsupported time-split fraction keys: {sorted(fractions)}"
            )

    return {
        name: np.sort(pool[local_idx]).astype(np.int64)
        for name, local_idx in local.items()
    }


def _build_nested_split(
    *,
    test_split: str,
    validation_split: str,
    scaffold_groups: np.ndarray,
    years: Optional[np.ndarray],
    val_fraction: float,
    test_fraction: float,
    include_val: bool,
    seed: int,
) -> dict[str, np.ndarray]:
    """Build train/val/test by carving test first, then val from the remainder.

    Args:
        test_split: Strategy for the test fold.
        validation_split: Strategy for the validation fold (ignored when
            ``include_val`` is ``False``).
        scaffold_groups: Per-row Murcko scaffold ids.
        years: Per-row years (for time); ``-1`` when missing.
        val_fraction: Target fraction of *all* rows for validation.
        test_fraction: Target fraction of *all* rows for test.
        include_val: If ``False``, return only train/test (remainder → train).
        seed: Random seed.

    Returns:
        Split mapping with ``train`` / ``test``, and ``val`` when requested.

    Raises:
        ValueError: If fractions are invalid or a required array is missing.
    """
    test_split = _validate_strategy("test_split", test_split)
    validation_split = _validate_strategy("validation_split", validation_split)
    if val_fraction < 0.0 or test_fraction < 0.0:
        raise ValueError("val_fraction and test_fraction must be non-negative.")
    if include_val and val_fraction + test_fraction >= 1.0:
        raise ValueError("val_fraction + test_fraction must be < 1.0.")
    if not include_val and test_fraction >= 1.0:
        raise ValueError("test_fraction must be < 1.0.")

    scaffold_groups = np.asarray(scaffold_groups)
    n = int(scaffold_groups.shape[0])
    all_idx = np.arange(n, dtype=np.int64)

    # Matched time strategies: keep legacy year-window semantics (val_fraction
    # sizes the middle year band even when it is later merged into train).
    if test_split == "time" and validation_split == "time":
        if years is None:
            raise ValueError("time splits require years.")
        val_year, test_year = year_cutoffs_from_fractions(
            np.asarray(years), val_fraction, test_fraction
        )
        return time_split(
            np.asarray(years), val_year, test_year, include_val=include_val
        )

    if include_val and test_split == validation_split:
        fractions = {
            "train": 1.0 - val_fraction - test_fraction,
            "val": val_fraction,
            "test": test_fraction,
        }
        return _partition_pool(
            all_idx,
            test_split,
            fractions,
            scaffold_groups=scaffold_groups,
            years=years,
            seed=seed,
        )

    if (not include_val) and test_split == validation_split:
        fractions = {"train": 1.0 - test_fraction, "test": test_fraction}
        return _partition_pool(
            all_idx,
            test_split,
            fractions,
            scaffold_groups=scaffold_groups,
            years=years,
            seed=seed,
        )

    rest_frac = 1.0 - test_fraction
    step1 = _partition_pool(
        all_idx,
        test_split,
        {"train": rest_frac, "test": test_fraction},
        scaffold_groups=scaffold_groups,
        years=years,
        seed=seed,
    )
    pool = step1["train"]
    test_idx = step1["test"]

    if not include_val:
        return {"train": pool, "test": test_idx}

    pool_n = int(pool.shape[0])
    if pool_n == 0:
        return {
            "train": np.empty(0, dtype=np.int64),
            "val": np.empty(0, dtype=np.int64),
            "test": test_idx,
        }
    val_rel = float(val_fraction) * n / float(pool_n)
    val_rel = float(min(max(val_rel, 0.0), 0.99))
    step2 = _partition_pool(
        pool,
        validation_split,
        {"train": 1.0 - val_rel, "val": val_rel},
        scaffold_groups=scaffold_groups,
        years=years,
        seed=seed + 1,
    )
    return {
        "train": step2["train"],
        "val": step2["val"],
        "test": test_idx,
    }


def get_or_create_nested_split(
    *,
    scaffold_groups: np.ndarray,
    signature: str,
    test_split: str = ml_config.DEFAULT_TEST_SPLIT,
    validation_split: str = ml_config.DEFAULT_VALIDATION_SPLIT,
    val_fraction: float = ml_config.VAL_FRACTION,
    test_fraction: float = ml_config.TEST_FRACTION,
    include_val: bool = False,
    years: Optional[np.ndarray] = None,
    seed: int = ml_config.RANDOM_SEED,
    splits_dir: str = "",
    verbose: bool = True,
) -> dict[str, np.ndarray]:
    """Return a cached nested train/val/test split or create a new one.

    Args:
        scaffold_groups: Per-row Murcko scaffold ids.
        signature: Dataset signature for cache binding.
        test_split: Strategy for the test fold (``scaffold`` or ``time``).
        validation_split: Strategy for the validation fold. Ignored when
            ``include_val`` is ``False``. When equal to ``test_split``, a
            single joint partition is used.
        val_fraction: Target fraction of all rows for validation.
        test_fraction: Target fraction of all rows for test.
        include_val: If ``False``, return train/test only.
        years: Required when either strategy is ``time``; ``-1`` if missing.
        seed: Random seed for scaffold assignment.
        splits_dir: Directory for split cache files.
        verbose: If ``True``, print reuse/create messages.

    Returns:
        Split mapping with ``train`` / ``test``, and ``val`` when requested.

    Raises:
        ValueError: If strategies/fractions are invalid or required inputs for
            a strategy are missing.
    """
    test_split = _validate_strategy("test_split", test_split)
    validation_split = _validate_strategy("validation_split", validation_split)
    needs_time = test_split == "time" or (
        include_val and validation_split == "time"
    )
    if needs_time and years is None:
        raise ValueError(
            "Time split requires dated ligands. Rebuild multilabel prepared "
            "tables from a Papyrus source that includes Year."
        )
    if needs_time:
        years_arr = np.asarray(years)
        if int(np.sum(years_arr >= 0)) == 0:
            raise ValueError(
                "Time split requires dated ligands. Rebuild multilabel "
                "prepared tables from a Papyrus source that includes Year."
            )
    else:
        years_arr = None

    scaffold_groups = np.asarray(scaffold_groups)
    n = int(scaffold_groups.shape[0])
    effective_val = validation_split if include_val else test_split
    split_type = (
        f"nested__test-{test_split}__val-{effective_val}"
        f"__val{100.0 * val_fraction:g}pct"
        f"__test{100.0 * test_fraction:g}pct"
        + ("" if include_val else "__val-merged")
    )

    def _builder() -> dict[str, np.ndarray]:
        """Build the nested multilabel split for caching."""
        return _build_nested_split(
            test_split=test_split,
            validation_split=validation_split,
            scaffold_groups=scaffold_groups,
            years=years_arr,
            val_fraction=val_fraction,
            test_fraction=test_fraction,
            include_val=include_val,
            seed=seed,
        )

    return _get_or_create_split(
        n_rows=n,
        signature=signature,
        split_type=split_type,
        seed=seed,
        splits_dir=splits_dir,
        verbose=verbose,
        builder=_builder,
        created_detail=(
            f"test_split={test_split} validation_split={validation_split}"
        ),
    )
