"""Ligand-only feature snapshots for multilabel family/target training."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import shutil
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterator, Optional, Sequence

import numpy as np
import pyarrow.parquet as pq

import config
from src.featurize import MurckoScaffoldResolver
from src.ligand_repr import CompositeLigandFeaturizer, canonical_ligand_repr, parse_ligand_repr
from src.lmdb_cache import EmbeddingCache, ScaffoldCache
from src.multilabel import config as ml_config
from src.multilabel.vocab import label_index, load_vocab

_ROW_CHUNK: int = 50_000
_PROGRESS_EVERY: int = 50_000


def _dataset_stem(source_path: str) -> str:
    """Return a readable, filesystem-safe dataset name.

    Args:
        source_path: Prepared Parquet path.

    Returns:
        Sanitized basename without its extension.
    """
    stem = os.path.splitext(os.path.basename(source_path))[0]
    return re.sub(r"[^0-9A-Za-z._-]+", "_", stem).strip("._") or "dataset"


def _source_identity(source_path: str) -> dict[str, Any]:
    """Describe the prepared source file for cache invalidation.

    Args:
        source_path: Prepared Parquet path.

    Returns:
        Resolved path, byte size and nanosecond modification time.
    """
    resolved = os.path.realpath(source_path)
    stat = os.stat(resolved)
    return {
        "path": resolved,
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _row_signature(
    source: dict[str, Any],
    vocab_path: str,
    task: str,
    activity_threshold_nm: float,
) -> str:
    """Hash every setting that changes row-level snapshot contents.

    Args:
        source: Source identity from :func:`_source_identity`.
        vocab_path: Vocabulary JSON path.
        task: ``family`` or ``target``.
        activity_threshold_nm: Binder cutoff used when the table was built.

    Returns:
        Sixteen-character hexadecimal signature.
    """
    vocab_identity = _source_identity(vocab_path)
    payload = {
        "storage_version": ml_config.STORAGE_VERSION,
        "source": source,
        "vocab": vocab_identity,
        "task": task,
        "activity_threshold_nm": float(activity_threshold_nm),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha1(encoded).hexdigest()[:16]


def _embedding_alias(identifier: str) -> str:
    """Create a short readable alias for a ligand embedding artifact.

    Args:
        identifier: Exact model id or ligand component cache id.

    Returns:
        Filesystem-safe alias such as ``MolFormerXL`` or ``morgan``.
    """
    lowered = identifier.lower()
    if "molformer-xl" in lowered:
        return "MolFormerXL"
    if identifier.startswith("morgan_"):
        return "morgan"
    if identifier.startswith("avalon_"):
        return "avalon"
    if identifier == "rdkit_descriptors":
        return "descriptors"
    basename = identifier.rstrip("/").rsplit("/", 1)[-1]
    alias = re.sub(r"[^0-9A-Za-z]+", "_", basename).strip("_")
    return alias or "ligand"


def _atomic_json(path: str, payload: dict[str, Any]) -> None:
    """Atomically write JSON metadata.

    Args:
        path: Final JSON path.
        payload: JSON-serializable mapping.

    Returns:
        None.
    """
    temporary = f"{path}.tmp-{os.getpid()}"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


@contextmanager
def _dataset_lock(dataset_directory: str) -> Iterator[None]:
    """Serialize builders that target the same dataset cache.

    Args:
        dataset_directory: Dataset-scoped cache directory.

    Yields:
        None after acquiring the advisory file lock.
    """
    os.makedirs(dataset_directory, exist_ok=True)
    lock_path = os.path.join(dataset_directory, ".features.lock")
    with open(lock_path, "a+b") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _save_array(directory: str, name: str, array: np.ndarray) -> None:
    """Save one NumPy array in a snapshot build directory.

    Args:
        directory: Destination directory.
        name: Filename ending in ``.npy``.
        array: Array to save.

    Returns:
        None.
    """
    np.save(os.path.join(directory, name), array, allow_pickle=False)


def _open_array(path: str, expected_shape: tuple[int, ...], dtype: np.dtype[Any]) -> np.ndarray:
    """Open and validate a NumPy array without loading it into RAM.

    Args:
        path: ``.npy`` path.
        expected_shape: Required array shape.
        dtype: Required NumPy dtype.

    Returns:
        Read-only memory-mapped array.

    Raises:
        ValueError: If shape or dtype differs.
    """
    array = np.load(path, mmap_mode="r", allow_pickle=False)
    if array.shape != expected_shape or array.dtype != np.dtype(dtype):
        raise ValueError(
            f"Invalid cache array {path}: expected {expected_shape} {np.dtype(dtype)}, "
            f"found {array.shape} {array.dtype}."
        )
    return array


@dataclass(slots=True)
class _IndexedLigands:
    """In-memory result of one ligand prepared-table index pass."""

    ligands: list[str]
    ligand_ids: np.ndarray
    activity_labels: np.ndarray
    scaffold_groups: np.ndarray
    years: np.ndarray
    n_scaffolds: int
    n_labels: int


def _index_prepared_table(
    source_path: str,
    vocab: Sequence[str],
    label_column: str,
    verbose: bool,
) -> _IndexedLigands:
    """Read a ligand prepared Parquet and build arrays + scaffold ids.

    Args:
        source_path: Ligand multilabel prepared Parquet.
        vocab: Ordered vocabulary.
        label_column: ``family_labels`` or ``target_labels``.
        verbose: Whether to print progress.

    Returns:
        Indexed ligands and label matrix.

    Raises:
        ValueError: If the table is empty or missing required columns.
    """
    table = pq.read_table(source_path)
    names = set(table.column_names)
    required = {"Ligand SMILES", label_column}
    missing = required - names
    if missing:
        raise ValueError(f"{source_path} missing columns: {sorted(missing)}")

    smiles_col = table.column("Ligand SMILES").to_pylist()
    label_col = table.column(label_column).to_pylist()
    year_col = (
        table.column("Year").to_pylist() if "Year" in names else [None] * len(smiles_col)
    )
    if not smiles_col:
        raise ValueError(f"No rows in {source_path}.")

    index = label_index(vocab)
    n = len(smiles_col)
    k = len(vocab)
    labels = np.zeros((n, k), dtype=np.uint8)
    years = np.full(n, -1, dtype=np.int32)
    scaffold_index: dict[str, int] = {}
    scaffold_ids: list[int] = []
    ligands: list[str] = []

    with ScaffoldCache() as scaffold_cache:
        resolver = MurckoScaffoldResolver(cache=scaffold_cache)
        for i, (smiles, label_list, year) in enumerate(zip(smiles_col, label_col, year_col)):
            smiles_text = str(smiles or "").strip()
            if not smiles_text:
                raise ValueError(f"Empty Ligand SMILES at row {i} in {source_path}.")
            ligands.append(smiles_text)
            scaffold = resolver.resolve(smiles_text, i)
            scaffold_ids.append(scaffold_index.setdefault(scaffold, len(scaffold_index)))
            if year is not None and str(year).strip() and str(year).lower() != "nan":
                try:
                    years[i] = int(year)
                except (TypeError, ValueError):
                    years[i] = -1
            if label_list is None:
                continue
            for label in label_list:
                col = index.get(str(label))
                if col is not None:
                    labels[i, col] = 1
            if verbose and (i + 1) % _PROGRESS_EVERY == 0:
                print(f"[features] indexed {i + 1}/{n} ligands", flush=True)
        resolver.flush()

    return _IndexedLigands(
        ligands=ligands,
        ligand_ids=np.arange(n, dtype=np.uint32),
        activity_labels=labels,
        scaffold_groups=np.asarray(scaffold_ids, dtype=np.int32),
        years=years,
        n_scaffolds=len(scaffold_index),
        n_labels=k,
    )


def _write_cached_vectors(
    entities: Sequence[str],
    cache: EmbeddingCache,
    component: Any,
    destination: str,
    dim: int,
    verbose: bool,
) -> None:
    """Write cached unique ligand vectors to an atomic ``.npy``.

    Args:
        entities: Ligand SMILES in dataset-local id order.
        cache: Filled embedding cache.
        component: Object exposing ``vectors_for(entities, cache)``.
        destination: Final ``.npy`` path.
        dim: Vector dimension.
        verbose: Whether to print progress.

    Returns:
        None.
    """
    temporary = f"{destination}.tmp-{os.getpid()}"
    matrix = np.lib.format.open_memmap(
        temporary, mode="w+", dtype=np.float32, shape=(len(entities), dim)
    )
    for start in range(0, len(entities), _ROW_CHUNK):
        chunk = list(entities[start : start + _ROW_CHUNK])
        block = component.vectors_for(chunk, cache)
        matrix[start : start + len(chunk)] = block
        if verbose and (start == 0 or start + _ROW_CHUNK >= len(entities)):
            print(
                f"[features] wrote {min(start + _ROW_CHUNK, len(entities))}/"
                f"{len(entities)} vectors to {os.path.basename(destination)}",
                flush=True,
            )
    matrix.flush()
    del matrix
    os.replace(temporary, destination)


def _manifest_entry(
    path: str,
    identifier: str,
    dimension: int,
    shape: tuple[int, int],
) -> dict[str, Any]:
    """Build metadata for a completed embedding matrix.

    Args:
        path: Matrix path.
        identifier: Exact model or component identifier.
        dimension: Vector dimension.
        shape: Matrix shape.

    Returns:
        Manifest mapping.
    """
    stat = os.stat(path)
    return {
        "identifier": identifier,
        "file": os.path.basename(path),
        "dimension": int(dimension),
        "dtype": "float32",
        "shape": list(shape),
        "size": int(stat.st_size),
    }


def _validate_alias(manifest: dict[str, Any], alias: str, identifier: str) -> None:
    """Reject a readable alias that already refers to another identifier.

    Args:
        manifest: Ligand embedding manifest.
        alias: Proposed short alias.
        identifier: Exact model/component identifier.

    Returns:
        None.

    Raises:
        ValueError: If the alias is already assigned to another identifier.
    """
    existing = manifest.get(alias)
    if existing is not None and existing.get("identifier") != identifier:
        raise ValueError(
            f"Ligand embedding alias {alias!r} maps to both "
            f"{existing.get('identifier')!r} and {identifier!r}."
        )


def _commit_snapshot(temporary: str, destination: str) -> None:
    """Replace a feature snapshot directory after successful validation.

    Args:
        temporary: Complete temporary snapshot.
        destination: Final ``features`` directory.

    Returns:
        None.
    """
    backup = f"{destination}.old-{os.getpid()}"
    if os.path.exists(backup):
        shutil.rmtree(backup)
    if os.path.exists(destination):
        os.replace(destination, backup)
    try:
        os.replace(temporary, destination)
    except Exception:
        if os.path.exists(backup) and not os.path.exists(destination):
            os.replace(backup, destination)
        raise
    if os.path.exists(backup):
        shutil.rmtree(backup)


def _recover_stale_snapshot_artifacts(dataset_directory: str) -> None:
    """Recover an interrupted directory swap and remove stale build folders.

    Args:
        dataset_directory: Dataset-scoped cache directory held under lock.

    Returns:
        None.
    """
    destination = os.path.join(dataset_directory, "features")
    names = os.listdir(dataset_directory)
    backups = sorted(
        (
            os.path.join(dataset_directory, name)
            for name in names
            if name.startswith("features.old-")
        ),
        key=os.path.getmtime,
        reverse=True,
    )
    if not os.path.exists(destination) and backups:
        os.replace(backups[0], destination)
        backups = backups[1:]
    for path in backups:
        if os.path.isdir(path):
            shutil.rmtree(path)
    for name in os.listdir(dataset_directory):
        if name.startswith("features.building-"):
            path = os.path.join(dataset_directory, name)
            if os.path.isdir(path):
                shutil.rmtree(path)


class LigandFeatureView:
    """Array-like, on-demand view over a ligand-only feature snapshot."""

    def __init__(
        self,
        dataset: "LigandFeatureDataset",
        indices: Optional[np.ndarray] = None,
    ) -> None:
        """Open selected ligand embedding matrices and an optional row mapping.

        Args:
            dataset: Feature snapshot descriptor.
            indices: Optional dataset-row indices represented by this view.
        """
        self.dataset = dataset
        self.indices = None if indices is None else np.asarray(indices, dtype=np.int64)
        self._ligand_ids = np.load(dataset.ligand_ids_path, mmap_mode="r")
        self._ligand_embeddings = [
            np.load(path, mmap_mode="r") for path in dataset.ligand_embedding_paths
        ]

    @property
    def shape(self) -> tuple[int, int]:
        """Return the logical matrix shape.

        Returns:
            ``(selected_rows, feature_dimension)``.
        """
        n_rows = self.dataset.n_rows if self.indices is None else len(self.indices)
        return n_rows, self.dataset.n_features

    def subset(self, indices: np.ndarray) -> "LigandFeatureView":
        """Return a zero-copy row subset.

        Args:
            indices: Indices relative to this view.

        Returns:
            New view carrying a composed dataset-row mapping.
        """
        local = np.asarray(indices, dtype=np.int64)
        mapped = local if self.indices is None else self.indices[local]
        return LigandFeatureView(self.dataset, mapped)

    def __getitem__(self, key: object) -> np.ndarray:
        """Gather one or more rows into a dense ligand feature array.

        Args:
            key: Integer, slice or integer index array relative to this view.

        Returns:
            Dense feature row or matrix following NumPy indexing semantics.
        """
        if self.indices is not None:
            rows = np.asarray(self.indices[key], dtype=np.int64)
        elif isinstance(key, slice):
            start, stop, step = key.indices(self.dataset.n_rows)
            rows = np.arange(start, stop, step, dtype=np.int64)
        else:
            rows = np.asarray(key, dtype=np.int64)
            rows = np.where(rows < 0, rows + self.dataset.n_rows, rows)
        scalar = rows.ndim == 0
        rows_1d = rows.reshape(1) if scalar else rows
        ligand_ids = np.asarray(self._ligand_ids[rows_1d], dtype=np.int64)
        blocks = [
            np.asarray(matrix[ligand_ids], dtype=np.float32)
            for matrix in self._ligand_embeddings
        ]
        gathered = (
            blocks[0] if len(blocks) == 1 else np.concatenate(blocks, axis=1)
        )
        return gathered[0] if scalar else gathered


@dataclass(frozen=True, slots=True)
class LigandFeatureDataset:
    """Descriptor for one ligand multilabel feature snapshot."""

    directory: str
    dataset_directory: str
    n_rows: int
    n_features: int
    ligand_dim: int
    n_labels: int
    signature: str
    ligand_aliases: tuple[str, ...]
    task: str
    vocab_path: str
    activity_threshold_nm: float = ml_config.ACTIVITY_THRESHOLD_NM

    @property
    def split_directory(self) -> str:
        """Return the dataset-level split-cache directory."""
        return os.path.join(self.dataset_directory, "splits")

    @property
    def ligand_ids_path(self) -> str:
        """Return the per-row ligand-id path."""
        return os.path.join(self.directory, "ligand_ids.npy")

    @property
    def ligand_embedding_paths(self) -> tuple[str, ...]:
        """Return selected ligand component matrix paths."""
        return tuple(
            os.path.join(self.directory, "ligand_embeddings", f"{alias}.npy")
            for alias in self.ligand_aliases
        )

    def load_y(self) -> np.ndarray:
        """Load multilabel activity matrix.

        Returns:
            ``uint8`` array of shape ``(n_rows, n_labels)``.
        """
        return np.load(os.path.join(self.directory, "activity_labels.npy"))

    def load_scaffold_groups(self) -> np.ndarray:
        """Load per-row Murcko scaffold ids.

        Returns:
            ``int32`` array of shape ``(n_rows,)``.
        """
        return np.load(os.path.join(self.directory, "scaffold_groups.npy"))

    def load_years(self) -> np.ndarray:
        """Load publication years, using ``-1`` for missing values.

        Returns:
            ``int32`` array of shape ``(n_rows,)``.
        """
        return np.load(os.path.join(self.directory, "years.npy"))

    def load_smiles(self) -> list[str]:
        """Load ligand SMILES in row order from ``ligands.txt``.

        Returns:
            List of SMILES strings of length ``n_rows``.
        """
        path = os.path.join(self.directory, "ligands.txt")
        if not os.path.exists(path):
            raise ValueError("Feature cache is missing ligands.txt.")
        with open(path, encoding="utf-8") as handle:
            smiles = [line.rstrip("\n") for line in handle]
        if len(smiles) != self.n_rows:
            raise ValueError(
                f"ligands.txt length {len(smiles)} != n_rows {self.n_rows}."
            )
        return smiles

    def feature_view(self, indices: Optional[np.ndarray] = None) -> LigandFeatureView:
        """Open an on-demand ligand feature view.

        Args:
            indices: Optional dataset-row selection.

        Returns:
            Array-like feature view.
        """
        return LigandFeatureView(self, indices)


def _validate_snapshot(
    directory: str,
    meta: dict[str, Any],
    expected_signature: str,
) -> None:
    """Validate shared row arrays and metadata before cache reuse.

    Args:
        directory: Feature snapshot directory.
        meta: Parsed ``meta.json``.
        expected_signature: Requested row-layout signature.

    Returns:
        None.

    Raises:
        ValueError: If metadata or arrays are stale or malformed.
    """
    if meta.get("storage_version") != ml_config.STORAGE_VERSION:
        raise ValueError("Ligand multilabel feature cache storage version is obsolete.")
    if meta.get("signature") != expected_signature:
        raise ValueError(
            "Feature cache does not match the requested source/configuration; "
            "use --rebuild-features."
        )
    n_rows = int(meta["n_rows"])
    n_ligands = int(meta["n_ligands"])
    n_labels = int(meta["n_labels"])
    ligand_ids = _open_array(
        os.path.join(directory, "ligand_ids.npy"), (n_rows,), np.dtype(np.uint32)
    )
    _open_array(
        os.path.join(directory, "activity_labels.npy"),
        (n_rows, n_labels),
        np.dtype(np.uint8),
    )
    _open_array(
        os.path.join(directory, "scaffold_groups.npy"), (n_rows,), np.dtype(np.int32)
    )
    _open_array(os.path.join(directory, "years.npy"), (n_rows,), np.dtype(np.int32))
    if n_rows and int(ligand_ids.max()) >= n_ligands:
        raise ValueError("Feature cache contains out-of-range ligand ids.")
    embeddings = meta.get("embeddings")
    if not isinstance(embeddings, dict) or not isinstance(embeddings.get("ligand"), dict):
        raise ValueError("Feature cache is missing its ligand embedding manifest.")
    for alias, entry in embeddings["ligand"].items():
        if not isinstance(entry, dict) or not entry.get("identifier"):
            raise ValueError(f"Malformed ligand embedding manifest entry {alias!r}.")
        dimension = int(entry["dimension"])
        filename = str(entry["file"])
        if filename != f"{alias}.npy":
            raise ValueError(
                f"Embedding manifest entry {alias!r} points to unexpected "
                f"filename {filename!r}."
            )
        path = os.path.join(directory, "ligand_embeddings", filename)
        expected_shape = (n_ligands, dimension)
        if entry.get("dtype") != "float32" or entry.get("shape") != list(expected_shape):
            raise ValueError(f"Embedding manifest shape/dtype is invalid for {path}.")
        _open_array(path, expected_shape, np.dtype(np.float32))
        if int(entry.get("size", -1)) != os.path.getsize(path):
            raise ValueError(f"Embedding matrix size changed: {path}.")


def _ensure_requested_embeddings(
    directory: str,
    meta: dict[str, Any],
    indexed: _IndexedLigands,
    ligand_featurizer: CompositeLigandFeaturizer,
    verbose: bool,
) -> tuple[str, ...]:
    """Materialize requested ligand component matrices.

    Args:
        directory: Feature snapshot directory.
        meta: Mutable snapshot metadata.
        indexed: Indexed ligands defining entity order.
        ligand_featurizer: Parsed ligand representation.
        verbose: Whether to print progress.

    Returns:
        Ordered ligand aliases.
    """
    embeddings = meta.setdefault("embeddings", {"ligand": {}})
    ligand_manifest: dict[str, Any] = embeddings.setdefault("ligand", {})
    ligand_featurizer.ensure_cached(indexed.ligands, verbose=verbose)
    ligand_aliases: list[str] = []
    for component in ligand_featurizer.components:
        alias = _embedding_alias(component.cache_id)
        _validate_alias(ligand_manifest, alias, component.cache_id)
        ligand_path = os.path.join(directory, "ligand_embeddings", f"{alias}.npy")
        if alias not in ligand_manifest or not os.path.exists(ligand_path):
            with EmbeddingCache(
                "ligand", component.cache_id, expected_dim=component.dim
            ) as ligand_cache:
                _write_cached_vectors(
                    indexed.ligands,
                    ligand_cache,
                    component,
                    ligand_path,
                    component.dim,
                    verbose,
                )
            ligand_manifest[alias] = _manifest_entry(
                ligand_path,
                component.cache_id,
                component.dim,
                (len(indexed.ligands), component.dim),
            )
        else:
            entry = ligand_manifest[alias]
            _open_array(
                ligand_path,
                (len(indexed.ligands), int(entry["dimension"])),
                np.dtype(np.float32),
            )
        ligand_aliases.append(alias)
    return tuple(ligand_aliases)


def _dataset_from_meta(
    directory: str,
    dataset_directory: str,
    meta: dict[str, Any],
    ligand_aliases: tuple[str, ...],
) -> LigandFeatureDataset:
    """Construct a feature descriptor from validated metadata.

    Args:
        directory: Feature snapshot directory.
        dataset_directory: Parent dataset cache directory.
        meta: Snapshot metadata.
        ligand_aliases: Selected ligand component aliases.

    Returns:
        Ligand feature dataset descriptor.
    """
    ligand_dim = sum(
        int(meta["embeddings"]["ligand"][alias]["dimension"])
        for alias in ligand_aliases
    )
    return LigandFeatureDataset(
        directory=directory,
        dataset_directory=dataset_directory,
        n_rows=int(meta["n_rows"]),
        n_features=ligand_dim,
        ligand_dim=ligand_dim,
        n_labels=int(meta["n_labels"]),
        signature=str(meta["signature"]),
        ligand_aliases=ligand_aliases,
        task=str(meta["task"]),
        vocab_path=str(meta["vocab_path"]),
        activity_threshold_nm=float(meta["activity_threshold_nm"]),
    )


def default_paths_for_task(task: str) -> tuple[str, str]:
    """Return default prepared Parquet and vocab paths for a task.

    Args:
        task: ``family`` or ``target``.

    Returns:
        ``(prepared_path, vocab_path)``.

    Raises:
        ValueError: If ``task`` is not recognized.
    """
    if task == "family":
        return ml_config.FAMILY_PREPARED, ml_config.FAMILY_VOCAB_PATH
    if task == "target":
        return ml_config.TARGET_PREPARED, ml_config.TARGET_VOCAB_PATH
    raise ValueError(f"task must be 'family' or 'target', got {task!r}.")


def build_ligand_features(
    source_path: str,
    vocab_path: str,
    task: str,
    ligand_model: str = ml_config.DEFAULT_LIGAND_MODEL,
    *,
    activity_threshold_nm: float = ml_config.ACTIVITY_THRESHOLD_NM,
    rebuild: bool = False,
    verbose: bool = True,
    device: object = None,
) -> LigandFeatureDataset:
    """Build or reuse a ligand-only multilabel feature snapshot.

    Args:
        source_path: Ligand prepared Parquet path.
        vocab_path: Vocabulary JSON path.
        task: ``family`` or ``target``.
        ligand_model: Ligand representation spec (same tokens as pair training).
        activity_threshold_nm: Binder cutoff recorded in metadata.
        rebuild: If ``True``, rebuild the snapshot from scratch.
        verbose: Whether to print progress.
        device: Optional torch device for HF ligand components.

    Returns:
        Opened ligand feature dataset descriptor.
    """
    label_column = "family_labels" if task == "family" else "target_labels"
    vocab = load_vocab(vocab_path)
    source = _source_identity(source_path)
    signature = _row_signature(source, vocab_path, task, activity_threshold_nm)
    stem = _dataset_stem(source_path)
    dataset_directory = os.path.join(config.DATASETS_CACHE_DIR, stem)
    features_directory = os.path.join(dataset_directory, "features")
    ligand_featurizer = parse_ligand_repr(ligand_model, device=device)
    canonical_repr = canonical_ligand_repr(ligand_model)

    with _dataset_lock(dataset_directory):
        _recover_stale_snapshot_artifacts(dataset_directory)
        meta_path = os.path.join(features_directory, "meta.json")
        if not rebuild and os.path.exists(meta_path):
            with open(meta_path, encoding="utf-8") as handle:
                meta = json.load(handle)
            try:
                _validate_snapshot(features_directory, meta, signature)
                aliases = []
                for component in ligand_featurizer.components:
                    alias = _embedding_alias(component.cache_id)
                    entry = meta["embeddings"]["ligand"].get(alias)
                    if entry is None or entry.get("identifier") != component.cache_id:
                        raise ValueError("Requested ligand embedding is not cached.")
                    aliases.append(alias)
                if verbose:
                    print(
                        f"[features] reusing ligand multilabel snapshot "
                        f"({meta['n_rows']} ligands, K={meta['n_labels']})",
                        flush=True,
                    )
                return _dataset_from_meta(
                    features_directory,
                    dataset_directory,
                    meta,
                    tuple(aliases),
                )
            except ValueError as exc:
                if verbose:
                    print(f"[features] rebuilding snapshot ({exc})", flush=True)

        if verbose:
            print(f"[features] building ligand multilabel snapshot for {stem}", flush=True)
        indexed = _index_prepared_table(source_path, vocab, label_column, verbose)
        temporary = os.path.join(
            dataset_directory, f"features.building-{os.getpid()}"
        )
        if os.path.exists(temporary):
            shutil.rmtree(temporary)
        os.makedirs(temporary)
        os.makedirs(os.path.join(temporary, "ligand_embeddings"))
        _save_array(temporary, "ligand_ids.npy", indexed.ligand_ids)
        _save_array(temporary, "activity_labels.npy", indexed.activity_labels)
        _save_array(temporary, "scaffold_groups.npy", indexed.scaffold_groups)
        _save_array(temporary, "years.npy", indexed.years)

        meta: dict[str, Any] = {
            "storage_version": ml_config.STORAGE_VERSION,
            "signature": signature,
            "task": task,
            "source": source,
            "vocab_path": os.path.realpath(vocab_path),
            "activity_threshold_nm": float(activity_threshold_nm),
            "n_rows": int(indexed.ligand_ids.shape[0]),
            "n_ligands": int(len(indexed.ligands)),
            "n_labels": int(indexed.n_labels),
            "n_scaffolds": int(indexed.n_scaffolds),
            "ligand_model": canonical_repr,
            "embeddings": {"ligand": {}},
        }
        aliases = _ensure_requested_embeddings(
            temporary, meta, indexed, ligand_featurizer, verbose
        )
        smiles_path = os.path.join(temporary, "ligands.txt")
        with open(smiles_path, "w", encoding="utf-8") as handle:
            for smiles in indexed.ligands:
                handle.write(smiles)
                handle.write("\n")
        meta["ligands_file"] = "ligands.txt"
        _atomic_json(os.path.join(temporary, "meta.json"), meta)
        _validate_snapshot(temporary, meta, signature)
        _commit_snapshot(temporary, features_directory)
        if verbose:
            print(
                f"[features] wrote {features_directory} "
                f"({meta['n_rows']} ligands, K={meta['n_labels']})",
                flush=True,
            )
        return _dataset_from_meta(
            features_directory, dataset_directory, meta, aliases
        )
