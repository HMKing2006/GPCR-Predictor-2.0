"""Build compact, dataset-scoped feature snapshots from embedding caches.

Prepared activity rows are indexed once into small per-row arrays. Unique
protein and ligand vectors are copied from the durable LMDB caches into named,
memory-mappable ``.npy`` matrices. :class:`FeatureView` gathers dense batches
on demand, avoiding both the former row-expanded ``X.dat`` and split copies.
"""

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
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

import config
from src.data_prep import binarize_pactivity, iter_prepared_rows
from src.embeddings import HFEmbedder, protein_embedder
from src.ligand_repr import CompositeLigandFeaturizer, canonical_ligand_repr, parse_ligand_repr
from src.lmdb_cache import EmbeddingCache

_ASSAY_TO_INDEX: dict[str, int] = {name: i for i, name in enumerate(config.ASSAY_TYPES)}
_ROW_CHUNK: int = 50_000
_PROGRESS_EVERY: int = 200_000
_STORAGE_VERSION: str = "id_gather_v1"


def murcko_scaffold_key(smiles: str, row_index: int) -> str:
    """Return a Bemis-Murcko scaffold key for a ligand SMILES.

    Args:
        smiles: Canonical ligand SMILES.
        row_index: Zero-based row index used for the orphan fallback key.

    Returns:
        Scaffold SMILES, or a unique row key when scaffolding fails.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return f"__orphan_{row_index}"
    try:
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
    except Exception:
        return f"__orphan_{row_index}"
    return scaffold or f"__orphan_{row_index}"


def assay_onehot(assay_type: str) -> np.ndarray:
    """One-hot encode an assay type in the configured order.

    Args:
        assay_type: One of ``config.ASSAY_TYPES``.

    Returns:
        A ``float32`` vector of length ``len(config.ASSAY_TYPES)``.
    """
    vector = np.zeros(len(config.ASSAY_TYPES), dtype=np.float32)
    vector[_ASSAY_TO_INDEX[assay_type]] = 1.0
    return vector


def assemble_matrix(
    protein_vecs: np.ndarray,
    ligand_vecs: np.ndarray,
    assay_indices: np.ndarray,
    ph: np.ndarray,
    temp: np.ndarray,
    include_assay_context: bool = config.INCLUDE_ASSAY_CONTEXT,
) -> np.ndarray:
    """Concatenate protein, ligand and optional assay features.

    Args:
        protein_vecs: Protein matrix ``(n, protein_dim)``.
        ligand_vecs: Ligand matrix ``(n, ligand_dim)``.
        assay_indices: Per-row assay indices.
        ph: Per-row pH values.
        temp: Per-row temperatures in Celsius.
        include_assay_context: Whether to append assay one-hot, pH and temp.

    Returns:
        Dense ``float32`` feature matrix.
    """
    n_rows = protein_vecs.shape[0]
    protein_dim = protein_vecs.shape[1]
    ligand_end = protein_dim + ligand_vecs.shape[1]
    if not include_assay_context:
        output = np.empty((n_rows, ligand_end), dtype=np.float32)
        output[:, :protein_dim] = protein_vecs
        output[:, protein_dim:ligand_end] = ligand_vecs
        return output

    n_assays = len(config.ASSAY_TYPES)
    output = np.empty(
        (n_rows, ligand_end + n_assays + config.NUM_SCALAR_FEATURES),
        dtype=np.float32,
    )
    output[:, :protein_dim] = protein_vecs
    output[:, protein_dim:ligand_end] = ligand_vecs
    onehot = np.zeros((n_rows, n_assays), dtype=np.float32)
    onehot[np.arange(n_rows), assay_indices] = 1.0
    output[:, ligand_end : ligand_end + n_assays] = onehot
    output[:, ligand_end + n_assays] = ph
    output[:, ligand_end + n_assays + 1] = temp
    return output


def _dataset_stem(source_path: str) -> str:
    """Return a readable, filesystem-safe dataset name.

    Args:
        source_path: Prepared CSV or Parquet path.

    Returns:
        Sanitized basename without its extension.
    """
    stem = os.path.splitext(os.path.basename(source_path))[0]
    return re.sub(r"[^0-9A-Za-z._-]+", "_", stem).strip("._") or "dataset"


def _source_identity(source_path: str) -> dict[str, Any]:
    """Describe the prepared source file for cache invalidation.

    Args:
        source_path: Prepared CSV or Parquet path.

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
    limit: Optional[int],
    activity_threshold_nm: float,
    include_assay_context: bool,
) -> str:
    """Hash every setting that changes row-level snapshot contents.

    Args:
        source: Source identity from :func:`_source_identity`.
        limit: Optional raw-row limit.
        activity_threshold_nm: Quantitative activity cutoff.
        include_assay_context: Whether assay sidecars are stored.

    Returns:
        Sixteen-character hexadecimal signature.
    """
    payload = {
        "storage_version": _STORAGE_VERSION,
        "source": source,
        "limit": limit,
        "activity_threshold_nm": float(activity_threshold_nm),
        "include_assay_context": bool(include_assay_context),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha1(encoded).hexdigest()[:16]


def _embedding_alias(identifier: str, modality: str) -> str:
    """Create a short readable alias for an embedding artifact.

    Args:
        identifier: Exact model id or ligand component cache id.
        modality: ``"protein"`` or ``"ligand"``.

    Returns:
        Filesystem-safe alias such as ``ESM650`` or ``MolFormerXL``.
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
    esm_match = re.search(r"esm2_t\d+_(\d+)([mb])", lowered)
    if modality == "protein" and esm_match:
        return f"ESM{esm_match.group(1)}{esm_match.group(2).upper() if esm_match.group(2) == 'b' else ''}"
    basename = identifier.rstrip("/").rsplit("/", 1)[-1]
    alias = re.sub(r"[^0-9A-Za-z]+", "_", basename).strip("_")
    return alias or modality


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
class _IndexedRows:
    """In-memory result of one streaming index pass."""

    proteins: list[str]
    ligands: list[str]
    protein_ids: np.ndarray
    ligand_ids: np.ndarray
    activity_labels: np.ndarray
    scaffold_groups: np.ndarray
    years: np.ndarray
    assay_idx: np.ndarray
    ph: np.ndarray
    temp: np.ndarray
    n_scaffolds: int


def _index_rows(
    source_path: str,
    limit: Optional[int],
    activity_threshold_nm: float,
    verbose: bool,
) -> _IndexedRows:
    """Stream prepared rows and assign stable dataset-local entity ids.

    Args:
        source_path: Prepared CSV or Parquet path.
        limit: Optional raw-row limit.
        activity_threshold_nm: Quantitative binder cutoff.
        verbose: Whether to print progress.

    Returns:
        Indexed rows and distinct entities in first-seen order.
    """
    ligand_index: dict[str, int] = {}
    protein_index: dict[str, int] = {}
    scaffold_index: dict[str, int] = {}
    ligand_ids: list[int] = []
    protein_ids: list[int] = []
    scaffold_ids: list[int] = []
    assay_ids: list[int] = []
    ph_values: list[float] = []
    temperatures: list[float] = []
    labels: list[int] = []
    years: list[int] = []

    if verbose:
        print("[features] streaming and indexing prepared rows", flush=True)
    for kept, row in enumerate(iter_prepared_rows(source_path, limit=limit), start=1):
        row_index = kept - 1
        ligand_ids.append(ligand_index.setdefault(row.smiles, len(ligand_index)))
        protein_ids.append(protein_index.setdefault(row.sequence, len(protein_index)))
        scaffold = murcko_scaffold_key(row.smiles, row_index)
        scaffold_ids.append(scaffold_index.setdefault(scaffold, len(scaffold_index)))
        assay_ids.append(_ASSAY_TO_INDEX[row.assay_type])
        ph_values.append(row.ph)
        temperatures.append(row.temp)
        years.append(-1 if row.year is None else int(row.year))
        if row.activity_label is not None:
            labels.append(int(row.activity_label))
        else:
            label = binarize_pactivity(
                np.asarray([row.pactivity]), threshold_nm=activity_threshold_nm
            )[0]
            labels.append(int(label))
        if verbose and kept % _PROGRESS_EVERY == 0:
            print(
                f"[features] indexed {kept} rows "
                f"({len(protein_index)} proteins, {len(ligand_index)} ligands)",
                flush=True,
            )

    if not labels:
        raise ValueError(f"No valid training rows produced from {source_path}.")
    return _IndexedRows(
        proteins=list(protein_index),
        ligands=list(ligand_index),
        protein_ids=np.asarray(protein_ids, dtype=np.uint32),
        ligand_ids=np.asarray(ligand_ids, dtype=np.uint32),
        activity_labels=np.asarray(labels, dtype=np.uint8),
        scaffold_groups=np.asarray(scaffold_ids, dtype=np.int32),
        years=np.asarray(years, dtype=np.int32),
        assay_idx=np.asarray(assay_ids, dtype=np.uint8),
        ph=np.asarray(ph_values, dtype=np.float32),
        temp=np.asarray(temperatures, dtype=np.float32),
        n_scaffolds=len(scaffold_index),
    )


def _write_cached_vectors(
    entities: Sequence[str],
    cache: EmbeddingCache,
    component: Any,
    destination: str,
    dim: int,
    verbose: bool,
) -> None:
    """Write cached unique vectors to an atomic, memory-mappable ``.npy``.

    Args:
        entities: Entity strings in dataset-local id order.
        cache: Filled embedding cache.
        component: Object exposing ``vectors_for(entities, cache)``; ``None``
            uses ``cache.get_many`` directly for proteins.
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
        if component is None:
            cached = cache.get_many(chunk)
            block = np.stack([cached[entity] for entity in chunk])
        else:
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


def _validate_alias(
    manifest: dict[str, Any],
    alias: str,
    identifier: str,
    modality: str,
) -> None:
    """Reject a readable alias that already refers to another identifier.

    Args:
        manifest: Modality-specific embedding manifest.
        alias: Proposed short alias.
        identifier: Exact model/component identifier.
        modality: Human-readable modality for errors.

    Returns:
        None.

    Raises:
        ValueError: If the alias is already assigned to another identifier.
    """
    existing = manifest.get(alias)
    if existing is not None and existing.get("identifier") != identifier:
        raise ValueError(
            f"{modality} embedding alias {alias!r} maps to both "
            f"{existing.get('identifier')!r} and {identifier!r}."
        )


class FeatureView:
    """Array-like, on-demand view over a compact feature snapshot."""

    def __init__(
        self,
        dataset: "FeatureDataset",
        indices: Optional[np.ndarray] = None,
    ) -> None:
        """Open selected embedding matrices and an optional row mapping.

        Args:
            dataset: Feature snapshot descriptor.
            indices: Optional dataset-row indices represented by this view.
        """
        self.dataset = dataset
        self.indices = (
            None if indices is None else np.asarray(indices, dtype=np.int64)
        )
        self._protein_ids = np.load(dataset.protein_ids_path, mmap_mode="r")
        self._ligand_ids = np.load(dataset.ligand_ids_path, mmap_mode="r")
        self._protein_embeddings = np.load(dataset.protein_embedding_path, mmap_mode="r")
        self._ligand_embeddings = [
            np.load(path, mmap_mode="r") for path in dataset.ligand_embedding_paths
        ]
        if dataset.include_assay_context:
            self._assay_idx = np.load(
                os.path.join(dataset.directory, "assay_idx.npy"), mmap_mode="r"
            )
            self._ph = np.load(os.path.join(dataset.directory, "ph.npy"), mmap_mode="r")
            self._temp = np.load(os.path.join(dataset.directory, "temp.npy"), mmap_mode="r")
        else:
            self._assay_idx = None
            self._ph = None
            self._temp = None

    @property
    def shape(self) -> tuple[int, int]:
        """Return the logical matrix shape.

        Returns:
            ``(selected_rows, feature_dimension)``.
        """
        n_rows = self.dataset.n_rows if self.indices is None else len(self.indices)
        return n_rows, self.dataset.n_features

    def subset(self, indices: np.ndarray) -> "FeatureView":
        """Return a zero-copy row subset.

        Args:
            indices: Indices relative to this view.

        Returns:
            New view carrying a composed dataset-row mapping.
        """
        local = np.asarray(indices, dtype=np.int64)
        mapped = local if self.indices is None else self.indices[local]
        return FeatureView(self.dataset, mapped)

    def __getitem__(self, key: object) -> np.ndarray:
        """Gather one or more rows into a dense feature array.

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
        protein_ids = np.asarray(self._protein_ids[rows_1d], dtype=np.int64)
        ligand_ids = np.asarray(self._ligand_ids[rows_1d], dtype=np.int64)
        protein = np.asarray(self._protein_embeddings[protein_ids], dtype=np.float32)
        ligand_blocks = [
            np.asarray(matrix[ligand_ids], dtype=np.float32)
            for matrix in self._ligand_embeddings
        ]
        ligand = (
            ligand_blocks[0]
            if len(ligand_blocks) == 1
            else np.concatenate(ligand_blocks, axis=1)
        )
        if self.dataset.include_assay_context:
            assert self._assay_idx is not None and self._ph is not None and self._temp is not None
            assay = np.asarray(self._assay_idx[rows_1d], dtype=np.int64)
            ph = np.asarray(self._ph[rows_1d], dtype=np.float32)
            temp = np.asarray(self._temp[rows_1d], dtype=np.float32)
        else:
            assay = np.zeros(rows_1d.size, dtype=np.int64)
            ph = np.zeros(rows_1d.size, dtype=np.float32)
            temp = np.zeros(rows_1d.size, dtype=np.float32)
        gathered = assemble_matrix(
            protein,
            ligand,
            assay,
            ph,
            temp,
            include_assay_context=self.dataset.include_assay_context,
        )
        return gathered[0] if scalar else gathered


@dataclass(frozen=True, slots=True)
class FeatureDataset:
    """Descriptor for one selected feature combination in a dataset snapshot."""

    directory: str
    dataset_directory: str
    n_rows: int
    n_features: int
    protein_dim: int
    ligand_dim: int
    signature: str
    protein_alias: str
    ligand_aliases: tuple[str, ...]
    labels_are_binary: bool = True
    activity_threshold_nm: float = config.ACTIVITY_THRESHOLD_NM
    include_assay_context: bool = config.INCLUDE_ASSAY_CONTEXT

    @property
    def split_directory(self) -> str:
        """Return the dataset-level split-cache directory."""
        return os.path.join(self.dataset_directory, "splits")

    @property
    def protein_ids_path(self) -> str:
        """Return the per-row protein-id path."""
        return os.path.join(self.directory, "protein_ids.npy")

    @property
    def ligand_ids_path(self) -> str:
        """Return the per-row ligand-id path."""
        return os.path.join(self.directory, "ligand_ids.npy")

    @property
    def protein_embedding_path(self) -> str:
        """Return the selected protein matrix path."""
        return os.path.join(
            self.directory, "protein_embeddings", f"{self.protein_alias}.npy"
        )

    @property
    def ligand_embedding_paths(self) -> tuple[str, ...]:
        """Return selected ligand component matrix paths."""
        return tuple(
            os.path.join(self.directory, "ligand_embeddings", f"{alias}.npy")
            for alias in self.ligand_aliases
        )

    def load_y(self) -> np.ndarray:
        """Load binary activity labels.

        Returns:
            ``uint8`` array of shape ``(n_rows,)``.
        """
        return np.load(os.path.join(self.directory, "activity_labels.npy"))

    def load_groups(self) -> np.ndarray:
        """Load protein group ids.

        Returns:
            Alias of the per-row protein ids.
        """
        return np.load(self.protein_ids_path)

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

    def feature_view(self, indices: Optional[np.ndarray] = None) -> FeatureView:
        """Open an on-demand feature view.

        Args:
            indices: Optional dataset-row selection.

        Returns:
            Array-like feature view.
        """
        return FeatureView(self, indices)


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
    if meta.get("storage_version") != _STORAGE_VERSION:
        raise ValueError("Feature cache storage version is obsolete.")
    if meta.get("signature") != expected_signature:
        raise ValueError(
            "Feature cache does not match the requested source/configuration; "
            "use --rebuild-features."
        )
    n_rows = int(meta["n_rows"])
    n_proteins = int(meta["n_proteins"])
    n_ligands = int(meta["n_ligands"])
    protein_ids = _open_array(
        os.path.join(directory, "protein_ids.npy"), (n_rows,), np.dtype(np.uint32)
    )
    ligand_ids = _open_array(
        os.path.join(directory, "ligand_ids.npy"), (n_rows,), np.dtype(np.uint32)
    )
    _open_array(
        os.path.join(directory, "activity_labels.npy"), (n_rows,), np.dtype(np.uint8)
    )
    _open_array(
        os.path.join(directory, "scaffold_groups.npy"), (n_rows,), np.dtype(np.int32)
    )
    _open_array(os.path.join(directory, "years.npy"), (n_rows,), np.dtype(np.int32))
    if n_rows and (
        int(protein_ids.max()) >= n_proteins or int(ligand_ids.max()) >= n_ligands
    ):
        raise ValueError("Feature cache contains out-of-range entity ids.")
    if bool(meta.get("include_assay_context", False)):
        _open_array(
            os.path.join(directory, "assay_idx.npy"), (n_rows,), np.dtype(np.uint8)
        )
        _open_array(os.path.join(directory, "ph.npy"), (n_rows,), np.dtype(np.float32))
        _open_array(os.path.join(directory, "temp.npy"), (n_rows,), np.dtype(np.float32))
    embeddings = meta.get("embeddings")
    if not isinstance(embeddings, dict):
        raise ValueError("Feature cache is missing its embedding manifest.")
    expected_counts = {"protein": n_proteins, "ligand": n_ligands}
    subdirectories = {
        "protein": "protein_embeddings",
        "ligand": "ligand_embeddings",
    }
    for modality in ("protein", "ligand"):
        manifest = embeddings.get(modality)
        if not isinstance(manifest, dict):
            raise ValueError(f"Feature cache has no {modality} embedding manifest.")
        for alias, entry in manifest.items():
            if not isinstance(entry, dict) or not entry.get("identifier"):
                raise ValueError(
                    f"Malformed {modality} embedding manifest entry {alias!r}."
                )
            dimension = int(entry["dimension"])
            filename = str(entry["file"])
            expected_filename = f"{alias}.npy"
            if filename != expected_filename:
                raise ValueError(
                    f"Embedding manifest entry {alias!r} points to unexpected "
                    f"filename {filename!r}."
                )
            path = os.path.join(directory, subdirectories[modality], filename)
            expected_shape = (expected_counts[modality], dimension)
            if entry.get("dtype") != "float32" or entry.get("shape") != list(
                expected_shape
            ):
                raise ValueError(
                    f"Embedding manifest shape/dtype is invalid for {path}."
                )
            _open_array(
                path,
                expected_shape,
                np.dtype(np.float32),
            )
            if int(entry.get("size", -1)) != os.path.getsize(path):
                raise ValueError(f"Embedding matrix size changed: {path}.")


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


def _write_row_snapshot(
    directory: str,
    indexed: _IndexedRows,
    include_assay_context: bool,
) -> None:
    """Persist shared row arrays in a temporary snapshot directory.

    Args:
        directory: Temporary feature directory.
        indexed: Indexed prepared rows.
        include_assay_context: Whether to persist assay arrays.

    Returns:
        None.
    """
    os.makedirs(directory, exist_ok=False)
    os.makedirs(os.path.join(directory, "protein_embeddings"))
    os.makedirs(os.path.join(directory, "ligand_embeddings"))
    _save_array(directory, "protein_ids.npy", indexed.protein_ids)
    _save_array(directory, "ligand_ids.npy", indexed.ligand_ids)
    _save_array(directory, "activity_labels.npy", indexed.activity_labels)
    _save_array(directory, "scaffold_groups.npy", indexed.scaffold_groups)
    _save_array(directory, "years.npy", indexed.years)
    if include_assay_context:
        _save_array(directory, "assay_idx.npy", indexed.assay_idx)
        _save_array(directory, "ph.npy", indexed.ph)
        _save_array(directory, "temp.npy", indexed.temp)


def _ensure_requested_embeddings(
    directory: str,
    meta: dict[str, Any],
    indexed: _IndexedRows,
    protein_model: str,
    ligand_featurizer: CompositeLigandFeaturizer,
    device: object,
    verbose: bool,
) -> tuple[str, tuple[str, ...]]:
    """Materialize requested protein and ligand component matrices.

    Args:
        directory: Feature snapshot directory.
        meta: Mutable snapshot metadata.
        indexed: Indexed rows defining entity order.
        protein_model: Exact protein model id.
        ligand_featurizer: Parsed ligand representation.
        device: Optional embedding device.
        verbose: Whether to print progress.

    Returns:
        Selected protein alias and ordered ligand aliases.
    """
    embeddings = meta.setdefault("embeddings", {"protein": {}, "ligand": {}})
    protein_manifest: dict[str, Any] = embeddings.setdefault("protein", {})
    ligand_manifest: dict[str, Any] = embeddings.setdefault("ligand", {})

    pemb: HFEmbedder = protein_embedder(protein_model, device=device)
    protein_alias = _embedding_alias(protein_model, "protein")
    _validate_alias(protein_manifest, protein_alias, protein_model, "protein")
    protein_path = os.path.join(
        directory, "protein_embeddings", f"{protein_alias}.npy"
    )
    if protein_alias not in protein_manifest or not os.path.exists(protein_path):
        with EmbeddingCache(
            "protein", protein_model, expected_dim=pemb.dim
        ) as protein_cache:
            pemb.ensure_cached(indexed.proteins, protein_cache, verbose=verbose)
            _write_cached_vectors(
                indexed.proteins,
                protein_cache,
                None,
                protein_path,
                pemb.dim,
                verbose,
            )
        protein_manifest[protein_alias] = _manifest_entry(
            protein_path,
            protein_model,
            pemb.dim,
            (len(indexed.proteins), pemb.dim),
        )
    else:
        entry = protein_manifest[protein_alias]
        _open_array(
            protein_path,
            (len(indexed.proteins), int(entry["dimension"])),
            np.dtype(np.float32),
        )
    protein_dim = int(protein_manifest[protein_alias]["dimension"])
    del pemb

    ligand_featurizer.ensure_cached(indexed.ligands, verbose=verbose)
    ligand_aliases: list[str] = []
    for component in ligand_featurizer.components:
        alias = _embedding_alias(component.cache_id, "ligand")
        _validate_alias(ligand_manifest, alias, component.cache_id, "ligand")
        ligand_path = os.path.join(
            directory, "ligand_embeddings", f"{alias}.npy"
        )
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

    return protein_alias, tuple(ligand_aliases)


def _dataset_from_meta(
    directory: str,
    dataset_directory: str,
    meta: dict[str, Any],
    protein_alias: str,
    ligand_aliases: tuple[str, ...],
) -> FeatureDataset:
    """Construct a feature descriptor from validated metadata.

    Args:
        directory: Feature snapshot directory.
        dataset_directory: Parent dataset cache directory.
        meta: Snapshot metadata.
        protein_alias: Selected protein matrix alias.
        ligand_aliases: Selected ligand component aliases in concatenation order.

    Returns:
        Feature dataset descriptor.
    """
    protein_dim = int(meta["embeddings"]["protein"][protein_alias]["dimension"])
    ligand_dim = sum(
        int(meta["embeddings"]["ligand"][alias]["dimension"])
        for alias in ligand_aliases
    )
    n_features = config.feature_dim(
        protein_dim,
        ligand_dim,
        include_assay_context=bool(meta["include_assay_context"]),
    )
    return FeatureDataset(
        directory=directory,
        dataset_directory=dataset_directory,
        n_rows=int(meta["n_rows"]),
        n_features=n_features,
        protein_dim=protein_dim,
        ligand_dim=ligand_dim,
        signature=str(meta["signature"]),
        protein_alias=protein_alias,
        ligand_aliases=ligand_aliases,
        labels_are_binary=True,
        activity_threshold_nm=float(meta["activity_threshold_nm"]),
        include_assay_context=bool(meta["include_assay_context"]),
    )


def build_features(
    csv_path: str = config.TRAIN_CSV,
    protein_model: str = config.DEFAULT_PROTEIN_MODEL,
    ligand_model: str = config.DEFAULT_LIGAND_MODEL,
    limit: Optional[int] = None,
    verbose: bool = True,
    rebuild: bool = False,
    device: object = None,
    activity_threshold_nm: float = config.ACTIVITY_THRESHOLD_NM,
    include_assay_context: bool = config.INCLUDE_ASSAY_CONTEXT,
) -> FeatureDataset:
    """Build or reuse a compact dataset feature snapshot.

    Args:
        csv_path: Prepared CSV or Parquet path.
        protein_model: Protein embedding model id.
        ligand_model: Ligand representation spec.
        limit: Optional raw-row cap.
        verbose: Whether to print progress.
        rebuild: Whether to replace shared row data and existing embeddings.
        device: Optional embedding device.
        activity_threshold_nm: Quantitative binder cutoff.
        include_assay_context: Whether to include assay, pH and temperature.

    Returns:
        Descriptor selecting the requested embedding combination.
    """
    source = _source_identity(csv_path)
    signature = _row_signature(
        source, limit, activity_threshold_nm, include_assay_context
    )
    dataset_directory = os.path.join(
        config.DATASETS_CACHE_DIR, _dataset_stem(csv_path)
    )
    directory = os.path.join(dataset_directory, "features")
    meta_path = os.path.join(directory, "meta.json")
    ligand_featurizer = parse_ligand_repr(ligand_model, device=device)
    protein_alias = _embedding_alias(protein_model, "protein")
    ligand_aliases = tuple(
        _embedding_alias(component.cache_id, "ligand")
        for component in ligand_featurizer.components
    )

    with _dataset_lock(dataset_directory):
        _recover_stale_snapshot_artifacts(dataset_directory)
        meta: Optional[dict[str, Any]] = None
        if os.path.exists(meta_path) and not rebuild:
            with open(meta_path, encoding="utf-8") as handle:
                meta = json.load(handle)
            _validate_snapshot(directory, meta, signature)
            protein_ready = (
                meta.get("embeddings", {})
                .get("protein", {})
                .get(protein_alias, {})
                .get("identifier")
                == protein_model
                and os.path.exists(
                    os.path.join(
                        directory, "protein_embeddings", f"{protein_alias}.npy"
                    )
                )
            )
            ligand_ready = all(
                meta.get("embeddings", {})
                .get("ligand", {})
                .get(alias, {})
                .get("identifier")
                == component.cache_id
                and os.path.exists(
                    os.path.join(directory, "ligand_embeddings", f"{alias}.npy")
                )
                for alias, component in zip(
                    ligand_aliases, ligand_featurizer.components
                )
            )
            if protein_ready and ligand_ready:
                if verbose:
                    print(
                        f"[features] reusing {directory} "
                        f"({meta['n_rows']} rows; {protein_alias} + "
                        f"{'+'.join(ligand_aliases)})"
                    )
                return _dataset_from_meta(
                    directory,
                    dataset_directory,
                    meta,
                    protein_alias,
                    ligand_aliases,
                )

        indexed = _index_rows(
            csv_path, limit, activity_threshold_nm, verbose=verbose
        )
        if meta is None or rebuild:
            temporary = f"{directory}.building-{os.getpid()}"
            if os.path.exists(temporary):
                shutil.rmtree(temporary)
            _write_row_snapshot(temporary, indexed, include_assay_context)
            meta = {
                "storage_version": _STORAGE_VERSION,
                "signature": signature,
                "source": source,
                "limit": limit,
                "n_rows": int(indexed.activity_labels.size),
                "n_proteins": len(indexed.proteins),
                "n_ligands": len(indexed.ligands),
                "n_scaffolds": indexed.n_scaffolds,
                "has_year": bool(np.any(indexed.years >= 0)),
                "n_dated_rows": int(np.sum(indexed.years >= 0)),
                "labels_are_binary": True,
                "activity_threshold_nm": float(activity_threshold_nm),
                "include_assay_context": bool(include_assay_context),
                "assay_types": list(config.ASSAY_TYPES),
                "embeddings": {"protein": {}, "ligand": {}},
            }
            selected_protein, selected_ligands = _ensure_requested_embeddings(
                temporary,
                meta,
                indexed,
                protein_model,
                ligand_featurizer,
                device,
                verbose,
            )
            _atomic_json(os.path.join(temporary, "meta.json"), meta)
            _validate_snapshot(temporary, meta, signature)
            _commit_snapshot(temporary, directory)
        else:
            if (
                indexed.protein_ids.shape[0] != int(meta["n_rows"])
                or not np.array_equal(
                    indexed.protein_ids,
                    np.load(os.path.join(directory, "protein_ids.npy")),
                )
                or not np.array_equal(
                    indexed.ligand_ids,
                    np.load(os.path.join(directory, "ligand_ids.npy")),
                )
            ):
                raise ValueError(
                    "Prepared rows no longer match the cached entity ordering; "
                    "use --rebuild-features."
                )
            selected_protein, selected_ligands = _ensure_requested_embeddings(
                directory,
                meta,
                indexed,
                protein_model,
                ligand_featurizer,
                device,
                verbose,
            )
            _atomic_json(meta_path, meta)

        if verbose:
            print(
                f"[features] ready: {directory} "
                f"({meta['n_rows']} rows; {selected_protein} + "
                f"{'+'.join(selected_ligands)})"
            )
        return _dataset_from_meta(
            directory,
            dataset_directory,
            meta,
            selected_protein,
            selected_ligands,
        )
