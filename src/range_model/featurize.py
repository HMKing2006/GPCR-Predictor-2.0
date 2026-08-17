"""Build range-model feature snapshots (IC50/EC50 bins + head ids)."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

import config
from src.data_prep import iter_prepared_rows
from src.featurize import (
    FeatureView,
    MurckoScaffoldResolver,
    _ASSAY_TO_INDEX,
    _PROGRESS_EVERY,
    _atomic_json,
    _commit_snapshot,
    _dataset_lock,
    _dataset_stem,
    _embedding_alias,
    _ensure_requested_embeddings,
    _open_array,
    _recover_stale_snapshot_artifacts,
    _save_array,
    _source_identity,
)
from src.ligand_repr import parse_ligand_repr
from src.lmdb_cache import ScaffoldCache
from src.range_model import config as range_config
from src.range_model.bins import BIN_LABELS, N_BINS, RANGE_EDGES_NM
from src.range_model.labels import iter_range_rows


@dataclass(slots=True)
class _IndexedRangeRows:
    """In-memory result of one range indexing pass."""

    proteins: list[str]
    ligands: list[str]
    protein_ids: np.ndarray
    ligand_ids: np.ndarray
    y_bin: np.ndarray
    head_idx: np.ndarray
    loss_weight: np.ndarray
    scaffold_groups: np.ndarray
    years: np.ndarray
    assay_idx: np.ndarray
    ph: np.ndarray
    temp: np.ndarray
    n_scaffolds: int


@dataclass(frozen=True, slots=True)
class RangeFeatureDataset:
    """Descriptor for one IC50/EC50 range feature combination."""

    directory: str
    dataset_directory: str
    n_rows: int
    n_features: int
    protein_dim: int
    ligand_dim: int
    signature: str
    protein_alias: str
    ligand_aliases: tuple[str, ...]
    include_assay_context: bool = config.INCLUDE_ASSAY_CONTEXT
    include_binary: bool = range_config.INCLUDE_BINARY_IN_RANGE
    n_bins: int = N_BINS

    @property
    def split_directory(self) -> str:
        """Return the dataset-level split-cache directory.

        Returns:
            Absolute splits directory path.
        """
        return os.path.join(self.dataset_directory, "splits")

    @property
    def protein_ids_path(self) -> str:
        """Return the per-row protein-id path.

        Returns:
            Path to ``protein_ids.npy``.
        """
        return os.path.join(self.directory, "protein_ids.npy")

    @property
    def ligand_ids_path(self) -> str:
        """Return the per-row ligand-id path.

        Returns:
            Path to ``ligand_ids.npy``.
        """
        return os.path.join(self.directory, "ligand_ids.npy")

    @property
    def protein_embedding_path(self) -> str:
        """Return the selected protein matrix path.

        Returns:
            Path to the protein embedding ``.npy``.
        """
        return os.path.join(
            self.directory, "protein_embeddings", f"{self.protein_alias}.npy"
        )

    @property
    def ligand_embedding_paths(self) -> tuple[str, ...]:
        """Return selected ligand component matrix paths.

        Returns:
            Ordered ligand embedding paths.
        """
        return tuple(
            os.path.join(self.directory, "ligand_embeddings", f"{alias}.npy")
            for alias in self.ligand_aliases
        )

    def load_y_bin(self) -> np.ndarray:
        """Load potency-bin labels.

        Returns:
            ``uint8`` array of shape ``(n_rows,)``.
        """
        return np.load(os.path.join(self.directory, "y_bin.npy"))

    def load_head_idx(self) -> np.ndarray:
        """Load assay-head indices (``0``=IC50, ``1``=EC50).

        Returns:
            ``uint8`` array of shape ``(n_rows,)``.
        """
        return np.load(os.path.join(self.directory, "head_idx.npy"))

    def load_loss_weight(self) -> np.ndarray:
        """Load per-row loss weights.

        Returns:
            ``float32`` array of shape ``(n_rows,)``.
        """
        path = os.path.join(self.directory, "loss_weight.npy")
        if not os.path.exists(path):
            return np.ones(self.n_rows, dtype=np.float32)
        return np.load(path)

    def load_groups(self) -> np.ndarray:
        """Load protein group ids.

        Returns:
            Per-row protein ids.
        """
        return np.load(self.protein_ids_path)

    def load_scaffold_groups(self) -> np.ndarray:
        """Load per-row Murcko scaffold ids.

        Returns:
            ``int32`` array of shape ``(n_rows,)``.
        """
        return np.load(os.path.join(self.directory, "scaffold_groups.npy"))

    def load_years(self) -> np.ndarray:
        """Load publication years (``-1`` when missing).

        Returns:
            ``int32`` array of shape ``(n_rows,)``.
        """
        return np.load(os.path.join(self.directory, "years.npy"))

    def feature_view(self, indices: Optional[np.ndarray] = None) -> FeatureView:
        """Open an on-demand feature view.

        Args:
            indices: Optional dataset-row selection.

        Returns:
            Array-like feature view compatible with the binder stack.
        """
        return FeatureView(self, indices)  # type: ignore[arg-type]


def _row_signature(
    source: dict[str, Any],
    limit: Optional[int],
    include_assay_context: bool,
    include_binary: bool,
    binary_inactive_bin: int,
    binary_loss_weight: float,
) -> str:
    """Hash settings that change range snapshot contents.

    Args:
        source: Source identity mapping.
        limit: Optional raw-row limit.
        include_assay_context: Whether assay sidecars are stored.
        include_binary: Whether binary weak labels are included.
        binary_inactive_bin: Weak-bin index for inactive binary rows.
        binary_loss_weight: Loss weight for binary rows.

    Returns:
        Sixteen-character hexadecimal signature.
    """
    payload = {
        "storage_version": range_config.STORAGE_VERSION,
        "source": source,
        "limit": limit,
        "include_assay_context": bool(include_assay_context),
        "include_binary": bool(include_binary),
        "binary_inactive_bin": int(binary_inactive_bin),
        "binary_loss_weight": float(binary_loss_weight),
        "range_edges_nm": list(RANGE_EDGES_NM),
        "n_bins": int(N_BINS),
        "heads": list(range_config.HEAD_NAMES),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha1(encoded).hexdigest()[:16]


def _index_range_rows(
    source_path: str,
    limit: Optional[int],
    *,
    include_binary: bool,
    binary_inactive_bin: int,
    binary_loss_weight: float,
    verbose: bool,
) -> _IndexedRangeRows:
    """Stream prepared rows and index IC50/EC50 range examples.

    Args:
        source_path: Prepared CSV or Parquet path.
        limit: Optional raw-row limit.
        include_binary: Whether inactive binary rows may enter the snapshot.
        binary_inactive_bin: Weak-bin index for inactive binary rows.
        binary_loss_weight: Loss weight for binary rows.
        verbose: Whether to print progress.

    Returns:
        Indexed range rows and distinct entities.

    Raises:
        ValueError: If no valid range rows are produced.
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
    y_bins: list[int] = []
    head_ids: list[int] = []
    weights: list[float] = []
    years: list[int] = []

    if verbose:
        print("[range-features] streaming and indexing IC50/EC50 rows", flush=True)
    with ScaffoldCache() as scaffold_cache:
        resolver = MurckoScaffoldResolver(cache=scaffold_cache)
        kept = 0
        for row in iter_range_rows(
            iter_prepared_rows(source_path, limit=limit),
            include_binary=include_binary,
            binary_inactive_bin=binary_inactive_bin,
            binary_loss_weight=binary_loss_weight,
        ):
            row_index = kept
            ligand_ids.append(ligand_index.setdefault(row.smiles, len(ligand_index)))
            protein_ids.append(protein_index.setdefault(row.sequence, len(protein_index)))
            scaffold = resolver.resolve(row.smiles, row_index)
            scaffold_ids.append(scaffold_index.setdefault(scaffold, len(scaffold_index)))
            assay_ids.append(_ASSAY_TO_INDEX[row.assay_type])
            ph_values.append(row.ph)
            temperatures.append(row.temp)
            years.append(-1 if row.year is None else int(row.year))
            y_bins.append(int(row.y_bin))
            head_ids.append(int(row.head_idx))
            weights.append(float(row.loss_weight))
            kept += 1
            if verbose and kept % _PROGRESS_EVERY == 0:
                print(
                    f"[range-features] indexed {kept} rows "
                    f"({len(protein_index)} proteins, {len(ligand_index)} ligands)",
                    flush=True,
                )
        resolver.flush()

    if kept == 0:
        raise ValueError(f"No valid IC50/EC50 range rows produced from {source_path}.")
    return _IndexedRangeRows(
        proteins=list(protein_index),
        ligands=list(ligand_index),
        protein_ids=np.asarray(protein_ids, dtype=np.uint32),
        ligand_ids=np.asarray(ligand_ids, dtype=np.uint32),
        y_bin=np.asarray(y_bins, dtype=np.uint8),
        head_idx=np.asarray(head_ids, dtype=np.uint8),
        loss_weight=np.asarray(weights, dtype=np.float32),
        scaffold_groups=np.asarray(scaffold_ids, dtype=np.int32),
        years=np.asarray(years, dtype=np.int32),
        assay_idx=np.asarray(assay_ids, dtype=np.uint8),
        ph=np.asarray(ph_values, dtype=np.float32),
        temp=np.asarray(temperatures, dtype=np.float32),
        n_scaffolds=len(scaffold_index),
    )


def _write_row_snapshot(
    directory: str,
    indexed: _IndexedRangeRows,
    include_assay_context: bool,
) -> None:
    """Persist shared row arrays in a temporary snapshot directory.

    Args:
        directory: Temporary feature directory.
        indexed: Indexed range rows.
        include_assay_context: Whether to persist assay arrays.

    Returns:
        None.
    """
    os.makedirs(directory, exist_ok=False)
    os.makedirs(os.path.join(directory, "protein_embeddings"))
    os.makedirs(os.path.join(directory, "ligand_embeddings"))
    _save_array(directory, "protein_ids.npy", indexed.protein_ids)
    _save_array(directory, "ligand_ids.npy", indexed.ligand_ids)
    _save_array(directory, "y_bin.npy", indexed.y_bin)
    _save_array(directory, "head_idx.npy", indexed.head_idx)
    _save_array(directory, "loss_weight.npy", indexed.loss_weight)
    _save_array(directory, "scaffold_groups.npy", indexed.scaffold_groups)
    _save_array(directory, "years.npy", indexed.years)
    if include_assay_context:
        _save_array(directory, "assay_idx.npy", indexed.assay_idx)
        _save_array(directory, "ph.npy", indexed.ph)
        _save_array(directory, "temp.npy", indexed.temp)


def _validate_snapshot(
    directory: str,
    meta: dict[str, Any],
    expected_signature: str,
) -> None:
    """Validate range snapshot arrays and metadata.

    Args:
        directory: Feature snapshot directory.
        meta: Parsed ``meta.json``.
        expected_signature: Requested row-layout signature.

    Returns:
        None.

    Raises:
        ValueError: If metadata or arrays are stale or malformed.
    """
    if meta.get("storage_version") != range_config.STORAGE_VERSION:
        raise ValueError("Range feature cache storage version is obsolete.")
    if meta.get("signature") != expected_signature:
        raise ValueError(
            "Range feature cache does not match the requested source/configuration; "
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
    _open_array(os.path.join(directory, "y_bin.npy"), (n_rows,), np.dtype(np.uint8))
    _open_array(os.path.join(directory, "head_idx.npy"), (n_rows,), np.dtype(np.uint8))
    _open_array(
        os.path.join(directory, "scaffold_groups.npy"), (n_rows,), np.dtype(np.int32)
    )
    _open_array(os.path.join(directory, "years.npy"), (n_rows,), np.dtype(np.int32))
    if n_rows and (
        int(protein_ids.max()) >= n_proteins or int(ligand_ids.max()) >= n_ligands
    ):
        raise ValueError("Range feature cache contains out-of-range entity ids.")
    if bool(meta.get("include_assay_context", False)):
        _open_array(
            os.path.join(directory, "assay_idx.npy"), (n_rows,), np.dtype(np.uint8)
        )
        _open_array(os.path.join(directory, "ph.npy"), (n_rows,), np.dtype(np.float32))
        _open_array(os.path.join(directory, "temp.npy"), (n_rows,), np.dtype(np.float32))
    embeddings = meta.get("embeddings")
    if not isinstance(embeddings, dict):
        raise ValueError("Range feature cache is missing its embedding manifest.")


def _dataset_from_meta(
    directory: str,
    dataset_directory: str,
    meta: dict[str, Any],
    protein_alias: str,
    ligand_aliases: tuple[str, ...],
) -> RangeFeatureDataset:
    """Construct a range feature descriptor from validated metadata.

    Args:
        directory: Feature snapshot directory.
        dataset_directory: Parent dataset cache directory.
        meta: Snapshot metadata.
        protein_alias: Selected protein matrix alias.
        ligand_aliases: Selected ligand component aliases.

    Returns:
        Range feature dataset descriptor.
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
    return RangeFeatureDataset(
        directory=directory,
        dataset_directory=dataset_directory,
        n_rows=int(meta["n_rows"]),
        n_features=n_features,
        protein_dim=protein_dim,
        ligand_dim=ligand_dim,
        signature=str(meta["signature"]),
        protein_alias=protein_alias,
        ligand_aliases=ligand_aliases,
        include_assay_context=bool(meta["include_assay_context"]),
        include_binary=bool(meta.get("include_binary", False)),
        n_bins=int(meta.get("n_bins", N_BINS)),
    )


def build_range_features(
    csv_path: str = config.TRAIN_CSV,
    protein_model: str = config.DEFAULT_PROTEIN_MODEL,
    ligand_model: str = config.DEFAULT_LIGAND_MODEL,
    limit: Optional[int] = None,
    verbose: bool = True,
    rebuild: bool = False,
    device: object = None,
    include_assay_context: bool = config.INCLUDE_ASSAY_CONTEXT,
    include_binary: bool = range_config.INCLUDE_BINARY_IN_RANGE,
    binary_inactive_bin: int = range_config.BINARY_INACTIVE_RANGE_BIN,
    binary_loss_weight: float = range_config.BINARY_RANGE_LOSS_WEIGHT,
) -> RangeFeatureDataset:
    """Build or reuse a compact IC50/EC50 range feature snapshot.

    Cache lives under ``cache/datasets/<stem>__range/`` so it never collides
    with the binder feature snapshot for the same prepared parquet.

    Args:
        csv_path: Prepared CSV or Parquet path.
        protein_model: Protein embedding model id.
        ligand_model: Ligand representation spec.
        limit: Optional raw-row cap.
        verbose: Whether to print progress.
        rebuild: Whether to replace shared row data and embeddings.
        device: Optional embedding device.
        include_assay_context: Whether to include assay, pH and temperature.
        include_binary: Whether inactive binary rows map into the weak bin.
        binary_inactive_bin: Weak-bin index for inactive binary rows.
        binary_loss_weight: Loss weight for binary-derived rows.

    Returns:
        Descriptor selecting the requested embedding combination.
    """
    source = _source_identity(csv_path)
    signature = _row_signature(
        source,
        limit,
        include_assay_context,
        include_binary,
        binary_inactive_bin,
        binary_loss_weight,
    )
    dataset_directory = os.path.join(
        config.DATASETS_CACHE_DIR, f"{_dataset_stem(csv_path)}__range"
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
                        f"[range-features] reusing {directory} "
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

        indexed = _index_range_rows(
            csv_path,
            limit,
            include_binary=include_binary,
            binary_inactive_bin=binary_inactive_bin,
            binary_loss_weight=binary_loss_weight,
            verbose=verbose,
        )
        # Adapt to the binder embedding writer which expects activity_labels.
        binder_like = indexed  # type: ignore[assignment]
        # _ensure_requested_embeddings only uses .proteins / .ligands

        if meta is None or rebuild:
            temporary = f"{directory}.building-{os.getpid()}"
            if os.path.exists(temporary):
                shutil.rmtree(temporary)
            _write_row_snapshot(temporary, indexed, include_assay_context)
            meta = {
                "storage_version": range_config.STORAGE_VERSION,
                "signature": signature,
                "source": source,
                "limit": limit,
                "n_rows": int(indexed.y_bin.size),
                "n_proteins": len(indexed.proteins),
                "n_ligands": len(indexed.ligands),
                "n_scaffolds": indexed.n_scaffolds,
                "has_year": bool(np.any(indexed.years >= 0)),
                "n_dated_rows": int(np.sum(indexed.years >= 0)),
                "include_assay_context": bool(include_assay_context),
                "include_binary": bool(include_binary),
                "binary_inactive_bin": int(binary_inactive_bin),
                "binary_loss_weight": float(binary_loss_weight),
                "range_edges_nm": list(RANGE_EDGES_NM),
                "bin_labels": list(BIN_LABELS),
                "n_bins": int(N_BINS),
                "heads": list(range_config.HEAD_NAMES),
                "embeddings": {"protein": {}, "ligand": {}},
            }
            selected_protein, selected_ligands = _ensure_requested_embeddings(
                temporary,
                meta,
                binder_like,  # has .proteins and .ligands
                protein_model,
                ligand_featurizer,
                device,
                verbose,
            )
            _atomic_json(os.path.join(temporary, "meta.json"), meta)
            _validate_snapshot(temporary, meta, signature)
            _commit_snapshot(temporary, directory)
        else:
            selected_protein, selected_ligands = _ensure_requested_embeddings(
                directory,
                meta,
                binder_like,
                protein_model,
                ligand_featurizer,
                device,
                verbose,
            )
            _atomic_json(meta_path, meta)

        if verbose:
            n_ic = int(np.sum(indexed.head_idx == range_config.HEAD_IC50))
            n_ec = int(np.sum(indexed.head_idx == range_config.HEAD_EC50))
            print(
                f"[range-features] ready: {directory} "
                f"({meta['n_rows']} rows; IC50={n_ic:,} EC50={n_ec:,}; "
                f"{selected_protein} + {'+'.join(selected_ligands)})"
            )
        return _dataset_from_meta(
            directory,
            dataset_directory,
            meta,
            selected_protein,
            selected_ligands,
        )
