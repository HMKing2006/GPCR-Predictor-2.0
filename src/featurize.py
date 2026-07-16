"""Assemble memmapped feature matrices from cached embeddings.

Feature construction runs in two streaming passes over the prepared rows:

1. Index the distinct ligands and proteins and record compact per-row integer
   references plus the scalar fields (assay type, pH, temperature, label) and
   Murcko scaffold ids.
2. Ensure every distinct entity is embedded (cache-aware), then fill an on-disk
   ``float32`` memmap ``X`` of shape ``(n_rows, feature_dim)`` in row chunks so
   peak RAM stays bounded even for tens of millions of rows.

Built datasets are keyed by a signature over the source CSV, row limit, protein
model, the canonical ligand-representation spec, activity threshold, and whether
assay context is included, so repeated ``train.py``/``grid_search.py``
invocations reuse the same matrix instead of recomputing it.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from typing import Optional

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


def murcko_scaffold_key(smiles: str, row_index: int) -> str:
    """Return a Bemis–Murcko scaffold key for a ligand SMILES.

    Failed or empty scaffolds fall back to a unique per-row key so orphans are
    never merged into a shared scaffold group.

    Args:
        smiles: Canonical ligand SMILES.
        row_index: Zero-based row index used for the orphan fallback key.

    Returns:
        Scaffold SMILES, or ``"__orphan_{row_index}"`` when scaffolding fails.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return f"__orphan_{row_index}"
    try:
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
    except Exception:
        return f"__orphan_{row_index}"
    if not scaffold:
        return f"__orphan_{row_index}"
    return scaffold


@dataclass(frozen=True, slots=True)
class FeatureDataset:
    """Handles to an on-disk feature matrix and its metadata.

    Attributes:
        directory: Folder holding ``X.dat``, ``y.npy``, ``groups.npy``,
            ``scaffold_groups.npy`` and ``meta.json``.
        n_rows: Number of training examples.
        n_features: Feature-vector length.
        protein_dim: Protein embedding dimensionality.
        ligand_dim: Ligand embedding dimensionality.
        signature: Content signature used for split reuse.
        labels_are_binary: If ``True``, ``y.npy`` already holds ``0``/``1``
            binder labels and should not be re-binarized at train time.
        activity_threshold_nm: Activity cutoff in nM used when binarizing
            quantitative rows during feature construction.
        include_assay_context: If ``True``, feature rows include assay one-hot
            and pH/temp scalars after the ligand block.
    """

    directory: str
    n_rows: int
    n_features: int
    protein_dim: int
    ligand_dim: int
    signature: str
    labels_are_binary: bool = False
    activity_threshold_nm: float = config.ACTIVITY_THRESHOLD_NM
    include_assay_context: bool = config.INCLUDE_ASSAY_CONTEXT

    @property
    def x_path(self) -> str:
        """Path to the feature memmap file.

        Returns:
            Absolute path of ``X.dat``.
        """
        return os.path.join(self.directory, "X.dat")

    def open_x(self, mode: str = "r") -> np.memmap:
        """Open the feature matrix as a memmap.

        Args:
            mode: Numpy memmap mode (e.g. ``"r"`` or ``"r+"``).

        Returns:
            A memmap of shape ``(n_rows, n_features)`` and dtype ``float32``.
        """
        return np.memmap(
            self.x_path, dtype=np.float32, mode=mode, shape=(self.n_rows, self.n_features)
        )

    def load_y(self) -> np.ndarray:
        """Load the label vector.

        Returns:
            A ``float32`` array of shape ``(n_rows,)``.
        """
        return np.load(os.path.join(self.directory, "y.npy"))

    def load_groups(self) -> np.ndarray:
        """Load the per-row protein group ids used for cold splitting.

        Returns:
            An ``int32`` array of shape ``(n_rows,)``.
        """
        return np.load(os.path.join(self.directory, "groups.npy"))

    def load_scaffold_groups(self) -> np.ndarray:
        """Load the per-row Murcko scaffold group ids.

        Returns:
            An ``int32`` array of shape ``(n_rows,)``.

        Raises:
            FileNotFoundError: If ``scaffold_groups.npy`` is missing (rebuild
                features with the current pipeline).
        """
        path = os.path.join(self.directory, "scaffold_groups.npy")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Missing {path}; rebuild features to enable double-cold splits."
            )
        return np.load(path)


def subset_to_memmap(dataset: "FeatureDataset", indices: np.ndarray, out_path: str) -> np.memmap:
    """Copy selected rows of a feature matrix into a new contiguous memmap.

    Materializing a split into its own contiguous memmap keeps peak RAM bounded
    (rows are copied in chunks) while letting the random forest read efficient
    contiguous shards during warm-start training.

    Args:
        dataset: The source feature dataset.
        indices: Row indices to copy, in the desired order.
        out_path: Destination memmap path.

    Returns:
        A memmap of shape ``(len(indices), dataset.n_features)``.
    """
    source = dataset.open_x("r")
    mm = np.memmap(
        out_path, dtype=np.float32, mode="w+", shape=(len(indices), dataset.n_features)
    )
    for start in range(0, len(indices), _ROW_CHUNK):
        end = min(start + _ROW_CHUNK, len(indices))
        mm[start:end] = source[indices[start:end]]
    mm.flush()
    return mm


def assay_onehot(assay_type: str) -> np.ndarray:
    """One-hot encode an assay type in the fixed ``config.ASSAY_TYPES`` order.

    Args:
        assay_type: One of ``config.ASSAY_TYPES``.

    Returns:
        A ``float32`` vector of length ``len(config.ASSAY_TYPES)``.

    Raises:
        KeyError: If ``assay_type`` is not a known assay.
    """
    vec = np.zeros(len(config.ASSAY_TYPES), dtype=np.float32)
    vec[_ASSAY_TO_INDEX[assay_type]] = 1.0
    return vec


def assemble_matrix(
    protein_vecs: np.ndarray,
    ligand_vecs: np.ndarray,
    assay_indices: np.ndarray,
    ph: np.ndarray,
    temp: np.ndarray,
    include_assay_context: bool = config.INCLUDE_ASSAY_CONTEXT,
) -> np.ndarray:
    """Concatenate component features into a dense matrix.

    Args:
        protein_vecs: Array ``(n, protein_dim)`` of protein embeddings.
        ligand_vecs: Array ``(n, ligand_dim)`` of ligand embeddings.
        assay_indices: Integer array ``(n,)`` of assay-type indices.
        ph: Array ``(n,)`` of pH values.
        temp: Array ``(n,)`` of temperatures in Celsius.
        include_assay_context: If ``True``, append assay one-hot and pH/temp;
            if ``False``, return ``[protein | ligand]`` only.

    Returns:
        A ``float32`` matrix ``(n, feature_dim)``. With assay context the layout
        is ``[protein | ligand | assay_onehot | pH | temp]``; otherwise
        ``[protein | ligand]``.
    """
    n = protein_vecs.shape[0]
    p = protein_vecs.shape[1]
    l = p + ligand_vecs.shape[1]
    if not include_assay_context:
        out = np.empty((n, l), dtype=np.float32)
        out[:, :p] = protein_vecs
        out[:, p:l] = ligand_vecs
        return out

    n_assay = len(config.ASSAY_TYPES)
    out = np.empty(
        (n, l + n_assay + config.NUM_SCALAR_FEATURES),
        dtype=np.float32,
    )
    out[:, :p] = protein_vecs
    out[:, p:l] = ligand_vecs
    onehot = np.zeros((n, n_assay), dtype=np.float32)
    onehot[np.arange(n), assay_indices] = 1.0
    out[:, l : l + n_assay] = onehot
    out[:, l + n_assay] = ph
    out[:, l + n_assay + 1] = temp
    return out


def _signature(
    csv_path: str,
    limit: Optional[int],
    protein_model: str,
    ligand_model: str,
    activity_threshold_nm: float,
    include_assay_context: bool,
) -> str:
    """Compute a stable content signature for a dataset build.

    Args:
        csv_path: Source CSV path.
        limit: Row limit used (or ``None``).
        protein_model: Protein model identifier.
        ligand_model: Ligand representation spec (canonicalized before hashing).
        activity_threshold_nm: Binder cutoff in nM for quantitative rows.
        include_assay_context: Whether assay/pH/temp features are included.

    Returns:
        A short hex digest identifying this configuration.
    """
    ligand_key = canonical_ligand_repr(ligand_model)
    key = (
        f"{os.path.basename(csv_path)}|{limit}|{protein_model}|{ligand_key}|"
        f"{activity_threshold_nm:g}|assayctx={int(include_assay_context)}"
    )
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]


def _build_ligand_emb_memmap(
    ligands: list[str],
    featurizer: CompositeLigandFeaturizer,
    path: str,
    verbose: bool = True,
) -> np.memmap:
    """Materialize unique concatenated ligand vectors into an indexable memmap.

    Args:
        ligands: Distinct ligand SMILES in index order.
        featurizer: Composite ligand featurizer whose component caches are filled.
        path: Destination memmap path.
        verbose: If ``True``, print periodic progress.

    Returns:
        A memmap of shape ``(len(ligands), featurizer.dim)``.
    """
    total = len(ligands)
    dim = featurizer.dim
    mm = np.memmap(path, dtype=np.float32, mode="w+", shape=(total, dim))
    for start in range(0, total, _ROW_CHUNK):
        chunk = ligands[start : start + _ROW_CHUNK]
        mm[start : start + len(chunk)] = featurizer.vectors_for(chunk)
        if verbose:
            print(
                f"[features] gathering ligand vectors {min(start + _ROW_CHUNK, total)}/{total}",
                flush=True,
            )
    mm.flush()
    return mm


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
    """Build (or reuse) the memmapped feature matrix for a dataset.

    Args:
        csv_path: Source BindingDB CSV.
        protein_model: ESM-2 model identifier.
        ligand_model: Ligand representation spec (HF model id, reserved RDKit
            token, or a comma-separated combination).
        limit: Optional cap on raw rows read (for smoke tests).
        verbose: If ``True``, print progress.
        rebuild: If ``True``, rebuild even when a cached build exists.
        device: Optional torch device for the embedders.
        activity_threshold_nm: Binder cutoff in nM for quantitative rows without
            an explicit ``Activity Label``.
        include_assay_context: If ``True``, append assay one-hot and pH/temp to
            each feature row; if ``False``, use protein and ligand only.

    Returns:
        A :class:`FeatureDataset` describing the on-disk matrix.
    """
    signature = _signature(
        csv_path,
        limit,
        protein_model,
        ligand_model,
        activity_threshold_nm,
        include_assay_context,
    )
    directory = os.path.join(config.FEATURES_DIR, signature)
    meta_path = os.path.join(directory, "meta.json")
    if os.path.exists(meta_path) and not rebuild:
        with open(meta_path) as handle:
            meta = json.load(handle)
        if verbose:
            print(f"[features] reusing cached build at {directory} ({meta['n_rows']} rows)")
        return FeatureDataset(
            directory=directory,
            n_rows=meta["n_rows"],
            n_features=meta["n_features"],
            protein_dim=meta["protein_dim"],
            ligand_dim=meta["ligand_dim"],
            signature=meta["signature"],
            labels_are_binary=bool(meta.get("labels_are_binary", False)),
            activity_threshold_nm=float(
                meta.get("activity_threshold_nm", config.ACTIVITY_THRESHOLD_NM)
            ),
            include_assay_context=bool(
                meta.get("include_assay_context", config.INCLUDE_ASSAY_CONTEXT)
            ),
        )

    os.makedirs(directory, exist_ok=True)

    # Pass 1: index entities and record compact per-row references.
    ligand_index: dict[str, int] = {}
    protein_index: dict[str, int] = {}
    scaffold_index: dict[str, int] = {}
    lig_ids: list[int] = []
    prot_ids: list[int] = []
    scaffold_ids: list[int] = []
    assay_ids: list[int] = []
    phs: list[float] = []
    temps: list[float] = []
    ys: list[float] = []
    if verbose:
        print("[features] pass 1: streaming + indexing rows", flush=True)
    for kept, row in enumerate(iter_prepared_rows(csv_path, limit=limit), start=1):
        row_idx = kept - 1
        lig_ids.append(ligand_index.setdefault(row.smiles, len(ligand_index)))
        prot_ids.append(protein_index.setdefault(row.sequence, len(protein_index)))
        scaffold_key = murcko_scaffold_key(row.smiles, row_idx)
        scaffold_ids.append(scaffold_index.setdefault(scaffold_key, len(scaffold_index)))
        assay_ids.append(_ASSAY_TO_INDEX[row.assay_type])
        phs.append(row.ph)
        temps.append(row.temp)
        if row.activity_label is not None:
            ys.append(float(row.activity_label))
        else:
            ys.append(
                float(
                    binarize_pactivity(
                        np.asarray([row.pactivity]),
                        threshold_nm=activity_threshold_nm,
                    )[0]
                )
            )
        if verbose and kept % _PROGRESS_EVERY == 0:
            print(
                f"[features] pass 1: {kept} rows kept "
                f"({len(protein_index)} proteins, {len(ligand_index)} ligands, "
                f"{len(scaffold_index)} scaffolds so far)",
                flush=True,
            )

    n_rows = len(ys)
    if n_rows == 0:
        raise ValueError("No valid training rows produced from the CSV.")
    ligands = list(ligand_index)
    proteins = list(protein_index)
    if verbose:
        print(
            f"[features] {n_rows} rows, {len(proteins)} proteins, "
            f"{len(ligands)} ligands, {len(scaffold_index)} scaffolds"
        )

    lig_ids_arr = np.asarray(lig_ids, dtype=np.int64)
    prot_ids_arr = np.asarray(prot_ids, dtype=np.int64)
    scaffold_ids_arr = np.asarray(scaffold_ids, dtype=np.int32)
    assay_arr = np.asarray(assay_ids, dtype=np.int64)
    ph_arr = np.asarray(phs, dtype=np.float32)
    temp_arr = np.asarray(temps, dtype=np.float32)
    y_arr = np.asarray(ys, dtype=np.float32)

    # Ensure embeddings exist for every distinct entity.
    pemb: HFEmbedder = protein_embedder(protein_model, device=device)
    with EmbeddingCache("protein", protein_model) as pcache:
        pemb.ensure_cached(proteins, pcache, verbose=verbose)
        protein_dim = pemb.dim
        protein_matrix = np.zeros((len(proteins), protein_dim), dtype=np.float32)
        got = pcache.get_many(proteins)
        for i, seq in enumerate(proteins):
            protein_matrix[i] = got[seq]
    del pemb

    ligand_featurizer = parse_ligand_repr(ligand_model, device=device)
    if verbose:
        print(
            f"[features] ligand representation: {ligand_featurizer.canonical_spec} "
            f"({ligand_featurizer.dim}-d: {ligand_featurizer.component_dims})",
            flush=True,
        )
    ligand_featurizer.ensure_cached(ligands, verbose=verbose)
    ligand_dim = ligand_featurizer.dim
    ligand_matrix = _build_ligand_emb_memmap(
        ligands,
        ligand_featurizer,
        os.path.join(directory, "ligand_emb.dat"),
        verbose=verbose,
    )

    # Pass 2: fill the feature memmap in row chunks.
    n_features = config.feature_dim(
        protein_dim, ligand_dim, include_assay_context=include_assay_context
    )
    x_path = os.path.join(directory, "X.dat")
    X = np.memmap(x_path, dtype=np.float32, mode="w+", shape=(n_rows, n_features))
    if verbose:
        ctx = "with assay context" if include_assay_context else "protein+ligand only"
        print(f"[features] pass 2: writing {n_rows}x{n_features} matrix ({ctx}) to {x_path}")
    for start in range(0, n_rows, _ROW_CHUNK):
        end = min(start + _ROW_CHUNK, n_rows)
        prot_vecs = protein_matrix[prot_ids_arr[start:end]]
        lig_vecs = np.asarray(ligand_matrix[lig_ids_arr[start:end]], dtype=np.float32)
        X[start:end] = assemble_matrix(
            prot_vecs,
            lig_vecs,
            assay_arr[start:end],
            ph_arr[start:end],
            temp_arr[start:end],
            include_assay_context=include_assay_context,
        )
        if verbose and (start // _ROW_CHUNK) % 10 == 0:
            print(f"[features] {end}/{n_rows} rows written", flush=True)
    X.flush()
    del X

    # Persist labels, protein/scaffold groups and metadata.
    np.save(os.path.join(directory, "y.npy"), y_arr)
    np.save(os.path.join(directory, "groups.npy"), prot_ids_arr.astype(np.int32))
    np.save(os.path.join(directory, "scaffold_groups.npy"), scaffold_ids_arr)
    meta = {
        "n_rows": n_rows,
        "n_features": n_features,
        "protein_dim": protein_dim,
        "ligand_dim": ligand_dim,
        "signature": signature,
        "ligand_model": ligand_featurizer.canonical_spec,
        "ligand_component_dims": ligand_featurizer.component_dims,
        "labels_are_binary": True,
        "activity_threshold_nm": activity_threshold_nm,
        "include_assay_context": include_assay_context,
        "assay_types": list(config.ASSAY_TYPES),
        "n_scaffolds": len(scaffold_index),
    }
    with open(meta_path, "w") as handle:
        json.dump(meta, handle, indent=2)

    if verbose:
        print(f"[features] build complete: {directory}")
    return FeatureDataset(
        directory=directory,
        n_rows=n_rows,
        n_features=n_features,
        protein_dim=protein_dim,
        ligand_dim=ligand_dim,
        signature=signature,
        labels_are_binary=True,
        activity_threshold_nm=activity_threshold_nm,
        include_assay_context=include_assay_context,
    )
