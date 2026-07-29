"""Sytravon + Genesis library screening helpers for the Gradio GUI.

Loads the ligand ID map, scores one protein against the full library with
:class:`predict.Predictor`, and builds top-k RDKit depictions plus a CSV export.

Ligand features are materialized once into a contiguous RAM matrix during
:meth:`ScreenLibrary.warm_ligands` so each :meth:`ScreenLibrary.screen` call
only embeds the protein and runs the classifier (no repeated LMDB membership
scans or ligand reloads).
"""

from __future__ import annotations

import io
import os
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import pandas as pd
from PIL import Image
from rdkit import Chem
from rdkit.Chem import Draw

import config
from predict import Predictor
from src.data_prep import canonicalize_smiles
from src.featurize import assemble_matrix

DEFAULT_LIGAND_MAP: str = os.path.join(
    config.DATA_DIR, "screen", "ligand_id_map.tsv"
)
DEFAULT_MODEL: str = config.DEFAULT_PAIR_MODEL_PATH
DEFAULT_ASSAY: str = "Ki"
DEFAULT_PH: float = float(config.DEFAULT_PH)
DEFAULT_TEMP: float = float(config.DEFAULT_TEMP_C)
_ASSAY_INDEX = {name: i for i, name in enumerate(config.ASSAY_TYPES)}
_SCORE_CHUNK = 20_000


@dataclass(frozen=True)
class TopHit:
    """One top-ranked library hit for UI display.

    Attributes:
        rank: 1-based rank by ``P(Active)``.
        smiles: Ligand SMILES.
        ligand_id: Compound ID (e.g. NCGC…).
        dataset: Source library name.
        p_active: Predicted binder probability.
        image: RDKit depiction, or ``None`` if parsing failed.
    """

    rank: int
    smiles: str
    ligand_id: str
    dataset: str
    p_active: float
    image: Optional[Image.Image]


class ScreenLibrary:
    """Scores Sytravon + Genesis ligands against a single protein sequence."""

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL,
        ligand_map_path: str = DEFAULT_LIGAND_MAP,
        *,
        assay: str = DEFAULT_ASSAY,
        ph: float = DEFAULT_PH,
        temp: float = DEFAULT_TEMP,
        verbose: bool = True,
    ) -> None:
        """Load the model and ligand library metadata.

        Args:
            model_path: Path to a saved joblib pair model.
            ligand_map_path: TSV with ``Ligand SMILES``, ``dataset``, ``ligand_id``.
            assay: Assay type used for all library pairs.
            ph: pH used for all library pairs.
            temp: Temperature (C) used for all library pairs.
            verbose: If ``True``, print embedding progress.

        Raises:
            FileNotFoundError: If the model or ligand map is missing.
            ValueError: If the ligand map is empty or missing columns.
        """
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        if not os.path.isfile(ligand_map_path):
            raise FileNotFoundError(f"Ligand map not found: {ligand_map_path}")
        if assay not in _ASSAY_INDEX:
            raise ValueError(f"Unknown assay {assay!r}; expected one of {config.ASSAY_TYPES}")

        self.assay = assay
        self.ph = float(ph)
        self.temp = float(temp)
        self.verbose = verbose
        self.predictor = Predictor(model_path, verbose=verbose)
        self.ligands = self._load_ligand_map(ligand_map_path)
        self._ligand_matrix: Optional[np.ndarray] = None
        self._warmed = False

    @staticmethod
    def _load_ligand_map(path: str) -> pd.DataFrame:
        """Read and validate the Sytravon/Genesis ligand ID map.

        Args:
            path: Path to ``ligand_id_map.tsv``.

        Returns:
            DataFrame with columns ``SMILES``, ``dataset``, ``ID``.

        Raises:
            ValueError: If required columns are missing or the table is empty.
        """
        df = pd.read_csv(path, sep="\t")
        rename = {
            "Ligand SMILES": "SMILES",
            "ligand_id": "ID",
            "dataset": "dataset",
        }
        missing = [c for c in ("Ligand SMILES", "dataset", "ligand_id") if c not in df.columns]
        if missing:
            raise ValueError(f"{path!r} missing columns: {missing}")
        out = df.rename(columns=rename)[["SMILES", "ID", "dataset"]].copy()
        out["SMILES"] = out["SMILES"].astype(str)
        out["ID"] = out["ID"].astype(str)
        out["dataset"] = out["dataset"].astype(str)
        out = out[out["SMILES"].str.strip().astype(bool)].reset_index(drop=True)
        if out.empty:
            raise ValueError(f"{path!r} has no ligand rows.")
        return out

    def warm_ligands(self) -> None:
        """Canonicalize, cache-fill, and load all library ligand features into RAM.

        Subsequent :meth:`screen` calls reuse the in-memory matrix and skip LMDB
        membership checks for ligands.

        Returns:
            None.

        Raises:
            ValueError: If no SMILES in the library can be canonicalized.
        """
        if self._warmed and self._ligand_matrix is not None:
            return

        raw_smiles = self.ligands["SMILES"].tolist()
        if self.verbose:
            print(f"[screen] warming {len(raw_smiles):,} ligand features…", flush=True)

        canon: list[str] = []
        keep: list[int] = []
        for i, smiles in enumerate(raw_smiles):
            c = canonicalize_smiles(smiles)
            if c is not None:
                keep.append(i)
                canon.append(c)
        if not canon:
            raise ValueError("No valid SMILES in the ligand library.")
        if len(keep) != len(self.ligands):
            dropped = len(self.ligands) - len(keep)
            if self.verbose:
                print(f"[screen] dropped {dropped:,} unparseable SMILES", flush=True)
            self.ligands = self.ligands.iloc[keep].reset_index(drop=True)

        featurizer = self.predictor.ligand_featurizer
        featurizer.ensure_cached(canon, verbose=self.verbose)
        if self.verbose:
            print(
                f"[screen] loading {len(canon):,} ligand vectors into RAM "
                f"({featurizer.dim}-d)…",
                flush=True,
            )
        self._ligand_matrix = np.ascontiguousarray(
            featurizer.vectors_for(canon), dtype=np.float32
        )
        self._warmed = True
        if self.verbose:
            nbytes = self._ligand_matrix.nbytes
            print(
                f"[screen] ligand matrix ready "
                f"({self._ligand_matrix.shape[0]:,} x {self._ligand_matrix.shape[1]}, "
                f"{nbytes / (1024**2):.0f} MiB)",
                flush=True,
            )

    def close(self) -> None:
        """Close the underlying predictor caches.

        Returns:
            None.
        """
        self.predictor.close()

    def screen(self, sequence: str) -> pd.DataFrame:
        """Score every library ligand against ``sequence``.

        Args:
            sequence: Amino-acid sequence.

        Returns:
            DataFrame with ``SMILES``, ``ID``, ``dataset``, ``P(Active)``, sorted
            by ``P(Active)`` descending.

        Raises:
            ValueError: If ``sequence`` is empty.
            RuntimeError: If the ligand matrix failed to warm.
        """
        seq = str(sequence or "").strip()
        if not seq:
            raise ValueError("Protein sequence is required.")
        self.warm_ligands()
        if self._ligand_matrix is None:
            raise RuntimeError("Ligand matrix was not warmed.")

        n = len(self.ligands)
        if self.verbose:
            print(f"[screen] scoring {n:,} ligands…", flush=True)

        pvecs = self.predictor._protein_vectors([seq])
        protein_vec = pvecs[seq]
        assay_idx_val = _ASSAY_INDEX[self.assay]
        preds = np.empty(n, dtype=np.float64)

        for start in range(0, n, _SCORE_CHUNK):
            end = min(start + _SCORE_CHUNK, n)
            m = end - start
            protein_block = np.broadcast_to(protein_vec, (m, protein_vec.shape[0])).copy()
            ligand_block = self._ligand_matrix[start:end]
            assay_idx = np.full(m, assay_idx_val, dtype=np.int64)
            ph_arr = np.full(m, self.ph, dtype=np.float32)
            temp_arr = np.full(m, self.temp, dtype=np.float32)
            matrix = assemble_matrix(
                protein_block,
                ligand_block,
                assay_idx,
                ph_arr,
                temp_arr,
                include_assay_context=self.predictor.include_assay_context,
            )
            preds[start:end] = self.predictor.model.predict(matrix)

        result = self.ligands.copy()
        result["P(Active)"] = preds
        result = result.sort_values("P(Active)", ascending=False, kind="mergesort")
        return result.reset_index(drop=True)

    def top_hits(self, results: pd.DataFrame, k: int = 10) -> list[TopHit]:
        """Build top-k hits with RDKit depictions.

        Args:
            results: Output of :meth:`screen`.
            k: Number of hits to return.

        Returns:
            A list of :class:`TopHit` of length up to ``k``.
        """
        hits: list[TopHit] = []
        for rank, (_, row) in enumerate(results.head(max(0, int(k))).iterrows(), start=1):
            smiles = str(row["SMILES"])
            hits.append(
                TopHit(
                    rank=rank,
                    smiles=smiles,
                    ligand_id=str(row["ID"]),
                    dataset=str(row["dataset"]),
                    p_active=float(row["P(Active)"]),
                    image=depict_smiles(smiles),
                )
            )
        return hits

    @staticmethod
    def results_to_csv_bytes(results: pd.DataFrame) -> bytes:
        """Serialize full screening results to CSV bytes.

        Args:
            results: Output of :meth:`screen`.

        Returns:
            UTF-8 CSV bytes with columns SMILES, ID, dataset, P(Active).
        """
        cols = ["SMILES", "ID", "dataset", "P(Active)"]
        buf = io.StringIO()
        results[cols].to_csv(buf, index=False, float_format="%.6f")
        return buf.getvalue().encode("utf-8")


def depict_smiles(smiles: str, size: tuple[int, int] = (640, 480)) -> Optional[Image.Image]:
    """Draw a 2D depiction of a SMILES string.

    Args:
        smiles: Ligand SMILES.
        size: Image width and height in pixels.

    Returns:
        A PIL RGB image, or ``None`` if RDKit cannot parse the SMILES.
    """
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None
    return Draw.MolToImage(mol, size=size)


def format_top_hits_markdown(hits: Sequence[TopHit]) -> str:
    """Format top hits as Markdown (without images).

    Args:
        hits: Top-ranked hits.

    Returns:
        Markdown table text.
    """
    if not hits:
        return "_No hits._"
    lines = [
        "| Rank | ID | Dataset | P(Active) | SMILES |",
        "| ---: | --- | --- | ---: | --- |",
    ]
    for hit in hits:
        lines.append(
            f"| {hit.rank} | `{hit.ligand_id}` | {hit.dataset} "
            f"| {hit.p_active:.4f} | `{hit.smiles}` |"
        )
    return "\n".join(lines)
