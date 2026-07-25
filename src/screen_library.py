"""Sytravon + Genesis library screening helpers for the Gradio GUI.

Loads the ligand ID map, scores one protein against the full library with
:class:`predict.Predictor`, and builds top-k RDKit depictions plus a CSV export.
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

DEFAULT_LIGAND_MAP: str = os.path.join(
    config.DATA_DIR, "screen", "ligand_id_map.tsv"
)
DEFAULT_MODEL: str = os.path.join(
    config.MODELS_DIR, "mlp_512x2_time_morgan_descriptors.joblib"
)
DEFAULT_ASSAY: str = "Ki"
DEFAULT_PH: float = float(config.DEFAULT_PH)
DEFAULT_TEMP: float = float(config.DEFAULT_TEMP_C)


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

        self.assay = assay
        self.ph = float(ph)
        self.temp = float(temp)
        self.verbose = verbose
        self.predictor = Predictor(model_path, verbose=verbose)
        self.ligands = self._load_ligand_map(ligand_map_path)
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
        """Ensure all library ligand features are present in the LMDB caches.

        Returns:
            None.
        """
        if self._warmed:
            return
        smiles = self.ligands["SMILES"].tolist()
        if self.verbose:
            print(f"[screen] warming {len(smiles):,} ligand features…", flush=True)
        self.predictor._ligand_vectors(smiles)
        self._warmed = True
        if self.verbose:
            print("[screen] ligand features ready", flush=True)

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
        """
        seq = str(sequence or "").strip()
        if not seq:
            raise ValueError("Protein sequence is required.")
        self.warm_ligands()
        n = len(self.ligands)
        smiles = self.ligands["SMILES"].tolist()
        sequences = [seq] * n
        assays = [self.assay] * n
        phs = [self.ph] * n
        temps = [self.temp] * n
        if self.verbose:
            print(f"[screen] scoring {n:,} ligands…", flush=True)
        preds = self.predictor.predict_aligned(smiles, sequences, assays, phs, temps)
        result = self.ligands.copy()
        result["P(Active)"] = preds.astype(np.float64)
        result = result.dropna(subset=["P(Active)"])
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


def depict_smiles(smiles: str, size: tuple[int, int] = (320, 240)) -> Optional[Image.Image]:
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
