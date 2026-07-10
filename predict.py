"""Predict binding affinity for new proteins and ligands.

Four input modes are supported:

1. ``--spreadsheet``: one spreadsheet carrying both ligand SMILES and protein
   sequence columns; a ``pActivity (Predicted)`` column is appended and written
   to ``*_predictions.<ext>`` (extension preserved).
2. ``--ligand`` + ``--protein``: a single ligand input crossed with a single
   protein input (each may itself contain many entries); every ligand-protein
   pair is written to one CSV.
3. ``--spreadsheet-dir``: a folder of mode-1 spreadsheets.
4. ``--ligand-dir`` + ``--protein-dir``: every ligand from every ligand file
   crossed with every protein from every protein file, into one CSV.

Unless overridden, each query is assumed to be a Ki measurement at pH 7.4 and
25 C. Embeddings for any new protein or ligand are computed and written back to
the LMDB caches during prediction.
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
from typing import Optional, Sequence

import numpy as np
import pandas as pd

import config
from src.data_prep import canonicalize_smiles
from src.embeddings import ligand_embedder, protein_embedder
from src.featurize import assemble_matrix
from src.io_utils import (
    FASTA_EXTS,
    SDF_EXTS,
    SMILES_FILE_EXTS,
    SPREADSHEET_EXTS,
    find_sequence_column,
    find_smiles_column,
    list_files,
    predictions_output_path,
    read_ligands,
    read_proteins,
    read_table,
    write_table,
)
from src.lmdb_cache import EmbeddingCache
from src.models import load_model

_PAIR_HEADER: list[str] = [
    "Ligand SMILES",
    "Protein Sequence",
    "Assay",
    "pH",
    "Temp (C)",
    "pActivity (Predicted)",
]
_ASSAY_INDEX = {name: i for i, name in enumerate(config.ASSAY_TYPES)}
_CHUNK = 20_000


class Predictor:
    """Wraps a trained model plus its embedders and caches for inference."""

    def __init__(self, model_path: str, verbose: bool = True) -> None:
        """Load the model and prepare embedders/caches from its metadata.

        Args:
            model_path: Path to a saved joblib model.
            verbose: If ``True``, print progress while embedding.
        """
        self.verbose = verbose
        self.model = load_model(model_path)
        self.protein_model: str = self.model.metadata["protein_model"]
        self.ligand_model: str = self.model.metadata["ligand_model"]
        self.pemb = protein_embedder(self.protein_model)
        self.lemb = ligand_embedder(self.ligand_model)
        self.pcache = EmbeddingCache("protein", self.protein_model)
        self.lcache = EmbeddingCache("ligand", self.ligand_model)

    def close(self) -> None:
        """Close the underlying embedding caches.

        Returns:
            None.
        """
        self.pcache.close()
        self.lcache.close()

    def _protein_vectors(self, sequences: Sequence[str]) -> dict[str, np.ndarray]:
        """Ensure sequences are embedded and return their vectors.

        Args:
            sequences: Protein sequences (duplicates tolerated).

        Returns:
            A mapping from sequence to its embedding vector.
        """
        self.pemb.ensure_cached(sequences, self.pcache, verbose=self.verbose)
        return self.pcache.get_many(sequences)

    def _ligand_vectors(self, smiles: Sequence[str]) -> dict[str, np.ndarray]:
        """Ensure ligands are embedded and return their vectors.

        Args:
            smiles: Canonical ligand SMILES (duplicates tolerated).

        Returns:
            A mapping from SMILES to its embedding vector.
        """
        self.lemb.ensure_cached(smiles, self.lcache, verbose=self.verbose)
        return self.lcache.get_many(smiles)

    def predict_aligned(
        self,
        smiles: Sequence[str],
        sequences: Sequence[str],
        assays: Sequence[str],
        ph: Sequence[float],
        temp: Sequence[float],
    ) -> np.ndarray:
        """Predict one value per aligned (ligand, protein, assay, pH, temp) row.

        Args:
            smiles: Raw ligand SMILES per row.
            sequences: Protein sequences per row.
            assays: Assay type per row (must be in ``config.ASSAY_TYPES``).
            ph: pH per row.
            temp: Temperature (C) per row.

        Returns:
            A ``float`` array of predictions; rows with an unparseable SMILES or
            empty sequence yield ``NaN``.
        """
        n = len(smiles)
        canon: list[Optional[str]] = [canonicalize_smiles(s) for s in smiles]
        valid = [
            i
            for i in range(n)
            if canon[i] is not None and str(sequences[i]).strip() and assays[i] in _ASSAY_INDEX
        ]
        out = np.full(n, np.nan, dtype=np.float64)
        if not valid:
            return out

        uniq_seqs = {str(sequences[i]).strip() for i in valid}
        uniq_ligs = {canon[i] for i in valid}
        pvecs = self._protein_vectors(list(uniq_seqs))
        lvecs = self._ligand_vectors([s for s in uniq_ligs if s is not None])

        for start in range(0, len(valid), _CHUNK):
            rows = valid[start : start + _CHUNK]
            protein_vecs = np.stack([pvecs[str(sequences[i]).strip()] for i in rows])
            ligand_vecs = np.stack([lvecs[canon[i]] for i in rows])
            assay_idx = np.array([_ASSAY_INDEX[assays[i]] for i in rows], dtype=np.int64)
            ph_arr = np.array([ph[i] for i in rows], dtype=np.float32)
            temp_arr = np.array([temp[i] for i in rows], dtype=np.float32)
            matrix = assemble_matrix(protein_vecs, ligand_vecs, assay_idx, ph_arr, temp_arr)
            out[rows] = self.model.predict(matrix)
        return out

    def predict_pairs_to_csv(
        self,
        ligands: Sequence[str],
        proteins: Sequence[str],
        assay: str,
        ph: float,
        temp: float,
        out_path: str,
    ) -> int:
        """Predict every ligand-protein pair and stream the results to CSV.

        Args:
            ligands: Raw ligand SMILES (canonicalized internally; invalid ones
                are skipped).
            proteins: Protein sequences.
            assay: Assay type for all pairs.
            ph: pH for all pairs.
            temp: Temperature (C) for all pairs.
            out_path: Destination CSV path.

        Returns:
            The number of prediction rows written.

        Raises:
            ValueError: If ``assay`` is not a known assay type.
        """
        if assay not in _ASSAY_INDEX:
            raise ValueError(f"Unknown assay {assay!r}; expected one of {config.ASSAY_TYPES}")

        canon_pairs: list[tuple[str, str]] = []
        for raw in ligands:
            c = canonicalize_smiles(raw)
            if c is not None:
                canon_pairs.append((raw, c))
        seqs = [str(s).strip() for s in proteins if str(s).strip()]
        if not canon_pairs or not seqs:
            raise ValueError("No valid ligands or proteins to predict.")

        pvecs = self._protein_vectors(seqs)
        lvecs = self._ligand_vectors([c for _, c in canon_pairs])
        protein_matrix = np.stack([pvecs[s] for s in seqs])
        assay_idx_full = np.full(len(seqs), _ASSAY_INDEX[assay], dtype=np.int64)
        ph_full = np.full(len(seqs), ph, dtype=np.float32)
        temp_full = np.full(len(seqs), temp, dtype=np.float32)

        written = 0
        with open(out_path, "w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(_PAIR_HEADER)
            for raw_smiles, canon in canon_pairs:
                lig_vec = lvecs[canon]
                ligand_matrix = np.repeat(lig_vec[None, :], len(seqs), axis=0)
                matrix = assemble_matrix(
                    protein_matrix, ligand_matrix, assay_idx_full, ph_full, temp_full
                )
                preds = self.model.predict(matrix)
                for seq, pred in zip(seqs, preds):
                    writer.writerow([raw_smiles, seq, assay, ph, temp, f"{float(pred):.4f}"])
                    written += 1
        return written


def _series_or_default(
    df: pd.DataFrame, candidates: tuple[str, ...], default: object, length: int
) -> list:
    """Return a per-row list from the first matching column, or a default fill.

    Args:
        df: The DataFrame to inspect.
        candidates: Lower-cased candidate column names.
        default: Value used to fill missing cells or when no column matches.
        length: Expected number of rows.

    Returns:
        A list of length ``length``.
    """
    lower_map = {str(c).strip().lower(): c for c in df.columns}
    for candidate in candidates:
        if candidate in lower_map:
            series = df[lower_map[candidate]]
            return [default if pd.isna(v) else v for v in series.tolist()]
    return [default] * length


def predict_spreadsheet(predictor: Predictor, path: str, assay: str, ph: float, temp: float) -> str:
    """Run mode 1 on a single spreadsheet and write the augmented output.

    Args:
        predictor: The loaded predictor.
        path: Input spreadsheet path (must contain SMILES + sequence columns).
        assay: Default assay when the spreadsheet has no assay column.
        ph: Default pH when the spreadsheet has no pH column.
        temp: Default temperature when the spreadsheet has no temperature column.

    Returns:
        The output file path.

    Raises:
        ValueError: If the required SMILES or sequence column is missing.
    """
    df = read_table(path)
    smiles_col = find_smiles_column(df)
    seq_col = find_sequence_column(df)
    if smiles_col is None or seq_col is None:
        raise ValueError(f"{path!r} must contain both a SMILES and a sequence column.")

    n = len(df)
    assays = _series_or_default(df, ("assay", "assay type", "assay_type"), assay, n)
    phs = _series_or_default(df, ("ph",), ph, n)
    temps = _series_or_default(df, ("temp (c)", "temp", "temperature"), temp, n)

    preds = predictor.predict_aligned(
        [str(v) for v in df[smiles_col].tolist()],
        [str(v) for v in df[seq_col].tolist()],
        [str(a) for a in assays],
        [float(p) for p in phs],
        [float(t) for t in temps],
    )
    df["pActivity (Predicted)"] = preds
    out_path = predictions_output_path(path)
    write_table(df, out_path)
    return out_path


def _gather_from_dir(directory: str, reader, exts: tuple[str, ...]) -> list[str]:
    """Collect entries from every supported file in a directory.

    Args:
        directory: Directory to scan.
        reader: A function mapping a file path to a list of entries.
        exts: Acceptable file extensions.

    Returns:
        A flat list of all entries across matching files.
    """
    entries: list[str] = []
    for path in list_files(directory, exts):
        entries.extend(reader(path))
    return entries


def run(args: argparse.Namespace) -> None:
    """Dispatch to the selected prediction mode.

    Args:
        args: Parsed CLI arguments.

    Returns:
        None.

    Raises:
        ValueError: If no valid combination of mode flags is supplied.
    """
    predictor = Predictor(args.model, verbose=not args.quiet)
    try:
        if args.spreadsheet:
            out = predict_spreadsheet(predictor, args.spreadsheet, args.assay, args.ph, args.temp)
            print(f"[predict] wrote {out}")
        elif args.spreadsheet_dir:
            for path in list_files(args.spreadsheet_dir, SPREADSHEET_EXTS):
                if os.path.splitext(path)[0].endswith("_predictions"):
                    continue  # Skip files this tool previously produced.
                out = predict_spreadsheet(predictor, path, args.assay, args.ph, args.temp)
                print(f"[predict] wrote {out}")
        elif args.ligand and args.protein:
            ligands = read_ligands(args.ligand)
            proteins = read_proteins(args.protein)
            n = predictor.predict_pairs_to_csv(
                ligands, proteins, args.assay, args.ph, args.temp, args.output
            )
            print(f"[predict] wrote {n} rows to {args.output}")
        elif args.ligand_dir and args.protein_dir:
            ligands = _gather_from_dir(
                args.ligand_dir, read_ligands, SMILES_FILE_EXTS + SDF_EXTS + SPREADSHEET_EXTS
            )
            proteins = _gather_from_dir(
                args.protein_dir, read_proteins, FASTA_EXTS + SPREADSHEET_EXTS
            )
            n = predictor.predict_pairs_to_csv(
                ligands, proteins, args.assay, args.ph, args.temp, args.output
            )
            print(f"[predict] wrote {n} rows to {args.output}")
        else:
            raise ValueError(
                "Provide one of: --spreadsheet, --spreadsheet-dir, "
                "--ligand with --protein, or --ligand-dir with --protein-dir."
            )
    finally:
        predictor.close()


def _default_model_path() -> str:
    """Pick a default model file from the models directory.

    Returns:
        ``models/best_model.joblib`` if present, else the first ``*.joblib``
        found, else the conventional best-model path (which may not exist).
    """
    best = os.path.join(config.MODELS_DIR, "best_model.joblib")
    if os.path.exists(best):
        return best
    candidates = sorted(glob.glob(os.path.join(config.MODELS_DIR, "*.joblib")))
    return candidates[0] if candidates else best


def build_arg_parser() -> argparse.ArgumentParser:
    """Construct the command-line argument parser.

    Returns:
        The configured :class:`argparse.ArgumentParser`.
    """
    p = argparse.ArgumentParser(description="Predict binding affinity for new inputs.")
    p.add_argument("--model", default=_default_model_path(), help="Trained joblib model path.")
    p.add_argument("--spreadsheet", default=None, help="Mode 1: single spreadsheet input.")
    p.add_argument("--spreadsheet-dir", default=None, help="Mode 3: folder of spreadsheets.")
    p.add_argument("--ligand", default=None, help="Mode 2: ligand input (SMILES/file).")
    p.add_argument("--protein", default=None, help="Mode 2: protein input (FASTA/file).")
    p.add_argument("--ligand-dir", default=None, help="Mode 4: folder of ligand inputs.")
    p.add_argument("--protein-dir", default=None, help="Mode 4: folder of protein inputs.")
    p.add_argument("--assay", default="Ki", choices=list(config.ASSAY_TYPES))
    p.add_argument("--pH", dest="ph", type=float, default=config.DEFAULT_PH)
    p.add_argument("--temp", type=float, default=config.DEFAULT_TEMP_C)
    p.add_argument("--output", default="predictions.csv", help="Output CSV for pair/folder modes.")
    p.add_argument("--quiet", action="store_true")
    return p


def main(argv: Optional[list[str]] = None) -> None:
    """CLI entry point.

    Args:
        argv: Optional argument list (defaults to ``sys.argv``).

    Returns:
        None.
    """
    args = build_arg_parser().parse_args(argv)
    run(args)


if __name__ == "__main__":
    main()
