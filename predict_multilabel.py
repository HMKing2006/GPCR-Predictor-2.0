"""Predict family or target multilabel probabilities for ligands."""

from __future__ import annotations

import argparse
import csv
import os
from typing import Optional, Sequence

import numpy as np

from src.data_prep import canonicalize_smiles
from src.io_utils import (
    SDF_EXTS,
    SMILES_FILE_EXTS,
    SPREADSHEET_EXTS,
    find_smiles_column,
    read_ligands,
    read_table,
    write_table,
)
from src.ligand_repr import parse_ligand_repr
from src.multilabel.models import load_model
from src.multilabel.vocab import load_vocab


def _load_vocab_for_model(model) -> list[str]:
    """Resolve the vocabulary list from model metadata.

    Args:
        model: Loaded multilabel model.

    Returns:
        Ordered label strings.

    Raises:
        SystemExit: If vocabulary cannot be resolved.
    """
    embedded = model.metadata.get("vocab")
    if isinstance(embedded, list) and embedded:
        return [str(x) for x in embedded]
    vocab_path = model.metadata.get("vocab_path")
    if vocab_path and os.path.exists(str(vocab_path)):
        return load_vocab(str(vocab_path))
    raise SystemExit(
        "Model metadata is missing an embedded vocab and vocab_path is unavailable."
    )


class MultilabelPredictor:
    """Wraps a trained multilabel model for ligand-only inference."""

    def __init__(self, model_path: str, verbose: bool = True) -> None:
        """Load the model and prepare the ligand featurizer.

        Args:
            model_path: Path to a saved multilabel joblib model.
            verbose: If ``True``, print embedding progress.
        """
        self.verbose = verbose
        self.model = load_model(model_path)
        self.vocab = _load_vocab_for_model(self.model)
        self.ligand_model: str = str(self.model.metadata["ligand_model"])
        self.ligand_featurizer = parse_ligand_repr(self.ligand_model)

    def predict_smiles(
        self,
        smiles: Sequence[str],
        *,
        top_k: Optional[int] = None,
    ) -> tuple[list[str], np.ndarray, list[str]]:
        """Score ligands and return probabilities aligned to the vocab.

        Args:
            smiles: Raw or canonical SMILES strings.
            top_k: Unused here; retained for caller convenience.

        Returns:
            ``(canonical_smiles, probs, skipped_raw)`` where ``probs`` has shape
            ``(n_kept, K)``.
        """
        del top_k
        kept: list[str] = []
        skipped: list[str] = []
        for raw in smiles:
            canon = canonicalize_smiles(str(raw))
            if canon is None:
                skipped.append(str(raw))
                continue
            kept.append(canon)
        if not kept:
            return [], np.zeros((0, len(self.vocab)), dtype=np.float32), skipped
        self.ligand_featurizer.ensure_cached(kept, verbose=self.verbose)
        matrix = self.ligand_featurizer.vectors_for(kept)
        probs = self.model.predict(matrix)
        return kept, probs, skipped


def _write_full_csv(
    path: str,
    smiles: Sequence[str],
    probs: np.ndarray,
    vocab: Sequence[str],
) -> None:
    """Write one probability column per vocabulary label.

    Args:
        path: Output CSV path.
        smiles: Canonical SMILES in row order.
        probs: Probability matrix ``(n, K)``.
        vocab: Label names.

    Returns:
        None.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Ligand SMILES", *vocab])
        for i, smi in enumerate(smiles):
            writer.writerow([smi, *[f"{float(p):.6f}" for p in probs[i]]])


def _write_topk_csv(
    path: str,
    smiles: Sequence[str],
    probs: np.ndarray,
    vocab: Sequence[str],
    top_k: int,
) -> None:
    """Write top-k label names and scores per ligand.

    Args:
        path: Output CSV path.
        smiles: Canonical SMILES in row order.
        probs: Probability matrix ``(n, K)``.
        vocab: Label names.
        top_k: Number of top labels to keep.

    Returns:
        None.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    k = min(int(top_k), len(vocab))
    header = ["Ligand SMILES"]
    for rank in range(1, k + 1):
        header.extend([f"label_{rank}", f"score_{rank}"])
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for i, smi in enumerate(smiles):
            order = np.argsort(-probs[i])[:k]
            row: list[object] = [smi]
            for j in order:
                row.extend([vocab[int(j)], f"{float(probs[i, j]):.6f}"])
            writer.writerow(row)


def build_arg_parser() -> argparse.ArgumentParser:
    """Construct the prediction CLI parser.

    Returns:
        Configured argument parser.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Predict experimental family/target multilabel probabilities for "
            "ligands using a saved multilabel joblib model."
        )
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Path to a models/multilabel/*.joblib file.",
    )
    parser.add_argument(
        "--ligand",
        default=None,
        help="SMILES string, .smi/.sdf file, or spreadsheet with a SMILES column.",
    )
    parser.add_argument(
        "--spreadsheet",
        default=None,
        help="Spreadsheet with a ligand SMILES column; writes *_predictions.<ext>.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output CSV path (default derived from input).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=0,
        help="If >0, write top-k labels instead of full vocab columns.",
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress progress.")
    return parser


def main(argv: Optional[list[str]] = None) -> None:
    """CLI entry point.

    Args:
        argv: Optional argument list.

    Returns:
        None.
    """
    args = build_arg_parser().parse_args(argv)
    if not args.ligand and not args.spreadsheet:
        raise SystemExit("Provide --ligand or --spreadsheet.")
    if args.ligand and args.spreadsheet:
        raise SystemExit("Pass only one of --ligand or --spreadsheet.")

    predictor = MultilabelPredictor(args.model, verbose=not args.quiet)
    top_k = int(args.top_k) if args.top_k and args.top_k > 0 else 0

    if args.spreadsheet:
        table = read_table(args.spreadsheet)
        smiles_col = find_smiles_column(table)
        if smiles_col is None:
            raise SystemExit(f"No SMILES column found in {args.spreadsheet}.")
        raw = [str(x) for x in table[smiles_col].tolist()]
        kept, probs, skipped = predictor.predict_smiles(raw)
        if skipped and not args.quiet:
            print(f"[predict] skipped {len(skipped)} unparseable SMILES", flush=True)
        # Align predictions back to original rows (NaN for skipped).
        full = np.full((len(raw), len(predictor.vocab)), np.nan, dtype=np.float32)
        kept_i = 0
        for i, raw_smi in enumerate(raw):
            canon = canonicalize_smiles(raw_smi)
            if canon is None:
                continue
            full[i] = probs[kept_i]
            kept_i += 1
        if top_k > 0:
            # For spreadsheets with top-k, append label/score columns.
            k = min(top_k, len(predictor.vocab))
            for rank in range(1, k + 1):
                table[f"label_{rank}"] = ""
                table[f"score_{rank}"] = np.nan
            for i in range(len(raw)):
                if np.isnan(full[i, 0]):
                    continue
                order = np.argsort(-full[i])[:k]
                for rank, j in enumerate(order, start=1):
                    table.at[table.index[i], f"label_{rank}"] = predictor.vocab[int(j)]
                    table.at[table.index[i], f"score_{rank}"] = float(full[i, j])
        else:
            for j, name in enumerate(predictor.vocab):
                table[name] = full[:, j]
        out = args.output
        if out is None:
            root, ext = os.path.splitext(args.spreadsheet)
            out = f"{root}_predictions{ext or '.csv'}"
        write_table(table, out)
        if not args.quiet:
            print(f"[predict] wrote {out}", flush=True)
        return

    assert args.ligand is not None
    ligand_arg = args.ligand
    if os.path.isfile(ligand_arg):
        ext = os.path.splitext(ligand_arg)[1].lower()
        if ext in SPREADSHEET_EXTS:
            table = read_table(ligand_arg)
            smiles_col = find_smiles_column(table)
            if smiles_col is None:
                raise SystemExit(f"No SMILES column found in {ligand_arg}.")
            raw = [str(x) for x in table[smiles_col].tolist()]
        elif ext in SMILES_FILE_EXTS or ext in SDF_EXTS:
            raw = list(read_ligands(ligand_arg))
        else:
            raw = [ligand_arg]
    else:
        raw = [ligand_arg]

    kept, probs, skipped = predictor.predict_smiles(raw)
    if skipped and not args.quiet:
        print(f"[predict] skipped {len(skipped)} unparseable SMILES", flush=True)
    out = args.output or "multilabel_predictions.csv"
    if top_k > 0:
        _write_topk_csv(out, kept, probs, predictor.vocab, top_k)
    else:
        _write_full_csv(out, kept, probs, predictor.vocab)
    if not args.quiet:
        print(f"[predict] wrote {out} ({len(kept)} ligands, K={len(predictor.vocab)})")


if __name__ == "__main__":
    main()
