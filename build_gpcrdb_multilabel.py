"""Build ligand-centric family/target multilabel tables from GPCRdb.

Streams an existing GPCRdb pair prepared Parquet, builds a protein sidecar with
``target_id`` = GPCRdb Entry name (``Target Name``), enriches Papyrus
``Classification`` by sequence when available, aggregates active annotations
per ligand, and writes GPCRdb-prefixed artifacts (Papyrus files untouched)::

    python build_gpcrdb.py
    python build_gpcrdb_multilabel.py

    python train_multilabel.py --task target \\
      --data data/train/GPCRdb_Ligand_target_prepared.parquet \\
      --vocab data/train/GPCRdb_target_vocab.json --rebuild-features

Family artifacts are skipped with a warning when Classification coverage is
too thin for a non-empty vocabulary; target artifacts are always required.
"""

from __future__ import annotations

import argparse
import os
import warnings
from typing import Optional

import pyarrow as pa
import pyarrow.parquet as pq

from src.multilabel import config as ml_config
from src.multilabel.labels import (
    aggregate_ligand_labels,
    build_label_matrix,
    count_family_actives,
    count_target_actives,
    years_array,
)
from src.multilabel.vocab import (
    build_family_vocab,
    build_target_vocab,
    filter_vocab_by_min_positives,
    save_vocab,
)

_COL_SEQ = "BindingDB Target Chain Sequence 1"
_COL_NAME = "Target Name"


def load_papyrus_classification_by_sequence(
    papyrus_proteins: Optional[str],
    *,
    version: str = "latest",
    papyrus_root: Optional[str] = None,
    verbose: bool = True,
) -> dict[str, dict[str, str]]:
    """Load Papyrus protein metadata keyed by sequence for Classification join.

    Prefers an existing proteins Parquet sidecar; falls back to Papyrus
    ``read_protein_set`` via :func:`build_papyrus_multilabel.load_protein_sidecar`.

    Args:
        papyrus_proteins: Optional path to a Papyrus-style proteins Parquet.
            When ``None``, tries :data:`ml_config.PROTEINS_SIDECAR`.
        version: Papyrus version used if the sidecar is missing.
        papyrus_root: Optional Papyrus data directory override.
        verbose: If ``True``, print load stats.

    Returns:
        Mapping ``sequence -> {Classification, TID, UniProtID, ...}``. Empty
        when neither a sidecar nor Papyrus data can be loaded.
    """
    path = papyrus_proteins or ml_config.PROTEINS_SIDECAR
    if path and os.path.isfile(path):
        table = pq.read_table(
            path,
            columns=["Sequence", "Classification", "TID", "UniProtID", "target_id"],
        )
        by_sequence: dict[str, dict[str, str]] = {}
        for row in table.to_pylist():
            sequence = str(row.get("Sequence") or "").strip()
            if not sequence:
                continue
            by_sequence[sequence] = {
                "target_id": str(row.get("target_id") or "").strip(),
                "Classification": str(row.get("Classification") or "").strip(),
                "TID": str(row.get("TID") or "").strip(),
                "UniProtID": str(row.get("UniProtID") or "").strip(),
                "Sequence": sequence,
            }
        if verbose:
            print(
                f"[gpcrdb-multilabel] loaded Papyrus Classification for "
                f"{len(by_sequence):,} sequences from {path}",
                flush=True,
            )
        return by_sequence

    try:
        from build_papyrus_multilabel import load_protein_sidecar

        return load_protein_sidecar(version, papyrus_root, verbose=verbose)
    except Exception as exc:
        if verbose:
            print(
                f"[gpcrdb-multilabel] Papyrus Classification unavailable "
                f"({exc!r}); family labels will be empty for unmatched sequences",
                flush=True,
            )
        return {}


def build_gpcrdb_protein_sidecar(
    activity_source: str,
    papyrus_by_sequence: dict[str, dict[str, str]],
    *,
    limit: Optional[int] = None,
    verbose: bool = True,
) -> dict[str, dict[str, str]]:
    """Build sequence-keyed protein metadata from a GPCRdb prepared Parquet.

    ``target_id`` is the GPCRdb Entry name stored in ``Target Name``. Papyrus
    ``Classification`` / ``TID`` / ``UniProtID`` are copied when the sequence
    matches.

    Args:
        activity_source: GPCRdb pair prepared Parquet path.
        papyrus_by_sequence: Optional Papyrus metadata keyed by sequence.
        limit: Optional cap on raw prepared rows scanned for unique proteins.
        verbose: If ``True``, print stats.

    Returns:
        Mapping ``sequence -> {target_id, Classification, TID, UniProtID, Sequence}``.

    Raises:
        FileNotFoundError: If ``activity_source`` is missing.
        ValueError: If required columns are absent.
    """
    if not os.path.isfile(activity_source):
        raise FileNotFoundError(
            f"GPCRdb activity source not found: {activity_source}\n"
            "Run build_gpcrdb.py first to write GPCRdb_prepared.parquet."
        )
    lower = activity_source.lower()
    if not lower.endswith((".parquet", ".pq")):
        raise ValueError(
            f"GPCRdb multilabel expects a prepared Parquet activity source, "
            f"got {activity_source!r}."
        )

    pf = pq.ParquetFile(activity_source)
    names = set(pf.schema_arrow.names)
    if _COL_SEQ not in names or _COL_NAME not in names:
        raise ValueError(
            f"{activity_source!r} must contain {_COL_NAME!r} and {_COL_SEQ!r}."
        )

    by_sequence: dict[str, dict[str, str]] = {}
    collisions = 0
    scanned = 0
    classified = 0
    for batch in pf.iter_batches(columns=[_COL_SEQ, _COL_NAME], batch_size=65_536):
        for row in batch.to_pylist():
            if limit is not None and scanned >= limit:
                break
            scanned += 1
            sequence = str(row.get(_COL_SEQ) or "").strip()
            target_id = str(row.get(_COL_NAME) or "").strip()
            if not sequence or not target_id:
                continue
            existing = by_sequence.get(sequence)
            if existing is not None:
                if existing["target_id"] != target_id:
                    collisions += 1
                continue
            papyrus = papyrus_by_sequence.get(sequence) or {}
            classification = str(papyrus.get("Classification") or "").strip()
            if classification:
                classified += 1
            by_sequence[sequence] = {
                "target_id": target_id,
                "Classification": classification,
                "TID": str(papyrus.get("TID") or "").strip(),
                "UniProtID": str(papyrus.get("UniProtID") or "").strip(),
                "Sequence": sequence,
            }
        if limit is not None and scanned >= limit:
            break

    if verbose:
        print(
            f"[gpcrdb-multilabel] protein sidecar: {len(by_sequence):,} sequences "
            f"(classified={classified:,}, skipped_collisions={collisions:,})",
            flush=True,
        )
    return by_sequence


def write_proteins_parquet(
    by_sequence: dict[str, dict[str, str]],
    path: str,
) -> None:
    """Write the protein Classification sidecar Parquet.

    Args:
        by_sequence: Sequence-keyed protein metadata.
        path: Destination Parquet path.

    Returns:
        None.
    """
    rows = list(by_sequence.values())
    table = pa.table(
        {
            "target_id": [r["target_id"] for r in rows],
            "Sequence": [r["Sequence"] for r in rows],
            "Classification": [r["Classification"] for r in rows],
            "TID": [r["TID"] for r in rows],
            "UniProtID": [r["UniProtID"] for r in rows],
        }
    )
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    temporary = f"{path}.tmp-{os.getpid()}.parquet"
    pq.write_table(table, temporary, compression="zstd")
    os.replace(temporary, path)


def _write_ligand_prepared(
    path: str,
    smiles: list[str],
    years: list[Optional[int]],
    label_lists: list[list[str]],
    label_column: str,
) -> None:
    """Write one ligand multilabel prepared Parquet atomically.

    Args:
        path: Destination Parquet path.
        smiles: Canonical ligand SMILES.
        years: Per-ligand max active year (``None`` when undated).
        label_lists: Per-ligand vocab label strings present.
        label_column: Column name for the label list (``family_labels`` or
            ``target_labels``).

    Returns:
        None.
    """
    table = pa.table(
        {
            "Ligand SMILES": smiles,
            "Year": years,
            label_column: label_lists,
        }
    )
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    temporary = f"{path}.tmp-{os.getpid()}.parquet"
    pq.write_table(table, temporary, compression="zstd")
    os.replace(temporary, path)


def build_multilabel_tables(
    activity_source: str = ml_config.DEFAULT_GPCRDB_ACTIVITY_SOURCE,
    papyrus_proteins: Optional[str] = None,
    version: str = "latest",
    papyrus_root: Optional[str] = None,
    threshold_nm: float = ml_config.ACTIVITY_THRESHOLD_NM,
    classification_depth: int = ml_config.CLASSIFICATION_DEPTH,
    min_positives: int = ml_config.MIN_POSITIVES,
    target_vocab_size: int = ml_config.TARGET_VOCAB_SIZE,
    limit: Optional[int] = None,
    verbose: bool = True,
) -> dict[str, str]:
    """Build GPCRdb protein sidecar, vocabularies, and ligand prepared tables.

    Args:
        activity_source: GPCRdb pair prepared Parquet path.
        papyrus_proteins: Optional Papyrus proteins Parquet for Classification.
        version: Papyrus version if the proteins sidecar must be rebuilt.
        papyrus_root: Optional Papyrus data directory.
        threshold_nm: Quantitative binder cutoff.
        classification_depth: Family Classification path depth.
        min_positives: Minimum unique active ligands per family/target label.
        target_vocab_size: Cap on target vocabulary size.
        limit: Optional activity-row stream limit.
        verbose: If ``True``, print progress.

    Returns:
        Mapping of artifact name to written path (family keys omitted when
        family artifacts are skipped).

    Raises:
        RuntimeError: When no active ligands or an empty target vocabulary.
    """
    papyrus_by_sequence = load_papyrus_classification_by_sequence(
        papyrus_proteins,
        version=version,
        papyrus_root=papyrus_root,
        verbose=verbose,
    )
    by_sequence = build_gpcrdb_protein_sidecar(
        activity_source,
        papyrus_by_sequence,
        limit=limit,
        verbose=verbose,
    )
    if not by_sequence:
        raise RuntimeError(
            f"No proteins extracted from {activity_source}. "
            "Check that Target Name and sequences are present."
        )
    write_proteins_parquet(by_sequence, ml_config.GPCRDB_PROTEINS_SIDECAR)
    if verbose:
        print(f"[gpcrdb-multilabel] wrote {ml_config.GPCRDB_PROTEINS_SIDECAR}", flush=True)

    by_ligand = aggregate_ligand_labels(
        activity_source,
        by_sequence,
        threshold_nm=threshold_nm,
        classification_depth=classification_depth,
        limit=limit,
        verbose=verbose,
    )
    if not by_ligand:
        raise RuntimeError(
            f"No active ligands aggregated from {activity_source}. "
            "Check the activity source and binder threshold."
        )

    smiles_order = sorted(by_ligand)
    years = years_array(smiles_order, by_ligand)
    year_cells: list[Optional[int]] = [None if int(y) < 0 else int(y) for y in years]

    written: dict[str, str] = {"proteins": ml_config.GPCRDB_PROTEINS_SIDECAR}

    # --- Target (required) -------------------------------------------------
    target_counts = count_target_actives(by_ligand)
    target_vocab = build_target_vocab(
        target_counts,
        min_positives=min_positives,
        max_size=target_vocab_size,
    )
    if not target_vocab:
        raise RuntimeError(
            "Target vocabulary is empty; lower --min-positives or check data."
        )
    save_vocab(
        ml_config.GPCRDB_TARGET_VOCAB_PATH,
        target_vocab,
        meta={
            "task": "target",
            "target_id_scheme": "gpcrdb_entry_name",
            "min_positives": int(min_positives),
            "max_size": int(target_vocab_size),
            "activity_threshold_nm": float(threshold_nm),
            "activity_source": os.path.realpath(activity_source),
            "active_counts": {tid: int(target_counts[tid]) for tid in target_vocab},
        },
    )
    target_set = set(target_vocab)
    target_lists = [
        sorted(label for label in by_ligand[s].targets if label in target_set)
        for s in smiles_order
    ]
    target_keep = [i for i, labels in enumerate(target_lists) if labels]
    if not target_keep:
        raise RuntimeError("No ligands remain after target vocab filtering.")
    _write_ligand_prepared(
        ml_config.GPCRDB_TARGET_PREPARED,
        [smiles_order[i] for i in target_keep],
        [year_cells[i] for i in target_keep],
        [target_lists[i] for i in target_keep],
        "target_labels",
    )
    written["target_prepared"] = ml_config.GPCRDB_TARGET_PREPARED
    written["target_vocab"] = ml_config.GPCRDB_TARGET_VOCAB_PATH
    if verbose:
        target_matrix = build_label_matrix(
            [smiles_order[i] for i in target_keep],
            by_ligand,
            target_vocab,
            kind="target",
        )
        print(
            f"[gpcrdb-multilabel] target: ligands={len(target_keep)} "
            f"K={len(target_vocab)} "
            f"mean_labels={float(target_matrix.sum(axis=1).mean()):.2f}",
            flush=True,
        )
        print(f"[gpcrdb-multilabel] wrote {ml_config.GPCRDB_TARGET_PREPARED}", flush=True)
        print(f"[gpcrdb-multilabel] wrote {ml_config.GPCRDB_TARGET_VOCAB_PATH}", flush=True)

    # --- Family (optional when Classification coverage is thin) ------------
    family_counts = count_family_actives(by_ligand)
    family_vocab = build_family_vocab(
        (meta["Classification"] for meta in by_sequence.values()),
        depth=classification_depth,
    )
    family_vocab = filter_vocab_by_min_positives(
        family_vocab,
        family_counts,
        min_positives=min_positives,
    )
    if not family_vocab:
        message = (
            "Family vocabulary is empty after filtering; skipping GPCRdb family "
            "artifacts (Classification join may be missing or too sparse). "
            "Target artifacts were written."
        )
        warnings.warn(message, UserWarning, stacklevel=2)
        if verbose:
            print(f"[gpcrdb-multilabel] WARNING: {message}", flush=True)
        return written

    save_vocab(
        ml_config.GPCRDB_FAMILY_VOCAB_PATH,
        family_vocab,
        meta={
            "task": "family",
            "classification_depth": classification_depth,
            "min_positives": int(min_positives),
            "activity_threshold_nm": float(threshold_nm),
            "activity_source": os.path.realpath(activity_source),
            "active_counts": {
                label: int(family_counts[label]) for label in family_vocab
            },
        },
    )
    family_set = set(family_vocab)
    family_lists = [
        sorted(label for label in by_ligand[s].families if label in family_set)
        for s in smiles_order
    ]
    family_keep = [i for i, labels in enumerate(family_lists) if labels]
    if not family_keep:
        message = (
            "No ligands remain after family vocab filtering; skipping GPCRdb "
            "family prepared table."
        )
        warnings.warn(message, UserWarning, stacklevel=2)
        if verbose:
            print(f"[gpcrdb-multilabel] WARNING: {message}", flush=True)
        written["family_vocab"] = ml_config.GPCRDB_FAMILY_VOCAB_PATH
        return written

    _write_ligand_prepared(
        ml_config.GPCRDB_FAMILY_PREPARED,
        [smiles_order[i] for i in family_keep],
        [year_cells[i] for i in family_keep],
        [family_lists[i] for i in family_keep],
        "family_labels",
    )
    written["family_prepared"] = ml_config.GPCRDB_FAMILY_PREPARED
    written["family_vocab"] = ml_config.GPCRDB_FAMILY_VOCAB_PATH
    if verbose:
        family_matrix = build_label_matrix(
            [smiles_order[i] for i in family_keep],
            by_ligand,
            family_vocab,
            kind="family",
        )
        print(
            f"[gpcrdb-multilabel] family: ligands={len(family_keep)} "
            f"K={len(family_vocab)} "
            f"mean_labels={float(family_matrix.sum(axis=1).mean()):.2f}",
            flush=True,
        )
        print(f"[gpcrdb-multilabel] wrote {ml_config.GPCRDB_FAMILY_PREPARED}", flush=True)
        print(f"[gpcrdb-multilabel] wrote {ml_config.GPCRDB_FAMILY_VOCAB_PATH}", flush=True)

    return written


def build_arg_parser() -> argparse.ArgumentParser:
    """Construct the CLI argument parser.

    Returns:
        Configured argument parser.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Build experimental ligand family/target multilabel prepared tables "
            "from a GPCRdb pair prepared Parquet. Target IDs are GPCRdb Entry "
            "names; Classification is joined from Papyrus by sequence when "
            "available."
        )
    )
    parser.add_argument(
        "--activity-source",
        default=ml_config.DEFAULT_GPCRDB_ACTIVITY_SOURCE,
        help="GPCRdb pair prepared Parquet used for active ligand aggregation.",
    )
    parser.add_argument(
        "--papyrus-proteins",
        default=None,
        help=(
            "Papyrus proteins Parquet for Classification join "
            f"(default: {ml_config.PROTEINS_SIDECAR} if present)."
        ),
    )
    parser.add_argument(
        "--version",
        default="latest",
        help="Papyrus version if proteins sidecar must be loaded from Papyrus.",
    )
    parser.add_argument(
        "--papyrus-root",
        default=None,
        help="Optional Papyrus data directory override.",
    )
    parser.add_argument(
        "--threshold-nm",
        type=float,
        default=ml_config.ACTIVITY_THRESHOLD_NM,
        help="Quantitative binder cutoff in nM (default matches pair training).",
    )
    parser.add_argument(
        "--classification-depth",
        type=int,
        default=ml_config.CLASSIFICATION_DEPTH,
        help="1-based Classification path depth for family labels (default 2).",
    )
    parser.add_argument(
        "--min-positives",
        type=int,
        default=ml_config.MIN_POSITIVES,
        help=(
            "Minimum unique active ligands required for a family or target "
            "label to be included in the vocabulary (default 100)."
        ),
    )
    parser.add_argument(
        "--target-vocab-size",
        type=int,
        default=ml_config.TARGET_VOCAB_SIZE,
        help="Maximum number of target_id labels (top by active count).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on streamed prepared activity rows.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output.",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> None:
    """CLI entry point.

    Args:
        argv: Optional argument list (defaults to ``sys.argv[1:]``).

    Returns:
        None.
    """
    args = build_arg_parser().parse_args(argv)
    build_multilabel_tables(
        activity_source=args.activity_source,
        papyrus_proteins=args.papyrus_proteins,
        version=args.version,
        papyrus_root=args.papyrus_root,
        threshold_nm=args.threshold_nm,
        classification_depth=args.classification_depth,
        min_positives=args.min_positives,
        target_vocab_size=args.target_vocab_size,
        limit=args.limit,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
