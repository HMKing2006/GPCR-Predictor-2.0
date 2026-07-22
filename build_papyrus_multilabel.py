"""Build ligand-centric family/target multilabel prepared tables from Papyrus.

Streams an existing pair prepared Parquet (unchanged schema), joins protein
``Classification`` / ``target_id`` via sequence, aggregates active annotations
per ligand, and writes:

* ``data/train/Papyrus_proteins_multilabel.parquet``
* ``data/train/Ligand_family_prepared.parquet`` + ``family_vocab.json``
* ``data/train/Ligand_target_prepared.parquet`` + ``target_vocab.json``
"""

from __future__ import annotations

import argparse
import os
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


def load_protein_sidecar(
    version: str,
    papyrus_root: Optional[str],
    verbose: bool = True,
) -> dict[str, dict[str, str]]:
    """Load Papyrus proteins keyed by amino-acid sequence.

    Args:
        version: Papyrus version string (e.g. ``"latest"``).
        papyrus_root: Optional Papyrus data directory override.
        verbose: If ``True``, print load stats.

    Returns:
        Mapping ``sequence -> {target_id, Classification, TID, UniProtID}``.
    """
    from build_papyrus import resolve_papyrus_version, sync_papyrus_version_layout
    from papyrus_scripts import read_protein_set

    sync_papyrus_version_layout(papyrus_root, verbose=verbose)
    resolved = resolve_papyrus_version(version, papyrus_root)
    proteins = read_protein_set(source_path=papyrus_root, version=resolved)
    by_sequence: dict[str, dict[str, str]] = {}
    collisions = 0
    for row in proteins.itertuples(index=False):
        target_id = str(getattr(row, "target_id", "") or "").strip()
        sequence = str(getattr(row, "Sequence", "") or "").strip()
        if not target_id or not sequence or sequence.lower() == "nan":
            continue
        classification = str(getattr(row, "Classification", "") or "").strip()
        if classification.lower() == "nan":
            classification = ""
        tid = str(getattr(row, "TID", "") or "").strip()
        uniprot = str(getattr(row, "UniProtID", "") or "").strip()
        if tid.lower() == "nan":
            tid = ""
        if uniprot.lower() == "nan":
            uniprot = ""
        record = {
            "target_id": target_id,
            "Classification": classification,
            "TID": tid,
            "UniProtID": uniprot,
            "Sequence": sequence,
        }
        existing = by_sequence.get(sequence)
        if existing is not None and existing["target_id"] != target_id:
            collisions += 1
            continue
        by_sequence[sequence] = record
    if verbose:
        print(
            f"[multilabel] loaded {len(by_sequence)} unique sequences "
            f"(skipped_collisions={collisions}) version={resolved!r}",
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
    activity_source: str = ml_config.DEFAULT_ACTIVITY_SOURCE,
    version: str = "latest",
    papyrus_root: Optional[str] = None,
    threshold_nm: float = ml_config.ACTIVITY_THRESHOLD_NM,
    classification_depth: int = ml_config.CLASSIFICATION_DEPTH,
    min_positives: int = ml_config.MIN_POSITIVES,
    target_vocab_size: int = ml_config.TARGET_VOCAB_SIZE,
    limit: Optional[int] = None,
    verbose: bool = True,
) -> dict[str, str]:
    """Build protein sidecar, vocabularies, and ligand prepared tables.

    Args:
        activity_source: Pair prepared Parquet / CSV path.
        version: Papyrus version for protein metadata.
        papyrus_root: Optional Papyrus data directory.
        threshold_nm: Quantitative binder cutoff.
        classification_depth: Family Classification path depth.
        min_positives: Minimum unique active ligands per family/target label.
        target_vocab_size: Cap on target vocabulary size.
        limit: Optional activity-row stream limit.
        verbose: If ``True``, print progress.

    Returns:
        Mapping of artifact name to written path.
    """
    by_sequence = load_protein_sidecar(version, papyrus_root, verbose=verbose)
    write_proteins_parquet(by_sequence, ml_config.PROTEINS_SIDECAR)
    if verbose:
        print(f"[multilabel] wrote {ml_config.PROTEINS_SIDECAR}", flush=True)

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
    family_counts = count_family_actives(by_ligand)
    family_vocab = build_family_vocab(
        (meta["Classification"] for meta in by_sequence.values()),
        depth=classification_depth,
    )
    # Restrict to families with enough unique active ligands.
    family_vocab = filter_vocab_by_min_positives(
        family_vocab,
        family_counts,
        min_positives=min_positives,
    )
    if not family_vocab:
        raise RuntimeError(
            "Family vocabulary is empty; lower --min-positives or check data."
        )

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
        ml_config.FAMILY_VOCAB_PATH,
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
    save_vocab(
        ml_config.TARGET_VOCAB_PATH,
        target_vocab,
        meta={
            "task": "target",
            "min_positives": int(min_positives),
            "max_size": int(target_vocab_size),
            "activity_threshold_nm": float(threshold_nm),
            "activity_source": os.path.realpath(activity_source),
            "active_counts": {tid: int(target_counts[tid]) for tid in target_vocab},
        },
    )

    years = years_array(smiles_order, by_ligand)
    year_cells: list[Optional[int]] = [
        None if int(y) < 0 else int(y) for y in years
    ]
    family_lists = [
        sorted(label for label in by_ligand[s].families if label in set(family_vocab))
        for s in smiles_order
    ]
    target_set = set(target_vocab)
    target_lists = [
        sorted(label for label in by_ligand[s].targets if label in target_set)
        for s in smiles_order
    ]

    # Drop ligands with an empty multi-hot after vocab filtering.
    family_keep = [i for i, labels in enumerate(family_lists) if labels]
    target_keep = [i for i, labels in enumerate(target_lists) if labels]
    if not family_keep:
        raise RuntimeError("No ligands remain after family vocab filtering.")
    if not target_keep:
        raise RuntimeError("No ligands remain after target vocab filtering.")

    _write_ligand_prepared(
        ml_config.FAMILY_PREPARED,
        [smiles_order[i] for i in family_keep],
        [year_cells[i] for i in family_keep],
        [family_lists[i] for i in family_keep],
        "family_labels",
    )
    _write_ligand_prepared(
        ml_config.TARGET_PREPARED,
        [smiles_order[i] for i in target_keep],
        [year_cells[i] for i in target_keep],
        [target_lists[i] for i in target_keep],
        "target_labels",
    )

    if verbose:
        family_matrix = build_label_matrix(
            [smiles_order[i] for i in family_keep],
            by_ligand,
            family_vocab,
            kind="family",
        )
        target_matrix = build_label_matrix(
            [smiles_order[i] for i in target_keep],
            by_ligand,
            target_vocab,
            kind="target",
        )
        print(
            f"[multilabel] family: ligands={len(family_keep)} K={len(family_vocab)} "
            f"mean_labels={float(family_matrix.sum(axis=1).mean()):.2f}",
            flush=True,
        )
        print(
            f"[multilabel] target: ligands={len(target_keep)} K={len(target_vocab)} "
            f"mean_labels={float(target_matrix.sum(axis=1).mean()):.2f}",
            flush=True,
        )
        print(f"[multilabel] wrote {ml_config.FAMILY_PREPARED}", flush=True)
        print(f"[multilabel] wrote {ml_config.TARGET_PREPARED}", flush=True)
        print(f"[multilabel] wrote {ml_config.FAMILY_VOCAB_PATH}", flush=True)
        print(f"[multilabel] wrote {ml_config.TARGET_VOCAB_PATH}", flush=True)

    return {
        "proteins": ml_config.PROTEINS_SIDECAR,
        "family_prepared": ml_config.FAMILY_PREPARED,
        "target_prepared": ml_config.TARGET_PREPARED,
        "family_vocab": ml_config.FAMILY_VOCAB_PATH,
        "target_vocab": ml_config.TARGET_VOCAB_PATH,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    """Construct the CLI argument parser.

    Returns:
        Configured argument parser.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Build experimental ligand family/target multilabel prepared tables "
            "from a pair prepared Parquet and Papyrus protein Classification."
        )
    )
    parser.add_argument(
        "--activity-source",
        default=ml_config.DEFAULT_ACTIVITY_SOURCE,
        help="Pair prepared Parquet/CSV used for active ligand aggregation.",
    )
    parser.add_argument(
        "--version",
        default="latest",
        help="Papyrus version for protein Classification metadata.",
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
