"""Download Papyrus and write a BindingDB-compatible prepared CSV.

Streams the Papyrus++ or full without_stereochemistry release, keeps exact
quantitative Ki/Kd/IC50/EC50 values, joins protein sequences, and writes a CSV
that :func:`src.data_prep.iter_prepared_rows` can consume unchanged::

    python build_papyrus.py --subset plusplus
    python train.py --csv data/train/Papyrus_pp_prepared.csv --rebuild-features

    python build_papyrus.py --subset full
    python train.py --csv data/train/Papyrus_full_prepared.csv --rebuild-features

The full subset is substantially larger (multi-GB download; the prepared CSV
may exceed 10 GB and take hours to build). Prefer ``--subset plusplus`` unless
you specifically need the full release. Use ``--limit`` for smoke tests.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from typing import Any, Iterator, Optional

import numpy as np
import pandas as pd

import config
from src.data_prep import canonicalize_smiles

# Output columns in the BindingDB prepared-CSV schema.
_OUTPUT_COLUMNS: list[str] = [
    "Ligand SMILES",
    "Target Name",
    "Target Source Organism According to Curator or DataSource",
    "Ki (nM)",
    "IC50 (nM)",
    "Kd (nM)",
    "EC50 (nM)",
    "pH",
    "Temp (C)",
    "BindingDB Target Chain Sequence 1",
]

# Papyrus type_* column -> BindingDB assay column. Order is the preference used
# when more than one type flag is set on a row (rare).
_TYPE_TO_ASSAY: list[tuple[str, str]] = [
    ("type_Ki", "Ki (nM)"),
    ("type_KD", "Kd (nM)"),
    ("type_IC50", "IC50 (nM)"),
    ("type_EC50", "EC50 (nM)"),
]

_ACTIVITY_TYPES: list[str] = ["Ki", "KD", "IC50", "EC50"]


def _require_papyrus() -> None:
    """Import-check papyrus-scripts and exit with a clear message if missing.

    Returns:
        None.

    Raises:
        SystemExit: If ``papyrus-scripts`` is not installed.
    """
    try:
        import papyrus_scripts  # noqa: F401
    except ImportError:
        print(
            "papyrus-scripts is required. Install with:\n"
            "  pip install 'papyrus-scripts>=2.0'",
            file=sys.stderr,
        )
        raise SystemExit(1) from None


def _default_output(subset: str) -> str:
    """Return the default prepared-CSV path for a subset name.

    Args:
        subset: Either ``"plusplus"`` or ``"full"``.

    Returns:
        Absolute path under ``data/train/``.
    """
    if subset == "plusplus":
        return config.PAPYRUS_PP_TRAIN_CSV
    return config.PAPYRUS_FULL_TRAIN_CSV


def _papyrus_data_root(papyrus_root: Optional[str]) -> str:
    """Return the on-disk ``papyrus/`` root directory used by papyrus-scripts.

    Args:
        papyrus_root: Optional override passed as ``outdir`` / ``source_path``.

    Returns:
        Absolute path of the Papyrus data root.
    """
    import pystow

    if papyrus_root is not None:
        os.environ["PYSTOW_HOME"] = os.path.abspath(papyrus_root)
    return pystow.module("papyrus").base.as_posix()


def sync_papyrus_version_layout(papyrus_root: Optional[str], verbose: bool = True) -> None:
    """Repair papyrus-scripts download/read folder mismatch.

    Recent Papyrus releases download into folders named like ``2024.09.2``, but
    :func:`papyrus_scripts.read_papyrus` looks up data under the legacy folder
    ``05.7``. This helper creates the missing legacy symlink and updates
    ``versions.json`` so readers accept ``latest`` / ``05.7``.

    Args:
        papyrus_root: Optional Papyrus data directory override.
        verbose: If ``True``, print repair actions.

    Returns:
        None.
    """
    import json
    from pathlib import Path

    from papyrus_scripts.utils.IO import PapyrusVersion

    root = Path(_papyrus_data_root(papyrus_root))
    versions_path = root / "versions.json"
    if not versions_path.exists():
        return

    recorded = json.loads(versions_path.read_text())
    if not isinstance(recorded, list):
        return

    updated = list(recorded)
    aliases = PapyrusVersion.aliases
    for entry in list(recorded):
        folder = root / str(entry)
        if not folder.exists():
            continue
        # Match either alias ("2024.09") or full "2024.09.2" / legacy "05.7".
        match = aliases.query(f'version == "{entry}" or alias == "{entry}"')
        if match.empty and "." in str(entry):
            # Strip trailing revision: 2024.09.2 -> 2024.09
            parts = str(entry).rsplit(".", 1)
            if len(parts) == 2 and parts[1].isdigit():
                match = aliases.query(f'alias == "{parts[0]}"')
        if match.empty:
            continue
        old_fmt = str(match.iloc[0]["version"])
        legacy = root / old_fmt
        if not legacy.exists():
            legacy.symlink_to(folder.name, target_is_directory=True)
            if verbose:
                print(f"[papyrus] linked {legacy.name} -> {folder.name}")
        if old_fmt not in updated:
            updated.append(old_fmt)

    if updated != recorded:
        versions_path.write_text(json.dumps(updated, indent=2) + "\n")
        if verbose:
            print(f"[papyrus] updated versions.json -> {updated}")


def resolve_papyrus_version(version: str, papyrus_root: Optional[str]) -> str:
    """Resolve a user version request to a reader-compatible version string.

    Args:
        version: User-supplied version (e.g. ``"latest"`` or ``"05.7"``).
        papyrus_root: Optional Papyrus data directory override.

    Returns:
        A version string accepted by ``read_papyrus`` / ``read_protein_set``
        (preferring the legacy ``05.x`` form when available).
    """
    from papyrus_scripts.utils.IO import PapyrusVersion, get_downloaded_versions

    sync_papyrus_version_layout(papyrus_root, verbose=False)
    available = get_downloaded_versions(papyrus_root)
    if not available:
        raise RuntimeError(
            "Papyrus data is not available after download. This often means "
            "papyrus-scripts aborted due to its free-disk-space check. Retry "
            "with --disk-margin 0 (default) or free more disk space."
        )
    if version == "latest":
        pv = PapyrusVersion(version="latest")
        if pv.version_old_fmt in available:
            return pv.version_old_fmt
        # Fall back to whatever was recorded as downloaded.
        return sorted(available)[-1]
    return version


def ensure_downloaded(
    version: str,
    plusplus: bool,
    papyrus_root: Optional[str],
    *,
    do_download: bool,
    disk_margin: float,
    verbose: bool,
) -> str:
    """Download the requested Papyrus release if missing (or always when asked).

    Args:
        version: Papyrus version string (e.g. ``"latest"`` or ``"05.7"``).
        plusplus: If ``True``, download only Papyrus++; otherwise the full
            without_stereochemistry set.
        papyrus_root: Optional override for the Papyrus data directory.
        do_download: If ``False``, skip the download call (assumes files exist).
        disk_margin: Fraction of total disk capacity that must remain free after
            download (passed through to ``download_papyrus``). Use ``0.0`` when
            the volume is nearly full but still has room for the archive.
        verbose: If ``True``, print download progress.

    Returns:
        Reader-compatible Papyrus version string.
    """
    if do_download:
        from papyrus_scripts import download_papyrus

        if verbose:
            label = "Papyrus++" if plusplus else "full Papyrus (without_stereochemistry)"
            print(f"[papyrus] ensuring {label} version={version!r} is available")
        download_papyrus(
            outdir=papyrus_root,
            version=version,
            nostereo=True,
            stereo=False,
            only_pp=plusplus,
            structures=False,
            descriptors=[],
            progress=verbose,
            disk_margin=disk_margin,
        )
    elif verbose:
        print("[papyrus] skipping download (--no-download)")

    sync_papyrus_version_layout(papyrus_root, verbose=verbose)
    resolved = resolve_papyrus_version(version, papyrus_root)
    if verbose:
        print(f"[papyrus] using version={resolved!r}")
    return resolved


def load_protein_table(
    version: str,
    papyrus_root: Optional[str],
    verbose: bool = True,
) -> dict[str, dict[str, str]]:
    """Load protein metadata keyed by ``target_id``.

    Args:
        version: Papyrus version string.
        papyrus_root: Optional Papyrus data directory override.
        verbose: If ``True``, print the number of proteins loaded.

    Returns:
        Mapping ``target_id -> {sequence, organism, name}``.
    """
    from papyrus_scripts import read_protein_set

    proteins = read_protein_set(source_path=papyrus_root, version=version)
    lookup: dict[str, dict[str, str]] = {}
    for row in proteins.itertuples(index=False):
        target_id = str(getattr(row, "target_id", "") or "").strip()
        sequence = str(getattr(row, "Sequence", "") or "").strip()
        if not target_id or not sequence or sequence.lower() == "nan":
            continue
        tid = str(getattr(row, "TID", "") or "").strip()
        uniprot = str(getattr(row, "UniProtID", "") or "").strip()
        name = tid if tid and tid.lower() != "nan" else uniprot
        organism = str(getattr(row, "Organism", "") or "").strip()
        if organism.lower() == "nan":
            organism = ""
        lookup[target_id] = {
            "sequence": sequence,
            "organism": organism,
            "name": name if name.lower() != "nan" else target_id,
        }
    if verbose:
        print(f"[papyrus] loaded {len(lookup)} proteins with sequences")
    return lookup


def pchembl_to_nm(pchembl: float) -> Optional[float]:
    """Convert a pChEMBL / pActivity value to activity in nM.

    Args:
        pchembl: Activity as ``-log10(M)``.

    Returns:
        Positive activity in nM, or ``None`` if the value is non-finite or
        non-positive after conversion.
    """
    if not math.isfinite(pchembl):
        return None
    activity_nm = 10.0 ** (9.0 - pchembl)
    if not math.isfinite(activity_nm) or activity_nm <= 0.0:
        return None
    return float(activity_nm)


def _assay_column(row: pd.Series) -> Optional[str]:
    """Pick the BindingDB assay column for a Papyrus activity row.

    Args:
        row: A filtered Papyrus activity row.

    Returns:
        One of ``Ki (nM)``, ``Kd (nM)``, ``IC50 (nM)``, ``EC50 (nM)``, or
        ``None`` if no recognized type flag is set.
    """
    for type_col, assay_col in _TYPE_TO_ASSAY:
        if type_col not in row.index:
            continue
        value = row[type_col]
        if pd.isna(value):
            continue
        if isinstance(value, (bool, np.bool_)) and bool(value):
            return assay_col
        if isinstance(value, (int, float, np.integer, np.floating)) and float(value) == 1.0:
            return assay_col
        if isinstance(value, str) and value.strip().lower() in {"1", "true", "yes"}:
            return assay_col
    return None


def convert_chunk(
    chunk: pd.DataFrame,
    proteins: dict[str, dict[str, str]],
    stats: dict[str, int],
    seen_ligands: set[str],
    seen_proteins: set[str],
    limit: Optional[int],
    rows_written: int,
) -> tuple[list[dict[str, Any]], int]:
    """Convert one Papyrus activity chunk into BindingDB-schema rows.

    Args:
        chunk: Filtered activity chunk (exact relation, selected assay types).
        proteins: Protein metadata keyed by ``target_id``.
        stats: Mutable skip/write counters updated in place.
        seen_ligands: Mutable set of canonical SMILES written so far.
        seen_proteins: Mutable set of sequences written so far.
        limit: Optional cap on total output rows across all chunks.
        rows_written: Number of rows already written before this chunk.

    Returns:
        A tuple ``(rows, rows_written_after)`` where ``rows`` is a list of
        dictionaries ready for ``csv.DictWriter``.
    """
    out: list[dict[str, Any]] = []
    for _, row in chunk.iterrows():
        if limit is not None and rows_written + len(out) >= limit:
            break

        relation = str(row.get("relation", "") or "").strip()
        if relation != "=":
            stats["skipped_relation"] += 1
            continue

        pchembl_raw = row.get("pchembl_value")
        try:
            pchembl = float(pchembl_raw)
        except (TypeError, ValueError):
            stats["skipped_activity"] += 1
            continue
        activity_nm = pchembl_to_nm(pchembl)
        if activity_nm is None:
            stats["skipped_activity"] += 1
            continue

        assay_col = _assay_column(row)
        if assay_col is None:
            stats["skipped_assay"] += 1
            continue

        target_id = str(row.get("target_id", "") or "").strip()
        protein = proteins.get(target_id)
        if protein is None:
            stats["skipped_sequence"] += 1
            continue

        smiles = canonicalize_smiles(str(row.get("SMILES", "") or ""))
        if smiles is None:
            stats["skipped_smiles"] += 1
            continue

        record: dict[str, Any] = {col: "" for col in _OUTPUT_COLUMNS}
        record["Ligand SMILES"] = smiles
        record["Target Name"] = protein["name"]
        record["Target Source Organism According to Curator or DataSource"] = protein["organism"]
        record[assay_col] = f"{activity_nm:.6g}"
        record["BindingDB Target Chain Sequence 1"] = protein["sequence"]
        out.append(record)
        seen_ligands.add(smiles)
        seen_proteins.add(protein["sequence"])
        stats["written"] += 1

    return out, rows_written + len(out)


def iter_filtered_chunks(
    version: str,
    plusplus: bool,
    papyrus_root: Optional[str],
    chunk_size: int,
    min_quality: str,
    verbose: bool,
) -> Iterator[pd.DataFrame]:
    """Yield filtered Papyrus activity chunks ready for conversion.

    Args:
        version: Papyrus version string.
        plusplus: If ``True``, read Papyrus++; otherwise the full set.
        papyrus_root: Optional Papyrus data directory override.
        chunk_size: Rows per streaming chunk.
        min_quality: Minimum quality tier for the full subset
            (``high`` / ``medium`` / ``low``). Ignored for Papyrus++.
        verbose: If ``True``, print progress notes.

    Yields:
        Filtered pandas DataFrames.
    """
    from papyrus_scripts import read_papyrus
    from papyrus_scripts.preprocess import keep_quality, keep_type

    if verbose:
        label = "Papyrus++" if plusplus else "full Papyrus"
        print(
            f"[papyrus] streaming {label} version={version!r} "
            f"chunk_size={chunk_size}",
            flush=True,
        )
    reader = read_papyrus(
        is3d=False,
        version=version,
        plusplus=plusplus,
        chunksize=chunk_size,
        source_path=papyrus_root,
    )
    # read_papyrus with chunksize returns an iterator / TextFileReader.
    stream: Any = reader
    if not plusplus:
        stream = keep_quality(data=stream, min_quality=min_quality)
        if verbose:
            print(f"[papyrus] applying min_quality={min_quality!r}", flush=True)
    stream = keep_type(data=stream, activity_types=_ACTIVITY_TYPES, njobs=1, verbose=False)

    for chunk_idx, chunk in enumerate(stream, start=1):
        if not isinstance(chunk, pd.DataFrame):
            continue
        if "relation" in chunk.columns:
            chunk = chunk[chunk["relation"].astype(str).str.strip() == "="]
        if chunk.empty:
            continue
        if verbose and chunk_idx % 5 == 0:
            print(f"[papyrus] processed {chunk_idx} chunks...", flush=True)
        yield chunk


def run_build(args: argparse.Namespace) -> str:
    """Download (optional), convert, and write the prepared Papyrus CSV.

    Args:
        args: Parsed CLI arguments.

    Returns:
        Path of the written prepared CSV.
    """
    _require_papyrus()
    verbose = not args.quiet
    plusplus = args.subset == "plusplus"
    output = args.output or _default_output(args.subset)
    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)

    resolved_version = ensure_downloaded(
        version=args.version,
        plusplus=plusplus,
        papyrus_root=args.papyrus_root,
        do_download=not args.no_download,
        disk_margin=args.disk_margin,
        verbose=verbose,
    )
    proteins = load_protein_table(resolved_version, args.papyrus_root, verbose=verbose)

    stats: dict[str, int] = {
        "written": 0,
        "skipped_relation": 0,
        "skipped_activity": 0,
        "skipped_assay": 0,
        "skipped_sequence": 0,
        "skipped_smiles": 0,
    }
    seen_ligands: set[str] = set()
    seen_proteins: set[str] = set()
    rows_written = 0

    with open(output, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=_OUTPUT_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for chunk in iter_filtered_chunks(
            version=resolved_version,
            plusplus=plusplus,
            papyrus_root=args.papyrus_root,
            chunk_size=args.chunk_size,
            min_quality=args.min_quality,
            verbose=verbose,
        ):
            rows, rows_written = convert_chunk(
                chunk,
                proteins,
                stats,
                seen_ligands,
                seen_proteins,
                args.limit,
                rows_written,
            )
            if rows:
                writer.writerows(rows)
                handle.flush()
            if args.limit is not None and rows_written >= args.limit:
                if verbose:
                    print(f"[papyrus] reached --limit {args.limit}", flush=True)
                break

    if verbose:
        print(
            f"\n[papyrus] subset={args.subset} wrote {stats['written']} rows -> {output}\n"
            f"[papyrus] unique ligands={len(seen_ligands)}  "
            f"unique proteins={len(seen_proteins)}\n"
            f"[papyrus] skipped: relation={stats['skipped_relation']}  "
            f"activity={stats['skipped_activity']}  assay={stats['skipped_assay']}  "
            f"sequence={stats['skipped_sequence']}  smiles={stats['skipped_smiles']}",
            flush=True,
        )
    if stats["written"] == 0:
        raise RuntimeError(
            "No rows written. Check that Papyrus data downloaded correctly and "
            "that filters (relation='=', assay types, sequences) are not too strict."
        )
    return output


def build_arg_parser() -> argparse.ArgumentParser:
    """Construct the command-line argument parser.

    Returns:
        The configured :class:`argparse.ArgumentParser`.
    """
    p = argparse.ArgumentParser(
        description=(
            "Build a BindingDB-compatible prepared CSV from Papyrus++ or the "
            "full without_stereochemistry Papyrus release. The full subset can "
            "produce a multi-GB CSV and take hours; prefer --subset plusplus "
            "unless you need the full release."
        )
    )
    p.add_argument(
        "--subset",
        choices=["plusplus", "full"],
        default="plusplus",
        help="Papyrus++ (default) or full without_stereochemistry release.",
    )
    p.add_argument(
        "--version",
        default="latest",
        help="Papyrus version string (default: latest).",
    )
    p.add_argument(
        "--output",
        default=None,
        help=(
            "Output CSV path (default: data/train/Papyrus_pp_prepared.csv or "
            "Papyrus_full_prepared.csv)."
        ),
    )
    p.add_argument(
        "--min-quality",
        choices=["high", "medium", "low"],
        default="low",
        help="Minimum quality for --subset full (default: low = all tiers).",
    )
    p.add_argument(
        "--no-download",
        action="store_true",
        help="Do not call download_papyrus; assume data is already present.",
    )
    p.add_argument(
        "--papyrus-root",
        default=None,
        help="Directory containing / receiving Papyrus data (default: pystow).",
    )
    p.add_argument(
        "--disk-margin",
        type=float,
        default=0.0,
        help=(
            "Fraction of total disk capacity that must remain free after download "
            "(papyrus-scripts default is 0.10; use 0.0 when free space is tight)."
        ),
    )
    p.add_argument(
        "--chunk-size",
        type=int,
        default=500_000,
        help="Rows per streaming chunk (default: 500000).",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Cap on output rows (smoke tests).",
    )
    p.add_argument("--quiet", action="store_true", help="Reduce progress output.")
    return p


def main(argv: Optional[list[str]] = None) -> None:
    """CLI entry point.

    Args:
        argv: Optional argument list (defaults to ``sys.argv``).

    Returns:
        None.
    """
    args = build_arg_parser().parse_args(argv)
    run_build(args)


if __name__ == "__main__":
    main()
