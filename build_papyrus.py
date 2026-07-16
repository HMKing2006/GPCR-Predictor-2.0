"""Download Papyrus and write a BindingDB-compatible prepared Parquet file.

Streams the Papyrus++ or full without_stereochemistry release, keeps exact
quantitative Ki/Kd/IC50/EC50/Other values, joins protein sequences, and writes
a Parquet file that :func:`src.data_prep.iter_prepared_rows` can consume::

    python build_papyrus.py --subset plusplus
    python train.py --csv data/train/Papyrus_pp_prepared.parquet --rebuild-features

    python build_papyrus.py --subset full
    python train.py --csv data/train/Papyrus_full_prepared.parquet --rebuild-features

    # Quant + Papyrus Activity_class binary rows (use --limit to cap size)
    python build_papyrus.py --subset full --include-binary --limit 5000000
    python train.py --csv data/train/Papyrus_full_binary_prepared.parquet --rebuild-features

    # Resume a failed binary build (skips quantitative pass, appends binary rows)
    python build_papyrus.py --subset full --include-binary --resume --no-download

The full subset is substantially larger (multi-GB download; the prepared
Parquet may exceed several GB and take hours to build). Prefer
``--subset plusplus`` unless you specifically need the full release. Use
``--limit`` for smoke tests.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from typing import Any, Iterator, Optional

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

import config
from src.data_prep import canonicalize_smiles

# Output columns in the BindingDB prepared schema (Parquet / CSV-compatible).
_OUTPUT_COLUMNS: list[str] = [
    "Ligand SMILES",
    "Target Name",
    "Target Source Organism According to Curator or DataSource",
    "Ki (nM)",
    "IC50 (nM)",
    "Kd (nM)",
    "EC50 (nM)",
    "Other (nM)",
    "Activity Label",
    "pH",
    "Temp (C)",
    "BindingDB Target Chain Sequence 1",
    "Year",
]

_PARQUET_SCHEMA = pa.schema(
    [
        ("Ligand SMILES", pa.string()),
        ("Target Name", pa.string()),
        ("Target Source Organism According to Curator or DataSource", pa.string()),
        ("Ki (nM)", pa.string()),
        ("IC50 (nM)", pa.string()),
        ("Kd (nM)", pa.string()),
        ("EC50 (nM)", pa.string()),
        ("Other (nM)", pa.string()),
        ("Activity Label", pa.string()),
        ("pH", pa.string()),
        ("Temp (C)", pa.string()),
        ("BindingDB Target Chain Sequence 1", pa.string()),
        ("Year", pa.int32()),
    ]
)

# Papyrus type_* column -> BindingDB assay column. Order is the preference used
# when more than one type flag is set on a row (rare).
_TYPE_TO_ASSAY: list[tuple[str, str]] = [
    ("type_Ki", "Ki (nM)"),
    ("type_KD", "Kd (nM)"),
    ("type_IC50", "IC50 (nM)"),
    ("type_EC50", "EC50 (nM)"),
    ("type_other", "Other (nM)"),
]

_TYPE_COLUMNS: list[str] = [col for col, _ in _TYPE_TO_ASSAY]
_QUALITY_LEVELS: tuple[str, ...] = ("low", "medium", "high")
_EXPLODE_COLUMNS: tuple[str, ...] = (
    "type_Ki",
    "type_KD",
    "type_IC50",
    "type_EC50",
    "type_other",
    "relation",
    "pchembl_value",
)
_ACTIVITY_CLASS_ACTIVE: frozenset[str] = frozenset({"y", "yes", "active", "1", "true"})
_ACTIVITY_CLASS_INACTIVE: frozenset[str] = frozenset({"n", "no", "inactive", "0", "false"})


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


def _default_output(subset: str, include_binary: bool = False) -> str:
    """Return the default prepared-CSV path for a subset name.

    Args:
        subset: Either ``"plusplus"`` or ``"full"``.
        include_binary: If ``True`` and ``subset`` is ``full``, return the
            quant+binary default path.

    Returns:
        Absolute path under ``data/train/``.
    """
    if subset == "plusplus":
        return config.PAPYRUS_PP_TRAIN_CSV
    if include_binary:
        return config.PAPYRUS_FULL_BINARY_TRAIN_CSV
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


def parse_year(raw: object) -> Optional[int]:
    """Parse a Papyrus publication year cell.

    Args:
        raw: Raw Year cell value.

    Returns:
        Integer year when parseable, else ``None``.
    """
    if raw is None or (isinstance(raw, float) and math.isnan(raw)):
        return None
    text = str(raw).strip()
    if not text or text.lower() == "nan":
        return None
    try:
        year = int(float(text))
    except (TypeError, ValueError):
        return None
    if year < 1900 or year > 2100:
        return None
    return year


def _records_to_table(rows: list[dict[str, Any]]) -> pa.Table:
    """Convert prepared row dictionaries to a typed Arrow table.

    Args:
        rows: Dictionaries with keys from ``_OUTPUT_COLUMNS``.

    Returns:
        A ``pyarrow.Table`` matching ``_PARQUET_SCHEMA``.
    """
    columns: dict[str, list[Any]] = {name: [] for name in _OUTPUT_COLUMNS}
    for record in rows:
        for name in _OUTPUT_COLUMNS:
            value = record.get(name)
            if name == "Year":
                columns[name].append(None if value in ("", None) else int(value))
            else:
                columns[name].append("" if value is None else str(value))
    arrays = []
    for field in _PARQUET_SCHEMA:
        if field.name == "Year":
            arrays.append(pa.array(columns[field.name], type=pa.int32()))
        else:
            arrays.append(pa.array(columns[field.name], type=pa.string()))
    return pa.Table.from_arrays(arrays, schema=_PARQUET_SCHEMA)


def _align_table_schema(table: pa.Table) -> pa.Table:
    """Cast / fill columns so ``table`` matches ``_PARQUET_SCHEMA``.

    Args:
        table: Source table that may be missing ``Year`` or use other types.

    Returns:
        A table with ``_PARQUET_SCHEMA``.
    """
    arrays = []
    for field in _PARQUET_SCHEMA:
        if field.name in table.column_names:
            col = table.column(field.name)
            arrays.append(
                col.cast(field.type, safe=False) if col.type != field.type else col
            )
        else:
            arrays.append(pa.nulls(table.num_rows, type=field.type))
    return pa.Table.from_arrays(arrays, schema=_PARQUET_SCHEMA)


def _copy_parquet_to_writer(source_path: str, writer: pq.ParquetWriter) -> int:
    """Stream all row groups from an existing Parquet file into a writer.

    Args:
        source_path: Existing Parquet path.
        writer: Open ``ParquetWriter`` receiving the copied batches.

    Returns:
        Number of rows copied.
    """
    copied = 0
    pf = pq.ParquetFile(source_path)
    for i in range(pf.num_row_groups):
        table = pf.read_row_group(i)
        if table.schema != _PARQUET_SCHEMA:
            table = _align_table_schema(table)
        writer.write_table(table)
        copied += table.num_rows
    return copied


def _validate_parquet_output(path: str, expected_rows: int) -> None:
    """Validate a completed prepared Parquet file before publication.

    Args:
        path: Temporary Parquet path.
        expected_rows: Number of rows produced by the conversion.

    Returns:
        None.

    Raises:
        ValueError: If the schema or row count is unexpected.
    """
    parquet = pq.ParquetFile(path)
    if parquet.metadata.num_rows != expected_rows:
        raise ValueError(
            f"Parquet validation failed for {path}: expected {expected_rows} rows, "
            f"found {parquet.metadata.num_rows}."
        )
    if parquet.schema_arrow != _PARQUET_SCHEMA:
        raise ValueError(
            f"Parquet validation failed for {path}: schema does not match "
            "the prepared-data schema."
        )


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


def _type_flag_true(series: pd.Series) -> pd.Series:
    """Return a boolean mask for Papyrus type_* columns set to active.

    Args:
        series: A ``type_Ki`` / ``type_KD`` / ``type_IC50`` / ``type_EC50`` /
            ``type_other`` column.

    Returns:
        Boolean Series aligned with ``series``.
    """
    if series.dtype == bool:
        return series.fillna(False)
    as_str = series.astype(str).str.strip().str.lower()
    return as_str.isin({"1", "1.0", "true", "yes"})


def _filter_quality(chunk: pd.DataFrame, min_quality: str) -> pd.DataFrame:
    """Keep rows at or above a Papyrus quality tier.

    Args:
        chunk: Raw Papyrus activity chunk.
        min_quality: Minimum tier (``high`` / ``medium`` / ``low``).

    Returns:
        Filtered chunk; unchanged when ``Quality`` is absent.
    """
    if "Quality" not in chunk.columns:
        return chunk
    tier = min_quality.lower()
    if tier not in _QUALITY_LEVELS:
        raise ValueError(f"min_quality must be one of {_QUALITY_LEVELS}")
    keep = _QUALITY_LEVELS[_QUALITY_LEVELS.index(tier) :]
    return chunk[chunk["Quality"].astype(str).str.lower().isin(keep)]


def _filter_exact_relation(chunk: pd.DataFrame) -> pd.DataFrame:
    """Keep only exact (``=``) quantitative relations.

    Args:
        chunk: Papyrus activity chunk.

    Returns:
        Filtered chunk; unchanged when ``relation`` is absent.
    """
    if "relation" not in chunk.columns:
        return chunk
    return chunk[chunk["relation"].astype(str).str.strip() == "="]


def _explode_semicolon_rows(chunk: pd.DataFrame, columns: tuple[str, ...]) -> pd.DataFrame:
    """Explode rows whose semicolon-separated fields encode multiple measurements.

    Args:
        chunk: Rows containing at least one semicolon-separated field.
        columns: Column names that may contain semicolon-separated values.

    Returns:
        One row per exploded measurement, with list lengths equalized by
        repeating the last value in each field (matching papyrus-scripts).
    """
    if chunk.empty:
        return chunk
    cols = [col for col in columns if col in chunk.columns]
    if not cols:
        return chunk

    exploded_rows: list[pd.Series] = []
    for _, row in chunk.iterrows():
        split_values: dict[str, list[str]] = {}
        max_len = 1
        for col in cols:
            parts = [part.strip() for part in str(row[col]).split(";")]
            split_values[col] = parts
            max_len = max(max_len, len(parts))
        for col in cols:
            parts = split_values[col]
            if len(parts) < max_len:
                parts = parts + [parts[-1]] * (max_len - len(parts))
            split_values[col] = parts
        for idx in range(max_len):
            new_row = row.copy()
            for col in cols:
                new_row[col] = split_values[col][idx]
            exploded_rows.append(new_row)
    if not exploded_rows:
        return chunk.iloc[0:0]
    return pd.DataFrame(exploded_rows)


def _filter_activity_types(chunk: pd.DataFrame) -> pd.DataFrame:
    """Keep Ki/Kd/IC50/EC50/Other rows without papyrus-scripts ``keep_type``.

    The upstream ``keep_type`` helper spins up joblib/swifter workers per chunk
    and, when chained through generators, can terminate the stream early. This
    replacement mirrors the simple and semicolon-exploded paths only.

    Args:
        chunk: Papyrus activity chunk.

    Returns:
        Rows whose assay-type flags match one of the supported types.
    """
    type_cols = [col for col in _TYPE_COLUMNS if col in chunk.columns]
    if not type_cols:
        return chunk.iloc[0:0]

    def _matching_types(df: pd.DataFrame) -> pd.DataFrame:
        active = pd.Series(False, index=df.index)
        clean = pd.Series(True, index=df.index)
        for col in type_cols:
            active |= _type_flag_true(df[col])
            clean &= ~df[col].astype(str).str.contains(";", na=False)
        return df[active & clean]

    multi_mask = pd.Series(False, index=chunk.index)
    for col in type_cols:
        multi_mask |= chunk[col].astype(str).str.contains(";", na=False)

    parts: list[pd.DataFrame] = []
    simple = _matching_types(chunk[~multi_mask])
    if not simple.empty:
        parts.append(simple)
    if multi_mask.any():
        exploded = _explode_semicolon_rows(chunk[multi_mask], _EXPLODE_COLUMNS)
        exploded = _filter_exact_relation(exploded)
        exploded = _matching_types(exploded)
        if not exploded.empty:
            parts.append(exploded)
    if not parts:
        return chunk.iloc[0:0]
    return pd.concat(parts, ignore_index=True)


def _expected_raw_rows(
    version: str,
    plusplus: bool,
    papyrus_root: Optional[str],
) -> Optional[int]:
    """Read the expected raw row count from Papyrus ``data_size.json``.

    Args:
        version: Papyrus version string.
        plusplus: Whether the Papyrus++ subset is being built.
        papyrus_root: Optional Papyrus data directory override.

    Returns:
        Expected raw row count, or ``None`` when metadata is unavailable.
    """
    import json
    from pathlib import Path

    from papyrus_scripts.utils.IO import process_data_version

    try:
        if papyrus_root is not None:
            os.environ["PYSTOW_HOME"] = os.path.abspath(papyrus_root)
        import pystow

        resolved = process_data_version(version=version, root_folder=papyrus_root)
        size_path = Path(pystow.module("papyrus", resolved.version_old_fmt).base.as_posix()) / "data_size.json"
        if not size_path.exists():
            return None
        payload = json.loads(size_path.read_text())
        key = "papyrus_++" if plusplus else "papyrus_2D"
        value = payload.get(key)
        return int(value) if value is not None else None
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return None


def _assay_column(row: pd.Series) -> Optional[str]:
    """Pick the BindingDB assay column for a Papyrus activity row.

    Args:
        row: A filtered Papyrus activity row.

    Returns:
        One of ``Ki (nM)``, ``Kd (nM)``, ``IC50 (nM)``, ``EC50 (nM)``,
        ``Other (nM)``, or ``None`` if no recognized type flag is set.
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


def _parse_activity_class(raw: Any) -> Optional[int]:
    """Map a Papyrus ``Activity_class`` cell to binder / non-binder.

    Args:
        raw: Raw cell value (e.g. ``"N"`` or ``"Y"``).

    Returns:
        ``0`` for inactive, ``1`` for active, or ``None`` if unrecognized /
        missing.
    """
    if raw is None or (isinstance(raw, float) and math.isnan(raw)):
        return None
    text = str(raw).strip().lower()
    if not text or text == "nan":
        return None
    if text in _ACTIVITY_CLASS_INACTIVE:
        return 0
    if text in _ACTIVITY_CLASS_ACTIVE:
        return 1
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
    """Convert one quantitative Papyrus chunk into BindingDB-schema rows.

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
        dictionaries ready for the Parquet writer.
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
        year = parse_year(row.get("Year"))
        record["Year"] = year if year is not None else ""
        out.append(record)
        seen_ligands.add(smiles)
        seen_proteins.add(protein["sequence"])
        stats["written"] += 1

    return out, rows_written + len(out)


def _count_existing_rows(path: str) -> tuple[int, int]:
    """Count quantitative and binary rows in an existing prepared Parquet file.

    Quantitative rows have an empty ``Activity Label``; binary rows set it to
    ``0`` or ``1``.

    Args:
        path: Path to a prepared Parquet file written by this script.

    Returns:
        A tuple ``(quant_rows, binary_rows)``.
    """
    quant_rows = 0
    binary_rows = 0
    pf = pq.ParquetFile(path)
    for i in range(pf.num_row_groups):
        table = pf.read_row_group(i, columns=["Activity Label"])
        labels = table.column("Activity Label").to_pylist()
        for label in labels:
            if str(label or "").strip():
                binary_rows += 1
            else:
                quant_rows += 1
    return quant_rows, binary_rows


def convert_binary_chunk(
    chunk: pd.DataFrame,
    proteins: dict[str, dict[str, str]],
    stats: dict[str, int],
    seen_ligands: set[str],
    seen_proteins: set[str],
    limit: Optional[int],
    rows_written: int,
    binary_skip: int = 0,
) -> tuple[list[dict[str, Any]], int, int]:
    """Convert Papyrus ``Activity_class`` rows into BindingDB-schema rows.

    Binary rows have empty ``pchembl_value``. Inactive (``N``) rows get a
    sentinel potency in ``Other (nM)`` and ``Activity Label=0``; active
    (``Y``) rows use a strong-binder sentinel and ``Activity Label=1``.

    Args:
        chunk: Raw Papyrus chunk (binary rows only).
        proteins: Protein metadata keyed by ``target_id``.
        stats: Mutable skip/write counters updated in place.
        seen_ligands: Mutable set of canonical SMILES written so far.
        seen_proteins: Mutable set of sequences written so far.
        limit: Optional cap on total output rows across all chunks.
        rows_written: Number of rows already written before this chunk.
        binary_skip: When resuming, skip this many converted binary rows before
            writing (already present in the output file).

    Returns:
        A tuple ``(rows, rows_written_after, binary_skip_remaining)``.
    """
    out: list[dict[str, Any]] = []
    for _, row in chunk.iterrows():
        label = _parse_activity_class(row.get("Activity_class"))
        if label is None:
            stats["skipped_binary_class"] += 1
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

        activity_nm = (
            config.BINARY_ACTIVE_NM if label == 1 else config.BINARY_INACTIVE_NM
        )
        record: dict[str, Any] = {col: "" for col in _OUTPUT_COLUMNS}
        record["Ligand SMILES"] = smiles
        record["Target Name"] = protein["name"]
        record["Target Source Organism According to Curator or DataSource"] = protein[
            "organism"
        ]
        record["Other (nM)"] = f"{activity_nm:.6g}"
        record["Activity Label"] = str(label)
        record["BindingDB Target Chain Sequence 1"] = protein["sequence"]
        year = parse_year(row.get("Year"))
        record["Year"] = year if year is not None else ""
        out.append(record)

    if binary_skip > 0:
        if binary_skip >= len(out):
            binary_skip -= len(out)
            out = []
        else:
            out = out[binary_skip:]
            binary_skip = 0

    if limit is not None and rows_written + len(out) > limit:
        out = out[: limit - rows_written]

    for record in out:
        seen_ligands.add(record["Ligand SMILES"])
        seen_proteins.add(record["BindingDB Target Chain Sequence 1"])
    stats["written"] += len(out)
    stats["written_binary"] += len(out)

    return out, rows_written + len(out), binary_skip


def _open_papyrus_reader(
    version: str,
    plusplus: bool,
    papyrus_root: Optional[str],
    chunk_size: int,
) -> Any:
    """Open a chunked Papyrus activity reader.

    Args:
        version: Papyrus version string.
        plusplus: If ``True``, read Papyrus++.
        papyrus_root: Optional data directory override.
        chunk_size: Rows per chunk.

    Returns:
        A pandas ``TextFileReader`` / chunk iterator.
    """
    from papyrus_scripts import read_papyrus

    return read_papyrus(
        is3d=False,
        version=version,
        plusplus=plusplus,
        chunksize=chunk_size,
        source_path=papyrus_root,
    )


def iter_filtered_chunks(
    version: str,
    plusplus: bool,
    papyrus_root: Optional[str],
    chunk_size: int,
    min_quality: str,
    verbose: bool,
) -> Iterator[pd.DataFrame]:
    """Yield filtered quantitative Papyrus activity chunks for conversion.

    Filters are applied directly per raw chunk instead of chaining
    ``papyrus_scripts.preprocess.keep_quality`` and ``keep_type`` generators,
    which can terminate early on the full ~60M-row release.

    Args:
        version: Papyrus version string.
        plusplus: If ``True``, read Papyrus++; otherwise the full set.
        papyrus_root: Optional Papyrus data directory override.
        chunk_size: Rows per streaming chunk.
        min_quality: Minimum quality tier for the full subset
            (``high`` / ``medium`` / ``low``). Ignored for Papyrus++.
        verbose: If ``True``, print progress notes.

    Yields:
        Filtered pandas DataFrames of exact quantitative assay rows.
    """
    if verbose:
        label = "Papyrus++" if plusplus else "full Papyrus"
        print(
            f"[papyrus] streaming quantitative {label} version={version!r} "
            f"chunk_size={chunk_size}",
            flush=True,
        )
        if not plusplus:
            print(f"[papyrus] applying min_quality={min_quality!r}", flush=True)
        expected = _expected_raw_rows(version, plusplus, papyrus_root)
        if expected is not None:
            print(f"[papyrus] expected raw rows in source file: {expected:,}", flush=True)

    reader = _open_papyrus_reader(version, plusplus, papyrus_root, chunk_size)
    raw_rows = 0
    raw_idx = 0
    for raw_idx, chunk in enumerate(reader, start=1):
        if not isinstance(chunk, pd.DataFrame) or chunk.empty:
            continue
        raw_rows += len(chunk)
        if not plusplus:
            chunk = _filter_quality(chunk, min_quality)
        # Skip class-labeled rows in the quantitative pass (handled separately).
        if "Activity_class" in chunk.columns:
            chunk = chunk[chunk["Activity_class"].isna()]
        chunk = _filter_activity_types(chunk)
        chunk = _filter_exact_relation(chunk)
        if chunk.empty:
            if verbose and raw_idx % 10 == 0:
                print(
                    f"[papyrus] quant: read {raw_idx} raw chunks ({raw_rows:,} rows)...",
                    flush=True,
                )
            continue
        if verbose and raw_idx % 10 == 0:
            print(
                f"[papyrus] quant: read {raw_idx} raw chunks ({raw_rows:,} rows), "
                f"latest filtered chunk={len(chunk):,}",
                flush=True,
            )
        yield chunk

    if verbose:
        print(
            f"[papyrus] finished quantitative pass ({raw_idx} raw chunks, "
            f"{raw_rows:,} rows)",
            flush=True,
        )


def iter_binary_chunks(
    version: str,
    papyrus_root: Optional[str],
    chunk_size: int,
    min_quality: str,
    verbose: bool,
) -> Iterator[pd.DataFrame]:
    """Yield Papyrus chunks that carry ``Activity_class`` binary labels.

    Args:
        version: Papyrus version string.
        papyrus_root: Optional Papyrus data directory override.
        chunk_size: Rows per streaming chunk.
        min_quality: Minimum quality tier (``high`` / ``medium`` / ``low``).
        verbose: If ``True``, print progress notes.

    Yields:
        DataFrames containing only rows with a non-null ``Activity_class``.
    """
    if verbose:
        print(
            f"[papyrus] streaming Activity_class rows version={version!r} "
            f"chunk_size={chunk_size}",
            flush=True,
        )

    reader = _open_papyrus_reader(version, plusplus=False, papyrus_root=papyrus_root, chunk_size=chunk_size)
    raw_rows = 0
    raw_idx = 0
    for raw_idx, chunk in enumerate(reader, start=1):
        if not isinstance(chunk, pd.DataFrame) or chunk.empty:
            continue
        raw_rows += len(chunk)
        chunk = _filter_quality(chunk, min_quality)
        if "Activity_class" not in chunk.columns:
            continue
        chunk = chunk[chunk["Activity_class"].notna()]
        if chunk.empty:
            if verbose and raw_idx % 20 == 0:
                print(
                    f"[papyrus] binary: read {raw_idx} raw chunks ({raw_rows:,} rows)...",
                    flush=True,
                )
            continue
        if verbose and raw_idx % 20 == 0:
            print(
                f"[papyrus] binary: read {raw_idx} raw chunks ({raw_rows:,} rows), "
                f"latest class chunk={len(chunk):,}",
                flush=True,
            )
        yield chunk

    if verbose:
        print(
            f"[papyrus] finished binary pass ({raw_idx} raw chunks, {raw_rows:,} rows)",
            flush=True,
        )


def run_build(args: argparse.Namespace) -> str:
    """Download (optional), convert, and write the prepared Papyrus Parquet file.

    Writes quantitative assay rows first. When ``--include-binary`` is set,
    streams a second pass over ``Activity_class`` rows into the same file.
    ``--limit`` caps total written rows across both phases.

    Args:
        args: Parsed CLI arguments.

    Returns:
        Path of the written prepared Parquet file.
    """
    _require_papyrus()
    verbose = not args.quiet
    plusplus = args.subset == "plusplus"
    include_binary = bool(getattr(args, "include_binary", False))
    resume = bool(getattr(args, "resume", False))
    if include_binary and plusplus:
        raise SystemExit("--include-binary requires --subset full (Papyrus++ has no Activity_class).")
    if resume and not include_binary:
        raise SystemExit("--resume requires --include-binary.")
    output = args.output or _default_output(args.subset, include_binary=include_binary)
    if not output.lower().endswith((".parquet", ".pq")):
        raise SystemExit(f"Papyrus output must be a .parquet path; got {output!r}")
    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    if resume and not os.path.isfile(output):
        raise SystemExit(f"--resume requested but output file does not exist: {output}")

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
        "written_binary": 0,
        "skipped_relation": 0,
        "skipped_activity": 0,
        "skipped_assay": 0,
        "skipped_sequence": 0,
        "skipped_smiles": 0,
        "skipped_binary_class": 0,
    }
    seen_ligands: set[str] = set()
    seen_proteins: set[str] = set()
    rows_written = 0
    binary_skip = 0
    expected_raw = _expected_raw_rows(resolved_version, plusplus, args.papyrus_root)

    if resume:
        if verbose:
            print(f"[papyrus] resuming from existing output: {output}", flush=True)
        quant_existing, binary_existing = _count_existing_rows(output)
        if quant_existing == 0:
            raise SystemExit(
                "--resume requires an existing Parquet file with quantitative rows "
                "(Activity Label empty)."
            )
        rows_written = quant_existing + binary_existing
        binary_skip = binary_existing
        stats["written"] = quant_existing
        stats["written_binary"] = binary_existing
        if verbose:
            print(
                f"[papyrus] found {quant_existing:,} quantitative and "
                f"{binary_existing:,} binary rows; skipping quant pass",
                flush=True,
            )
        if args.limit is not None and rows_written >= args.limit:
            if verbose:
                print(
                    f"[papyrus] existing row count already meets --limit {args.limit}; "
                    "nothing to do",
                    flush=True,
                )
            return output

    # Every build publishes atomically; Parquet is invalid until its footer is
    # written, so an interrupted writer must never target the final path.
    write_path = f"{output}.__building__.parquet"
    if os.path.exists(write_path):
        os.remove(write_path)

    writer = pq.ParquetWriter(write_path, _PARQUET_SCHEMA, compression="zstd")
    completed = False
    try:
        if resume:
            if verbose:
                print("[papyrus] copying existing Parquet rows into rebuild file", flush=True)
            _copy_parquet_to_writer(output, writer)

        if not resume:
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
                    writer.write_table(_records_to_table(rows))
                if args.limit is not None and rows_written >= args.limit:
                    if verbose:
                        print(
                            f"[papyrus] reached --limit {args.limit} during quantitative pass",
                            flush=True,
                        )
                    break

        if include_binary and (args.limit is None or rows_written < args.limit):
            if verbose:
                if resume:
                    print(
                        f"[papyrus] resuming Activity_class pass "
                        f"(skip {binary_skip:,} binary rows)",
                        flush=True,
                    )
                else:
                    print(
                        f"[papyrus] quantitative rows written so far: {rows_written:,}; "
                        "starting Activity_class pass",
                        flush=True,
                    )
            for chunk in iter_binary_chunks(
                version=resolved_version,
                papyrus_root=args.papyrus_root,
                chunk_size=args.chunk_size,
                min_quality=args.min_quality,
                verbose=verbose,
            ):
                rows, rows_written, binary_skip = convert_binary_chunk(
                    chunk,
                    proteins,
                    stats,
                    seen_ligands,
                    seen_proteins,
                    args.limit,
                    rows_written,
                    binary_skip=binary_skip,
                )
                if rows:
                    writer.write_table(_records_to_table(rows))
                if args.limit is not None and rows_written >= args.limit:
                    if verbose:
                        print(
                            f"[papyrus] reached --limit {args.limit} during binary pass",
                            flush=True,
                        )
                    break
        completed = True
    finally:
        writer.close()
        if not completed and os.path.exists(write_path):
            os.remove(write_path)

    try:
        _validate_parquet_output(write_path, rows_written)
        os.replace(write_path, output)
    except Exception:
        if os.path.exists(write_path):
            os.remove(write_path)
        raise

    if verbose:
        print(
            f"\n[papyrus] subset={args.subset} include_binary={include_binary} "
            f"wrote {stats['written']} rows "
            f"(binary={stats['written_binary']}) -> {output}\n"
            f"[papyrus] unique ligands={len(seen_ligands)}  "
            f"unique proteins={len(seen_proteins)}\n"
            f"[papyrus] skipped: relation={stats['skipped_relation']}  "
            f"activity={stats['skipped_activity']}  assay={stats['skipped_assay']}  "
            f"sequence={stats['skipped_sequence']}  smiles={stats['skipped_smiles']}  "
            f"binary_class={stats['skipped_binary_class']}",
            flush=True,
        )
        if (
            not plusplus
            and not include_binary
            and args.limit is None
            and expected_raw is not None
            and stats["written"] < expected_raw * 0.01
        ):
            print(
                "[papyrus] warning: output row count is far below the raw Papyrus 2D "
                f"size ({expected_raw:,}). This usually means the source file was "
                "not fully streamed; rebuild with --no-download after verifying the "
                "download under ~/.data/papyrus.",
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
            "Build a BindingDB-compatible prepared Parquet file from Papyrus++ "
            "or the full without_stereochemistry Papyrus release. The full subset "
            "can produce a multi-GB file and take hours; prefer --subset plusplus "
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
            "Output Parquet path (default: Papyrus_pp_prepared.parquet, "
            "Papyrus_full_prepared.parquet, or Papyrus_full_binary_prepared.parquet "
            "when --include-binary is set)."
        ),
    )
    p.add_argument(
        "--include-binary",
        action="store_true",
        help=(
            "For --subset full: after quantitative rows, also append "
            "Activity_class binary rows (Other assay + Activity Label). "
            "Reuse --limit to cap total size (write quant first)."
        ),
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help=(
            "With --include-binary: rebuild from an existing partial Parquet file, "
            "skipping the quantitative pass and continuing the Activity_class "
            "pass from the row count already on disk."
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
        help=(
            "Cap on total output rows (smoke tests, or binary subsampling: "
            "quant is written first, then Activity_class rows fill the remainder)."
        ),
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
