"""Build a BindingDB-compatible prepared Parquet file from GPCRdb / ChEMBL CSV.

Reads ``data/train/gpcrdb_data.csv`` (ChEMBL-style activity export), resolves
ligand SMILES via the ChEMBL REST API and protein sequences via UniProt, and
writes a Parquet file that :func:`src.data_prep.iter_prepared_rows` can
consume::

    python build_gpcrdb.py
    python train.py --data data/train/GPCRdb_prepared.parquet --rebuild-features

    python build_gpcrdb.py --limit 10000   # smoke test

Entity lookups are cached under ``cache/gpcrdb/`` so reruns skip network work.
``Year`` is left empty (the source CSV has no publication year); prefer
cold-protein or double-cold splits for this dataset, especially when
concatenating with Papyrus for joint training.
"""

from __future__ import annotations

import argparse
import json
import os
import ssl
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Iterable, Iterator, Mapping, Optional

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

import config
from src.data_prep import canonicalize_smiles
from src.prepared_schema import OUTPUT_COLUMNS, PARQUET_SCHEMA

# GPCRdb / ChEMBL column names on the raw CSV.
COL_ENTRY_NAME = "Entry name"
COL_ACCESSION = "Accession"
COL_MOL_ID = "parent_molecule_chembl_id"
COL_RELATION = "standard_relation"
COL_TYPE = "standard_type"
COL_UNITS = "standard_units"
COL_VALUE = "standard_value"

# Map ChEMBL standard_type -> BindingDB assay column.
_ASSAY_COLUMNS: dict[str, str] = {
    "Ki": "Ki (nM)",
    "IC50": "IC50 (nM)",
    "EC50": "EC50 (nM)",
    "Kd": "Kd (nM)",
    "Potency": "Other (nM)",
    "AC50": "Other (nM)",
    "ED50": "Other (nM)",
    "XC50": "Other (nM)",
}

_CHEMBL_MOLECULE_URL = "https://www.ebi.ac.uk/chembl/api/data/molecule.json"
_UNIPROT_ACCESSIONS_URL = "https://rest.uniprot.org/uniprotkb/accessions"

_HTTP_RETRIES = 5
_HTTP_BACKOFF_S = 1.5


def _ssl_context() -> ssl.SSLContext:
    """Build an SSL context, preferring certifi CA roots when installed.

    Returns:
        An :class:`ssl.SSLContext` suitable for HTTPS API calls.
    """
    try:
        import certifi

        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        return ssl.create_default_context()


def _http_json(
    url: str,
    *,
    data: Optional[bytes] = None,
    headers: Optional[Mapping[str, str]] = None,
    timeout: float = 120.0,
) -> Any:
    """GET or POST JSON from ``url`` with retries.

    Args:
        url: Request URL.
        data: Optional POST body.
        headers: Optional HTTP headers.
        timeout: Socket timeout in seconds.

    Returns:
        Parsed JSON payload.

    Raises:
        urllib.error.URLError: When all retries fail.
        json.JSONDecodeError: When the response is not valid JSON.
    """
    hdrs = {"Accept": "application/json", "User-Agent": "GPCR-Predictor-2.0/build_gpcrdb"}
    if headers:
        hdrs.update(headers)
    ctx = _ssl_context()
    last_error: Optional[BaseException] = None
    for attempt in range(_HTTP_RETRIES):
        req = urllib.request.Request(url, data=data, headers=hdrs, method="POST" if data else "GET")
        try:
            with urllib.request.urlopen(req, context=ctx, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            last_error = exc
            # 4xx (except 429) are not worth retrying.
            if exc.code == 429 or exc.code >= 500:
                time.sleep(_HTTP_BACKOFF_S * (2**attempt))
                continue
            raise
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            last_error = exc
            time.sleep(_HTTP_BACKOFF_S * (2**attempt))
    assert last_error is not None
    raise last_error


def _load_json_cache(path: str) -> dict[str, Any]:
    """Load a JSON object cache from disk.

    Args:
        path: Cache file path.

    Returns:
        Parsed mapping, or an empty dict when the file is absent.
    """
    if not os.path.isfile(path):
        return {}
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Cache {path!r} must contain a JSON object.")
    return payload


def _save_json_cache(path: str, payload: Mapping[str, Any]) -> None:
    """Atomically write a JSON object cache.

    Args:
        path: Destination path.
        payload: Mapping to serialize.

    Returns:
        None.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    temporary = f"{path}.tmp"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=0, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def _batched(items: list[str], size: int) -> Iterator[list[str]]:
    """Yield successive batches from ``items``.

    Args:
        items: Sequence to split.
        size: Maximum batch length (must be positive).

    Yields:
        Contiguous sublists of ``items``.
    """
    if size <= 0:
        raise ValueError("batch size must be positive")
    for start in range(0, len(items), size):
        yield items[start : start + size]


def resolve_chembl_smiles(
    chembl_ids: Iterable[str],
    cache_path: str,
    *,
    batch_size: int = 100,
    verbose: bool = True,
) -> dict[str, Optional[str]]:
    """Resolve ChEMBL molecule IDs to canonical SMILES with a disk cache.

    Args:
        chembl_ids: ChEMBL IDs to resolve (e.g. ``CHEMBL25``).
        cache_path: JSON cache path (``id -> smiles | null``).
        batch_size: IDs per ChEMBL ``__in`` request.
        verbose: Print progress.

    Returns:
        Mapping of requested IDs to raw ChEMBL canonical SMILES, or ``None``
        when ChEMBL has no structure for that ID.
    """
    cache: dict[str, Any] = _load_json_cache(cache_path)
    needed = sorted({cid.strip() for cid in chembl_ids if cid and cid.strip() and cid not in cache})
    if verbose:
        print(
            f"[gpcrdb] ChEMBL SMILES: {len(cache):,} cached, {len(needed):,} to fetch",
            flush=True,
        )
    for batch_index, batch in enumerate(_batched(needed, batch_size), start=1):
        query = urllib.parse.urlencode(
            {
                "molecule_chembl_id__in": ",".join(batch),
                "limit": str(len(batch)),
            }
        )
        payload = _http_json(f"{_CHEMBL_MOLECULE_URL}?{query}")
        found: set[str] = set()
        for molecule in payload.get("molecules") or []:
            mol_id = str(molecule.get("molecule_chembl_id") or "").strip()
            if not mol_id:
                continue
            structures = molecule.get("molecule_structures") or {}
            smiles = structures.get("canonical_smiles")
            cache[mol_id] = str(smiles).strip() if smiles else None
            found.add(mol_id)
        for mol_id in batch:
            if mol_id not in found:
                cache[mol_id] = None
        if batch_index % 25 == 0 or batch_index == (len(needed) + batch_size - 1) // batch_size:
            _save_json_cache(cache_path, cache)
            if verbose:
                print(
                    f"[gpcrdb] ChEMBL progress {min(batch_index * batch_size, len(needed)):,}"
                    f"/{len(needed):,}",
                    flush=True,
                )
    if needed:
        _save_json_cache(cache_path, cache)
    return {cid: cache.get(cid) for cid in {c.strip() for c in chembl_ids if c and c.strip()}}


def resolve_uniprot_sequences(
    accessions: Iterable[str],
    cache_path: str,
    *,
    batch_size: int = 100,
    verbose: bool = True,
) -> dict[str, dict[str, str]]:
    """Resolve UniProt accessions to sequences (and organism) with a disk cache.

    Args:
        accessions: UniProt accession IDs.
        cache_path: JSON cache path
            (``accession -> {sequence, organism}``).
        batch_size: Accessions per UniProt request.
        verbose: Print progress.

    Returns:
        Mapping of requested accessions to ``{"sequence", "organism"}`` for
        IDs that resolved; missing accessions are omitted.
    """
    cache: dict[str, Any] = _load_json_cache(cache_path)
    needed = sorted(
        {
            acc.strip().upper()
            for acc in accessions
            if acc and str(acc).strip() and str(acc).strip().upper() not in cache
        }
    )
    if verbose:
        print(
            f"[gpcrdb] UniProt sequences: {len(cache):,} cached, {len(needed):,} to fetch",
            flush=True,
        )
    for batch_index, batch in enumerate(_batched(needed, batch_size), start=1):
        query = urllib.parse.urlencode(
            {
                "accessions": ",".join(batch),
                "format": "json",
                "fields": "accession,sequence,organism_name",
            }
        )
        payload = _http_json(f"{_UNIPROT_ACCESSIONS_URL}?{query}")
        found: set[str] = set()
        for entry in payload.get("results") or []:
            acc = str(entry.get("primaryAccession") or "").strip().upper()
            if not acc:
                continue
            sequence = str((entry.get("sequence") or {}).get("value") or "").strip()
            organism = str((entry.get("organism") or {}).get("scientificName") or "").strip()
            if sequence:
                cache[acc] = {"sequence": sequence, "organism": organism}
                found.add(acc)
        for acc in batch:
            if acc not in found:
                # Record a negative cache entry so we do not refetch forever.
                cache[acc] = None
        _save_json_cache(cache_path, cache)
        if verbose:
            print(
                f"[gpcrdb] UniProt progress {min(batch_index * batch_size, len(needed)):,}"
                f"/{len(needed):,}",
                flush=True,
            )
    out: dict[str, dict[str, str]] = {}
    for acc in accessions:
        key = str(acc).strip().upper()
        value = cache.get(key)
        if isinstance(value, dict) and value.get("sequence"):
            out[key] = {
                "sequence": str(value["sequence"]),
                "organism": str(value.get("organism") or ""),
            }
    return out


def _records_to_table(rows: list[dict[str, Any]]) -> pa.Table:
    """Convert prepared row dictionaries to a typed Arrow table.

    Args:
        rows: Dictionaries with keys from ``OUTPUT_COLUMNS``.

    Returns:
        A ``pyarrow.Table`` matching ``PARQUET_SCHEMA``.
    """
    columns: dict[str, list[Any]] = {name: [] for name in OUTPUT_COLUMNS}
    for record in rows:
        for name in OUTPUT_COLUMNS:
            value = record.get(name)
            if name == "Year":
                columns[name].append(None if value in ("", None) else int(value))
            else:
                columns[name].append("" if value is None else str(value))
    arrays = []
    for field in PARQUET_SCHEMA:
        if field.name == "Year":
            arrays.append(pa.array(columns[field.name], type=pa.int32()))
        else:
            arrays.append(pa.array(columns[field.name], type=pa.string()))
    return pa.Table.from_arrays(arrays, schema=PARQUET_SCHEMA)


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
    if parquet.schema_arrow != PARQUET_SCHEMA:
        raise ValueError(
            f"Parquet validation failed for {path}: schema does not match "
            "the prepared-data schema."
        )


def _row_passes_filters(row: Mapping[str, Any]) -> bool:
    """Return whether a raw GPCRdb row should be considered for output.

    Args:
        row: Raw CSV row mapping.

    Returns:
        ``True`` when relation, units, and assay type are acceptable.
    """
    relation = str(row.get(COL_RELATION) or "").strip()
    units = str(row.get(COL_UNITS) or "").strip()
    assay = str(row.get(COL_TYPE) or "").strip()
    if relation != "=" or units != "nM":
        return False
    if assay not in _ASSAY_COLUMNS:
        return False
    try:
        value = float(row.get(COL_VALUE))
    except (TypeError, ValueError):
        return False
    if not (value > 0.0):
        return False
    mol_id = str(row.get(COL_MOL_ID) or "").strip()
    accession = str(row.get(COL_ACCESSION) or "").strip()
    return bool(mol_id and accession)


def collect_entity_ids(
    input_path: str,
    *,
    limit: Optional[int],
    chunk_size: int,
    verbose: bool,
) -> tuple[set[str], set[str], dict[str, str], int]:
    """Scan the raw CSV for ChEMBL IDs and UniProt accessions to resolve.

    Args:
        input_path: Path to ``gpcrdb_data.csv``.
        limit: Optional cap on qualifying raw rows (smoke tests).
        chunk_size: Pandas chunk size.
        verbose: Print progress.

    Returns:
        A tuple ``(chembl_ids, accessions, entry_names, qualifying_rows)`` where
        ``entry_names`` maps accession → first-seen ``Entry name``.
    """
    chembl_ids: set[str] = set()
    accessions: set[str] = set()
    entry_names: dict[str, str] = {}
    qualifying = 0
    usecols = [COL_ENTRY_NAME, COL_ACCESSION, COL_MOL_ID, COL_RELATION, COL_TYPE, COL_UNITS, COL_VALUE]
    for chunk in pd.read_csv(input_path, usecols=usecols, chunksize=chunk_size):
        for row in chunk.to_dict(orient="records"):
            if limit is not None and qualifying >= limit:
                if verbose:
                    print(
                        f"[gpcrdb] collected entities from {qualifying:,} qualifying rows "
                        f"({len(chembl_ids):,} ligands, {len(accessions):,} proteins)",
                        flush=True,
                    )
                return chembl_ids, accessions, entry_names, qualifying
            if not _row_passes_filters(row):
                continue
            mol_id = str(row[COL_MOL_ID]).strip()
            accession = str(row[COL_ACCESSION]).strip().upper()
            chembl_ids.add(mol_id)
            accessions.add(accession)
            name = str(row.get(COL_ENTRY_NAME) or "").strip()
            if name and accession not in entry_names:
                entry_names[accession] = name
            qualifying += 1
    if verbose:
        print(
            f"[gpcrdb] collected entities from {qualifying:,} qualifying rows "
            f"({len(chembl_ids):,} ligands, {len(accessions):,} proteins)",
            flush=True,
        )
    return chembl_ids, accessions, entry_names, qualifying


def run_build(args: argparse.Namespace) -> str:
    """Execute the GPCRdb → prepared Parquet conversion.

    Args:
        args: Parsed CLI arguments.

    Returns:
        Path to the published prepared Parquet file.
    """
    input_path = os.path.abspath(args.input)
    output_path = os.path.abspath(args.output)
    cache_dir = os.path.abspath(args.cache_dir)
    verbose = not args.quiet

    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"GPCRdb input CSV not found: {input_path}")

    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    smiles_cache = os.path.join(cache_dir, "chembl_smiles.json")
    sequence_cache = os.path.join(cache_dir, "uniprot_sequences.json")

    chembl_ids, accessions, entry_names, _qualifying = collect_entity_ids(
        input_path,
        limit=args.limit,
        chunk_size=args.chunk_size,
        verbose=verbose,
    )
    smiles_by_chembl = resolve_chembl_smiles(
        chembl_ids,
        smiles_cache,
        batch_size=args.chembl_batch_size,
        verbose=verbose,
    )
    proteins_by_accession = resolve_uniprot_sequences(
        accessions,
        sequence_cache,
        batch_size=args.uniprot_batch_size,
        verbose=verbose,
    )
    if verbose:
        resolved_smiles = sum(1 for cid in chembl_ids if smiles_by_chembl.get(cid))
        print(
            f"[gpcrdb] resolved SMILES for {resolved_smiles:,}/{len(chembl_ids):,} ligands; "
            f"sequences for {len(proteins_by_accession):,}/{len(accessions):,} proteins",
            flush=True,
        )

    write_path = f"{output_path}.tmp"
    if os.path.exists(write_path):
        os.remove(write_path)

    writer = pq.ParquetWriter(write_path, PARQUET_SCHEMA, compression="zstd")
    stats = {
        "written": 0,
        "skipped_smiles": 0,
        "skipped_sequence": 0,
    }
    buffer: list[dict[str, Any]] = []
    buffer_size = max(1, int(args.write_buffer))

    try:
        # Re-scan to count skips accurately while writing.
        usecols = [
            COL_ENTRY_NAME,
            COL_ACCESSION,
            COL_MOL_ID,
            COL_RELATION,
            COL_TYPE,
            COL_UNITS,
            COL_VALUE,
        ]
        emitted = 0
        for chunk in pd.read_csv(input_path, usecols=usecols, chunksize=args.chunk_size):
            for row in chunk.to_dict(orient="records"):
                if args.limit is not None and emitted >= args.limit:
                    break
                if not _row_passes_filters(row):
                    continue
                # Count this qualifying row against --limit even if later skipped.
                emitted += 1
                mol_id = str(row[COL_MOL_ID]).strip()
                accession = str(row[COL_ACCESSION]).strip().upper()
                raw_smiles = smiles_by_chembl.get(mol_id)
                if not raw_smiles:
                    stats["skipped_smiles"] += 1
                    continue
                smiles = canonicalize_smiles(raw_smiles)
                if smiles is None:
                    stats["skipped_smiles"] += 1
                    continue
                protein = proteins_by_accession.get(accession)
                if protein is None or not protein.get("sequence"):
                    stats["skipped_sequence"] += 1
                    continue
                assay_col = _ASSAY_COLUMNS[str(row[COL_TYPE]).strip()]
                activity_nm = float(row[COL_VALUE])
                target_name = entry_names.get(accession) or str(
                    row.get(COL_ENTRY_NAME) or ""
                ).strip()
                record: dict[str, Any] = {col: "" for col in OUTPUT_COLUMNS}
                record["Ligand SMILES"] = smiles
                record["Target Name"] = target_name
                record["Target Source Organism According to Curator or DataSource"] = (
                    protein.get("organism", "")
                )
                record[assay_col] = f"{activity_nm:.6g}"
                record["BindingDB Target Chain Sequence 1"] = protein["sequence"]
                record["Year"] = ""
                buffer.append(record)
                stats["written"] += 1
                if len(buffer) >= buffer_size:
                    writer.write_table(_records_to_table(buffer))
                    buffer.clear()
                    if verbose and stats["written"] % (buffer_size * 5) < buffer_size:
                        print(f"[gpcrdb] wrote {stats['written']:,} rows", flush=True)
            if args.limit is not None and emitted >= args.limit:
                break
        if buffer:
            writer.write_table(_records_to_table(buffer))
            buffer.clear()
    finally:
        writer.close()

    if stats["written"] == 0:
        if os.path.exists(write_path):
            os.remove(write_path)
        raise RuntimeError(
            "No rows written. Check that ChEMBL/UniProt lookups succeeded and "
            "that filters (relation='=', units=nM, assay types) are not too strict."
        )

    _validate_parquet_output(write_path, stats["written"])
    os.replace(write_path, output_path)
    if verbose:
        print(
            f"[gpcrdb] done: wrote {stats['written']:,} rows -> {output_path}\n"
            f"  skipped_smiles={stats['skipped_smiles']:,}  "
            f"skipped_sequence={stats['skipped_sequence']:,}",
            flush=True,
        )
    return output_path


def build_arg_parser() -> argparse.ArgumentParser:
    """Construct the command-line argument parser.

    Returns:
        The configured :class:`argparse.ArgumentParser`.
    """
    p = argparse.ArgumentParser(
        description=(
            "Build a BindingDB-compatible prepared Parquet file from a GPCRdb "
            "/ ChEMBL activity CSV. Resolves SMILES via ChEMBL and sequences via "
            "UniProt (cached under cache/gpcrdb/)."
        )
    )
    p.add_argument(
        "--input",
        default=config.GPCRDB_RAW_CSV,
        help=f"Raw GPCRdb CSV path (default: {config.GPCRDB_RAW_CSV}).",
    )
    p.add_argument(
        "--output",
        default=config.GPCRDB_TRAIN_CSV,
        help=f"Output Parquet path (default: {config.GPCRDB_TRAIN_CSV}).",
    )
    p.add_argument(
        "--cache-dir",
        default=os.path.join(config.CACHE_DIR, "gpcrdb"),
        help="Directory for ChEMBL/UniProt JSON caches (default: cache/gpcrdb).",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Cap on qualifying raw rows after filters (smoke tests).",
    )
    p.add_argument(
        "--chunk-size",
        type=int,
        default=100_000,
        help="Pandas read_csv chunk size (default: 100000).",
    )
    p.add_argument(
        "--chembl-batch-size",
        type=int,
        default=100,
        help="ChEMBL molecule IDs per API request (default: 100).",
    )
    p.add_argument(
        "--uniprot-batch-size",
        type=int,
        default=100,
        help="UniProt accessions per API request (default: 100).",
    )
    p.add_argument(
        "--write-buffer",
        type=int,
        default=50_000,
        help="Prepared rows buffered before each Parquet write (default: 50000).",
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
