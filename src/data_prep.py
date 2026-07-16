"""Streaming cleaning and label preparation for BindingDB CSV / Papyrus Parquet.

This module reads prepared training tables row by row (never loading the whole
file into memory), cleans ligand SMILES (salt removal + canonicalization),
converts activity measurements to pActivity, explodes rows that carry several
assay measurements, drops censored/missing values, and yields tidy
``PreparedRow`` records ready for embedding and featurization.

BindingDB inputs remain CSV; Papyrus prepared builds are Parquet (``.parquet``).
"""

from __future__ import annotations

import csv
import math
import sys
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Iterable, Iterator, Mapping, Optional

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem.MolStandardize import rdMolStandardize

import config

RDLogger.DisableLog("rdApp.*")

# Raw CSV / Parquet column names.
COL_SMILES = "Ligand SMILES"
COL_PH = "pH"
COL_TEMP = "Temp (C)"
COL_SEQ = "BindingDB Target Chain Sequence 1"
COL_YEAR = "Year"

# Mapping from assay type to its raw activity column (values in nM).
ASSAY_COLUMNS: dict[str, str] = {
    "IC50": "IC50 (nM)",
    "EC50": "EC50 (nM)",
    "Ki": "Ki (nM)",
    "Kd": "Kd (nM)",
    "Other": "Other (nM)",
}

COL_ACTIVITY_LABEL = "Activity Label"

# Allow very long single-cell sequences without hitting the csv field limit.
csv.field_size_limit(min(sys.maxsize, 2**31 - 1))


@dataclass(frozen=True, slots=True)
class PreparedRow:
    """A single cleaned training example.

    Attributes:
        smiles: Canonical, desalted ligand SMILES.
        sequence: Protein chain amino-acid sequence.
        assay_type: One of ``config.ASSAY_TYPES``.
        ph: Assay pH (imputed to the default when missing).
        temp: Assay temperature in Celsius (imputed to the default when missing).
        pactivity: Continuous target ``-log10(activity_nM * 1e-9)``.
        activity_label: Optional explicit binder class (``0`` / ``1``) from
            Papyrus ``Activity_class`` rows. When set, featurization prefers
            this over binarizing ``pactivity``.
        year: Optional publication year from Papyrus (used for temporal splits).
    """

    smiles: str
    sequence: str
    assay_type: str
    ph: float
    temp: float
    pactivity: float
    activity_label: Optional[float] = None
    year: Optional[int] = None


@lru_cache(maxsize=200_000)
def canonicalize_smiles(smiles: str) -> Optional[str]:
    """Desalt and canonicalize a SMILES string.

    The largest organic fragment is kept (removing salts and counter-ions) and
    the result is returned as a canonical SMILES. Results are memoized because
    the same ligand recurs many times across the dataset.

    Args:
        smiles: Raw SMILES string.

    Returns:
        The canonical SMILES of the largest fragment, or ``None`` if the input
        cannot be parsed.
    """
    smiles = smiles.strip()
    if not smiles:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        mol = rdMolStandardize.LargestFragmentChooser().choose(mol)
    except Exception:
        return None
    if mol is None or mol.GetNumAtoms() == 0:
        return None
    return Chem.MolToSmiles(mol)


def parse_temperature(raw: str) -> Optional[float]:
    """Parse a raw temperature cell such as ``"25.00 C"``.

    Args:
        raw: Raw temperature string; may include a trailing ``C`` unit or be
            empty.

    Returns:
        The temperature in Celsius if it parses and falls inside the plausible
        physical range (``config.TEMP_MIN_C`` .. ``config.TEMP_MAX_C``),
        otherwise ``None``.
    """
    text = raw.strip().rstrip("Cc").strip()
    if not text:
        return None
    try:
        value = float(text)
    except ValueError:
        return None
    if not math.isfinite(value) or value < config.TEMP_MIN_C or value > config.TEMP_MAX_C:
        return None
    return value


def parse_ph(raw: str) -> Optional[float]:
    """Parse a raw pH cell.

    Args:
        raw: Raw pH string; may be empty.

    Returns:
        The pH as a float when parseable and within ``0 <= pH <= 14``, else
        ``None``.
    """
    text = raw.strip()
    if not text:
        return None
    try:
        value = float(text)
    except ValueError:
        return None
    if not math.isfinite(value) or value < 0.0 or value > 14.0:
        return None
    return value


def _parse_activity_nm(raw: str) -> Optional[float]:
    """Parse a raw activity cell in nM, rejecting censored values.

    Args:
        raw: Raw activity string, e.g. ``"0.24"``. Values carrying a relational
            operator (``>``/``<``) are treated as censored and rejected.

    Returns:
        The positive activity in nM, or ``None`` if empty, censored, or invalid.
    """
    text = raw.strip()
    if not text:
        return None
    if "<" in text or ">" in text:
        return None
    try:
        value = float(text)
    except ValueError:
        return None
    if not math.isfinite(value) or value <= 0.0:
        return None
    return value


def activity_to_pactivity(activity_nm: float) -> float:
    """Convert an activity in nanomolar to pActivity.

    Args:
        activity_nm: Activity concentration in nM (must be positive).

    Returns:
        ``-log10(activity_nm * 1e-9)`` (i.e. pActivity on the molar scale).
    """
    return -math.log10(activity_nm * 1e-9)


def binarize_pactivity(
    y: np.ndarray,
    threshold_nm: float = config.ACTIVITY_THRESHOLD_NM,
) -> np.ndarray:
    """Convert continuous pActivity labels to binder / non-binder classes.

    A row is labeled active (``1``) when its activity is at most
    ``threshold_nm`` nM, i.e. when ``pActivity >= -log10(threshold_nm * 1e-9)``.

    Args:
        y: Continuous pActivity values.
        threshold_nm: Activity cutoff in nM (default ``config.ACTIVITY_THRESHOLD_NM``).

    Returns:
        A ``float32`` array of ``0.0`` / ``1.0`` labels with the same shape as ``y``.
    """
    thr = activity_to_pactivity(threshold_nm)
    return (np.asarray(y) >= thr).astype(np.float32)


def _parse_activity_label(raw: str) -> Optional[float]:
    """Parse an optional explicit binder label cell.

    Args:
        raw: Cell value such as ``"0"``, ``"1"``, ``"N"``, or ``"Y"``.

    Returns:
        ``0.0`` / ``1.0`` when recognized, else ``None``.
    """
    text = raw.strip().lower()
    if not text:
        return None
    if text in {"0", "0.0", "n", "inactive", "false"}:
        return 0.0
    if text in {"1", "1.0", "y", "active", "true"}:
        return 1.0
    try:
        value = float(text)
    except ValueError:
        return None
    if value in (0.0, 1.0):
        return value
    return None


def _parse_year(raw: object) -> Optional[int]:
    """Parse an optional publication-year cell.

    Args:
        raw: Cell value from CSV or Parquet.

    Returns:
        Integer year in ``[1900, 2100]``, or ``None`` when missing/invalid.
    """
    if raw is None:
        return None
    if isinstance(raw, float) and math.isnan(raw):
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


def _cell(row: Mapping[str, Any], key: str) -> str:
    """Return a string cell value from a mapping, treating nulls as empty.

    Args:
        row: Row mapping.
        key: Column name.

    Returns:
        String value (possibly empty).
    """
    value = row.get(key)
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return str(value)


def _yield_prepared_from_raw(row: Mapping[str, Any]) -> Iterator[PreparedRow]:
    """Convert one raw prepared-table row into zero or more ``PreparedRow``s.

    Args:
        row: Mapping of column name to cell value.

    Yields:
        Cleaned assay-exploded ``PreparedRow`` instances.
    """
    sequence = _cell(row, COL_SEQ).strip()
    if not sequence:
        return

    smiles = canonicalize_smiles(_cell(row, COL_SMILES))
    if smiles is None:
        return

    ph = parse_ph(_cell(row, COL_PH))
    if ph is None:
        ph = config.DEFAULT_PH
    temp = parse_temperature(_cell(row, COL_TEMP))
    if temp is None:
        temp = config.DEFAULT_TEMP_C

    activity_label = _parse_activity_label(_cell(row, COL_ACTIVITY_LABEL))
    year = _parse_year(row.get(COL_YEAR))

    for assay_type, column in ASSAY_COLUMNS.items():
        activity = _parse_activity_nm(_cell(row, column))
        if activity is None:
            continue
        yield PreparedRow(
            smiles=smiles,
            sequence=sequence,
            assay_type=assay_type,
            ph=ph,
            temp=temp,
            pactivity=activity_to_pactivity(activity),
            activity_label=activity_label,
            year=year,
        )


def _iter_csv_rows(path: str, limit: Optional[int]) -> Iterator[Mapping[str, Any]]:
    """Stream raw rows from a prepared CSV.

    Args:
        path: CSV path.
        limit: Optional cap on raw input rows.

    Yields:
        Row mappings.
    """
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)
        for read_count, row in enumerate(reader):
            if limit is not None and read_count >= limit:
                break
            yield row


def _iter_parquet_rows(path: str, limit: Optional[int]) -> Iterator[Mapping[str, Any]]:
    """Stream raw rows from a prepared Parquet file.

    Args:
        path: Parquet path.
        limit: Optional cap on raw input rows.

    Yields:
        Row mappings.
    """
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(path)
    emitted = 0
    for batch in pf.iter_batches(batch_size=65_536):
        rows = batch.to_pylist()
        for row in rows:
            if limit is not None and emitted >= limit:
                return
            yield row
            emitted += 1


def iter_prepared_rows(
    csv_path: str = config.TRAIN_CSV,
    limit: Optional[int] = None,
) -> Iterator[PreparedRow]:
    """Stream cleaned training examples from a prepared CSV or Parquet file.

    Each raw row may yield multiple ``PreparedRow`` records: one per assay type
    that carries a valid (non-censored) measurement. Rows lacking a parseable
    SMILES or sequence, or with no valid activity, are skipped.

    When an ``Activity Label`` column is present and parseable, that explicit
    class is attached to every assay yield from the row (Papyrus binary path).
    An optional ``Year`` column is attached when present (Papyrus Parquet).

    Args:
        csv_path: Path to the prepared BindingDB CSV or Papyrus Parquet file.
        limit: Optional cap on the number of *raw input rows* read (useful for
            smoke tests). ``None`` reads the whole file.

    Yields:
        ``PreparedRow`` instances in file order.
    """
    lower = csv_path.lower()
    if lower.endswith((".parquet", ".pq")):
        raw_rows = _iter_parquet_rows(csv_path, limit)
    else:
        raw_rows = _iter_csv_rows(csv_path, limit)
    for row in raw_rows:
        yield from _yield_prepared_from_raw(row)


def unique_entities(rows: Iterable[PreparedRow]) -> tuple[list[str], list[str]]:
    """Collect the distinct ligands and proteins from prepared rows.

    Args:
        rows: Iterable of prepared rows.

    Returns:
        A tuple ``(ligand_smiles, protein_sequences)`` of de-duplicated,
        insertion-ordered lists.
    """
    ligands: dict[str, None] = {}
    proteins: dict[str, None] = {}
    for row in rows:
        ligands.setdefault(row.smiles, None)
        proteins.setdefault(row.sequence, None)
    return list(ligands), list(proteins)
