"""Streaming cleaning and label preparation for the BindingDB CSV.

This module reads the raw BindingDB export row by row (never loading the whole
file into memory), cleans ligand SMILES (salt removal + canonicalization),
converts activity measurements to pActivity, explodes rows that carry several
assay measurements, drops censored/missing values, and yields tidy
``PreparedRow`` records ready for embedding and featurization.
"""

from __future__ import annotations

import csv
import math
import sys
from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable, Iterator, Optional

from rdkit import Chem, RDLogger
from rdkit.Chem.MolStandardize import rdMolStandardize

import config

RDLogger.DisableLog("rdApp.*")

# Raw CSV column names.
COL_SMILES = "Ligand SMILES"
COL_PH = "pH"
COL_TEMP = "Temp (C)"
COL_SEQ = "BindingDB Target Chain Sequence 1"

# Mapping from assay type to its raw activity column (values in nM).
ASSAY_COLUMNS: dict[str, str] = {
    "IC50": "IC50 (nM)",
    "EC50": "EC50 (nM)",
    "Ki": "Ki (nM)",
    "Kd": "Kd (nM)",
}

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
        pactivity: Target label, ``-log10(activity_nM * 1e-9)``.
    """

    smiles: str
    sequence: str
    assay_type: str
    ph: float
    temp: float
    pactivity: float


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


def iter_prepared_rows(
    csv_path: str = config.TRAIN_CSV,
    limit: Optional[int] = None,
) -> Iterator[PreparedRow]:
    """Stream cleaned training examples from the BindingDB CSV.

    Each raw row may yield multiple ``PreparedRow`` records: one per assay type
    that carries a valid (non-censored) measurement. Rows lacking a parseable
    SMILES or sequence, or with no valid activity, are skipped.

    Args:
        csv_path: Path to the prepared BindingDB CSV.
        limit: Optional cap on the number of *raw input rows* read (useful for
            smoke tests). ``None`` reads the whole file.

    Yields:
        ``PreparedRow`` instances in file order.
    """
    with open(csv_path, newline="") as handle:
        reader = csv.DictReader(handle)
        for read_count, row in enumerate(reader):
            if limit is not None and read_count >= limit:
                break

            sequence = (row.get(COL_SEQ) or "").strip()
            if not sequence:
                continue

            smiles = canonicalize_smiles(row.get(COL_SMILES) or "")
            if smiles is None:
                continue

            ph = parse_ph(row.get(COL_PH) or "")
            if ph is None:
                ph = config.DEFAULT_PH
            temp = parse_temperature(row.get(COL_TEMP) or "")
            if temp is None:
                temp = config.DEFAULT_TEMP_C

            for assay_type, column in ASSAY_COLUMNS.items():
                activity = _parse_activity_nm(row.get(column) or "")
                if activity is None:
                    continue
                yield PreparedRow(
                    smiles=smiles,
                    sequence=sequence,
                    assay_type=assay_type,
                    ph=ph,
                    temp=temp,
                    pactivity=activity_to_pactivity(activity),
                )


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
