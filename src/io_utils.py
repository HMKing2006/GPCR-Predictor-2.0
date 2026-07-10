"""Readers and writers for spreadsheet, SMILES, SDF and FASTA inputs.

These helpers normalize the many accepted prediction input formats into plain
Python lists/DataFrames and route outputs back to the correct file type while
preserving the input extension.
"""

from __future__ import annotations

import os
from typing import Optional

import pandas as pd
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

SPREADSHEET_EXTS: tuple[str, ...] = (".csv", ".tsv", ".xls", ".xlsx")
SMILES_FILE_EXTS: tuple[str, ...] = (".smi", ".smiles")
SDF_EXTS: tuple[str, ...] = (".sdf",)
FASTA_EXTS: tuple[str, ...] = (".fasta", ".fa", ".faa", ".fna", ".txt")

# Candidate column headers (matched case-insensitively) for auto-detection.
_SMILES_COLUMNS: tuple[str, ...] = ("ligand smiles", "smiles", "canonical_smiles", "canonical smiles")
_SEQUENCE_COLUMNS: tuple[str, ...] = (
    "bindingdb target chain sequence 1",
    "sequence",
    "protein sequence",
    "protein",
    "fasta",
    "target sequence",
)


def ext_of(path: str) -> str:
    """Return the lower-cased file extension.

    Args:
        path: A file path.

    Returns:
        The extension including the leading dot (e.g. ``".csv"``).
    """
    return os.path.splitext(path)[1].lower()


def read_table(path: str) -> pd.DataFrame:
    """Read a csv/tsv/xls/xlsx file into a DataFrame.

    Args:
        path: Path to a spreadsheet file.

    Returns:
        The parsed DataFrame.

    Raises:
        ValueError: If the extension is not a supported spreadsheet type.
    """
    ext = ext_of(path)
    if ext == ".csv":
        return pd.read_csv(path)
    if ext == ".tsv":
        return pd.read_csv(path, sep="\t")
    if ext == ".xlsx":
        return pd.read_excel(path, engine="openpyxl")
    if ext == ".xls":
        return pd.read_excel(path, engine="xlrd")
    raise ValueError(f"Unsupported spreadsheet extension: {ext!r}")


def write_table(df: pd.DataFrame, path: str) -> None:
    """Write a DataFrame to csv/tsv/xls/xlsx based on the path extension.

    Args:
        df: The DataFrame to write.
        path: Destination path; its extension selects the format.

    Returns:
        None.

    Raises:
        ValueError: If the extension is not a supported spreadsheet type.
    """
    ext = ext_of(path)
    if ext == ".csv":
        df.to_csv(path, index=False)
    elif ext == ".tsv":
        df.to_csv(path, sep="\t", index=False)
    elif ext == ".xlsx":
        df.to_excel(path, index=False, engine="openpyxl")
    elif ext == ".xls":
        _write_xls(df, path)
    else:
        raise ValueError(f"Unsupported spreadsheet extension: {ext!r}")


def _write_xls(df: pd.DataFrame, path: str) -> None:
    """Write a DataFrame to a legacy ``.xls`` file using ``xlwt``.

    Modern pandas no longer routes ``.xls`` writing through an engine, so the
    file is produced directly with ``xlwt``.

    Args:
        df: The DataFrame to write.
        path: Destination ``.xls`` path.

    Returns:
        None.

    Raises:
        ImportError: If ``xlwt`` is not installed.
    """
    try:
        import xlwt
    except ImportError as exc:  # pragma: no cover - depends on optional dep
        raise ImportError(
            "Writing .xls files requires the 'xlwt' package. Install it or use a "
            ".xlsx/.csv output extension instead."
        ) from exc
    book = xlwt.Workbook()
    sheet = book.add_sheet("Sheet1")
    for col, name in enumerate(df.columns):
        sheet.write(0, col, str(name))
    for row_idx, (_, row) in enumerate(df.iterrows(), start=1):
        for col, value in enumerate(row.tolist()):
            sheet.write(row_idx, col, None if pd.isna(value) else value)
    book.save(path)


def predictions_output_path(input_path: str, suffix: str = "_predictions") -> str:
    """Build the output path for a per-spreadsheet prediction file.

    Args:
        input_path: The input spreadsheet path.
        suffix: Suffix inserted before the extension.

    Returns:
        A sibling path ``<stem><suffix><ext>`` preserving the extension.
    """
    stem, ext = os.path.splitext(input_path)
    return f"{stem}{suffix}{ext}"


def _find_column(df: pd.DataFrame, candidates: tuple[str, ...]) -> Optional[str]:
    """Find the first DataFrame column matching any candidate name.

    Args:
        df: The DataFrame to inspect.
        candidates: Lower-cased candidate column names.

    Returns:
        The actual column name if found, else ``None``.
    """
    lower_map = {str(c).strip().lower(): c for c in df.columns}
    for candidate in candidates:
        if candidate in lower_map:
            return lower_map[candidate]
    return None


def find_smiles_column(df: pd.DataFrame) -> Optional[str]:
    """Locate the ligand SMILES column in a DataFrame.

    Args:
        df: The DataFrame to inspect.

    Returns:
        The SMILES column name, or ``None`` if none matches.
    """
    return _find_column(df, _SMILES_COLUMNS)


def find_sequence_column(df: pd.DataFrame) -> Optional[str]:
    """Locate the protein sequence column in a DataFrame.

    Args:
        df: The DataFrame to inspect.

    Returns:
        The sequence column name, or ``None`` if none matches.
    """
    return _find_column(df, _SEQUENCE_COLUMNS)


def _read_smi_file(path: str) -> list[str]:
    """Read SMILES from a ``.smi``/``.smiles`` file (one per line).

    Args:
        path: Path to the SMILES file.

    Returns:
        A list of SMILES strings (first whitespace token per non-empty line).
    """
    smiles: list[str] = []
    with open(path) as handle:
        for line in handle:
            token = line.strip().split()
            if token:
                smiles.append(token[0])
    return smiles


def _read_sdf_file(path: str) -> list[str]:
    """Read molecules from an SDF file and return canonical SMILES.

    Args:
        path: Path to the ``.sdf`` file.

    Returns:
        A list of SMILES strings for each successfully parsed molecule.
    """
    smiles: list[str] = []
    supplier = Chem.SDMolSupplier(path)
    for mol in supplier:
        if mol is not None:
            smiles.append(Chem.MolToSmiles(mol))
    return smiles


def read_ligands(source: str) -> list[str]:
    """Read ligand SMILES from any supported ligand input.

    Args:
        source: A raw SMILES string, or a path to a spreadsheet, ``.smi``/
            ``.smiles`` file, or ``.sdf`` file.

    Returns:
        A list of SMILES strings.

    Raises:
        ValueError: If a spreadsheet input lacks a recognizable SMILES column.
    """
    if not os.path.exists(source):
        return [source.strip()]
    ext = ext_of(source)
    if ext in SMILES_FILE_EXTS:
        return _read_smi_file(source)
    if ext in SDF_EXTS:
        return _read_sdf_file(source)
    if ext in SPREADSHEET_EXTS:
        df = read_table(source)
        col = find_smiles_column(df)
        if col is None:
            raise ValueError(f"No SMILES column found in {source!r}")
        return [str(v) for v in df[col].dropna().tolist()]
    # Fall back to treating the file as a line-delimited SMILES list.
    return _read_smi_file(source)


def _parse_fasta_text(text: str) -> list[str]:
    """Parse FASTA-formatted text into sequences.

    Args:
        text: FASTA text, or a single bare sequence with no header.

    Returns:
        A list of amino-acid sequences (headers stripped).
    """
    sequences: list[str] = []
    current: list[str] = []
    saw_header = False
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            saw_header = True
            if current:
                sequences.append("".join(current))
                current = []
        else:
            current.append(line)
    if current:
        sequences.append("".join(current))
    if not saw_header and not sequences and text.strip():
        sequences.append("".join(text.split()))
    return sequences


def read_proteins(source: str) -> list[str]:
    """Read protein sequences from any supported protein input.

    Args:
        source: A raw FASTA/sequence string, or a path to a FASTA file or a
            spreadsheet with a sequence column.

    Returns:
        A list of amino-acid sequences.

    Raises:
        ValueError: If a spreadsheet input lacks a recognizable sequence column.
    """
    if not os.path.exists(source):
        return _parse_fasta_text(source)
    ext = ext_of(source)
    if ext in SPREADSHEET_EXTS:
        df = read_table(source)
        col = find_sequence_column(df)
        if col is None:
            raise ValueError(f"No sequence column found in {source!r}")
        return [str(v).strip() for v in df[col].dropna().tolist()]
    with open(source) as handle:
        return _parse_fasta_text(handle.read())


def list_files(directory: str, exts: tuple[str, ...]) -> list[str]:
    """List files in a directory whose extension is in ``exts``.

    Args:
        directory: Directory to scan (non-recursively).
        exts: Acceptable lower-cased extensions.

    Returns:
        A sorted list of matching absolute file paths.
    """
    out: list[str] = []
    for name in sorted(os.listdir(directory)):
        path = os.path.join(directory, name)
        if os.path.isfile(path) and ext_of(path) in exts:
            out.append(path)
    return out
