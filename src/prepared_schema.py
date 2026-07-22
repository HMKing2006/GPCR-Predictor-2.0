"""Shared BindingDB-compatible prepared-table schema.

Builders (:mod:`build_papyrus`, :mod:`build_gpcrdb`) and the training streamer
(:func:`src.data_prep.iter_prepared_rows`) agree on this column set so prepared
Parquet files from different sources can be concatenated for joint training.
"""

from __future__ import annotations

from typing import Final

import pyarrow as pa

# Column order for BindingDB-compatible prepared CSV / Parquet.
OUTPUT_COLUMNS: Final[list[str]] = [
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

PARQUET_SCHEMA: Final[pa.Schema] = pa.schema(
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
