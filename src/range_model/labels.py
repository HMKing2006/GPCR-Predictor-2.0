"""Filter prepared rows into IC50/EC50 range-training examples."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Optional

from src.data_prep import PreparedRow
from src.range_model import config as range_config
from src.range_model.bins import pactivity_to_bin


@dataclass(frozen=True, slots=True)
class RangeRow:
    """One IC50 or EC50 example with a discrete potency bin.

    Attributes:
        smiles: Canonical ligand SMILES.
        sequence: Protein amino-acid sequence.
        assay_type: ``IC50`` or ``EC50``.
        head_idx: ``0`` for IC50, ``1`` for EC50.
        y_bin: Potency bin index in ``[0, N_BINS)``.
        ph: Assay pH.
        temp: Assay temperature in Celsius.
        year: Optional publication year.
        loss_weight: Per-row loss multiplier (binary weak labels may differ).
        from_binary: Whether this row came from a Papyrus Activity_class row.
    """

    smiles: str
    sequence: str
    assay_type: str
    head_idx: int
    y_bin: int
    ph: float
    temp: float
    year: Optional[int]
    loss_weight: float = 1.0
    from_binary: bool = False


def _quant_range_row(row: PreparedRow) -> Optional[RangeRow]:
    """Build a range row from a quantitative IC50/EC50 prepared row.

    Args:
        row: Prepared activity row without an explicit activity label.

    Returns:
        Range row, or ``None`` when the assay is not IC50/EC50.
    """
    if row.assay_type not in range_config.RANGE_ASSAY_TYPES:
        return None
    head_idx = (
        range_config.HEAD_IC50 if row.assay_type == "IC50" else range_config.HEAD_EC50
    )
    return RangeRow(
        smiles=row.smiles,
        sequence=row.sequence,
        assay_type=row.assay_type,
        head_idx=head_idx,
        y_bin=pactivity_to_bin(row.pactivity),
        ph=row.ph,
        temp=row.temp,
        year=row.year,
        loss_weight=1.0,
        from_binary=False,
    )


def _binary_inactive_rows(
    row: PreparedRow,
    *,
    binary_inactive_bin: int,
    binary_loss_weight: float,
) -> list[RangeRow]:
    """Map an inactive binary row onto both IC50 and EC50 weak bins.

    Papyrus ``Activity_class`` rows are stored under ``Other (nM)``. When the
    include-binary hook is enabled they contribute once per head at the
    configured weak bin (default ``≥10 µM``).

    Args:
        row: Prepared row with ``activity_label == 0``.
        binary_inactive_bin: Target bin index.
        binary_loss_weight: Per-row loss weight.

    Returns:
        Two :class:`RangeRow` instances (IC50 and EC50), or one when the
        prepared row already carries an IC50/EC50 assay type.
    """
    if row.assay_type in range_config.RANGE_ASSAY_TYPES:
        head_idx = (
            range_config.HEAD_IC50
            if row.assay_type == "IC50"
            else range_config.HEAD_EC50
        )
        return [
            RangeRow(
                smiles=row.smiles,
                sequence=row.sequence,
                assay_type=row.assay_type,
                head_idx=head_idx,
                y_bin=int(binary_inactive_bin),
                ph=row.ph,
                temp=row.temp,
                year=row.year,
                loss_weight=float(binary_loss_weight),
                from_binary=True,
            )
        ]
    return [
        RangeRow(
            smiles=row.smiles,
            sequence=row.sequence,
            assay_type=name,
            head_idx=idx,
            y_bin=int(binary_inactive_bin),
            ph=row.ph,
            temp=row.temp,
            year=row.year,
            loss_weight=float(binary_loss_weight),
            from_binary=True,
        )
        for idx, name in enumerate(range_config.HEAD_NAMES)
    ]


def include_prepared_row(
    row: PreparedRow,
    *,
    include_binary: bool = range_config.INCLUDE_BINARY_IN_RANGE,
    binary_inactive_bin: int = range_config.BINARY_INACTIVE_RANGE_BIN,
    binary_loss_weight: float = range_config.BINARY_RANGE_LOSS_WEIGHT,
) -> list[RangeRow]:
    """Convert a prepared row into zero or more range examples.

    Quantitative IC50/EC50 rows (no ``activity_label``) map via pActivity bins.
    When ``include_binary`` is True, inactive binary rows map to
    ``binary_inactive_bin`` (typically for both heads when the source assay is
    ``Other``). Active binary rows are skipped.

    Args:
        row: Streamed prepared activity row.
        include_binary: Whether inactive binary rows may enter training.
        binary_inactive_bin: Bin index for inactive binary sentinels.
        binary_loss_weight: Loss weight applied to binary-derived rows.

    Returns:
        Zero or more :class:`RangeRow` instances.
    """
    if row.activity_label is None:
        converted = _quant_range_row(row)
        return [converted] if converted is not None else []

    if not include_binary:
        return []
    if int(row.activity_label) != 0:
        return []
    return _binary_inactive_rows(
        row,
        binary_inactive_bin=binary_inactive_bin,
        binary_loss_weight=binary_loss_weight,
    )


def iter_range_rows(
    rows: Iterator[PreparedRow],
    *,
    include_binary: bool = range_config.INCLUDE_BINARY_IN_RANGE,
    binary_inactive_bin: int = range_config.BINARY_INACTIVE_RANGE_BIN,
    binary_loss_weight: float = range_config.BINARY_RANGE_LOSS_WEIGHT,
) -> Iterator[RangeRow]:
    """Yield range examples from a prepared-row stream.

    Args:
        rows: Iterable of :class:`PreparedRow`.
        include_binary: Whether inactive binary rows may enter training.
        binary_inactive_bin: Bin index for inactive binary sentinels.
        binary_loss_weight: Loss weight applied to binary-derived rows.

    Yields:
        Filtered :class:`RangeRow` instances.
    """
    for row in rows:
        yield from include_prepared_row(
            row,
            include_binary=include_binary,
            binary_inactive_bin=binary_inactive_bin,
            binary_loss_weight=binary_loss_weight,
        )
