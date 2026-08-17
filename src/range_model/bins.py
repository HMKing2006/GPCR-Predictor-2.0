"""Potency bin edges and conversions for the range classifier."""

from __future__ import annotations

from typing import Final, Sequence

import numpy as np

# Upper edges in nM for bins 0..3; bin 4 is everything at/above the last edge.
# Bins: <10 nM | 10–100 nM | 100 nM–1 µM | 1–10 µM | ≥10 µM
RANGE_EDGES_NM: Final[tuple[float, ...]] = (10.0, 100.0, 1000.0, 10_000.0)
N_BINS: Final[int] = len(RANGE_EDGES_NM) + 1
BIN_LABELS: Final[tuple[str, ...]] = (
    "<10 nM",
    "10–100 nM",
    "100 nM–1 µM",
    "1–10 µM",
    "≥10 µM",
)


def activity_nm_to_bin(
    activity_nm: float,
    edges_nm: Sequence[float] = RANGE_EDGES_NM,
) -> int:
    """Map a positive activity in nM to a left-closed / right-open bin index.

    Bin ``i`` covers ``[edges[i-1], edges[i])`` with ``edges[-1]=0`` conceptually
    and the final bin covering ``[edges[-1], ∞)``.

    Args:
        activity_nm: Potency in nanomolar (must be finite and ``> 0``).
        edges_nm: Ascending upper edges for all but the last bin.

    Returns:
        Integer bin index in ``[0, len(edges_nm)]``.

    Raises:
        ValueError: If ``activity_nm`` is not a positive finite number.
    """
    value = float(activity_nm)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"activity_nm must be finite and > 0; got {activity_nm!r}.")
    for index, edge in enumerate(edges_nm):
        if value < float(edge):
            return int(index)
    return int(len(edges_nm))


def pactivity_to_nm(pactivity: float) -> float:
    """Convert pActivity back to nanomolar concentration.

    Args:
        pactivity: Continuous ``-log10(M)`` potency.

    Returns:
        Activity in nM.
    """
    return float(10.0 ** (9.0 - float(pactivity)))


def pactivity_to_bin(
    pactivity: float,
    edges_nm: Sequence[float] = RANGE_EDGES_NM,
) -> int:
    """Map a pActivity value to a potency bin.

    Args:
        pactivity: Continuous ``-log10(M)`` potency.
        edges_nm: Ascending upper edges for all but the last bin.

    Returns:
        Integer bin index.
    """
    return activity_nm_to_bin(pactivity_to_nm(pactivity), edges_nm=edges_nm)


def activity_nm_array_to_bins(
    activity_nm: np.ndarray,
    edges_nm: Sequence[float] = RANGE_EDGES_NM,
) -> np.ndarray:
    """Vectorized nM → bin mapping.

    Args:
        activity_nm: Array of positive nM values.
        edges_nm: Ascending upper edges for all but the last bin.

    Returns:
        ``uint8`` bin indices with the same shape as ``activity_nm``.
    """
    values = np.asarray(activity_nm, dtype=np.float64)
    bins = np.full(values.shape, len(edges_nm), dtype=np.uint8)
    for index, edge in enumerate(edges_nm):
        bins = np.where((bins == len(edges_nm)) & (values < float(edge)), index, bins)
    return bins.astype(np.uint8, copy=False)
