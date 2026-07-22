"""Aggregate prepared activity rows into ligand-centric multilabel tables."""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Iterable, Iterator, Mapping, Optional

import numpy as np

from src.data_prep import binarize_pactivity, canonicalize_smiles, iter_prepared_rows
from src.multilabel import config as ml_config
from src.multilabel.vocab import label_index, level_tokens


@dataclass
class LigandAccumulator:
    """Mutable per-ligand aggregation state for multilabel construction.

    Attributes:
        families: Level-2 Classification tokens from active annotations.
        targets: ``target_id`` values from active annotations.
        year: Maximum dated publication year among active annotations, or
            ``None`` when no dated active row was observed.
    """

    families: set[str] = field(default_factory=set)
    targets: set[str] = field(default_factory=set)
    year: Optional[int] = None


def is_active_row(
    activity_label: Optional[float],
    pactivity: float,
    threshold_nm: float = ml_config.ACTIVITY_THRESHOLD_NM,
) -> bool:
    """Decide whether a prepared activity row counts as a binder positive.

    Explicit ``Activity Label`` wins when present; otherwise quantitative
    pActivity is binarized at ``threshold_nm``.

    Args:
        activity_label: Optional explicit ``0`` / ``1`` label.
        pactivity: Continuous pActivity value.
        threshold_nm: Quantitative binder cutoff in nM.

    Returns:
        ``True`` when the row is treated as active.
    """
    if activity_label is not None:
        return int(activity_label) == 1
    return bool(
        binarize_pactivity(
            np.asarray([pactivity], dtype=np.float32),
            threshold_nm=threshold_nm,
        )[0]
    )


def update_year(current: Optional[int], year: Optional[int]) -> Optional[int]:
    """Return the max of two optional years.

    Args:
        current: Existing max year, or ``None``.
        year: Candidate year, or ``None``.

    Returns:
        The maximum dated year, or ``None`` if both are missing.
    """
    if year is None:
        return current
    if current is None:
        return int(year)
    return max(int(current), int(year))


def aggregate_ligand_labels(
    activity_source: str,
    sequence_to_protein: Mapping[str, Mapping[str, str]],
    *,
    threshold_nm: float = ml_config.ACTIVITY_THRESHOLD_NM,
    classification_depth: int = ml_config.CLASSIFICATION_DEPTH,
    limit: Optional[int] = None,
    verbose: bool = True,
) -> dict[str, LigandAccumulator]:
    """Stream prepared activity rows and aggregate active labels per ligand.

    Only active rows contribute family/target labels and years. Ligands that
    never appear as active are omitted.

    Args:
        activity_source: Prepared pair Parquet / CSV path.
        sequence_to_protein: Mapping ``sequence -> {target_id, Classification}``.
        threshold_nm: Binder cutoff for quantitative rows.
        classification_depth: 1-based Classification path depth for families.
        limit: Optional prepared-row limit forwarded to the streamer.
        verbose: If ``True``, print progress.

    Returns:
        Mapping from canonical ligand SMILES to accumulated label state.
    """
    by_ligand: dict[str, LigandAccumulator] = {}
    n_active = 0
    n_missing_protein = 0
    for kept, row in enumerate(iter_prepared_rows(activity_source, limit=limit), start=1):
        if not is_active_row(row.activity_label, row.pactivity, threshold_nm):
            continue
        protein = sequence_to_protein.get(row.sequence)
        if protein is None:
            n_missing_protein += 1
            continue
        n_active += 1
        state = by_ligand.setdefault(row.smiles, LigandAccumulator())
        state.families.update(
            level_tokens(protein.get("Classification", ""), depth=classification_depth)
        )
        target_id = str(protein.get("target_id", "") or "").strip()
        if target_id:
            state.targets.add(target_id)
        state.year = update_year(state.year, row.year)
        if verbose and kept % 200_000 == 0:
            print(
                f"[multilabel] scanned {kept} rows "
                f"(active_kept={n_active}, ligands={len(by_ligand)})",
                flush=True,
            )
    if verbose:
        print(
            f"[multilabel] aggregation done: ligands={len(by_ligand)} "
            f"active_rows={n_active} missing_protein={n_missing_protein}",
            flush=True,
        )
    return by_ligand


def count_family_actives(by_ligand: Mapping[str, LigandAccumulator]) -> Counter[str]:
    """Count unique active ligands per family label.

    Args:
        by_ligand: Aggregated ligand label state.

    Returns:
        ``Counter`` of family labels to unique-ligand counts.
    """
    counts: Counter[str] = Counter()
    for state in by_ligand.values():
        for family in state.families:
            counts[family] += 1
    return counts


def count_target_actives(by_ligand: Mapping[str, LigandAccumulator]) -> Counter[str]:
    """Count unique active ligands per ``target_id``.

    Args:
        by_ligand: Aggregated ligand label state.

    Returns:
        ``Counter`` of target ids to unique-ligand counts.
    """
    counts: Counter[str] = Counter()
    for state in by_ligand.values():
        for target_id in state.targets:
            counts[target_id] += 1
    return counts


def multi_hot(
    labels: Iterable[str],
    vocab: Sequence[str],
) -> np.ndarray:
    """Encode a label set as a ``uint8`` multi-hot vector.

    Args:
        labels: Label strings present for one ligand.
        vocab: Ordered vocabulary.

    Returns:
        ``uint8`` vector of shape ``(len(vocab),)``.
    """
    vector = np.zeros(len(vocab), dtype=np.uint8)
    index = label_index(vocab)
    for label in labels:
        col = index.get(label)
        if col is not None:
            vector[col] = 1
    return vector


def build_label_matrix(
    smiles_order: Sequence[str],
    by_ligand: Mapping[str, LigandAccumulator],
    vocab: Sequence[str],
    *,
    kind: str,
) -> np.ndarray:
    """Build a dense multilabel matrix for ligands in a fixed order.

    Args:
        smiles_order: Canonical SMILES in row order.
        by_ligand: Aggregated ligand state.
        vocab: Ordered vocabulary.
        kind: ``"family"`` or ``"target"`` selecting which label set to encode.

    Returns:
        ``uint8`` array of shape ``(n_ligands, len(vocab))``.

    Raises:
        ValueError: If ``kind`` is not recognized.
    """
    if kind not in {"family", "target"}:
        raise ValueError(f"kind must be 'family' or 'target', got {kind!r}.")
    matrix = np.zeros((len(smiles_order), len(vocab)), dtype=np.uint8)
    index = label_index(vocab)
    for row_i, smiles in enumerate(smiles_order):
        state = by_ligand[smiles]
        labels = state.families if kind == "family" else state.targets
        for label in labels:
            col = index.get(label)
            if col is not None:
                matrix[row_i, col] = 1
    return matrix


def years_array(
    smiles_order: Sequence[str],
    by_ligand: Mapping[str, LigandAccumulator],
) -> np.ndarray:
    """Build a per-ligand year array (``-1`` when undated).

    Args:
        smiles_order: Canonical SMILES in row order.
        by_ligand: Aggregated ligand state.

    Returns:
        ``int32`` array of shape ``(n_ligands,)``.
    """
    years = np.full(len(smiles_order), -1, dtype=np.int32)
    for i, smiles in enumerate(smiles_order):
        year = by_ligand[smiles].year
        if year is not None:
            years[i] = int(year)
    return years


def iter_canonical_smiles(raw_smiles: Iterable[str]) -> Iterator[str]:
    """Yield desalted canonical SMILES, skipping unparseable inputs.

    Args:
        raw_smiles: Raw SMILES strings.

    Yields:
        Canonical SMILES strings.
    """
    for smiles in raw_smiles:
        canon = canonicalize_smiles(smiles)
        if canon is not None:
            yield canon


def finite_or_none(value: Any) -> Optional[float]:
    """Coerce a value to float when finite, else ``None``.

    Args:
        value: Arbitrary cell value.

    Returns:
        Finite float or ``None``.
    """
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number
