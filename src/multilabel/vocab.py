"""Family and target vocabulary construction and persistence."""

from __future__ import annotations

import json
import os
from collections import Counter
from typing import Any, Iterable, Optional, Sequence

from src.multilabel import config as ml_config


def parse_classification_paths(raw: object) -> list[list[str]]:
    """Parse a Papyrus ``Classification`` cell into path token lists.

    Paths are separated by ``";"`` and levels within a path by ``"->"``.

    Args:
        raw: Raw Classification cell value.

    Returns:
        A list of non-empty token paths (each path is a list of level strings).
    """
    if raw is None:
        return []
    text = str(raw).strip()
    if not text or text.lower() == "nan":
        return []
    paths: list[list[str]] = []
    for chunk in text.split(";"):
        tokens = [part.strip() for part in chunk.split("->") if part.strip()]
        if tokens:
            paths.append(tokens)
    return paths


def level_tokens(raw: object, depth: int = ml_config.CLASSIFICATION_DEPTH) -> set[str]:
    """Extract Classification tokens at a fixed 1-based path depth.

    Args:
        raw: Raw Classification cell value.
        depth: 1-based depth (``2`` selects the level-2 family token).

    Returns:
        Unique tokens present at ``depth`` across all paths.
    """
    if depth < 1:
        raise ValueError(f"Classification depth must be >= 1, got {depth}.")
    index = depth - 1
    tokens: set[str] = set()
    for path in parse_classification_paths(raw):
        if len(path) > index:
            tokens.add(path[index])
    return tokens


def save_vocab(path: str, labels: Sequence[str], meta: Optional[dict[str, Any]] = None) -> None:
    """Atomically write a vocabulary JSON sidecar.

    Args:
        path: Destination JSON path.
        labels: Ordered label strings (index = column in multi-hot).
        meta: Optional extra metadata merged into the file.

    Returns:
        None.
    """
    payload: dict[str, Any] = {
        "labels": list(labels),
        "size": len(labels),
    }
    if meta:
        payload.update(meta)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    temporary = f"{path}.tmp-{os.getpid()}"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def load_vocab(path: str) -> list[str]:
    """Load an ordered vocabulary from a JSON sidecar.

    Args:
        path: Vocabulary JSON path.

    Returns:
        Ordered label strings.

    Raises:
        ValueError: If the file is missing required fields.
    """
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    labels = payload.get("labels")
    if not isinstance(labels, list) or not all(isinstance(x, str) for x in labels):
        raise ValueError(f"Invalid vocabulary file: {path}")
    return list(labels)


def build_family_vocab(
    classifications: Iterable[object],
    depth: int = ml_config.CLASSIFICATION_DEPTH,
) -> list[str]:
    """Build a sorted family vocabulary from Classification cells.

    Args:
        classifications: Iterable of Classification cell values.
        depth: 1-based Classification path depth.

    Returns:
        Sorted unique level tokens.
    """
    labels: set[str] = set()
    for raw in classifications:
        labels.update(level_tokens(raw, depth=depth))
    return sorted(labels)


def filter_vocab_by_min_positives(
    labels: Sequence[str],
    active_counts: Counter[str],
    min_positives: int = ml_config.MIN_POSITIVES,
) -> list[str]:
    """Keep vocabulary labels with enough unique active ligands.

    Args:
        labels: Candidate vocabulary labels (order preserved among survivors).
        active_counts: Mapping ``label -> unique active ligand count``.
        min_positives: Minimum unique active ligands required for inclusion.

    Returns:
        Filtered label list in the same relative order as ``labels``.
    """
    return [
        label
        for label in labels
        if int(active_counts.get(label, 0)) >= int(min_positives)
    ]


def build_target_vocab(
    target_active_counts: Counter[str],
    min_positives: int = ml_config.MIN_POSITIVES,
    max_size: int = ml_config.TARGET_VOCAB_SIZE,
) -> list[str]:
    """Select the top targets by active-ligand count.

    Args:
        target_active_counts: Mapping ``target_id -> unique active ligand count``.
        min_positives: Minimum unique active ligands required for inclusion.
        max_size: Maximum vocabulary size after filtering.

    Returns:
        Target ids ordered by descending active count, then lexicographically.
    """
    eligible = [
        (target_id, count)
        for target_id, count in target_active_counts.items()
        if count >= min_positives
    ]
    eligible.sort(key=lambda item: (-item[1], item[0]))
    return [target_id for target_id, _ in eligible[:max_size]]


def label_index(vocab: Sequence[str]) -> dict[str, int]:
    """Map vocabulary strings to column indices.

    Args:
        vocab: Ordered vocabulary.

    Returns:
        Mapping from label string to integer index.
    """
    return {label: i for i, label in enumerate(vocab)}
