"""Defaults and paths for the experimental ligand multilabel stack."""

from __future__ import annotations

import os
from typing import Any, Final

import config as root_config

# --- Filesystem layout -------------------------------------------------------

TRAIN_DIR: Final[str] = os.path.join(root_config.DATA_DIR, "train")
PROTEINS_SIDECAR: Final[str] = os.path.join(
    TRAIN_DIR, "Papyrus_proteins_multilabel.parquet"
)
FAMILY_PREPARED: Final[str] = os.path.join(TRAIN_DIR, "Ligand_family_prepared.parquet")
TARGET_PREPARED: Final[str] = os.path.join(TRAIN_DIR, "Ligand_target_prepared.parquet")
FAMILY_VOCAB_PATH: Final[str] = os.path.join(TRAIN_DIR, "family_vocab.json")
TARGET_VOCAB_PATH: Final[str] = os.path.join(TRAIN_DIR, "target_vocab.json")

# GPCRdb multilabel artifacts (separate from Papyrus so both can coexist).
GPCRDB_PROTEINS_SIDECAR: Final[str] = os.path.join(
    TRAIN_DIR, "GPCRdb_proteins_multilabel.parquet"
)
GPCRDB_FAMILY_PREPARED: Final[str] = os.path.join(
    TRAIN_DIR, "GPCRdb_Ligand_family_prepared.parquet"
)
GPCRDB_TARGET_PREPARED: Final[str] = os.path.join(
    TRAIN_DIR, "GPCRdb_Ligand_target_prepared.parquet"
)
GPCRDB_FAMILY_VOCAB_PATH: Final[str] = os.path.join(TRAIN_DIR, "GPCRdb_family_vocab.json")
GPCRDB_TARGET_VOCAB_PATH: Final[str] = os.path.join(TRAIN_DIR, "GPCRdb_target_vocab.json")

MODELS_DIR: Final[str] = os.path.join(root_config.MODELS_DIR, "multilabel")

# Default activity source for aggregation (pair prepared Parquet, unchanged).
DEFAULT_ACTIVITY_SOURCE: Final[str] = root_config.PAPYRUS_FULL_BINARY_TRAIN_CSV
DEFAULT_GPCRDB_ACTIVITY_SOURCE: Final[str] = root_config.GPCRDB_TRAIN_CSV

# --- Vocabulary defaults -----------------------------------------------------

CLASSIFICATION_DEPTH: Final[int] = 2  # 1-based path depth (level-2 token)
MIN_POSITIVES: Final[int] = 100  # unique active ligands required per label
TARGET_VOCAB_SIZE: Final[int] = 1024

# --- Split / training defaults -----------------------------------------------

RANDOM_SEED: Final[int] = root_config.RANDOM_SEED
TEST_FRACTION: Final[float] = root_config.TEST_FRACTION
VAL_FRACTION: Final[float] = root_config.GRID_VAL_FRACTION
ACTIVITY_THRESHOLD_NM: Final[float] = root_config.ACTIVITY_THRESHOLD_NM
DEFAULT_LIGAND_MODEL: Final[str] = "morgan,descriptors"

# Outer fold strategies for ``--test-split`` / ``--validation-split``.
# ``scaffold`` = Murcko-scaffold cold; ``time`` = publication-year.
SPLIT_STRATEGIES: Final[tuple[str, ...]] = ("scaffold", "time")
DEFAULT_TEST_SPLIT: Final[str] = "scaffold"
DEFAULT_VALIDATION_SPLIT: Final[str] = "scaffold"

# Multilabel MLP defaults (ligand-only trunk; no bilinear / protein slice).
MLP_DEFAULTS: Final[dict[str, Any]] = {
    "hidden_dim": int(root_config.MLP_DEFAULTS["hidden_dim"]),
    "num_layers": int(root_config.MLP_DEFAULTS["num_layers"]),
    "dropout": float(root_config.MLP_DEFAULTS["dropout"]),
    "batch_size": int(root_config.MLP_DEFAULTS["batch_size"]),
    "epochs": int(root_config.MLP_DEFAULTS["epochs"]),
    "learning_rate": float(root_config.MLP_DEFAULTS["learning_rate"]),
    "weight_decay": float(root_config.MLP_DEFAULTS["weight_decay"]),
    "patience": int(root_config.MLP_DEFAULTS["patience"]),
    "es_val_fraction": float(root_config.MLP_DEFAULTS["es_val_fraction"]),
    "es_min_delta": float(root_config.MLP_DEFAULTS["es_min_delta"]),
    "class_weights": True,
}

STORAGE_VERSION: Final[str] = "ligand_multilabel_v1"
