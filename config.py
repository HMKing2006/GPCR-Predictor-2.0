"""Central configuration for the GPCR binding-affinity predictor.

This module holds default model identifiers, filesystem paths, feature layout
constants, and default hyperparameters shared across the training, grid-search
and prediction entry points. Values here are intentionally plain module-level
constants so they can be imported cheaply from any script.
"""

from __future__ import annotations

import os
from typing import Any, Final

# --- Filesystem layout -------------------------------------------------------

PROJECT_ROOT: Final[str] = os.path.dirname(os.path.abspath(__file__))
DATA_DIR: Final[str] = os.path.join(PROJECT_ROOT, "data")
TRAIN_CSV: Final[str] = os.path.join(DATA_DIR, "train", "BindingDB_all_prepared.csv")
CACHE_DIR: Final[str] = os.path.join(PROJECT_ROOT, "cache")
SPLITS_DIR: Final[str] = os.path.join(DATA_DIR, "splits")
MODELS_DIR: Final[str] = os.path.join(PROJECT_ROOT, "models")
FEATURES_DIR: Final[str] = os.path.join(CACHE_DIR, "features")

# --- Default embedding models ------------------------------------------------

DEFAULT_PROTEIN_MODEL: Final[str] = "facebook/esm2_t33_650M_UR50D"
DEFAULT_LIGAND_MODEL: Final[str] = "ibm-research/MoLFormer-XL-both-10pct"

# Embedding dimensionalities for the default models. These are validated at
# runtime against the actual model output and only used for pre-allocation.
PROTEIN_EMB_DIM: Final[int] = 1280
LIGAND_EMB_DIM: Final[int] = 768

# Models that ship custom modeling code on the Hub and must be loaded with
# ``trust_remote_code=True`` (e.g. MoLFormer's linear-attention implementation).
TRUST_REMOTE_CODE_MODELS: Final[tuple[str, ...]] = (
    "ibm-research/MoLFormer-XL-both-10pct",
    "ibm/MoLFormer-XL-both-10pct",
)

# --- Feature layout ----------------------------------------------------------

# One-hot column order for the assay type. This order is fixed so that saved
# models remain compatible across runs.
ASSAY_TYPES: Final[tuple[str, ...]] = ("IC50", "EC50", "Ki", "Kd")

# Scalar auxiliary features appended after the assay one-hot block.
DEFAULT_PH: Final[float] = 7.4
DEFAULT_TEMP_C: Final[float] = 25.0

# Plausible physical bounds used to reject junk temperature values.
TEMP_MIN_C: Final[float] = -80.0
TEMP_MAX_C: Final[float] = 100.0

# Number of trailing scalar features: pH and temperature.
NUM_SCALAR_FEATURES: Final[int] = 2


def feature_dim(protein_dim: int = PROTEIN_EMB_DIM, ligand_dim: int = LIGAND_EMB_DIM) -> int:
    """Compute the total feature-vector length.

    Args:
        protein_dim: Dimensionality of the protein embedding.
        ligand_dim: Dimensionality of the ligand embedding.

    Returns:
        The concatenated feature dimension: protein + ligand + assay one-hot +
        the scalar auxiliary features (pH and temperature).
    """
    return protein_dim + ligand_dim + len(ASSAY_TYPES) + NUM_SCALAR_FEATURES


# --- Default hyperparameters -------------------------------------------------

DEFAULT_MODEL_TYPE: Final[str] = "rf"
RANDOM_SEED: Final[int] = 42
TEST_FRACTION: Final[float] = 0.20
GRID_VAL_FRACTION: Final[float] = 0.10
GRID_TEST_FRACTION: Final[float] = 0.10

# Random-forest defaults (warm-start incremental training).
RF_DEFAULTS: Final[dict[str, int]] = {
    "n_estimators": 200,
    "batch_trees": 20,
    "shard_rows": 100_000,
    "max_depth": 0,  # 0 means unlimited (None)
    "min_samples_leaf": 1,
    "n_jobs": -1,
}

# Torch MLP defaults.
#
# ``patience`` and ``es_val_fraction`` drive early stopping: a random
# ``es_val_fraction`` of the training rows is held out each fit, validation RMSE
# is tracked after every epoch, the best weights are restored, and training
# stops once RMSE fails to improve for ``patience`` consecutive epochs. Set
# ``patience`` to ``0`` to disable early stopping and always run ``epochs``.
# ``use_batchnorm`` inserts BatchNorm1d after each hidden linear layer.
# ``use_bilinear`` adds a learned protein/ligand bilinear interaction block: the
# protein and ligand embeddings are projected to ``bilinear_dim`` and combined by
# ``nn.Bilinear`` before being concatenated back into the MLP trunk.
MLP_DEFAULTS: Final[dict[str, Any]] = {
    "hidden_dim": 1024,
    "num_layers": 3,
    "dropout": 0.1,
    "batch_size": 512,
    "epochs": 20,
    "learning_rate": 1e-3,
    "weight_decay": 1e-5,
    "patience": 4,
    "es_val_fraction": 0.05,
    "es_min_delta": 1e-4,
    "use_batchnorm": False,
    "use_bilinear": False,
    "bilinear_dim": 256,
}

# --- Embedding runtime -------------------------------------------------------

PROTEIN_BATCH_SIZE: Final[int] = 8
LIGAND_BATCH_SIZE: Final[int] = 64
MAX_PROTEIN_LEN: Final[int] = 1022
