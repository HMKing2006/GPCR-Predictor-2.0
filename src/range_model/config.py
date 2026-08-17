"""Configuration for the IC50/EC50 potency-range model."""

from __future__ import annotations

from typing import Final

import config as root_config

# Assay heads (index into HEAD_NAMES).
HEAD_IC50: Final[int] = 0
HEAD_EC50: Final[int] = 1
HEAD_NAMES: Final[tuple[str, ...]] = ("IC50", "EC50")
RANGE_ASSAY_TYPES: Final[frozenset[str]] = frozenset(HEAD_NAMES)

# When True, inactive Papyrus binary rows (Activity Label == 0) can contribute
# to the weakest bin via :data:`BINARY_INACTIVE_RANGE_BIN`. Active binary rows
# are still skipped (no reliable potency). Default keeps binary out of training.
INCLUDE_BINARY_IN_RANGE: Final[bool] = False
BINARY_INACTIVE_RANGE_BIN: Final[int] = 4  # >=10 µM
BINARY_RANGE_LOSS_WEIGHT: Final[float] = 1.0

# Feature-cache storage marker (separate from binder snapshots).
STORAGE_VERSION: Final[str] = "range_id_gather_v1"

# Default MLP schedule (mirrors binder defaults where sensible).
RANGE_MLP_DEFAULTS: Final[dict[str, object]] = {
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
    "es_metric": "macro_f1",
    "class_weights": True,
}

DEFAULT_RANGE_MODEL_PATH: Final[str] = (
    f"{root_config.MODELS_DIR}/mlp_range_ic50_ec50_morgan_descriptors.joblib"
)
