"""Grid-search multilabel MLP hyperparameters with nested outer folds.

Candidates are trained on the train split and scored on the validation split by
micro-AUPRC. The winner is evaluated on the held-out test set (optional per-label
spreadsheet via ``--label-metrics``). Isolated from the pair binder pipeline.
"""

from __future__ import annotations

import argparse
import os
from typing import Any, Optional

from src.multilabel import config as ml_config
from src.multilabel.featurize import build_ligand_features, default_paths_for_task
from src.multilabel.models import MultilabelMLP
from src.multilabel.splits import SPLIT_STRATEGIES
from src.multilabel.vocab import load_vocab
from train_multilabel import (
    _build_model_metadata,
    _resolve_split,
    evaluate_multilabel_on_indices,
    fit_multilabel_on_indices,
)

# MLP candidates; ``epochs`` is a ceiling (early stopping may halt sooner).
NN_GRID: list[dict[str, Any]] = [
    {"hidden_dim": 512, "num_layers": 2, "learning_rate": 1e-3},
    {"hidden_dim": 1024, "num_layers": 3, "learning_rate": 5e-4},
    {"hidden_dim": 1536, "num_layers": 4, "learning_rate": 3e-4},
]


def _merge_defaults(overrides: dict[str, Any]) -> dict[str, Any]:
    """Merge grid overrides onto multilabel MLP defaults.

    Args:
        overrides: Hyperparameters to override.

    Returns:
        Complete hyperparameter dictionary for :class:`MultilabelMLP`.
    """
    base: dict[str, Any] = {
        "hidden_dim": int(ml_config.MLP_DEFAULTS["hidden_dim"]),
        "num_layers": int(ml_config.MLP_DEFAULTS["num_layers"]),
        "dropout": float(ml_config.MLP_DEFAULTS["dropout"]),
        "batch_size": int(ml_config.MLP_DEFAULTS["batch_size"]),
        "epochs": int(ml_config.MLP_DEFAULTS["epochs"]),
        "learning_rate": float(ml_config.MLP_DEFAULTS["learning_rate"]),
        "weight_decay": float(ml_config.MLP_DEFAULTS["weight_decay"]),
        "patience": int(ml_config.MLP_DEFAULTS["patience"]),
        "es_val_fraction": float(ml_config.MLP_DEFAULTS["es_val_fraction"]),
        "es_min_delta": float(ml_config.MLP_DEFAULTS["es_min_delta"]),
        "class_weights": bool(ml_config.MLP_DEFAULTS["class_weights"]),
    }
    base.update(overrides)
    return base


def run_grid_search(args: argparse.Namespace) -> str:
    """Run the multilabel grid search and save the best model.

    Args:
        args: Parsed CLI arguments.

    Returns:
        Path of the saved best-model joblib file.

    Raises:
        SystemExit: If prepared data/vocab are missing.
    """
    default_data, default_vocab = default_paths_for_task(args.task)
    data_path = args.data or default_data
    vocab_path = args.vocab or default_vocab
    if not os.path.exists(data_path):
        raise SystemExit(
            f"Prepared data not found: {data_path}\n"
            "Run: python build_papyrus_multilabel.py"
        )
    if not os.path.exists(vocab_path):
        raise SystemExit(
            f"Vocabulary not found: {vocab_path}\n"
            "Run: python build_papyrus_multilabel.py"
        )

    verbose = not args.quiet
    dataset = build_ligand_features(
        data_path,
        vocab_path,
        args.task,
        ligand_model=args.ligand_model,
        activity_threshold_nm=args.threshold_nm,
        rebuild=args.rebuild_features,
        verbose=verbose,
    )
    split = _resolve_split(dataset, args, include_val=True)
    if "val" not in split:
        raise SystemExit("Grid search requires a validation fold (include_val=True).")
    vocab = load_vocab(vocab_path)

    best_model: Optional[MultilabelMLP] = None
    best_score = float("-inf")
    best_desc = ""

    for index, overrides in enumerate(NN_GRID):
        hyperparams = _merge_defaults(overrides)
        hyperparams["class_weights"] = bool(args.class_weights)
        desc = f"mlp {overrides}"
        print(f"\n[grid] ({index + 1}/{len(NN_GRID)}) training {desc}", flush=True)
        metadata = _build_model_metadata(
            dataset=dataset,
            vocab=vocab,
            hyperparams=hyperparams,
            args=args,
            vocab_path=vocab_path,
        )
        model = fit_multilabel_on_indices(
            dataset,
            split["train"],
            hyperparams,
            seed=args.seed,
            metadata=metadata,
            verbose=verbose,
        )
        scores = evaluate_multilabel_on_indices(
            model,
            dataset,
            split["val"],
            vocab,
            label=f"[grid] {desc} val",
            verbose=True,
        )
        val_score = float(scores["micro_auprc"])
        print(f"[grid] {desc} val_micro_auprc={val_score:.4f}", flush=True)
        if val_score > best_score:
            best_score = val_score
            best_model = model
            best_desc = desc

    assert best_model is not None, "Grid search produced no model."
    print(f"\n[grid] best (val micro_AUPRC={best_score:.4f}): {best_desc}", flush=True)
    print("[grid] best model test-set performance:", flush=True)
    evaluate_multilabel_on_indices(
        best_model,
        dataset,
        split["test"],
        vocab,
        label="test",
        verbose=True,
        label_metrics_path=args.label_metrics,
    )

    os.makedirs(ml_config.MODELS_DIR, exist_ok=True)
    output = args.output or os.path.join(
        ml_config.MODELS_DIR,
        (
            f"{args.task}_multilabel_grid"
            f"__test-{args.test_split}"
            f"__val-{args.validation_split}.joblib"
        ),
    )
    best_model.save(output)
    print(f"[grid] saved best model to {output}", flush=True)
    return output


def build_arg_parser() -> argparse.ArgumentParser:
    """Construct the multilabel grid-search CLI parser.

    Returns:
        Configured argument parser.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Grid-search ligand multilabel MLP hyperparameters. Selects by "
            "validation micro-AUPRC. Isolated from the pair binder pipeline."
        )
    )
    parser.add_argument(
        "--task",
        choices=["family", "target"],
        required=True,
        help="Which multilabel vocabulary / prepared table to train on.",
    )
    parser.add_argument(
        "--data",
        default=None,
        help="Ligand prepared Parquet (default depends on --task).",
    )
    parser.add_argument(
        "--vocab",
        default=None,
        help="Vocabulary JSON path (default depends on --task).",
    )
    parser.add_argument(
        "--ligand-model",
        default=ml_config.DEFAULT_LIGAND_MODEL,
        help="Ligand representation (same tokens as pair --ligand-model).",
    )
    parser.add_argument(
        "--rebuild-features",
        action="store_true",
        help="Force rebuild of the ligand feature snapshot.",
    )
    parser.add_argument(
        "--test-split",
        choices=list(SPLIT_STRATEGIES),
        default=ml_config.DEFAULT_TEST_SPLIT,
        help=(
            "Strategy for the held-out test fold: Murcko-scaffold cold "
            "(default) or publication-year time."
        ),
    )
    parser.add_argument(
        "--validation-split",
        choices=list(SPLIT_STRATEGIES),
        default=ml_config.DEFAULT_VALIDATION_SPLIT,
        help=(
            "Strategy for the validation fold used for candidate selection "
            "(default: scaffold)."
        ),
    )
    parser.add_argument(
        "--test-fraction",
        type=float,
        default=ml_config.TEST_FRACTION,
        help="Test fraction of all ligands.",
    )
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=ml_config.VAL_FRACTION,
        help="Validation fraction of all ligands.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=ml_config.RANDOM_SEED,
        help="Random seed.",
    )
    parser.add_argument(
        "--threshold-nm",
        type=float,
        default=ml_config.ACTIVITY_THRESHOLD_NM,
        help="Activity threshold recorded in feature metadata.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output joblib path for the best model.",
    )
    parser.add_argument(
        "--label-metrics",
        default=None,
        metavar="PATH",
        help=(
            "Optional spreadsheet path for per-label test metrics of the "
            "winning model (.xlsx/.csv/.tsv)."
        ),
    )
    parser.add_argument(
        "--class-weights",
        action=argparse.BooleanOptionalAction,
        default=bool(ml_config.MLP_DEFAULTS["class_weights"]),
        help="Per-label BCE pos_weight = n_neg/n_pos (default on).",
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress progress.")
    return parser


def main(argv: Optional[list[str]] = None) -> None:
    """CLI entry point.

    Args:
        argv: Optional argument list.

    Returns:
        None.
    """
    args = build_arg_parser().parse_args(argv)
    run_grid_search(args)


if __name__ == "__main__":
    main()
