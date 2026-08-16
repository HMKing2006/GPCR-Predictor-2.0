"""Search model types and hyperparameters with nested outer folds.

Each candidate is trained on the train split and scored on the validation split,
with metrics printed as soon as the candidate finishes. The candidate with the
highest validation macro per-target AUROC is finally evaluated on the held-out
test split (with protein/scaffold novelty breakouts) and saved. Use
``--test-split`` / ``--validation-split`` (defaults: time test, time-protein
val); e.g. ``--test-split time --validation-split protein`` to select for
protein transfer only.
"""

from __future__ import annotations

import argparse
import os
from typing import Any, Optional

import config
from src.featurize import build_features
from src.models import BaseRegressor
from src.splits import SPLIT_STRATEGIES
from src.target_balance import TARGET_BALANCE_MODES, TARGET_BALANCE_RATIOS, TARGET_BCE_REDUCTIONS
from train import _resolve_split, evaluate_on_indices, fit_on_indices

# Every entry inherits MLP defaults including early stopping; ``epochs`` is a
# ceiling, not the actual run length.
NN_GRID: list[dict[str, Any]] = [
    {"hidden_dim": 512, "num_layers": 2, "learning_rate": 1e-3},
    {"hidden_dim": 1024, "num_layers": 3, "learning_rate": 5e-4},
    {"hidden_dim": 1024, "num_layers": 3, "learning_rate": 5e-4, "dropout": 0.2},
    {"hidden_dim": 1536, "num_layers": 4, "learning_rate": 3e-4, "dropout": 0.2, "weight_decay": 0.0},
    {"hidden_dim": 1024, "num_layers": 3, "learning_rate": 5e-4, "use_batchnorm": True},
    # {"hidden_dim": 1024, "num_layers": 3, "learning_rate": 5e-4, "use_bilinear": True, "bilinear_dim": 256},
    # {"hidden_dim": 1024, "num_layers": 3, "learning_rate": 3e-4, "dropout": 0.2, "use_bilinear": True, "bilinear_dim": 512},
]

# Optional random-forest baselines, enabled with ``--include-rf``.
RF_GRID: list[dict[str, Any]] = [
    {"n_estimators": 200, "max_depth": 0},
    {"n_estimators": 400, "max_depth": 30},
]


def build_grid(include_rf: bool) -> list[tuple[str, dict[str, Any]]]:
    """Assemble the ``(model_type, overrides)`` candidate list.

    Args:
        include_rf: If ``True``, prepend the random-forest baselines.

    Returns:
        The ordered list of candidates to train and score.
    """
    grid: list[tuple[str, dict[str, Any]]] = []
    if include_rf:
        grid += [("rf", overrides) for overrides in RF_GRID]
    grid += [("mlp", overrides) for overrides in NN_GRID]
    return grid


def _merge_defaults(model_type: str, overrides: dict[str, Any]) -> dict[str, Any]:
    """Merge grid overrides onto the model's default hyperparameters.

    Args:
        model_type: Either ``"rf"`` or ``"mlp"``.
        overrides: Hyperparameters to override.

    Returns:
        A complete hyperparameter dictionary for :func:`build_model`.
    """
    if model_type == "rf":
        base: dict[str, Any] = {
            "n_estimators": config.RF_DEFAULTS["n_estimators"],
            "batch_trees": config.RF_DEFAULTS["batch_trees"],
            "shard_rows": config.RF_DEFAULTS["shard_rows"],
            "max_depth": config.RF_DEFAULTS["max_depth"],
            "min_samples_leaf": config.RF_DEFAULTS["min_samples_leaf"],
            "n_jobs": config.RF_DEFAULTS["n_jobs"],
        }
    else:
        base = {
            "hidden_dim": int(config.MLP_DEFAULTS["hidden_dim"]),
            "num_layers": int(config.MLP_DEFAULTS["num_layers"]),
            "dropout": float(config.MLP_DEFAULTS["dropout"]),
            "batch_size": int(config.MLP_DEFAULTS["batch_size"]),
            "epochs": int(config.MLP_DEFAULTS["epochs"]),
            "learning_rate": float(config.MLP_DEFAULTS["learning_rate"]),
            "weight_decay": float(config.MLP_DEFAULTS["weight_decay"]),
            "patience": int(config.MLP_DEFAULTS["patience"]),
            "es_val_fraction": float(config.MLP_DEFAULTS["es_val_fraction"]),
            "es_metric": str(config.MLP_DEFAULTS["es_metric"]),
            "class_weights": bool(config.MLP_DEFAULTS["class_weights"]),
            "target_balance": str(config.MLP_DEFAULTS["target_balance"]),
            "target_balance_ratio": str(config.MLP_DEFAULTS["target_balance_ratio"]),
            "target_bce_reduction": str(config.MLP_DEFAULTS["target_bce_reduction"]),
            "rank_loss_weight": float(config.MLP_DEFAULTS["rank_loss_weight"]),
            "rank_targets_per_batch": int(config.MLP_DEFAULTS["rank_targets_per_batch"]),
            "rank_samples_per_class": int(config.MLP_DEFAULTS["rank_samples_per_class"]),
            "target_size_exponent": float(config.MLP_DEFAULTS["target_size_exponent"]),
            "use_batchnorm": bool(config.MLP_DEFAULTS["use_batchnorm"]),
            "use_bilinear": bool(config.MLP_DEFAULTS["use_bilinear"]),
            "bilinear_dim": int(config.MLP_DEFAULTS["bilinear_dim"]),
        }
    base.update(overrides)
    return base


def run_grid_search(args: argparse.Namespace) -> str:
    """Run the grid search and save the best model by validation macro AUROC.

    Args:
        args: Parsed CLI arguments.

    Returns:
        The path of the saved best-model file.
    """
    verbose = not args.quiet
    dataset = build_features(
        csv_path=args.data,
        protein_model=args.protein_model,
        ligand_model=args.ligand_model,
        limit=args.limit,
        verbose=verbose,
        rebuild=args.rebuild_features,
        activity_threshold_nm=args.activity_threshold_nm,
        include_assay_context=args.include_assay_context,
    )
    split = _resolve_split(dataset, args, include_val=True, verbose=verbose)

    grid = build_grid(args.include_rf)
    best_model: Optional[BaseRegressor] = None
    best_score = float("-inf")
    best_desc = ""

    for index, (model_type, overrides) in enumerate(grid):
        hyperparams = _merge_defaults(model_type, overrides)
        if model_type == "mlp":
            hyperparams["class_weights"] = bool(args.class_weights)
            hyperparams["target_balance"] = str(args.target_balance)
            hyperparams["target_balance_ratio"] = str(args.target_balance_ratio)
            hyperparams["target_bce_reduction"] = str(args.target_bce_reduction)
            hyperparams["rank_loss_weight"] = float(args.rank_loss_weight)
            hyperparams["rank_targets_per_batch"] = int(args.rank_targets_per_batch)
            hyperparams["rank_samples_per_class"] = int(args.rank_samples_per_class)
            hyperparams["target_size_exponent"] = float(args.target_size_exponent)
        desc = f"{model_type} {overrides}"
        print(f"\n[grid] ({index + 1}/{len(grid)}) training {desc}")
        model = fit_on_indices(
            dataset,
            split["train"],
            model_type,
            hyperparams,
            args.protein_model,
            args.ligand_model,
            seed=args.seed,
            verbose=verbose,
            tag="grid_train",
        )
        scores = evaluate_on_indices(
            model, dataset, split["val"], label=f"[grid] {desc} val", verbose=True, tag="grid_val"
        )
        # Prefer macro per-target AUROC; fall back to pooled when undefined.
        if scores.get("macro_auroc_n_evaluable", 0) > 0:
            score = float(scores["macro_auroc"])
            score_name = "macroAUROC"
        else:
            score = float(scores["auroc"])
            score_name = "AUROC"
            print(
                f"[grid] {desc}: no evaluable two-class val targets; "
                "using pooled AUROC for selection",
                flush=True,
            )
        print(
            f"[grid] {desc}: select_score={score:.4f} ({score_name}) "
            f"pooled_AUROC={scores['auroc']:.4f}",
            flush=True,
        )
        if score > best_score:
            best_score = score
            best_model = model
            best_desc = desc

    assert best_model is not None, "Grid search produced no model."
    print(f"\n[grid] best (val macroAUROC/AUROC={best_score:.4f}): {best_desc}")
    print("[grid] best model test-set performance:")
    evaluate_on_indices(
        best_model,
        dataset,
        split["test"],
        label="test",
        verbose=True,
        tag="grid_test",
        train_idx=split["train"],
    )

    os.makedirs(config.MODELS_DIR, exist_ok=True)
    output = args.output or os.path.join(config.MODELS_DIR, "best_model.joblib")
    best_model.save(output)
    print(f"[grid] saved best model to {output}")
    return output


def build_arg_parser() -> argparse.ArgumentParser:
    """Construct the command-line argument parser.

    Returns:
        The configured :class:`argparse.ArgumentParser`.
    """
    p = argparse.ArgumentParser(description="Grid-search binder classifiers and hyperparameters.")
    p.add_argument(
        "--data",
        default=config.TRAIN_CSV,
        help="Prepared training data path (CSV or Parquet).",
    )
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--protein-model", default=config.DEFAULT_PROTEIN_MODEL)
    p.add_argument(
        "--ligand-model",
        default=config.DEFAULT_LIGAND_MODEL,
        help=(
            "Ligand representation: HF model id, reserved token "
            "(morgan, avalon, descriptors, molformer), or a comma-separated "
            "combination (e.g. morgan,avalon,molformer)."
        ),
    )
    p.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    p.add_argument(
        "--test-split",
        choices=list(SPLIT_STRATEGIES),
        default=config.DEFAULT_TEST_SPLIT,
        help=(
            "Strategy for the held-out test fold (default: time): cold-protein, "
            "double-cold protein+scaffold, publication-year time, or "
            "time-protein (latest year of unseen proteins)."
        ),
    )
    p.add_argument(
        "--validation-split",
        choices=list(SPLIT_STRATEGIES),
        default=config.DEFAULT_VALIDATION_SPLIT,
        help=(
            "Strategy for the validation fold (default: time-protein). Test is "
            "carved first; val is carved from the remainder."
        ),
    )
    p.add_argument(
        "--val-fraction",
        type=float,
        default=config.GRID_VAL_FRACTION,
        help="Validation fraction of all rows (strategy-dependent assignment).",
    )
    p.add_argument(
        "--test-fraction",
        type=float,
        default=config.GRID_TEST_FRACTION,
        help="Test fraction of all rows (strategy-dependent assignment).",
    )
    p.add_argument("--output", default=None, help="Output joblib path for the best model.")
    p.add_argument(
        "--rebuild-features",
        action="store_true",
        help=(
            "Replace this dataset's row snapshot and local embedding matrices "
            "(global LMDB embeddings are retained)."
        ),
    )
    p.add_argument(
        "--activity-threshold-nm",
        type=float,
        default=config.ACTIVITY_THRESHOLD_NM,
        help=(
            "Binder cutoff in nM for quantitative rows (default: 50). "
            "Papyrus Activity_class rows keep their explicit Activity Label."
        ),
    )
    p.add_argument(
        "--include-assay-context",
        action="store_true",
        default=config.INCLUDE_ASSAY_CONTEXT,
        help=(
            "Append assay type one-hot, pH, and temperature to feature rows "
            "(off by default; protein+ligand only)."
        ),
    )
    p.add_argument(
        "--include-rf",
        action="store_true",
        help="Also evaluate the random-forest baselines (off by default).",
    )
    p.add_argument(
        "--class-weights",
        action=argparse.BooleanOptionalAction,
        default=bool(config.MLP_DEFAULTS["class_weights"]),
        help=(
            "Weight MLP BCE positives by n_neg/n_pos on the fit rows "
            f"(default: {'on' if config.MLP_DEFAULTS['class_weights'] else 'off'}). "
            "Incompatible with --target-balance other than none."
        ),
    )
    p.add_argument(
        "--target-balance",
        choices=list(TARGET_BALANCE_MODES),
        default=str(config.MLP_DEFAULTS["target_balance"]),
        help=(
            "Per-target MLP balancing: none (default), weights, downsample, "
            "or upsample. Excludes single-class targets."
        ),
    )
    p.add_argument(
        "--target-balance-ratio",
        choices=list(TARGET_BALANCE_RATIOS),
        default=str(config.MLP_DEFAULTS["target_balance_ratio"]),
        help=(
            "Active-fraction goal for --target-balance: equal (0.5, default) "
            "or dataset (fit-set mean prevalence)."
        ),
    )
    p.add_argument(
        "--target-bce-reduction",
        choices=list(TARGET_BCE_REDUCTIONS),
        default=str(config.MLP_DEFAULTS["target_bce_reduction"]),
        help=(
            "MLP BCE reduction: pooled (default) or mean (stratified "
            "per-target BCE). "
        ),
    )
    p.add_argument(
        "--rank-loss-weight",
        type=float,
        default=float(config.MLP_DEFAULTS["rank_loss_weight"]),
        help=(
            "Weight on within-target RankNet (default 0). Requires "
            "--target-bce-reduction mean."
        ),
    )
    p.add_argument(
        "--rank-targets-per-batch",
        type=int,
        default=int(config.MLP_DEFAULTS["rank_targets_per_batch"]),
        help="Proteins T per stratified batch when reduction is mean.",
    )
    p.add_argument(
        "--rank-samples-per-class",
        type=int,
        default=int(config.MLP_DEFAULTS["rank_samples_per_class"]),
        help="Positives and negatives k per protein in a stratified batch.",
    )
    p.add_argument(
        "--target-size-exponent",
        type=float,
        default=float(config.MLP_DEFAULTS["target_size_exponent"]),
        help="Scale row weights by 1/n_t**alpha after class balancing (0=off).",
    )
    p.add_argument("--quiet", action="store_true")
    return p


def main(argv: Optional[list[str]] = None) -> None:
    """CLI entry point.

    Args:
        argv: Optional argument list (defaults to ``sys.argv``).

    Returns:
        None.
    """
    args = build_arg_parser().parse_args(argv)
    if args.target_balance != "none" and args.class_weights:
        raise SystemExit(
            "Cannot combine --class-weights with "
            f"--target-balance={args.target_balance}. "
            "Use --no-class-weights or --target-balance none."
        )
    if args.rank_loss_weight < 0.0:
        raise SystemExit("--rank-loss-weight must be >= 0.")
    if args.target_size_exponent < 0.0:
        raise SystemExit("--target-size-exponent must be >= 0.")
    if args.rank_loss_weight > 0.0 and args.target_bce_reduction != "mean":
        raise SystemExit(
            "--rank-loss-weight > 0 requires --target-bce-reduction mean."
        )
    if args.rank_targets_per_batch < 1 or args.rank_samples_per_class < 1:
        raise SystemExit(
            "--rank-targets-per-batch and --rank-samples-per-class must be >= 1."
        )
    run_grid_search(args)


if __name__ == "__main__":
    main()
