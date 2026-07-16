"""Train a single 50 nM binder / non-binder classifier.

Builds (or reuses) the cached feature matrix, applies an 80/20 cold-protein
split, trains the requested model (random forest by default, or an MLP), prints
classification metrics on the held-out test set, and saves the model as a
joblib file. The core routines are imported by ``grid_search.py``.
"""

from __future__ import annotations

import argparse
import os
from typing import Any, Optional

import numpy as np

import config
from src.data_prep import binarize_pactivity
from src.featurize import FeatureDataset, build_features, subset_to_memmap
from src.ligand_repr import canonical_ligand_repr
from src.metrics import print_metrics
from src.models import BaseRegressor, build_model
from src.splits import train_test_split


def collect_hyperparams(args: argparse.Namespace, model_type: str) -> dict[str, Any]:
    """Extract model hyperparameters from parsed CLI arguments.

    Args:
        args: Parsed argument namespace.
        model_type: Either ``"rf"`` or ``"mlp"``.

    Returns:
        A dictionary of hyperparameters for :func:`src.models.build_model`.
    """
    if model_type == "rf":
        return {
            "n_estimators": args.n_estimators,
            "batch_trees": args.rf_batch_trees,
            "shard_rows": args.rf_shard_rows,
            "max_depth": args.max_depth,
            "min_samples_leaf": args.min_samples_leaf,
            "n_jobs": args.n_jobs,
        }
    return {
        "hidden_dim": args.hidden_dim,
        "num_layers": args.num_layers,
        "dropout": args.dropout,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "patience": args.patience,
        "es_val_fraction": args.es_val_fraction,
        "use_batchnorm": args.batchnorm,
        "use_bilinear": args.bilinear,
        "bilinear_dim": args.bilinear_dim,
    }


def fit_on_indices(
    dataset: FeatureDataset,
    train_idx: np.ndarray,
    model_type: str,
    hyperparams: dict[str, Any],
    protein_model: str,
    ligand_model: str,
    seed: int = config.RANDOM_SEED,
    verbose: bool = True,
    tag: str = "train",
) -> BaseRegressor:
    """Materialize the training rows and fit a model on them.

    Labels from the feature cache are used as binder classes when
    ``dataset.labels_are_binary`` is set (default after rebuild); otherwise
    continuous pActivity is binarized at ``config.ACTIVITY_THRESHOLD_NM``.

    Args:
        dataset: The feature dataset.
        train_idx: Row indices to train on.
        model_type: Either ``"rf"`` or ``"mlp"``.
        hyperparams: Model hyperparameters.
        protein_model: Protein embedding model id (stored in metadata).
        ligand_model: Ligand embedding model id (stored in metadata).
        seed: Random seed.
        verbose: If ``True``, print progress.
        tag: Filename tag for the temporary split memmap.

    Returns:
        The fitted model with populated ``metadata``.
    """
    tmp_path = os.path.join(dataset.directory, f"split_{tag}.dat")
    X_train = subset_to_memmap(dataset, train_idx, tmp_path)
    y_raw = dataset.load_y()[train_idx]
    y_train = (
        y_raw
        if dataset.labels_are_binary
        else binarize_pactivity(y_raw, threshold_nm=dataset.activity_threshold_nm)
    )
    build_kwargs = dict(hyperparams)
    if model_type == "mlp":
        build_kwargs["protein_dim"] = dataset.protein_dim
        build_kwargs["ligand_dim"] = dataset.ligand_dim
    model = build_model(model_type, seed=seed, **build_kwargs)
    model.metadata = {
        "protein_model": protein_model,
        "ligand_model": canonical_ligand_repr(ligand_model),
        "protein_dim": dataset.protein_dim,
        "ligand_dim": dataset.ligand_dim,
        "n_features": dataset.n_features,
        "model_type": model_type,
        "hyperparams": hyperparams,
        "activity_threshold_nm": dataset.activity_threshold_nm,
        "task": "classification",
    }
    if verbose:
        pos_frac = float(np.mean(y_train))
        print(
            f"[train] fitting {model_type} on {len(train_idx)} rows "
            f"(active_frac={pos_frac:.3f} @ {dataset.activity_threshold_nm:g} nM)"
        )
    fit_kwargs: dict[str, Any] = {"verbose": verbose}
    if model_type == "mlp":
        # Align protein ids with the materialized train memmap for cold-protein ES.
        fit_kwargs["groups"] = dataset.load_groups()[train_idx]
    model.fit(X_train, y_train, **fit_kwargs)
    del X_train
    return model


def evaluate_on_indices(
    model: BaseRegressor,
    dataset: FeatureDataset,
    idx: np.ndarray,
    label: str,
    verbose: bool = True,
    tag: str = "eval",
) -> dict[str, float]:
    """Evaluate a fitted model on a subset of rows.

    Args:
        model: A fitted model.
        dataset: The feature dataset.
        idx: Row indices to evaluate on.
        label: Label used in the printed metrics line.
        verbose: If ``True``, print the metrics.
        tag: Filename tag for the temporary split memmap.

    Returns:
        The metric mapping (``accuracy``, ``precision``, ``recall``, ``f1``,
        ``auroc``, ``auprc``).
    """
    tmp_path = os.path.join(dataset.directory, f"split_{tag}.dat")
    X_eval = subset_to_memmap(dataset, idx, tmp_path)
    y_raw = dataset.load_y()[idx]
    y_eval = (
        y_raw
        if dataset.labels_are_binary
        else binarize_pactivity(y_raw, threshold_nm=dataset.activity_threshold_nm)
    )
    preds = model.predict(X_eval)
    del X_eval
    from src.metrics import compute_metrics

    scores = compute_metrics(y_eval, preds)
    if verbose:
        print_metrics(y_eval, preds, label=label)
    return scores


def run_training(args: argparse.Namespace) -> str:
    """Execute the full training workflow from parsed arguments.

    Args:
        args: Parsed CLI arguments.

    Returns:
        The path of the saved model file.
    """
    verbose = not args.quiet
    dataset = build_features(
        csv_path=args.csv,
        protein_model=args.protein_model,
        ligand_model=args.ligand_model,
        limit=args.limit,
        verbose=verbose,
        rebuild=args.rebuild_features,
        activity_threshold_nm=args.activity_threshold_nm,
    )
    groups = dataset.load_groups()
    split = train_test_split(
        groups, dataset.signature, test_fraction=args.test_fraction, seed=args.seed, verbose=verbose
    )

    hyperparams = collect_hyperparams(args, args.model)
    model = fit_on_indices(
        dataset,
        split["train"],
        args.model,
        hyperparams,
        args.protein_model,
        args.ligand_model,
        seed=args.seed,
        verbose=verbose,
    )
    print("\n[train] test-set performance:")
    evaluate_on_indices(model, dataset, split["test"], label="test", verbose=True)

    os.makedirs(config.MODELS_DIR, exist_ok=True)
    output = args.output or os.path.join(config.MODELS_DIR, f"{args.model}_model.joblib")
    model.save(output)
    print(f"\n[train] saved model to {output}")
    return output


def build_arg_parser() -> argparse.ArgumentParser:
    """Construct the command-line argument parser.

    Returns:
        The configured :class:`argparse.ArgumentParser`.
    """
    p = argparse.ArgumentParser(description="Train a 50 nM binder / non-binder classifier.")
    p.add_argument("--model", choices=["rf", "mlp"], default=config.DEFAULT_MODEL_TYPE)
    p.add_argument("--csv", default=config.TRAIN_CSV, help="Training CSV path.")
    p.add_argument("--limit", type=int, default=None, help="Cap on raw rows (smoke tests).")
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
    p.add_argument("--test-fraction", type=float, default=config.TEST_FRACTION)
    p.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    p.add_argument("--output", default=None, help="Output joblib path.")
    p.add_argument("--rebuild-features", action="store_true", help="Force feature rebuild.")
    p.add_argument(
        "--activity-threshold-nm",
        type=float,
        default=config.ACTIVITY_THRESHOLD_NM,
        help=(
            "Binder cutoff in nM for quantitative rows (default: 50). "
            "Papyrus Activity_class rows keep their explicit Activity Label."
        ),
    )
    p.add_argument("--quiet", action="store_true", help="Reduce progress output.")

    # Random-forest hyperparameters.
    p.add_argument("--n-estimators", type=int, default=config.RF_DEFAULTS["n_estimators"])
    p.add_argument("--rf-batch-trees", type=int, default=config.RF_DEFAULTS["batch_trees"])
    p.add_argument("--rf-shard-rows", type=int, default=config.RF_DEFAULTS["shard_rows"])
    p.add_argument("--max-depth", type=int, default=config.RF_DEFAULTS["max_depth"])
    p.add_argument("--min-samples-leaf", type=int, default=config.RF_DEFAULTS["min_samples_leaf"])
    p.add_argument("--n-jobs", type=int, default=config.RF_DEFAULTS["n_jobs"])

    # MLP hyperparameters.
    p.add_argument("--hidden-dim", type=int, default=int(config.MLP_DEFAULTS["hidden_dim"]))
    p.add_argument("--num-layers", type=int, default=int(config.MLP_DEFAULTS["num_layers"]))
    p.add_argument("--dropout", type=float, default=float(config.MLP_DEFAULTS["dropout"]))
    p.add_argument("--batch-size", type=int, default=int(config.MLP_DEFAULTS["batch_size"]))
    p.add_argument("--epochs", type=int, default=int(config.MLP_DEFAULTS["epochs"]))
    p.add_argument("--learning-rate", type=float, default=float(config.MLP_DEFAULTS["learning_rate"]))
    p.add_argument("--weight-decay", type=float, default=float(config.MLP_DEFAULTS["weight_decay"]))
    p.add_argument(
        "--patience",
        type=int,
        default=int(config.MLP_DEFAULTS["patience"]),
        help="MLP early-stopping patience in epochs (0 disables).",
    )
    p.add_argument(
        "--es-val-fraction",
        type=float,
        default=float(config.MLP_DEFAULTS["es_val_fraction"]),
        help="Target fraction of training rows for the cold-protein MLP ES holdout.",
    )
    p.add_argument(
        "--batchnorm",
        action="store_true",
        help="Insert BatchNorm1d after each MLP hidden layer.",
    )
    p.add_argument(
        "--bilinear",
        action="store_true",
        help="Append a learned bilinear protein/ligand interaction block.",
    )
    p.add_argument(
        "--bilinear-dim",
        type=int,
        default=int(config.MLP_DEFAULTS["bilinear_dim"]),
        help="Projection width for the bilinear protein/ligand interaction block.",
    )
    return p


def main(argv: Optional[list[str]] = None) -> None:
    """CLI entry point.

    Args:
        argv: Optional argument list (defaults to ``sys.argv``).

    Returns:
        None.
    """
    args = build_arg_parser().parse_args(argv)
    run_training(args)


if __name__ == "__main__":
    main()
