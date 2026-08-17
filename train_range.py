"""Train the IC50/EC50 multi-head potency-range classifier."""

from __future__ import annotations

import argparse
import os
from typing import Any

import numpy as np

import config
from src.range_model import config as range_config
from src.range_model.bins import BIN_LABELS, N_BINS, RANGE_EDGES_NM
from src.range_model.featurize import RangeFeatureDataset, build_range_features
from src.range_model.metrics import compute_range_metrics, print_range_metrics
from src.range_model.models import RangeMLP
from src.splits import SPLIT_STRATEGIES, get_or_create_nested_split


def collect_hyperparams(args: argparse.Namespace) -> dict[str, Any]:
    """Extract range-MLP hyperparameters from CLI arguments.

    Args:
        args: Parsed argument namespace.

    Returns:
        Keyword arguments for :class:`RangeMLP`.
    """
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
        "es_min_delta": args.es_min_delta,
        "class_weights": args.class_weights,
        "n_bins": N_BINS,
        "seed": args.seed,
    }


def _resolve_split(
    dataset: RangeFeatureDataset,
    args: argparse.Namespace,
    *,
    include_val: bool,
    verbose: bool,
) -> dict[str, np.ndarray]:
    """Build the requested train/val/test split for a range dataset.

    Args:
        dataset: Range feature dataset.
        args: Parsed CLI arguments.
        include_val: Whether a validation fold is required.
        verbose: Progress logging.

    Returns:
        Split mapping with at least ``train`` and ``test``.

    Raises:
        SystemExit: If a time split is requested without dated rows.
    """
    test_split = args.test_split
    validation_split = args.validation_split
    protein_groups = dataset.load_groups()
    needs_dc = test_split == "double-cold" or (
        include_val and validation_split == "double-cold"
    )
    needs_time = test_split in ("time", "time-protein") or (
        include_val and validation_split in ("time", "time-protein")
    )
    scaffold_groups = dataset.load_scaffold_groups() if needs_dc else None
    years = None
    if needs_time:
        try:
            years = dataset.load_years()
        except FileNotFoundError as exc:
            raise SystemExit(str(exc)) from exc
    try:
        return get_or_create_nested_split(
            protein_groups=protein_groups,
            signature=dataset.signature,
            test_split=test_split,
            validation_split=validation_split,
            val_fraction=float(args.val_fraction),
            test_fraction=float(args.test_fraction),
            include_val=include_val,
            scaffold_groups=scaffold_groups,
            years=years,
            seed=args.seed,
            splits_dir=dataset.split_directory,
            verbose=verbose,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc


def fit_on_indices(
    dataset: RangeFeatureDataset,
    train_idx: np.ndarray,
    hyperparams: dict[str, Any],
    *,
    protein_model: str,
    ligand_model: str,
    verbose: bool,
) -> RangeMLP:
    """Fit a range MLP on selected dataset rows.

    Args:
        dataset: Range feature snapshot.
        train_idx: Global row indices used for fitting.
        hyperparams: :class:`RangeMLP` constructor kwargs.
        protein_model: Protein embedding id (stored in metadata).
        ligand_model: Ligand representation spec (stored in metadata).
        verbose: Progress logging.

    Returns:
        Trained :class:`RangeMLP`.
    """
    view = dataset.feature_view(train_idx)
    y_bin = dataset.load_y_bin()[train_idx]
    head_idx = dataset.load_head_idx()[train_idx]
    loss_weight = dataset.load_loss_weight()[train_idx]
    groups = dataset.load_groups()[train_idx]
    scaffolds = dataset.load_scaffold_groups()[train_idx]
    years = dataset.load_years()[train_idx]
    model = RangeMLP(**hyperparams)
    model.fit(
        view,
        y_bin,
        head_idx,
        loss_weight=loss_weight,
        groups=groups,
        scaffold_groups=scaffolds,
        years=years,
        verbose=verbose,
    )
    model.metadata = {
        "task": "ic50_ec50_range",
        "protein_model": protein_model,
        "ligand_model": ligand_model,
        "n_bins": N_BINS,
        "bin_labels": list(BIN_LABELS),
        "range_edges_nm": list(RANGE_EDGES_NM),
        "heads": list(range_config.HEAD_NAMES),
        "include_binary": dataset.include_binary,
        "n_train_rows": int(train_idx.shape[0]),
        "signature": dataset.signature,
    }
    return model


def evaluate_on_indices(
    model: RangeMLP,
    dataset: RangeFeatureDataset,
    idx: np.ndarray,
    *,
    label: str,
) -> dict[str, Any]:
    """Evaluate range predictions on selected rows.

    Args:
        model: Trained range model.
        dataset: Range feature snapshot.
        idx: Global row indices to score.
        label: Print label.

    Returns:
        Metrics dictionary from :func:`compute_range_metrics`.
    """
    view = dataset.feature_view(idx)
    y_true = dataset.load_y_bin()[idx]
    head_idx = dataset.load_head_idx()[idx]
    y_pred = model.predict_bins(view, head_idx=head_idx)
    metrics = compute_range_metrics(y_true, y_pred, head_idx, n_bins=model.n_bins)
    print_range_metrics(metrics, label=label)
    return metrics


def run_training(args: argparse.Namespace) -> str:
    """Execute the range-model training workflow.

    Args:
        args: Parsed CLI arguments.

    Returns:
        Path of the saved model file.
    """
    verbose = not args.quiet
    dataset = build_range_features(
        csv_path=args.data,
        protein_model=args.protein_model,
        ligand_model=args.ligand_model,
        limit=args.limit,
        verbose=verbose,
        rebuild=args.rebuild_features,
        include_assay_context=args.include_assay_context,
        include_binary=args.include_binary,
        binary_inactive_bin=args.binary_inactive_bin,
        binary_loss_weight=args.binary_loss_weight,
    )
    split = _resolve_split(dataset, args, include_val=False, verbose=verbose)
    train_idx = split["train"]
    hyperparams = collect_hyperparams(args)
    model = fit_on_indices(
        dataset,
        train_idx,
        hyperparams,
        protein_model=args.protein_model,
        ligand_model=args.ligand_model,
        verbose=verbose,
    )
    if not args.skip_test:
        evaluate_on_indices(model, dataset, split["test"], label="test")
    else:
        print("\n[train-range] skipping full test eval (--skip-test)", flush=True)

    os.makedirs(config.MODELS_DIR, exist_ok=True)
    output = args.output or range_config.DEFAULT_RANGE_MODEL_PATH
    model.save(output)
    print(f"\n[train-range] saved model to {output}")
    return output


def build_arg_parser() -> argparse.ArgumentParser:
    """Construct the range-model CLI parser.

    Returns:
        Configured argument parser.
    """
    p = argparse.ArgumentParser(
        description=(
            "Train a multi-head IC50/EC50 potency-range classifier "
            f"({', '.join(BIN_LABELS)})."
        )
    )
    p.add_argument(
        "--data",
        default=config.TRAIN_CSV,
        help="Prepared training data path (CSV or Parquet).",
    )
    p.add_argument("--limit", type=int, default=None, help="Cap on raw rows (smoke tests).")
    p.add_argument("--protein-model", default=config.DEFAULT_PROTEIN_MODEL)
    p.add_argument(
        "--ligand-model",
        default=config.DEFAULT_LIGAND_MODEL,
        help="Ligand representation spec (default: morgan,descriptors).",
    )
    p.add_argument("--test-fraction", type=float, default=config.TEST_FRACTION)
    p.add_argument(
        "--val-fraction",
        type=float,
        default=config.GRID_VAL_FRACTION,
        help="Validation fraction when a nested val fold is kept (unused by default).",
    )
    p.add_argument(
        "--test-split",
        choices=sorted(SPLIT_STRATEGIES),
        default=config.DEFAULT_TEST_SPLIT,
    )
    p.add_argument(
        "--validation-split",
        choices=sorted(SPLIT_STRATEGIES),
        default=config.DEFAULT_VALIDATION_SPLIT,
        help="Strategy used only when a nested val fold is requested.",
    )
    p.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    p.add_argument(
        "--output",
        default=None,
        help=f"Model output path (default: {range_config.DEFAULT_RANGE_MODEL_PATH}).",
    )
    p.add_argument(
        "--rebuild-features",
        action="store_true",
        help="Rebuild the range feature snapshot even if a cache exists.",
    )
    p.add_argument(
        "--include-assay-context",
        action="store_true",
        help="Append assay one-hot, pH and temperature to features.",
    )
    p.add_argument(
        "--include-binary",
        action="store_true",
        help=(
            "Map inactive Papyrus binary rows into the weak potency bin "
            f"(default bin index {range_config.BINARY_INACTIVE_RANGE_BIN}: "
            f"{BIN_LABELS[range_config.BINARY_INACTIVE_RANGE_BIN]}). Off by default."
        ),
    )
    p.add_argument(
        "--binary-inactive-bin",
        type=int,
        default=range_config.BINARY_INACTIVE_RANGE_BIN,
        help="Bin index used for inactive binary rows when --include-binary is set.",
    )
    p.add_argument(
        "--binary-loss-weight",
        type=float,
        default=range_config.BINARY_RANGE_LOSS_WEIGHT,
        help="Per-row loss weight for binary-derived examples.",
    )
    p.add_argument("--skip-test", action="store_true", help="Skip held-out test evaluation.")
    p.add_argument("--quiet", action="store_true", help="Reduce progress output.")

    p.add_argument("--hidden-dim", type=int, default=int(range_config.RANGE_MLP_DEFAULTS["hidden_dim"]))
    p.add_argument("--num-layers", type=int, default=int(range_config.RANGE_MLP_DEFAULTS["num_layers"]))
    p.add_argument("--dropout", type=float, default=float(range_config.RANGE_MLP_DEFAULTS["dropout"]))
    p.add_argument("--batch-size", type=int, default=int(range_config.RANGE_MLP_DEFAULTS["batch_size"]))
    p.add_argument("--epochs", type=int, default=int(range_config.RANGE_MLP_DEFAULTS["epochs"]))
    p.add_argument(
        "--learning-rate",
        type=float,
        default=float(range_config.RANGE_MLP_DEFAULTS["learning_rate"]),
    )
    p.add_argument(
        "--weight-decay",
        type=float,
        default=float(range_config.RANGE_MLP_DEFAULTS["weight_decay"]),
    )
    p.add_argument(
        "--patience",
        type=int,
        default=int(range_config.RANGE_MLP_DEFAULTS["patience"]),
        help="Early-stopping patience in epochs (0 disables).",
    )
    p.add_argument(
        "--es-val-fraction",
        type=float,
        default=float(range_config.RANGE_MLP_DEFAULTS["es_val_fraction"]),
    )
    p.add_argument(
        "--es-min-delta",
        type=float,
        default=float(range_config.RANGE_MLP_DEFAULTS["es_min_delta"]),
    )
    class_group = p.add_mutually_exclusive_group()
    class_group.add_argument(
        "--class-weights",
        dest="class_weights",
        action="store_true",
        help="Use inverse-frequency class weights per head (default).",
    )
    class_group.add_argument(
        "--no-class-weights",
        dest="class_weights",
        action="store_false",
        help="Disable per-head class weights.",
    )
    p.set_defaults(class_weights=bool(range_config.RANGE_MLP_DEFAULTS["class_weights"]))
    return p


def main(argv: list[str] | None = None) -> None:
    """CLI entry point.

    Args:
        argv: Optional argument vector (defaults to ``sys.argv[1:]``).

    Returns:
        None.
    """
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    run_training(args)


if __name__ == "__main__":
    main()
