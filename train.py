"""Train a single binder / non-binder classifier.

Builds (or reuses) a compact feature snapshot, applies nested outer folds via
``--test-split`` / ``--validation-split`` (defaults: time test, time-protein
val), trains the requested model, prints classification metrics and novelty
breakouts on the held-out test set, and saves the model as a joblib file. The
core routines are imported by ``grid_search.py``.
"""

from __future__ import annotations

import argparse
import os
from typing import Any, Optional

import numpy as np

import config
from src.data_prep import binarize_pactivity
from src.featurize import FeatureDataset, build_features
from src.ligand_repr import canonical_ligand_repr
from src.metrics import print_breakdowns, print_metrics
from src.models import BaseRegressor, build_model
from src.splits import SPLIT_STRATEGIES, get_or_create_nested_split
from src.target_balance import TARGET_BALANCE_MODES, TARGET_BALANCE_RATIOS, TARGET_BCE_REDUCTIONS


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
        "es_metric": args.es_metric,
        "class_weights": args.class_weights,
        "target_balance": args.target_balance,
        "target_balance_ratio": args.target_balance_ratio,
        "target_bce_reduction": args.target_bce_reduction,
        "rank_loss_weight": args.rank_loss_weight,
        "listwise_loss_weight": args.listwise_loss_weight,
        "rank_targets_per_batch": args.rank_targets_per_batch,
        "rank_samples_per_class": args.rank_samples_per_class,
        "target_size_exponent": args.target_size_exponent,
        "use_batchnorm": args.batchnorm,
        "use_bilinear": args.bilinear,
        "bilinear_dim": args.bilinear_dim,
        "use_film": args.film,
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
    """Fit a model through an on-demand view of the selected training rows.

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
        tag: Legacy progress tag retained for caller compatibility.

    Returns:
        The fitted model with populated ``metadata``.
    """
    del tag
    X_train = dataset.feature_view(train_idx)
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
        "include_assay_context": dataset.include_assay_context,
        "feature_storage_version": "id_gather_v1",
        "feature_signature": dataset.signature,
        "feature_directory": dataset.directory,
        "protein_embedding_alias": dataset.protein_alias,
        "ligand_embedding_aliases": list(dataset.ligand_aliases),
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
        # Align protein/scaffold/year ids with the selected FeatureView for ES.
        fit_kwargs["groups"] = dataset.load_groups()[train_idx]
        fit_kwargs["scaffold_groups"] = dataset.load_scaffold_groups()[train_idx]
        fit_kwargs["years"] = dataset.load_years()[train_idx]
    model.fit(X_train, y_train, **fit_kwargs)
    if model_type == "mlp" and getattr(model, "balance_summary", None) is not None:
        model.metadata["balance_summary"] = model.balance_summary
    del X_train
    return model


def evaluate_on_indices(
    model: BaseRegressor,
    dataset: FeatureDataset,
    idx: np.ndarray,
    label: str,
    verbose: bool = True,
    tag: str = "eval",
    train_idx: Optional[np.ndarray] = None,
) -> dict[str, float]:
    """Evaluate a fitted model through an on-demand row view.

    Args:
        model: A fitted model.
        dataset: The feature dataset.
        idx: Row indices to evaluate on.
        label: Label used in the printed metrics line.
        verbose: If ``True``, print the metrics.
        tag: Legacy progress tag retained for caller compatibility.
        train_idx: Optional training indices used to compute known/unknown
            protein and scaffold breakouts relative to train.

    Returns:
        The metric mapping (``accuracy``, ``precision``, ``recall``, ``f1``,
        ``auroc``, ``auprc``, plus macro per-target ``auroc`` / ``auprc`` /
        ``precision`` / ``recall`` / ``f1`` and evaluable/skipped counts when
        protein ids are available).
    """
    del tag
    X_eval = dataset.feature_view(idx)
    y_raw = dataset.load_y()[idx]
    y_eval = (
        y_raw
        if dataset.labels_are_binary
        else binarize_pactivity(y_raw, threshold_nm=dataset.activity_threshold_nm)
    )
    preds = model.predict(X_eval)
    del X_eval
    from src.metrics import compute_metrics

    eval_proteins = dataset.load_groups()[idx]
    scores = compute_metrics(y_eval, preds, protein_ids=eval_proteins)
    if verbose:
        print_metrics(y_eval, preds, label=label, protein_ids=eval_proteins)
        if train_idx is not None:
            train_proteins = dataset.load_groups()[train_idx]
            train_scaffolds = dataset.load_scaffold_groups()[train_idx]
            eval_scaffolds = dataset.load_scaffold_groups()[idx]
            protein_known = np.isin(eval_proteins, train_proteins)
            scaffold_known = np.isin(eval_scaffolds, train_scaffolds)
            print_breakdowns(
                y_eval,
                preds,
                protein_known,
                scaffold_known,
                label=label,
                protein_ids=eval_proteins,
            )
    return scores


def _resolve_split(
    dataset: FeatureDataset,
    args: argparse.Namespace,
    *,
    include_val: bool,
    verbose: bool,
) -> dict[str, np.ndarray]:
    """Build the requested train/val/test split for a dataset.

    Test is carved with ``args.test_split`` first; when ``include_val`` is
    ``True``, validation is carved from the remainder with
    ``args.validation_split`` (defaulting to the same strategy as test).

    Args:
        dataset: Feature dataset with group / year arrays on disk.
        args: Parsed CLI arguments (expects ``test_split``, ``validation_split``,
            fractions, seed).
        include_val: Whether a validation fold is required. When ``False``,
            ``validation_split`` is ignored and all non-test rows go to train.
        verbose: Progress logging.

    Returns:
        Split mapping with at least ``train`` and ``test``.

    Raises:
        SystemExit: If a time split is requested without dated rows.
    """
    test_split = getattr(args, "test_split", config.DEFAULT_TEST_SPLIT)
    validation_split = getattr(
        args, "validation_split", config.DEFAULT_VALIDATION_SPLIT
    )

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


def _cap_rows_per_protein(
    indices: np.ndarray,
    protein_ids: np.ndarray,
    max_rows: int,
    seed: int,
) -> np.ndarray:
    """Downsample training indices to at most ``max_rows`` per protein.

    Args:
        indices: Global dataset row indices.
        protein_ids: Protein id for every dataset row (full length).
        max_rows: Maximum rows retained for each protein.
        seed: RNG seed for within-protein sampling.

    Returns:
        Subsampled index array (sorted).
    """
    if max_rows < 1:
        raise ValueError(f"max_rows must be >= 1; got {max_rows}.")
    rng = np.random.default_rng(seed)
    groups = protein_ids[indices]
    order = np.argsort(groups, kind="mergesort")
    idx_sorted = indices[order]
    g_sorted = groups[order]
    uniq, starts = np.unique(g_sorted, return_index=True)
    ends = list(starts[1:]) + [int(g_sorted.shape[0])]
    kept: list[np.ndarray] = []
    for start, end in zip(starts.tolist(), ends):
        block = idx_sorted[start:end]
        if block.size <= max_rows:
            kept.append(block)
        else:
            kept.append(rng.choice(block, size=max_rows, replace=False))
    out = np.concatenate(kept) if kept else np.empty(0, dtype=np.int64)
    return np.sort(out.astype(np.int64, copy=False))


def _downsample_high_prevalence(
    indices: np.ndarray,
    protein_ids: np.ndarray,
    labels: np.ndarray,
    prev_cut: float,
    keep_frac: float,
    seed: int,
) -> tuple[np.ndarray, int]:
    """Downsample rows from high-prevalence two-class proteins.

    Targets with empirical active fraction ``> prev_cut`` keep only
    ``keep_frac`` of their rows (stratified by label when possible). Other
    targets are kept in full. This shifts training mass toward mid/low
    prevalence proteins that dominate the unknown-protein precision gap.

    Args:
        indices: Global training row indices.
        protein_ids: Full-length protein id vector.
        labels: Full-length binary labels.
        prev_cut: Prevalence above which a two-class target is downsampled.
        keep_frac: Fraction of rows to keep for high-prevalence targets.
        seed: RNG seed.

    Returns:
        Tuple ``(filtered_indices, n_high_prev_targets_thinned)``.

    Raises:
        ValueError: If ``prev_cut`` / ``keep_frac`` are out of range.
    """
    if not (0.0 < prev_cut < 1.0):
        raise ValueError(f"prev_cut must be in (0, 1); got {prev_cut}.")
    if not (0.0 < keep_frac <= 1.0):
        raise ValueError(f"keep_frac must be in (0, 1]; got {keep_frac}.")
    rng = np.random.default_rng(seed)
    groups = protein_ids[indices]
    y = labels[indices]
    order = np.argsort(groups, kind="mergesort")
    idx_sorted = indices[order]
    g_sorted = groups[order]
    y_sorted = y[order]
    _, starts = np.unique(g_sorted, return_index=True)
    ends = list(starts[1:]) + [int(g_sorted.shape[0])]
    kept: list[np.ndarray] = []
    n_hi = 0
    for start, end in zip(starts.tolist(), ends):
        block = idx_sorted[start:end]
        yb = y_sorted[start:end]
        if yb.min() == yb.max():
            kept.append(block)
            continue
        prev = float(yb.mean())
        if prev <= prev_cut:
            kept.append(block)
            continue
        n_hi += 1
        n_keep = max(2, int(round(block.size * keep_frac)))
        n_keep = min(n_keep, block.size)
        # Stratify keep by label when both classes remain large enough.
        pos = block[yb > 0.5]
        neg = block[yb <= 0.5]
        n_pos = max(1, int(round(n_keep * prev)))
        n_pos = min(n_pos, pos.size)
        n_neg = min(n_keep - n_pos, neg.size)
        if n_pos + n_neg < n_keep:
            n_pos = min(pos.size, n_keep - n_neg)
        chosen = np.concatenate(
            [
                rng.choice(pos, size=n_pos, replace=False),
                rng.choice(neg, size=n_neg, replace=False),
            ]
        )
        kept.append(chosen)
    out = np.concatenate(kept) if kept else np.empty(0, dtype=np.int64)
    return np.sort(out.astype(np.int64, copy=False)), n_hi


def run_training(args: argparse.Namespace) -> str:
    """Execute the full training workflow from parsed arguments.

    Args:
        args: Parsed CLI arguments.

    Returns:
        The path of the saved model file.
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
    split = _resolve_split(dataset, args, include_val=False, verbose=verbose)
    train_idx = split["train"]
    if args.max_rows_per_protein is not None:
        before = int(train_idx.shape[0])
        train_idx = _cap_rows_per_protein(
            train_idx,
            dataset.load_groups(),
            int(args.max_rows_per_protein),
            seed=int(args.seed),
        )
        if verbose:
            print(
                f"[train] capped rows/protein={args.max_rows_per_protein}: "
                f"{before:,} → {train_idx.shape[0]:,} train rows",
                flush=True,
            )
    if args.high_prev_keep_frac < 1.0:
        before = int(train_idx.shape[0])
        y_all = dataset.load_y()
        if not dataset.labels_are_binary:
            y_all = binarize_pactivity(
                y_all, threshold_nm=dataset.activity_threshold_nm
            )
        train_idx, n_hi = _downsample_high_prevalence(
            train_idx,
            dataset.load_groups(),
            y_all,
            prev_cut=float(args.high_prev_cut),
            keep_frac=float(args.high_prev_keep_frac),
            seed=int(args.seed),
        )
        if verbose:
            print(
                f"[train] high-prev downsample cut={args.high_prev_cut} "
                f"keep_frac={args.high_prev_keep_frac}: "
                f"{before:,} → {train_idx.shape[0]:,} rows "
                f"({n_hi} high-prev targets thinned)",
                flush=True,
            )

    hyperparams = collect_hyperparams(args, args.model)
    model = fit_on_indices(
        dataset,
        train_idx,
        args.model,
        hyperparams,
        args.protein_model,
        args.ligand_model,
        seed=args.seed,
        verbose=verbose,
    )
    if not args.skip_test:
        print("\n[train] test-set performance:")
        evaluate_on_indices(
            model,
            dataset,
            split["test"],
            label="test",
            verbose=True,
            train_idx=train_idx,
        )
    else:
        print("\n[train] skipping full test eval (--skip-test)", flush=True)

    os.makedirs(config.MODELS_DIR, exist_ok=True)
    output = args.output or config.DEFAULT_PAIR_MODEL_PATH
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
        help=(
            "Ligand representation: HF model id, reserved token "
            "(morgan, avalon, descriptors, molformer), or a comma-separated "
            "combination (e.g. morgan,avalon,molformer)."
        ),
    )
    p.add_argument("--test-fraction", type=float, default=config.TEST_FRACTION)
    p.add_argument(
        "--val-fraction",
        type=float,
        default=config.GRID_VAL_FRACTION,
        help=(
            "Validation fraction of all rows (grid search). For time validation "
            "this sizes the year window; time-protein uses a ~1-year cold "
            "window instead. In train.py the val fold is merged into train "
            "unless grid search is used."
        ),
    )
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
            "carved first; val is carved from the remainder when a val fold is "
            "used. Example: --test-split time --validation-split protein."
        ),
    )
    p.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    p.add_argument(
        "--output",
        default=None,
        help=f"Output joblib path (default: {config.DEFAULT_PAIR_MODEL_PATH}).",
    )
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
        help="Minimum-size bar for the MLP ES holdout (~0.25 of this fraction of train rows).",
    )
    p.add_argument(
        "--es-metric",
        choices=["auroc", "auprc", "precision", "recall", "f1"],
        default=str(config.MLP_DEFAULTS["es_metric"]),
        help="Macro per-target metric used for MLP early stopping.",
    )
    p.add_argument(
        "--class-weights",
        action=argparse.BooleanOptionalAction,
        default=bool(config.MLP_DEFAULTS["class_weights"]),
        help=(
            "Weight BCE positives by n_neg/n_pos on the fit rows "
            f"(default: {'on' if config.MLP_DEFAULTS['class_weights'] else 'off'}). "
            "Incompatible with --target-balance other than none. "
            "Use --class-weights / --no-class-weights to override."
        ),
    )
    p.add_argument(
        "--target-balance",
        choices=list(TARGET_BALANCE_MODES),
        default=str(config.MLP_DEFAULTS["target_balance"]),
        help=(
            "Per-target MLP balancing: none (default), weights (per-target "
            "pos weight), downsample majority, or upsample minority. "
            "Excludes single-class targets and prints kept/excluded counts."
        ),
    )
    p.add_argument(
        "--target-balance-ratio",
        default=str(config.MLP_DEFAULTS["target_balance_ratio"]),
        help=(
            "Active-fraction goal for --target-balance: equal (0.5), "
            "dataset (fit-set mean prevalence), or a float in (0, 1). "
            "Applies to weights, downsample, and upsample."
        ),
    )
    p.add_argument(
        "--target-bce-reduction",
        choices=list(TARGET_BCE_REDUCTIONS),
        default=str(config.MLP_DEFAULTS["target_bce_reduction"]),
        help=(
            "MLP BCE reduction: pooled (default random mixed-target batches) "
            "or mean (stratified batches; average of per-target BCEs)."
        ),
    )
    p.add_argument(
        "--rank-loss-weight",
        type=float,
        default=float(config.MLP_DEFAULTS["rank_loss_weight"]),
        help=(
            "Weight on within-target RankNet added to mean-target BCE "
            "(default 0). Requires --target-bce-reduction mean."
        ),
    )
    p.add_argument(
        "--listwise-loss-weight",
        type=float,
        default=float(config.MLP_DEFAULTS["listwise_loss_weight"]),
        help=(
            "Weight on within-target ListNet CE added to mean-target BCE "
            "(default 0). Requires --target-bce-reduction mean."
        ),
    )
    p.add_argument(
        "--rank-targets-per-batch",
        type=int,
        default=int(config.MLP_DEFAULTS["rank_targets_per_batch"]),
        help="Proteins T per stratified batch when --target-bce-reduction mean.",
    )
    p.add_argument(
        "--rank-samples-per-class",
        type=int,
        default=int(config.MLP_DEFAULTS["rank_samples_per_class"]),
        help=(
            "Positives and negatives k drawn per protein in a stratified batch "
            "(effective batch size = T * 2 * k)."
        ),
    )
    p.add_argument(
        "--target-size-exponent",
        type=float,
        default=float(config.MLP_DEFAULTS["target_size_exponent"]),
        help=(
            "Scale row weights by 1/n_t**alpha after class balancing "
            "(default 0 = off). Most useful with pooled BCE."
        ),
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
    p.add_argument(
        "--film",
        action="store_true",
        help=(
            "Use FiLM-conditioned ligand scoring (protein generates scale/shift). "
            "Mutually exclusive with --bilinear."
        ),
    )
    p.add_argument(
        "--max-rows-per-protein",
        type=int,
        default=None,
        help=(
            "After the train/test split, keep at most this many training rows "
            "per protein (fast hypothesis screens)."
        ),
    )
    p.add_argument(
        "--skip-test",
        action="store_true",
        help="Skip full test evaluation after fit (use a separate eval script).",
    )
    p.add_argument(
        "--high-prev-cut",
        type=float,
        default=0.8,
        help=(
            "With --high-prev-keep-frac < 1, two-class train targets whose "
            "active fraction exceeds this cut are downsampled."
        ),
    )
    p.add_argument(
        "--high-prev-keep-frac",
        type=float,
        default=1.0,
        help=(
            "Fraction of rows to keep for high-prevalence train targets "
            "(default 1.0 = off). Use e.g. 0.25 to focus learning on mid/low "
            "prevalence proteins."
        ),
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
    if args.model == "mlp" and args.target_balance != "none" and args.class_weights:
        raise SystemExit(
            "Cannot combine --class-weights with "
            f"--target-balance={args.target_balance}. "
            "Use --no-class-weights or --target-balance none."
        )
    if args.model == "rf" and args.target_balance != "none":
        raise SystemExit(
            "--target-balance is only supported for --model mlp."
        )
    if args.model == "rf" and (
        args.target_bce_reduction != "pooled"
        or args.rank_loss_weight != 0.0
        or args.listwise_loss_weight != 0.0
        or args.target_size_exponent != 0.0
    ):
        raise SystemExit(
            "--target-bce-reduction / --rank-loss-weight / "
            "--listwise-loss-weight / --target-size-exponent are only "
            "supported for --model mlp."
        )
    if args.rank_loss_weight < 0.0:
        raise SystemExit("--rank-loss-weight must be >= 0.")
    if args.listwise_loss_weight < 0.0:
        raise SystemExit("--listwise-loss-weight must be >= 0.")
    if args.target_size_exponent < 0.0:
        raise SystemExit("--target-size-exponent must be >= 0.")
    if args.rank_loss_weight > 0.0 and args.target_bce_reduction != "mean":
        raise SystemExit(
            "--rank-loss-weight > 0 requires --target-bce-reduction mean."
        )
    if args.listwise_loss_weight > 0.0 and args.target_bce_reduction != "mean":
        raise SystemExit(
            "--listwise-loss-weight > 0 requires --target-bce-reduction mean."
        )
    if args.film and args.bilinear:
        raise SystemExit("--film and --bilinear are mutually exclusive.")
    if args.rank_targets_per_batch < 1 or args.rank_samples_per_class < 1:
        raise SystemExit(
            "--rank-targets-per-batch and --rank-samples-per-class must be >= 1."
        )
    run_training(args)


if __name__ == "__main__":
    main()
