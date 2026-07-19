"""Train an experimental ligand family or target multilabel classifier."""

from __future__ import annotations

import argparse
import os
from typing import Any, Optional

import numpy as np

from src.ligand_repr import canonical_ligand_repr
from src.multilabel import config as ml_config
from src.multilabel.featurize import build_ligand_features, default_paths_for_task
from src.multilabel.metrics import print_multilabel_metrics, write_per_label_metrics
from src.multilabel.models import MultilabelMLP
from src.multilabel.splits import SPLIT_STRATEGIES, get_or_create_nested_split
from src.multilabel.vocab import load_vocab


def collect_hyperparams(args: argparse.Namespace) -> dict[str, Any]:
    """Extract multilabel MLP hyperparameters from CLI arguments.

    Args:
        args: Parsed argument namespace.

    Returns:
        Hyperparameter dictionary for :class:`MultilabelMLP`.
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
        "class_weights": args.class_weights,
    }


def _resolve_split(dataset, args: argparse.Namespace) -> dict[str, np.ndarray]:
    """Create or reuse nested scaffold/time outer folds.

    Test is carved with ``args.test_split`` first. ``train_multilabel`` merges
    any validation fold into train (``include_val=False``), matching pair
    ``train.py`` behavior.

    Args:
        dataset: Ligand feature dataset.
        args: Parsed CLI arguments.

    Returns:
        Mapping with at least ``train`` and ``test`` index arrays.

    Raises:
        SystemExit: If a time split is requested without dated ligands.
    """
    test_split = args.test_split
    validation_split = args.validation_split
    needs_time = test_split == "time" or validation_split == "time"
    years = None
    if needs_time:
        try:
            years = dataset.load_years()
        except FileNotFoundError as exc:
            raise SystemExit(str(exc)) from exc

    try:
        return get_or_create_nested_split(
            scaffold_groups=dataset.load_scaffold_groups(),
            signature=dataset.signature,
            test_split=test_split,
            validation_split=validation_split,
            val_fraction=float(args.val_fraction),
            test_fraction=float(args.test_fraction),
            include_val=False,
            years=years,
            seed=args.seed,
            splits_dir=dataset.split_directory,
            verbose=not args.quiet,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc


def build_arg_parser() -> argparse.ArgumentParser:
    """Construct the training CLI parser.

    Returns:
        Configured argument parser.
    """
    defaults = ml_config.MLP_DEFAULTS
    parser = argparse.ArgumentParser(
        description=(
            "Train an experimental ligand-centric family or target multilabel "
            "MLP. Isolated from the pair binder pipeline."
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
            "Strategy for the validation fold (default: scaffold). Test is "
            "carved first; val is carved from the remainder when a val fold is "
            "kept. train_multilabel merges val into train."
        ),
    )
    parser.add_argument(
        "--test-fraction",
        type=float,
        default=ml_config.TEST_FRACTION,
        help="Test fraction of all ligands (strategy-dependent assignment).",
    )
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=ml_config.VAL_FRACTION,
        help=(
            "Validation fraction of all ligands. Used to size time windows and "
            "nested val carves; merged into train by this entrypoint."
        ),
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
        help="Output joblib path (default under models/multilabel/).",
    )
    parser.add_argument(
        "--label-metrics",
        default=None,
        metavar="PATH",
        help=(
            "Optional spreadsheet path for per-label test metrics "
            "(AUROC, AUPRC, Precision, Recall, active_frac). "
            "Extension selects format (.xlsx/.csv/.tsv)."
        ),
    )
    parser.add_argument("--hidden-dim", type=int, default=int(defaults["hidden_dim"]))
    parser.add_argument("--num-layers", type=int, default=int(defaults["num_layers"]))
    parser.add_argument("--dropout", type=float, default=float(defaults["dropout"]))
    parser.add_argument("--batch-size", type=int, default=int(defaults["batch_size"]))
    parser.add_argument("--epochs", type=int, default=int(defaults["epochs"]))
    parser.add_argument(
        "--learning-rate", type=float, default=float(defaults["learning_rate"])
    )
    parser.add_argument(
        "--weight-decay", type=float, default=float(defaults["weight_decay"])
    )
    parser.add_argument("--patience", type=int, default=int(defaults["patience"]))
    parser.add_argument(
        "--es-val-fraction",
        type=float,
        default=float(defaults["es_val_fraction"]),
    )
    parser.add_argument(
        "--class-weights",
        action=argparse.BooleanOptionalAction,
        default=bool(defaults["class_weights"]),
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
    split = _resolve_split(dataset, args)
    train_idx = split["train"]
    test_idx = split["test"]
    vocab = load_vocab(vocab_path)
    hyperparams = collect_hyperparams(args)

    X_train = dataset.feature_view(train_idx)
    y_train = np.asarray(dataset.load_y()[train_idx], dtype=np.float32)
    model = MultilabelMLP(n_labels=dataset.n_labels, seed=args.seed, **hyperparams)
    model.metadata = {
        "task": f"{args.task}_multilabel",
        "vocab_path": os.path.realpath(vocab_path),
        "vocab": vocab,
        "ligand_model": canonical_ligand_repr(args.ligand_model),
        "ligand_dim": dataset.ligand_dim,
        "n_features": dataset.n_features,
        "n_labels": dataset.n_labels,
        "hyperparams": hyperparams,
        "activity_threshold_nm": dataset.activity_threshold_nm,
        "feature_storage_version": ml_config.STORAGE_VERSION,
        "feature_signature": dataset.signature,
        "feature_directory": dataset.directory,
        "ligand_embedding_aliases": list(dataset.ligand_aliases),
        "test_split": args.test_split,
        "validation_split": args.validation_split,
    }
    if verbose:
        mean_labels = float(y_train.sum(axis=1).mean()) if y_train.size else 0.0
        print(
            f"[train] fitting multilabel MLP on {len(train_idx)} ligands "
            f"(K={dataset.n_labels}, mean_labels={mean_labels:.2f})",
            flush=True,
        )
    model.fit(
        X_train,
        y_train,
        verbose=verbose,
        scaffold_groups=dataset.load_scaffold_groups()[train_idx],
    )
    del X_train

    X_test = dataset.feature_view(test_idx)
    y_test = np.asarray(dataset.load_y()[test_idx], dtype=np.float32)
    probs = model.predict(X_test)
    del X_test
    print_multilabel_metrics(y_test, probs, label="test", vocab=vocab)
    if args.label_metrics:
        write_per_label_metrics(
            args.label_metrics,
            y_test,
            probs,
            vocab=vocab,
        )
        if verbose:
            print(f"[train] wrote label metrics to {args.label_metrics}", flush=True)

    output = args.output
    if output is None:
        os.makedirs(ml_config.MODELS_DIR, exist_ok=True)
        output = os.path.join(
            ml_config.MODELS_DIR,
            (
                f"{args.task}_multilabel"
                f"__test-{args.test_split}"
                f"__val-{args.validation_split}.joblib"
            ),
        )
    model.save(output)
    if verbose:
        print(f"[train] saved {output}", flush=True)


if __name__ == "__main__":
    main()
