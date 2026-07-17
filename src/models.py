"""Binary classifiers with a shared fit/predict/save/load interface.

Two model families are provided:

* :class:`RandomForestModel` - a scikit-learn ``RandomForestClassifier`` grown
  with ``warm_start=True``. Trees are added in batches, each batch fit on a
  gathered row shard, keeping peak memory near one shard plus the incremental
  tree batch rather than the whole matrix and full forest at once.
* :class:`MLPModel` - a small PyTorch multilayer perceptron trained with
  minibatch SGD (Adam) gathered from an array-like feature view, with input
  standardization.

Both expose ``fit``, ``predict``, ``save`` and ``load`` and carry a ``metadata``
dict (embedding model ids, dims) so prediction can reconstruct features
identically. ``predict`` returns positive-class probabilities in ``[0, 1]``.
"""

from __future__ import annotations

import math
import os
from typing import Any, Optional

import joblib
import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
from torch import nn

import config
from src.embeddings import select_device
from src.metrics import auroc
from src.splits import cold_protein_split, double_cold_split

_PREDICT_CHUNK: int = 100_000


class BaseRegressor:
    """Common interface and joblib persistence for classifiers.

    The historical ``BaseRegressor`` name is kept so existing imports continue to
    work after the binder/non-binder cutover.
    """

    model_type: str = "base"

    def __init__(self) -> None:
        """Initialize shared state (metadata carried into the saved file)."""
        self.metadata: dict[str, Any] = {}

    def fit(self, X: Any, y: np.ndarray, verbose: bool = True) -> "BaseRegressor":
        """Fit the model.

        Args:
            X: Feature matrix ``(n, d)`` (may be a memmap).
            y: Binary target vector ``(n,)`` with labels in ``{0, 1}``.
            verbose: If ``True``, print progress.

        Returns:
            ``self``.

        Raises:
            NotImplementedError: Always; subclasses must override.
        """
        raise NotImplementedError

    def predict(self, X: Any) -> np.ndarray:
        """Predict positive-class probabilities for ``X``.

        Args:
            X: Feature matrix ``(n, d)``.

        Returns:
            Probabilities ``(n,)`` in ``[0, 1]``.

        Raises:
            NotImplementedError: Always; subclasses must override.
        """
        raise NotImplementedError

    def _state(self) -> dict[str, Any]:
        """Return the picklable payload for this model.

        Returns:
            A dictionary of everything needed to restore the model.

        Raises:
            NotImplementedError: Always; subclasses must override.
        """
        raise NotImplementedError

    def save(self, path: str) -> None:
        """Serialize the model to a joblib file.

        Args:
            path: Destination path.

        Returns:
            None.
        """
        payload = {
            "model_type": self.model_type,
            "state": self._state(),
            "metadata": self.metadata,
        }
        temporary = f"{path}.tmp-{os.getpid()}"
        try:
            joblib.dump(payload, temporary)
            verified = joblib.load(temporary)
            if verified.get("model_type") != self.model_type:
                raise ValueError(f"Model verification failed for {temporary}.")
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.remove(temporary)


class RandomForestModel(BaseRegressor):
    """Warm-start random forest classifier with sharded, memory-bounded training."""

    model_type = "rf"

    def __init__(
        self,
        n_estimators: int = config.RF_DEFAULTS["n_estimators"],
        batch_trees: int = config.RF_DEFAULTS["batch_trees"],
        shard_rows: int = config.RF_DEFAULTS["shard_rows"],
        max_depth: int = config.RF_DEFAULTS["max_depth"],
        min_samples_leaf: int = config.RF_DEFAULTS["min_samples_leaf"],
        n_jobs: int = config.RF_DEFAULTS["n_jobs"],
        seed: int = config.RANDOM_SEED,
    ) -> None:
        """Configure the forest.

        Args:
            n_estimators: Total number of trees to grow.
            batch_trees: Trees added per warm-start round.
            shard_rows: Rows per training shard (bounds peak memory).
            max_depth: Maximum tree depth; ``0`` means unlimited.
            min_samples_leaf: Minimum samples per leaf.
            n_jobs: Parallel jobs for sklearn (``-1`` uses all cores).
            seed: Random seed.
        """
        super().__init__()
        self.n_estimators = int(n_estimators)
        self.batch_trees = int(batch_trees)
        self.shard_rows = int(shard_rows)
        self.seed = int(seed)
        self.forest = RandomForestClassifier(
            n_estimators=0,
            warm_start=True,
            max_depth=None if max_depth in (0, None) else int(max_depth),
            min_samples_leaf=int(min_samples_leaf),
            n_jobs=int(n_jobs),
            random_state=self.seed,
        )

    def fit(self, X: Any, y: np.ndarray, verbose: bool = True) -> "RandomForestModel":
        """Grow the forest incrementally over rotating row shards.

        Args:
            X: Feature matrix ``(n, d)`` (may be a memmap).
            y: Binary target vector ``(n,)`` with labels in ``{0, 1}``.
            verbose: If ``True``, print progress after each round.

        Returns:
            ``self``.
        """
        n_rows = X.shape[0]
        shard = min(self.shard_rows, n_rows)
        n_rounds = math.ceil(self.n_estimators / self.batch_trees)
        trees = 0
        y = np.asarray(y)
        for round_idx in range(n_rounds):
            trees = min(self.n_estimators, trees + self.batch_trees)
            self.forest.n_estimators = trees
            start = (round_idx * shard) % n_rows
            end = start + shard
            if end <= n_rows:
                xb = np.asarray(X[start:end])
                yb = y[start:end]
            else:
                xb = np.concatenate([np.asarray(X[start:n_rows]), np.asarray(X[0 : end - n_rows])])
                yb = np.concatenate([y[start:n_rows], y[0 : end - n_rows]])
            self.forest.fit(xb, yb)
            if verbose:
                print(
                    f"[rf] round {round_idx + 1}/{n_rounds}: {trees}/{self.n_estimators} trees "
                    f"(shard {len(yb)} rows)",
                    flush=True,
                )
        return self

    def predict(self, X: Any) -> np.ndarray:
        """Predict binder probabilities in row chunks to bound memory.

        Args:
            X: Feature matrix ``(n, d)``.

        Returns:
            Positive-class probabilities ``(n,)``.
        """
        n_rows = X.shape[0]
        out = np.empty(n_rows, dtype=np.float32)
        classes = np.asarray(self.forest.classes_)
        pos_matches = np.flatnonzero(classes == 1)
        if pos_matches.size == 0:
            pos_matches = np.flatnonzero(classes == 1.0)
        pos_col = int(pos_matches[0])
        for start in range(0, n_rows, _PREDICT_CHUNK):
            end = min(start + _PREDICT_CHUNK, n_rows)
            probs = self.forest.predict_proba(np.asarray(X[start:end]))
            out[start:end] = probs[:, pos_col]
        return out

    def _state(self) -> dict[str, Any]:
        """Return the picklable payload (the fitted forest and settings).

        Returns:
            A dictionary with the sklearn estimator and hyperparameters.
        """
        return {
            "forest": self.forest,
            "n_estimators": self.n_estimators,
            "batch_trees": self.batch_trees,
            "shard_rows": self.shard_rows,
            "seed": self.seed,
        }


class _MLP(nn.Module):
    """Feed-forward network over pooled features with an optional bilinear head.

    The input row is laid out as ``[protein | ligand]`` by default, or
    ``[protein | ligand | assay_onehot | pH | temp]`` when assay context is
    enabled. When bilinear mode is enabled, the protein and ligand slices are
    projected to a shared width, mixed through ``nn.Bilinear``, and the
    resulting interaction vector is concatenated back onto the original feature
    row before the MLP trunk.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
        *,
        protein_dim: int,
        ligand_dim: int,
        use_batchnorm: bool = False,
        use_bilinear: bool = False,
        bilinear_dim: int = 256,
    ) -> None:
        """Build the optional bilinear head and MLP trunk.

        Args:
            input_dim: Full feature-vector length.
            hidden_dim: Hidden layer width.
            num_layers: Number of hidden layers.
            dropout: Dropout probability between layers.
            protein_dim: Length of the leading protein-embedding slice.
            ligand_dim: Length of the ligand-embedding slice that follows it.
            use_batchnorm: If ``True``, insert ``BatchNorm1d`` after each hidden
                linear layer.
            use_bilinear: If ``True``, append a learned bilinear protein/ligand
                interaction vector before the MLP trunk.
            bilinear_dim: Shared width for the protein/ligand projections and the
                bilinear interaction block.
        """
        super().__init__()
        self.protein_dim = int(protein_dim)
        self.ligand_dim = int(ligand_dim)
        self.use_bilinear = bool(use_bilinear)
        trunk_input_dim = input_dim
        if self.use_bilinear:
            self.protein_proj = nn.Linear(self.protein_dim, bilinear_dim)
            self.ligand_proj = nn.Linear(self.ligand_dim, bilinear_dim)
            self.bilinear = nn.Bilinear(bilinear_dim, bilinear_dim, bilinear_dim)
            trunk_input_dim += bilinear_dim

        layers: list[nn.Module] = []
        prev = trunk_input_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(prev, hidden_dim))
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers += [nn.ReLU(), nn.Dropout(dropout)]
            prev = hidden_dim
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a forward pass.

        Args:
            x: Input batch ``(batch, input_dim)``.

        Returns:
            Output batch ``(batch,)``.
        """
        if self.use_bilinear:
            protein = x[:, : self.protein_dim]
            ligand = x[:, self.protein_dim : self.protein_dim + self.ligand_dim]
            protein_proj = self.protein_proj(protein)
            ligand_proj = self.ligand_proj(ligand)
            bilinear = self.bilinear(protein_proj, ligand_proj)
            x = torch.cat([x, bilinear], dim=1)
        return self.net(x).squeeze(-1)


class MLPModel(BaseRegressor):
    """PyTorch MLP classifier trained from gathered minibatches."""

    model_type = "mlp"

    def __init__(
        self,
        hidden_dim: int = int(config.MLP_DEFAULTS["hidden_dim"]),
        num_layers: int = int(config.MLP_DEFAULTS["num_layers"]),
        dropout: float = float(config.MLP_DEFAULTS["dropout"]),
        batch_size: int = int(config.MLP_DEFAULTS["batch_size"]),
        epochs: int = int(config.MLP_DEFAULTS["epochs"]),
        learning_rate: float = float(config.MLP_DEFAULTS["learning_rate"]),
        weight_decay: float = float(config.MLP_DEFAULTS["weight_decay"]),
        patience: int = int(config.MLP_DEFAULTS["patience"]),
        es_val_fraction: float = float(config.MLP_DEFAULTS["es_val_fraction"]),
        es_min_delta: float = float(config.MLP_DEFAULTS["es_min_delta"]),
        class_weights: bool = bool(config.MLP_DEFAULTS["class_weights"]),
        use_batchnorm: bool = bool(config.MLP_DEFAULTS["use_batchnorm"]),
        use_bilinear: bool = bool(config.MLP_DEFAULTS["use_bilinear"]),
        bilinear_dim: int = int(config.MLP_DEFAULTS["bilinear_dim"]),
        protein_dim: int = config.PROTEIN_EMB_DIM,
        ligand_dim: int = config.LIGAND_EMB_DIM,
        seed: int = config.RANDOM_SEED,
        device: Optional[torch.device] = None,
    ) -> None:
        """Configure the MLP and training schedule.

        Args:
            hidden_dim: Hidden layer width.
            num_layers: Number of hidden layers.
            dropout: Dropout probability.
            batch_size: Minibatch size.
            epochs: Maximum number of training epochs.
            learning_rate: Adam learning rate.
            weight_decay: Adam weight decay.
            patience: Epochs without validation-AUROC improvement tolerated before
                stopping early. ``0`` disables early stopping.
            es_val_fraction: Target fraction of training *rows* held out as a
                cold-protein early-stopping set (proteins disjoint from the
                fit set).
            es_min_delta: Minimum validation-AUROC increase counted as an
                improvement.
            class_weights: If ``True``, set BCE ``pos_weight`` to
                ``n_neg / n_pos`` on the fit rows (inverse class frequency).
            use_batchnorm: If ``True``, add BatchNorm1d after each hidden layer.
            use_bilinear: If ``True``, append a learned bilinear protein/ligand
                interaction vector before the MLP trunk.
            bilinear_dim: Shared width for the protein/ligand projections and the
                bilinear interaction block.
            protein_dim: Length of the protein-embedding slice in each row.
            ligand_dim: Length of the ligand-embedding slice in each row.
            seed: Random seed.
            device: Torch device; auto-selected when ``None``.
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.patience = int(patience)
        self.es_val_fraction = float(es_val_fraction)
        self.es_min_delta = float(es_min_delta)
        self.class_weights = bool(class_weights)
        self.use_batchnorm = bool(use_batchnorm)
        self.use_bilinear = bool(use_bilinear)
        self.bilinear_dim = int(bilinear_dim)
        self.protein_dim = int(protein_dim)
        self.ligand_dim = int(ligand_dim)
        self.input_dim: Optional[int] = None
        self.seed = seed
        self.device = device if device is not None else select_device()
        self.net: Optional[_MLP] = None
        self.feature_mean: Optional[np.ndarray] = None
        self.feature_std: Optional[np.ndarray] = None
        self.pos_weight: Optional[float] = None

    def _standardize_stats(self, X: Any, rows: Optional[np.ndarray] = None) -> None:
        """Compute per-feature mean and std over selected rows in chunks.

        Args:
            X: Feature matrix ``(n, d)``.
            rows: Optional row indices to include. When ``None``, uses every row.

        Returns:
            None. Sets ``feature_mean`` and ``feature_std`` in place.
        """
        if rows is None:
            rows = np.arange(X.shape[0], dtype=np.int64)
        else:
            rows = np.asarray(rows, dtype=np.int64)
        n = int(rows.shape[0])
        d = X.shape[1]
        total = np.zeros(d, dtype=np.float64)
        total_sq = np.zeros(d, dtype=np.float64)
        for start in range(0, n, _PREDICT_CHUNK):
            batch_rows = rows[start : start + _PREDICT_CHUNK]
            chunk = np.asarray(X[batch_rows], dtype=np.float64)
            total += chunk.sum(axis=0)
            total_sq += (chunk**2).sum(axis=0)
        mean = total / max(n, 1)
        var = np.maximum(total_sq / max(n, 1) - mean**2, 1e-8)
        self.feature_mean = mean.astype(np.float32)
        self.feature_std = np.sqrt(var).astype(np.float32)

    def _batch_tensor(self, X: Any, rows: np.ndarray) -> torch.Tensor:
        """Gather and standardize a batch of rows as a device tensor.

        Args:
            X: Feature matrix ``(n, d)``.
            rows: Integer row indices for the batch.

        Returns:
            A standardized ``float32`` tensor on the model device.
        """
        batch = np.asarray(X[rows], dtype=np.float32)
        batch = (batch - self.feature_mean) / self.feature_std
        return torch.from_numpy(batch).to(self.device)

    def _predict_logits(self, X: Any, rows: np.ndarray) -> np.ndarray:
        """Run the network on selected rows and return raw logits on CPU.

        Args:
            X: Feature matrix ``(n, d)``.
            rows: Integer row indices to score.

        Returns:
            Logits as a ``float32`` numpy array ``(len(rows),)``.
        """
        assert self.net is not None
        self.net.eval()
        chunks: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, rows.shape[0], _PREDICT_CHUNK):
                batch_rows = rows[start : start + _PREDICT_CHUNK]
                logits = self.net(self._batch_tensor(X, batch_rows)).detach().to("cpu").numpy()
                chunks.append(logits.astype(np.float32, copy=False))
        return np.concatenate(chunks) if chunks else np.empty(0, dtype=np.float32)

    def _eval_auroc(self, X: Any, rows: np.ndarray, y: np.ndarray) -> float:
        """Compute validation AUROC over a set of rows without tracking grads.

        Args:
            X: Feature matrix ``(n, d)``.
            rows: Integer row indices to evaluate.
            y: Full binary target vector ``(n,)`` (indexed by ``rows``).

        Returns:
            The AUROC on ``rows``.
        """
        logits = self._predict_logits(X, rows)
        probs = 1.0 / (1.0 + np.exp(-logits))
        return auroc(y[rows], probs)

    def fit(
        self,
        X: Any,
        y: np.ndarray,
        verbose: bool = True,
        groups: Optional[np.ndarray] = None,
        scaffold_groups: Optional[np.ndarray] = None,
    ) -> "MLPModel":
        """Train the MLP with minibatch Adam and validation early stopping.

        When ``patience > 0``, a holdout of about ``es_val_fraction`` of the
        rows is carved so no protein (and, when ``scaffold_groups`` is provided,
        no Murcko scaffold) appears in both the fit set and the early-stopping
        set. After each epoch the holdout AUROC is measured; the best-performing
        weights are cached and restored at the end, and training stops early
        once AUROC fails to improve by ``es_min_delta`` for ``patience``
        consecutive epochs. Early stopping is skipped when ``patience`` is
        ``0``. When ``class_weights`` is enabled, BCE uses
        ``pos_weight = n_neg / n_pos`` computed on the fit rows.

        Args:
            X: Feature matrix ``(n, d)`` (may be a memmap).
            y: Binary target vector ``(n,)`` with labels in ``{0, 1}``.
            verbose: If ``True``, print periodic batch and per-epoch loss.
            groups: Per-row protein group ids aligned with ``X`` / ``y``.
                Required when early stopping is enabled.
            scaffold_groups: Optional per-row Murcko scaffold ids aligned with
                ``X`` / ``y``. When provided with ``groups``, the early-stopping
                holdout is double-cold (protein and scaffold disjoint).

        Returns:
            ``self``.

        Raises:
            ValueError: If early stopping is enabled but ``groups`` is missing
                or has the wrong length.
        """
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        n, d = X.shape
        self.input_dim = int(d)
        y_np = np.asarray(y, dtype=np.float32)
        y_t = torch.from_numpy(y_np)

        if self.patience > 0:
            if groups is None:
                raise ValueError(
                    "MLP early stopping requires protein groups for a cold holdout."
                )
            groups_arr = np.asarray(groups)
            if groups_arr.shape[0] != n:
                raise ValueError(
                    f"groups length {groups_arr.shape[0]} does not match X rows {n}."
                )
            fractions = {
                "train": 1.0 - self.es_val_fraction,
                "es_val": self.es_val_fraction,
            }
            min_es_rows = max(1, int(0.25 * self.es_val_fraction * n))

            def _usable(split: dict[str, np.ndarray]) -> bool:
                """Return True when fit/holdout sizes are non-empty and oriented."""
                n_train = int(split["train"].shape[0])
                n_val = int(split["es_val"].shape[0])
                return n_train > 0 and n_val >= min_es_rows and n_train >= n_val

            if scaffold_groups is not None:
                scaffold_arr = np.asarray(scaffold_groups)
                if scaffold_arr.shape[0] != n:
                    raise ValueError(
                        f"scaffold_groups length {scaffold_arr.shape[0]} "
                        f"does not match X rows {n}."
                    )
                es_split = double_cold_split(
                    groups_arr, scaffold_arr, fractions, seed=self.seed
                )
                if _usable(es_split):
                    es_kind = "double-cold"
                else:
                    if verbose:
                        print(
                            "[mlp] double-cold ES holdout too small; "
                            "falling back to cold-protein",
                            flush=True,
                        )
                    es_split = cold_protein_split(
                        groups_arr, fractions, seed=self.seed
                    )
                    es_kind = "cold-protein"
            else:
                es_split = cold_protein_split(groups_arr, fractions, seed=self.seed)
                es_kind = "cold-protein"
            train_rows = es_split["train"]
            val_rows = es_split["es_val"]
            early_stopping = _usable(es_split)
            if verbose and early_stopping:
                n_train_prot = int(np.unique(groups_arr[train_rows]).shape[0])
                n_val_prot = int(np.unique(groups_arr[val_rows]).shape[0])
                print(
                    f"[mlp] {es_kind} early-stop holdout: "
                    f"train={train_rows.shape[0]} rows / {n_train_prot} proteins, "
                    f"es_val={val_rows.shape[0]} rows / {n_val_prot} proteins",
                    flush=True,
                )
        else:
            train_rows = np.arange(n, dtype=np.int64)
            val_rows = np.empty(0, dtype=np.int64)
            early_stopping = False

        if not early_stopping and self.patience > 0:
            train_rows = np.arange(n, dtype=np.int64)
            if verbose:
                print("[mlp] early stopping disabled (empty cold holdout)", flush=True)

        self._standardize_stats(X, train_rows)
        self.net = _MLP(
            d,
            self.hidden_dim,
            self.num_layers,
            self.dropout,
            protein_dim=self.protein_dim,
            ligand_dim=self.ligand_dim,
            use_batchnorm=self.use_batchnorm,
            use_bilinear=self.use_bilinear,
            bilinear_dim=self.bilinear_dim,
        ).to(self.device)
        optimizer = torch.optim.Adam(
            self.net.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )
        y_fit = y_np[train_rows]
        n_pos = float(y_fit.sum())
        n_neg = float(y_fit.shape[0]) - n_pos
        if self.class_weights and n_pos > 0.0:
            self.pos_weight = n_neg / n_pos
            loss_fn = nn.BCEWithLogitsLoss(
                pos_weight=torch.tensor([self.pos_weight], device=self.device)
            )
            if verbose:
                print(
                    f"[mlp] BCE pos_weight={self.pos_weight:.4f} "
                    f"(n_pos={int(n_pos)}, n_neg={int(n_neg)})",
                    flush=True,
                )
        else:
            self.pos_weight = None
            loss_fn = nn.BCEWithLogitsLoss()
            if verbose and self.class_weights and n_pos <= 0.0:
                print(
                    "[mlp] class_weights requested but no positives in fit "
                    "set; using unweighted BCE",
                    flush=True,
                )
        best_auroc = -math.inf
        best_state: Optional[dict[str, torch.Tensor]] = None
        best_epoch = 0
        no_improve = 0

        for epoch in range(self.epochs):
            self.net.train()
            order = rng.permutation(train_rows)
            running = 0.0
            seen = 0
            for bstart in range(0, order.shape[0], self.batch_size):
                rows = order[bstart : bstart + self.batch_size]
                # BatchNorm1d cannot normalize a singleton batch in train mode.
                if self.use_batchnorm and len(rows) < 2:
                    continue
                xb = self._batch_tensor(X, rows)
                yb = y_t[rows].to(self.device)
                optimizer.zero_grad()
                pred = self.net(xb)
                loss = loss_fn(pred, yb)
                loss.backward()
                optimizer.step()
                running += float(loss.item()) * len(rows)
                seen += len(rows)
                if verbose and (bstart // self.batch_size) % 200 == 0:
                    print(
                        f"[mlp] epoch {epoch + 1}/{self.epochs} "
                        f"{seen}/{order.shape[0]} rows  loss={running / seen:.4f}",
                        flush=True,
                    )

            val_auroc = self._eval_auroc(X, val_rows, y_np) if early_stopping else math.nan
            if verbose:
                tail = f"  val_auroc={val_auroc:.4f}" if early_stopping else ""
                print(
                    f"[mlp] epoch {epoch + 1}/{self.epochs} done  "
                    f"train_bce={running / max(seen, 1):.4f}{tail}",
                    flush=True,
                )

            if not early_stopping:
                continue
            if val_auroc - best_auroc > self.es_min_delta:
                best_auroc = val_auroc
                best_epoch = epoch + 1
                best_state = {k: v.detach().cpu().clone() for k, v in self.net.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= self.patience:
                    if verbose:
                        print(
                            f"[mlp] early stop at epoch {epoch + 1}; "
                            f"best val_auroc={best_auroc:.4f} @ epoch {best_epoch}",
                            flush=True,
                        )
                    break

        if best_state is not None:
            self.net.load_state_dict(best_state)
            if verbose:
                print(
                    f"[mlp] restored best weights (val_auroc={best_auroc:.4f} @ epoch {best_epoch})"
                )
        return self
    def predict(self, X: Any) -> np.ndarray:
        """Predict binder probabilities in chunks with the trained network.

        Args:
            X: Feature matrix ``(n, d)``.

        Returns:
            Positive-class probabilities ``(n,)``.
        """
        assert self.net is not None, "Model must be fit or loaded before predict."
        rows = np.arange(X.shape[0])
        logits = self._predict_logits(X, rows)
        return (1.0 / (1.0 + np.exp(-logits))).astype(np.float32)

    def _state(self) -> dict[str, Any]:
        """Return the picklable payload (weights, scaler and hyperparameters).

        Returns:
            A dictionary sufficient to rebuild and restore the network.
        """
        assert self.net is not None, "Model must be fit before saving."
        assert self.input_dim is not None, "Model must be fit before saving."
        return {
            "state_dict": {k: v.cpu() for k, v in self.net.state_dict().items()},
            "input_dim": int(self.input_dim),
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "class_weights": self.class_weights,
            "pos_weight": self.pos_weight,
            "use_batchnorm": self.use_batchnorm,
            "use_bilinear": self.use_bilinear,
            "bilinear_dim": self.bilinear_dim,
            "protein_dim": self.protein_dim,
            "ligand_dim": self.ligand_dim,
            "seed": self.seed,
            "feature_mean": self.feature_mean,
            "feature_std": self.feature_std,
        }


def build_model(model_type: str, seed: int = config.RANDOM_SEED, **hyperparams: Any) -> BaseRegressor:
    """Construct an untrained model of the requested type.

    Args:
        model_type: Either ``"rf"`` or ``"mlp"``.
        seed: Random seed passed to the model.
        **hyperparams: Model-specific hyperparameters overriding defaults.

    Returns:
        An unfitted classifier (:class:`BaseRegressor` interface).

    Raises:
        ValueError: If ``model_type`` is unknown.
    """
    if model_type == "rf":
        return RandomForestModel(seed=seed, **hyperparams)
    if model_type == "mlp":
        return MLPModel(seed=seed, **hyperparams)
    raise ValueError(f"Unknown model type: {model_type!r}")


def load_model(path: str) -> BaseRegressor:
    """Load a model previously written by :meth:`BaseRegressor.save`.

    Args:
        path: Path to the joblib file.

    Returns:
        The restored model with its ``metadata`` populated.

    Raises:
        ValueError: If the file references an unknown model type.
    """
    blob = joblib.load(path)
    model_type = blob["model_type"]
    state = blob["state"]
    if model_type == "rf":
        model = RandomForestModel(
            n_estimators=state["n_estimators"],
            batch_trees=state["batch_trees"],
            shard_rows=state["shard_rows"],
            seed=state["seed"],
        )
        model.forest = state["forest"]
    elif model_type == "mlp":
        use_batchnorm = bool(state.get("use_batchnorm", False))
        use_bilinear = bool(state.get("use_bilinear", False))
        bilinear_dim = int(state.get("bilinear_dim", config.MLP_DEFAULTS["bilinear_dim"]))
        protein_dim = int(state.get("protein_dim", config.PROTEIN_EMB_DIM))
        ligand_dim = int(state.get("ligand_dim", config.LIGAND_EMB_DIM))
        class_weights = bool(
            state.get(
                "class_weights",
                state.get(
                    "balance_pos_weight", config.MLP_DEFAULTS["class_weights"]
                ),
            )
        )
        model = MLPModel(
            hidden_dim=state["hidden_dim"],
            num_layers=state["num_layers"],
            dropout=state["dropout"],
            batch_size=state["batch_size"],
            epochs=state["epochs"],
            learning_rate=state["learning_rate"],
            weight_decay=state["weight_decay"],
            class_weights=class_weights,
            use_batchnorm=use_batchnorm,
            use_bilinear=use_bilinear,
            bilinear_dim=bilinear_dim,
            protein_dim=protein_dim,
            ligand_dim=ligand_dim,
            seed=state["seed"],
        )
        model.input_dim = int(state["input_dim"])
        model.pos_weight = (
            None if state.get("pos_weight") is None else float(state["pos_weight"])
        )
        model.net = _MLP(
            state["input_dim"],
            state["hidden_dim"],
            state["num_layers"],
            state["dropout"],
            protein_dim=protein_dim,
            ligand_dim=ligand_dim,
            use_batchnorm=use_batchnorm,
            use_bilinear=use_bilinear,
            bilinear_dim=bilinear_dim,
        )
        model.net.load_state_dict(state["state_dict"])
        model.net.to(model.device)
        model.feature_mean = state["feature_mean"]
        model.feature_std = state["feature_std"]
    else:
        raise ValueError(f"Unknown model type in {path!r}: {model_type!r}")
    model.metadata = blob.get("metadata", {})
    return model
