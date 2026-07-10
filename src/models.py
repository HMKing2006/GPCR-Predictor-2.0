"""Regressors with a shared fit/predict/save/load interface.

Two model families are provided:

* :class:`RandomForestModel` - a scikit-learn ``RandomForestRegressor`` grown
  with ``warm_start=True``. Trees are added in batches, each batch fit on a
  contiguous row-shard read from the (memmapped) training matrix, keeping peak
  memory near a single shard plus the incremental tree batch rather than the
  whole matrix and full forest at once.
* :class:`MLPModel` - a small PyTorch multilayer perceptron trained with
  minibatch SGD (Adam) streamed from the memmap, with input standardization.

Both expose ``fit``, ``predict``, ``save`` and ``load`` and carry a ``metadata``
dict (embedding model ids, dims) so prediction can reconstruct features
identically.
"""

from __future__ import annotations

import math
from typing import Any, Optional

import joblib
import numpy as np
import torch
from sklearn.ensemble import RandomForestRegressor
from torch import nn

import config
from src.embeddings import select_device

_PREDICT_CHUNK: int = 100_000


class BaseRegressor:
    """Common interface and joblib persistence for regressors."""

    model_type: str = "base"

    def __init__(self) -> None:
        """Initialize shared state (metadata carried into the saved file)."""
        self.metadata: dict[str, Any] = {}

    def fit(self, X: np.ndarray, y: np.ndarray, verbose: bool = True) -> "BaseRegressor":
        """Fit the model.

        Args:
            X: Feature matrix ``(n, d)`` (may be a memmap).
            y: Target vector ``(n,)``.
            verbose: If ``True``, print progress.

        Returns:
            ``self``.

        Raises:
            NotImplementedError: Always; subclasses must override.
        """
        raise NotImplementedError

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict targets for ``X``.

        Args:
            X: Feature matrix ``(n, d)``.

        Returns:
            Predicted values ``(n,)``.

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
        joblib.dump(
            {"model_type": self.model_type, "state": self._state(), "metadata": self.metadata},
            path,
        )


class RandomForestModel(BaseRegressor):
    """Warm-start random forest with sharded, memory-bounded training."""

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
        self.forest = RandomForestRegressor(
            n_estimators=0,
            warm_start=True,
            max_depth=None if max_depth in (0, None) else int(max_depth),
            min_samples_leaf=int(min_samples_leaf),
            n_jobs=int(n_jobs),
            random_state=self.seed,
        )

    def fit(self, X: np.ndarray, y: np.ndarray, verbose: bool = True) -> "RandomForestModel":
        """Grow the forest incrementally over rotating row shards.

        Args:
            X: Feature matrix ``(n, d)`` (may be a memmap).
            y: Target vector ``(n,)``.
            verbose: If ``True``, print progress after each round.

        Returns:
            ``self``.
        """
        n_rows = X.shape[0]
        shard = min(self.shard_rows, n_rows)
        n_rounds = math.ceil(self.n_estimators / self.batch_trees)
        trees = 0
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

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict in row chunks to bound memory.

        Args:
            X: Feature matrix ``(n, d)``.

        Returns:
            Predicted values ``(n,)``.
        """
        n_rows = X.shape[0]
        out = np.empty(n_rows, dtype=np.float32)
        for start in range(0, n_rows, _PREDICT_CHUNK):
            end = min(start + _PREDICT_CHUNK, n_rows)
            out[start:end] = self.forest.predict(np.asarray(X[start:end]))
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

    The input row is laid out as ``[protein | ligand | assay_onehot | pH | temp]``.
    When bilinear mode is enabled, the protein and ligand slices are projected to
    a shared width, mixed through ``nn.Bilinear``, and the resulting interaction
    vector is concatenated back onto the original feature row before the MLP
    trunk.
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
    """PyTorch MLP regressor trained with minibatch SGD from a memmap."""

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
            patience: Epochs without validation-RMSE improvement tolerated before
                stopping early. ``0`` disables early stopping.
            es_val_fraction: Fraction of training rows held out to monitor
                validation RMSE for early stopping.
            es_min_delta: Minimum validation-RMSE decrease counted as an
                improvement.
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

    def _standardize_stats(self, X: np.ndarray) -> None:
        """Compute per-feature mean and std over the training matrix in chunks.

        Args:
            X: Feature matrix ``(n, d)``.

        Returns:
            None. Sets ``feature_mean`` and ``feature_std`` in place.
        """
        n, d = X.shape
        total = np.zeros(d, dtype=np.float64)
        total_sq = np.zeros(d, dtype=np.float64)
        for start in range(0, n, _PREDICT_CHUNK):
            end = min(start + _PREDICT_CHUNK, n)
            chunk = np.asarray(X[start:end], dtype=np.float64)
            total += chunk.sum(axis=0)
            total_sq += (chunk**2).sum(axis=0)
        mean = total / n
        var = np.maximum(total_sq / n - mean**2, 1e-8)
        self.feature_mean = mean.astype(np.float32)
        self.feature_std = np.sqrt(var).astype(np.float32)

    def _batch_tensor(self, X: np.ndarray, rows: np.ndarray) -> torch.Tensor:
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

    def _eval_rmse(self, X: np.ndarray, rows: np.ndarray, y: np.ndarray) -> float:
        """Compute validation RMSE over a set of rows without tracking grads.

        Args:
            X: Feature matrix ``(n, d)``.
            rows: Integer row indices to evaluate.
            y: Full target vector ``(n,)`` (indexed by ``rows``).

        Returns:
            The root-mean-squared error on ``rows``.
        """
        assert self.net is not None
        self.net.eval()
        sq_err = 0.0
        with torch.no_grad():
            for start in range(0, rows.shape[0], _PREDICT_CHUNK):
                batch_rows = rows[start : start + _PREDICT_CHUNK]
                pred = self.net(self._batch_tensor(X, batch_rows)).detach().to("cpu").numpy()
                sq_err += float(np.sum((pred - y[batch_rows]) ** 2))
        return math.sqrt(sq_err / rows.shape[0])

    def fit(self, X: np.ndarray, y: np.ndarray, verbose: bool = True) -> "MLPModel":
        """Train the MLP with minibatch Adam and validation early stopping.

        A random ``es_val_fraction`` of the rows is held out as an internal
        validation set. After each epoch the validation RMSE is measured; the
        best-performing weights are cached and restored at the end, and training
        stops early once RMSE fails to improve by ``es_min_delta`` for
        ``patience`` consecutive epochs. Early stopping is skipped when
        ``patience`` is ``0`` or the dataset is too small to hold out a batch.

        Args:
            X: Feature matrix ``(n, d)`` (may be a memmap).
            y: Target vector ``(n,)``.
            verbose: If ``True``, print periodic batch and per-epoch loss.

        Returns:
            ``self``.
        """
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        n, d = X.shape
        self.input_dim = int(d)
        self._standardize_stats(X)
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
        loss_fn = nn.MSELoss()
        y_np = np.asarray(y, dtype=np.float32)
        y_t = torch.from_numpy(y_np)

        perm = rng.permutation(n)
        val_count = int(round(self.es_val_fraction * n)) if self.patience > 0 else 0
        val_count = min(val_count, n - 1) if val_count > 0 else 0
        val_rows = perm[:val_count]
        train_rows = perm[val_count:]
        early_stopping = val_count > 0
        best_rmse = math.inf
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

            val_rmse = self._eval_rmse(X, val_rows, y_np) if early_stopping else math.nan
            if verbose:
                tail = f"  val_rmse={val_rmse:.4f}" if early_stopping else ""
                print(
                    f"[mlp] epoch {epoch + 1}/{self.epochs} done  "
                    f"train_mse={running / max(seen, 1):.4f}{tail}",
                    flush=True,
                )

            if not early_stopping:
                continue
            if best_rmse - val_rmse > self.es_min_delta:
                best_rmse = val_rmse
                best_epoch = epoch + 1
                best_state = {k: v.detach().cpu().clone() for k, v in self.net.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= self.patience:
                    if verbose:
                        print(
                            f"[mlp] early stop at epoch {epoch + 1}; "
                            f"best val_rmse={best_rmse:.4f} @ epoch {best_epoch}",
                            flush=True,
                        )
                    break

        if best_state is not None:
            self.net.load_state_dict(best_state)
            if verbose:
                print(f"[mlp] restored best weights (val_rmse={best_rmse:.4f} @ epoch {best_epoch})")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict in chunks with the trained network.

        Args:
            X: Feature matrix ``(n, d)``.

        Returns:
            Predicted values ``(n,)``.
        """
        assert self.net is not None, "Model must be fit or loaded before predict."
        self.net.eval()
        n = X.shape[0]
        out = np.empty(n, dtype=np.float32)
        with torch.no_grad():
            for start in range(0, n, _PREDICT_CHUNK):
                end = min(start + _PREDICT_CHUNK, n)
                rows = np.arange(start, end)
                xb = self._batch_tensor(X, rows)
                out[start:end] = self.net(xb).detach().to("cpu").numpy()
        return out

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
        An unfitted :class:`BaseRegressor`.

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
        model = MLPModel(
            hidden_dim=state["hidden_dim"],
            num_layers=state["num_layers"],
            dropout=state["dropout"],
            batch_size=state["batch_size"],
            epochs=state["epochs"],
            learning_rate=state["learning_rate"],
            weight_decay=state["weight_decay"],
            use_batchnorm=use_batchnorm,
            use_bilinear=use_bilinear,
            bilinear_dim=bilinear_dim,
            protein_dim=protein_dim,
            ligand_dim=ligand_dim,
            seed=state["seed"],
        )
        model.input_dim = int(state["input_dim"])
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
