"""Ligand multilabel MLP with BCEWithLogits over a fixed vocabulary."""

from __future__ import annotations

import math
import os
from typing import Any, Optional

import joblib
import numpy as np
import torch
from torch import nn

from src.embeddings import select_device
from src.multilabel import config as ml_config
from src.multilabel.metrics import micro_auprc
from src.multilabel.splits import scaffold_cold_split

_PREDICT_CHUNK: int = 50_000


def build_active_fraction_mask(
    y_fit: np.ndarray,
    active_fraction: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Build a fixed per-label mask that downsamples negatives to a target density.

    For each label, all positives are kept. Negatives are randomly subsampled
    once so that ``n_pos / (n_pos + n_neg_kept)`` is approximately
    ``active_fraction``. Labels already at or above that density keep every
    negative. Labels with no positives keep all zeros (no sampling).

    Args:
        y_fit: Train label matrix ``(n_train, K)`` with values in ``{0, 1}``.
        active_fraction: Target positive fraction in ``(0, 1]``.
        rng: NumPy Generator used for the fixed negative draw.

    Returns:
        Boolean mask ``(n_train, K)`` where ``True`` cells contribute to BCE.

    Raises:
        ValueError: If ``active_fraction`` is outside ``(0, 1]`` or ``y_fit``
            is not 2-D.
    """
    fraction = float(active_fraction)
    if not (0.0 < fraction <= 1.0):
        raise ValueError(
            f"active_fraction must be in (0, 1], got {active_fraction!r}."
        )
    y_arr = np.asarray(y_fit)
    if y_arr.ndim != 2:
        raise ValueError(f"y_fit must be 2-D, got shape {y_arr.shape}.")
    n_train, n_labels = y_arr.shape
    positives = y_arr > 0.5
    mask = positives.copy()
    if fraction >= 1.0:
        return mask
    inv_ratio = (1.0 - fraction) / fraction
    for label_idx in range(n_labels):
        pos_rows = np.flatnonzero(positives[:, label_idx])
        n_pos = int(pos_rows.shape[0])
        if n_pos == 0:
            continue
        neg_rows = np.flatnonzero(~positives[:, label_idx])
        n_neg = int(neg_rows.shape[0])
        if n_neg == 0:
            continue
        density = n_pos / float(n_pos + n_neg)
        if density >= fraction:
            mask[neg_rows, label_idx] = True
            continue
        n_neg_keep = int(round(n_pos * inv_ratio))
        n_neg_keep = max(0, min(n_neg, n_neg_keep))
        if n_neg_keep == 0:
            continue
        chosen = rng.choice(neg_rows, size=n_neg_keep, replace=False)
        mask[chosen, label_idx] = True
    return mask


class _MultilabelMLP(nn.Module):
    """Feed-forward network mapping ligand features to ``K`` logits."""

    def __init__(
        self,
        input_dim: int,
        n_labels: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        """Build the MLP trunk and multilabel output head.

        Args:
            input_dim: Ligand feature dimension.
            n_labels: Vocabulary size ``K``.
            hidden_dim: Hidden layer width.
            num_layers: Number of hidden layers.
            dropout: Dropout probability between layers.
        """
        super().__init__()
        layers: list[nn.Module] = []
        prev = input_dim
        for _ in range(num_layers):
            layers += [nn.Linear(prev, hidden_dim), nn.ReLU(), nn.Dropout(dropout)]
            prev = hidden_dim
        layers.append(nn.Linear(prev, n_labels))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a forward pass.

        Args:
            x: Input batch ``(batch, input_dim)``.

        Returns:
            Logits ``(batch, K)``.
        """
        return self.net(x)


class MultilabelMLP:
    """PyTorch multilabel classifier trained with BCEWithLogitsLoss."""

    model_type = "multilabel_mlp"

    def __init__(
        self,
        n_labels: int,
        hidden_dim: int = int(ml_config.MLP_DEFAULTS["hidden_dim"]),
        num_layers: int = int(ml_config.MLP_DEFAULTS["num_layers"]),
        dropout: float = float(ml_config.MLP_DEFAULTS["dropout"]),
        batch_size: int = int(ml_config.MLP_DEFAULTS["batch_size"]),
        epochs: int = int(ml_config.MLP_DEFAULTS["epochs"]),
        learning_rate: float = float(ml_config.MLP_DEFAULTS["learning_rate"]),
        weight_decay: float = float(ml_config.MLP_DEFAULTS["weight_decay"]),
        patience: int = int(ml_config.MLP_DEFAULTS["patience"]),
        es_val_fraction: float = float(ml_config.MLP_DEFAULTS["es_val_fraction"]),
        es_min_delta: float = float(ml_config.MLP_DEFAULTS["es_min_delta"]),
        class_weights: bool = bool(ml_config.MLP_DEFAULTS["class_weights"]),
        active_fraction: Optional[float] = ml_config.MLP_DEFAULTS["active_fraction"],
        seed: int = ml_config.RANDOM_SEED,
        device: Optional[torch.device] = None,
    ) -> None:
        """Configure the multilabel MLP and training schedule.

        Args:
            n_labels: Vocabulary size ``K``.
            hidden_dim: Hidden layer width.
            num_layers: Number of hidden layers.
            dropout: Dropout probability.
            batch_size: Minibatch size.
            epochs: Maximum number of training epochs.
            learning_rate: Adam learning rate.
            weight_decay: Adam weight decay.
            patience: Epochs without micro-AUPRC improvement before stopping.
                ``0`` disables early stopping.
            es_val_fraction: Target fraction of training rows for a scaffold-cold
                early-stopping holdout.
            es_min_delta: Minimum micro-AUPRC increase counted as improvement.
            class_weights: If ``True``, use per-label ``pos_weight = n_neg/n_pos``.
            active_fraction: If set, keep a fixed per-label train mask so
                positives are approximately this fraction of supervised cells.
                ``None`` disables masking.
            seed: Random seed.
            device: Torch device; auto-selected when ``None``.

        Raises:
            ValueError: If ``active_fraction`` is outside ``(0, 1]``.
        """
        self.n_labels = int(n_labels)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.dropout = float(dropout)
        self.batch_size = int(batch_size)
        self.epochs = int(epochs)
        self.learning_rate = float(learning_rate)
        self.weight_decay = float(weight_decay)
        self.patience = int(patience)
        self.es_val_fraction = float(es_val_fraction)
        self.es_min_delta = float(es_min_delta)
        self.class_weights = bool(class_weights)
        if active_fraction is None:
            self.active_fraction: Optional[float] = None
        else:
            fraction = float(active_fraction)
            if not (0.0 < fraction <= 1.0):
                raise ValueError(
                    f"active_fraction must be in (0, 1], got {active_fraction!r}."
                )
            self.active_fraction = fraction
        self.seed = int(seed)
        self.device = device if device is not None else select_device()
        self.input_dim: Optional[int] = None
        self.net: Optional[_MultilabelMLP] = None
        self.feature_mean: Optional[np.ndarray] = None
        self.feature_std: Optional[np.ndarray] = None
        self.pos_weight: Optional[np.ndarray] = None
        self.metadata: dict[str, Any] = {}

    def _standardize_stats(self, X: Any, rows: np.ndarray) -> None:
        """Compute per-feature mean and std over selected rows in chunks.

        Args:
            X: Feature matrix ``(n, d)``.
            rows: Row indices to include.

        Returns:
            None.
        """
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
            Standardized ``float32`` tensor on the model device.
        """
        batch = np.asarray(X[rows], dtype=np.float32)
        batch = (batch - self.feature_mean) / self.feature_std
        return torch.from_numpy(batch).to(self.device)

    def _predict_logits(self, X: Any, rows: np.ndarray) -> np.ndarray:
        """Run the network on selected rows and return logits on CPU.

        Args:
            X: Feature matrix ``(n, d)``.
            rows: Integer row indices to score.

        Returns:
            Logits ``(len(rows), K)`` as ``float32``.
        """
        assert self.net is not None
        self.net.eval()
        chunks: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, rows.shape[0], _PREDICT_CHUNK):
                batch_rows = rows[start : start + _PREDICT_CHUNK]
                logits = self.net(self._batch_tensor(X, batch_rows)).detach().to("cpu").numpy()
                chunks.append(logits.astype(np.float32, copy=False))
        if not chunks:
            return np.empty((0, self.n_labels), dtype=np.float32)
        return np.concatenate(chunks, axis=0)

    def _eval_micro_auprc(self, X: Any, rows: np.ndarray, y: np.ndarray) -> float:
        """Compute micro AUPRC over a set of rows.

        Args:
            X: Feature matrix ``(n, d)``.
            rows: Integer row indices to evaluate.
            y: Full label matrix ``(n, K)``.

        Returns:
            Micro AUPRC on ``rows``.
        """
        logits = self._predict_logits(X, rows)
        probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -50.0, 50.0)))
        return micro_auprc(y[rows], probs)

    def fit(
        self,
        X: Any,
        y: np.ndarray,
        verbose: bool = True,
        scaffold_groups: Optional[np.ndarray] = None,
    ) -> "MultilabelMLP":
        """Train the multilabel MLP with optional scaffold-cold early stopping.

        When ``active_fraction`` is set, a fixed per-label negative mask is
        built on the fit rows after the early-stopping carve. Train BCE uses
        only masked cells; early-stop micro-AUPRC still uses the full labels.

        Args:
            X: Feature matrix ``(n, d)``.
            y: Multilabel matrix ``(n, K)`` with values in ``{0, 1}``.
            verbose: If ``True``, print progress.
            scaffold_groups: Per-row Murcko scaffold ids for early stopping.

        Returns:
            ``self``.

        Raises:
            ValueError: If label width mismatches ``n_labels`` or early stopping
                is enabled without scaffold groups.
        """
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        n, d = X.shape
        y_np = np.asarray(y, dtype=np.float32)
        if y_np.ndim != 2 or y_np.shape[0] != n or y_np.shape[1] != self.n_labels:
            raise ValueError(
                f"y shape {y_np.shape} incompatible with X rows={n} "
                f"n_labels={self.n_labels}."
            )
        self.input_dim = int(d)
        y_t = torch.from_numpy(y_np)

        if self.patience > 0:
            if scaffold_groups is None:
                raise ValueError(
                    "Multilabel early stopping requires scaffold_groups."
                )
            scaffold_arr = np.asarray(scaffold_groups)
            if scaffold_arr.shape[0] != n:
                raise ValueError(
                    f"scaffold_groups length {scaffold_arr.shape[0]} != X rows {n}."
                )
            fractions = {
                "train": 1.0 - self.es_val_fraction,
                "es_val": self.es_val_fraction,
            }
            es_split = scaffold_cold_split(scaffold_arr, fractions, seed=self.seed)
            train_rows = es_split["train"]
            val_rows = es_split["es_val"]
            min_es_rows = max(1, int(0.25 * self.es_val_fraction * n))
            early_stopping = (
                train_rows.shape[0] > 0
                and val_rows.shape[0] >= min_es_rows
                and train_rows.shape[0] >= val_rows.shape[0]
            )
            if verbose and early_stopping:
                print(
                    f"[multilabel-mlp] scaffold-cold early-stop holdout: "
                    f"train={train_rows.shape[0]} es_val={val_rows.shape[0]}",
                    flush=True,
                )
            if not early_stopping:
                train_rows = np.arange(n, dtype=np.int64)
                val_rows = np.empty(0, dtype=np.int64)
                if verbose:
                    print(
                        "[multilabel-mlp] early stopping disabled "
                        "(empty scaffold holdout)",
                        flush=True,
                    )
        else:
            train_rows = np.arange(n, dtype=np.int64)
            val_rows = np.empty(0, dtype=np.int64)
            early_stopping = False

        self._standardize_stats(X, train_rows)
        self.net = _MultilabelMLP(
            d,
            self.n_labels,
            self.hidden_dim,
            self.num_layers,
            self.dropout,
        ).to(self.device)
        optimizer = torch.optim.Adam(
            self.net.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )

        y_fit = y_np[train_rows]
        train_mask: Optional[np.ndarray] = None
        mask_t: Optional[torch.Tensor] = None
        if self.active_fraction is not None:
            mask_rng = np.random.default_rng(self.seed)
            train_mask = build_active_fraction_mask(
                y_fit, self.active_fraction, mask_rng
            )
            full_mask = np.zeros((n, self.n_labels), dtype=np.float32)
            full_mask[train_rows] = train_mask.astype(np.float32)
            mask_t = torch.from_numpy(full_mask).to(self.device)
            if verbose:
                positives = y_fit > 0.5
                n_pos_col = positives.sum(axis=0).astype(np.float64)
                n_kept_neg = ((~positives) & train_mask).sum(axis=0).astype(
                    np.float64
                )
                with np.errstate(divide="ignore", invalid="ignore"):
                    dens = np.where(
                        n_pos_col > 0,
                        n_pos_col / np.maximum(n_pos_col + n_kept_neg, 1.0),
                        np.nan,
                    )
                finite_dens = dens[np.isfinite(dens)]
                dens_msg = (
                    f"median={float(np.median(finite_dens)):.3f}"
                    if finite_dens.size
                    else "n/a"
                )
                print(
                    f"[multilabel-mlp] active_fraction={self.active_fraction:.3f} "
                    f"fixed per-label mask; kept-neg median="
                    f"{float(np.median(n_kept_neg)):.0f} "
                    f"effective_pos_frac {dens_msg}",
                    flush=True,
                )

        if train_mask is not None:
            n_pos = (y_fit * train_mask.astype(np.float32)).sum(axis=0)
            n_neg = ((1.0 - y_fit) * train_mask.astype(np.float32)).sum(axis=0)
        else:
            n_pos = y_fit.sum(axis=0)
            n_neg = float(y_fit.shape[0]) - n_pos
        pos_weight_t: Optional[torch.Tensor] = None
        if self.class_weights:
            with np.errstate(divide="ignore", invalid="ignore"):
                weights = np.where(n_pos > 0, n_neg / np.maximum(n_pos, 1.0), 1.0)
            self.pos_weight = weights.astype(np.float32)
            pos_weight_t = torch.from_numpy(self.pos_weight).to(self.device)
            if verbose:
                finite = self.pos_weight[np.isfinite(self.pos_weight)]
                print(
                    f"[multilabel-mlp] per-label pos_weight "
                    f"median={float(np.median(finite)):.2f} "
                    f"max={float(np.max(finite)):.2f}",
                    flush=True,
                )
        else:
            self.pos_weight = None

        if mask_t is None:
            loss_fn = (
                nn.BCEWithLogitsLoss(pos_weight=pos_weight_t)
                if pos_weight_t is not None
                else nn.BCEWithLogitsLoss()
            )
        else:
            loss_fn = (
                nn.BCEWithLogitsLoss(reduction="none", pos_weight=pos_weight_t)
                if pos_weight_t is not None
                else nn.BCEWithLogitsLoss(reduction="none")
            )

        best_score = -math.inf
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
                xb = self._batch_tensor(X, rows)
                yb = y_t[rows].to(self.device)
                optimizer.zero_grad()
                pred = self.net(xb)
                if mask_t is None:
                    loss = loss_fn(pred, yb)
                    batch_loss = float(loss.item())
                else:
                    per_cell = loss_fn(pred, yb)
                    mb = mask_t[rows]
                    kept = mb.sum().clamp_min(1.0)
                    loss = (per_cell * mb).sum() / kept
                    batch_loss = float(loss.item())
                loss.backward()
                optimizer.step()
                running += batch_loss * len(rows)
                seen += len(rows)
                if verbose and (bstart // self.batch_size) % 200 == 0:
                    print(
                        f"[multilabel-mlp] epoch {epoch + 1}/{self.epochs} "
                        f"{seen}/{order.shape[0]} rows  loss={running / max(seen, 1):.4f}",
                        flush=True,
                    )

            val_score = (
                self._eval_micro_auprc(X, val_rows, y_np) if early_stopping else math.nan
            )
            if verbose:
                tail = f"  val_micro_auprc={val_score:.4f}" if early_stopping else ""
                print(
                    f"[multilabel-mlp] epoch {epoch + 1}/{self.epochs} done  "
                    f"train_bce={running / max(seen, 1):.4f}{tail}",
                    flush=True,
                )

            if not early_stopping:
                continue
            if val_score - best_score > self.es_min_delta:
                best_score = val_score
                best_epoch = epoch + 1
                best_state = {
                    k: v.detach().cpu().clone() for k, v in self.net.state_dict().items()
                }
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= self.patience:
                    if verbose:
                        print(
                            f"[multilabel-mlp] early stop at epoch {epoch + 1}; "
                            f"best val_micro_auprc={best_score:.4f} @ epoch {best_epoch}",
                            flush=True,
                        )
                    break

        if best_state is not None:
            self.net.load_state_dict(best_state)
            if verbose:
                print(
                    f"[multilabel-mlp] restored best weights "
                    f"(val_micro_auprc={best_score:.4f} @ epoch {best_epoch})"
                )
        return self

    def predict(self, X: Any) -> np.ndarray:
        """Predict per-label probabilities.

        Args:
            X: Feature matrix ``(n, d)``.

        Returns:
            Probabilities ``(n, K)`` in ``[0, 1]``.
        """
        rows = np.arange(X.shape[0], dtype=np.int64)
        logits = self._predict_logits(X, rows)
        return 1.0 / (1.0 + np.exp(-np.clip(logits, -50.0, 50.0)))

    def _state(self) -> dict[str, Any]:
        """Return the picklable payload for this model.

        Returns:
            Dictionary of network weights and training settings.
        """
        assert self.net is not None
        return {
            "n_labels": self.n_labels,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "patience": self.patience,
            "es_val_fraction": self.es_val_fraction,
            "es_min_delta": self.es_min_delta,
            "class_weights": self.class_weights,
            "active_fraction": self.active_fraction,
            "seed": self.seed,
            "input_dim": self.input_dim,
            "feature_mean": self.feature_mean,
            "feature_std": self.feature_std,
            "pos_weight": self.pos_weight,
            "state_dict": {k: v.detach().cpu() for k, v in self.net.state_dict().items()},
        }

    def save(self, path: str) -> None:
        """Serialize the model to a joblib file.

        Args:
            path: Destination path.

        Returns:
            None.
        """
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
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

    @classmethod
    def load(cls, path: str) -> "MultilabelMLP":
        """Load a multilabel model from a joblib file.

        Args:
            path: Saved model path.

        Returns:
            Restored :class:`MultilabelMLP`.

        Raises:
            ValueError: If the file is not a multilabel MLP payload.
        """
        payload = joblib.load(path)
        if payload.get("model_type") != cls.model_type:
            raise ValueError(
                f"{path} is not a multilabel MLP model "
                f"(found model_type={payload.get('model_type')!r})."
            )
        state = payload["state"]
        model = cls(
            n_labels=int(state["n_labels"]),
            hidden_dim=int(state["hidden_dim"]),
            num_layers=int(state["num_layers"]),
            dropout=float(state["dropout"]),
            batch_size=int(state["batch_size"]),
            epochs=int(state["epochs"]),
            learning_rate=float(state["learning_rate"]),
            weight_decay=float(state["weight_decay"]),
            patience=int(state["patience"]),
            es_val_fraction=float(state["es_val_fraction"]),
            es_min_delta=float(state["es_min_delta"]),
            class_weights=bool(state["class_weights"]),
            active_fraction=state.get("active_fraction"),
            seed=int(state["seed"]),
        )
        model.input_dim = int(state["input_dim"])
        model.feature_mean = state["feature_mean"]
        model.feature_std = state["feature_std"]
        model.pos_weight = state["pos_weight"]
        model.metadata = dict(payload.get("metadata") or {})
        assert model.input_dim is not None
        model.net = _MultilabelMLP(
            model.input_dim,
            model.n_labels,
            model.hidden_dim,
            model.num_layers,
            model.dropout,
        )
        model.net.load_state_dict(state["state_dict"])
        model.net.to(model.device)
        model.net.eval()
        return model


def load_model(path: str) -> MultilabelMLP:
    """Load a multilabel joblib model.

    Args:
        path: Saved model path.

    Returns:
        Restored multilabel MLP.
    """
    return MultilabelMLP.load(path)
