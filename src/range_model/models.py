"""Shared-trunk dual-head MLP for IC50/EC50 potency-range classification."""

from __future__ import annotations

import os
from typing import Any, Optional

import joblib
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

import config
from src.embeddings import select_device
from src.range_model import config as range_config
from src.range_model.bins import BIN_LABELS, N_BINS, RANGE_EDGES_NM
from src.range_model.metrics import compute_range_metrics
from src.splits import cold_protein_split, double_cold_split, time_protein_split

_PREDICT_CHUNK: int = 50_000
_IGNORE_INDEX: int = -100


class _RangeMLPNet(nn.Module):
    """Feed-forward trunk with separate IC50 and EC50 classification heads."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
        n_bins: int,
    ) -> None:
        """Build the shared trunk and two ``n_bins``-way heads.

        Args:
            input_dim: Feature dimension.
            hidden_dim: Hidden layer width.
            num_layers: Number of hidden layers.
            dropout: Dropout probability.
            n_bins: Number of potency classes per head.
        """
        super().__init__()
        layers: list[nn.Module] = []
        prev = input_dim
        for _ in range(num_layers):
            layers += [nn.Linear(prev, hidden_dim), nn.ReLU(), nn.Dropout(dropout)]
            prev = hidden_dim
        self.trunk = nn.Sequential(*layers)
        self.ic50_head = nn.Linear(prev, n_bins)
        self.ec50_head = nn.Linear(prev, n_bins)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Run a forward pass.

        Args:
            x: Input batch ``(batch, input_dim)``.

        Returns:
            ``(ic50_logits, ec50_logits)`` each ``(batch, n_bins)``.
        """
        hidden = self.trunk(x)
        return self.ic50_head(hidden), self.ec50_head(hidden)


def _class_weights_for_head(
    y_bin: np.ndarray,
    head_idx: np.ndarray,
    head: int,
    n_bins: int,
) -> Optional[torch.Tensor]:
    """Inverse-frequency class weights for one assay head.

    Args:
        y_bin: Bin labels on fit rows.
        head_idx: Head ids on fit rows.
        head: Head index to weight.
        n_bins: Number of classes.

    Returns:
        Float tensor of shape ``(n_bins,)``, or ``None`` when the head has no
        rows or a zero count would make weights undefined.
    """
    mask = head_idx == head
    if not np.any(mask):
        return None
    counts = np.bincount(y_bin[mask].astype(np.int64), minlength=n_bins).astype(
        np.float64
    )
    if np.any(counts <= 0):
        counts = np.maximum(counts, 1.0)
    weights = counts.sum() / (n_bins * counts)
    return torch.tensor(weights, dtype=torch.float32)


def masked_range_ce_loss(
    ic50_logits: torch.Tensor,
    ec50_logits: torch.Tensor,
    y_bin: torch.Tensor,
    head_idx: torch.Tensor,
    *,
    ic50_weight: Optional[torch.Tensor] = None,
    ec50_weight: Optional[torch.Tensor] = None,
    row_weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Sum of per-head cross-entropies with inactive-head masking.

    Args:
        ic50_logits: Logits ``(batch, n_bins)`` for the IC50 head.
        ec50_logits: Logits ``(batch, n_bins)`` for the EC50 head.
        y_bin: Bin targets ``(batch,)``.
        head_idx: Head ids ``(batch,)`` (``0``=IC50, ``1``=EC50).
        ic50_weight: Optional class weights for the IC50 head.
        ec50_weight: Optional class weights for the EC50 head.
        row_weight: Optional per-row multipliers ``(batch,)``.

    Returns:
        Scalar loss (mean over rows that contribute to either head).
    """
    targets_ic = y_bin.clone()
    targets_ec = y_bin.clone()
    targets_ic = targets_ic.masked_fill(head_idx != range_config.HEAD_IC50, _IGNORE_INDEX)
    targets_ec = targets_ec.masked_fill(head_idx != range_config.HEAD_EC50, _IGNORE_INDEX)

    loss_ic = F.cross_entropy(
        ic50_logits,
        targets_ic,
        weight=ic50_weight,
        ignore_index=_IGNORE_INDEX,
        reduction="none",
    )
    loss_ec = F.cross_entropy(
        ec50_logits,
        targets_ec,
        weight=ec50_weight,
        ignore_index=_IGNORE_INDEX,
        reduction="none",
    )
    per_row = loss_ic + loss_ec
    active = (head_idx == range_config.HEAD_IC50) | (head_idx == range_config.HEAD_EC50)
    if row_weight is not None:
        per_row = per_row * row_weight
    if not bool(torch.any(active)):
        return per_row.sum() * 0.0
    return per_row[active].mean()


class RangeMLP:
    """Dual-head potency-range classifier with shared trunk."""

    model_type = "range_mlp"

    def __init__(
        self,
        hidden_dim: int = int(range_config.RANGE_MLP_DEFAULTS["hidden_dim"]),
        num_layers: int = int(range_config.RANGE_MLP_DEFAULTS["num_layers"]),
        dropout: float = float(range_config.RANGE_MLP_DEFAULTS["dropout"]),
        batch_size: int = int(range_config.RANGE_MLP_DEFAULTS["batch_size"]),
        epochs: int = int(range_config.RANGE_MLP_DEFAULTS["epochs"]),
        learning_rate: float = float(range_config.RANGE_MLP_DEFAULTS["learning_rate"]),
        weight_decay: float = float(range_config.RANGE_MLP_DEFAULTS["weight_decay"]),
        patience: int = int(range_config.RANGE_MLP_DEFAULTS["patience"]),
        es_val_fraction: float = float(
            range_config.RANGE_MLP_DEFAULTS["es_val_fraction"]
        ),
        es_min_delta: float = float(range_config.RANGE_MLP_DEFAULTS["es_min_delta"]),
        es_metric: str = str(range_config.RANGE_MLP_DEFAULTS["es_metric"]),
        class_weights: bool = bool(range_config.RANGE_MLP_DEFAULTS["class_weights"]),
        n_bins: int = N_BINS,
        seed: int = config.RANDOM_SEED,
        device: Optional[torch.device] = None,
    ) -> None:
        """Configure the range MLP and training schedule.

        Args:
            hidden_dim: Hidden layer width.
            num_layers: Number of hidden layers.
            dropout: Dropout probability.
            batch_size: Minibatch size.
            epochs: Maximum training epochs.
            learning_rate: Adam learning rate.
            weight_decay: Adam weight decay.
            patience: Early-stopping patience (``0`` disables).
            es_val_fraction: Target fraction for the ES holdout size bar.
            es_min_delta: Minimum macro-F1 improvement to reset patience.
            es_metric: Currently only ``macro_f1`` is used for selection.
            class_weights: If ``True``, use inverse-frequency weights per head.
            n_bins: Number of potency classes.
            seed: RNG seed.
            device: Optional torch device.
        """
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
        self.es_metric = str(es_metric)
        self.class_weights = bool(class_weights)
        self.n_bins = int(n_bins)
        self.seed = int(seed)
        self.device = device or select_device()
        self.net: Optional[_RangeMLPNet] = None
        self.input_dim: Optional[int] = None
        self.feature_mean: Optional[np.ndarray] = None
        self.feature_std: Optional[np.ndarray] = None
        self.metadata: dict[str, Any] = {}
        self.ic50_class_weight: Optional[np.ndarray] = None
        self.ec50_class_weight: Optional[np.ndarray] = None

    def _standardize_stats(self, X: Any, rows: np.ndarray) -> None:
        """Fit mean/std on selected rows.

        Args:
            X: Feature matrix.
            rows: Row indices used for statistics.

        Returns:
            None.
        """
        sample = np.asarray(X[rows], dtype=np.float64)
        mean = sample.mean(axis=0)
        std = sample.std(axis=0)
        std = np.where(std < 1e-6, 1.0, std)
        self.feature_mean = mean.astype(np.float32)
        self.feature_std = std.astype(np.float32)

    def _transform(self, batch: np.ndarray) -> torch.Tensor:
        """Standardize a numpy batch and move it to the model device.

        Args:
            batch: Dense float features.

        Returns:
            Torch float tensor on ``self.device``.
        """
        assert self.feature_mean is not None and self.feature_std is not None
        scaled = (batch - self.feature_mean) / self.feature_std
        return torch.from_numpy(scaled.astype(np.float32, copy=False)).to(self.device)

    def _carve_es(
        self,
        n: int,
        groups: np.ndarray,
        scaffold_groups: Optional[np.ndarray],
        years: Optional[np.ndarray],
        verbose: bool,
    ) -> tuple[np.ndarray, np.ndarray, bool]:
        """Carve an early-stopping holdout (time-protein preferred).

        Args:
            n: Number of fit rows.
            groups: Protein group ids.
            scaffold_groups: Optional scaffold ids.
            years: Optional publication years.
            verbose: Whether to print progress.

        Returns:
            ``(train_rows, val_rows, early_stopping_enabled)``.
        """
        if self.patience <= 0:
            return np.arange(n, dtype=np.int64), np.empty(0, dtype=np.int64), False

        fractions = {
            "train": 1.0 - self.es_val_fraction,
            "es_val": self.es_val_fraction,
        }
        min_es_rows = max(1, int(0.25 * self.es_val_fraction * n))

        def _usable(split: dict[str, np.ndarray]) -> bool:
            n_train = int(split["train"].shape[0])
            n_val = int(split["es_val"].shape[0])
            return n_train > 0 and n_val >= min_es_rows and n_train >= n_val

        es_split: Optional[dict[str, np.ndarray]] = None
        es_kind = "time-protein"
        if years is not None and int(np.sum(years >= 0)) > 0:
            try:
                two_way = time_protein_split(
                    years,
                    groups,
                    n_years=int(config.TIME_PROTEIN_YEARS),
                    min_val_rows=min_es_rows,
                )
                candidate = {"train": two_way["train"], "es_val": two_way["val"]}
                if _usable(candidate):
                    es_split = candidate
            except ValueError:
                es_split = None

        if es_split is None:
            if verbose and years is not None:
                print(
                    "[range-mlp] time-protein ES holdout too small / unavailable; "
                    "falling back to double-cold / cold-protein",
                    flush=True,
                )
            if scaffold_groups is not None:
                es_split = double_cold_split(
                    groups, scaffold_groups, fractions, seed=self.seed
                )
                if _usable(es_split):
                    es_kind = "double-cold"
                else:
                    if verbose:
                        print(
                            "[range-mlp] double-cold ES holdout too small; "
                            "falling back to cold-protein",
                            flush=True,
                        )
                    es_split = cold_protein_split(groups, fractions, seed=self.seed)
                    es_kind = "cold-protein"
            else:
                es_split = cold_protein_split(groups, fractions, seed=self.seed)
                es_kind = "cold-protein"

        train_rows = es_split["train"]
        val_rows = es_split["es_val"]
        early_stopping = _usable(es_split)
        if verbose and early_stopping:
            print(
                f"[range-mlp] {es_kind} early-stop holdout: "
                f"train={train_rows.shape[0]:,}  es_val={val_rows.shape[0]:,}",
                flush=True,
            )
        if not early_stopping:
            if verbose:
                print(
                    "[range-mlp] early stopping disabled (empty cold holdout)",
                    flush=True,
                )
            return np.arange(n, dtype=np.int64), np.empty(0, dtype=np.int64), False
        return train_rows, val_rows, True

    def fit(
        self,
        X: Any,
        y_bin: np.ndarray,
        head_idx: np.ndarray,
        *,
        loss_weight: Optional[np.ndarray] = None,
        groups: Optional[np.ndarray] = None,
        scaffold_groups: Optional[np.ndarray] = None,
        years: Optional[np.ndarray] = None,
        verbose: bool = True,
    ) -> "RangeMLP":
        """Train the dual-head range MLP.

        Args:
            X: Feature matrix ``(n, d)``.
            y_bin: Potency bin labels ``(n,)``.
            head_idx: Head ids ``(n,)``.
            loss_weight: Optional per-row loss weights ``(n,)``.
            groups: Protein group ids (required when early stopping is on).
            scaffold_groups: Optional scaffold ids for double-cold ES.
            years: Optional publication years for time-protein ES.
            verbose: Whether to print progress.

        Returns:
            ``self``.
        """
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        n, d = X.shape
        self.input_dim = int(d)
        y_np = np.asarray(y_bin, dtype=np.int64)
        head_np = np.asarray(head_idx, dtype=np.int64)
        weight_np = (
            np.ones(n, dtype=np.float32)
            if loss_weight is None
            else np.asarray(loss_weight, dtype=np.float32)
        )

        groups_arr: Optional[np.ndarray] = None
        if groups is not None:
            groups_arr = np.asarray(groups)
            if groups_arr.shape[0] != n:
                raise ValueError(
                    f"groups length {groups_arr.shape[0]} does not match X rows {n}."
                )

        if self.patience > 0 and groups_arr is None:
            raise ValueError(
                "Range MLP early stopping requires protein groups for a cold holdout."
            )

        if groups_arr is None:
            train_rows = np.arange(n, dtype=np.int64)
            val_rows = np.empty(0, dtype=np.int64)
            early_stopping = False
        else:
            scaf = None if scaffold_groups is None else np.asarray(scaffold_groups)
            yrs = None if years is None else np.asarray(years)
            train_rows, val_rows, early_stopping = self._carve_es(
                n, groups_arr, scaf, yrs, verbose
            )

        self._standardize_stats(X, train_rows)
        self.net = _RangeMLPNet(
            d, self.hidden_dim, self.num_layers, self.dropout, self.n_bins
        ).to(self.device)
        optimizer = torch.optim.Adam(
            self.net.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )

        ic_w_t: Optional[torch.Tensor] = None
        ec_w_t: Optional[torch.Tensor] = None
        if self.class_weights:
            ic_w = _class_weights_for_head(
                y_np[train_rows], head_np[train_rows], range_config.HEAD_IC50, self.n_bins
            )
            ec_w = _class_weights_for_head(
                y_np[train_rows], head_np[train_rows], range_config.HEAD_EC50, self.n_bins
            )
            if ic_w is not None:
                self.ic50_class_weight = ic_w.detach().cpu().numpy()
                ic_w_t = ic_w.to(self.device)
            if ec_w is not None:
                self.ec50_class_weight = ec_w.detach().cpu().numpy()
                ec_w_t = ec_w.to(self.device)

        best_state: Optional[dict[str, torch.Tensor]] = None
        best_score = -float("inf")
        best_epoch = 0
        no_improve = 0

        for epoch in range(self.epochs):
            self.net.train()
            order = rng.permutation(train_rows)
            running = 0.0
            n_batches = 0
            for start in range(0, order.shape[0], self.batch_size):
                batch_rows = order[start : start + self.batch_size]
                xb = self._transform(np.asarray(X[batch_rows], dtype=np.float32))
                yb = torch.from_numpy(y_np[batch_rows]).to(self.device)
                hb = torch.from_numpy(head_np[batch_rows]).to(self.device)
                wb = torch.from_numpy(weight_np[batch_rows]).to(self.device)
                ic_logits, ec_logits = self.net(xb)
                loss = masked_range_ce_loss(
                    ic_logits,
                    ec_logits,
                    yb,
                    hb,
                    ic50_weight=ic_w_t,
                    ec50_weight=ec_w_t,
                    row_weight=wb,
                )
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                running += float(loss.detach().cpu())
                n_batches += 1
            train_loss = running / max(1, n_batches)

            val_msg = ""
            if early_stopping and val_rows.size > 0:
                pred = self.predict_bins(
                    X, head_idx=head_np[val_rows], rows=val_rows
                )
                metrics = compute_range_metrics(
                    y_np[val_rows],
                    pred,
                    head_np[val_rows],
                    n_bins=self.n_bins,
                )
                score = float(metrics["mean_macro_f1"])
                val_msg = (
                    f"  val_mean_macro_f1={score:.4f}  "
                    + " ".join(
                        f"val_{name}_f1={head['macro_f1']:.4f}"
                        for name, head in metrics["per_head"].items()
                        if head["n"] > 0
                    )
                )
                if score > best_score + self.es_min_delta:
                    best_score = score
                    best_epoch = epoch + 1
                    best_state = {
                        k: v.detach().cpu().clone()
                        for k, v in self.net.state_dict().items()
                    }
                    no_improve = 0
                else:
                    no_improve += 1

            if verbose:
                print(
                    f"[range-mlp] epoch {epoch + 1}/{self.epochs}  "
                    f"train_ce={train_loss:.4f}{val_msg}",
                    flush=True,
                )
            if early_stopping and no_improve >= self.patience:
                if verbose:
                    print(
                        f"[range-mlp] early stop at epoch {epoch + 1}; "
                        f"best val_mean_macro_f1={best_score:.4f} @ epoch {best_epoch}",
                        flush=True,
                    )
                break

        if best_state is not None:
            self.net.load_state_dict(best_state)
            if verbose:
                print(
                    f"[range-mlp] restored best weights "
                    f"(val_mean_macro_f1={best_score:.4f} @ epoch {best_epoch})",
                    flush=True,
                )
        return self

    def _predict_logits(
        self, X: Any, rows: Optional[np.ndarray] = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Predict IC50/EC50 logits for selected rows.

        Args:
            X: Feature matrix.
            rows: Optional row indices; defaults to all rows.

        Returns:
            ``(ic50_logits, ec50_logits)`` each ``(n_sel, n_bins)``.
        """
        assert self.net is not None
        if rows is None:
            rows = np.arange(X.shape[0], dtype=np.int64)
        else:
            rows = np.asarray(rows, dtype=np.int64)
        ic_parts: list[np.ndarray] = []
        ec_parts: list[np.ndarray] = []
        self.net.eval()
        with torch.no_grad():
            for start in range(0, rows.shape[0], _PREDICT_CHUNK):
                batch_rows = rows[start : start + _PREDICT_CHUNK]
                xb = self._transform(np.asarray(X[batch_rows], dtype=np.float32))
                ic_logits, ec_logits = self.net(xb)
                ic_parts.append(ic_logits.detach().cpu().numpy())
                ec_parts.append(ec_logits.detach().cpu().numpy())
        return np.concatenate(ic_parts, axis=0), np.concatenate(ec_parts, axis=0)

    def predict_proba(
        self,
        X: Any,
        head_idx: np.ndarray,
        rows: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Predict class probabilities for the active head of each row.

        Args:
            X: Feature matrix.
            head_idx: Per-row head ids aligned with ``rows`` (or all rows).
            rows: Optional row indices into ``X``.

        Returns:
            Probabilities ``(n_sel, n_bins)``.
        """
        if rows is None:
            rows = np.arange(X.shape[0], dtype=np.int64)
            heads = np.asarray(head_idx, dtype=np.int64)
        else:
            rows = np.asarray(rows, dtype=np.int64)
            heads = np.asarray(head_idx, dtype=np.int64)
            if heads.shape[0] == X.shape[0]:
                heads = heads[rows]
        ic_logits, ec_logits = self._predict_logits(X, rows)
        ic_prob = _softmax_np(ic_logits)
        ec_prob = _softmax_np(ec_logits)
        out = np.empty((rows.shape[0], self.n_bins), dtype=np.float32)
        ic_mask = heads == range_config.HEAD_IC50
        ec_mask = heads == range_config.HEAD_EC50
        out[ic_mask] = ic_prob[ic_mask]
        out[ec_mask] = ec_prob[ec_mask]
        # Rows with unknown heads get a uniform distribution.
        other = ~(ic_mask | ec_mask)
        if np.any(other):
            out[other] = 1.0 / float(self.n_bins)
        return out

    def predict_bins(
        self,
        X: Any,
        head_idx: np.ndarray,
        rows: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Predict the most probable potency bin for each row's active head.

        Args:
            X: Feature matrix.
            head_idx: Per-selected-row head ids (or full-length when ``rows``
                indexes into them).
            rows: Optional row indices into ``X``.

        Returns:
            Predicted bin indices ``(n_sel,)``.
        """
        probs = self.predict_proba(X, head_idx, rows=rows)
        return probs.argmax(axis=1).astype(np.int64)

    def expected_bin(
        self,
        X: Any,
        head_idx: np.ndarray,
        rows: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Return the probability-weighted expected bin index.

        Args:
            X: Feature matrix.
            head_idx: Per-row head ids.
            rows: Optional row indices into ``X``.

        Returns:
            Float expected bins ``(n_sel,)``.
        """
        probs = self.predict_proba(X, head_idx, rows=rows)
        centers = np.arange(self.n_bins, dtype=np.float64)
        return (probs * centers).sum(axis=1).astype(np.float32)

    def save(self, path: str) -> None:
        """Serialize the trained model to ``path``.

        Args:
            path: Destination joblib path.

        Returns:
            None.
        """
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        joblib.dump(
            {
                "model_type": self.model_type,
                "state": self._state(),
                "metadata": self.metadata,
            },
            path,
        )

    def _state(self) -> dict[str, Any]:
        """Return the picklable payload.

        Returns:
            State dictionary.
        """
        assert self.net is not None and self.input_dim is not None
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
            "patience": self.patience,
            "es_val_fraction": self.es_val_fraction,
            "es_min_delta": self.es_min_delta,
            "es_metric": self.es_metric,
            "class_weights": self.class_weights,
            "n_bins": self.n_bins,
            "seed": self.seed,
            "feature_mean": self.feature_mean,
            "feature_std": self.feature_std,
            "ic50_class_weight": self.ic50_class_weight,
            "ec50_class_weight": self.ec50_class_weight,
            "range_edges_nm": list(RANGE_EDGES_NM),
            "bin_labels": list(BIN_LABELS),
            "heads": list(range_config.HEAD_NAMES),
        }


def _softmax_np(logits: np.ndarray) -> np.ndarray:
    """Numerically stable softmax over the last axis.

    Args:
        logits: Array ``(..., n_bins)``.

    Returns:
        Probabilities with the same shape.
    """
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return (exp / exp.sum(axis=-1, keepdims=True)).astype(np.float32)


def load_range_model(path: str) -> RangeMLP:
    """Load a range model previously written by :meth:`RangeMLP.save`.

    Args:
        path: Path to the joblib file.

    Returns:
        Restored :class:`RangeMLP` with metadata populated.

    Raises:
        ValueError: If the file is not a range MLP checkpoint.
    """
    blob = joblib.load(path)
    if blob.get("model_type") != RangeMLP.model_type:
        raise ValueError(f"Unsupported model_type {blob.get('model_type')!r}.")
    state = blob["state"]
    model = RangeMLP(
        hidden_dim=int(state["hidden_dim"]),
        num_layers=int(state["num_layers"]),
        dropout=float(state["dropout"]),
        batch_size=int(state["batch_size"]),
        epochs=int(state["epochs"]),
        learning_rate=float(state["learning_rate"]),
        weight_decay=float(state["weight_decay"]),
        patience=int(state.get("patience", 0)),
        es_val_fraction=float(
            state.get("es_val_fraction", range_config.RANGE_MLP_DEFAULTS["es_val_fraction"])
        ),
        es_min_delta=float(
            state.get("es_min_delta", range_config.RANGE_MLP_DEFAULTS["es_min_delta"])
        ),
        es_metric=str(state.get("es_metric", "macro_f1")),
        class_weights=bool(state.get("class_weights", True)),
        n_bins=int(state.get("n_bins", N_BINS)),
        seed=int(state["seed"]),
    )
    model.input_dim = int(state["input_dim"])
    model.feature_mean = state["feature_mean"]
    model.feature_std = state["feature_std"]
    model.ic50_class_weight = state.get("ic50_class_weight")
    model.ec50_class_weight = state.get("ec50_class_weight")
    model.net = _RangeMLPNet(
        model.input_dim,
        model.hidden_dim,
        model.num_layers,
        model.dropout,
        model.n_bins,
    )
    model.net.load_state_dict(state["state_dict"])
    model.net.to(model.device)
    model.metadata = dict(blob.get("metadata") or {})
    return model
