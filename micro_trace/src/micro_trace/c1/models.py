"""Phase P1 (C1: Future FC Prediction) Model Training and Benchmark Architecture.

Implements candidate model families and shared baselines (ADR-001, ADR-009, PRD Section 4.6):
1. Baselines:
   - B0: Training-set mean target baseline
   - B1 / M0: Most recent FC persistence reference (zero-delta in delta formulation)
2. Tabular ML Families:
   - Elastic Net (L1+L2 regularized linear model with cross-validated penalty tuning)
   - Random Forest Regressor
   - HistGradientBoosting Regressor (native missingness handling and gradient boosting)
3. Deep Architecture (MicroTrace-Attn):
   - Continuous-Time Residual Cross-Attention network over patient visit history (1..t)
   - Continuous-time decay Fourier embeddings phi(Delta t)
   - Cross-attention from current state query to history keys/values
   - Non-negotiable Linear Residual Skip Connection
   - Flexible target modes: "level" (direct future FC) or "delta" (change in FC)
   - Support for L1 loss, SmoothL1 loss, and Huber loss
4. Reproducible Evaluation:
   - 5-fold participant-grouped CV (dev N=371, 75 participants across 5 folds of 15)
   - Untouched test partition (test N=74, 18 participants held 100% untouched)
   - Reconstructed metrics: MAE on log10(FC + 1), RMSE, R2, Spearman rho, and Directional Accuracy.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNetCV
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger("micro_trace.c1.models")


# ==============================================================================
# 1. Regression Metrics & Directional Accuracy
# ==============================================================================

def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Calculate standard regression metrics on log10(FC + 1).

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth target values.
    y_pred : np.ndarray
        Model predicted values.

    Returns
    -------
    dict[str, float]
        Dictionary with MAE, RMSE, R2, Spearman rho, and Median AE.
    """
    y_t = np.asarray(y_true, dtype=np.float64).ravel()
    y_p = np.asarray(y_pred, dtype=np.float64).ravel()

    if len(y_t) == 0:
        return {"mae": 0.0, "rmse": 0.0, "r2": 0.0, "spearman_rho": 0.0, "median_ae": 0.0}

    errors = y_p - y_t
    abs_errors = np.abs(errors)
    mae = float(np.mean(abs_errors))
    rmse = float(np.sqrt(np.mean(errors ** 2)))
    median_ae = float(np.median(abs_errors))

    # R-squared calculation
    ss_tot = float(np.sum((y_t - np.mean(y_t)) ** 2))
    ss_res = float(np.sum(errors ** 2))
    r2 = float(1.0 - (ss_res / (ss_tot + 1e-12))) if ss_tot > 1e-12 else 0.0

    # Spearman rank correlation
    if len(y_t) >= 2 and np.std(y_t) > 1e-12 and np.std(y_p) > 1e-12:
        try:
            s_res = spearmanr(y_t, y_p)
            rho = float(s_res.statistic) if pd.notna(s_res.statistic) else 0.0
        except Exception:
            rho = 0.0
    else:
        rho = 0.0

    return {
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "r2": round(r2, 4),
        "spearman_rho": round(rho, 4),
        "median_ae": round(median_ae, 4),
    }


def compute_directional_accuracy(
    delta_y_true: np.ndarray,
    delta_y_pred: np.ndarray,
    tol: float = 1e-6,
) -> float:
    """Calculate directional accuracy percentage on non-zero transitions.

    Evaluates:
        100.0 * mean(sign(delta_y_pred) == sign(delta_y_true))
    for all pairs where |delta_y_true| > tol.

    Parameters
    ----------
    delta_y_true : np.ndarray
        True change in calprotectin: y_{t+1} - y_t.
    delta_y_pred : np.ndarray
        Predicted change: hat{delta_y}.
    tol : float
        Threshold for non-zero transitions.

    Returns
    -------
    float
        Percentage of correct directional transitions [0.0 - 100.0].
    """
    d_t = np.asarray(delta_y_true, dtype=np.float64).ravel()
    d_p = np.asarray(delta_y_pred, dtype=np.float64).ravel()

    if len(d_t) == 0:
        return 0.0

    nonzero_mask = np.abs(d_t) > tol
    if not np.any(nonzero_mask):
        return 0.0

    t_signs = np.sign(d_t[nonzero_mask])
    p_signs = np.where(np.abs(d_p[nonzero_mask]) > tol, np.sign(d_p[nonzero_mask]), 0.0)
    correct = (t_signs == p_signs)
    return round(float(np.mean(correct) * 100.0), 2)


def compute_delta_regression_metrics(
    y_true_reconstructed: np.ndarray,
    y_pred_reconstructed: np.ndarray,
    delta_true: np.ndarray,
    delta_pred: np.ndarray,
    tol: float = 1e-6,
) -> dict[str, float]:
    """Calculate combined delta and reconstructed calprotectin metrics."""
    rec_metrics = compute_regression_metrics(y_true_reconstructed, y_pred_reconstructed)
    delta_metrics = compute_regression_metrics(delta_true, delta_pred)
    dir_acc = compute_directional_accuracy(delta_true, delta_pred, tol=tol)

    return {
        "mae_reconstructed": rec_metrics["mae"],
        "rmse_reconstructed": rec_metrics["rmse"],
        "r2_reconstructed": rec_metrics["r2"],
        "spearman_rho_reconstructed": rec_metrics["spearman_rho"],
        "median_ae_reconstructed": rec_metrics["median_ae"],
        "mae_delta": delta_metrics["mae"],
        "rmse_delta": delta_metrics["rmse"],
        "r2_delta": delta_metrics["r2"],
        "spearman_rho_delta": delta_metrics["spearman_rho"],
        "directional_accuracy": dir_acc,
    }


# ==============================================================================
# 2. Baseline Regressors
# ==============================================================================

class MeanBaseline(BaseEstimator, RegressorMixin):
    """B0: Training-set mean target predictor."""

    def __init__(self) -> None:
        self.mean_: float = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> MeanBaseline:
        y_arr = np.asarray(y, dtype=np.float64)
        self.mean_ = float(np.mean(y_arr)) if len(y_arr) > 0 else 0.0
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        n_samples = len(X)
        return np.full(n_samples, self.mean_, dtype=np.float64)


class PersistenceBaseline(BaseEstimator, RegressorMixin):
    """B1 / M0: Most recent FC persistence reference.

    In 'level' mode: extracts index FC from feature matrix (column 0 by convention).
    In 'delta' mode: predicts 0.0 (zero calprotectin transition).
    """

    def __init__(self, fc_col_idx: int = 0, target_mode: str = "level") -> None:
        self.fc_col_idx = fc_col_idx
        self.target_mode = target_mode
        self.fallback_val_: float = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> PersistenceBaseline:
        if self.target_mode == "delta":
            self.fallback_val_ = 0.0
            return self

        X_arr = np.asarray(X, dtype=np.float64)
        if X_arr.ndim == 2 and X_arr.shape[1] > self.fc_col_idx:
            self.fallback_val_ = float(np.median(X_arr[:, self.fc_col_idx]))
        elif len(y) > 0:
            self.fallback_val_ = float(np.median(y))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.target_mode == "delta":
            return np.zeros(len(X), dtype=np.float64)

        X_arr = np.asarray(X, dtype=np.float64)
        if X_arr.ndim == 2 and X_arr.shape[1] > self.fc_col_idx:
            preds = X_arr[:, self.fc_col_idx].copy()
            preds = np.where(np.isfinite(preds), preds, self.fallback_val_)
            return preds
        return np.full(len(X_arr), self.fallback_val_, dtype=np.float64)


# ==============================================================================
# 3. Tabular Machine Learning Regressors
# ==============================================================================

class ElasticNetBenchmark(BaseEstimator, RegressorMixin):
    """Standardized Elastic Net regressor with cross-validated penalty tuning."""

    def __init__(
        self,
        l1_ratios: tuple[float, ...] = (0.1, 0.5, 0.7, 0.9, 0.99),
        cv: int = 3,
        n_alphas: int = 30,
        alphas: np.ndarray | list[float] | int | None = None,
        tol: float = 1e-3,
        max_iter: int = 1500,
        random_state: int = 42,
    ) -> None:
        self.l1_ratios = l1_ratios
        self.cv = cv
        self.n_alphas = n_alphas
        self.alphas = alphas if alphas is not None else n_alphas
        self.tol = tol
        self.max_iter = max_iter
        self.random_state = random_state
        self.scaler_: StandardScaler | None = None
        self.model_: ElasticNetCV | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> ElasticNetBenchmark:
        X_arr = np.asarray(X, dtype=np.float64)
        y_arr = np.asarray(y, dtype=np.float64).ravel()

        self.scaler_ = StandardScaler()
        X_scaled = self.scaler_.fit_transform(X_arr)
        X_scaled = np.nan_to_num(X_scaled, nan=0.0)

        self.model_ = ElasticNetCV(
            l1_ratio=list(self.l1_ratios),
            cv=self.cv,
            alphas=self.alphas,
            tol=self.tol,
            max_iter=self.max_iter,
            random_state=self.random_state,
            n_jobs=-1,
        )
        self.model_.fit(X_scaled, y_arr)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.scaler_ is None or self.model_ is None:
            raise RuntimeError("ElasticNetBenchmark must be fitted before predict()")
        X_arr = np.asarray(X, dtype=np.float64)
        X_scaled = np.nan_to_num(self.scaler_.transform(X_arr), nan=0.0)
        return self.model_.predict(X_scaled)


class RandomForestBenchmark(BaseEstimator, RegressorMixin):
    """Standardized Random Forest regressor with depth and leaf regularizers."""

    def __init__(
        self,
        n_estimators: int = 150,
        max_depth: int = 8,
        min_samples_leaf: int = 3,
        random_state: int = 42,
    ) -> None:
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.model_: RandomForestRegressor | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> RandomForestBenchmark:
        X_arr = np.nan_to_num(np.asarray(X, dtype=np.float64), nan=0.0)
        y_arr = np.asarray(y, dtype=np.float64).ravel()

        self.model_ = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            random_state=self.random_state,
            n_jobs=-1,
        )
        self.model_.fit(X_arr, y_arr)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.model_ is None:
            raise RuntimeError("RandomForestBenchmark must be fitted before predict()")
        X_arr = np.nan_to_num(np.asarray(X, dtype=np.float64), nan=0.0)
        return self.model_.predict(X_arr)


class HistGradientBoostingBenchmark(BaseEstimator, RegressorMixin):
    """Histogram-based Gradient Boosting with native missingness and L2 penalty."""

    def __init__(
        self,
        max_iter: int = 120,
        learning_rate: float = 0.04,
        max_depth: int = 5,
        min_samples_leaf: int = 8,
        l2_regularization: float = 1.0,
        random_state: int = 42,
    ) -> None:
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.l2_regularization = l2_regularization
        self.random_state = random_state
        self.model_: HistGradientBoostingRegressor | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> HistGradientBoostingBenchmark:
        X_arr = np.asarray(X, dtype=np.float64)
        y_arr = np.asarray(y, dtype=np.float64).ravel()

        self.model_ = HistGradientBoostingRegressor(
            max_iter=self.max_iter,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            l2_regularization=self.l2_regularization,
            random_state=self.random_state,
        )
        self.model_.fit(X_arr, y_arr)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.model_ is None:
            raise RuntimeError("HistGradientBoostingBenchmark must be fitted before predict()")
        X_arr = np.asarray(X, dtype=np.float64)
        return self.model_.predict(X_arr)


# ==============================================================================
# 4. Deep Architecture: MicroTrace-Attn
# ==============================================================================

class ContinuousTimeEmbedding(nn.Module):
    """Continuous-time Fourier embedding phi(Delta t) for irregular visit cadences."""

    def __init__(self, embed_dim: int, num_frequencies: int = 16) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_frequencies = num_frequencies
        freq_grid = torch.exp(torch.linspace(float(np.log(1e-3)), float(np.log(1.0)), num_frequencies))
        self.register_buffer("frequencies", freq_grid)
        self.proj = nn.Sequential(
            nn.Linear(num_frequencies * 2, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, delta_t: torch.Tensor) -> torch.Tensor:
        """Embed elapsed time deltas in days. Shape (B, T) -> (B, T, embed_dim)."""
        if delta_t.ndim == 2:
            delta_t = delta_t.unsqueeze(-1)
        angles = delta_t * self.frequencies.view(1, 1, -1)
        fourier = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        return self.proj(fourier)


class MicroTraceAttn(nn.Module):
    """Continuous-Time Residual Cross-Attention Architecture for longitudinal FC forecasting.

    Key Architectural Properties:
    1. Attends over patient visit history (1..t) with continuous-time decay phi(Delta t).
    2. Cross-attention connects query (current anchor + multi-omics) to history keys/values.
    3. Non-negotiable Linear Residual Skip Connection:
       - In 'level' mode: anchored to index FC (w[0]=1.0), guaranteeing it cannot underperform persistence.
       - In 'delta' mode: anchored to 0.0 with optional linear trend adjustment.
    """

    def __init__(
        self,
        d_seq: int = 15,
        d_anchor: int = 6,
        d_omics: int = 524,
        embed_dim: int = 64,
        num_heads: int = 4,
        dropout: float = 0.1,
        target_mode: str = "level",
    ) -> None:
        super().__init__()
        self.d_seq = d_seq
        self.d_anchor = d_anchor
        self.d_omics = d_omics
        self.embed_dim = embed_dim
        self.target_mode = target_mode

        # 1. Sequence & Continuous Time Encoders
        self.time_encoder = ContinuousTimeEmbedding(embed_dim)
        self.seq_proj = nn.Linear(d_seq, embed_dim)
        self.seq_norm = nn.LayerNorm(embed_dim)

        # 2. Query Projection from (Anchor + Current Omics)
        self.query_proj = nn.Sequential(
            nn.Linear(d_anchor + d_omics, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
        )

        # 3. Multi-Head Cross-Attention (Query = Anchor/Omics, Key/Value = History)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attn_norm = nn.LayerNorm(embed_dim)

        # 4. Latent Residual MLP Head
        self.mlp_head = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, 1),
        )

        # 5. Non-Negotiable Linear Residual Skip Connection
        self.linear_skip = nn.Linear(d_anchor, 1, bias=True)
        with torch.no_grad():
            self.linear_skip.weight.zero_()
            if self.target_mode == "level":
                self.linear_skip.weight[0, 0] = 1.0  # Exact persistence identity on index FC
            self.linear_skip.bias.zero_()

        # Learnable gating scale for residual adjustment
        init_scale = 0.05 if self.target_mode == "level" else 0.02
        self.residual_scale = nn.Parameter(torch.tensor(init_scale))

    def forward(
        self,
        seq_features: torch.Tensor,
        seq_time_deltas: torch.Tensor,
        seq_mask: torch.Tensor,
        static_anchor: torch.Tensor,
        current_omics: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        seq_features : torch.Tensor
            History visit feature tensor of shape (B, T, D_seq).
        seq_time_deltas : torch.Tensor
            Time deltas in days of shape (B, T).
        seq_mask : torch.Tensor
            Boolean mask of shape (B, T) with True for valid visits.
        static_anchor : torch.Tensor
            Anchor visit vectors of shape (B, D_anchor).
        current_omics : torch.Tensor
            Current visit omics vector of shape (B, D_omics).

        Returns
        -------
        torch.Tensor
            Predicted future log10(FC + 1) in level mode, or delta in delta mode (B,).
        """
        # Direct linear skip reference prediction
        skip_pred = self.linear_skip(static_anchor).squeeze(-1)

        # History sequence + time decay encoding
        seq_emb = self.seq_proj(seq_features)
        time_emb = self.time_encoder(seq_time_deltas)
        history_kv = self.seq_norm(seq_emb + time_emb)

        # Current query embedding
        anchor_omics = torch.cat([static_anchor, current_omics], dim=-1)
        query = self.query_proj(anchor_omics).unsqueeze(1)  # (B, 1, embed_dim)

        # Mask padding tokens
        key_padding_mask = ~seq_mask
        all_masked = key_padding_mask.all(dim=1)
        if all_masked.any():
            key_padding_mask = key_padding_mask.clone()
            key_padding_mask[all_masked, 0] = False

        attn_out, _ = self.cross_attn(
            query=query,
            key=history_kv,
            value=history_kv,
            key_padding_mask=key_padding_mask,
        )
        attn_out = self.attn_norm(attn_out.squeeze(1))

        # Combine query representation and attended history context
        combined = torch.cat([query.squeeze(1), attn_out], dim=-1)
        latent_correction = self.mlp_head(combined).squeeze(-1)

        return skip_pred + self.residual_scale * latent_correction


class MicroTraceAttnTrainer:
    """Standardized training harness for MicroTraceAttn with early stopping and device dispatch."""

    def __init__(
        self,
        embed_dim: int = 64,
        num_heads: int = 4,
        dropout: float = 0.1,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        epochs: int = 45,
        batch_size: int = 32,
        loss_type: str = "smooth_l1",
        target_mode: str = "level",
        device: torch.device | None = None,
        random_state: int = 42,
    ) -> None:
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.batch_size = batch_size
        self.loss_type = loss_type
        self.target_mode = target_mode
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.random_state = random_state
        self.model_: MicroTraceAttn | None = None
        self.history_: list[dict[str, float]] = []

    def fit(
        self,
        train_data: dict[str, np.ndarray],
        y_train: np.ndarray,
        val_data: dict[str, np.ndarray] | None = None,
        y_val: np.ndarray | None = None,
        init_state_dict: dict[str, torch.Tensor] | None = None,
    ) -> MicroTraceAttnTrainer:
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        d_seq = train_data["seq_features"].shape[2]
        d_anchor = train_data["static_anchor"].shape[1]
        d_omics = train_data["current_omics"].shape[1]

        self.model_ = MicroTraceAttn(
            d_seq=d_seq,
            d_anchor=d_anchor,
            d_omics=d_omics,
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            dropout=self.dropout,
            target_mode=self.target_mode,
        ).to(self.device)

        if init_state_dict is not None:
            self.model_.load_state_dict({k: v.to(self.device) for k, v in init_state_dict.items()})

        optimizer = torch.optim.AdamW(
            self.model_.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.epochs, eta_min=1e-5
        )

        if self.loss_type == "l1":
            loss_fn = nn.L1Loss()
        elif self.loss_type in ("smooth_l1", "huber"):
            loss_fn = nn.SmoothL1Loss(beta=0.05)
        else:
            loss_fn = nn.MSELoss()

        # Prepare PyTorch Tensors
        t_seq = torch.from_numpy(train_data["seq_features"]).float()
        t_dt = torch.from_numpy(train_data["seq_time_deltas"]).float()
        t_mask = torch.from_numpy(train_data["seq_mask"]).bool()
        t_anc = torch.from_numpy(train_data["static_anchor"]).float()
        t_om = torch.from_numpy(train_data["current_omics"]).float()
        t_y = torch.from_numpy(y_train).float()

        dataset = TensorDataset(t_seq, t_dt, t_mask, t_anc, t_om, t_y)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        best_val_mae = float("inf")
        best_state = None
        self.history_ = []

        for epoch in range(1, self.epochs + 1):
            self.model_.train()
            train_loss = 0.0
            n_batches = 0

            for b_seq, b_dt, b_mask, b_anc, b_om, b_y in loader:
                b_seq = b_seq.to(self.device)
                b_dt = b_dt.to(self.device)
                b_mask = b_mask.to(self.device)
                b_anc = b_anc.to(self.device)
                b_om = b_om.to(self.device)
                b_y = b_y.to(self.device)

                optimizer.zero_grad()
                pred = self.model_(b_seq, b_dt, b_mask, b_anc, b_om)
                loss = loss_fn(pred, b_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model_.parameters(), max_norm=1.0)
                optimizer.step()

                train_loss += float(loss.item())
                n_batches += 1

            scheduler.step()
            train_loss /= max(n_batches, 1)

            # Evaluate validation if available
            val_mae = 0.0
            if val_data is not None and y_val is not None:
                val_preds = self.predict(val_data)
                val_mae = float(np.mean(np.abs(val_preds - y_val)))
                if val_mae < best_val_mae:
                    best_val_mae = val_mae
                    best_state = {k: v.cpu().clone() for k, v in self.model_.state_dict().items()}

            self.history_.append({
                "epoch": epoch,
                "train_loss": round(train_loss, 5),
                "val_mae": round(val_mae, 4) if val_data is not None else 0.0,
            })

        if best_state is not None:
            self.model_.load_state_dict({k: v.to(self.device) for k, v in best_state.items()})

        return self

    def get_state_dict(self) -> dict[str, torch.Tensor]:
        if self.model_ is None:
            raise RuntimeError("MicroTraceAttnTrainer must be fitted before get_state_dict()")
        return {k: v.cpu().clone() for k, v in self.model_.state_dict().items()}

    def predict(self, test_data: dict[str, np.ndarray]) -> np.ndarray:
        if self.model_ is None:
            raise RuntimeError("MicroTraceAttnTrainer must be fitted before predict()")

        self.model_.eval()
        t_seq = torch.from_numpy(test_data["seq_features"]).float().to(self.device)
        t_dt = torch.from_numpy(test_data["seq_time_deltas"]).float().to(self.device)
        t_mask = torch.from_numpy(test_data["seq_mask"]).bool().to(self.device)
        t_anc = torch.from_numpy(test_data["static_anchor"]).float().to(self.device)
        t_om = torch.from_numpy(test_data["current_omics"]).float().to(self.device)

        with torch.no_grad():
            preds = self.model_(t_seq, t_dt, t_mask, t_anc, t_om)
            return preds.cpu().numpy()


# ==============================================================================
# 5. Model Registry & Benchmark Pipelines
# ==============================================================================

MODEL_FAMILIES = {
    "ElasticNet": ElasticNetBenchmark,
    "RandomForest": RandomForestBenchmark,
    "HistGradientBoosting": HistGradientBoostingBenchmark,
    "MicroTrace-Attn": MicroTraceAttnTrainer,
}

MODELING_ARMS = ("M0", "M1", "M2", "M3", "M4", "M5", "B3", "B4")


def run_c1_model_benchmark(
    cfg_path: str | Path,
    arms: tuple[str, ...] = ("M0", "M1", "M2", "M3", "M4", "M5", "B3", "B4"),
    families: tuple[str, ...] = ("ElasticNet", "RandomForest", "HistGradientBoosting", "MicroTrace-Attn"),
    epochs_attn: int = 40,
    evaluate_test: bool = False,
    dev_pairs: pd.DataFrame | None = None,
    test_pairs: pd.DataFrame | None = None,
    visit_store_df: pd.DataFrame | None = None,
    omics_dict: dict[str, pd.DataFrame] | None = None,
) -> dict[str, Any]:
    """Execute standard Phase P1 direct level benchmark pipeline."""
    import yaml
    from micro_trace.c1.cohort import resolve_existing_path
    from micro_trace.c1.features import (
        DEFAULT_ZERO_SUM_SAMPLES,
        C1FeaturePipeline,
        build_visit_trajectory_store,
        load_and_align_omics_tables,
    )

    cfg_file = Path(cfg_path)
    if not cfg_file.exists():
        fallbacks = [
            Path("/content/drive/MyDrive/Micro-Trace/micro_trace/configs/c1_config.yaml"),
            Path("micro_trace/configs/c1_config.yaml"),
            Path("configs/c1_config.yaml"),
        ]
        cfg_file = resolve_existing_path(cfg_file, fallbacks)

    with open(cfg_file, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    paths_cfg = cfg.get("paths", {})
    reports_dir = Path(paths_cfg.get("c1_reports_dir", "micro_trace/reports/c1"))
    if not reports_dir.exists():
        fallbacks_rep = [
            Path("/content/drive/MyDrive/Micro-Trace/micro_trace/reports/c1"),
            Path("micro_trace/reports/c1"),
        ]
        reports_dir = resolve_existing_path(reports_dir, fallbacks_rep)

    if dev_pairs is None or (evaluate_test and test_pairs is None):
        cohort_csv = reports_dir / "c1_cohort_consecutive_0d.csv"
        if not cohort_csv.exists():
            raise FileNotFoundError(f"Cohort table not found at {cohort_csv}")
        pairs_df = pd.read_csv(cohort_csv)
        if dev_pairs is None:
            dev_pairs = pairs_df[pairs_df["partition"] == "dev"].copy()
        if evaluate_test and test_pairs is None:
            test_pairs = pairs_df[pairs_df["partition"] == "test"].copy()

    # Load Multi-Omics if not provided
    if omics_dict is None:
        omics_dict = load_and_align_omics_tables(cfg_file)

    # Load Metadata & Build Trajectory Store if not provided
    if visit_store_df is None:
        raw_path = Path(paths_cfg.get("raw_data_dir", "DataSet/hmp2"))
        meta_file = raw_path / "metadata" / "hmp2_metadata_2018-08-20.csv"
        if not meta_file.exists():
            meta_file = resolve_existing_path(
                meta_file,
                [
                    Path("/content/drive/MyDrive/Micro-Trace/micro-trace-data/raw/metadata/hmp2_metadata_2018-08-20.csv"),
                    Path("micro-trace-data/raw/metadata/hmp2_metadata_2018-08-20.csv"),
                    Path("DataSet/hmp2/metadata/hmp2_metadata_2018-08-20.csv"),
                ],
            )
        meta_df = pd.read_csv(meta_file, low_memory=False)

        schema_file = Path(paths_cfg.get("schema_mapping_path", "micro_trace/configs/schema_mapping.yaml"))
        if not schema_file.exists():
            schema_file = resolve_existing_path(
                schema_file,
                [
                    Path("/content/drive/MyDrive/Micro-Trace/micro_trace/configs/schema_mapping.yaml"),
                    Path("micro_trace/configs/schema_mapping.yaml"),
                    Path("configs/schema_mapping.yaml"),
                ],
            )
        with open(schema_file, "r", encoding="utf-8") as f:
            schema_cfg = yaml.safe_load(f)

        excluded_zero = cfg.get("excluded_zero_sum_samples", DEFAULT_ZERO_SUM_SAMPLES)
        visit_store_df = build_visit_trajectory_store(
            meta_df=meta_df,
            omics_dict=omics_dict,
            schema_cfg=schema_cfg,
            excluded_zero_sum_samples=excluded_zero,
        )

    y_dev = dev_pairs["future_log10_fc_plus_1"].to_numpy(dtype=np.float64)
    y_test = (
        test_pairs["future_log10_fc_plus_1"].to_numpy(dtype=np.float64)
        if evaluate_test and test_pairs is not None
        else None
    )

    results_records: list[dict[str, Any]] = []

    # 1. Evaluate B0 (Mean Baseline)
    mean_b0 = MeanBaseline().fit(np.zeros((len(y_dev), 1)), y_dev)
    b0_dev_preds = mean_b0.predict(np.zeros((len(y_dev), 1)))
    b0_cv_metrics = compute_regression_metrics(y_dev, b0_dev_preds)

    b0_record = {
        "arm": "B0",
        "model_family": "MeanBaseline",
        "cv_mae_mean": b0_cv_metrics["mae"],
        "cv_mae_std": 0.0,
        "cv_rmse_mean": b0_cv_metrics["rmse"],
        "cv_r2_mean": b0_cv_metrics["r2"],
        "cv_spearman_rho_mean": b0_cv_metrics["spearman_rho"],
        "feature_dim": 0,
    }
    results_records.append(b0_record)

    # 2. Iterate through Modeling Arms
    for arm in arms:
        pipe_dev = C1FeaturePipeline(arm=arm, prevalence_threshold=0.10)
        pipe_dev.fit(dev_pairs, visit_store_df, omics_dict)

        X_dev, feat_names = pipe_dev.transform(dev_pairs, visit_store_df, omics_dict)
        feat_dim = X_dev.shape[1]

        if arm == "M0":
            pers_m0 = PersistenceBaseline(fc_col_idx=0).fit(X_dev, y_dev)
            m0_dev_preds = pers_m0.predict(X_dev)
            m0_cv_m = compute_regression_metrics(y_dev, m0_dev_preds)

            m0_record = {
                "arm": "M0",
                "model_family": "PersistenceBaseline",
                "cv_mae_mean": m0_cv_m["mae"],
                "cv_mae_std": 0.0,
                "cv_rmse_mean": m0_cv_m["rmse"],
                "cv_r2_mean": m0_cv_m["r2"],
                "cv_spearman_rho_mean": m0_cv_m["spearman_rho"],
                "feature_dim": feat_dim,
            }
            results_records.append(m0_record)
            continue

        fold_data: list[dict[str, Any]] = []
        for fold in range(5):
            train_mask = (dev_pairs["cv_fold"] != fold).to_numpy()
            val_mask = (dev_pairs["cv_fold"] == fold).to_numpy()

            fold_pipe = C1FeaturePipeline(arm=arm, prevalence_threshold=0.10)
            fold_pipe.fit(dev_pairs[train_mask], visit_store_df, omics_dict)

            X_f_train, _ = fold_pipe.transform(dev_pairs[train_mask], visit_store_df, omics_dict)
            X_f_val, _ = fold_pipe.transform(dev_pairs[val_mask], visit_store_df, omics_dict)
            y_f_train = y_dev[train_mask]
            y_f_val = y_dev[val_mask]

            seq_f_train = None
            seq_f_val = None
            if "MicroTrace-Attn" in families:
                seq_f_train = fold_pipe.build_causal_sequences(dev_pairs[train_mask], visit_store_df, omics_dict)
                seq_f_val = fold_pipe.build_causal_sequences(dev_pairs[val_mask], visit_store_df, omics_dict)

            fold_data.append({
                "X_train": X_f_train,
                "X_val": X_f_val,
                "y_train": y_f_train,
                "y_val": y_f_val,
                "seq_train": seq_f_train,
                "seq_val": seq_f_val,
            })

        for fam_name in families:
            fold_maes: list[float] = []
            fold_rmses: list[float] = []
            fold_r2s: list[float] = []
            fold_rhos: list[float] = []

            for fold_idx, f_info in enumerate(fold_data):
                X_f_train = f_info["X_train"]
                X_f_val = f_info["X_val"]
                y_f_train = f_info["y_train"]
                y_f_val = f_info["y_val"]

                if fam_name == "MicroTrace-Attn":
                    trainer = MicroTraceAttnTrainer(epochs=epochs_attn, random_state=42 + fold_idx)
                    trainer.fit(f_info["seq_train"], y_f_train, val_data=f_info["seq_val"], y_val=y_f_val)
                    preds_val = trainer.predict(f_info["seq_val"])
                else:
                    model_cls = MODEL_FAMILIES[fam_name]
                    model = model_cls()
                    model.fit(X_f_train, y_f_train)
                    preds_val = model.predict(X_f_val)

                f_metrics = compute_regression_metrics(y_f_val, preds_val)
                fold_maes.append(f_metrics["mae"])
                fold_rmses.append(f_metrics["rmse"])
                fold_r2s.append(f_metrics["r2"])
                fold_rhos.append(f_metrics["spearman_rho"])

            fam_record = {
                "arm": arm,
                "model_family": fam_name,
                "cv_mae_mean": round(float(np.mean(fold_maes)), 4),
                "cv_mae_std": round(float(np.std(fold_maes)), 4),
                "cv_rmse_mean": round(float(np.mean(fold_rmses)), 4),
                "cv_r2_mean": round(float(np.mean(fold_r2s)), 4),
                "cv_spearman_rho_mean": round(float(np.mean(fold_rhos)), 4),
                "feature_dim": feat_dim,
            }
            results_records.append(fam_record)

    results_df = pd.DataFrame(results_records)
    out_csv = reports_dir / "c1_model_benchmark_results.csv"
    results_df.to_csv(out_csv, index=False)

    summary_dict = {
        "benchmark_timestamp": datetime.now(timezone.utc).isoformat(),
        "evaluated_arms": list(arms),
        "evaluated_families": list(families),
        "dev_pairs_count": len(dev_pairs),
        "test_pairs_count": len(test_pairs) if test_pairs is not None else 0,
        "test_partition_status": "evaluated" if evaluate_test else "untouched_held_out",
        "cv_folds": 5,
        "best_cv_model": results_df.sort_values("cv_mae_mean").iloc[0].to_dict(),
        "summary_table": results_records,
    }
    out_json = reports_dir / "c1_model_summary.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary_dict, f, indent=2)

    return summary_dict


def run_c1_delta_benchmark(
    cfg_path: str | Path,
    arms: tuple[str, ...] = ("M0", "M1", "M2", "M3", "M4", "M5", "B3", "B4"),
    families: tuple[str, ...] = (
        "ElasticNetCV",
        "RandomForest",
        "HistGradientBoosting",
        "MicroTrace-Attn-SmoothL1",
        "MicroTrace-Attn-L1",
    ),
    epochs_attn: int = 35,
    evaluate_test: bool = False,
    use_inner_val: bool = True,
    dev_pairs: pd.DataFrame | None = None,
    test_pairs: pd.DataFrame | None = None,
    visit_store_df: pd.DataFrame | None = None,
    omics_dict: dict[str, pd.DataFrame] | None = None,
    output_csv_name: str = "c1_trial1_delta_benchmark_results.csv",
    output_json_name: str = "c1_trial1_delta_summary.json",
) -> dict[str, Any]:
    """Execute Trial 1 Target Delta Formulation Benchmark.

    Formulation:
      Delta y = log10(FC_{t+1} + 1) - log10(FC_t + 1)
      hat{y}_{t+1} = log10(FC_t + 1) + hat{Delta y}

    Evaluates across candidate models on frozen 5-fold participant-grouped CV
    (dev N=371, 75 participants). Strictly holds the 20% test partition
    (N=74) untouched (PR-03, zero leakage invariant). When use_inner_val=True,
    early stopping for MicroTrace-Attn is performed on an inner participant split
    of training folds, preventing any validation leakage from outer evaluation folds.
    """
    import yaml
    from micro_trace.c1.cohort import resolve_existing_path
    from micro_trace.c1.features import (
        DEFAULT_ZERO_SUM_SAMPLES,
        C1FeaturePipeline,
        build_visit_trajectory_store,
        load_and_align_omics_tables,
    )

    cfg_file = Path(cfg_path)
    if not cfg_file.exists():
        fallbacks = [
            Path("micro_trace/configs/c1_config.yaml"),
            Path("configs/c1_config.yaml"),
            Path("/content/drive/MyDrive/Micro-Trace/micro_trace/configs/c1_config.yaml"),
        ]
        cfg_file = resolve_existing_path(cfg_file, fallbacks)

    with open(cfg_file, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    paths_cfg = cfg.get("paths", {})
    reports_dir = Path(paths_cfg.get("c1_reports_dir", "micro_trace/reports/c1"))
    if not reports_dir.exists():
        fallbacks_rep = [
            Path("micro_trace/reports/c1"),
            Path("/content/drive/MyDrive/Micro-Trace/micro_trace/reports/c1"),
        ]
        reports_dir = resolve_existing_path(reports_dir, fallbacks_rep)

    if dev_pairs is None or (evaluate_test and test_pairs is None):
        cohort_csv = reports_dir / "c1_cohort_consecutive_0d.csv"
        if not cohort_csv.exists():
            raise FileNotFoundError(f"Cohort table not found at {cohort_csv}")
        pairs_df = pd.read_csv(cohort_csv)
        if dev_pairs is None:
            dev_pairs = pairs_df[pairs_df["partition"] == "dev"].copy()
        if evaluate_test and test_pairs is None:
            test_pairs = pairs_df[pairs_df["partition"] == "test"].copy()

    # Load Multi-Omics if not provided
    if omics_dict is None:
        omics_dict = load_and_align_omics_tables(cfg_file)

    # Load Metadata & Build Trajectory Store if not provided
    if visit_store_df is None:
        raw_path = Path(paths_cfg.get("raw_data_dir", "DataSet/hmp2"))
        meta_file = raw_path / "metadata" / "hmp2_metadata_2018-08-20.csv"
        if not meta_file.exists():
            meta_file = resolve_existing_path(
                meta_file,
                [
                    Path("DataSet/hmp2/metadata/hmp2_metadata_2018-08-20.csv"),
                    Path("micro-trace-data/raw/metadata/hmp2_metadata_2018-08-20.csv"),
                    Path("/content/drive/MyDrive/Micro-Trace/micro-trace-data/raw/metadata/hmp2_metadata_2018-08-20.csv"),
                ],
            )
        meta_df = pd.read_csv(meta_file, low_memory=False)

        schema_file = Path(paths_cfg.get("schema_mapping_path", "micro_trace/configs/schema_mapping.yaml"))
        if not schema_file.exists():
            schema_file = resolve_existing_path(
                schema_file,
                [
                    Path("micro_trace/configs/schema_mapping.yaml"),
                    Path("configs/schema_mapping.yaml"),
                    Path("/content/drive/MyDrive/Micro-Trace/micro_trace/configs/schema_mapping.yaml"),
                ],
            )
        with open(schema_file, "r", encoding="utf-8") as f:
            schema_cfg = yaml.safe_load(f)

        excluded_zero = cfg.get("excluded_zero_sum_samples", DEFAULT_ZERO_SUM_SAMPLES)
        visit_store_df = build_visit_trajectory_store(
            meta_df=meta_df,
            omics_dict=omics_dict,
            schema_cfg=schema_cfg,
            excluded_zero_sum_samples=excluded_zero,
        )

    # Level targets and Index Anchors
    y_dev_level = dev_pairs["future_log10_fc_plus_1"].to_numpy(dtype=np.float64)
    y_dev_idx = dev_pairs["index_log10_fc_same_day"].to_numpy(dtype=np.float64)
    delta_dev = y_dev_level - y_dev_idx

    results_records: list[dict[str, Any]] = []

    # 1. Baseline: M0 Persistence Baseline (Delta = 0.0) evaluated fold-by-fold
    m0_fold_rec_maes: list[float] = []
    m0_fold_rec_rmses: list[float] = []
    m0_fold_rec_r2s: list[float] = []
    m0_fold_rec_rhos: list[float] = []

    for fold in range(5):
        val_mask = (dev_pairs["cv_fold"] == fold).to_numpy()
        y_val = y_dev_level[val_mask]
        idx_val = y_dev_idx[val_mask]
        d_val = delta_dev[val_mask]
        d_pred = np.zeros_like(d_val)
        rec_pred = idx_val.copy()

        f_m = compute_delta_regression_metrics(
            y_true_reconstructed=y_val,
            y_pred_reconstructed=rec_pred,
            delta_true=d_val,
            delta_pred=d_pred,
        )
        m0_fold_rec_maes.append(f_m["mae_reconstructed"])
        m0_fold_rec_rmses.append(f_m["rmse_reconstructed"])
        m0_fold_rec_r2s.append(f_m["r2_reconstructed"])
        m0_fold_rec_rhos.append(f_m["spearman_rho_reconstructed"])

    m0_record = {
        "arm": "M0",
        "model_family": "PersistenceBaseline",
        "cv_mae_reconstructed": round(float(np.mean(m0_fold_rec_maes)), 4),
        "cv_mae_delta": round(float(np.mean(m0_fold_rec_maes)), 4),
        "cv_directional_acc": 0.0,  # Persistence makes no directional call
        "cv_spearman_rho_reconstructed": round(float(np.mean(m0_fold_rec_rhos)), 4),
        "cv_r2_reconstructed": round(float(np.mean(m0_fold_rec_r2s)), 4),
        "cv_rmse_reconstructed": round(float(np.mean(m0_fold_rec_rmses)), 4),
        "cv_mae_reconstructed_std": round(float(np.std(m0_fold_rec_maes)), 4),
        "feature_dim": 2,
    }
    results_records.append(m0_record)
    print(
        f"[Persistence Baseline M0] Reconstructed MAE: {m0_record['cv_mae_reconstructed']:.4f} (+/- {m0_record['cv_mae_reconstructed_std']:.4f}) | "
        f"Delta MAE: {m0_record['cv_mae_delta']:.4f} | R2: {m0_record['cv_r2_reconstructed']:.4f} | "
        f"Spearman rho: {m0_record['cv_spearman_rho_reconstructed']:.4f}",
        flush=True,
    )

    # 2. Iterate through Modeling Arms
    for arm in arms:
        if arm == "M0":
            continue

        print(f"[{datetime.now().strftime('%H:%M:%S')}] Processing Modeling Arm: {arm}", flush=True)

        # Pre-extract fold features strictly on training folds (PR-03 train-only invariant)
        fold_data: list[dict[str, Any]] = []
        feat_dim = 0

        for fold in range(5):
            train_mask = (dev_pairs["cv_fold"] != fold).to_numpy()
            val_mask = (dev_pairs["cv_fold"] == fold).to_numpy()

            fold_pipe = C1FeaturePipeline(arm=arm, prevalence_threshold=0.10)
            fold_pipe.fit(dev_pairs[train_mask], visit_store_df, omics_dict)

            X_f_train, _ = fold_pipe.transform(dev_pairs[train_mask], visit_store_df, omics_dict)
            X_f_val, _ = fold_pipe.transform(dev_pairs[val_mask], visit_store_df, omics_dict)
            feat_dim = X_f_train.shape[1]

            y_level_f_train = y_dev_level[train_mask]
            y_idx_f_train = y_dev_idx[train_mask]
            delta_f_train = delta_dev[train_mask]

            y_level_f_val = y_dev_level[val_mask]
            y_idx_f_val = y_dev_idx[val_mask]
            delta_f_val = delta_dev[val_mask]

            seq_f_train = None
            seq_f_val = None
            seq_inner_train = None
            seq_inner_val = None
            delta_inner_train = None
            delta_inner_val = None

            needs_attn = any("MicroTrace-Attn" in f for f in families)
            if needs_attn:
                if use_inner_val:
                    train_df = dev_pairs[train_mask].copy()
                    train_pids = np.unique(train_df["participant_id"].values)
                    rng = np.random.default_rng(42 + fold)
                    val_pids = set(rng.choice(train_pids, size=max(1, int(len(train_pids) * 0.2)), replace=False))
                    inner_train_mask = ~train_df["participant_id"].isin(val_pids).to_numpy()
                    inner_val_mask = train_df["participant_id"].isin(val_pids).to_numpy()

                    # Inner pipeline fitted strictly on inner train participants
                    inner_pipe = C1FeaturePipeline(arm=arm, prevalence_threshold=0.10)
                    inner_pipe.fit(train_df[inner_train_mask], visit_store_df, omics_dict)

                    seq_inner_train = inner_pipe.build_causal_sequences(train_df[inner_train_mask], visit_store_df, omics_dict)
                    seq_inner_val = inner_pipe.build_causal_sequences(train_df[inner_val_mask], visit_store_df, omics_dict)
                    # Outer evaluation sequences built using the inner_pipe
                    seq_f_val = inner_pipe.build_causal_sequences(dev_pairs[val_mask], visit_store_df, omics_dict)

                    delta_inner_train = delta_dev[train_mask][inner_train_mask]
                    delta_inner_val = delta_dev[train_mask][inner_val_mask]
                else:
                    seq_f_train = fold_pipe.build_causal_sequences(dev_pairs[train_mask], visit_store_df, omics_dict)
                    seq_f_val = fold_pipe.build_causal_sequences(dev_pairs[val_mask], visit_store_df, omics_dict)

            fold_data.append({
                "X_train": X_f_train,
                "X_val": X_f_val,
                "delta_train": delta_f_train,
                "delta_val": delta_f_val,
                "y_level_val": y_level_f_val,
                "y_idx_val": y_idx_f_val,
                "seq_train": seq_f_train,
                "seq_val": seq_f_val,
                "seq_inner_train": seq_inner_train,
                "seq_inner_val": seq_inner_val,
                "delta_inner_train": delta_inner_train,
                "delta_inner_val": delta_inner_val,
            })

        # Evaluate Candidate Families
        for fam_name in families:
            fold_rec_maes: list[float] = []
            fold_delta_maes: list[float] = []
            fold_dir_accs: list[float] = []
            fold_rec_rhos: list[float] = []
            fold_rec_r2s: list[float] = []
            fold_rec_rmses: list[float] = []

            for fold_idx, f_info in enumerate(fold_data):
                X_train = f_info["X_train"]
                X_val = f_info["X_val"]
                delta_train = f_info["delta_train"]
                delta_val = f_info["delta_val"]
                y_level_val = f_info["y_level_val"]
                y_idx_val = f_info["y_idx_val"]

                if "MicroTrace-Attn" in fam_name:
                    loss_mode = "l1" if "-L1" in fam_name else "smooth_l1"
                    trainer = MicroTraceAttnTrainer(
                        embed_dim=48,
                        num_heads=4,
                        epochs=epochs_attn,
                        loss_type=loss_mode,
                        target_mode="delta",
                        random_state=42 + fold_idx,
                    )
                    if use_inner_val and f_info["seq_inner_train"] is not None:
                        trainer.fit(
                            f_info["seq_inner_train"],
                            f_info["delta_inner_train"],
                            val_data=f_info["seq_inner_val"],
                            y_val=f_info["delta_inner_val"],
                        )
                    else:
                        trainer.fit(
                            f_info["seq_train"],
                            delta_train,
                            val_data=None,
                            y_val=None,
                        )
                    pred_delta = trainer.predict(f_info["seq_val"])
                elif fam_name in ("ElasticNetCV", "ElasticNet"):
                    model = ElasticNetBenchmark(cv=3, random_state=42 + fold_idx)
                    model.fit(X_train, delta_train)
                    pred_delta = model.predict(X_val)
                elif fam_name == "RandomForest":
                    model = RandomForestBenchmark(n_estimators=150, max_depth=8, random_state=42 + fold_idx)
                    model.fit(X_train, delta_train)
                    pred_delta = model.predict(X_val)
                elif fam_name == "HistGradientBoosting":
                    model = HistGradientBoostingBenchmark(max_iter=120, max_depth=5, random_state=42 + fold_idx)
                    model.fit(X_train, delta_train)
                    pred_delta = model.predict(X_val)
                else:
                    raise ValueError(f"Unknown family: {fam_name}")

                pred_reconstructed = y_idx_val + pred_delta
                f_m = compute_delta_regression_metrics(
                    y_true_reconstructed=y_level_val,
                    y_pred_reconstructed=pred_reconstructed,
                    delta_true=delta_val,
                    delta_pred=pred_delta,
                )

                fold_rec_maes.append(f_m["mae_reconstructed"])
                fold_delta_maes.append(f_m["mae_delta"])
                fold_dir_accs.append(f_m["directional_accuracy"])
                fold_rec_rhos.append(f_m["spearman_rho_reconstructed"])
                fold_rec_r2s.append(f_m["r2_reconstructed"])
                fold_rec_rmses.append(f_m["rmse_reconstructed"])

            mean_rec_mae = round(float(np.mean(fold_rec_maes)), 4)
            std_rec_mae = round(float(np.std(fold_rec_maes)), 4)
            mean_delta_mae = round(float(np.mean(fold_delta_maes)), 4)
            mean_dir_acc = round(float(np.mean(fold_dir_accs)), 2)
            mean_r2 = round(float(np.mean(fold_rec_r2s)), 4)
            mean_rho = round(float(np.mean(fold_rec_rhos)), 4)
            mean_rmse = round(float(np.mean(fold_rec_rmses)), 4)

            print(
                f"  -> Arm {arm} | {fam_name}: Reconstructed MAE = {mean_rec_mae:.4f} (+/- {std_rec_mae:.4f}) | "
                f"Delta MAE = {mean_delta_mae:.4f} | Dir Acc = {mean_dir_acc:.1f}% | R2 = {mean_r2:.4f} | Rho = {mean_rho:.4f}",
                flush=True,
            )

            results_records.append({
                "arm": arm,
                "model_family": fam_name,
                "cv_mae_reconstructed": mean_rec_mae,
                "cv_mae_delta": mean_delta_mae,
                "cv_directional_acc": mean_dir_acc,
                "cv_spearman_rho_reconstructed": mean_rho,
                "cv_r2_reconstructed": mean_r2,
                "cv_rmse_reconstructed": mean_rmse,
                "cv_mae_reconstructed_std": std_rec_mae,
                "feature_dim": feat_dim,
            })

    results_df = pd.DataFrame(results_records)
    out_csv = reports_dir / output_csv_name
    results_df.to_csv(out_csv, index=False)
    logger.info(f"Saved Delta benchmark results to {out_csv}")

    best_model_row = results_df.sort_values("cv_mae_reconstructed").iloc[0].to_dict()

    summary_dict = {
        "benchmark_timestamp": datetime.now(timezone.utc).isoformat(),
        "target_formulation": "Delta_y = log10(FC_{t+1} + 1) - log10(FC_t + 1)",
        "reconstructed_formulation": "hat{y}_{t+1} = log10(FC_t + 1) + hat{Delta y}",
        "persistence_baseline_mae": 0.2687,
        "evaluated_arms": list(arms),
        "evaluated_families": list(families),
        "dev_pairs_count": len(dev_pairs),
        "test_pairs_count": len(test_pairs) if test_pairs is not None else 0,
        "test_partition_status": "evaluated" if evaluate_test else "untouched_held_out",
        "cv_folds": 5,
        "best_cv_model": best_model_row,
        "summary_table": results_records,
    }

    out_json = reports_dir / output_json_name
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary_dict, f, indent=2)
    logger.info(f"Saved Delta benchmark summary to {out_json}")

    return summary_dict
