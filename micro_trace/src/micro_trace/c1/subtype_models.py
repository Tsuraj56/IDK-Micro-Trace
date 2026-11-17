"""Phase P1 (C1: Future FC Prediction) — Subtype-Stratified Prediction Benchmark.

Evaluates whether IBD subtype-aware modeling (CD, UC, nonIBD) improves longitudinal
fecal calprotectin (FC) delta forecasting under strict 5-fold participant-grouped
dev CV (N=371 dev pairs across 75 participants; 18 held-out test participants / 74 pairs
kept 100% untouched, evaluate_test=False, seed=42).

Compares against:
  - Persistence Baseline (M0: delta_y = 0)
  - Global Pooled Reference (trained on all 75 dev participants, evaluated per subtype)

Across compact arms (M1, M_guild_delta) in Target Delta mode (delta_y = y_{t+1} - y_t):
  - Method 1 (Compact Subtype-Only): Separate ElasticNetCV & MicroTrace-Attn-L1 per subtype.
  - Method 2 (Hierarchical Partial Pooling / Subtype-Interaction):
      * ElasticNetCV-Interaction: Pooled train-fold ElasticNetCV with [X, is_CD, is_UC, is_CD*X, is_UC*X].
      * MicroTrace-Attn-SubtypeHead: Shared-Trunk + Subtype-Head MicroTraceAttn.
  - Method 3 (Pooled Multi-Horizon Pre-Training -> Subtype Fine-Tuning):
      * Pre-train MicroTraceAttn strictly on train-fold multi-horizon [7, 90]d pairs,
        then fine-tune separate models on CD, UC, and nonIBD training pairs.
  - Method 4 (Subtype OOF Stacking):
      * Leak-free nested OOF alpha-blending per subtype combining subtype-adapted
        MicroTrace-Attn + ElasticNetCV.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from micro_trace.c1.cohort import DEFAULT_ZERO_SUM_SAMPLES, resolve_existing_path
from micro_trace.c1.features import (
    C1FeaturePipeline,
    build_visit_trajectory_store,
    load_and_align_omics_tables,
)
from micro_trace.c1.models import (
    ContinuousTimeEmbedding,
    ElasticNetBenchmark,
    MicroTraceAttnTrainer,
    compute_delta_regression_metrics,
)
from micro_trace.c1.trials_combinations import (
    _get_delta_targets,
    _split_inner_participants,
    filter_multi_horizon_pretrain_df,
    tune_oof_stack_alpha,
)

logger = logging.getLogger("micro_trace.c1.subtype_models")

SUBTYPES: tuple[str, ...] = ("CD", "UC", "nonIBD")
COMPACT_ARMS: tuple[str, ...] = ("M1", "M_guild_delta")


def _sanitize_json_value(val: Any) -> Any:
    """Recursively replace NaN/Inf floats with None/0.0 so json.dump(..., allow_nan=False) never fails."""
    if isinstance(val, dict):
        return {k: _sanitize_json_value(v) for k, v in val.items()}
    if isinstance(val, list):
        return [_sanitize_json_value(v) for v in val]
    if isinstance(val, (float, np.floating)):
        f = float(val)
        return 0.0 if (math.isnan(f) or math.isinf(f)) else f
    if isinstance(val, (bool, np.bool_)):
        return bool(val)
    if isinstance(val, (int, np.integer)):
        return int(val)
    return val


def filter_subtype_df(df: pd.DataFrame, subtype: str) -> pd.DataFrame:
    """Filter dataframe strictly to a single diagnosis subtype with explicit contamination check."""
    if subtype not in SUBTYPES:
        raise ValueError(f"Unsupported subtype '{subtype}'. Expected one of {SUBTYPES}.")
    sub_df = df[df["diagnosis"].astype(str) == subtype].copy()
    if not sub_df.empty:
        found = set(sub_df["diagnosis"].astype(str))
        assert found == {subtype}, f"Cross-subtype contamination detected: {found} != {{{subtype}}}"
    return sub_df


def build_subtype_interaction_features(
    X_train: np.ndarray,
    diag_train: pd.Series | np.ndarray,
    X_eval: np.ndarray,
    diag_eval: pd.Series | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Build train-fitted scaled subtype interaction features: [X, is_CD, is_UC, is_CD*X, is_UC*X].

    StandardScaler is fitted strictly on X_train to guarantee zero validation/test leakage.
    nonIBD serves as the reference group (is_CD=0, is_UC=0), allowing smaller subtypes
    to borrow strength from shared main effects X while ElasticNet regularizes subtype deviations.
    """
    X_tr_raw = np.nan_to_num(np.asarray(X_train, dtype=np.float64), nan=0.0)
    X_ev_raw = np.nan_to_num(np.asarray(X_eval, dtype=np.float64), nan=0.0)

    scaler = StandardScaler()
    X_tr = np.nan_to_num(scaler.fit_transform(X_tr_raw), nan=0.0)
    X_ev = np.nan_to_num(scaler.transform(X_ev_raw), nan=0.0)

    d_tr = np.asarray(diag_train, dtype=str)
    d_ev = np.asarray(diag_eval, dtype=str)

    cd_tr = (d_tr == "CD").astype(np.float64)[:, None]
    uc_tr = (d_tr == "UC").astype(np.float64)[:, None]
    cd_ev = (d_ev == "CD").astype(np.float64)[:, None]
    uc_ev = (d_ev == "UC").astype(np.float64)[:, None]

    X_tr_int = np.hstack([X_tr, cd_tr, uc_tr, cd_tr * X_tr, uc_tr * X_tr])
    X_ev_int = np.hstack([X_ev, cd_ev, uc_ev, cd_ev * X_ev, uc_ev * X_ev])
    return X_tr_int, X_ev_int


class SubtypeHeadMicroTraceAttn(nn.Module):
    """Shared-Trunk + Subtype-Head MicroTraceAttn for hierarchical partial pooling in delta mode.

    All subtypes share the continuous-time cross-attention trunk and shared residual head,
    while each subtype (CD=0, UC=1, nonIBD=2) has a lightweight residual adaptation head.
    """

    def __init__(
        self,
        d_seq: int = 15,
        d_anchor: int = 6,
        d_omics: int = 10,
        embed_dim: int = 48,
        num_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.time_encoder = ContinuousTimeEmbedding(embed_dim)
        self.seq_proj = nn.Linear(d_seq, embed_dim)
        self.seq_norm = nn.LayerNorm(embed_dim)

        self.query_proj = nn.Sequential(
            nn.Linear(d_anchor + d_omics, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
        )

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attn_norm = nn.LayerNorm(embed_dim)

        # Shared residual head
        self.shared_mlp_head = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, 1),
        )
        self.shared_linear_skip = nn.Linear(d_anchor, 1, bias=True)
        with torch.no_grad():
            self.shared_linear_skip.weight.zero_()
            self.shared_linear_skip.bias.zero_()
        self.shared_scale = nn.Parameter(torch.tensor(0.02))

        # Subtype-specific lightweight residual adaptation heads (0=CD, 1=UC, 2=nonIBD)
        self.subtype_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim * 2, embed_dim // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(embed_dim // 2, 1),
            )
            for _ in range(3)
        ])
        self.subtype_scales = nn.Parameter(torch.full((3,), 0.01))

    def forward(
        self,
        seq_features: torch.Tensor,
        seq_time_deltas: torch.Tensor,
        seq_mask: torch.Tensor,
        static_anchor: torch.Tensor,
        current_omics: torch.Tensor,
        subtype_idx: torch.Tensor,
    ) -> torch.Tensor:
        skip_pred = self.shared_linear_skip(static_anchor).squeeze(-1)

        seq_emb = self.seq_proj(seq_features)
        time_emb = self.time_encoder(seq_time_deltas)
        history_kv = self.seq_norm(seq_emb + time_emb)

        anchor_omics = torch.cat([static_anchor, current_omics], dim=-1)
        query = self.query_proj(anchor_omics).unsqueeze(1)

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

        combined = torch.cat([query.squeeze(1), attn_out], dim=-1)
        shared_corr = self.shared_mlp_head(combined).squeeze(-1)

        # Compute subtype-specific adaptation
        head_outputs = torch.stack(
            [head(combined).squeeze(-1) for head in self.subtype_heads],
            dim=1,
        )  # (B, 3)
        idx_clamped = subtype_idx.clamp(0, 2)
        sub_corr = head_outputs.gather(1, idx_clamped.unsqueeze(1)).squeeze(1)
        sub_scale = self.subtype_scales[idx_clamped]

        return skip_pred + self.shared_scale * shared_corr + sub_scale * sub_corr


class SubtypeHeadMicroTraceAttnTrainer:
    """Training harness for SubtypeHeadMicroTraceAttn with L1 loss and early stopping."""

    def __init__(
        self,
        embed_dim: int = 48,
        num_heads: int = 4,
        dropout: float = 0.1,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        epochs: int = 30,
        batch_size: int = 32,
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
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.random_state = random_state
        self.model_: SubtypeHeadMicroTraceAttn | None = None

    @staticmethod
    def _encode_subtypes(diagnoses: pd.Series | np.ndarray) -> np.ndarray:
        mapping = {"CD": 0, "UC": 1, "nonIBD": 2}
        return np.array([mapping.get(str(d), 2) for d in diagnoses], dtype=np.int64)

    def fit(
        self,
        train_data: dict[str, np.ndarray],
        y_train: np.ndarray,
        diag_train: pd.Series | np.ndarray,
        val_data: dict[str, np.ndarray] | None = None,
        y_val: np.ndarray | None = None,
        diag_val: pd.Series | np.ndarray | None = None,
    ) -> SubtypeHeadMicroTraceAttnTrainer:
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        d_seq = train_data["seq_features"].shape[2]
        d_anchor = train_data["static_anchor"].shape[1]
        d_omics = train_data["current_omics"].shape[1]

        self.model_ = SubtypeHeadMicroTraceAttn(
            d_seq=d_seq,
            d_anchor=d_anchor,
            d_omics=d_omics,
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            dropout=self.dropout,
        ).to(self.device)

        optimizer = torch.optim.AdamW(
            self.model_.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(self.epochs, 1), eta_min=1e-5
        )
        loss_fn = nn.L1Loss()

        t_seq = torch.from_numpy(train_data["seq_features"]).float()
        t_dt = torch.from_numpy(train_data["seq_time_deltas"]).float()
        t_mask = torch.from_numpy(train_data["seq_mask"]).bool()
        t_anc = torch.from_numpy(train_data["static_anchor"]).float()
        t_om = torch.from_numpy(train_data["current_omics"]).float()
        t_sub = torch.from_numpy(self._encode_subtypes(diag_train)).long()
        t_y = torch.from_numpy(y_train).float()

        dataset = TensorDataset(t_seq, t_dt, t_mask, t_anc, t_om, t_sub, t_y)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        best_val_mae = float("inf")
        best_state = None

        for _ in range(1, self.epochs + 1):
            self.model_.train()
            for b_seq, b_dt, b_mask, b_anc, b_om, b_sub, b_y in loader:
                optimizer.zero_grad()
                pred = self.model_(
                    b_seq.to(self.device),
                    b_dt.to(self.device),
                    b_mask.to(self.device),
                    b_anc.to(self.device),
                    b_om.to(self.device),
                    b_sub.to(self.device),
                )
                loss = loss_fn(pred, b_y.to(self.device))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model_.parameters(), max_norm=1.0)
                optimizer.step()
            scheduler.step()

            if val_data is not None and y_val is not None and diag_val is not None and len(y_val) > 0:
                val_preds = self.predict(val_data, diag_val)
                val_mae = float(np.mean(np.abs(val_preds - y_val)))
                if val_mae < best_val_mae:
                    best_val_mae = val_mae
                    best_state = {k: v.cpu().clone() for k, v in self.model_.state_dict().items()}

        if best_state is not None:
            self.model_.load_state_dict({k: v.to(self.device) for k, v in best_state.items()})
        return self

    def predict(self, eval_data: dict[str, np.ndarray], diag_eval: pd.Series | np.ndarray) -> np.ndarray:
        if self.model_ is None:
            raise RuntimeError("SubtypeHeadMicroTraceAttnTrainer must be fitted before predict()")
        self.model_.eval()
        with torch.no_grad():
            preds = self.model_(
                torch.from_numpy(eval_data["seq_features"]).float().to(self.device),
                torch.from_numpy(eval_data["seq_time_deltas"]).float().to(self.device),
                torch.from_numpy(eval_data["seq_mask"]).bool().to(self.device),
                torch.from_numpy(eval_data["static_anchor"]).float().to(self.device),
                torch.from_numpy(eval_data["current_omics"]).float().to(self.device),
                torch.from_numpy(self._encode_subtypes(diag_eval)).long().to(self.device),
            )
            return preds.cpu().numpy()


def _fit_predict_enet(
    X_tr: np.ndarray,
    d_tr: np.ndarray,
    X_va: np.ndarray,
    seed: int,
) -> np.ndarray:
    """Fit ElasticNetBenchmark safely on potentially small subtype training folds."""
    n_samples = len(d_tr)
    if n_samples < 2 or np.std(d_tr) < 1e-12:
        mean_val = float(np.mean(d_tr)) if n_samples > 0 else 0.0
        return np.full(len(X_va), mean_val, dtype=np.float64)
    cv_folds = min(3, n_samples)
    model = ElasticNetBenchmark(cv=cv_folds, tol=5e-3, max_iter=10000, random_state=seed).fit(X_tr, d_tr)
    return model.predict(X_va)


def _fit_predict_attn_l1(
    arm: str,
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    visit_store_df: pd.DataFrame,
    omics_dict: dict[str, pd.DataFrame],
    epochs: int,
    seed: int,
    pretrain_df: pd.DataFrame | None = None,
    pretrain_epochs: int = 20,
    pipe_fit_df: pd.DataFrame | None = None,
) -> np.ndarray:
    """Fit MicroTrace-Attn-L1 in delta mode with optional multi-horizon pretraining."""
    if len(train_df) == 0:
        return np.zeros(len(eval_df), dtype=np.float64)

    inner_tr_mask, inner_val_mask = _split_inner_participants(train_df, seed=seed)
    has_inner_val = bool(inner_val_mask.any() and inner_tr_mask.any())
    fit_df = train_df[inner_tr_mask].copy() if has_inner_val else train_df.copy()
    val_df = train_df[inner_val_mask].copy() if has_inner_val else None

    pipe = C1FeaturePipeline(arm=arm, prevalence_threshold=0.10)
    pipe.fit(pipe_fit_df if pipe_fit_df is not None else fit_df, visit_store_df, omics_dict)

    seq_fit = pipe.build_causal_sequences(fit_df, visit_store_df, omics_dict)
    _, _, d_fit = _get_delta_targets(fit_df)

    seq_val = None
    d_val = None
    if val_df is not None and not val_df.empty:
        seq_val = pipe.build_causal_sequences(val_df, visit_store_df, omics_dict)
        _, _, d_val = _get_delta_targets(val_df)

    init_state: dict[str, torch.Tensor] | None = None
    if pretrain_df is not None and not pretrain_df.empty:
        # Exclude any inner-val or outer-val or test participants from pretraining
        forbidden_pids = set(eval_df["participant_id"].astype(str))
        if val_df is not None:
            forbidden_pids |= set(val_df["participant_id"].astype(str))
        pre_clean = pretrain_df[~pretrain_df["participant_id"].astype(str).isin(forbidden_pids)].copy()
        if not pre_clean.empty:
            seq_pre = pipe.build_causal_sequences(pre_clean, visit_store_df, omics_dict)
            _, _, d_pre = _get_delta_targets(pre_clean)
            pre_trainer = MicroTraceAttnTrainer(
                embed_dim=48,
                num_heads=4,
                lr=1e-3,
                epochs=pretrain_epochs,
                loss_type="l1",
                target_mode="delta",
                random_state=seed,
            )
            pre_trainer.fit(seq_pre, d_pre, val_data=seq_val, y_val=d_val)
            init_state = pre_trainer.get_state_dict()

    trainer = MicroTraceAttnTrainer(
        embed_dim=48,
        num_heads=4,
        lr=5e-4 if init_state is not None else 1e-3,
        epochs=epochs,
        loss_type="l1",
        target_mode="delta",
        random_state=seed,
    )
    trainer.fit(seq_fit, d_fit, val_data=seq_val, y_val=d_val, init_state_dict=init_state)
    seq_eval = pipe.build_causal_sequences(eval_df, visit_store_df, omics_dict)
    return trainer.predict(seq_eval)


def _fit_predict_method3_fold(
    arm: str,
    tr_df: pd.DataFrame,
    va_df: pd.DataFrame,
    pre_df: pd.DataFrame,
    visit_store_df: pd.DataFrame,
    omics_dict: dict[str, pd.DataFrame],
    epochs: int,
    pretrain_epochs: int,
    seed: int,
) -> np.ndarray:
    """Method 3: Pre-train once on pooled train-fold multi-horizon pairs, then fine-tune per subtype."""
    inner_tr_mask, inner_val_mask = _split_inner_participants(tr_df, seed=seed)
    has_inner_val = bool(inner_val_mask.any() and inner_tr_mask.any())
    fit_pool_df = tr_df[inner_tr_mask].copy() if has_inner_val else tr_df.copy()
    val_pool_df = tr_df[inner_val_mask].copy() if has_inner_val else None

    pipe = C1FeaturePipeline(arm=arm, prevalence_threshold=0.10)
    pipe.fit(fit_pool_df, visit_store_df, omics_dict)

    fit_pids = set(fit_pool_df["participant_id"].astype(str))
    pre_clean = pre_df[pre_df["participant_id"].astype(str).isin(fit_pids)].copy() if not pre_df.empty else pd.DataFrame()

    seq_val_pool = None
    d_val_pool = None
    if val_pool_df is not None and not val_pool_df.empty:
        seq_val_pool = pipe.build_causal_sequences(val_pool_df, visit_store_df, omics_dict)
        _, _, d_val_pool = _get_delta_targets(val_pool_df)

    init_state: dict[str, torch.Tensor] | None = None
    if not pre_clean.empty:
        seq_pre = pipe.build_causal_sequences(pre_clean, visit_store_df, omics_dict)
        _, _, d_pre = _get_delta_targets(pre_clean)
        pre_trainer = MicroTraceAttnTrainer(
            embed_dim=48,
            num_heads=4,
            lr=1e-3,
            epochs=pretrain_epochs,
            loss_type="l1",
            target_mode="delta",
            random_state=seed,
        )
        pre_trainer.fit(seq_pre, d_pre, val_data=seq_val_pool, y_val=d_val_pool)
        init_state = pre_trainer.get_state_dict()

    preds_out = np.zeros(len(va_df), dtype=np.float64)
    for subtype in SUBTYPES:
        va_sub_mask = (va_df["diagnosis"].astype(str) == subtype).to_numpy()
        if not va_sub_mask.any():
            continue
        va_sub_df = filter_subtype_df(va_df, subtype)
        fit_sub_df = filter_subtype_df(fit_pool_df, subtype)
        if fit_sub_df.empty:
            continue

        seq_fit_sub = pipe.build_causal_sequences(fit_sub_df, visit_store_df, omics_dict)
        _, _, d_fit_sub = _get_delta_targets(fit_sub_df)

        seq_val_sub = seq_val_pool
        d_val_sub = d_val_pool
        if val_pool_df is not None:
            val_sub_df = filter_subtype_df(val_pool_df, subtype)
            if not val_sub_df.empty:
                seq_val_sub = pipe.build_causal_sequences(val_sub_df, visit_store_df, omics_dict)
                _, _, d_val_sub = _get_delta_targets(val_sub_df)

        ft_trainer = MicroTraceAttnTrainer(
            embed_dim=48,
            num_heads=4,
            lr=5e-4 if init_state is not None else 1e-3,
            epochs=epochs,
            loss_type="l1",
            target_mode="delta",
            random_state=seed,
        )
        ft_trainer.fit(seq_fit_sub, d_fit_sub, val_data=seq_val_sub, y_val=d_val_sub, init_state_dict=init_state)
        seq_va_sub = pipe.build_causal_sequences(va_sub_df, visit_store_df, omics_dict)
        preds_out[va_sub_mask] = ft_trainer.predict(seq_va_sub)

    return preds_out


def _fit_predict_method2_attn_fold(
    arm: str,
    tr_df: pd.DataFrame,
    va_df: pd.DataFrame,
    visit_store_df: pd.DataFrame,
    omics_dict: dict[str, pd.DataFrame],
    epochs: int,
    seed: int,
) -> np.ndarray:
    """Method 2 (Neural): Shared-Trunk + Subtype-Head MicroTraceAttn."""
    inner_tr_mask, inner_val_mask = _split_inner_participants(tr_df, seed=seed)
    has_inner_val = bool(inner_val_mask.any() and inner_tr_mask.any())
    fit_df = tr_df[inner_tr_mask].copy() if has_inner_val else tr_df.copy()
    val_df = tr_df[inner_val_mask].copy() if has_inner_val else None

    pipe = C1FeaturePipeline(arm=arm, prevalence_threshold=0.10).fit(fit_df, visit_store_df, omics_dict)
    seq_fit = pipe.build_causal_sequences(fit_df, visit_store_df, omics_dict)
    _, _, d_fit = _get_delta_targets(fit_df)

    seq_val = None
    d_val = None
    diag_val = None
    if val_df is not None and not val_df.empty:
        seq_val = pipe.build_causal_sequences(val_df, visit_store_df, omics_dict)
        _, _, d_val = _get_delta_targets(val_df)
        diag_val = val_df["diagnosis"]

    trainer = SubtypeHeadMicroTraceAttnTrainer(
        embed_dim=48,
        num_heads=4,
        lr=1e-3,
        epochs=epochs,
        random_state=seed,
    )
    trainer.fit(
        seq_fit,
        d_fit,
        fit_df["diagnosis"],
        val_data=seq_val,
        y_val=d_val,
        diag_val=diag_val,
    )
    seq_va = pipe.build_causal_sequences(va_df, visit_store_df, omics_dict)
    return trainer.predict(seq_va, va_df["diagnosis"])


def _evaluate_predictions_by_subtype(
    dev_pairs: pd.DataFrame,
    outer_preds: dict[int, np.ndarray],
    method: str,
    arm: str,
    model_family: str,
    extra: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Compute 5-fold CV metrics per subtype (CD, UC, nonIBD) and Overall_Weighted."""
    records: list[dict[str, Any]] = []
    targets = list(SUBTYPES) + ["Overall_Weighted"]

    for target_sub in targets:
        fold_metrics: list[dict[str, float]] = []
        sub_mask_all = (
            np.ones(len(dev_pairs), dtype=bool)
            if target_sub == "Overall_Weighted"
            else (dev_pairs["diagnosis"].astype(str) == target_sub).to_numpy()
        )
        n_pairs = int(sub_mask_all.sum())
        n_pts = int(dev_pairs.loc[sub_mask_all, "participant_id"].nunique())

        for fold in range(5):
            va_df = dev_pairs[dev_pairs["cv_fold"] == fold]
            if va_df.empty:
                continue
            pred_fold = outer_preds[fold]
            if target_sub != "Overall_Weighted":
                sub_m = (va_df["diagnosis"].astype(str) == target_sub).to_numpy()
                if not sub_m.any():
                    continue
                va_sub = va_df[sub_m]
                pred_sub = pred_fold[sub_m]
            else:
                va_sub = va_df
                pred_sub = pred_fold

            y_lvl, y_idx, d_true = _get_delta_targets(va_sub)
            m = compute_delta_regression_metrics(
                y_true_reconstructed=y_lvl,
                y_pred_reconstructed=y_idx + pred_sub,
                delta_true=d_true,
                delta_pred=pred_sub,
            )
            fold_metrics.append(m)

        if not fold_metrics:
            continue

        rec: dict[str, Any] = {
            "method": method,
            "arm": arm,
            "model_family": model_family,
            "subtype": target_sub,
            "n_pairs": n_pairs,
            "n_participants": n_pts,
            "cv_mae_reconstructed": round(float(np.mean([m["mae_reconstructed"] for m in fold_metrics])), 4),
            "cv_mae_delta": round(float(np.mean([m["mae_delta"] for m in fold_metrics])), 4),
            "cv_directional_acc": round(float(np.mean([m["directional_accuracy"] for m in fold_metrics])), 2),
            "cv_r2_reconstructed": round(float(np.mean([m["r2_reconstructed"] for m in fold_metrics])), 4),
            "cv_rmse_reconstructed": round(float(np.mean([m["rmse_reconstructed"] for m in fold_metrics])), 4),
            "cv_spearman_rho_reconstructed": round(float(np.mean([m["spearman_rho_reconstructed"] for m in fold_metrics])), 4),
            "cv_mae_reconstructed_std": round(float(np.std([m["mae_reconstructed"] for m in fold_metrics])), 4),
        }
        if extra:
            rec.update(extra)
        records.append(rec)

    return records


def evaluate_subtype_oof_stack(
    dev_pairs: pd.DataFrame,
    inner_oof_attn: dict[int, np.ndarray],
    outer_attn: dict[int, np.ndarray],
    inner_oof_enet: dict[int, np.ndarray],
    outer_enet: dict[int, np.ndarray],
    method_label: str,
    arm_label: str,
    model_label: str,
) -> list[dict[str, Any]]:
    """Leak-free nested OOF alpha-blending per subtype within each outer CV fold."""
    stacked_outer: dict[int, np.ndarray] = {}
    subtype_alphas: dict[str, list[float]] = {s: [] for s in SUBTYPES}

    for fold in range(5):
        tr_df = dev_pairs[dev_pairs["cv_fold"] != fold].reset_index(drop=True)
        va_df = dev_pairs[dev_pairs["cv_fold"] == fold].reset_index(drop=True)
        _, _, d_tr = _get_delta_targets(tr_df)

        pred_va = np.zeros(len(va_df), dtype=np.float64)
        for subtype in SUBTYPES:
            tr_sub_m = (tr_df["diagnosis"].astype(str) == subtype).to_numpy()
            va_sub_m = (va_df["diagnosis"].astype(str) == subtype).to_numpy()
            if not va_sub_m.any():
                continue

            if tr_sub_m.any():
                alpha_k_s, _ = tune_oof_stack_alpha(
                    inner_oof_attn[fold][tr_sub_m],
                    inner_oof_enet[fold][tr_sub_m],
                    d_tr[tr_sub_m],
                )
            else:
                alpha_k_s = 0.5
            subtype_alphas[subtype].append(alpha_k_s)
            pred_va[va_sub_m] = (
                alpha_k_s * outer_attn[fold][va_sub_m]
                + (1.0 - alpha_k_s) * outer_enet[fold][va_sub_m]
            )
        stacked_outer[fold] = pred_va

    records = _evaluate_predictions_by_subtype(
        dev_pairs=dev_pairs,
        outer_preds=stacked_outer,
        method=method_label,
        arm=arm_label,
        model_family=model_label,
    )
    for rec in records:
        sub = rec["subtype"]
        if sub in subtype_alphas and subtype_alphas[sub]:
            rec["mean_tuned_alpha"] = round(float(np.mean(subtype_alphas[sub])), 4)
        elif sub == "Overall_Weighted":
            all_a = [a for vals in subtype_alphas.values() for a in vals]
            rec["mean_tuned_alpha"] = round(float(np.mean(all_a)), 4) if all_a else 0.5
    return records


def run_c1_subtype_benchmark(
    cfg_path: str | Path = "micro_trace/configs/c1_config.yaml",
    epochs_attn: int = 30,
    pretrain_epochs: int = 20,
    evaluate_test: bool = False,
    dev_pairs: pd.DataFrame | None = None,
    all_horizons_df: pd.DataFrame | None = None,
    visit_store_df: pd.DataFrame | None = None,
    omics_dict: dict[str, pd.DataFrame] | None = None,
    output_csv_name: str = "c1_subtype_benchmark.csv",
    output_json_name: str = "c1_subtype_summary.json",
) -> dict[str, Any]:
    """Execute the C1 Subtype-Stratified Prediction Benchmark (CD, UC, nonIBD) on dev CV only."""
    if evaluate_test:
        raise ValueError("Test partition must remain 100% untouched (evaluate_test=False).")

    np.random.seed(42)
    torch.manual_seed(42)

    cfg_file = Path(cfg_path)
    if not cfg_file.exists():
        cfg_file = resolve_existing_path(
            cfg_file,
            [Path("micro_trace/configs/c1_config.yaml"), Path("configs/c1_config.yaml")],
        )
    with open(cfg_file, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    paths_cfg = cfg.get("paths", {})
    reports_dir = Path(paths_cfg.get("c1_reports_dir", "micro_trace/reports/c1"))
    if not reports_dir.exists():
        reports_dir = resolve_existing_path(
            reports_dir,
            [Path("micro_trace/reports/c1"), Path("reports/c1")],
        )
    reports_dir.mkdir(parents=True, exist_ok=True)

    if dev_pairs is None:
        cohort_csv = reports_dir / "c1_cohort_consecutive_0d.csv"
        pairs_df = pd.read_csv(cohort_csv)
        dev_pairs = pairs_df[pairs_df["partition"] == "dev"].copy()

    # Safety assertion: ensure zero test rows in dev_pairs
    if "partition" in dev_pairs.columns:
        assert set(dev_pairs["partition"].astype(str)) == {"dev"}, "Non-dev rows detected in dev_pairs!"

    if all_horizons_df is None:
        all_h_csv = reports_dir / "c1_cohort_all_horizons_0d.csv"
        all_horizons_df = pd.read_csv(all_h_csv) if all_h_csv.exists() else pd.DataFrame()

    if omics_dict is None:
        omics_dict = load_and_align_omics_tables(cfg_file)

    if visit_store_df is None:
        processed_dir_cfg = paths_cfg.get("processed_c1_dir")
        processed_dir = resolve_existing_path(
            processed_dir_cfg or "",
            [Path("micro-trace-data/processed/c1"), Path("../micro-trace-data/processed/c1")],
        )
        store_cache = processed_dir / "visit_trajectory_store.csv.gz"
        if store_cache.exists():
            visit_store_df = pd.read_csv(store_cache, compression="gzip")
            visit_store_df["event_date"] = pd.to_datetime(visit_store_df["event_date"])
        else:
            raw_path = resolve_existing_path(
                paths_cfg.get("raw_data_dir", ""),
                [Path("micro-trace-data/raw"), Path("../micro-trace-data/raw")],
            )
            meta_df = pd.read_csv(raw_path / "metadata" / "hmp2_metadata_2018-08-20.csv", low_memory=False)
            schema_file = resolve_existing_path(
                paths_cfg.get("schema_mapping_path", ""),
                [Path("micro_trace/configs/schema_mapping.yaml"), Path("configs/schema_mapping.yaml")],
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

    all_records: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    # 0. Persistence Baseline (M0)
    # ------------------------------------------------------------------
    m0_outer: dict[int, np.ndarray] = {}
    for fold in range(5):
        va_df = dev_pairs[dev_pairs["cv_fold"] == fold]
        m0_outer[fold] = np.zeros(len(va_df), dtype=np.float64)
    m0_records = _evaluate_predictions_by_subtype(
        dev_pairs=dev_pairs,
        outer_preds=m0_outer,
        method="Baseline_M0",
        arm="M0",
        model_family="PersistenceBaseline",
    )
    for r in m0_records:
        r["cv_directional_acc"] = 0.0
    all_records.extend(m0_records)

    # Cache outer & inner-OOF predictions for Method 4 stacking:
    # key: (method_key, arm, family) -> {fold: np.ndarray}
    outer_cache: dict[tuple[str, str, str], dict[int, np.ndarray]] = {}
    inner_cache: dict[tuple[str, str, str], dict[int, np.ndarray]] = {}

    def _run_model_on_split(
        method_key: str,
        arm: str,
        fam: str,
        tr_df: pd.DataFrame,
        va_df: pd.DataFrame,
        fold_seed: int,
    ) -> np.ndarray:
        if method_key == "Global_Pooled":
            if fam == "ElasticNetCV":
                pipe = C1FeaturePipeline(arm=arm, prevalence_threshold=0.10).fit(tr_df, visit_store_df, omics_dict)
                X_tr, _ = pipe.transform(tr_df, visit_store_df, omics_dict)
                X_va, _ = pipe.transform(va_df, visit_store_df, omics_dict)
                _, _, d_tr = _get_delta_targets(tr_df)
                return _fit_predict_enet(X_tr, d_tr, X_va, seed=fold_seed)
            return _fit_predict_attn_l1(
                arm=arm,
                train_df=tr_df,
                eval_df=va_df,
                visit_store_df=visit_store_df,
                omics_dict=omics_dict,
                epochs=epochs_attn,
                seed=fold_seed,
            )

        if method_key == "Method1_SubtypeOnly":
            preds = np.zeros(len(va_df), dtype=np.float64)
            for subtype in SUBTYPES:
                va_sub_m = (va_df["diagnosis"].astype(str) == subtype).to_numpy()
                if not va_sub_m.any():
                    continue
                tr_sub = filter_subtype_df(tr_df, subtype)
                va_sub = filter_subtype_df(va_df, subtype)
                if tr_sub.empty:
                    continue
                if fam == "ElasticNetCV":
                    pipe = C1FeaturePipeline(arm=arm, prevalence_threshold=0.10).fit(tr_sub, visit_store_df, omics_dict)
                    X_tr_s, _ = pipe.transform(tr_sub, visit_store_df, omics_dict)
                    X_va_s, _ = pipe.transform(va_sub, visit_store_df, omics_dict)
                    _, _, d_tr_s = _get_delta_targets(tr_sub)
                    preds[va_sub_m] = _fit_predict_enet(X_tr_s, d_tr_s, X_va_s, seed=fold_seed)
                else:
                    preds[va_sub_m] = _fit_predict_attn_l1(
                        arm=arm,
                        train_df=tr_sub,
                        eval_df=va_sub,
                        visit_store_df=visit_store_df,
                        omics_dict=omics_dict,
                        epochs=epochs_attn,
                        seed=fold_seed,
                    )
            return preds

        if method_key == "Method2_PartialPooling":
            if fam == "ElasticNetCV-Interaction":
                pipe = C1FeaturePipeline(arm=arm, prevalence_threshold=0.10).fit(tr_df, visit_store_df, omics_dict)
                X_tr, _ = pipe.transform(tr_df, visit_store_df, omics_dict)
                X_va, _ = pipe.transform(va_df, visit_store_df, omics_dict)
                X_tr_int, X_va_int = build_subtype_interaction_features(
                    X_tr, tr_df["diagnosis"], X_va, va_df["diagnosis"]
                )
                _, _, d_tr = _get_delta_targets(tr_df)
                return _fit_predict_enet(X_tr_int, d_tr, X_va_int, seed=fold_seed)
            return _fit_predict_method2_attn_fold(
                arm=arm,
                tr_df=tr_df,
                va_df=va_df,
                visit_store_df=visit_store_df,
                omics_dict=omics_dict,
                epochs=epochs_attn,
                seed=fold_seed,
            )

        if method_key == "Method3_PretrainSubtypeFT":
            tr_pids = set(tr_df["participant_id"].astype(str))
            pre_df = filter_multi_horizon_pretrain_df(all_horizons_df, tr_pids, min_days=7, max_days=90)
            return _fit_predict_method3_fold(
                arm=arm,
                tr_df=tr_df,
                va_df=va_df,
                pre_df=pre_df,
                visit_store_df=visit_store_df,
                omics_dict=omics_dict,
                epochs=epochs_attn,
                pretrain_epochs=pretrain_epochs,
                seed=fold_seed,
            )

        raise ValueError(f"Unknown method_key: {method_key}")

    def _run_and_cache(method_key: str, arm: str, fam: str, compute_inner: bool = False) -> None:
        key = (method_key, arm, fam)
        if key not in outer_cache:
            outer_cache[key] = {}
            for fold in range(5):
                tr_df = dev_pairs[dev_pairs["cv_fold"] != fold]
                va_df = dev_pairs[dev_pairs["cv_fold"] == fold]
                outer_cache[key][fold] = _run_model_on_split(
                    method_key=method_key,
                    arm=arm,
                    fam=fam,
                    tr_df=tr_df,
                    va_df=va_df,
                    fold_seed=42 + fold,
                )
            recs = _evaluate_predictions_by_subtype(
                dev_pairs=dev_pairs,
                outer_preds=outer_cache[key],
                method=method_key,
                arm=arm,
                model_family=fam,
            )
            all_records.extend(recs)
            for r in recs:
                if r["subtype"] == "Overall_Weighted":
                    logger.info(
                        f"  [{method_key}] {arm} | {fam} (Overall): MAE={r['cv_mae_reconstructed']:.4f} | "
                        f"DirAcc={r['cv_directional_acc']:.1f}% | R2={r['cv_r2_reconstructed']:.4f}"
                    )

        if compute_inner and key not in inner_cache:
            inner_cache[key] = {}
            for fold in range(5):
                tr_df = dev_pairs[dev_pairs["cv_fold"] != fold].reset_index(drop=True)
                oof_vec = np.zeros(len(tr_df), dtype=np.float64)
                inner_folds = sorted(tr_df["cv_fold"].unique())
                for ifold in inner_folds:
                    itr_mask = (tr_df["cv_fold"] != ifold).to_numpy()
                    iva_mask = (tr_df["cv_fold"] == ifold).to_numpy()
                    oof_vec[iva_mask] = _run_model_on_split(
                        method_key=method_key,
                        arm=arm,
                        fam=fam,
                        tr_df=tr_df[itr_mask],
                        va_df=tr_df[iva_mask],
                        fold_seed=42 + fold * 10 + int(ifold),
                    )
                inner_cache[key][fold] = oof_vec

    # ------------------------------------------------------------------
    # 1. Global Pooled Reference & Methods 1, 2, 3 across compact arms
    # ------------------------------------------------------------------
    for arm in COMPACT_ARMS:
        logger.info(f"=== Subtype Benchmark Arm: {arm} ===")
        # Global Pooled Reference
        _run_and_cache("Global_Pooled", arm, "ElasticNetCV", compute_inner=False)
        _run_and_cache("Global_Pooled", arm, "MicroTrace-Attn-L1", compute_inner=False)

        # Method 1: Compact Subtype-Only Models (need inner OOF for Method 4 stacking)
        _run_and_cache("Method1_SubtypeOnly", arm, "ElasticNetCV", compute_inner=True)
        _run_and_cache("Method1_SubtypeOnly", arm, "MicroTrace-Attn-L1", compute_inner=True)

        # Method 2: Hierarchical Partial Pooling / Subtype-Interaction
        _run_and_cache("Method2_PartialPooling", arm, "ElasticNetCV-Interaction", compute_inner=True)
        _run_and_cache("Method2_PartialPooling", arm, "MicroTrace-Attn-SubtypeHead", compute_inner=True)

        # Method 3: Pooled Multi-Horizon Pre-Training -> Subtype Fine-Tuning
        _run_and_cache("Method3_PretrainSubtypeFT", arm, "MicroTrace-Attn-Pretrain-SubtypeFT", compute_inner=True)

        # --------------------------------------------------------------
        # 2. Method 4: Subtype OOF Stacking
        # --------------------------------------------------------------
        stack_pairs = [
            (
                ("Method1_SubtypeOnly", arm, "MicroTrace-Attn-L1"),
                ("Method1_SubtypeOnly", arm, "ElasticNetCV"),
                "OOFStack(M1_AttnL1+M1_ENet)",
            ),
            (
                ("Method2_PartialPooling", arm, "MicroTrace-Attn-SubtypeHead"),
                ("Method2_PartialPooling", arm, "ElasticNetCV-Interaction"),
                "OOFStack(M2_SubtypeHeadAttn+M2_ENetInt)",
            ),
            (
                ("Method3_PretrainSubtypeFT", arm, "MicroTrace-Attn-Pretrain-SubtypeFT"),
                ("Method1_SubtypeOnly", arm, "ElasticNetCV"),
                "OOFStack(M3_PretrainFT+M1_ENet)",
            ),
        ]
        for attn_k, enet_k, mod_lbl in stack_pairs:
            s_recs = evaluate_subtype_oof_stack(
                dev_pairs=dev_pairs,
                inner_oof_attn=inner_cache[attn_k],
                outer_attn=outer_cache[attn_k],
                inner_oof_enet=inner_cache[enet_k],
                outer_enet=outer_cache[enet_k],
                method_label="Method4_SubtypeOOFStack",
                arm_label=arm,
                model_label=mod_lbl,
            )
            all_records.extend(s_recs)

    results_df = pd.DataFrame(all_records)
    csv_path = reports_dir / output_csv_name
    results_df.to_csv(csv_path, index=False)

    # Build concise best-per-subtype summary
    best_by_subtype: dict[str, Any] = {}
    for sub in list(SUBTYPES) + ["Overall_Weighted"]:
        sub_rows = [r for r in all_records if r["subtype"] == sub]
        if sub_rows:
            best_mae = min(sub_rows, key=lambda x: x["cv_mae_reconstructed"])
            best_dir = max(sub_rows, key=lambda x: x["cv_directional_acc"])
            best_by_subtype[sub] = {
                "best_mae_model": f"{best_mae['method']} | {best_mae['arm']} | {best_mae['model_family']}",
                "best_mae": best_mae["cv_mae_reconstructed"],
                "best_dir_model": f"{best_dir['method']} | {best_dir['arm']} | {best_dir['model_family']}",
                "best_directional_acc": best_dir["cv_directional_acc"],
            }

    summary = _sanitize_json_value({
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "seed": 42,
        "test_partition_status": "untouched_held_out",
        "evaluate_test": False,
        "dev_pairs": int(len(dev_pairs)),
        "dev_participants": int(dev_pairs["participant_id"].nunique()),
        "subtypes": list(SUBTYPES),
        "arms": list(COMPACT_ARMS),
        "best_by_subtype": best_by_subtype,
        "summary_table": all_records,
    })

    json_path = reports_dir / output_json_name
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, allow_nan=False)

    logger.info(f"Saved subtype benchmark CSV -> {csv_path}")
    logger.info(f"Saved subtype summary JSON  -> {json_path}")
    return summary


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    parser = argparse.ArgumentParser(description="Run C1 Subtype-Stratified Prediction Benchmark.")
    parser.add_argument("--cfg-path", type=str, default="micro_trace/configs/c1_config.yaml")
    parser.add_argument("--epochs-attn", type=int, default=30)
    parser.add_argument("--pretrain-epochs", type=int, default=20)
    args = parser.parse_args()

    t0 = time.time()
    summary = run_c1_subtype_benchmark(
        cfg_path=args.cfg_path,
        epochs_attn=args.epochs_attn,
        pretrain_epochs=args.pretrain_epochs,
        evaluate_test=False,
    )
    elapsed = time.time() - t0
    logger.info(f"Completed C1 Subtype Benchmark in {elapsed:.1f}s ({len(summary['summary_table'])} rows).")


if __name__ == "__main__":
    main()
