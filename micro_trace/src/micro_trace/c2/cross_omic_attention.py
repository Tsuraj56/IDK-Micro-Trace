"""Phase P2 (C2) Cross-Omic Attention ([Bacteria] <-> [Pathways] <-> [FC]).

Implements a deterministic bipartite cross-attention bottleneck trained across
5 participant-grouped folds for both Current FC and Future FC to extract
reproducible bacterium-pathway teamwork weights (>= 4/5 folds).
"""

from __future__ import annotations

import random
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import spearmanr
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

from micro_trace.c2.lme_signatures import fit_transform_clr_train_only


def _set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class CrossOmicAttentionNet(nn.Module):
    """Bipartite Cross-Omic Attention: [Bacteria] <-> [Pathways] <-> [FC]."""

    def __init__(
        self,
        n_taxa: int,
        n_pathways: int,
        n_clinical: int,
        d_model: int = 16,
        dropout: float = 0.30,
    ) -> None:
        super().__init__()
        self.n_taxa = n_taxa
        self.n_pathways = n_pathways
        self.d_model = d_model

        # Identity embeddings + abundance modulation so attention learns stable biological coupling
        self.tax_emb = nn.Parameter(torch.randn(n_taxa, d_model) * 0.05)
        self.pw_emb = nn.Parameter(torch.randn(n_pathways, d_model) * 0.05)
        self.tax_val_proj = nn.Parameter(torch.randn(1, 1, d_model) * 0.05)
        self.pw_val_proj = nn.Parameter(torch.randn(1, 1, d_model) * 0.05)

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)

        self.tax_gate = nn.Linear(d_model, 1)
        self.dropout = nn.Dropout(dropout)

        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )
        self.omic_scale = nn.Parameter(torch.tensor(0.25))
        self.clinical_skip = nn.Linear(max(1, n_clinical), 1)

    def forward(
        self,
        x_tax: torch.Tensor,
        x_pw: torch.Tensor,
        x_clin: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (y_pred, attn_matrix [B, T, P], tax_weights [B, T])."""
        e_tax = self.tax_emb.unsqueeze(0) + x_tax.unsqueeze(-1) * self.tax_val_proj  # [B, T, d]
        e_pw = self.pw_emb.unsqueeze(0) + x_pw.unsqueeze(-1) * self.pw_val_proj     # [B, P, d]

        q = self.q_proj(e_tax)  # [B, T, d]
        k = self.k_proj(e_pw)   # [B, P, d]
        v = self.v_proj(e_pw)   # [B, P, d]

        scale = float(np.sqrt(self.d_model))
        attn_logits = torch.bmm(q, k.transpose(1, 2)) / scale  # [B, T, P]
        attn_weights = torch.softmax(attn_logits, dim=-1)      # [B, T, P]

        # Cross-attended pathway context per taxon
        ctx = torch.bmm(self.dropout(attn_weights), v) + e_tax  # [B, T, d]

        # Taxon-to-FC pooling
        tax_scores = self.tax_gate(ctx).squeeze(-1)             # [B, T]
        tax_weights = torch.softmax(tax_scores, dim=-1)         # [B, T]
        pooled = torch.sum(ctx * tax_weights.unsqueeze(-1), dim=1)  # [B, d]

        y_omic = self.omic_scale * self.head(pooled).squeeze(-1)
        y_skip = self.clinical_skip(x_clin).squeeze(-1)
        return y_omic + y_skip, attn_weights, tax_weights


def _extract_clinical_matrix(df: pd.DataFrame, target_name: str) -> np.ndarray:
    cols = []
    if target_name == "future_fc" and "index_log10_fc_same_day" in df.columns:
        cols.append(pd.to_numeric(df["index_log10_fc_same_day"], errors="coerce").fillna(0.0).values)
        cols.append(pd.to_numeric(df["elapsed_days"], errors="coerce").fillna(56.0).values)
    cols.append((df["diagnosis"].astype(str) == "CD").astype(float).values)
    cols.append((df["diagnosis"].astype(str) == "UC").astype(float).values)
    for c in ["antibiotics_flag", "immunosuppressants_flag"]:
        if c in df.columns:
            cols.append(pd.to_numeric(df[c], errors="coerce").fillna(0.0).values)
    return np.column_stack(cols) if cols else np.zeros((len(df), 1), dtype=float)


def _screen_top_features(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    col_names: list[str],
    max_features: int,
) -> tuple[list[int], list[str]]:
    """Train-fold-only top-K feature screening by absolute Pearson correlation with y_tr."""
    if len(col_names) <= max_features:
        idx = list(range(len(col_names)))
        return idx, col_names
    y_c = y_tr - np.mean(y_tr)
    y_norm = np.sqrt(np.sum(y_c ** 2)) + 1e-12
    X_c = X_tr - np.mean(X_tr, axis=0, keepdims=True)
    X_norm = np.sqrt(np.sum(X_c ** 2, axis=0)) + 1e-12
    corrs = np.abs(np.sum(X_c * y_c[:, None], axis=0) / (X_norm * y_norm))
    top_idx = np.argsort(corrs)[::-1][:max_features].tolist()
    return top_idx, [col_names[i] for i in top_idx]


def train_and_extract_cross_omic_attention(
    cohort_df: pd.DataFrame,
    species_relab: pd.DataFrame,
    pathways_relab: pd.DataFrame,
    target_name: str,
    prevalence_threshold: float = 0.10,
    max_taxa: int = 35,
    max_pathways: int = 50,
    top_k_pairs_per_fold: int = 150,
    epochs: int = 80,
    lr: float = 1e-3,
    weight_decay: float = 1e-1,
    seed: int = 42,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Train 5-fold Cross-Omic Attention and extract reproducible [Bacterium]<->[Pathway] pairs."""
    _set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    folds = sorted([int(f) for f in cohort_df["cv_fold"].unique() if int(f) >= 0])
    if len(folds) != 5:
        raise ValueError(f"Expected 5 dev folds (0..4), got {folds}")

    oof_preds = np.zeros(len(cohort_df), dtype=float)
    oof_true = cohort_df["target_fc_log10"].astype(float).values

    pair_records: dict[tuple[str, str], dict[str, Any]] = {}
    fold_metrics: list[dict[str, Any]] = []

    for fold in folds:
        _set_seed(seed + fold)
        tr_mask = (cohort_df["cv_fold"] != fold).values
        va_mask = (cohort_df["cv_fold"] == fold).values
        tr_df = cohort_df.loc[tr_mask].reset_index(drop=True)
        va_df = cohort_df.loc[va_mask].reset_index(drop=True)

        tr_ids = tr_df["clean_mgx_sample_id"].astype(str).tolist()
        va_ids = va_df["clean_mgx_sample_id"].astype(str).tolist()

        sp_tr, sp_va, sp_cols, _ = fit_transform_clr_train_only(
            species_relab, tr_ids, va_ids, prevalence_threshold=prevalence_threshold
        )
        pw_tr, pw_va, pw_cols, _ = fit_transform_clr_train_only(
            pathways_relab, tr_ids, va_ids, prevalence_threshold=prevalence_threshold
        )

        sp_tr_arr = sp_tr.loc[tr_ids].values
        sp_va_arr = sp_va.loc[va_ids].values
        pw_tr_arr = pw_tr.loc[tr_ids].values
        pw_va_arr = pw_va.loc[va_ids].values

        y_tr_np = tr_df["target_fc_log10"].astype(float).values
        y_mean = float(np.mean(y_tr_np))
        y_tr_centered = y_tr_np - y_mean
        y_va = va_df["target_fc_log10"].astype(float).values

        # Train-fold-only top-K screening to prevent 43,320-pair overfitting
        sp_idx, sp_cols = _screen_top_features(sp_tr_arr, y_tr_centered, sp_cols, max_taxa)
        pw_idx, pw_cols = _screen_top_features(pw_tr_arr, y_tr_centered, pw_cols, max_pathways)
        sp_tr_arr = sp_tr_arr[:, sp_idx]
        sp_va_arr = sp_va_arr[:, sp_idx]
        pw_tr_arr = pw_tr_arr[:, pw_idx]
        pw_va_arr = pw_va_arr[:, pw_idx]

        sc_sp = StandardScaler().fit(sp_tr_arr)
        sc_pw = StandardScaler().fit(pw_tr_arr)
        x_sp_tr = torch.tensor(sc_sp.transform(sp_tr_arr), dtype=torch.float32, device=device)
        x_sp_va = torch.tensor(sc_sp.transform(sp_va_arr), dtype=torch.float32, device=device)
        x_pw_tr = torch.tensor(sc_pw.transform(pw_tr_arr), dtype=torch.float32, device=device)
        x_pw_va = torch.tensor(sc_pw.transform(pw_va_arr), dtype=torch.float32, device=device)

        clin_tr_raw = _extract_clinical_matrix(tr_df, target_name)
        clin_va_raw = _extract_clinical_matrix(va_df, target_name)
        sc_cl = StandardScaler().fit(clin_tr_raw)
        clin_tr_s = sc_cl.transform(clin_tr_raw)
        clin_va_s = sc_cl.transform(clin_va_raw)
        x_cl_tr = torch.tensor(clin_tr_s, dtype=torch.float32, device=device)
        x_cl_va = torch.tensor(clin_va_s, dtype=torch.float32, device=device)

        # Inner participant-grouped validation split (20% of train participants) for early stopping
        tr_pids = tr_df["participant_id"].unique()
        rng_split = np.random.default_rng(seed + fold)
        n_in_val = max(3, int(round(0.20 * len(tr_pids))))
        in_val_pids = set(rng_split.choice(tr_pids, size=min(n_in_val, len(tr_pids) - 1), replace=False))
        in_val_mask = tr_df["participant_id"].isin(in_val_pids).values
        in_tr_mask = ~in_val_mask

        y_in_tr = torch.tensor(y_tr_centered[in_tr_mask], dtype=torch.float32, device=device)
        y_in_va = torch.tensor(y_tr_centered[in_val_mask], dtype=torch.float32, device=device)

        model = CrossOmicAttentionNet(
            n_taxa=len(sp_cols),
            n_pathways=len(pw_cols),
            n_clinical=clin_tr_raw.shape[1],
            d_model=16,
            dropout=0.30,
        ).to(device)

        # Anchor clinical skip with RidgeCV fit on inner-train clinical covariates
        ridge_skip = RidgeCV(alphas=np.logspace(-1, 3, 10)).fit(clin_tr_s[in_tr_mask], y_tr_centered[in_tr_mask])
        with torch.no_grad():
            model.clinical_skip.weight.copy_(
                torch.tensor(ridge_skip.coef_.reshape(1, -1), dtype=torch.float32, device=device)
            )
            model.clinical_skip.bias.copy_(
                torch.tensor([float(ridge_skip.intercept_)], dtype=torch.float32, device=device)
            )

        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        loss_fn = nn.SmoothL1Loss(beta=0.25)

        best_loss = float("inf")
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        with torch.no_grad():
            init_val_pred, _, _ = model(x_sp_tr[in_val_mask], x_pw_tr[in_val_mask], x_cl_tr[in_val_mask])
            best_loss = float(loss_fn(init_val_pred, y_in_va).item())

        for _ in range(epochs):
            model.train()
            optimizer.zero_grad()
            pred, _, _ = model(x_sp_tr[in_tr_mask], x_pw_tr[in_tr_mask], x_cl_tr[in_tr_mask])
            loss = loss_fn(pred, y_in_tr)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            model.eval()
            with torch.no_grad():
                val_pred, _, _ = model(x_sp_tr[in_val_mask], x_pw_tr[in_val_mask], x_cl_tr[in_val_mask])
                val_loss = float(loss_fn(val_pred, y_in_va).item())
            if val_loss < best_loss:
                best_loss = val_loss
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

        model.eval()
        with torch.no_grad():
            va_pred, va_attn, va_tax_w = model(x_sp_va, x_pw_va, x_cl_va)
            va_pred_np = va_pred.cpu().numpy() + y_mean
            oof_preds[va_mask] = va_pred_np

            # Joint bacterium-pathway-FC teamwork score: E[tax_w_i * attn_{i,j}]
            joint_score = (va_tax_w.unsqueeze(-1) * va_attn).mean(dim=0).cpu().numpy()  # [T, P]

        fold_metrics.append(
            {
                "fold": fold,
                "mae": float(mean_absolute_error(y_va, va_pred_np)),
                "rmse": float(np.sqrt(mean_squared_error(y_va, va_pred_np))),
                "r2": float(r2_score(y_va, va_pred_np)),
            }
        )

        # Record top-K pairs for this fold
        flat_idx = np.argsort(joint_score.ravel())[::-1][:top_k_pairs_per_fold]
        for idx in flat_idx:
            t_i, p_j = divmod(int(idx), len(pw_cols))
            key = (sp_cols[t_i], pw_cols[p_j])
            if key not in pair_records:
                pair_records[key] = {
                    "bacterium": sp_cols[t_i],
                    "pathway": pw_cols[p_j],
                    "target": target_name,
                    "fold_hits": 0,
                    "scores": [],
                }
            pair_records[key]["fold_hits"] += 1
            pair_records[key]["scores"].append(float(joint_score[t_i, p_j]))

    rows = []
    for key, rec in pair_records.items():
        rows.append(
            {
                "bacterium": rec["bacterium"],
                "pathway": rec["pathway"],
                "target": rec["target"],
                "fold_hits": rec["fold_hits"],
                "stability": rec["fold_hits"] / 5.0,
                "mean_teamwork_score": float(np.mean(rec["scores"])),
                "is_stable_ge4of5": rec["fold_hits"] >= 4,
            }
        )

    pairs_df = pd.DataFrame(rows).sort_values(
        ["fold_hits", "mean_teamwork_score"], ascending=[False, False]
    ).reset_index(drop=True)

    rho, _ = spearmanr(oof_true, oof_preds)
    summary = {
        "target": target_name,
        "oof_mae": float(mean_absolute_error(oof_true, oof_preds)),
        "oof_rmse": float(np.sqrt(mean_squared_error(oof_true, oof_preds))),
        "oof_r2": float(r2_score(oof_true, oof_preds)),
        "oof_spearman_rho": float(rho) if np.isfinite(rho) else 0.0,
        "n_unique_pairs_top_k": len(pairs_df),
        "n_stable_pairs_ge4of5": int(pairs_df["is_stable_ge4of5"].sum()) if len(pairs_df) > 0 else 0,
        "fold_metrics": fold_metrics,
    }
    return pairs_df, summary
