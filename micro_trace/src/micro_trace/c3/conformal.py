"""Phase P3 (C3) Uncertainty Calibration & Selective Abstention.

Implements:
  - U0: Homoscedastic Cross-Conformal intervals around point predictor
  - U1: Quantile Regression intervals (HistGradientBoosting quantile 0.05 / 0.95)
  - U2: 5-fold Participant-Grouped Cross-Conformal (CV+) with locally-weighted
        residuals r_i = |y_i - y_hat_i| / sigma_hat(x_i)
  - U3: Selective Abstention (Accept_i = 1 if width_i <= tau) with train-fold-only
        tau calibration and Risk-Coverage curves
  - Subgroup Calibration Audit across diagnosis, FC range, and forecast horizon
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNetCV, RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler


def conformal_quantile(scores: np.ndarray, alpha: float = 0.10) -> float:
    """Finite-sample conformal quantile at level ceil((n + 1) * (1 - alpha)) / n."""
    s = np.asarray(scores, dtype=float)
    s = s[np.isfinite(s)]
    n = len(s)
    if n == 0:
        raise ValueError("Cannot compute conformal quantile on empty scores.")
    level = min(1.0, float(np.ceil((n + 1) * (1.0 - alpha)) / n))
    return float(np.quantile(s, level, method="higher"))


def participant_cluster_conformal_quantile(
    scores: np.ndarray,
    participant_ids: np.ndarray,
    alpha: float = 0.10,
) -> float:
    """Cluster-aware conformal quantile: 1/m_p weighted empirical distribution across participants."""
    s = np.asarray(scores, dtype=float)
    pids = np.asarray(participant_ids)
    valid = np.isfinite(s)
    s = s[valid]
    pids = pids[valid]
    if len(s) == 0:
        raise ValueError("Empty scores for cluster conformal quantile.")

    unique_pids, counts = np.unique(pids, return_counts=True)
    pid_to_weight = {pid: 1.0 / (len(unique_pids) * cnt) for pid, cnt in zip(unique_pids, counts)}
    weights = np.array([pid_to_weight[pid] for pid in pids], dtype=float)
    weights = weights / weights.sum()

    order = np.argsort(s)
    s_sorted = s[order]
    w_sorted = weights[order]
    cdf = np.cumsum(w_sorted)
    target = min(1.0, 1.0 - alpha + 1.0 / (len(unique_pids) + 1.0))
    idx = int(np.searchsorted(cdf, target, side="left"))
    idx = min(idx, len(s_sorted) - 1)
    return float(s_sorted[idx])


def compute_interval_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> dict[str, float]:
    """Compute empirical coverage, mean/median width, MAE, and RMSE."""
    y_t = np.asarray(y_true, dtype=float)
    y_p = np.asarray(y_pred, dtype=float)
    lo = np.asarray(lower, dtype=float)
    hi = np.asarray(upper, dtype=float)

    covered = (y_t >= lo) & (y_t <= hi)
    widths = np.maximum(0.0, hi - lo)
    return {
        "n": int(len(y_t)),
        "coverage": float(np.mean(covered)) if len(y_t) > 0 else np.nan,
        "mean_width": float(np.mean(widths)) if len(y_t) > 0 else np.nan,
        "median_width": float(np.median(widths)) if len(y_t) > 0 else np.nan,
        "width_std": float(np.std(widths)) if len(y_t) > 0 else np.nan,
        "mae": float(mean_absolute_error(y_t, y_p)) if len(y_t) > 0 else np.nan,
        "rmse": float(np.sqrt(mean_squared_error(y_t, y_p))) if len(y_t) > 0 else np.nan,
    }


def _fit_base_point_and_sigma(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    p_tr: np.ndarray,
    X_va: np.ndarray,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fit base point predictor and locally-weighted uncertainty scaler sigma_hat(x) strictly on train."""
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_va_s = scaler.transform(X_va)

    # Base point predictor: ElasticNetCV (strongest tabular family in C1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        base = ElasticNetCV(l1_ratio=[0.1, 0.5, 0.9], cv=4, random_state=seed, max_iter=3000, tol=1e-3)
        base.fit(X_tr_s, y_tr)

    # Inner-CV OOF residuals on training set to fit conformal quantile & sigma_hat(x) without leakage
    n_unique_p = len(np.unique(p_tr))
    n_inner = min(4, max(2, n_unique_p))
    gkf = GroupKFold(n_splits=n_inner)
    oof_tr_pred = np.zeros(len(y_tr), dtype=float)
    oof_tr_rf_std = np.zeros(len(y_tr), dtype=float)

    for in_tr, in_cal in gkf.split(X_tr_s, y_tr, groups=p_tr):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m_in = ElasticNetCV(l1_ratio=[0.1, 0.5, 0.9], cv=3, random_state=seed, max_iter=2000, tol=1e-3)
            m_in.fit(X_tr_s[in_tr], y_tr[in_tr])
            oof_tr_pred[in_cal] = m_in.predict(X_tr_s[in_cal])

        rf_in = RandomForestRegressor(
            n_estimators=100, max_depth=8, min_samples_leaf=5, random_state=seed, n_jobs=-1
        )
        rf_in.fit(X_tr_s[in_tr], y_tr[in_tr])
        tree_preds = np.array([t.predict(X_tr_s[in_cal]) for t in rf_in.estimators_])
        oof_tr_rf_std[in_cal] = np.std(tree_preds, axis=0)

    # Fit full-train RF for validation tree-spread std
    rf_full = RandomForestRegressor(
        n_estimators=100, max_depth=8, min_samples_leaf=5, random_state=seed, n_jobs=-1
    )
    rf_full.fit(X_tr_s, y_tr)
    va_tree_preds = np.array([t.predict(X_va_s) for t in rf_full.estimators_])
    va_rf_std = np.std(va_tree_preds, axis=0)

    # Auxiliary residual predictor on |y_tr - oof_tr_pred| (strictly OOF for sigma_tr)
    abs_res_tr = np.abs(y_tr - oof_tr_pred)
    log_abs_res_tr = np.log1p(abs_res_tr)
    oof_tr_res_hat = np.zeros(len(y_tr), dtype=float)
    for in_tr, in_cal in gkf.split(X_tr_s, y_tr, groups=p_tr):
        res_in = RidgeCV(alphas=np.logspace(-2, 3, 12))
        res_in.fit(X_tr_s[in_tr], log_abs_res_tr[in_tr])
        oof_tr_res_hat[in_cal] = np.expm1(np.clip(res_in.predict(X_tr_s[in_cal]), 0.0, 3.0))

    res_reg = RidgeCV(alphas=np.logspace(-2, 3, 12))
    res_reg.fit(X_tr_s, log_abs_res_tr)
    va_res_hat = np.expm1(np.clip(res_reg.predict(X_va_s), 0.0, 3.0))

    # Combine RF tree spread + OOF residual magnitude predictor, floored at sigma_min
    sigma_tr = 0.5 * oof_tr_rf_std + 0.5 * oof_tr_res_hat
    sigma_va = 0.5 * va_rf_std + 0.5 * va_res_hat
    sigma_floor = max(0.05, float(np.percentile(sigma_tr, 5)))
    sigma_tr = np.maximum(sigma_tr, sigma_floor)
    sigma_va = np.maximum(sigma_va, sigma_floor)

    y_hat_va = base.predict(X_va_s)
    return oof_tr_pred, y_hat_va, sigma_tr, sigma_va


def evaluate_uncertainty_systems_cv(
    fold_data: list[dict[str, Any]],
    alpha: float = 0.10,
    acceptance_grid: tuple[float, ...] = (1.0, 0.9, 0.8, 0.7, 0.6, 0.5),
    seed: int = 42,
) -> dict[str, Any]:
    """Evaluate U0, U1, U2, and U3 across 5 participant-grouped CV folds.

    Each element of fold_data must contain:
      - fold: int
      - X_tr, y_tr, p_tr
      - X_va, y_va, p_va
      - meta_va: DataFrame with optional columns ['diagnosis', 'index_fc_same_day', 'future_fc', 'elapsed_days']
    """
    oof_rows = []
    tau_records: dict[float, list[float]] = {acc: [] for acc in acceptance_grid}

    for fd in fold_data:
        fold = int(fd["fold"])
        X_tr = np.asarray(fd["X_tr"], dtype=float)
        y_tr = np.asarray(fd["y_tr"], dtype=float)
        p_tr = np.asarray(fd["p_tr"])
        X_va = np.asarray(fd["X_va"], dtype=float)
        y_va = np.asarray(fd["y_va"], dtype=float)
        p_va = np.asarray(fd["p_va"])
        meta_va = fd.get("meta_va")

        oof_tr_pred, y_hat_va, sigma_tr, sigma_va = _fit_base_point_and_sigma(
            X_tr, y_tr, p_tr, X_va, seed=seed + fold
        )

        # U0: Homoscedastic conformal
        abs_res_tr = np.abs(y_tr - oof_tr_pred)
        q_u0 = conformal_quantile(abs_res_tr, alpha=alpha)
        q_u0_cluster = participant_cluster_conformal_quantile(abs_res_tr, p_tr, alpha=alpha)
        u0_lo = y_hat_va - q_u0
        u0_hi = y_hat_va + q_u0

        # U1: Quantile Regression (HGB quantile 0.05, 0.50, 0.95)
        q_lo_m = HistGradientBoostingRegressor(
            loss="quantile", quantile=alpha / 2.0, max_depth=4, max_iter=150, random_state=seed + fold
        )
        q_mid_m = HistGradientBoostingRegressor(
            loss="quantile", quantile=0.50, max_depth=4, max_iter=150, random_state=seed + fold
        )
        q_hi_m = HistGradientBoostingRegressor(
            loss="quantile", quantile=1.0 - alpha / 2.0, max_depth=4, max_iter=150, random_state=seed + fold
        )
        q_lo_m.fit(X_tr, y_tr)
        q_mid_m.fit(X_tr, y_tr)
        q_hi_m.fit(X_tr, y_tr)
        u1_lo = q_lo_m.predict(X_va)
        u1_mid = q_mid_m.predict(X_va)
        u1_hi = np.maximum(u1_lo + 1e-4, q_hi_m.predict(X_va))

        # U2: Locally-Weighted Cross-Conformal (CV+)
        norm_res_tr = abs_res_tr / sigma_tr
        q_u2 = conformal_quantile(norm_res_tr, alpha=alpha)
        q_u2_cluster = participant_cluster_conformal_quantile(norm_res_tr, p_tr, alpha=alpha)
        u2_half = q_u2 * sigma_va
        u2_lo = y_hat_va - u2_half
        u2_hi = y_hat_va + u2_half

        # U3: Train-fold-only tau calibration for each target acceptance rate
        tr_widths_u2 = 2.0 * q_u2 * sigma_tr
        fold_taus = {
            acc: (float("inf") if acc >= 1.0 else float(np.quantile(tr_widths_u2, acc)))
            for acc in acceptance_grid
        }
        for acc, tau_val in fold_taus.items():
            tau_records[acc].append(tau_val)

        for i in range(len(y_va)):
            row: dict[str, Any] = {
                "fold": fold,
                "participant_id": p_va[i],
                "y_true": float(y_va[i]),
                "y_pred_base": float(y_hat_va[i]),
                "sigma_hat": float(sigma_va[i]),
                "u0_lo": float(u0_lo[i]),
                "u0_hi": float(u0_hi[i]),
                "u0_width": float(u0_hi[i] - u0_lo[i]),
                "u0_cluster_width": float(2.0 * q_u0_cluster),
                "u1_pred": float(u1_mid[i]),
                "u1_lo": float(u1_lo[i]),
                "u1_hi": float(u1_hi[i]),
                "u1_width": float(u1_hi[i] - u1_lo[i]),
                "u2_lo": float(u2_lo[i]),
                "u2_hi": float(u2_hi[i]),
                "u2_width": float(u2_hi[i] - u2_lo[i]),
                "u2_cluster_lo": float(y_hat_va[i] - q_u2_cluster * sigma_va[i]),
                "u2_cluster_hi": float(y_hat_va[i] + q_u2_cluster * sigma_va[i]),
            }
            for acc in acceptance_grid:
                row[f"u3_accept_{int(round(acc * 100))}"] = int((u2_hi[i] - u2_lo[i]) <= fold_taus[acc])
            if meta_va is not None and i < len(meta_va):
                for col in ["diagnosis", "index_fc_same_day", "future_fc", "elapsed_days"]:
                    if col in meta_va.columns:
                        row[col] = meta_va.iloc[i][col]
            oof_rows.append(row)

    oof_df = pd.DataFrame(oof_rows)

    system_summary = {
        "U0_homoscedastic_conformal": compute_interval_metrics(
            oof_df["y_true"], oof_df["y_pred_base"], oof_df["u0_lo"], oof_df["u0_hi"]
        ),
        "U1_quantile_regression": compute_interval_metrics(
            oof_df["y_true"], oof_df["u1_pred"], oof_df["u1_lo"], oof_df["u1_hi"]
        ),
        "U2_locally_weighted_cv_plus": compute_interval_metrics(
            oof_df["y_true"], oof_df["y_pred_base"], oof_df["u2_lo"], oof_df["u2_hi"]
        ),
        "U2_cluster_weighted_sensitivity": compute_interval_metrics(
            oof_df["y_true"], oof_df["y_pred_base"], oof_df["u2_cluster_lo"], oof_df["u2_cluster_hi"]
        ),
    }

    # U3 Risk-Coverage curve
    risk_cov_rows = []
    for acc in acceptance_grid:
        col = f"u3_accept_{int(round(acc * 100))}"
        accepted = oof_df[oof_df[col] == 1]
        m = compute_interval_metrics(
            accepted["y_true"], accepted["y_pred_base"], accepted["u2_lo"], accepted["u2_hi"]
        )
        m["target_acceptance"] = float(acc)
        m["observed_acceptance"] = float(len(accepted) / len(oof_df)) if len(oof_df) > 0 else 0.0
        finite_taus = [t for t in tau_records[acc] if np.isfinite(t)]
        m["mean_train_tau"] = float(np.mean(finite_taus)) if finite_taus else float("inf")
        risk_cov_rows.append(m)

    risk_cov_df = pd.DataFrame(risk_cov_rows)
    subgroup_df = compute_subgroup_calibration_audit(oof_df)

    return {
        "oof_df": oof_df,
        "system_summary": system_summary,
        "risk_coverage_df": risk_cov_df,
        "subgroup_audit_df": subgroup_df,
    }


def compute_subgroup_calibration_audit(oof_df: pd.DataFrame) -> pd.DataFrame:
    """Audit U0, U1, U2 coverage, width, and MAE across diagnosis, FC range, and horizon."""
    df = oof_df.copy()

    subgroups: list[tuple[str, pd.Series]] = [("overall", pd.Series(True, index=df.index))]

    if "diagnosis" in df.columns:
        for diag in ["CD", "UC", "nonIBD"]:
            subgroups.append((f"diagnosis={diag}", df["diagnosis"] == diag))

    if "future_fc" in df.columns:
        fc = pd.to_numeric(df["future_fc"], errors="coerce")
        subgroups.append(("fc_range=remission_le50", fc <= 50.0))
        subgroups.append(("fc_range=mild_50_250", (fc > 50.0) & (fc < 250.0)))
        subgroups.append(("fc_range=flare_ge250", fc >= 250.0))

    if "elapsed_days" in df.columns:
        dt = pd.to_numeric(df["elapsed_days"], errors="coerce")
        subgroups.append(("horizon=short_le35d", dt <= 35.0))
        subgroups.append(("horizon=mid_36_70d", (dt > 35.0) & (dt <= 70.0)))
        subgroups.append(("horizon=long_gt70d", dt > 70.0))

    rows = []
    for name, mask in subgroups:
        sub = df[mask]
        if len(sub) == 0:
            continue
        for sys_name, pred_col, lo_col, hi_col in [
            ("U0", "y_pred_base", "u0_lo", "u0_hi"),
            ("U1", "u1_pred", "u1_lo", "u1_hi"),
            ("U2", "y_pred_base", "u2_lo", "u2_hi"),
        ]:
            m = compute_interval_metrics(sub["y_true"], sub[pred_col], sub[lo_col], sub[hi_col])
            m["subgroup"] = name
            m["system"] = sys_name
            rows.append(m)

    return pd.DataFrame(rows)
