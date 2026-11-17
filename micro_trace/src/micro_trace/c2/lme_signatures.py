"""Phase P2 (C2) Linear Mixed-Effects (LME), Guilds, and 5-Fold Stability Selection.

Implements:
  - Dual outcome scope (ADR-002):
      1. Current FC (same-day MGX+FC visits, N=503 raw / 502 non-zero-sum)
      2. Future FC (consecutive next-FC pairs, N=445 total / 371 dev pairs)
  - Strict train-only compositional preprocessing (PR-03):
      prevalence >= 10% on dev/train fold, pseudocount = 0.5 * min positive, CLR.
  - Participant random-intercept Linear Mixed-Effects (1 | participant_id)
    with Benjamini-Hochberg FDR control per hypothesis family.
  - 5-fold participant-grouped stability selection (>= 4/5 folds) across
    LME, ElasticNetCV, and RandomForest, plus pairwise Jaccard stability.
  - Subtype directional consistency audit (CD, UC, nonIBD).
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import optimize, stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNetCV
from sklearn.preprocessing import StandardScaler

GUILD_FEATURE_COLS = [
    "guild_scfa_butyrate_log",
    "guild_pathobiont_log",
    "guild_protective_vs_pathobiont_logratio",
    "fguild_scfa_butanoate_log",
    "fguild_sulfur_h2s_log",
    "fguild_oxidative_stress_log",
    "fguild_mucin_hexuronide_log",
    "fguild_lps_enterobacterial_log",
    "shannon_species",
    "simpson_species",
    "species_richness",
]


def _encode_binary_flag(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.astype(float)
    s = series.astype(str).str.strip().str.lower()
    return s.isin({"yes", "true", "1", "1.0", "y"}).astype(float)


def _extract_metadata_covariates(metadata_path: Path) -> pd.DataFrame:
    """Extract visit-level clinical covariates from raw HMP2 metadata."""
    usecols = [
        "site_sub_coll",
        "Participant ID",
        "data_type",
        "consent_age",
        "sex",
        "Antibiotics",
        "Immunosuppressants (e.g. oral corticosteroids)",
    ]
    raw = pd.read_csv(metadata_path, usecols=lambda c: c in usecols, low_memory=False)
    mgx = raw[raw["data_type"] == "metagenomics"].copy()
    mgx = mgx.drop_duplicates(subset=["site_sub_coll"], keep="first")
    mgx = mgx.rename(
        columns={
            "site_sub_coll": "collection_event_id",
            "Participant ID": "participant_id",
            "Immunosuppressants (e.g. oral corticosteroids)": "immunosuppressants",
            "Antibiotics": "antibiotics",
        }
    )
    mgx["antibiotics_flag"] = _encode_binary_flag(mgx["antibiotics"])
    mgx["immunosuppressants_flag"] = _encode_binary_flag(mgx["immunosuppressants"])
    mgx["consent_age"] = pd.to_numeric(mgx["consent_age"], errors="coerce")
    mgx["sex_male"] = (mgx["sex"].astype(str).str.strip().str.lower() == "male").astype(float)
    return mgx[
        [
            "collection_event_id",
            "consent_age",
            "sex_male",
            "antibiotics_flag",
            "immunosuppressants_flag",
        ]
    ]


def build_c2_current_fc_cohort(
    trajectory_store_path: Path,
    split_audit_path: Path,
    metadata_path: Path | None = None,
    dev_only: bool = True,
) -> pd.DataFrame:
    """Build Current FC cohort (same-day MGX+FC visits, excluding zero-sum MGX)."""
    store = pd.read_csv(trajectory_store_path, low_memory=False)
    split = pd.read_csv(split_audit_path)

    same_day = store[store["has_mgx_sample"].astype(bool) & store["fc_log10"].notna()].copy()
    same_day = same_day[~same_day["is_zero_sum_mgx_sample"].astype(bool)].copy()

    if metadata_path is not None and metadata_path.exists():
        covs = _extract_metadata_covariates(metadata_path)
        same_day = same_day.merge(covs, on="collection_event_id", how="left")
    else:
        for col in ["consent_age", "sex_male", "antibiotics_flag", "immunosuppressants_flag"]:
            if col not in same_day.columns:
                same_day[col] = 0.0

    same_day = same_day.merge(
        split[["participant_id", "partition", "cv_fold"]],
        on="participant_id",
        how="left",
    )
    same_day["partition"] = same_day["partition"].fillna("unassigned")
    same_day["cv_fold"] = same_day["cv_fold"].fillna(-2).astype(int)

    if dev_only:
        same_day = same_day[same_day["partition"] == "dev"].copy()

    same_day["target_fc_log10"] = same_day["fc_log10"].astype(float)
    same_day = same_day.sort_values(["participant_id", "event_date", "visit_num"]).reset_index(drop=True)
    return same_day


def build_c2_future_fc_cohort(
    consecutive_cohort_path: Path,
    trajectory_store_path: Path,
    dev_only: bool = True,
) -> pd.DataFrame:
    """Build Future FC cohort (consecutive next-FC pairs, excluding zero-sum MGX)."""
    consec = pd.read_csv(consecutive_cohort_path, low_memory=False)
    store = pd.read_csv(trajectory_store_path, low_memory=False)

    consec = consec[~consec["is_zero_sum_mgx_sample"].astype(bool)].copy()
    if dev_only:
        consec = consec[consec["partition"] == "dev"].copy()

    consec["antibiotics_flag"] = _encode_binary_flag(consec["Antibiotics"])
    consec["immunosuppressants_flag"] = _encode_binary_flag(
        consec["Immunosuppressants (e.g. oral corticosteroids)"]
    )
    consec["consent_age"] = pd.to_numeric(consec["consent_age"], errors="coerce")
    consec["sex_male"] = (consec["sex"].astype(str).str.strip().str.lower() == "male").astype(float)

    guild_cols = ["collection_event_id", "clean_mgx_sample_id"] + [
        c for c in GUILD_FEATURE_COLS if c in store.columns
    ]
    store_sub = store[guild_cols].drop_duplicates(subset=["collection_event_id"], keep="first")

    merged = consec.merge(
        store_sub,
        left_on="index_collection_event",
        right_on="collection_event_id",
        how="left",
    )
    merged["target_fc_log10"] = merged["future_log10_fc_plus_1"].astype(float)
    merged["clean_mgx_sample_id"] = merged["clean_mgx_sample_id"].fillna(merged["index_mgx_sample_id"])
    merged = merged.sort_values(["participant_id", "index_date", "index_visit_num"]).reset_index(drop=True)
    return merged


def fit_transform_clr_train_only(
    relab_df: pd.DataFrame,
    train_sample_ids: list[str],
    eval_sample_ids: list[str] | None = None,
    prevalence_threshold: float = 0.10,
) -> tuple[pd.DataFrame, pd.DataFrame | None, list[str], float]:
    """Fit prevalence filter (>=10%) and pseudocount on train_sample_ids ONLY, then CLR-transform."""
    if "Unnamed: 0" in relab_df.columns:
        df = relab_df.set_index("Unnamed: 0")
    else:
        df = relab_df.copy()
    df.index = df.index.astype(str)

    train_idx = [s for s in train_sample_ids if s in df.index]
    if not train_idx:
        raise ValueError("No training sample IDs found in abundance table.")

    train_mat = df.loc[train_idx].astype(float)
    prevalence = (train_mat > 0.0).mean(axis=0)
    kept_cols = prevalence[prevalence >= prevalence_threshold].index.tolist()
    if not kept_cols:
        raise ValueError(f"No features passed prevalence >= {prevalence_threshold} in training set.")

    train_sub = train_mat[kept_cols]
    pos_vals = train_sub.values[train_sub.values > 0.0]
    pseudocount = 0.5 * float(np.min(pos_vals)) if len(pos_vals) > 0 else 1e-6

    def _clr(sub_df: pd.DataFrame) -> pd.DataFrame:
        x = np.log(sub_df.values + pseudocount)
        x_clr = x - np.mean(x, axis=1, keepdims=True)
        return pd.DataFrame(x_clr, index=sub_df.index, columns=sub_df.columns)

    train_clr = _clr(train_sub)
    eval_clr = None
    if eval_sample_ids is not None:
        eval_idx = [s for s in eval_sample_ids if s in df.index]
        eval_sub = df.loc[eval_idx, kept_cols].astype(float)
        eval_clr = _clr(eval_sub)

    return train_clr, eval_clr, kept_cols, pseudocount


def fit_lme_single_feature(
    df: pd.DataFrame,
    feature_values: np.ndarray,
    outcome_col: str = "target_fc_log10",
    group_col: str = "participant_id",
    covariate_cols: list[str] | None = None,
) -> dict[str, Any]:
    """Fit participant random-intercept LME: outcome ~ feature + covariates + (1 | participant_id)."""
    work = pd.DataFrame(
        {
            "y": pd.to_numeric(df[outcome_col], errors="coerce").values,
            "x": pd.to_numeric(feature_values, errors="coerce"),
            "group": df[group_col].astype(str).values,
        }
    )

    cov_names: list[str] = []
    if covariate_cols:
        for col in covariate_cols:
            if col not in df.columns:
                continue
            if col == "diagnosis":
                work["diag_CD"] = (df[col].astype(str) == "CD").astype(float).values
                work["diag_UC"] = (df[col].astype(str) == "UC").astype(float).values
                cov_names.extend(["diag_CD", "diag_UC"])
            else:
                vals = pd.to_numeric(df[col], errors="coerce").values
                if np.isnan(vals).any():
                    med = np.nanmedian(vals)
                    vals = np.where(np.isnan(vals), 0.0 if np.isnan(med) else med, vals)
                if np.nanstd(vals) > 1e-12:
                    work[col] = vals
                    cov_names.append(col)

    work = work.dropna(subset=["y", "x"]).reset_index(drop=True)
    n_obs = len(work)
    n_participants = work["group"].nunique()
    if n_obs < 15 or n_participants < 5 or np.std(work["x"]) < 1e-12:
        return {
            "beta": 0.0,
            "se": np.nan,
            "ci_low": np.nan,
            "ci_high": np.nan,
            "p_value": 1.0,
            "direction": "neutral",
            "converged": False,
            "method": "skipped",
            "n_obs": n_obs,
            "n_participants": n_participants,
        }

    # Standardize x for comparable effect sizes (SD change in outcome per 1-SD change in feature)
    x_std = (work["x"] - work["x"].mean()) / (work["x"].std(ddof=0) + 1e-12)
    work["x_std"] = x_std

    X_cols = ["x_std"] + cov_names
    X = np.column_stack([np.ones(n_obs, dtype=float), work[X_cols].values.astype(float)])
    y_vec = work["y"].values.astype(float)
    groups = work["group"].values

    unique_groups, group_inv = np.unique(groups, return_inverse=True)
    group_indices = [np.where(group_inv == g)[0] for g in range(len(unique_groups))]
    group_sizes = np.array([len(idx) for idx in group_indices], dtype=float)

    def _gls_given_gamma(gamma: float) -> tuple[float, np.ndarray, np.ndarray, float]:
        p = X.shape[1]
        xtx = np.zeros((p, p), dtype=float)
        xty = np.zeros(p, dtype=float)
        for idx, n_g in zip(group_indices, group_sizes):
            Xg = X[idx]
            yg = y_vec[idx]
            c = gamma / (1.0 + n_g * gamma) if gamma > 0 else 0.0
            Xg_sum = Xg.sum(axis=0, keepdims=True)
            yg_sum = yg.sum()
            xtx += Xg.T @ Xg - c * (Xg_sum.T @ Xg_sum)
            xty += Xg.T @ yg - c * (Xg_sum.ravel() * yg_sum)

        xtx_reg = xtx + 1e-10 * np.eye(p)
        beta_hat = np.linalg.solve(xtx_reg, xty)
        resid = y_vec - X @ beta_hat

        rss_weighted = 0.0
        for idx, n_g in zip(group_indices, group_sizes):
            rg = resid[idx]
            c = gamma / (1.0 + n_g * gamma) if gamma > 0 else 0.0
            rss_weighted += float(np.dot(rg, rg) - c * (rg.sum() ** 2))

        sigma2_e = max(1e-12, rss_weighted / n_obs)
        logdet = float(np.sum(np.log1p(group_sizes * gamma))) if gamma > 0 else 0.0
        nll2 = n_obs * np.log(sigma2_e) + logdet
        cov_beta = sigma2_e * np.linalg.inv(xtx_reg)
        return nll2, beta_hat, cov_beta, sigma2_e

    converged = False
    beta = 0.0
    se = np.nan
    p_val = 1.0
    method = "lme_profile_ml"

    try:
        opt = optimize.minimize_scalar(
            lambda g: _gls_given_gamma(float(g))[0],
            bounds=(0.0, 50.0),
            method="bounded",
            options={"xatol": 1e-4, "maxiter": 60},
        )
        gamma_opt = float(opt.x)
        # Check boundary gamma = 0.0 as well
        nll0, _, _, _ = _gls_given_gamma(0.0)
        if nll0 <= float(opt.fun):
            gamma_opt = 0.0
        _, beta_hat, cov_beta, _ = _gls_given_gamma(gamma_opt)
        beta = float(beta_hat[1])  # index 1 is x_std (index 0 is intercept)
        var_b = float(cov_beta[1, 1])
        if np.isfinite(beta) and np.isfinite(var_b) and var_b > 0:
            se = float(np.sqrt(var_b))
            z_stat = beta / se
            p_val = float(2.0 * stats.norm.sf(abs(z_stat)))
            converged = True
    except Exception:
        converged = False
        method = "failed"

    ci_low = float(beta - 1.96 * se) if np.isfinite(se) else np.nan
    ci_high = float(beta + 1.96 * se) if np.isfinite(se) else np.nan
    direction = "positive" if beta > 0 else ("negative" if beta < 0 else "neutral")

    return {
        "beta": beta,
        "se": se,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "p_value": p_val if np.isfinite(p_val) else 1.0,
        "direction": direction,
        "converged": bool(converged),
        "method": method,
        "n_obs": int(n_obs),
        "n_participants": int(n_participants),
    }


def run_lme_family_scan(
    cohort_df: pd.DataFrame,
    feature_df: pd.DataFrame,
    family_name: str,
    target_name: str,
    covariate_cols: list[str],
    outcome_col: str = "target_fc_log10",
) -> pd.DataFrame:
    """Run participant random-intercept LME across all features in a family with BH-FDR."""
    rows = []
    for col in feature_df.columns:
        res = fit_lme_single_feature(
            df=cohort_df,
            feature_values=feature_df[col].values,
            outcome_col=outcome_col,
            group_col="participant_id",
            covariate_cols=covariate_cols,
        )
        res["feature"] = col
        res["family"] = family_name
        res["target"] = target_name
        rows.append(res)

    out = pd.DataFrame(rows)
    if len(out) > 0:
        p_vals = out["p_value"].values.astype(float)
        n_tests = len(p_vals)
        order = np.argsort(p_vals)
        ranked = p_vals[order]
        q_vals = ranked * n_tests / np.arange(1, n_tests + 1)
        q_vals = np.minimum.accumulate(q_vals[::-1])[::-1]
        q_vals = np.clip(q_vals, 0.0, 1.0)
        fdr_vals = np.empty_like(q_vals)
        fdr_vals[order] = q_vals
        out["fdr_bh"] = fdr_vals
        out = out.sort_values(["fdr_bh", "p_value", "feature"], ascending=[True, True, True]).reset_index(drop=True)
    else:
        out["fdr_bh"] = []
    return out


def _pairwise_jaccard(fold_sets: list[set[str]]) -> float:
    if len(fold_sets) < 2:
        return 0.0
    scores = []
    for i in range(len(fold_sets)):
        for j in range(i + 1, len(fold_sets)):
            union = fold_sets[i] | fold_sets[j]
            if len(union) == 0:
                scores.append(0.0)
            else:
                scores.append(len(fold_sets[i] & fold_sets[j]) / len(union))
    return float(np.mean(scores)) if scores else 0.0


def run_5fold_stability_selection(
    cohort_df: pd.DataFrame,
    species_relab: pd.DataFrame,
    pathways_relab: pd.DataFrame,
    target_name: str,
    covariate_cols: list[str],
    prevalence_threshold: float = 0.10,
    seed: int = 42,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Run 5-fold participant-grouped stability selection (LME + ElasticNet + RF)."""
    folds = sorted([int(f) for f in cohort_df["cv_fold"].unique() if int(f) >= 0])
    if len(folds) != 5:
        raise ValueError(f"Expected 5 dev folds (0..4), got {folds}")

    enet_fold_sets: list[set[str]] = []
    rf_fold_sets: list[set[str]] = []
    lme_fold_sets: list[set[str]] = []

    feature_records: dict[str, dict[str, Any]] = {}

    def _ensure_rec(feat: str, fam: str) -> dict[str, Any]:
        if feat not in feature_records:
            feature_records[feat] = {
                "feature": feat,
                "family": fam,
                "target": target_name,
                "lme_sig_folds": 0,
                "lme_betas": [],
                "enet_sel_folds": 0,
                "enet_coefs": [],
                "rf_top_folds": 0,
                "rf_imps": [],
            }
        return feature_records[feat]

    for fold in folds:
        tr_df = cohort_df[cohort_df["cv_fold"] != fold].reset_index(drop=True)
        tr_ids = tr_df["clean_mgx_sample_id"].astype(str).tolist()

        sp_tr, _, sp_cols, _ = fit_transform_clr_train_only(
            species_relab, tr_ids, prevalence_threshold=prevalence_threshold
        )
        pw_tr, _, pw_cols, _ = fit_transform_clr_train_only(
            pathways_relab, tr_ids, prevalence_threshold=prevalence_threshold
        )

        # Align rows to tr_df
        sp_tr = sp_tr.loc[tr_ids].reset_index(drop=True)
        pw_tr = pw_tr.loc[tr_ids].reset_index(drop=True)
        guild_cols = [c for c in GUILD_FEATURE_COLS if c in tr_df.columns]
        guild_tr = tr_df[guild_cols].astype(float).reset_index(drop=True)

        # 1. Fold-level LME on guilds + species + pathways
        lme_set = set()
        for fam, mat in [("guild", guild_tr), ("species", sp_tr), ("pathway", pw_tr)]:
            scan = run_lme_family_scan(
                tr_df, mat, family_name=fam, target_name=target_name, covariate_cols=covariate_cols
            )
            for _, row in scan.iterrows():
                rec = _ensure_rec(row["feature"], fam)
                rec["lme_betas"].append(float(row["beta"]))
                if float(row["p_value"]) < 0.05:
                    rec["lme_sig_folds"] += 1
                    lme_set.add(str(row["feature"]))
        lme_fold_sets.append(lme_set)

        # 2. Combined multi-feature ElasticNetCV + RandomForest
        joint_X = pd.concat([guild_tr, sp_tr, pw_tr], axis=1)
        joint_names = joint_X.columns.tolist()
        joint_fams = (
            ["guild"] * len(guild_cols)
            + ["species"] * len(sp_cols)
            + ["pathway"] * len(pw_cols)
        )

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(joint_X.values)
        y_tr = tr_df["target_fc_log10"].astype(float).values

        enet = ElasticNetCV(
            l1_ratio=[0.1, 0.5, 0.9],
            cv=4,
            random_state=seed,
            max_iter=3000,
            tol=1e-3,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            enet.fit(X_scaled, y_tr)

        enet_set = set()
        for name, fam, coef in zip(joint_names, joint_fams, enet.coef_):
            rec = _ensure_rec(name, fam)
            rec["enet_coefs"].append(float(coef))
            if abs(float(coef)) > 1e-6:
                rec["enet_sel_folds"] += 1
                enet_set.add(name)
        enet_fold_sets.append(enet_set)

        rf = RandomForestRegressor(
            n_estimators=200, max_depth=8, min_samples_leaf=5, random_state=seed, n_jobs=-1
        )
        rf.fit(X_scaled, y_tr)
        imps = rf.feature_importances_
        top_k = min(30, len(joint_names))
        top_idx = set(np.argsort(imps)[::-1][:top_k].tolist())
        rf_set = set()
        for idx, (name, fam, imp) in enumerate(zip(joint_names, joint_fams, imps)):
            rec = _ensure_rec(name, fam)
            rec["rf_imps"].append(float(imp))
            if idx in top_idx and imp > 0:
                rec["rf_top_folds"] += 1
                rf_set.add(name)
        rf_fold_sets.append(rf_set)

    rows = []
    for feat, rec in feature_records.items():
        betas = rec["lme_betas"]
        coefs = rec["enet_coefs"]
        imps = rec["rf_imps"]
        sign_cons = 0.0
        if betas:
            pos_frac = np.mean([b > 0 for b in betas])
            neg_frac = np.mean([b < 0 for b in betas])
            sign_cons = float(max(pos_frac, neg_frac))

        rows.append(
            {
                "feature": feat,
                "family": rec["family"],
                "target": target_name,
                "lme_sig_folds": rec["lme_sig_folds"],
                "lme_stability": rec["lme_sig_folds"] / 5.0,
                "lme_mean_beta": float(np.mean(betas)) if betas else 0.0,
                "lme_sign_consistency": sign_cons,
                "enet_sel_folds": rec["enet_sel_folds"],
                "enet_stability": rec["enet_sel_folds"] / 5.0,
                "enet_mean_coef": float(np.mean(coefs)) if coefs else 0.0,
                "rf_top_folds": rec["rf_top_folds"],
                "rf_stability": rec["rf_top_folds"] / 5.0,
                "rf_mean_importance": float(np.mean(imps)) if imps else 0.0,
                "is_stable_lme_ge4of5": rec["lme_sig_folds"] >= 4 and sign_cons >= 0.8,
                "is_stable_enet_ge4of5": rec["enet_sel_folds"] >= 4,
                "is_stable_rf_ge4of5": rec["rf_top_folds"] >= 4,
            }
        )

    stab_df = pd.DataFrame(rows).sort_values(
        ["lme_sig_folds", "enet_sel_folds", "rf_top_folds", "lme_mean_beta"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)

    summary = {
        "target": target_name,
        "n_folds": len(folds),
        "jaccard_lme_p05": _pairwise_jaccard(lme_fold_sets),
        "jaccard_elasticnet": _pairwise_jaccard(enet_fold_sets),
        "jaccard_rf_top30": _pairwise_jaccard(rf_fold_sets),
        "n_stable_lme_ge4of5": int(stab_df["is_stable_lme_ge4of5"].sum()),
        "n_stable_enet_ge4of5": int(stab_df["is_stable_enet_ge4of5"].sum()),
        "n_stable_rf_ge4of5": int(stab_df["is_stable_rf_ge4of5"].sum()),
    }
    return stab_df, summary


def run_subtype_sensitivity(
    cohort_df: pd.DataFrame,
    feature_df: pd.DataFrame,
    features_to_check: list[str],
    target_name: str,
    covariate_cols: list[str],
) -> pd.DataFrame:
    """Check directional consistency across CD, UC, and nonIBD subtypes."""
    sub_covs = [c for c in covariate_cols if c != "diagnosis"]
    rows = []
    for diag in ["CD", "UC", "nonIBD"]:
        sub = cohort_df[cohort_df["diagnosis"] == diag].reset_index(drop=True)
        mask = (cohort_df["diagnosis"] == diag).values
        sub_feat = feature_df.loc[mask].reset_index(drop=True)
        for feat in features_to_check:
            if feat not in sub_feat.columns:
                continue
            res = fit_lme_single_feature(
                sub, sub_feat[feat].values, covariate_cols=sub_covs
            )
            res["feature"] = feat
            res["diagnosis_subgroup"] = diag
            res["target"] = target_name
            rows.append(res)
    return pd.DataFrame(rows)
