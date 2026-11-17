"""Subtype-stratified Phase P2 (C2) analysis across CD, UC, and nonIBD.

Runs the full C2 Guild + Species + Pathway participant random-intercept LME
pipeline (with BH-FDR computed within each subgroup & hypothesis family) and
5-fold participant-grouped stability selection independently within each
diagnosis subgroup (CD, UC, nonIBD) on the 75 development participants only,
keeping the 18 held-out test participants 100% untouched.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from micro_trace.c2.lme_signatures import (
    GUILD_FEATURE_COLS,
    build_c2_current_fc_cohort,
    build_c2_future_fc_cohort,
    fit_transform_clr_train_only,
    run_5fold_stability_selection,
    run_lme_family_scan,
)

SUBTYPES: tuple[str, ...] = ("CD", "UC", "nonIBD")


def _safe_float(val: Any) -> float | None:
    try:
        f = float(val)
    except (TypeError, ValueError):
        return None
    return f if np.isfinite(f) else None


def _ensure_five_participant_folds(sub_df: pd.DataFrame) -> pd.DataFrame:
    """Use existing cv_fold if all 5 folds (0..4) are populated; else assign deterministically by participant."""
    folds = sorted([int(f) for f in sub_df["cv_fold"].unique() if int(f) >= 0])
    if folds == [0, 1, 2, 3, 4]:
        return sub_df
    pids = sorted(sub_df["participant_id"].astype(str).unique().tolist())
    if len(pids) < 5:
        raise ValueError(
            f"Subgroup requires at least 5 participants for 5-fold stability selection, got {len(pids)}."
        )
    pid_to_fold = {pid: idx % 5 for idx, pid in enumerate(pids)}
    out = sub_df.copy()
    out["cv_fold"] = out["participant_id"].astype(str).map(pid_to_fold).astype(int)
    return out


def run_subtype_stratified_analysis(
    curr_dev: pd.DataFrame,
    fut_dev: pd.DataFrame,
    species_relab: pd.DataFrame,
    pathways_relab: pd.DataFrame,
    prevalence_threshold: float = 0.10,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Run full LME + BH-FDR and 5-fold stability selection separately for CD, UC, and nonIBD."""
    # Verify zero test leakage in inputs
    if "partition" in curr_dev.columns and not (curr_dev["partition"] == "dev").all():
        raise ValueError("curr_dev must contain only 'dev' partition rows.")
    if "partition" in fut_dev.columns and not (fut_dev["partition"] == "dev").all():
        raise ValueError("fut_dev must contain only 'dev' partition rows.")

    curr_covs = ["antibiotics_flag", "immunosuppressants_flag", "consent_age", "sex_male"]
    fut_covs = ["index_log10_fc_same_day", "elapsed_days"] + curr_covs

    lme_frames: list[pd.DataFrame] = []
    stab_frames: list[pd.DataFrame] = []
    cohort_stats: dict[str, dict[str, Any]] = {"current_fc": {}, "future_fc": {}}
    stability_summaries: dict[str, dict[str, Any]] = {"current_fc": {}, "future_fc": {}}

    for target_name, df_dev, covs in [
        ("current_fc", curr_dev, curr_covs),
        ("future_fc", fut_dev, fut_covs),
    ]:
        seen_pids_across_subtypes: set[str] = set()
        for subtype in SUBTYPES:
            sub_df = df_dev[df_dev["diagnosis"].astype(str) == subtype].reset_index(drop=True)
            if len(sub_df) == 0:
                raise ValueError(f"No rows found for subtype={subtype} in target={target_name}.")

            # Strict subgroup isolation check
            sub_pids = set(sub_df["participant_id"].astype(str).unique())
            overlap = seen_pids_across_subtypes & sub_pids
            if overlap:
                raise ValueError(f"Participant overlap detected across subtypes: {sorted(overlap)}")
            seen_pids_across_subtypes.update(sub_pids)
            if set(sub_df["diagnosis"].astype(str).unique()) != {subtype}:
                raise ValueError(f"Cross-subtype contamination detected in {subtype} subset.")

            sub_df = _ensure_five_participant_folds(sub_df)
            sample_ids = sub_df["clean_mgx_sample_id"].astype(str).tolist()

            # Train/subgroup-only CLR preprocessing (zero cross-subtype contamination)
            sp_clr, _, sp_cols, _ = fit_transform_clr_train_only(
                species_relab, sample_ids, prevalence_threshold=prevalence_threshold
            )
            pw_clr, _, pw_cols, _ = fit_transform_clr_train_only(
                pathways_relab, sample_ids, prevalence_threshold=prevalence_threshold
            )

            sp_clr_aligned = sp_clr.loc[sample_ids].reset_index(drop=True)
            pw_clr_aligned = pw_clr.loc[sample_ids].reset_index(drop=True)
            guild_cols = [c for c in GUILD_FEATURE_COLS if c in sub_df.columns]
            guild_aligned = sub_df[guild_cols].astype(float).reset_index(drop=True)

            # LME scans with BH-FDR computed strictly within (subtype, family, target)
            for fam, mat in [
                ("guild", guild_aligned),
                ("species", sp_clr_aligned),
                ("pathway", pw_clr_aligned),
            ]:
                lme_df = run_lme_family_scan(sub_df, mat, fam, target_name, covs)
                lme_df.insert(0, "diagnosis_subgroup", subtype)
                lme_frames.append(lme_df)

            # 5-fold participant-grouped stability selection within subgroup
            stab_df, stab_sum = run_5fold_stability_selection(
                sub_df,
                species_relab,
                pathways_relab,
                target_name,
                covs,
                prevalence_threshold=prevalence_threshold,
                seed=seed,
            )
            stab_df.insert(0, "diagnosis_subgroup", subtype)
            stab_frames.append(stab_df)

            cohort_stats[target_name][subtype] = {
                "n_obs": int(len(sub_df)),
                "n_participants": int(len(sub_pids)),
                "n_guilds_tested": int(len(guild_cols)),
                "n_species_tested": int(len(sp_cols)),
                "n_pathways_tested": int(len(pw_cols)),
            }
            stability_summaries[target_name][subtype] = stab_sum

    lme_all_df = pd.concat(lme_frames, ignore_index=True)
    stab_all_df = pd.concat(stab_frames, ignore_index=True)

    comparison = build_cross_subtype_comparison_summary(
        lme_all_df=lme_all_df,
        stab_all_df=stab_all_df,
        cohort_stats=cohort_stats,
        stability_summaries=stability_summaries,
        seed=seed,
    )
    return lme_all_df, stab_all_df, comparison


def _compute_pairwise_beta_corr(
    df_t: pd.DataFrame,
    s1: str,
    s2: str,
    family: str | None = None,
) -> dict[str, Any]:
    sub1 = df_t[df_t["diagnosis_subgroup"] == s1]
    sub2 = df_t[df_t["diagnosis_subgroup"] == s2]
    if family is not None:
        sub1 = sub1[sub1["family"] == family]
        sub2 = sub2[sub2["family"] == family]

    merged = sub1[["feature", "family", "beta", "p_value", "fdr_bh"]].merge(
        sub2[["feature", "family", "beta", "p_value", "fdr_bh"]],
        on=["feature", "family"],
        suffixes=(f"_{s1}", f"_{s2}"),
    )
    merged = merged.dropna(subset=[f"beta_{s1}", f"beta_{s2}"])
    n_shared = int(len(merged))
    if n_shared < 3:
        return {
            "n_shared_features": n_shared,
            "pearson_r": None,
            "pearson_p": None,
            "spearman_rho": None,
            "spearman_p": None,
            "sign_concordance_all": None,
            "n_nominal_union": 0,
            "sign_concordance_nominal_union": None,
        }

    b1 = merged[f"beta_{s1}"].astype(float).values
    b2 = merged[f"beta_{s2}"].astype(float).values
    if np.std(b1) > 1e-12 and np.std(b2) > 1e-12:
        pr, pp = stats.pearsonr(b1, b2)
        sr, sp = stats.spearmanr(b1, b2)
    else:
        pr, pp, sr, sp = np.nan, np.nan, np.nan, np.nan

    same_sign = np.sign(b1) == np.sign(b2)
    nom_union = merged[
        (merged[f"p_value_{s1}"] < 0.05) | (merged[f"p_value_{s2}"] < 0.05)
    ]
    if len(nom_union) > 0:
        nom_same = float(
            np.mean(
                np.sign(nom_union[f"beta_{s1}"].values)
                == np.sign(nom_union[f"beta_{s2}"].values)
            )
        )
    else:
        nom_same = None

    return {
        "n_shared_features": n_shared,
        "pearson_r": _safe_float(pr),
        "pearson_p": _safe_float(pp),
        "spearman_rho": _safe_float(sr),
        "spearman_p": _safe_float(sp),
        "sign_concordance_all": float(np.mean(same_sign)),
        "n_nominal_union": int(len(nom_union)),
        "sign_concordance_nominal_union": nom_same,
    }


def _overlap_breakdown(sets_by_subtype: dict[str, set[str]]) -> dict[str, Any]:
    cd = sets_by_subtype.get("CD", set())
    uc = sets_by_subtype.get("UC", set())
    ni = sets_by_subtype.get("nonIBD", set())
    return {
        "counts": {
            "CD": len(cd),
            "UC": len(uc),
            "nonIBD": len(ni),
            "CD_only": len(cd - uc - ni),
            "UC_only": len(uc - cd - ni),
            "nonIBD_only": len(ni - cd - uc),
            "CD_and_UC_shared": len(cd & uc),
            "CD_and_nonIBD_shared": len(cd & ni),
            "UC_and_nonIBD_shared": len(uc & ni),
            "all_three_shared": len(cd & uc & ni),
        },
        "CD_and_UC_shared_features": sorted(cd & uc),
        "all_three_shared_features": sorted(cd & uc & ni),
        "CD_only_top25": sorted(cd - uc - ni)[:25],
        "UC_only_top25": sorted(uc - cd - ni)[:25],
        "nonIBD_only_top25": sorted(ni - cd - uc)[:25],
    }


def build_cross_subtype_comparison_summary(
    lme_all_df: pd.DataFrame,
    stab_all_df: pd.DataFrame,
    cohort_stats: dict[str, dict[str, Any]],
    stability_summaries: dict[str, dict[str, Any]],
    seed: int = 42,
) -> dict[str, Any]:
    """Build structured cross-subtype comparison summary JSON."""
    targets_summary: dict[str, Any] = {}

    for target_name in ["current_fc", "future_fc"]:
        df_t = lme_all_df[lme_all_df["target"] == target_name].copy()
        st_t = stab_all_df[stab_all_df["target"] == target_name].copy()

        sig_counts: dict[str, dict[str, dict[str, int]]] = {}
        top_hits: dict[str, list[dict[str, Any]]] = {}
        fdr10_hits: dict[str, list[dict[str, Any]]] = {}
        stable_hits: dict[str, list[dict[str, Any]]] = {}

        fdr05_sets: dict[str, set[str]] = {}
        fdr10_sets: dict[str, set[str]] = {}
        nom05_sets: dict[str, set[str]] = {}
        stab4_sets: dict[str, set[str]] = {}

        for subtype in SUBTYPES:
            sub_lme = df_t[df_t["diagnosis_subgroup"] == subtype].sort_values(
                ["fdr_bh", "p_value", "feature"], ascending=[True, True, True]
            )
            sub_st = st_t[st_t["diagnosis_subgroup"] == subtype].sort_values(
                ["lme_sig_folds", "enet_sel_folds", "rf_top_folds", "feature"],
                ascending=[False, False, False, True],
            )

            sig_counts[subtype] = {}
            for fam in ["guild", "species", "pathway", "all"]:
                fam_df = sub_lme if fam == "all" else sub_lme[sub_lme["family"] == fam]
                sig_counts[subtype][fam] = {
                    "n_tested": int(len(fam_df)),
                    "fdr_lt_0_05": int((fam_df["fdr_bh"] < 0.05).sum()),
                    "fdr_lt_0_10": int((fam_df["fdr_bh"] < 0.10).sum()),
                    "nominal_p_lt_0_05": int((fam_df["p_value"] < 0.05).sum()),
                }

            fdr05_sets[subtype] = set(sub_lme.loc[sub_lme["fdr_bh"] < 0.05, "feature"].astype(str))
            fdr10_sets[subtype] = set(sub_lme.loc[sub_lme["fdr_bh"] < 0.10, "feature"].astype(str))
            nom05_sets[subtype] = set(sub_lme.loc[sub_lme["p_value"] < 0.05, "feature"].astype(str))
            stab4_sets[subtype] = set(
                sub_st.loc[sub_st["is_stable_lme_ge4of5"].astype(bool), "feature"].astype(str)
            )

            def _row_to_dict(r: pd.Series) -> dict[str, Any]:
                return {
                    "feature": str(r["feature"]),
                    "family": str(r["family"]),
                    "beta": _safe_float(r["beta"]),
                    "se": _safe_float(r["se"]),
                    "p_value": _safe_float(r["p_value"]),
                    "fdr_bh": _safe_float(r["fdr_bh"]),
                    "direction": str(r["direction"]),
                }

            top_hits[subtype] = [_row_to_dict(r) for _, r in sub_lme.head(10).iterrows()]
            fdr10_hits[subtype] = [
                _row_to_dict(r) for _, r in sub_lme[sub_lme["fdr_bh"] < 0.10].iterrows()
            ]
            stable_hits[subtype] = [
                {
                    "feature": str(r["feature"]),
                    "family": str(r["family"]),
                    "lme_sig_folds": int(r["lme_sig_folds"]),
                    "lme_stability": _safe_float(r["lme_stability"]),
                    "lme_mean_beta": _safe_float(r["lme_mean_beta"]),
                    "enet_sel_folds": int(r["enet_sel_folds"]),
                    "rf_top_folds": int(r["rf_top_folds"]),
                }
                for _, r in sub_st[sub_st["is_stable_lme_ge4of5"].astype(bool)].head(15).iterrows()
            ]

        pairwise_corr: dict[str, dict[str, Any]] = {}
        for s1, s2 in [("CD", "UC"), ("CD", "nonIBD"), ("UC", "nonIBD")]:
            pair_key = f"{s1}_vs_{s2}"
            pairwise_corr[pair_key] = {
                "all": _compute_pairwise_beta_corr(df_t, s1, s2, family=None),
                "guild": _compute_pairwise_beta_corr(df_t, s1, s2, family="guild"),
                "species": _compute_pairwise_beta_corr(df_t, s1, s2, family="species"),
                "pathway": _compute_pairwise_beta_corr(df_t, s1, s2, family="pathway"),
            }

        targets_summary[target_name] = {
            "cohorts": cohort_stats[target_name],
            "significance_counts": sig_counts,
            "stability_selection": stability_summaries[target_name],
            "pairwise_beta_correlations": pairwise_corr,
            "overlap_fdr_lt_0_05": _overlap_breakdown(fdr05_sets),
            "overlap_fdr_lt_0_10": _overlap_breakdown(fdr10_sets),
            "overlap_nominal_p_lt_0_05": _overlap_breakdown(nom05_sets),
            "overlap_stable_lme_ge4of5": _overlap_breakdown(stab4_sets),
            "fdr_lt_0_10_hits": fdr10_hits,
            "top_10_lme_hits": top_hits,
            "top_stable_lme_ge4of5_hits": stable_hits,
        }

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "test_partition_status": "untouched_held_out",
        "subtypes": list(SUBTYPES),
        "targets": targets_summary,
    }


def run_subtype_stratified_c2_pipeline(
    project_root: Path | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    """Execute the full subtype-stratified C2 pipeline on real HMP2 dev data and save artifacts."""
    root = project_root or Path(__file__).resolve().parents[3]
    if not (root / "micro_trace").exists():
        root = root.parent

    processed_dir = root / "micro-trace-data" / "processed" / "c1"
    c1_reports = root / "micro_trace" / "reports" / "c1"
    c2_reports = root / "micro_trace" / "reports" / "c2"
    c2_reports.mkdir(parents=True, exist_ok=True)

    metadata_path = root / "DataSet" / "hmp2" / "metadata" / "hmp2_metadata_2018-08-20.csv"
    store_path = processed_dir / "visit_trajectory_store.csv.gz"
    split_path = c1_reports / "split_audit_80_20.csv"
    consec_path = c1_reports / "c1_cohort_consecutive_0d.csv"

    species_relab = pd.read_csv(processed_dir / "species_relab.csv.gz")
    pathways_relab = pd.read_csv(processed_dir / "pathways_unstrat_relab.csv.gz")

    curr_all = build_c2_current_fc_cohort(store_path, split_path, metadata_path, dev_only=False)
    curr_dev = curr_all[curr_all["partition"] == "dev"].reset_index(drop=True)

    fut_all = build_c2_future_fc_cohort(consec_path, store_path, dev_only=False)
    fut_dev = fut_all[fut_all["partition"] == "dev"].reset_index(drop=True)

    # Strictly verify 18 held-out test participants remain untouched
    assert len(set(curr_dev["participant_id"]) & set(curr_all[curr_all["partition"] == "test"]["participant_id"])) == 0
    assert len(set(fut_dev["participant_id"]) & set(fut_all[fut_all["partition"] == "test"]["participant_id"])) == 0

    lme_all_df, stab_all_df, summary = run_subtype_stratified_analysis(
        curr_dev=curr_dev,
        fut_dev=fut_dev,
        species_relab=species_relab,
        pathways_relab=pathways_relab,
        prevalence_threshold=0.10,
        seed=seed,
    )

    lme_all_df.to_csv(c2_reports / "c2_subtype_full_lme.csv", index=False)
    stab_all_df.to_csv(c2_reports / "c2_subtype_full_stability.csv", index=False)
    with open(c2_reports / "c2_subtype_comparison_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary


if __name__ == "__main__":
    res = run_subtype_stratified_c2_pipeline()
    print(json.dumps(res, indent=2))
