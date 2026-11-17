"""End-to-end runner for Phase P2 (C2: Taxonomic, Functional, Guild & Cross-Omic Signatures)."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from micro_trace.c2.cross_omic_attention import train_and_extract_cross_omic_attention
from micro_trace.c2.lme_signatures import (
    GUILD_FEATURE_COLS,
    build_c2_current_fc_cohort,
    build_c2_future_fc_cohort,
    fit_transform_clr_train_only,
    run_5fold_stability_selection,
    run_lme_family_scan,
    run_subtype_sensitivity,
)


def run_c2_pipeline(project_root: Path | None = None, seed: int = 42) -> dict[str, Any]:
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

    # 1. Build cohorts (dev_only=True keeps the 18 test participants 100% untouched)
    curr_all = build_c2_current_fc_cohort(store_path, split_path, metadata_path, dev_only=False)
    curr_dev = curr_all[curr_all["partition"] == "dev"].reset_index(drop=True)

    fut_all = build_c2_future_fc_cohort(consec_path, store_path, dev_only=False)
    fut_dev = fut_all[fut_all["partition"] == "dev"].reset_index(drop=True)

    assert len(set(curr_dev["participant_id"]) & set(curr_all[curr_all["partition"] == "test"]["participant_id"])) == 0
    assert len(set(fut_dev["participant_id"]) & set(fut_all[fut_all["partition"] == "test"]["participant_id"])) == 0

    curr_covs = ["diagnosis", "antibiotics_flag", "immunosuppressants_flag", "consent_age", "sex_male"]
    fut_covs = ["index_log10_fc_same_day", "elapsed_days"] + curr_covs

    all_lme_results = []
    stability_summaries = {}
    cross_omic_summaries = {}
    subtype_results = []

    for target_name, df_dev, covs in [
        ("current_fc", curr_dev, curr_covs),
        ("future_fc", fut_dev, fut_covs),
    ]:
        sample_ids = df_dev["clean_mgx_sample_id"].astype(str).tolist()
        sp_clr, _, _, _ = fit_transform_clr_train_only(species_relab, sample_ids, prevalence_threshold=0.10)
        pw_clr, _, _, _ = fit_transform_clr_train_only(pathways_relab, sample_ids, prevalence_threshold=0.10)

        sp_clr_aligned = sp_clr.loc[sample_ids].reset_index(drop=True)
        pw_clr_aligned = pw_clr.loc[sample_ids].reset_index(drop=True)
        guild_cols = [c for c in GUILD_FEATURE_COLS if c in df_dev.columns]
        guild_aligned = df_dev[guild_cols].astype(float).reset_index(drop=True)

        # LME family scans with BH-FDR
        lme_guild = run_lme_family_scan(df_dev, guild_aligned, "guild", target_name, covs)
        lme_sp = run_lme_family_scan(df_dev, sp_clr_aligned, "species", target_name, covs)
        lme_pw = run_lme_family_scan(df_dev, pw_clr_aligned, "pathway", target_name, covs)
        all_lme_results.extend([lme_guild, lme_sp, lme_pw])

        # 5-fold stability selection
        stab_df, stab_sum = run_5fold_stability_selection(
            df_dev, species_relab, pathways_relab, target_name, covs, seed=seed
        )
        stab_df.to_csv(c2_reports / f"c2_stability_{target_name}.csv", index=False)
        stability_summaries[target_name] = stab_sum

        # Subtype directional consistency on guilds
        sub_df = run_subtype_sensitivity(df_dev, guild_aligned, guild_cols, target_name, covs)
        subtype_results.append(sub_df)

        # Cross-Omic Attention ([Bacteria] <-> [Pathways] <-> [FC])
        pairs_df, co_sum = train_and_extract_cross_omic_attention(
            df_dev, species_relab, pathways_relab, target_name, seed=seed
        )
        pairs_df.to_csv(c2_reports / f"c2_cross_omic_pairs_{target_name}.csv", index=False)
        cross_omic_summaries[target_name] = co_sum

    lme_all_df = pd.concat(all_lme_results, ignore_index=True)
    lme_all_df.to_csv(c2_reports / "c2_lme_all_associations.csv", index=False)

    subtype_all_df = pd.concat(subtype_results, ignore_index=True)
    subtype_all_df.to_csv(c2_reports / "c2_subtype_sensitivity.csv", index=False)

    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "test_partition_status": "untouched_held_out",
        "cohorts": {
            "current_fc_total_clean_visits": int(len(curr_all)),
            "current_fc_total_participants": int(curr_all["participant_id"].nunique()),
            "current_fc_dev_visits": int(len(curr_dev)),
            "current_fc_dev_participants": int(curr_dev["participant_id"].nunique()),
            "future_fc_total_clean_pairs": int(len(fut_all)),
            "future_fc_total_participants": int(fut_all["participant_id"].nunique()),
            "future_fc_dev_pairs": int(len(fut_dev)),
            "future_fc_dev_participants": int(fut_dev["participant_id"].nunique()),
        },
        "lme_fdr_lt_0_10_counts": {
            f"{t}_{f}": int(
                ((lme_all_df["target"] == t) & (lme_all_df["family"] == f) & (lme_all_df["fdr_bh"] < 0.10)).sum()
            )
            for t in ["current_fc", "future_fc"]
            for f in ["guild", "species", "pathway"]
        },
        "lme_fdr_lt_0_05_counts": {
            f"{t}_{f}": int(
                ((lme_all_df["target"] == t) & (lme_all_df["family"] == f) & (lme_all_df["fdr_bh"] < 0.05)).sum()
            )
            for t in ["current_fc", "future_fc"]
            for f in ["guild", "species", "pathway"]
        },
        "stability_selection": stability_summaries,
        "cross_omic_attention": cross_omic_summaries,
    }

    with open(c2_reports / "c2_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary


if __name__ == "__main__":
    res = run_c2_pipeline()
    print(json.dumps(res, indent=2))
