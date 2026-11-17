"""End-to-end runner for Phase P3 (C3: Uncertainty Calibration & Selective Abstention)."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from micro_trace.c1.features import C1FeaturePipeline
from micro_trace.c3.conformal import evaluate_uncertainty_systems_cv


def run_c3_pipeline(
    project_root: Path | None = None,
    arm: str = "M1",
    alpha: float = 0.10,
    seed: int = 42,
) -> dict[str, Any]:
    root = project_root or Path(__file__).resolve().parents[3]
    if not (root / "micro_trace").exists():
        root = root.parent

    processed_dir = root / "micro-trace-data" / "processed" / "c1"
    c1_reports = root / "micro_trace" / "reports" / "c1"
    c3_reports = root / "micro_trace" / "reports" / "c3"
    c3_reports.mkdir(parents=True, exist_ok=True)

    consec = pd.read_csv(c1_reports / "c1_cohort_consecutive_0d.csv", low_memory=False)
    dev_df = consec[consec["partition"] == "dev"].copy().reset_index(drop=True)
    test_df = consec[consec["partition"] == "test"].copy()

    # Strict zero-overlap check
    assert len(set(dev_df["participant_id"]) & set(test_df["participant_id"])) == 0

    store_df = pd.read_csv(processed_dir / "visit_trajectory_store.csv.gz", low_memory=False)
    omics_dict = {
        "species_relab": pd.read_csv(processed_dir / "species_relab.csv.gz", index_col=0),
        "genus_relab": pd.read_csv(processed_dir / "genus_relab.csv.gz", index_col=0),
        "pathways_unstrat_relab": pd.read_csv(processed_dir / "pathways_unstrat_relab.csv.gz", index_col=0),
    }

    folds = sorted([int(f) for f in dev_df["cv_fold"].unique() if int(f) >= 0])
    if len(folds) != 5:
        raise ValueError(f"Expected 5 dev folds (0..4), got {folds}")

    arms_to_run = [arm] if arm != "ALL" else ["M1", "M_guild_delta", "M5"]
    all_summaries: dict[str, Any] = {}

    for curr_arm in arms_to_run:
        fold_data = []
        for fold in folds:
            tr_df = dev_df[dev_df["cv_fold"] != fold].reset_index(drop=True)
            va_df = dev_df[dev_df["cv_fold"] == fold].reset_index(drop=True)

            pipe = C1FeaturePipeline(arm=curr_arm, prevalence_threshold=0.10)
            pipe.fit(tr_df, store_df, omics_dict)
            X_tr_arr, _ = pipe.transform(tr_df, store_df, omics_dict, standardize=False)
            X_va_arr, _ = pipe.transform(va_df, store_df, omics_dict, standardize=False)

            fold_data.append(
                {
                    "fold": fold,
                    "X_tr": X_tr_arr,
                    "y_tr": tr_df["future_log10_fc_plus_1"].astype(float).values,
                    "p_tr": tr_df["participant_id"].values,
                    "X_va": X_va_arr,
                    "y_va": va_df["future_log10_fc_plus_1"].astype(float).values,
                    "p_va": va_df["participant_id"].values,
                    "meta_va": va_df[["diagnosis", "index_fc_same_day", "future_fc", "elapsed_days"]].reset_index(drop=True),
                }
            )

        res = evaluate_uncertainty_systems_cv(fold_data, alpha=alpha, seed=seed)
        res["oof_df"].to_csv(c3_reports / f"c3_oof_intervals_{curr_arm}.csv", index=False)
        res["risk_coverage_df"].to_csv(c3_reports / f"c3_risk_coverage_{curr_arm}.csv", index=False)
        res["subgroup_audit_df"].to_csv(c3_reports / f"c3_subgroup_audit_{curr_arm}.csv", index=False)

        all_summaries[curr_arm] = {
            "system_summary": res["system_summary"],
            "risk_coverage": res["risk_coverage_df"].to_dict(orient="records"),
        }

    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "nominal_coverage": 1.0 - alpha,
        "alpha": alpha,
        "dev_pairs_count": int(len(dev_df)),
        "dev_participants_count": int(dev_df["participant_id"].nunique()),
        "test_pairs_count": int(len(test_df)),
        "test_partition_status": "untouched_held_out",
        "arms": all_summaries,
    }

    with open(c3_reports / "c3_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary


if __name__ == "__main__":
    res = run_c3_pipeline(arm="ALL")
    print(json.dumps(res, indent=2))
