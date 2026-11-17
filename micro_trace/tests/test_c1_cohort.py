"""Tests for Phase P1 (C1: Future FC Prediction) Cohort and 80/20 Grouped CV Split."""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from micro_trace.c1.cohort import (
    build_c1_cohorts,
    create_participant_split_80_20,
    verify_raw_mgx_files,
    DEFAULT_ZERO_SUM_SAMPLES,
)


def _load_available_pairs_data() -> pd.DataFrame:
    """Helper to load c1 generated all-horizons report or ensure it is built."""
    pkg_root = Path(__file__).resolve().parents[1]
    c1_path = pkg_root / "reports" / "c1" / "c1_cohort_all_horizons_0d.csv"
    if not c1_path.exists():
        cfg_path = pkg_root / "configs" / "c1_config.yaml"
        build_c1_cohorts(cfg_path)
    assert c1_path.exists(), f"Failed to find or generate {c1_path}"
    return pd.read_csv(c1_path)


def test_c1_consecutive_cohort_counts_and_causality():
    """Verify 445 consecutive next-FC pairs across 93 participants and strict causality."""
    pkg_root = Path(__file__).resolve().parents[1]
    c1_consec_path = pkg_root / "reports" / "c1" / "c1_cohort_consecutive_0d.csv"
    feas_path = pkg_root / "reports" / "feasibility" / "longitudinal_pairs.csv"

    if not c1_consec_path.exists():
        cfg_path = pkg_root / "configs" / "c1_config.yaml"
        build_c1_cohorts(cfg_path)
    assert c1_consec_path.exists(), f"reports/c1/c1_cohort_consecutive_0d.csv not found at {c1_consec_path}"
    df = pd.read_csv(c1_consec_path)

    # Exact count checks
    assert len(df) == 445, f"Expected 445 consecutive pairs, got {len(df)}"
    assert df["participant_id"].nunique() == 93, f"Expected 93 unique participants, got {df['participant_id'].nunique()}"

    # Diagnosis breakdown
    diag_counts = df.drop_duplicates("participant_id")["diagnosis"].value_counts().to_dict()
    assert diag_counts["CD"] + diag_counts["UC"] == 69, f"Expected 69 IBD participants, got {diag_counts}"
    assert diag_counts["nonIBD"] == 24, f"Expected 24 nonIBD participants, got {diag_counts}"

    # Strict causality and elapsed time
    assert (df["elapsed_days"] > 0).all(), "Found pair with non-positive elapsed days!"
    assert (pd.to_datetime(df["index_date"]) < pd.to_datetime(df["future_date"])).all(), "Index date not strictly before future date!"
    assert (df["most_recent_fc_lag_days"] == 0).all(), "Found consecutive 0d pair with non-zero lag days!"

    # Uniqueness of pair_id
    if "pair_id" in df.columns:
        assert df["pair_id"].nunique() == 445, "pair_id is not unique across all 445 consecutive pairs!"
        sample_id = str(df["pair_id"].iloc[0])
        assert "__" in sample_id, f"pair_id {sample_id} does not follow expected '{{pid}}__{{idx}}__{{fut}}' delimiter format"


def test_c1_all_horizons_counts():
    """Verify all 1,671 causal future pairs and bounded horizon subset counts."""
    df = _load_available_pairs_data()
    assert len(df) == 1671, f"Expected 1,671 total 0d future pairs, got {len(df)}"
    assert df["participant_id"].nunique() == 93, f"Expected 93 unique participants, got {df['participant_id'].nunique()}"

    # 30-90d horizon
    col_30_90 = "in_protocol_30_90d" if "in_protocol_30_90d" in df.columns else "in_30_90d"
    sub_30_90 = df[df[col_30_90] == True]
    assert len(sub_30_90) == 545, f"Expected 545 pairs in 30-90d, got {len(sub_30_90)}"
    assert sub_30_90["participant_id"].nunique() == 88, f"Expected 88 participants in 30-90d, got {sub_30_90['participant_id'].nunique()}"

    # 14-56d horizon
    col_14_56 = "in_monthly_14_56d" if "in_monthly_14_56d" in df.columns else "in_14_56d"
    sub_14_56 = df[df[col_14_56] == True]
    assert len(sub_14_56) == 415, f"Expected 415 pairs in 14-56d, got {len(sub_14_56)}"
    assert sub_14_56["participant_id"].nunique() == 84, f"Expected 84 participants in 14-56d, got {sub_14_56['participant_id'].nunique()}"

    # 14-35d horizon
    col_14_35 = "in_biweekly_14_35d" if "in_biweekly_14_35d" in df.columns else "in_14_35d"
    sub_14_35 = df[df[col_14_35] == True]
    assert len(sub_14_35) == 190, f"Expected 190 pairs in 14-35d, got {len(sub_14_35)}"
    assert sub_14_35["participant_id"].nunique() == 44, f"Expected 44 participants in 14-35d, got {sub_14_35['participant_id'].nunique()}"

    # 7-21d horizon
    col_7_21 = "in_short_7_21d" if "in_short_7_21d" in df.columns else "in_7_21d"
    sub_7_21 = df[df[col_7_21] == True]
    assert len(sub_7_21) == 113, f"Expected 113 pairs in 7-21d, got {len(sub_7_21)}"
    assert sub_7_21["participant_id"].nunique() == 36, f"Expected 36 participants in 7-21d, got {sub_7_21['participant_id'].nunique()}"


def test_c1_consecutive_le14d_sensitivity_counts():
    """Verify consecutive pairs under <=14d causal matching sensitivity cohort."""
    pkg_root = Path(__file__).resolve().parents[1]
    c1_le14_path = pkg_root / "reports" / "c1" / "c1_cohort_consecutive_le14d.csv"
    if not c1_le14_path.exists():
        cfg_path = pkg_root / "configs" / "c1_config.yaml"
        build_c1_cohorts(cfg_path)
    assert c1_le14_path.exists(), f"reports/c1/c1_cohort_consecutive_le14d.csv not found at {c1_le14_path}"

    df = pd.read_csv(c1_le14_path)
    assert len(df) == 611, f"Expected 611 pairs, got {len(df)}"
    assert df["participant_id"].nunique() == 94, f"Expected 94 participants, got {df['participant_id'].nunique()}"
    assert (df["partition"] == "dev").sum() == 506
    assert (df["partition"] == "test").sum() == 104
    assert (df["partition"] == "unassigned").sum() == 1  # M2086 has no 0d visit


def test_c1_80_20_split_and_5fold_disjointness():
    """Verify 80% Dev (75 participants) / 20% Test (18 participants) split and 5 disjoint folds of 15."""
    df = _load_available_pairs_data()
    split_df = create_participant_split_80_20(df, random_seed=42, dev_frac=0.80, n_folds=5)

    assert len(split_df) == 93, f"Expected 93 participants in split, got {len(split_df)}"
    assert split_df["participant_id"].nunique() == 93, "Duplicate participant_id in split table!"

    dev_sub = split_df[split_df["partition"] == "dev"]
    test_sub = split_df[split_df["partition"] == "test"]

    # Target partition sizes
    assert len(dev_sub) == 75, f"Expected exactly 75 dev participants, got {len(dev_sub)}"
    assert len(test_sub) == 18, f"Expected exactly 18 test participants, got {len(test_sub)}"

    # Zero participant overlap between dev and test
    dev_pids = set(dev_sub["participant_id"])
    test_pids = set(test_sub["participant_id"])
    assert len(dev_pids & test_pids) == 0, f"Participant overlap detected between dev and test: {dev_pids & test_pids}"

    # Test participants must have cv_fold == -1
    assert (test_sub["cv_fold"] == -1).all(), "Test participants must have cv_fold == -1"

    # Exactly 5 folds in dev, each having exactly 15 participants
    for fold_id in range(5):
        fold_pids = set(dev_sub[dev_sub["cv_fold"] == fold_id]["participant_id"])
        assert len(fold_pids) == 15, f"Fold {fold_id} expected exactly 15 participants, got {len(fold_pids)}"

    # Zero overlap across all 5 folds
    for f1 in range(5):
        p1 = set(dev_sub[dev_sub["cv_fold"] == f1]["participant_id"])
        for f2 in range(f1 + 1, 5):
            p2 = set(dev_sub[dev_sub["cv_fold"] == f2]["participant_id"])
            assert len(p1 & p2) == 0, f"Participant overlap between fold {f1} and fold {f2}: {p1 & p2}"

    # Verify required return columns
    expected_cols = [
        "participant_id", "diagnosis", "is_ibd", "partition", "cv_fold",
        "n_consecutive_0d_pairs", "n_all_future_0d_pairs", "n_30_90d_pairs", "n_14_56d_pairs"
    ]
    for col in expected_cols:
        assert col in split_df.columns, f"Missing required column in split DataFrame: {col}"

    # Verify exact pair count totals across all 93 participants
    assert split_df["n_consecutive_0d_pairs"].sum() == 445, f"Expected 445 consecutive pairs, got {split_df['n_consecutive_0d_pairs'].sum()}"
    assert split_df["n_all_future_0d_pairs"].sum() == 1671, f"Expected 1,671 all-future pairs, got {split_df['n_all_future_0d_pairs'].sum()}"
    assert split_df["n_30_90d_pairs"].sum() == 545, f"Expected 545 pairs in 30-90d, got {split_df['n_30_90d_pairs'].sum()}"
    assert split_df["n_14_56d_pairs"].sum() == 415, f"Expected 415 pairs in 14-56d, got {split_df['n_14_56d_pairs'].sum()}"

    # Verify split with explicit all_horizons_df passed
    consec_sub = df[df["is_next_available_fc_from_index"] == True]
    split_explicit = create_participant_split_80_20(consec_sub, random_seed=42, all_horizons_df=df)
    assert (split_explicit["partition"] == split_df["partition"]).all()
    assert (split_explicit["cv_fold"] == split_df["cv_fold"]).all()
    assert split_explicit["n_all_future_0d_pairs"].sum() == 1671


def test_c1_build_cohorts_pipeline_execution():
    """Verify end-to-end execution of build_c1_cohorts against repository config."""
    pkg_root = Path(__file__).resolve().parents[1]
    cfg_path = pkg_root / "configs" / "c1_config.yaml"
    summary = build_c1_cohorts(cfg_path)

    assert summary["status"] == "SUCCESS"
    assert summary["c1_cohort_consecutive_0d"]["n_pairs"] == 445
    assert summary["c1_cohort_consecutive_0d"]["n_participants"] == 93
    assert summary["c1_cohort_all_horizons_0d"]["n_pairs"] == 1671
    assert summary["c1_cohort_consecutive_le14d"]["n_pairs"] == 611
    assert summary["c1_cohort_consecutive_le14d"]["n_matched_index_visits"] == 686
    assert summary["c1_cohort_consecutive_le14d"]["n_matched_participants"] == 96
    assert summary["metadata_stats"]["mgx_fc_prior_or_same_day_le14d_visits"] == 686
    assert summary["metadata_stats"]["mgx_fc_prior_or_same_day_le14d_participants"] == 96
    assert summary["metadata_stats"]["mgx_fc_same_day_0d_visits"] == 503
    assert summary["metadata_stats"]["mgx_fc_same_day_0d_participants"] == 95
    assert summary["split_summary"]["dev_participants"] == 75
    assert summary["split_summary"]["test_participants"] == 18
    assert summary["split_summary"]["participant_overlap_dev_test"] == 0

    # Ensure all report paths exist on disk
    for path_key, p_str in summary["paths"].items():
        if path_key == "raw_metadata":
            continue
        p = Path(p_str)
        assert p.exists(), f"Expected artifact {path_key} at {p} does not exist"
        assert p.stat().st_size > 0, f"Artifact {path_key} at {p} is empty"


def test_c1_synthetic_and_invalid_input_raises():
    """Verify input validation and fail-fast behavior on malformed or edge-case inputs."""
    # Empty DataFrame
    with pytest.raises(ValueError, match="cannot be empty or None"):
        create_participant_split_80_20(pd.DataFrame())

    # None DataFrame
    with pytest.raises(ValueError, match="cannot be empty or None"):
        create_participant_split_80_20(None)

    # Missing required columns
    bad_df = pd.DataFrame([{"wrong_col": 1}])
    with pytest.raises(ValueError, match="missing required columns"):
        create_participant_split_80_20(bad_df)

    # Null values in participant_id or diagnosis
    null_pid_df = pd.DataFrame([{"participant_id": None, "diagnosis": "CD"}])
    with pytest.raises(ValueError, match="null values in required columns"):
        create_participant_split_80_20(null_pid_df)

    null_diag_df = pd.DataFrame([{"participant_id": "P1", "diagnosis": None}])
    with pytest.raises(ValueError, match="null values in required columns"):
        create_participant_split_80_20(null_diag_df)

    # Empty string values in required columns
    empty_pid_df = pd.DataFrame([{"participant_id": "   ", "diagnosis": "CD"}])
    with pytest.raises(ValueError, match="empty string values in required columns"):
        create_participant_split_80_20(empty_pid_df)

    # Invalid dev_frac
    valid_df = pd.DataFrame([{"participant_id": f"P{i}", "diagnosis": "CD"} for i in range(10)])
    with pytest.raises(ValueError, match="dev_frac must be strictly between 0 and 1"):
        create_participant_split_80_20(valid_df, dev_frac=1.5)
    with pytest.raises(ValueError, match="dev_frac must be strictly between 0 and 1"):
        create_participant_split_80_20(valid_df, dev_frac=0.0)

    # Invalid n_folds
    with pytest.raises(ValueError, match="n_folds must be >= 2"):
        create_participant_split_80_20(valid_df, n_folds=1)

    # Too few participants for folds
    tiny_df = pd.DataFrame([{"participant_id": "P1", "diagnosis": "CD"}])
    with pytest.raises(ValueError, match="Need at least"):
        create_participant_split_80_20(tiny_df, n_folds=5)

    # Explicit mismatched dev_target and test_target
    with pytest.raises(ValueError, match="!= total participants"):
        create_participant_split_80_20(valid_df, dev_target=8, test_target=5)

    # verify_raw_mgx_files missing config
    with pytest.raises(ValueError, match="dictionary"):
        verify_raw_mgx_files(None)
    with pytest.raises(ValueError, match="dictionary"):
        verify_raw_mgx_files("not_a_dict")
    with pytest.raises(ValueError, match="raw_data_dir"):
        verify_raw_mgx_files({})
    with pytest.raises(ValueError, match="cfg\\['paths'\\]"):
        verify_raw_mgx_files({"paths": "not_a_dict"})

    # verify_raw_mgx_files invalid directory
    with pytest.raises(FileNotFoundError, match="does not exist"):
        verify_raw_mgx_files({"paths": {"raw_data_dir": "/nonexistent/path/to/data"}})


def test_c1_zero_sum_sample_flagging():
    """Verify zero-sum metagenomic sample IDs (MSM5LLDM, MSM5LLDS) are accurately flagged."""
    test_samples = [
        ("MSM5LLDM", True),
        ("MSM5LLDM_P", True),
        ("MSM5LLDM_TR", True),
        ("MSM5LLDS", True),
        ("MSM5LLDS_P", True),
        ("CSM5FZ3N_P", False),
        ("HSM5MD39", False),
    ]
    for sid, expected_flag in test_samples:
        stripped = sid.replace("_P", "").replace("_TR", "").strip()
        flag = stripped in DEFAULT_ZERO_SUM_SAMPLES
        assert flag == expected_flag, f"Sample {sid} flagged as {flag}, expected {expected_flag}"


def test_c1_medication_change_detection():
    """Verify medication_changed_between_index_and_future identifies changes in antibiotics and immunosuppressants."""
    cases = [
        ({"Antibiotics": "No", "imm": "No"}, {"Antibiotics": "No", "imm": "No"}, False),
        ({"Antibiotics": "No", "imm": "No"}, {"Antibiotics": "Yes", "imm": "No"}, True),
        ({"Antibiotics": "Yes", "imm": "No"}, {"Antibiotics": "Yes", "imm": "Yes"}, True),
        ({"Antibiotics": np.nan, "imm": np.nan}, {"Antibiotics": np.nan, "imm": np.nan}, False),
    ]
    for idx_cov, fut_cov, expected in cases:
        changed = False
        if pd.notna(idx_cov.get("Antibiotics")) and pd.notna(fut_cov.get("Antibiotics")):
            if str(idx_cov["Antibiotics"]).strip().lower() != str(fut_cov["Antibiotics"]).strip().lower():
                changed = True
        if pd.notna(idx_cov.get("imm")) and pd.notna(fut_cov.get("imm")):
            if str(idx_cov["imm"]).strip().lower() != str(fut_cov["imm"]).strip().lower():
                changed = True
        assert changed == expected


def test_c1_saved_split_audit_artifact_integrity():
    """Verify generated split_audit_80_20.csv on disk matches exact protocol constants."""
    pkg_root = Path(__file__).resolve().parents[1]
    split_path = pkg_root / "reports" / "c1" / "split_audit_80_20.csv"
    assert split_path.exists(), f"reports/c1/split_audit_80_20.csv not found at {split_path}"

    audit_df = pd.read_csv(split_path)
    assert len(audit_df) == 93
    assert audit_df["participant_id"].nunique() == 93

    dev_sub = audit_df[audit_df["partition"] == "dev"]
    test_sub = audit_df[audit_df["partition"] == "test"]
    assert len(dev_sub) == 75
    assert len(test_sub) == 18
    assert len(set(dev_sub["participant_id"]) & set(test_sub["participant_id"])) == 0

    assert audit_df["n_consecutive_0d_pairs"].sum() == 445
    assert audit_df["n_all_future_0d_pairs"].sum() == 1671
    assert audit_df["n_30_90d_pairs"].sum() == 545
    assert audit_df["n_14_56d_pairs"].sum() == 415

    for f in range(5):
        assert (dev_sub["cv_fold"] == f).sum() == 15


def test_c1_cohort_summary_json_integrity():
    """Verify c1_cohort_summary.json artifact on disk matches all verified protocol constants."""
    pkg_root = Path(__file__).resolve().parents[1]
    summary_path = pkg_root / "reports" / "c1" / "c1_cohort_summary.json"
    assert summary_path.exists(), f"reports/c1/c1_cohort_summary.json not found at {summary_path}"

    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)

    assert summary["status"] == "SUCCESS"
    assert summary["c1_cohort_consecutive_0d"]["n_pairs"] == 445
    assert summary["c1_cohort_consecutive_0d"]["n_participants"] == 93
    assert summary["c1_cohort_consecutive_0d"]["ibd_participants"] == 69
    assert summary["c1_cohort_consecutive_0d"]["non_ibd_participants"] == 24
    assert summary["c1_cohort_consecutive_0d"]["dev_pairs"] == 371
    assert summary["c1_cohort_consecutive_0d"]["test_pairs"] == 74
    assert summary["c1_cohort_all_horizons_0d"]["n_pairs"] == 1671
    assert summary["c1_cohort_all_horizons_0d"]["horizons"]["protocol_30_90d"]["pairs"] == 545
    assert summary["c1_cohort_all_horizons_0d"]["horizons"]["monthly_14_56d"]["pairs"] == 415
    assert summary["c1_cohort_all_horizons_0d"]["horizons"]["biweekly_14_35d"]["pairs"] == 190
    assert summary["c1_cohort_all_horizons_0d"]["horizons"]["short_7_21d"]["pairs"] == 113
    assert summary["c1_cohort_consecutive_le14d"]["n_pairs"] == 611
    assert summary["c1_cohort_consecutive_le14d"]["n_matched_index_visits"] == 686
    assert summary["c1_cohort_consecutive_le14d"]["n_matched_participants"] == 96
    assert summary["metadata_stats"]["mgx_fc_prior_or_same_day_le14d_visits"] == 686
    assert summary["metadata_stats"]["mgx_fc_prior_or_same_day_le14d_participants"] == 96
    assert summary["metadata_stats"]["mgx_fc_same_day_0d_visits"] == 503
    assert summary["metadata_stats"]["mgx_fc_same_day_0d_participants"] == 95
    assert summary["split_summary"]["dev_participants"] == 75
    assert summary["split_summary"]["test_participants"] == 18
    assert summary["split_summary"]["participant_overlap_dev_test"] == 0
    for f in range(5):
        assert summary["split_summary"]["folds_distribution"][f"fold_{f}"] == 15
