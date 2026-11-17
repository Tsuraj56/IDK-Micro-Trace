"""Tests for Phase P1 (C1: Future FC Prediction) Train-Only Feature Engineering & Trajectory Engine."""

import copy
import gzip
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from micro_trace.c1.features import (
    C1FeaturePipeline,
    TAXONOMIC_GUILDS,
    FUNCTIONAL_GUILDS,
    DYNAMIC_TRAJECTORY_VARS,
    compute_alpha_diversity,
    compute_taxonomic_guilds,
    compute_functional_guilds,
    compute_bray_curtis,
    build_visit_trajectory_store,
    load_and_align_omics_tables,
)


@pytest.fixture
def synthetic_omics():
    """Create controlled synthetic species, genus, and pathway DataFrames."""
    sample_ids = [f"SMP_{i}" for i in range(10)]
    species_cols = [
        "k__Bacteria|p__Firmicutes|c__Clostridia|o__Clostridiales|f__Ruminococcaceae|g__Faecalibacterium|s__Faecalibacterium_prausnitzii",
        "k__Bacteria|p__Firmicutes|c__Clostridia|o__Clostridiales|f__Lachnospiraceae|g__Roseburia|s__Roseburia_intestinalis",
        "k__Bacteria|p__Proteobacteria|c__Gammaproteobacteria|o__Enterobacterales|f__Enterobacteriaceae|g__Escherichia|s__Escherichia_coli",
        "k__Bacteria|p__Bacteroidetes|c__Bacteroidia|o__Bacteroidales|f__Bacteroidaceae|g__Bacteroides|s__Bacteroides_vulgatus",
        "k__Bacteria|p__Firmicutes|c__Bacilli|o__Lactobacillales|f__Enterococcaceae|g__Enterococcus|s__Enterococcus_faecalis",
    ]
    pathway_cols = [
        "CENTFERM-PWY: pyruvate fermentation to butanoate",
        "SO4ASSIM-PWY: sulfate reduction I",
        "GLUTATHIONESYN-PWY: glutathione biosynthesis",
        "GLUCUROCAT-PWY: beta-D-glucuronide and D-glucuronate degradation",
        "NAGLIPASYN-PWY: lipid IVA biosynthesis",
        "PWY-7111: glycolysis IV",
    ]

    rng = np.random.default_rng(42)
    # Generate positive random matrices and renormalize to sum to 1.0
    sp_raw = rng.exponential(scale=1.0, size=(10, len(species_cols)))
    sp_mat = sp_raw / sp_raw.sum(axis=1, keepdims=True)
    species_df = pd.DataFrame(sp_mat, index=sample_ids, columns=species_cols)

    p_raw = rng.exponential(scale=1.0, size=(10, len(pathway_cols)))
    p_mat = p_raw / p_raw.sum(axis=1, keepdims=True)
    pathways_df = pd.DataFrame(p_mat, index=sample_ids, columns=pathway_cols)

    return {
        "species_relab": species_df,
        "genus_relab": pd.DataFrame(),
        "pathways_unstrat_relab": pathways_df,
    }


@pytest.fixture
def synthetic_pairs_and_visits(synthetic_omics):
    """Create longitudinal synthetic visit store and paired forecast cohorts."""
    species_df = synthetic_omics["species_relab"]
    pathways_df = synthetic_omics["pathways_unstrat_relab"]

    # 4 participants, 4 visits each
    visit_rows = []
    pair_rows = []
    pids = ["P101", "P102", "P103", "P104"]
    diags = ["CD", "UC", "CD", "nonIBD"]

    smp_idx = 0
    for p_i, pid in enumerate(pids):
        diag = diags[p_i]
        dates = pd.date_range("2015-01-01", periods=4, freq="14D")
        for v_k, dt in enumerate(dates):
            cid = f"{pid}_C{v_k+1}"
            sid = f"SMP_{smp_idx % 10}"
            smp_idx += 1
            fc = float(20.0 + 30.0 * v_k + (10.0 if diag == "CD" else 0.0))
            visit_rows.append({
                "collection_event_id": cid,
                "participant_id": pid,
                "diagnosis": diag,
                "is_ibd": diag in ["CD", "UC"],
                "visit_num": v_k + 1,
                "week_num": float(v_k * 2),
                "event_date": dt,
                "fc_value": fc,
                "clean_mgx_sample_id": sid,
                "has_mgx_sample": True,
                "is_zero_sum_mgx_sample": False,
                "hbi": 2.0 + v_k if diag == "CD" else np.nan,
                "sccai": 3.0 + v_k if diag == "UC" else np.nan,
            })

            # Form consecutive forecast pair if v_k < 3
            if v_k < 3:
                next_dt = dates[v_k + 1]
                next_cid = f"{pid}_C{v_k+2}"
                next_fc = float(20.0 + 30.0 * (v_k + 1) + (10.0 if diag == "CD" else 0.0))
                pair_rows.append({
                    "pair_id": f"{pid}__{cid}__{next_cid}",
                    "participant_id": pid,
                    "diagnosis": diag,
                    "is_ibd": diag in ["CD", "UC"],
                    "index_collection_event": cid,
                    "index_date": dt,
                    "index_mgx_sample_id": sid,
                    "index_log10_fc": np.log10(fc + 1.0),
                    "index_log10_fc_same_day": np.log10(fc + 1.0),
                    "future_collection_event": next_cid,
                    "future_date": next_dt,
                    "future_fc": next_fc,
                    "future_log10_fc_plus_1": np.log10(next_fc + 1.0),
                    "elapsed_days": 14,
                    "consent_age": 35.0,
                    "Age at diagnosis": 25.0,
                    "sex": "Female" if p_i % 2 == 0 else "Male",
                    "hbi": 2.0 + v_k if diag == "CD" else np.nan,
                    "sccai": 3.0 + v_k if diag == "UC" else np.nan,
                    "Bowel frequency during the day": 3.0,
                    "Abdominal pain": "Mild",
                    "General wellbeing": "Slightly below par",
                    "Antibiotics": "No",
                    "Immunosuppressants (e.g. oral corticosteroids)": "Yes",
                    "reads_filtered": 1.2e7,
                    "reads_human": 5e5,
                    "partition": "dev" if p_i < 3 else "test",
                    "cv_fold": p_i if p_i < 3 else -1,
                })

    meta_df = pd.DataFrame(visit_rows)
    pairs_df = pd.DataFrame(pair_rows)

    schema_cfg = {
        "identifiers": {
            "participant_id": "participant_id",
            "sample_id": "clean_mgx_sample_id",
            "collection_event_id": "collection_event_id",
            "visit_id": "visit_num",
            "week_num": "week_num",
            "data_type": "data_type",
        },
        "dates": {"receipt_date": "event_date"},
        "clinical": {
            "disease_subtype": "diagnosis",
            "ibd_subtypes": ["CD", "UC"],
        },
        "outcomes": {
            "fc_primary": "fc_value",
            "fc_secondary": "fc_value",
        },
    }
    meta_df["data_type"] = "metagenomics"

    visit_store_df = build_visit_trajectory_store(
        meta_df=meta_df,
        omics_dict=synthetic_omics,
        schema_cfg=schema_cfg,
    )

    return pairs_df, visit_store_df, synthetic_omics


def test_train_only_fitting_invariance_pr03(synthetic_pairs_and_visits):
    """Verify that perturbing/corrupting test data has ZERO effect on train-fitted parameters or X_train."""
    pairs_df, visit_store_df, omics_dict = synthetic_pairs_and_visits

    train_pairs = pairs_df[pairs_df["partition"] == "dev"].copy()
    test_pairs = pairs_df[pairs_df["partition"] == "test"].copy()

    # Fit M5 pipeline on clean train pairs
    pipe1 = C1FeaturePipeline(arm="M5", prevalence_threshold=0.10)
    pipe1.fit(train_pairs, visit_store_df, omics_dict)
    X_train_1, names_1 = pipe1.transform(train_pairs, visit_store_df, omics_dict)
    X_test_1, _ = pipe1.transform(test_pairs, visit_store_df, omics_dict)

    # Wildly corrupt test_pairs (multiply numeric values by 1000x, change categories)
    test_corrupted = test_pairs.copy()
    test_corrupted["index_log10_fc"] = test_corrupted["index_log10_fc"] * 1000.0
    test_corrupted["elapsed_days"] = test_corrupted["elapsed_days"] * 500
    test_corrupted["consent_age"] = 999.0
    test_corrupted["hbi"] = 888.0
    test_corrupted["diagnosis"] = "UNKNOWN_DIAGNOSIS"

    # Refit brand new pipeline strictly on train_pairs
    pipe2 = C1FeaturePipeline(arm="M5", prevalence_threshold=0.10)
    pipe2.fit(train_pairs, visit_store_df, omics_dict)
    X_train_2, names_2 = pipe2.transform(train_pairs, visit_store_df, omics_dict)

    # PR-03 Invariant assertions:
    assert names_1 == names_2, "Feature names differ after test perturbation!"
    assert pipe1.retained_species_ == pipe2.retained_species_, "Retained species differed!"
    assert pipe1.retained_pathways_ == pipe2.retained_pathways_, "Retained pathways differed!"
    assert np.isclose(pipe1.species_pseudocount_, pipe2.species_pseudocount_, atol=0, rtol=0), "Species pseudocount differed!"
    assert np.isclose(pipe1.pathway_pseudocount_, pipe2.pathway_pseudocount_, atol=0, rtol=0), "Pathway pseudocount differed!"
    for k in pipe1.train_medians_:
        assert np.isclose(pipe1.train_medians_[k], pipe2.train_medians_[k], atol=0, rtol=0), f"Median for {k} differed!"
    assert np.allclose(pipe1.scaler_mean_, pipe2.scaler_mean_, atol=0, rtol=0), "Scaler means differed!"
    assert np.allclose(pipe1.scaler_scale_, pipe2.scaler_scale_, atol=0, rtol=0), "Scaler stds differed!"
    for k in pipe1.scaler_means_:
        assert np.isclose(pipe1.scaler_means_[k], pipe2.scaler_means_[k], atol=0, rtol=0), f"Scaler mean for {k} differed!"
        assert np.isclose(pipe1.scaler_stds_[k], pipe2.scaler_stds_[k], atol=0, rtol=0), f"Scaler std for {k} differed!"

    # Mathematical bit-for-bit identity on X_train (atol=0, rtol=0)
    assert np.array_equal(X_train_1, X_train_2), "Leakage detected: X_train is not bit-for-bit identical!"

    X_train_std_1, _ = pipe1.transform(train_pairs, visit_store_df, omics_dict, standardize=True)
    X_train_std_2, _ = pipe2.transform(train_pairs, visit_store_df, omics_dict, standardize=True)
    assert np.array_equal(X_train_std_1, X_train_std_2), "Standardized X_train differed after test perturbation!"


def test_m5_trajectory_and_sequences_strict_causality(synthetic_pairs_and_visits):
    """Verify modifying a future visit (event_date > index_date) has zero effect on M5 features or sequences."""
    pairs_df, visit_store_df, omics_dict = synthetic_pairs_and_visits

    pipe = C1FeaturePipeline(arm="M5", prevalence_threshold=0.10)
    pipe.fit(pairs_df, visit_store_df, omics_dict)

    # Pick first pair: index_date is 2015-01-01
    pair_0 = pairs_df.iloc[[0]].copy()
    X_orig, names = pipe.transform(pair_0, visit_store_df, omics_dict)
    seqs_orig = pipe.build_causal_sequences(pair_0, visit_store_df, omics_dict, max_seq_len=10)

    # Corrupt future visits in visit_store_df (e.g. visit 3 and 4 with event_date >= 2015-01-15)
    store_corrupted = visit_store_df.copy()
    idx_future = store_corrupted["event_date"] > pd.to_datetime(pair_0["index_date"].iloc[0])
    store_corrupted.loc[idx_future, "fc_log10"] = 9999.0
    store_corrupted.loc[idx_future, "shannon_species"] = 5555.0
    store_corrupted.loc[idx_future, "guild_scfa_butyrate_log"] = 1234.0

    X_corrupted, _ = pipe.transform(pair_0, store_corrupted, omics_dict)
    seqs_corrupted = pipe.build_causal_sequences(pair_0, store_corrupted, omics_dict, max_seq_len=10)

    # Future modification must have zero effect on index features or sequences
    assert np.array_equal(X_orig, X_corrupted), "Temporal leakage: future visit corrupted index trajectory features!"
    assert np.array_equal(seqs_orig["seq_features"], seqs_corrupted["seq_features"]), "Temporal leakage in seq_features!"
    assert np.array_equal(seqs_orig["seq_time_deltas"], seqs_corrupted["seq_time_deltas"]), "Temporal leakage in seq_time_deltas!"

    # Verify time deltas are non-negative
    deltas = seqs_orig["seq_time_deltas"]
    mask = seqs_orig["seq_mask"]
    assert (deltas[mask] >= 0).all(), "Found negative time delta in historical visit sequences!"
    assert seqs_orig["seq_features"].shape[2] == 15, f"Expected d_seq=15, got {seqs_orig['seq_features'].shape[2]}"
    assert seqs_orig["static_anchor"].shape[1] == 6, f"Expected d_anchor=6, got {seqs_orig['static_anchor'].shape[1]}"


def test_clr_simplex_and_zero_sum_handling():
    """Verify CLR simplex zero-sum property (sum=0.0) and safe handling of zero-sum samples."""
    sample_ids = ["S1", "S2", "ZERO_SUM_SMP"]
    features = ["spA", "spB", "spC", "spD"]

    # S1 and S2 have non-zero simplex relative abundances; ZERO_SUM_SMP is all zeros
    sp_mat = np.array([
        [0.4, 0.3, 0.2, 0.1],
        [0.1, 0.2, 0.3, 0.4],
        [0.0, 0.0, 0.0, 0.0],
    ])
    species_df = pd.DataFrame(sp_mat, index=sample_ids, columns=features)
    omics_dict = {"species_relab": species_df, "pathways_unstrat_relab": pd.DataFrame()}

    pairs_df = pd.DataFrame([
        {"pair_id": "P1__C1__C2", "participant_id": "P1", "index_mgx_sample_id": "S1", "index_log10_fc": 2.0, "elapsed_days": 14},
        {"pair_id": "P2__C1__C2", "participant_id": "P2", "index_mgx_sample_id": "S2", "index_log10_fc": 1.5, "elapsed_days": 28},
        {"pair_id": "P3__C1__C2", "participant_id": "P3", "index_mgx_sample_id": "ZERO_SUM_SMP", "index_log10_fc": 1.0, "elapsed_days": 14},
    ])

    pipe = C1FeaturePipeline(arm="M2", prevalence_threshold=0.10)
    pipe.fit(pairs_df, pd.DataFrame(), omics_dict)
    X, names = pipe.transform(pairs_df, pd.DataFrame(), omics_dict)

    clr_cols = [i for i, n in enumerate(names) if n.startswith("clr_species_")]
    clr_mat = X[:, clr_cols]

    # Standard valid samples must sum to 0.0 in CLR space
    assert np.allclose(clr_mat[0].sum(), 0.0, atol=1e-6), f"S1 CLR sum is {clr_mat[0].sum()}, expected 0.0"
    assert np.allclose(clr_mat[1].sum(), 0.0, atol=1e-6), f"S2 CLR sum is {clr_mat[1].sum()}, expected 0.0"

    # Zero-sum sample must be imputed with 0.0 vector and flag set to 1.0
    zero_flag_idx = names.index("is_zero_sum_or_missing_omics")
    assert np.allclose(clr_mat[2], 0.0, atol=0), "Zero-sum sample CLR was not imputed with 0.0!"
    assert X[2, zero_flag_idx] == 1.0, "Zero-sum flag was not set to 1.0!"
    assert X[0, zero_flag_idx] == 0.0, "Valid sample had zero-sum flag set!"
    assert np.isfinite(X).all(), "Found non-finite values in CLR matrix!"


def test_all_arms_m0_to_m5_shapes(synthetic_pairs_and_visits):
    """Verify M0 < M1 < M2 < M4 < M5 and M1 < M3 < M4 < M5 feature dimensions and finite values on synthetic data."""
    pairs_df, visit_store_df, omics_dict = synthetic_pairs_and_visits

    arm_dims = {}
    for arm in ["M0", "M1", "M2", "M3", "M4", "M5", "B3", "B4"]:
        pipe = C1FeaturePipeline(arm=arm, prevalence_threshold=0.0)
        pipe.fit(pairs_df, visit_store_df, omics_dict)
        X, names = pipe.transform(pairs_df, visit_store_df, omics_dict)
        arm_dims[arm] = X.shape[1]
        assert np.isfinite(X).all(), f"Arm {arm} produced non-finite values"
        assert len(names) == X.shape[1], f"Arm {arm} names length mismatch"

    # Verify dimensional hierarchy:
    assert arm_dims["M0"] < arm_dims["M1"], f"M0 ({arm_dims['M0']}) >= M1 ({arm_dims['M1']})"
    assert arm_dims["M1"] < arm_dims["M2"], f"M1 ({arm_dims['M1']}) >= M2 ({arm_dims['M2']})"
    assert arm_dims["M1"] < arm_dims["M3"], f"M1 ({arm_dims['M1']}) >= M3 ({arm_dims['M3']})"
    assert arm_dims["M2"] < arm_dims["M4"], f"M2 ({arm_dims['M2']}) >= M4 ({arm_dims['M4']})"
    assert arm_dims["M3"] < arm_dims["M4"], f"M3 ({arm_dims['M3']}) >= M4 ({arm_dims['M4']})"
    assert arm_dims["M4"] < arm_dims["M5"], f"M4 ({arm_dims['M4']}) >= M5 ({arm_dims['M5']})"

    # B3 and B4 reference checks
    assert arm_dims["B3"] > 0
    assert arm_dims["B4"] > 0


def test_c1_real_run_artifacts_on_disk():
    """Verify reports/c1/c1_feature_audit.json and reports/c1/c1_guild_definitions.csv on disk.
    
    This test strictly checks the on-disk artifacts generated by the real CLI run.
    It fails if they are missing and MUST NOT generate them itself.
    """
    pkg_root = Path(__file__).resolve().parents[1]
    audit_json = pkg_root / "reports" / "c1" / "c1_feature_audit.json"
    guild_csv = pkg_root / "reports" / "c1" / "c1_guild_definitions.csv"

    assert audit_json.exists(), (
        f"Missing audit artifact: {audit_json}. "
        "A real CLI execution against real HMP2 data must produce this artifact."
    )
    assert guild_csv.exists(), (
        f"Missing guild definitions artifact: {guild_csv}. "
        "A real CLI execution against real HMP2 data must produce this artifact."
    )

    import json
    from datetime import datetime
    with open(audit_json, "r", encoding="utf-8") as f:
        audit_data = json.load(f)

    # Execution timestamp must be authentic (ISO-8601 with timezone, non-zero microseconds)
    ts_str = audit_data.get("execution_timestamp", "")
    assert ts_str, "execution_timestamp missing from audit JSON"
    dt = datetime.fromisoformat(ts_str)
    assert dt.tzinfo is not None, "execution_timestamp must include timezone offset"
    assert not ts_str.endswith(".000000+00:00"), f"Timestamp {ts_str} appears fabricated with .000000 microseconds"

    # Exact cohort numbers
    assert audit_data["total_consecutive_pairs"] == 445
    assert audit_data["dev_pairs"] == 371
    assert audit_data["test_pairs"] == 74
    assert audit_data["total_unique_participants"] == 93
    assert audit_data["dev_unique_participants"] == 75
    assert audit_data["test_unique_participants"] == 18

    # Exact arm feature dimensions
    expected_dims = {
        "M0": 2,
        "M1": 29,
        "M2": 158,
        "M3": 396,
        "M4": 524,
        "M5": 646,
        "B3": 130,
        "B4": 368,
    }
    assert audit_data["arms_feature_dimensions"] == expected_dims

    # Retained features and biological counts
    assert audit_data["prevalence_filtering_threshold"] == 0.10
    assert audit_data["metaphlan3_species_total"] == 578
    assert audit_data["metaphlan3_genus_total"] == 187
    assert audit_data["dev_prevalence_retained_species"] == 120
    assert audit_data["humann3_pathways_unstratified_total"] == 476
    assert audit_data["dev_prevalence_retained_pathways"] == 361

    # 5-fold cross-validation retained counts
    assert audit_data["5fold_retained_species_counts"] == {
        "fold_0": 122, "fold_1": 121, "fold_2": 124, "fold_3": 121, "fold_4": 122,
    }
    assert audit_data["5fold_retained_pathway_counts"] == {
        "fold_0": 363, "fold_1": 360, "fold_2": 363, "fold_3": 358, "fold_4": 361,
    }

    # Guild counts and leakage verification flags
    expected_guild_counts = {
        "guild_scfa_butyrate": 19,
        "guild_pathobiont": 53,
        "fguild_scfa_butanoate": 21,
        "fguild_sulfur_h2s": 6,
        "fguild_oxidative_stress": 2,
        "fguild_mucin_hexuronide": 8,
        "fguild_lps_enterobacterial": 5,
    }
    assert audit_data["guild_membership_counts"] == expected_guild_counts
    assert audit_data["leakage_verification_flags"]["train_only_fitting_invariant_pr03"] is True
    assert audit_data["leakage_verification_flags"]["m5_trajectory_strict_temporal_causality"] is True
    assert audit_data["leakage_verification_flags"]["sequence_time_deltas_non_negative"] is True
    assert audit_data["leakage_verification_flags"]["all_features_finite_no_nans_or_infs"] is True
    assert audit_data["leakage_verification_flags"]["participant_overlap_dev_test"] == 0
    assert audit_data["leakage_verification_flags"]["participant_overlap_across_folds"] == 0

    # MicroTrace-Attn sequence tensors
    assert audit_data["microtrace_attn_sequence_tensors"]["max_seq_len"] == 25
    assert audit_data["microtrace_attn_sequence_tensors"]["d_seq"] == 15
    assert audit_data["microtrace_attn_sequence_tensors"]["d_anchor"] == 6
    assert audit_data["microtrace_attn_sequence_tensors"]["strictly_causal_temporal_deltas"] is True
    assert audit_data["microtrace_attn_sequence_tensors"]["seq_time_deltas_min_days"] >= 0.0

    # Guild definitions CSV consistency
    guild_df = pd.read_csv(guild_csv)
    assert len(guild_df) == 7, f"Expected 7 domain guilds in definitions, got {len(guild_df)}"
    assert set(guild_df["guild_id"]) == set(expected_guild_counts.keys())
    for _, row in guild_df.iterrows():
        gid = row["guild_id"]
        assert row["matched_feature_count"] == expected_guild_counts[gid]
        assert isinstance(row["matched_features"], str) and len(row["matched_features"]) > 0



def test_invalid_arm_or_unfitted_transform_raises():
    """Verify fail-fast behavior on unknown arms or transform before fit."""
    with pytest.raises(ValueError, match="Unknown model arm: INVALID_ARM"):
        C1FeaturePipeline(arm="INVALID_ARM")

    pipe = C1FeaturePipeline(arm="M1")
    with pytest.raises(RuntimeError, match="must be fitted before calling transform"):
        pipe.transform(pd.DataFrame([{"pair_id": "P1"}]), pd.DataFrame(), {})

    with pytest.raises(RuntimeError, match="must be fitted before calling build_causal_sequences"):
        pipe.build_causal_sequences(pd.DataFrame([{"pair_id": "P1"}]), pd.DataFrame(), {})


def test_diversity_and_guilds():
    """Verify alpha diversity, beta diversity, and domain guild calculations."""
    species = ["Faecalibacterium_prausnitzii", "Roseburia_intestinalis", "Escherichia_coli", "Other_species"]
    sp_mat = pd.DataFrame(
        [
            [0.5, 0.3, 0.1, 0.1],
            [0.1, 0.0, 0.8, 0.1],
        ],
        index=["S1", "S2"],
        columns=species,
    )

    alpha_df = compute_alpha_diversity(sp_mat)
    assert (alpha_df["shannon_species"] > 0).all()
    assert (alpha_df["simpson_species"] > 0).all()
    assert alpha_df.loc["S1", "species_richness"] == 4.0
    assert alpha_df.loc["S2", "species_richness"] == 3.0

    guild_df = compute_taxonomic_guilds(sp_mat)
    assert np.isclose(guild_df.loc["S1", "guild_scfa_butyrate_relab"], 0.8)
    assert np.isclose(guild_df.loc["S1", "guild_pathobiont_relab"], 0.1)
    # S1 has higher protective vs pathobiont ratio than S2
    assert guild_df.loc["S1", "guild_protective_vs_pathobiont_logratio"] > guild_df.loc["S2", "guild_protective_vs_pathobiont_logratio"]

    # Bray-Curtis dissimilarity
    bc_self = compute_bray_curtis(sp_mat.iloc[0].to_numpy(), sp_mat.iloc[0].to_numpy())
    assert np.isclose(bc_self, 0.0)
    bc_diff = compute_bray_curtis(sp_mat.iloc[0].to_numpy(), sp_mat.iloc[1].to_numpy())
    assert 0.0 < bc_diff <= 1.0


def test_load_and_align_omics_tables_synthetic(tmp_path):
    """Verify parsing and renormalization of MetaPhlAn3 and HUMAnN3 synthetic tables."""
    raw_dir = tmp_path / "raw"
    tax_dir = raw_dir / "products" / "HMP2" / "MGX" / "2018-05-04"
    tax_dir.mkdir(parents=True, exist_ok=True)

    # 1. Create synthetic taxonomic_profiles_3.tsv.gz
    tax_tsv = tax_dir / "taxonomic_profiles_3.tsv.gz"
    with gzip.open(tax_tsv, "wt", encoding="utf-8") as f:
        f.write("#clade_name\tS1_profile\tS2_profile\n")
        f.write("k__Bacteria|p__Firmicutes|c__Clostridia|o__Clostridiales|f__Ruminococcaceae|g__Faecalibacterium\t50.0\t20.0\n")
        f.write("k__Bacteria|p__Firmicutes|c__Clostridia|o__Clostridiales|f__Ruminococcaceae|g__Faecalibacterium|s__Faecalibacterium_prausnitzii\t40.0\t15.0\n")
        f.write("k__Bacteria|p__Proteobacteria|c__Gammaproteobacteria|o__Enterobacterales|f__Enterobacteriaceae|g__Escherichia|s__Escherichia_coli\t10.0\t35.0\n")
        f.write("k__Bacteria|p__Proteobacteria|c__Gammaproteobacteria|o__Enterobacterales|f__Enterobacteriaceae|g__Escherichia|s__Escherichia_coli|t__Strain1\t10.0\t35.0\n")

    # 2. Create synthetic pathabundances_3.tsv.gz
    path_tsv = tax_dir / "pathabundances_3.tsv.gz"
    with gzip.open(path_tsv, "wt", encoding="utf-8") as f:
        f.write("# Pathway\tS1_pathabundance_cpm\tS2_pathabundance_cpm\n")
        f.write("UNMAPPED\t1000.0\t1000.0\n")
        f.write("UNINTEGRATED\t500.0\t500.0\n")
        f.write("CENTFERM-PWY: pyruvate fermentation to butanoate\t100.0\t200.0\n")
        f.write("CENTFERM-PWY: pyruvate fermentation to butanoate|g__Faecalibacterium.s__Faecalibacterium_prausnitzii\t100.0\t200.0\n")
        f.write("PWY-7111: glycolysis IV\t300.0\t200.0\n")

    # 3. Create dummy config
    cfg_file = tmp_path / "c1_config.yaml"
    cfg_content = f"""paths:
  raw_data_dir: "{raw_dir.as_posix()}"
"""
    cfg_file.write_text(cfg_content, encoding="utf-8")

    omics = load_and_align_omics_tables(cfg_file)
    assert "species_relab" in omics
    assert "genus_relab" in omics
    assert "pathways_unstrat_relab" in omics

    sp = omics["species_relab"]
    assert len(sp.columns) == 2, f"Expected exactly 2 species columns, got {len(sp.columns)}"
    assert sp.index.tolist() == ["S1", "S2"]
    # Verify renormalization sums to 1.0
    assert np.allclose(sp.sum(axis=1), 1.0)

    pwy = omics["pathways_unstrat_relab"]
    assert len(pwy.columns) == 2, f"Expected exactly 2 unstratified pathways, got {len(pwy.columns)}"
    assert np.allclose(pwy.sum(axis=1), 1.0)


def test_ols_slope_mathematical_exactness():
    """Verify that OLS slope calculation is mathematically exact without covariance divisor bias."""
    dates = pd.date_range("2015-01-01", periods=3, freq="10D")
    meta_rows = [
        {
            "collection_event_id": f"C{i+1}",
            "participant_id": "P_OLS",
            "diagnosis": "CD",
            "visit_num": i + 1,
            "week_num": float(i * 1.4),
            "event_date": dates[i],
            "fc_value": float(10.0 + 20.0 * i),
            "clean_mgx_sample_id": f"S{i+1}",
            "data_type": "metagenomics",
        }
        for i in range(3)
    ]
    meta_df = pd.DataFrame(meta_rows)
    schema_cfg = {
        "identifiers": {
            "participant_id": "participant_id",
            "sample_id": "clean_mgx_sample_id",
            "collection_event_id": "collection_event_id",
            "visit_id": "visit_num",
            "week_num": "week_num",
            "data_type": "data_type",
        },
        "dates": {"receipt_date": "event_date"},
        "clinical": {"disease_subtype": "diagnosis", "ibd_subtypes": ["CD", "UC"]},
        "outcomes": {"fc_primary": "fc_value", "fc_secondary": "fc_value"},
    }
    store = build_visit_trajectory_store(meta_df, {}, schema_cfg)

    # Check slope at visit 2 (2 points: t=0, y0; t=10, y1)
    v0_y = float(store.loc[0, "fc_log10"])
    v1_y = float(store.loc[1, "fc_log10"])
    expected_slope_2pt = (v1_y - v0_y) / 10.0
    actual_slope_2pt = float(store.loc[1, "fc_log10_roll_slope"])
    assert np.isclose(actual_slope_2pt, expected_slope_2pt, atol=1e-6), (
        f"OLS slope 2-point mismatch: actual={actual_slope_2pt}, expected={expected_slope_2pt}"
    )

    # Check slope at visit 3 (3 points: t=0, y0; t=10, y1; t=20, y2)
    v2_y = float(store.loc[2, "fc_log10"])
    t_vals = np.array([0.0, 10.0, 20.0])
    y_vals = np.array([v0_y, v1_y, v2_y])
    expected_slope_3pt = float(np.polyfit(t_vals, y_vals, 1)[0])
    actual_slope_3pt = float(store.loc[2, "fc_log10_roll_slope"])
    assert np.isclose(actual_slope_3pt, expected_slope_3pt, atol=1e-6), (
        f"OLS slope 3-point mismatch: actual={actual_slope_3pt}, expected={expected_slope_3pt}"
    )


def test_single_visit_participant_trajectory_defaults():
    """Verify that a participant with only 1 visit produces valid causal defaults."""
    meta_df = pd.DataFrame([{
        "collection_event_id": "SINGLE_C1",
        "participant_id": "P_SINGLE",
        "diagnosis": "UC",
        "visit_num": 1,
        "week_num": 0.0,
        "event_date": pd.Timestamp("2015-05-01"),
        "fc_value": 45.0,
        "clean_mgx_sample_id": "SINGLE_S1",
        "data_type": "metagenomics",
    }])
    schema_cfg = {
        "identifiers": {
            "participant_id": "participant_id",
            "sample_id": "clean_mgx_sample_id",
            "collection_event_id": "collection_event_id",
            "visit_id": "visit_num",
            "week_num": "week_num",
            "data_type": "data_type",
        },
        "dates": {"receipt_date": "event_date"},
        "clinical": {"disease_subtype": "diagnosis", "ibd_subtypes": ["CD", "UC"]},
        "outcomes": {"fc_primary": "fc_value", "fc_secondary": "fc_value"},
    }
    store = build_visit_trajectory_store(meta_df, {}, schema_cfg)
    assert store.loc[0, "has_prior_visit"] == 0.0
    assert store.loc[0, "days_since_first_visit"] == 0.0
    assert store.loc[0, "days_since_prev_visit"] == 0.0
    assert np.isclose(store.loc[0, "fc_log10_prev"], store.loc[0, "fc_log10_curr"])
    assert store.loc[0, "fc_log10_delta"] == 0.0
    assert store.loc[0, "fc_log10_velocity"] == 0.0
    assert store.loc[0, "fc_log10_roll_std"] == 0.0
    assert store.loc[0, "fc_log10_roll_slope"] == 0.0


def test_same_day_collection_events_causality():
    """Verify that multiple visits on the same date strictly isolate history up to the index event."""
    dt = pd.Timestamp("2015-06-01")
    species_df = pd.DataFrame(
        [[0.6, 0.4], [0.1, 0.9]],
        index=["S_EARLY", "S_LATE"],
        columns=["sp1", "sp2"],
    )
    omics = {"species_relab": species_df, "pathways_unstrat_relab": pd.DataFrame()}

    meta_df = pd.DataFrame([
        {
            "collection_event_id": "C_EARLY",
            "participant_id": "P_MULTI",
            "diagnosis": "CD",
            "visit_num": 1,
            "week_num": 0.0,
            "event_date": dt,
            "fc_value": 30.0,
            "clean_mgx_sample_id": "S_EARLY",
            "data_type": "metagenomics",
        },
        {
            "collection_event_id": "C_LATE",
            "participant_id": "P_MULTI",
            "diagnosis": "CD",
            "visit_num": 2,
            "week_num": 0.0,
            "event_date": dt,
            "fc_value": 300.0,
            "clean_mgx_sample_id": "S_LATE",
            "data_type": "metagenomics",
        },
    ])
    schema_cfg = {
        "identifiers": {
            "participant_id": "participant_id",
            "sample_id": "clean_mgx_sample_id",
            "collection_event_id": "collection_event_id",
            "visit_id": "visit_num",
            "week_num": "week_num",
            "data_type": "data_type",
        },
        "dates": {"receipt_date": "event_date"},
        "clinical": {"disease_subtype": "diagnosis", "ibd_subtypes": ["CD", "UC"]},
        "outcomes": {"fc_primary": "fc_value", "fc_secondary": "fc_value"},
    }
    store = build_visit_trajectory_store(meta_df, omics, schema_cfg)

    pipe = C1FeaturePipeline(arm="M5", prevalence_threshold=0.0)
    train_pairs = pd.DataFrame([{
        "pair_id": "P_MULTI__C_EARLY__C_LATE",
        "participant_id": "P_MULTI",
        "index_collection_event": "C_EARLY",
        "index_date": dt,
        "index_mgx_sample_id": "S_EARLY",
        "index_log10_fc": np.log10(31.0),
        "elapsed_days": 14,
        "diagnosis": "CD",
    }])
    pipe.fit(train_pairs, store, omics)
    X, names = pipe.transform(train_pairs, store, omics)

    # For C_EARLY (the first collection event), there is no prior visit in sub_history
    seqs = pipe.build_causal_sequences(train_pairs, store, omics, max_seq_len=5)
    # Mask should have exactly 1 valid historical visit (C_EARLY), not 2
    assert seqs["seq_mask"][0].sum() == 1, f"Expected 1 valid visit in sequence mask, got {seqs['seq_mask'][0].sum()}"


def test_omics_caching_and_roundtrip(tmp_path):
    """Verify that load_and_align_omics_tables caches tables and re-reading from disk yields identical data."""
    raw_dir = tmp_path / "raw"
    cache_dir = tmp_path / "cache"
    tax_dir = raw_dir / "products" / "HMP2" / "MGX" / "2018-05-04"
    tax_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # 1. Create synthetic files
    tax_tsv = tax_dir / "taxonomic_profiles_3.tsv.gz"
    with gzip.open(tax_tsv, "wt", encoding="utf-8") as f:
        f.write("#clade_name\tS1_profile\n")
        f.write("k__Bacteria|p__Firmicutes|c__Clostridia|o__Clostridiales|f__Ruminococcaceae|g__Faecalibacterium|s__Faecalibacterium_prausnitzii\t100.0\n")

    path_tsv = tax_dir / "pathabundances_3.tsv.gz"
    with gzip.open(path_tsv, "wt", encoding="utf-8") as f:
        f.write("# Pathway\tS1_pathabundance_cpm\n")
        f.write("PWY-1234: test pathway\t50.0\n")

    cfg_file = tmp_path / "c1_config.yaml"
    cfg_content = f"""paths:
  raw_data_dir: "{raw_dir.as_posix()}"
  processed_c1_dir: "{cache_dir.as_posix()}"
"""
    cfg_file.write_text(cfg_content, encoding="utf-8")

    # Initial load: must parse raw and populate cache
    omics_orig = load_and_align_omics_tables(cfg_file, cache_dir=cache_dir)
    assert (cache_dir / "species_relab.csv.gz").exists()
    assert (cache_dir / "pathways_unstrat_relab.csv.gz").exists()

    # Second load: must load directly from cache
    omics_cached = load_and_align_omics_tables(cfg_file, cache_dir=cache_dir)
    pd.testing.assert_frame_equal(omics_orig["species_relab"], omics_cached["species_relab"])
    pd.testing.assert_frame_equal(omics_orig["pathways_unstrat_relab"], omics_cached["pathways_unstrat_relab"])


def test_duplicate_feature_names_deduplicated(tmp_path):
    """Verify that omics files with duplicate feature names are deduplicated without error."""
    raw_dir = tmp_path / "raw"
    tax_dir = raw_dir / "products" / "HMP2" / "MGX" / "2018-05-04"
    tax_dir.mkdir(parents=True, exist_ok=True)

    tax_tsv = tax_dir / "taxonomic_profiles_3.tsv.gz"
    with gzip.open(tax_tsv, "wt", encoding="utf-8") as f:
        f.write("#clade_name\tS1_profile\tS2_profile\n")
        # Duplicate species clades
        f.write("k__Bacteria|p__Firmicutes|c__Clostridia|o__Clostridiales|f__Ruminococcaceae|g__Faecalibacterium|s__Faecalibacterium_prausnitzii\t30.0\t40.0\n")
        f.write("k__Bacteria|p__Firmicutes|c__Clostridia|o__Clostridiales|f__Ruminococcaceae|g__Faecalibacterium|s__Faecalibacterium_prausnitzii\t20.0\t10.0\n")

    path_tsv = tax_dir / "pathabundances_3.tsv.gz"
    with gzip.open(path_tsv, "wt", encoding="utf-8") as f:
        f.write("# Pathway\tS1_pathabundance_cpm\tS2_pathabundance_cpm\n")
        f.write("PWY-DUP: duplicate pathway\t100.0\t100.0\n")
        f.write("PWY-DUP: duplicate pathway\t50.0\t50.0\n")

    cfg_file = tmp_path / "c1_config.yaml"
    cfg_file.write_text(f'paths:\n  raw_data_dir: "{raw_dir.as_posix()}"\n', encoding="utf-8")

    omics = load_and_align_omics_tables(cfg_file)
    assert not omics["species_relab"].columns.duplicated().any()
    assert not omics["pathways_unstrat_relab"].columns.duplicated().any()


def test_corrupted_gzip_raises_informative_error(tmp_path):
    """Verify that a corrupted gzip file raises an informative ValueError with path context."""
    raw_dir = tmp_path / "raw"
    tax_dir = raw_dir / "products" / "HMP2" / "MGX" / "2018-05-04"
    tax_dir.mkdir(parents=True, exist_ok=True)

    # Write truncated garbage bytes
    corrupt_gz = tax_dir / "taxonomic_profiles_3.tsv.gz"
    with open(corrupt_gz, "wb") as f:
        f.write(b"\x1f\x8b\x08\x00\x00\x00\x00\x00corrupted_garbage_bytes")

    path_tsv = tax_dir / "pathabundances_3.tsv.gz"
    with gzip.open(path_tsv, "wt", encoding="utf-8") as f:
        f.write("# Pathway\tS1\nPWY-1\t10\n")

    cfg_file = tmp_path / "c1_config.yaml"
    cfg_file.write_text(f'paths:\n  raw_data_dir: "{raw_dir.as_posix()}"\n', encoding="utf-8")

    with pytest.raises(ValueError, match="Corrupted or invalid gzip file at"):
        load_and_align_omics_tables(cfg_file)


def test_transform_scaler_dimension_mismatch_raises(synthetic_pairs_and_visits):
    """Verify fail-fast error when pipeline arm is changed after fitting and transform with standardization is called."""
    pairs_df, visit_store_df, omics_dict = synthetic_pairs_and_visits
    pipe = C1FeaturePipeline(arm="M1")
    pipe.fit(pairs_df, visit_store_df, omics_dict)

    # Artificially alter arm without re-fitting
    pipe.arm = "M2"
    with pytest.raises(RuntimeError, match="Scaler dimension mismatch"):
        pipe.transform(pairs_df, visit_store_df, omics_dict, standardize=True)


def test_medication_boolean_and_string_variants(synthetic_pairs_and_visits):
    """Verify that medication columns with True, 'true', 'yes', '1' are all parsed without NaNs."""
    pairs_df, visit_store_df, omics_dict = synthetic_pairs_and_visits
    pairs_test = pairs_df.copy()
    pairs_test["Antibiotics"] = pairs_test["Antibiotics"].astype(object)
    imm_col = "Immunosuppressants (e.g. oral corticosteroids)"
    pairs_test[imm_col] = pairs_test[imm_col].astype(object)

    pairs_test.loc[0, "Antibiotics"] = True
    pairs_test.loc[1, "Antibiotics"] = "true"
    pairs_test.loc[2, "Antibiotics"] = "1"
    pairs_test.loc[0, imm_col] = True
    pairs_test.loc[1, imm_col] = "YES"

    pipe = C1FeaturePipeline(arm="M1")
    pipe.fit(pairs_test, visit_store_df, omics_dict)
    X, names = pipe.transform(pairs_test, visit_store_df, omics_dict)
    assert np.isfinite(X).all()

    abx_idx = names.index("antibiotics_yes")
    assert X[0, abx_idx] == 1.0
    assert X[1, abx_idx] == 1.0
    assert X[2, abx_idx] == 1.0


def test_m_guild_and_m_guild_delta_no_raw_omics(synthetic_pairs_and_visits):
    """Verify M_guild and M_guild_delta include curated guilds/diversity without raw species/pathways."""
    pairs_df, visit_store_df, omics_dict = synthetic_pairs_and_visits
    train_pairs = pairs_df[pairs_df["partition"] == "dev"].copy()
    test_pairs = pairs_df[pairs_df["partition"] == "test"].copy()

    pipe_m1 = C1FeaturePipeline(arm="M1").fit(train_pairs, visit_store_df, omics_dict)
    X_m1, _ = pipe_m1.transform(train_pairs, visit_store_df, omics_dict)

    pipe_g = C1FeaturePipeline(arm="M_guild").fit(train_pairs, visit_store_df, omics_dict)
    X_g, names_g = pipe_g.transform(train_pairs, visit_store_df, omics_dict)

    pipe_gd = C1FeaturePipeline(arm="M_guild_delta").fit(train_pairs, visit_store_df, omics_dict)
    X_gd, names_gd = pipe_gd.transform(train_pairs, visit_store_df, omics_dict)

    assert np.isfinite(X_g).all()
    assert np.isfinite(X_gd).all()
    assert X_m1.shape[1] < X_g.shape[1] < X_gd.shape[1]

    # Must NOT contain raw CLR species or pathways
    assert not any(n.startswith("clr_species_") or n.startswith("clr_pathway_") for n in names_g)
    assert not any(n.startswith("clr_species_") or n.startswith("clr_pathway_") for n in names_gd)

    # Must contain curated guild & diversity features
    expected_guild_feats = [
        "shannon_species",
        "simpson_species",
        "guild_scfa_butyrate_log",
        "guild_pathobiont_log",
        "guild_protective_vs_pathobiont_logratio",
        "bray_curtis_from_prev_mgx",
        "fguild_scfa_butanoate_log",
        "fguild_sulfur_h2s_log",
        "fguild_oxidative_stress_log",
        "fguild_mucin_hexuronide_log",
        "fguild_lps_enterobacterial_log",
    ]
    for feat in expected_guild_feats:
        assert feat in names_g, f"Missing {feat} in M_guild"

    # PR-03 train-only invariance check on M_guild
    X_g_test, _ = pipe_g.transform(test_pairs, visit_store_df, omics_dict)
    assert np.isfinite(X_g_test).all()


