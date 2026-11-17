"""Unit and leakage tests for Phase P2 (C2: LME, Stability Selection & Cross-Omic Attention)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from micro_trace.c2.cross_omic_attention import (
    CrossOmicAttentionNet,
    train_and_extract_cross_omic_attention,
)
from micro_trace.c2.lme_signatures import (
    fit_lme_single_feature,
    fit_transform_clr_train_only,
    run_5fold_stability_selection,
    run_lme_family_scan,
)
from micro_trace.c2.subtype_stratified import run_subtype_stratified_analysis


def _make_synthetic_c2_data(n_participants: int = 25, visits_per_p: int = 5, seed: int = 42):
    rng = np.random.default_rng(seed)
    rows = []
    sp_rows = []
    pw_rows = []
    sample_ids = []

    for p_idx in range(n_participants):
        pid = f"P{p_idx:03d}"
        diag = ["CD", "UC", "nonIBD"][p_idx % 3]
        fold = p_idx % 5
        p_intercept = rng.normal(0.0, 0.3)
        for v in range(visits_per_p):
            sid = f"{pid}_V{v}"
            sample_ids.append(sid)
            x_true = rng.normal(0.0, 1.0)
            y = 1.8 + p_intercept + 0.45 * x_true + (0.3 if diag == "CD" else 0.0) + rng.normal(0, 0.2)
            rows.append(
                {
                    "participant_id": pid,
                    "clean_mgx_sample_id": sid,
                    "cv_fold": fold,
                    "partition": "dev",
                    "diagnosis": diag,
                    "antibiotics_flag": float(v == 2),
                    "immunosuppressants_flag": float(diag == "CD"),
                    "consent_age": 30.0 + p_idx,
                    "sex_male": float(p_idx % 2),
                    "target_fc_log10": y,
                    "guild_scfa_butyrate_log": -x_true + rng.normal(0, 0.1),
                    "guild_pathobiont_log": x_true + rng.normal(0, 0.1),
                    "guild_protective_vs_pathobiont_logratio": -2.0 * x_true,
                    "shannon_species": 2.5 - 0.2 * x_true,
                }
            )
            sp = rng.dirichlet(np.ones(8))
            sp[0] = np.clip(0.25 + 0.15 * np.tanh(x_true), 0.01, 0.8)
            sp = sp / sp.sum()
            sp_rows.append(sp)

            pw = rng.dirichlet(np.ones(10))
            pw[0] = np.clip(0.20 + 0.12 * np.tanh(x_true), 0.01, 0.8)
            pw = pw / pw.sum()
            pw_rows.append(pw)

    cohort_df = pd.DataFrame(rows)
    sp_df = pd.DataFrame(sp_rows, columns=[f"sp_{i}" for i in range(8)])
    sp_df.insert(0, "Unnamed: 0", sample_ids)
    pw_df = pd.DataFrame(pw_rows, columns=[f"pw_{i}" for i in range(10)])
    pw_df.insert(0, "Unnamed: 0", sample_ids)
    return cohort_df, sp_df, pw_df


def test_fit_transform_clr_train_only_no_leakage():
    cohort_df, sp_df, _ = _make_synthetic_c2_data(n_participants=10, visits_per_p=4)
    # Put a rare species that is 0 in train but non-zero in eval
    sp_df["rare_eval_only"] = 0.0
    eval_ids = cohort_df[cohort_df["cv_fold"] == 0]["clean_mgx_sample_id"].tolist()
    train_ids = cohort_df[cohort_df["cv_fold"] != 0]["clean_mgx_sample_id"].tolist()
    sp_df.loc[sp_df["Unnamed: 0"].isin(eval_ids), "rare_eval_only"] = 0.5

    tr_clr, ev_clr, kept, pseudo = fit_transform_clr_train_only(
        sp_df, train_ids, eval_ids, prevalence_threshold=0.10
    )
    assert "rare_eval_only" not in kept
    assert pseudo > 0.0
    assert np.allclose(tr_clr.values.mean(axis=1), 0.0, atol=1e-6)
    assert ev_clr is not None and np.allclose(ev_clr.values.mean(axis=1), 0.0, atol=1e-6)


def test_lme_single_feature_and_bh_fdr():
    cohort_df, _, _ = _make_synthetic_c2_data(n_participants=20, visits_per_p=5)
    feats = cohort_df[["guild_pathobiont_log", "guild_scfa_butyrate_log"]].copy()
    feats["noise_feat"] = np.random.default_rng(123).normal(0, 1, size=len(cohort_df))

    res = run_lme_family_scan(
        cohort_df,
        feats,
        family_name="guild",
        target_name="current_fc",
        covariate_cols=["diagnosis", "antibiotics_flag", "consent_age", "sex_male"],
    )
    assert len(res) == 3
    assert "fdr_bh" in res.columns
    patho_row = res[res["feature"] == "guild_pathobiont_log"].iloc[0]
    scfa_row = res[res["feature"] == "guild_scfa_butyrate_log"].iloc[0]
    assert patho_row["beta"] > 0.2
    assert scfa_row["beta"] < -0.2
    assert patho_row["fdr_bh"] < 0.05
    assert (res["fdr_bh"] >= res["p_value"] - 1e-12).all()


def test_lme_handles_constant_or_tiny_input_gracefully():
    cohort_df, _, _ = _make_synthetic_c2_data(n_participants=10, visits_per_p=3)
    res = fit_lme_single_feature(cohort_df, np.ones(len(cohort_df)))
    assert res["converged"] is False
    assert res["p_value"] == 1.0


def test_5fold_stability_selection_synthetic():
    cohort_df, sp_df, pw_df = _make_synthetic_c2_data(n_participants=25, visits_per_p=4)
    stab_df, summary = run_5fold_stability_selection(
        cohort_df,
        sp_df,
        pw_df,
        target_name="current_fc",
        covariate_cols=["diagnosis", "antibiotics_flag"],
        seed=42,
    )
    assert summary["n_folds"] == 5
    assert 0.0 <= summary["jaccard_lme_p05"] <= 1.0
    assert 0.0 <= summary["jaccard_elasticnet"] <= 1.0
    assert summary["n_stable_lme_ge4of5"] >= 1
    assert stab_df["lme_sig_folds"].max() >= 4


def test_cross_omic_attention_architecture_and_5fold():
    net = CrossOmicAttentionNet(n_taxa=6, n_pathways=8, n_clinical=3, d_model=16)
    x_t = torch.randn(4, 6)
    x_p = torch.randn(4, 8)
    x_c = torch.randn(4, 3)
    y, attn, tw = net(x_t, x_p, x_c)
    assert y.shape == (4,)
    assert attn.shape == (4, 6, 8)
    assert tw.shape == (4, 6)
    assert torch.allclose(attn.sum(dim=-1), torch.ones(4, 6), atol=1e-5)
    assert torch.allclose(tw.sum(dim=-1), torch.ones(4), atol=1e-5)

    cohort_df, sp_df, pw_df = _make_synthetic_c2_data(n_participants=20, visits_per_p=4)
    pairs_df, co_sum = train_and_extract_cross_omic_attention(
        cohort_df, sp_df, pw_df, target_name="current_fc", epochs=10, top_k_pairs_per_fold=15, seed=42
    )
    assert len(pairs_df) > 0
    assert np.isfinite(co_sum["oof_mae"])
    assert len(co_sum["fold_metrics"]) == 5


def test_subtype_stratified_isolation_and_no_test_leakage():
    cohort_df, sp_df, pw_df = _make_synthetic_c2_data(n_participants=36, visits_per_p=4, seed=42)
    # Mark last 6 participants (2 per subtype) as held-out test
    test_pids = sorted(cohort_df["participant_id"].unique())[-6:]
    cohort_df.loc[cohort_df["participant_id"].isin(test_pids), "partition"] = "test"
    cohort_df.loc[cohort_df["participant_id"].isin(test_pids), "cv_fold"] = -1

    # Add future_fc covariates
    cohort_df["index_log10_fc_same_day"] = cohort_df["target_fc_log10"] * 0.7
    cohort_df["elapsed_days"] = 28.0

    # Add test-only species and CD-only species to verify zero leakage & zero cross-subtype contamination
    sp_df["test_only_sp"] = 0.0
    test_sids = set(cohort_df.loc[cohort_df["partition"] == "test", "clean_mgx_sample_id"])
    sp_df.loc[sp_df["Unnamed: 0"].isin(test_sids), "test_only_sp"] = 0.5

    sp_df["cd_only_sp"] = 0.0
    cd_dev_sids = set(
        cohort_df.loc[(cohort_df["partition"] == "dev") & (cohort_df["diagnosis"] == "CD"), "clean_mgx_sample_id"]
    )
    sp_df.loc[sp_df["Unnamed: 0"].isin(cd_dev_sids), "cd_only_sp"] = 0.25

    dev_df = cohort_df[cohort_df["partition"] == "dev"].reset_index(drop=True)
    lme_df, stab_df, summary = run_subtype_stratified_analysis(
        curr_dev=dev_df,
        fut_dev=dev_df,
        species_relab=sp_df,
        pathways_relab=pw_df,
        prevalence_threshold=0.10,
        seed=42,
    )

    assert summary["test_partition_status"] == "untouched_held_out"
    assert set(lme_df["diagnosis_subgroup"].unique()) == {"CD", "UC", "nonIBD"}
    assert set(lme_df["target"].unique()) == {"current_fc", "future_fc"}
    assert set(stab_df["diagnosis_subgroup"].unique()) == {"CD", "UC", "nonIBD"}
    assert set(stab_df["target"].unique()) == {"current_fc", "future_fc"}

    # Zero test-participant leakage into feature universe
    assert "test_only_sp" not in set(lme_df["feature"])
    assert "test_only_sp" not in set(stab_df["feature"])

    # Zero cross-subtype contamination: cd_only_sp must only be tested in CD
    cd_only_rows = lme_df[lme_df["feature"] == "cd_only_sp"]
    assert len(cd_only_rows) > 0
    assert set(cd_only_rows["diagnosis_subgroup"].unique()) == {"CD"}

    # Finite LME and stability outputs
    assert np.isfinite(lme_df["beta"]).all()
    assert np.isfinite(lme_df["p_value"]).all()
    assert np.isfinite(lme_df["fdr_bh"]).all()
    assert (lme_df["fdr_bh"] >= lme_df["p_value"] - 1e-12).all()
    assert np.isfinite(stab_df["lme_stability"]).all()
    assert np.isfinite(stab_df["enet_stability"]).all()
    assert np.isfinite(stab_df["rf_stability"]).all()

    for t in ["current_fc", "future_fc"]:
        assert t in summary["targets"]
        assert "CD_vs_UC" in summary["targets"][t]["pairwise_beta_correlations"]


def test_subtype_stratified_rejects_test_partition_leakage():
    cohort_df, sp_df, pw_df = _make_synthetic_c2_data(n_participants=18, visits_per_p=4, seed=42)
    cohort_df.loc[cohort_df["participant_id"] == "P000", "partition"] = "test"
    cohort_df["index_log10_fc_same_day"] = cohort_df["target_fc_log10"] * 0.7
    cohort_df["elapsed_days"] = 28.0
    with pytest.raises(ValueError, match="dev"):
        run_subtype_stratified_analysis(
            curr_dev=cohort_df,
            fut_dev=cohort_df,
            species_relab=sp_df,
            pathways_relab=pw_df,
        )
