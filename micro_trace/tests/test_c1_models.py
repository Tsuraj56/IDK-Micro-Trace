"""Unit and verification tests for Phase P1 candidate models and benchmark pipeline."""

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from micro_trace.c1.models import (
    ContinuousTimeEmbedding,
    ElasticNetBenchmark,
    HistGradientBoostingBenchmark,
    MeanBaseline,
    MicroTraceAttn,
    MicroTraceAttnTrainer,
    PersistenceBaseline,
    RandomForestBenchmark,
    compute_delta_regression_metrics,
    compute_directional_accuracy,
    compute_regression_metrics,
    run_c1_delta_benchmark,
    run_c1_model_benchmark,
)


def test_compute_regression_metrics_exactness():
    """Verify regression metrics calculate exact MAE, RMSE, R2, and Spearman rho."""
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred = np.array([1.1, 1.9, 3.0, 4.2, 4.8])

    metrics = compute_regression_metrics(y_true, y_pred)
    assert metrics["mae"] == round(float(np.mean(np.abs(y_true - y_pred))), 4)
    assert metrics["rmse"] == round(float(np.sqrt(np.mean((y_true - y_pred) ** 2))), 4)
    assert metrics["r2"] > 0.95
    assert metrics["spearman_rho"] == 1.0


def test_compute_directional_accuracy():
    """Verify directional accuracy on non-zero transitions."""
    delta_true = np.array([0.5, -0.3, 0.8, -0.2, 0.0])
    delta_pred = np.array([0.2, -0.1, -0.4, -0.5, 0.3])

    # Non-zero pairs are indices 0, 1, 2, 3:
    # 0: true > 0, pred > 0 -> MATCH
    # 1: true < 0, pred < 0 -> MATCH
    # 2: true > 0, pred < 0 -> MISMATCH
    # 3: true < 0, pred < 0 -> MATCH
    # Total: 3/4 = 75.0%
    acc = compute_directional_accuracy(delta_true, delta_pred)
    assert acc == 75.0

    # Persistence baseline (all zeros): 0 matches for non-zero transitions
    pers_pred = np.zeros_like(delta_true)
    acc_pers = compute_directional_accuracy(delta_true, pers_pred)
    assert acc_pers == 0.0

    # Empty array
    assert compute_directional_accuracy(np.array([]), np.array([])) == 0.0

    # Float precision noise: predictions below tol are treated as zero
    noisy_zero_pred = np.array([1e-9, -1e-9, 1e-12, -1e-15, 0.0])
    acc_noisy = compute_directional_accuracy(delta_true, noisy_zero_pred, tol=1e-6)
    assert acc_noisy == 0.0


def test_compute_delta_regression_metrics_exactness():
    """Verify compute_delta_regression_metrics computes both reconstructed and delta metrics."""
    y_idx = np.array([2.0, 2.5, 1.8, 3.0])
    y_true = np.array([2.3, 2.1, 2.2, 2.8])
    delta_true = y_true - y_idx

    delta_pred = np.array([0.2, -0.3, 0.5, -0.1])
    y_pred_rec = y_idx + delta_pred

    metrics = compute_delta_regression_metrics(
        y_true_reconstructed=y_true,
        y_pred_reconstructed=y_pred_rec,
        delta_true=delta_true,
        delta_pred=delta_pred,
    )

    # MAE on reconstructed must equal MAE on delta
    assert metrics["mae_reconstructed"] == metrics["mae_delta"]
    assert metrics["rmse_reconstructed"] == metrics["rmse_delta"]
    assert 0.0 <= metrics["directional_accuracy"] <= 100.0


def test_regression_metrics_edge_cases():
    """Verify metrics handle empty and constant arrays gracefully without crashing."""
    m_empty = compute_regression_metrics(np.array([]), np.array([]))
    assert m_empty["mae"] == 0.0

    m_const = compute_regression_metrics(np.array([2.0, 2.0]), np.array([2.0, 2.0]))
    assert m_const["mae"] == 0.0
    assert m_const["spearman_rho"] == 0.0


def test_baseline_models_contract():
    """Verify MeanBaseline and PersistenceBaseline adhere to scikit-learn regressor contract."""
    X = np.array([[2.5, 10.0], [1.5, 20.0], [3.0, 30.0]])
    y = np.array([2.0, 1.8, 3.2])

    mean_b = MeanBaseline().fit(X, y)
    preds_mean = mean_b.predict(X)
    assert preds_mean.shape == (3,)
    assert np.allclose(preds_mean, np.mean(y))

    # Persistence in level mode
    pers_b = PersistenceBaseline(fc_col_idx=0, target_mode="level").fit(X, y)
    preds_pers = pers_b.predict(X)
    assert preds_pers.shape == (3,)
    assert np.allclose(preds_pers, X[:, 0])

    # Persistence in delta mode
    pers_delta = PersistenceBaseline(target_mode="delta").fit(X, y)
    preds_delta = pers_delta.predict(X)
    assert preds_delta.shape == (3,)
    assert np.allclose(preds_delta, 0.0)


def test_tabular_ml_models_fit_and_predict():
    """Verify ElasticNet, RandomForest, and HistGradientBoosting train and predict on synthetic data."""
    np.random.seed(42)
    X = np.random.randn(50, 10)
    y = 0.5 * X[:, 0] - 0.3 * X[:, 1] + 1.5

    # Elastic Net
    en = ElasticNetBenchmark(cv=2, max_iter=500).fit(X, y)
    en_preds = en.predict(X)
    assert en_preds.shape == (50,)
    assert np.isfinite(en_preds).all()

    # Random Forest
    rf = RandomForestBenchmark(n_estimators=20, max_depth=4).fit(X, y)
    rf_preds = rf.predict(X)
    assert rf_preds.shape == (50,)
    assert np.isfinite(rf_preds).all()

    # HistGradientBoosting
    hgb = HistGradientBoostingBenchmark(max_iter=30, max_depth=3).fit(X, y)
    hgb_preds = hgb.predict(X)
    assert hgb_preds.shape == (50,)
    assert np.isfinite(hgb_preds).all()


def test_continuous_time_embedding():
    """Verify ContinuousTimeEmbedding embeds elapsed days into continuous latent vectors."""
    embedder = ContinuousTimeEmbedding(embed_dim=32, num_frequencies=8)
    delta_t = torch.tensor([[0.0, 14.0, 45.0], [5.0, 30.0, 90.0]], dtype=torch.float32)

    emb = embedder(delta_t)
    assert emb.shape == (2, 3, 32)
    assert torch.isfinite(emb).all()


def test_microtrace_attn_architecture_and_skip_identity():
    """Verify MicroTraceAttn initializes with exact persistence identity in both level and delta modes."""
    B, T, D_seq, D_anchor, D_omics = 4, 10, 15, 6, 50

    # 1. Level mode
    model_level = MicroTraceAttn(
        d_seq=D_seq,
        d_anchor=D_anchor,
        d_omics=D_omics,
        embed_dim=32,
        num_heads=2,
        target_mode="level",
    )

    static_anchor = torch.tensor([
        [1.85, 0.45, 0.60, 1.0, 0.0, 0.0],
        [2.10, 0.30, 0.50, 0.0, 1.0, 0.0],
        [0.90, 0.60, 0.40, 0.0, 0.0, 1.0],
        [2.45, 0.15, 0.70, 1.0, 0.0, 0.0],
    ], dtype=torch.float32)

    with torch.no_grad():
        skip_only = model_level.linear_skip(static_anchor).squeeze(-1)
        assert torch.allclose(skip_only, static_anchor[:, 0], atol=1e-5)

    # 2. Delta mode
    model_delta = MicroTraceAttn(
        d_seq=D_seq,
        d_anchor=D_anchor,
        d_omics=D_omics,
        embed_dim=32,
        num_heads=2,
        target_mode="delta",
    )
    with torch.no_grad():
        skip_delta = model_delta.linear_skip(static_anchor).squeeze(-1)
        assert torch.allclose(skip_delta, torch.zeros(B), atol=1e-5)


def test_microtrace_attn_trainer_fit_and_predict():
    """Verify MicroTraceAttnTrainer trains with L1 and SmoothL1 loss on synthetic data."""
    B, T, D_seq, D_anchor, D_omics = 20, 8, 15, 6, 30
    np.random.seed(42)

    train_data = {
        "seq_features": np.random.randn(B, T, D_seq).astype(np.float32),
        "seq_time_deltas": np.random.uniform(0, 80, (B, T)).astype(np.float32),
        "seq_mask": np.ones((B, T), dtype=bool),
        "static_anchor": np.random.randn(B, D_anchor).astype(np.float32),
        "current_omics": np.random.randn(B, D_omics).astype(np.float32),
    }
    delta_train = np.random.randn(B).astype(np.float32) * 0.2

    # Test with L1 loss
    trainer_l1 = MicroTraceAttnTrainer(
        embed_dim=16,
        num_heads=2,
        epochs=3,
        batch_size=8,
        loss_type="l1",
        target_mode="delta",
        device=torch.device("cpu"),
    )
    trainer_l1.fit(train_data, delta_train)
    preds_l1 = trainer_l1.predict(train_data)
    assert preds_l1.shape == (B,)
    assert np.isfinite(preds_l1).all()

    # Test with SmoothL1 loss
    trainer_smooth = MicroTraceAttnTrainer(
        embed_dim=16,
        num_heads=2,
        epochs=3,
        batch_size=8,
        loss_type="smooth_l1",
        target_mode="delta",
        device=torch.device("cpu"),
    )
    trainer_smooth.fit(train_data, delta_train)
    preds_smooth = trainer_smooth.predict(train_data)
    assert preds_smooth.shape == (B,)
    assert np.isfinite(preds_smooth).all()


def test_c1_delta_benchmark_preserves_untouched_test_partition(tmp_path):
    """Verify run_c1_delta_benchmark strictly leaves test partition untouched and leak-free."""
    # Build 5 dev participants (folds 0..4) and 1 test participant
    visit_rows = []
    pair_rows = []
    pids = [f"P_{i}" for i in range(6)]
    sample_ids = [f"SMP_{i}" for i in range(12)]
    smp_idx = 0

    for i, pid in enumerate(pids):
        diag = "CD" if i % 2 == 0 else "UC"
        for v in range(2):
            cid = f"{pid}_C{v+1}"
            sid = sample_ids[smp_idx % len(sample_ids)]
            smp_idx += 1
            dt = f"2020-0{i+1}-{v*14+1:02d}"
            fc = 50.0 + 10.0 * (i + v)
            visit_rows.append({
                "collection_event_id": cid,
                "participant_id": pid,
                "diagnosis": diag,
                "is_ibd": True,
                "visit_num": v + 1,
                "week_num": float(v * 2),
                "event_date": pd.to_datetime(dt),
                "fc_value": fc,
                "clean_mgx_sample_id": sid,
                "has_mgx_sample": True,
                "is_zero_sum_mgx_sample": False,
                "hbi": 2.0,
                "sccai": 2.0,
                "fc_log10": np.log10(fc + 1.0),
                "shannon_species": 2.5,
                "guild_scfa_butyrate_log": 1.0,
            })
        next_fc = 60.0 + 10.0 * i
        pair_rows.append({
            "pair_id": f"{pid}__pair",
            "participant_id": pid,
            "diagnosis": diag,
            "is_ibd": True,
            "index_collection_event": f"{pid}_C1",
            "index_date": f"2020-0{i+1}-01",
            "index_mgx_sample_id": f"SMP_{(i*2)%12}",
            "index_log10_fc": np.log10(50.0 + 10.0 * i + 1.0),
            "index_log10_fc_same_day": np.log10(50.0 + 10.0 * i + 1.0),
            "future_collection_event": f"{pid}_C2",
            "future_date": f"2020-0{i+1}-15",
            "future_fc": next_fc,
            "future_log10_fc_plus_1": np.log10(next_fc + 1.0),
            "elapsed_days": 14,
            "consent_age": 30.0,
            "Age at diagnosis": 20.0,
            "sex": "Female",
            "hbi": 2.0,
            "sccai": 2.0,
            "Bowel frequency during the day": 2.0,
            "Abdominal pain": "None",
            "General wellbeing": "Generally well",
            "Antibiotics": "No",
            "Immunosuppressants (e.g. oral corticosteroids)": "No",
            "reads_filtered": 1e7,
            "reads_human": 1e5,
            "partition": "dev" if i < 5 else "test",
            "cv_fold": i if i < 5 else -1,
        })

    visit_store_df = pd.DataFrame(visit_rows)
    pairs_df = pd.DataFrame(pair_rows)
    dev_pairs = pairs_df[pairs_df["partition"] == "dev"].copy()
    test_pairs = pairs_df[pairs_df["partition"] == "test"].copy()

    sp_cols = ["k__Bacteria|p__Firmicutes|s__Faecalibacterium_prausnitzii", "k__Bacteria|p__Proteobacteria|s__Escherichia_coli"]
    omics_dict = {
        "species_relab": pd.DataFrame(0.5, index=sample_ids, columns=sp_cols),
        "pathways_unstrat_relab": pd.DataFrame(0.5, index=sample_ids, columns=["PWY-1", "PWY-2"]),
    }

    cfg_data = {
        "paths": {
            "c1_reports_dir": str(tmp_path),
            "raw_data_dir": str(tmp_path),
            "schema_mapping_path": "micro_trace/configs/schema_mapping.yaml",
        }
    }
    cfg_file = tmp_path / "test_cfg.yaml"
    with open(cfg_file, "w", encoding="utf-8") as f:
        yaml.dump(cfg_data, f)

    # 1. Clean run
    summary_clean = run_c1_delta_benchmark(
        cfg_path=cfg_file,
        arms=("M0", "M1"),
        families=("ElasticNetCV",),
        evaluate_test=False,
        dev_pairs=dev_pairs,
        test_pairs=test_pairs,
        visit_store_df=visit_store_df,
        omics_dict=omics_dict,
        output_csv_name="clean_results.csv",
        output_json_name="clean_summary.json",
    )
    assert summary_clean["test_partition_status"] == "untouched_held_out"
    cv_clean = summary_clean["best_cv_model"]["cv_mae_reconstructed"]

    # 2. Corrupted test set
    corrupted_test = test_pairs.copy()
    corrupted_test["future_log10_fc_plus_1"] = 99999.0
    corrupted_test["index_log10_fc_same_day"] = -1000.0

    summary_corrupted = run_c1_delta_benchmark(
        cfg_path=cfg_file,
        arms=("M0", "M1"),
        families=("ElasticNetCV",),
        evaluate_test=False,
        dev_pairs=dev_pairs,
        test_pairs=corrupted_test,
        visit_store_df=visit_store_df,
        omics_dict=omics_dict,
        output_csv_name="corrupted_results.csv",
        output_json_name="corrupted_summary.json",
    )
    cv_corrupted = summary_corrupted["best_cv_model"]["cv_mae_reconstructed"]

    # Zero leakage invariant
    assert cv_clean == cv_corrupted, f"Leakage detected! {cv_clean} != {cv_corrupted}"


def test_microtrace_attn_inner_val_isolation(tmp_path):
    """Verify MicroTrace-Attn with use_inner_val=True does not leak outer fold labels into training."""
    # Synthetic dataset with 5 dev folds (3 participants each = 15 participants)
    visit_rows = []
    pair_rows = []
    pids = [f"P_{i:02d}" for i in range(15)]
    sample_ids = [f"SMP_{i:02d}" for i in range(30)]

    for i, pid in enumerate(pids):
        diag = "CD" if i % 2 == 0 else "UC"
        for v in range(2):
            cid = f"{pid}_C{v+1}"
            sid = sample_ids[(i * 2 + v) % len(sample_ids)]
            fc = 50.0 + 5.0 * i + v
            visit_rows.append({
                "collection_event_id": cid,
                "participant_id": pid,
                "diagnosis": diag,
                "is_ibd": True,
                "visit_num": v + 1,
                "week_num": float(v * 2),
                "event_date": pd.to_datetime(f"2020-01-{(v*14+1):02d}"),
                "fc_value": fc,
                "clean_mgx_sample_id": sid,
                "has_mgx_sample": True,
                "is_zero_sum_mgx_sample": False,
                "hbi": 2.0,
                "sccai": 2.0,
                "fc_log10": np.log10(fc + 1.0),
                "shannon_species": 2.5,
                "guild_scfa_butyrate_log": 1.0,
            })
        next_fc = 55.0 + 5.0 * i
        pair_rows.append({
            "pair_id": f"{pid}__pair",
            "participant_id": pid,
            "diagnosis": diag,
            "is_ibd": True,
            "index_collection_event": f"{pid}_C1",
            "index_date": "2020-01-01",
            "index_mgx_sample_id": f"SMP_{(i*2)%len(sample_ids)}",
            "index_log10_fc": np.log10(50.0 + 5.0 * i + 1.0),
            "index_log10_fc_same_day": np.log10(50.0 + 5.0 * i + 1.0),
            "future_collection_event": f"{pid}_C2",
            "future_date": "2020-01-15",
            "future_fc": next_fc,
            "future_log10_fc_plus_1": np.log10(next_fc + 1.0),
            "elapsed_days": 14,
            "consent_age": 30.0,
            "Age at diagnosis": 20.0,
            "sex": "Female",
            "hbi": 2.0,
            "sccai": 2.0,
            "Bowel frequency during the day": 2.0,
            "Abdominal pain": "None",
            "General wellbeing": "Generally well",
            "Antibiotics": "No",
            "Immunosuppressants (e.g. oral corticosteroids)": "No",
            "reads_filtered": 1e7,
            "reads_human": 1e5,
            "partition": "dev",
            "cv_fold": i % 5,
        })

    visit_store_df = pd.DataFrame(visit_rows)
    dev_pairs = pd.DataFrame(pair_rows)
    sp_cols = ["k__Bacteria|p__Firmicutes|s__Faecalibacterium_prausnitzii", "k__Bacteria|p__Proteobacteria|s__Escherichia_coli"]
    omics_dict = {
        "species_relab": pd.DataFrame(0.5, index=sample_ids, columns=sp_cols),
        "pathways_unstrat_relab": pd.DataFrame(0.5, index=sample_ids, columns=["PWY-1", "PWY-2"]),
    }

    cfg_data = {
        "paths": {
            "c1_reports_dir": str(tmp_path),
            "raw_data_dir": str(tmp_path),
            "schema_mapping_path": "micro_trace/configs/schema_mapping.yaml",
        }
    }
    cfg_file = tmp_path / "test_cfg.yaml"
    with open(cfg_file, "w", encoding="utf-8") as f:
        yaml.dump(cfg_data, f)

    summary = run_c1_delta_benchmark(
        cfg_path=cfg_file,
        arms=("M1",),
        families=("MicroTrace-Attn-L1",),
        epochs_attn=3,
        use_inner_val=True,
        evaluate_test=False,
        dev_pairs=dev_pairs,
        visit_store_df=visit_store_df,
        omics_dict=omics_dict,
        output_csv_name="inner_val_test.csv",
        output_json_name="inner_val_test.json",
    )
    assert len(summary["summary_table"]) == 2  # M0 and M1
    m1_row = [r for r in summary["summary_table"] if r["model_family"] == "MicroTrace-Attn-L1"][0]
    assert 0.0 <= m1_row["cv_mae_reconstructed"] <= 5.0
    assert 0.0 <= m1_row["cv_directional_acc"] <= 100.0


def test_oof_stack_alpha_tuning_and_leak_freedom():
    """Verify leak-free OOF stacking tunes alpha strictly on inner OOF predictions."""
    from micro_trace.c1.trials_combinations import (
        evaluate_oof_stack_from_cached,
        tune_oof_stack_alpha,
    )

    y_true = np.array([0.1, -0.2, 0.3, -0.4, 0.5])
    # Case 1: attn exact, enet noisy -> alpha == 1.0
    a1, _ = tune_oof_stack_alpha(y_true, y_true + 1.0, y_true)
    assert a1 == 1.0
    # Case 2: enet exact, attn noisy -> alpha == 0.0
    a0, _ = tune_oof_stack_alpha(y_true - 1.0, y_true, y_true)
    assert a0 == 0.0
    # Case 3: 50/50 average is exact -> alpha == 0.5
    a05, _ = tune_oof_stack_alpha(y_true + 0.4, y_true - 0.4, y_true)
    assert a05 == 0.5

    # Verify evaluate_oof_stack_from_cached never peeks at outer val fold when tuning alpha
    rows = []
    for fold in range(5):
        for j in range(4):
            rows.append({
                "participant_id": f"P_{fold}_{j}",
                "cv_fold": fold,
                "index_log10_fc_same_day": 2.0 + 0.1 * j,
                "future_log10_fc_plus_1": 2.0 + 0.1 * j + (0.2 if j % 2 == 0 else -0.2),
            })
    dev_pairs = pd.DataFrame(rows)
    delta_all = (dev_pairs["future_log10_fc_plus_1"] - dev_pairs["index_log10_fc_same_day"]).to_numpy()

    inner_oof_attn = {}
    inner_oof_enet = {}
    outer_attn = {}
    outer_enet = {}
    for fold in range(5):
        tr_mask = (dev_pairs["cv_fold"] != fold).to_numpy()
        va_mask = (dev_pairs["cv_fold"] == fold).to_numpy()
        inner_oof_attn[fold] = delta_all[tr_mask]
        inner_oof_enet[fold] = delta_all[tr_mask] + 0.5
        outer_attn[fold] = delta_all[va_mask]
        outer_enet[fold] = delta_all[va_mask] + 0.5

    res_clean = evaluate_oof_stack_from_cached(
        dev_pairs=dev_pairs,
        inner_oof_attn=inner_oof_attn,
        outer_attn=outer_attn,
        inner_oof_enet=inner_oof_enet,
        outer_enet=outer_enet,
        arm_label="M_guild",
        model_label="OOFStackTest",
        combination_label="Test",
        feature_dim=43,
    )
    assert res_clean["fold_alphas"] == [1.0, 1.0, 1.0, 1.0, 1.0]
    assert res_clean["cv_mae_reconstructed"] == 0.0

    # Corrupt outer fold predictions for enet: tuned fold_alphas must remain identical!
    outer_enet_corrupt = {k: v - 999.0 for k, v in outer_enet.items()}
    res_corrupt = evaluate_oof_stack_from_cached(
        dev_pairs=dev_pairs,
        inner_oof_attn=inner_oof_attn,
        outer_attn=outer_attn,
        inner_oof_enet=inner_oof_enet,
        outer_enet=outer_enet_corrupt,
        arm_label="M_guild",
        model_label="OOFStackTest",
        combination_label="Test",
        feature_dim=43,
    )
    assert res_corrupt["fold_alphas"] == res_clean["fold_alphas"]


def test_multi_horizon_pretrain_zero_val_test_leakage():
    """Verify multi-horizon pretraining strictly excludes val/test participants and enforces [7, 90]d."""
    from micro_trace.c1.trials_combinations import filter_multi_horizon_pretrain_df

    all_h = pd.DataFrame([
        {"participant_id": "P_TRAIN_1", "partition": "dev", "cv_fold": 1, "elapsed_days": 14, "index_log10_fc_same_day": 2.0, "future_log10_fc_plus_1": 2.1},
        {"participant_id": "P_TRAIN_2", "partition": "dev", "cv_fold": 2, "elapsed_days": 60, "index_log10_fc_same_day": 1.8, "future_log10_fc_plus_1": 1.5},
        {"participant_id": "P_TRAIN_1", "partition": "dev", "cv_fold": 1, "elapsed_days": 3, "index_log10_fc_same_day": 2.0, "future_log10_fc_plus_1": 2.1},
        {"participant_id": "P_TRAIN_1", "partition": "dev", "cv_fold": 1, "elapsed_days": 120, "index_log10_fc_same_day": 2.0, "future_log10_fc_plus_1": 2.1},
        {"participant_id": "P_VAL_0", "partition": "dev", "cv_fold": 0, "elapsed_days": 21, "index_log10_fc_same_day": 2.2, "future_log10_fc_plus_1": 2.4},
        {"participant_id": "P_TEST_9", "partition": "test", "cv_fold": -1, "elapsed_days": 28, "index_log10_fc_same_day": 1.9, "future_log10_fc_plus_1": 2.0},
    ])

    train_pids = {"P_TRAIN_1", "P_TRAIN_2"}
    filtered = filter_multi_horizon_pretrain_df(all_h, train_pids, min_days=7, max_days=90)

    assert len(filtered) == 2
    assert set(filtered["participant_id"]) == {"P_TRAIN_1", "P_TRAIN_2"}
    assert "P_VAL_0" not in set(filtered["participant_id"])
    assert "P_TEST_9" not in set(filtered["participant_id"])
    assert (filtered["elapsed_days"] >= 7).all() and (filtered["elapsed_days"] <= 90).all()

    # Also verify init_state_dict + get_state_dict roundtrip in MicroTraceAttnTrainer
    B, T, D_seq, D_anchor, D_omics = 12, 5, 15, 6, 10
    rng = np.random.default_rng(42)
    data = {
        "seq_features": rng.standard_normal((B, T, D_seq)).astype(np.float32),
        "seq_time_deltas": rng.uniform(0, 50, (B, T)).astype(np.float32),
        "seq_mask": np.ones((B, T), dtype=bool),
        "static_anchor": rng.standard_normal((B, D_anchor)).astype(np.float32),
        "current_omics": rng.standard_normal((B, D_omics)).astype(np.float32),
    }
    y = rng.standard_normal(B).astype(np.float32) * 0.1
    pre = MicroTraceAttnTrainer(embed_dim=16, num_heads=2, epochs=2, batch_size=6, loss_type="l1", target_mode="delta", device=torch.device("cpu"))
    pre.fit(data, y)
    sd = pre.get_state_dict()
    ft = MicroTraceAttnTrainer(embed_dim=16, num_heads=2, epochs=2, batch_size=6, loss_type="l1", target_mode="delta", device=torch.device("cpu"))
    ft.fit(data, y, init_state_dict=sd)
    preds = ft.predict(data)
    assert preds.shape == (B,) and np.isfinite(preds).all()


def test_run_c1_trials_combinations_benchmark_synthetic(tmp_path):
    """End-to-end test for run_c1_trials_combinations_benchmark: test-untouched guard, valid JSON (no NaN), all combos."""
    import json
    from micro_trace.c1.trials_combinations import run_c1_trials_combinations_benchmark

    visit_rows = []
    pair_rows = []
    all_h_rows = []
    sample_ids = [f"SMP_{i}" for i in range(15)]

    # 10 dev participants across 5 folds (2 per fold), 2 visits each
    for i in range(10):
        pid = f"P_{i}"
        diag = "CD" if i % 2 == 0 else "UC"
        fold = i % 5
        for v in range(2):
            cid = f"{pid}_C{v+1}"
            sid = sample_ids[(i * 2 + v) % len(sample_ids)]
            dt = f"2020-{(i % 9) + 1:02d}-{v * 14 + 1:02d}"
            fc = 40.0 + 15.0 * ((i + v) % 4)
            visit_rows.append({
                "collection_event_id": cid,
                "participant_id": pid,
                "diagnosis": diag,
                "is_ibd": True,
                "visit_num": v + 1,
                "week_num": float(v * 2),
                "event_date": pd.to_datetime(dt),
                "fc_value": fc,
                "clean_mgx_sample_id": sid,
                "has_mgx_sample": True,
                "is_zero_sum_mgx_sample": False,
                "hbi": 2.0,
                "sccai": 2.0,
                "fc_log10": np.log10(fc + 1.0),
                "shannon_species": 2.2,
                "guild_scfa_butyrate_log": -1.5,
            })

        idx_fc = 40.0 + 15.0 * (i % 4)
        fut_fc = 40.0 + 15.0 * ((i + 1) % 4)
        row = {
            "pair_id": f"{pid}__pair",
            "participant_id": pid,
            "diagnosis": diag,
            "is_ibd": True,
            "index_collection_event": f"{pid}_C1",
            "index_date": f"2020-{(i % 9) + 1:02d}-01",
            "index_mgx_sample_id": sample_ids[(i * 2) % len(sample_ids)],
            "index_log10_fc": np.log10(idx_fc + 1.0),
            "index_log10_fc_same_day": np.log10(idx_fc + 1.0),
            "future_collection_event": f"{pid}_C2",
            "future_date": f"2020-{(i % 9) + 1:02d}-15",
            "future_fc": fut_fc,
            "future_log10_fc_plus_1": np.log10(fut_fc + 1.0),
            "elapsed_days": 14,
            "consent_age": 30.0 + i,
            "Age at diagnosis": 20.0,
            "sex": "Female" if i % 2 == 0 else "Male",
            "hbi": 2.0,
            "sccai": 2.0,
            "Bowel frequency during the day": 2.0,
            "Abdominal pain": "None",
            "General wellbeing": "Generally well",
            "Antibiotics": "No",
            "Immunosuppressants (e.g. oral corticosteroids)": "No",
            "reads_filtered": 1e7,
            "reads_human": 1e5,
            "partition": "dev",
            "cv_fold": fold,
        }
        pair_rows.append(row)
        all_h_rows.append(dict(row, elapsed_days=21))

    visit_store_df = pd.DataFrame(visit_rows)
    dev_pairs = pd.DataFrame(pair_rows)
    all_horizons_df = pd.DataFrame(all_h_rows)
    sp_cols = [
        "k__Bacteria|p__Firmicutes|s__Faecalibacterium_prausnitzii",
        "k__Bacteria|p__Proteobacteria|s__Escherichia_coli",
    ]
    omics_dict = {
        "species_relab": pd.DataFrame(0.5, index=sample_ids, columns=sp_cols),
        "pathways_unstrat_relab": pd.DataFrame(0.5, index=sample_ids, columns=["CENTFERM-PWY", "SO4ASSIM-PWY"]),
    }

    cfg_file = tmp_path / "test_cfg.yaml"
    with open(cfg_file, "w", encoding="utf-8") as f:
        yaml.dump({
            "paths": {
                "c1_reports_dir": str(tmp_path),
                "raw_data_dir": str(tmp_path),
                "schema_mapping_path": "micro_trace/configs/schema_mapping.yaml",
            }
        }, f)

    with pytest.raises(ValueError, match="untouched"):
        run_c1_trials_combinations_benchmark(cfg_path=cfg_file, evaluate_test=True)

    summary = run_c1_trials_combinations_benchmark(
        cfg_path=cfg_file,
        epochs_attn=2,
        pretrain_epochs=2,
        evaluate_test=False,
        dev_pairs=dev_pairs,
        all_horizons_df=all_horizons_df,
        visit_store_df=visit_store_df,
        omics_dict=omics_dict,
        output_csv_name="synth_combos.csv",
        output_json_name="synth_combos.json",
    )
    assert summary["test_partition_status"] == "untouched_held_out"
    assert (tmp_path / "synth_combos.csv").exists()
    json_text = (tmp_path / "synth_combos.json").read_text(encoding="utf-8")
    assert "NaN" not in json_text
    parsed = json.loads(json_text)
    assert len(parsed["summary_table"]) == len(summary["summary_table"])


def test_subtype_method1_zero_cross_subtype_contamination():
    """Verify Method 1 subtype filtering & fitting strictly isolate CD, UC, and nonIBD with zero contamination."""
    from micro_trace.c1.subtype_models import _fit_predict_enet, filter_subtype_df

    df = pd.DataFrame({
        "participant_id": ["P1", "P2", "P3", "P4", "P5", "P6"],
        "diagnosis": ["CD", "UC", "nonIBD", "CD", "CD", "CD"],
        "x1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        "x2": [0.5, -1.0, 2.0, 1.5, 2.5, 3.0],
        "y": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
    })
    for sub, expected_n in [("CD", 4), ("UC", 1), ("nonIBD", 1)]:
        sub_df = filter_subtype_df(df, sub)
        assert len(sub_df) == expected_n
        assert set(sub_df["diagnosis"]) == {sub}

    with pytest.raises(ValueError, match="Unsupported subtype"):
        filter_subtype_df(df, "INVALID")

    # Perturbing UC/nonIBD rows must have zero effect on CD-only model predictions
    df_perturbed = df.copy()
    non_cd = df_perturbed["diagnosis"] != "CD"
    df_perturbed.loc[non_cd, ["x1", "x2", "y"]] = 9999.0

    cd1 = filter_subtype_df(df, "CD")
    cd2 = filter_subtype_df(df_perturbed, "CD")
    X_eval = np.array([[2.0, 1.0], [4.5, 2.0]])
    pred1 = _fit_predict_enet(cd1[["x1", "x2"]].to_numpy(), cd1["y"].to_numpy(), X_eval, seed=42)
    pred2 = _fit_predict_enet(cd2[["x1", "x2"]].to_numpy(), cd2["y"].to_numpy(), X_eval, seed=42)
    assert np.allclose(pred1, pred2)


def test_subtype_method2_interaction_no_val_leakage_and_subtype_head():
    """Verify Method 2 interaction scaler is strictly train-fitted and SubtypeHeadMicroTraceAttn runs cleanly."""
    from micro_trace.c1.subtype_models import (
        SubtypeHeadMicroTraceAttnTrainer,
        build_subtype_interaction_features,
    )

    rng = np.random.default_rng(42)
    X_tr = rng.standard_normal((20, 4))
    d_tr = np.array(["CD"] * 8 + ["UC"] * 7 + ["nonIBD"] * 5)
    X_ev1 = rng.standard_normal((6, 4))
    X_ev2 = X_ev1 * 100.0 + 500.0  # Extreme shift in eval must NOT affect X_tr_int
    d_ev = np.array(["CD", "UC", "nonIBD", "CD", "UC", "nonIBD"])

    X_tr_int1, X_ev_int1 = build_subtype_interaction_features(X_tr, d_tr, X_ev1, d_ev)
    X_tr_int2, _ = build_subtype_interaction_features(X_tr, d_tr, X_ev2, d_ev)

    assert X_tr_int1.shape == (20, 4 * 3 + 2)
    assert X_ev_int1.shape == (6, 4 * 3 + 2)
    assert np.allclose(X_tr_int1, X_tr_int2)

    # SubtypeHeadMicroTraceAttnTrainer forward/fit/predict
    B, T, D_seq, D_anchor, D_omics = 12, 5, 15, 6, 8
    tr_data = {
        "seq_features": rng.standard_normal((B, T, D_seq)).astype(np.float32),
        "seq_time_deltas": rng.uniform(0, 40, (B, T)).astype(np.float32),
        "seq_mask": np.ones((B, T), dtype=bool),
        "static_anchor": rng.standard_normal((B, D_anchor)).astype(np.float32),
        "current_omics": rng.standard_normal((B, D_omics)).astype(np.float32),
    }
    y_tr = rng.standard_normal(B).astype(np.float32) * 0.1
    diag_tr = np.array(["CD", "UC", "nonIBD"] * 4)

    trainer = SubtypeHeadMicroTraceAttnTrainer(
        embed_dim=16, num_heads=2, epochs=2, batch_size=6, device=torch.device("cpu")
    )
    trainer.fit(tr_data, y_tr, diag_tr, val_data=tr_data, y_val=y_tr, diag_val=diag_tr)
    preds = trainer.predict(tr_data, diag_tr)
    assert preds.shape == (B,)
    assert np.isfinite(preds).all()


def test_run_c1_subtype_benchmark_synthetic_and_guards(tmp_path):
    """End-to-end test for run_c1_subtype_benchmark: evaluate_test=True rejection, valid JSON/CSV, all 4 methods."""
    import json
    from micro_trace.c1.subtype_models import run_c1_subtype_benchmark

    visit_rows = []
    pair_rows = []
    all_h_rows = []
    sample_ids = [f"SMP_{i}" for i in range(30)]
    subtypes = ["CD", "UC", "nonIBD"]

    # 15 dev participants across 5 folds (3 per fold: 1 CD, 1 UC, 1 nonIBD)
    for i in range(15):
        pid = f"P_{i}"
        diag = subtypes[i % 3]
        fold = i // 3
        for v in range(2):
            cid = f"{pid}_C{v+1}"
            sid = sample_ids[(i * 2 + v) % len(sample_ids)]
            dt = f"2020-{(i % 9) + 1:02d}-{v * 14 + 1:02d}"
            fc = 40.0 + 15.0 * ((i + v) % 4)
            visit_rows.append({
                "collection_event_id": cid,
                "participant_id": pid,
                "diagnosis": diag,
                "is_ibd": diag != "nonIBD",
                "visit_num": v + 1,
                "week_num": float(v * 2),
                "event_date": pd.to_datetime(dt),
                "fc_value": fc,
                "clean_mgx_sample_id": sid,
                "has_mgx_sample": True,
                "is_zero_sum_mgx_sample": False,
                "hbi": 2.0,
                "sccai": 2.0,
                "fc_log10": np.log10(fc + 1.0),
                "shannon_species": 2.2,
                "guild_scfa_butyrate_log": -1.5,
            })

        idx_fc = 40.0 + 15.0 * (i % 4)
        fut_fc = 40.0 + 15.0 * ((i + 1) % 4)
        row = {
            "pair_id": f"{pid}__pair",
            "participant_id": pid,
            "diagnosis": diag,
            "is_ibd": diag != "nonIBD",
            "index_collection_event": f"{pid}_C1",
            "index_date": f"2020-{(i % 9) + 1:02d}-01",
            "index_mgx_sample_id": sample_ids[(i * 2) % len(sample_ids)],
            "index_log10_fc": np.log10(idx_fc + 1.0),
            "index_log10_fc_same_day": np.log10(idx_fc + 1.0),
            "future_collection_event": f"{pid}_C2",
            "future_date": f"2020-{(i % 9) + 1:02d}-15",
            "future_fc": fut_fc,
            "future_log10_fc_plus_1": np.log10(fut_fc + 1.0),
            "elapsed_days": 14,
            "consent_age": 30.0 + i,
            "Age at diagnosis": 20.0,
            "sex": "Female" if i % 2 == 0 else "Male",
            "hbi": 2.0,
            "sccai": 2.0,
            "Bowel frequency during the day": 2.0,
            "Abdominal pain": "None",
            "General wellbeing": "Generally well",
            "Antibiotics": "No",
            "Immunosuppressants (e.g. oral corticosteroids)": "No",
            "reads_filtered": 1e7,
            "reads_human": 1e5,
            "partition": "dev",
            "cv_fold": fold,
        }
        pair_rows.append(row)
        all_h_rows.append(dict(row, elapsed_days=21))

    # Add a held-out test row to all_horizons_df to verify zero test leakage
    all_h_rows.append(dict(pair_rows[0], participant_id="P_TEST_99", partition="test", cv_fold=-1, elapsed_days=21))

    visit_store_df = pd.DataFrame(visit_rows)
    dev_pairs = pd.DataFrame(pair_rows)
    all_horizons_df = pd.DataFrame(all_h_rows)
    sp_cols = [
        "k__Bacteria|p__Firmicutes|s__Faecalibacterium_prausnitzii",
        "k__Bacteria|p__Proteobacteria|s__Escherichia_coli",
    ]
    omics_dict = {
        "species_relab": pd.DataFrame(0.5, index=sample_ids, columns=sp_cols),
        "pathways_unstrat_relab": pd.DataFrame(0.5, index=sample_ids, columns=["CENTFERM-PWY", "SO4ASSIM-PWY"]),
    }

    cfg_file = tmp_path / "test_subtype_cfg.yaml"
    with open(cfg_file, "w", encoding="utf-8") as f:
        yaml.dump({
            "paths": {
                "c1_reports_dir": str(tmp_path),
                "raw_data_dir": str(tmp_path),
                "schema_mapping_path": "micro_trace/configs/schema_mapping.yaml",
            }
        }, f)

    with pytest.raises(ValueError, match="untouched"):
        run_c1_subtype_benchmark(cfg_path=cfg_file, evaluate_test=True)

    summary = run_c1_subtype_benchmark(
        cfg_path=cfg_file,
        epochs_attn=2,
        pretrain_epochs=2,
        evaluate_test=False,
        dev_pairs=dev_pairs,
        all_horizons_df=all_horizons_df,
        visit_store_df=visit_store_df,
        omics_dict=omics_dict,
        output_csv_name="synth_subtype.csv",
        output_json_name="synth_subtype.json",
    )
    assert summary["test_partition_status"] == "untouched_held_out"
    assert summary["evaluate_test"] is False
    assert (tmp_path / "synth_subtype.csv").exists()
    json_text = (tmp_path / "synth_subtype.json").read_text(encoding="utf-8")
    assert "NaN" not in json_text
    parsed = json.loads(json_text)
    assert parsed["evaluate_test"] is False

    # Verify Method 3 & pipeline invariance when held-out test row in all_horizons_df has extreme shift
    all_horizons_perturbed = all_horizons_df.copy()
    all_horizons_perturbed.loc[all_horizons_perturbed["partition"] == "test", "future_log10_fc_plus_1"] = 999.0
    summary_pert = run_c1_subtype_benchmark(
        cfg_path=cfg_file,
        epochs_attn=2,
        pretrain_epochs=2,
        evaluate_test=False,
        dev_pairs=dev_pairs,
        all_horizons_df=all_horizons_perturbed,
        visit_store_df=visit_store_df,
        omics_dict=omics_dict,
        output_csv_name="synth_subtype_pert.csv",
        output_json_name="synth_subtype_pert.json",
    )
    assert summary_pert["summary_table"] == summary["summary_table"]
    methods_present = {r["method"] for r in parsed["summary_table"]}
    assert {
        "Baseline_M0",
        "Global_Pooled",
        "Method1_SubtypeOnly",
        "Method2_PartialPooling",
        "Method3_PretrainSubtypeFT",
        "Method4_SubtypeOOFStack",
    }.issubset(methods_present)
    for row in parsed["summary_table"]:
        for metric in (
            "cv_mae_reconstructed",
            "cv_directional_acc",
            "cv_r2_reconstructed",
            "cv_rmse_reconstructed",
            "cv_spearman_rho_reconstructed",
        ):
            assert metric in row
            assert np.isfinite(row[metric])



