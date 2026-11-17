"""Phase P1 (C1: Future FC Prediction) — Trials 2, 3, 4 & Promising Combinations Benchmark.

Implements and benchmarks under strict 5-fold participant-grouped CV (N=371 dev pairs,
75 participants; 20% test partition N=74 held 100% untouched, evaluate_test=False, seed=42):

1. Combination A (Most Promising): Trial 1 (Delta + L1/SmoothL1) + Trial 2 (M_guild / M_guild_delta)
2. Combination B (High Promise): Trial 1 + Trial 2 + Trial 3 (Leak-free OOF Stacking: MicroTrace-Attn + ElasticNetCV)
3. Trial 4 & Combination C (Exploratory): Trial 4 (Leak-free Multi-Horizon Pre-training on [7, 90]d transitions
   filtered strictly to training-fold participants) + Combinations (T1+T2+T4, T1+T2+T3+T4).
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml

from micro_trace.c1.cohort import DEFAULT_ZERO_SUM_SAMPLES, resolve_existing_path
from micro_trace.c1.features import (
    C1FeaturePipeline,
    build_visit_trajectory_store,
    load_and_align_omics_tables,
)
from micro_trace.c1.models import (
    ElasticNetBenchmark,
    HistGradientBoostingBenchmark,
    MicroTraceAttnTrainer,
    RandomForestBenchmark,
    compute_delta_regression_metrics,
)

logger = logging.getLogger("micro_trace.c1.trials_combinations")


def tune_oof_stack_alpha(
    oof_attn: np.ndarray,
    oof_enet: np.ndarray,
    y_true: np.ndarray,
    alpha_grid: np.ndarray | None = None,
) -> tuple[float, float]:
    """Grid-search blending weight alpha in [0, 1] on inner OOF predictions minimizing MAE."""
    if alpha_grid is None:
        alpha_grid = np.linspace(0.0, 1.0, 21)

    a_arr = np.asarray(oof_attn, dtype=np.float64).ravel()
    e_arr = np.asarray(oof_enet, dtype=np.float64).ravel()
    y_arr = np.asarray(y_true, dtype=np.float64).ravel()

    best_alpha = 0.5
    best_mae = float("inf")
    for alpha in alpha_grid:
        blend = float(alpha) * a_arr + (1.0 - float(alpha)) * e_arr
        mae = float(np.mean(np.abs(blend - y_arr)))
        if mae < best_mae:
            best_mae = mae
            best_alpha = float(alpha)

    return round(best_alpha, 4), round(best_mae, 6)


def filter_multi_horizon_pretrain_df(
    all_horizons_df: pd.DataFrame,
    train_participant_ids: set[str],
    min_days: int = 7,
    max_days: int = 90,
) -> pd.DataFrame:
    """Filter multi-horizon transitions strictly to training-fold participants and [min_days, max_days].

    Guarantees zero validation or test participant leakage.
    """
    if all_horizons_df is None or all_horizons_df.empty:
        return pd.DataFrame()

    df = all_horizons_df.copy()
    if "partition" in df.columns:
        df = df[df["partition"] == "dev"].copy()

    pid_str = df["participant_id"].astype(str)
    mask = pid_str.isin(train_participant_ids)
    if "elapsed_days" in df.columns:
        elapsed = pd.to_numeric(df["elapsed_days"], errors="coerce")
        mask &= (elapsed >= min_days) & (elapsed <= max_days)

    fc_col = "index_log10_fc_same_day" if "index_log10_fc_same_day" in df.columns else "index_log10_fc"
    idx_fc = pd.to_numeric(df[fc_col], errors="coerce")
    fut_fc = pd.to_numeric(df["future_log10_fc_plus_1"], errors="coerce")
    mask &= np.isfinite(idx_fc) & np.isfinite(fut_fc)

    filtered = df[mask].copy()
    leaked = set(filtered["participant_id"].astype(str)) - set(train_participant_ids)
    assert not leaked, f"Leakage detected in multi-horizon pretraining: {leaked}"
    return filtered


def _split_inner_participants(
    train_df: pd.DataFrame,
    seed: int,
    val_frac: float = 0.2,
) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic participant-grouped inner train/val split for early stopping."""
    train_pids = np.unique(train_df["participant_id"].astype(str).values)
    if len(train_pids) < 2:
        mask = np.ones(len(train_df), dtype=bool)
        return mask, ~mask
    rng = np.random.default_rng(seed)
    n_val = max(1, int(len(train_pids) * val_frac))
    val_pids = set(rng.choice(train_pids, size=n_val, replace=False))
    inner_val_mask = train_df["participant_id"].astype(str).isin(val_pids).to_numpy()
    inner_train_mask = ~inner_val_mask
    return inner_train_mask, inner_val_mask


def _get_delta_targets(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (y_level, y_idx, delta_y) arrays."""
    y_level = df["future_log10_fc_plus_1"].to_numpy(dtype=np.float64)
    fc_col = "index_log10_fc_same_day" if "index_log10_fc_same_day" in df.columns else "index_log10_fc"
    y_idx = df[fc_col].to_numpy(dtype=np.float64)
    return y_level, y_idx, y_level - y_idx


def _fit_predict_attn_single(
    arm: str,
    loss_mode: str,
    target_mode: str,
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    visit_store_df: pd.DataFrame,
    omics_dict: dict[str, pd.DataFrame],
    epochs: int,
    seed: int,
    pretrain_df: pd.DataFrame | None = None,
    pretrain_epochs: int = 20,
) -> np.ndarray:
    """Fit MicroTrace-Attn strictly on train_df (with inner participant val) and predict on eval_df."""
    inner_tr_mask, inner_val_mask = _split_inner_participants(train_df, seed=seed)
    has_inner_val = bool(inner_val_mask.any() and inner_tr_mask.any())

    if has_inner_val:
        fit_df = train_df[inner_tr_mask].copy()
        val_df = train_df[inner_val_mask].copy()
    else:
        fit_df = train_df.copy()
        val_df = None

    pipe = C1FeaturePipeline(arm=arm, prevalence_threshold=0.10)
    pipe.fit(fit_df, visit_store_df, omics_dict)

    seq_fit = pipe.build_causal_sequences(fit_df, visit_store_df, omics_dict)
    y_fit_lvl, _, d_fit = _get_delta_targets(fit_df)
    y_fit_target = d_fit if target_mode == "delta" else y_fit_lvl

    seq_val = None
    y_val_target = None
    if val_df is not None:
        seq_val = pipe.build_causal_sequences(val_df, visit_store_df, omics_dict)
        y_val_lvl, _, d_val = _get_delta_targets(val_df)
        y_val_target = d_val if target_mode == "delta" else y_val_lvl

    init_state: dict[str, torch.Tensor] | None = None
    if pretrain_df is not None and not pretrain_df.empty:
        fit_pids = set(fit_df["participant_id"].astype(str))
        # Ensure pretrain_df only uses fit_pids so even inner_val_df remains untouched
        pre_filtered = pretrain_df[pretrain_df["participant_id"].astype(str).isin(fit_pids)].copy()
        if not pre_filtered.empty:
            seq_pre = pipe.build_causal_sequences(pre_filtered, visit_store_df, omics_dict)
            y_pre_lvl, _, d_pre = _get_delta_targets(pre_filtered)
            y_pre_target = d_pre if target_mode == "delta" else y_pre_lvl
            pre_trainer = MicroTraceAttnTrainer(
                embed_dim=48,
                num_heads=4,
                lr=1e-3,
                epochs=pretrain_epochs,
                loss_type=loss_mode,
                target_mode=target_mode,
                random_state=seed,
            )
            pre_trainer.fit(seq_pre, y_pre_target, val_data=seq_val, y_val=y_val_target)
            init_state = pre_trainer.get_state_dict()

    trainer = MicroTraceAttnTrainer(
        embed_dim=48,
        num_heads=4,
        lr=5e-4 if init_state is not None else 1e-3,
        epochs=epochs,
        loss_type=loss_mode,
        target_mode=target_mode,
        random_state=seed,
    )
    trainer.fit(seq_fit, y_fit_target, val_data=seq_val, y_val=y_val_target, init_state_dict=init_state)

    seq_eval = pipe.build_causal_sequences(eval_df, visit_store_df, omics_dict)
    preds = trainer.predict(seq_eval)
    if target_mode == "level":
        _, y_eval_idx, _ = _get_delta_targets(eval_df)
        return preds - y_eval_idx
    return preds


def _aggregate_fold_metrics(
    arm: str,
    model_family: str,
    combination: str,
    feature_dim: int,
    fold_metrics_list: list[dict[str, float]],
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Aggregate 5-fold delta & reconstructed metrics into benchmark record."""
    rec = {
        "arm": arm,
        "model_family": model_family,
        "combination": combination,
        "cv_mae_reconstructed": round(float(np.mean([m["mae_reconstructed"] for m in fold_metrics_list])), 4),
        "cv_mae_delta": round(float(np.mean([m["mae_delta"] for m in fold_metrics_list])), 4),
        "cv_directional_acc": round(float(np.mean([m["directional_accuracy"] for m in fold_metrics_list])), 2),
        "cv_r2_reconstructed": round(float(np.mean([m["r2_reconstructed"] for m in fold_metrics_list])), 4),
        "cv_rmse_reconstructed": round(float(np.mean([m["rmse_reconstructed"] for m in fold_metrics_list])), 4),
        "cv_spearman_rho_reconstructed": round(float(np.mean([m["spearman_rho_reconstructed"] for m in fold_metrics_list])), 4),
        "cv_mae_reconstructed_std": round(float(np.std([m["mae_reconstructed"] for m in fold_metrics_list])), 4),
        "feature_dim": int(feature_dim),
    }
    if extra:
        rec.update(extra)
    return rec


def evaluate_oof_stack_from_cached(
    dev_pairs: pd.DataFrame,
    inner_oof_attn: dict[int, np.ndarray],
    outer_attn: dict[int, np.ndarray],
    inner_oof_enet: dict[int, np.ndarray],
    outer_enet: dict[int, np.ndarray],
    arm_label: str,
    model_label: str,
    combination_label: str,
    feature_dim: int,
) -> dict[str, Any]:
    """Evaluate leak-free OOF stacking ensemble from cached inner-OOF and outer-fold predictions."""
    fold_metrics: list[dict[str, float]] = []
    fold_alphas: list[float] = []

    for fold in range(5):
        train_mask = (dev_pairs["cv_fold"] != fold).to_numpy()
        val_mask = (dev_pairs["cv_fold"] == fold).to_numpy()

        _, _, delta_train = _get_delta_targets(dev_pairs[train_mask])
        y_lvl_val, y_idx_val, delta_val = _get_delta_targets(dev_pairs[val_mask])

        alpha_k, _ = tune_oof_stack_alpha(
            inner_oof_attn[fold],
            inner_oof_enet[fold],
            delta_train,
        )
        fold_alphas.append(alpha_k)

        pred_delta_val = alpha_k * outer_attn[fold] + (1.0 - alpha_k) * outer_enet[fold]
        pred_rec_val = y_idx_val + pred_delta_val

        f_m = compute_delta_regression_metrics(
            y_true_reconstructed=y_lvl_val,
            y_pred_reconstructed=pred_rec_val,
            delta_true=delta_val,
            delta_pred=pred_delta_val,
        )
        fold_metrics.append(f_m)

    return _aggregate_fold_metrics(
        arm=arm_label,
        model_family=model_label,
        combination=combination_label,
        feature_dim=feature_dim,
        fold_metrics_list=fold_metrics,
        extra={
            "mean_tuned_alpha": round(float(np.mean(fold_alphas)), 4),
            "fold_alphas": fold_alphas,
        },
    )


def run_c1_trials_combinations_benchmark(
    cfg_path: str | Path = "micro_trace/configs/c1_config.yaml",
    epochs_attn: int = 30,
    pretrain_epochs: int = 20,
    evaluate_test: bool = False,
    dev_pairs: pd.DataFrame | None = None,
    test_pairs: pd.DataFrame | None = None,
    all_horizons_df: pd.DataFrame | None = None,
    visit_store_df: pd.DataFrame | None = None,
    omics_dict: dict[str, pd.DataFrame] | None = None,
    output_csv_name: str = "c1_trials_combinations_benchmark.csv",
    output_json_name: str = "c1_trials_combinations_summary.json",
) -> dict[str, Any]:
    """Execute Trials 2, 3, 4 and their combinations in promising-to-exploratory order."""
    if evaluate_test:
        raise ValueError("Test partition must remain 100% untouched (evaluate_test=False).")

    cfg_file = Path(cfg_path)
    if not cfg_file.exists():
        cfg_file = resolve_existing_path(
            cfg_file,
            [
                Path("micro_trace/configs/c1_config.yaml"),
                Path("configs/c1_config.yaml"),
            ],
        )
    with open(cfg_file, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    paths_cfg = cfg.get("paths", {})
    reports_dir = Path(paths_cfg.get("c1_reports_dir", "micro_trace/reports/c1"))
    if not reports_dir.exists():
        reports_dir = resolve_existing_path(
            reports_dir,
            [Path("micro_trace/reports/c1"), Path("reports/c1")],
        )
    reports_dir.mkdir(parents=True, exist_ok=True)

    if dev_pairs is None:
        cohort_csv = reports_dir / "c1_cohort_consecutive_0d.csv"
        pairs_df = pd.read_csv(cohort_csv)
        dev_pairs = pairs_df[pairs_df["partition"] == "dev"].copy()
        test_pairs = pairs_df[pairs_df["partition"] == "test"].copy()

    if all_horizons_df is None:
        all_h_csv = reports_dir / "c1_cohort_all_horizons_0d.csv"
        if all_h_csv.exists():
            all_horizons_df = pd.read_csv(all_h_csv)
        else:
            all_horizons_df = pd.DataFrame()

    if omics_dict is None:
        omics_dict = load_and_align_omics_tables(cfg_file)

    if visit_store_df is None:
        processed_dir_cfg = paths_cfg.get("processed_c1_dir")
        processed_dir = resolve_existing_path(
            processed_dir_cfg or "",
            [Path("micro-trace-data/processed/c1"), Path("../micro-trace-data/processed/c1")],
        )
        store_cache = processed_dir / "visit_trajectory_store.csv.gz"
        if store_cache.exists():
            visit_store_df = pd.read_csv(store_cache, compression="gzip")
            visit_store_df["event_date"] = pd.to_datetime(visit_store_df["event_date"])
        else:
            raw_path = resolve_existing_path(
                paths_cfg.get("raw_data_dir", ""),
                [Path("micro-trace-data/raw"), Path("../micro-trace-data/raw")],
            )
            meta_df = pd.read_csv(raw_path / "metadata" / "hmp2_metadata_2018-08-20.csv", low_memory=False)
            schema_file = resolve_existing_path(
                paths_cfg.get("schema_mapping_path", ""),
                [Path("micro_trace/configs/schema_mapping.yaml"), Path("configs/schema_mapping.yaml")],
            )
            with open(schema_file, "r", encoding="utf-8") as f:
                schema_cfg = yaml.safe_load(f)
            excluded_zero = cfg.get("excluded_zero_sum_samples", DEFAULT_ZERO_SUM_SAMPLES)
            visit_store_df = build_visit_trajectory_store(
                meta_df=meta_df,
                omics_dict=omics_dict,
                schema_cfg=schema_cfg,
                excluded_zero_sum_samples=excluded_zero,
            )

    results_records: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    # 0. M0 Persistence Baseline
    # ------------------------------------------------------------------
    m0_metrics: list[dict[str, float]] = []
    for fold in range(5):
        val_mask = (dev_pairs["cv_fold"] == fold).to_numpy()
        y_lvl_val, y_idx_val, d_val = _get_delta_targets(dev_pairs[val_mask])
        d_pred = np.zeros_like(d_val)
        m0_metrics.append(
            compute_delta_regression_metrics(
                y_true_reconstructed=y_lvl_val,
                y_pred_reconstructed=y_idx_val.copy(),
                delta_true=d_val,
                delta_pred=d_pred,
            )
        )
    m0_rec = _aggregate_fold_metrics(
        arm="M0",
        model_family="PersistenceBaseline",
        combination="Baseline",
        feature_dim=2,
        fold_metrics_list=m0_metrics,
    )
    m0_rec["cv_directional_acc"] = 0.0
    results_records.append(m0_rec)

    # ------------------------------------------------------------------
    # 1. Combination A (Most Promising): Trial 1 (Delta + L1/SmoothL1) + Trial 2 (M_guild, M_guild_delta)
    # ------------------------------------------------------------------
    logger.info("=== Combination A: Trial 1 + Trial 2 (M_guild, M_guild_delta) ===")
    arm_dims: dict[str, int] = {}
    # Also cache outer and inner-OOF predictions for M_guild, M1, M5 for Combination B
    outer_preds: dict[tuple[str, str], dict[int, np.ndarray]] = {}
    inner_oof_preds: dict[tuple[str, str], dict[int, np.ndarray]] = {}

    for arm in ("M_guild", "M_guild_delta"):
        combo_label = "ComboA_T1+T2" if arm == "M_guild" else "ComboA_T1+T2_Delta"
        families_a = (
            "ElasticNetCV",
            "RandomForest",
            "HistGradientBoosting",
            "MicroTrace-Attn-SmoothL1",
            "MicroTrace-Attn-L1",
        )
        fold_feats: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
        for fold in range(5):
            train_mask = (dev_pairs["cv_fold"] != fold).to_numpy()
            val_mask = (dev_pairs["cv_fold"] == fold).to_numpy()
            tr_df = dev_pairs[train_mask]
            va_df = dev_pairs[val_mask]
            pipe = C1FeaturePipeline(arm=arm, prevalence_threshold=0.10)
            pipe.fit(tr_df, visit_store_df, omics_dict)
            X_tr, _ = pipe.transform(tr_df, visit_store_df, omics_dict)
            X_va, _ = pipe.transform(va_df, visit_store_df, omics_dict)
            arm_dims[arm] = X_tr.shape[1]
            _, _, d_tr = _get_delta_targets(tr_df)
            y_lvl_va, y_idx_va, d_va = _get_delta_targets(va_df)
            fold_feats[fold] = (X_tr, X_va, d_tr, y_lvl_va, y_idx_va, d_va)

        for fam in families_a:
            f_metrics: list[dict[str, float]] = []
            outer_preds[(arm, fam)] = {}
            for fold in range(5):
                X_tr, X_va, d_tr, y_lvl_va, y_idx_va, d_va = fold_feats[fold]
                tr_df = dev_pairs[dev_pairs["cv_fold"] != fold]
                va_df = dev_pairs[dev_pairs["cv_fold"] == fold]

                if "MicroTrace-Attn" in fam:
                    loss_mode = "l1" if "-L1" in fam else "smooth_l1"
                    pred_delta = _fit_predict_attn_single(
                        arm=arm,
                        loss_mode=loss_mode,
                        target_mode="delta",
                        train_df=tr_df,
                        eval_df=va_df,
                        visit_store_df=visit_store_df,
                        omics_dict=omics_dict,
                        epochs=epochs_attn,
                        seed=42 + fold,
                    )
                elif fam == "ElasticNetCV":
                    m = ElasticNetBenchmark(cv=3, max_iter=5000, random_state=42 + fold).fit(X_tr, d_tr)
                    pred_delta = m.predict(X_va)
                elif fam == "RandomForest":
                    m = RandomForestBenchmark(n_estimators=150, max_depth=8, random_state=42 + fold).fit(X_tr, d_tr)
                    pred_delta = m.predict(X_va)
                elif fam == "HistGradientBoosting":
                    m = HistGradientBoostingBenchmark(max_iter=120, max_depth=5, random_state=42 + fold).fit(X_tr, d_tr)
                    pred_delta = m.predict(X_va)
                else:
                    raise ValueError(f"Unknown family: {fam}")

                outer_preds[(arm, fam)][fold] = pred_delta
                f_metrics.append(
                    compute_delta_regression_metrics(
                        y_true_reconstructed=y_lvl_va,
                        y_pred_reconstructed=y_idx_va + pred_delta,
                        delta_true=d_va,
                        delta_pred=pred_delta,
                    )
                )

            rec = _aggregate_fold_metrics(
                arm=arm,
                model_family=fam,
                combination=combo_label,
                feature_dim=arm_dims[arm],
                fold_metrics_list=f_metrics,
            )
            results_records.append(rec)
            logger.info(
                f"  [{combo_label}] {arm} | {fam}: MAE={rec['cv_mae_reconstructed']:.4f} | "
                f"DirAcc={rec['cv_directional_acc']:.1f}% | R2={rec['cv_r2_reconstructed']:.4f}"
            )

    # ------------------------------------------------------------------
    # 2. Combination B (High Promise): Trial 1 + Trial 2 + Trial 3 (Leak-Free OOF Stacking)
    # ------------------------------------------------------------------
    logger.info("=== Combination B: Leak-Free OOF Stacking (MicroTrace-Attn + ElasticNetCV) ===")

    # Helper to populate outer + inner OOF predictions for any (arm, family) needed in stacking
    def ensure_cached_oof(arm: str, fam: str) -> None:
        key = (arm, fam)
        if key not in outer_preds:
            outer_preds[key] = {}
            f_metrics: list[dict[str, float]] = []
            for fold in range(5):
                tr_df = dev_pairs[dev_pairs["cv_fold"] != fold]
                va_df = dev_pairs[dev_pairs["cv_fold"] == fold]
                y_lvl_va, y_idx_va, d_va = _get_delta_targets(va_df)
                if "MicroTrace-Attn" in fam:
                    loss_mode = "l1" if "-L1" in fam else "smooth_l1"
                    pred_delta = _fit_predict_attn_single(
                        arm=arm,
                        loss_mode=loss_mode,
                        target_mode="delta",
                        train_df=tr_df,
                        eval_df=va_df,
                        visit_store_df=visit_store_df,
                        omics_dict=omics_dict,
                        epochs=epochs_attn,
                        seed=42 + fold,
                    )
                    if arm not in arm_dims:
                        p_tmp = C1FeaturePipeline(arm=arm, prevalence_threshold=0.10).fit(tr_df, visit_store_df, omics_dict)
                        arm_dims[arm] = len(p_tmp.feature_names_)
                elif fam == "ElasticNetCV":
                    pipe = C1FeaturePipeline(arm=arm, prevalence_threshold=0.10).fit(tr_df, visit_store_df, omics_dict)
                    X_tr, _ = pipe.transform(tr_df, visit_store_df, omics_dict)
                    X_va, _ = pipe.transform(va_df, visit_store_df, omics_dict)
                    arm_dims[arm] = X_tr.shape[1]
                    _, _, d_tr = _get_delta_targets(tr_df)
                    m = ElasticNetBenchmark(cv=3, max_iter=5000, random_state=42 + fold).fit(X_tr, d_tr)
                    pred_delta = m.predict(X_va)
                else:
                    raise ValueError(f"Unsupported stack component: {fam}")

                outer_preds[key][fold] = pred_delta
                f_metrics.append(
                    compute_delta_regression_metrics(
                        y_true_reconstructed=y_lvl_va,
                        y_pred_reconstructed=y_idx_va + pred_delta,
                        delta_true=d_va,
                        delta_pred=pred_delta,
                    )
                )
            results_records.append(
                _aggregate_fold_metrics(
                    arm=arm,
                    model_family=fam,
                    combination="Trial1_Reference",
                    feature_dim=arm_dims[arm],
                    fold_metrics_list=f_metrics,
                )
            )

        if key not in inner_oof_preds:
            inner_oof_preds[key] = {}
            for fold in range(5):
                tr_df = dev_pairs[dev_pairs["cv_fold"] != fold].reset_index(drop=True)
                oof_vec = np.zeros(len(tr_df), dtype=np.float64)
                inner_folds = sorted(tr_df["cv_fold"].unique())
                for ifold in inner_folds:
                    itr_mask = (tr_df["cv_fold"] != ifold).to_numpy()
                    iva_mask = (tr_df["cv_fold"] == ifold).to_numpy()
                    itr_df = tr_df[itr_mask]
                    iva_df = tr_df[iva_mask]
                    if "MicroTrace-Attn" in fam:
                        loss_mode = "l1" if "-L1" in fam else "smooth_l1"
                        oof_vec[iva_mask] = _fit_predict_attn_single(
                            arm=arm,
                            loss_mode=loss_mode,
                            target_mode="delta",
                            train_df=itr_df,
                            eval_df=iva_df,
                            visit_store_df=visit_store_df,
                            omics_dict=omics_dict,
                            epochs=epochs_attn,
                            seed=42 + fold * 10 + int(ifold),
                        )
                    elif fam == "ElasticNetCV":
                        pipe = C1FeaturePipeline(arm=arm, prevalence_threshold=0.10).fit(itr_df, visit_store_df, omics_dict)
                        X_itr, _ = pipe.transform(itr_df, visit_store_df, omics_dict)
                        X_iva, _ = pipe.transform(iva_df, visit_store_df, omics_dict)
                        _, _, d_itr = _get_delta_targets(itr_df)
                        m = ElasticNetBenchmark(cv=3, max_iter=5000, random_state=42 + fold * 10 + int(ifold)).fit(X_itr, d_itr)
                        oof_vec[iva_mask] = m.predict(X_iva)
                inner_oof_preds[key][fold] = oof_vec

    stack_specs = [
        ("M_guild", "MicroTrace-Attn-L1", "M_guild", "ElasticNetCV", "M_guild", "OOFStack(M_guild_AttnL1+M_guild_ENet)", "ComboB_T1+T2+T3"),
        ("M_guild", "MicroTrace-Attn-SmoothL1", "M_guild", "ElasticNetCV", "M_guild", "OOFStack(M_guild_AttnSmoothL1+M_guild_ENet)", "ComboB_T1+T2+T3"),
        ("M_guild", "MicroTrace-Attn-L1", "M_guild_delta", "ElasticNetCV", "M_guild+M_guild_delta", "OOFStack(M_guild_AttnL1+M_guild_delta_ENet)", "ComboB_T1+T2+T3"),
        ("M1", "MicroTrace-Attn-L1", "M1", "ElasticNetCV", "M1", "OOFStack(M1_AttnL1+M1_ENet)", "ComboB_T1+T3"),
        ("M5", "MicroTrace-Attn-L1", "M5", "ElasticNetCV", "M5", "OOFStack(M5_AttnL1+M5_ENet)", "ComboB_T1+T3"),
        ("M1", "MicroTrace-Attn-L1", "M5", "ElasticNetCV", "M1+M5", "OOFStack(M1_AttnL1+M5_ENet)", "ComboB_CrossArm_T1+T3"),
        ("M_guild", "MicroTrace-Attn-L1", "M5", "ElasticNetCV", "M_guild+M5", "OOFStack(M_guild_AttnL1+M5_ENet)", "ComboB_CrossArm_T1+T2+T3"),
        ("M1", "MicroTrace-Attn-L1", "M_guild", "ElasticNetCV", "M1+M_guild", "OOFStack(M1_AttnL1+M_guild_ENet)", "ComboB_CrossArm_T1+T2+T3"),
    ]

    for attn_arm, attn_fam, enet_arm, enet_fam, arm_lbl, mod_lbl, combo_lbl in stack_specs:
        ensure_cached_oof(attn_arm, attn_fam)
        ensure_cached_oof(enet_arm, enet_fam)
        fdim = arm_dims[attn_arm] if attn_arm == enet_arm else arm_dims[attn_arm] + arm_dims[enet_arm]
        rec = evaluate_oof_stack_from_cached(
            dev_pairs=dev_pairs,
            inner_oof_attn=inner_oof_preds[(attn_arm, attn_fam)],
            outer_attn=outer_preds[(attn_arm, attn_fam)],
            inner_oof_enet=inner_oof_preds[(enet_arm, enet_fam)],
            outer_enet=outer_preds[(enet_arm, enet_fam)],
            arm_label=arm_lbl,
            model_label=mod_lbl,
            combination_label=combo_lbl,
            feature_dim=fdim,
        )
        results_records.append(rec)
        logger.info(
            f"  [{combo_lbl}] {mod_lbl}: MAE={rec['cv_mae_reconstructed']:.4f} | "
            f"DirAcc={rec['cv_directional_acc']:.1f}% | R2={rec['cv_r2_reconstructed']:.4f} | alpha={rec['mean_tuned_alpha']:.2f}"
        )

    # ------------------------------------------------------------------
    # 3. Trial 4 & Combination C (Exploratory): Multi-Horizon Pre-training
    # ------------------------------------------------------------------
    logger.info("=== Trial 4 & Combination C: Leak-Free Multi-Horizon Pre-training ===")
    if not all_horizons_df.empty:
        pretrain_specs = [
            ("M1", "smooth_l1", "level", "MicroTrace-Attn-Pretrain-Level", "Trial4_Pretrain_Level"),
            ("M1", "l1", "delta", "MicroTrace-Attn-Pretrain-L1", "Trial4_Pretrain_Delta"),
            ("M_guild", "l1", "delta", "MicroTrace-Attn-Pretrain-L1", "ComboC_T1+T2+T4"),
            ("M_guild", "smooth_l1", "delta", "MicroTrace-Attn-Pretrain-SmoothL1", "ComboC_T1+T2+T4"),
        ]

        for arm, loss_mode, target_mode, mod_lbl, combo_lbl in pretrain_specs:
            key = (arm, mod_lbl)
            outer_preds[key] = {}
            f_metrics: list[dict[str, float]] = []

            for fold in range(5):
                tr_df = dev_pairs[dev_pairs["cv_fold"] != fold]
                va_df = dev_pairs[dev_pairs["cv_fold"] == fold]
                train_pids = set(tr_df["participant_id"].astype(str))
                pre_df = filter_multi_horizon_pretrain_df(all_horizons_df, train_pids, min_days=7, max_days=90)

                pred_delta = _fit_predict_attn_single(
                    arm=arm,
                    loss_mode=loss_mode,
                    target_mode=target_mode,
                    train_df=tr_df,
                    eval_df=va_df,
                    visit_store_df=visit_store_df,
                    omics_dict=omics_dict,
                    epochs=epochs_attn,
                    seed=42 + fold,
                    pretrain_df=pre_df,
                    pretrain_epochs=pretrain_epochs,
                )
                outer_preds[key][fold] = pred_delta
                y_lvl_va, y_idx_va, d_va = _get_delta_targets(va_df)
                f_metrics.append(
                    compute_delta_regression_metrics(
                        y_true_reconstructed=y_lvl_va,
                        y_pred_reconstructed=y_idx_va + pred_delta,
                        delta_true=d_va,
                        delta_pred=pred_delta,
                    )
                )

            rec = _aggregate_fold_metrics(
                arm=arm,
                model_family=mod_lbl,
                combination=combo_lbl,
                feature_dim=arm_dims[arm],
                fold_metrics_list=f_metrics,
            )
            results_records.append(rec)
            logger.info(
                f"  [{combo_lbl}] {arm} | {mod_lbl}: MAE={rec['cv_mae_reconstructed']:.4f} | "
                f"DirAcc={rec['cv_directional_acc']:.1f}% | R2={rec['cv_r2_reconstructed']:.4f}"
            )

        # Combination C Full Stack: T1 + T2 + T3 + T4 (Pretrained M_guild Attn-L1 + ElasticNetCV)
        pre_key = ("M_guild", "MicroTrace-Attn-Pretrain-L1")
        inner_oof_preds[pre_key] = {}
        for fold in range(5):
            tr_df = dev_pairs[dev_pairs["cv_fold"] != fold].reset_index(drop=True)
            oof_vec = np.zeros(len(tr_df), dtype=np.float64)
            inner_folds = sorted(tr_df["cv_fold"].unique())
            for ifold in inner_folds:
                itr_mask = (tr_df["cv_fold"] != ifold).to_numpy()
                iva_mask = (tr_df["cv_fold"] == ifold).to_numpy()
                itr_df = tr_df[itr_mask]
                iva_df = tr_df[iva_mask]
                itr_pids = set(itr_df["participant_id"].astype(str))
                pre_idf = filter_multi_horizon_pretrain_df(all_horizons_df, itr_pids, min_days=7, max_days=90)
                oof_vec[iva_mask] = _fit_predict_attn_single(
                    arm="M_guild",
                    loss_mode="l1",
                    target_mode="delta",
                    train_df=itr_df,
                    eval_df=iva_df,
                    visit_store_df=visit_store_df,
                    omics_dict=omics_dict,
                    epochs=epochs_attn,
                    seed=42 + fold * 10 + int(ifold),
                    pretrain_df=pre_idf,
                    pretrain_epochs=pretrain_epochs,
                )
            inner_oof_preds[pre_key][fold] = oof_vec

        for enet_arm, arm_lbl in [
            ("M_guild", "M_guild"),
            ("M_guild_delta", "M_guild+M_guild_delta"),
            ("M5", "M_guild+M5"),
        ]:
            fdim = arm_dims["M_guild"] if enet_arm == "M_guild" else (arm_dims["M_guild_delta"] if enet_arm == "M_guild_delta" else arm_dims["M_guild"] + arm_dims["M5"])
            rec_full = evaluate_oof_stack_from_cached(
                dev_pairs=dev_pairs,
                inner_oof_attn=inner_oof_preds[pre_key],
                outer_attn=outer_preds[pre_key],
                inner_oof_enet=inner_oof_preds[(enet_arm, "ElasticNetCV")],
                outer_enet=outer_preds[(enet_arm, "ElasticNetCV")],
                arm_label=arm_lbl,
                model_label=f"OOFStack(M_guild_PretrainAttnL1+{enet_arm}_ENet)",
                combination_label="ComboC_FullStack_T1+T2+T3+T4",
                feature_dim=fdim,
            )
            results_records.append(rec_full)
            logger.info(
                f"  [ComboC_FullStack_T1+T2+T3+T4] {rec_full['model_family']}: "
                f"MAE={rec_full['cv_mae_reconstructed']:.4f} | DirAcc={rec_full['cv_directional_acc']:.1f}% | "
                f"R2={rec_full['cv_r2_reconstructed']:.4f} | alpha={rec_full['mean_tuned_alpha']:.2f}"
            )

    # ------------------------------------------------------------------
    # 4. Rank by cv_mae_reconstructed and export CSV + JSON
    # ------------------------------------------------------------------
    results_df = pd.DataFrame(results_records)
    results_df = results_df.sort_values(
        by=["cv_mae_reconstructed", "cv_r2_reconstructed"],
        ascending=[True, False],
    ).reset_index(drop=True)

    cols_order = [
        "arm",
        "model_family",
        "combination",
        "cv_mae_reconstructed",
        "cv_mae_delta",
        "cv_directional_acc",
        "cv_r2_reconstructed",
        "cv_rmse_reconstructed",
        "cv_spearman_rho_reconstructed",
        "cv_mae_reconstructed_std",
        "feature_dim",
    ]
    extra_cols = [c for c in results_df.columns if c not in cols_order]
    results_df = results_df[cols_order + extra_cols]

    out_csv = reports_dir / output_csv_name
    results_df.to_csv(out_csv, index=False)

    summary_records = results_df.astype(object).where(pd.notna(results_df), None).to_dict(orient="records")
    summary_dict = {
        "benchmark_timestamp": datetime.now(timezone.utc).isoformat(),
        "seed": 42,
        "cv_folds": 5,
        "dev_pairs_count": int(len(dev_pairs)),
        "dev_participants_count": int(dev_pairs["participant_id"].nunique()),
        "test_pairs_count": int(len(test_pairs)) if test_pairs is not None else 0,
        "test_partition_status": "untouched_held_out",
        "arm_dimensions": arm_dims,
        "best_cv_model": results_df.iloc[0].dropna().to_dict(),
        "summary_table": summary_records,
    }

    out_json = reports_dir / output_json_name
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary_dict, f, indent=2, default=str, allow_nan=False)

    return summary_dict


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    parser = argparse.ArgumentParser(description="Run C1 Trials 2, 3, 4 & Combinations Benchmark")
    parser.add_argument("--config", type=str, default="micro_trace/configs/c1_config.yaml")
    parser.add_argument("--epochs-attn", type=int, default=30)
    parser.add_argument("--pretrain-epochs", type=int, default=20)
    args = parser.parse_args()

    t0 = time.time()
    summary = run_c1_trials_combinations_benchmark(
        cfg_path=args.config,
        epochs_attn=args.epochs_attn,
        pretrain_epochs=args.pretrain_epochs,
        evaluate_test=False,
    )
    elapsed = time.time() - t0
    logger.info(f"Completed all trials & combinations in {elapsed:.1f}s")
    logger.info(f"Best CV model: {summary['best_cv_model']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
