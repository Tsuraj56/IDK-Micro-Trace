import sys
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
import torch

from micro_trace.c1.models import (
    MicroTraceAttnTrainer,
    compute_delta_regression_metrics,
)
from micro_trace.c1.features import (
    C1FeaturePipeline,
    build_visit_trajectory_store,
    load_and_align_omics_tables,
    DEFAULT_ZERO_SUM_SAMPLES,
)

def test_inner_split():
    cfg_file = Path("micro_trace/configs/c1_config.yaml")
    with open(cfg_file, "r") as f:
        cfg = yaml.safe_load(f)
    paths_cfg = cfg.get("paths", {})
    reports_dir = Path(paths_cfg.get("c1_reports_dir", "micro_trace/reports/c1"))
    cohort_csv = reports_dir / "c1_cohort_consecutive_0d.csv"
    pairs_df = pd.read_csv(cohort_csv)
    dev_pairs = pairs_df[pairs_df["partition"] == "dev"].copy()

    omics_dict = load_and_align_omics_tables(cfg_file)
    raw_path = Path(paths_cfg.get("raw_data_dir", "DataSet/hmp2"))
    meta_file = raw_path / "metadata" / "hmp2_metadata_2018-08-20.csv"
    meta_df = pd.read_csv(meta_file, low_memory=False)
    schema_file = Path(paths_cfg.get("schema_mapping_path", "micro_trace/configs/schema_mapping.yaml"))
    with open(schema_file, "r") as f:
        schema_cfg = yaml.safe_load(f)

    visit_store_df = build_visit_trajectory_store(
        meta_df=meta_df,
        omics_dict=omics_dict,
        schema_cfg=schema_cfg,
        excluded_zero_sum_samples=DEFAULT_ZERO_SUM_SAMPLES,
    )

    y_dev_level = dev_pairs["future_log10_fc_plus_1"].to_numpy(dtype=np.float64)
    y_dev_idx = dev_pairs["index_log10_fc_same_day"].to_numpy(dtype=np.float64)
    delta_dev = y_dev_level - y_dev_idx

    # Let's test with inner split (hold out 12 of 60 training participants for early stopping)
    fold_maes_inner = []
    fold_dir_accs_inner = []

    for fold in range(5):
        train_mask = (dev_pairs["cv_fold"] != fold).to_numpy()
        val_mask = (dev_pairs["cv_fold"] == fold).to_numpy()
        train_df = dev_pairs[train_mask].copy()

        # Grouped split on train_df participants
        train_pids = np.unique(train_df["participant_id"].values)
        rng = np.random.default_rng(42 + fold)
        val_pids = set(rng.choice(train_pids, size=int(len(train_pids) * 0.2), replace=False))
        
        inner_train_mask = ~train_df["participant_id"].isin(val_pids).to_numpy()
        inner_val_mask = train_df["participant_id"].isin(val_pids).to_numpy()

        fold_pipe = C1FeaturePipeline(arm="M1", prevalence_threshold=0.10)
        # fit on inner train
        fold_pipe.fit(train_df[inner_train_mask], visit_store_df, omics_dict)

        seq_inner_train = fold_pipe.build_causal_sequences(train_df[inner_train_mask], visit_store_df, omics_dict)
        seq_inner_val = fold_pipe.build_causal_sequences(train_df[inner_val_mask], visit_store_df, omics_dict)
        seq_outer_val = fold_pipe.build_causal_sequences(dev_pairs[val_mask], visit_store_df, omics_dict)

        delta_inner_train = delta_dev[train_mask][inner_train_mask]
        delta_inner_val = delta_dev[train_mask][inner_val_mask]
        delta_outer_val = delta_dev[val_mask]
        y_level_outer_val = y_dev_level[val_mask]
        y_idx_outer_val = y_dev_idx[val_mask]

        trainer = MicroTraceAttnTrainer(
            embed_dim=48,
            num_heads=4,
            epochs=35,
            loss_type="l1",
            target_mode="delta",
            random_state=42 + fold,
        )
        trainer.fit(seq_inner_train, delta_inner_train, val_data=seq_inner_val, y_val=delta_inner_val)
        pred_delta = trainer.predict(seq_outer_val)
        pred_rec = y_idx_outer_val + pred_delta

        m = compute_delta_regression_metrics(
            y_true_reconstructed=y_level_outer_val,
            y_pred_reconstructed=pred_rec,
            delta_true=delta_outer_val,
            delta_pred=pred_delta,
        )
        fold_maes_inner.append(m["mae_reconstructed"])
        fold_dir_accs_inner.append(m["directional_accuracy"])
        print(f"Fold {fold}: Inner Val MAE={m['mae_reconstructed']:.4f}, DirAcc={m['directional_accuracy']:.1f}%")

    print(f"INNER SPLIT Mean MAE: {np.mean(fold_maes_inner):.4f} +/- {np.std(fold_maes_inner):.4f}")
    print(f"INNER SPLIT Mean DirAcc: {np.mean(fold_dir_accs_inner):.2f}%")

if __name__ == "__main__":
    test_inner_split()
