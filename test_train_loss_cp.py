import sys
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from micro_trace.c1.models import (
    MicroTraceAttn,
    compute_delta_regression_metrics,
)
from micro_trace.c1.features import (
    C1FeaturePipeline,
    build_visit_trajectory_store,
    load_and_align_omics_tables,
    DEFAULT_ZERO_SUM_SAMPLES,
)

def test_train_loss_checkpoint():
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

    fold_maes = []
    fold_dir_accs = []

    for fold in range(5):
        train_mask = (dev_pairs["cv_fold"] != fold).to_numpy()
        val_mask = (dev_pairs["cv_fold"] == fold).to_numpy()

        fold_pipe = C1FeaturePipeline(arm="M1", prevalence_threshold=0.10)
        fold_pipe.fit(dev_pairs[train_mask], visit_store_df, omics_dict)

        seq_f_train = fold_pipe.build_causal_sequences(dev_pairs[train_mask], visit_store_df, omics_dict)
        seq_f_val = fold_pipe.build_causal_sequences(dev_pairs[val_mask], visit_store_df, omics_dict)

        delta_train = delta_dev[train_mask]
        delta_val = delta_dev[val_mask]
        y_level_val = y_dev_level[val_mask]
        y_idx_val = y_dev_idx[val_mask]

        torch.manual_seed(42 + fold)
        np.random.seed(42 + fold)
        d_seq = seq_f_train["seq_features"].shape[2]
        d_anchor = seq_f_train["static_anchor"].shape[1]
        d_omics = seq_f_train["current_omics"].shape[1]

        model = MicroTraceAttn(
            d_seq=d_seq,
            d_anchor=d_anchor,
            d_omics=d_omics,
            embed_dim=48,
            num_heads=4,
            target_mode="delta",
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30, eta_min=1e-5)
        loss_fn = nn.L1Loss()

        t_seq = torch.from_numpy(seq_f_train["seq_features"]).float()
        t_dt = torch.from_numpy(seq_f_train["seq_time_deltas"]).float()
        t_mask = torch.from_numpy(seq_f_train["seq_mask"]).bool()
        t_anc = torch.from_numpy(seq_f_train["static_anchor"]).float()
        t_om = torch.from_numpy(seq_f_train["current_omics"]).float()
        t_y = torch.from_numpy(delta_train).float()

        dataset = TensorDataset(t_seq, t_dt, t_mask, t_anc, t_om, t_y)
        loader = DataLoader(dataset, batch_size=32, shuffle=True)

        best_loss = float("inf")
        best_state = None

        for epoch in range(1, 31):
            model.train()
            train_loss = 0.0
            n_b = 0
            for b_seq, b_dt, b_mask, b_anc, b_om, b_y in loader:
                optimizer.zero_grad()
                pred = model(b_seq, b_dt, b_mask, b_anc, b_om)
                loss = loss_fn(pred, b_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                train_loss += float(loss.item())
                n_b += 1
            scheduler.step()
            train_loss /= max(n_b, 1)
            if train_loss < best_loss:
                best_loss = train_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if best_state is not None:
            model.load_state_dict(best_state)

        model.eval()
        with torch.no_grad():
            v_seq = torch.from_numpy(seq_f_val["seq_features"]).float()
            v_dt = torch.from_numpy(seq_f_val["seq_time_deltas"]).float()
            v_mask = torch.from_numpy(seq_f_val["seq_mask"]).bool()
            v_anc = torch.from_numpy(seq_f_val["static_anchor"]).float()
            v_om = torch.from_numpy(seq_f_val["current_omics"]).float()
            pred_delta = model(v_seq, v_dt, v_mask, v_anc, v_om).numpy()

        pred_rec = y_idx_val + pred_delta
        m = compute_delta_regression_metrics(
            y_true_reconstructed=y_level_val,
            y_pred_reconstructed=pred_rec,
            delta_true=delta_val,
            delta_pred=pred_delta,
        )
        fold_maes.append(m["mae_reconstructed"])
        fold_dir_accs.append(m["directional_accuracy"])
        print(f"Fold {fold}: MAE={m['mae_reconstructed']:.4f}, DirAcc={m['directional_accuracy']:.1f}%")

    print(f"TRAIN LOSS CHECKPOINT Mean MAE: {np.mean(fold_maes):.4f} +/- {np.std(fold_maes):.4f}")
    print(f"TRAIN LOSS CHECKPOINT Mean DirAcc: {np.mean(fold_dir_accs):.2f}%")

if __name__ == "__main__":
    test_train_loss_checkpoint()
