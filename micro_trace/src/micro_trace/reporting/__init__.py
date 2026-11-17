"""Visualizations and comprehensive Markdown report generation."""
import os
import json
import datetime
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def generate_figures(
    fig_dir: str,
    inv_df: pd.DataFrame,
    part_df: pd.DataFrame,
    fc_df: pd.DataFrame,
    meta_df: pd.DataFrame,
    match_df: pd.DataFrame,
    gap_df: pd.DataFrame,
    pairs_df: pd.DataFrame,
    tax_df: pd.DataFrame,
    func_df: pd.DataFrame,
    flow_df: pd.DataFrame,
    is_synthetic: bool = False,
) -> list[str]:
    """Generate the 8 mandatory visual outputs from inspected data."""
    os.makedirs(fig_dir, exist_ok=True)
    tag = " [SYNTHETIC DEMO]" if is_synthetic else " (Actual HMP2 / IBDMDB Data)"
    saved = []

    # 1. Data inventory overview
    fig, ax = plt.subplots(figsize=(8, 4.5))
    layer_mb = inv_df.groupby("layer")["size_mb"].sum().sort_values(ascending=True)
    layer_cnt = inv_df.groupby("layer")["file_name"].count().reindex(layer_mb.index)
    bars = ax.barh(layer_mb.index.astype(str), layer_mb.values, color="#2b5c8f")
    ax.set_xlabel("Total Storage Size (MB)")
    ax.set_title(f"1. Data Inventory Overview by Omics Layer{tag}")
    for bar, cnt in zip(bars, layer_cnt.values):
        ax.text(bar.get_width() + max(layer_mb.max() * 0.01, 0.5), bar.get_y() + bar.get_height()/2,
                f"{bar.get_width():.1f} MB ({cnt} files)", va="center", fontsize=8)
    ax.set_xlim(0, max(layer_mb.max() * 1.28, 10))
    plt.tight_layout()
    p1 = os.path.join(fig_dir, "01_data_inventory_overview.png")
    plt.savefig(p1, dpi=160)
    plt.close(fig)
    saved.append(p1)

    # 2. Participant and visit distribution
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    diag_cnt = part_df["diagnosis"].value_counts()
    axes[0].bar(diag_cnt.index.astype(str), diag_cnt.values, color=["#c44e52", "#dd8452", "#4c72b0"][:len(diag_cnt)])
    axes[0].set_title(f"Participants by Subtype (N={len(part_df)}){tag}")
    axes[0].set_ylabel("Unique Participants")
    for idx, val in enumerate(diag_cnt.values):
        axes[0].text(idx, val + 0.5, str(val), ha="center", fontsize=9, fontweight="bold")
    axes[1].hist(part_df["n_mgx_collection_events"], bins=15, color="#4c72b0", edgecolor="black", alpha=0.85, label="MGX visits")
    axes[1].hist(part_df["n_mgx_and_fc_events"], bins=15, color="#55a868", edgecolor="black", alpha=0.75, label="MGX + FC visits")
    axes[1].set_title("Visits per Participant Distribution")
    axes[1].set_xlabel("Collection Events per Participant")
    axes[1].set_ylabel("Number of Participants")
    axes[1].legend()
    plt.tight_layout()
    p2 = os.path.join(fig_dir, "02_participant_visit_distribution.png")
    plt.savefig(p2, dpi=160)
    plt.close(fig)
    saved.append(p2)

    # 3. FC distribution and missingness
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    fc_s = pd.to_numeric(meta_df.get("fecalcal", pd.Series(dtype=float)), errors="coerce").combine_first(
        pd.to_numeric(meta_df.get("fecalcal_ng_ml", pd.Series(dtype=float)), errors="coerce")
    ).dropna()
    if len(fc_s) > 0:
        axes[0].hist(fc_s, bins=25, color="#c44e52", edgecolor="black", alpha=0.85)
        axes[0].set_title(f"Raw FC Distribution (n={len(fc_s)} rows){tag}")
        axes[0].set_xlabel("Fecal Calprotectin (ug/g scale)")
        axes[0].set_ylabel("Count")
        axes[1].hist(np.log10(fc_s + 1.0), bins=25, color="#8172b2", edgecolor="black", alpha=0.85)
        axes[1].set_title("Transformed Target: log10(FC + 1)")
        axes[1].set_xlabel("log10(FC + 1)")
        axes[1].set_ylabel("Count")
    plt.tight_layout()
    p3 = os.path.join(fig_dir, "03_fc_distribution_and_missingness.png")
    plt.savefig(p3, dpi=160)
    plt.close(fig)
    saved.append(p3)

    # 4. Microbiome and FC matching coverage across windows
    fig, ax = plt.subplots(figsize=(8, 4.2))
    win_cols = [c for c in match_df.columns if c.startswith("matched_causal_within_")]
    if win_cols:
        labels = [c.replace("matched_causal_within_", "<= ") for c in win_cols]
        counts = [int(match_df[c].sum()) for c in win_cols]
        pids = [int(match_df.loc[match_df[c], "participant_id"].nunique()) for c in win_cols]
        x = np.arange(len(labels))
        w = 0.36
        b1 = ax.bar(x - w/2, counts, width=w, color="#4c72b0", label="Matched MGX Events")
        b2 = ax.bar(x + w/2, pids, width=w, color="#55a868", label="Unique Participants")
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel("Count")
        ax.set_title(f"4. Causal Index FC Matching Coverage Across Windows (Total MGX Events={len(match_df)}){tag}")
        ax.legend()
        for b in list(b1) + list(b2):
            ax.text(b.get_x() + b.get_width()/2, b.get_height() + 2, str(int(b.get_height())), ha="center", fontsize=8)
    plt.tight_layout()
    p4 = os.path.join(fig_dir, "04_microbiome_fc_matching_coverage.png")
    plt.savefig(p4, dpi=160)
    plt.close(fig)
    saved.append(p4)

    # 5. Visit-gap distribution
    fig, ax = plt.subplots(figsize=(8, 4.2))
    p_next = pairs_df[pairs_df["has_same_day_index_fc"] & pairs_df["is_next_available_fc_from_index"]] if not pairs_df.empty else pd.DataFrame()
    if not p_next.empty:
        ax.hist(p_next["elapsed_days"], bins=30, color="#2b5c8f", edgecolor="black", alpha=0.85)
        ax.axvline(float(p_next["elapsed_days"].median()), color="red", linestyle="--", label=f"Median={p_next['elapsed_days'].median():.0f}d")
        ax.set_title(f"5. Follow-up Interval Distribution (Index MGX+FC to Next FC, n={len(p_next)}){tag}")
        ax.set_xlabel("Elapsed Days (t_future - t_index)")
        ax.set_ylabel("Number of Consecutive Pairs")
        ax.legend()
    plt.tight_layout()
    p5 = os.path.join(fig_dir, "05_visit_gap_distribution.png")
    plt.savefig(p5, dpi=160)
    plt.close(fig)
    saved.append(p5)

    # 6. Eligible future-prediction pair counts by candidate horizon
    fig, ax = plt.subplots(figsize=(9, 4.5))
    p_same = pairs_df[pairs_df["has_same_day_index_fc"]] if not pairs_df.empty else pd.DataFrame()
    h_cols = [c for c in p_same.columns if c.startswith("in_")]
    if h_cols and not p_same.empty:
        h_labels = [c.replace("in_", "") for c in h_cols]
        pair_cnts = [int(p_same[c].sum()) for c in h_cols]
        part_cnts = [int(p_same.loc[p_same[c], "participant_id"].nunique()) for c in h_cols]
        ibd_part_cnts = [int(p_same.loc[p_same[c] & p_same["is_ibd"], "participant_id"].nunique()) for c in h_cols]
        x = np.arange(len(h_labels))
        w = 0.26
        b1 = ax.bar(x - w, pair_cnts, width=w, color="#4c72b0", label="Eligible Pairs")
        b2 = ax.bar(x, part_cnts, width=w, color="#55a868", label="All Unique Participants")
        b3 = ax.bar(x + w, ibd_part_cnts, width=w, color="#c44e52", label="IBD Participants (CD/UC)")
        ax.set_xticks(x)
        ax.set_xticklabels(h_labels, rotation=15, ha="right")
        ax.set_ylabel("Count")
        ax.set_title(f"6. Eligible Prediction Pairs & Unique Participants by Candidate Horizon{tag}")
        ax.legend()
        for b in list(b1) + list(b2) + list(b3):
            ax.text(b.get_x() + b.get_width()/2, b.get_height() + 2, str(int(b.get_height())), ha="center", fontsize=7.5)
    plt.tight_layout()
    p6 = os.path.join(fig_dir, "06_pairs_by_candidate_horizon.png")
    plt.savefig(p6, dpi=160)
    plt.close(fig)
    saved.append(p6)

    # 7. Taxonomy and functional feature availability
    fig, ax = plt.subplots(figsize=(9, 4.5))
    comb_tf = pd.concat([tax_df, func_df], ignore_index=True) if (not tax_df.empty or not func_df.empty) else pd.DataFrame()
    if not comb_tf.empty:
        x = np.arange(len(comb_tf))
        w = 0.35
        b1 = ax.bar(x - w/2, comb_tf["n_samples_in_table"], width=w, color="#4c72b0", label="Samples in Table")
        b2 = ax.bar(x + w/2, comb_tf["features_prev_ge_10pct"], width=w, color="#dd8452", label="Features >=10% Prevalence (in top 5k)")
        ax.set_xticks(x)
        ax.set_xticklabels(comb_tf["table_name"], rotation=25, ha="right", fontsize=8)
        ax.set_ylabel("Count")
        ax.set_title(f"7. Taxonomic & Functional Table Coverage and Prevalent Features{tag}")
        ax.legend()
    plt.tight_layout()
    p7 = os.path.join(fig_dir, "07_taxonomy_functional_availability.png")
    plt.savefig(p7, dpi=160)
    plt.close(fig)
    saved.append(p7)

    # 8. Eligibility flow diagram
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    if not flow_df.empty:
        sub_fl = flow_df.iloc[:10].iloc[::-1]
        y = np.arange(len(sub_fl))
        ax.barh(y - 0.18, sub_fl["n_observations_or_events"], height=0.35, color="#4c72b0", label="Events / Index Visits")
        ax.barh(y + 0.18, sub_fl["n_unique_participants"], height=0.35, color="#55a868", label="Unique Participants")
        ax.set_yticks(y)
        ax.set_yticklabels(sub_fl["step_name"], fontsize=8)
        ax.set_xlabel("Count")
        ax.set_title(f"8. Cohort & Sample Eligibility Flow{tag}")
        ax.legend()
    plt.tight_layout()
    p8 = os.path.join(fig_dir, "08_eligibility_flow_diagram.png")
    plt.savefig(p8, dpi=160)
    plt.close(fig)
    saved.append(p8)

    return saved
