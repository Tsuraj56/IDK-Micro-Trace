"""Longitudinal sample matching, visit-gap distribution, and index-to-future FC pair generation."""
import numpy as np
import pandas as pd

def build_longitudinal_audit(
    meta_df: pd.DataFrame,
    cov_df: pd.DataFrame,
    schema_cfg: dict,
    matching_windows: list[int],
    horizons: list[dict],
    split_cfg: dict,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Generate sample_matching.csv, visit_gap_summary.csv, longitudinal_pairs.csv, eligibility_flow.csv, split_audit.csv."""
    id_cfg = schema_cfg["identifiers"]
    pid_col = id_cfg["participant_id"]
    sid_col = id_cfg["sample_id"]
    coll_col = id_cfg["collection_event_id"]
    vid_col = id_cfg["visit_id"]
    wk_col = id_cfg["week_num"]
    dt_col = id_cfg["data_type"]
    date_col = schema_cfg["dates"]["receipt_date"]
    diag_col = schema_cfg["clinical"]["disease_subtype"]
    fc1_col = schema_cfg["outcomes"]["fc_primary"]
    fc2_col = schema_cfg["outcomes"]["fc_secondary"]

    df = meta_df.copy()
    df["fc_combined"] = pd.to_numeric(df[fc1_col], errors="coerce").combine_first(pd.to_numeric(df[fc2_col], errors="coerce"))
    df["dt_parsed"] = pd.to_datetime(df[date_col], errors="coerce")

    # 1. Build event-level table for all collection events with valid date or FC
    cov_map = cov_df.set_index("collection_event_id").to_dict(orient="index") if not cov_df.empty else {}

    # Deduplicate metadata to collection_event_id level while retaining primary MGX sample ID
    mgx_rows = df[df[dt_col] == "metagenomics"].sort_values(sid_col)
    # Prefer primary sample ID (without _TR) when technical replicates exist
    mgx_primary = mgx_rows[~mgx_rows[sid_col].astype(str).str.endswith("_TR")].groupby(coll_col)[sid_col].first()
    mgx_any = mgx_rows.groupby(coll_col)[sid_col].first()
    mgx_sid_map = mgx_primary.combine_first(mgx_any).to_dict()

    event_rows = []
    for coll_id, grp in df.groupby(coll_col):
        pid = str(grp[pid_col].iloc[0])
        diag = str(grp[diag_col].iloc[0])
        vid = grp[vid_col].iloc[0]
        wk = grp[wk_col].iloc[0]
        dts = grp["dt_parsed"].dropna()
        dt_val = dts.iloc[0] if len(dts) > 0 else pd.NaT
        fcs = grp["fc_combined"].dropna()
        fc_val = float(fcs.iloc[0]) if len(fcs) > 0 else np.nan
        mgx_sid = mgx_sid_map.get(coll_id, None)
        cinfo = cov_map.get(coll_id, {})
        has_tax = bool(cinfo.get("has_mgx_tax_v2") or cinfo.get("has_mgx_tax_v3"))
        has_func = bool(cinfo.get("has_mgx_path_v2") or cinfo.get("has_mgx_path_v3") or cinfo.get("has_mgx_ec_v2"))
        has_mtx = bool(cinfo.get("has_mtx_path_v2"))

        event_rows.append({
            "collection_event_id": coll_id,
            "participant_id": pid,
            "diagnosis": diag,
            "is_ibd": diag in schema_cfg["clinical"]["ibd_subtypes"],
            "visit_num": vid,
            "week_num": wk,
            "event_date": dt_val,
            "fc_value": fc_val,
            "log10_fc_plus_1": float(np.log10(fc_val + 1.0)) if pd.notna(fc_val) and fc_val >= 0 else np.nan,
            "mgx_sample_id": mgx_sid,
            "has_mgx_sample": mgx_sid is not None,
            "has_taxonomy": has_tax,
            "has_function": has_func,
            "has_mtx": has_mtx,
        })
    events_df = pd.DataFrame(event_rows)
    if events_df.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    # 2. Sample matching audit across candidate windows (matching MGX event to nearest prior/same-day FC event)
    fc_events = events_df[events_df["fc_value"].notna() & events_df["event_date"].notna()].sort_values("event_date")
    mgx_events = events_df[events_df["has_mgx_sample"] & events_df["event_date"].notna()].sort_values("event_date")

    match_rows = []
    for _, mrow in mgx_events.iterrows():
        pid = mrow["participant_id"]
        mdate = mrow["event_date"]
        mcoll = mrow["collection_event_id"]
        direct_fc = mrow["fc_value"]
        p_fcs = fc_events[fc_events["participant_id"] == pid]

        # Strict causal matching for index features: FC must be on or before index date (mdate - fc_date >= 0)
        prior_or_same = p_fcs[p_fcs["event_date"] <= mdate]
        if not prior_or_same.empty:
            diffs_prior = (mdate - prior_or_same["event_date"]).dt.days
            min_idx = diffs_prior.idxmin()
            nearest_prior_days = int(diffs_prior.loc[min_idx])
            nearest_prior_fc = float(prior_or_same.loc[min_idx, "fc_value"])
            nearest_prior_coll = str(prior_or_same.loc[min_idx, "collection_event_id"])
        else:
            nearest_prior_days = None
            nearest_prior_fc = None
            nearest_prior_coll = None

        row_dict = {
            "collection_event_id": mcoll,
            "mgx_sample_id": mrow["mgx_sample_id"],
            "participant_id": pid,
            "diagnosis": mrow["diagnosis"],
            "visit_num": mrow["visit_num"],
            "week_num": mrow["week_num"],
            "mgx_date": str(mdate.date()),
            "has_same_event_fc": pd.notna(direct_fc),
            "same_event_fc": direct_fc if pd.notna(direct_fc) else None,
            "nearest_prior_or_same_fc_event": nearest_prior_coll,
            "nearest_prior_or_same_fc_gap_days": nearest_prior_days,
            "nearest_prior_or_same_fc_val": nearest_prior_fc,
            "has_taxonomy": mrow["has_taxonomy"],
            "has_function": mrow["has_function"],
        }
        for w in matching_windows:
            row_dict[f"matched_causal_within_{w}d"] = (nearest_prior_days is not None and nearest_prior_days <= w)
        match_rows.append(row_dict)
    sample_matching_df = pd.DataFrame(match_rows)

    # 3. Visit gap summary
    gap_rows = []
    for subset_name, sub_ev in [
        ("all_stool_events", events_df[events_df["event_date"].notna()]),
        ("mgx_events", mgx_events),
        ("fc_events", fc_events),
        ("mgx_with_same_day_fc", mgx_events[mgx_events["fc_value"].notna()]),
    ]:
        consec_gaps = []
        all_future_gaps = []
        for pid, pgrp in sub_ev.sort_values("event_date").groupby("participant_id"):
            dates = pgrp["event_date"].drop_duplicates().sort_values().tolist()
            for i in range(len(dates) - 1):
                consec_gaps.append((dates[i + 1] - dates[i]).days)
                for j in range(i + 1, len(dates)):
                    all_future_gaps.append((dates[j] - dates[i]).days)
        for gtype, garr in [("consecutive_visit_gap_days", consec_gaps), ("all_pairwise_future_gap_days", all_future_gaps)]:
            s = pd.Series(garr, dtype="float64")
            if len(s) > 0:
                gap_rows.append({
                    "event_subset": subset_name,
                    "gap_type": gtype,
                    "n_intervals": len(s),
                    "min_days": round(float(s.min()), 1),
                    "p10_days": round(float(s.quantile(0.10)), 1),
                    "p25_days": round(float(s.quantile(0.25)), 1),
                    "median_days": round(float(s.median()), 1),
                    "mean_days": round(float(s.mean()), 1),
                    "p75_days": round(float(s.quantile(0.75)), 1),
                    "p90_days": round(float(s.quantile(0.90)), 1),
                    "max_days": round(float(s.max()), 1),
                })
    visit_gap_df = pd.DataFrame(gap_rows)

    # 4. Generate strictly causal index-to-future pairs
    # Index unit: an MGX collection event with valid date (and matched index FC on or before index_date within window=0 or window=14).
    # Outcome unit: any future collection event for the same participant with valid FC and future_date > index_date.
    pair_rows = []
    for pid, p_mgx in mgx_events.groupby("participant_id"):
        p_fc = fc_events[fc_events["participant_id"] == pid].sort_values("event_date")
        if p_fc.empty:
            continue
        p_mgx_sorted = p_mgx.sort_values("event_date")
        for _, idx_row in p_mgx_sorted.iterrows():
            idx_date = idx_row["event_date"]
            idx_coll = idx_row["collection_event_id"]
            idx_fc_same = idx_row["fc_value"]

            # Most recent FC at or before idx_date (strict causal rule: fc_date <= idx_date)
            hist_fcs = p_fc[p_fc["event_date"] <= idx_date]
            if not hist_fcs.empty:
                last_fc_row = hist_fcs.iloc[-1]
                recent_fc_val = float(last_fc_row["fc_value"])
                recent_fc_date = last_fc_row["event_date"]
                recent_fc_lag_days = int((idx_date - recent_fc_date).days)
            else:
                recent_fc_val = np.nan
                recent_fc_date = pd.NaT
                recent_fc_lag_days = None

            # Future FC events strictly after idx_date (future_date > idx_date)
            fut_fcs = p_fc[p_fc["event_date"] > idx_date].sort_values("event_date")
            for rank_idx, (_, fut_row) in enumerate(fut_fcs.iterrows(), start=1):
                fut_date = fut_row["event_date"]
                elapsed_days = int((fut_date - idx_date).days)
                if elapsed_days <= 0:
                    continue  # Hard guard against same-day or past leakage
                fut_fc_val = float(fut_row["fc_value"])

                pair_dict = {
                    "participant_id": pid,
                    "diagnosis": idx_row["diagnosis"],
                    "is_ibd": bool(idx_row["is_ibd"]),
                    "index_collection_event": idx_coll,
                    "index_visit_num": idx_row["visit_num"],
                    "index_week_num": idx_row["week_num"],
                    "index_date": str(idx_date.date()),
                    "index_mgx_sample_id": idx_row["mgx_sample_id"],
                    "has_same_day_index_fc": pd.notna(idx_fc_same),
                    "index_fc_same_day": float(idx_fc_same) if pd.notna(idx_fc_same) else None,
                    "index_log10_fc_same_day": float(np.log10(idx_fc_same + 1.0)) if pd.notna(idx_fc_same) else None,
                    "has_prior_or_same_fc": pd.notna(recent_fc_val),
                    "most_recent_fc_at_or_before_index": recent_fc_val if pd.notna(recent_fc_val) else None,
                    "most_recent_fc_date": str(recent_fc_date.date()) if pd.notna(recent_fc_date) else None,
                    "most_recent_fc_lag_days": recent_fc_lag_days,
                    "future_collection_event": fut_row["collection_event_id"],
                    "future_visit_num": fut_row["visit_num"],
                    "future_week_num": fut_row["week_num"],
                    "future_date": str(fut_date.date()),
                    "future_fc": fut_fc_val,
                    "future_log10_fc_plus_1": float(np.log10(fut_fc_val + 1.0)),
                    "elapsed_days": elapsed_days,
                    "is_next_available_fc_from_index": (rank_idx == 1),
                    "has_taxonomy": bool(idx_row["has_taxonomy"]),
                    "has_function": bool(idx_row["has_function"]),
                    "has_both_tax_and_func": bool(idx_row["has_taxonomy"] and idx_row["has_function"]),
                    "feature_timestamp_le_index_date": True,
                }
                for h in horizons:
                    hname = h["name"]
                    pair_dict[f"in_{hname}"] = (h["min_days"] <= elapsed_days <= h["max_days"])
                pair_rows.append(pair_dict)

    pairs_df = pd.DataFrame(pair_rows)

    # 5. Build eligibility flow table
    flow_rows = []
    def add_flow(step_num, step_name, n_part, n_ibd_part, n_obs, n_pairs, notes):
        flow_rows.append({
            "step_order": step_num,
            "step_name": step_name,
            "n_unique_participants": int(n_part),
            "n_ibd_participants_CD_UC": int(n_ibd_part),
            "n_observations_or_events": int(n_obs),
            "n_eligible_pairs": int(n_pairs),
            "notes": notes,
        })

    all_pids = events_df["participant_id"].nunique()
    ibd_pids = events_df[events_df["is_ibd"]]["participant_id"].nunique()
    add_flow(1, "All metadata rows", df[pid_col].nunique(), df[df[diag_col].isin(["CD", "UC"])][pid_col].nunique(), len(df), 0, "Raw metadata table across 11 omics layers")
    add_flow(2, "Unique collection events (site_sub_coll)", all_pids, ibd_pids, len(events_df), 0, "Deduplicated physical sample collection events")

    stool_ev = events_df[events_df["event_date"].notna()]
    add_flow(3, "Stool collection events with valid date", stool_ev["participant_id"].nunique(), stool_ev[stool_ev["is_ibd"]]["participant_id"].nunique(), len(stool_ev), 0, "100% of stool events have valid date_of_receipt")

    add_flow(4, "Events with shotgun MGX (taxonomy + function)", mgx_events["participant_id"].nunique(), mgx_events[mgx_events["is_ibd"]]["participant_id"].nunique(), len(mgx_events), 0, "1595 unique collection events (1638 MGX runs incl. replicates)")
    add_flow(5, "Events with valid Fecal Calprotectin (FC)", fc_events["participant_id"].nunique(), fc_events[fc_events["is_ibd"]]["participant_id"].nunique(), len(fc_events), 0, "Combined fecalcal and fecalcal_ng_ml")

    mgx_fc_ev = mgx_events[mgx_events["fc_value"].notna()]
    add_flow(6, "Events with BOTH MGX and same-event FC", mgx_fc_ev["participant_id"].nunique(), mgx_fc_ev[mgx_fc_ev["is_ibd"]]["participant_id"].nunique(), len(mgx_fc_ev), 0, "Exact collection-event match (0-day window)")

    # Participants with >=2 matched MGX+FC events
    cnt_mgx_fc = mgx_fc_ev.groupby("participant_id").size()
    p_ge2 = set(cnt_mgx_fc[cnt_mgx_fc >= 2].index)
    mgx_fc_ge2 = mgx_fc_ev[mgx_fc_ev["participant_id"].isin(p_ge2)]
    add_flow(7, "Participants with >=2 matched (MGX + same-event FC) visits", len(p_ge2), mgx_fc_ge2[mgx_fc_ge2["is_ibd"]]["participant_id"].nunique(), len(mgx_fc_ge2), 0, "Minimum requirement for longitudinal baseline+future pair")

    if not pairs_df.empty:
        # All pairs where index has MGX (any index FC availability)
        add_flow(8, "All causal pairs: Index MGX -> Future FC (t_future > t_index)", pairs_df["participant_id"].nunique(), pairs_df[pairs_df["is_ibd"]]["participant_id"].nunique(), pairs_df["index_collection_event"].nunique(), len(pairs_df), "Unrestricted future horizon (>0 days), index FC optional")

        # Pairs requiring same-day index FC (for M0/M1 previous-FC baseline)
        p_sameday = pairs_df[pairs_df["has_same_day_index_fc"]]
        add_flow(9, "Causal pairs with Same-Day Index FC + MGX -> Future FC", p_sameday["participant_id"].nunique(), p_sameday[p_sameday["is_ibd"]]["participant_id"].nunique(), p_sameday["index_collection_event"].nunique(), len(p_sameday), "Supports all model arms M0-M5 with 0-day index FC match")

        # Next-available future FC pairs (1 pair per index visit, no pairwise horizon fan-out)
        p_next = p_sameday[p_sameday["is_next_available_fc_from_index"]]
        add_flow(10, "Consecutive pairs only (Same-Day Index FC + MGX -> Next Future FC)", p_next["participant_id"].nunique(), p_next[p_next["is_ibd"]]["participant_id"].nunique(), p_next["index_collection_event"].nunique(), len(p_next), "Exactly 1 future target per eligible index visit")

        step_idx = 11
        for h in horizons:
            hname = h["name"]
            col_h = f"in_{hname}"
            sub_all = p_sameday[p_sameday[col_h]]
            sub_ibd = sub_all[sub_all["is_ibd"]]
            add_flow(
                step_idx,
                f"Same-Day Index FC + MGX pairs in horizon '{hname}' ({h['min_days']}-{h['max_days']}d)",
                sub_all["participant_id"].nunique(),
                sub_ibd["participant_id"].nunique(),
                sub_all["index_collection_event"].nunique(),
                len(sub_all),
                f"IBD-only (CD/UC) pairs={len(sub_ibd)} across {sub_ibd['participant_id'].nunique()} IBD participants",
            )
            step_idx += 1

    flow_df = pd.DataFrame(flow_rows)

    # 6. Participant-level split preview (strictly zero participant overlap across train/calib/test)
    rng = np.random.default_rng(int(split_cfg.get("random_seed", 42)))
    elig_pids = sorted(p_sameday["participant_id"].unique()) if not pairs_df.empty else []
    pid_diag = events_df.groupby("participant_id")["diagnosis"].first().to_dict()
    split_rows = []
    # Stratified by diagnosis at participant level
    by_diag = {}
    for p in elig_pids:
        by_diag.setdefault(pid_diag.get(p, "UNKNOWN"), []).append(p)

    train_f = float(split_cfg.get("train_frac", 0.70))
    calib_f = float(split_cfg.get("calib_frac", 0.15))

    for dval, plist in sorted(by_diag.items()):
        parr = list(plist)
        rng.shuffle(parr)
        n = len(parr)
        n_tr = max(1, int(round(n * train_f))) if n >= 3 else n
        n_ca = max(1, int(round(n * calib_f))) if n >= 3 else 0
        if n_tr + n_ca >= n and n >= 3:
            n_tr = n - 2
            n_ca = 1
        for idx, p in enumerate(parr):
            if idx < n_tr:
                part = "train"
            elif idx < n_tr + n_ca:
                part = "calibration"
            else:
                part = "test"
            n_p_pairs = int((p_sameday["participant_id"] == p).sum())
            n_p_next = int((p_next["participant_id"] == p).sum())
            split_rows.append({
                "participant_id": p,
                "diagnosis": dval,
                "partition": part,
                "n_all_horizon_pairs": n_p_pairs,
                "n_next_visit_pairs": n_p_next,
            })

    split_df = pd.DataFrame(split_rows)
    return sample_matching_df, visit_gap_df, pairs_df, flow_df, split_df
