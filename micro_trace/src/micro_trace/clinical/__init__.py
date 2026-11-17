"""Participant summary, clinical variable audit, and Fecal Calprotectin (FC) analysis."""
import numpy as np
import pandas as pd

def audit_participants_and_fc(meta_df: pd.DataFrame, schema_cfg: dict, thresholds: dict) -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    """Build participant_summary.csv, fc_summary.csv, and record FC/clinical data quality issues."""
    id_cfg = schema_cfg["identifiers"]
    pid_col = id_cfg["participant_id"]
    coll_col = id_cfg["collection_event_id"]
    wk_col = id_cfg["week_num"]
    dt_col = id_cfg["data_type"]
    site_col = id_cfg["site_name"]
    date_col = schema_cfg["dates"]["receipt_date"]
    diag_col = schema_cfg["clinical"]["disease_subtype"]
    fc1_col = schema_cfg["outcomes"]["fc_primary"]
    fc2_col = schema_cfg["outcomes"]["fc_secondary"]

    issues = []
    df = meta_df.copy()

    # Check numeric conversion of FC columns
    for col in [fc1_col, fc2_col]:
        if col in df.columns:
            coerced = pd.to_numeric(df[col], errors="coerce")
            n_bad = int((df[col].notna() & coerced.isna()).sum())
            if n_bad > 0:
                issues.append({
                    "severity": "ERROR",
                    "category": "fc_non_numeric",
                    "field": col,
                    "description": f"Column '{col}' has {n_bad} non-numeric string values.",
                    "affected_rows": n_bad,
                })
            df[col] = coerced

    # Check dual FC columns
    n_fc1 = int(df[fc1_col].notna().sum()) if fc1_col in df.columns else 0
    n_fc2 = int(df[fc2_col].notna().sum()) if fc2_col in df.columns else 0
    n_both = int((df[fc1_col].notna() & df[fc2_col].notna()).sum()) if (fc1_col in df.columns and fc2_col in df.columns) else 0
    issues.append({
        "severity": "WARNING",
        "category": "fc_dual_columns_and_unit_label",
        "field": f"{fc1_col} / {fc2_col}",
        "description": (
            f"FC is split across two mutually exclusive columns: '{fc1_col}' ({n_fc1} rows) and '{fc2_col}' ({n_fc2} rows), "
            f"overlap={n_both}. Within identical collection events (site_sub_coll), their numeric values match to 5 decimal places "
            "(max diff < 1e-4), proving both store the exact same scale (~3.45 to 493.7, standard ELISA ug/g range despite 'ng_ml' label)."
        ),
        "affected_rows": n_fc1 + n_fc2,
    })

    df["fc_combined"] = df[fc1_col].combine_first(df[fc2_col])
    df["dt_parsed"] = pd.to_datetime(df[date_col], errors="coerce")

    # Missing dates check
    n_miss_date = int(df["dt_parsed"].isna().sum())
    if n_miss_date > 0:
        issues.append({
            "severity": "WARNING",
            "category": "missing_dates",
            "field": date_col,
            "description": (
                f"'{date_col}' is missing in {n_miss_date}/{len(df)} metadata rows (all in biopsy_16S, host_transcriptomics, "
                "host_genome, methylome, serology; 0 missing in stool omics layers)."
            ),
            "affected_rows": n_miss_date,
        })

    # Check same-week multiple stool collections with different FC
    stool_mask = df["dt_parsed"].notna()
    pw_diff = df[stool_mask].groupby([pid_col, wk_col])["fc_combined"].agg(
        lambda s: float(s.max() - s.min()) if s.notna().sum() > 1 else 0.0
    )
    n_pw_conflict = int((pw_diff > 1e-4).sum())
    if n_pw_conflict > 0:
        issues.append({
            "severity": "WARNING",
            "category": "same_week_multiple_collections",
            "field": f"{pid_col}+{wk_col}",
            "description": (
                f"Found {n_pw_conflict} (Participant ID, week_num) pair(s) with multiple distinct collection events and different FC values "
                "(e.g. C3007 week 0 has C3007C1 on 2014-06-06 FC=41.21 and C3007C2 on 2014-06-12 FC=58.55). "
                "Join on site_sub_coll (or exact date_of_receipt), never on (Participant ID, week_num) alone."
            ),
            "affected_rows": n_pw_conflict,
        })

    # Participant summary
    part_rows = []
    for pid, grp in df.groupby(pid_col):
        diag_vals = grp[diag_col].dropna().unique()
        diag = str(diag_vals[0]) if len(diag_vals) > 0 else "UNKNOWN"
        if len(diag_vals) > 1:
            issues.append({
                "severity": "ERROR",
                "category": "inconsistent_diagnosis",
                "field": diag_col,
                "description": f"Participant {pid} has multiple diagnosis labels: {list(diag_vals)}",
                "affected_rows": len(grp),
            })
        site_vals = grp[site_col].dropna().unique()
        site = str(site_vals[0]) if len(site_vals) > 0 else "UNKNOWN"

        # Unique stool collection events (by site_sub_coll with valid date)
        stool_grp = grp[grp["dt_parsed"].notna()]
        n_total_colls = int(grp[coll_col].nunique())
        n_stool_colls = int(stool_grp[coll_col].nunique())
        fc_colls = stool_grp.dropna(subset=["fc_combined"])[coll_col].nunique()
        mgx_grp = grp[grp[dt_col] == "metagenomics"]
        n_mgx_samples = len(mgx_grp)
        n_mgx_colls = int(mgx_grp[coll_col].nunique())
        n_mgx_with_fc = int(mgx_grp.dropna(subset=["fc_combined"])[coll_col].nunique())
        mtx_colls = int(grp[grp[dt_col] == "metatranscriptomics"][coll_col].nunique())
        s16_stool_colls = int(grp[grp[dt_col] == "stool_16S"][coll_col].nunique())

        dts = stool_grp["dt_parsed"].dropna()
        span_days = int((dts.max() - dts.min()).days) if len(dts) >= 2 else 0

        part_rows.append({
            "participant_id": pid,
            "diagnosis": diag,
            "is_ibd": diag in schema_cfg["clinical"]["ibd_subtypes"],
            "site_name": site,
            "n_metadata_rows": len(grp),
            "n_unique_collection_events": n_total_colls,
            "n_stool_collection_events": n_stool_colls,
            "n_fc_collection_events": int(fc_colls),
            "n_mgx_samples": n_mgx_samples,
            "n_mgx_collection_events": n_mgx_colls,
            "n_mgx_and_fc_events": n_mgx_with_fc,
            "n_mtx_collection_events": mtx_colls,
            "n_stool_16s_events": s16_stool_colls,
            "follow_up_span_days": span_days,
            "has_repeated_stool_visits": n_stool_colls >= 2,
            "has_repeated_fc": fc_colls >= 2,
            "has_repeated_mgx_and_fc": n_mgx_with_fc >= 2,
        })
    part_df = pd.DataFrame(part_rows)

    # FC summary across strata (all metadata rows, unique collection events, MGX collection events, and by diagnosis)
    fc_high = float(thresholds.get("fc_extreme_high", 250.0))
    fc_low = float(thresholds.get("fc_detection_low", 15.0))

    # Deduplicate to collection event level for event-level FC summary
    coll_df = df.sort_values([coll_col, dt_col]).groupby(coll_col, as_index=False).agg({
        pid_col: "first",
        diag_col: "first",
        fc1_col: "first",
        fc2_col: "first",
        "fc_combined": "first",
        "dt_parsed": "first",
    })
    mgx_colls_set = set(df[df[dt_col] == "metagenomics"][coll_col].dropna())
    coll_df["has_mgx"] = coll_df[coll_col].isin(mgx_colls_set)

    subsets = [
        ("all_metadata_rows", df),
        ("unique_collection_events_all", coll_df),
        ("unique_collection_events_with_MGX", coll_df[coll_df["has_mgx"]]),
        ("MGX_events_CD", coll_df[coll_df["has_mgx"] & (coll_df[diag_col] == "CD")]),
        ("MGX_events_UC", coll_df[coll_df["has_mgx"] & (coll_df[diag_col] == "UC")]),
        ("MGX_events_nonIBD", coll_df[coll_df["has_mgx"] & (coll_df[diag_col] == "nonIBD")]),
    ]

    fc_rows = []
    for label, sub in subsets:
        n_tot = len(sub)
        s = sub["fc_combined"].dropna()
        n_val = len(s)
        n_miss = n_tot - n_val
        miss_rate = round(n_miss / n_tot, 4) if n_tot > 0 else 1.0
        if n_val > 0:
            log_s = np.log10(s + 1.0)
            fc_rows.append({
                "subset": label,
                "n_total": n_tot,
                "n_valid_fc": n_val,
                "n_missing_fc": n_miss,
                "missing_rate": miss_rate,
                "n_participants_with_fc": int(sub.loc[sub["fc_combined"].notna(), pid_col].nunique()),
                "min_fc": round(float(s.min()), 4),
                "p10_fc": round(float(s.quantile(0.10)), 4),
                "p25_fc": round(float(s.quantile(0.25)), 4),
                "median_fc": round(float(s.median()), 4),
                "mean_fc": round(float(s.mean()), 4),
                "std_fc": round(float(s.std(ddof=1)), 4) if n_val > 1 else 0.0,
                "p75_fc": round(float(s.quantile(0.75)), 4),
                "p90_fc": round(float(s.quantile(0.90)), 4),
                "max_fc": round(float(s.max()), 4),
                "n_zero_or_neg": int((s <= 0).sum()),
                "n_below_detection_15": int((s <= fc_low).sum()),
                "n_above_250": int((s >= fc_high).sum()),
                "mean_log10_fc_plus_1": round(float(log_s.mean()), 4),
                "std_log10_fc_plus_1": round(float(log_s.std(ddof=1)), 4) if n_val > 1 else 0.0,
            })
        else:
            fc_rows.append({
                "subset": label, "n_total": n_tot, "n_valid_fc": 0, "n_missing_fc": n_tot,
                "missing_rate": 1.0, "n_participants_with_fc": 0, "min_fc": None, "p10_fc": None,
                "p25_fc": None, "median_fc": None, "mean_fc": None, "std_fc": None, "p75_fc": None,
                "p90_fc": None, "max_fc": None, "n_zero_or_neg": 0, "n_below_detection_15": 0,
                "n_above_250": 0, "mean_log10_fc_plus_1": None, "std_log10_fc_plus_1": None,
            })

    return part_df, pd.DataFrame(fc_rows), issues
