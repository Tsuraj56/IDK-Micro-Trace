"""Identifier, join cardinality, and temporal integrity auditing."""
import pandas as pd

def audit_identifiers_and_joins(meta_df: pd.DataFrame, schema_cfg: dict, omics_stats: dict) -> tuple[pd.DataFrame, list[dict]]:
    """Audit key cardinality, uniqueness, and omics-to-metadata join integrity."""
    id_cfg = schema_cfg["identifiers"]
    pid_col = id_cfg["participant_id"]
    sid_col = id_cfg["sample_id"]
    coll_col = id_cfg["collection_event_id"]
    vid_col = id_cfg["visit_id"]
    wk_col = id_cfg["week_num"]
    dt_col = id_cfg["data_type"]
    date_col = schema_cfg["dates"]["receipt_date"]

    issues = []
    audit_rows = []

    # 1. Check missing identifiers
    for col in [pid_col, sid_col, coll_col, vid_col, wk_col, dt_col]:
        n_miss = int(meta_df[col].isna().sum())
        if n_miss > 0:
            issues.append({
                "severity": "ERROR",
                "category": "missing_identifier",
                "field": col,
                "description": f"Identifier column '{col}' has {n_miss} missing values.",
                "affected_rows": n_miss,
            })

    # 2. Check duplicate External ID across entire metadata vs within data_type
    total_rows = len(meta_df)
    uniq_ext = int(meta_df[sid_col].nunique(dropna=True))
    dup_ext_rows = int(meta_df.duplicated(subset=[sid_col], keep=False).sum())
    dup_ext_dt_rows = int(meta_df.duplicated(subset=[sid_col, dt_col], keep=False).sum())
    dup_coll_dt_rows = int(meta_df.duplicated(subset=[coll_col, dt_col], keep=False).sum())

    if dup_ext_rows > 0:
        issues.append({
            "severity": "WARNING",
            "category": "many_to_many_join_risk",
            "field": sid_col,
            "description": (
                f"External ID is NOT globally unique in metadata ({uniq_ext} unique across {total_rows} rows; "
                f"{dup_ext_rows} rows share an External ID across multiple omics layers). "
                "Joining an omics table to unfiltered metadata on External ID alone causes many-to-many row multiplication. "
                "Must filter metadata by data_type (or deduplicate to collection_event_id) before joining."
            ),
            "affected_rows": dup_ext_rows,
        })

    if dup_ext_dt_rows > 0:
        issues.append({
            "severity": "ERROR",
            "category": "duplicate_sample_within_datatype",
            "field": f"{sid_col}+{dt_col}",
            "description": f"Found {dup_ext_dt_rows} duplicate (External ID, data_type) rows in metadata.",
            "affected_rows": dup_ext_dt_rows,
        })

    if dup_coll_dt_rows > 0:
        issues.append({
            "severity": "WARNING",
            "category": "technical_replicates_within_collection",
            "field": f"{coll_col}+{dt_col}",
            "description": (
                f"Found {dup_coll_dt_rows} rows sharing the same (site_sub_coll, data_type) — e.g., technical replicates (_TR / _P) "
                "or multiple biopsy locations (Rectum vs Ileum)."
            ),
            "affected_rows": dup_coll_dt_rows,
        })

    # 3. Audit per data_type in metadata
    for dt_val, grp in meta_df.groupby(dt_col, dropna=False):
        n_r = len(grp)
        n_p = int(grp[pid_col].nunique(dropna=True))
        n_s = int(grp[sid_col].nunique(dropna=True))
        n_c = int(grp[coll_col].nunique(dropna=True))
        n_d = int(grp[date_col].notna().sum()) if date_col in grp.columns else 0
        audit_rows.append({
            "source_table": f"metadata[{dt_val}]",
            "modality": str(dt_val),
            "n_rows": n_r,
            "n_unique_participants": n_p,
            "n_unique_external_ids": n_s,
            "n_unique_collection_events": n_c,
            "n_valid_dates": n_d,
            "is_external_id_unique": (n_r == n_s),
            "matched_in_metadata_dt": n_s,
            "unmatched_in_metadata_dt": 0,
            "join_cardinality": "1:1" if n_r == n_s else "N:1 (replicates/biopsies)",
        })

    # 4. Audit omics tables against metadata
    dt_map = {
        "MGX_WGS": "metagenomics",
        "16S_amplicon_biopsy": "biopsy_16S",
        "16S_amplicon_stool_pilot": "stool_16S",
        "MTX_RNAseq": "metatranscriptomics",
    }
    for tname, st in omics_stats.items():
        if not st.get("exists"):
            continue
        assay = st.get("assay", "")
        expected_dt = dt_map.get(assay, "")
        sids = st["sample_ids"]
        n_s = len(sids)
        n_uniq_s = len(set(sids))
        if n_s != n_uniq_s:
            issues.append({
                "severity": "ERROR",
                "category": "duplicate_omics_column",
                "field": tname,
                "description": f"Omics table '{tname}' contains {n_s - n_uniq_s} duplicate sample ID columns.",
                "affected_rows": n_s - n_uniq_s,
            })
        sub_meta = meta_df[meta_df[dt_col] == expected_dt] if expected_dt else meta_df
        meta_sids = set(sub_meta[sid_col].astype(str))
        matched = set(sids) & meta_sids
        unmatched = set(sids) - meta_sids
        matched_pids = int(sub_meta[sub_meta[sid_col].astype(str).isin(matched)][pid_col].nunique())
        matched_colls = int(sub_meta[sub_meta[sid_col].astype(str).isin(matched)][coll_col].nunique())
        matched_dates = int(sub_meta[sub_meta[sid_col].astype(str).isin(matched)][date_col].notna().sum())

        if len(unmatched) > 0:
            issues.append({
                "severity": "WARNING",
                "category": "unmatched_omics_samples",
                "field": tname,
                "description": f"Omics table '{tname}' has {len(unmatched)} sample IDs not found in metadata[{expected_dt}].",
                "affected_rows": len(unmatched),
            })

        audit_rows.append({
            "source_table": tname,
            "modality": f"{st.get('modality')}:{st.get('tool')}",
            "n_rows": st.get("n_features", 0),
            "n_unique_participants": matched_pids,
            "n_unique_external_ids": n_uniq_s,
            "n_unique_collection_events": matched_colls,
            "n_valid_dates": matched_dates,
            "is_external_id_unique": (n_s == n_uniq_s),
            "matched_in_metadata_dt": len(matched),
            "unmatched_in_metadata_dt": len(unmatched),
            "join_cardinality": "1:1 (on External ID + data_type)" if (n_s == n_uniq_s and len(unmatched) == 0) else "check_unmatched_or_dup",
        })

    return pd.DataFrame(audit_rows), issues
