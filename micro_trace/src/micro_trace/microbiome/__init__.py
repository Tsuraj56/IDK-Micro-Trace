"""Taxonomic and functional microbiome profile auditing."""
import os
import pandas as pd
from micro_trace.loading import read_omics_header_and_stats

def audit_microbiome_tables(raw_dir: str, meta_df: pd.DataFrame, schema_cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """Inspect all configured taxonomic and functional tables and compute sample coverage."""
    id_cfg = schema_cfg["identifiers"]
    pid_col = id_cfg["participant_id"]
    sid_col = id_cfg["sample_id"]
    coll_col = id_cfg["collection_event_id"]
    dt_col = id_cfg["data_type"]

    omics_cfg = schema_cfg.get("omics_tables", {})
    tax_rows = []
    func_rows = []
    omics_stats = {}
    table_sample_sets = {}

    for tname, tinfo in omics_cfg.items():
        fp = os.path.join(raw_dir, tinfo["path"])
        st = read_omics_header_and_stats(fp, suffix_strip=tinfo.get("sample_id_suffix_strip", ""), max_rows=5000)
        st.update(tinfo)
        omics_stats[tname] = st
        table_sample_sets[tname] = set(st.get("sample_ids", []))

        row_common = {
            "table_name": tname,
            "relative_path": tinfo["path"],
            "exists": st["exists"],
            "assay": tinfo["assay"],
            "tool_or_database": tinfo["tool"],
            "representation": tinfo["representation"],
            "n_samples_in_table": st.get("n_samples", 0),
            "n_features_total": st.get("n_features", 0),
            "n_features_sampled_for_prev": st.get("n_features_sampled", 0),
            "unmapped_or_ungrouped_rows": st.get("unmapped_or_ungrouped_rows", 0),
            "stratified_rows": st.get("stratified_rows", 0),
            "unstratified_rows": st.get("unstratified_rows", 0),
            "mean_feature_prevalence": round(st.get("mean_prevalence", 0.0), 4),
            "median_feature_prevalence": round(st.get("median_prevalence", 0.0), 4),
            "zero_fraction": round(st.get("zero_fraction", 1.0), 4),
            "features_prev_ge_1pct": st.get("prev_ge_1pct", 0),
            "features_prev_ge_5pct": st.get("prev_ge_5pct", 0),
            "features_prev_ge_10pct": st.get("prev_ge_10pct", 0),
            "features_prev_ge_20pct": st.get("prev_ge_20pct", 0),
            "sample_sum_median": round(st.get("col_sum_median", 0.0), 4),
            "sample_sum_min": round(st.get("col_sum_min", 0.0), 4),
            "sample_sum_max": round(st.get("col_sum_max", 0.0), 4),
        }

        if tinfo["modality"] == "taxonomy":
            tl = st.get("tax_levels", {})
            row_common.update({
                "kingdom_rows": tl.get("k__", 0),
                "phylum_rows": tl.get("p__", 0),
                "class_rows": tl.get("c__", 0),
                "order_rows": tl.get("o__", 0),
                "family_rows": tl.get("f__", 0),
                "genus_rows": tl.get("g__", 0),
                "species_rows": tl.get("s__", 0),
                "strain_rows": tl.get("t__", 0),
            })
            tax_rows.append(row_common)
        else:
            row_common.update({
                "functional_quantity": "DNA_functional_potential" if tinfo["modality"] == "function_potential" else "RNA_active_expression",
            })
            func_rows.append(row_common)

    # Build microbiome_sample_coverage across collection events
    df = meta_df.copy()
    fc1 = schema_cfg["outcomes"]["fc_primary"]
    fc2 = schema_cfg["outcomes"]["fc_secondary"]
    df["fc_combined"] = pd.to_numeric(df[fc1], errors="coerce").combine_first(pd.to_numeric(df[fc2], errors="coerce"))

    # Map each External ID to which actual omics tables contain it
    cov_rows = []
    for coll_id, grp in df.groupby(coll_col):
        pid = grp[pid_col].iloc[0]
        diag = grp[schema_cfg["clinical"]["disease_subtype"]].iloc[0]
        wk = grp[id_cfg["week_num"]].iloc[0]
        dt_rec = grp[schema_cfg["dates"]["receipt_date"]].dropna()
        rec_date = str(dt_rec.iloc[0]) if len(dt_rec) > 0 else None
        fc_vals = grp["fc_combined"].dropna()
        fc_val = float(fc_vals.iloc[0]) if len(fc_vals) > 0 else None

        mgx_sids = set(grp.loc[grp[dt_col] == "metagenomics", sid_col].astype(str))
        mtx_sids = set(grp.loc[grp[dt_col] == "metatranscriptomics", sid_col].astype(str))
        s16_stool_sids = set(grp.loc[grp[dt_col] == "stool_16S", sid_col].astype(str))
        s16_biopsy_sids = set(grp.loc[grp[dt_col] == "biopsy_16S", sid_col].astype(str))

        has_mgx_tax_v2 = bool(mgx_sids & table_sample_sets.get("mgx_taxonomy_v2", set()))
        has_mgx_tax_v3 = bool(mgx_sids & table_sample_sets.get("mgx_taxonomy_v3", set()))
        has_mgx_path_v2 = bool(mgx_sids & table_sample_sets.get("mgx_pathways_v2", set()))
        has_mgx_path_v3 = bool(mgx_sids & table_sample_sets.get("mgx_pathways_v3", set()))
        has_mgx_ec_v2 = bool(mgx_sids & table_sample_sets.get("mgx_ecs_v2", set()))
        has_mgx_ec_v3 = bool(mgx_sids & table_sample_sets.get("mgx_ecs_v3", set()))
        has_mtx_path = bool(mtx_sids & table_sample_sets.get("mtx_pathways_v2", set()))
        has_16s_pilot = bool(s16_stool_sids & table_sample_sets.get("s16_pilot_stool_taxonomy", set()))
        has_16s_biopsy = bool(s16_biopsy_sids & table_sample_sets.get("s16_biopsy_taxonomy", set()))

        cov_rows.append({
            "collection_event_id": coll_id,
            "participant_id": pid,
            "diagnosis": diag,
            "week_num": wk,
            "receipt_date": rec_date,
            "has_fc": fc_val is not None,
            "fc_value": fc_val,
            "mgx_sample_ids": ";".join(sorted(mgx_sids)),
            "has_mgx_tax_v2": has_mgx_tax_v2,
            "has_mgx_tax_v3": has_mgx_tax_v3,
            "has_mgx_path_v2": has_mgx_path_v2,
            "has_mgx_path_v3": has_mgx_path_v3,
            "has_mgx_ec_v2": has_mgx_ec_v2,
            "has_mgx_ec_v3": has_mgx_ec_v3,
            "has_mtx_path_v2": has_mtx_path,
            "has_stool_16s_metadata": len(s16_stool_sids) > 0,
            "has_stool_16s_pilot_profile": has_16s_pilot,
            "has_biopsy_16s_profile": has_16s_biopsy,
        })

    return pd.DataFrame(tax_rows), pd.DataFrame(func_rows), pd.DataFrame(cov_rows), omics_stats
