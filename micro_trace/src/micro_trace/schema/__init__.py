"""Schema inspection and mapping validation."""
import pandas as pd

def inspect_schema(meta_df: pd.DataFrame, schema_cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    """Validate mapped columns against meta_df and return (table_schema_df, feature_availability_df, issues)."""
    issues = []
    mapped_lookup = {}
    for group in ["identifiers", "dates", "outcomes", "clinical", "medications", "qc_variables"]:
        sub = schema_cfg.get(group, {})
        if isinstance(sub, dict):
            for k, col_name in sub.items():
                if isinstance(col_name, str) and not k.endswith("_assumed"):
                    mapped_lookup[col_name] = f"{group}.{k}"
    for dcol in schema_cfg.get("dietary_variables", []):
        mapped_lookup[dcol] = "dietary_variables"

    # Check if any mapped column is missing from meta_df
    for col_name, role in mapped_lookup.items():
        if col_name not in meta_df.columns:
            issues.append({
                "severity": "ERROR",
                "category": "schema_mapping",
                "field": col_name,
                "description": f"Configured column '{col_name}' ({role}) not found in metadata table.",
                "affected_rows": len(meta_df),
            })

    total_rows = len(meta_df)
    dt_col = schema_cfg["identifiers"]["data_type"]
    mgx_mask = (meta_df[dt_col] == "metagenomics") if dt_col in meta_df.columns else pd.Series(False, index=meta_df.index)
    n_mgx = int(mgx_mask.sum())

    schema_rows = []
    for idx, col in enumerate(meta_df.columns):
        s = meta_df[col]
        nn = int(s.notna().sum())
        miss_frac = round(1.0 - (nn / total_rows), 4) if total_rows > 0 else 1.0
        nn_mgx = int(s[mgx_mask].notna().sum()) if n_mgx > 0 else 0
        miss_mgx = round(1.0 - (nn_mgx / n_mgx), 4) if n_mgx > 0 else 1.0
        n_uniq = int(s.nunique(dropna=True))
        role = mapped_lookup.get(col, "unmapped_metadata")
        schema_rows.append({
            "column_index": idx,
            "column_name": col,
            "mapped_role": role,
            "dtype": str(s.dtype),
            "non_null_total": nn,
            "missing_frac_total": miss_frac,
            "non_null_mgx": nn_mgx,
            "missing_frac_mgx": miss_mgx,
            "n_unique": n_uniq,
        })
        if nn == 0:
            issues.append({
                "severity": "WARNING",
                "category": "empty_column",
                "field": col,
                "description": f"Column '{col}' is 100% missing (0/5533 non-null).",
                "affected_rows": total_rows,
            })

    schema_df = pd.DataFrame(schema_rows)
    feat_df = schema_df[schema_df["mapped_role"] != "unmapped_metadata"].copy()
    return schema_df, feat_df, issues
