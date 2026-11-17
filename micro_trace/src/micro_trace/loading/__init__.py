"""Configuration and safe data loading utilities."""
import os
import gzip
import yaml
import pandas as pd

def load_yaml(path: str) -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(f"YAML configuration file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Invalid YAML structure in {path}: expected mapping.")
    return data

def load_metadata(raw_dir: str, schema_cfg: dict) -> pd.DataFrame:
    rel_path = schema_cfg.get("metadata_table")
    if not rel_path:
        raise ValueError("Missing 'metadata_table' in schema_mapping configuration.")
    fp = os.path.join(raw_dir, rel_path)
    if not os.path.exists(fp):
        raise FileNotFoundError(f"Metadata table not found at {fp}")
    df = pd.read_csv(fp, low_memory=False)
    if df.empty:
        raise ValueError(f"Metadata table at {fp} is empty (0 rows).")
    return df

def read_omics_header_and_stats(fp: str, suffix_strip: str = "", max_rows: int = 5000) -> dict:
    """Read header sample IDs and compute prevalence/zero stats on up to max_rows features."""
    if not os.path.exists(fp):
        return {"exists": False, "sample_ids": [], "raw_sample_cols": [], "n_features": 0}
    op = gzip.open if fp.endswith(".gz") else open
    sample_cols_raw = []
    n_features = 0
    non_zero_counts = []
    col_sums = None
    unmapped_rows = 0
    stratified_rows = 0
    unstratified_rows = 0
    tax_levels = {"k__": 0, "p__": 0, "c__": 0, "o__": 0, "f__": 0, "g__": 0, "s__": 0, "t__": 0}

    with op(fp, "rt", encoding="utf-8", errors="ignore") as f:
        header_line = f.readline()
        if not header_line:
            return {"exists": True, "sample_ids": [], "raw_sample_cols": [], "n_features": 0}
        cols = header_line.rstrip("\r\n").split("\t")
        # Exclude first column (feature ID) and trailing 'taxonomy' column if present
        has_trailing_tax = (cols[-1].lower() == "taxonomy")
        sample_cols_raw = cols[1:-1] if has_trailing_tax else cols[1:]
        n_samples = len(sample_cols_raw)
        col_sums = [0.0] * n_samples

        for line in f:
            n_features += 1
            parts = line.rstrip("\r\n").split("\t")
            feat_name = parts[0]
            if feat_name in ("UNKNOWN", "UNMAPPED", "UNINTEGRATED", "UNGROUPED") or feat_name.startswith(("UNMAPPED", "UNINTEGRATED", "UNGROUPED")):
                unmapped_rows += 1
            if "|" in feat_name:
                stratified_rows += 1
            else:
                unstratified_rows += 1
            # Check lowest taxonomic rank in MetaPhlAn strings
            if "k__" in feat_name:
                last_clade = feat_name.split("|")[-1]
                prefix = last_clade[:3]
                if prefix in tax_levels:
                    tax_levels[prefix] += 1

            if n_features <= max_rows and n_samples > 0:
                vals = parts[1:-1] if has_trailing_tax else parts[1:]
                nz = 0
                for idx, v in enumerate(vals):
                    try:
                        fv = float(v)
                        if fv > 0.0:
                            nz += 1
                            col_sums[idx] += fv
                    except ValueError:
                        pass
                non_zero_counts.append(nz / n_samples)

    clean_ids = []
    for c in sample_cols_raw:
        if suffix_strip and c.endswith(suffix_strip):
            clean_ids.append(c[:-len(suffix_strip)])
        else:
            clean_ids.append(c)

    import numpy as np
    prev_arr = np.array(non_zero_counts) if non_zero_counts else np.array([])
    sums_arr = np.array(col_sums) if col_sums else np.array([])
    return {
        "exists": True,
        "raw_sample_cols": sample_cols_raw,
        "sample_ids": clean_ids,
        "n_samples": len(clean_ids),
        "n_features": n_features,
        "n_features_sampled": len(non_zero_counts),
        "unmapped_or_ungrouped_rows": unmapped_rows,
        "stratified_rows": stratified_rows,
        "unstratified_rows": unstratified_rows,
        "tax_levels": tax_levels,
        "mean_prevalence": float(np.mean(prev_arr)) if prev_arr.size > 0 else 0.0,
        "median_prevalence": float(np.median(prev_arr)) if prev_arr.size > 0 else 0.0,
        "zero_fraction": float(1.0 - np.mean(prev_arr)) if prev_arr.size > 0 else 1.0,
        "prev_ge_1pct": int(np.sum(prev_arr >= 0.01)) if prev_arr.size > 0 else 0,
        "prev_ge_5pct": int(np.sum(prev_arr >= 0.05)) if prev_arr.size > 0 else 0,
        "prev_ge_10pct": int(np.sum(prev_arr >= 0.10)) if prev_arr.size > 0 else 0,
        "prev_ge_20pct": int(np.sum(prev_arr >= 0.20)) if prev_arr.size > 0 else 0,
        "col_sum_median": float(np.median(sums_arr)) if sums_arr.size > 0 else 0.0,
        "col_sum_min": float(np.min(sums_arr)) if sums_arr.size > 0 else 0.0,
        "col_sum_max": float(np.max(sums_arr)) if sums_arr.size > 0 else 0.0,
    }
