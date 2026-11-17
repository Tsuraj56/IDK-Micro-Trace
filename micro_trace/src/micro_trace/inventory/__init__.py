"""File discovery and data inventory module."""
import os
import gzip
import pandas as pd

def build_inventory(raw_dir: str) -> pd.DataFrame:
    """Scan raw_dir and return a comprehensive DataFrame of all files, sizes, formats, and dimensions."""
    if not os.path.exists(raw_dir):
        raise FileNotFoundError(f"Raw data directory not found: {raw_dir}")
    records = []
    for root, _, files in sorted(os.walk(raw_dir)):
        for fn in sorted(files):
            if fn.endswith(".part"):
                continue
            fp = os.path.join(root, fn)
            rel_path = os.path.relpath(fp, raw_dir).replace("\\", "/")
            size_bytes = os.path.getsize(fp)
            size_mb = round(size_bytes / 1e6, 4)
            
            # Determine format
            if fn.endswith(".tsv.gz"):
                fmt = "tsv.gz"
            elif fn.endswith(".biom.gz"):
                fmt = "biom.gz"
            elif fn.endswith(".log.gz"):
                fmt = "log.gz"
            elif fn.endswith(".csv"):
                fmt = "csv"
            elif fn.endswith(".tsv"):
                fmt = "tsv"
            elif fn.endswith(".biom"):
                fmt = "biom"
            elif fn.endswith(".xlsx"):
                fmt = "xlsx"
            else:
                fmt = os.path.splitext(fn)[1].lstrip(".") or "unknown"
            
            # Determine layer from path
            parts = rel_path.split("/")
            if "metadata" in parts or fn == "download_manifest.csv":
                layer = "META"
            elif len(parts) >= 3 and parts[0] == "products":
                layer = parts[2] if parts[1] == "HMP2" else f"{parts[1]}_{parts[2]}"
            else:
                layer = parts[0]
            
            n_rows, n_cols = None, None
            if fmt in ("csv", "tsv", "tsv.gz"):
                try:
                    sep = "," if fmt == "csv" else "\t"
                    op = gzip.open if fmt.endswith(".gz") else open
                    with op(fp, "rt", encoding="utf-8", errors="ignore") as f:
                        first_line = f.readline()
                        if first_line:
                            n_cols = len(first_line.rstrip("\r\n").split(sep))
                            n_rows = sum(1 for _ in f)
                        else:
                            n_cols, n_rows = 0, 0
                except Exception:
                    n_rows, n_cols = -1, -1
            
            records.append({
                "file_name": fn,
                "relative_path": rel_path,
                "layer": layer,
                "format": fmt,
                "size_bytes": size_bytes,
                "size_mb": size_mb,
                "n_rows": n_rows,
                "n_cols": n_cols,
            })
    return pd.DataFrame(records)
