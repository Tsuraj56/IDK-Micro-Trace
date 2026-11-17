"""Phase P1 (C1: Future FC Prediction) Cohort and Modeling Pipeline.

Constructs:
  - c1_cohort_consecutive_0d.csv (445 pairs across 93 participants)
  - c1_cohort_all_horizons_0d.csv (1,671 pairs across 93 participants)
  - c1_cohort_consecutive_le14d.csv (<=14d matching sensitivity cohort, 611 pairs)
  - split_audit_80_20.csv (Frozen ADR-007 75 dev [5 folds of 15] / 18 test participants)
  - c1_cohort_summary.json (JSON execution manifest)
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

REQUIRED_RAW_FILES: dict[str, str] = {
    "metadata": "metadata/hmp2_metadata_2018-08-20.csv",
    "mgx_taxonomy_v3": "products/HMP2/MGX/2018-05-04/taxonomic_profiles_3.tsv.gz",
    "mgx_pathways_v3": "products/HMP2/MGX/2018-05-04/pathabundances_3.tsv.gz",
    "mgx_taxonomy_v2": "products/HMP2/MGX/2018-05-04/taxonomic_profiles.tsv.gz",
    "mgx_pathways_v2": "products/HMP2/MGX/2018-05-04/pathabundance_relab.tsv.gz",
}

DEFAULT_ZERO_SUM_SAMPLES: list[str] = ["MSM5LLDM", "MSM5LLDS"]

CLINICAL_COVARIATE_COLS: list[str] = [
    "consent_age",
    "Age at diagnosis",
    "sex",
    "race",
    "site_name",
    "hbi",
    "sccai",
    "Bowel frequency during the day",
    "Abdominal pain",
    "General wellbeing",
    "Antibiotics",
    "Immunosuppressants (e.g. oral corticosteroids)",
    "reads_filtered",
    "reads_human",
]


def resolve_existing_path(configured_path: str | Path, fallback_candidates: list[Path] | None = None) -> Path:
    """Resolve a configured path, checking fallbacks if the path does not exist."""
    p = Path(configured_path)
    if p.exists():
        return p.resolve()
    if fallback_candidates:
        for fb in fallback_candidates:
            if fb.exists():
                return fb.resolve()
    return p


def verify_raw_mgx_files(cfg: dict[str, Any]) -> dict[str, str]:
    """Verify required HMP2 raw files exist in cfg['paths']['raw_data_dir'].

    Checks Google Drive (/content/drive/MyDrive/Micro-Trace/micro-trace-data/raw)
    with fallback to local repository paths (DataSet/hmp2).

    Raises FileNotFoundError with specific path if any required file is missing or empty.
    Returns dict mapping key -> resolved path string.
    """
    if not isinstance(cfg, dict):
        raise ValueError("cfg must be a dictionary")

    paths_cfg = cfg.get("paths", {})
    if not isinstance(paths_cfg, dict):
        raise ValueError("Configuration missing valid dictionary at cfg['paths']")
    raw_dir_cfg = paths_cfg.get("raw_data_dir")
    if not raw_dir_cfg:
        raise ValueError("Configuration missing cfg['paths']['raw_data_dir']")

    raw_path = Path(raw_dir_cfg)
    repo_root = Path(__file__).resolve().parents[4]
    pkg_root = Path(__file__).resolve().parents[2]

    resolved_raw_dir = None
    if raw_path.exists() and raw_path.is_dir():
        resolved_raw_dir = raw_path.resolve()
    elif (repo_root / raw_path).exists() and (repo_root / raw_path).is_dir():
        resolved_raw_dir = (repo_root / raw_path).resolve()
    elif str(raw_dir_cfg).replace("\\", "/").endswith("DataSet/hmp2") or str(raw_dir_cfg).startswith("/content/drive") or "micro-trace-data" in str(raw_dir_cfg):
        candidate_raw_dirs = [
            Path("/content/drive/MyDrive/Micro-Trace/micro-trace-data/raw"),
            pkg_root.parent / "micro-trace-data" / "raw",
            repo_root.parent / "micro-trace-data" / "raw",
            repo_root / "DataSet" / "hmp2",
            repo_root / "micro-trace-data" / "raw",
            Path.cwd() / "DataSet" / "hmp2",
            Path.cwd() / "micro-trace-data" / "raw",
        ]
        for cand in candidate_raw_dirs:
            if cand.exists() and cand.is_dir():
                resolved_raw_dir = cand.resolve()
                break

    if not resolved_raw_dir:
        raise FileNotFoundError(
            f"Required raw data directory does not exist: {raw_dir_cfg}"
        )

    resolved_files: dict[str, str] = {}
    for key, rel_subpath in REQUIRED_RAW_FILES.items():
        candidate_paths = [
            (resolved_raw_dir / rel_subpath).resolve(),
            (Path("/content/drive/MyDrive/Micro-Trace/micro-trace-data/raw") / rel_subpath).resolve(),
            (repo_root / "DataSet" / "hmp2" / rel_subpath).resolve(),
        ]
        # Also check simplified local directory structure (e.g. mgx/ filename)
        fname = Path(rel_subpath).name
        candidate_paths.append((resolved_raw_dir / "mgx" / fname).resolve())
        candidate_paths.append((resolved_raw_dir / fname).resolve())
        candidate_paths.append((repo_root / "DataSet" / "hmp2" / "mgx" / fname).resolve())
        if key == "metadata":
            candidate_paths.append((resolved_raw_dir / "metadata" / fname).resolve())
            candidate_paths.append((repo_root / "DataSet" / "hmp2" / "metadata" / fname).resolve())

        found_path: Path | None = None
        for cp in candidate_paths:
            if cp.exists() and cp.is_file() and cp.stat().st_size > 0:
                found_path = cp
                break

        if found_path is None:
            raise FileNotFoundError(f"Required raw file missing: {rel_subpath}")
        resolved_files[key] = str(found_path)

    return resolved_files


def create_participant_split_80_20(
    pairs_df: pd.DataFrame,
    random_seed: int = 42,
    dev_frac: float = 0.80,
    n_folds: int = 5,
    all_horizons_df: pd.DataFrame | None = None,
    dev_target: int | None = None,
    test_target: int | None = None,
) -> pd.DataFrame:
    """Deterministically partition unique eligible participants into 80% Dev and 20% Test.

    Stratified by diagnosis ('CD', 'UC', 'nonIBD'):
      - 'dev': exactly 75 participants (80% for N=93), assigned to 5 diagnosis-stratified
               participant-grouped folds (cv_fold in {0, 1, 2, 3, 4}, exactly 15 participants per fold)
      - 'test': exactly 18 participants (20% for N=93), with cv_fold = -1
    Enforces zero participant overlap between dev and test and across all 5 cv_folds.

    Returns DataFrame with columns:
      ['participant_id', 'diagnosis', 'is_ibd', 'partition', 'cv_fold',
       'n_consecutive_0d_pairs', 'n_all_future_0d_pairs', 'n_30_90d_pairs', 'n_14_56d_pairs']
    """
    if pairs_df is None or pairs_df.empty:
        raise ValueError("pairs_df cannot be empty or None")

    required_cols = {"participant_id", "diagnosis"}
    missing = required_cols - set(pairs_df.columns)
    if missing:
        raise ValueError(f"pairs_df missing required columns: {sorted(missing)}")

    null_cols = [c for c in sorted(required_cols) if pairs_df[c].isna().any()]
    if null_cols:
        raise ValueError(f"pairs_df contains null values in required columns: {null_cols}")

    empty_cols = [c for c in sorted(required_cols) if (pairs_df[c].astype(str).str.strip() == "").any()]
    if empty_cols:
        raise ValueError(f"pairs_df contains empty string values in required columns: {empty_cols}")

    if not isinstance(dev_frac, (int, float)) or not (0.0 < dev_frac < 1.0):
        raise ValueError(f"dev_frac must be strictly between 0 and 1, got {dev_frac}")

    if not isinstance(n_folds, int) or n_folds < 2:
        raise ValueError(f"n_folds must be >= 2, got {n_folds}")

    p_meta = pairs_df.drop_duplicates("participant_id").copy()
    total_participants = len(p_meta)

    if total_participants < n_folds + 1:
        raise ValueError(
            f"Need at least {n_folds + 1} unique participants to partition into dev and test, got {total_participants}"
        )

    if "is_ibd" in p_meta.columns:
        p_meta["is_ibd"] = p_meta["is_ibd"].astype(bool)
    else:
        p_meta["is_ibd"] = p_meta["diagnosis"].isin(["CD", "UC"])

    # Determine target counts for dev and test
    if dev_target is not None:
        target_dev = int(dev_target)
        target_test = total_participants - target_dev if test_target is None else int(test_target)
        if target_dev + target_test != total_participants:
            raise ValueError(
                f"dev_target ({target_dev}) + test_target ({target_test}) != total participants ({total_participants})"
            )
        if target_dev < n_folds or target_test < 1:
            raise ValueError(
                f"Invalid targets: target_dev ({target_dev}) must be >= n_folds ({n_folds}) and target_test ({target_test}) >= 1"
            )
    else:
        # Round dev count to nearest multiple of n_folds for equal-sized folds
        raw_dev = total_participants * dev_frac
        target_dev = int(round(raw_dev / n_folds) * n_folds)
        if target_dev >= total_participants:
            target_dev = total_participants - n_folds
        elif target_dev < n_folds:
            target_dev = n_folds
        target_test = total_participants - target_dev

    # Largest Remainder quota per diagnosis
    diag_groups: dict[str, list[str]] = {}
    for diag, grp in p_meta.groupby("diagnosis"):
        diag_groups[str(diag)] = sorted(grp["participant_id"].astype(str).tolist())

    quotas: dict[str, int] = {}
    remainders: dict[str, float] = {}
    for diag, plist in diag_groups.items():
        exact = len(plist) * (target_dev / total_participants)
        floor_val = int(np.floor(exact))
        quotas[diag] = floor_val
        remainders[diag] = exact - floor_val

    allocated = sum(quotas.values())
    diff = target_dev - allocated
    sorted_by_rem = sorted(remainders.items(), key=lambda x: (x[1], x[0]), reverse=True)
    for i in range(diff):
        d_name = sorted_by_rem[i % len(sorted_by_rem)][0]
        quotas[d_name] += 1

    rng = np.random.default_rng(random_seed)
    dev_pids: list[str] = []
    test_pids: list[str] = []
    split_info: dict[str, dict[str, Any]] = {}

    for diag in sorted(diag_groups.keys()):
        plist = list(diag_groups[diag])
        shuffled = rng.permutation(plist).tolist()
        q = quotas[diag]
        dev_diag = shuffled[:q]
        test_diag = shuffled[q:]
        dev_pids.extend(dev_diag)
        test_pids.extend(test_diag)
        for p in dev_diag:
            split_info[p] = {"partition": "dev"}
        for p in test_diag:
            split_info[p] = {"partition": "test", "cv_fold": -1}

    # Assign cv_fold to dev participants with balanced diagnosis round-robin
    fold_ptr = 0
    p_meta_indexed = p_meta.set_index("participant_id")
    for diag in sorted(diag_groups.keys()):
        dev_for_d = [p for p in dev_pids if str(p_meta_indexed.loc[p, "diagnosis"]) == diag]
        for p in dev_for_d:
            split_info[p]["cv_fold"] = fold_ptr % n_folds
            fold_ptr += 1

    # Verification assertions
    assert len(set(dev_pids) & set(test_pids)) == 0, "Leakage: Participant overlap between dev and test!"
    assert len(dev_pids) == target_dev, f"Dev participants mismatch: {len(dev_pids)} != {target_dev}"
    assert len(test_pids) == target_test, f"Test participants mismatch: {len(test_pids)} != {target_test}"

    # Pair count calculations
    is_next_col = "is_next_available_fc_from_index"
    consec_sub = pairs_df[pairs_df[is_next_col]] if is_next_col in pairs_df.columns else pairs_df
    consec_counts = consec_sub.groupby("participant_id").size().to_dict()

    # Determine source for all future pairs and horizon windows
    horizons_source = all_horizons_df if all_horizons_df is not None else pairs_df
    all_future_counts = horizons_source.groupby("participant_id").size().to_dict()

    p_30_90 = pd.DataFrame()
    for c in ["in_protocol_30_90d", "in_30_90d"]:
        if c in horizons_source.columns:
            p_30_90 = horizons_source[horizons_source[c].fillna(False).astype(bool)]
            break
    counts_30_90 = p_30_90.groupby("participant_id").size().to_dict() if not p_30_90.empty else {}

    p_14_56 = pd.DataFrame()
    for c in ["in_monthly_14_56d", "in_14_56d"]:
        if c in horizons_source.columns:
            p_14_56 = horizons_source[horizons_source[c].fillna(False).astype(bool)]
            break
    counts_14_56 = p_14_56.groupby("participant_id").size().to_dict() if not p_14_56.empty else {}

    rows: list[dict[str, Any]] = []
    for _, r in p_meta.iterrows():
        pid = str(r["participant_id"])
        s = split_info[pid]
        rows.append({
            "participant_id": pid,
            "diagnosis": str(r["diagnosis"]),
            "is_ibd": bool(r["is_ibd"]),
            "partition": s["partition"],
            "cv_fold": int(s["cv_fold"]),
            "n_consecutive_0d_pairs": int(consec_counts.get(pid, 0)),
            "n_all_future_0d_pairs": int(all_future_counts.get(pid, 0)),
            "n_30_90d_pairs": int(counts_30_90.get(pid, 0)),
            "n_14_56d_pairs": int(counts_14_56.get(pid, 0)),
        })

    split_df = pd.DataFrame(rows).sort_values(["partition", "cv_fold", "participant_id"]).reset_index(drop=True)
    return split_df


def extract_clinical_covariates_per_event(
    meta_df: pd.DataFrame,
    schema_cfg: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """Extract clinical covariates from site_sub_coll collection events.

    Returns dict mapping collection_event_id -> dict of covariate values.
    """
    coll_col = schema_cfg["identifiers"]["collection_event_id"]
    dt_col = schema_cfg["identifiers"]["data_type"]

    cov_cols = [
        "consent_age",
        "Age at diagnosis",
        "sex",
        "race",
        "site_name",
        "hbi",
        "sccai",
        "Bowel frequency during the day",
        "Abdominal pain",
        "General wellbeing",
        "Antibiotics",
        "Immunosuppressants (e.g. oral corticosteroids)",
    ]

    # MGX-specific sequencing read counts
    mgx_sub = meta_df[meta_df[dt_col] == "metagenomics"].dropna(subset=[coll_col])
    reads_map = mgx_sub.groupby(coll_col)[["reads_filtered", "reads_human"]].first().to_dict(orient="index")

    # Clinical demographic and questionnaire variables
    existing_covs = [c for c in cov_cols if c in meta_df.columns]
    clin_map = meta_df.dropna(subset=[coll_col]).groupby(coll_col)[existing_covs].first().to_dict(orient="index")

    cov_map: dict[str, dict[str, Any]] = {}
    for cid_raw in meta_df[coll_col].dropna().unique():
        cid = str(cid_raw)
        c_dict: dict[str, Any] = {}
        r = reads_map.get(cid_raw, {})
        c_dict["reads_filtered"] = float(r["reads_filtered"]) if pd.notna(r.get("reads_filtered")) else np.nan
        c_dict["reads_human"] = float(r["reads_human"]) if pd.notna(r.get("reads_human")) else np.nan
        c_data = clin_map.get(cid_raw, {})
        for cov_col in cov_cols:
            c_dict[cov_col] = c_data.get(cov_col, np.nan)
        cov_map[cid] = c_dict

    return cov_map


def build_c1_cohorts(cfg_path: str | Path) -> dict[str, Any]:
    """Load HMP2 metadata and schema configuration from cfg_path, construct and save:

      1. c1_cohort_consecutive_0d.csv (445 pairs, 93 participants)
      2. c1_cohort_all_horizons_0d.csv (1,671 pairs, 93 participants)
      3. c1_cohort_consecutive_le14d.csv (<=14d matching sensitivity cohort, 611 pairs)
      4. split_audit_80_20.csv (75 dev / 18 test participants, 0 overlap)
      5. c1_cohort_summary.json (JSON execution manifest)

    Merges 'partition' and 'cv_fold' onto the saved cohort CSVs and flags whether
    index_mgx_sample_id (stripped of _P/_TR) is in excluded_zero_sum_samples.
    Returns the summary dictionary written to c1_cohort_summary.json.
    """
    repo_root = Path(__file__).resolve().parents[4]
    cfg_file = Path(cfg_path)
    if not cfg_file.exists():
        fallbacks = [
            repo_root / "micro_trace" / "configs" / "c1_config.yaml",
            Path.cwd() / "configs" / "c1_config.yaml",
            Path.cwd() / "micro_trace" / "configs" / "c1_config.yaml",
        ]
        cfg_file = resolve_existing_path(cfg_file, fallbacks)
        if not cfg_file.exists():
            raise FileNotFoundError(f"Config file not found: {cfg_path}")

    with open(cfg_file, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Resolve schema mapping
    schema_path_cfg = cfg.get("paths", {}).get("schema_mapping_path")
    schema_path = Path(schema_path_cfg) if schema_path_cfg else Path()
    schema_fallbacks = [
        repo_root / "micro_trace" / "configs" / "schema_mapping.yaml",
        Path.cwd() / "configs" / "schema_mapping.yaml",
        Path.cwd() / "micro_trace" / "configs" / "schema_mapping.yaml",
    ]
    resolved_schema_path = resolve_existing_path(schema_path, schema_fallbacks)
    if not resolved_schema_path.exists():
        raise FileNotFoundError(f"Schema mapping file not found at {schema_path_cfg}")

    with open(resolved_schema_path, "r", encoding="utf-8") as f:
        schema_cfg = yaml.safe_load(f)

    # Verify raw MGX files
    raw_files = verify_raw_mgx_files(cfg)

    # Load metadata
    meta_df = pd.read_csv(raw_files["metadata"], low_memory=False)
    if meta_df.empty:
        raise ValueError(f"Raw metadata table at {raw_files['metadata']} is empty")

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

    meta_df["fc_combined"] = pd.to_numeric(meta_df[fc1_col], errors="coerce").combine_first(
        pd.to_numeric(meta_df[fc2_col], errors="coerce")
    )
    meta_df["dt_parsed"] = pd.to_datetime(meta_df[date_col], errors="coerce")

    # Extract event-level clinical covariates
    event_cov_map = extract_clinical_covariates_per_event(meta_df, schema_cfg)

    # Deduplicate MGX runs to collection_event_id level, preferring non-_TR sample IDs
    mgx_rows = meta_df[meta_df[dt_col] == "metagenomics"].sort_values(sid_col)
    mgx_primary = mgx_rows[~mgx_rows[sid_col].astype(str).str.endswith("_TR")].groupby(coll_col)[sid_col].first()
    mgx_any = mgx_rows.groupby(coll_col)[sid_col].first()
    mgx_sid_map = mgx_primary.combine_first(mgx_any).to_dict()

    # Build event-level records (vectorized)
    valid_events = meta_df.dropna(subset=[coll_col])
    first_events = valid_events.groupby(coll_col)[[pid_col, diag_col, vid_col, wk_col]].first()
    dt_first = valid_events.groupby(coll_col)["dt_parsed"].first()
    fc_first = valid_events.groupby(coll_col)["fc_combined"].first()

    event_rows: list[dict[str, Any]] = []
    for cid_raw in first_events.index:
        cid = str(cid_raw)
        row = first_events.loc[cid_raw]
        pid = str(row[pid_col])
        diag = str(row[diag_col])
        vid = row[vid_col]
        wk = row[wk_col]
        dt_val = dt_first.get(cid_raw, pd.NaT)
        fc_raw = fc_first.get(cid_raw, np.nan)
        fc_val = float(fc_raw) if pd.notna(fc_raw) and fc_raw >= 0 else np.nan
        mgx_sid = mgx_sid_map.get(cid_raw, None)

        event_rows.append({
            "collection_event_id": cid,
            "participant_id": pid,
            "diagnosis": diag,
            "is_ibd": diag in schema_cfg["clinical"]["ibd_subtypes"],
            "visit_num": vid,
            "week_num": wk,
            "event_date": dt_val,
            "fc_value": fc_val,
            "mgx_sample_id": mgx_sid,
            "has_mgx_sample": mgx_sid is not None,
        })

    events_df = pd.DataFrame(event_rows)

    # Filter to events with valid dates
    fc_events = events_df[events_df["fc_value"].notna() & events_df["event_date"].notna()].sort_values("event_date")
    mgx_events = events_df[events_df["has_mgx_sample"] & events_df["event_date"].notna()].sort_values("event_date")

    # Matched visits accounting for metadata_stats
    matched_0d_visits = set(
        events_df[events_df["has_mgx_sample"] & events_df["fc_value"].notna() & events_df["event_date"].notna()]["collection_event_id"]
    )
    matched_0d_pids = set(
        events_df[events_df["collection_event_id"].isin(matched_0d_visits)]["participant_id"]
    )

    # Causal prior/same-day <=14d matched index visits across all MGX visits
    le14d_index_visits: list[str] = []
    le14d_index_pids: set[str] = set()
    fc_by_pid = {p: grp.sort_values("event_date") for p, grp in fc_events.groupby("participant_id")}
    for _, m_row in mgx_events.iterrows():
        m_pid = str(m_row["participant_id"])
        m_date = m_row["event_date"]
        p_fcs = fc_by_pid.get(m_pid)
        if p_fcs is not None and not p_fcs.empty:
            prior_fcs = p_fcs[p_fcs["event_date"] <= m_date]
            if not prior_fcs.empty:
                min_lag = int((m_date - prior_fcs.iloc[-1]["event_date"]).days)
                if min_lag <= 14:
                    le14d_index_visits.append(str(m_row["collection_event_id"]))
                    le14d_index_pids.add(m_pid)

    excluded_zero_sum = cfg.get("excluded_zero_sum_samples", DEFAULT_ZERO_SUM_SAMPLES)

    # Generate causal pairs (t_future > t_index)
    pair_rows: list[dict[str, Any]] = []
    for pid, p_mgx in mgx_events.groupby("participant_id"):
        p_fc = fc_events[fc_events["participant_id"] == pid].sort_values("event_date")
        if p_fc.empty:
            continue
        p_mgx_sorted = p_mgx.sort_values("event_date")

        for _, idx_row in p_mgx_sorted.iterrows():
            idx_date = idx_row["event_date"]
            idx_coll = str(idx_row["collection_event_id"])
            idx_fc_same = idx_row["fc_value"]
            idx_mgx_sid = str(idx_row["mgx_sample_id"])

            # Causal prior or same-day FC (fc_date <= idx_date)
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

            # Causal future FCs strictly after idx_date (elapsed_days > 0)
            fut_fcs = p_fc[p_fc["event_date"] > idx_date].sort_values("event_date")
            for rank_idx, (_, fut_row) in enumerate(fut_fcs.iterrows(), start=1):
                fut_date = fut_row["event_date"]
                elapsed_days = int((fut_date - idx_date).days)
                if elapsed_days <= 0:
                    continue

                fut_coll = str(fut_row["collection_event_id"])
                fut_fc_val = float(fut_row["fc_value"])

                # Zero-sum check on stripped MGX sample ID
                stripped_mgx_sid = idx_mgx_sid.replace("_P", "").replace("_TR", "").strip()
                is_zero_sum = stripped_mgx_sid in excluded_zero_sum

                # Clinical covariates at index
                idx_covs = event_cov_map.get(idx_coll, {})
                fut_covs = event_cov_map.get(fut_coll, {})

                # Future medication change tracking (P1-09 sensitivity filter)
                fut_abx = fut_covs.get("Antibiotics", np.nan)
                fut_imm = fut_covs.get("Immunosuppressants (e.g. oral corticosteroids)", np.nan)
                idx_abx = idx_covs.get("Antibiotics", np.nan)
                idx_imm = idx_covs.get("Immunosuppressants (e.g. oral corticosteroids)", np.nan)

                med_changed = False
                if pd.notna(idx_abx) and pd.notna(fut_abx):
                    if str(idx_abx).strip().lower() != str(fut_abx).strip().lower():
                        med_changed = True
                if pd.notna(idx_imm) and pd.notna(fut_imm):
                    if str(idx_imm).strip().lower() != str(fut_imm).strip().lower():
                        med_changed = True

                pair_id = f"{pid}__{idx_coll}__{fut_coll}"

                pair_dict: dict[str, Any] = {
                    "pair_id": pair_id,
                    "participant_id": str(pid),
                    "diagnosis": str(idx_row["diagnosis"]),
                    "is_ibd": bool(idx_row["is_ibd"]),
                    "index_collection_event": idx_coll,
                    "index_visit_num": idx_row["visit_num"],
                    "index_week_num": idx_row["week_num"],
                    "index_date": str(idx_date.date()),
                    "index_mgx_sample_id": idx_mgx_sid,
                    "is_zero_sum_mgx_sample": is_zero_sum,
                    "has_same_day_index_fc": pd.notna(idx_fc_same),
                    "index_fc_same_day": float(idx_fc_same) if pd.notna(idx_fc_same) else np.nan,
                    "index_log10_fc_same_day": float(np.log10(idx_fc_same + 1.0)) if pd.notna(idx_fc_same) else np.nan,
                    "has_prior_or_same_fc": pd.notna(recent_fc_val),
                    "most_recent_fc_at_or_before_index": recent_fc_val if pd.notna(recent_fc_val) else np.nan,
                    "most_recent_fc_date": str(recent_fc_date.date()) if pd.notna(recent_fc_date) else None,
                    "most_recent_fc_lag_days": recent_fc_lag_days,
                    "future_collection_event": fut_coll,
                    "future_visit_num": fut_row["visit_num"],
                    "future_week_num": fut_row["week_num"],
                    "future_date": str(fut_date.date()),
                    "future_fc": fut_fc_val,
                    "future_log10_fc_plus_1": float(np.log10(fut_fc_val + 1.0)),
                    "elapsed_days": elapsed_days,
                    "is_next_available_fc_from_index": (rank_idx == 1),
                    # Horizon flags
                    "in_short_7_21d": (7 <= elapsed_days <= 21),
                    "in_7_21d": (7 <= elapsed_days <= 21),
                    "in_biweekly_14_35d": (14 <= elapsed_days <= 35),
                    "in_14_35d": (14 <= elapsed_days <= 35),
                    "in_monthly_14_56d": (14 <= elapsed_days <= 56),
                    "in_14_56d": (14 <= elapsed_days <= 56),
                    "in_protocol_30_90d": (30 <= elapsed_days <= 90),
                    "in_30_90d": (30 <= elapsed_days <= 90),
                    "in_any_future_gt0d": True,
                    # Clinical covariates at index
                    "consent_age": idx_covs.get("consent_age", np.nan),
                    "Age at diagnosis": idx_covs.get("Age at diagnosis", np.nan),
                    "sex": idx_covs.get("sex", np.nan),
                    "race": idx_covs.get("race", np.nan),
                    "site_name": idx_covs.get("site_name", np.nan),
                    "hbi": idx_covs.get("hbi", np.nan),
                    "sccai": idx_covs.get("sccai", np.nan),
                    "Bowel frequency during the day": idx_covs.get("Bowel frequency during the day", np.nan),
                    "Abdominal pain": idx_covs.get("Abdominal pain", np.nan),
                    "General wellbeing": idx_covs.get("General wellbeing", np.nan),
                    "Antibiotics": idx_covs.get("Antibiotics", np.nan),
                    "Immunosuppressants (e.g. oral corticosteroids)": idx_covs.get("Immunosuppressants (e.g. oral corticosteroids)", np.nan),
                    "reads_filtered": idx_covs.get("reads_filtered", np.nan),
                    "reads_human": idx_covs.get("reads_human", np.nan),
                    # Future medication flags
                    "future_antibiotics": fut_abx,
                    "future_immunosuppressants": fut_imm,
                    "medication_changed_between_index_and_future": med_changed,
                    "feature_timestamp_le_index_date": True,
                }
                pair_rows.append(pair_dict)

    pairs_df = pd.DataFrame(pair_rows)
    if pairs_df.empty:
        raise ValueError("Failed to generate any valid index-to-future pairs from metadata")

    # Cohort 1: Primary consecutive next-FC with 0d match
    c1_consecutive_0d = pairs_df[
        pairs_df["has_same_day_index_fc"] & pairs_df["is_next_available_fc_from_index"]
    ].copy().reset_index(drop=True)

    # Cohort 2: All causal future pairs with 0d match
    c1_all_horizons_0d = pairs_df[pairs_df["has_same_day_index_fc"]].copy().reset_index(drop=True)

    # Cohort 3: Consecutive next-FC with <=14d causal matching
    c1_consecutive_le14d = pairs_df[
        pairs_df["has_prior_or_same_fc"]
        & (pairs_df["most_recent_fc_lag_days"] <= 14)
        & pairs_df["is_next_available_fc_from_index"]
    ].copy().reset_index(drop=True)

    # Generate Frozen 80/20 (5-fold grouped CV) Split
    split_cfg = cfg.get("split", {})
    random_seed = int(split_cfg.get("random_seed", 42))
    dev_frac = float(split_cfg.get("dev_frac", 0.80))
    n_folds = int(split_cfg.get("n_cv_folds", 5))

    split_df = create_participant_split_80_20(
        c1_all_horizons_0d,
        random_seed=random_seed,
        dev_frac=dev_frac,
        n_folds=n_folds,
        dev_target=split_cfg.get("dev_participants_target"),
        test_target=split_cfg.get("test_participants_target"),
    )

    # Merge partition and cv_fold onto cohort tables
    part_map = split_df.set_index("participant_id")[["partition", "cv_fold"]].to_dict(orient="index")

    for cdf in [c1_consecutive_0d, c1_all_horizons_0d, c1_consecutive_le14d]:
        cdf["partition"] = cdf["participant_id"].map(lambda p: part_map.get(str(p), {}).get("partition", "unassigned"))
        cdf["cv_fold"] = cdf["participant_id"].map(lambda p: part_map.get(str(p), {}).get("cv_fold", -1)).astype(int)

    # Determine report output directory
    c1_reports_dir_cfg = cfg.get("paths", {}).get("c1_reports_dir", "micro_trace/reports/c1")
    rep_path = Path(c1_reports_dir_cfg)
    rep_fallbacks = [
        repo_root / "micro_trace" / "reports" / "c1",
        Path.cwd() / "micro_trace" / "reports" / "c1",
    ]
    output_dir = resolve_existing_path(rep_path, rep_fallbacks)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save output tables
    path_consec_0d = output_dir / "c1_cohort_consecutive_0d.csv"
    path_all_0d = output_dir / "c1_cohort_all_horizons_0d.csv"
    path_consec_le14d = output_dir / "c1_cohort_consecutive_le14d.csv"
    path_split = output_dir / "split_audit_80_20.csv"
    path_summary = output_dir / "c1_cohort_summary.json"

    c1_consecutive_0d.to_csv(path_consec_0d, index=False)
    c1_all_horizons_0d.to_csv(path_all_0d, index=False)
    c1_consecutive_le14d.to_csv(path_consec_le14d, index=False)
    split_df.to_csv(path_split, index=False)

    # Compute summary metrics
    dev_pids = set(split_df[split_df["partition"] == "dev"]["participant_id"])
    test_pids = set(split_df[split_df["partition"] == "test"]["participant_id"])
    overlap_count = len(dev_pids & test_pids)

    fold_counts = {
        f"fold_{f}": int((split_df["cv_fold"] == f).sum())
        for f in range(n_folds)
    }

    summary: dict[str, Any] = {
        "status": "SUCCESS",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "paths": {
            "raw_metadata": raw_files["metadata"],
            "c1_cohort_consecutive_0d": str(path_consec_0d),
            "c1_cohort_all_horizons_0d": str(path_all_0d),
            "c1_cohort_consecutive_le14d": str(path_consec_le14d),
            "split_audit_80_20": str(path_split),
            "c1_cohort_summary": str(path_summary),
        },
        "metadata_stats": {
            "total_rows": int(len(meta_df)),
            "unique_participants": int(meta_df[pid_col].nunique()),
            "unique_collection_events": int(meta_df[coll_col].nunique()),
            "mgx_visits_deduplicated": int(events_df[events_df["has_mgx_sample"]]["collection_event_id"].nunique()),
            "fc_events_total": int(len(fc_events)),
            "mgx_fc_same_day_0d_visits": len(matched_0d_visits),
            "mgx_fc_same_day_0d_participants": len(matched_0d_pids),
            "mgx_fc_prior_or_same_day_le14d_visits": len(le14d_index_visits),
            "mgx_fc_prior_or_same_day_le14d_participants": len(le14d_index_pids),
        },
        "c1_cohort_consecutive_0d": {
            "n_pairs": int(len(c1_consecutive_0d)),
            "n_participants": int(c1_consecutive_0d["participant_id"].nunique()),
            "participants_by_diagnosis": {
                str(k): int(v)
                for k, v in c1_consecutive_0d.drop_duplicates("participant_id")["diagnosis"].value_counts().to_dict().items()
            },
            "pairs_by_diagnosis": {
                str(k): int(v)
                for k, v in c1_consecutive_0d["diagnosis"].value_counts().to_dict().items()
            },
            "ibd_participants": int(c1_consecutive_0d[c1_consecutive_0d["is_ibd"]]["participant_id"].nunique()),
            "ibd_pairs": int(c1_consecutive_0d["is_ibd"].sum()),
            "non_ibd_participants": int(c1_consecutive_0d[~c1_consecutive_0d["is_ibd"]]["participant_id"].nunique()),
            "non_ibd_pairs": int((~c1_consecutive_0d["is_ibd"]).sum()),
            "dev_pairs": int((c1_consecutive_0d["partition"] == "dev").sum()),
            "test_pairs": int((c1_consecutive_0d["partition"] == "test").sum()),
            "median_elapsed_days": float(c1_consecutive_0d["elapsed_days"].median()),
            "iqr_elapsed_days": [
                float(c1_consecutive_0d["elapsed_days"].quantile(0.25)),
                float(c1_consecutive_0d["elapsed_days"].quantile(0.75)),
            ],
            "all_temporal_order_verified": bool((c1_consecutive_0d["elapsed_days"] > 0).all()),
        },
        "c1_cohort_all_horizons_0d": {
            "n_pairs": int(len(c1_all_horizons_0d)),
            "n_participants": int(c1_all_horizons_0d["participant_id"].nunique()),
            "horizons": {
                "short_7_21d": {
                    "pairs": int(c1_all_horizons_0d["in_7_21d"].sum()),
                    "participants": int(c1_all_horizons_0d[c1_all_horizons_0d["in_7_21d"]]["participant_id"].nunique()),
                },
                "biweekly_14_35d": {
                    "pairs": int(c1_all_horizons_0d["in_14_35d"].sum()),
                    "participants": int(c1_all_horizons_0d[c1_all_horizons_0d["in_14_35d"]]["participant_id"].nunique()),
                },
                "monthly_14_56d": {
                    "pairs": int(c1_all_horizons_0d["in_14_56d"].sum()),
                    "participants": int(c1_all_horizons_0d[c1_all_horizons_0d["in_14_56d"]]["participant_id"].nunique()),
                },
                "protocol_30_90d": {
                    "pairs": int(c1_all_horizons_0d["in_30_90d"].sum()),
                    "participants": int(c1_all_horizons_0d[c1_all_horizons_0d["in_30_90d"]]["participant_id"].nunique()),
                },
                "any_future_gt0d": {
                    "pairs": int(len(c1_all_horizons_0d)),
                    "participants": int(c1_all_horizons_0d["participant_id"].nunique()),
                },
            },
        },
        "c1_cohort_consecutive_le14d": {
            "n_matched_index_visits": len(le14d_index_visits),
            "n_matched_participants": len(le14d_index_pids),
            "n_pairs": int(len(c1_consecutive_le14d)),
            "n_participants": int(c1_consecutive_le14d["participant_id"].nunique()),
            "dev_pairs": int((c1_consecutive_le14d["partition"] == "dev").sum()),
            "test_pairs": int((c1_consecutive_le14d["partition"] == "test").sum()),
            "unassigned_pairs": int((c1_consecutive_le14d["partition"] == "unassigned").sum()),
        },
        "split_summary": {
            "dev_participants": int(len(dev_pids)),
            "test_participants": int(len(test_pids)),
            "participant_overlap_dev_test": overlap_count,
            "folds_distribution": fold_counts,
            "dev_participants_by_diagnosis": {
                str(k): int(v)
                for k, v in split_df[split_df["partition"] == "dev"]["diagnosis"].value_counts().to_dict().items()
            },
            "test_participants_by_diagnosis": {
                str(k): int(v)
                for k, v in split_df[split_df["partition"] == "test"]["diagnosis"].value_counts().to_dict().items()
            },
            "total_consecutive_0d_pairs": int(split_df["n_consecutive_0d_pairs"].sum()),
            "total_all_future_0d_pairs": int(split_df["n_all_future_0d_pairs"].sum()),
            "total_30_90d_pairs": int(split_df["n_30_90d_pairs"].sum()),
            "total_14_56d_pairs": int(split_df["n_14_56d_pairs"].sum()),
        },
        "zero_sum_sample_audit": {
            "configured_excluded": excluded_zero_sum,
            "consecutive_0d_flagged": int(c1_consecutive_0d["is_zero_sum_mgx_sample"].sum()),
            "all_horizons_0d_flagged": int(c1_all_horizons_0d["is_zero_sum_mgx_sample"].sum()),
        },
    }

    with open(path_summary, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="MICRO-TRACE C1 Cohort and 80/20 Grouped CV Split Builder")
    parser.add_argument("--config", type=str, default="micro_trace/configs/c1_config.yaml", help="Path to c1_config.yaml")
    args = parser.parse_args()

    summary = build_c1_cohorts(args.config)
    print(f"C1 Cohort Construction Completed: status={summary.get('status')}")
    print(f"  Consecutive 0d pairs: {summary['c1_cohort_consecutive_0d']['n_pairs']} across {summary['c1_cohort_consecutive_0d']['n_participants']} participants")
    print(f"  All future 0d pairs:  {summary['c1_cohort_all_horizons_0d']['n_pairs']} across {summary['c1_cohort_all_horizons_0d']['n_participants']} participants")
    print(f"  Dev participants:     {summary['split_summary']['dev_participants']} (Folds: {summary['split_summary']['folds_distribution']})")
    print(f"  Test participants:    {summary['split_summary']['test_participants']}")
    print(f"  Participant overlap:  {summary['split_summary']['participant_overlap_dev_test']}")


if __name__ == "__main__":
    main()
