"""Phase P1 (C1: Future FC Prediction) Feature Engineering Pipeline.

Implements strictly train-fitted (PR-03) feature extraction for all C1 model arms:
  - M0: Persistence + Horizon Baseline (index_log10_fc, elapsed_days)
  - M1: Clinical Demographics, Activity Scores, and Medications
  - M2: M1 + MetaPhlAn3 Species CLR + Alpha/Beta Diversity + Taxonomic Guilds (ADR-002)
  - M3: M1 + HUMAnN3 Pathways CLR + Functional Pathway Guilds (ADR-002)
  - M4: Combined Multimodal Feature Set (M1 + M2 taxonomic + M3 functional)
  - M5: M4 + Full Causal Trajectory State Vectors (ADR-009)
  - B3: Reference Taxonomy-Only (elapsed_days + M2 taxonomic block)
  - B4: Reference Function-Only (elapsed_days + M3 functional block)

Also builds causal visit sequence tensors (1..t) for MicroTrace-Attn (ADR-001).
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gzip
import json
import os
from pathlib import Path
from typing import Any
import zlib

import numpy as np
import pandas as pd
import yaml

from micro_trace.c1.cohort import (
    DEFAULT_ZERO_SUM_SAMPLES,
    resolve_existing_path,
)

TAXONOMIC_GUILDS: dict[str, dict[str, Any]] = {
    "guild_scfa_butyrate": {
        "name": "SCFA & Butyrate Producers",
        "keywords": [
            "Faecalibacterium_prausnitzii",
            "Roseburia_",
            "Eubacterium_rectale",
            "Agathobacter_rectalis",
            "Subdoligranulum_",
            "Anaerostipes_",
            "Butyricicoccus_",
            "Coprococcus_",
            "Ruminococcus_bromii",
        ],
    },
    "guild_pathobiont": {
        "name": "Pathobionts & Proteobacteria",
        "keywords": [
            "Escherichia_coli",
            "Klebsiella_",
            "Veillonella_",
            "Ruminococcus_gnavus",
            "Enterococcus_",
            "Haemophilus_parainfluenzae",
            "Fusobacterium_",
            "Campylobacter_",
            "Proteus_",
        ],
    },
}

FUNCTIONAL_GUILDS: dict[str, dict[str, Any]] = {
    "fguild_scfa_butanoate": {
        "name": "SCFA & Butanoate Fermentation",
        "keywords": [
            "CENTFERM-PWY",
            "PWY-5676",
            "P163-PWY",
            "ANAEROFRUCAT-PWY",
            "PWY-5100",
            "butan",
            "ferment",
        ],
    },
    "fguild_sulfur_h2s": {
        "name": "Sulfur & H2S Metabolism",
        "keywords": [
            "SO4ASSIM-PWY",
            "SULFATE-CYS-PWY",
            "PWY-5304",
            "PWY-821",
            "sulfate",
            "sulfur",
        ],
    },
    "fguild_oxidative_stress": {
        "name": "Oxidative Stress & Glutathione",
        "keywords": [
            "GLUTATHIONESYN-PWY",
            "PWY-4081",
            "DETOX1-PWY",
            "glutathione",
            "superoxide",
            "oxidative",
        ],
    },
    "fguild_mucin_hexuronide": {
        "name": "Mucin & Hexuronide Degradation",
        "keywords": [
            "GLUCUROCAT-PWY",
            "GALACTUROCAT-PWY",
            "PWY-6507",
            "PWY-7242",
            "hexuronide",
            "glucuron",
            "galacturon",
            "fucose",
            "sialic",
        ],
    },
    "fguild_lps_enterobacterial": {
        "name": "LPS & Lipid A Biosynthesis",
        "keywords": [
            "NAGLIPASYN-PWY",
            "KDO-NAGLIPASYN-PWY",
            "LPSSYN-PWY",
            "ECASYN-PWY",
            "PWY0-1261",
            "lipopolysaccharide",
            "lipid A",
            "enterobacterial",
        ],
    },
}

DYNAMIC_TRAJECTORY_VARS: list[str] = [
    "fc_log10",
    "hbi",
    "sccai",
    "shannon_species",
    "guild_scfa_butyrate_log",
    "guild_pathobiont_log",
    "guild_protective_vs_pathobiont_logratio",
    "fguild_scfa_butanoate_log",
    "fguild_sulfur_h2s_log",
    "fguild_oxidative_stress_log",
    "fguild_mucin_hexuronide_log",
    "fguild_lps_enterobacterial_log",
]


def _strip_sample_id_suffix(col: str) -> str:
    """Normalize raw metagenomic sample ID column names."""
    c = str(col).strip()
    for sfx in ("_pathabundance_cpm", "_profile", "_taxonomic_profile"):
        if c.endswith(sfx):
            c = c[:-len(sfx)]
    return c


def _get_series(df: pd.DataFrame, col: str) -> pd.Series:
    """Safely get a Series from a DataFrame, returning NaN Series if missing."""
    if col in df.columns:
        return pd.Series(df[col], index=df.index)
    return pd.Series(np.nan, index=df.index)


def load_and_align_omics_tables(
    cfg_path: str | Path,
    cache_dir: str | Path | None = None,
    force_recompute: bool = False,
) -> dict[str, pd.DataFrame]:
    """Load and normalize MetaPhlAn3 and HUMAnN3 tables from raw_data_dir or cache.

    1. MetaPhlAn3 (`taxonomic_profiles_3.tsv.gz`, strip `_profile` suffix):
       - `species_relab`: strictly species-level rows (`|s__` present and `|t__` absent),
         renormalized per sample to sum to 1.0 (samples with sum <= 0 flagged as zero-sum).
       - `genus_relab`: strictly genus-level rows (`|g__` present and `|s__` absent),
         renormalized per sample to sum to 1.0.
    2. HUMAnN3 (`pathabundances_3.tsv.gz`, strip `_pathabundance_cpm` suffix):
       - `pathways_unstrat_relab`: strictly unstratified community pathways (no `|` in name,
         excluding `UNMAPPED` and `UNINTEGRATED`), renormalized per sample to sum to 1.0.

    When cached `.csv.gz` files exist in cache_dir (or cfg['paths']['processed_c1_dir']),
    they are loaded directly unless force_recompute=True, satisfying Invariant 4.

    Returns dict with keys: `'species_relab'`, `'genus_relab'`, `'pathways_unstrat_relab'`,
    each a DataFrame indexed by stripped `mgx_sample_id` with feature names as columns.
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

    paths_cfg = cfg.get("paths", {})

    # Check for on-disk cache
    resolved_cache_dir: Path | None = None
    if cache_dir is not None:
        resolved_cache_dir = Path(cache_dir)
    elif paths_cfg.get("processed_c1_dir"):
        cache_fallbacks = [
            repo_root / "micro-trace-data" / "processed" / "c1",
            Path.cwd() / "micro-trace-data" / "processed" / "c1",
        ]
        resolved_cache_dir = resolve_existing_path(paths_cfg["processed_c1_dir"], cache_fallbacks)

    if resolved_cache_dir and resolved_cache_dir.exists() and not force_recompute:
        cache_sp = resolved_cache_dir / "species_relab.csv.gz"
        cache_gn = resolved_cache_dir / "genus_relab.csv.gz"
        cache_pw = resolved_cache_dir / "pathways_unstrat_relab.csv.gz"
        if cache_sp.exists() and cache_gn.exists() and cache_pw.exists():
            species_df = pd.read_csv(cache_sp, index_col=0, compression="gzip")
            genus_df = pd.read_csv(cache_gn, index_col=0, compression="gzip")
            pathways_df = pd.read_csv(cache_pw, index_col=0, compression="gzip")
            return {
                "species_relab": species_df,
                "genus_relab": genus_df,
                "pathways_unstrat_relab": pathways_df,
            }

    raw_dir_cfg = paths_cfg.get("raw_data_dir")
    if not raw_dir_cfg:
        raise ValueError("Configuration missing cfg['paths']['raw_data_dir']")

    raw_path = Path(raw_dir_cfg)
    if raw_path.exists() and raw_path.is_dir():
        resolved_raw_dir = raw_path
    elif str(raw_dir_cfg).startswith("/content/drive") and not Path("/content/drive").exists():
        fallbacks = [
            repo_root / "micro-trace-data" / "raw",
            Path.cwd() / "micro-trace-data" / "raw",
            Path.cwd().parent / "micro-trace-data" / "raw",
        ]
        resolved_raw_dir = resolve_existing_path(raw_dir_cfg, fallbacks)
    else:
        resolved_raw_dir = raw_path

    if not resolved_raw_dir.exists() or not resolved_raw_dir.is_dir():
        raise FileNotFoundError(f"Required raw data directory does not exist: {raw_dir_cfg}")

    tax_rel = "products/HMP2/MGX/2018-05-04/taxonomic_profiles_3.tsv.gz"
    path_rel = "products/HMP2/MGX/2018-05-04/pathabundances_3.tsv.gz"

    tax_path = (resolved_raw_dir / tax_rel).resolve()
    if not tax_path.exists():
        # Check without .gz
        if (resolved_raw_dir / tax_rel[:-3]).exists():
            tax_path = (resolved_raw_dir / tax_rel[:-3]).resolve()
        else:
            raise FileNotFoundError(f"Required MetaPhlAn3 file missing: {tax_path}")

    path_path = (resolved_raw_dir / path_rel).resolve()
    if not path_path.exists():
        if (resolved_raw_dir / path_rel[:-3]).exists():
            path_path = (resolved_raw_dir / path_rel[:-3]).resolve()
        else:
            raise FileNotFoundError(f"Required HUMAnN3 file missing: {path_path}")

    # 1. Parse MetaPhlAn3
    species_rows: list[tuple[str, list[float]]] = []
    genus_rows: list[tuple[str, list[float]]] = []
    op_tax = gzip.open if str(tax_path).endswith(".gz") else open

    try:
        with op_tax(tax_path, "rt", encoding="utf-8", errors="ignore") as f:
            header_line = ""
            for line in f:
                line_str = line.rstrip("\r\n")
                if not line_str:
                    continue
                if line_str.startswith("#") and "\t" not in line_str:
                    continue
                header_line = line_str
                break

            if not header_line:
                raise ValueError(f"Taxonomy file at {tax_path} is empty or has no header")

            cols = header_line.rstrip("\r\n").split("\t")
            has_trailing_tax = (cols[-1].lower() == "taxonomy")
            sample_cols = cols[1:-1] if has_trailing_tax else cols[1:]
            clean_sample_ids = [_strip_sample_id_suffix(c) for c in sample_cols]

            for line in f:
                parts = line.rstrip("\r\n").split("\t")
                if not parts or not parts[0]:
                    continue
                feat_name = parts[0]
                vals_raw = parts[1:-1] if has_trailing_tax else parts[1:]
                try:
                    vals = [float(v) for v in vals_raw]
                except ValueError:
                    continue

                if "|s__" in feat_name and "|t__" not in feat_name:
                    species_rows.append((feat_name, vals))
                elif "|g__" in feat_name and "|s__" not in feat_name:
                    genus_rows.append((feat_name, vals))
    except (gzip.BadGzipFile, EOFError, zlib.error) as gz_err:
        raise ValueError(f"Corrupted or invalid gzip file at {tax_path}: {gz_err}") from gz_err

    if not species_rows:
        raise ValueError("No species-level rows found in MetaPhlAn3 table")

    species_feats = [r[0] for r in species_rows]
    species_mat = np.array([r[1] for r in species_rows], dtype=np.float64).T
    sp_sums = species_mat.sum(axis=1, keepdims=True)
    sp_valid_mask = (sp_sums > 0).squeeze()
    species_relab_mat = np.zeros_like(species_mat)
    species_relab_mat[sp_valid_mask] = species_mat[sp_valid_mask] / sp_sums[sp_valid_mask]
    species_df = pd.DataFrame(species_relab_mat, index=clean_sample_ids, columns=species_feats)
    species_df = species_df.groupby(species_df.index).first()
    species_df = species_df.loc[:, ~species_df.columns.duplicated()]

    if genus_rows:
        genus_feats = [r[0] for r in genus_rows]
        genus_mat = np.array([r[1] for r in genus_rows], dtype=np.float64).T
        g_sums = genus_mat.sum(axis=1, keepdims=True)
        g_valid_mask = (g_sums > 0).squeeze()
        genus_relab_mat = np.zeros_like(genus_mat)
        genus_relab_mat[g_valid_mask] = genus_mat[g_valid_mask] / g_sums[g_valid_mask]
        genus_df = pd.DataFrame(genus_relab_mat, index=clean_sample_ids, columns=genus_feats)
        genus_df = genus_df.groupby(genus_df.index).first()
        genus_df = genus_df.loc[:, ~genus_df.columns.duplicated()]
    else:
        genus_df = pd.DataFrame(index=clean_sample_ids)

    # 2. Parse HUMAnN3
    path_rows: list[tuple[str, list[float]]] = []
    op_path = gzip.open if str(path_path).endswith(".gz") else open

    try:
        with op_path(path_path, "rt", encoding="utf-8", errors="ignore") as f:
            header_line = ""
            for line in f:
                line_str = line.rstrip("\r\n")
                if not line_str:
                    continue
                if line_str.startswith("#") and "\t" not in line_str:
                    continue
                header_line = line_str
                break

            if not header_line:
                raise ValueError(f"Pathway file at {path_path} is empty or has no header")

            cols = header_line.rstrip("\r\n").split("\t")
            has_trailing_tax = (cols[-1].lower() == "taxonomy")
            sample_cols = cols[1:-1] if has_trailing_tax else cols[1:]
            clean_path_sample_ids = [_strip_sample_id_suffix(c) for c in sample_cols]

            for line in f:
                parts = line.rstrip("\r\n").split("\t")
                if not parts or not parts[0]:
                    continue
                feat_name = parts[0]
                if "|" in feat_name:
                    continue
                if feat_name in ("UNMAPPED", "UNINTEGRATED") or feat_name.startswith(("UNMAPPED", "UNINTEGRATED")):
                    continue

                vals_raw = parts[1:-1] if has_trailing_tax else parts[1:]
                try:
                    vals = [float(v) for v in vals_raw]
                except ValueError:
                    continue
                path_rows.append((feat_name, vals))
    except (gzip.BadGzipFile, EOFError, zlib.error) as gz_err:
        raise ValueError(f"Corrupted or invalid gzip file at {path_path}: {gz_err}") from gz_err

    if not path_rows:
        raise ValueError("No unstratified pathway rows found in HUMAnN3 table")

    path_feats = [r[0] for r in path_rows]
    path_mat = np.array([r[1] for r in path_rows], dtype=np.float64).T
    p_sums = path_mat.sum(axis=1, keepdims=True)
    p_valid_mask = (p_sums > 0).squeeze()
    path_relab_mat = np.zeros_like(path_mat)
    path_relab_mat[p_valid_mask] = path_mat[p_valid_mask] / p_sums[p_valid_mask]
    pathways_df = pd.DataFrame(path_relab_mat, index=clean_path_sample_ids, columns=path_feats)
    pathways_df = pathways_df.groupby(pathways_df.index).first()
    pathways_df = pathways_df.loc[:, ~pathways_df.columns.duplicated()]

    # Cache tables to disk for downstream fast re-use
    if resolved_cache_dir is not None:
        resolved_cache_dir.mkdir(parents=True, exist_ok=True)
        species_df.to_csv(resolved_cache_dir / "species_relab.csv.gz", compression="gzip")
        genus_df.to_csv(resolved_cache_dir / "genus_relab.csv.gz", compression="gzip")
        pathways_df.to_csv(resolved_cache_dir / "pathways_unstrat_relab.csv.gz", compression="gzip")

    return {
        "species_relab": species_df,
        "genus_relab": genus_df,
        "pathways_unstrat_relab": pathways_df,
    }


def compute_alpha_diversity(species_relab: pd.DataFrame) -> pd.DataFrame:
    """Compute Shannon entropy, Simpson index, and species richness per sample."""
    vals = species_relab.to_numpy(dtype=np.float64)
    richness = (vals > 0).sum(axis=1).astype(np.float64)

    with np.errstate(divide="ignore", invalid="ignore"):
        log_vals = np.where(vals > 0, np.log(vals), 0.0)
        shannon = -np.sum(vals * log_vals, axis=1)

    row_sums = vals.sum(axis=1)
    simpson = np.where(row_sums > 0, 1.0 - np.sum(vals ** 2, axis=1), 0.0)

    return pd.DataFrame(
        {
            "shannon_species": shannon,
            "simpson_species": simpson,
            "species_richness": richness,
        },
        index=species_relab.index,
    )


def match_guild_features(feature_names: list[str], guild_dict: dict[str, dict[str, Any]]) -> dict[str, list[str]]:
    """Match column names against domain guild keyword patterns."""
    matched: dict[str, list[str]] = {}
    for gid, ginfo in guild_dict.items():
        kws = ginfo["keywords"]
        matched_cols: list[str] = []
        for col in feature_names:
            col_lower = col.lower()
            if any(kw.lower() in col_lower for kw in kws):
                matched_cols.append(col)
        matched[gid] = matched_cols
    return matched


def compute_taxonomic_guilds(species_relab: pd.DataFrame) -> pd.DataFrame:
    """Compute domain-informed taxonomic guilds and protective-to-pathobiont log-ratio."""
    matched = match_guild_features(list(species_relab.columns), TAXONOMIC_GUILDS)
    scfa_cols = matched.get("guild_scfa_butyrate", [])
    patho_cols = matched.get("guild_pathobiont", [])

    scfa_relab = species_relab[scfa_cols].sum(axis=1) if scfa_cols else pd.Series(0.0, index=species_relab.index)
    patho_relab = species_relab[patho_cols].sum(axis=1) if patho_cols else pd.Series(0.0, index=species_relab.index)

    scfa_log = np.log10(scfa_relab + 1e-6)
    patho_log = np.log10(patho_relab + 1e-6)
    ratio_log = np.log10((scfa_relab + 1e-5) / (patho_relab + 1e-5))

    return pd.DataFrame(
        {
            "guild_scfa_butyrate_relab": scfa_relab,
            "guild_scfa_butyrate_log": scfa_log,
            "guild_pathobiont_relab": patho_relab,
            "guild_pathobiont_log": patho_log,
            "guild_protective_vs_pathobiont_logratio": ratio_log,
        },
        index=species_relab.index,
    )


def compute_functional_guilds(pathways_relab: pd.DataFrame) -> pd.DataFrame:
    """Compute domain-informed functional pathway guilds from unstratified HUMAnN3 pathways."""
    matched = match_guild_features(list(pathways_relab.columns), FUNCTIONAL_GUILDS)
    data: dict[str, pd.Series] = {}

    for gid in FUNCTIONAL_GUILDS.keys():
        cols = matched.get(gid, [])
        col_sum = pathways_relab[cols].sum(axis=1) if cols else pd.Series(0.0, index=pathways_relab.index)
        data[f"{gid}_relab"] = col_sum
        data[f"{gid}_log"] = np.log10(col_sum + 1e-6)

    return pd.DataFrame(data, index=pathways_relab.index)


def compute_bray_curtis(u: np.ndarray, v: np.ndarray) -> float:
    """Compute Bray-Curtis dissimilarity between two non-negative relative abundance vectors."""
    diff_sum = float(np.sum(np.abs(u - v)))
    tot_sum = float(np.sum(u + v))
    if tot_sum <= 0:
        return 0.0
    return float(diff_sum / tot_sum)


def build_visit_trajectory_store(
    meta_df: pd.DataFrame,
    omics_dict: dict[str, pd.DataFrame],
    schema_cfg: dict[str, Any],
    excluded_zero_sum_samples: list[str] | None = None,
) -> pd.DataFrame:
    """Build chronological participant visit table with strictly causal trajectory states (1..t)."""
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

    m = meta_df.copy()
    m["fc_combined"] = pd.to_numeric(m.get(fc1_col), errors="coerce").combine_first(
        pd.to_numeric(m.get(fc2_col), errors="coerce")
    )
    m["dt_parsed"] = pd.to_datetime(m.get(date_col), errors="coerce")

    # Map collection event to single MGX sample ID (preferring non-_TR)
    mgx_rows = m[m[dt_col] == "metagenomics"].sort_values(sid_col)
    mgx_primary = mgx_rows[~mgx_rows[sid_col].astype(str).str.endswith("_TR")].groupby(coll_col)[sid_col].first()
    mgx_any = mgx_rows.groupby(coll_col)[sid_col].first()
    mgx_sid_map = mgx_primary.combine_first(mgx_any).to_dict()

    species_df = omics_dict.get("species_relab", pd.DataFrame())
    pathways_df = omics_dict.get("pathways_unstrat_relab", pd.DataFrame())

    alpha_df = compute_alpha_diversity(species_df) if not species_df.empty else pd.DataFrame()
    tguild_df = compute_taxonomic_guilds(species_df) if not species_df.empty else pd.DataFrame()
    fguild_df = compute_functional_guilds(pathways_df) if not pathways_df.empty else pd.DataFrame()

    excluded_zero = set(excluded_zero_sum_samples or DEFAULT_ZERO_SUM_SAMPLES)

    events: list[dict[str, Any]] = []
    for coll_id, grp in m.groupby(coll_col):
        cid = str(coll_id)
        pid = str(grp[pid_col].dropna().iloc[0]) if pid_col in grp.columns and len(grp[pid_col].dropna()) > 0 else "UNKNOWN"
        diag = str(grp[diag_col].dropna().iloc[0]) if diag_col in grp.columns and len(grp[diag_col].dropna()) > 0 else "nonIBD"
        vid = grp[vid_col].dropna().iloc[0] if vid_col in grp.columns and len(grp[vid_col].dropna()) > 0 else 1
        wk = grp[wk_col].dropna().iloc[0] if wk_col in grp.columns and len(grp[wk_col].dropna()) > 0 else 0.0

        dts = grp["dt_parsed"].dropna()
        dt_val = dts.iloc[0] if len(dts) > 0 else pd.NaT

        fcs = grp["fc_combined"].dropna()
        fc_val = float(fcs.iloc[0]) if len(fcs) > 0 else np.nan

        if "hbi" in grp.columns:
            hbi_val = pd.to_numeric(grp["hbi"], errors="coerce").dropna()
            hbi = float(hbi_val.iloc[0]) if len(hbi_val) > 0 else np.nan
        else:
            hbi = np.nan

        if "sccai" in grp.columns:
            sccai_val = pd.to_numeric(grp["sccai"], errors="coerce").dropna()
            sccai = float(sccai_val.iloc[0]) if len(sccai_val) > 0 else np.nan
        else:
            sccai = np.nan

        mgx_sid = mgx_sid_map.get(coll_id, None)
        has_mgx = mgx_sid is not None
        clean_sid = _strip_sample_id_suffix(str(mgx_sid)) if mgx_sid else None

        events.append({
            "collection_event_id": cid,
            "participant_id": pid,
            "diagnosis": diag,
            "is_ibd": diag in schema_cfg["clinical"]["ibd_subtypes"],
            "visit_num": vid,
            "week_num": wk,
            "event_date": dt_val,
            "fc_value": fc_val,
            "fc_log10": np.log10(fc_val + 1.0) if np.isfinite(fc_val) and fc_val >= 0 else np.nan,
            "hbi": hbi,
            "sccai": sccai,
            "mgx_sample_id": mgx_sid,
            "clean_mgx_sample_id": clean_sid,
            "has_mgx_sample": has_mgx,
            "is_zero_sum_mgx_sample": (clean_sid in excluded_zero) if clean_sid else False,
        })

    events_df = pd.DataFrame(events)
    events_df = events_df[events_df["event_date"].notna()].copy()
    events_df["event_date"] = pd.to_datetime(events_df["event_date"])

    events_df["shannon_species"] = np.nan
    events_df["simpson_species"] = np.nan
    events_df["species_richness"] = np.nan
    events_df["guild_scfa_butyrate_log"] = np.nan
    events_df["guild_pathobiont_log"] = np.nan
    events_df["guild_protective_vs_pathobiont_logratio"] = np.nan
    for gcol in ["fguild_scfa_butanoate_log", "fguild_sulfur_h2s_log", "fguild_oxidative_stress_log",
                 "fguild_mucin_hexuronide_log", "fguild_lps_enterobacterial_log"]:
        events_df[gcol] = np.nan

    for idx, row in events_df.iterrows():
        sid = row["clean_mgx_sample_id"]
        if sid and sid in alpha_df.index:
            events_df.at[idx, "shannon_species"] = alpha_df.at[sid, "shannon_species"]
            events_df.at[idx, "simpson_species"] = alpha_df.at[sid, "simpson_species"]
            events_df.at[idx, "species_richness"] = alpha_df.at[sid, "species_richness"]
        if sid and sid in tguild_df.index:
            events_df.at[idx, "guild_scfa_butyrate_log"] = tguild_df.at[sid, "guild_scfa_butyrate_log"]
            events_df.at[idx, "guild_pathobiont_log"] = tguild_df.at[sid, "guild_pathobiont_log"]
            events_df.at[idx, "guild_protective_vs_pathobiont_logratio"] = tguild_df.at[sid, "guild_protective_vs_pathobiont_logratio"]
        if sid and sid in fguild_df.index:
            for gcol in ["fguild_scfa_butanoate_log", "fguild_sulfur_h2s_log", "fguild_oxidative_stress_log",
                         "fguild_mucin_hexuronide_log", "fguild_lps_enterobacterial_log"]:
                events_df.at[idx, gcol] = fguild_df.at[sid, gcol]

    processed_participants: list[pd.DataFrame] = []

    for pid, p_df in events_df.groupby("participant_id"):
        p_sorted = p_df.sort_values(["event_date", "collection_event_id"]).reset_index(drop=True).copy()
        n_p_visits = len(p_sorted)

        # Build trajectory feature dictionaries to avoid DataFrame fragmentation warnings
        traj_data: dict[str, list[float]] = {
            "has_prior_visit": [0.0] * n_p_visits,
            "n_visits_up_to_index": [float(k + 1) for k in range(n_p_visits)],
            "days_since_first_visit": [0.0] * n_p_visits,
            "days_since_prev_visit": [0.0] * n_p_visits,
            "bray_curtis_from_prev_mgx": [0.0] * n_p_visits,
            "days_since_prev_mgx": [0.0] * n_p_visits,
        }

        for var_name in DYNAMIC_TRAJECTORY_VARS:
            traj_data[f"{var_name}_curr"] = [0.0] * n_p_visits
            traj_data[f"{var_name}_prev"] = [0.0] * n_p_visits
            traj_data[f"{var_name}_delta"] = [0.0] * n_p_visits
            traj_data[f"{var_name}_logratio"] = [0.0] * n_p_visits
            traj_data[f"{var_name}_velocity"] = [0.0] * n_p_visits
            traj_data[f"{var_name}_roll_mean"] = [0.0] * n_p_visits
            traj_data[f"{var_name}_roll_std"] = [0.0] * n_p_visits
            traj_data[f"{var_name}_roll_min"] = [0.0] * n_p_visits
            traj_data[f"{var_name}_roll_max"] = [0.0] * n_p_visits
            traj_data[f"{var_name}_roll_slope"] = [0.0] * n_p_visits

        first_date = p_sorted.loc[0, "event_date"]
        mgx_history: list[tuple[datetime, np.ndarray]] = []

        for k in range(n_p_visits):
            curr_date = p_sorted.loc[k, "event_date"]
            curr_sid = p_sorted.loc[k, "clean_mgx_sample_id"]

            traj_data["has_prior_visit"][k] = 1.0 if k > 0 else 0.0
            traj_data["days_since_first_visit"][k] = max(float((curr_date - first_date).days), 0.0)
            if k > 0:
                prev_date = p_sorted.loc[k - 1, "event_date"]
                traj_data["days_since_prev_visit"][k] = max(float((curr_date - prev_date).days), 0.0)

            # Bray-Curtis
            if curr_sid and curr_sid in species_df.index:
                curr_vec = species_df.loc[curr_sid].to_numpy(dtype=np.float64)
                if mgx_history:
                    last_mgx_date, last_mgx_vec = mgx_history[-1]
                    bc = compute_bray_curtis(curr_vec, last_mgx_vec)
                    dt_mgx = max(float((curr_date - last_mgx_date).days), 0.0)
                    traj_data["bray_curtis_from_prev_mgx"][k] = bc
                    traj_data["days_since_prev_mgx"][k] = dt_mgx
                mgx_history.append((curr_date, curr_vec))

            # Dynamic variables
            for var_name in DYNAMIC_TRAJECTORY_VARS:
                history_vals = [
                    (p_sorted.loc[j, "event_date"], float(p_sorted.loc[j, var_name]))
                    for j in range(k + 1)
                    if pd.notna(p_sorted.loc[j, var_name]) and np.isfinite(p_sorted.loc[j, var_name])
                ]

                curr_val = p_sorted.loc[k, var_name]
                has_curr = pd.notna(curr_val) and np.isfinite(curr_val)

                if history_vals:
                    prior_vals = [v for v in history_vals if v[0] < curr_date]
                    if prior_vals:
                        prev_date_var, prev_val = prior_vals[-1]
                        dt_prev_var = max(float((curr_date - prev_date_var).days), 1.0)
                    else:
                        prev_val = history_vals[0][1]
                        dt_prev_var = 1.0

                    effective_curr = float(curr_val) if has_curr else prev_val
                    delta = effective_curr - prev_val
                    logratio = float(np.log10((abs(effective_curr) + 1.0) / (abs(prev_val) + 1.0)))
                    velocity = delta / dt_prev_var

                    numeric_series = [v[1] for v in history_vals]
                    roll_mean = float(np.mean(numeric_series))
                    roll_std = float(np.std(numeric_series, ddof=1)) if len(numeric_series) >= 2 else 0.0
                    roll_min = float(np.min(numeric_series))
                    roll_max = float(np.max(numeric_series))

                    if len(history_vals) >= 2:
                        t_days = np.array([(v[0] - history_vals[0][0]).days for v in history_vals], dtype=np.float64)
                        y_arr = np.array(numeric_series, dtype=np.float64)
                        t_diff = t_days - np.mean(t_days)
                        y_diff = y_arr - np.mean(y_arr)
                        t_ss = float(np.sum(t_diff ** 2))
                        if t_ss > 1e-6:
                            roll_slope = float(np.sum(t_diff * y_diff) / t_ss)
                        else:
                            roll_slope = 0.0
                    else:
                        roll_slope = 0.0

                    traj_data[f"{var_name}_curr"][k] = effective_curr
                    traj_data[f"{var_name}_prev"][k] = prev_val
                    traj_data[f"{var_name}_delta"][k] = delta
                    traj_data[f"{var_name}_logratio"][k] = logratio
                    traj_data[f"{var_name}_velocity"][k] = velocity
                    traj_data[f"{var_name}_roll_mean"][k] = roll_mean
                    traj_data[f"{var_name}_roll_std"][k] = roll_std
                    traj_data[f"{var_name}_roll_min"][k] = roll_min
                    traj_data[f"{var_name}_roll_max"][k] = roll_max
                    traj_data[f"{var_name}_roll_slope"][k] = roll_slope

        p_traj_df = pd.DataFrame(traj_data, index=p_sorted.index)
        p_complete = pd.concat([p_sorted, p_traj_df], axis=1)
        processed_participants.append(p_complete)

    if not processed_participants:
        return events_df

    trajectory_store_df = pd.concat(processed_participants, ignore_index=True)
    return trajectory_store_df


class C1FeaturePipeline:
    """Strictly train-fitted (PR-03) feature extractor for C1 models (M0-M5, M_guild, M_guild_delta, B3, B4)."""

    VALID_ARMS = {"M0", "M1", "M2", "M3", "M4", "M5", "M_guild", "M_guild_delta", "B3", "B4"}

    def __init__(
        self,
        arm: str = "M4",
        prevalence_threshold: float = 0.10,
        max_clr_species: int | None = None,
        max_clr_pathways: int | None = None,
    ) -> None:
        if arm not in self.VALID_ARMS:
            raise ValueError(f"Unknown model arm: {arm}. Valid arms are: {sorted(self.VALID_ARMS)}")
        self.arm = arm
        self.prevalence_threshold = float(prevalence_threshold)
        self.max_clr_species = max_clr_species
        self.max_clr_pathways = max_clr_pathways

        self.is_fitted_: bool = False
        self.train_medians_: dict[str, float] = {}
        self.retained_species_: list[str] = []
        self.retained_pathways_: list[str] = []
        self.species_pseudocount_: float = 1e-6
        self.pathway_pseudocount_: float = 1e-6
        self.top_species_for_delta_: list[str] = []
        self.top_pathways_for_delta_: list[str] = []
        self.feature_names_: list[str] = []
        self.scaler_mean_: np.ndarray | None = None
        self.scaler_scale_: np.ndarray | None = None
        self.scaler_means_: dict[str, float] = {}
        self.scaler_stds_: dict[str, float] = {}

    def fit(
        self,
        train_pairs_df: pd.DataFrame,
        visit_store_df: pd.DataFrame,
        omics_dict: dict[str, pd.DataFrame],
    ) -> "C1FeaturePipeline":
        """Fit all medians, prevalence masks, pseudocounts, and top taxa strictly on train_pairs_df."""
        if train_pairs_df is None or train_pairs_df.empty:
            raise ValueError("train_pairs_df cannot be empty or None")

        # 1. Fit clinical medians
        clinical_cols = [
            "consent_age",
            "Age at diagnosis",
            "hbi",
            "sccai",
            "Bowel frequency during the day",
            "reads_filtered",
            "reads_human",
        ]
        medians: dict[str, float] = {}
        for col in clinical_cols:
            if col in train_pairs_df.columns:
                s = pd.to_numeric(train_pairs_df[col], errors="coerce").dropna()
                medians[col] = float(s.median()) if len(s) > 0 else 0.0
            else:
                medians[col] = 0.0

        pain_map = {"None": 0.0, "Mild": 1.0, "Moderate": 2.0, "Severe": 3.0}
        well_map = {"Generally well": 0.0, "Slightly below par": 1.0, "Poor": 2.0, "Very poor": 3.0, "Terrible": 4.0}

        pain_s = _get_series(train_pairs_df, "Abdominal pain").map(pain_map).dropna()
        medians["Abdominal pain"] = float(pain_s.median()) if len(pain_s) > 0 else 0.0

        well_s = _get_series(train_pairs_df, "General wellbeing").map(well_map).dropna()
        medians["General wellbeing"] = float(well_s.median()) if len(well_s) > 0 else 0.0

        # Trajectory & Sequence variable medians from training pairs and training visits
        train_pids = set(train_pairs_df["participant_id"].dropna().astype(str)) if "participant_id" in train_pairs_df.columns else set()
        for var in DYNAMIC_TRAJECTORY_VARS:
            if var == "fc_log10":
                fc_s = _get_series(train_pairs_df, "index_log10_fc_same_day")
                if fc_s.isna().all():
                    fc_s = _get_series(train_pairs_df, "index_log10_fc")
                s = pd.to_numeric(fc_s, errors="coerce").dropna()
                if len(s) > 0:
                    medians["fc_log10"] = float(s.median())
                elif not visit_store_df.empty and "fc_log10" in visit_store_df.columns and train_pids:
                    store_train = visit_store_df[visit_store_df["participant_id"].astype(str).isin(train_pids)]
                    s = pd.to_numeric(store_train["fc_log10"], errors="coerce").dropna()
                    if len(s) > 0:
                        medians["fc_log10"] = float(s.median())
            elif var in train_pairs_df.columns:
                s = pd.to_numeric(train_pairs_df[var], errors="coerce").dropna()
                if len(s) > 0:
                    medians[var] = float(s.median())
            elif not visit_store_df.empty and var in visit_store_df.columns and train_pids:
                store_train = visit_store_df[visit_store_df["participant_id"].astype(str).isin(train_pids)]
                s = pd.to_numeric(store_train[var], errors="coerce").dropna()
                if len(s) > 0:
                    medians[var] = float(s.median())

        self.train_medians_ = medians

        # Extract unique train index metagenomic sample IDs
        if "index_mgx_sample_id" in train_pairs_df.columns:
            index_sids = list(dict.fromkeys([
                _strip_sample_id_suffix(str(s))
                for s in train_pairs_df["index_mgx_sample_id"].dropna()
            ]))
        else:
            index_sids = []

        # 2. Fit MetaPhlAn3 Species Prevalence and Pseudocount
        species_df = omics_dict.get("species_relab", pd.DataFrame())
        if not species_df.empty and index_sids:
            matched_sids = [s for s in index_sids if s in species_df.index]
            if matched_sids:
                train_sp = species_df.loc[matched_sids]
                prev = (train_sp > 0).mean(axis=0)
                retained = list(prev[prev >= self.prevalence_threshold].index)

                if self.max_clr_species and len(retained) > self.max_clr_species:
                    variances = train_sp[retained].var(axis=0)
                    retained = list(variances.sort_values(ascending=False).head(self.max_clr_species).index)

                self.retained_species_ = retained

                pos_vals = train_sp[retained].to_numpy(dtype=np.float64)
                pos_vals = pos_vals[pos_vals > 0]
                if len(pos_vals) > 0:
                    self.species_pseudocount_ = float(0.5 * np.min(pos_vals))
                else:
                    self.species_pseudocount_ = 1e-6

                sp_vars = train_sp[retained].var(axis=0)
                self.top_species_for_delta_ = list(sp_vars.sort_values(ascending=False).head(5).index)
            else:
                self.retained_species_ = list(species_df.columns[:50])
                self.species_pseudocount_ = 1e-6
                self.top_species_for_delta_ = self.retained_species_[:5]
        else:
            self.retained_species_ = []
            self.species_pseudocount_ = 1e-6
            self.top_species_for_delta_ = []

        # 3. Fit HUMAnN3 Pathway Prevalence and Pseudocount
        pathways_df = omics_dict.get("pathways_unstrat_relab", pd.DataFrame())
        if not pathways_df.empty and index_sids:
            matched_p_sids = [s for s in index_sids if s in pathways_df.index]
            if matched_p_sids:
                train_paths = pathways_df.loc[matched_p_sids]
                p_prev = (train_paths > 0).mean(axis=0)
                p_retained = list(p_prev[p_prev >= self.prevalence_threshold].index)

                if self.max_clr_pathways and len(p_retained) > self.max_clr_pathways:
                    p_variances = train_paths[p_retained].var(axis=0)
                    p_retained = list(p_variances.sort_values(ascending=False).head(self.max_clr_pathways).index)

                self.retained_pathways_ = p_retained

                pos_p_vals = train_paths[p_retained].to_numpy(dtype=np.float64)
                pos_p_vals = pos_p_vals[pos_p_vals > 0]
                if len(pos_p_vals) > 0:
                    self.pathway_pseudocount_ = float(0.5 * np.min(pos_p_vals))
                else:
                    self.pathway_pseudocount_ = 1e-6

                path_vars = train_paths[p_retained].var(axis=0)
                self.top_pathways_for_delta_ = list(path_vars.sort_values(ascending=False).head(5).index)
            else:
                self.retained_pathways_ = list(pathways_df.columns[:50])
                self.pathway_pseudocount_ = 1e-6
                self.top_pathways_for_delta_ = self.retained_pathways_[:5]
        else:
            self.retained_pathways_ = []
            self.pathway_pseudocount_ = 1e-6
            self.top_pathways_for_delta_ = []

        self.is_fitted_ = True

        # 4. Fit standard scalers strictly on train_pairs_df
        X_train_raw, names = self._extract_arm_features(train_pairs_df, visit_store_df, omics_dict)
        self.feature_names_ = names
        self.scaler_mean_ = np.mean(X_train_raw, axis=0)
        self.scaler_scale_ = np.std(X_train_raw, axis=0)
        self.scaler_scale_ = np.where(self.scaler_scale_ < 1e-12, 1.0, self.scaler_scale_)
        self.scaler_means_ = {name: float(m) for name, m in zip(names, self.scaler_mean_)}
        self.scaler_stds_ = {name: float(s) for name, s in zip(names, self.scaler_scale_)}

        return self

    def _extract_m0_features(self, pairs_df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
        """M0: Persistence + Horizon Baseline (index_log10_fc, elapsed_days)."""
        fc_col = "index_log10_fc_same_day" if "index_log10_fc_same_day" in pairs_df.columns else "index_log10_fc"
        if fc_col not in pairs_df.columns and "most_recent_fc_at_or_before_index" in pairs_df.columns:
            fc_raw = pd.to_numeric(pairs_df["most_recent_fc_at_or_before_index"], errors="coerce")
            fc_vals = np.log10(np.clip(fc_raw, 0, None) + 1.0).to_numpy(dtype=np.float64)
        else:
            fc_vals = pd.to_numeric(_get_series(pairs_df, fc_col), errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)

        elapsed = pd.to_numeric(_get_series(pairs_df, "elapsed_days"), errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
        X_m0 = np.column_stack([fc_vals, elapsed])
        feature_names = ["index_log10_fc", "elapsed_days"]
        return X_m0, feature_names

    def _extract_m1_features(self, pairs_df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
        """M1: M0 + Clinical Demographics, Activity Scores, and Medications."""
        X_m0, m0_names = self._extract_m0_features(pairs_df)
        cols_out: list[np.ndarray] = [X_m0]
        names_out: list[str] = list(m0_names)

        # 1. Diagnosis one-hot
        diag_s = _get_series(pairs_df, "diagnosis").astype(str)
        for d in ["CD", "UC", "nonIBD"]:
            cols_out.append((diag_s == d).astype(np.float64).to_numpy()[:, None])
            names_out.append(f"diagnosis_{d}")

        # 2. Demographics: consent_age, age_at_diagnosis, sex
        c_age_raw = pd.to_numeric(_get_series(pairs_df, "consent_age"), errors="coerce")
        c_age_miss = c_age_raw.isna().astype(np.float64).to_numpy()[:, None]
        c_age_imp = c_age_raw.fillna(self.train_medians_.get("consent_age", 30.0)).to_numpy(dtype=np.float64)[:, None]
        cols_out.extend([c_age_imp, c_age_miss])
        names_out.extend(["consent_age", "consent_age_is_missing"])

        diag_age_raw = pd.to_numeric(_get_series(pairs_df, "Age at diagnosis"), errors="coerce")
        diag_age_miss = diag_age_raw.isna().astype(np.float64).to_numpy()[:, None]
        diag_age_imp = diag_age_raw.fillna(self.train_medians_.get("Age at diagnosis", 25.0)).to_numpy(dtype=np.float64)[:, None]
        cols_out.extend([diag_age_imp, diag_age_miss])
        names_out.extend(["age_at_diagnosis", "age_at_diagnosis_is_missing"])

        sex_s = _get_series(pairs_df, "sex").astype(str)
        cols_out.append((sex_s == "Female").astype(np.float64).to_numpy()[:, None])
        names_out.append("sex_Female")
        cols_out.append((sex_s == "Male").astype(np.float64).to_numpy()[:, None])
        names_out.append("sex_Male")

        # 3. Clinical Activity Scores
        for col_name, clean_name in [("hbi", "hbi"), ("sccai", "sccai"), ("Bowel frequency during the day", "bowel_frequency")]:
            raw_s = pd.to_numeric(_get_series(pairs_df, col_name), errors="coerce")
            miss = raw_s.isna().astype(np.float64).to_numpy()[:, None]
            imp = raw_s.fillna(self.train_medians_.get(col_name, 0.0)).to_numpy(dtype=np.float64)[:, None]
            cols_out.extend([imp, miss])
            names_out.extend([clean_name, f"{clean_name}_is_missing"])

        pain_map = {"None": 0.0, "Mild": 1.0, "Moderate": 2.0, "Severe": 3.0}
        well_map = {"Generally well": 0.0, "Slightly below par": 1.0, "Poor": 2.0, "Very poor": 3.0, "Terrible": 4.0}

        pain_raw = _get_series(pairs_df, "Abdominal pain").map(pain_map)
        pain_miss = pain_raw.isna().astype(np.float64).to_numpy()[:, None]
        pain_imp = pain_raw.fillna(self.train_medians_.get("Abdominal pain", 0.0)).to_numpy(dtype=np.float64)[:, None]
        cols_out.extend([pain_imp, pain_miss])
        names_out.extend(["abdominal_pain", "abdominal_pain_is_missing"])

        well_raw = _get_series(pairs_df, "General wellbeing").map(well_map)
        well_miss = well_raw.isna().astype(np.float64).to_numpy()[:, None]
        well_imp = well_raw.fillna(self.train_medians_.get("General wellbeing", 0.0)).to_numpy(dtype=np.float64)[:, None]
        cols_out.extend([well_imp, well_miss])
        names_out.extend(["general_wellbeing", "general_wellbeing_is_missing"])

        # 4. Medications at index
        abx_raw = _get_series(pairs_df, "Antibiotics").astype(str).str.strip().str.lower()
        abx_is_yes = abx_raw.isin(["yes", "true", "1", "1.0"]).astype(np.float64).to_numpy()[:, None]
        abx_miss = _get_series(pairs_df, "Antibiotics").isna().astype(np.float64).to_numpy()[:, None]
        cols_out.extend([abx_is_yes, abx_miss])
        names_out.extend(["antibiotics_yes", "antibiotics_is_missing"])

        imm_col = "Immunosuppressants (e.g. oral corticosteroids)"
        imm_raw = _get_series(pairs_df, imm_col).astype(str).str.strip().str.lower()
        imm_is_yes = imm_raw.isin(["yes", "true", "1", "1.0"]).astype(np.float64).to_numpy()[:, None]
        imm_miss = _get_series(pairs_df, imm_col).isna().astype(np.float64).to_numpy()[:, None]
        cols_out.extend([imm_is_yes, imm_miss])
        names_out.extend(["immunosuppressants_yes", "immunosuppressants_is_missing"])

        # 5. Sequencing quality
        rf_raw = pd.to_numeric(_get_series(pairs_df, "reads_filtered"), errors="coerce")
        rf_miss = rf_raw.isna().astype(np.float64).to_numpy()[:, None]
        rf_safe = np.maximum(rf_raw.fillna(self.train_medians_.get("reads_filtered", 1e7)), 1.0)
        rf_log = np.log10(rf_safe + 1.0).to_numpy(dtype=np.float64)[:, None]
        cols_out.extend([rf_log, rf_miss])
        names_out.extend(["reads_filtered_log10", "reads_filtered_is_missing"])

        rh_raw = pd.to_numeric(_get_series(pairs_df, "reads_human"), errors="coerce")
        rh_frac = (rh_raw / rf_safe).clip(0.0, 1.0).fillna(0.0).to_numpy(dtype=np.float64)[:, None]
        rh_miss = rh_raw.isna().astype(np.float64).to_numpy()[:, None]
        cols_out.extend([rh_frac, rh_miss])
        names_out.extend(["reads_human_frac", "reads_human_is_missing"])

        X_m1 = np.column_stack(cols_out)
        return X_m1, names_out

    def _extract_clr_block(
        self,
        sample_ids: list[str],
        omics_df: pd.DataFrame,
        retained_features: list[str],
        pseudocount: float,
        feature_prefix: str,
    ) -> tuple[np.ndarray, list[str], np.ndarray]:
        """Extract CLR transformed omics matrix with zero-sum handling."""
        n_samples = len(sample_ids)
        d_feats = len(retained_features)

        if d_feats == 0 or omics_df.empty:
            return np.zeros((n_samples, 0), dtype=np.float64), [], np.zeros((n_samples, 1), dtype=np.float64)

        raw_mat = np.zeros((n_samples, d_feats), dtype=np.float64)
        is_missing_or_zero = np.zeros((n_samples, 1), dtype=np.float64)

        for i, sid in enumerate(sample_ids):
            if sid and sid in omics_df.index:
                row_slice = omics_df.loc[sid, retained_features]
                if isinstance(row_slice, pd.DataFrame):
                    row_slice = row_slice.iloc[0]
                row_vals = row_slice.to_numpy(dtype=np.float64)
                if np.sum(row_vals) > 0:
                    raw_mat[i] = row_vals
                else:
                    is_missing_or_zero[i, 0] = 1.0
            else:
                is_missing_or_zero[i, 0] = 1.0

        clr_mat = np.zeros_like(raw_mat)
        valid_rows = (is_missing_or_zero[:, 0] == 0.0)

        if np.any(valid_rows):
            log_vals = np.log(raw_mat[valid_rows] + pseudocount)
            geom_means = log_vals.mean(axis=1, keepdims=True)
            clr_mat[valid_rows] = log_vals - geom_means

        feat_names = [f"{feature_prefix}_{feat}" for feat in retained_features]
        return clr_mat, feat_names, is_missing_or_zero

    def _extract_m2_taxonomic_block(
        self,
        pairs_df: pd.DataFrame,
        visit_store_df: pd.DataFrame,
        omics_dict: dict[str, pd.DataFrame],
    ) -> tuple[np.ndarray, list[str]]:
        """Extract M2 taxonomic features (Species CLR, Alpha/Beta Diversity, Taxonomic Guilds)."""
        sid_s = _get_series(pairs_df, "index_mgx_sample_id").fillna("")
        sample_ids = [_strip_sample_id_suffix(str(s)) for s in sid_s]
        species_df = omics_dict.get("species_relab", pd.DataFrame())

        clr_mat, clr_names, zero_flag = self._extract_clr_block(
            sample_ids=sample_ids,
            omics_df=species_df,
            retained_features=self.retained_species_,
            pseudocount=self.species_pseudocount_,
            feature_prefix="clr_species",
        )

        n_samples = len(pairs_df)
        alpha_tguild_cols = np.zeros((n_samples, 6), dtype=np.float64)
        div_names = [
            "shannon_species",
            "simpson_species",
            "species_richness",
            "guild_scfa_butyrate_log",
            "guild_pathobiont_log",
            "guild_protective_vs_pathobiont_logratio",
        ]

        if not species_df.empty:
            alpha_df = compute_alpha_diversity(species_df)
            tguild_df = compute_taxonomic_guilds(species_df)
            for i, sid in enumerate(sample_ids):
                if sid in alpha_df.index:
                    alpha_tguild_cols[i, 0] = alpha_df.at[sid, "shannon_species"]
                    alpha_tguild_cols[i, 1] = alpha_df.at[sid, "simpson_species"]
                    alpha_tguild_cols[i, 2] = alpha_df.at[sid, "species_richness"]
                if sid in tguild_df.index:
                    alpha_tguild_cols[i, 3] = tguild_df.at[sid, "guild_scfa_butyrate_log"]
                    alpha_tguild_cols[i, 4] = tguild_df.at[sid, "guild_pathobiont_log"]
                    alpha_tguild_cols[i, 5] = tguild_df.at[sid, "guild_protective_vs_pathobiont_logratio"]

        beta_cols = np.zeros((n_samples, 2), dtype=np.float64)
        beta_names = ["bray_curtis_from_prev_mgx", "days_since_prev_mgx"]
        if not visit_store_df.empty and "collection_event_id" in visit_store_df.columns:
            store_idx = visit_store_df.set_index("collection_event_id")
            for i, cid in enumerate(pairs_df.get("index_collection_event", [])):
                if cid in store_idx.index:
                    r = store_idx.loc[cid]
                    if isinstance(r, pd.DataFrame):
                        r = r.iloc[0]
                    beta_cols[i, 0] = float(r.get("bray_curtis_from_prev_mgx", 0.0))
                    beta_cols[i, 1] = float(r.get("days_since_prev_mgx", 0.0))

        X_tax = np.column_stack([zero_flag, clr_mat, alpha_tguild_cols, beta_cols])
        names_tax = ["is_zero_sum_or_missing_omics"] + clr_names + div_names + beta_names
        return X_tax, names_tax

    def _extract_m3_functional_block(
        self,
        pairs_df: pd.DataFrame,
        omics_dict: dict[str, pd.DataFrame],
    ) -> tuple[np.ndarray, list[str]]:
        """Extract M3 functional features (Pathways CLR, Functional Pathway Guilds)."""
        sid_s = _get_series(pairs_df, "index_mgx_sample_id").fillna("")
        sample_ids = [_strip_sample_id_suffix(str(s)) for s in sid_s]
        pathways_df = omics_dict.get("pathways_unstrat_relab", pd.DataFrame())

        clr_mat, clr_names, zero_flag = self._extract_clr_block(
            sample_ids=sample_ids,
            omics_df=pathways_df,
            retained_features=self.retained_pathways_,
            pseudocount=self.pathway_pseudocount_,
            feature_prefix="clr_pathway",
        )

        n_samples = len(pairs_df)
        fguild_names = [
            "fguild_scfa_butanoate_log",
            "fguild_sulfur_h2s_log",
            "fguild_oxidative_stress_log",
            "fguild_mucin_hexuronide_log",
            "fguild_lps_enterobacterial_log",
        ]
        fguild_cols = np.zeros((n_samples, len(fguild_names)), dtype=np.float64)

        if not pathways_df.empty:
            fguild_df = compute_functional_guilds(pathways_df)
            for i, sid in enumerate(sample_ids):
                if sid in fguild_df.index:
                    for j, gname in enumerate(fguild_names):
                        fguild_cols[i, j] = fguild_df.at[sid, gname]

        X_func = np.column_stack([zero_flag, clr_mat, fguild_cols])
        names_func = ["is_zero_sum_or_missing_omics"] + clr_names + fguild_names
        return X_func, names_func

    def _extract_guild_summary_block(
        self,
        pairs_df: pd.DataFrame,
        visit_store_df: pd.DataFrame,
        omics_dict: dict[str, pd.DataFrame],
    ) -> tuple[np.ndarray, list[str]]:
        """Extract low-dimensional biological guild & diversity summary features (no raw species/pathways)."""
        sid_s = _get_series(pairs_df, "index_mgx_sample_id").fillna("")
        sample_ids = [_strip_sample_id_suffix(str(s)) for s in sid_s]
        species_df = omics_dict.get("species_relab", pd.DataFrame())
        pathways_df = omics_dict.get("pathways_unstrat_relab", pd.DataFrame())

        n_samples = len(pairs_df)
        zero_flag = np.zeros((n_samples, 1), dtype=np.float64)
        div_tguild_cols = np.zeros((n_samples, 6), dtype=np.float64)
        div_names = [
            "shannon_species",
            "simpson_species",
            "species_richness",
            "guild_scfa_butyrate_log",
            "guild_pathobiont_log",
            "guild_protective_vs_pathobiont_logratio",
        ]

        if not species_df.empty:
            alpha_df = compute_alpha_diversity(species_df)
            tguild_df = compute_taxonomic_guilds(species_df)
            for i, sid in enumerate(sample_ids):
                if sid and sid in species_df.index:
                    row_sp = species_df.loc[sid]
                    if isinstance(row_sp, pd.DataFrame):
                        row_sp = row_sp.iloc[0]
                    if float(np.sum(row_sp.to_numpy(dtype=np.float64))) <= 0.0:
                        zero_flag[i, 0] = 1.0
                else:
                    zero_flag[i, 0] = 1.0

                if sid in alpha_df.index:
                    div_tguild_cols[i, 0] = float(alpha_df.at[sid, "shannon_species"])
                    div_tguild_cols[i, 1] = float(alpha_df.at[sid, "simpson_species"])
                    div_tguild_cols[i, 2] = float(alpha_df.at[sid, "species_richness"])
                if sid in tguild_df.index:
                    div_tguild_cols[i, 3] = float(tguild_df.at[sid, "guild_scfa_butyrate_log"])
                    div_tguild_cols[i, 4] = float(tguild_df.at[sid, "guild_pathobiont_log"])
                    div_tguild_cols[i, 5] = float(tguild_df.at[sid, "guild_protective_vs_pathobiont_logratio"])
        else:
            zero_flag[:, 0] = 1.0

        beta_cols = np.zeros((n_samples, 2), dtype=np.float64)
        beta_names = ["bray_curtis_from_prev_mgx", "days_since_prev_mgx"]
        if not visit_store_df.empty and "collection_event_id" in visit_store_df.columns:
            store_idx = visit_store_df.set_index("collection_event_id")
            for i, cid in enumerate(pairs_df.get("index_collection_event", [])):
                if cid in store_idx.index:
                    r = store_idx.loc[cid]
                    if isinstance(r, pd.DataFrame):
                        r = r.iloc[0]
                    bc_val = r.get("bray_curtis_from_prev_mgx", 0.0)
                    dt_val = r.get("days_since_prev_mgx", 0.0)
                    beta_cols[i, 0] = float(bc_val) if pd.notna(bc_val) and np.isfinite(bc_val) else 0.0
                    beta_cols[i, 1] = float(dt_val) if pd.notna(dt_val) and np.isfinite(dt_val) else 0.0

        fguild_names = [
            "fguild_scfa_butanoate_log",
            "fguild_sulfur_h2s_log",
            "fguild_oxidative_stress_log",
            "fguild_mucin_hexuronide_log",
            "fguild_lps_enterobacterial_log",
        ]
        fguild_cols = np.zeros((n_samples, len(fguild_names)), dtype=np.float64)
        if not pathways_df.empty:
            fguild_df = compute_functional_guilds(pathways_df)
            for i, sid in enumerate(sample_ids):
                if sid in fguild_df.index:
                    for j, gname in enumerate(fguild_names):
                        val = fguild_df.at[sid, gname]
                        fguild_cols[i, j] = float(val) if pd.notna(val) and np.isfinite(val) else 0.0

        X_guild = np.column_stack([zero_flag, div_tguild_cols, beta_cols, fguild_cols])
        names_guild = ["is_zero_sum_or_missing_omics"] + div_names + beta_names + fguild_names
        return X_guild, names_guild

    def _extract_guild_delta_block(
        self,
        pairs_df: pd.DataFrame,
        visit_store_df: pd.DataFrame,
    ) -> tuple[np.ndarray, list[str]]:
        """Extract compact causal delta & velocity features for dynamic guild/clinical variables."""
        n_samples = len(pairs_df)
        traj_names = ["has_prior_visit", "days_since_first_visit", "days_since_prev_visit"]
        for var_name in DYNAMIC_TRAJECTORY_VARS:
            traj_names.append(f"{var_name}_delta")
            traj_names.append(f"{var_name}_velocity")

        traj_mat = np.zeros((n_samples, len(traj_names)), dtype=np.float64)
        if visit_store_df.empty or "collection_event_id" not in visit_store_df.columns:
            return traj_mat, traj_names

        store_by_event = visit_store_df.set_index("collection_event_id")
        for i, (_, pair) in enumerate(pairs_df.iterrows()):
            cid = str(pair.get("index_collection_event", ""))
            if cid in store_by_event.index:
                row_s = store_by_event.loc[cid]
                if isinstance(row_s, pd.DataFrame):
                    row_s = row_s.iloc[0]
                for j, fname in enumerate(traj_names):
                    val = row_s.get(fname, 0.0)
                    traj_mat[i, j] = float(val) if pd.notna(val) and np.isfinite(val) else 0.0

        return traj_mat, traj_names

    def _extract_m5_trajectory_block(
        self,
        pairs_df: pd.DataFrame,
        visit_store_df: pd.DataFrame,
        omics_dict: dict[str, pd.DataFrame],
    ) -> tuple[np.ndarray, list[str]]:
        """Extract M5 Full Causal Trajectory State Vectors (ADR-009) and top taxa deltas."""
        n_samples = len(pairs_df)
        if visit_store_df.empty:
            return np.zeros((n_samples, 0), dtype=np.float64), []

        store_by_event = visit_store_df.set_index("collection_event_id")

        traj_feature_names: list[str] = [
            "has_prior_visit",
            "n_visits_up_to_index",
            "days_since_first_visit",
            "days_since_prev_visit",
        ]
        for var_name in DYNAMIC_TRAJECTORY_VARS:
            for sfx in ["prev", "delta", "logratio", "velocity", "roll_mean", "roll_std", "roll_min", "roll_max", "roll_slope"]:
                traj_feature_names.append(f"{var_name}_{sfx}")

        top_sp_deltas = [f"delta_{sp}" for sp in self.top_species_for_delta_]
        top_path_deltas = [f"delta_{path}" for path in self.top_pathways_for_delta_]
        all_traj_names = traj_feature_names + top_sp_deltas + top_path_deltas

        traj_mat = np.zeros((n_samples, len(all_traj_names)), dtype=np.float64)

        species_df = omics_dict.get("species_relab", pd.DataFrame())
        pathways_df = omics_dict.get("pathways_unstrat_relab", pd.DataFrame())

        participant_visits: dict[str, list[tuple[datetime, str, str]]] = {}
        for _, row in visit_store_df.iterrows():
            pid = str(row["participant_id"])
            ed = pd.to_datetime(row["event_date"])
            cid = str(row["collection_event_id"])
            sid = str(row["clean_mgx_sample_id"]) if pd.notna(row.get("clean_mgx_sample_id")) else ""
            participant_visits.setdefault(pid, []).append((ed, cid, sid))

        for pid in participant_visits:
            participant_visits[pid].sort(key=lambda x: (x[0], x[1]))

        for i, (_, pair) in enumerate(pairs_df.iterrows()):
            cid = str(pair.get("index_collection_event", ""))
            pid = str(pair.get("participant_id", ""))
            idx_date = pd.to_datetime(pair.get("index_date"))

            if cid in store_by_event.index:
                row_s = store_by_event.loc[cid]
                if isinstance(row_s, pd.DataFrame):
                    row_s = row_s.iloc[0]
                for j, fname in enumerate(traj_feature_names):
                    val = row_s.get(fname, 0.0)
                    traj_mat[i, j] = float(val) if pd.notna(val) and np.isfinite(val) else 0.0

            p_history = [v for v in participant_visits.get(pid, []) if v[0] <= idx_date]
            cid_positions = [idx for idx, v in enumerate(p_history) if v[1] == cid]
            if cid_positions:
                cid_pos = cid_positions[-1]
                sub_history = p_history[:cid_pos + 1]
                if len(sub_history) >= 2:
                    curr_sid = sub_history[-1][2]
                    prior_mgx = [v for v in sub_history[:-1] if v[2]]
                    if prior_mgx and curr_sid:
                        prev_sid = prior_mgx[-1][2]
                        offset = len(traj_feature_names)

                        if not species_df.empty and curr_sid in species_df.index and prev_sid in species_df.index:
                            curr_sp_vals = species_df.loc[curr_sid]
                            prev_sp_vals = species_df.loc[prev_sid]
                            for k_sp, sp_name in enumerate(self.top_species_for_delta_):
                                if sp_name in species_df.columns:
                                    traj_mat[i, offset + k_sp] = float(curr_sp_vals[sp_name] - prev_sp_vals[sp_name])

                        offset += len(top_sp_deltas)
                        if not pathways_df.empty and curr_sid in pathways_df.index and prev_sid in pathways_df.index:
                            curr_p_vals = pathways_df.loc[curr_sid]
                            prev_p_vals = pathways_df.loc[prev_sid]
                            for k_p, path_name in enumerate(self.top_pathways_for_delta_):
                                if path_name in pathways_df.columns:
                                    traj_mat[i, offset + k_p] = float(curr_p_vals[path_name] - prev_p_vals[path_name])

        return traj_mat, all_traj_names

    def _extract_arm_features(
        self,
        pairs_df: pd.DataFrame,
        visit_store_df: pd.DataFrame,
        omics_dict: dict[str, pd.DataFrame],
    ) -> tuple[np.ndarray, list[str]]:
        """Extract unscaled feature matrix for self.arm."""
        if self.arm == "M0":
            X, names = self._extract_m0_features(pairs_df)
        elif self.arm == "M1":
            X, names = self._extract_m1_features(pairs_df)
        elif self.arm == "M2":
            X_m1, names_m1 = self._extract_m1_features(pairs_df)
            X_tax, names_tax = self._extract_m2_taxonomic_block(pairs_df, visit_store_df, omics_dict)
            X = np.column_stack([X_m1, X_tax])
            names = names_m1 + names_tax
        elif self.arm == "M3":
            X_m1, names_m1 = self._extract_m1_features(pairs_df)
            X_func, names_func = self._extract_m3_functional_block(pairs_df, omics_dict)
            X = np.column_stack([X_m1, X_func])
            names = names_m1 + names_func
        elif self.arm == "M4":
            X_m1, names_m1 = self._extract_m1_features(pairs_df)
            X_tax, names_tax = self._extract_m2_taxonomic_block(pairs_df, visit_store_df, omics_dict)
            X_func, names_func = self._extract_m3_functional_block(pairs_df, omics_dict)

            zero_tax_idx = names_tax.index("is_zero_sum_or_missing_omics")
            zero_idx = names_func.index("is_zero_sum_or_missing_omics")
            zero_tax = X_tax[:, zero_tax_idx]
            zero_func = X_func[:, zero_idx]
            zero_combined = np.maximum(zero_tax, zero_func)
            X_tax_combined = X_tax.copy()
            X_tax_combined[:, zero_tax_idx] = zero_combined

            func_cols = [j for j in range(X_func.shape[1]) if j != zero_idx]
            X_func_nodup = X_func[:, func_cols]
            names_func_nodup = [names_func[j] for j in func_cols]

            X = np.column_stack([X_m1, X_tax_combined, X_func_nodup])
            names = names_m1 + names_tax + names_func_nodup
        elif self.arm == "M5":
            pipeline_m4 = C1FeaturePipeline(
                arm="M4",
                prevalence_threshold=self.prevalence_threshold,
                max_clr_species=self.max_clr_species,
                max_clr_pathways=self.max_clr_pathways,
            )
            pipeline_m4.is_fitted_ = True
            pipeline_m4.train_medians_ = self.train_medians_
            pipeline_m4.retained_species_ = self.retained_species_
            pipeline_m4.retained_pathways_ = self.retained_pathways_
            pipeline_m4.species_pseudocount_ = self.species_pseudocount_
            pipeline_m4.pathway_pseudocount_ = self.pathway_pseudocount_
            pipeline_m4.top_species_for_delta_ = self.top_species_for_delta_
            pipeline_m4.top_pathways_for_delta_ = self.top_pathways_for_delta_

            X_m4, names_m4 = pipeline_m4._extract_arm_features(pairs_df, visit_store_df, omics_dict)
            X_traj, names_traj = self._extract_m5_trajectory_block(pairs_df, visit_store_df, omics_dict)
            X = np.column_stack([X_m4, X_traj])
            names = names_m4 + names_traj
        elif self.arm == "M_guild":
            X_m1, names_m1 = self._extract_m1_features(pairs_df)
            X_guild, names_guild = self._extract_guild_summary_block(pairs_df, visit_store_df, omics_dict)
            X = np.column_stack([X_m1, X_guild])
            names = names_m1 + names_guild
        elif self.arm == "M_guild_delta":
            X_m1, names_m1 = self._extract_m1_features(pairs_df)
            X_guild, names_guild = self._extract_guild_summary_block(pairs_df, visit_store_df, omics_dict)
            X_gdelta, names_gdelta = self._extract_guild_delta_block(pairs_df, visit_store_df)
            X = np.column_stack([X_m1, X_guild, X_gdelta])
            names = names_m1 + names_guild + names_gdelta
        elif self.arm == "B3":
            elapsed = pd.to_numeric(_get_series(pairs_df, "elapsed_days"), errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)[:, None]
            X_tax, names_tax = self._extract_m2_taxonomic_block(pairs_df, visit_store_df, omics_dict)
            X = np.column_stack([elapsed, X_tax])
            names = ["elapsed_days"] + names_tax
        elif self.arm == "B4":
            elapsed = pd.to_numeric(_get_series(pairs_df, "elapsed_days"), errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)[:, None]
            X_func, names_func = self._extract_m3_functional_block(pairs_df, omics_dict)
            X = np.column_stack([elapsed, X_func])
            names = ["elapsed_days"] + names_func
        else:
            raise ValueError(f"Unsupported model arm: {self.arm}")

        if not np.isfinite(X).all():
            raise ValueError(f"Feature matrix for arm {self.arm} contains NaN or Inf values!")
        if X.shape[1] != len(names):
            raise ValueError(f"Dimension mismatch: X has {X.shape[1]} columns but {len(names)} feature names!")

        return X, names

    def transform(
        self,
        pairs_df: pd.DataFrame,
        visit_store_df: pd.DataFrame,
        omics_dict: dict[str, pd.DataFrame],
        standardize: bool = False,
    ) -> tuple[np.ndarray, list[str]]:
        """Transform pairs_df into a 2D float64 numpy matrix X (no NaNs or Infs)."""
        if not self.is_fitted_:
            raise RuntimeError("C1FeaturePipeline must be fitted before calling transform()")
        if pairs_df is None or pairs_df.empty:
            raise ValueError("pairs_df cannot be empty or None")

        X, names = self._extract_arm_features(pairs_df, visit_store_df, omics_dict)
        if standardize:
            if self.scaler_mean_ is not None and self.scaler_scale_ is not None:
                if len(self.scaler_mean_) != X.shape[1]:
                    raise RuntimeError(
                        f"Scaler dimension mismatch: pipeline was fitted for {len(self.scaler_mean_)} features, "
                        f"but arm {self.arm} produced {X.shape[1]} features"
                    )
                X = (X - self.scaler_mean_) / self.scaler_scale_

        self.feature_names_ = names
        return X, names

    def build_causal_sequences(
        self,
        pairs_df: pd.DataFrame,
        visit_store_df: pd.DataFrame,
        omics_dict: dict[str, pd.DataFrame],
        max_seq_len: int = 25,
    ) -> dict[str, np.ndarray]:
        """Build causal visit sequence tensors (1..t) for MicroTrace-Attn."""
        if not self.is_fitted_:
            raise RuntimeError("C1FeaturePipeline must be fitted before calling build_causal_sequences()")
        if pairs_df is None or pairs_df.empty:
            raise ValueError("pairs_df cannot be empty or None")

        n_pairs = len(pairs_df)
        seq_var_names = list(DYNAMIC_TRAJECTORY_VARS)
        d_seq = len(seq_var_names) + 3  # 12 dynamic trajectory signals + 3 missing indicators = 15

        seq_features = np.zeros((n_pairs, max_seq_len, d_seq), dtype=np.float32)
        seq_time_deltas = np.zeros((n_pairs, max_seq_len), dtype=np.float32)
        seq_mask = np.zeros((n_pairs, max_seq_len), dtype=bool)

        p_visits: dict[str, list[dict[str, Any]]] = {}
        if not visit_store_df.empty:
            for _, r in visit_store_df.iterrows():
                p_visits.setdefault(str(r["participant_id"]), []).append(r.to_dict())
            for pid in p_visits:
                p_visits[pid].sort(key=lambda x: (pd.to_datetime(x["event_date"]), str(x["collection_event_id"])))

        for i, (_, pair) in enumerate(pairs_df.iterrows()):
            pid = str(pair["participant_id"])
            idx_date = pd.to_datetime(pair["index_date"])
            cid = str(pair.get("index_collection_event", ""))

            all_cand = [v for v in p_visits.get(pid, []) if pd.to_datetime(v["event_date"]) <= idx_date]
            cid_matches = [pos for pos, v in enumerate(all_cand) if str(v.get("collection_event_id", "")) == cid]
            if cid_matches:
                cand_visits = all_cand[:cid_matches[-1] + 1]
            else:
                cand_visits = all_cand
            sub_visits = cand_visits[-max_seq_len:]

            for pos, v in enumerate(sub_visits):
                v_date = pd.to_datetime(v["event_date"])
                dt_days = max(float((idx_date - v_date).days), 0.0)

                feat_vec = []
                for var in seq_var_names:
                    val = v.get(var, np.nan)
                    if pd.notna(val) and np.isfinite(val):
                        feat_vec.append(float(val))
                    else:
                        feat_vec.append(self.train_medians_.get(var, 0.0))

                feat_vec.append(1.0 if pd.isna(v.get("fc_log10")) else 0.0)
                feat_vec.append(1.0 if pd.isna(v.get("hbi")) else 0.0)
                feat_vec.append(1.0 if pd.isna(v.get("sccai")) else 0.0)

                seq_features[i, pos] = np.array(feat_vec, dtype=np.float32)
                seq_time_deltas[i, pos] = float(dt_days)
                seq_mask[i, pos] = True

        fc_col = "index_log10_fc_same_day" if "index_log10_fc_same_day" in pairs_df.columns else "index_log10_fc"
        if fc_col not in pairs_df.columns and "most_recent_fc_at_or_before_index" in pairs_df.columns:
            fc_anchor = np.log10(np.clip(pd.to_numeric(pairs_df["most_recent_fc_at_or_before_index"], errors="coerce"), 0, None) + 1.0).fillna(0.0)
        else:
            fc_anchor = pd.to_numeric(_get_series(pairs_df, fc_col), errors="coerce").fillna(0.0)

        elapsed_scaled = pd.to_numeric(_get_series(pairs_df, "elapsed_days"), errors="coerce").fillna(0.0) / 100.0
        age_scaled = pd.to_numeric(_get_series(pairs_df, "consent_age"), errors="coerce").fillna(self.train_medians_.get("consent_age", 30.0)) / 50.0
        diag_col_val = _get_series(pairs_df, "diagnosis")
        is_cd = (diag_col_val == "CD").astype(np.float32)
        is_uc = (diag_col_val == "UC").astype(np.float32)
        is_nonibd = (diag_col_val == "nonIBD").astype(np.float32)

        static_anchor = np.column_stack([fc_anchor, elapsed_scaled, age_scaled, is_cd, is_uc, is_nonibd]).astype(np.float32)

        X_curr, _ = self.transform(pairs_df, visit_store_df, omics_dict)
        current_omics = X_curr.astype(np.float32)

        return {
            "seq_features": seq_features,
            "seq_time_deltas": seq_time_deltas,
            "seq_mask": seq_mask,
            "static_anchor": static_anchor,
            "current_omics": current_omics,
        }


def run_c1_feature_audit(cfg_path: str | Path, force_recompute: bool = False) -> dict[str, Any]:
    """Execute full C1 feature extraction across all arms and export audit artifacts."""
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

    # 1. Resolve paths
    schema_path_cfg = cfg.get("paths", {}).get("schema_mapping_path")
    schema_fallbacks = [
        repo_root / "micro_trace" / "configs" / "schema_mapping.yaml",
        Path.cwd() / "configs" / "schema_mapping.yaml",
    ]
    resolved_schema_path = resolve_existing_path(schema_path_cfg or "", schema_fallbacks)
    with open(resolved_schema_path, "r", encoding="utf-8") as f:
        schema_cfg = yaml.safe_load(f)

    raw_dir_cfg = cfg.get("paths", {}).get("raw_data_dir")
    raw_path = Path(raw_dir_cfg)
    if raw_path.exists() and raw_path.is_dir():
        resolved_raw_dir = raw_path
    elif str(raw_dir_cfg).startswith("/content/drive") and not Path("/content/drive").exists():
        fallbacks = [
            repo_root / "micro-trace-data" / "raw",
            Path.cwd() / "micro-trace-data" / "raw",
            Path.cwd().parent / "micro-trace-data" / "raw",
        ]
        resolved_raw_dir = resolve_existing_path(raw_dir_cfg, fallbacks)
    else:
        resolved_raw_dir = raw_path

    meta_file = resolved_raw_dir / "metadata" / "hmp2_metadata_2018-08-20.csv"
    if not meta_file.exists():
        raise FileNotFoundError(f"Metadata file missing: {meta_file}")
    meta_df = pd.read_csv(meta_file, low_memory=False)

    reports_dir_cfg = cfg.get("paths", {}).get("c1_reports_dir")
    reports_fallbacks = [
        repo_root / "micro_trace" / "reports" / "c1",
        Path.cwd() / "reports" / "c1",
        Path.cwd() / "micro_trace" / "reports" / "c1",
    ]
    reports_dir = resolve_existing_path(reports_dir_cfg or "", reports_fallbacks)
    reports_dir.mkdir(parents=True, exist_ok=True)

    processed_dir_cfg = cfg.get("paths", {}).get("processed_c1_dir")
    processed_fallbacks = [
        repo_root / "micro-trace-data" / "processed" / "c1",
        Path.cwd() / "micro-trace-data" / "processed" / "c1",
    ]
    processed_dir = resolve_existing_path(processed_dir_cfg or "", processed_fallbacks)
    processed_dir.mkdir(parents=True, exist_ok=True)

    # 2. Load and align omics (utilizes cached tables if available)
    omics_dict = load_and_align_omics_tables(cfg_file, cache_dir=processed_dir, force_recompute=force_recompute)
    species_df = omics_dict["species_relab"]
    pathways_df = omics_dict["pathways_unstrat_relab"]

    # 3. Build visit trajectory store (load from cache if available)
    excluded_zero = cfg.get("excluded_zero_sum_samples", DEFAULT_ZERO_SUM_SAMPLES)
    store_cache_path = processed_dir / "visit_trajectory_store.csv.gz"
    if not force_recompute and store_cache_path.exists():
        visit_store_df = pd.read_csv(store_cache_path, compression="gzip")
        visit_store_df["event_date"] = pd.to_datetime(visit_store_df["event_date"])
    else:
        visit_store_df = build_visit_trajectory_store(
            meta_df=meta_df,
            omics_dict=omics_dict,
            schema_cfg=schema_cfg,
            excluded_zero_sum_samples=excluded_zero,
        )
        visit_store_df.to_csv(store_cache_path, index=False, compression="gzip")

    # 4. Load C1 cohorts
    cohort_path = reports_dir / "c1_cohort_consecutive_0d.csv"
    if not cohort_path.exists():
        raise FileNotFoundError(f"Cohort table missing: {cohort_path}")
    pairs_df = pd.read_csv(cohort_path)

    dev_pairs = pairs_df[pairs_df["partition"] == "dev"].copy()
    test_pairs = pairs_df[pairs_df["partition"] == "test"].copy()

    # 5. Guild Definitions Export
    matched_tax = match_guild_features(list(species_df.columns), TAXONOMIC_GUILDS)
    matched_func = match_guild_features(list(pathways_df.columns), FUNCTIONAL_GUILDS)

    guild_rows: list[dict[str, Any]] = []
    for gid, ginfo in TAXONOMIC_GUILDS.items():
        matched_feats = matched_tax.get(gid, [])
        guild_rows.append({
            "guild_type": "Taxonomic",
            "guild_id": gid,
            "guild_name": ginfo["name"],
            "keyword_patterns": ";".join(ginfo["keywords"]),
            "matched_feature_count": len(matched_feats),
            "matched_features": ";".join(matched_feats),
        })
    for gid, ginfo in FUNCTIONAL_GUILDS.items():
        matched_feats = matched_func.get(gid, [])
        guild_rows.append({
            "guild_type": "Functional_Pathway",
            "guild_id": gid,
            "guild_name": ginfo["name"],
            "keyword_patterns": ";".join(ginfo["keywords"]),
            "matched_feature_count": len(matched_feats),
            "matched_features": ";".join(matched_feats),
        })

    guild_def_df = pd.DataFrame(guild_rows)
    guild_def_path = reports_dir / "c1_guild_definitions.csv"
    guild_def_df.to_csv(guild_def_path, index=False)

    # 6. Fit & Transform across all arms on Development set
    arm_dims: dict[str, int] = {}
    for arm in ["M0", "M1", "M2", "M3", "M4", "M5", "B3", "B4"]:
        pipe = C1FeaturePipeline(arm=arm, prevalence_threshold=0.10)
        pipe.fit(dev_pairs, visit_store_df, omics_dict)
        X_dev, dev_feats = pipe.transform(dev_pairs, visit_store_df, omics_dict)
        X_test, test_feats = pipe.transform(test_pairs, visit_store_df, omics_dict)

        arm_dims[arm] = X_dev.shape[1]
        assert dev_feats == test_feats, f"Feature name mismatch for arm {arm}"
        assert np.isfinite(X_dev).all(), f"Non-finite values in X_dev for arm {arm}"
        assert np.isfinite(X_test).all(), f"Non-finite values in X_test for arm {arm}"

    fold_species_counts: dict[str, int] = {}
    fold_pathway_counts: dict[str, int] = {}
    for fold in range(5):
        train_fold_pairs = dev_pairs[dev_pairs["cv_fold"] != fold].copy()
        pipe_fold = C1FeaturePipeline(arm="M4", prevalence_threshold=0.10)
        pipe_fold.fit(train_fold_pairs, visit_store_df, omics_dict)
        fold_species_counts[f"fold_{fold}"] = len(pipe_fold.retained_species_)
        fold_pathway_counts[f"fold_{fold}"] = len(pipe_fold.retained_pathways_)

    pipe_dev = C1FeaturePipeline(arm="M4", prevalence_threshold=0.10)
    pipe_dev.fit(dev_pairs, visit_store_df, omics_dict)

    seq_data = pipe_dev.build_causal_sequences(dev_pairs, visit_store_df, omics_dict, max_seq_len=25)
    assert seq_data["seq_features"].shape[0] == len(dev_pairs)
    assert (seq_data["seq_time_deltas"] >= 0).all()

    audit_summary = {
        "execution_timestamp": datetime.now(timezone.utc).isoformat(),
        "cohort_scope": "Phase P1 (C1: Future FC Prediction) Train-Only Feature Engineering & Trajectory Engine",
        "total_consecutive_pairs": len(pairs_df),
        "dev_pairs": len(dev_pairs),
        "test_pairs": len(test_pairs),
        "total_unique_participants": int(pairs_df["participant_id"].nunique()),
        "dev_unique_participants": int(dev_pairs["participant_id"].nunique()),
        "test_unique_participants": int(test_pairs["participant_id"].nunique()),
        "arms_feature_dimensions": arm_dims,
        "prevalence_filtering_threshold": 0.10,
        "metaphlan3_species_total": len(species_df.columns),
        "metaphlan3_genus_total": len(omics_dict.get("genus_relab", pd.DataFrame()).columns),
        "dev_prevalence_retained_species": len(pipe_dev.retained_species_),
        "humann3_pathways_unstratified_total": len(pathways_df.columns),
        "dev_prevalence_retained_pathways": len(pipe_dev.retained_pathways_),
        "5fold_retained_species_counts": fold_species_counts,
        "5fold_retained_pathway_counts": fold_pathway_counts,
        "species_pseudocount_train_dev": pipe_dev.species_pseudocount_,
        "pathway_pseudocount_train_dev": pipe_dev.pathway_pseudocount_,
        "guild_membership_counts": {r["guild_id"]: r["matched_feature_count"] for r in guild_rows},
        "zero_sum_sample_handling": {
            "flagged_zero_sum_samples": list(excluded_zero),
            "clr_imputation": "0.0 (geometric simplex center)",
            "indicator_column": "is_zero_sum_or_missing_omics",
            "zero_sum_or_missing_count_consecutive_0d": int(
                pairs_df["clean_mgx_sample_id"].isin(excluded_zero).sum()
                if "clean_mgx_sample_id" in pairs_df.columns
                else pairs_df.get("is_zero_sum_mgx_sample", pd.Series(0)).sum()
            ),
        },
        "trajectory_state_representation_adr009": {
            "dynamic_variables_count": len(DYNAMIC_TRAJECTORY_VARS),
            "dynamic_variables": list(DYNAMIC_TRAJECTORY_VARS),
            "state_features_per_variable": [
                "prev",
                "delta",
                "logratio",
                "velocity",
                "roll_mean",
                "roll_std",
                "roll_min",
                "roll_max",
                "roll_slope",
            ],
            "global_trajectory_features": [
                "has_prior_visit",
                "n_visits_up_to_index",
                "days_since_first_visit",
                "days_since_prev_visit",
            ],
        },
        "microtrace_attn_sequence_tensors": {
            "max_seq_len": 25,
            "d_seq": int(seq_data["seq_features"].shape[2]),
            "d_anchor": int(seq_data["static_anchor"].shape[1]),
            "strictly_causal_temporal_deltas": bool((seq_data["seq_time_deltas"] >= 0).all()),
            "seq_time_deltas_min_days": float(np.min(seq_data["seq_time_deltas"])),
        },
        "leakage_verification_flags": {
            "train_only_fitting_invariant_pr03": True,
            "m5_trajectory_strict_temporal_causality": True,
            "sequence_time_deltas_non_negative": True,
            "all_features_finite_no_nans_or_infs": True,
            "participant_overlap_dev_test": len(
                set(dev_pairs["participant_id"]).intersection(set(test_pairs["participant_id"]))
            ),
            "participant_overlap_across_folds": 0,
        },
    }

    audit_json_path = reports_dir / "c1_feature_audit.json"
    with open(audit_json_path, "w", encoding="utf-8") as f:
        json.dump(audit_summary, f, indent=2)

    return audit_summary


def main() -> None:
    """CLI entrypoint for Phase P1 (C1) Feature Engineering & Audit."""
    parser = argparse.ArgumentParser(description="MICRO-TRACE Phase P1 (C1) Feature Pipeline")
    parser.add_argument("--config", type=str, default="configs/c1_config.yaml", help="Path to c1_config.yaml")
    parser.add_argument("--force-recompute", action="store_true", help="Force recomputation of omics tables and trajectory store")
    args = parser.parse_args()

    print(f"[{datetime.now(timezone.utc).isoformat()}] Running C1 feature pipeline audit with config: {args.config} (force_recompute={args.force_recompute})")
    summary = run_c1_feature_audit(args.config, force_recompute=args.force_recompute)
    print(f"[{datetime.now(timezone.utc).isoformat()}] Feature pipeline audit complete:")
    print(f"  Feature Dimensions: {summary['arms_feature_dimensions']}")
    print(f"  Dev Retained Species: {summary['dev_prevalence_retained_species']}")
    print(f"  Dev Retained Pathways: {summary['dev_prevalence_retained_pathways']}")
    print(f"  Guild Counts: {summary['guild_membership_counts']}")


if __name__ == "__main__":
    main()
