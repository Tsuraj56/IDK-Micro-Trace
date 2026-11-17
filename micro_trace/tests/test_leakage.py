"""Tests 3, 4, 9: Future FC cannot be used as index feature; post-index dates rejected; participant split disjointness."""
import pandas as pd
from micro_trace.longitudinal import build_longitudinal_audit

def test_03_future_fc_cannot_be_used_as_index_feature(base_schema_cfg):
    """Req 3: If index visit has NO FC and the only FC for the participant is in the future, index_fc and most_recent_fc_at_or_before_index must be None/False."""
    meta = pd.DataFrame([
        {"Participant ID": "P1", "External ID": "S1", "site_sub_coll": "P1C1", "visit_num": 1, "week_num": 0, "data_type": "metagenomics", "date_of_receipt": "2020-01-01", "diagnosis": "CD", "fecalcal": None, "fecalcal_ng_ml": None, "site_name": "MGH"},
        {"Participant ID": "P1", "External ID": "S2", "site_sub_coll": "P1C2", "visit_num": 2, "week_num": 2, "data_type": "metagenomics", "date_of_receipt": "2020-01-15", "diagnosis": "CD", "fecalcal": 150.0, "fecalcal_ng_ml": None, "site_name": "MGH"},
    ])
    cov = pd.DataFrame([
        {"collection_event_id": "P1C1", "has_mgx_tax_v2": True, "has_mgx_path_v2": True},
        {"collection_event_id": "P1C2", "has_mgx_tax_v2": True, "has_mgx_path_v2": True},
    ])
    match_df, _, pairs_df, _, _ = build_longitudinal_audit(meta, cov, base_schema_cfg, [0, 14, 30], [], {})
    # At P1C1 (2020-01-01), the future FC on 2020-01-15 must NEVER be matched as an index feature even under a 30-day window
    m_c1 = match_df[match_df["collection_event_id"] == "P1C1"].iloc[0]
    assert pd.isna(m_c1["nearest_prior_or_same_fc_val"])
    assert m_c1["matched_causal_within_30d"] == False
    # And in pairs_df, P1C1 -> P1C2 has no prior/same index FC
    p_row = pairs_df.iloc[0]
    assert p_row["has_same_day_index_fc"] == False
    assert p_row["has_prior_or_same_fc"] == False
    assert pd.isna(p_row["most_recent_fc_at_or_before_index"])

def test_04_date_after_future_observation_rejected_as_input_feature(base_schema_cfg):
    """Req 4: A visit occurring on or after a candidate outcome date is never paired as an index visit for that outcome."""
    meta = pd.DataFrame([
        {"Participant ID": "P1", "External ID": "S1", "site_sub_coll": "P1C1", "visit_num": 1, "week_num": 0, "data_type": "metagenomics", "date_of_receipt": "2020-01-10", "diagnosis": "CD", "fecalcal": 50.0, "fecalcal_ng_ml": None, "site_name": "MGH"},
        {"Participant ID": "P1", "External ID": "S2", "site_sub_coll": "P1C2", "visit_num": 2, "week_num": 2, "data_type": "metagenomics", "date_of_receipt": "2020-01-20", "diagnosis": "CD", "fecalcal": 90.0, "fecalcal_ng_ml": None, "site_name": "MGH"},
    ])
    cov = pd.DataFrame([
        {"collection_event_id": "P1C1", "has_mgx_tax_v2": True, "has_mgx_path_v2": True},
        {"collection_event_id": "P1C2", "has_mgx_tax_v2": True, "has_mgx_path_v2": True},
    ])
    _, _, pairs_df, _, _ = build_longitudinal_audit(meta, cov, base_schema_cfg, [0], [], {})
    # S2 (2020-01-20) can never be used as index to predict S1 (2020-01-10) or itself (2020-01-20)
    assert all(pairs_df["elapsed_days"] > 0)
    assert all(pd.to_datetime(pairs_df["index_date"]) < pd.to_datetime(pairs_df["future_date"]))
    assert not any(pairs_df["index_collection_event"] == "P1C2")

def test_09_participants_not_split_across_partitions(base_schema_cfg):
    """Req 9: Participants cannot be split across train/calibration/test partitions."""
    rows = []
    cov_rows = []
    for i in range(1, 13):
        pid = f"P{i}"
        diag = "CD" if i <= 6 else "UC"
        for v, dt in [(1, "2020-01-01"), (2, "2020-01-15"), (3, "2020-02-01")]:
            cid = f"{pid}C{v}"
            rows.append({
                "Participant ID": pid, "External ID": f"{cid}_S", "site_sub_coll": cid,
                "visit_num": v, "week_num": (v-1)*2, "data_type": "metagenomics",
                "date_of_receipt": dt, "diagnosis": diag, "fecalcal": 50.0 + v*10,
                "fecalcal_ng_ml": None, "site_name": "MGH"
            })
            cov_rows.append({"collection_event_id": cid, "has_mgx_tax_v2": True, "has_mgx_path_v2": True})
    meta = pd.DataFrame(rows)
    cov = pd.DataFrame(cov_rows)
    _, _, _, _, split_df = build_longitudinal_audit(
        meta, cov, base_schema_cfg, [0], [], {"random_seed": 42, "train_frac": 0.7, "calib_frac": 0.15, "test_frac": 0.15}
    )
    tr = set(split_df[split_df["partition"] == "train"]["participant_id"])
    ca = set(split_df[split_df["partition"] == "calibration"]["participant_id"])
    te = set(split_df[split_df["partition"] == "test"]["participant_id"])
    assert len(tr & ca) == 0
    assert len(tr & te) == 0
    assert len(ca & te) == 0
    assert len(tr | ca | te) == 12
