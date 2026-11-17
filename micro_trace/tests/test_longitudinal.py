"""Tests 1 & 2: Two visits -> 1 pair; Three visits -> 3 all-future pairs and 2 consecutive pairs."""
import pandas as pd
from micro_trace.longitudinal import build_longitudinal_audit

def test_01_participant_with_two_valid_visits_generates_one_pair(base_schema_cfg):
    """Req 1: A participant with two valid visits generates exactly one future pair."""
    meta = pd.DataFrame([
        {"Participant ID": "P1", "External ID": "S1", "site_sub_coll": "P1C1", "visit_num": 1, "week_num": 0, "data_type": "metagenomics", "date_of_receipt": "2020-01-01", "diagnosis": "CD", "fecalcal": 40.0, "fecalcal_ng_ml": None, "site_name": "MGH"},
        {"Participant ID": "P1", "External ID": "S2", "site_sub_coll": "P1C2", "visit_num": 2, "week_num": 2, "data_type": "metagenomics", "date_of_receipt": "2020-01-15", "diagnosis": "CD", "fecalcal": 120.0, "fecalcal_ng_ml": None, "site_name": "MGH"},
    ])
    cov = pd.DataFrame([
        {"collection_event_id": "P1C1", "has_mgx_tax_v2": True, "has_mgx_path_v2": True},
        {"collection_event_id": "P1C2", "has_mgx_tax_v2": True, "has_mgx_path_v2": True},
    ])
    _, _, pairs_df, _, _ = build_longitudinal_audit(
        meta, cov, base_schema_cfg, [0, 7], [{"name": "biweekly", "min_days": 7, "max_days": 21}], {}
    )
    assert len(pairs_df) == 1
    row = pairs_df.iloc[0]
    assert row["index_collection_event"] == "P1C1"
    assert row["future_collection_event"] == "P1C2"
    assert row["elapsed_days"] == 14
    assert row["in_biweekly"] == True

def test_02_participant_with_three_visits_generates_correct_pairs(base_schema_cfg):
    """Req 2: A participant with 3 valid visits at days 0, 14, 42 generates 3 total causal pairs, 2 consecutive pairs, and 2 pairs in 7-30d horizon."""
    meta = pd.DataFrame([
        {"Participant ID": "P1", "External ID": "S1", "site_sub_coll": "P1C1", "visit_num": 1, "week_num": 0, "data_type": "metagenomics", "date_of_receipt": "2020-01-01", "diagnosis": "UC", "fecalcal": 30.0, "fecalcal_ng_ml": None, "site_name": "MGH"},
        {"Participant ID": "P1", "External ID": "S2", "site_sub_coll": "P1C2", "visit_num": 2, "week_num": 2, "data_type": "metagenomics", "date_of_receipt": "2020-01-15", "diagnosis": "UC", "fecalcal": 80.0, "fecalcal_ng_ml": None, "site_name": "MGH"},
        {"Participant ID": "P1", "External ID": "S3", "site_sub_coll": "P1C3", "visit_num": 3, "week_num": 6, "data_type": "metagenomics", "date_of_receipt": "2020-02-12", "diagnosis": "UC", "fecalcal": 200.0, "fecalcal_ng_ml": None, "site_name": "MGH"},
    ])
    cov = pd.DataFrame([
        {"collection_event_id": "P1C1", "has_mgx_tax_v2": True, "has_mgx_path_v2": True},
        {"collection_event_id": "P1C2", "has_mgx_tax_v2": True, "has_mgx_path_v2": True},
        {"collection_event_id": "P1C3", "has_mgx_tax_v2": True, "has_mgx_path_v2": True},
    ])
    _, _, pairs_df, _, _ = build_longitudinal_audit(
        meta, cov, base_schema_cfg, [0], [{"name": "h7_30", "min_days": 7, "max_days": 30}], {}
    )
    assert len(pairs_df) == 3  # (C1->C2: 14d), (C1->C3: 42d), (C2->C3: 28d)
    assert int(pairs_df["is_next_available_fc_from_index"].sum()) == 2  # (C1->C2) and (C2->C3)
    assert int(pairs_df["in_h7_30"].sum()) == 2  # 14d and 28d are in [7, 30]; 42d is False
