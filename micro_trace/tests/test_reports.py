"""Test report generation and cross-table metric consistency."""
import pandas as pd
from micro_trace.clinical import audit_participants_and_fc
from micro_trace.longitudinal import build_longitudinal_audit
from micro_trace.feasibility import evaluate_feasibility

def test_report_metrics_match_source_tables(base_schema_cfg):
    """Verify that evaluate_feasibility summary counts agree with participant_summary and longitudinal_pairs."""
    meta = pd.DataFrame([
        {"Participant ID": "P1", "External ID": "S1", "site_sub_coll": "P1C1", "visit_num": 1, "week_num": 0, "data_type": "metagenomics", "date_of_receipt": "2020-01-01", "diagnosis": "CD", "fecalcal": 25.0, "fecalcal_ng_ml": None, "site_name": "MGH"},
        {"Participant ID": "P1", "External ID": "S2", "site_sub_coll": "P1C2", "visit_num": 2, "week_num": 2, "data_type": "metagenomics", "date_of_receipt": "2020-01-15", "diagnosis": "CD", "fecalcal": 75.0, "fecalcal_ng_ml": None, "site_name": "MGH"},
        {"Participant ID": "P2", "External ID": "S3", "site_sub_coll": "P2C1", "visit_num": 1, "week_num": 0, "data_type": "metagenomics", "date_of_receipt": "2020-01-05", "diagnosis": "UC", "fecalcal": 100.0, "fecalcal_ng_ml": None, "site_name": "MGH"},
    ])
    cov = pd.DataFrame([
        {"collection_event_id": "P1C1", "has_mgx_tax_v2": True, "has_mgx_path_v2": True},
        {"collection_event_id": "P1C2", "has_mgx_tax_v2": True, "has_mgx_path_v2": True},
        {"collection_event_id": "P2C1", "has_mgx_tax_v2": True, "has_mgx_path_v2": True},
    ])
    part_df, fc_df, issues = audit_participants_and_fc(meta, base_schema_cfg, {})
    match_df, gap_df, pairs_df, flow_df, split_df = build_longitudinal_audit(
        meta, cov, base_schema_cfg, [0], [{"name": "monthly_14_56d", "min_days": 14, "max_days": 56}], {}
    )
    metrics = evaluate_feasibility(
        pd.DataFrame({"size_mb": [1.0]}), pd.DataFrame({"non_null_total": [3]}),
        pd.DataFrame(), part_df, fc_df, pd.DataFrame(), pd.DataFrame(), pairs_df, flow_df, split_df, issues
    )
    assert metrics["n_unique_participants"] == 2
    assert metrics["n_participants_repeated_mgx_and_fc"] == 1
    assert metrics["n_pairs_sameday_index_fc_next_visit"] == 1
