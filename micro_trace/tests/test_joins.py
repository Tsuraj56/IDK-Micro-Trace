"""Tests 5, 6, 7, 8: Duplicate sample keys, many-to-many join flags, missing dates, unmatched records."""
import pandas as pd
from micro_trace.integrity import audit_identifiers_and_joins
from micro_trace.clinical import audit_participants_and_fc

def test_05_duplicate_sample_keys_detected(base_schema_cfg):
    """Req 5: Duplicate sample keys within a modality or omics table are detected."""
    meta = pd.DataFrame([
        {"Participant ID": "P1", "External ID": "S1", "site_sub_coll": "P1C1", "visit_num": 1, "week_num": 0, "data_type": "metagenomics", "date_of_receipt": "2020-01-01", "diagnosis": "CD", "fecalcal": 50.0, "fecalcal_ng_ml": None, "site_name": "MGH"},
        {"Participant ID": "P1", "External ID": "S1", "site_sub_coll": "P1C1", "visit_num": 1, "week_num": 0, "data_type": "metagenomics", "date_of_receipt": "2020-01-01", "diagnosis": "CD", "fecalcal": 50.0, "fecalcal_ng_ml": None, "site_name": "MGH"},
    ])
    omics_stats = {
        "mgx_tax": {"exists": True, "assay": "MGX_WGS", "modality": "taxonomy", "tool": "MetaPhlAn2", "sample_ids": ["S1", "S1"], "n_features": 10}
    }
    _, issues = audit_identifiers_and_joins(meta, base_schema_cfg, omics_stats)
    cats = [i["category"] for i in issues]
    assert "duplicate_sample_within_datatype" in cats
    assert "duplicate_omics_column" in cats

def test_06_many_to_many_joins_flagged(base_schema_cfg):
    """Req 6: Shared External ID across omics layers is flagged as a many-to-many join risk."""
    meta = pd.DataFrame([
        {"Participant ID": "P1", "External ID": "SHARED_ID", "site_sub_coll": "P1C1", "visit_num": 1, "week_num": 0, "data_type": "metagenomics", "date_of_receipt": "2020-01-01", "diagnosis": "CD", "fecalcal": 50.0, "fecalcal_ng_ml": None, "site_name": "MGH"},
        {"Participant ID": "P1", "External ID": "SHARED_ID", "site_sub_coll": "P1C1", "visit_num": 1, "week_num": 0, "data_type": "metabolomics", "date_of_receipt": "2020-01-01", "diagnosis": "CD", "fecalcal": 50.0, "fecalcal_ng_ml": None, "site_name": "MGH"},
    ])
    _, issues = audit_identifiers_and_joins(meta, base_schema_cfg, {})
    cats = [i["category"] for i in issues]
    assert "many_to_many_join_risk" in cats

def test_07_missing_dates_reported(base_schema_cfg):
    """Req 7: Missing dates are explicitly reported in quality issues."""
    meta = pd.DataFrame([
        {"Participant ID": "P1", "External ID": "S1", "site_sub_coll": "P1C1", "visit_num": 1, "week_num": 0, "data_type": "biopsy_16S", "date_of_receipt": None, "diagnosis": "CD", "fecalcal": None, "fecalcal_ng_ml": None, "site_name": "MGH"},
    ])
    _, _, issues = audit_participants_and_fc(meta, base_schema_cfg, {})
    cats = [i["category"] for i in issues]
    assert "missing_dates" in cats

def test_08_unmatched_microbiome_and_clinical_records_reported(base_schema_cfg):
    """Req 8: Unmatched microbiome sample IDs not present in metadata are reported."""
    meta = pd.DataFrame([
        {"Participant ID": "P1", "External ID": "S1", "site_sub_coll": "P1C1", "visit_num": 1, "week_num": 0, "data_type": "metagenomics", "date_of_receipt": "2020-01-01", "diagnosis": "CD", "fecalcal": 50.0, "fecalcal_ng_ml": None, "site_name": "MGH"},
    ])
    omics_stats = {
        "mgx_tax": {"exists": True, "assay": "MGX_WGS", "modality": "taxonomy", "tool": "MetaPhlAn2", "sample_ids": ["S1", "UNMATCHED_SAMPLE_99"], "n_features": 10}
    }
    id_df, issues = audit_identifiers_and_joins(meta, base_schema_cfg, omics_stats)
    cats = [i["category"] for i in issues]
    assert "unmatched_omics_samples" in cats
    row = id_df[id_df["source_table"] == "mgx_tax"].iloc[0]
    assert row["unmatched_in_metadata_dt"] == 1
