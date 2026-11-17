"""Test 10 & inventory checks: Empty dataset handling and file discovery."""
import os
import pytest
import pandas as pd
from micro_trace.inventory import build_inventory
from micro_trace.clinical import audit_participants_and_fc
from micro_trace.longitudinal import build_longitudinal_audit

def test_10_empty_dataset_does_not_fabricate_statistics(tmp_path, base_schema_cfg):
    """Req 10: An empty dataset does not produce fabricated summary statistics."""
    inv = build_inventory(str(tmp_path))
    assert inv.empty

    empty_meta = pd.DataFrame(columns=[
        "Participant ID", "External ID", "site_sub_coll", "visit_num",
        "week_num", "interval_days", "data_type", "site_name", "Project",
        "date_of_receipt", "fecalcal", "fecalcal_ng_ml", "Tube B:Fecal Calprotectin",
        "diagnosis", "Antibiotics", "Water", "reads_raw"
    ])
    part_df, fc_df, issues = audit_participants_and_fc(empty_meta, base_schema_cfg, {})
    assert len(part_df) == 0
    assert (fc_df["n_valid_fc"] == 0).all()

    match_df, gap_df, pairs_df, flow_df, split_df = build_longitudinal_audit(
        empty_meta, pd.DataFrame(), base_schema_cfg, [0, 7], [{"name": "h30", "min_days": 1, "max_days": 30}], {}
    )
    assert len(pairs_df) == 0
    assert len(split_df) == 0
