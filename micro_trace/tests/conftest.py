"""Shared synthetic fixtures for MICRO-TRACE QA & leakage tests."""
import pytest
import pandas as pd

@pytest.fixture
def base_schema_cfg():
    return {
        "metadata_table": "meta.csv",
        "identifiers": {
            "participant_id": "Participant ID",
            "sample_id": "External ID",
            "collection_event_id": "site_sub_coll",
            "visit_id": "visit_num",
            "week_num": "week_num",
            "interval_days": "interval_days",
            "data_type": "data_type",
            "site_name": "site_name",
            "project": "Project",
        },
        "dates": {"receipt_date": "date_of_receipt"},
        "outcomes": {
            "fc_primary": "fecalcal",
            "fc_secondary": "fecalcal_ng_ml",
            "fc_tube_id": "Tube B:Fecal Calprotectin",
        },
        "clinical": {
            "disease_subtype": "diagnosis",
            "ibd_subtypes": ["CD", "UC"],
            "control_subtypes": ["nonIBD"],
        },
        "medications": {"antibiotics": "Antibiotics"},
        "dietary_variables": ["Water"],
        "qc_variables": {"reads_raw": "reads_raw"},
        "omics_tables": {},
    }
