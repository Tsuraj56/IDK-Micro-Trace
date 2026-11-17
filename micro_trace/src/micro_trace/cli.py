"""Command-line interface and orchestrator for the MICRO-TRACE Dataset Audit & Feasibility Pipeline."""
import os
import sys
import json
import logging
import argparse
import pandas as pd

from micro_trace.loading import load_yaml, load_metadata
from micro_trace.inventory import build_inventory
from micro_trace.schema import inspect_schema
from micro_trace.microbiome import audit_microbiome_tables
from micro_trace.integrity import audit_identifiers_and_joins
from micro_trace.clinical import audit_participants_and_fc
from micro_trace.longitudinal import build_longitudinal_audit
from micro_trace.feasibility import evaluate_feasibility
from micro_trace.reporting import generate_figures
from micro_trace.reporting.markdown_reports import (
    write_agent1_deliverables,
    write_agent2_to_6_and_final_reports,
)

def run_pipeline(config_path: str, is_synthetic: bool = False) -> dict:
    cfg = load_yaml(config_path)
    paths = cfg["paths"]
    raw_dir = paths["raw_data_dir"]
    rep_dir = paths["reports_dir"]
    fig_dir = paths["figures_dir"]
    log_dir = paths["logs_dir"]
    schema_path = paths["schema_mapping_path"]

    os.makedirs(rep_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    log_file = os.path.join(log_dir, "audit_pipeline.log")
    logging.basicConfig(
        level=getattr(logging, cfg.get("logging", {}).get("level", "INFO")),
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(log_file, mode="w", encoding="utf-8"), logging.StreamHandler(sys.stdout)],
        force=True,
    )
    logger = logging.getLogger("micro_trace")
    logger.info("Starting MICRO-TRACE Audit Pipeline with config: %s", config_path)

    schema_cfg = load_yaml(schema_path)

    # 1. Inventory
    logger.info("Step 1: Building file inventory from %s", raw_dir)
    inv_df = build_inventory(raw_dir)
    inv_df.to_csv(os.path.join(rep_dir, "data_inventory.csv"), index=False)
    inv_df.to_csv(os.path.join(rep_dir, "table_inventory.csv"), index=False)

    # 2. Load Metadata & Inspect Schema
    logger.info("Step 2: Loading metadata and inspecting schema")
    meta_df = load_metadata(raw_dir, schema_cfg)
    schema_df, feat_df, schema_issues = inspect_schema(meta_df, schema_cfg)
    schema_df.to_csv(os.path.join(rep_dir, "table_schema.csv"), index=False)
    feat_df.to_csv(os.path.join(rep_dir, "feature_availability.csv"), index=False)

    # 3. Microbiome Availability Audit
    logger.info("Step 3: Auditing taxonomic and functional microbiome tables")
    tax_df, func_df, cov_df, omics_stats = audit_microbiome_tables(raw_dir, meta_df, schema_cfg)
    tax_df.to_csv(os.path.join(rep_dir, "taxonomy_availability.csv"), index=False)
    func_df.to_csv(os.path.join(rep_dir, "functional_availability.csv"), index=False)
    cov_df.to_csv(os.path.join(rep_dir, "microbiome_sample_coverage.csv"), index=False)

    # 4. Identifier & Join Integrity Audit
    logger.info("Step 4: Validating identifiers and join cardinality")
    id_audit_df, join_issues = audit_identifiers_and_joins(meta_df, schema_cfg, omics_stats)
    id_audit_df.to_csv(os.path.join(rep_dir, "identifier_audit.csv"), index=False)

    # 5. Participant & FC Audit
    logger.info("Step 5: Auditing participant summary and Fecal Calprotectin (FC)")
    part_df, fc_df, clin_issues = audit_participants_and_fc(meta_df, schema_cfg, cfg.get("thresholds", {}))
    part_df.to_csv(os.path.join(rep_dir, "participant_summary.csv"), index=False)
    fc_df.to_csv(os.path.join(rep_dir, "fc_summary.csv"), index=False)

    # 6. Longitudinal Matching & Future Pair Generation
    logger.info("Step 6: Constructing longitudinal pairs and eligibility flow")
    match_df, gap_df, pairs_df, flow_df, split_df = build_longitudinal_audit(
        meta_df,
        cov_df,
        schema_cfg,
        cfg.get("matching_windows_days", [0, 3, 7, 14]),
        cfg.get("candidate_horizons_days", []),
        cfg.get("split_preview", {}),
    )
    match_df.to_csv(os.path.join(rep_dir, "sample_matching.csv"), index=False)
    gap_df.to_csv(os.path.join(rep_dir, "visit_gap_summary.csv"), index=False)
    pairs_df.to_csv(os.path.join(rep_dir, "longitudinal_pairs.csv"), index=False)
    flow_df.to_csv(os.path.join(rep_dir, "eligibility_flow.csv"), index=False)
    split_df.to_csv(os.path.join(rep_dir, "split_audit.csv"), index=False)

    # Combine all quality issues
    all_issues = schema_issues + join_issues + clin_issues
    issues_df = pd.DataFrame(all_issues)
    issues_df.to_csv(os.path.join(rep_dir, "data_quality_issues.csv"), index=False)

    # 7. Feasibility Evaluation
    logger.info("Step 7: Evaluating G1-G8 gates and C1-C4 feasibility")
    metrics = evaluate_feasibility(
        inv_df, schema_df, id_audit_df, part_df, fc_df, tax_df, func_df, pairs_df, flow_df, split_df, all_issues
    )
    with open(os.path.join(rep_dir, "audit_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # 8. Figures & Markdown Reports
    logger.info("Step 8: Generating 8 figures and Markdown reports")
    generate_figures(
        fig_dir, inv_df, part_df, fc_df, meta_df, match_df, gap_df, pairs_df, tax_df, func_df, flow_df, is_synthetic=is_synthetic
    )
    write_agent1_deliverables(rep_dir, inv_df)
    write_agent2_to_6_and_final_reports(
        rep_dir, inv_df, schema_df, feat_df, id_audit_df, part_df, fc_df,
        match_df, gap_df, pairs_df, flow_df, split_df, tax_df, func_df, cov_df, issues_df, metrics
    )

    logger.info("MICRO-TRACE Audit Pipeline completed successfully. Outputs in %s", rep_dir)
    return metrics

def main():
    parser = argparse.ArgumentParser(description="Run MICRO-TRACE IBDMDB Audit & Feasibility Pipeline")
    parser.add_argument(
        "--config",
        default="/content/drive/MyDrive/Micro-Trace/micro_trace/configs/audit_config.yaml",
        help="Path to audit_config.yaml",
    )
    parser.add_argument("--synthetic", action="store_true", help="Flag run as synthetic demo")
    args = parser.parse_args()
    run_pipeline(args.config, is_synthetic=args.synthetic)

if __name__ == "__main__":
    main()
