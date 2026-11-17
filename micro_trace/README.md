# MICRO-TRACE: IBDMDB Dataset Audit & Feasibility Pipeline

Reproducible Python audit and feasibility assessment pipeline for the **MICRO-TRACE** research programme on the **iHMP / IBDMDB (HMP2)** multi-omics cohort.

## Directory Structure
```
micro_trace/
├── configs/
│   ├── audit_config.yaml        # Paths, candidate matching windows, forecast horizons, thresholds
│   └── schema_mapping.yaml      # Metadata & omics table schema definitions
├── src/micro_trace/
│   ├── inventory/               # File discovery, format & dimension inventory
│   ├── loading/                 # YAML and streaming gzipped TSV loaders
│   ├── schema/                  # Schema validation & feature availability audit
│   ├── integrity/               # Identifier uniqueness & many-to-many join cardinality checks
│   ├── clinical/                # Participant summary & Fecal Calprotectin (FC) distribution
│   ├── microbiome/              # Taxonomic (MetaPhlAn2/v3, 16S) & Functional (HUMAnN2/v3, MTX) audits
│   ├── longitudinal/            # Causal sample matching, visit gaps, index-to-future pair construction
│   ├── feasibility/             # G1-G8 gate evaluation & C1-C4 feasibility synthesis
│   ├── reporting/               # 8 matplotlib figures & multi-agent Markdown reports
│   └── cli.py                   # End-to-end CLI entrypoint
├── tests/                       # 10 mandatory synthetic QA & leakage unit tests
├── reports/feasibility/         # Generated CSVs, JSON metrics, Markdown reports, and figures/
├── logs/                        # Execution logs
└── pyproject.toml
```

## Installation & Execution

### Run Automated QA & Leakage Test Suite (Agent 6)
```bash
PYTHONPATH=src pytest -v tests/
```

### Run Real-Data IBDMDB Audit Pipeline
```bash
PYTHONPATH=src python -m micro_trace.cli --config configs/audit_config.yaml
```
