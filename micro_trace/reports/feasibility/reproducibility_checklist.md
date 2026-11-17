# Reproducibility Checklist (Agent 6)

- [x] **Immutable Raw Data:** All 46 raw files (1.45 GB) stored untouched in `/content/drive/MyDrive/Micro-Trace/micro-trace-data/raw/`.
- [x] **Configuration-Driven Schema Mapping:** `configs/schema_mapping.yaml` and `configs/audit_config.yaml` define all column names, suffixes, horizons, and windows.
- [x] **Deterministic Seed:** Participant partition preview seeded with `random_seed: 42` in `audit_config.yaml`.
- [x] **Environment Recorded:** Python `3.13.18`, pandas `2.2.3`, executed on `2026-11-08T19:17:36.755904+00:00`.
- [x] **Automated QA Test Suite:** 10/10 synthetic unit & leakage tests verified via `pytest`.
- [x] **Independent Verification:** All summary counts in `final_feasibility_report.md` cross-checked directly against generated CSV tables (`audit_metrics.json`).
