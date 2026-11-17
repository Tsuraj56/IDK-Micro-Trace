# MICRO-TRACE — Project Instructions & Memory System

## Project Overview

**MICRO-TRACE** (*Microbiome-Informed Temporal and Reliable Analysis for Calprotectin Estimation*) is a four-part reproducible research and machine learning system on the longitudinal **IBDMDB / HMP2** cohort (`131` participants, `1,595` deduplicated shotgun metagenomic visits, `636` Fecal Calprotectin visits):
- **C1 (`Phase P1`)**: Causal future $\log_{10}(\text{FC}+1)$ forecasting (`M0`–`M5`) across `Elastic Net`, `Random Forest`, `HistGradientBoosting`, and `MicroTrace-Attn` (Continuous-Time Residual Attention over visits $1 \dots t$ with full trajectory state vectors).
- **C2 (`Phase P2`)**: Reproducible taxonomic (`MetaPhlAn3`/`v2`) and functional (`HUMAnN3`/`v2`) signature discovery for **both `Current FC` (`503` visits) and `Future FC` (`445` pairs)** across single features (`LME` + BH-FDR), Multi-Feature Biological Guilds, 5-fold Stability Selection, and **Cross-Omic Attention (`[Bacteria] <-> [Pathways] <-> [FC]`)**.
- **C3 (`Phase P3`)**: Patient-specific prediction intervals via **5-fold Participant-Grouped Cross-Conformal (`CV+`) with locally-weighted residuals** ($r_i = |y_i - \hat{y}_i|/\hat{\sigma}(x_i)$), subgroup calibration audit, and selective abstention (`U0`–`U3`).
- **C4 (`Phase P4`)**: Evidence-constrained dietary-pattern decision support (`R1` guidelines only, `R2` + clinical context, `R3` + supported microbiome links) with strict safety filters and abstention.

---

## Project Memory System

This project maintains institutional knowledge in `docs/project_notes/` for consistency across sessions and AI tools.

### Memory Files

- **[docs/project_notes/bugs.md](docs/project_notes/bugs.md)** — Bug log with dates, root causes, solutions, and prevention notes (including HMP2 join rules and replicate handling).
- **[docs/project_notes/decisions.md](docs/project_notes/decisions.md)** — Architectural Decision Records (`ADR-001` to `ADR-010`) with context and trade-offs.
- **[docs/project_notes/key_facts.md](docs/project_notes/key_facts.md)** — Verified HMP2 cohort numbers, file paths (local & Google Drive/Colab), outcome semantics, and frozen 80/20 participant split constants.
- **[docs/project_notes/issues.md](docs/project_notes/issues.md)** — Phase-by-phase work log (`P0` completed; `P1`–`P5` roadmap) linked to [MICRO_TRACE_Project_Tracker.md](MICRO_TRACE_Project_Tracker.md) and [MICRO_TRACE_Research_PRD_and_Protocol.md](MICRO_TRACE_Research_PRD_and_Protocol.md).

### Memory-Aware Protocols

**Before proposing architectural or protocol changes:**
- Check `docs/project_notes/decisions.md` for existing ADRs (`ADR-001` through `ADR-010`).
- Verify the proposed approach does not conflict with frozen protocol decisions in `MICRO_TRACE_Research_PRD_and_Protocol.md`.
- If a change is warranted, document the rationale and update both `docs/project_notes/decisions.md` and `MICRO_TRACE_Project_Tracker.xlsx` (`.md`).

**When encountering errors, join discrepancies, or bugs:**
- Search `docs/project_notes/bugs.md` for known HMP2 data pitfalls (e.g., `External ID` reuse across omics layers, `site_sub_coll` vs `week_num`, `_TR`/`_P` replicates).
- Document new bugs and solutions in `docs/project_notes/bugs.md` when resolved.

**When looking up dataset counts, paths, or configuration:**
- Check `docs/project_notes/key_facts.md` first rather than guessing paths or re-deriving cohort counts from scratch.

**When completing phase tasks (`P1`–`P5`):**
- Log completed work in `docs/project_notes/issues.md` and update task statuses in `MICRO_TRACE_Project_Tracker.xlsx` and `MICRO_TRACE_Project_Tracker.md`.

### Style Guidelines for Memory Files

- **Prefer bullet lists over wide tables** inside `docs/project_notes/` for simplicity and fast scanning.
- **Keep entries concise** and factual with exact verified metrics.
- **Always include dates (`YYYY-MM-DD`)** for temporal context.
