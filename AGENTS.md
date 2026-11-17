# MICRO-TRACE — Multi-Agent & AI Tool Instructions (`AGENTS.md`)

## Project Overview

**MICRO-TRACE** (*Microbiome-Informed Temporal and Reliable Analysis for Calprotectin Estimation*) is a four-part reproducible research and machine learning pipeline on the **IBDMDB / HMP2** longitudinal cohort:
- **C1 (`Phase P1`)**: Future $\log_{10}(\text{FC}+1)$ prediction (`M0`–`M5`) comparing `Elastic Net`, `Random Forest`, `HistGradientBoosting`, and `MicroTrace-Attn` (Continuous-Time Residual Attention over patient history $1 \dots t$ + full trajectory state vectors).
- **C2 (`Phase P2`)**: Reproducible taxonomic (`MetaPhlAn3`/`v2`) and shotgun functional (`HUMAnN3`/`v2`) signatures for **both `Current FC` (`503` visits) and `Future FC` (`445` pairs)** using participant random-intercept `LME` (`BH-FDR`), Multi-Feature Biological Guilds, 5-fold Stability Selection, and **Cross-Omic Attention (`[Bacteria] <-> [Pathways] <-> [FC]`)**.
- **C3 (`Phase P3`)**: Uncertainty-aware intervals via **5-fold Participant-Grouped Cross-Conformal (`CV+`) with locally-weighted residuals** and selective abstention (`U0`–`U3`).
- **C4 (`Phase P4`)**: Evidence-constrained dietary-pattern decision support (`R1`/`R2`/`R3`) with clinical safety filters and abstention.

---

## Project Memory System

This project maintains institutional knowledge in `docs/project_notes/` for consistency across sessions and across AI coding tools (Antigravity, Claude Code, Cursor, Gemini CLI, etc.).

### Memory Files

- **[docs/project_notes/bugs.md](docs/project_notes/bugs.md)** — Bug log with dates, root causes, solutions, and prevention notes.
- **[docs/project_notes/decisions.md](docs/project_notes/decisions.md)** — Architectural Decision Records (`ADR-001` to `ADR-010`) with context and trade-offs.
- **[docs/project_notes/key_facts.md](docs/project_notes/key_facts.md)** — Verified HMP2 cohort constants, local/Colab paths, join rules, and split definitions.
- **[docs/project_notes/issues.md](docs/project_notes/issues.md)** — Work log tracking completed (`P0`) and upcoming (`P1`–`P5`) phases.

### Companion Tracking & Protocol Files

- **[MICRO_TRACE_Project_Tracker.md](MICRO_TRACE_Project_Tracker.md)** (and `.xlsx`) — Master execution checklist (`P0-01` to `P5-04`), Dataset Audit gates (`G1`–`G8`), Experiment Registry (`C1-E1` to `EXT-E1`), and Decision Log (`D-001` to `D-010`).
- **[MICRO_TRACE_Research_PRD_and_Protocol.md](MICRO_TRACE_Research_PRD_and_Protocol.md)** (and `.docx`) — Full experimental specification (`Version 1.1`).

### Memory-Aware Protocols

**Before proposing architectural changes:**
- Check `docs/project_notes/decisions.md` for existing decisions (`ADR-001`–`ADR-010`).
- Verify the proposed approach doesn't conflict with past choices.
- If it does conflict, acknowledge the existing decision and explain why a change is warranted.

**When encountering errors or bugs:**
- Search `docs/project_notes/bugs.md` for similar issues.
- Apply known solutions if found.
- Document new bugs and solutions when resolved.

**When looking up project configuration or cohort numbers:**
- Check `docs/project_notes/key_facts.md` for verified HMP2 counts, paths, and split parameters.
- Prefer documented facts over assumptions.

**When completing work on phases/tasks:**
- Log completed work in `docs/project_notes/issues.md` and update `MICRO_TRACE_Project_Tracker.xlsx` / `MICRO_TRACE_Project_Tracker.md`.
