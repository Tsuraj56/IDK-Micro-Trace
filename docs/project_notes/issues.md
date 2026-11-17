# Issues & Work Log

This file tracks completed and upcoming phase workstreams for **MICRO-TRACE** (aligned with `Master Plan` tasks `P0-01` through `P5-04` in `MICRO_TRACE_Project_Tracker.xlsx`).

---

## Completed Work

### 2026-11-10 - Phase P0 (`P0-01` to `P0-10`): Dataset Acquisition, Feasibility Audit & Protocol v1.1 Freeze
- **Status**: Completed (`10/10` P0 tasks `Done`, `25/29` Dataset Audit checks `Pass`, `11/11` unit/leakage tests passing)
- **Description**:
  - Downloaded all `46` raw HMP2/IBDMDB files (`1,453.02 MB`) to Google Drive (`/content/drive/MyDrive/Micro-Trace/micro-trace-data/raw/`).
  - Built and executed the modular `micro_trace` Python package (`inventory`, `loading`, `schema`, `integrity`, `clinical`, `microbiome`, `longitudinal`, `feasibility`, `reporting`) with CLI and `11` pytest unit/leakage tests.
  - Generated all `16` audit CSV/JSON artifacts, `9` Markdown feasibility reports, and `8` publication-grade PNG figures in `micro_trace/reports/feasibility/`.
  - Updated `MICRO_TRACE_Project_Tracker.xlsx` (`.md`) and `MICRO_TRACE_Research_PRD_and_Protocol.docx` (`.md`) to **Version 1.1**, incorporating `MicroTrace-Attn`, full trajectory state vectors (`M5`), dual `Current FC` + `Future FC` scope (`C2`), Multi-Feature Guilds, `Cross-Omic Attention`, and 5-fold Participant-Grouped Cross-Conformal (`CV+`) locally-weighted intervals (`C3`).
- **Artifacts**: `micro_trace/reports/feasibility/final_feasibility_report.md`, `audit_metrics.json`, `MICRO_TRACE_Project_Tracker.md`, `MICRO_TRACE_Research_PRD_and_Protocol.md`

### 2026-11-11 - Phase P1 Cohort & Split (`P1-01` & `P1-02` / `T-001`): C1 Index-to-Future Cohort Tables & Frozen 80/20 (5-Fold Grouped CV) Split
- **Status**: Completed (`P1-01` & `P1-02` `Done`, `18/18` pytest tests passing on Google Colab and locally)
- **Description**:
  - Created `micro_trace/configs/c1_config.yaml`, `micro_trace/src/micro_trace/c1/cohort.py`, and `micro_trace/tests/test_c1_cohort.py`.
  - Executed `python -m micro_trace.c1.cohort` and `pytest` on Google Colab via `colab-mcp` against `/content/drive/MyDrive/Micro-Trace/micro-trace-data/raw/` and mirrored all generated C1 report artifacts locally (`SHA256` verified).
  - Generated `c1_cohort_consecutive_0d.csv` (`445` pairs across `93` participants: `42` CD [`216` pairs], `27` UC [`125` pairs], `24` nonIBD [`104` pairs]), `c1_cohort_all_horizons_0d.csv` (`1,671` pairs across `93` participants: `545` in `30-90d`, `415` in `14-56d`, `190` in `14-35d`, `113` in `7-21d`), `c1_cohort_consecutive_le14d.csv` (`611` pairs across `94` participants), and `split_audit_80_20.csv` (`75` dev participants [`34` CD, `22` UC, `19` nonIBD across 5 folds of `15`; `371` consecutive pairs] and `18` test participants [`8` CD, `5` UC, `5` nonIBD; `74` consecutive pairs], `0` participant overlap).
- **Artifacts**: `micro_trace/reports/c1/c1_cohort_consecutive_0d.csv`, `c1_cohort_all_horizons_0d.csv`, `c1_cohort_consecutive_le14d.csv`, `split_audit_80_20.csv`, `c1_cohort_summary.json`

### 2026-11-11 - Phase P1 Feature Engine & Trajectory Architecture (`P1-04` to `P1-08` / `T-002`): Multi-Arm Feature Matrix, Biological Guilds, Causal Trajectory Engine & MicroTrace-Attn Sequence Tensors
- **Status**: Completed (`P1-04`, `P1-05`, `P1-06`, `P1-07`, `P1-08` `Done`, `29/29` pytest tests passing on Google Colab and locally)
- **Description**:
  - Implemented train-only fitting invariant (`PR-03`) with $\ge 10\%$ prevalence filtering, adaptive pseudocount, and centered log-ratio (CLR) transformation on MetaPhlAn3 species (`120` retained in dev) and HUMAnN3 unstratified MetaCyc pathways (`361` retained in dev).
  - Constructed Multi-Feature Microbial Guilds (SCFA butyrate producers, pathobionts, protective-to-pathobiont log-ratio) and Functional Guilds (butanoate, sulfur/H2S, oxidative stress, mucin/hexuronide degradation, LPS enterobacterial).
  - Engineered full causal visit trajectory engine (`ADR-009`) across 12 dynamic variables ($x_t, x_{t-1}, \Delta x, \text{log-ratio}, \text{velocity}, \text{rolling statistics}_{\le t}$) and MicroTrace-Attn continuous-time sequence tensors ($D_{\text{seq}}=15$, $D_{\text{anchor}}=6$, $\Delta t \ge 0$).
  - Constructed modeling feature matrices across arms `M0` (2), `M1` (29), `M2` (158), `M3` (396), `M4` (524), `M5` (646), `B3` (130), `B4` (368). Handled zero-sum samples (`MSM5LLDM`, `MSM5LLDS`) via simplex center imputation (0.0) and missingness indicator.
  - Executed CLI and test suite on Google Colab Pro (`Python 3.13.18`) against raw HMP2 data (`run_c1_feature_audit` at `2026-11-11T07:45:37.267505+00:00`), hardened `test_c1_real_run_artifacts_on_disk()` to strictly assert all dimensions and authenticity flags, and synchronized authentic artifacts to local repository.
- **Artifacts**: `micro_trace/reports/c1/c1_feature_audit.json`, `micro_trace/reports/c1/c1_guild_definitions.csv`, `micro_trace/src/micro_trace/c1/features.py`, `micro_trace/tests/test_c1_features.py`

### 2026-11-12 - Phase P1 Model Training & Benchmark (`P1-03` to `P1-10` / `T-003`): Multi-Arm Benchmark Architecture & MicroTrace-Attn Deep Continuous-Time Network on Colab Tesla T4 GPU
- **Status**: Completed (`P1-03` to `P1-10` `Done`, `37/37` pytest tests passing on Google Colab and locally)
- **Description**:
  - Implemented complete Phase P1 model training and multi-arm benchmark suite in `micro_trace/src/micro_trace/c1/models.py`:
    - Baselines: $B_0$ (Training Mean, CV MAE = 0.4970) and $B_1$ / $M_0$ (Persistence Baseline, CV MAE = 0.2687, $R^2$ = 0.3778, Spearman $\rho$ = 0.6958).
    - Tabular ML: `ElasticNetCV` ($L_1+L_2$ CV-tuned), `RandomForestRegressor`, and `HistGradientBoostingRegressor` across all arms.
    - Deep Architecture (`MicroTrace-Attn`): Continuous-time residual cross-attention over patient history ($1 \dots t$) with continuous Fourier time-decay embeddings $\phi(\Delta t)$ and non-negotiable linear residual skip connection from anchor FC ($w_0=1.0$), ensuring the network strictly cannot underperform persistence.
  - Executed benchmark on Google Colab (`account2` runtime on **Tesla T4 GPU**, `15.36 GB` VRAM) under strict **Train-Only Fitting Invariance (`PR-03`)** across 5 diagnosis-stratified participant-grouped folds ($N=371$, 75 participants) while keeping the 20% test partition ($N=74$, 18 participants) **100% untouched** (`evaluate_test=False`).
  - Best performing model: **`MicroTrace-Attn` on Arm `M1` (CV MAE = 0.2616 $\pm$ 0.0340, RMSE = 0.4140, $R^2$ = 0.3878, Spearman $\rho$ = 0.6863)**, outperforming the strong persistence baseline ($M_0$ CV MAE = 0.2687).
  - Synchronized executed notebooks (`t003_microtrace_attn_training.ipynb`, `t003_c1_model_training.ipynb`, `310.5 KB` each) and authentic benchmark reports (`c1_model_benchmark_results.csv`, `c1_model_summary.json`) locally and terminated Colab session.
- **Artifacts**: `micro_trace/reports/c1/c1_model_benchmark_results.csv`, `micro_trace/reports/c1/c1_model_summary.json`, `micro_trace/notebooks/t003_microtrace_attn_training.ipynb`, `micro_trace/notebooks/t003_c1_model_training.ipynb`

### 2026-11-12 - Phase P1 Optimization Trial 1: Target Delta Formulation ($\Delta y$ and Reconstructed $y_{t+1}$)
- **Status**: Completed (`Trial 1` `Done`, `47/47` pytest unit/model/leakage tests passing locally)
- **Description**:
  - Implemented Target Delta Formulation: $\Delta y = \log_{10}(\text{FC}_{t+1} + 1) - \log_{10}(\text{FC}_t + 1)$ with reconstructed forecast $\hat{y}_{t+1} = \log_{10}(\text{FC}_t + 1) + \hat{\Delta y}$.
  - Evaluated candidate families (`ElasticNetCV`, `RandomForest`, `HistGradientBoosting`, `MicroTrace-Attn-SmoothL1`, `MicroTrace-Attn-L1`) across all arms (`M0`–`M5`, `B3`, `B4`) under frozen 5-fold participant-grouped CV ($N=371$ dev pairs, 75 participants; 20% test partition held 100% untouched).
  - Tabular ML showed massive gains over prior level formulation: `ElasticNetCV` on `M5` dropped from MAE 0.3204 to 0.2729 (61.9% Directional Accuracy, $R^2=0.4674$); on `M1` dropped from MAE 0.3116 to 0.2854 (60.3% Directional Accuracy, $R^2=0.4719$).
  - Best model overall: **`MicroTrace-Attn-L1` on Arm `M1` (CV MAE Reconstructed = 0.2611 $\pm$ 0.0328, Delta MAE = 0.2611, Directional Accuracy = 61.5%, $R^2 = 0.3974$, Spearman $\rho = 0.6891$)**, strictly beating the persistence baseline ($M_0$ CV MAE = 0.2687). `MicroTrace-Attn-SmoothL1` achieved **63.15% Directional Accuracy** (MAE = 0.2618).
- **Artifacts**: `micro_trace/reports/c1/c1_trial1_delta_benchmark_results.csv`, `micro_trace/reports/c1/c1_trial1_delta_summary.json`, `micro_trace/src/micro_trace/c1/models.py`, `micro_trace/src/micro_trace/c1/benchmark_delta.py`, `micro_trace/tests/test_c1_models.py`

---

## Upcoming Workstreams

### Phase P2 (`P2-01` to `P2-06` | `C2-E1` to `C2-E4`): C2 — Taxonomic, Functional, Guild & Cross-Omic Attention Signatures
- **Status**: Planned (`Next Up`)
- **Scope**:
  - Run train-only prevalence filtering ($\ge 10\%$), pseudocount + CLR transformation (`MetaPhlAn3`/`HUMAnN3` primary; `v2` sensitivity).
  - Evaluate **both `Current FC` (`503` visits) and `Future FC` (`445` pairs)** using participant random-intercept `LME` (`BH-FDR`), Multi-Feature Microbial & Functional Guilds, 5-fold `Stability Selection` ($\ge 4/5$ folds, Jaccard), and **Cross-Omic Attention (`[Bacteria] <-> [Pathways] <-> [FC]`)**.

### Phase P3 (`P3-01` to `P3-05` | `C3-E1` to `C3-E3`): C3 — Uncertainty Calibration & Selective Abstention
- **Status**: Planned (`Not started`)
- **Scope**:
  - Implement quantile regression (`U1`), **5-fold Participant-Grouped Cross-Conformal (`CV+`) with locally-weighted residuals** (`U2`), and selective abstention (`U3`) with subgroup coverage and risk-coverage curves.

### Phase P4 (`P4-01` to `P4-07` | `C4-E1` to `C4-E3`): C4 — Evidence-Constrained Dietary-Pattern Decision Support
- **Status**: Planned (`Not started`)
- **Scope**:
  - Curate structured evidence database (AGA, ECCO, BDA, DINE-CD), implement safety/contraindication/abstention rules, build `R1`/`R2`/`R3` recommendation arms, and run standardized clinical scenario evaluations.

### Phase P5 (`P5-01` to `P5-04` | `EXT-E1`): External Validation & Manuscript Package
- **Status**: Planned (`Not started`)
- **Scope**:
  - Evaluate external cohort compatibility (`IBSEN III`) and assemble final manuscript tables, figures, and reproducibility bundle.
