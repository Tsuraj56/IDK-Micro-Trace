# MICRO-TRACE | Scientific Reproducibility Guide for Research Readers

This document provides complete instructions for independent researchers and peer reviewers to reproduce all findings, cohort tables, machine learning benchmark metrics, and publication figures from the **MICRO-TRACE** paper.

---

## 1. Computational Environment & Requirements

- **Python Version**: Python 3.10, 3.11, or 3.12 (Tested on Python 3.13)
- **Hardware**: Standard x86_64 CPU (minimum 16 GB RAM) or Google Colab (Free T4 GPU runtime).
- **Core Dependencies**:
  - `numpy >= 1.26.0`
  - `pandas >= 2.2.0`
  - `scipy >= 1.13.0`
  - `scikit-learn >= 1.5.0`
  - `torch >= 2.4.0` (PyTorch with or without CUDA)
  - `matplotlib >= 3.8.0`, `seaborn >= 0.13.0`
  - `pytest >= 8.0.0`

Install all dependencies in a virtual environment:
```bash
git clone <repository_url> micro-trace
cd micro-trace
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

pip install -e micro_trace/
```

---

## 2. Public Data Acquisition & Integrity

MICRO-TRACE uses the publicly available **IBDMDB / HMP2** (Integrative Human Microbiome Project) multi-omic longitudinal cohort. No proprietary or closed-access datasets are required.

### Official Public Sources:
1. **Clinical & Master Metadata**:
   - Source: IBDMDB Portal (`https://ibdmdb.org/tunnel/products/HMP2/metadata/hmp2_metadata_2018-08-20.csv`)
   - Path: `DataSet/hmp2/metadata/hmp2_metadata_2018-08-20.csv`
   - Expected Size: `5,533` rows across `131` participants
2. **Shotgun Metagenomic Taxonomic Profiles (MetaPhlAn3)**:
   - Source: `https://g-227ca.190ebd.75bc.data.globus.org/ibdmdb/products/HMP2/MGX/2018-05-04/taxonomic_profiles_3.tsv.gz`
   - Path: `DataSet/hmp2/mgx/taxonomic_profiles_3.tsv.gz`
3. **Shotgun Metagenomic Functional Pathways (HUMAnN3)**:
   - Source: `https://g-227ca.190ebd.75bc.data.globus.org/ibdmdb/products/HMP2/MGX/2018-05-04/pathabundances_3.tsv.gz`
   - Path: `DataSet/hmp2/mgx/pathabundances_3.tsv.gz`

All cohort exclusion criteria (e.g. zero-sum stool MGX samples `MSM5LLDM` and `MSM5LLDS`) are automated in code per `ADR-004`.

---

## 3. Deterministic Pipeline Execution

All random processes are locked to `seed = 42`. The pipeline can be executed via the standard command-line interface or inside a Jupyter notebook.

### Step 1: Cohort Construction & Participant Partitioning
```bash
python -m micro_trace.c1.cohort --config micro_trace/configs/c1_config.yaml
```
- **Expected Artifacts** in `micro_trace/reports/c1/`:
  - `c1_cohort_consecutive_0d.csv`: Exactly `445` pairs across `93` participants (`69` IBD, `24` nonIBD).
  - `split_audit_80_20.csv`: Exactly `75` development participants (5 folds of 15) and `18` untouched test participants (`0` overlap).
  - `c1_cohort_summary.json`: Provenance manifest.

### Step 2: Train-Only Feature Engineering & Guild Construction
```bash
python -m micro_trace.c1.features --config micro_trace/configs/c1_config.yaml
```
- **Invariants**: All transformers (prevalence filtering $\ge 10\%$, pseudocount, Centered Log-Ratio `CLR`, scaling) are fit **strictly on training folds** (`PR-03`). Test participants are transformed using frozen training parameters with zero leakage.

### Step 3: Model Training, Cross-Validation & Benchmark Ablation
```bash
python -m micro_trace.c1.models --config micro_trace/configs/c1_config.yaml
```
- Evaluates 6 model estimators across modeling arms `M0` through `M5`:
  - `B0`: Mean baseline
  - `B1` / `M0`: Most-recent persistence reference
  - `ElasticNet`: $L_1 + L_2$ penalized linear model
  - `RandomForest`: Regularized decision forest
  - `HistGradientBoosting`: Gradient boosted regression trees
  - `MicroTrace-Attn`: Continuous-Time Residual Cross-Attention network with linear skip connection anchored on index state
- Generates `c1_model_benchmark_results.csv`, `c1_model_summary.json`, and model checkpoints.

### Step 4: Participant-Clustered Bootstrap & Sensitivity Suite
```bash
python -m micro_trace.c1.evaluation --config micro_trace/configs/c1_config.yaml
```
- Computes `MAE` on $\log_{10}(\text{FC}+1)$, `RMSE`, $R^2$, Spearman rank correlation $\rho$, and 1,000-iteration participant-cluster bootstrap 95% confidence intervals for $\Delta\text{MAE}$ vs baseline.
- Runs pre-specified sensitivity analyses: horizon windows (`30–90d`, `14–56d`, `14–35d`, `7–21d`), causal $\le 14\text{d}$ matching, subtype stratification (`CD`, `UC`, `nonIBD`), and medication change subsets.

### Step 5: Publication Figures & Tables
```bash
python -m micro_trace.c1.reporting --config micro_trace/configs/c1_config.yaml
```
- Generates publication-ready figures in `micro_trace/reports/c1/figures/`:
  - `fig1_c1_model_ablation_mae.png`: Primary metric comparison across arms `M0`–`M5`.
  - `fig2_c1_bootstrap_delta_mae.png`: Participant-clustered bootstrap distributions of incremental gain.
  - `fig3_c1_predictions_by_subtype.png`: Scatter plots and calibration curves stratified by `CD`, `UC`, and `nonIBD`.
  - `fig4_c1_horizon_and_matching_sensitivity.png`: Sensitivity to forecast horizons and matching intervals.
  - `fig5_c1_trajectory_attention_importance.png`: Cross-attention weights over longitudinal history.

---

## 4. One-Click Google Colab Reproduction

For researchers without a local GPU or scientific Python stack, the complete pipeline is executable via Google Colab:
- Notebook: `micro_trace/notebooks/c1_reproduce_benchmark.ipynb`
- Can be run end-to-end on a free Google Colab T4 GPU instance in `< 45` minutes.
- Includes step-by-step Markdown explanations, inline plots, and automatic assertion gates verifying that every metric matches the published paper tables.

---

## 5. Automated Verification Suite

Run the full automated test suite to confirm zero regressions and complete pipeline integrity:
```bash
pytest micro_trace/tests -v
```
All tests must pass (`0` failures, `0` warnings of failure).
