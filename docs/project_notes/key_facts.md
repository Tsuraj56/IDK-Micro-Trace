# Key Project Facts & Configuration

This file stores verified non-sensitive constants, dataset metrics, file paths, and environment details for the **MICRO-TRACE** project.

---

## 1. Environments & Paths

### Local Windows Workspace
- **Project Root**: `d:\IDK-projects\IDK--PS-projects\IDK-Micro-Trace`
- **Python Package Root**: `d:\IDK-projects\IDK--PS-projects\IDK-Micro-Trace\micro_trace`
- **Local Virtual Environment**: `d:\IDK-projects\IDK--PS-projects\IDK-Micro-Trace\micro_trace\.venv\Scripts\python.exe`
  - Installed packages include: `pandas`, `numpy`, `scipy`, `matplotlib`, `seaborn`, `pyyaml`, `pytest`, `openpyxl`, `python-docx`
- **Phase P0 Feasibility Reports & Figures**:
  - CSV/JSON/MD outputs: `d:\IDK-projects\IDK--PS-projects\IDK-Micro-Trace\micro_trace\reports\feasibility\`
  - Figures (8 PNGs): `d:\IDK-projects\IDK--PS-projects\IDK-Micro-Trace\micro_trace\reports\feasibility\figures\`
- **Phase P1 (`C1`) Cohort & Split Reports**:
  - CSV/JSON outputs: `d:\IDK-projects\IDK--PS-projects\IDK-Micro-Trace\micro_trace\reports\c1\`
- **Tracking & Protocol Documents**:
  - Excel Tracker: `d:\IDK-projects\IDK--PS-projects\IDK-Micro-Trace\MICRO_TRACE_Project_Tracker.xlsx` (and `.md`)
  - Research PRD & Protocol (`v1.1`): `d:\IDK-projects\IDK--PS-projects\IDK-Micro-Trace\MICRO_TRACE_Research_PRD_and_Protocol.docx` (and `.md`)

### Google Colab Pro & Google Drive Runtime
- **MCP Server**: `colab-mcp` (`open_colab_browser_connection`)
- **Google Drive Root**: `/content/drive/MyDrive/Micro-Trace`
- **Raw HMP2 Dataset Directory (`46` files, `1,453.02 MB`)**: `/content/drive/MyDrive/Micro-Trace/micro-trace-data/raw`
- **Download Manifest**: `/content/drive/MyDrive/Micro-Trace/micro-trace-data/download_manifest.csv`
- **Colab Package Mirror**: `/content/drive/MyDrive/Micro-Trace/micro_trace`

---

## 2. Verified HMP2 / IBDMDB Cohort & Audit Constants (Frozen from Phase P0)

### Overall Cohort & Metadata (`hmp2_metadata_2018-08-20.csv`)
- **Total Metadata Rows**: `5,533` rows across `713` columns and `11` `data_type` layers
- **Unique Participants**: `131` (`66` CD, `38` UC, `27` nonIBD -> `104` total IBD)
- **Unique Physical Collection Events (`site_sub_coll`)**: `2,302` (`1,769` stool collection events; `100%` have valid `date_of_receipt` with `0` temporal inversions against `week_num`)
- **Unique `External ID`s**: `2,892` across all layers (`100%` unique within a single `data_type`, e.g., `1,638/1,638` in `metagenomics`, but reused across omics layers)

### Fecal Calprotectin (FC) Outcome Semantics
- **Source Columns**: `fecalcal` (`1,883` non-null rows) and `fecalcal_ng_ml` (`121` non-null rows)
- **Harmonization Rule**: `fc_combined = df["fecalcal"].combine_first(df["fecalcal_ng_ml"])` (verified identical to `< 1e-4` across shared `site_sub_coll` visits; values `3.45–493.7` in standard ELISA $\mu\text{g/g}$ stool scale)
- **Unique Stool Collection Events with FC**: `636` visits across `98` participants (`median = 70.62`, `IQR = 19.39–192.63`; `116` visits $\le 15.0$, `95` visits $\ge 250.0$)
- **Target Transformation**: $y = \log_{10}(\text{FC} + 1)$

### Shotgun Metagenomics (`MGX`) & Multi-Omics Availability
- **Raw `MGX` Samples**: `1,638` samples $\rightarrow$ **`1,595` unique `site_sub_coll` physical visits** after deduplicating `43` `_TR` / `_P` technical replicates
- **Taxonomic Tables (`100%` matched: `1,638/1,638` samples)**:
  - **Primary (`MetaPhlAn3`)**: `taxonomic_profiles_3.tsv.gz` (`932` clades: `578` species, `187` genera; `159` species $\ge 10\%$ prevalence; `2` zero-sum samples `MSM5LLDM`, `MSM5LLDS` excluded)
  - **Sensitivity (`MetaPhlAn2`)**: `taxonomic_profiles.tsv.gz` (`1,479` clades)
- **Functional Tables (`100%` matched: `1,638/1,638` samples)**:
  - **Primary (`HUMAnN3`)**: `pathabundances_3.tsv.gz` (`22,113` total features; **`478` unstratified MetaCyc pathways**, `410` at $\ge 10\%$ prevalence) & `ecs_3.tsv.gz` (`167,854` features)
  - **Sensitivity (`HUMAnN2`)**: `pathabundances.tsv.gz` (`10,884` features) & `ecs.tsv.gz` (`108,433` features)
- **Other Modalities**:
  - **`16S` (`taxonomic_profiles.tsv.gz`)**: `178` samples — `100%` `biopsy_16S` (`0` FC rows; excluded from stool FC modeling)
  - **`MTX` (Metatranscriptomics RNA)**: `833` samples (`795` unique `site_sub_coll` visits) — secondary expression sensitivity only

### Sample Matching & Longitudinal Forecast Pairs
- **Exact Same-Visit (`0d` via `site_sub_coll`) `MGX + FC` Matches**: **`503` unique visits** (`522` MGX runs) across **`95` participants** (Primary C2 Current-FC cohort & Primary C1 index cohort)
- **Causal Prior/Same-Day (`<=14d`) `MGX + FC` Matches**: **`686` unique visits** across **`96` participants** (Primary C1 matching sensitivity cohort)
- **Participants with $\ge 2$ Matched `MGX + FC` Visits**: **`90` participants** (`66` IBD: `44` CD, `22` UC; `24` nonIBD)
- **Consecutive Next-FC Forecast Pairs (Primary C1 & C2 Future-FC Cohort)**:
  - **`445` pairs across `93` participants** (`69` IBD [`42` CD (`216` pairs), `27` UC (`125` pairs) -> `341` IBD pairs]; `24` nonIBD [`104` pairs]; median gap `56.0d`, `IQR = 22.0–70.0d`)
- **Consecutive Next-FC Pairs with Causal Prior/Same-Day (`<=14d`) Index FC (`c1_cohort_consecutive_le14d.csv`)**:
  - **`611` pairs across `94` participants** (`506` dev pairs, `104` test pairs, `1` unassigned pair [`M2086`, who has no `0d` index visit])
- **All-Future Causal Pairs (`1,671` pairs across `93` participants) by Horizon Window**:
  - `7–21d`: `113` pairs (`36` participants, `26` IBD)
  - `14–35d`: `190` pairs (`44` participants, `35` IBD)
  - `14–56d`: `415` pairs (`84` participants, `63` IBD)
  - `30–90d`: `545` pairs (`88` participants, `65` IBD)

### Frozen Participant Partition (`seed = 42`, `split_audit_80_20.csv`)
- **80% Development Set**: **`75` participants** (`34` CD, `22` UC, `19` nonIBD; `371` consecutive `0d` pairs; partitioned into **5 diagnosis-stratified participant-grouped folds** `0..4` with exactly `15` participants per fold for CV tuning, C2 stability selection, and C3 `CV+` cross-conformal calibration)
- **20% Untouched Final Test Set**: **`18` participants** (`8` CD, `5` UC, `5` nonIBD; `74` consecutive `0d` pairs; `cv_fold = -1`, strictly held out until final evaluation)
- **Participant Overlap**: **`0` participants** shared between Development and Test sets (`split_audit_80_20.csv`)

---

## 3. Verified Phase P1 (`C1`) Feature Engineering & Trajectory Engine Constants

### Real HMP2 Data Execution (`c1_feature_audit.json`, timestamp `2026-11-11T07:45:37.267505+00:00`)
- **Execution Environment**: Google Colab Pro / Linux (`Python 3.13.18`), executed against raw HMP2 tables in `/content/drive/MyDrive/Micro-Trace/micro-trace-data/raw`.
- **Artifact Verification**:
  - `c1_feature_audit.json` (3,005 bytes, SHA-256: `6EEC14DB747ACC681D11E627EA86F6FBD0A39C7E27B17C4D2702AA4B81768A36`)
  - `c1_guild_definitions.csv` (12,560 bytes, SHA-256: `BCDA67503247ED2F188487DC2AA289CE438516606004A6FAF4283D207D05D381`)
- **Primary Cohort**: `c1_cohort_consecutive_0d.csv` (`445` pairs across `93` participants: `371` dev pairs [`75` participants], `74` test pairs [`18` participants]).
- **Prevalence Threshold**: $\ge 10\%$ in training development set (`dev_pairs`, `N=371`).
  - Total MetaPhlAn3 species: `578` (genera: `187`); **retained in dev**: `120` species.
  - Total HUMAnN3 unstratified MetaCyc pathways: `476`; **retained in dev**: `361` pathways.
  - 5-Fold retained species counts: `fold_0`: 122, `fold_1`: 121, `fold_2`: 124, `fold_3`: 121, `fold_4`: 122.
  - 5-Fold retained pathway counts: `fold_0`: 363, `fold_1`: 360, `fold_2`: 363, `fold_3`: 358, `fold_4`: 361.
- **CLR Simplex Pseudocounts**:
  - `species_pseudocount_train_dev`: $7.50 \times 10^{-7}$ (0.5 * min positive relative abundance in train dev)
  - `pathway_pseudocount_train_dev`: $5.85 \times 10^{-7}$ (0.5 * min positive relative abundance in train dev)
- **Biological Guild Matches (`c1_guild_definitions.csv`)**:
  - `guild_scfa_butyrate`: 19 species
  - `guild_pathobiont`: 53 species
  - `fguild_scfa_butanoate`: 21 pathways
  - `fguild_sulfur_h2s`: 6 pathways
  - `fguild_oxidative_stress`: 2 pathways
  - `fguild_mucin_hexuronide`: 8 pathways
  - `fguild_lps_enterobacterial`: 5 pathways
- **Arm Feature Dimensions**:
  - `M0` (Minimal Clinical): `2`
  - `M1` (Expanded Clinical): `29`
  - `M2` (Clinical + Alpha/Beta Diversity + Guilds): `158`
  - `M3` (Clinical + Filtered CLR Species): `396`
  - `M4` (Clinical + Filtered CLR Species + Filtered CLR Pathways): `524`
  - `M5` (M4 + Trajectory State Representation ADR-009): `646`
  - `B3` (Microbiome-Only Filtered CLR Species): `130`
  - `B4` (Microbiome-Only Filtered CLR Species + Filtered CLR Pathways): `368`
- **MicroTrace-Attn Tensor Dimensions**:
  - `max_seq_len`: 25
  - `d_seq`: 15
  - `d_anchor`: 6
  - `strictly_causal_temporal_deltas`: True (min $\Delta t = 0.0$ days)
- **Zero-Sum Sample Handling**:
  - Flagged zero-sum samples: `MSM5LLDM`, `MSM5LLDS` (imputed to geometric simplex center 0.0 with indicator `is_zero_sum_or_missing_omics=1`).

---

## 4. Verified Phase P1 (`C1`) Model Benchmark Constants (`T-003`, `2026-11-12`)

### Real GPU Execution (`c1_model_summary.json`, timestamp `2026-11-12T06:46:05.990664+00:00`)
- **Execution Environment**: Google Colab Pro (`account2` runtime on **Tesla T4 GPU**, `15.36 GB` VRAM, Driver `580.82.07`, CUDA `13.0`, PyTorch `2.6.0+cu124`).
- **Artifact Verification**:
  - `c1_model_benchmark_results.csv` (`1,786` bytes, SHA-256: `F0D11A7BB17BD60F2582F72B9735A75AEB049E5D9051EC9CA4E75E54784BE746`)
  - `c1_model_summary.json` (`8,154` bytes)
  - `t003_microtrace_attn_training.ipynb` (`310,488` bytes, fully executed with outputs)
  - `t003_c1_model_training.ipynb` (`310,488` bytes, fully executed with outputs)
- **Evaluated Modeling Arms (`8`)**: `M0`, `M1`, `M2`, `M3`, `M4`, `M5`, `B3`, `B4`.
- **Evaluated Model Families (`4`)**: `ElasticNetCV`, `RandomForestRegressor`, `HistGradientBoostingRegressor`, and `MicroTrace-Attn`.
- **Participant-Grouped 5-Fold Development Set**: `dev_pairs_count = 371` across `75` disjoint participants (folds `0..4`, `15` participants/fold).
- **Test Partition Status**: `test_pairs_count = 0`, `test_partition_status = "untouched_held_out"` (`18` participants, `74` pairs strictly preserved).
- **Benchmark Performance Metrics (Sorted by 5-Fold CV MAE on $\log_{10}(\text{FC}+1)$)**:
  - **$B_0$ (Mean Baseline)**: CV MAE = `0.4970`, RMSE = `0.5367`, $R^2$ = `0.0000`, Spearman $\rho$ = `0.0000`
  - **$B_1$ / $M_0$ (Persistence Baseline)**: CV MAE = `0.2687`, RMSE = `0.4233`, $R^2$ = `0.3778`, Spearman $\rho$ = `0.6958`
  - **Top Model — `MicroTrace-Attn` on Arm `M1`**: CV MAE = `0.2616` ($\pm$ `0.0340`), RMSE = `0.4140`, $R^2$ = `0.3878`, Spearman $\rho$ = `0.6863` (`-2.64%` error reduction vs persistence)
  - **`MicroTrace-Attn` on Arm `M5` (Trajectory Vectors)**: CV MAE = `0.2635` ($\pm$ `0.0324`), RMSE = `0.4144`, $R^2$ = `0.3879`, Spearman $\rho$ = `0.6870`
  - **`MicroTrace-Attn` on Arm `B3`**: CV MAE = `0.2640` ($\pm$ `0.0329`)
  - **`MicroTrace-Attn` on Arm `B4`**: CV MAE = `0.2641` ($\pm$ `0.0327`)
  - **`MicroTrace-Attn` on Arm `M4` (Full Multi-Omics)**: CV MAE = `0.2642` ($\pm$ `0.0323`)
  - **`MicroTrace-Attn` on Arm `M2`**: CV MAE = `0.2643` ($\pm$ `0.0326`)
  - **`MicroTrace-Attn` on Arm `M3`**: CV MAE = `0.2645` ($\pm$ `0.0331`)
  - **ElasticNet Top**: Arm `M1` CV MAE = `0.3116` ($\pm$ `0.0306`), Arm `M5` CV MAE = `0.3204`
  - **Random Forest Top**: Arm `M1` CV MAE = `0.3230` ($\pm$ `0.0342`), Arm `M5` CV MAE = `0.3313`
  - **HistGradientBoosting Top**: Arm `M1` CV MAE = `0.3359` ($\pm$ `0.0391`), Arm `M5` CV MAE = `0.3558`
  - **Non-Omics Tabular Baselines ($B_3$, $B_4$)**: Collapse without omics (CV MAE `0.4669`–`0.4922`, negative or near-zero $R^2$).
- **Test Suite Verification**: `37/37` pytest unit and leakage tests passed cleanly in the live Colab runtime (`8.00s`) and locally (`15.67s`).


