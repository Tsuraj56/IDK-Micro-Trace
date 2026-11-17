# MICRO-TRACE: Final Dataset Audit & Feasibility Report (Agent 0 — Orchestrator)

## 1. Executive Summary
This report presents the verified, evidence-backed feasibility audit of the **IBDMDB / HMP2** multi-omics dataset for the **MICRO-TRACE** research programme (C1–C4).
- **Completion State:**
  - `Pipeline implemented`: **YES** (`micro_trace` Python package in `/content/drive/MyDrive/Micro-Trace/micro_trace/` and mirrored locally).
  - `Pipeline tested on synthetic data`: **YES** (10/10 mandatory QA and leakage tests passed).
  - `Actual IBDMDB data inspected`: **YES** (All 46 official merged HMP2 data products — 1.45 GB across all 11 omics layers — downloaded and audited).
  - `Actual feasibility established`: **YES** for C1, C2 (Taxonomy & Function), C3, and C4 internal IBDMDB components; **NOT YET DETERMINABLE** for external validation (IBSEN III).

| Research component | Feasibility | Evidence | Remaining blockers |
|---|---|---|---|
| **C1 — Future FC** | **FEASIBLE WITH LIMITATIONS** | 90 unique participants (66 IBD: CD/UC) have >=2 visits with both MGX and same-day FC, yielding 445 consecutive index-to-next-FC pairs (1671 all-future pairs; 415 in 14-56d; 545 in 30-90d). Limitation: FC is missing in 68.1% of MGX visits (522/1638 present), and independent participant count (90 with >=2 MGX+FC visits [66 IBD]; 93 with >=1 MGX+FC index visit and a future FC [69 IBD]) requires parsimonious models (Elastic Net / Random Forest) rather than deep learning. | Principal investigator must freeze the primary forecast horizon (e.g. next-visit within 14-56d vs 30-90d) and confirm whether nonIBD controls are excluded from C1 training (PRD 4.2 specifies CD/UC). |
| **C2 — Taxonomy** | **FEASIBLE** | Both MetaPhlAn2 (1,638 samples x 1,479 clades) and MetaPhlAn3 (1,638 samples x 932 clades) shotgun WGS taxonomic profiles are downloaded, verified, and 100% matched (1,638/1,638) to metadata External IDs across 130 participants. Biopsy 16S (178 samples) and Pilot Stool 16S (58 samples) are also available. | Must choose between MetaPhlAn2 (original publication baseline) and MetaPhlAn3 (updated taxonomy) before freezing P2. |
| **C2 — Function** | **FEASIBLE** | Shotgun WGS functional potential profiles are downloaded and 100% matched (1,638/1,638 samples): HUMAnN2 MetaCyc pathways (10,884 features), HUMAnN3 MetaCyc pathways (22,113 features), HUMAnN2 EC enzymes (108,433 features), and HUMAnN3 EC enzymes (167,854 features, 1,635 samples). Additionally, active RNA expression (MTX HUMAnN2 pathways/ECs, 735 samples) and fecal proteomics (450 samples) are available for separate comparison. | Retain strict distinction between metagenomic DNA potential (MGX) and metatranscriptomic RNA expression (MTX). |
| **C3 — Uncertainty** | **FEASIBLE WITH LIMITATIONS** | Participant-grouped 70/15/15 split yields 65 train, 14 calibration, and 14 test participants (overlap=0). However, calibration partition has only ~13 independent participants (~10 IBD). | With ~10-13 calibration participants, participant-level split-conformal quantiles have high finite-sample variance; recommend cross-conformal / CV+ grouped by participant rather than a single 15% calibration holdout. |
| **C4 — Dietary patterns** | **FEASIBLE WITH LIMITATIONS** | 21 longitudinal food-frequency variables (yogurt, dairy, probiotics, fruits, vegetables, beans, whole grains, starch, eggs, processed/red/white meat, shellfish, fish, sweets, beverages) are recorded in >98% of MGX visits (~1,600/1,638), alongside diagnosis (100%), medications/antibiotics (100%), HBI (693 MGX rows), and SCCAI (452 MGX rows). | IBDMDB records coarse food-group recall frequencies, not standardized Mediterranean or SCD adherence scores; DINE-CD and clinical guidelines (AGA/ECCO/BDA) must be curated externally to build the R1-R3 evidence knowledgebase. |
| **External validation** | **NOT YET DETERMINABLE** | IBSEN III and DINE-CD datasets are not present locally and require external institutional data-access requests. | Submit external data request for IBSEN III; note that 16S external cohorts can only validate genus-level taxonomic models, not shotgun functional pathways. |

---

## 2. Dataset Access and Provenance
- **Source:** Official iHMP / IBDMDB Globus endpoint (`https://g-227ca.190ebd.75bc.data.globus.org/ibdmdb/`) indexed by `https://ibdmdb.org/downloads/`.
- **Access Tier:** Open public release (46 merged tables, 1,453.02 MB). Controlled-access host raw DNA sequences (dbGaP `phs001626.v1.p1`) are not required.
- **Guide Correction:** Contrary to `HMP2_IBDMDB_Dataset_Guide.md`, merged tables for `MGX` (MetaPhlAn2/v3, HUMAnN2/v3), `MTX`, `16S`, `MPX`, and `MVX` exist on the official portal and are now stored in `/content/drive/MyDrive/Micro-Trace/micro-trace-data/raw/`.

## 3. Data Inventory
- **Files Inventoried:** `47` files (`1453.02` MB total). Full details in `data_inventory.csv` and `table_inventory.csv`.
| file_name | relative_path | layer | format | size_bytes | size_mb | n_rows | n_cols |
| --- | --- | --- | --- | --- | --- | --- | --- |
| download_manifest.csv | download_manifest.csv | META | csv | 12708 | 0.0127 | 72.0 | 6.0 |
| anadama_run.log.gz | metadata/anadama_run.log.gz | META | log.gz | 11681060 | 11.6811 | nan | nan |
| hmp2_metadata_2018-08-20.csv | metadata/hmp2_metadata_2018-08-20.csv | META | csv | 9074342 | 9.0743 | 5533.0 | 602.0 |
| taxonomic_profiles.biom.gz | products/HMP2/16S/2018-01-07/taxonomic_profiles.biom.gz | 16S | biom.gz | 69382 | 0.0694 | nan | nan |
| taxonomic_profiles.tsv.gz | products/HMP2/16S/2018-01-07/taxonomic_profiles.tsv.gz | 16S | tsv.gz | 55417 | 0.0554 | 982.0 | 180.0 |
| host_tx_counts.biom.gz | products/HMP2/HTX/host_tx_counts.biom.gz | HTX | biom.gz | 27034786 | 27.0348 | nan | nan |
| host_tx_counts.tsv.gz | products/HMP2/HTX/host_tx_counts.tsv.gz | HTX | tsv.gz | 11016783 | 11.0168 | 55765.0 | 255.0 |
| C18n_Metabolites_ID_AfterPublication_2021-05-17.xlsx | products/HMP2/MBX/C18n_Metabolites_ID_AfterPublication_2021-05-17.xlsx | MBX | xlsx | 11363 | 0.0114 | nan | nan |
| HMP2_metabolomics.biom | products/HMP2/MBX/HMP2_metabolomics.biom | MBX | biom | 528617375 | 528.6174 | nan | nan |
| HMP2_metabolomics_w_metadata.biom.gz | products/HMP2/MBX/HMP2_metabolomics_w_metadata.biom.gz | MBX | biom.gz | 145459585 | 145.4596 | nan | nan |
| ecs_3.tsv.gz | products/HMP2/MGX/2018-05-04/ecs_3.tsv.gz | MGX | tsv.gz | 113323013 | 113.323 | 167854.0 | 1636.0 |
| ecs_relab.biom.gz | products/HMP2/MGX/2018-05-04/ecs_relab.biom.gz | MGX | biom.gz | 139831389 | 139.8314 | nan | nan |
| ecs_relab.tsv.gz | products/HMP2/MGX/2018-05-04/ecs_relab.tsv.gz | MGX | tsv.gz | 88776917 | 88.7769 | 108433.0 | 1639.0 |
| pathabundance_relab.biom.gz | products/HMP2/MGX/2018-05-04/pathabundance_relab.biom.gz | MGX | biom.gz | 14280126 | 14.2801 | nan | nan |
| pathabundance_relab.tsv.gz | products/HMP2/MGX/2018-05-04/pathabundance_relab.tsv.gz | MGX | tsv.gz | 9037246 | 9.0372 | 10884.0 | 1639.0 |
| pathabundances_3.tsv.gz | products/HMP2/MGX/2018-05-04/pathabundances_3.tsv.gz | MGX | tsv.gz | 13307715 | 13.3077 | 22113.0 | 1639.0 |
| taxonomic_profiles.biom.gz | products/HMP2/MGX/2018-05-04/taxonomic_profiles.biom.gz | MGX | biom.gz | 1704023 | 1.704 | nan | nan |
| taxonomic_profiles.tsv.gz | products/HMP2/MGX/2018-05-04/taxonomic_profiles.tsv.gz | MGX | tsv.gz | 666634 | 0.6666 | 1479.0 | 1639.0 |
| taxonomic_profiles_3.tsv.gz | products/HMP2/MGX/2018-05-04/taxonomic_profiles_3.tsv.gz | MGX | tsv.gz | 539337 | 0.5393 | 932.0 | 1639.0 |
| HMP2_proteomics_1pep-pro_1p_FDR_5ppm.biom.gz | products/HMP2/MPX/2017-03-20/HMP2_proteomics_1pep-pro_1p_FDR_5ppm.biom.gz | MPX | biom.gz | 1807367 | 1.8074 | nan | nan |

_... truncated (27 additional rows in CSV artifact) ..._

## 4. Participant and Visit Characteristics
- **Unique Participants:** `131` (`66 CD`, `38 UC`, `27 nonIBD`; `104 IBD total`).
- **Unique Physical Collection Events (`site_sub_coll`):** `2,302` (`1,769` stool events with exact `date_of_receipt`).
- **Repeated Visits:** `130/131` participants have repeated stool visits; `97/131` have repeated FC visits; `90/131` (`66 IBD`, `24 nonIBD`) have `>=2` visits with **both** shotgun MGX and same-day FC.

## 5. FC Data Availability and Quality
- **Dual Columns Resolved:** `fecalcal` (1,883 rows) and `fecalcal_ng_ml` (121 rows) are mutually exclusive per row but identical to 5 decimal places within `site_sub_coll` (`max diff < 1e-4`), yielding **2,004 rows (636 unique collection events across 98 participants)** when combined.
- **Distribution:** Range `3.45` to `493.7` (median `70.62` across unique events; `116` events `<= 15.0`; `95` events `>= 250.0`). Zero negative or non-numeric values. Full stats in `fc_summary.csv`.

## 6. Taxonomic Data Availability
- **Primary Modality (`MGX` Shotgun WGS):** `MetaPhlAn2` (`1,638` samples, `1,479` clades) and `MetaPhlAn3` (`1,638` samples, `932` clades) are **100% matched** (`1,638/1,638`) to metadata `External ID` across `130` participants (`1,595` unique collection events).
- **16S Amplicon Caveat:** The main 16S table (`178` samples) is **100% intestinal biopsy (`biopsy_16S`)** with `0` FC measurements; the Pilot 16S stool table covers only `58/180` stool 16S samples. Therefore, C1–C3 should use **shotgun MGX taxonomy**.

## 7. Functional Data Availability
- **Shotgun Functional Potential (`MGX` DNA):** `HUMAnN2` MetaCyc pathways (`10,884` rows x `1,638` samples), `HUMAnN3` pathways (`22,113` rows x `1,638` samples), `HUMAnN2` ECs (`108,433` rows x `1,638` samples), and `HUMAnN3` ECs (`167,854` rows x `1,635` samples) are verified on disk and 100% matched.
- **Active Transcription (`MTX` RNA):** `HUMAnN2` pathways (`6,061` rows x `735` samples) and ECs (`70,711` rows x `735` samples) are available for complementary RNA expression analysis.

## 8. Clinical, Medication, and Dietary (C4) Coverage
- **Clinical & Medications:** `diagnosis` (100%), `Antibiotics` (100%), `Immunosuppressants` (100%), `hbi` (`693/1,638` MGX rows), `sccai` (`452/1,638` MGX rows). Note: `SES-CD Score` (`481` rows) and `SIBDQ Score` (`110` rows) are recorded on biopsy/baseline rows (`0` direct MGX rows) and require participant/visit linkage if used.
- **C4 Dietary Recall:** 21 food-frequency variables are populated in **97.7%–98.4% of MGX visits** (`1,600–1,612 / 1,638`).

## 9. Longitudinal Matching Results
- **Same-Event (0-day) Match:** `503` unique collection events (`522` MGX runs across `95` participants) have **both** shotgun MGX (taxonomy + function) and same-day FC.
- **Causal Window Sensitivity (`fc_date <= mgx_date`):** Relaxing the causal index-FC lookup window from `0d` to `3d`, `7d`, and `14d` increases matched MGX index visits from `504` (`0d`, including 1 same-day dual-collection event) to `505` (`3d`), `522` (`7d`), and `686` (`14d`). Full breakdown in `sample_matching.csv`.

## 10. Future FC Pair Feasibility
- **Strict Future Rule (`t_future > t_index`):**
  - **All causal pairs (Index MGX -> Future FC, any index FC):** `3,867` pairs across `95` participants (`71` IBD).
  - **Complete-baseline causal pairs (Same-Day Index FC + MGX -> All Future FC):** `1,671` pairs across `93` participants (`69` IBD; `24` nonIBD).
  - **Consecutive next-FC pairs (1 target per index visit, Same-Day Index FC + MGX -> Next Available FC):** `445` pairs across `93` participants (`69` IBD, `355` IBD pairs). Median consecutive MGX+FC visit gap is **56.0 days** (IQR `28.0–70.0` days).
  - **By Horizon Window (Same-Day Index FC + MGX):**
    - `short_7_21d` (7–21d): `113` pairs across `36` participants (`31` IBD).
    - `biweekly_14_35d` (14–35d): `190` pairs across `44` participants (`39` IBD).
    - `monthly_14_56d` (14–56d): `415` pairs across `84` participants (`63` IBD).
    - `protocol_30_90d` (30–90d): `545` pairs across `88` participants (`65` IBD).

## 11. Data Leakage and Integrity Risks
1. **Unfiltered `External ID` Join Multiplication:** Prevented by filtering metadata by `data_type` or aggregating to `site_sub_coll`.
2. **Technical Replicates (`_TR` / `_P`):** Prevented by deduplicating `1,638` MGX runs to `1,595` unique `site_sub_coll` events.
3. **Retrospective Recall Columns (`In the past 6 months...`):** Flagged; only features recorded at or before `index_date` are permitted.
4. **Pairwise Horizon Fan-Out:** Using all `1,671` pairwise future outcomes creates correlated targets per index visit; primary C1 evaluation should either use **next-available future FC within a bounded horizon (e.g. 14–56d or 14–90d)** or cluster-robust mixed/bootstrap inference.

## 12. C1–C4 Feasibility Assessment & Gate Outcomes (G1–G8)
- **G1 (Linkage):** PASS — 100% of MGX/MTX/16S sample IDs link to metadata External ID when filtered by data_type; site_sub_coll uniquely links multi-omics from the same physical collection.
- **G2 (Time):** PASS — 100% of stool samples (4,352/4,352) have exact date_of_receipt (monotonic with week_num across all participants).
- **G3 (FC):** PASS WITH CAVEAT — 2,004 rows (636 unique collection events across 119 participants) have valid numeric FC; dual columns (fecalcal and fecalcal_ng_ml) must be combined via combine_first.
- **G4 (Repeated Observations):** PASS WITH CAVEAT — 90 participants (66 IBD, 23 nonIBD) have >=2 matched MGX+FC visits (445 consecutive pairs).
- **G5 (Matched Microbiome):** PASS — 522 MGX runs (503 unique collection events across 95 participants) have same-event FC; 686 MGX events (96 participants) have a causal prior/same-day FC within 14 days.
- **G6 (Functional Data):** PASS — HUMAnN2 and HUMAnN3 pathway and EC tables (shotgun WGS) are present and 100% sample-matched.
- **G7 (External Compatibility):** PENDING — IBSEN III not yet accessed.
- **G8 (Permissions):** PASS (Open IBDMDB release) / PENDING (dbGaP host genomics not required; DINE-CD/IBSEN III pending).

## 13. External Validation Readiness
- External cohort **IBSEN III** is not yet available locally (`NOT YET DETERMINABLE`). Because IBSEN III uses 16S amplicon sequencing, external validation must be restricted to harmonized genus-level taxonomy (`B5/M2`) and clinical baselines (`B1/B2`), never shotgun pathways (`B6/M3`).

## 14. Limitations
1. **FC Missingness (68.1% of MGX visits lack same-day FC):** Reduces usable MGX+FC events from `1,595` to `503` (`445` consecutive pairs).
2. **Modest Independent Participant Sample Size (`90` with >=2 MGX+FC visits [66 IBD]; `93` contributing causal pairs [69 IBD]):** Precludes high-capacity deep learning; mandates regularized linear models (`Elastic Net`) and tree ensembles (`Random Forest`) with participant-grouped cross-validation.
3. **Calibration Partition Size (~13 participants in a 15% split):** Standard split-conformal intervals on 13 clusters have wide coverage variance; **cross-conformal (CV+) grouped by participant** is strongly recommended for C3.

## 15. Recommended Next Research Decisions (To Freeze in Decision Log)
1. **D-005 (Forecast Horizon):** Adopt **Next-Available FC within 14–56 days (or 14–90 days)** as primary target, and report both consecutive-pair and fixed-window sensitivity analyses.
2. **D-006 (Matching Window):** Use **0-day (`site_sub_coll` exact match)** as primary, with **<=14-day causal prior FC** as a pre-specified sensitivity arm (which boosts index MGX visits from 504 to 686).
3. **D-007 (Validation & Calibration Design):** Use **5-fold Participant-Grouped Cross-Validation + Cross-Conformal (CV+)** on the training/development cohort (80% of participants) plus an untouched 20% participant test holdout.

## 16. Reproducibility Information
- **Pipeline CLI:** `PYTHONPATH=src python -m micro_trace.cli --config configs/audit_config.yaml`
- **All 14 CSV/JSON artifacts, 8 figures, and 9 Markdown reports** are saved in `/content/drive/MyDrive/Micro-Trace/micro_trace/reports/feasibility/`.
