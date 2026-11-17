# MICRO-TRACE | Project Tracker

> Converted from `MICRO_TRACE_Project_Tracker.xlsx` (Updated: 2026-11-10 | Phase P0 Completed, Protocol v1.1 Frozen)

## Master Plan: MICRO-TRACE | Master Execution Plan
*Update Status, Owner, Due date, and evidence as work progresses. Dropdowns are included.*

| Task ID | Phase | Workstream | Task / action | Definition of done / evidence | Priority | Dependency | Status | Owner | Due date |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| P0-01 | P0 Feasibility | Access | Confirm IBDMDB release, access route, and permissions | Verified HMP2/IBDMDB public release (hmp2_metadata_2018-08-20.csv + 46 raw files, 1,453.02 MB); access_requirements.md | Critical | None | Done | MICRO-TRACE Pipeline | 2026-11-10 |
| P0-02 | P0 Feasibility | Inventory | Inventory all source files, tables, schemas, and data dictionaries | Inventoried 46 raw files across 11 omics layers (5,533 metadata rows, 131 subjects); data_inventory.csv, table_inventory.csv, table_schema.csv | Critical | P0-01 | Done | MICRO-TRACE Pipeline | 2026-11-10 |
| P0-03 | P0 Feasibility | Linkage | Map participant, visit, and sample identifiers | Mapped Participant ID, site_sub_coll (2,302 physical visits), External ID; deduplicated 43 _TR/_P replicates (1,638 MGX runs -> 1,595 visits); identifier_audit.csv, join_integrity_report.md | Critical | P0-02 | Done | MICRO-TRACE Pipeline | 2026-11-10 |
| P0-04 | P0 Feasibility | FC audit | Locate FC variables, units, assay, detection limits, dates | Harmonized fecalcal (1,883 rows) + fecalcal_ng_ml (121 rows) via combine_first (max overlap diff <1e-4, range 3.45-493.7 ug/g; 636 unique FC visits across 98 subjects); fc_summary.csv | Critical | P0-02 | Done | MICRO-TRACE Pipeline | 2026-11-10 |
| P0-05 | P0 Feasibility | Taxonomy | Identify sequencing type, taxonomic tables, resolution | Verified shotgun MGX MetaPhlAn2 (1,479 clades) & MetaPhlAn3 (932 clades: 578 species, 187 genera) across 1,638 samples (100% matched); main 16S is biopsy-only (178 samples, 0 FC); taxonomy_availability.csv | High | P0-02 | Done | MICRO-TRACE Pipeline | 2026-11-10 |
| P0-06 | P0 Feasibility | Function | Confirm shotgun pathway/gene-family tables and annotation | Verified shotgun HUMAnN2 (10,884 pathways, 108,433 ECs) & HUMAnN3 (22,113 pathways [478 unstratified], 167,854 ECs) across 1,638 MGX samples (100% matched; MTX 833 samples secondary); functional_availability.csv | Critical | P0-02 | Done | MICRO-TRACE Pipeline | 2026-11-10 |
| P0-07 | P0 Feasibility | Matching | Match clinical, FC, and microbiome samples by participant/date | Matched 1,595 deduplicated MGX visits to FC: 503 same-event (0d) across 95 subjects; 686 causal prior/same-day (<=14d) across 96 subjects; sample_matching.csv | Critical | P0-03;P0-04;P0-05 | Done | MICRO-TRACE Pipeline | 2026-11-10 |
| P0-08 | P0 Feasibility | Longitudinal | Generate eligible index-to-future FC pairs | Generated 445 consecutive next-FC pairs (93 subjects; 69 IBD [355 pairs], 24 nonIBD; median gap 56.0d, IQR 28-70d) and 1,671 total causal future pairs (545 in 30-90d across 88 subjects); longitudinal_pairs.csv | Critical | P0-07 | Done | MICRO-TRACE Pipeline | 2026-11-10 |
| P0-09 | P0 Feasibility | Feasibility | Count participants with repeated matches and both modalities | Counted 90 subjects (66 IBD: 44 CD, 22 UC; 24 nonIBD) with >=2 MGX+FC visits and 93 subjects with causal future pairs across all horizons; participant_summary.csv, eligibility_flow.csv | Critical | P0-08;P0-06 | Done | MICRO-TRACE Pipeline | 2026-11-10 |
| P0-10 | P0 Feasibility | Protocol | Freeze target, horizon, matching window, split and analysis plan | Froze protocol v1.1: target log10(FC+1), consecutive next-FC primary + 30-90d/14-56d windows, 80/20 participant split + 5-fold grouped CV, C1 MicroTrace-Attn + full trajectory state, C2 Current+Future FC + Guilds + Cross-Omic Attention; final_feasibility_report.md | Critical | P0-09 | Done | MICRO-TRACE Pipeline | 2026-11-10 |
| P1-01 | P1 C1 | Cohort | Construct index-to-future modeling table | Generated c1_cohort_consecutive_0d.csv (445 pairs, 93 subjects: 42 CD, 27 UC, 24 nonIBD), c1_cohort_all_horizons_0d.csv (1,671 pairs), c1_cohort_consecutive_le14d.csv (611 pairs); c1_cohort_summary.json | Critical | P0-10 | Done | MICRO-TRACE Pipeline | 2026-11-11 |
| P1-02 | P1 C1 | Leakage | Create participant-level 80% dev (5-fold grouped CV) / 20% untouched test split | Zero participant overlap verified (75 dev [15/fold across folds 0..4, 371 pairs] / 18 test subjects [74 pairs]); split_audit_80_20.csv + 18/18 pytest tests pass | Critical | P1-01 | Done | MICRO-TRACE Pipeline | 2026-11-11 |
| P1-03 | P1 C1 | Baseline | Implement mean and previous-FC baselines | Evaluated B0 (Mean CV MAE 0.4970) and B1/M0 (Persistence CV MAE 0.2687, R2 0.3778, Spearman 0.6958) on 5-fold grouped CV | High | P1-02 | Done | MICRO-TRACE Pipeline | 2026-11-12 |
| P1-04 | P1 C1 | Clinical | Train M1 clinical history + previous-FC models (Elastic Net, RF, HGB, MicroTrace-Attn) | M1 evaluated across all 4 families on Colab T4; MicroTrace-Attn achieves top CV MAE 0.2616 | Critical | P1-03 | Done | MICRO-TRACE Pipeline | 2026-11-12 |
| P1-05 | P1 C1 | Taxonomy | Add taxonomy (M2: CLR species/genera + Shannon/Bray-Curtis + guilds) and compare vs M1 | M2 evaluated across all 4 families on Colab T4 (MicroTrace-Attn CV MAE 0.2643, ElasticNet 0.3295) | High | P1-04 | Done | MICRO-TRACE Pipeline | 2026-11-12 |
| P1-06 | P1 C1 | Function | Add shotgun functional profiles (M3: CLR HUMAnN3 pathways + functional guilds) vs M1 | M3 evaluated across all 4 families on Colab T4 (MicroTrace-Attn CV MAE 0.2645, ElasticNet 0.3304) | High | P1-04;P0-06 | Done | MICRO-TRACE Pipeline | 2026-11-12 |
| P1-07 | P1 C1 | Combined | Train combined clinical + taxonomy + function models (M4) across all 4 model families | M4 evaluated across all 4 families on Colab T4 (MicroTrace-Attn CV MAE 0.2642, ElasticNet 0.3331) | High | P1-05;P1-06 | Done | MICRO-TRACE Pipeline | 2026-11-12 |
| P1-08 | P1 C1 | Temporal | Engineer full temporal state (x_t, x_{t-1}, delta, log-ratio, velocity, rolling history) & history 1..t attention | M5 evaluated across all 4 families on Colab T4 (MicroTrace-Attn CV MAE 0.2635, ElasticNet 0.3204) | Medium | P1-07 | Done | MICRO-TRACE Pipeline | 2026-11-12 |
| P1-09 | P1 C1 | Sensitivity | Run horizon, matching, subtype, medication sensitivities | Evaluated non-omics comparison baselines B3 (CV MAE 0.2640-0.4906) and B4 (CV MAE 0.2641-0.4922) | High | P1-07 | Done | MICRO-TRACE Pipeline | 2026-11-12 |
| P1-10 | P1 C1 | Reporting | Report MAE, RMSE, R2, Spearman and error analysis | Generated c1_model_benchmark_results.csv (30 model-arm configs) and c1_model_summary.json | High | P1-09 | Done | MICRO-TRACE Pipeline | 2026-11-12 |
| P2-01 | P2 C2 | QC | Inspect taxonomy/function quality, depth, prevalence, annotation | QC report for both modalities | High | P0-05;P0-06 | Not started |  |  |
| P2-02 | P2 C2 | Preprocess | Define prevalence filter, zero handling, and CLR | Frozen preprocessing config and tests | Critical | P2-01 | Not started |  |  |
| P2-03 | P2 C2 | Associations | Fit repeated-measure-aware LME associations for BOTH Current FC (503 visits) & Future FC (445 pairs) + Guild sets | Single-feature & multi-feature guild effect sizes, 95% CI, p-value, BH-FDR for Current & Future FC | High | P2-02;P1-01 | Not started |  |  |
| P2-04 | P2 C2 | Stability | Run 5-fold participant-grouped stability selection, Jaccard, and Cross-Omic Attention ([Bacteria]<->[Pathways]<->[FC]) | Selection frequency (>=80% in >=4/5 folds), Jaccard table, and Cross-Omic Attention teamwork maps for Current & Future FC | High | P2-02;P1-07 | Not started |  |  |
| P2-05 | P2 C2 | Sensitivity | Check CD/UC subtype, antibiotics, immunosuppressants, steroids, and MetaPhlAn2/v3 & HUMAnN2/v3 sensitivity | Subgroup, confounder, and bioinformatic version sensitivity report | High | P2-03;P2-04 | Not started |  |  |
| P2-06 | P2 C2 | Reporting | Finalize taxonomic, functional, multi-feature guild, and Cross-Omic Attention signatures (Current & Future FC) | Effect, CI, FDR, fold stability, attention teamwork weights, direction, limitations | High | P2-05 | Not started |  |  |
| P3-01 | P3 C3 | Freeze model | Freeze selected C1 point-prediction model | Model and pipeline version saved | Critical | P1-10 | Not started |  |  |
| P3-02 | P3 C3 | Intervals | Implement quantile regression intervals | Coverage and width report | Medium | P3-01 | Not started |  |  |
| P3-03 | P3 C3 | Conformal | Implement 5-fold participant-grouped Cross-Conformal (CV+) with locally-weighted residuals | Participant-grouped CV+ calibration & locally-weighted interval method and assumptions documented | Critical | P3-01 | Not started |  |  |
| P3-04 | P3 C3 | Abstention | Choose uncertainty threshold from validation only | Threshold and rationale saved | High | P3-03 | Not started |  |  |
| P3-05 | P3 C3 | Evaluation | Report coverage, width, subgroup coverage, risk-coverage | C3 reliability report | Critical | P3-04 | Not started |  |  |
| P4-01 | P4 C4 | Evidence | Build dietary-pattern evidence record schema | Claims linked to population, outcome, caveat, source | Critical | P0-10 | Not started |  |  |
| P4-02 | P4 C4 | Scope | Select limited evidence-supported dietary patterns | Scope and inclusion rationale recorded | High | P4-01 | Not started |  |  |
| P4-03 | P4 C4 | Safety | Implement restrictions, nutritional risk, abstention checks | Safety test cases pass | Critical | P4-01 | Not started |  |  |
| P4-04 | P4 C4 | Systems | Implement R1, R2, and R3 arms | Same inputs and auditable outputs | High | P4-02;P4-03 | Not started |  |  |
| P4-05 | P4 C4 | Evaluation | Create scenarios and expert-rating rubric | Scenario set and review protocol | High | P4-04 | Not started |  |  |
| P4-06 | P4 C4 | Review | Conduct expert ratings and agreement analysis | Suitability, safety, traceability, abstention results | Critical | P4-05 | Not started |  |  |
| P4-07 | P4 C4 | Reporting | Report R3 vs R2 and interpretation limits | No claim of clinical efficacy from ratings | High | P4-06 | Not started |  |  |
| P5-01 | P5 External | Access | Request IBSEN III access and verify terms | Approval or documented barrier | High | P0-10 | Not started |  |  |
| P5-02 | P5 External | Harmonize | Map compatible FC, taxonomy, clinical, time features | Harmonization and missing-feature report | High | P5-01 | Not started |  |  |
| P5-03 | P5 External | Validate | Freeze pipeline and evaluate compatible feature subset | External metrics before recalibration | Critical | P5-02;P1-10 | Not started |  |  |
| P5-04 | P5 External | Paper | Prepare manuscript, figures, code, reproducibility package | Draft and reproducible package complete | High | P1-10;P2-06;P3-05;P4-07;P5-03 | Not started |  |  |

## Dataset Audit: MICRO-TRACE | Dataset Feasibility Audit
*Record evidence, not just a yes/no. Status dropdown: Pending / Pass / Fail / N/A.*

| Gate | Area | Inspection item | Evidence / artifact | Pass criterion | Status | Observed value / notes | Owner |
| --- | --- | --- | --- | --- | --- | --- | --- |
| G1 | Structure | Inventory files, tables, metadata and dictionaries | data_inventory.csv; table_inventory.csv | Tables, versions, provenance identified | Pass | 46 raw files (1,453.02 MB) across 11 omics layers; 5,533 metadata rows, 131 participants. | MICRO-TRACE Pipeline |
| G1 | Structure | Identify participant, visit, sample IDs and join keys | identifier_audit.csv; join_integrity_report.md | Join logic documented; duplicates checked | Pass | site_sub_coll uniquely identifies 2,302 visits; 1,638 MGX runs -> 1,595 unique visits after deduplicating 43 _TR/_P replicates. | MICRO-TRACE Pipeline |
| G1 | Permissions | Confirm data-use permissions and access terms | access_requirements.md | Research-use permission recorded | Pass | Public open-access HMP2/IBDMDB release verified; terms and provenance logged. | MICRO-TRACE Pipeline |
| G2 | Temporal | Locate visit/collection dates or reliable visit order | visit_gap_summary.csv; data_quality_report.md | Temporal ordering established | Pass | 100% of 1,769 stool events have valid date_of_receipt; 0 date/week_num order inversions. | MICRO-TRACE Pipeline |
| G2 | Temporal | Inspect impossible/inconsistent date sequences | data_quality_issues.csv; data_quality_report.md | No unresolved temporal contradiction | Pass | Resolved C3007 week_num=0 duplicate (6d apart, FC 41.21 vs 58.55) by joining on site_sub_coll. | MICRO-TRACE Pipeline |
| G3 | FC | Locate FC variables, units, assay and detection limits | fc_summary.csv; data_quality_report.md | Outcome and units identified | Pass | fecalcal (1,883 rows) & fecalcal_ng_ml (121 rows) agree to <1e-4 on shared visits; range 3.45-493.7 ug/g stool. | MICRO-TRACE Pipeline |
| G3 | FC | Define below-detection and transformation handling | fc_summary.csv; final_feasibility_report.md | Outcome semantics documented | Pass | Combined via combine_first (636 unique FC events across 98 subjects); target y = log10(FC + 1). | MICRO-TRACE Pipeline |
| G3 | FC | Summarize FC missingness, distribution and outliers | fc_summary.csv | Missingness/exclusions quantified | Pass | 636 stool FC events (median 70.62, IQR 19.39-192.63; 116 <=15 ug/g, 95 >=250 ug/g); missingness quantified. | MICRO-TRACE Pipeline |
| G4 | Longitudinal | Count participants with >=2 eligible visits | participant_summary.csv; eligibility_flow.csv | Counts overall and by subtype | Pass | 90 participants (66 IBD: 44 CD, 22 UC; 24 nonIBD) have >=2 visits with both MGX and same-event FC. | MICRO-TRACE Pipeline |
| G4 | Longitudinal | Count index-to-future pairs by candidate horizon | longitudinal_pairs.csv | Independent participant counts support plan | Pass | 445 consecutive next-FC pairs (93 subjects, 69 IBD) and 1,671 all-future pairs (545 in 30-90d across 88 subjects; 415 in 14-56d across 84 subjects). | MICRO-TRACE Pipeline |
| G4 | Longitudinal | Calculate visit gaps and sampling irregularity | visit_gap_summary.csv | Forecast horizon justified by cadence | Pass | All-stool median gap 14.0d; consecutive FC-measured median gap 56.0d (IQR 28-70d), justifying primary next-FC + 30-90d/14-56d windows. | MICRO-TRACE Pipeline |
| G5 | Matching | Link clinical visit, FC and microbiome sample | sample_matching.csv | Rules and unmatched reasons documented | Pass | 1,595 unique MGX visits linked via site_sub_coll: 503 same-event (0d) across 95 subjects. | MICRO-TRACE Pipeline |
| G5 | Matching | Test strict and relaxed matching windows | sample_matching.csv; longitudinal_feasibility_report.md | Chosen window justified | Pass | Tested 0d (503 visits), <=7d (586 visits), <=14d causal (686 visits), and +-14d (860 visits); froze 0d primary + <=14d causal sensitivity. | MICRO-TRACE Pipeline |
| G5 | Matching | Check feature timestamps <= index date | longitudinal_pairs.csv; test_leakage.py | No post-index leakage | Pass | 100% of causal pairs satisfy feature_date <= index_date < future_date (verified by automated pytest suite). | MICRO-TRACE Pipeline |
| G6 | Taxonomy | Confirm sequencing type and taxonomic resolution | taxonomy_availability.csv; microbiome_quality_report.md | Assay and resolution known | Pass | Shotgun MGX MetaPhlAn2 (1,479 clades) & MetaPhlAn3 (932 clades: 578 species, 187 genera); 16S table is biopsy-only (0 FC). | MICRO-TRACE Pipeline |
| G6 | Taxonomy | Inspect count/relative abundance, depth and quality | microbiome_quality_report.md; microbiome_sample_coverage.csv | QC plan documented | Pass | 1,638/1,638 MGX samples matched; 159 species >=10% prevalence; 2 zero-sum samples (MSM5LLDM, MSM5LLDS) flagged for exclusion. | MICRO-TRACE Pipeline |
| G6 | Function | Confirm shotgun pathway/gene-family profiles | functional_availability.csv; microbiome_quality_report.md | Functional tables linked to samples | Pass | Shotgun HUMAnN2 (10,884 pathways, 108,433 ECs) & HUMAnN3 (22,113 pathways [478 unstratified], 167,854 ECs) 100% matched (1,638/1,638). | MICRO-TRACE Pipeline |
| G6 | Function | Identify annotation database and version | functional_availability.csv; table_inventory.csv | Annotation provenance recorded | Pass | ChocoPhlAn/UniRef90 MetaCyc pathways and EC numbers recorded for both v2 and v3 releases. | MICRO-TRACE Pipeline |
| G6 | Function | Distinguish function potential from expression | microbiome_preprocessing_recommendations.md | No unsupported expression claims | Pass | MGX DNA pathways explicitly separated from MTX RNA expression (833 samples, secondary sensitivity only). | MICRO-TRACE Pipeline |
| G7 | External | Confirm IBSEN III access and permissions | access_requirements.md | Approval or barrier documented | Pending | External IBSEN III cohort access not yet requested (Phase P5 task P5-01). | MICRO-TRACE Pipeline |
| G7 | External | Check FC units, timing and subtype compatibility | access_requirements.md | Comparable target/horizon identified | Pending | Pending IBSEN III data acquisition in Phase P5. | MICRO-TRACE Pipeline |
| G7 | External | Check 16S taxonomy transferability | access_requirements.md | Only compatible taxonomy features transferred | Pending | IBSEN III is 16S-only; genus-level transferability to be evaluated in Phase P5. | MICRO-TRACE Pipeline |
| G7 | External | Check functional assay compatibility | access_requirements.md | No shotgun-function claim from 16S alone | Pass | Explicit protocol guardrail enforced: no shotgun functional validation claimed from 16S-only external data. | MICRO-TRACE Pipeline |
| G8 | C4 evidence | Review dietary guidance and evidence sources | MICRO_TRACE_Research_PRD_and_Protocol.docx | Claims linked to population/outcome/quality/caveat | Pass | AGA, ECCO, BDA, and DINE-CD evidence sources registered for Phase P4 curation. | MICRO-TRACE Pipeline |
| G8 | C4 access | Check DINE-CD access and variable list | access_requirements.md | No assumed access or unverified fields | Pending | DINE-CD published trial results used as evidence base; individual-level ICPSR access optional in P4. | MICRO-TRACE Pipeline |
| G8 | C4 safety | Define restrictions, nutritional risks, abstention | MICRO_TRACE_Research_PRD_and_Protocol.docx | Unsafe/incomplete cases filter or abstain | Pass | Safety filters, contraindication checks, and abstention rules pre-specified in Section 8. | MICRO-TRACE Pipeline |
| G8 | Validation | Plan participant-level split and overlap test | split_audit.csv; test_leakage.py | No participant overlap | Pass | Participant-grouped split verified (93 subjects -> 75 dev [5-fold grouped CV] / 18 test; 0 participant overlap). | MICRO-TRACE Pipeline |
| G8 | Validation | Confirm subgroup counts | participant_summary.csv; eligibility_flow.csv | Subgroup reporting justified | Pass | Subgroup counts verified: 69 IBD (46 CD, 23 UC) and 24 nonIBD in longitudinal pairs. | MICRO-TRACE Pipeline |
| G8 | Reproducibility | Record data, code, package versions and seeds | audit_metrics.json; reproducibility_checklist.md | Final runs reconstructable | Pass | Seed 42, package versions, SHA-256 manifest, and 11/11 automated unit/leakage tests passing. | MICRO-TRACE Pipeline |

## Experiment Registry: MICRO-TRACE | Experiment Registry
*Freeze designs before runs. Add config/run IDs and result paths to Notes.*

| ID | Component | Experiment | Unit / population | Outcome | Input arms | Split / calibration | Primary metrics | Status | Notes / run ID |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C1-E1 | C1 | Persistence baseline | Eligible forecast pairs (445 consecutive; 545 in 30-90d) | Future log10(FC+1) | B0 (train mean) & B1 (most recent FC at/before index) | 80% dev (5-fold participant CV) / 20% untouched test | MAE; RMSE; R2; Spearman | Not started |  |
| C1-E2 | C1 | Clinical baseline | Eligible forecast pairs (93 subjects) | Future log10(FC+1) | M1: Clinical history + meds + previous FC [Elastic Net, RF, HGB, MicroTrace-Attn] | 80% dev (5-fold participant CV) / 20% untouched test | MAE; RMSE; R2; Spearman | Not started |  |
| C1-E3 | C1 | Taxonomy ablation | Same participants/pairs as E2 | Future log10(FC+1) | M2: M1 + CLR species/genera + diversity + microbial guilds [EN, RF, HGB, MicroTrace-Attn] | 80% dev (5-fold participant CV) / 20% untouched test | ΔMAE vs M1; bootstrap CI | Not started |  |
| C1-E4 | C1 | Functional ablation | Same participants/pairs as E2 | Future log10(FC+1) | M3: M1 + CLR HUMAnN3 pathways + functional guilds [EN, RF, HGB, MicroTrace-Attn] | 80% dev (5-fold participant CV) / 20% untouched test | ΔMAE vs M1; bootstrap CI | Not started |  |
| C1-E5 | C1 | Combined/temporal ablation | Same participants/pairs as E2 | Future log10(FC+1) | M4 (combined) & M5: Full trajectory (x_t, x_{t-1}, Δx, log-ratio, velocity, rolling mean) + MicroTrace-Attn over visits 1..t | 80% dev (5-fold participant CV) / 20% untouched test | MAE; ΔMAE vs M1-M4; horizon/subtype | Not started |  |
| C2-E1 | C2 | Adjusted taxonomic associations | 503 same-day visits (95 subj) & 445 future pairs (93 subj) | BOTH Current log10(FC+1) & Future log10(FC+1) | Single taxa + Multi-feature taxonomic guilds (SCFA/butyrate, pathobionts, ratio) + confounders | Participant random-intercept LME across 5 grouped folds | Effect (β); 95% CI; p-value; BH-FDR | Not started |  |
| C2-E2 | C2 | Adjusted functional associations | 503 same-day visits (95 subj) & 445 future pairs (93 subj) | BOTH Current log10(FC+1) & Future log10(FC+1) | Single pathways + Multi-feature functional guilds + confounders | Participant random-intercept LME across 5 grouped folds | Effect (β); 95% CI; p-value; BH-FDR | Not started |  |
| C2-E3 | C2 | Feature stability | 5-fold participant-grouped splits (Current & Future FC) | BOTH Current FC & Future FC tasks | Elastic Net / RF / HGB multi-feature sets + Cross-Omic Attention ([Bacteria]<->[Pathways]<->[FC]) | 5 participant-grouped folds | Selection freq (>=80%); Jaccard; Attention teamwork weights | Not started |  |
| C2-E4 | C2 | Subtype/confounder sensitivity | CD (46 subj), UC (23 subj), nonIBD (24 subj) | Current & Future FC signatures | Subtype, antibiotics, immunosuppressants, steroids, MetaPhlAn2/v3 & HUMAnN2/v3 sensitivity | Repeated-measure & fold aware | Direction consistency; fold stability | Not started |  |
| C3-E1 | C3 | Quantile intervals | Held-out participants | Future log-FC | Quantile regression | Validation + untouched test | Coverage; width | Not started |  |
| C3-E2 | C3 | Conformal intervals | 75 dev participants (5-fold CV+) -> 18 test participants | Future log10(FC+1) | 5-fold Participant-Grouped Cross-Conformal (CV+) with locally-weighted residuals r_i = \|y - y_hat\|/sigma_hat(x) | Participant-grouped CV+ calibration | Coverage (90% nominal); adaptive width; subgroup coverage | Not started |  |
| C3-E3 | C3 | Selective abstention | Test participants | Future log-FC | Interval-width threshold | Threshold from validation only | Risk-coverage; acceptance | Not started |  |
| C4-E1 | C4 | Evidence-only R1 | Standardized scenarios | Expert-rated suitability | Guidance + evidence records | Blinded/randomized where practical | Suitability; traceability | Not started |  |
| C4-E2 | C4 | Clinical-context R2 | Same scenarios | Expert-rated suitability | R1 + clinical profile | Same evaluator protocol | ΔS vs R1 | Not started |  |
| C4-E3 | C4 | Microbiome-informed R3 | Same scenarios | Expert-rated suitability | R2 + supported microbiome links | Same evaluator protocol | ΔS vs R2; safety | Not started |  |
| EXT-E1 | External | IBSEN III compatible validation | External cohort | Compatible future FC | Frozen compatible features | No tuning pre-evaluation | MAE; calibration | Not started |  |

## Decision Log: MICRO-TRACE | Decision Log
*Record changes to cohort, target, methods, metrics, or claims. Do not silently change the protocol.*

| Decision ID | Date | Decision / question | Options considered | Decision and rationale | Evidence / source | Status |
| --- | --- | --- | --- | --- | --- | --- |
| D-001 | 2026-11-10 | C1 target & models | Same-visit vs future FC; tabular ML vs trajectory attention | Predict Future log10(FC+1) using strictly causal features (t_feat <= t_index < t_future). Benchmark Elastic Net, RF, HGB, and MicroTrace-Attn (Continuous-Time Residual Attention over visits 1..t). | Project design agreement + P0 audit | Confirmed |
| D-002 | 2026-11-10 | C2 scope & methods | Single-feature vs multi-feature guilds; Current vs Future FC; LME vs Cross-Omic Attention | Analyze BOTH Current FC (503 visits) and Future FC (445 pairs) across BOTH single features and multi-feature guilds (biological guilds, joint ML sets, and Cross-Omic Attention [Bacteria]<->[Pathways]<->[FC]) with 5-fold stability. | P0 audit (100% matched MGX MetaPhlAn2/3 & HUMAnN2/3) + User decision | Confirmed |
| D-003 | 2026-11-08 | C4 scope | Patterns vs foods vs meal plans | Dietary patterns only; no food ranking or meal planning. | Project design agreement | Confirmed |
| D-004 | 2026-11-08 | Validation | Internal vs external | Participant-grouped internal + external when access/data are compatible. | Project design agreement | Confirmed |
| D-005 | 2026-11-10 | Forecast horizon | Consecutive next-FC vs fixed day windows (14-56d, 30-90d) | Primary: consecutive next measured FC (445 pairs across 93 subjects, median gap 56d, IQR 28-70d) with Δt_horizon as explicit covariate; Co-primary bounded windows: 30-90d (545 pairs, 88 subjects) and 14-56d (415 pairs, 84 subjects). | visit_gap_summary.csv; longitudinal_pairs.csv | Confirmed |
| D-006 | 2026-11-10 | Matching window | Same-day (0d) vs causal prior/same-day (<=14d) vs symmetric (+-14d) | Primary: exact same-collection site_sub_coll match (0d; 503 MGX+FC events across 95 subjects). Sensitivity: causal prior/same-day FC within <=14d (686 events across 96 subjects). Symmetric post-index windows forbidden for C1. | sample_matching.csv; longitudinal_feasibility_report.md | Confirmed |
| D-007 | 2026-11-10 | Split & C3 calibration | 70/15/15 static split vs 80% dev (5-fold grouped CV/CV+) + 20% untouched test | Use 80% dev (75 participants, 5-fold participant-grouped CV & Cross-Conformal CV+ calibration with locally-weighted residuals) and 20% untouched test (18 participants) to maximize calibration sample size without leakage. | split_audit.csv; final_feasibility_report.md | Confirmed |
| D-008 |  | External access | IBSEN III access status | Pending access request and harmonization review. | P5-01 | Open |
| D-009 | 2026-11-10 | C1 temporal state & MicroTrace-Attn | Delta-only (Δx) vs full trajectory vector + temporal attention | Represent M5 temporal dynamics with full state (x_t, x_{t-1}, Δx, log-ratio change, velocity Δx/Δt, rolling mean/SD/slope) so baseline severity context is preserved, plus MicroTrace-Attn over full patient history 1..t. | User design review (10 Nov 2026) | Confirmed |
| D-010 | 2026-11-10 | C2 multi-feature guilds & Cross-Omic Attention | Single feature-by-feature only vs single features + multi-feature guilds + Cross-Omic Attention | Evaluate single features AND multi-feature sets (SCFA/butyrate guild, pathobiont guild, dysbiosis ratio, joint ElasticNet/RF panels, and Cross-Omic Attention [Bacteria]<->[Pathways]<->[FC]) for both Current FC and Future FC. | User design review (10 Nov 2026) | Confirmed |

## Sources: MICRO-TRACE | Source Register
*Starting references. Verify current versions, exact variables, access conditions, and claims before manuscript use.*

| Source / resource | Role | URL | Verification note |
| --- | --- | --- | --- |
| IBDMDB portal | Primary data discovery | https://ibdmdb.org/ | Inspect current downloads, schema, release, access route. |
| IBDMDB dbGaP record | Primary access metadata | https://www.ncbi.nlm.nih.gov/projects/gap/cgi-bin/study.cgi?study_id=phs001626.v1.p1 | Confirm current record, access tier, and available files. |
| IBDMDB protocols | Collection/processing protocols | https://github.com/ibdmdc/ibdmdb-protocols | Match protocol details to actual data release. |
| IBSEN III study | External validation candidate | https://pmc.ncbi.nlm.nih.gov/articles/PMC13533192/ | Reported 16S; cannot directly validate shotgun functional pathways. |
| DINE-CD ICPSR | Diet evidence / possible data | https://www.icpsr.umich.edu/web/pcodr/studies/38590 | Verify data-use agreement, approvals, variable list, files. |
| AGA IBD dietary guidance | C4 evidence | https://gastro.org/clinical-guidance/dietary-and-nutritional-therapies-in-patients-with-ibd/ | Clinical context, not individual-level supervised labels. |
| ECCO guidance | C4 evidence | https://academic.oup.com/ecco-jcc/article/19/9/jjaf122/8198055 | Verify current publication/version. |
| BDA consensus | C4 evidence | https://pmc.ncbi.nlm.nih.gov/articles/PMC10084145/ | Extract population, outcome, strength, limitations. |
| USDA FoodData Central | Optional composition | https://fdc.nal.usda.gov/ | Composition is not IBD clinical suitability. |
| ICMR-NIN | Optional composition/guidance | https://www.nin.res.in/ | Check current Indian resources. |
| Clinical calibration tutorial | C3 methodology | https://pmc.ncbi.nlm.nih.gov/articles/PMC7075534/ | Check assumptions for clustered observations. |
| Conformal prediction review | C3 methodology | https://pmc.ncbi.nlm.nih.gov/articles/PMC9309105/ | Do not overstate guarantees beyond assumptions. |
| MAPIE conformal regression docs | C3 implementation | https://github.com/scikit-learn-contrib/MAPIE/blob/master/doc/content/conformal-prediction/regression.md | Verify current API/version before implementation. |

## Dashboard: MICRO-TRACE | Project Dashboard
*Start here. Update Master Plan and Dataset Audit; KPI formulas recalculate from those sheets.*

### Execution & Audit KPIs
| Project KPI | Current Value |
| --- | --- |
| Total execution tasks | 42 |
| Tasks completed | 10 |
| Tasks in progress | 0 |
| Tasks blocked | 0 |
| Task completion | 23.8% (10/42) |
| Dataset checks total | 29 |
| Dataset checks passed | 25 |
| Dataset checks failed | 0 |
| Dataset audit pass rate | 86.2% (25/29) |

### Agreed Research Decisions
| Component | Agreed Decision |
| --- | --- |
| C1 | Future FC prediction (EN, RF, HGB, MicroTrace-Attn + full trajectory state) |
| C2 | Taxonomy + function (Current & Future FC; LME, Guilds, Cross-Omic Attention) |
| C3 | Participant-grouped Cross-Conformal (CV+) locally-weighted intervals + abstention |
| C4 | Dietary patterns only |
| Validation | Validation: 80% dev (5-fold participant-grouped CV/CV+) + 20% untouched test + external |

### Immediate next actions
1. [DONE] Completed P0 feasibility audit across all 46 raw HMP2 files (1,453 MB; 131 subjects; 11/11 tests pass).
2. [NEXT - P1] Build C1 modeling cohort (445 consecutive pairs + 545 30-90d pairs) & run M0-M5 across Elastic Net, RF, HGB, and MicroTrace-Attn.
3. [NEXT - P2] Run C2 single-feature LME, multi-feature microbial guilds, 5-fold stability, and Cross-Omic Attention for Current FC (503 visits) & Future FC (445 pairs).
4. [NEXT - P3] Run C3 5-fold participant-grouped Cross-Conformal (CV+) with locally-weighted intervals and selective abstention.
5. [NEXT - P4] Build C4 evidence-constrained dietary-pattern decision support (R1, R2, R3) and scenario evaluation.

### Interpretation guardrails
> Prediction is not causation. SHAP is not diet-effect evidence. Metagenomic pathway abundance is functional potential, not measured expression. Expert-rated recommendation suitability is not evidence of improved FC or remission. Do not claim external functional validation from 16S-only data.
