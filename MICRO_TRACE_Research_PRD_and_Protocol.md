**RESEARCH PRODUCT REQUIREMENTS & EXPERIMENTAL PROTOCOL**

# MICRO-TRACE


Microbiome-Informed Temporal and Reliable Analysis for Calprotectin Estimation


Working title: Longitudinal Microbiome-Informed Prediction of Intestinal Inflammation with Uncertainty-Aware, Evidence-Guided Dietary Decision Support in IBD


*Version 1.1  |  10 November 2026  |  Status: Phase P0 Data Feasibility Confirmed; Protocol Frozen for C1–C4 Execution*


> **Purpose**
> This document is the project’s working specification. It records the agreed research scope, experimental protocols, dataset feasibility checks, implementation sequence, acceptance criteria, and decision boundaries. Use the companion Excel tracker to record execution status and evidence.


| Decision area | Final decision |
| --- | --- |
| C1 — Prediction target | Future fecal calprotectin log10(FC+1) using strictly causal features (t_feat <= t_index < t_future), benchmarking Elastic Net, Random Forest, HistGradientBoosting, and MicroTrace-Attn (Continuous-Time Residual Attention over visits 1..t with full trajectory state vectors). |
| C2 — Microbiome resolution | Both taxonomic (MetaPhlAn3/v2) and shotgun functional potential (HUMAnN3/v2), analyzing BOTH Current FC (503 visits) and Future FC (445 pairs) across single features, multi-feature microbial/functional guilds, 5-fold stability selection, and Cross-Omic Attention ([Bacteria]<->[Pathways]<->[FC]). |
| C3 — Reliability | 5-fold participant-grouped Cross-Conformal (CV+) intervals with locally-weighted patient-specific widths, quantile regression comparison, subgroup calibration audit, and selective abstention. |
| C4 — Decision support | Dietary-pattern recommendations only; no individual-food or meal-plan generator. |
| Validation | 80% development (75 participants, 5-fold participant-grouped CV/CV+) + 20% untouched test (18 participants) with zero participant overlap, plus external validation where compatible. |


### How to use this document

- Treat the data feasibility gates as go/no-go requirements, not paperwork.
- Use the Master Plan and Dataset Audit tabs in the companion workbook to track evidence and progress.
- Do not change primary outcomes, horizons, splits, or evaluation metrics after looking at the final test results.
- Record every material protocol change in the Decision Log with a reason and date.

### Document control


| Field | Value |
| --- | --- |
| Project name | MICRO-TRACE |
| Document type | Research PRD + experimental protocol + implementation plan |
| Version | 1.1 (10 November 2026) |
| Status | Phase P0 Data Feasibility Confirmed (11/11 tests pass); Protocol Frozen for C1–C4 Execution |
| Primary population | People with Crohn’s disease or ulcerative colitis represented in the selected datasets |
| Primary outcome | Future log10(FC + 1) for C1/C3; Both Current log10(FC + 1) and Future log10(FC + 1) for C2 |
| Primary data candidate | IBDMDB / HMP2 verified locally & on Drive (46 files, 1,453 MB, 131 subjects, 1,595 MGX visits, 503 same-day MGX+FC visits, 445 consecutive pairs) |
| External validation candidate | IBSEN III, pending data access and feature compatibility |
| Diet evidence candidate | DINE-CD plus current clinical dietary guidance |


## 1. Executive summary


MICRO-TRACE is a four-part research programme connecting longitudinal inflammation prediction, reproducible microbial signal discovery, uncertainty-aware forecasting, and evidence-constrained dietary-pattern decision support in inflammatory bowel disease (IBD).


The core scientific question is whether historical microbial and clinical information—modeled via both regularized/tree-based learners and Continuous-Time Residual Attention (MicroTrace-Attn) over full trajectory state vectors—improves prediction of future FC beyond a strong clinical baseline that includes the most recent FC. The project then identifies reproducible taxonomic, functional, multi-feature guild, and Cross-Omic Attention signatures for both Current FC (same-visit inflammation) and Future FC (early-warning transition), quantifies patient-specific prediction uncertainty via participant-grouped Cross-Conformal (CV+) intervals with selective abstention, and evaluates whether microbiome information improves expert-rated dietary-pattern decision support.


> **Non-negotiable interpretation boundary**
> Predictive association is not causation. SHAP describes how a fitted model uses features; it does not show that changing a microbe or diet will change FC. C4 evaluates decision-support quality, not clinical efficacy of a diet.


### 1.1 Research components


| ID | Workstream | Main question | Primary deliverable |
| --- | --- | --- | --- |
| C1 | Future FC prediction | Do historical microbiome trajectories (modeled via EN, RF, HGB, and MicroTrace-Attn) add value beyond previous FC and clinical history? | Participant-held-out future-FC predictions and ablation results |
| C2 | Taxonomic + functional signatures | Which single microbes/pathways, multi-feature guilds, and Cross-Omic Attention pairs associate with Current FC and Future FC across participant folds? | Dual Current/Future FC LME, guild, 5-fold stability, and Cross-Omic Attention report |
| C3 | Uncertainty + abstention | Can the system quantify uncertainty and identify unreliable cases? | Calibrated intervals, coverage/width metrics, risk–coverage analysis |
| C4 | Dietary-pattern decision support | Does microbiome information improve expert-rated suitability beyond clinical context and evidence? | Evidence-linked prototype and blinded expert evaluation |


## 2. Product and research requirements


### 2.1 Intended users and use context

- Research users: student researchers, supervisors, data scientists, bioinformaticians, and clinical collaborators evaluating the model and evidence base.
- Potential future users: IBD clinicians and registered dietitians reviewing decision-support outputs. The prototype is not a patient-facing prescribing tool.
- The product output is a research-grade prediction and evidence summary, not a diagnosis, treatment instruction, or substitute for clinician judgement.

### 2.2 Goals

- Estimate future FC from data available at a clearly defined index time.
- Quantify the incremental predictive value of clinical, taxonomic, and functional information.
- Identify reproducible single-feature, multi-feature microbial guild, and Cross-Omic Attention ([Bacteria] <-> [Pathways] <-> [FC]) signatures for both Current FC (concurrent inflammation) and Future FC (next-visit flare risk) while controlling for repeated measurements, confounding, and multiple testing.
- Quantify prediction uncertainty and support abstention when reliability is inadequate.
- Compare guideline-only, clinical-context, and microbiome-informed dietary-pattern decision support.
- Produce a reproducible, auditable implementation with explicit data and model limitations.

### 2.3 Out of scope

- Clinical deployment or autonomous treatment decisions.
- Claims that model-selected microbes are causal therapeutic targets.
- Claims that a recommended dietary pattern will reduce FC for a particular patient.
- Individual-food ranking, meal planning, calorie prescriptions, or automated nutrient prescriptions.
- Unconstrained deep learning without linear/clinical residual grounding or participant-grouped validation when the number of independent participants is modest (MicroTrace-Attn uses a strict low-rank residual attention bottleneck anchored on linear clinical/lag predictors).
- External validation claims for features that are not measured comparably in the external cohort.

### 2.4 Product requirements


| Requirement ID | Requirement | Acceptance evidence |
| --- | --- | --- |
| PR-01 | All predictions must have an explicit index date and future outcome date. | Audit table demonstrates feature timestamp ≤ index date < future outcome date. |
| PR-02 | All visits from one participant must stay within one data partition. | Split audit shows zero participant overlap between train/calibration/test. |
| PR-03 | All preprocessing must be fitted on training data only. | Saved preprocessing pipeline and test proving no fit on held-out data. |
| PR-04 | The C1 model must be compared against previous-FC and clinical baselines. | Baseline/ablation results table. |
| PR-05 | C2 must report feature stability and multiple-testing control. | Feature table with effect, confidence interval, FDR, and stability. |
| PR-06 | C3 must report coverage and interval width, not just point error. | Calibration and subgroup reliability report. |
| PR-07 | C4 outputs must link recommendations to evidence and limitations. | Evidence record and traceable rationale for every pattern shown. |
| PR-08 | C4 must have safety filters and abstention. | Test cases for restrictions, missing context, and insufficient evidence. |
| PR-09 | External validation must not be used for tuning unless explicitly labelled recalibration. | Frozen pipeline evaluation report. |
| PR-10 | All exclusions and protocol deviations must be documented. | Data-quality report and decision log. |


## 3. Dataset strategy and feasibility


Phase P0 feasibility auditing on the complete IBDMDB / HMP2 release (46 raw files, 1,453.02 MB, 5,533 metadata rows, 131 unique participants: 66 CD, 38 UC, 27 nonIBD) has been completed and verified (11/11 automated unit and leakage tests passing). All internal feasibility gates (G1–G6, G8) passed: 1,595 deduplicated shotgun metagenomic (MGX) physical visits are 100% matched across MetaPhlAn2/v3 and HUMAnN2/v3; 636 unique visits have harmonized Fecal Calprotectin (FC); 503 MGX visits (95 subjects) have exact same-visit FC (0d); 686 MGX visits (96 subjects) have causal prior/same-day FC (<=14d); and 93 participants (69 IBD, 24 nonIBD) contribute 445 consecutive next-FC forecast pairs (median gap 56.0 days, IQR 28–70 days) and 1,671 total causal future pairs (545 pairs in 30–90 days across 88 participants).


| Dataset | Role | Potentially useful data | Critical caveat / gate |
| --- | --- | --- | --- |
| IBDMDB / HMP2 | Primary candidate for C1–C3 | Verified: 131 participants (66 CD, 38 UC, 27 nonIBD), 1,595 deduplicated MGX visits, 636 FC visits, 100% matched MetaPhlAn2/3 & HUMAnN2/3. | PASSED P0 Audit: 503 same-day MGX+FC visits (95 subjects), 686 causal <=14d visits (96 subjects), 445 consecutive next-FC pairs (93 subjects), 545 30-90d pairs (88 subjects). |
| IBSEN III | External validation candidate | Longitudinal 16S microbiome and clinical/FC data. | Access may require a request; 16S does not directly validate shotgun functional pathways. |
| DINE-CD | C4 evidence / diet-outcome context | Randomized SCD vs Mediterranean-style diet study in mild-to-moderate Crohn’s disease; trial outcomes include FC. | Restricted data-use conditions may apply. Do not assume microbiome sequencing is included or that the sample can support personalized treatment-effect ML. |
| AGA / ECCO / BDA guidance | C4 evidence base | Clinical dietary and nutrition recommendations, cautions, and context. | Guidance informs evidence records; it does not provide individualized supervised labels. |
| USDA FoodData Central / ICMR-NIN | Optional composition reference | Food composition and nutrient information. | Nutrient composition is not proof of IBD clinical suitability; C4 is patterns-only. |


### 3.1 Feasibility gates


| Gate | Evidence required | If not satisfied |
| --- | --- | --- |
| G1 — Linkage | Stable participant, visit, and sample IDs across tables. | PASS — site_sub_coll uniquely links 2,302 visits; 43 _TR/_P replicates deduplicated (1,638 MGX runs -> 1,595 visits). |
| G2 — Time | Reliable visit/collection dates or defensible visit ordering. | PASS — 100% of 1,769 stool events have valid date_of_receipt; 0 inversions with week_num. |
| G3 — FC | Numeric FC, units, assay/metadata, and missingness profile. | PASS — fecalcal + fecalcal_ng_ml harmonized (636 stool FC visits across 98 subjects, range 3.45–493.7 ug/g). |
| G4 — Repeated observations | Sufficient participants with eligible index/future observations. | PASS — 90 subjects have >=2 MGX+FC visits; 93 subjects yield 445 consecutive pairs and 545 pairs in 30–90d. |
| G5 — Matched microbiome | Stool/microbiome profile linked to the index clinical visit. | PASS — 503 exact same-visit (0d) MGX+FC matches (95 subjects); 686 causal <=14d matches (96 subjects). |
| G6 — Functional data | Shotgun-derived functional pathway/gene-family profiles and annotation version. | PASS — 100% of 1,638 MGX samples have HUMAnN2 (10,884 pathways) & HUMAnN3 (22,113 pathways; 478 unstratified). |
| G7 — External compatibility | Comparable outcome, feature definitions, and forecast horizon. | PENDING P5 — External IBSEN III is 16S-only; protocol restricts external transfer to compatible genus features. |
| G8 — Permissions | Access approvals and research-use terms. | PASS — Open-access HMP2 release terms verified and documented in access_requirements.md. |


### 3.2 Required data audit outputs


| Artifact | Minimum contents |
| --- | --- |
| participant_summary.csv | Unique participants, visits per participant, disease subtype, inclusion flags. |
| sample_matching.csv | Clinical–FC–microbiome links, sample IDs, dates, matching gaps. |
| fc_summary.csv | Units, assay metadata, distribution, detection limits if present, missingness, outliers. |
| visit_gap_summary.csv | Time between visits and distributions of eligible forecast horizons. |
| longitudinal_pairs.csv | All eligible index-to-future pairs, participant IDs, horizon days, feature timestamps. |
| feature_availability.csv | Availability by feature group: clinical, taxonomy, function, diet, medication. |
| split_audit.csv | Participant assignments and proof of no participant overlap. |
| data_quality_report.md | Cleaning choices, exclusions, unresolved problems, and gate outcomes. |


## 4. Shared experimental protocol


### 4.1 Outcome and forecast estimand


Define transformed FC as y(i,t) = log10(FC(i,t) + 1), where FC is harmonized from fecalcal and fecalcal_ng_ml via combine_first (verified identical to <1e-4 on shared collection events, range 3.45–493.7 ug/g). The primary future prediction estimand is consecutive next-visit y(i,t+h) (445 pairs across 93 participants, with elapsed horizon Δt_horizon included as an explicit feature), supplemented by co-primary bounded horizon windows of 30–90 days (545 pairs, 88 participants) and 14–56 days (415 pairs, 84 participants).


> **Forecasting rule**
> Every row must distinguish the index time from the future outcome time. Features measured or recorded after the index time are excluded, even if they are otherwise clinically useful.


### 4.2 Cohort eligibility

- Confirmed Crohn’s disease or UC as defined by the source cohort.
- At least one eligible index microbiome profile for microbiome model arms.
- FC associated with the future outcome visit, with interpretable units.
- Valid dates or defensible visit order.
- Clinical variables needed by the chosen model recorded by the index time.
- Include explicit missingness flags and report how many participants/visits are removed at each stage.

### 4.3 Matching windows


Based on the Phase P0 matching sensitivity audit across 1,595 deduplicated MGX physical collection events (site_sub_coll), the primary index matching rule is exact same-collection matching (0-day gap: 503 MGX+FC visits across 95 participants). A causal prior/same-day matching window of <=14 days (where index FC was measured at or up to 14 days before the MGX stool sample: 686 visits across 96 participants) is pre-specified as the primary sensitivity cohort. Post-index FC values are strictly excluded from index features.


### 4.4 Participant-level data partition


Verified participant-level partition: 80% development set (75 participants) and 20% untouched final test set (18 participants), stratified by diagnosis (CD/UC/nonIBD) with zero participant overlap. Inside the 80% development set, 5-fold participant-grouped cross-validation is used for hyperparameter tuning, stability selection (C2), and out-of-fold Cross-Conformal (CV+) residual calibration (C3). The final 20% test set is never used for feature selection, threshold selection, model tuning, or calibration.


> **Splitting caveat**
> If cohort size is small, the percentages may need to change. The invariant is participant separation and an untouched test set, not a particular percentage split.


### 4.5 Preprocessing and leakage controls

- Fit imputation, scaling, prevalence filters, zero handling, feature selection, dimensionality reduction, and hyperparameter selection using training data only.
- Apply frozen transformations to calibration and test partitions.
- No future FC, future medication, future diet, or post-index clinical values can be predictors.
- Store a participant-level split audit and a feature-availability timestamp audit.
- Keep original data immutable; write derived tables to interim/processed directories.
- Version every data release, annotation database, and preprocessing configuration.

### 4.6 Candidate models and shared baselines


| Model | Inputs | Purpose |
| --- | --- | --- |
| B0 | Training-set mean FC | Trivial reference. |
| B1 | Most recent FC | Persistence reference. |
| B2 / M1 | Clinical history + previous FC | Strong clinical baseline. |
| B3 | Taxonomy only | Microbiome-only reference. |
| B4 | Function only | Functional-only reference. |
| B5 / M2 | Clinical + taxonomy | Incremental taxonomic value. |
| B6 / M3 | Clinical + function | Incremental functional value. |
| B7 / M4 | Clinical + taxonomy + function | Combined clinical + taxonomy + function across EN, RF, HGB, and MicroTrace-Attn. |


*Evaluate four complementary model families across identical participant splits: (1) Elastic Net (L1+L2 regularized linear model), (2) Random Forest, (3) HistGradientBoosting (HGB), and (4) MicroTrace-Attn—a lightweight Continuous-Time Residual Cross-Attention architecture that attends over a patient’s full causal visit trajectory (visits 1..t) with time-decay embeddings φ(Δt) and a linear residual skip connection from (FC_t, Δt_horizon, clinical state) so it cannot overfit or underperform the linear persistence baseline.*


### 4.7 Common performance reporting

- Primary: MAE on log10(FC + 1), pre-specified before final testing.
- Secondary: RMSE, R-squared, Spearman correlation, and error by disease subtype, FC range, and horizon.
- Use participant-level bootstrap resampling for uncertainty in performance estimates where feasible; do not bootstrap individual visits as if independent.
- Report original-scale FC errors as a supplementary view, noting that inverse-transforming a log prediction is not automatically an unbiased estimate of the original-scale mean.

## 5. C1 — Future FC prediction


> **Research question**
> Can historical microbiome and clinical information improve prediction of future FC compared with previous FC and clinical history alone?


### 5.1 Hypotheses

- H1: Clinical history plus previous FC predicts future FC better than a trivial or persistence-only reference.
- H2: Adding taxonomy improves prediction over the clinical baseline.
- H3: Adding functional profiles provides additional value beyond taxonomy, if available.
- H4: Performance differs by disease subtype and forecast horizon.

### 5.2 Prediction-pair schema


| Field | Definition |
| --- | --- |
| participant_id | Stable participant identifier. |
| index_date / index_visit_id | Date and visit used to generate the prediction. |
| index_fc | Most recent FC available on or before the index date. |
| index_microbiome_id | Microbiome sample associated with the index date under pre-defined matching rules. |
| future_date / future_fc | Future outcome measurement. |
| horizon_days | Future date minus index date. |
| disease_subtype | Crohn’s disease or UC. |
| feature_snapshot | Explicit list/version of values available at index date. |


### 5.3 Model arms / ablation


| Arm | Inputs |
| --- | --- |
| M0 | Most recent FC only. |
| M1 | Clinical history + medications + previous FC (trained with Elastic Net, RF, HGB, and MicroTrace-Attn). |
| M2 | M1 + taxonomic features (CLR species/genera, alpha/beta diversity, SCFA/pathobiont guilds). |
| M3 | M1 + shotgun functional features (CLR HUMAnN3 pathways + functional metabolic guilds). |
| M4 | M1 + taxonomy + shotgun function (combined multi-omic benchmark across EN, RF, HGB, MicroTrace-Attn). |
| M5 | M4 + full causal trajectory state (x_t, x_{t-1}, Δx, log-ratio change, velocity Δx/Δt, rolling history <=t) + MicroTrace-Attn over visits 1..t. |


### 5.4 Temporal features


Because an identical raw delta (e.g., ΔFC = +20 from 100->120 vs 180->200) has different clinical meaning depending on baseline severity and elapsed time, M5 represents each longitudinal feature using its full causal trajectory state vector rather than Δ alone: (1) Current value x(i,t), (2) Previous value x(i,t−1), (3) Raw change Δx = x(i,t) − x(i,t−1), (4) Relative log-ratio change log((x(i,t)+1)/(x(i,t−1)+1)), (5) Rate of change (velocity) Δx / Δt_prev, and (6) Rolling participant baseline summaries (mean, SD, min, max, linear slope across all visits <= t), alongside Bray–Curtis dissimilarity and alpha-diversity shifts. Furthermore, MicroTrace-Attn ingests the entire sequence of prior visits (1..t) with continuous time-gap encoding.


### 5.5 Sensitivity analyses

- Exclude or separately analyze participants with major medication changes between index and future outcome.
- Compare strict and relaxed microbiome–visit matching windows.
- Compare short and longer prediction horizons supported by the data.
- Fit subtype-specific models only if participant counts are adequate.
- Quantify how much of model performance is explained by previous FC alone.

### 5.6 C1 acceptance criteria

- Eligible prediction pairs and participant counts are reported transparently.
- All model arms use identical participant splits and comparable evaluation subsets.
- M2/M3/M4 are compared against M1 on untouched test participants.
- Performance differences include confidence intervals or another justified uncertainty estimate.
- No claim of added microbiome value is made unless improvement is reproducible and not merely a single split artifact.

## 6. C2 — Reproducible taxonomic and functional signatures


> Research question | Which individual microbial taxa/pathways, multi-feature biological guilds, and Cross-Omic Attention ([Bacteria] <-> [Pathways] <-> [FC]) teamwork pairs are reproducibly associated with Current FC (same-visit inflammation) and Future FC (next-visit transition) across participant-grouped folds?


### 6.1 Taxonomic scope


Analyze species and genus representations (MetaPhlAn3 primary: 578 species, 187 genera; MetaPhlAn2 sensitivity: 1,479 clades) plus alpha diversity (Shannon, Simpson, richness), beta-diversity trajectories (Bray–Curtis), and domain-informed Multi-Feature Microbial Guilds: (1) SCFA / Butyrate Producer Guild (Faecalibacterium prausnitzii, Roseburia spp., Eubacterium rectale, Subdoligranulum, Anaerostipes), (2) Pathobiont / Proteobacteria Bloom Guild (Escherichia coli, Klebsiella, Veillonella, Ruminococcus gnavus, Enterococcus), and (3) Protective-to-Pathobiont Log-Ratio Index.


### 6.2 Functional scope


Use shotgun-metagenomic functional pathway and EC profiles (HUMAnN3 primary: 478 unstratified MetaCyc pathways; HUMAnN2 sensitivity), analyzing both individual pathways and Multi-Feature Functional Guilds (butyrate/propionate fermentation, sulfur/H2S metabolism, oxidative stress response, mucin degradation, and lipopolysaccharide biosynthesis). Metatranscriptomic (MTX) RNA pathways (833 samples) are analyzed separately as a secondary expression sensitivity check.


> **Functional interpretation**
> Metagenomic pathway abundance indicates functional potential, not necessarily active expression. Metatranscriptomics is a different measurement and should be analyzed separately.


### 6.3 Compositional preprocessing

- Inspect count tables, library sizes, prevalence, and sequencing quality.
- Define low-prevalence filtering on training data and document the threshold.
- Specify zero handling before log-ratio transformation and run sensitivity checks.
- Use CLR where appropriate: CLR(x_j) = log(x_j / geometric_mean(x)).
- Retain raw counts, relative abundances, and transformed profiles as distinct representations.
- Do not interpret relative-abundance changes as absolute microbial growth.

### 6.4 Three analysis experiments


| Experiment | Method | Output |
| --- | --- | --- |
| A — Adjusted association | Repeated-measures Linear Mixed-Effects (LME with participant random intercept (1\|Participant_ID)) for BOTH Current log10(FC+1) (503 visits) and Future log10(FC+1) (445 pairs), testing both single features and multi-feature biological guilds (SCFA/butyrate guild, pathobiont guild, dysbiosis ratio, functional guilds), adjusted for previous FC (in future models), diagnosis subtype, medications, antibiotics, age/sex, and elapsed time. | Effect size, interval, p-value, FDR, direction. |
| B — Predictive contribution | Multi-feature predictive sets (Elastic Net L1+L2, Random Forest, HistGradientBoosting) PLUS Cross-Omic Attention ([Every Bacterium] <-> [Every Pathway] <-> [FC]) trained across 5 participant-grouped folds for both Current FC and Future FC. | Multi-feature importance, SHAP, and Cross-Omic Attention bacterium–pathway teamwork weights. |
| C — Stability selection | Repeat LME, multi-feature ML selection, and Cross-Omic Attention across 5 participant-grouped folds (and 100 participant-cluster bootstraps); require >=80% fold stability (>=4/5 folds) and calculate Jaccard similarity. | Stable versus split-sensitive features. |


All C2 analyses are performed on BOTH (1) Current FC (concurrent same-visit log10(FC+1) across 503 visits / 95 participants, capturing active biological coupling) and (2) Future FC (next-visit log10(FC+1) across 445 pairs / 93 participants, capturing early-warning lead/lag signals). Stability(feature j) = folds selecting j / 5 participant-grouped folds (requiring >=80% selection, i.e., >=4/5 folds, plus bootstrap CI excluding 0). Apply Benjamini–Hochberg FDR correction across pre-specified hypothesis families, and train a Cross-Omic Attention network ([Every Bacterium] <-> [Every Pathway] <-> [FC]) across all 5 participant folds to extract reproducible multi-omic teamwork pairs.


### 6.5 C2 acceptance criteria

- Report sample/participant counts and missingness per feature group.
- Repeated measurements and key confounders are handled explicitly.
- Report effect size, confidence interval, FDR, stability frequency, and direction for reported signals.
- Distinguish association from prediction and both from causal interpretation.
- External replication is claimed only for comparable feature types and assays.

## 7. C3 — Uncertainty-aware prediction and abstention


> **Research question**
> Can the system quantify the uncertainty of a future-FC prediction and identify cases where it should not issue a confident prediction?


### 7.1 Experimental systems


| System | Output |
| --- | --- |
| U0 | Standard regression point prediction. |
| U1 | Quantile regression interval. |
| U2 | 5-fold Participant-Grouped Cross-Conformal (CV+) interval with locally-weighted patient-specific widths. |
| U3 | Conformal interval plus selective abstention. |


### 7.2 Conformal procedure


To maximize calibration sample size while strictly respecting participant clustering, implement 5-fold Participant-Grouped Cross-Conformal Prediction (CV+) on the 75 development participants alongside Locally-Weighted Conformal residuals r_i = |y_i − ŷ_i| / σ̂(x_i), where σ̂(x_i) is an out-of-fold uncertainty scaler (ensemble spread / quantile width). This yields patient-specific adaptive interval widths [ŷ_i − q_α·σ̂(x_i), ŷ_i + q_α·σ̂(x_i)] at 90% nominal coverage, evaluated alongside cluster-aware participant-level quantiles.


> Repeated-measurement & locally-weighted conformal protocol | Because visits from the same participant are correlated and cohort size is 93 participants, C3 uses 5-fold Participant-Grouped Cross-Conformal (CV+) calibration on the 75 development participants with locally-weighted residuals r_i = |y_i − ŷ_i| / σ̂(x_i). This yields patient-specific interval widths while ensuring all visits of a participant remain in the same fold, evaluated alongside participant-cluster quantiles on the 18 untouched test participants.


### 7.3 Metrics


| Metric | Definition / interpretation |
| --- | --- |
| Coverage | Fraction of test outcomes contained within their prediction intervals. |
| Mean interval width | Average upper minus lower interval bound. |
| Risk–coverage | Prediction error among accepted examples as the accepted fraction changes. |
| Acceptance rate | Fraction of examples for which the model does not abstain. |
| Subgroup coverage | Coverage by subtype, FC range, forecast horizon, and other supported subgroups. |


### 7.4 Selective prediction


Define an acceptance rule using uncertainty, for example Accept_i = 1 if interval_width_i ≤ τ, otherwise 0. Choose τ using validation data only. An abstention should state that uncertainty is too high or the input is outside the validated feature domain; it must not be translated into a clinical diagnosis.


### 7.5 C3 acceptance criteria

- Report nominal versus observed coverage and average interval width.
- Report subgroup coverage; overall coverage alone is insufficient.
- Show risk–coverage curves and acceptance rates.
- Document exchangeability assumptions and the participant-aware calibration method.
- Do not claim clinical safety from statistical calibration alone.

## 8. C4 — Evidence-constrained dietary-pattern decision support


> **Research question**
> Does microbiome information improve the relevance and evidence consistency of dietary-pattern decision support beyond clinical information and established dietary guidance?


### 8.1 Three systems to compare


| System | Information |
| --- | --- |
| R1 — Evidence only | Clinical dietary guidance and curated evidence records. |
| R2 — Evidence + clinical context | R1 plus subtype, disease context, medications, restrictions, and preferences. |
| R3 — Evidence + clinical + microbiome | R2 plus microbiome information only where independent evidence justifies its use. |


The primary comparison is R3 versus R2. If the microbiome does not improve expert-rated suitability, report that result rather than forcing a microbiome-specific recommendation.


### 8.2 Dietary-pattern scope


Limit the prototype to a small, explicitly supported set such as Mediterranean-style and Specific Carbohydrate Diet patterns, plus other patterns only when their evidence and context are sufficiently defined. This project does not generate individual-food rankings, meal plans, calorie targets, or nutrient prescriptions.


### 8.3 Evidence-record schema


| Field | Purpose |
| --- | --- |
| diet_pattern_id | Stable identifier for the pattern. |
| population / clinical_context | Population and context to which evidence applies. |
| study_design | Guideline, randomized trial, cohort, or other design. |
| outcome | FC, symptoms, quality of life, nutritional status, etc. |
| evidence_direction / strength | Structured summary based on documented appraisal. |
| contraindications / nutritional_risks | Known restrictions and potential harms. |
| source_reference | DOI, guideline URL, or stable source identifier. |
| limitations | Population, duration, adherence, and generalizability constraints. |


### 8.4 Recommendation workflow

1. Check whether the clinical profile is sufficiently complete.
2. Apply allergy, intolerance, restriction, and nutritional-safety filters.
3. Retrieve evidence applicable to the patient’s disease subtype and clinical context.
4. Use microbiome features only when an independently supported evidence link exists.
5. Rank eligible patterns or abstain; show rationale, evidence strength, uncertainty, and limitations.
6. Refer restrictive or clinically consequential changes to a qualified IBD dietitian/clinician.

### 8.5 Expert evaluation


Create 30–50 standardized scenarios as an initial feasibility pilot. The final scenario count must be justified by scenario diversity, reviewer availability, and the planned statistical analysis. Scenarios should vary disease subtype, disease activity, medication, nutritional restrictions, preferences, microbiome profile, and completeness of information.


Have qualified IBD dietitians or clinicians independently evaluate anonymized R1/R2/R3 outputs. Randomize presentation order and hide system identity where practical. Rate clinical appropriateness, evidence traceability, nutritional safety, preference compatibility, and abstention appropriateness.


Primary comparison: ΔS = mean suitability(R3) − mean suitability(R2). Also report safety-violation rate, traceability, preference compatibility, abstention appropriateness, and inter-rater agreement. For two raters consider weighted Cohen’s kappa for ordinal ratings; for more raters use an appropriate multi-rater statistic.


### 8.6 C4 acceptance criteria

- Every displayed pattern is linked to an evidence record and source.
- Missing or unsafe context triggers filtering or abstention rather than a confident recommendation.
- R1, R2, and R3 are compared using the same standardized scenarios.
- Expert evaluation is blinded/randomized where practical and inter-rater agreement is reported.
- No claim of improved FC or remission is made from expert suitability scores.

## 9. Internal and external validation


### 9.1 Internal validation

- Assign participants—not visits—to train, calibration, and final test partitions.
- Perform model selection and feature selection only inside training data.
- Use identical splits for ablation comparisons where possible.
- Report participant and visit counts per partition and per disease subtype.
- Use participant-level bootstrap or another cluster-aware method for confidence intervals.

### 9.2 External validation


IBSEN III is a candidate external cohort subject to data-access approval and compatibility review. Its reported 16S profiling can support compatible taxonomic validation; it cannot directly reproduce shotgun functional profiles. Freeze the primary model and preprocessing pipeline before external evaluation. Report external performance before any recalibration. If recalibration is conducted, report it separately and explicitly identify which external data were used.


### 9.3 Domain shift reporting

- Compare cohort demographics, disease subtype, assay, sample handling, sequencing technology, taxonomy resolution, FC units, and visit schedule.
- Report missing/incompatible features and any feature harmonization performed.
- Do not silently impute an entire unavailable modality or claim full-model external validation when only a subset transfers.

## 10. Implementation architecture and repository


Recommended implementation: Python, pandas/NumPy, scikit-learn, and appropriate microbiome/compositional-data tooling. Use notebooks for exploration and scripts/modules for repeatable processing and evaluation. Keep raw data immutable.


| Directory | Contents |
| --- | --- |
| data/raw/ | Original downloaded/approved data; read-only. |
| data/interim/ | Matched samples, audit tables, intermediate representations. |
| data/processed/ | Model-ready, versioned feature tables. |
| notebooks/ | 01 inventory; 02 quality; 03 longitudinal feasibility; 04 exploratory analysis. |
| src/data/ | Import, joins, validation, longitudinal pair creation. |
| src/preprocessing/ | Training-fitted transforms and microbiome processing. |
| src/features/ | Temporal, taxonomic, and functional features. |
| src/models/ | Baselines, model training, tuning. |
| src/evaluation/ | Metrics, bootstrap, subgroup, external validation. |
| src/uncertainty/ | Quantile/conformal and abstention. |
| src/recommendations/ | Evidence records, safety filters, R1/R2/R3. |
| configs/ | Versioned experiment settings. |
| tests/ | Unit and leakage tests. |
| results/ | Tables, figures, serialized models, logs. |
| reports/ | Protocol, data audit, manuscript outputs. |


### 10.1 Engineering requirements

- Use a fixed random seed and record package versions.
- Use a configuration file for target, horizon, matching windows, splits, model settings, and calibration.
- Add tests for participant leakage, temporal leakage, duplicate matching, and train-only fitting.
- Log dataset version, source table versions, preprocessing version, and model parameters per run.
- Save metrics and predictions at participant/forecast-example level for audit.
- Keep a data dictionary and a reproducible run command for each experiment.

## 11. Implementation roadmap and milestones


| Phase | Work | Exit deliverable / gate |
| --- | --- | --- |
| P0 — Feasibility | Inventory data, access, matching, repeated FC, taxonomy/function availability. | COMPLETED (10 Nov 2026) — All G1–G6 & G8 gates passed; 16 CSV/JSON artifacts, 9 reports, 8 figures, and 11/11 tests verified. |
| P1 — C1 | Construct index/future pairs; train M0–M5 across Elastic Net, RF, HGB, and MicroTrace-Attn (visits 1..t + full trajectory state); run horizon/matching sensitivities. | Prediction table and future-FC performance report. |
| P2 — C2 | Taxonomic + functional CLR preprocessing; Current FC (503 visits) & Future FC (445 pairs) LME, multi-feature guilds, 5-fold stability, and Cross-Omic Attention. | Feature signature and reproducibility report. |
| P3 — C3 | Quantile regression & 5-fold participant-grouped Cross-Conformal (CV+) locally-weighted intervals; subgroup coverage; selective abstention. | Coverage, width, subgroup, and risk–coverage report. |
| P4 — C4 | Evidence database, R1/R2/R3 prototype, scenario construction, expert ratings. | Decision-support prototype and evaluation report. |
| P5 — External + paper | Request/inspect IBSEN III; freeze pipeline; validate compatible subsets; manuscript. | External validation report, paper, code, and reproducibility package. |


*Planning estimate only: P0 approximately 2–4 weeks; P1 3–4 weeks; P2 3–4 weeks; P3 2–3 weeks; P4 4–6 weeks; P5 3–5 weeks. Access approvals and data availability can extend these ranges. Start external access requests early.*


## 12. Risks, controls, and decision rules


| Risk | Control / response |
| --- | --- |
| Too few matched future-FC pairs | Run feasibility audit first; adjust horizon only with scientific justification; narrow claims if needed. |
| Functional profiles unavailable | Keep C2 taxonomy-only or identify a compatible functional cohort; do not infer function from 16S data. |
| Previous FC dominates signal | Report honestly; quantify incremental microbiome contribution rather than overclaiming. |
| Repeated-measure leakage | Group by participant and run automated overlap tests. |
| Compositional artifacts | Use documented zero handling and log-ratio methods; run sensitivity analyses. |
| Medication/antibiotic confounding | Pre-specify adjustment and stratified sensitivity analyses. |
| Poor conformal coverage under clustering | Use participant-aware calibration and state the supported guarantee; widen or abstain as needed. |
| External cohort mismatch | Validate only harmonized features; report domain shift and avoid unsupported transfer claims. |
| Unsafe dietary suggestions | Evidence traceability, safety filters, expert review, and abstention. |
| Overclaiming novelty | Compare against current literature; position contribution as integrated, rigorously evaluated capabilities. |


## 13. Definition of done and go/no-go checklist


| Area | Done when |
| --- | --- |
| Data feasibility | All gate decisions have evidence; inclusion flow and usable participant/pair counts are reproducible. |
| C1 | Pre-registered forecast horizon/target; baselines/ablations; untouched test results; sensitivity analyses. |
| C2 | Taxonomic and functional data availability established; repeated-measure-aware analysis; FDR and stability reported. |
| C3 | Calibration assumptions documented; coverage/width/subgroup results; abstention analysis completed. |
| C4 | Evidence records complete; R1/R2/R3 tested; expert protocol and safety evaluation reported. |
| External | Data access and compatibility confirmed; frozen model evaluated; limitations reported. |
| Reproducibility | Code, configs, environment/version record, data dictionary, tables, figures, and run instructions available. |


> **Go/no-go rule for the full programme**
> Proceed with all four contributions only if the source data support longitudinal matching and sufficient independent participants, functional profiles are truly available for C2, and clinical evidence can support safe pattern-level guidance for C4. Otherwise, narrow the scope and document the reason rather than fabricating completeness.


## 14. Starting references and data portals


*These are starting points carried forward from the research design. Verify the current version, data access terms, exact variables, and suitability before citing or using them in a submitted manuscript.*


IBDMDB portal — https://ibdmdb.org/


IBDMDB / HMP2 dbGaP study record — https://www.ncbi.nlm.nih.gov/projects/gap/cgi-bin/study.cgi?study_id=phs001626.v1.p1


IBDMDB protocol repository — https://github.com/ibdmdc/ibdmdb-protocols


IBSEN III longitudinal microbiome and FC study — https://pmc.ncbi.nlm.nih.gov/articles/PMC13533192/


DINE-CD ICPSR study record — https://www.icpsr.umich.edu/web/pcodr/studies/38590


AGA dietary and nutritional therapies in IBD — https://gastro.org/clinical-guidance/diet-and-nutritional-therapies-in-patients-with-ibd/


ECCO dietary guidance / consensus — https://academic.oup.com/ecco-jcc/article/19/9/jjaf122/8198055


BDA consensus on dietary management in IBD — https://pmc.ncbi.nlm.nih.gov/articles/PMC10084145/


USDA FoodData Central — https://fdc.nal.usda.gov/


ICMR–National Institute of Nutrition — https://www.nin.res.in/


Clinical calibration tutorial — https://pmc.ncbi.nlm.nih.gov/articles/PMC7075534/


Conformal prediction in clinical medical sciences — https://pmc.ncbi.nlm.nih.gov/articles/PMC9309105/


MAPIE conformal regression documentation — https://github.com/scikit-learn-contrib/MAPIE/blob/master/doc/content/conformal-prediction/regression.md


## Appendix A. Mandatory inspection questions

1. How many unique participants are present in each source table?
2. How many participants have at least two eligible visits with matched microbiome and FC?
3. How many valid index-to-future pairs exist within the chosen horizon?
4. How many matched samples have both taxonomy and functional profiles?
5. How many participants remain after all inclusion and quality criteria?
6. Are FC units and assay metadata consistent across visits and cohorts?
7. Do timestamps support a genuine forecast rather than same-visit prediction?
8. Can all participants be assigned to disjoint train/calibration/test partitions?

## Appendix B. Minimum manuscript figure/table set


| Type | Content |
| --- | --- |
| Figure 1 | MICRO-TRACE research architecture. |
| Figure 2 | Cohort and participant/sample flow diagram. |
| Figure 3 | Future-FC prediction performance and ablation. |
| Figure 4 | Taxonomic and functional association signatures. |
| Figure 5 | Feature stability across participant folds. |
| Figure 6 | Prediction interval coverage and width. |
| Figure 7 | Risk–coverage curve for selective abstention. |
| Figure 8 | C4 recommendation evaluation. |
| Table 1 | Participant demographics and clinical characteristics. |
| Table 2 | Data availability, matching, and missingness. |
| Table 3 | Model comparisons and ablations. |
| Table 4 | Microbial association and stability results. |
| Table 5 | Uncertainty and subgroup reliability metrics. |
| Table 6 | Expert dietary-pattern evaluation. |
