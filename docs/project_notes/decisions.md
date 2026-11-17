# Architectural Decisions (ADRs)

This file logs all architectural and experimental protocol decisions (`ADR-001` through `ADR-010`, aligned with `D-001`–`D-010` in `MICRO_TRACE_Project_Tracker.xlsx`) for the MICRO-TRACE project.

---

### ADR-001: C1 Future FC Target & Four Model Families including `MicroTrace-Attn` (2026-11-10)

**Context:**
- Same-visit FC prediction is clinically less actionable than early-warning prediction of the next visit's inflammation level.
- Tabular models (`Elastic Net`, `Random Forest`, `HistGradientBoosting`) handle high-dimensional microbiome features well on 93 participants, but standard tabular models only see fixed lag columns rather than the patient's variable-length visit trajectory $1 \dots t$.

**Decision:**
- Predict future transformed Fecal Calprotectin $y_{i, t+h} = \log_{10}(\text{FC}_{i, t+h} + 1)$ using strictly causal features ($t_{\text{feature}} \le t_{\text{index}} < t_{\text{future}}$).
- Benchmark four model families across identical participant splits (`M0`–`M5`):
  1. `Elastic Net` (L1+L2 regularized linear model)
  2. `Random Forest`
  3. `HistGradientBoosting` (`HGB`)
  4. `MicroTrace-Attn` — Continuous-Time Residual Cross-Attention over full patient history $1 \dots t$ with time-decay encoding $\phi(\Delta t)$ and a linear residual skip connection anchored on $(\text{FC}_t, \Delta t_{\text{horizon}}, \text{clinical state})$.

**Alternatives Considered:**
- Same-visit FC only for C1 -> Rejected: does not test genuine forecasting utility.
- Unconstrained deep Transformer without residual anchor -> Rejected: high overfitting risk on 93 independent participants.

**Consequences:**
- ✅ Tests both classic regularizers/trees and continuous-time trajectory attention on identical participant folds.
- ✅ Residual skip connection guarantees `MicroTrace-Attn` cannot collapse below the linear persistence baseline.

---

### ADR-002: C2 Dual Current/Future FC Scope, Multi-Feature Guilds, and Cross-Omic Attention (2026-11-10)

**Context:**
- Testing features one-by-one against Future FC (`445` pairs across variable $14\text{–}90\text{d}$ gaps) misses two crucial biological realities:
  1. Many bacteria/pathways drive **active same-day inflammation (`Current FC`, `503` visits)** even if their signal decays over a 56-day gap, while others act as **early-warning indicators (`Future FC`, `445` pairs)**.
  2. Gut inflammation is rarely driven by a single isolated bacterium—microbes act in **functional guilds/teams** and interact with metabolic pathways.

**Decision:**
- Run all C2 signature discovery analyses on **BOTH**:
  1. **Current FC (`503` same-day visits across `95` participants)** — active concurrent biological coupling.
  2. **Future FC (`445` next-visit pairs across `93` participants)** — early-warning transition signal (adjusted for current FC).
- Evaluate **BOTH single features AND multi-feature sets**:
  - **Domain-informed Microbial & Functional Guilds**: SCFA/butyrate producer guild (*Faecalibacterium prausnitzii*, *Roseburia* spp., *Eubacterium rectale*, *Subdoligranulum*, *Anaerostipes*), Pathobiont/Proteobacteria bloom guild (*Escherichia coli*, *Klebsiella*, *Veillonella*, *Ruminococcus gnavus*, *Enterococcus*), Protective-to-Pathobiont log-ratio index, and HUMAnN3 metabolic pathway guilds (SCFA fermentation, sulfur/$\text{H}_2\text{S}$, oxidative stress, mucin degradation, LPS biosynthesis).
  - **Joint Multi-Feature ML Panels**: `Elastic Net`, `Random Forest`, and `HGB` across 5 participant-grouped folds.
  - **Cross-Omic Attention (`[Every Bacterium] <-> [Every Pathway] <-> [FC]`)**: Bipartite cross-attention bottleneck trained across all 5 participant folds to extract reproducible bacterium-pathway teamwork weights ($\ge 4/5$ folds).

**Alternatives Considered:**
- Single feature-by-feature LME on Future FC only -> Rejected: ignores microbial teamwork and conflates biological coupling with 56-day temporal decay.

**Consequences:**
- ✅ Distinguishes concurrent inflammation drivers (`Current FC`) from early-warning flare predictors (`Future FC`).
- ✅ Captures both additive biological guilds and learned bacterium-pathway interactions (`Cross-Omic Attention`).

---

### ADR-003: C4 Dietary-Pattern Scope & Safety Guardrails (2026-11-08)

**Context:**
- Clinical IBD dietary evidence (AGA, ECCO, BDA, DINE-CD trial) supports whole dietary patterns (e.g., Mediterranean diet, Specific Carbohydrate Diet) rather than individualized single-food or calorie prescriptions from observational microbiome data.

**Decision:**
- Restrict C4 (`R1` guideline-only, `R2` + clinical context, `R3` + microbiome links) strictly to evidence-graded dietary patterns with mandatory allergy, stricture/malnutrition safety filters, C3 uncertainty abstention pass-through, and source traceability.

**Alternatives Considered:**
- Individual food ranking or meal-plan generation -> Rejected: unsupported by causal clinical evidence and unsafe without dietitian supervision.

**Consequences:**
- ✅ 100% traceable to peer-reviewed clinical guidelines and randomized trials.
- ✅ Prevents hallucinated or unsafe nutrition advice.

---

### ADR-004: Participant-Grouped Validation & External Cohort Guardrails (2026-11-08)

**Context:**
- Repeated longitudinal visits from the same participant are strongly correlated; random visit-level splits leak participant baseline identity across train and test sets.
- Candidate external cohort (`IBSEN III`) uses 16S amplicon sequencing rather than shotgun metagenomics.

**Decision:**
- Enforce strict participant-level partitioning (zero participant overlap across train/validation/test) verified by automated unit tests (`tests/test_leakage.py`).
- Restrict external 16S validation (`Phase P5`) to harmonized genus-level taxonomic features; never claim external validation of shotgun functional pathways from 16S data.

**Alternatives Considered:**
- Visit-level random split or inferring functional pathways from 16S via PICRUSt -> Rejected: causes severe data leakage and unverified functional claims.

**Consequences:**
- ✅ Zero participant leakage (`split_audit.csv`).
- ✅ Scientifically defensible external validation claims.

---

### ADR-005: Forecast Horizon Specification (2026-11-10)

**Context:**
- Phase P0 cadence audit showed all-stool median gap is `14.0d`, while consecutive FC-measured visits have median gap `56.0d` (`IQR 28–70d`), yielding `445` consecutive pairs (`93` subjects) and `1,671` total causal future pairs (`545` in `30–90d` across `88` subjects; `415` in `14–56d` across `84` subjects).

**Decision:**
- Use **consecutive next measured FC (`445` pairs across `93` subjects)** with explicit elapsed horizon $\Delta t_{\text{horizon}}$ as the primary C1 formulation, alongside co-primary bounded horizon windows of **`30–90` days (`545` pairs, `88` subjects)** and **`14–56` days (`415` pairs, `84` subjects)**, plus short-term **`14–35` days (`190` pairs, `44` subjects)** sensitivity.

**Alternatives Considered:**
- Restricting solely to a narrow `7–21d` window (`113` pairs, `36` subjects) -> Rejected: discards 60% of participants due to HMP2's ~4–8 week FC assay cadence.

**Consequences:**
- ✅ Preserves all `93` longitudinal participants while explicitly testing horizon-window sensitivity.

---

### ADR-006: Modality Matching Window (`0d` Primary + `<=14d` Causal Sensitivity) (2026-11-10)

**Context:**
- Out of `1,595` deduplicated MGX stool visits, `503` visits (`95` subjects) have FC measured on the exact same `site_sub_coll` collection (`0d`), and `686` visits (`96` subjects) have a causal prior/same-day FC within `<=14` days before the MGX sample.

**Decision:**
- Lock **exact same-collection matching (`0d`, `503` visits across `95` subjects)** as the primary matching rule, and **causal prior/same-day matching (`<=14d`, `686` visits across `96` subjects)** as the primary sensitivity cohort.
- Strictly forbid symmetric (`+-14d`) post-index matching for C1 index features.

**Alternatives Considered:**
- Symmetric `+-14d` window (`860` visits) -> Rejected for C1 index features because using an FC measured after $t_{\text{index}}$ violates strict temporal causality.

**Consequences:**
- ✅ Zero post-index temporal leakage (`100%` of pairs satisfy $t_{\text{feature}} \le t_{\text{index}} < t_{\text{future}}$).

---

### ADR-007: 80/20 Participant Partition + 5-Fold Participant-Grouped Cross-Conformal (`CV+`) with Locally-Weighted Residuals (2026-11-10)

**Context:**
- With `93` longitudinal participants (`69` IBD, `24` nonIBD), a static `70/15/15` split leaves only ~14 participants in the calibration set, which is noisy for 90% quantile estimation.
- Standard split-conformal uses a constant interval width $\pm q$ for every patient, which cannot trigger selective abstention (`U3`) because every patient receives the same width.

**Decision:**
- Partition participants into an **80% Development set (`75` participants)** and a **20% Untouched Final Test set (`18` participants)** stratified by diagnosis (`CD`, `UC`, `nonIBD`).
- Inside the `75` development participants, use **5-fold Participant-Grouped Cross-Conformal Prediction (`CV+`)** so all `75` dev participants contribute out-of-fold calibration residuals without participant leakage.
- Compute **Locally-Weighted Conformal Residuals** $r_i = |y_i - \hat{y}_i| / \hat{\sigma}(x_i)$, where $\hat{\sigma}(x_i)$ is an out-of-fold uncertainty scaler (ensemble spread / quantile width), yielding patient-specific intervals $[\hat{y}_i - q_\alpha \hat{\sigma}(x_i),\ \hat{y}_i + q_\alpha \hat{\sigma}(x_i)]$ for selective abstention (`U3`).

**Alternatives Considered:**
- Static 15% (`14`-participant) unscaled split conformal -> Rejected: small calibration sample size and constant interval width across all patients.

**Consequences:**
- ✅ 5x larger calibration pool (`75` dev participants via `CV+`) while keeping `18` test participants 100% untouched.
- ✅ Adaptive patient-specific interval widths enable meaningful selective abstention (`U3`) and risk-coverage evaluation.

---

### ADR-008: External Cohort Harmonization Scope (2026-11-10)

**Context:**
- External validation on `IBSEN III` requires formal data access (`Phase P5`) and harmonization between HMP2 shotgun metagenomics and IBSEN III 16S profiles.

**Decision:**
- Keep `ADR-008` (`D-008`) open for Phase P5 execution; freeze internal HMP2 pipelines before any external evaluation and restrict external transfer to compatible clinical + genus-level features.

**Alternatives Considered:**
- Blocking internal P1–P4 development on external cohort access -> Rejected: internal participant-held-out evaluation is self-contained and ready now.

**Consequences:**
- ✅ Unblocks P1–P4 execution while preserving strict external validation integrity for P5.

---

### ADR-009: C1 `M5` Full Causal Trajectory State Representation (2026-11-10)

**Context:**
- A raw delta alone ($\Delta \text{FC} = +20$) means something completely different when moving from $100 \to 120$ vs $180 \to 200$, or over 14 days vs 70 days.

**Decision:**
- For every dynamic feature in `M5` (FC, clinical activity scores, medications, alpha/beta diversity, top CLR taxa/pathways, and guilds), provide the **complete causal state vector**:
  1. Current value $x_{i,t}$
  2. Previous value $x_{i,t-1}$
  3. Raw change $\Delta x = x_{i,t} - x_{i,t-1}$
  4. Relative log-ratio change $\log((x_{i,t}+1)/(x_{i,t-1}+1))$
  5. Velocity (rate of change per day) $\Delta x / \Delta t_{\text{prev}}$
  6. Rolling participant history up to $t$ ($\text{mean}_{\le t}, \text{SD}_{\le t}, \text{min}_{\le t}, \text{max}_{\le t}, \text{slope}_{\le t}$)

**Alternatives Considered:**
- Providing $\Delta x = x_t - x_{t-1}$ alone -> Rejected: loses baseline severity context and elapsed time rate.

**Consequences:**
- ✅ Preserves both current/previous levels and rate of change for tree, linear, and attention models.

---

### ADR-010: Bioinformatic Version Policy (`v3` Primary + `v2` Sensitivity) (2026-11-10)

**Context:**
- The HMP2 dataset provides both legacy (`MetaPhlAn2` / `HUMAnN2`) and updated (`MetaPhlAn3` / `HUMAnN3`) tables across all `1,638` MGX samples (`100%` matched).

**Decision:**
- Use **`MetaPhlAn3` (`578` species, `187` genera) and `HUMAnN3` (`478` unstratified MetaCyc pathways)** as the primary representations for C1 and C2, and run **`MetaPhlAn2` / `HUMAnN2`** as a pre-specified bioinformatic version sensitivity check (`P2-05`).

**Alternatives Considered:**
- Mixing `v2` and `v3` features in the same model arm -> Rejected: creates colinearity and duplicate biological clades.

**Consequences:**
- ✅ Clean, non-redundant primary feature matrices with a built-in pipeline reproducibility sensitivity check.
