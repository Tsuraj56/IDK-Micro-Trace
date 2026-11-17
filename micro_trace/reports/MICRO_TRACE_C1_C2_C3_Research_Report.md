# MICRO-TRACE: Longitudinal Microbiome-Informed Prediction, Biological Signatures, and Uncertainty Calibration for Fecal Calprotectin in IBD (Components C1–C3 Research Report)

**Date:** 2026-11-16  
**Cohort:** IBDMDB / Human Microbiome Project 2 (HMP2)  
**Validation Status:** 5-Fold Participant-Grouped Development Cross-Validation (`N = 75` participants, `0` participant overlap; `N = 18` test participants / `74` consecutive pairs strictly held out as `untouched_held_out`)  
**Software Verification:** `63/63` automated unit and leakage tests passing (`pytest micro_trace/tests -v`, `seed = 42`)

---

## Abstract

Predicting longitudinal mucosal inflammation—quantified via fecal calprotectin (FC)—in Inflammatory Bowel Disease (IBD) is critical for proactive disease management, yet prior machine learning studies frequently conflate cross-sectional classification with prospective continuous forecasting or suffer from within-patient data leakage. Using the multi-omic IBDMDB/HMP2 cohort under a strictly causal, participant-grouped evaluation protocol, we report integrated findings across three research components:

1. **C1 (Prospective Continuous Forecasting):** Without baseline calprotectin anchoring, taxonomic (`B3`) and functional (`B4`) microbiome models collapse ($R^2 = 0.006\text{--}0.036$). Reformulating the target from absolute future level to **longitudinal change ($\Delta y = y_{t+1} - y_t$)**, compressing 600+ sparse omics features into **curated biological guilds (`M_guild_delta`, 70 features)**, pre-training a continuous-time residual cross-attention network (`MicroTrace-Attn`) across multi-horizon transitions ($N = 1,671$ train-fold pairs), and blending with `ElasticNetCV` via leak-free out-of-fold (OOF) stacking improves explained variance ($R^2$) from `0.3721` (Persistence `M0`) to **`0.4307`–`0.4801`**, achieves **`62.34%`–`63.15%` directional accuracy** (vs `0%` for persistence), and lowers CV MAE to **`0.2604`–`0.2608`** (vs `0.2669` persistence).
2. **C2 (Reproducible & Subtype-Specific Microbiome Signatures):** Participant random-intercept Linear Mixed-Effects (`LME`) modeling reveals strong, FDR-controlled (`BH-FDR < 0.05`) signatures for **concurrent inflammation (`Current FC`, 502 visits)**: **2 biological guilds** (pathobiont guild $\beta = +0.0840$, `FDR = 0.0017`; protective-to-pathobiont ratio $\beta = -0.0745$, `FDR = 0.0067`), **4 bacterial species** (led by *Veillonella dispar*, $\beta = +0.0883$, `FDR = 0.0157`), and **20 MetaCyc pathways**. Pooled **`Future FC` (444 pairs)** yields `0` global univariate `FDR < 0.10` hits; however, **subtype-stratified LME** (`CD`, `UC`, `nonIBD` separately) uncovers subtype-specific `Future FC` predictors (`FDR < 0.05`) including *Barnesiella intestinihominis* ($\beta = -0.1618$, `FDR = 0.0006`) and *Dialister invisus* ($\beta = +0.0995$, `FDR = 0.0092`) in `CD`, and *Fusicatenibacter saccharivorans* ($\beta = -0.1841$, `FDR = 0.0053`) in `UC`, with `0` shared stable future predictors between `CD` and `UC` ($r = -0.117$). Ridge-anchored Cross-Omic Attention achieves positive OOF $R^2$ (`+0.0398` current, `+0.4657` future) but diffuse pair-level attention weights (`0` pairs stable at $\ge 4/5$ folds).
3. **C3 (Uncertainty Calibration & Selective Abstention):** Off-the-shelf Quantile Regression (`U1`) severely under-covers at nominal `90%` (`69.8%–71.4%` overall) and collapses to **`37.7%` coverage on active flares (`FC >= 250`)**. Out-of-fold locally-weighted participant-grouped Cross-Conformal prediction (`U2`) restores calibrated **`90.57%–91.11%` overall coverage** and **`88.5%` flare coverage**, while selective abstention (`U3`) monotonically reduces MAE (`0.3115` $\to$ `0.2828` on `M1`) while maintaining $\ge 90\%$ coverage.

---

## 1. Cohort Construction, Splits & Preprocessing (`P0` / `T-001` / `T-002`)

### 1.1 Cohort & Strict Zero-Leakage Partition
* **Source Data:** Complete IBDMDB/HMP2 release (`46` raw files, `1,453.02 MB`, `131` unique participants, `1,638` raw MGX runs deduplicated to `1,595` unique physical stool collection events `site_sub_coll`).
* **Outcome Harmonization:** `fecalcal` and `fecalcal_ng_ml` harmonized on the ELISA $\mu\text{g/g}$ scale (`3.45–493.7`) and log-transformed as $y = \log_{10}(\text{FC} + 1)$.
* **Causal Alignment (`PR-01`):**
  * **Current FC Cohort (`C2`):** `503` raw / `502` clean same-day (`0d`) `MGX + FC` visits across `95` participants (`415` dev visits across `75` participants).
  * **Consecutive Future FC Cohort (`C1` & `C2`):** `445` raw / `444` clean consecutive `0d` pairs across `93` participants (`42` CD, `27` UC, `24` nonIBD).
  * **All-Horizons Causal Cohort:** `1,671` causal pairs across `93` participants (used strictly within training folds for multi-horizon pre-training).
* **Participant-Grouped 80/20 Split (`PR-02`, `seed = 42`):**
  * **Development Set (80%):** `75` participants (`34` CD, `22` UC, `19` nonIBD; `371` consecutive pairs) partitioned into **5 diagnosis-stratified participant-grouped folds** (`15` participants/fold).
  * **Untouched Test Set (20%):** `18` participants (`8` CD, `5` UC, `5` nonIBD; `74` consecutive pairs) locked with `0` participant overlap.

### 1.2 Train-Only Compositional & Clinical Preprocessing (`PR-03`, `ADR-004`)
All transformations are fitted strictly on training-fold visits and applied to validation/test folds:
* **Zero-Sum MGX Handling (`ADR-004`):** Samples `MSM5LLDM` and `MSM5LLDS` (all-zero abundances) imputed to `0.0` in CLR space with explicit indicator `is_zero_sum_mgx_sample = 1.0`.
* **Prevalence Filtering & CLR Transform:** Retained features with $\ge 10\%$ prevalence in the training partition (`120/578` MetaPhlAn3 species; `361/476` unstratified HUMAnN3 MetaCyc pathways in dev), added train-derived half-minimum pseudocounts (`7.50e-7` species; `5.85e-7` pathways), and applied Centered Log-Ratio (CLR) transformation.
* **Clinical & Missingness Encoding:** Train-median imputation for `HBI`, `SCCAI`, `bowel_freq`, `abdominal_pain`, `wellbeing`, and `consent_age` paired with explicit binary `_is_missing` flags; standardized medication indicators (`Antibiotics`, `Immunosuppressants`, etc.); train-fitted standard scaling.
* **Curated Biological Guilds (`c1_guild_definitions.csv`):**
  * **Microbial Guilds:** SCFA butyrate producers (`19` species), Pathobionts (`53` species), and Protective-to-Pathobiont log-ratio.
  * **Functional Guilds:** SCFA butanoate fermentation (`21` pathways), Sulfur/$H_2S$ (`6`), Oxidative stress (`2`), Mucin/hexuronide degradation (`8`), and Enterobacterial LPS (`5`).

---

## 2. Component C1: Prospective Future Calprotectin Forecasting

### 2.1 Architecture: `MicroTrace-Attn`
`MicroTrace-Attn` is a continuous-time residual cross-attention network designed for irregularly sampled clinical visits ($\Delta t \in [14, 70]$ days):
* **Inputs:**
  1. Causal visit history tensor `(B, 25, 15)` (12 dynamic trajectory variables + 3 missingness flags for visits $1 \dots t \le t_{\text{index}}$).
  2. Continuous calendar time deltas `(B, 25)` encoded via Fourier time-decay embeddings $\phi(\Delta t)$.
  3. Static anchor vector `(B, 6)` (index FC, elapsed days, age, CD/UC/nonIBD indicators).
  4. Current-visit multi-omic/clinical vector `(B, D_arm)`.
* **Residual Anchor Skip:** Fixed linear skip connection anchored to index FC ($w_0 = 1.0$ in level mode) or zero-change ($0.0$ in delta mode), preventing mean-collapse on modest sample sizes ($N=371$).

### 2.2 Level Prediction Baseline Benchmark (`T-003`, 5-Fold Dev CV)
| Arm | Model Family | Dim ($D$) | CV MAE | CV RMSE | CV $R^2$ | Spearman $\rho$ |
|---|---|---:|---:|---:|---:|---:|
| **`M1`** | **`MicroTrace-Attn`** | **29** | **0.2616 $\pm$ 0.034** | **0.4140** | **0.3878** | **0.6863** |
| `M5` | `MicroTrace-Attn` | 646 | 0.2635 $\pm$ 0.032 | 0.4144 | 0.3879 | 0.6870 |
| `M4` | `MicroTrace-Attn` | 524 | 0.2642 $\pm$ 0.032 | 0.4158 | 0.3837 | 0.6850 |
| **`M0`** | **`PersistenceBaseline`** | **2** | **0.2687** | **0.4233** | **0.3778** | **0.6958** |
| `M1` | `ElasticNet` | 29 | 0.3116 $\pm$ 0.031 | 0.3860 | 0.4684 | 0.6847 |
| `M1` | `RandomForest` | 29 | 0.3230 $\pm$ 0.034 | 0.4156 | 0.3842 | 0.6272 |
| `M1` | `HistGradientBoosting` | 29 | 0.3359 $\pm$ 0.039 | 0.4296 | 0.3434 | 0.5990 |
| `B3` | `RandomForest` (Taxa only, no FC) | 130 | 0.4669 $\pm$ 0.031 | 0.5205 | 0.0364 | 0.2736 |
| `B4` | `RandomForest` (Function only, no FC) | 368 | 0.4765 $\pm$ 0.014 | 0.5296 | 0.0059 | 0.2344 |
| `B0` | `MeanBaseline` | 0 | 0.4970 | 0.5367 | 0.0000 | 0.0000 |

### 2.3 Optimization Trials & Combinations (`Trials 1–4` + `Combos A–C`)
Four targeted improvements were evaluated individually and in combination (`c1_trials_combinations_benchmark.csv`):
1. **Trial 1 — Target Delta ($\Delta y = y_{t+1} - y_t$) + L1 Loss:** Eliminated shrinkage toward the cohort mean, dropping `ElasticNetCV` MAE from `0.3204` $\to$ `0.2729` (`M5`) and raising $R^2$ to `0.4674`–`0.4719`.
2. **Trial 2 — Biological Guild Bottleneck (`M_guild` / `M_guild_delta`):** Compressing 648 raw features into **70 guild + clinical delta features** yielded the **highest single-model $R^2$ (`0.4801`)** and **lowest single-model RMSE (`0.3821`)** (`M_guild_delta ElasticNetCV`, `62.19%` directional accuracy).
3. **Trial 3 — Leak-Free OOF Stacking (`MicroTrace-Attn` + `ElasticNetCV`):** Nested inner-CV $\alpha$-blending combined `MicroTrace-Attn`'s low MAE with `ElasticNetCV`'s high variance capture.
4. **Trial 4 — Multi-Horizon Pre-Training ($N=1,671$ train-fold pairs):** Raised rank correlation on `M_guild` ($\rho = 0.6912\text{--}0.6916$), boosting stacked ensemble synergy.

| Rank | Combination | Arm | Model Family | Dim ($D$) | CV MAE | Dir. Acc | CV $R^2$ | CV RMSE | Spearman $\rho$ |
|:---:|---|---|---|---:|---:|---:|---:|---:|---:|
| **1** | **Combo C (Full Stack)** | `M_guild+M5` | `OOFStack(M_guild_PretrainAttnL1 + M5_ENet)` | 691 | **0.2604** | 62.23% | 0.4289 | 0.4003 | 0.6878 |
| **2** | **Combo C (Compact Champion)** | `M_guild+M_guild_delta` | `OOFStack(M_guild_PretrainAttnL1 + M_guild_delta_ENet)` | **70** | **0.2608** | **62.34%** | **0.4307** | **0.3996** | **0.6922** |
| **3** | **Combo B (Cross-Arm)** | `M1+M5` | `OOFStack(M1_AttnL1 + M5_ENet)` | 677 | 0.2608 | 61.05% | 0.4274 | 0.4008 | 0.6861 |
| **9** | **Combo B (T1+T3)** | `M5` | `OOFStack(M5_AttnL1 + M5_ENet)` | 648 | 0.2624 | **63.12%** | **0.4322** | **0.3990** | 0.6857 |
| **19** | **Baseline** | `M0` | `PersistenceBaseline` | 2 | **0.2669** | 0.00% | 0.3721 | 0.4196 | 0.6876 |
| **24** | **Combo A (Best Single $R^2$)** | `M_guild_delta` | `ElasticNetCV` | 70 | 0.2769 | 62.19% | **0.4801** | **0.3821** | 0.6854 |

### 2.4 C1 Key Scientific Takeaways
* **Baseline FC is Essential:** Microbiome alone cannot predict absolute future FC (`B3`/`B4` $R^2 \approx 0$).
* **Guilds Beat Raw Omics:** 70 curated biological guild features outperform 648 raw species/pathways on $R^2$, RMSE, and Spearman $\rho$.
* **Delta + Stacking Unlocks Directionality & Variance:** Relative $R^2$ improves by **+15.7% to +29.0%** over persistence, with **62.3%–63.1% directional accuracy** (vs `0%` persistence), while point MAE reduction remains modest (~2.4%) due to stool assay variability and unmeasured inter-visit triggers.

---

## 3. Component C2: Biological Signatures & Cross-Omic Attention

Evaluated using participant random-intercept Linear Mixed-Effects (`LME`) models (`BH-FDR` controlled), 5-fold Stability Selection, and Ridge-anchored Cross-Omic Attention (`c2_summary.json`).

### 3.1 Concurrent Inflammation (`Current FC`, `415` dev visits / `75` participants)
* **FDR-Significant Discoveries (`BH-FDR < 0.05`):** **2 biological guilds**, **4 species**, and **20 MetaCyc pathways** (`9` species and `32` pathways at `FDR < 0.10`).
* **Top Verified Associations:**
  * `guild_pathobiont_log`: $\beta = +0.0840$, $p = 1.57 \times 10^{-4}$, `FDR = 0.0017`
  * `guild_protective_vs_pathobiont_logratio`: $\beta = -0.0745$, $p = 1.21 \times 10^{-3}$, `FDR = 0.0067`
  * *Veillonella dispar*: $\beta = +0.0883$, $p = 2.09 \times 10^{-4}$, `FDR = 0.0157`
  * `KETOGLUCONMET-PWY` (ketogluconate metabolism): $\beta = +0.1003$, $p = 2.69 \times 10^{-6}$, `FDR = 0.0005`
  * `PWY-5675` (nitrate reduction V): $\beta = +0.0976$, $p = 2.86 \times 10^{-6}$, `FDR = 0.0005`
  * `P105-PWY` (TCA cycle IV): $\beta = +0.0923$, $p = 8.95 \times 10^{-6}$, `FDR = 0.0009`
  * `PWY-6628` (L-phenylalanine biosynthesis): $\beta = +0.0916$, $p = 9.73 \times 10^{-6}$, `FDR = 0.0009`
* **5-Fold Stability:** Jaccard stability `0.464` (LME), `0.320` (ElasticNet), `0.348` (RF), with `59` LME-stable, `31` ENet-stable, and `17` RF-stable features ($\ge 4/5$ folds).

### 3.2 Prospective Inflammation (`Future FC`, `370` dev pairs / `75` participants)
* **Global FDR:** `0` individual features pass `BH-FDR < 0.10` for `Future FC` alone—confirming that single taxa do not independently determine future FC without baseline anchoring.
* **Cross-Fold Stable Signals ($\ge 4/5$ folds):** `7` LME-stable, `11` ENet-stable, and `12` RF-stable features persist across folds, led by *Blautia obeum* (`5/5` LME folds, $\beta = -0.0665$), `COLANSYN-PWY` (`4/5` LME & ENet folds, $\beta = -0.0581$), and `PRPP-PWY` (`4/5` LME folds, $\beta = -0.0586$).

### 3.3 Cross-Omic Attention (`[Bacteria] <-> [Pathways] <-> [FC]`)
* Using train-fold top-K Pearson screening (`35` taxa $\times$ `50` pathways) and a Ridge-anchored clinical skip:
  * **Current FC:** OOF MAE = `0.4481`, OOF $R^2$ = **`+0.0398`** (`4/5` folds positive), Spearman $\rho$ = `0.3055`.
  * **Future FC:** OOF MAE = `0.2938`, OOF $R^2$ = **`+0.4657`** (`5/5` folds positive), Spearman $\rho$ = `0.6727`.
* **Interpretation Caveat:** While predictive $R^2$ is positive, individual `[Bacterium] <-> [Pathway]` softmax attention weights remain diffuse across patient folds (`0` pairs stable at $\ge 4/5$ folds; max `3/5` folds).

### 3.4 Subtype-Stratified Analysis (`CD` vs. `UC` vs. `nonIBD`)
To separate within-disease inflammation drivers from between-disease baseline confounding, the full Guild + Species + Pathway `LME` (`BH-FDR` computed within each subgroup and family) and 5-fold stability selection were executed separately within `CD` (`34` dev participants), `UC` (`22`), and `nonIBD` (`19`) (`c2_subtype_full_lme.csv`, `c2_subtype_full_stability.csv`, `c2_subtype_comparison_summary.json`):

* **Concurrent Inflammation (`Current FC`):**
  * **`CD` (`196` visits / `34` pts):** `3` features at `FDR < 0.05` (`14` at `FDR < 0.10`; `44` LME-stable at $\ge 4/5$ folds), led by *Bacteroides fragilis* ($\beta = +0.1383$, `FDR = 0.0107`), *Alistipes indistinctus* ($\beta = -0.1205$, `FDR = 0.0107`), *Barnesiella intestinihominis* ($\beta = -0.1346$, `FDR = 0.0398`), and `guild_pathobiont_log` ($\beta = +0.0816$, `FDR = 0.0575`).
  * **`UC` (`121` visits / `22` pts):** `1` feature at `FDR < 0.05` (`22` LME-stable at $\ge 4/5$ folds): `GOLPDLCAT-PWY` glycerol degradation ($\beta = -0.1923$, `FDR = 0.0099`).
  * **`nonIBD` (`98` visits / `19` pts):** `106` features at `FDR < 0.05` (`3` guilds, `103` pathways; `112` LME-stable), led by `PWY-5103` L-isoleucine biosynthesis III ($\beta = -0.1523$, `FDR = 0.0060`) and `BRANCHED-CHAIN-AA-SYN-PWY` ($\beta = -0.1514$, `FDR = 0.0060`).
  * **Shared `CD` & `UC` `Current FC` Signatures ($\ge 4/5$ folds):** Only `3` features were stable in both subtypes (`PWY-5675: nitrate reduction V`, *Ruminococcus bicirculans*, and *Veillonella dispar*), with modest cross-subtype effect correlation ($r = 0.180$, $p = 9.3 \times 10^{-5}$; guild $r = 0.594$).
* **Prospective Inflammation (`Future FC`):**
  * **`CD` (`177` pairs / `34` pts):** `2` species at `FDR < 0.05` (`3` at `FDR < 0.10`; `7` LME-stable): *Barnesiella intestinihominis* ($\beta = -0.1618$, `FDR = 0.0006`, `5/5` folds across LME, ENet, and RF), *Dialister invisus* ($\beta = +0.0995$, `FDR = 0.0092`), and `PRPP-PWY` ($\beta = -0.0864$, `FDR = 0.0703`, `5/5` folds across LME, ENet, and RF).
  * **`UC` (`108` pairs / `22` pts):** `1` species at `FDR < 0.05` (`6` LME-stable): *Fusicatenibacter saccharivorans* ($\beta = -0.1841$, `FDR = 0.0053`, `5/5` folds across LME, ENet, and RF; *Bacteroides massiliensis* also `5/5` across LME, ENet, and RF).
  * **`nonIBD` (`85` pairs / `19` pts):** `24` features at `FDR < 0.05` (`39` LME-stable), led by `fguild_oxidative_stress_log` ($\beta = -0.1175$, `FDR = 0.0021`) and methionine biosynthesis (`PWY-5347`, `METSYN-PWY`, `MET-SAM-PWY`, `FDR = 0.0054`).
  * **Why Pooled `Future FC` Was Masked:** `CD` and `UC` shared **`0`** `>= 4/5` LME-stable or `FDR < 0.10` `Future FC` features and exhibited an overall negative effect-size correlation ($r = -0.117$, $p = 0.011$), demonstrating that pooling distinct IBD subtypes cancels out subtype-specific prospective signals.
* **Small-Sample Sensitivity Caveat (`nonIBD`):** Because `nonIBD` contains only `19` participants (including `E5002` with elevated mean $\log_{10}(\text{FC}+1) = 2.38$), the `106` `nonIBD` `Current FC` FDR hits should be treated as exploratory pending leave-one-participant-out sensitivity verification.

---

## 4. Component C3: Uncertainty Calibration & Selective Abstention

Evaluated at **`90%` nominal coverage** ($\alpha = 0.10$) across `M1`, `M_guild_delta`, and `M5` using strictly out-of-fold residual estimation inside 5-fold participant-grouped `CV+` (`c3_summary.json`).

### 4.1 System Comparison (`90%` Target Coverage)
| Arm | `U0` Homoscedastic Conformal | `U1` Quantile Regression | **`U2` Locally-Weighted `CV+`** | `U2` Cluster-Weighted Sensitivity |
|---|:---:|:---:|:---:|:---:|
| **`M1`** (29 dim) | `90.30%` (width `1.292`) | `71.43%` (width `0.929`) | **`91.11%`** (width `1.327`, std `0.198`) | `92.72%` (width `1.475`) |
| **`M_guild_delta`** (70 dim) | `89.76%` (width `1.280`) | `69.81%` (width `0.922`) | **`90.57%`** (width `1.321`, std `0.177`) | `91.64%` (width `1.466`) |
| **`M5`** (646 dim) | `90.03%` (width `1.287`) | `71.43%` (width `0.931`) | **`90.57%`** (width `1.320`, std `0.175`) | `91.64%` (width `1.463`) |

### 4.2 Subgroup Calibration Audit (`M_guild_delta`: Active Flares vs. Remission)
* **Failure of Standard Quantile Regression (`U1`):** `U1` severely under-covers overall (`69.8%`) and collapses on **active flares (`FC >= 250`, $n=61$)** to **`37.70%` coverage** (`width = 0.930`).
* **Adaptivity of Locally-Weighted `CV+` (`U2`):** `U2` automatically widens intervals on high-variance active flares (`width = 1.379` vs `1.273` in remission) to recover **`88.52%` coverage on flares**, `94.49%` on mild activity (`50–250`), `93.82%` on Crohn's Disease, and `90.57%` overall.

### 4.3 Selective Abstention (`U3` Risk–Coverage Tradeoff)
Abstaining on visits with the widest predicted uncertainty ($\hat{\sigma}$) monotonically reduces prediction error while preserving `~90%` coverage at every acceptance level (e.g., on `M1`):
* `100.0%` acceptance (`n = 371`): Coverage = `91.11%`, MAE = `0.3115`, RMSE = `0.3901`
* `77.4%` acceptance (`n = 287`): Coverage = `90.24%`, MAE = `0.3036`, RMSE = `0.3831`
* `59.6%` acceptance (`n = 221`): Coverage = `90.05%`, MAE = `0.2899`, RMSE = `0.3696`
* `49.1%` acceptance (`n = 182`): Coverage = `90.11%`, MAE = **`0.2828`**, RMSE = **`0.3610`**

---

## 5. Limitations & Pending Workstreams

1. **Untouched Test Partition:** All results above are 5-fold participant-grouped CV on the `75` development participants (`371` consecutive pairs). The `18` test participants (`74` pairs) remain **`untouched_held_out`** until final locked evaluation.
2. **Sensitivity & Pending Components:** `MetaPhlAn2`/`HUMAnN2` (`v2`) sensitivity (`P2-05`) and `Component C4` (evidence-constrained dietary-pattern decision support using `C3` `U3` abstention) are pending execution.
3. **Interpretation Guardrail:** Predictive association is not causation; metagenomic pathway abundances represent functional potential rather than measured transcript/protein expression.
