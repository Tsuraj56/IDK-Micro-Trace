# MICRO-TRACE: Simple English Summary Report (Parts C1, C2 & C3)

**Date:** 2026-11-16  
**Dataset:** Human Microbiome Project 2 (HMP2 / IBDMDB)  
**How We Tested It:** Tested on `75` patients across 5 patient-grouped rounds (no patient's data ever appeared in both training and testing at the same time). Another `18` patients (`74` visit pairs) are still locked away untouched for final testing.  
**Software Checks:** All `63/63` automated tests passed.

---

## The Big Picture (In One Minute)

We set out to answer three questions about Inflammatory Bowel Disease (IBD) using stool bacteria data and **fecal calprotectin (FC)** (a stool test that measures gut inflammation):

1. **Part C1 (Predicting Future Inflammation):** Can gut bacteria help predict next month's inflammation?
   * **Short answer:** Not by themselves. You *must* start with today's inflammation score. However, when we asked the AI to predict **how much inflammation will change** (instead of guessing the exact raw number), grouped 600+ messy bacteria into **70 clean "bacterial team" features**, and blended a deep-learning model with a simple linear model, our predictions explained **16% to 29% more of the real ups-and-downs** and guessed the **up/down direction correctly ~62%–63% of the time** (compared to 0% for just copying today's score).
2. **Part C2 (Which Bacteria Matter Most?):** Which specific bacteria and bacterial activities link to inflammation?
   * **Short answer:** For **today's** inflammation, there is a clear signal: "bad" inflammation-linked bacteria teams go up, and "good" protective bacteria teams go down. For **next month's** inflammation, when we lumped all patients together, no single bacteria stood out. **But when we split Crohn's Disease (`CD`) and Ulcerative Colitis (`UC`) apart**, clear future warning-sign bacteria appeared for each disease—they had been canceling each other out when mixed together!
3. **Part C3 (Knowing When the AI is Guessing):** Can the AI give a reliable safety range and say "I'm not sure" on unpredictable patients?
   * **Short answer:** Yes. A standard off-the-shelf confidence method failed badly on the sickest patients (only right **37.7%** of the time during active flares instead of the promised 90%). Our custom safety-range method automatically widens its range for unstable patients, hitting **~91% accuracy overall** and **88.5% even during active flares**. When it skips the most uncertain ~50% of visits, its error drops cleanly.

---

## 1. How We Prepared the Data (Zero Cheating)

* **Starting Data:** `131` participants, `1,595` unique stool collections (after removing `43` duplicate lab runs).
* **Log Scale:** Because calprotectin numbers jump wildly (`3.45` to `493.7`), we converted them to a compressed $\log_{10}(\text{FC}+1)$ scale so extreme spikes don't break the math.
* **Two Main Checklists:**
  * **Same-Day Visits (for C2):** `502` clean visits (`95` patients) where stool bacteria and calprotectin were measured on the exact same day.
  * **Next-Visit Pairs (for C1 & C2):** `444` clean pairs of consecutive visits (`93` patients), plus `1,671` wider-gap visit pairs used strictly inside training rounds to give the AI extra practice.
* **Strict Separation (`80/20` Split):**
  * **Training/Testing Group (80%):** `75` patients (`34` Crohn's, `22` Ulcerative Colitis, `19` non-IBD; `371` consecutive pairs), split into 5 groups of 15 patients.
  * **Locked Final Exam Group (20%):** `18` patients (`74` pairs) kept completely untouched.
* **Clean Microbiome Math (Done Inside Each Training Group Only):**
  * Fixed 2 stool samples (`MSM5LLDM`, `MSM5LLDS`) that had all zeros.
  * Removed super-rare bacteria/pathways (kept `120` common species and `361` common pathways) and converted raw percentages using a standard microbiome ratio method (CLR) so percentages don't create fake correlations.
  * Filled in missing symptom scores with the training group's middle value and added a `"was missing"` flag so the AI knew what was filled in.
  * Grouped bacteria and pathways into **7 biological teams ("guilds")**: good butyrate-producing bacteria (`19` species), bad pathobionts (`53` species), their good-to-bad ratio, plus 5 functional teams (butyrate fermentation, sulfur/$H_2S$ gas, oxidative stress, mucus breakdown, and *E. coli*-like LPS).

---

## 2. Part C1: Predicting Next Month's Inflammation

### What Goes Into `MicroTrace-Attn` (Our Custom AI)
Think of `MicroTrace-Attn` as a doctor reviewing a patient's chart:
1. **Past Medical Diary:** Up to `25` past visits (`15` numbers per visit: calprotectin, symptom scores, gut diversity, bacterial team scores, and missing-data flags).
2. **Time Gaps:** Exact number of days since each past visit (because real patients don't visit on a fixed schedule).
3. **Today's Snapshot:** Today's calprotectin, days until next visit, age, and diagnosis (`CD`, `UC`, or `non-IBD`).
4. **Today's Full Features:** Depending on the test, anywhere from `29` clinical features up to `648` full microbiome features.

Crucially, the AI has a **built-in safety anchor**: it starts directly from today's calprotectin and only predicts a small nudge up or down, so it can never collapse to a terrible guess.

### What Happened in Our First Tests
* **Microbiome Alone Fails:** If you hide today's calprotectin and try to predict next month using *only* bacteria (`B3`) or bacterial functions (`B4`), the models score near zero ($R^2 = 0.006\text{--}0.036$, error `0.467`–`0.477`, barely better than guessing the average `0.497`).
* **Copying Today's Score (`M0 Persistence`) is Hard to Beat:** Simply guessing *"next month = today"* gets an error (MAE) of **`0.2669`–`0.2687`**.
* **Standard AI Models Did Worse:** Under standard settings, `ElasticNet` (`0.3116`), `RandomForest` (`0.3230`), and `HistGradientBoosting` (`0.3359`) all had *worse* error than copying today's score. Only `MicroTrace-Attn` beat it (`0.2616`).

### The 4 Breakthroughs That Improved Everything
We tested 4 improvements and combined them:
1. **Predict the *Change* ($\Delta y$), Not the Raw Number:** Asking *"how much will inflammation go up or down?"* stopped models from getting pulled toward the average. `ElasticNetCV`'s error dropped from `0.3204` down to `0.2729`, and its ability to explain real swings ($R^2$) jumped to `0.47`.
2. **Use 70 "Bacterial Team" Features Instead of 648 Raw Ones:** Replacing 600+ noisy individual bacteria with our biological teams (`M_guild_delta`, `70` features) produced the **highest single-model swing-explaining power ($R^2 = 0.4801$)** and lowest RMSE (`0.3821`).
3. **Blend the Two Best Models (`OOFStack`):** Combining `MicroTrace-Attn` (steady, low average error) with `ElasticNetCV` (great at catching big swings) beat every individual model.
4. **Extra Practice on All 1,671 Visit Pairs:** Pre-training the AI on wider-gap visits helped it rank patients from healthiest to sickest (`Spearman = 0.6922`).

### Final C1 Scoreboard
| Model | Features Used | Average Error (MAE) | Up/Down Direction Accuracy | Swing Explained ($R^2$) |
|---|---:|---:|---:|---:|
| **Just Copy Today (`M0 Persistence`)** | 2 | `0.2669` | `0%` (always guesses 0 change) | `0.3721` |
| **Best Overall Blend (`M_guild + M5`)** | 691 | **`0.2604`** | `62.23%` | `0.4289` |
| **Best Compact Blend (`M_guild + M_guild_delta`)** | **70** | **`0.2608`** | **`62.34%`** | **`0.4307`** |
| **Best Single Linear Model (`M_guild_delta ElasticNet`)** | 70 | `0.2769` | `62.19%` | **`0.4801`** |

* **Plain-English Takeaway:** Average error only dropped slightly (~2.4%) because stool tests have natural day-to-day lab noise and unexpected events happen between visits (stress, infections, diet). The real win is **16%–29% more variance explained**, **~62%–63% up/down direction accuracy**, and proving **70 clean bacterial-team features beat 648 raw ones**.

---

## 3. Part C2: Which Gut Bacteria & Pathways Actually Connect to Inflammation

We used longitudinal mixed-effects statistics (which properly handle the same patient having multiple visits) to check both **Today's** and **Next Month's** inflammation.

### 3.1 Today's Inflammation (`Current FC`): Strong, Clear Signals
* **2 bacterial teams**, **4 individual species**, and **20 metabolic pathways** passed strict false-discovery checks (`FDR < 0.05`).
* **The Team Balance:** More **"bad" pathobiont bacteria** meant significantly **higher** calprotectin ($\beta = +0.084$, `FDR = 0.0017`). A higher **good-to-bad bacteria ratio** meant significantly **lower** calprotectin ($\beta = -0.075$, `FDR = 0.0067`).
* **Key Individual Culprits:** *Veillonella dispar* (linked to inflammation) and bacterial pathways for **nitrate reduction** and **ketogluconate/stress metabolism**—classic signs of bacteria surviving in an inflamed gut.

### 3.2 Next Month's Inflammation (`Future FC`): Why Splitting Crohn's vs. UC Changed Everything
* **When All Patients Were Mixed Together:** **Zero** individual bacteria passed strict statistical checks (`FDR < 0.10`) for predicting next month's calprotectin. Only a few steady clues—like the helpful bacterium *Blautia obeum* (`5/5` folds) and `COLANSYN-PWY` (`4/5` folds)—showed up consistently.
* **The Big Discovery — Splitting Crohn's (`CD`) vs. Ulcerative Colitis (`UC`):**
  When we analyzed each disease separately, **clear next-month predictors appeared**:
  * **In Crohn's Disease (`CD`):** *Barnesiella intestinihominis* strongly predicted **lower** future inflammation (`FDR = 0.0006`, consistent in `5/5` folds across all methods), while *Dialister invisus* predicted **higher** future inflammation (`FDR = 0.0092`).
  * **In Ulcerative Colitis (`UC`):** *Fusicatenibacter saccharivorans* strongly predicted **lower** future inflammation (`FDR = 0.0053`, consistent in `5/5` folds across all methods).
  * **In Non-IBD Controls:** `24` features passed (`FDR < 0.05`), though with only `19` people (including one person `E5002` with unusually high calprotectin), those non-IBD hits should be treated cautiously.
* **Why Mixing Them Hid the Signal:** Crohn's and Ulcerative Colitis had **zero overlapping** top future-inflammation bacteria and actually showed a slight **negative correlation** ($r = -0.117$) between their effects. Pooling them together canceled the signals out!

### 3.3 Cross-Omic Attention (Bacteria + Pathways Together)
* When anchored on clinical history, the network predicted future calprotectin decently ($R^2 = +0.4657$, `5/5` folds positive), **but** specific `[Bacterium] <-> [Pathway]` pairs did not repeat consistently across patient groups (`0` pairs stable in `4/5` folds). We report this honestly: the model spreads attention broadly rather than locking onto one fixed bacteria–pathway pair.

---

## 4. Part C3: Safety Ranges & Knowing When to Say "I'm Not Sure"

Instead of only giving a single number, Part C3 gives doctors a **90% confidence range** (where the true lab value should land 90% of the time) and an **"Abstain"** option for unpredictable patients.

### 4.1 Standard Confidence Ranges Fail Dangerously on Sick Patients
We tested a common off-the-shelf method (**Quantile Regression `U1`**) against our **Locally-Weighted Cross-Conformal method (`U2`)**, both aiming for **90% accuracy**:

| Patient Group | Standard Method (`U1`) | Our Conformal Method (`U2`) | What Happened |
|---|---:|---:|---|
| **All Patients Overall (`M_guild_delta`)** | `69.8%` | **`90.6%`** | Standard method missed the 90% target by 20 points. |
| **Patients in Active Severe Flare (`FC >= 250`)** | **`37.7%`** | **`88.5%`** | Standard method was **wrong >62% of the time** on the sickest patients! `U2` widened its range (`1.379` vs `1.273` in remission) and stayed safe. |
| **Crohn's Disease (`CD`)** | `71.9%` | **`93.8%`** | Reliable coverage across subtypes. |

### 4.2 The "Abstain When Unsure" Safety Switch (`U3`)
When we let our `U2` system **refuse to guess** on the patients with the widest uncertainty ranges:
* Keeping **100%** of visits (`371`): Error (MAE) = `0.3115`, Coverage = `91.11%`
* Keeping the **77%** most certain visits (`287`): Error drops to `0.3036`, Coverage = `90.24%`
* Keeping the **49%** most certain visits (`182`): Error drops to **`0.2828`**, Coverage = `90.11%`

Every time the AI steps back from an unpredictable patient, accuracy improves on the rest while keeping its **90% safety guarantee**.

---

## 5. What Still Needs to Be Done
1. **Final Exam on the Locked 20% Test Set:** All numbers above are from the `75` development patients; the `18` held-out patients (`74` pairs) will be tested **once** at the very end.
2. **Older Microbiome Version Check (`P2-05`):** Re-running C2 on the older `MetaPhlAn2`/`HUMAnN2` tables to confirm results hold across software versions.
3. **Part C4 (Dietary Decision Support):** Connecting C3's "Abstain" safety switch to evidence-based dietary pattern recommendations.
