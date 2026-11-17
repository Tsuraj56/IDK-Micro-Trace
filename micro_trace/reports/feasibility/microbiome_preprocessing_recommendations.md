# Compositional Microbiome Preprocessing Recommendations (Agent 4)

1. **Rank & Unstratified Filtering (Before Transformation):**
   - **MetaPhlAn2/v3:** Because clade tables contain nested hierarchical levels (`k__` through `s__`/`t__`), summing across all rows violates simplex closure. Filter strictly to a single taxonomic rank (e.g., **species-level** rows matching `s__` without `|t__`, or **genus-level** rows matching `g__` without `|s__`) and renormalize sample columns to sum to 1.0 (excluding or separately tracking `UNKNOWN`).
   - **HUMAnN2/v3 Pathways & ECs:** For C1/C2 primary functional modeling, separate **unstratified** community-level pathways (rows without `|`, excluding `UNMAPPED` and `UNINTEGRATED`) from taxon-stratified rows (`|`), as stratified rows duplicate community totals across species.
2. **Train-Only Prevalence & Abundance Filtering (PR-03):**
   - Fit prevalence thresholds (e.g., `>= 10%` non-zero across training samples) **exclusively on the training partition**. Apply the frozen feature mask to calibration and test partitions.
3. **Zero Replacement & Centered Log-Ratio (`CLR`) Transformation:**
   - Microbiome relative abundances are compositional (`sum = 1` or `100%` or `10^6 CPM`).
   - Before applying `CLR(x_j) = log(x_j / geometric_mean(x))`, zeros must be handled using a documented, train-fitted multiplicative replacement or pseudocount sensitivity grid (e.g., half the minimum non-zero relative abundance observed in the training set vs. Bayesian-multiplicative replacement).
4. **Sequencing Depth & Batch Covariates:**
   - Retain `reads_filtered`, `reads_human`, and `site_name` / `LCSET` from `hmp2_metadata_2018-08-20.csv` as technical QC and batch covariates in C2 sensitivity checks.
