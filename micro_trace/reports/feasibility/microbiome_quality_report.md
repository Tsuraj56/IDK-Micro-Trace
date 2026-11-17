# Microbiome Taxonomic & Functional Quality Report (Agent 4)

## 1. Taxonomic Profile Availability & Resolution (`Verified in actual files`)
- **Shotgun WGS Taxonomy (`MetaPhlAn2` & `MetaPhlAn3`):**
  - `MGX/2018-05-04/taxonomic_profiles.tsv.gz`: **1,638 samples**, **1,479 clade rows** (hierarchical `k__|p__|c__|o__|f__|g__|s__|t__`, relative abundance in `[0, 1]` or `[0, 100]`), **100% matched** (`1,638/1,638`) to `metadata[data_type == 'metagenomics']`.
  - `MGX/2018-05-04/taxonomic_profiles_3.tsv.gz`: **1,638 samples** (after stripping `_profile` suffix), **932 clade rows** (includes `UNKNOWN` row), **100% matched**.
- **16S Amplicon Taxonomy:**
  - `16S/2018-01-07/taxonomic_profiles.tsv.gz`: **178 samples**, **982 OTUs** (raw read counts + trailing `taxonomy` lineage column). Matches **100% of `biopsy_16S`** (`178/178`), **0% of `stool_16S`**. Since intestinal biopsies have no paired FC (`0/178`), biopsy 16S cannot be used for C1 future-FC prediction.
  - `HMP2_Pilot/16S/taxonomic_profiles.tsv`: **58 samples**, matches `58/180` of `stool_16S`. Thus, **shotgun MGX (`1,638` samples) is the primary taxonomic modality for MICRO-TRACE**.

| table_name | relative_path | exists | assay | tool_or_database | representation | n_samples_in_table | n_features_total | n_features_sampled_for_prev | unmapped_or_ungrouped_rows | stratified_rows | unstratified_rows | mean_feature_prevalence | median_feature_prevalence | zero_fraction | features_prev_ge_1pct | features_prev_ge_5pct | features_prev_ge_10pct | features_prev_ge_20pct | sample_sum_median | sample_sum_min | sample_sum_max | kingdom_rows | phylum_rows | class_rows | order_rows | family_rows | genus_rows | species_rows | strain_rows |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mgx_taxonomy_v2 | products/HMP2/MGX/2018-05-04/taxonomic_profiles.tsv.gz | True | MGX_WGS | MetaPhlAn2 | relative_abundance | 1638 | 1479 | 1479 | 0 | 1474 | 5 | 0.12 | 0.0104 | 0.88 | 742 | 464 | 347 | 283 | 7.9436 | 1.0 | 8.0 | 4 | 17 | 27 | 39 | 86 | 199 | 572 | 534 |
| mgx_taxonomy_v3 | products/HMP2/MGX/2018-05-04/taxonomic_profiles_3.tsv.gz | True | MGX_WGS | MetaPhlAn3 | relative_abundance_percent | 1638 | 932 | 932 | 1 | 928 | 4 | 0.1286 | 0.014 | 0.8714 | 504 | 312 | 247 | 201 | 700.0 | 100.0 | 700.0001 | 3 | 15 | 28 | 43 | 77 | 187 | 578 | 0 |
| s16_biopsy_taxonomy | products/HMP2/16S/2018-01-07/taxonomic_profiles.tsv.gz | True | 16S_amplicon_biopsy | 16S_OTU_pipeline | read_counts | 178 | 982 | 982 | 0 | 0 | 982 | 0.0834 | 0.0169 | 0.9166 | 635 | 291 | 184 | 118 | 16627.5 | 34.0 | 31781.0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| s16_pilot_stool_taxonomy | products/HMP2_Pilot/16S/taxonomic_profiles.tsv | True | 16S_amplicon_stool_pilot | 16S_OTU_pipeline | relative_abundance | 58 | 503 | 503 | 0 | 0 | 503 | 0.0627 | 0.0172 | 0.9373 | 503 | 166 | 86 | 33 | 12094.0 | 10.0 | 84308.0 | 503 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

## 2. Functional Profile Availability & Biological Quantity (`Verified in actual files`)
- **Shotgun WGS Functional Potential (`MGX` — DNA-level potential, NOT active expression):**
  - `pathabundance_relab.tsv.gz` (`HUMAnN2` MetaCyc pathways, relative abundance): **1,638 samples**, **10,884 rows** (558 unstratified community pathways + 10,326 taxon-stratified pathways).
  - `pathabundances_3.tsv.gz` (`HUMAnN3` MetaCyc pathways, copies per million `CPM`): **1,638 samples** (strip `_pathabundance_cpm`), **22,113 rows** (604 unstratified + 21,509 stratified).
  - `ecs_relab.tsv.gz` (`HUMAnN2` Level-4 EC enzymes, relative abundance): **1,638 samples**, **108,433 rows**.
  - `ecs_3.tsv.gz` (`HUMAnN3` Level-4 EC enzymes, `CPM`): **1,635 samples** (strip `_level4ec`), **167,854 rows**.
- **Metatranscriptomics (`MTX` — Active RNA Transcription, distinct biological measurement):**
  - `MTX/2017-12-14/pathabundance_relab.tsv.gz` & `ecs_relab.tsv.gz`: **735 samples** (`6,061` pathway rows; `70,711` EC rows). Must be analyzed separately from DNA functional potential (`MGX`).

| table_name | relative_path | exists | assay | tool_or_database | representation | n_samples_in_table | n_features_total | n_features_sampled_for_prev | unmapped_or_ungrouped_rows | stratified_rows | unstratified_rows | mean_feature_prevalence | median_feature_prevalence | zero_fraction | features_prev_ge_1pct | features_prev_ge_5pct | features_prev_ge_10pct | features_prev_ge_20pct | sample_sum_median | sample_sum_min | sample_sum_max | functional_quantity |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mgx_pathways_v2 | products/HMP2/MGX/2018-05-04/pathabundance_relab.tsv.gz | True | MGX_WGS | HUMAnN2_MetaCyc | relative_abundance | 1638 | 10884 | 5000 | 0 | 10418 | 466 | 0.1067 | 0.0092 | 0.8933 | 2429 | 1538 | 1176 | 938 | 0.6952 | 0.0 | 1.5375 | DNA_functional_potential |
| mgx_pathways_v3 | products/HMP2/MGX/2018-05-04/pathabundances_3.tsv.gz | True | MGX_WGS | HUMAnN3_MetaCyc | cpm | 1638 | 22113 | 5000 | 432 | 21635 | 478 | 0.0766 | 0.0092 | 0.9234 | 2397 | 1230 | 882 | 565 | 1697458.8301 | 0.0 | 2098807.9217 | DNA_functional_potential |
| mgx_ecs_v2 | products/HMP2/MGX/2018-05-04/ecs_relab.tsv.gz | True | MGX_WGS | HUMAnN2_EC | relative_abundance | 1638 | 108433 | 5000 | 0 | 106260 | 2173 | 0.1155 | 0.0122 | 0.8845 | 2612 | 1644 | 1283 | 1020 | 0.0955 | 0.0 | 0.1594 | DNA_functional_potential |
| mgx_ecs_v3 | products/HMP2/MGX/2018-05-04/ecs_3.tsv.gz | True | MGX_WGS | HUMAnN3_EC | cpm | 1635 | 167854 | 5000 | 523 | 165402 | 2452 | 0.0919 | 0.0116 | 0.9081 | 2601 | 1475 | 1078 | 740 | 1644246.3652 | 1044801.9766 | 1771879.8616 | DNA_functional_potential |
| mtx_pathways_v2 | products/HMP2/MTX/2017-12-14/pathabundance_relab.tsv.gz | True | MTX_RNAseq | HUMAnN2_MetaCyc | relative_abundance | 735 | 6061 | 5000 | 0 | 5631 | 430 | 0.0723 | 0.0082 | 0.9277 | 2322 | 1225 | 864 | 527 | 1.1196 | 0.0 | 1.6842 | RNA_active_expression |
| mtx_ecs_v2 | products/HMP2/MTX/2017-12-14/ecs_relab.tsv.gz | True | MTX_RNAseq | HUMAnN2_EC | relative_abundance | 735 | 70711 | 5000 | 0 | 68629 | 2082 | 0.0814 | 0.0122 | 0.9186 | 2660 | 1452 | 1010 | 646 | 0.2113 | 0.0358 | 0.7246 | RNA_active_expression |
