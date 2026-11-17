# Join Integrity & Schema Audit Report (Agent 2)

## 1. Identifier Hierarchy & Join Rules (`Verified in actual files`)
1. **`Participant ID` (131 unique subjects):** Top-level subject identifier (`C3001`, `H4001`, `M2001`, `P6001`, `E5001`). Zero missing values across all 5,533 rows.
2. **`site_sub_coll` (2,302 unique collection events):** True physical sample-collection event key (`{ParticipantID}C{visit}` for stool/blood, `{ParticipantID}CSC{num}` for biopsy).
   - **Proven Invariant:** Fecal Calprotectin (`fc_combined`) has **0 conflicts** (`max - min < 1e-4`) within any `site_sub_coll`.
   - **Trap Flagged:** Joining on `(Participant ID, week_num)` has **1 real conflict** (`C3007` at `week_num=0.0` has two distinct collections 6 days apart: `C3007C1` on `2014-06-06` with FC=`41.21` and `C3007C2` on `2014-06-12` with FC=`58.55`). Therefore, multi-omics linkage **must** use `site_sub_coll` (or `(Participant ID, date_of_receipt)`), never `(Participant ID, week_num)`.
3. **`External ID` (2,892 unique across 5,533 rows):** Assay run / sample file identifier.
   - **Many-to-Many Join Hazard:** Because the same physical stool aliquot (`CSM5FZ3N`) is reused across `metabolomics`, `proteomics`, and `stool_16S` rows, `External ID` is **not** unique across the raw metadata table (3,949 rows share an `External ID` with another omics layer).
   - **Safe Join Key:** Within any single stool omics layer (`(External ID, data_type)`), `External ID` is **100% unique** (`1,638/1,638` for `metagenomics`, `835/835` for `metatranscriptomics`, `546/546` for `metabolomics`, `450/450` for `proteomics`, `180/180` for `stool_16S`, `703/703` for `viromics`).
   - **Technical Replicates within `site_sub_coll`:** In `metagenomics`, 1,638 `External ID` runs map to 1,595 unique `site_sub_coll` events because 43 runs are technical replicates (`_TR` suffix) or pilot re-runs (`_P` suffix). Our longitudinal pipeline deduplicates to primary `site_sub_coll` collection events before pair construction to prevent replicate inflation.

## 2. Identifier & Join Cardinality Audit Table
| source_table | modality | n_rows | n_unique_participants | n_unique_external_ids | n_unique_collection_events | n_valid_dates | is_external_id_unique | matched_in_metadata_dt | unmatched_in_metadata_dt | join_cardinality |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| metadata[biopsy_16S] | biopsy_16S | 178 | 81 | 178 | 178 | 0 | True | 178 | 0 | 1:1 |
| metadata[host_genome] | host_genome | 92 | 92 | 92 | 92 | 0 | True | 92 | 0 | 1:1 |
| metadata[host_transcriptomics] | host_transcriptomics | 252 | 90 | 252 | 249 | 0 | True | 252 | 0 | 1:1 |
| metadata[metabolomics] | metabolomics | 546 | 106 | 546 | 546 | 546 | True | 546 | 0 | 1:1 |
| metadata[metagenomics] | metagenomics | 1638 | 130 | 1638 | 1595 | 1638 | True | 1638 | 0 | 1:1 |
| metadata[metatranscriptomics] | metatranscriptomics | 835 | 109 | 835 | 815 | 835 | True | 835 | 0 | 1:1 |
| metadata[methylome] | methylome | 449 | 94 | 449 | 449 | 0 | True | 449 | 0 | 1:1 |
| metadata[proteomics] | proteomics | 450 | 89 | 450 | 450 | 450 | True | 450 | 0 | 1:1 |
| metadata[serology] | serology | 210 | 67 | 210 | 210 | 0 | True | 210 | 0 | 1:1 |
| metadata[stool_16S] | stool_16S | 180 | 36 | 180 | 180 | 180 | True | 180 | 0 | 1:1 |
| metadata[viromics] | viromics | 703 | 105 | 703 | 703 | 703 | True | 703 | 0 | 1:1 |
| mgx_taxonomy_v2 | taxonomy:MetaPhlAn2 | 1479 | 130 | 1638 | 1595 | 1638 | True | 1638 | 0 | 1:1 (on External ID + data_type) |
| mgx_taxonomy_v3 | taxonomy:MetaPhlAn3 | 932 | 130 | 1638 | 1595 | 1638 | True | 1638 | 0 | 1:1 (on External ID + data_type) |
| s16_biopsy_taxonomy | taxonomy:16S_OTU_pipeline | 982 | 81 | 178 | 178 | 0 | True | 178 | 0 | 1:1 (on External ID + data_type) |
| s16_pilot_stool_taxonomy | taxonomy:16S_OTU_pipeline | 503 | 22 | 58 | 58 | 58 | True | 58 | 0 | 1:1 (on External ID + data_type) |
| mgx_pathways_v2 | function_potential:HUMAnN2_MetaCyc | 10884 | 130 | 1638 | 1595 | 1638 | True | 1638 | 0 | 1:1 (on External ID + data_type) |
| mgx_pathways_v3 | function_potential:HUMAnN3_MetaCyc | 22113 | 130 | 1638 | 1595 | 1638 | True | 1638 | 0 | 1:1 (on External ID + data_type) |
| mgx_ecs_v2 | function_potential:HUMAnN2_EC | 108433 | 130 | 1638 | 1595 | 1638 | True | 1638 | 0 | 1:1 (on External ID + data_type) |
| mgx_ecs_v3 | function_potential:HUMAnN3_EC | 167854 | 129 | 1635 | 1593 | 1635 | True | 1635 | 0 | 1:1 (on External ID + data_type) |
| mtx_pathways_v2 | function_expression:HUMAnN2_MetaCyc | 6061 | 104 | 735 | 716 | 735 | True | 735 | 0 | 1:1 (on External ID + data_type) |
| mtx_ecs_v2 | function_expression:HUMAnN2_EC | 70711 | 104 | 735 | 716 | 735 | True | 735 | 0 | 1:1 (on External ID + data_type) |

## 3. Logged Data Quality Issues (38 total)
| severity | category | field | description | affected_rows |
| --- | --- | --- | --- | --- |
| ERROR | schema_mapping | Cimzia (Certolizumab) | Configured column 'Cimzia (Certolizumab)' (medications.cimzia) not found in metadata table. | 5533 |
| ERROR | schema_mapping | Azathioprine | Configured column 'Azathioprine' (medications.azathioprine) not found in metadata table. | 5533 |
| ERROR | schema_mapping | Mercaptopurine | Configured column 'Mercaptopurine' (medications.mercaptopurine) not found in metadata table. | 5533 |
| WARNING | empty_column | Delta | Column 'Delta' is 100% missing (0/5533 non-null). | 5533 |
| WARNING | empty_column | FFQ/YAQ results: | Column 'FFQ/YAQ results:' is 100% missing (0/5533 non-null). | 5533 |
| WARNING | empty_column | Other inflamed cell biopsy: | Column 'Other inflamed cell biopsy:' is 100% missing (0/5533 non-null). | 5533 |
| WARNING | empty_column | Subject is unwilling or unable to provide biopsy, blood, and stool samples: | Column 'Subject is unwilling or unable to provide biopsy, blood, and stool samples:' is 100% missing (0/5533 non-null). | 5533 |
| WARNING | empty_column | Subject has an immune mediated disease (i.e. rheumatoid arthritis, lupus, T1DM): | Column 'Subject has an immune mediated disease (i.e. rheumatoid arthritis, lupus, T1DM):' is 100% missing (0/5533 non-null). | 5533 |
| WARNING | empty_column | Subject is pregnant: | Column 'Subject is pregnant:' is 100% missing (0/5533 non-null). | 5533 |
| WARNING | empty_column | Subject has a known bleeding disorder: | Column 'Subject has a known bleeding disorder:' is 100% missing (0/5533 non-null). | 5533 |
| WARNING | empty_column | Subject has an acute gastrointestinal infection: | Column 'Subject has an acute gastrointestinal infection:' is 100% missing (0/5533 non-null). | 5533 |
| WARNING | empty_column | Subject is actively being treated for a malignancy with chemotherapy: | Column 'Subject is actively being treated for a malignancy with chemotherapy:' is 100% missing (0/5533 non-null). | 5533 |
| WARNING | empty_column | Subject has been diagnosed with indeterminate colitis: | Column 'Subject has been diagnosed with indeterminate colitis:' is 100% missing (0/5533 non-null). | 5533 |
| WARNING | empty_column | Subject has a stoma or pouch: | Column 'Subject has a stoma or pouch:' is 100% missing (0/5533 non-null). | 5533 |
| WARNING | empty_column | Subject has an acute gastrointestinal illness: | Column 'Subject has an acute gastrointestinal illness:' is 100% missing (0/5533 non-null). | 5533 |
| WARNING | empty_column | Subject has an acute gastrointestinal infection or IBD: | Column 'Subject has an acute gastrointestinal infection or IBD:' is 100% missing (0/5533 non-null). | 5533 |
| WARNING | empty_column | Subject is 18 years of age or older: | Column 'Subject is 18 years of age or older:' is 100% missing (0/5533 non-null). | 5533 |
| WARNING | empty_column | Subject is 17 years of age or younger: | Column 'Subject is 17 years of age or younger:' is 100% missing (0/5533 non-null). | 5533 |
| WARNING | empty_column | Subject has a confirmed diagnosis of IBD: | Column 'Subject has a confirmed diagnosis of IBD:' is 100% missing (0/5533 non-null). | 5533 |
| WARNING | empty_column | Subject was diagnosed within the last 6 months: | Column 'Subject was diagnosed within the last 6 months:' is 100% missing (0/5533 non-null). | 5533 |
| WARNING | empty_column | Subject is NOT on an anti-TNF inhibitor: | Column 'Subject is NOT on an anti-TNF inhibitor:' is 100% missing (0/5533 non-null). | 5533 |
| WARNING | empty_column | Subject has had IBD for 5 years or more: | Column 'Subject has had IBD for 5 years or more:' is 100% missing (0/5533 non-null). | 5533 |
| WARNING | empty_column | Reason for stopping lomotil: | Column 'Reason for stopping lomotil:' is 100% missing (0/5533 non-null). | 5533 |
| WARNING | empty_column | Reason for stopping FOS: | Column 'Reason for stopping FOS:' is 100% missing (0/5533 non-null). | 5533 |
| WARNING | empty_column | Reason for stopping Tysabri: | Column 'Reason for stopping Tysabri:' is 100% missing (0/5533 non-null). | 5533 |
| WARNING | empty_column | Duration of Tysabri use (months): | Column 'Duration of Tysabri use (months):' is 100% missing (0/5533 non-null). | 5533 |
| WARNING | empty_column | Reason for stopping: | Column 'Reason for stopping:' is 100% missing (0/5533 non-null). | 5533 |
| WARNING | empty_column | Reason for stopping Apriso: | Column 'Reason for stopping Apriso:' is 100% missing (0/5533 non-null). | 5533 |
| WARNING | empty_column | Type of surgery: | Column 'Type of surgery:' is 100% missing (0/5533 non-null). | 5533 |
| WARNING | empty_column | Type of surgery:.1 | Column 'Type of surgery:.1' is 100% missing (0/5533 non-null). | 5533 |
| WARNING | empty_column | Screening Status | Column 'Screening Status' is 100% missing (0/5533 non-null). | 5533 |
| WARNING | empty_column | stool_id | Column 'stool_id' is 100% missing (0/5533 non-null). | 5533 |
| ERROR | missing_identifier | week_num | Identifier column 'week_num' has 260 missing values. | 260 |
| WARNING | many_to_many_join_risk | External ID | External ID is NOT globally unique in metadata (2892 unique across 5533 rows; 4008 rows share an External ID across multiple omics layers). Joining an omics table to unfiltered metadata on External ID alone causes many-to-many row multiplication. Must filter metadata by data_type (or deduplicate to collection_event_id) before joining. | 4008 |
| WARNING | technical_replicates_within_collection | site_sub_coll+data_type | Found 130 rows sharing the same (site_sub_coll, data_type) — e.g., technical replicates (_TR / _P) or multiple biopsy locations (Rectum vs Ileum). | 130 |
| WARNING | fc_dual_columns_and_unit_label | fecalcal / fecalcal_ng_ml | FC is split across two mutually exclusive columns: 'fecalcal' (1883 rows) and 'fecalcal_ng_ml' (121 rows), overlap=0. Within identical collection events (site_sub_coll), their numeric values match to 5 decimal places (max diff < 1e-4), proving both store the exact same scale (~3.45 to 493.7, standard ELISA ug/g range despite 'ng_ml' label). | 2004 |
| WARNING | missing_dates | date_of_receipt | 'date_of_receipt' is missing in 1181/5533 metadata rows (all in biopsy_16S, host_transcriptomics, host_genome, methylome, serology; 0 missing in stool omics layers). | 1181 |
| WARNING | same_week_multiple_collections | Participant ID+week_num | Found 1 (Participant ID, week_num) pair(s) with multiple distinct collection events and different FC values (e.g. C3007 week 0 has C3007C1 on 2014-06-06 FC=41.21 and C3007C2 on 2014-06-12 FC=58.55). Join on site_sub_coll (or exact date_of_receipt), never on (Participant ID, week_num) alone. | 1 |
