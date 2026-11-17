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


## 4. Mapped Feature Availability Summary
| column_index | column_name | mapped_role | dtype | non_null_total | missing_frac_total | non_null_mgx | missing_frac_mgx | n_unique |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | Project | identifiers.project | object | 5533 | 0.0 | 1638 | 0.0 | 5490 |
| 1 | External ID | identifiers.sample_id | object | 5533 | 0.0 | 1638 | 0.0 | 2892 |
| 2 | Participant ID | identifiers.participant_id | object | 5533 | 0.0 | 1638 | 0.0 | 131 |
| 3 | site_sub_coll | identifiers.collection_event_id | object | 5533 | 0.0 | 1638 | 0.0 | 2302 |
| 4 | data_type | identifiers.data_type | object | 5533 | 0.0 | 1638 | 0.0 | 11 |
| 5 | week_num | identifiers.week_num | float64 | 5273 | 0.047 | 1638 | 0.0 | 61 |
| 6 | date_of_receipt | dates.receipt_date | object | 4352 | 0.2134 | 1638 | 0.0 | 462 |
| 7 | interval_days | identifiers.interval_days | float64 | 4352 | 0.2134 | 1638 | 0.0 | 61 |
| 8 | visit_num | identifiers.visit_id | int64 | 5533 | 0.0 | 1638 | 0.0 | 31 |
| 10 | PDO Number | qc_variables.pdo_number | object | 3062 | 0.4466 | 1617 | 0.0128 | 56 |
| 13 | LCSET | qc_variables.lcset | object | 400 | 0.9277 | 322 | 0.8034 | 4 |
| 17 | reads_raw | qc_variables.reads_raw | float64 | 3176 | 0.426 | 1638 | 0.0 | 3175 |
| 18 | reads_filtered | qc_variables.reads_filtered | float64 | 3176 | 0.426 | 1638 | 0.0 | 3174 |
| 19 | reads_qc_fail | qc_variables.reads_qc_fail | float64 | 3176 | 0.426 | 1638 | 0.0 | 3149 |
| 20 | reads_human | qc_variables.reads_human | float64 | 2473 | 0.553 | 1638 | 0.0 | 2459 |
| 21 | reads_ribosomal | qc_variables.reads_ribosomal | float64 | 835 | 0.8491 | 0 | 1.0 | 834 |
| 22 | reads_viral | qc_variables.reads_viral | float64 | 703 | 0.8729 | 0 | 1.0 | 547 |
| 27 | site_name | identifiers.site_name | object | 5533 | 0.0 | 1638 | 0.0 | 5 |
| 28 | Age at diagnosis | clinical.age_diagnosis | float64 | 4149 | 0.2501 | 1198 | 0.2686 | 39 |
| 69 | consent_age | clinical.age_consent | float64 | 5473 | 0.0108 | 1619 | 0.0116 | 50 |
| 70 | diagnosis | clinical.disease_subtype | object | 5533 | 0.0 | 1638 | 0.0 | 3 |
| 71 | Soft drinks, tea or coffee with sugar (corn syrup, maple syrup, cane sugar, etc) | dietary_variables | object | 4285 | 0.2256 | 1606 | 0.0195 | 5 |
| 72 | Diet soft drinks, tea or coffee with sugar (Stevia, Equal, Splenda etc) | dietary_variables | object | 4293 | 0.2241 | 1609 | 0.0177 | 5 |
| 73 | Fruit juice (orange, apple, cranberry, prune etc.) | dietary_variables | object | 4306 | 0.2218 | 1611 | 0.0165 | 5 |
| 74 | Water | dietary_variables | object | 4307 | 0.2216 | 1612 | 0.0159 | 5 |
| 75 | Alcohol (beer, brandy, spirits, hard liquor, wine, aperitif, etc.) | dietary_variables | object | 4303 | 0.2223 | 1611 | 0.0165 | 5 |
| 76 | Yogurt or other foods containing active bacterial cultures (kefir, sauerkraut) | dietary_variables | object | 4290 | 0.2247 | 1608 | 0.0183 | 5 |
| 77 | Dairy (milk, cream, ice cream, cheese, cream cheese) | dietary_variables | object | 4293 | 0.2241 | 1609 | 0.0177 | 5 |
| 78 | Probiotic | dietary_variables | object | 4286 | 0.2254 | 1607 | 0.0189 | 5 |
| 79 | Fruits (no juice) (Apples, raisins, bananas, oranges, strawberries, blueberries | dietary_variables | object | 4287 | 0.2252 | 1604 | 0.0208 | 5 |
| 80 | Vegetables (salad, tomatoes, onions, greens, carrots, peppers, green beans, etc) | dietary_variables | object | 4281 | 0.2263 | 1607 | 0.0189 | 5 |
| 82 | Beans (tofu, soy, soy burgers, lentils, Mexican beans, lima beans etc) | dietary_variables | object | 4283 | 0.2259 | 1603 | 0.0214 | 5 |
| 83 | Whole grains (wheat, oats, brown rice, rye, quinoa, wheat bread, wheat pasta) | dietary_variables | object | 4292 | 0.2243 | 1610 | 0.0171 | 5 |
| 84 | Starch (white rice, bread, pizza, potatoes, yams, cereals, pancakes, etc.) | dietary_variables | object | 4289 | 0.2248 | 1606 | 0.0195 | 5 |
| 85 | Eggs | dietary_variables | object | 4267 | 0.2288 | 1600 | 0.0232 | 5 |
| 86 | Processed meat (other red or white meat such as lunch meat, ham, salami, bologna | dietary_variables | object | 4295 | 0.2237 | 1608 | 0.0183 | 5 |
| 87 | Red meat (beef, hamburger, pork, lamb) | dietary_variables | object | 4297 | 0.2234 | 1610 | 0.0171 | 5 |
| 88 | White meat (chicken, turkey, etc.) | dietary_variables | object | 4281 | 0.2263 | 1604 | 0.0208 | 5 |
| 89 | Shellfish (shrimp, lobster, scallops, etc.) | dietary_variables | object | 4289 | 0.2248 | 1608 | 0.0183 | 5 |
| 90 | Fish (fish nuggets, breaded fish, fish cakes, salmon, tuna, etc.) | dietary_variables | object | 4288 | 0.225 | 1607 | 0.0189 | 5 |
| 91 | Sweets (pies, jam, chocolate, cake, cookies, etc.) | dietary_variables | object | 4295 | 0.2237 | 1609 | 0.0177 | 5 |
| 92 | Antibiotics | medications.antibiotics | object | 5533 | 0.0 | 1638 | 0.0 | 2 |
| 94 | Immunosuppressants (e.g. oral corticosteroids) | medications.immunosuppressants | object | 5533 | 0.0 | 1638 | 0.0 | 2 |
| 100 | Tea or coffee no sugar and no sugar replacement | dietary_variables | object | 4300 | 0.2228 | 1607 | 0.0189 | 5 |
| 186 | Has the subject used any antibiotics since the last visit? | medications.antibiotics_since_last | object | 324 | 0.9414 | 0 | 1.0 | 2 |
| 187 | fecalcal_ng_ml | outcomes.fc_secondary | float64 | 121 | 0.9781 | 66 | 0.9597 | 105 |
| 225 | General wellbeing | clinical.wellbeing | object | 2224 | 0.598 | 719 | 0.5611 | 5 |
| 226 | Abdominal pain | clinical.abdominal_pain | object | 1180 | 0.7867 | 379 | 0.7686 | 3 |
| 230 | hbi | clinical.hbi | float64 | 2145 | 0.6123 | 693 | 0.5769 | 19 |
| 333 | Prednisone | medications.prednisone | object | 505 | 0.9087 | 0 | 1.0 | 4 |
| 357 | Remicade (Infliximab) | medications.remicade | object | 505 | 0.9087 | 0 | 1.0 | 4 |
| 360 | Humira (Adalimumab) | medications.humira | object | 505 | 0.9087 | 0 | 1.0 | 3 |
| 417 | race | clinical.race | object | 5533 | 0.0 | 1638 | 0.0 | 5 |
| 425 | Bowel frequency during the day | clinical.bowel_freq | object | 1338 | 0.7582 | 453 | 0.7234 | 4 |
| 430 | sccai | clinical.sccai | float64 | 1335 | 0.7587 | 452 | 0.7241 | 12 |
| 456 | SES-CD Score | clinical.ses_cd | float64 | 481 | 0.9131 | 0 | 1.0 | 17 |
| 458 | sex | clinical.sex | object | 5533 | 0.0 | 1638 | 0.0 | 2 |
| 469 | SIBDQ Score | clinical.sibdq | float64 | 110 | 0.9801 | 0 | 1.0 | 30 |
| 476 | Tube B:Fecal Calprotectin | outcomes.fc_tube_id | object | 2206 | 0.6013 | 610 | 0.6276 | 751 |
| 477 | fecalcal | outcomes.fc_primary | float64 | 1883 | 0.6597 | 456 | 0.7216 | 624 |