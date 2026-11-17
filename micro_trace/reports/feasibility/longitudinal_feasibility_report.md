# Longitudinal & Clinical Feasibility Report (Agent 3)

## 1. Participant & Visit Structure (`Verified in actual files`)
- **Total Unique Participants:** `131` (`CD`: `66`, `UC`: `38`, `nonIBD`: `27`).
- **Participants with Repeated FC (>=2 FC visits):** `97` participants.
- **Participants with Repeated Matched MGX + Same-Day FC (>=2 visits):** `90` total participants (`66` IBD [CD/UC] participants).

## 2. Fecal Calprotectin (FC) Distribution & Semantics
- **Dual Column Resolution:** `fecalcal` (1,883 rows) and `fecalcal_ng_ml` (121 rows) have zero overlap in any single row, but cross-checking across `site_sub_coll` proves they store the **exact same numerical quantity** (range `3.45` to `493.7`, standard ELISA `ug/g` stool range).
- **Extreme & Boundary Values Retained:** No outliers or boundary values were dropped. Values <=15 (`n_below_detection_15`) and >=250 (`n_above_250`) are explicitly flagged in `fc_summary.csv`.
| subset | n_total | n_valid_fc | n_missing_fc | missing_rate | n_participants_with_fc | min_fc | p10_fc | p25_fc | median_fc | mean_fc | std_fc | p75_fc | p90_fc | max_fc | n_zero_or_neg | n_below_detection_15 | n_above_250 | mean_log10_fc_plus_1 | std_log10_fc_plus_1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| all_metadata_rows | 5533 | 2004 | 3529 | 0.6378 | 98 | 3.45 | 13.5451 | 16.65 | 59.0822 | 118.7599 | 116.6847 | 217.4153 | 288.4 | 493.7 | 0 | 386 | 315 | 1.7964 | 0.5373 |
| unique_collection_events_all | 2302 | 636 | 1666 | 0.7237 | 98 | 3.45 | 13.554 | 16.9996 | 70.6171 | 122.5563 | 115.8554 | 218.1983 | 282.6585 | 493.7 | 0 | 116 | 95 | 1.8215 | 0.5333 |
| unique_collection_events_with_MGX | 1595 | 503 | 1092 | 0.6846 | 95 | 3.62 | 13.4952 | 16.7337 | 61.8487 | 118.7429 | 115.734 | 212.475 | 282.6734 | 493.7 | 0 | 96 | 77 | 1.8 | 0.5356 |
| MGX_events_CD | 736 | 241 | 495 | 0.6726 | 43 | 7.87 | 14.5107 | 24.8 | 136.72 | 138.6265 | 110.4891 | 225.2186 | 276.8321 | 401.4 | 0 | 26 | 41 | 1.9294 | 0.4929 |
| MGX_events_UC | 445 | 141 | 304 | 0.6831 | 28 | 4.18 | 12.716 | 17.4462 | 107.8 | 145.3684 | 133.9011 | 241.09 | 336.5243 | 493.7 | 0 | 25 | 34 | 1.8697 | 0.5845 |
| MGX_events_nonIBD | 414 | 121 | 293 | 0.7077 | 24 | 3.62 | 12.9246 | 14.1 | 16.9792 | 48.1134 | 63.581 | 45.6542 | 145.75 | 292.9457 | 0 | 45 | 2 | 1.4612 | 0.4008 |

## 3. Visit-Gap & Follow-Up Cadence Distribution
| event_subset | gap_type | n_intervals | min_days | p10_days | p25_days | median_days | mean_days | p75_days | p90_days | max_days |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| all_stool_events | consecutive_visit_gap_days | 1636 | 1.0 | 12.0 | 14.0 | 14.0 | 21.0 | 24.0 | 42.0 | 202.0 |
| all_stool_events | all_pairwise_future_gap_days | 14051 | 1.0 | 26.0 | 54.0 | 105.0 | 118.7 | 176.0 | 237.0 | 405.0 |
| mgx_events | consecutive_visit_gap_days | 1463 | 1.0 | 12.0 | 14.0 | 15.0 | 21.5 | 27.0 | 42.8 | 138.0 |
| mgx_events | all_pairwise_future_gap_days | 11967 | 1.0 | 26.0 | 52.5 | 100.0 | 116.0 | 169.0 | 227.0 | 405.0 |
| fc_events | consecutive_visit_gap_days | 538 | 2.0 | 14.0 | 16.0 | 54.0 | 48.2 | 69.0 | 84.0 | 202.0 |
| fc_events | all_pairwise_future_gap_days | 2107 | 2.0 | 28.0 | 55.0 | 103.0 | 123.5 | 186.0 | 257.0 | 366.0 |
| mgx_with_same_day_fc | consecutive_visit_gap_days | 408 | 2.0 | 14.0 | 28.0 | 56.0 | 54.5 | 70.0 | 88.6 | 182.0 |
| mgx_with_same_day_fc | all_pairwise_future_gap_days | 1315 | 2.0 | 28.0 | 56.0 | 105.0 | 121.3 | 181.5 | 246.6 | 366.0 |

## 4. Eligibility Flow & Pair Counts Across Candidate Horizons
- **Strict Causal Rule Enforced:** Every candidate pair satisfies `feature_timestamp <= index_date < future_date` (`elapsed_days >= 1`).
- **Observation vs. Participant Distinction:** All counts below report both unique participants (and IBD-only participants) and index visits / pair counts.
| step_order | step_name | n_unique_participants | n_ibd_participants_CD_UC | n_observations_or_events | n_eligible_pairs | notes |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | All metadata rows | 131 | 104 | 5533 | 0 | Raw metadata table across 11 omics layers |
| 2 | Unique collection events (site_sub_coll) | 131 | 104 | 2302 | 0 | Deduplicated physical sample collection events |
| 3 | Stool collection events with valid date | 131 | 104 | 1769 | 0 | 100% of stool events have valid date_of_receipt |
| 4 | Events with shotgun MGX (taxonomy + function) | 130 | 103 | 1595 | 0 | 1595 unique collection events (1638 MGX runs incl. replicates) |
| 5 | Events with valid Fecal Calprotectin (FC) | 98 | 74 | 636 | 0 | Combined fecalcal and fecalcal_ng_ml |
| 6 | Events with BOTH MGX and same-event FC | 95 | 71 | 503 | 0 | Exact collection-event match (0-day window) |
| 7 | Participants with >=2 matched (MGX + same-event FC) visits | 90 | 66 | 498 | 0 | Minimum requirement for longitudinal baseline+future pair |
| 8 | All causal pairs: Index MGX -> Future FC (t_future > t_index) | 95 | 71 | 1201 | 3867 | Unrestricted future horizon (>0 days), index FC optional |
| 9 | Causal pairs with Same-Day Index FC + MGX -> Future FC | 93 | 69 | 445 | 1671 | Supports all model arms M0-M5 with 0-day index FC match |
| 10 | Consecutive pairs only (Same-Day Index FC + MGX -> Next Future FC) | 93 | 69 | 445 | 445 | Exactly 1 future target per eligible index visit |
| 11 | Same-Day Index FC + MGX pairs in horizon 'short_7_21d' (7-21d) | 36 | 31 | 111 | 113 | IBD-only (CD/UC) pairs=101 across 31 IBD participants |
| 12 | Same-Day Index FC + MGX pairs in horizon 'biweekly_14_35d' (14-35d) | 44 | 39 | 128 | 190 | IBD-only (CD/UC) pairs=175 across 39 IBD participants |
| 13 | Same-Day Index FC + MGX pairs in horizon 'monthly_14_56d' (14-56d) | 84 | 63 | 246 | 415 | IBD-only (CD/UC) pairs=365 across 63 IBD participants |
| 14 | Same-Day Index FC + MGX pairs in horizon 'protocol_30_90d' (30-90d) | 88 | 65 | 396 | 545 | IBD-only (CD/UC) pairs=453 across 65 IBD participants |
| 15 | Same-Day Index FC + MGX pairs in horizon 'any_future_gt0d' (1-400d) | 93 | 69 | 445 | 1671 | IBD-only (CD/UC) pairs=1358 across 69 IBD participants |

## 5. Participant-Grouped Split Preview (`split_audit.csv`)
- **Zero Overlap Verified:** `split_participant_overlap = 0`.
| partition | diagnosis | n_participants | total_all_pairs | total_next_visit_pairs |
| --- | --- | --- | --- | --- |
| calibration | CD | 6 | 165 | 36 |
| calibration | UC | 4 | 79 | 21 |
| calibration | nonIBD | 4 | 47 | 16 |
| test | CD | 7 | 148 | 34 |
| test | UC | 4 | 36 | 12 |
| test | nonIBD | 3 | 30 | 11 |
| train | CD | 29 | 593 | 146 |
| train | UC | 19 | 337 | 92 |
| train | nonIBD | 17 | 236 | 77 |
