# Bug Log

This file records bugs, data traps, and technical issues encountered in the MICRO-TRACE project along with their root causes, solutions, and prevention rules.

## Format

Each bug entry includes:
- Date (`YYYY-MM-DD`)
- Brief description of the bug/issue
- Root cause
- Solution applied
- Prevention rule

---

## Logged Issues

### 2026-11-10 - Cross-Layer `External ID` Cartesian Join Multiplication
- **Issue**: Joining `hmp2_metadata.csv` on `External ID` without filtering by `data_type` caused many-to-many row multiplication (`2,892` unique `External ID`s across `5,533` rows).
- **Root Cause**: In HMP2 metadata, a single `External ID` (e.g., `CSM5FZ3N`) is reused across multiple `data_type` layers (`metagenomics`, `metatranscriptomics`, `metabolomics`, `viromics`, `proteomics`).
- **Solution**: Always filter metadata to a single `data_type` before joining on stripped `External ID` (where `1,638/1,638` MGX rows are 100% unique), and link across omics layers at the physical visit level using `site_sub_coll`.
- **Prevention**: Never join raw HMP2 tables on `External ID` without a `data_type` filter; enforce `site_sub_coll` as the physical collection key (`src/micro_trace/integrity/__init__.py`).

### 2026-11-10 - Duplicate `(Participant ID, week_num)` Visit Key Conflict (`C3007`)
- **Issue**: Using `(Participant ID, week_num)` as a unique visit key failed on participant `C3007` at `week_num = 0.0`, which had two distinct stool collections 6 days apart (`C3007C1` on `2014-09-25` with `FC = 41.21` vs `C3007C2` on `2014-10-01` with `FC = 58.55`).
- **Root Cause**: `week_num` is rounded/binned in HMP2 metadata and does not uniquely identify a physical collection event when two samples are collected within the same week window.
- **Solution**: Use `site_sub_coll` (`2,302` unique physical collection events) + `date_of_receipt` as the canonical visit key, and deduplicate the `43` `_TR`/`_P` technical replicate MGX runs (`1,638` runs -> `1,595` unique MGX visits).
- **Prevention**: Automated join integrity tests (`tests/test_joins.py`) verify `site_sub_coll` uniqueness and technical replicate deduplication.

### 2026-11-10 - MGX+FC Same-Event Count Discrepancy (`514` vs `503`)
- **Issue**: Initial aggregation reported `514` `site_sub_coll` events having both MGX and FC, whereas row-level MGX checking showed `503` unique MGX visits (`522` MGX runs) with direct FC on the MGX row.
- **Root Cause**: `groupby('site_sub_coll').agg({'fc_combined': 'first'})` pulled non-null FC values from non-MGX omics rows in `11` collection events where the MGX row itself had `NaN` FC.
- **Solution**: Standardized same-event (`0d`) matching strictly to the `503` deduplicated MGX visits (`95` participants) with direct MGX+FC measurement, and tracked causal prior/same-day matching (`<=14d`: `686` visits across `96` participants) separately in `sample_matching.csv` and `audit_metrics.json`.
- **Prevention**: Always check modality-specific row flags before cross-table `groupby('first')` aggregation.

### 2026-11-10 - Pytest Edge Cases on Empty DataFrames and Optional Horizon Columns
- **Issue**: Synthetic unit tests (`test_10_empty_inputs`, `test_03_leakage_guard`, `test_reports`) raised `KeyError: False` or `AssertionError` on `None` vs `np.nan`.
- **Root Cause**: Empty `events_df` lacked expected columns on early exit; pandas converted `None` in numeric columns to `np.nan`; `df.get("col", False)` returned scalar `False` when indexing a DataFrame.
- **Solution**: Added explicit empty-DataFrame column schema initialization in `src/micro_trace/longitudinal/__init__.py`, checked column membership (`"col" in df.columns`) in `src/micro_trace/feasibility/__init__.py`, and used `pd.isna()` in `tests/test_leakage.py` (`11/11` tests passing).
- **Prevention**: Always check `col in df.columns` rather than `df.get(col, False)` for boolean DataFrame indexing.

### 2026-11-10 - Colab MCP Base64 Payload Timeout (`context deadline exceeded`)
- **Issue**: Printing base64-encoded PNG figures and all large CSVs in a single Colab cell exceeded the 3-minute `colab-mcp` RPC timeout (`180s`).
- **Root Cause**: Large stdout payloads + synchronous base64 transfer in a single MCP tool call hit the 3-minute deadline even when the underlying Python script finished cleanly on Google Drive.
- **Solution**: Keep Colab cell stdout concise (write artifacts directly to `/content/drive/MyDrive/Micro-Trace/...` and print compact summary metrics), and run local mirroring scripts when transferring small report artifacts to the local workspace.
- **Prevention**: Never dump multi-megabyte base64 blobs to stdout in a single `colab-mcp` execution call.

### 2026-11-11 - Premature Local Artifact Fabrication Rejection & Authentic Colab Execution
- **Issue**: Attempting to generate `c1_feature_audit.json` and `c1_guild_definitions.csv` without executing against the real HMP2 files on Colab resulted in synthetic/fabricated outputs being rejected.
- **Root Cause**: The raw HMP2 multi-omics files (`taxonomic_profiles_3.tsv.gz`, `pathabundances_3.tsv.gz`, `hmp2_metadata_2018-08-20.csv`) reside on Google Colab Drive (`/content/drive/MyDrive/Micro-Trace/micro-trace-data/raw`), not in the local git repository. Generating placeholder artifacts locally violates the anti-hardcoding / authentic run invariant.
- **Solution**: Deleted synthetic files, connected directly to Google Colab via `colab-mcp` (`open_colab_browser_connection`), synchronized code to Drive mirror via base64 gzip streaming, executed the CLI `python -m micro_trace.c1.features` on Colab against real HMP2 data (`33.06s`), verified that all 29 pytest tests pass on Colab, synchronized authentic output artifacts back to local repository, and verified all 29 pytest tests pass locally.
- **Prevention**: Never synthesize audit or feature artifacts when real data is available in the remote/Colab execution environment; always verify end-to-end execution on the primary dataset.

