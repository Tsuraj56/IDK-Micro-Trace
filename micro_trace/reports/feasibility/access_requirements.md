# IBDMDB & Companion Datasets — Access Requirements & Provenance (Agent 1)

## 1. Access Classification Summary
- **Openly Accessible & Downloaded Locally (`Verified in actual files`):**
  - All 46 merged processed data products across all 11 HMP2 omics layers (1.45 GB total) hosted at `https://g-227ca.190ebd.75bc.data.globus.org/ibdmdb/`.
  - Includes master clinical/sample metadata (`hmp2_metadata_2018-08-20.csv`), WGS metagenomic taxonomy (`MetaPhlAn2` and `MetaPhlAn3`), WGS functional potential (`HUMAnN2` and `HUMAnN3` pathways and EC enzymes), metatranscriptomics (`MTX`), 16S amplicon profiles, fecal metabolomics (`MBX`), fecal proteomics (`MPX`), viromics (`MVX`), host biopsy transcriptomics (`HTX` count matrix), and serology (`ELISA`).
- **Controlled Access — Not Required for C1–C4 (`Verified in official metadata/documentation`):**
  - Human host raw genome/exome sequencing reads and identifiable patient records require NIH dbGaP authorization under study `phs001626.v1.p1`. MICRO-TRACE C1–C4 does not use raw host DNA sequences.
- **External Validation & Dietary Trial Datasets (`Not yet verified`):**
  - **IBSEN III:** Requires formal institutional data-transfer request to the IBSEN III study group. Contains 16S amplicon profiles, not shotgun metagenomic pathways.
  - **DINE-CD (ICPSR 38590):** Requires ICPSR terms-of-use agreement for patient-level trial tables; published trial summary statistics (Lewis et al., *Gastroenterology* 2021) are openly available for C4 evidence cards.

## 2. Correction of Local Dataset Guide Discrepancies
1. **Merged Microbiome Tables Exist (`Verified in actual files`):** The local file `HMP2_IBDMDB_Dataset_Guide.md` erroneously stated *"No single merged file on portal"* for MGX, MTX, 16S, and Viromics. Direct inspection of `ibdmdb.org/downloads/html/products_*.html` confirmed official merged tables for all modalities, which are now downloaded and verified.
2. **Local Guide File-Size & Filename Errors (`Verified in actual files`):**
   - `host_tx_counts.tsv.gz` is **10.51 MiB (11.02 MB)** on disk and on Globus, whereas the local guide listed 36.3 MB.
   - `host_tx_counts.biom.gz` is **25.78 MiB (27.03 MB)**, whereas the local guide listed 0.9 MB.
   - The serology file on the portal is `products/HMP2/Serology/2017-10-05/hmp2_serology_Compiled_ELISA_Data.tsv` (9 KB), whereas `download_hmp2.py` attempted to fetch a non-existent `HMP2_serology_Compiled_CLIA_lab_results_2.tsv`.
3. **Root `DataDictionary_Microbes.csv` Provenance (`Verified in actual files`):**
   - The file `DataDictionary_Microbes.csv` in the workspace root (1,979 binary indicator fields) is from a different processed binary microbiome indicator schema and does **not** match the column headers of `hmp2_metadata_2018-08-20.csv` or the HMP2 MetaPhlAn2/v3 clade strings. It must not be used as the HMP2 metadata dictionary.
