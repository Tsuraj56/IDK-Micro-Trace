# Dataset Documentation & Provenance Report (Agent 1)

## 1. Authoritative Repository & Protocol Mapping
- **Cohort:** Integrative Human Microbiome Project (iHMP / HMP2) Inflammatory Bowel Disease Multi'omics Database (IBDMDB), Lloyd-Price et al., *Nature* 569:655–662 (2019) [`Verified in actual files` & `Reported in a publication`].
- **Collection Protocol:** Up to 1 year of biweekly home stool collection (shipped via FedEx to Broad Institute / MGH) plus baseline/follow-up clinic visits and colonoscopies across 5 clinical sites (Cedars-Sinai, Emory, Cincinnati Children's, MGH, MGH Pediatrics) [`Verified in official metadata/documentation`].

## 2. Mapping of Documented Resources to Planned Experiments (C1–C4)
| Experiment | Required Data | Authoritative HMP2 Resource | Verification Status |
|---|---|---|---|
| **C1 — Future FC** | Longitudinal FC + dates + clinical baseline + microbiome link | `metadata/hmp2_metadata_2018-08-20.csv` (`fecalcal`, `fecalcal_ng_ml`, `date_of_receipt`, `site_sub_coll`) | **Verified in actual files** (2,004 rows / 636 collection events with FC) |
| **C2 — Taxonomy** | Species/genus profiles linked to visits | `MGX/2018-05-04/taxonomic_profiles.tsv.gz` (MetaPhlAn2, 1,638 samples) & `taxonomic_profiles_3.tsv.gz` (MetaPhlAn3, 1,638 samples) | **Verified in actual files** (100% matched to metadata) |
| **C2 — Function** | Shotgun pathway & enzyme profiles (DNA potential) | `MGX/2018-05-04/pathabundance_relab.tsv.gz`, `pathabundances_3.tsv.gz`, `ecs_relab.tsv.gz`, `ecs_3.tsv.gz` | **Verified in actual files** (HUMAnN2 & HUMAnN3, 100% matched) |
| **C3 — Uncertainty** | Repeated participant measurements & subgroup covariates | `Participant ID` (131 subjects), `diagnosis` (CD/UC/nonIBD), `site_name`, `week_num` | **Verified in actual files** (89 subjects with >=2 matched MGX+FC visits) |
| **C4 — Dietary Support** | Clinical context, medications, dietary recall, external guidelines | 21 food-frequency columns + 22 medication columns in `hmp2_metadata_2018-08-20.csv` + AGA/ECCO/BDA & DINE-CD literature | **Verified in actual files** (metadata) / **Reported in publication** (guidelines) |

## 3. Source Registry
| source_id | name | url | access_tier | verification_status | role |
| --- | --- | --- | --- | --- | --- |
| SRC-01 | IBDMDB Official Data Portal | https://ibdmdb.org/downloads/ | Open Public | Verified in actual files & HTTP index | Primary multi-omics product repository |
| SRC-02 | Globus HMP2 Endpoint | https://g-227ca.190ebd.75bc.data.globus.org/ibdmdb/ | Open Public | Verified in actual files (46 merged tables downloaded) | Direct file server for HMP2 products |
| SRC-03 | Lloyd-Price et al. (2019) Nature | https://doi.org/10.1038/s41586-019-1237-9 | Open Publication | Reported in publication & verified against metadata | Primary cohort publication (iHMP IBDMDB) |
| SRC-04 | NCBI BioProject PRJNA398089 | https://www.ncbi.nlm.nih.gov/bioproject/PRJNA398089 | Open Public | Verified in official metadata/documentation | Raw microbial sequencing reads |
| SRC-05 | NCBI dbGaP phs001626.v1.p1 | https://www.ncbi.nlm.nih.gov/projects/gap/cgi-bin/study.cgi?study_id=phs001626.v1.p1 | Controlled Access (NIH dbGaP) | Verified in official metadata/documentation | Host human WGS/exome raw reads (not required for C1-C4) |
| SRC-06 | IBSEN III Study | https://pmc.ncbi.nlm.nih.gov/articles/PMC13533192/ | Controlled / Institutional Request | Not yet verified (External cohort) | Candidate external validation cohort (16S + FC) |
| SRC-07 | DINE-CD Trial (ICPSR 38590) | https://www.icpsr.umich.edu/web/pcodr/studies/38590 | Controlled / Data Use Agreement | Not yet verified (External trial) | Dietary RCT evidence (SCD vs Mediterranean in CD) |
| SRC-08 | AGA / ECCO / BDA IBD Dietary Guidelines | https://gastro.org/clinical-guidance/diet-and-nutritional-therapies-in-patients-with-ibd/ | Open Clinical Guidance | Verified in official documentation | C4 dietary-pattern clinical evidence base |
