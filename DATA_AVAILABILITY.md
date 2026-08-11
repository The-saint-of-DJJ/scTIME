# Data Availability and Acquisition

This copy is distributed inside the journal submission package. Commands below
assume that `code/` is the working repository root; the package's derived
source tables are in the sibling `../source_data/` directory.

This repository contains code and derived computational source tables. It does
not redistribute large public sequencing,
TCGA, or CPTAC files. Obtain those resources from their primary public
repositories and respect their current access terms.

## Resource Roles

| Resource | Workflow role | Retrieval route |
|---|---|---|
| GSE243013 | NSCLC anti-PD-1 single-cell/TCR immune-state summaries | NCBI GEO |
| GSE207422 | NSCLC neoadjuvant bulk and single-cell data; MPR endpoint | NCBI GEO |
| GSE126044 | NSCLC anti-PD-1 bulk response cohort | NCBI GEO |
| GSE135222 | Exploratory PFS association/projection | NCBI GEO |
| GSE274975 | Histology-only projection; not a response-modeling cohort | NCBI GEO |
| TCGA-LUAD/LUSC clinical | Non-ICI survival/projection context | NCI GDC Cases API |
| TCGA-LUAD/LUSC expression | Non-ICI transcriptomic projection | UCSC Xena GDC Hub |
| CPTAC LUAD/LSCC | Proteome/phosphoproteome projection | LinkedOmics public portal |

## GEO Inputs

`scripts/download_CGZ_public_data.ps1` downloads the named GEO supplementary
files and emits `CGZ_public_data_manifest.tsv`, including expected and observed
byte counts. From the repository root:

```powershell
powershell -ExecutionPolicy Bypass -File "scripts\download_CGZ_public_data.ps1" -SkipHuge
```

`-SkipHuge` omits `GSE243013_NSCLC_immune_scRNA_counts.mtx.gz`, the 7.1 GB
matrix. Omit the flag to retrieve the complete GEO input set. Add
`-IncludeOptional` only when the optional GEO RAW archive is required.

The script writes files below `data/geo/` with the exact filenames consumed by
the analysis. Its URLs point to NCBI's GEO FTP service; no login or local
credential is required.

## TCGA and Xena Inputs

Run the public acquisition helper after installing the environment:

```bash
python scripts/download_external_public_data.py --all
```

It performs two documented public operations:

1. Queries `https://api.gdc.cancer.gov/cases` for TCGA-LUAD and TCGA-LUSC,
   using a project filter and case-level clinical fields. The script normalizes
   the returned TSV into the fields consumed by the workflow and writes:

   ```text
   data/gdc_clinical/TCGA-LUAD/clinical/clinical.tsv
   data/gdc_clinical/TCGA-LUSC/clinical/clinical.tsv
   ```

2. Downloads the public Xena GDC Hub releases:

   ```text
   https://gdc-hub.s3.us-east-1.amazonaws.com/download/TCGA-LUAD.star_counts.tsv.gz
   https://gdc-hub.s3.us-east-1.amazonaws.com/download/TCGA-LUSC.star_counts.tsv.gz
   ```

   They are stored at `data/xena/TCGA-LUAD.star_counts.tsv.gz` and
   `data/xena/TCGA-LUSC.star_counts.tsv.gz`, respectively.

The helper records the resolved GDC request URL, source URL, UTC retrieval
time, file size, and SHA-256 checksum in `external_public_data_manifest.tsv`.
Use `--verify` to check expected external paths without downloading them.

## CPTAC / LinkedOmics Inputs

The analysis expects the following unmodified exports under
`data/cptac/LinkedOmics/`:

```text
HS_CPTAC_LUAD_proteome_ratio_NArm_TUMOR.cct
HS_CPTAC_LUAD_phosphoproteome_ratio_norm_NArm_TUMOR.cct
HS_CPTAC_LUAD_cli.tsi
HS_CPTAC_LSCC_2020_proteome_ratio_NArm_TUMOR.cct
HS_CPTAC_LSCC_2020_phospho_ratio_norm_NArm_TUMOR.cct
HS_CPTAC_LSCC_2020_clinical_phenotypes_TUMOR.tsi
```

Use the official [LinkedOmics portal](https://linkedomics.org/) to select and
download these CPTAC LUAD/LSCC tables. The
[Proteomic Data Commons](https://pdc.cancer.gov/pdc/) is the authoritative
public CPTAC archive for checking dataset provenance. The analysis was written
for the named LinkedOmics export format, so do not silently substitute a
different PDC export format without adapting the parser.

Portal download links can be session-dependent. To avoid publishing invented
or unstable URLs, the repository does not automate that portal step. After
downloading the six files from the official interface, stage and checksum them
with:

```bash
python scripts/download_external_public_data.py \
  --stage-cptac /absolute/path/to/LinkedOmics_exports
python scripts/download_external_public_data.py --verify
```

The staging command accepts either the directory containing the six files or a
parent directory with a `LinkedOmics/` child directory. It refuses to replace
existing staged files unless `--force` is supplied.

## Included and Derived Tables

Generated tables, figures, and models are written under `results/` and are not
committed to the public code repository. The journal submission package carries
derived, plot-ready tables under `../source_data/` for review and figure-source
inspection.

## Access Limits and DOI

No controlled-access source is required by the scripted public workflow. A
Zenodo DOI has not been issued for this revision. `zenodo.json` is prepared for
the author-controlled deposit; a DOI should be cited only after a release has
been published and its metadata verified.
