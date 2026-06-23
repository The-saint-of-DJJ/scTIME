# Data Availability

This repository does not include large public sequencing, TCGA or CPTAC files. These data should be obtained from the original public resources and placed under `data/` before running the full workflow.

## Public Resources

| Resource | Role in scTIME-AI workflow |
|---|---|
| GSE243013 | NSCLC anti-PD-1 single-cell/TCR resource for immune-cell composition, TCR ecology and response-linked state analysis |
| GSE207422 | NSCLC neoadjuvant immunotherapy single-cell and bulk RNA-seq resource for immune-state discovery, MPR endpoint modeling and scRNA validation |
| GSE126044 | NSCLC anti-PD-1 bulk RNA-seq response cohort for reciprocal cross-cohort validation |
| GSE135222 | NSCLC ICI/pembrolizumab bulk RNA-seq cohort for exploratory PFS validation |
| GSE274975 | NSCLC bulk RNA-seq resource used for projection/histology-level immune profiling when endpoint metadata are available locally |
| TCGA-LUAD/LUSC | External non-ICI transcriptomic projection resources |
| CPTAC LUAD/LSCC | External proteome/phosphoproteome projection resources |

## GEO Download Helper

The PowerShell downloader retrieves the GEO files used by the workflow and writes `CGZ_public_data_manifest.tsv` with file-size checks:

```powershell
powershell -ExecutionPolicy Bypass -File "scripts\download_CGZ_public_data.ps1" -SkipHuge
```

Use the command without `-SkipHuge` to download the largest GSE243013 matrix as well.

## Local Input Data

Small cell-validation tables are included in `Biodata/` because they are required to regenerate Figure 8:

- `tumor_qPCR_long.csv`
- `Tcell_qPCR_long.csv`
- `secretome_viability_long.csv`
- `CD8_migration_long.csv`
- `Tcell_activation_killing_long.csv`
- `score_by_replicate.csv`
- `summary_indices_by_condition.csv`
- `statistics.csv`
- `experimental_design.csv`
- `reference_mapping.csv`
- `replicate_matrix.csv`

## Generated Data

The workflow writes generated tables, figures and models under `results/`. These outputs are excluded from version control and can be regenerated from public input data plus `Biodata/`.

## Controlled-access Data

No controlled-access data are required to run the public workflow. If future users add controlled-access resources, those files should not be committed to this repository.
