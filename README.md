# scTIME-AI

Interpretable artificial intelligence workflow for modeling immunotherapy-associated immune states in non-small cell lung cancer (NSCLC) from patient-derived public multi-omics resources.

This repository accompanies the scTIME-AI manuscript. The code integrates public single-cell RNA-seq, TCR, bulk RNA-seq, survival, TCGA and CPTAC projection resources to build an interpretable immune-state score and generate manuscript figures/source tables.

## Study Overview

scTIME-AI is a single-cell-informed Tumor Immune MicroEnvironment AI score. The workflow:

1. Derives immune-state modules from NSCLC single-cell resources.
2. Projects these modules into bulk RNA-seq cohorts.
3. Trains an interpretable ElasticNet model for ICI/MPR response-associated stratification.
4. Evaluates cross-validation, reciprocal cross-cohort validation, calibration, decision-curve behavior and patient-level feature contributions.
5. Projects the score across TCGA/CPTAC multi-omics resources.
6. Generates publication figures and panel-level source data.
7. Visualizes cell-perturbation validation data for AI-nominated responder and exclusion axes.

Key biological axes represented in the model include IFN response, antigen presentation, cytotoxic CD8, CXCL10-CXCR3 signaling, CD274 expression, SPP1 macrophage, TGF-beta/EMT and hypoxia.

## Repository Layout

```text
.
├── Biodata/                         # Small cell-validation input tables used for Figure 8
├── scripts/
│   ├── download_CGZ_public_data.ps1  # GEO public-data downloader
│   ├── run_cgz_analysis.py           # Main computational workflow
│   ├── design_publication_figures.py # Main/supplementary figure generation
│   ├── design_figure8_cell_validation.py
│   ├── export_figure_panel_data_and_results.py
│   └── export_main_figure_individual_panels.py
├── DATA_AVAILABILITY.md
├── environment.yml
├── requirements.txt
└── README.md
```

Large public datasets and generated outputs are intentionally not committed. The scripts create `data/`, `results/`, `logs/` and `results/models/` at runtime.

## Installation

Create a fresh environment:

```bash
conda env create -f environment.yml
conda activate sctime-ai
```

Alternatively, use pip:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The full enhanced workflow uses optional packages including `scanpy`, `xgboost` and `lightgbm`. If optional packages are unavailable, the main script records the missing enhanced module in status tables where possible.

## Data Setup

On Windows PowerShell:

```powershell
powershell -ExecutionPolicy Bypass -File "scripts\download_CGZ_public_data.ps1" -SkipHuge
```

The `-SkipHuge` flag skips the largest GSE243013 count matrix. The current workflow uses official GSE243013 cell annotations and TCR tables for full-cohort summaries and does not need to stream the 7.1 GB matrix for the manuscript-level analyses.

To download all core files, including the largest matrix:

```powershell
powershell -ExecutionPolicy Bypass -File "scripts\download_CGZ_public_data.ps1"
```

To include the optional GEO RAW archive:

```powershell
powershell -ExecutionPolicy Bypass -File "scripts\download_CGZ_public_data.ps1" -IncludeOptional
```

See `DATA_AVAILABILITY.md` for dataset roles and external resources.

## Run the Analysis

Run commands from the repository root:

```bash
python scripts/run_cgz_analysis.py
python scripts/design_publication_figures.py
python scripts/design_figure8_cell_validation.py
python scripts/export_figure_panel_data_and_results.py
python scripts/export_main_figure_individual_panels.py
```

Expected output folders:

```text
results/tables/          # TSV matrices, model predictions and statistics
results/figures/         # PNG/PDF publication figures
results/figure_design/   # Panel-level source data and figure metadata
results/models/          # Serialized scTIME-AI model and metadata
logs/                    # Runtime logs
```

## Main Outputs

The main workflow produces:

- `results/tables/model_performance.tsv`
- `results/tables/model_predictions.tsv`
- `results/tables/scTIME_AI_scores_all_bulk_cohorts.tsv`
- `results/tables/elasticnet_coefficients.tsv`
- `results/tables/elasticnet_linear_shap_style_contributions.tsv`
- `results/models/scTIME_AI_elasticnet_pipeline.pkl`
- `results/models/scTIME_AI_elasticnet_metadata.json`

The contribution table contains exact linear logit feature contributions from the final ElasticNet model; no external SHAP package is required.

## Reproducibility Notes

- Public-data downloads are checked against expected file sizes by `download_CGZ_public_data.ps1`.
- The main script records analysis status in `results/tables/enhanced_analysis_status.tsv`.
- TCGA/CPTAC projections are data-driven: local expression/proteome matrices are analyzed when present; otherwise status tables document missing inputs.
- Generated model files (`*.pkl`, `*.h5ad`) and raw public data are excluded from version control.

## Citation

Please cite the associated manuscript when available. Until publication, cite this repository as:

> scTIME-AI: interpretable AI modeling of NSCLC immunotherapy-associated immune states from patient-derived public multi-omics resources.

## License

Choose and add the intended repository license before public release. If the journal or institution does not require a specific license, MIT or BSD-3-Clause are common choices for analysis code.
