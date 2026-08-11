# scTIME Reproducibility Code

This directory is the GitHub-ready executable code snapshot for the revised
scTIME public-data analysis. The public workflow name is **scTIME**. Legacy
strings such as `scTIME-AI`, `scTIME_AI_score`, and
`scTIME_AI_elasticnet_pipeline.pkl` remain in historical filenames and table
columns for compatibility; they do not identify a separate model.

The canonical public repository is
`https://github.com/The-saint-of-DJJ/scTIME`.

## What This Package Contains

- `../source_data/results_tables/`: archived derived analysis tables, including
  held-out prediction inputs.
- `../source_data/panel_data/`: plot-ready panel source tables and their
  manifest.
- `scripts/`: analysis, source-data export, figure-rendering, and audit code.
- `requirements.lock.txt` and `environment.yml`: pinned tested environment.
- `DATA_AVAILABILITY.md`: external-data acquisition and staging instructions.
- `GITHUB_RELEASE_CHECKLIST.md`: a short checklist for updating the public
  repository without committing caches, raw external data, or local artifacts.

The package does not redistribute large GEO, TCGA, or CPTAC files. It does not
have a Zenodo DOI yet; `zenodo.json` is a metadata template that requires a
verified depositor name and an author-controlled release before a DOI can be
cited. See [DEPOSITION.md](DEPOSITION.md) for the required release steps.

The retained scripts cover the submitted data figures, supplementary figures,
held-out model audit, survival sensitivity audit, source-data export, and
external public-data acquisition.

## Install

Run from `submission_package_frontiers_20260615_122936/`:

```bash
conda env create -f code/environment.yml
conda activate sctime
```

Or, from `code/`:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.lock.txt
```

The tested environment is Python 3.13.0. The lock includes the full Scanpy
Leiden path, XGBoost, LightGBM, OpenPyXL, and statsmodels because those are
needed for the complete computational workflow.

## Archived-Result Reproduction

These commands use only the deposited source-data tables. They are suitable
for inspecting the revision, regenerating the supplied figure outputs, and
re-running the held-out audit without downloading large raw datasets.

```bash
python code/scripts/audit_heldout_model_revision.py
python code/scripts/audit_gse135222_survival_sensitivity.py
python code/scripts/design_publication_figures.py
python code/scripts/export_figure_panel_data_and_results.py
python code/scripts/export_main_figure_individual_panels.py
```

Run them in this order: the audit scripts refresh the key revised statistics,
`design_publication_figures.py` regenerates the retained composite data
figures, and the panel-data exporters refresh the main and supplementary source
tables. Figure files are written to `figures_jpg/`; independent panel images
are written to `figures_jpg/main_figure_panels/`.

## Full Public-Data Rerun

Use the public repository for the full from-scratch workflow. If running the
snapshot here, first enter `code/`; the analysis writes raw-data outputs below
that directory:

```bash
cd code
powershell -ExecutionPolicy Bypass -File scripts/download_CGZ_public_data.ps1 -SkipHuge
python scripts/download_external_public_data.py --all
python scripts/download_external_public_data.py \
  --stage-cptac /absolute/path/to/LinkedOmics_exports
python scripts/download_external_public_data.py --verify
python scripts/run_cgz_analysis.py
```

`download_CGZ_public_data.ps1` can also be invoked through PowerShell 7
(`pwsh`) on macOS/Linux. `-SkipHuge` omits the 7.1 GB GSE243013 count matrix;
omit the flag to retrieve it. Read [DATA_AVAILABILITY.md](DATA_AVAILABILITY.md)
before obtaining any external data, especially the six portal-selected
LinkedOmics files.

The GDC/Xena helper downloads or normalizes the required public inputs and
writes `external_public_data_manifest.tsv` with source URLs and checksums. It
does not hard-code unverified LinkedOmics portal URLs; users download those
exact exports through the official portal and stage them with `--stage-cptac`.

The public repository's figure scripts read the newly generated
`results/` tables. In contrast, this journal-package snapshot intentionally
renders from `../source_data/` so the reviewed source tables and delivered
figures remain directly traceable. Do not describe the latter archive-rendering
route as a fresh model fit.

## Held-Out Model Audit

`scripts/audit_heldout_model_revision.py` starts from the archived labeled
feature matrix and pooled five-fold out-of-fold ElasticNet predictions. It
writes these files to `../source_data/results_tables/` by default:

```text
heldout_predictions.tsv
heldout_stratification.tsv
per_cohort_oof_sensitivity.tsv
nested_selection_sensitivity.tsv
coefficient_audit.tsv
```

The script documents deterministic nested-selection choices in its output. It
does not regenerate gene-expression features or fit the final deployment model.

`scripts/audit_gse135222_survival_sensitivity.py` fits the archived exploratory
GSE135222 Cox sensitivity models using the full-refit scTIME projection and the
available age/sex covariates. It writes
`source_data/results_tables/gse135222_multivariable_sensitivity.tsv`; smoking,
performance status, stage and treatment-line covariates are unavailable, so
these rows are not independent prognostic validation.

## License

Code in this directory is available under the [MIT License](LICENSE). Public
source datasets retain the terms set by their original repositories.
