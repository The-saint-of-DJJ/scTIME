# Script Index

This folder contains the retained scTIME analysis and release-support scripts.

- `audit_heldout_model_revision.py`: audits pooled OOF predictions, matched
  scTIME/CD274 stratification, per-cohort sensitivity, nested model selection,
  and ElasticNet coefficients.
- `audit_gse135222_survival_sensitivity.py`: fits the exploratory age/sex
  adjusted GSE135222 PFS sensitivity analysis from archived result tables.
- `build_submission_tables.py`: rebuilds submission-facing tables from archived
  result tables.
- `check_duplicate_table_content.py`: checks for duplicated table content in
  submission tables.
- `design_publication_figures.py`: regenerates retained composite data figures
  and supplementary figures from archived source tables.
- `download_CGZ_public_data.ps1`: downloads public GEO resources for the
  from-scratch workflow.
- `download_external_public_data.py`: documents and stages TCGA/GDC/Xena and
  CPTAC/LinkedOmics inputs for from-scratch reruns.
- `export_figure_panel_data_and_results.py`: exports plot-ready panel source
  tables and the panel-data manifest for retained figures.
- `export_main_figure_individual_panels.py`: exports independent main-figure
  panel images from panel source tables.
- `figure6_heldout.py`: shared utilities for the revised held-out Figure 6
  stratification and feature-contrast analysis.
- `run_cgz_analysis.py`: full public-data analysis workflow after required
  external data have been downloaded or staged.
