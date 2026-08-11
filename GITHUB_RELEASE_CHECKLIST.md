# GitHub Release Checklist

Use this checklist before updating `https://github.com/The-saint-of-DJJ/scTIME`.

## Include

- `README.md`
- `DATA_AVAILABILITY.md`
- `DEPOSITION.md`
- `LICENSE`
- `requirements.txt`
- `requirements.lock.txt`
- `environment.yml`
- `zenodo.json`
- `scripts/*.py`
- `scripts/download_CGZ_public_data.ps1`

If the GitHub repository is organized like the submission package, also include
the reviewed `source_data/results_tables/` and `source_data/panel_data/`
directories from the package root so that archived-result reproduction works.

## Exclude

- `scripts/__pycache__/`
- `*.pyc`
- local virtual environments
- local raw downloads under `code/data/`
- local rerun outputs under `code/results/`
- private notes, manuscript drafts, reviewer-response Word files, and temporary
  editing artifacts

## Pre-push checks

From the package root, run:

```bash
python code/scripts/audit_heldout_model_revision.py
python code/scripts/audit_gse135222_survival_sensitivity.py
python -m py_compile code/scripts/*.py
```

Then confirm:

```bash
find code -type d -name __pycache__
find code -type f -name "*.pyc"
```

Both commands should return no tracked release files after cleanup.
