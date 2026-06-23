# Cell-validation Input Tables

This folder contains small tabular inputs for the cell-perturbation validation figure.

The corresponding script is:

```bash
python scripts/design_figure8_cell_validation.py
```

These files encode the G1-G5 perturbation series:

- G1: control
- G2: IFN activation
- G3: IFN + CXCL10 blockade
- G4: IFN + exclusion pressure
- G5: exclusion rescue

The script writes regenerated outputs to `results/figures/` and panel-level tables to `results/figure_design/panel_data/`.
