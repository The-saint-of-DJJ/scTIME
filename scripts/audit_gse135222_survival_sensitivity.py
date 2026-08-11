#!/usr/bin/env python
"""Audit the available clinical adjustment sensitivity for GSE135222 PFS.

The archived GSE135222 metadata contain age and sex, but not smoking history,
performance status, stage, or treatment-line covariates. This script therefore
fits only pre-specified score-plus-age/sex sensitivity models and records their
limited, exploratory status. The score is the full-refit scTIME projection and
is not an independent validation estimate.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.duration.hazard_regression import PHReg


SCRIPT_DIR = Path(__file__).resolve().parent
PACKAGE_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_TABLES = PACKAGE_ROOT / "source_data" / "results_tables"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tables-dir",
        type=Path,
        default=DEFAULT_TABLES,
        help="Directory containing scTIME_AI_scores_all_bulk_cohorts.tsv and receiving the audit TSV.",
    )
    return parser.parse_args()


def fit_model(data: pd.DataFrame, variables: list[str], model_label: str) -> pd.DataFrame:
    x = data[variables].astype(float)
    fit = PHReg(data["pfs_time"].astype(float), x, status=data["pfs_event"].astype(int)).fit(disp=0)
    ci = np.asarray(fit.conf_int(), dtype=float)
    rows = []
    for index, variable in enumerate(variables):
        rows.append(
            {
                "dataset": "GSE135222",
                "analysis": model_label,
                "variable": variable,
                "n": len(data),
                "events": int(data["pfs_event"].sum()),
                "coefficient": float(fit.params[index]),
                "hazard_ratio": float(np.exp(fit.params[index])),
                "ci_low": float(np.exp(ci[index, 0])),
                "ci_high": float(np.exp(ci[index, 1])),
                "p_value": float(fit.pvalues[index]),
                "covariates": ";".join(variables),
                "score_provenance": "full-refit scTIME score projected to GSE135222 for exploratory PFS association; not an independent response-modeling cohort",
                "interpretation": "Exploratory sensitivity only; no smoking history, performance status, stage, or treatment-line covariates were available.",
            }
        )
    return pd.DataFrame(rows)


def main() -> int:
    args = parse_args()
    tables = args.tables_dir.resolve()
    source = tables / "scTIME_AI_scores_all_bulk_cohorts.tsv"
    if not source.exists():
        raise FileNotFoundError(f"Missing archived score table: {source}")
    scores = pd.read_csv(source, sep="\t")
    data = scores.loc[scores["dataset"].eq("GSE135222")].copy()
    data["scTIME_score"] = pd.to_numeric(data["scTIME_AI_score"], errors="coerce")
    data["pfs_time"] = pd.to_numeric(data["pfs_time"], errors="coerce")
    data["pfs_event"] = pd.to_numeric(data["pfs_event"], errors="coerce")
    data["age"] = pd.to_numeric(data["age"], errors="coerce")
    data["female"] = data["gender"].astype(str).str.strip().str.lower().eq("female").astype(float)
    data = data.dropna(subset=["scTIME_score", "pfs_time", "pfs_event", "age", "gender"]).copy()
    if len(data) != 27 or int(data["pfs_event"].sum()) != 21 or int(data["female"].sum()) != 5:
        raise ValueError("Expected 27 GSE135222 samples, 21 events, and 5 female samples with complete age/sex.")

    model_specs = [
        ("univariate Cox PH", ["scTIME_score"]),
        ("age-adjusted Cox PH sensitivity", ["scTIME_score", "age"]),
        ("sex-adjusted Cox PH sensitivity", ["scTIME_score", "female"]),
        ("age-and-sex-adjusted Cox PH sensitivity", ["scTIME_score", "age", "female"]),
    ]
    outputs = []
    for label, variables in model_specs:
        outputs.append(fit_model(data, variables, label))
    output = pd.concat(outputs, ignore_index=True)
    output.to_csv(tables / "gse135222_multivariable_sensitivity.tsv", sep="\t", index=False)
    score_row = output.loc[
        output["analysis"].eq("age-and-sex-adjusted Cox PH sensitivity")
        & output["variable"].eq("scTIME_score")
    ].iloc[0]
    print(f"Wrote {tables / 'gse135222_multivariable_sensitivity.tsv'}")
    print(
        "Age-and-sex-adjusted scTIME sensitivity: "
        f"HR={score_row['hazard_ratio']:.6f}, P={score_row['p_value']:.6f}, "
        f"95% CI={score_row['ci_low']:.6f}-{score_row['ci_high']:.6f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
