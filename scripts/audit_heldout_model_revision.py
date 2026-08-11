#!/usr/bin/env python
"""Create reproducible reviewer-revision audits for the scTIME response model.

This script intentionally starts from the archived labeled feature matrix and
the archived five-fold predictions.  It does not regenerate expression scores
or refit the final deployment model, so it can be run from the revision
package without the large public input matrices.  Outputs are written to
``source_data/results_tables`` by default.

The nested sensitivity analysis uses a stratified five-fold outer split
(``random_state=1``) and a stratified four-fold inner split with
``random_state=100 + outer_fold``.  Every candidate model is scored by inner-fold ROC-AUC;
the model with the largest value is fitted to the outer training fold and used
once for the outer test fold.  Alphabetical model name resolves exact ties.
These fixed choices are reported in the output table rather than presented as
a tuning search.
"""

from __future__ import annotations

import argparse
import sys
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import clone
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict


SCRIPT_DIR = Path(__file__).resolve().parent
PACKAGE_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_TABLES = PACKAGE_ROOT / "source_data" / "results_tables"

# Import the canonical model factory rather than duplicating its hyperparameters.
sys.path.insert(0, str(SCRIPT_DIR))
from run_cgz_analysis import SIGNATURES, make_models, nested_candidate_selection_predictions  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tables-dir",
        type=Path,
        default=DEFAULT_TABLES,
        help="Directory containing archived results tables and receiving audit TSVs.",
    )
    return parser.parse_args()


def validate_inputs(tables_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    required = {
        "labeled feature matrix": tables_dir / "labeled_bulk_model_matrix.tsv",
        "model predictions": tables_dir / "model_predictions.tsv",
        "ElasticNet coefficients": tables_dir / "elasticnet_coefficients.tsv",
    }
    missing = [f"{label}: {path}" for label, path in required.items() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required audit inputs:\n" + "\n".join(missing))

    matrix = pd.read_csv(required["labeled feature matrix"], sep="\t")
    predictions = pd.read_csv(required["model predictions"], sep="\t")
    coefficients = pd.read_csv(required["ElasticNet coefficients"], sep="\t")
    needed = {"dataset", "sample", "benefit", "CD274_expr"}
    z_features = {f"z_{feature}" for feature in list(SIGNATURES) + ["CD274_expr"]}
    missing_cols = (needed | z_features) - set(matrix.columns)
    if missing_cols:
        raise ValueError(f"Labeled feature matrix is missing columns: {sorted(missing_cols)}")
    if len(matrix) != 40 or int(pd.to_numeric(matrix["benefit"], errors="raise").sum()) != 14:
        raise ValueError("Expected the archived 40-patient, 14-benefit labeled matrix.")
    return matrix, predictions, coefficients


def fisher_two_group_pvalue(first: pd.DataFrame, second: pd.DataFrame) -> float:
    table = [[int(first["benefit"].sum()), int((1 - first["benefit"]).sum())],
             [int(second["benefit"].sum()), int((1 - second["benefit"]).sum())]]
    return float(stats.fisher_exact(table, alternative="two-sided").pvalue)


def heldout_tables(matrix: pd.DataFrame, predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return one OOF row per person plus matched score-stratification summaries."""
    oof = predictions.loc[predictions["model"].eq("ElasticNet_logistic")].copy()
    key_columns = ["dataset", "sample"]
    if oof.duplicated(key_columns).any() or len(oof) != len(matrix):
        raise ValueError("Expected exactly one pooled-CV ElasticNet prediction per labeled participant.")
    audit = matrix.merge(
        oof[key_columns + ["prediction"]],
        on=key_columns,
        how="inner",
        validate="one_to_one",
    )
    if len(audit) != len(matrix):
        raise ValueError("Could not match all labeled participants to archived OOF predictions.")
    audit = audit[["dataset", "sample", "endpoint", "endpoint_text", "benefit", "CD274_expr", "prediction"]].copy()
    audit = audit.rename(columns={"prediction": "sctime_oof_probability"})
    audit["score_provenance"] = "pooled_5fold_out_of_fold_ElasticNet"

    grouped = audit.groupby("dataset", observed=True)
    audit["sctime_q25"] = grouped["sctime_oof_probability"].transform(lambda values: values.quantile(0.25))
    audit["sctime_q75"] = grouped["sctime_oof_probability"].transform(lambda values: values.quantile(0.75))
    audit["cd274_q25"] = grouped["CD274_expr"].transform(lambda values: values.quantile(0.25))
    audit["cd274_q75"] = grouped["CD274_expr"].transform(lambda values: values.quantile(0.75))
    audit["sctime_median_cutoff"] = grouped["sctime_oof_probability"].transform("median")
    audit["cd274_median_cutoff"] = grouped["CD274_expr"].transform("median")

    def quartile_group(values: pd.Series, low: pd.Series, high: pd.Series) -> pd.Series:
        return np.select(
            [values.le(low), values.ge(high)],
            ["bottom_quartile", "top_quartile"],
            default="interquartile",
        )

    audit["sctime_quartile_group"] = quartile_group(audit["sctime_oof_probability"], audit["sctime_q25"], audit["sctime_q75"])
    audit["cd274_quartile_group"] = quartile_group(audit["CD274_expr"], audit["cd274_q25"], audit["cd274_q75"])
    audit["sctime_median_group"] = np.where(
        audit["sctime_oof_probability"].ge(audit["sctime_median_cutoff"]), "high_median_half", "low_median_half"
    )
    audit["cd274_median_group"] = np.where(
        audit["CD274_expr"].ge(audit["cd274_median_cutoff"]), "high_median_half", "low_median_half"
    )
    audit = audit.sort_values(["dataset", "sample"]).reset_index(drop=True)

    rows: list[dict[str, object]] = []
    for score, split, column, low_group, high_group, provenance in [
        ("scTIME", "quartile", "sctime_quartile_group", "bottom_quartile", "top_quartile", "pooled_5fold_out_of_fold_ElasticNet"),
        ("CD274", "quartile", "cd274_quartile_group", "bottom_quartile", "top_quartile", "within_cohort_expression"),
        ("scTIME", "median", "sctime_median_group", "low_median_half", "high_median_half", "pooled_5fold_out_of_fold_ElasticNet"),
        ("CD274", "median", "cd274_median_group", "low_median_half", "high_median_half", "within_cohort_expression"),
    ]:
        low = audit.loc[audit[column].eq(low_group)]
        high = audit.loc[audit[column].eq(high_group)]
        pvalue = fisher_two_group_pvalue(high, low)
        for direction, subset in [("high_or_top", high), ("low_or_bottom", low)]:
            rows.append(
                {
                    "score": score,
                    "score_provenance": provenance,
                    "split": split,
                    "direction": direction,
                    "group_label": high_group if direction == "high_or_top" else low_group,
                    "cutoff_definition": "within-cohort quantiles, then pooled",
                    "n": len(subset),
                    "benefit_mpr": int(subset["benefit"].sum()),
                    "no_benefit_nmpr": int((1 - subset["benefit"]).sum()),
                    "benefit_mpr_rate": float(subset["benefit"].mean()),
                    "two_sided_fisher_p": pvalue,
                }
            )
    return audit, pd.DataFrame(rows), oof


def per_cohort_sensitivity(audit: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dataset, subset in audit.groupby("dataset", observed=True):
        y = subset["benefit"].astype(int)
        probabilities = subset["sctime_oof_probability"]
        rows.append(
            {
                "dataset": dataset,
                "score_provenance": "pooled_5fold_out_of_fold_ElasticNet",
                "n": len(subset),
                "benefit_mpr": int(y.sum()),
                "roc_auc": float(roc_auc_score(y, probabilities)),
                "pr_auc": float(average_precision_score(y, probabilities)),
                "brier": float(brier_score_loss(y, probabilities)),
                "interpretation": "pooled-CV subgroup sensitivity; not leave-one-cohort-out validation",
            }
        )
    return pd.DataFrame(rows)


def nested_selection_sensitivity(matrix: pd.DataFrame) -> pd.DataFrame:
    features = list(SIGNATURES) + ["CD274_expr"]
    y = matrix["benefit"].astype(int).to_numpy()
    nested, selection = nested_candidate_selection_predictions(matrix, features, y, make_models(random_state=1), 5)
    rows: list[dict[str, object]] = []
    for record in selection.to_dict("records"):
        rows.append(
            {
                "record_type": "inner_candidate",
                **record,
                "selected_in_outer_fold": record["candidate_model"] == record["selected_model"],
                "outer_split": "StratifiedKFold(n_splits=5, shuffle=True, random_state=1)",
                "inner_split": f"StratifiedKFold(n_splits=4, shuffle=True, random_state={100 + int(record['outer_fold'])})",
                "preprocessing": "dataset-specific scaling estimated inside each training fold",
            }
        )
    probabilities = pd.to_numeric(nested["prediction"], errors="raise")
    selected_counts = Counter(selection.drop_duplicates("outer_fold")["selected_model"])
    rows.append(
        {
            "record_type": "pooled_nested_summary",
            "outer_fold": "all",
            "n_outer_train": 32,
            "n_outer_test": len(y),
            "outer_roc_auc": float(roc_auc_score(y, probabilities)),
            "outer_pr_auc": float(average_precision_score(y, probabilities)),
            "outer_brier": float(brier_score_loss(y, probabilities)),
            "selected_model_counts": "; ".join(f"{name}={count}" for name, count in sorted(selected_counts.items())),
            "outer_split": "StratifiedKFold(n_splits=5, shuffle=True, random_state=1)",
            "inner_split": "StratifiedKFold(n_splits=4, shuffle=True, random_state=100 + outer_fold)",
            "preprocessing": "dataset-specific scaling estimated inside each training fold",
            "selection_rule": "maximum inner ROC-AUC; alphabetical model name breaks exact ties",
        }
    )
    return pd.DataFrame(rows)


def coefficient_audit(coefficients: pd.DataFrame) -> pd.DataFrame:
    out = coefficients.copy()
    out["absolute_coefficient"] = out["coefficient"].abs()
    out["coefficient_rank_by_absolute_value"] = out["absolute_coefficient"].rank(method="min", ascending=False).astype(int)
    out["is_exactly_zero"] = out["coefficient"].eq(0.0)
    out["interpretation"] = np.select(
        [out["coefficient"].gt(0), out["coefficient"].lt(0)],
        ["positive final-refit logit contribution", "negative final-refit logit contribution"],
        default="zero final-refit logit contribution",
    )
    return out.sort_values(["coefficient_rank_by_absolute_value", "feature"]).reset_index(drop=True)


def main() -> None:
    args = parse_args()
    tables_dir = args.tables_dir.resolve()
    tables_dir.mkdir(parents=True, exist_ok=True)
    matrix, predictions, coefficients = validate_inputs(tables_dir)
    heldout, stratification, _ = heldout_tables(matrix, predictions)
    sensitivity = per_cohort_sensitivity(heldout)
    nested = nested_selection_sensitivity(matrix)
    coefficient_table = coefficient_audit(coefficients)

    heldout.to_csv(tables_dir / "heldout_predictions.tsv", sep="\t", index=False)
    stratification.to_csv(tables_dir / "heldout_stratification.tsv", sep="\t", index=False)
    sensitivity.to_csv(tables_dir / "per_cohort_oof_sensitivity.tsv", sep="\t", index=False)
    nested.to_csv(tables_dir / "nested_selection_sensitivity.tsv", sep="\t", index=False)
    coefficient_table.to_csv(tables_dir / "coefficient_audit.tsv", sep="\t", index=False)

    summary = nested.loc[nested["record_type"].eq("pooled_nested_summary")].iloc[0]
    print(f"Wrote audit tables to {tables_dir}")
    print(
        "Nested selection sensitivity: "
        f"ROC-AUC={summary['outer_roc_auc']:.6f}, "
        f"PR-AUC={summary['outer_pr_auc']:.6f}, "
        f"Brier={summary['outer_brier']:.6f}"
    )


if __name__ == "__main__":
    main()
