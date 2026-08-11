#!/usr/bin/env python
"""Prepare provenance-labeled held-out data for Figure 6.

The response-labelled samples in Figure 6 must use each patient's pooled
five-fold out-of-fold ElasticNet prediction.  The final refit is retained only
as an audit column and is never used for response stratification.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


LABELED_DATASETS = ("GSE126044", "GSE207422_bulk")
OOF_MODEL = "ElasticNet_logistic"
FINAL_REFIT_MODEL = "ElasticNet_logistic_final_fit"
OOF_SCORE_PROVENANCE = (
    "Pooled five-fold out-of-fold ElasticNet probability; each labeled sample "
    "was scored by a fold model that did not fit that sample."
)
FINAL_REFIT_PROVENANCE = (
    "Final ElasticNet refit probability on all 40 labeled samples; retained "
    "for audit only and not used for Figure 6 stratification."
)


def _require_unique_predictions(predictions: pd.DataFrame, model: str) -> pd.DataFrame:
    selected = predictions.loc[
        predictions["model"].eq(model) & predictions["dataset"].isin(LABELED_DATASETS),
        ["dataset", "sample", "prediction"],
    ].copy()
    if selected.duplicated(["dataset", "sample"]).any():
        raise ValueError(f"{model} predictions are not unique by dataset and sample.")
    return selected


def _assign_ranked_splits(
    frame: pd.DataFrame,
    score_col: str,
    prefix: str,
) -> pd.DataFrame:
    """Assign deterministic, within-cohort quartile and median groups.

    Ranking by score and sample ID avoids ambiguous membership in the unlikely
    event of tied values while preserving exact group sizes (4/6 quartile and
    8/12 median samples in the two labeled cohorts).
    """

    out = frame.copy()
    quartile_group = pd.Series(index=out.index, dtype=object)
    median_group = pd.Series(index=out.index, dtype=object)
    lower_cutoff = pd.Series(index=out.index, dtype=float)
    upper_cutoff = pd.Series(index=out.index, dtype=float)
    median_cutoff = pd.Series(index=out.index, dtype=float)

    for _, idx in out.groupby("dataset", observed=True).groups.items():
        cohort = out.loc[idx].copy()
        cohort[score_col] = pd.to_numeric(cohort[score_col], errors="coerce")
        if cohort[score_col].isna().any():
            raise ValueError(f"Missing {score_col} values in Figure 6 cohort.")
        cohort = cohort.sort_values([score_col, "sample"], kind="mergesort")
        n = len(cohort)
        n_quartile = int(round(n * 0.25))
        n_median = n // 2
        if n_quartile < 1 or n_median < 1:
            raise ValueError("Figure 6 cohorts require at least four labeled samples.")

        display_prefix = "scTIME OOF" if prefix == "scTIME" else prefix
        q_labels = np.full(n, f"{display_prefix} intermediate", dtype=object)
        q_labels[:n_quartile] = f"{display_prefix} bottom quartile"
        q_labels[-n_quartile:] = f"{display_prefix} top quartile"
        m_labels = np.full(n, f"{display_prefix} high (median)", dtype=object)
        m_labels[:n_median] = f"{display_prefix} low (median)"

        quartile_group.loc[cohort.index] = q_labels
        median_group.loc[cohort.index] = m_labels
        lower_cutoff.loc[cohort.index] = cohort.iloc[n_quartile - 1][score_col]
        upper_cutoff.loc[cohort.index] = cohort.iloc[-n_quartile][score_col]
        median_cutoff.loc[cohort.index] = cohort.iloc[n_median][score_col]

    out[f"{prefix}_quartile_group"] = quartile_group
    out[f"{prefix}_median_group"] = median_group
    out[f"{prefix}_bottom_quartile_upper_cutoff"] = lower_cutoff
    out[f"{prefix}_top_quartile_lower_cutoff"] = upper_cutoff
    out[f"{prefix}_median_high_lower_cutoff"] = median_cutoff
    return out


def prepare_heldout_figure6_labeled(tables: Path) -> pd.DataFrame:
    """Merge public-cohort features with OOF ElasticNet probabilities."""

    bulk = pd.read_csv(tables / "scTIME_AI_scores_all_bulk_cohorts.tsv", sep="\t")
    predictions = pd.read_csv(tables / "model_predictions.tsv", sep="\t")
    labeled = bulk.loc[
        bulk["dataset"].isin(LABELED_DATASETS),
    ].dropna(subset=["benefit", "CD274_expr"]).copy()

    oof = _require_unique_predictions(predictions, OOF_MODEL).rename(
        columns={"prediction": "scTIME_oof_score"}
    )
    final_refit = _require_unique_predictions(predictions, FINAL_REFIT_MODEL).rename(
        columns={"prediction": "scTIME_final_refit_score"}
    )
    labeled = labeled.merge(oof, on=["dataset", "sample"], how="inner", validate="one_to_one")
    labeled = labeled.merge(final_refit, on=["dataset", "sample"], how="left", validate="one_to_one")
    expected_samples = len(labeled)
    if expected_samples != 40 or len(oof) != expected_samples:
        raise ValueError(
            f"Figure 6 requires 40 labeled OOF predictions; found {expected_samples} merged and {len(oof)} OOF rows."
        )

    labeled["benefit"] = pd.to_numeric(labeled["benefit"], errors="raise").astype(int)
    labeled["response"] = np.where(labeled["benefit"].eq(1), "benefit/MPR", "no benefit")
    labeled["score_provenance"] = OOF_SCORE_PROVENANCE
    labeled["final_refit_provenance"] = FINAL_REFIT_PROVENANCE
    labeled = _assign_ranked_splits(labeled, "scTIME_oof_score", "scTIME")
    labeled = _assign_ranked_splits(labeled, "CD274_expr", "CD274")
    labeled["combined_median_group"] = (
        labeled["CD274_median_group"] + " / " + labeled["scTIME_median_group"]
    )
    return labeled


def _fisher_by_groups(frame: pd.DataFrame, group_col: str, first: str, second: str) -> float:
    first_values = frame.loc[frame[group_col].eq(first), "benefit"]
    second_values = frame.loc[frame[group_col].eq(second), "benefit"]
    table = [
        [int(first_values.sum()), int(len(first_values) - first_values.sum())],
        [int(second_values.sum()), int(len(second_values) - second_values.sum())],
    ]
    return float(stats.fisher_exact(table, alternative="two-sided").pvalue)


def build_stratification_table(labeled: pd.DataFrame) -> pd.DataFrame:
    """Return matched within-cohort quartile and median response summaries."""

    strategies = [
        (
            "scTIME OOF quartile",
            "scTIME_oof_score",
            "scTIME_quartile_group",
            ["scTIME OOF bottom quartile", "scTIME OOF intermediate", "scTIME OOF top quartile"],
            "Within-cohort rank-defined bottom/top 25%; pooled after group assignment.",
            "scTIME OOF top quartile vs scTIME OOF bottom quartile",
            "scTIME OOF top quartile",
            "scTIME OOF bottom quartile",
            OOF_SCORE_PROVENANCE,
        ),
        (
            "CD274 quartile",
            "CD274_expr",
            "CD274_quartile_group",
            ["CD274 bottom quartile", "CD274 intermediate", "CD274 top quartile"],
            "Within-cohort rank-defined bottom/top 25%; pooled after group assignment.",
            "CD274 top quartile vs CD274 bottom quartile",
            "CD274 top quartile",
            "CD274 bottom quartile",
            "Within-cohort CD274 expression-module rank; no fitted-model probability.",
        ),
        (
            "scTIME OOF median",
            "scTIME_oof_score",
            "scTIME_median_group",
            ["scTIME OOF low (median)", "scTIME OOF high (median)"],
            "Within-cohort rank-defined median split; pooled after group assignment.",
            "scTIME OOF high (median) vs scTIME OOF low (median)",
            "scTIME OOF high (median)",
            "scTIME OOF low (median)",
            OOF_SCORE_PROVENANCE,
        ),
        (
            "CD274 median",
            "CD274_expr",
            "CD274_median_group",
            ["CD274 low (median)", "CD274 high (median)"],
            "Within-cohort rank-defined median split; pooled after group assignment.",
            "CD274 high (median) vs CD274 low (median)",
            "CD274 high (median)",
            "CD274 low (median)",
            "Within-cohort CD274 expression-module rank; no fitted-model probability.",
        ),
    ]
    rows: list[dict[str, object]] = []
    for strategy, score_col, group_col, order, cutoff_rule, comparison, first, second, provenance in strategies:
        fisher_p = _fisher_by_groups(labeled, group_col, first, second)
        for group in order:
            sub = labeled.loc[labeled[group_col].eq(group)].copy()
            n = len(sub)
            response_count = int(sub["benefit"].sum())
            rate = response_count / n if n else np.nan
            rows.append(
                {
                    "strategy": strategy,
                    "score_column": score_col,
                    "score_provenance": provenance,
                    "cutoff_rule": cutoff_rule,
                    "group": group,
                    "n": n,
                    "responses": response_count,
                    "benefit_rate": rate,
                    "mean_score": float(pd.to_numeric(sub[score_col], errors="coerce").mean()),
                    "mean_scTIME_oof_score": float(pd.to_numeric(sub["scTIME_oof_score"], errors="coerce").mean()),
                    "mean_cd274": float(pd.to_numeric(sub["CD274_expr"], errors="coerce").mean()),
                    "fisher_comparison": comparison,
                    "fisher_exact_p_two_sided": fisher_p,
                }
            )
    out = pd.DataFrame(rows)
    p = out["benefit_rate"].astype(float)
    n = out["n"].astype(float)
    z = 1.96
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt((p * (1 - p) + z**2 / (4 * n)) / n) / denom
    out["ci_low"] = (center - half).clip(lower=0)
    out["ci_high"] = (center + half).clip(upper=1)
    return out


def build_combined_response_groups(labeled: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        labeled.groupby("combined_median_group", observed=True)
        .agg(
            n=("sample", "size"),
            responses=("benefit", "sum"),
            benefit_rate=("benefit", "mean"),
            mean_scTIME_oof_score=("scTIME_oof_score", "mean"),
            mean_cd274=("CD274_expr", "mean"),
        )
        .reset_index()
    )
    grouped["score_provenance"] = OOF_SCORE_PROVENANCE
    grouped["cutoff_rule"] = "Within-cohort rank-defined median splits; pooled after group assignment."
    p = grouped["benefit_rate"].astype(float)
    n = grouped["n"].astype(float)
    z = 1.96
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt((p * (1 - p) + z**2 / (4 * n)) / n) / denom
    grouped["ci_low"] = (center - half).clip(lower=0)
    grouped["ci_high"] = (center + half).clip(upper=1)
    return grouped


def benjamini_hochberg(p_values: pd.Series) -> pd.Series:
    p = pd.to_numeric(p_values, errors="coerce")
    out = pd.Series(np.nan, index=p.index, dtype=float)
    valid = p.dropna().sort_values()
    if valid.empty:
        return out
    adjusted = valid * len(valid) / np.arange(1, len(valid) + 1)
    adjusted = np.minimum.accumulate(adjusted.iloc[::-1]).iloc[::-1].clip(upper=1.0)
    out.loc[adjusted.index] = adjusted
    return out


def build_feature_contrast(labeled: pd.DataFrame) -> pd.DataFrame:
    """Contrast OOF-scTIME high patients with CD274-high/OOF-scTIME-low patients."""

    features = [
        "IFN_response",
        "Antigen_presentation",
        "Cytotoxic_CD8",
        "CXCL10_CXCR3_axis",
        "SPP1_macrophage",
        "TGF_beta_EMT",
        "Hypoxia",
    ]
    high_score = labeled.loc[labeled["scTIME_median_group"].eq("scTIME OOF high (median)")]
    cd274_high_score_low = labeled.loc[
        labeled["CD274_median_group"].eq("CD274 high (median)")
        & labeled["scTIME_median_group"].eq("scTIME OOF low (median)")
    ]
    rows: list[dict[str, object]] = []
    for feature in features:
        first = pd.to_numeric(high_score[feature], errors="coerce").dropna()
        second = pd.to_numeric(cd274_high_score_low[feature], errors="coerce").dropna()
        p_value = (
            float(stats.mannwhitneyu(first, second, alternative="two-sided").pvalue)
            if len(first) >= 2 and len(second) >= 2
            else np.nan
        )
        rows.append(
            {
                "feature": feature,
                "high_oof_sctime_mean": first.mean(),
                "cd274_high_oof_sctime_low_mean": second.mean(),
                "difference": first.mean() - second.mean(),
                "p_value": p_value,
                "n_high_oof_sctime": len(first),
                "n_cd274_high_oof_sctime_low": len(second),
                "group_rule": "Within-cohort median groups using held-out scTIME OOF predictions.",
                "score_provenance": OOF_SCORE_PROVENANCE,
            }
        )
    out = pd.DataFrame(rows)
    out["fdr"] = benjamini_hochberg(out["p_value"])
    return out


def panel_a_frame(labeled: pd.DataFrame) -> pd.DataFrame:
    """Return concise plot-ready source data rather than cohort metadata dumps."""

    columns = [
        "dataset",
        "sample",
        "endpoint",
        "endpoint_text",
        "benefit",
        "response",
        "CD274_expr",
        "scTIME_oof_score",
        "scTIME_final_refit_score",
        "score_provenance",
        "final_refit_provenance",
        "scTIME_quartile_group",
        "scTIME_median_group",
        "CD274_quartile_group",
        "CD274_median_group",
        "scTIME_bottom_quartile_upper_cutoff",
        "scTIME_top_quartile_lower_cutoff",
        "scTIME_median_high_lower_cutoff",
        "CD274_bottom_quartile_upper_cutoff",
        "CD274_top_quartile_lower_cutoff",
        "CD274_median_high_lower_cutoff",
    ]
    return labeled.loc[:, [column for column in columns if column in labeled.columns]].copy()
