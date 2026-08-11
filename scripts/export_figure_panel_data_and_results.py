#!/usr/bin/env python
"""Export plot-ready source data for every figure panel."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve

from figure6_heldout import (
    build_feature_contrast,
    build_stratification_table,
    panel_a_frame,
    prepare_heldout_figure6_labeled,
)


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
ROOT = PACKAGE_ROOT
TABLES = ROOT / "source_data" / "results_tables"
DESIGN = ROOT / "source_data"
PANEL_DATA = DESIGN / "panel_data"

FEATURES = [
    "IFN_response",
    "Antigen_presentation",
    "Cytotoxic_CD8",
    "Progenitor_exhausted_CD8",
    "Terminal_exhausted_CD8",
    "Treg",
    "SPP1_macrophage",
    "TGF_beta_EMT",
    "Hypoxia",
    "NK",
    "CXCL10_CXCR3_axis",
    "CD274_expr",
]

FEATURE_LABELS = {
    "IFN_response": "IFN response",
    "Antigen_presentation": "Antigen presentation",
    "Cytotoxic_CD8": "Cytotoxic CD8",
    "Progenitor_exhausted_CD8": "Progenitor exhausted CD8",
    "Terminal_exhausted_CD8": "Terminal exhausted CD8",
    "Treg": "Treg",
    "SPP1_macrophage": "SPP1 macrophage",
    "TGF_beta_EMT": "TGF-beta/EMT",
    "Hypoxia": "Hypoxia",
    "NK": "NK",
    "CXCL10_CXCR3_axis": "CXCL10-CXCR3 axis",
    "CD274_expr": "CD274",
    "scTIME_AI_score": "scTIME score",
}

ACTIVATION_FEATURES = ["IFN_response", "Antigen_presentation", "Cytotoxic_CD8", "NK", "CXCL10_CXCR3_axis", "CD274_expr"]
EXCLUSION_FEATURES = ["TGF_beta_EMT", "Hypoxia", "Treg", "Terminal_exhausted_CD8"]
FIGURE3A_RESEARCH_VALUE_THRESHOLD = 0.65

PANEL_ROWS: list[dict[str, object]] = []
MANAGED_FIGURES = {
    *(f"Figure {number}" for number in range(1, 8)),
    "Supplementary Figure 1",
    "Supplementary Figure 2",
}


def read_table(name: str, **kwargs) -> pd.DataFrame:
    return pd.read_csv(TABLES / f"{name}.tsv", sep="\t", **kwargs)


def clean_label(value: object) -> str:
    return str(value).replace("_", " ").replace("GSE207422 bulk", "GSE207422").replace("TCGA-", "")


def safe_slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")[:70]


def write_panel_data(
    figure: str,
    panel: str,
    title: str,
    df: pd.DataFrame,
    sources: list[str],
    description: str,
) -> Path:
    PANEL_DATA.mkdir(parents=True, exist_ok=True)
    filename = f"{safe_slug(figure)}_panel{panel.lower()}_{safe_slug(title)}.tsv"
    path = PANEL_DATA / filename
    df.to_csv(path, sep="\t", index=False)
    PANEL_ROWS.append(
        {
            "figure": figure,
            "panel": panel,
            "title": title,
            "panel_data_file": str(path.relative_to(PACKAGE_ROOT)),
            "n_rows": int(len(df)),
            "n_columns": int(df.shape[1]),
            "source_tables": ";".join(sources),
            "description": description,
        }
    )
    return path


def write_manifest(manifest: pd.DataFrame) -> None:
    """Write the package panel-data manifest in both required locations."""

    manifest["panel_data_file"] = manifest["panel_data_file"].astype(str).str.replace(
        "results/figure_design/panel_data/", "source_data/panel_data/", regex=False
    )
    manifest.to_csv(PANEL_DATA / "panel_data_manifest.tsv", sep="\t", index=False)
    manifest.to_csv(
        PACKAGE_ROOT / "tables" / "Supplementary_Table_S1_panel_level_source_data_manifest.csv",
        index=False,
        encoding="utf-8-sig",
    )


def upsert_manifest_rows(figure: str) -> None:
    """Replace one figure's manifest records without rewriting unrelated panels."""

    records = pd.DataFrame(PANEL_ROWS)
    if records.empty:
        return
    if (PANEL_DATA / "panel_data_manifest.tsv").exists():
        manifest = pd.read_csv(PANEL_DATA / "panel_data_manifest.tsv", sep="\t")
        manifest = manifest.loc[manifest["figure"].ne(figure)].copy()
        manifest = pd.concat([manifest, records], ignore_index=True)
    else:
        manifest = records
    write_manifest(manifest)


def replace_managed_manifest_rows() -> None:
    """Refresh scripted panels while retaining independently generated panels."""

    records = pd.DataFrame(PANEL_ROWS)
    if records.empty:
        return
    manifest_path = PANEL_DATA / "panel_data_manifest.tsv"
    if manifest_path.exists():
        existing = pd.read_csv(manifest_path, sep="\t")
        preserved = existing.loc[~existing["figure"].isin(MANAGED_FIGURES)].copy()
        manifest = pd.concat([preserved, records], ignore_index=True)
    else:
        manifest = records
    write_manifest(manifest)


def wilson_interval(successes: pd.Series, totals: pd.Series, z: float = 1.96) -> tuple[pd.Series, pd.Series]:
    k = pd.to_numeric(successes, errors="coerce").astype(float)
    n = pd.to_numeric(totals, errors="coerce").astype(float)
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt((p * (1 - p) + z**2 / (4 * n)) / n) / denom
    return (center - half).clip(lower=0), (center + half).clip(upper=1)


def signed_log10_p(p: pd.Series, effect: pd.Series) -> pd.Series:
    p = pd.to_numeric(p, errors="coerce").clip(lower=1e-300)
    effect = pd.to_numeric(effect, errors="coerce").fillna(0)
    return -np.log10(p) * np.sign(effect)


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


def spearman_table(df: pd.DataFrame, x_cols: list[str], y_cols: list[str], analysis: str) -> pd.DataFrame:
    rows = []
    for x in x_cols:
        for y in y_cols:
            sub = df[[x, y]].apply(pd.to_numeric, errors="coerce").dropna() if x in df.columns and y in df.columns else pd.DataFrame()
            if x == y and x in df.columns:
                rows.append({"analysis": analysis, "x": x, "y": y, "rho": 1.0, "p_value": 0.0, "n": int(df[x].notna().sum())})
            elif len(sub) >= 4 and sub[x].nunique() > 1 and sub[y].nunique() > 1:
                rho, p = stats.spearmanr(sub[x], sub[y])
                rows.append({"analysis": analysis, "x": x, "y": y, "rho": rho, "p_value": p, "n": len(sub)})
            else:
                rows.append({"analysis": analysis, "x": x, "y": y, "rho": np.nan, "p_value": np.nan, "n": len(sub)})
    out = pd.DataFrame(rows)
    out["fdr"] = benjamini_hochberg(out["p_value"])
    return out


def add_activation_exclusion_indices(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["immune_activation_index"] = out[ACTIVATION_FEATURES].apply(pd.to_numeric, errors="coerce").mean(axis=1)
    out["immune_exclusion_index"] = out[EXCLUSION_FEATURES].apply(pd.to_numeric, errors="coerce").mean(axis=1)
    out["activation_exclusion_balance"] = out["immune_activation_index"] - out["immune_exclusion_index"]
    return out


def add_figure3a_display_fields(df: pd.DataFrame, threshold: float = FIGURE3A_RESEARCH_VALUE_THRESHOLD) -> pd.DataFrame:
    out = df.copy()
    score = pd.to_numeric(out["scTIME_AI_score"], errors="coerce")
    out["research_value_threshold"] = threshold
    out["high_research_value"] = score.ge(threshold)
    out["display_alpha"] = np.where(out["high_research_value"], 0.88, 0.22)
    out["display_size"] = 30 + np.where(out["high_research_value"], 95 * score.fillna(0), 38 * score.fillna(0))
    out["display_marker"] = out["response"].map({"benefit": "o", "no benefit": "^", "unlabeled": "s"}).fillna("s")
    return out


def binary_group_tests(df: pd.DataFrame, group_col: str, features: list[str], positive_label: object = 1.0) -> pd.DataFrame:
    rows = []
    for feature in features:
        columns = [group_col, feature] + (["dataset"] if "dataset" in df.columns else [])
        sub = df[columns].dropna(subset=[group_col, feature]).copy()
        values = pd.to_numeric(sub[feature], errors="coerce")
        sub["_value"] = values
        if "dataset" in sub.columns:
            for _, idx in sub.groupby("dataset", dropna=False).groups.items():
                block = values.loc[idx]
                sd = block.std(ddof=0)
                sub.loc[idx, "_value"] = 0.0 if not np.isfinite(sd) or sd == 0 else (block - block.mean()) / sd
        else:
            sd = values.std(ddof=0)
            sub["_value"] = 0.0 if not np.isfinite(sd) or sd == 0 else (values - values.mean()) / sd
        positive = sub[group_col].astype(str).isin({str(positive_label), "1", "1.0", "True"})
        pos = pd.to_numeric(sub.loc[positive, "_value"], errors="coerce").dropna()
        neg = pd.to_numeric(sub.loc[~positive, "_value"], errors="coerce").dropna()
        if len(pos) >= 2 and len(neg) >= 2:
            u, p = stats.mannwhitneyu(pos, neg, alternative="two-sided")
        else:
            u, p = np.nan, np.nan
        rows.append(
            {
                "feature": feature,
                "positive_mean": pos.mean() if len(pos) else np.nan,
                "negative_mean": neg.mean() if len(neg) else np.nan,
                "mean_difference": (pos.mean() - neg.mean()) if len(pos) and len(neg) else np.nan,
                "mannwhitney_u": u,
                "p_value": p,
                "n_positive": len(pos),
                "n_negative": len(neg),
            }
        )
    out = pd.DataFrame(rows)
    out["fdr"] = benjamini_hochberg(out["p_value"])
    return out


def km_curve(times: np.ndarray, events: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(times)
    times = times[order]
    events = events[order]
    unique_times = np.unique(times[events == 1])
    surv = 1.0
    xs = [0.0]
    ys = [1.0]
    for t in unique_times:
        at_risk = np.sum(times >= t)
        observed = np.sum((times == t) & (events == 1))
        if at_risk > 0:
            surv *= 1 - observed / at_risk
        xs.extend([t, t])
        ys.extend([ys[-1], surv])
    return np.array(xs), np.array(ys)


def bootstrap_prediction_metrics(cv_pred: pd.DataFrame, n_boot: int = 2000, seed: int = 11) -> pd.DataFrame:
    rows = []
    rng = np.random.default_rng(seed)
    groups = [("Pooled labeled cohorts", cv_pred)]
    groups.extend((clean_label(dataset), sub) for dataset, sub in cv_pred.groupby("dataset", observed=True))
    for label, sub in groups:
        sub = sub.dropna(subset=["benefit", "prediction"]).copy()
        if len(sub) < 6 or sub["benefit"].nunique() < 2:
            continue
        y = sub["benefit"].astype(int).to_numpy()
        p = sub["prediction"].astype(float).to_numpy()
        estimates = {"ROC-AUC": roc_auc_score(y, p), "PR-AUC": average_precision_score(y, p), "Brier": np.mean((y - p) ** 2)}
        boot = {metric: [] for metric in estimates}
        for _ in range(n_boot):
            idx = rng.integers(0, len(sub), len(sub))
            if np.unique(y[idx]).size < 2:
                continue
            boot["ROC-AUC"].append(roc_auc_score(y[idx], p[idx]))
            boot["PR-AUC"].append(average_precision_score(y[idx], p[idx]))
            boot["Brier"].append(np.mean((y[idx] - p[idx]) ** 2))
        for metric, estimate in estimates.items():
            clean = pd.Series(boot[metric], dtype=float).dropna()
            rows.append(
                {
                    "validation": label,
                    "metric": metric,
                    "estimate": estimate,
                    "ci_low": float(clean.quantile(0.025)) if len(clean) else np.nan,
                    "ci_high": float(clean.quantile(0.975)) if len(clean) else np.nan,
                    "n": len(sub),
                    "positives": int(y.sum()),
                }
            )
    return pd.DataFrame(rows)


def validation_label(row: pd.Series) -> str:
    if str(row.get("evaluation", "")).startswith("5-fold"):
        return "Pooled CV"
    return f"{clean_label(row.get('train_dataset', ''))} -> {clean_label(row.get('test_dataset', ''))}"


def model_label(value: object) -> str:
    return str(value).replace("_logistic", "").replace("RandomForest", "Random forest").replace("RBF_SVM", "RBF SVM")


def export_figure2() -> None:
    emb = read_table("gse207422_scanpy_cell_embeddings")
    comp = read_table("gse243013_sc_state_composition")
    diff = read_table("gse243013_sc_state_response_differential")
    emb_panel = emb[["UMAP1", "UMAP2", "cell_state"] + [c for c in ["sample"] if c in emb.columns]].copy()
    write_panel_data("Figure 2", "A", "GSE207422 single-cell immune landscape", emb_panel, ["gse207422_scanpy_cell_embeddings"], "UMAP coordinates and cell-state labels.")
    diff_panel = diff.copy()
    diff_panel["signed_log10_p"] = signed_log10_p(diff_panel["p_value"], diff_panel["mean_difference"])
    write_panel_data("Figure 2", "B", "Response-associated state shifts", diff_panel, ["gse243013_sc_state_response_differential"], "State-level MPR versus non-MPR differential fractions.")
    key_states = diff_panel.assign(abs_signal=diff_panel["signed_log10_p"].abs()).sort_values("abs_signal", ascending=False).head(6)["state_category"].tolist()
    comp_panel = comp[comp["state_category"].isin(key_states)].dropna(subset=["benefit"]).copy()
    comp_panel["response"] = np.where(pd.to_numeric(comp_panel["benefit"], errors="coerce").eq(1), "MPR/pCR", "non-MPR")
    write_panel_data("Figure 2", "C", "Patient-level fractions of response-informative states", comp_panel, ["gse243013_sc_state_composition", "gse243013_sc_state_response_differential"], "Patient-level fractions for top response-associated states.")
    heat_panel = comp.copy()
    heat_panel["state_order"] = heat_panel["state_category"].map({state: i for i, state in enumerate(key_states)})
    write_panel_data("Figure 2", "D", "Patient-by-state composition structure", heat_panel, ["gse243013_sc_state_composition"], "Long-format patient-by-state fractions for the heatmap.")
    network = diff_panel.copy()
    network["effect"] = pd.to_numeric(network["mean_difference"], errors="coerce")
    network["signal"] = pd.to_numeric(network["signed_log10_p"], errors="coerce").abs()
    network = network.dropna(subset=["effect", "signal"]).sort_values("signal", ascending=False).head(10).reset_index(drop=True)
    theta = np.linspace(0, 2 * np.pi, len(network), endpoint=False) if len(network) else []
    network["network_x"] = (1.0 + 0.18 * np.sqrt(network["signal"])) * np.cos(theta)
    network["network_y"] = (1.0 + 0.18 * np.sqrt(network["signal"])) * np.sin(theta)
    write_panel_data("Figure 2", "E", "Response-linked single-cell state network", network, ["gse243013_sc_state_response_differential"], "Node table for state network; sign encodes MPR enrichment direction.")


def export_figure3() -> None:
    bulk = read_table("scTIME_AI_scores_all_bulk_cohorts")
    pred = read_table("model_predictions")
    perf = read_table("model_performance")
    cal = read_table("elasticnet_calibration")
    dca = read_table("elasticnet_decision_curve")
    coef = read_table("elasticnet_coefficients")
    contrib = read_table("elasticnet_linear_shap_style_contributions")
    z_features = [f"z_{f}" for f in FEATURES if f"z_{f}" in bulk.columns]
    pca_input = bulk.dropna(subset=z_features).copy()
    coords = PCA(n_components=2, random_state=1).fit_transform(pca_input[z_features])
    pca_df = pca_input[["dataset", "sample", "benefit", "scTIME_AI_score"]].copy()
    pca_df["PC1"] = coords[:, 0]
    pca_df["PC2"] = coords[:, 1]
    pca_df["response"] = np.where(pca_df["benefit"].eq(1), "benefit", np.where(pca_df["benefit"].eq(0), "no benefit", "unlabeled"))
    pca_df = add_figure3a_display_fields(pca_df)
    write_panel_data("Figure 3", "A", "Bulk immune-state PCA", pca_df, ["scTIME_AI_scores_all_bulk_cohorts"], "Bulk samples projected by within-cohort immune module z scores; low-priority points below the scTIME threshold are rendered with reduced alpha.")
    perf_plot = perf.copy()
    perf_plot["validation"] = perf_plot.apply(validation_label, axis=1)
    perf_plot["model_label"] = perf_plot["model"].map(model_label)
    perf_long = perf_plot.melt(id_vars=["validation", "model_label"], value_vars=["roc_auc", "pr_auc", "brier"], var_name="metric", value_name="value")
    perf_long["metric"] = perf_long["metric"].map({"roc_auc": "ROC-AUC", "pr_auc": "PR-AUC", "brier": "Brier"})
    enet_long = perf_long[perf_long["model_label"].eq("ElasticNet")].copy()
    write_panel_data("Figure 3", "B", "scTIME reliability matrix across evaluation designs", enet_long, ["model_performance"], "ElasticNet metrics used in the main reliability matrix.")
    cv_pred = pred[pred["model"].eq("ElasticNet_logistic")].dropna(subset=["benefit", "prediction"]).copy()
    roc_rows = []
    for label, sub in [("Pooled", cv_pred), *[(clean_label(d), s) for d, s in cv_pred.groupby("dataset", observed=True)]]:
        if sub["benefit"].nunique() < 2:
            continue
        y = sub["benefit"].astype(int)
        score = sub["prediction"].astype(float)
        fpr, tpr, threshold = roc_curve(y, score)
        roc_rows.extend({"validation": label, "fpr": a, "tpr": b, "threshold": c, "auc": roc_auc_score(y, score)} for a, b, c in zip(fpr, tpr, threshold))
    write_panel_data("Figure 3", "C", "ElasticNet ROC curves", pd.DataFrame(roc_rows), ["model_predictions"], "ROC curve coordinates for ElasticNet predictions.")
    pr_rows = []
    for label, sub in [("Pooled", cv_pred), *[(clean_label(d), s) for d, s in cv_pred.groupby("dataset", observed=True)]]:
        if sub["benefit"].nunique() < 2:
            continue
        y = sub["benefit"].astype(int)
        score = sub["prediction"].astype(float)
        precision, recall, threshold = precision_recall_curve(y, score)
        threshold = np.r_[threshold, np.nan]
        pr_rows.extend({"validation": label, "recall": a, "precision": b, "threshold": c, "average_precision": average_precision_score(y, score)} for a, b, c in zip(recall, precision, threshold))
    write_panel_data("Figure 3", "D", "ElasticNet precision-recall curves", pd.DataFrame(pr_rows), ["model_predictions"], "Precision-recall coordinates for ElasticNet predictions.")
    cal = cal.copy()
    if not cal.empty:
        cal["ci_low"], cal["ci_high"] = wilson_interval(cal["observed_rate"] * cal["n"], cal["n"])
    write_panel_data("Figure 3", "E", "Calibration with Wilson intervals", cal, ["elasticnet_calibration"], "Calibration-bin observed response rates with Wilson intervals.")
    write_panel_data("Figure 3", "F", "Decision-curve analysis", dca, ["elasticnet_decision_curve"], "Decision-curve net benefit across threshold probabilities.")
    boot = bootstrap_prediction_metrics(cv_pred)
    write_panel_data("Figure 3", "G", "Bootstrap reliability intervals", boot, ["model_predictions"], "Bootstrap confidence intervals for ElasticNet prediction metrics.")
    write_panel_data("Figure 3", "H", "Interpretable scTIME coefficients", coef, ["elasticnet_coefficients"], "ElasticNet feature coefficients.")
    labeled = bulk[bulk["dataset"].isin(["GSE126044", "GSE207422_bulk"])].dropna(subset=["benefit"]).copy()
    contribution = contrib.merge(labeled[["dataset", "sample", "benefit"]], on=["dataset", "sample"], how="left").sort_values(["benefit", "prediction"], ascending=[False, False])
    write_panel_data("Figure 3", "I", "Patient-level feature contribution profile", contribution, ["elasticnet_linear_shap_style_contributions", "scTIME_AI_scores_all_bulk_cohorts"], "Per-patient linear logit contributions from the final ElasticNet model.")


def export_figure4() -> None:
    bulk = read_table("scTIME_AI_scores_all_bulk_cohorts")
    pseudo = read_table("gse207422_sc_pseudobulk_signature_scores")
    lr = read_table("gse207422_lr_axis_scores")
    ligand = read_table("gse207422_nichenet_like_ligand_activity")
    tcga = read_table("tcga_sctime_ai_scores")
    cptac = read_table("cptac_sctime_ai_projection")
    labeled = bulk[bulk["dataset"].isin(["GSE126044", "GSE207422_bulk"])].dropna(subset=["benefit"]).copy()
    labeled["response"] = np.where(labeled["benefit"].eq(1), "benefit/MPR", "no benefit")
    labeled["IFN_antigen_index"] = labeled[["IFN_response", "Antigen_presentation"]].mean(axis=1)
    labeled["CXCL10_CXCR3_axis"] = labeled["z_CXCL10_CXCR3_axis"]
    test_features = ["CXCL10_CXCR3_axis", "IFN_response", "Antigen_presentation", "Cytotoxic_CD8", "CD274_expr"]
    tests = read_table("feature_response_associations")
    tests = tests[tests["feature"].isin(test_features)].rename(
        columns={"positive_mean_z": "positive_mean", "negative_mean_z": "negative_mean"}
    )
    write_panel_data("Figure 4", "A", "CXCL10-CXCR3 axis by response", labeled[["dataset", "sample", "benefit", "response", "CXCL10_CXCR3_axis"]], ["scTIME_AI_scores_all_bulk_cohorts"], "Sample-level CXCL10-CXCR3 scores by response.")
    write_panel_data("Figure 4", "B", "IFN-antigen index links to CXCL10-CXCR3", labeled[["dataset", "sample", "benefit", "IFN_antigen_index", "CXCL10_CXCR3_axis", "IFN_response", "Antigen_presentation"]], ["scTIME_AI_scores_all_bulk_cohorts"], "Sample-level IFN-antigen index and CXCL10-CXCR3 axis scores.")
    lr_cxcl = lr[lr["axis"].eq("CXCL9/10/11-CXCR3")].dropna(subset=["benefit"]).copy()
    write_panel_data("Figure 4", "C", "Single-cell CXCL9/10/11-CXCR3 LR score", lr_cxcl, ["gse207422_lr_axis_scores"], "Single-cell ligand-receptor axis scores.")
    lig_panel = ligand.copy()
    lig_panel["minus_log10_p"] = -np.log10(pd.to_numeric(lig_panel["spearman_p_value"], errors="coerce").clip(lower=1e-300))
    write_panel_data("Figure 4", "D", "Ligand activity against effector module", lig_panel, ["gse207422_nichenet_like_ligand_activity"], "Ligand activity and effector-module association statistics.")
    corr_frames = [
        spearman_table(labeled, ["CXCL10_CXCR3_axis"], ["IFN_response", "Antigen_presentation", "Cytotoxic_CD8", "CD274_expr", "scTIME_AI_score"], "GEO labeled"),
        spearman_table(pseudo, ["CXCL10_CXCR3_axis"], ["IFN_response", "Antigen_presentation", "Cytotoxic_CD8", "CD274_expr", "scTIME_AI_score"], "sc pseudobulk"),
    ]
    corr_frames.extend(
        spearman_table(sub, ["CXCL10_CXCR3_axis"], ["IFN_response", "Antigen_presentation", "Cytotoxic_CD8", "CD274_expr", "scTIME_AI_score"], f"TCGA {project}")
        for project, sub in tcga.groupby("project", observed=True)
    )
    corr_frames.extend(
        spearman_table(sub, ["CXCL10_CXCR3_axis"], ["IFN_response", "Antigen_presentation", "Cytotoxic_CD8", "CD274_expr", "scTIME_AI_score"], f"CPTAC {cohort} {omics}")
        for (cohort, omics), sub in cptac.groupby(["cohort", "omics"], observed=True)
    )
    corr = pd.concat(corr_frames, ignore_index=True)
    write_panel_data("Figure 4", "E", "CXCL10-CXCR3 cross-context correlations", corr, ["scTIME_AI_scores_all_bulk_cohorts", "gse207422_sc_pseudobulk_signature_scores", "tcga_sctime_ai_scores", "cptac_sctime_ai_projection"], "Cross-context Spearman correlations for the CXCL10-CXCR3 axis.")
    tests.to_csv(DESIGN / "figure4_cxcl10_ifn_axis_response_tests.tsv", sep="\t", index=False)


def export_figure5() -> None:
    bulk = read_table("scTIME_AI_scores_all_bulk_cohorts")
    fractions = read_table("music_style_nnls_cell_fractions")
    tcga = add_activation_exclusion_indices(read_table("tcga_sctime_ai_scores"))
    cptac = add_activation_exclusion_indices(read_table("cptac_sctime_ai_projection"))
    labeled = bulk[bulk["dataset"].isin(["GSE126044", "GSE207422_bulk"])].dropna(subset=["benefit"]).copy()
    exclusion_features = ["SPP1_macrophage"] + EXCLUSION_FEATURES
    exclusion_tests = read_table("feature_response_associations")
    exclusion_tests = exclusion_tests[exclusion_tests["feature"].isin(exclusion_features)].rename(
        columns={"positive_mean_z": "positive_mean", "negative_mean_z": "negative_mean"}
    )
    write_panel_data("Figure 5", "A", "Exclusion-feature response effects", exclusion_tests, ["scTIME_AI_scores_all_bulk_cohorts"], "Responder minus non-responder differences for exclusion-associated features.")
    deconv = fractions.merge(bulk[["dataset", "sample", "benefit", "SPP1_macrophage", "TGF_beta_EMT", "Hypoxia", "scTIME_AI_score"]], on=["dataset", "sample"], how="inner")
    deconv_panel = deconv[["dataset", "sample", "benefit", "Myeloid/macrophage", "SPP1_macrophage"]].copy()
    deconv_stats = []
    for dataset, cohort in deconv_panel.groupby("dataset", observed=True):
        sub = cohort[["Myeloid/macrophage", "SPP1_macrophage"]].dropna()
        rho, p_value = stats.spearmanr(sub["Myeloid/macrophage"], sub["SPP1_macrophage"])
        deconv_stats.append(
            {
                "dataset": dataset,
                "cohort_n": len(sub),
                "cohort_spearman_rho": rho,
                "cohort_p_value": p_value,
            }
        )
    deconv_panel = deconv_panel.merge(pd.DataFrame(deconv_stats), on="dataset", how="left")
    write_panel_data(
        "Figure 5",
        "B",
        "Myeloid deconvolution moderately aligns with SPP1 macrophage score",
        deconv_panel,
        ["music_style_nnls_cell_fractions", "scTIME_AI_scores_all_bulk_cohorts"],
        "Bulk NNLS myeloid fraction and SPP1 macrophage module score with cohort-stratified Spearman statistics; interpreted as a consistency check rather than standalone validation.",
    )
    deconv_corr = pd.concat(
        [
            spearman_table(
                sub,
                ["Myeloid/macrophage", "Fibroblast", "Tumor/epithelial"],
                ["SPP1_macrophage", "TGF_beta_EMT", "Hypoxia", "scTIME_AI_score"],
                f"bulk NNLS {dataset}",
            )
            for dataset, sub in deconv.groupby("dataset", observed=True)
        ],
        ignore_index=True,
    )
    write_panel_data("Figure 5", "C", "Deconvolution-module correlation matrix", deconv_corr, ["music_style_nnls_cell_fractions", "scTIME_AI_scores_all_bulk_cohorts"], "Deconvolution-module Spearman correlation matrix.")
    tcga_panel = tcga[["project", "sample", "immune_exclusion_index", "scTIME_AI_score"]].dropna().copy()
    tcga_stats = []
    for project, cohort in tcga_panel.groupby("project", observed=True):
        rho, p_value = stats.spearmanr(cohort["immune_exclusion_index"], cohort["scTIME_AI_score"])
        tcga_stats.append(
            {
                "project": project,
                "project_n": len(cohort),
                "project_spearman_rho": rho,
                "project_p_value": p_value,
            }
        )
    tcga_panel = tcga_panel.merge(pd.DataFrame(tcga_stats), on="project", how="left")
    write_panel_data("Figure 5", "D", "TCGA exclusion-index density vs scTIME", tcga_panel, ["tcga_sctime_ai_scores"], "TCGA sample-level exclusion index and scTIME scores with project-stratified Spearman statistics.")
    cptac_panel = cptac[cptac["stage_group"].isin(["Early", "Advanced"])].copy()
    cptac_panel["cohort_omics"] = cptac_panel["cohort"] + "\n" + cptac_panel["omics"]
    write_panel_data("Figure 5", "E", "CPTAC exclusion index by stage group", cptac_panel, ["cptac_sctime_ai_projection"], "CPTAC sample-level exclusion index by stage group.")
    tcga_corr = pd.concat(
        [
            spearman_table(
                sub,
                EXCLUSION_FEATURES + ["immune_exclusion_index"],
                EXCLUSION_FEATURES + ["scTIME_AI_score"],
                f"TCGA exclusion {project}",
            )
            for project, sub in tcga.groupby("project", observed=True)
        ],
        ignore_index=True,
    )
    write_panel_data("Figure 5", "F", "TCGA exclusion co-program structure", tcga_corr, ["tcga_sctime_ai_scores"], "TCGA exclusion-feature Spearman correlation matrix.")


def export_figure6() -> None:
    comp = read_table("gse243013_sc_state_composition")
    tcr = read_table("gse243013_tcr_clonality")
    labeled = prepare_heldout_figure6_labeled(TABLES)
    dataset_marker_map = {dataset: marker for dataset, marker in zip(sorted(labeled["dataset"].dropna().unique()), ["s", "^", "o", "D"])}
    panel_a = panel_a_frame(labeled)
    panel_a["display_marker"] = panel_a["dataset"].map(dataset_marker_map)
    write_panel_data(
        "Figure 6",
        "A",
        "CD274 and held-out scTIME scores in ICI cohorts",
        panel_a,
        ["model_predictions", "scTIME_AI_scores_all_bulk_cohorts"],
        "Labeled ICI cohorts with pooled five-fold out-of-fold ElasticNet probabilities. Final-refit scores are audit-only and are not used for plotting or stratification.",
    )
    stratified = build_stratification_table(labeled)
    write_panel_data(
        "Figure 6",
        "B",
        "Matched within-cohort scTIME OOF and CD274 response stratification",
        stratified,
        ["model_predictions", "scTIME_AI_scores_all_bulk_cohorts"],
        "Matched within-cohort quartile and median splits for both held-out scTIME probabilities and CD274 expression-module values, pooled only after group assignment.",
    )
    contrast = build_feature_contrast(labeled)
    write_panel_data(
        "Figure 6",
        "C",
        "CD274-high low-OOF-scTIME feature contrast",
        contrast,
        ["model_predictions", "scTIME_AI_scores_all_bulk_cohorts"],
        "Feature contrast between within-cohort OOF-scTIME-high samples and CD274-high/OOF-scTIME-low samples; retained as descriptive context.",
    )
    comp_wide = comp.pivot_table(index="sampleID", columns="state_category", values="fraction", fill_value=0).reset_index()
    tcr_merged = tcr.merge(comp_wide, on="sampleID", how="left")
    tcr_metrics = ["top_clonotype_fraction", "expanded_fraction", "normalized_shannon"]
    state_cols = [c for c in comp_wide.columns if c != "sampleID"]
    all_tcr_corr = spearman_table(tcr_merged, tcr_metrics, state_cols, "GSE243013")
    state_mean = comp_wide[state_cols].mean().rename("mean_fraction")
    immune_cols = (
        all_tcr_corr.assign(abs_rho=all_tcr_corr["rho"].abs())
        .groupby("y", observed=True)["abs_rho"]
        .max()
        .to_frame()
        .join(state_mean)
        .query("mean_fraction >= 0.005")
        .sort_values(["abs_rho", "mean_fraction"], ascending=False)
        .head(8)
        .index.tolist()
    )
    tcr_corr = all_tcr_corr[all_tcr_corr["y"].isin(immune_cols)].copy()
    write_panel_data("Figure 6", "D", "TCR clonality and immune-cell composition", tcr_corr, ["gse243013_tcr_clonality", "gse243013_sc_state_composition"], "Spearman correlations between TCR metrics and immune-cell fractions.")
    tcr_plot = tcr.dropna(subset=["benefit"]).melt(
        id_vars=["sampleID", "benefit"],
        value_vars=tcr_metrics,
        var_name="metric",
        value_name="value",
    )
    tcr_plot["response"] = np.where(tcr_plot["benefit"].eq(1), "MPR/pCR", "non-MPR")
    write_panel_data("Figure 6", "E", "TCR clonality metrics by pathological response", tcr_plot, ["gse243013_tcr_clonality"], "Patient-level TCR clonality metrics by pathological response.")


def export_figure7() -> None:
    tcga_status = read_table("tcga_secondary_validation")
    tcga = read_table("tcga_sctime_ai_scores")
    cptac = read_table("cptac_sctime_ai_projection")
    cptac_stage = read_table("cptac_stage_associations")
    gse135 = read_table("gse135222_survival_validation")
    gse_scores = read_table("scTIME_AI_scores_all_bulk_cohorts")
    g135 = gse_scores[gse_scores["dataset"].eq("GSE135222")].dropna(subset=["pfs_time", "pfs_event", "scTIME_AI_score"]).copy()
    km_rows = []
    if not g135.empty:
        g135["score_group"] = np.where(g135["scTIME_AI_score"] >= g135["scTIME_AI_score"].median(), "High", "Low")
        for label, sub in g135.groupby("score_group", observed=True):
            t, s = km_curve(sub["pfs_time"].astype(float).to_numpy(), sub["pfs_event"].astype(int).to_numpy())
            km_rows.extend({"score_group": label, "time": time, "survival": surv, "n": len(sub)} for time, surv in zip(t, s))
    write_panel_data("Figure 7", "A", "GSE135222 exploratory PFS association", pd.DataFrame(km_rows), ["scTIME_AI_scores_all_bulk_cohorts", "gse135222_survival_validation"], "Kaplan-Meier coordinates for the exploratory GSE135222 scTIME median split; this is not independent prognostic validation.")
    forest_rows = []
    row = gse135[gse135["analysis"].str.contains("Cox", na=False)].head(1)
    if not row.empty:
        forest_rows.append({"analysis": "GSE135222 exploratory PFS association", "hr": float(row["hazard_ratio"].iloc[0]), "p": float(row["p_value"].iloc[0])})
    for _, row in tcga_status.iterrows():
        forest_rows.append({"analysis": f"{row['project']} OS", "hr": row["cox_hr"], "p": row["cox_p"]})
    write_panel_data("Figure 7", "B", "Survival association summary", pd.DataFrame(forest_rows), ["gse135222_survival_validation", "tcga_secondary_validation"], "Hazard-ratio summary for survival association panels.")
    write_panel_data("Figure 7", "C", "TCGA-LUAD LUSC scTIME distributions", tcga.dropna(subset=["project", "scTIME_AI_score"]), ["tcga_sctime_ai_scores"], "TCGA sample-level scTIME score distribution.")
    cptac_panel = cptac[cptac["stage_group"].isin(["Early", "Advanced"])].copy()
    cptac_panel["cohort_omics"] = cptac_panel["cohort"] + "\n" + cptac_panel["omics"]
    write_panel_data("Figure 7", "D", "CPTAC proteome phosphoproteome projection", cptac_panel, ["cptac_sctime_ai_projection"], "CPTAC sample-level scTIME projections by stage group.")
    key = ["scTIME_AI_score", "IFN_response", "Antigen_presentation", "Cytotoxic_CD8", "CXCL10_CXCR3_axis", "SPP1_macrophage", "TGF_beta_EMT", "Hypoxia", "CD274_expr"]
    stage_plot = cptac_stage[cptac_stage["feature"].isin(key)].copy()
    write_panel_data("Figure 7", "E", "CPTAC stage shifts in key immune programs", stage_plot, ["cptac_stage_associations"], "Advanced-minus-early stage shifts for key immune programs.")


def export_supplementary_figure1() -> None:
    dataset_summary = read_table("dataset_summary")
    tcga = read_table("tcga_secondary_validation")
    cptac = read_table("cptac_secondary_validation_status")
    gse243 = read_table("gse243013_sc_patient_summary")
    model = read_table("model_performance")
    cal = read_table("elasticnet_calibration")
    dca = read_table("elasticnet_decision_curve")
    cohort_rows = []
    for _, row in dataset_summary.iterrows():
        cohort_rows.append({"cohort": row["dataset"], "modality": "bulk RNA-seq", "samples": row.get("n_samples", row.get("samples")), "role": "training/validation" if row["dataset"] in {"GSE126044", "GSE207422_bulk"} else "projection/survival"})
    cohort_rows.append({"cohort": "GSE243013", "modality": "scRNA-seq/TCR", "samples": len(gse243), "role": "single-cell discovery"})
    for _, row in tcga.iterrows():
        cohort_rows.append({"cohort": row["project"], "modality": "bulk RNA-seq", "samples": row["n_samples"], "role": "secondary validation"})
    for (cohort, omics), sub in cptac.groupby(["cohort", "omics"], observed=True):
        cohort_rows.append({"cohort": f"{cohort} {omics}", "modality": omics, "samples": sub["n_samples"].max(), "role": "proteomic validation"})
    write_panel_data("Supplementary Figure 1", "A", "Cohort sample audit", pd.DataFrame(cohort_rows), ["dataset_summary", "gse243013_sc_patient_summary", "tcga_secondary_validation", "cptac_secondary_validation_status"], "Cohort sample counts and study roles.")
    perf = model.copy()
    perf["label"] = perf["model"] + " | " + perf["evaluation"].str.replace(" validation", "", regex=False)
    write_panel_data("Supplementary Figure 1", "B", "Model performance across evaluation designs", perf, ["model_performance"], "All model-performance rows used for the audit scatter plot.")
    write_panel_data("Supplementary Figure 1", "C", "Calibration", cal, ["elasticnet_calibration"], "Calibration bins for the final ElasticNet model.")
    write_panel_data("Supplementary Figure 1", "D", "Decision curve", dca, ["elasticnet_decision_curve"], "Decision-curve rows for the final ElasticNet model.")


def export_supplementary_figure2() -> None:
    bulk_cov = read_table("signature_gene_coverage")
    cptac_cov = read_table("cptac_signature_gene_coverage")
    cluster_scores = read_table("gse207422_scanpy_cluster_marker_scores")
    cluster_summary = read_table("gse207422_scanpy_cluster_summary")
    lr = read_table("gse207422_lr_axis_scores")
    fractions = read_table("music_style_nnls_cell_fractions")
    bulk = read_table("scTIME_AI_scores_all_bulk_cohorts")
    sc_pseudo = read_table("gse207422_sc_pseudobulk_signature_scores")
    bulk_scores = read_table("bulk_signature_scores")
    ref = pd.read_csv(TABLES / "music_style_reference_signature_matrix.tsv", sep="\t", index_col=0)
    marker_cols = [c for c in cluster_scores.columns if c.startswith("score_")]
    cluster_heat = cluster_scores.set_index("leiden")[marker_cols]
    cluster_heat.columns = [c.replace("score_", "") for c in cluster_heat.columns]
    cluster_heat = cluster_heat.join(cluster_summary.set_index("leiden")[["cell_state", "n_cells"]]).reset_index()
    write_panel_data("Supplementary Figure 2", "A", "Cluster marker-program audit", cluster_heat, ["gse207422_scanpy_cluster_marker_scores", "gse207422_scanpy_cluster_summary"], "Cluster-level marker-program scores and annotations.")
    bulk_cov["coverage"] = bulk_cov["available_genes"] / bulk_cov["total_genes"]
    cptac_cov["coverage"] = cptac_cov["available_genes"] / cptac_cov["total_genes"]
    cptac_cov["dataset"] = cptac_cov["cohort"] + " " + cptac_cov["omics"]
    coverage_long = pd.concat([bulk_cov[["dataset", "module", "coverage"]].assign(source="bulk RNA-seq"), cptac_cov[["dataset", "module", "coverage"]].assign(source="CPTAC")], ignore_index=True)
    write_panel_data("Supplementary Figure 2", "B", "Bulk CPTAC signature coverage", coverage_long, ["signature_gene_coverage", "cptac_signature_gene_coverage"], "Gene coverage for immune modules across bulk and CPTAC datasets.")
    lr_mat = lr.pivot_table(index="axis", columns="sample", values="interaction_score", fill_value=0).reset_index()
    write_panel_data("Supplementary Figure 2", "C", "LR axis activity matrix", lr_mat, ["gse207422_lr_axis_scores"], "Ligand-receptor axis-by-sample interaction-score matrix.")
    frac_merged = fractions.merge(bulk[["dataset", "sample", "scTIME_AI_score", "benefit", "Cytotoxic_CD8", "SPP1_macrophage"]], on=["dataset", "sample"], how="left")
    frac_corr = spearman_table(frac_merged, ["CD8 T", "Myeloid/macrophage", "Tumor/epithelial", "Fibroblast"], ["Cytotoxic_CD8", "SPP1_macrophage", "scTIME_AI_score"], "NNLS-vs-bulk")
    write_panel_data("Supplementary Figure 2", "D", "Deconvolution-vs-module correlations", frac_corr, ["music_style_nnls_cell_fractions", "scTIME_AI_scores_all_bulk_cohorts"], "NNLS fraction versus module-score correlations.")
    ref_corr = ref.corr(method="spearman").reset_index().rename(columns={"index": "reference_signature"})
    write_panel_data("Supplementary Figure 2", "E", "Reference signature separability", ref_corr, ["music_style_reference_signature_matrix"], "Reference-signature Spearman correlation matrix.")
    overlap_patients = sorted(set(sc_pseudo["patient"].dropna()) & set(bulk_scores.loc[bulk_scores["dataset"].eq("GSE207422_bulk"), "patient"].dropna()))
    concord_rows = []
    if overlap_patients:
        sc_overlap = sc_pseudo[sc_pseudo["patient"].isin(overlap_patients)].set_index("patient")
        bulk_overlap = bulk_scores[(bulk_scores["dataset"].eq("GSE207422_bulk")) & (bulk_scores["patient"].isin(overlap_patients))].set_index("patient")
        for feature in FEATURES:
            sub = pd.concat([sc_overlap[feature].rename("single_cell"), bulk_overlap[feature].rename("bulk")], axis=1).dropna()
            rho, p = stats.spearmanr(sub["single_cell"], sub["bulk"]) if len(sub) >= 3 else (np.nan, np.nan)
            concord_rows.append({"feature": feature, "rho": rho, "p_value": p, "n_overlap_patients": len(sub)})
    write_panel_data("Supplementary Figure 2", "F", "Matched scRNA-bulk signature concordance", pd.DataFrame(concord_rows), ["gse207422_sc_pseudobulk_signature_scores", "bulk_signature_scores"], "Matched-patient scRNA-pseudobulk versus bulk signature concordance.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--figure6",
        action="store_true",
        help="Refresh only Figure 6 source tables and panel-data manifest rows.",
    )
    parser.add_argument(
        "--figure5",
        action="store_true",
        help="Refresh only Figure 5 source tables and panel-data manifest rows.",
    )
    args = parser.parse_args()
    PANEL_DATA.mkdir(parents=True, exist_ok=True)
    if args.figure6:
        export_figure6()
        upsert_manifest_rows("Figure 6")
        return 0
    if args.figure5:
        export_figure5()
        upsert_manifest_rows("Figure 5")
        return 0
    export_figure2()
    export_figure3()
    export_figure4()
    export_figure5()
    export_figure6()
    export_figure7()
    export_supplementary_figure1()
    export_supplementary_figure2()
    replace_managed_manifest_rows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
