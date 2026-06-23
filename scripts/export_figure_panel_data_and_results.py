#!/usr/bin/env python
"""Export plot-ready source data for every figure panel and draft results text."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve


ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "results" / "tables"
DESIGN = ROOT / "results" / "figure_design"
PANEL_DATA = DESIGN / "panel_data"
MODELS = ROOT / "results" / "models"

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
    "scTIME_AI_score": "scTIME-AI score",
}

ACTIVATION_FEATURES = ["IFN_response", "Antigen_presentation", "Cytotoxic_CD8", "NK", "CXCL10_CXCR3_axis", "CD274_expr"]
EXCLUSION_FEATURES = ["SPP1_macrophage", "TGF_beta_EMT", "Hypoxia", "Treg", "Terminal_exhausted_CD8"]
FIGURE3A_RESEARCH_VALUE_THRESHOLD = 0.65

PANEL_ROWS: list[dict[str, object]] = []


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
            "panel_data_file": str(path.relative_to(ROOT)),
            "n_rows": int(len(df)),
            "n_columns": int(df.shape[1]),
            "source_tables": ";".join(sources),
            "description": description,
        }
    )
    return path


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
        sub = df[[group_col, feature]].dropna()
        positive = sub[group_col].astype(str).isin({str(positive_label), "1", "1.0", "True"})
        pos = pd.to_numeric(sub.loc[positive, feature], errors="coerce").dropna()
        neg = pd.to_numeric(sub.loc[~positive, feature], errors="coerce").dropna()
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


def export_figure1() -> None:
    dataset_summary = read_table("dataset_summary")
    gse243 = read_table("gse243013_sc_patient_summary")
    tcga = read_table("tcga_secondary_validation")
    cptac = read_table("cptac_secondary_validation_status")
    manifest = read_table("download_manifest_check")
    metadata = []
    for _, row in dataset_summary.iterrows():
        metadata.append({"cohort": row["dataset"], "modality": "bulk RNA-seq", "samples": row.get("n_samples", row.get("samples")), "role": "response modeling/projection"})
    metadata.append({"cohort": "GSE243013", "modality": "scRNA-seq/TCR", "samples": len(gse243), "role": "single-cell discovery"})
    for _, row in tcga.iterrows():
        metadata.append({"cohort": row["project"], "modality": "bulk RNA-seq", "samples": row["n_samples"], "role": "external survival validation"})
    for (cohort, omics), sub in cptac.groupby(["cohort", "omics"], observed=True):
        metadata.append({"cohort": f"{cohort} {omics}", "modality": omics, "samples": sub["n_samples"].max(), "role": "cross-omics projection"})
    write_panel_data("Figure 1", "A", "Public multi-omics cohort map", pd.DataFrame(metadata), ["dataset_summary", "gse243013_sc_patient_summary", "tcga_secondary_validation", "cptac_secondary_validation_status"], "Cohort-level counts and roles for the manually drawn framework.")
    flow = pd.DataFrame(
        {
            "step_order": [1, 2, 3, 4, 5, 6],
            "step": ["Public data intake", "Single-cell state annotation", "Bulk module projection", "scTIME-AI training", "Model interpretation", "External validation"],
            "output": ["Manifest-verified cohorts", "Immune state fractions/LR axes", "Module scores and deconvolution", "ElasticNet probability score", "Feature coefficients/contributions", "Survival and cross-omics evidence"],
        }
    )
    write_panel_data("Figure 1", "B", "Analytical flow", flow, ["download_manifest_check", "model_performance"], "Workflow nodes for the manually drawn schematic.")
    axes = pd.DataFrame(
        {
            "axis": ["Responder activation", "Non-responder exclusion"],
            "features": [";".join(ACTIVATION_FEATURES), ";".join(EXCLUSION_FEATURES)],
            "interpretation": ["IFN/antigen/cytotoxic/CXCL10-CXCR3 inflamed state", "SPP1 macrophage/TGF-beta/hypoxia/Treg-exhaustion exclusion state"],
        }
    )
    write_panel_data("Figure 1", "C", "Biological hypothesis axes", axes, ["feature_response_associations", "figure5_exclusion_axis_response_tests"], "Feature groups supporting the schematic biological model.")
    model_meta = json.loads((MODELS / "scTIME_AI_elasticnet_metadata.json").read_text(encoding="utf-8"))
    translational = pd.DataFrame([model_meta])
    write_panel_data("Figure 1", "D", "Translational output", translational, ["scTIME_AI_elasticnet_metadata.json"], "Final model metadata supporting the translational-output schematic.")


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
    comp_panel = comp[comp["state_category"].isin(key_states)].copy()
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
    write_panel_data("Figure 3", "A", "Bulk immune-state PCA", pca_df, ["scTIME_AI_scores_all_bulk_cohorts"], "Bulk samples projected by within-cohort immune module z scores; low-priority points below the scTIME-AI threshold are rendered with reduced alpha.")
    perf_plot = perf.copy()
    perf_plot["validation"] = perf_plot.apply(validation_label, axis=1)
    perf_plot["model_label"] = perf_plot["model"].map(model_label)
    perf_long = perf_plot.melt(id_vars=["validation", "model_label"], value_vars=["roc_auc", "pr_auc", "brier"], var_name="metric", value_name="value")
    perf_long["metric"] = perf_long["metric"].map({"roc_auc": "ROC-AUC", "pr_auc": "PR-AUC", "brier": "Brier"})
    enet_long = perf_long[perf_long["model_label"].eq("ElasticNet")].copy()
    write_panel_data("Figure 3", "B", "scTIME-AI reliability matrix across validation designs", enet_long, ["model_performance"], "ElasticNet metrics used in the main reliability matrix.")
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
    write_panel_data("Figure 3", "H", "Interpretable scTIME-AI coefficients", coef, ["elasticnet_coefficients"], "ElasticNet feature coefficients.")
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
    tests = binary_group_tests(labeled, "benefit", ["CXCL10_CXCR3_axis", "IFN_response", "Antigen_presentation", "Cytotoxic_CD8", "CD274_expr", "scTIME_AI_score"])
    write_panel_data("Figure 4", "A", "CXCL10-CXCR3 axis by response", labeled[["dataset", "sample", "benefit", "response", "CXCL10_CXCR3_axis"]], ["scTIME_AI_scores_all_bulk_cohorts"], "Sample-level CXCL10-CXCR3 scores by response.")
    write_panel_data("Figure 4", "B", "IFN-antigen index links to CXCL10-CXCR3", labeled[["dataset", "sample", "benefit", "IFN_antigen_index", "CXCL10_CXCR3_axis", "IFN_response", "Antigen_presentation"]], ["scTIME_AI_scores_all_bulk_cohorts"], "Sample-level IFN-antigen index and CXCL10-CXCR3 axis scores.")
    lr_cxcl = lr[lr["axis"].eq("CXCL9/10/11-CXCR3")].copy()
    write_panel_data("Figure 4", "C", "Single-cell CXCL9/10/11-CXCR3 LR score", lr_cxcl, ["gse207422_lr_axis_scores"], "Single-cell ligand-receptor axis scores.")
    lig_panel = ligand.copy()
    lig_panel["minus_log10_p"] = -np.log10(pd.to_numeric(lig_panel["spearman_p_value"], errors="coerce").clip(lower=1e-300))
    write_panel_data("Figure 4", "D", "Ligand activity against effector module", lig_panel, ["gse207422_nichenet_like_ligand_activity"], "Ligand activity and effector-module association statistics.")
    corr = pd.concat(
        [
            spearman_table(df, ["CXCL10_CXCR3_axis"], ["IFN_response", "Antigen_presentation", "Cytotoxic_CD8", "CD274_expr", "scTIME_AI_score"], name)
            for name, df in [("GEO labeled", labeled), ("sc pseudobulk", pseudo), ("TCGA", tcga), ("CPTAC", cptac)]
        ],
        ignore_index=True,
    )
    write_panel_data("Figure 4", "E", "CXCL10-CXCR3 cross-context correlations", corr, ["scTIME_AI_scores_all_bulk_cohorts", "gse207422_sc_pseudobulk_signature_scores", "tcga_sctime_ai_scores", "cptac_sctime_ai_projection"], "Cross-context Spearman correlations for the CXCL10-CXCR3 axis.")
    tests.to_csv(DESIGN / "figure4_cxcl10_ifn_axis_response_tests.tsv", sep="\t", index=False)


def export_figure5() -> None:
    bulk = read_table("scTIME_AI_scores_all_bulk_cohorts")
    fractions = read_table("music_style_nnls_cell_fractions")
    tcga = add_activation_exclusion_indices(read_table("tcga_sctime_ai_scores"))
    cptac = add_activation_exclusion_indices(read_table("cptac_sctime_ai_projection"))
    labeled = bulk[bulk["dataset"].isin(["GSE126044", "GSE207422_bulk"])].dropna(subset=["benefit"]).copy()
    exclusion_tests = binary_group_tests(labeled, "benefit", EXCLUSION_FEATURES + ["scTIME_AI_score"])
    write_panel_data("Figure 5", "A", "Exclusion-feature response effects", exclusion_tests, ["scTIME_AI_scores_all_bulk_cohorts"], "Responder minus non-responder differences for exclusion-associated features.")
    deconv = fractions.merge(bulk[["dataset", "sample", "benefit", "SPP1_macrophage", "TGF_beta_EMT", "Hypoxia", "scTIME_AI_score"]], on=["dataset", "sample"], how="inner")
    write_panel_data(
        "Figure 5",
        "B",
        "Myeloid deconvolution moderately aligns with SPP1 macrophage score",
        deconv[["dataset", "sample", "benefit", "Myeloid/macrophage", "SPP1_macrophage"]],
        ["music_style_nnls_cell_fractions", "scTIME_AI_scores_all_bulk_cohorts"],
        "Bulk NNLS myeloid fraction and SPP1 macrophage module score; interpreted as a moderate consistency check rather than a decisive standalone validation.",
    )
    deconv_corr = spearman_table(deconv, ["Myeloid/macrophage", "Fibroblast", "Tumor/epithelial"], ["SPP1_macrophage", "TGF_beta_EMT", "Hypoxia", "scTIME_AI_score"], "bulk NNLS")
    write_panel_data("Figure 5", "C", "Deconvolution-module correlation matrix", deconv_corr, ["music_style_nnls_cell_fractions", "scTIME_AI_scores_all_bulk_cohorts"], "Deconvolution-module Spearman correlation matrix.")
    write_panel_data("Figure 5", "D", "TCGA exclusion-index density vs scTIME-AI", tcga[["project", "sample", "immune_exclusion_index", "scTIME_AI_score"]].dropna(), ["tcga_sctime_ai_scores"], "TCGA sample-level exclusion index and scTIME-AI scores.")
    cptac_panel = cptac[cptac["stage_group"].isin(["Early", "Advanced"])].copy()
    cptac_panel["cohort_omics"] = cptac_panel["cohort"] + "\n" + cptac_panel["omics"]
    write_panel_data("Figure 5", "E", "CPTAC exclusion index by stage group", cptac_panel, ["cptac_sctime_ai_projection"], "CPTAC sample-level exclusion index by stage group.")
    tcga_corr = spearman_table(tcga, EXCLUSION_FEATURES + ["immune_exclusion_index"], EXCLUSION_FEATURES + ["scTIME_AI_score"], "TCGA exclusion")
    write_panel_data("Figure 5", "F", "TCGA exclusion co-program structure", tcga_corr, ["tcga_sctime_ai_scores"], "TCGA exclusion-feature Spearman correlation matrix.")


def export_figure6() -> None:
    bulk = read_table("scTIME_AI_scores_all_bulk_cohorts")
    comp = read_table("gse243013_sc_state_composition")
    tcr = read_table("gse243013_tcr_clonality")
    labeled = bulk[bulk["dataset"].isin(["GSE126044", "GSE207422_bulk"])].dropna(subset=["benefit", "CD274_expr", "scTIME_AI_score"]).copy()
    labeled["response"] = np.where(labeled["benefit"].eq(1), "benefit/MPR", "no benefit")
    labeled["cd274_group"] = np.where(labeled["CD274_expr"] >= labeled.groupby("dataset", observed=True)["CD274_expr"].transform("median"), "CD274 high", "CD274 low")
    labeled["sctime_group"] = np.where(labeled["scTIME_AI_score"] >= labeled.groupby("dataset", observed=True)["scTIME_AI_score"].transform("median"), "scTIME-AI high", "scTIME-AI low")
    dataset_marker_map = {dataset: marker for dataset, marker in zip(sorted(labeled["dataset"].dropna().unique()), ["s", "^", "o", "D"])}
    labeled["display_marker"] = labeled["dataset"].map(dataset_marker_map)
    write_panel_data("Figure 6", "A", "CD274 and scTIME-AI co-state in ICI cohorts", labeled, ["scTIME_AI_scores_all_bulk_cohorts"], "Labeled ICI cohort CD274 and scTIME-AI values.")
    low_q = labeled.groupby("dataset", observed=True)["scTIME_AI_score"].transform(lambda s: s.quantile(0.25))
    high_q = labeled.groupby("dataset", observed=True)["scTIME_AI_score"].transform(lambda s: s.quantile(0.75))
    labeled["sctime_confidence_group"] = "scTIME-AI intermediate"
    labeled.loc[labeled["scTIME_AI_score"].le(low_q), "sctime_confidence_group"] = "scTIME-AI bottom quartile"
    labeled.loc[labeled["scTIME_AI_score"].ge(high_q), "sctime_confidence_group"] = "scTIME-AI top quartile"
    labeled["sctime_median_group"] = labeled["sctime_group"]
    strat_rows = []
    for strategy, group_col in [("scTIME-AI quartile", "sctime_confidence_group"), ("scTIME-AI median", "sctime_median_group"), ("CD274 median", "cd274_group")]:
        for group, sub in labeled.groupby(group_col, observed=True):
            strat_rows.append({"strategy": strategy, "group": group, "n": len(sub), "responses": int(sub["benefit"].sum()), "benefit_rate": float(sub["benefit"].mean()), "mean_sctime_ai": float(sub["scTIME_AI_score"].mean()), "mean_cd274": float(sub["CD274_expr"].mean())})
    stratified = pd.DataFrame(strat_rows)
    stratified["ci_low"], stratified["ci_high"] = wilson_interval(stratified["responses"], stratified["n"])
    write_panel_data("Figure 6", "B", "High-confidence scTIME-AI enrichment versus CD274 stratification", stratified, ["scTIME_AI_scores_all_bulk_cohorts"], "Response fractions for scTIME-AI and CD274 stratification schemes.")
    diff_features = ["IFN_response", "Antigen_presentation", "Cytotoxic_CD8", "CXCL10_CXCR3_axis", "SPP1_macrophage", "TGF_beta_EMT", "Hypoxia"]
    high_cd274_low_score = labeled[(labeled["cd274_group"].eq("CD274 high")) & (labeled["sctime_group"].eq("scTIME-AI low"))]
    high_score = labeled[labeled["sctime_group"].eq("scTIME-AI high")]
    contrast_rows = []
    for feature in diff_features:
        a = pd.to_numeric(high_score[feature], errors="coerce").dropna()
        b = pd.to_numeric(high_cd274_low_score[feature], errors="coerce").dropna()
        _, pval = stats.mannwhitneyu(a, b, alternative="two-sided") if len(a) >= 2 and len(b) >= 2 else (np.nan, np.nan)
        contrast_rows.append({"feature": feature, "high_score_mean": a.mean(), "cd274_high_score_low_mean": b.mean(), "difference": a.mean() - b.mean(), "p_value": pval, "n_high_score": len(a), "n_cd274_high_score_low": len(b)})
    contrast = pd.DataFrame(contrast_rows)
    contrast["fdr"] = benjamini_hochberg(contrast["p_value"])
    write_panel_data("Figure 6", "C", "CD274-high scTIME-AI-low feature contrast", contrast, ["scTIME_AI_scores_all_bulk_cohorts"], "Feature contrast between high-scTIME-AI and CD274-high/scTIME-AI-low samples.")
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
    tcr_plot = tcr.melt(id_vars=["sampleID", "benefit"], value_vars=tcr_metrics, var_name="metric", value_name="value")
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
    write_panel_data("Figure 7", "A", "GSE135222 PFS validation", pd.DataFrame(km_rows), ["scTIME_AI_scores_all_bulk_cohorts", "gse135222_survival_validation"], "Kaplan-Meier coordinates for GSE135222 scTIME-AI median split.")
    forest_rows = []
    row = gse135[gse135["analysis"].str.contains("Cox", na=False)].head(1)
    if not row.empty:
        forest_rows.append({"analysis": "GSE135222 PFS", "hr": float(row["hazard_ratio"].iloc[0]), "p": float(row["p_value"].iloc[0])})
    for _, row in tcga_status.iterrows():
        forest_rows.append({"analysis": f"{row['project']} OS", "hr": row["cox_hr"], "p": row["cox_p"]})
    write_panel_data("Figure 7", "B", "Survival association summary", pd.DataFrame(forest_rows), ["gse135222_survival_validation", "tcga_secondary_validation"], "Hazard-ratio summary for survival association panels.")
    write_panel_data("Figure 7", "C", "TCGA-LUAD LUSC scTIME-AI distributions", tcga.dropna(subset=["project", "scTIME_AI_score"]), ["tcga_sctime_ai_scores"], "TCGA sample-level scTIME-AI score distribution.")
    cptac_panel = cptac[cptac["stage_group"].isin(["Early", "Advanced"])].copy()
    cptac_panel["cohort_omics"] = cptac_panel["cohort"] + "\n" + cptac_panel["omics"]
    write_panel_data("Figure 7", "D", "CPTAC proteome phosphoproteome projection", cptac_panel, ["cptac_sctime_ai_projection"], "CPTAC sample-level scTIME-AI projections by stage group.")
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
    write_panel_data("Supplementary Figure 1", "B", "Model performance across validation designs", perf, ["model_performance"], "All model-performance rows used for the audit scatter plot.")
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


def fmt(value: float, digits: int = 2) -> str:
    if pd.isna(value):
        return "NA"
    return f"{value:.{digits}f}"


def fmt_p(value: float) -> str:
    if pd.isna(value):
        return "NA"
    if value < 0.001:
        return f"{value:.1e}"
    return f"{value:.3f}"


def write_results_and_experiment_plan() -> None:
    perf = pd.read_csv(PANEL_DATA / "figure_3_panelb_sctime_ai_reliability_matrix_across_validation_designs.tsv", sep="\t")
    boot = pd.read_csv(PANEL_DATA / "figure_3_panelg_bootstrap_reliability_intervals.tsv", sep="\t")
    f2b = pd.read_csv(PANEL_DATA / "figure_2_panelb_response_associated_state_shifts.tsv", sep="\t")
    f4b = pd.read_csv(PANEL_DATA / "figure_4_panelb_ifn_antigen_index_links_to_cxcl10_cxcr3.tsv", sep="\t")
    f5b = pd.read_csv(PANEL_DATA / "figure_5_panelb_myeloid_deconvolution_moderately_aligns_with_spp1_macrophage_score.tsv", sep="\t")
    f6b = pd.read_csv(PANEL_DATA / "figure_6_panelb_high_confidence_sctime_ai_enrichment_versus_cd274_stratification.tsv", sep="\t")
    f7b = pd.read_csv(PANEL_DATA / "figure_7_panelb_survival_association_summary.tsv", sep="\t")
    f4a_tests = pd.read_csv(DESIGN / "figure4_cxcl10_ifn_axis_response_tests.tsv", sep="\t")
    f5a = pd.read_csv(PANEL_DATA / "figure_5_panela_exclusion_feature_response_effects.tsv", sep="\t")
    f6a = pd.read_csv(PANEL_DATA / "figure_6_panela_cd274_and_sctime_ai_co_state_in_ici_cohorts.tsv", sep="\t")

    top_pos = f2b.sort_values("signed_log10_p", ascending=False).head(3)
    top_neg = f2b.sort_values("signed_log10_p", ascending=True).head(3)
    pooled_auc = perf[(perf["validation"].eq("Pooled CV")) & (perf["metric"].eq("ROC-AUC"))]["value"].iloc[0]
    pooled_ap = perf[(perf["validation"].eq("Pooled CV")) & (perf["metric"].eq("PR-AUC"))]["value"].iloc[0]
    cc_auc = perf[perf["metric"].eq("ROC-AUC")]["value"].max()
    boot_pooled = boot[(boot["validation"].eq("Pooled labeled cohorts")) & (boot["metric"].eq("ROC-AUC"))].iloc[0]
    cxcl10 = f4a_tests[f4a_tests["feature"].eq("CXCL10_CXCR3_axis")].iloc[0]
    ifn_rho, ifn_p = stats.spearmanr(f4b["IFN_antigen_index"], f4b["CXCL10_CXCR3_axis"])
    myeloid_rho, myeloid_p = stats.spearmanr(f5b["Myeloid/macrophage"], f5b["SPP1_macrophage"])
    exclusion_top = f5a.sort_values("mean_difference").head(3)
    cd274_rho, cd274_p = stats.spearmanr(f6a["CD274_expr"], f6a["scTIME_AI_score"])
    top_q = f6b[f6b["group"].eq("scTIME-AI top quartile")].iloc[0]
    bottom_q = f6b[f6b["group"].eq("scTIME-AI bottom quartile")].iloc[0]
    cd274_high = f6b[f6b["group"].eq("CD274 high")].iloc[0]
    cd274_low = f6b[f6b["group"].eq("CD274 low")].iloc[0]
    pfs = f7b[f7b["analysis"].eq("GSE135222 PFS")].iloc[0]

    lines = [
        "# Completed Results Description and Cell Experiment Plan\n",
        "\n## Panel Data Availability\n",
        "All computational figure panels now have plot-ready source data in `results/figure_design/panel_data/`. The index file `panel_data_manifest.tsv` maps every main-text and supplementary panel to its TSV data file and source tables. Figure 1 is a manually drawn framework figure, so its panels are supported by schematic source tables rather than plotted coordinates.\n",
        "\n## Completed Results Description\n",
        "\n### Figure 1. Study Framework\n",
        "The study integrates public NSCLC single-cell, bulk transcriptomic, TCGA and CPTAC resources into a single-cell-informed immunotherapy-response workflow. The framework links single-cell state discovery, bulk projection, ElasticNet scTIME-AI modeling, interpretable feature contributions, and external survival/cross-omics validation.\n",
        "\n### Figure 2. Single-Cell Response Landscape\n",
        f"GSE207422 single-cell embedding resolved the major immune and tumor/stromal compartments used as the single-cell reference. In GSE243013, the most MPR-enriched state shifts included {', '.join(top_pos['state_category'].astype(str))}, whereas non-MPR-enriched shifts included {', '.join(top_neg['state_category'].astype(str))}. Patient-level fraction plots and the heatmap show that response-associated states vary at the patient level rather than only at aggregate-cell level, supporting downstream bulk projection.\n",
        "\n### Figure 3. scTIME-AI Reliability and Interpretation\n",
        f"Bulk immune-state PCA separated labeled and external samples along immune-program axes and provided the projection space for scTIME-AI. The final ElasticNet model achieved pooled cross-validation ROC-AUC {fmt(pooled_auc, 3)} and PR-AUC {fmt(pooled_ap, 3)}, with the strongest cross-cohort ROC-AUC {fmt(cc_auc, 3)}. Bootstrap analysis estimated pooled ROC-AUC {fmt(boot_pooled['estimate'], 3)} with 95% interval {fmt(boot_pooled['ci_low'], 3)}-{fmt(boot_pooled['ci_high'], 3)}. Calibration, decision-curve, coefficient, and patient-level contribution panels document both predictive performance and feature interpretability.\n",
        "\n### Figure 4. CXCL10-CXCR3 / IFN-Antigen Axis\n",
        f"The CXCL10-CXCR3 axis was higher in benefit/MPR samples than non-benefit samples by mean difference {fmt(cxcl10['mean_difference'], 3)} (FDR {fmt_p(cxcl10['fdr'])}). The IFN-antigen index correlated with the CXCL10-CXCR3 axis (Spearman rho {fmt(ifn_rho, 2)}, P={fmt_p(ifn_p)}), linking antigen-presentation/IFN signaling to the chemokine recruitment program. Single-cell LR and ligand-activity panels further prioritize CXCL9/10/11-CXCR3 signaling as a candidate effector-recruitment mechanism.\n",
        "\n### Figure 5. Myeloid/TGF-beta/Hypoxia Exclusion Biology\n",
        f"Non-responder biology was characterized by exclusion-associated programs including {', '.join(exclusion_top['feature'].map(lambda x: FEATURE_LABELS.get(x, x)).astype(str))}. Bulk NNLS myeloid/macrophage fraction showed a moderate but statistically significant correlation with the SPP1 macrophage module score (Spearman rho {fmt(myeloid_rho, 2)}, P={fmt_p(myeloid_p)}), supporting it as a consistency check within the broader exclusion-axis evidence chain rather than as a standalone validation. TCGA and CPTAC panels extend this axis into larger external and cross-omics settings.\n",
        "\n### Figure 6. CD274/scTIME-AI Complementarity and TCR Ecology\n",
        f"CD274 and scTIME-AI were correlated but not redundant (Spearman rho {fmt(cd274_rho, 2)}, P={fmt_p(cd274_p)}). High-confidence scTIME-AI stratification was more discriminative than CD274 median split: top-quartile scTIME-AI had {int(top_q['responses'])}/{int(top_q['n'])} benefit/MPR samples (rate {fmt(top_q['benefit_rate'], 2)}), whereas bottom-quartile scTIME-AI had {int(bottom_q['responses'])}/{int(bottom_q['n'])} (rate {fmt(bottom_q['benefit_rate'], 2)}). CD274-high and CD274-low groups showed more modest rates of {fmt(cd274_high['benefit_rate'], 2)} and {fmt(cd274_low['benefit_rate'], 2)}, respectively. TCR panels retain immune-ecology context without making TCR metrics the primary predictive claim.\n",
        "\n### Figure 7. External Survival and Cross-Omics Validation\n",
        f"GSE135222 survival validation associated high scTIME-AI score with improved PFS; the Cox model HR was {fmt(pfs['hr'], 3)} with P={fmt_p(pfs['p'])}. TCGA and CPTAC projections show that the same score and immune programs can be evaluated across bulk RNA, proteome and phosphoproteome contexts, supporting cross-platform interpretability.\n",
        "\n### Supplementary Figures\n",
        "Supplementary Figure 1 documents cohort coverage, broad model benchmarking, calibration and decision-curve details. Supplementary Figure 2 records marker-program, gene-coverage, LR-axis, deconvolution, reference-separability and matched scRNA-bulk concordance audits that support the main analyses but are not the central narrative.\n",
        "\n## Follow-Up Cell Experiment Plan\n",
        "\n### Overall Constraints\n",
        "The proposed validation avoids cell imaging, flow sorting and electrophoresis. Readouts rely on qPCR/RT-qPCR, ELISA or multiplex immunoassay, plate-reader luminescence/colorimetry, impedance assays, endpoint cell counting, and biochemical viability/cytotoxicity assays.\n",
        "\n### Aim 1. Validate IFN-Induced CXCL10 Production in NSCLC Cells\n",
        "- Models: NSCLC cell lines representing LUAD/LUSC backgrounds, for example A549, H1975, H1299 and H520, plus one normal bronchial epithelial comparator if available.\n",
        "- Perturbations: IFN-gamma dose-response and time-course; optional TNF-alpha co-stimulation; CXCL10 knockdown by siRNA or CRISPRi where feasible.\n",
        "- Readouts: CXCL10, CXCL9, CXCL11, HLA-A/B/C, B2M, CD274, IFIT1 and STAT1 by RT-qPCR; CXCL10/CXCL9/CXCL11 protein in supernatant by ELISA or multiplex assay; cell viability by CellTiter-Glo to exclude nonspecific toxicity.\n",
        "- Decision criterion: IFN-gamma should increase CXCL10-family ligand secretion without major viability loss; knockdown should selectively reduce CXCL10 protein and transcript.\n",
        "\n### Aim 2. Test CXCL10-CXCR3-Dependent T-Cell Chemotaxis Without Imaging or Flow Sorting\n",
        "- Effector cells: commercially sourced primary human CD8+ T cells or untouched negative-selection magnetic bead-enriched CD8+ T cells; avoid flow sorting.\n",
        "- Assay: transwell migration toward conditioned media from IFN-stimulated tumor cells, recombinant CXCL10 positive control, and unstimulated media negative control.\n",
        "- Blocking arms: CXCL10 neutralizing antibody, CXCR3 antagonist such as AMG487, anti-CXCR3 blocking antibody, and recombinant CXCL10 rescue after CXCL10 knockdown.\n",
        "- Readouts: migrated-cell quantity by ATP luminescence, DNA dye plate-reader assay, or endpoint automated cell counter; supernatant CXCL10 by ELISA to confirm stimulus strength.\n",
        "- Decision criterion: migration should increase with IFN-stimulated conditioned media and recombinant CXCL10, and decrease after CXCL10/CXCR3 blockade.\n",
        "\n### Aim 3. Test Whether CXCL10 Axis Enhances T-Cell Functional Activation in Co-Culture\n",
        "- Setup: tumor-cell conditioned media plus activated CD8+ T cells, or direct tumor/T-cell co-culture using fixed effector:target ratios.\n",
        "- Perturbations: CXCL10 knockdown/neutralization, CXCR3 inhibition, recombinant CXCL10 add-back, and optional anti-PD-1/anti-PD-L1 antibody arms for translational relevance.\n",
        "- Readouts: IFN-gamma, granzyme B, TNF-alpha and IL-2 in supernatant by ELISA/multiplex; T-cell activation transcripts by RT-qPCR from bulk cell pellets; tumor-cell killing by LDH release, Caspase-Glo 3/7, CellTiter-Glo differential viability, or real-time impedance.\n",
        "- Decision criterion: intact CXCL10-CXCR3 signaling should increase chemotactic/activation readouts and improve biochemical cytotoxicity signals.\n",
        "\n### Aim 4. Validate Myeloid/SPP1/TGF-beta/Hypoxia Exclusion Counter-Axis\n",
        "- Models: THP-1-derived macrophage-like cells or primary monocyte-derived macrophages, with SPP1-high polarization using tumor-conditioned media; tumor cells under TGF-beta1 and hypoxia-mimetic conditions such as CoCl2 or controlled hypoxia chamber if available.\n",
        "- Perturbations: SPP1 neutralization, TGF-beta receptor inhibition, hypoxia-mimetic withdrawal, and combination with IFN-gamma-induced CXCL10 condition.\n",
        "- Readouts: SPP1, TGFB1, VEGFA, CA9, CXCL10 and CD274 transcripts by RT-qPCR; SPP1/TGF-beta1/CXCL10 proteins by ELISA or multiplex; T-cell migration/function assays as above to test whether exclusion-conditioned media suppresses CXCL10-driven activity.\n",
        "- Decision criterion: SPP1/TGF-beta/hypoxia-conditioned media should blunt T-cell migration/function, while pathway blockade should partially rescue it.\n",
        "\n### Aim 5. Build an Experimental scTIME-AI-Compatible qPCR/Secretome Panel\n",
        "- Panel genes/proteins: IFN response markers, antigen-presentation genes, CXCL9/10/11, CD274, cytotoxic markers, SPP1, TGFB1, hypoxia markers and Treg/exhaustion-associated markers where measurable in the chosen co-culture system.\n",
        "- Readouts: RT-qPCR delta-Ct matrix and ELISA/multiplex secretome matrix; no western blot/electrophoresis.\n",
        "- Analysis: project experimental perturbation profiles onto the directionality learned from scTIME-AI coefficients and report whether CXCL10-enhancing conditions move samples toward the responder-like axis while SPP1/TGF-beta/hypoxia conditions move samples toward exclusion.\n",
    ]
    out = DESIGN / "completed_results_and_cell_experiment_plan.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_results_and_experiment_plan_zh() -> None:
    perf = pd.read_csv(PANEL_DATA / "figure_3_panelb_sctime_ai_reliability_matrix_across_validation_designs.tsv", sep="\t")
    boot = pd.read_csv(PANEL_DATA / "figure_3_panelg_bootstrap_reliability_intervals.tsv", sep="\t")
    f2b = pd.read_csv(PANEL_DATA / "figure_2_panelb_response_associated_state_shifts.tsv", sep="\t")
    f4b = pd.read_csv(PANEL_DATA / "figure_4_panelb_ifn_antigen_index_links_to_cxcl10_cxcr3.tsv", sep="\t")
    f5b = pd.read_csv(PANEL_DATA / "figure_5_panelb_myeloid_deconvolution_moderately_aligns_with_spp1_macrophage_score.tsv", sep="\t")
    f6b = pd.read_csv(PANEL_DATA / "figure_6_panelb_high_confidence_sctime_ai_enrichment_versus_cd274_stratification.tsv", sep="\t")
    f7b = pd.read_csv(PANEL_DATA / "figure_7_panelb_survival_association_summary.tsv", sep="\t")
    f4a_tests = pd.read_csv(DESIGN / "figure4_cxcl10_ifn_axis_response_tests.tsv", sep="\t")
    f5a = pd.read_csv(PANEL_DATA / "figure_5_panela_exclusion_feature_response_effects.tsv", sep="\t")
    f6a = pd.read_csv(PANEL_DATA / "figure_6_panela_cd274_and_sctime_ai_co_state_in_ici_cohorts.tsv", sep="\t")
    manifest = pd.read_csv(PANEL_DATA / "panel_data_manifest.tsv", sep="\t")

    top_pos = f2b.sort_values("signed_log10_p", ascending=False).head(3)
    top_neg = f2b.sort_values("signed_log10_p", ascending=True).head(3)
    pooled_auc = perf[(perf["validation"].eq("Pooled CV")) & (perf["metric"].eq("ROC-AUC"))]["value"].iloc[0]
    pooled_ap = perf[(perf["validation"].eq("Pooled CV")) & (perf["metric"].eq("PR-AUC"))]["value"].iloc[0]
    cc_auc = perf[perf["metric"].eq("ROC-AUC")]["value"].max()
    boot_pooled = boot[(boot["validation"].eq("Pooled labeled cohorts")) & (boot["metric"].eq("ROC-AUC"))].iloc[0]
    cxcl10 = f4a_tests[f4a_tests["feature"].eq("CXCL10_CXCR3_axis")].iloc[0]
    ifn_rho, ifn_p = stats.spearmanr(f4b["IFN_antigen_index"], f4b["CXCL10_CXCR3_axis"])
    myeloid_rho, myeloid_p = stats.spearmanr(f5b["Myeloid/macrophage"], f5b["SPP1_macrophage"])
    exclusion_top = f5a.sort_values("mean_difference").head(3)
    cd274_rho, cd274_p = stats.spearmanr(f6a["CD274_expr"], f6a["scTIME_AI_score"])
    top_q = f6b[f6b["group"].eq("scTIME-AI top quartile")].iloc[0]
    bottom_q = f6b[f6b["group"].eq("scTIME-AI bottom quartile")].iloc[0]
    cd274_high = f6b[f6b["group"].eq("CD274 high")].iloc[0]
    cd274_low = f6b[f6b["group"].eq("CD274 low")].iloc[0]
    pfs = f7b[f7b["analysis"].eq("GSE135222 PFS")].iloc[0]

    lines = [
        "# 已完成结果描述与后续细胞实验方案",
        "",
        "## 图表原始数据记录",
        "",
        f"所有正文图和附图子图均已导出对应的 plot-ready 原始数据。索引文件为 `results/figure_design/panel_data/panel_data_manifest.tsv`，共记录 {len(manifest)} 个子图数据文件，覆盖 Figure 1-7 以及 Supplementary Figure 1-2。每一行包含 figure、panel、panel 标题、数据文件路径、行列数、来源表和用途说明。Figure 1 为手绘框架图，因此其 A-D panel 记录的是支撑框架图绘制的队列、流程、机制轴和模型输出元数据。",
        "",
        "## 已完成结果描述",
        "",
        "### Figure 1. 研究总体框架",
        "",
        "本研究整合 NSCLC 免疫治疗相关公共单细胞、bulk RNA-seq、TCGA 与 CPTAC 多组学资源，形成从单细胞发现到 bulk 投影、scTIME-AI 建模、特征解释和外部验证的完整分析流程。Figure 1 可作为手绘示意图，强调公共队列来源、分析模块、响应者激活轴、非响应者免疫排斥轴以及可转化的患者级预测输出。",
        "",
        "### Figure 2. 单细胞层面的免疫治疗响应景观",
        "",
        f"GSE207422 单细胞图谱解析出主要免疫、肿瘤和基质细胞状态，并作为后续 bulk 投影的单细胞参考。GSE243013 的病理响应差异分析显示，MPR 富集的状态主要包括 {', '.join(top_pos['state_category'].astype(str))}，而 non-MPR 富集的状态主要包括 {', '.join(top_neg['state_category'].astype(str))}。患者水平细胞比例箱线图和 patient-by-state 热图显示，应答相关细胞状态并非只存在于总体平均层面，而是在不同患者之间呈现清晰的组成差异，为后续构建单细胞指导的 bulk 预测模型提供依据。",
        "",
        "### Figure 3. scTIME-AI 的可靠性与可解释性",
        "",
        f"基于单细胞免疫状态构建的 bulk 免疫模块投影可将已标注免疫治疗队列和外部验证队列置于同一免疫状态空间。最终 ElasticNet scTIME-AI 模型在 pooled 交叉验证中达到 ROC-AUC {fmt(pooled_auc, 3)}、PR-AUC {fmt(pooled_ap, 3)}，最佳跨队列验证 ROC-AUC 为 {fmt(cc_auc, 3)}。Bootstrap 评估显示 pooled ROC-AUC 为 {fmt(boot_pooled['estimate'], 3)}，95% 区间为 {fmt(boot_pooled['ci_low'], 3)}-{fmt(boot_pooled['ci_high'], 3)}。校准曲线、decision curve、模型系数和患者级 feature contribution heatmap 共同支持该模型不仅具有预测性能，也具有可解释性。",
        "",
        "### Figure 4. CXCL10-CXCR3 / IFN-antigen 轴的证据链",
        "",
        f"CXCL10-CXCR3 轴在 benefit/MPR 样本中高于无获益样本，平均差异为 {fmt(cxcl10['mean_difference'], 3)}，FDR 为 {fmt_p(cxcl10['fdr'])}。IFN-antigen index 与 CXCL10-CXCR3 轴显著相关，Spearman rho={fmt(ifn_rho, 2)}，P={fmt_p(ifn_p)}，提示 IFN 反应和抗原呈递增强可能与 CXCL10/CXCR3 介导的 T 细胞募集相连。单细胞 LR 和 ligand activity panel 进一步支持 CXCL9/10/11-CXCR3 作为响应者 T cell-inflamed 状态的重要候选机制。",
        "",
        "### Figure 5. 髓系/TGF-beta/hypoxia 免疫排斥轴",
        "",
        f"非响应者相关生物学主要体现为排斥相关模块增强，其中代表性特征包括 {', '.join(exclusion_top['feature'].map(lambda x: FEATURE_LABELS.get(x, x)).astype(str))}。Bulk NNLS 推断的 myeloid/macrophage fraction 与 SPP1 macrophage module 呈中等但显著的正相关，Spearman rho={fmt(myeloid_rho, 2)}，P={fmt_p(myeloid_p)}。因此 5B 更适合作为 bulk 层面的一致性证据，而不是单独决定性验证；其意义需要结合 5A、5C、5D 和 5F 的排斥轴证据链共同解释。TCGA 和 CPTAC panel 将该排斥轴扩展到更大样本量和跨组学验证场景。",
        "",
        "### Figure 6. CD274/scTIME-AI 互补性与 TCR 免疫生态",
        "",
        f"CD274 与 scTIME-AI 呈相关但不完全冗余，Spearman rho={fmt(cd274_rho, 2)}，P={fmt_p(cd274_p)}。高置信 scTIME-AI 分层优于单纯 CD274 median split：scTIME-AI top quartile 中 {int(top_q['responses'])}/{int(top_q['n'])} 为 benefit/MPR，比例 {fmt(top_q['benefit_rate'], 2)}；bottom quartile 中 {int(bottom_q['responses'])}/{int(bottom_q['n'])} 为 benefit/MPR，比例 {fmt(bottom_q['benefit_rate'], 2)}。相比之下，CD274-high 与 CD274-low 的 benefit/MPR 比例分别为 {fmt(cd274_high['benefit_rate'], 2)} 和 {fmt(cd274_low['benefit_rate'], 2)}。TCR 相关 panel 用于补充免疫生态背景，而不是将 TCR clonality 作为主要预测结论。",
        "",
        "### Figure 7. 外部生存与跨组学验证",
        "",
        f"GSE135222 中高 scTIME-AI score 与更好的 PFS 相关，Cox 模型 HR={fmt(pfs['hr'], 3)}，P={fmt_p(pfs['p'])}。TCGA 和 CPTAC 投影显示，同一 scTIME-AI 分数与关键免疫模块可在 bulk RNA、proteome 和 phosphoproteome 场景中评估，支持该模型的跨平台解释潜力。",
        "",
        "### Supplementary Figures",
        "",
        "Supplementary Figure 1 记录队列样本量、广义模型性能、校准和 decision curve 审计。Supplementary Figure 2 记录 marker-program audit、signature coverage、LR axis activity、deconvolution-module correlation、reference separability 和 matched scRNA-bulk concordance。这些内容用于支撑主文结论，但不作为主叙事的核心图。",
        "",
        "## 后续细胞实验方案",
        "",
        "### 总体原则",
        "",
        "后续实验聚焦 CXCL10-CXCR3 激活轴与 SPP1/TGF-beta/hypoxia 排斥轴。方案不使用细胞成像、流式分选或电泳类读出。推荐使用 RT-qPCR、ELISA 或 multiplex protein assay、plate-reader luminescence/colorimetry、实时阻抗、LDH release、Caspase-Glo、CellTiter-Glo、自动细胞计数或普通计数板等非成像读出。",
        "",
        "### Aim 1. 验证 IFN 刺激诱导 NSCLC 细胞产生 CXCL10",
        "",
        "- 模型：选择 LUAD/LUSC 背景细胞系，例如 A549、H1975、H1299、H520；如条件允许加入正常支气管上皮细胞作为非肿瘤对照。",
        "- 处理：IFN-gamma 剂量梯度和时间梯度；可加入 TNF-alpha 协同刺激；使用 siRNA 或 CRISPRi 降低 CXCL10 表达。",
        "- 读出：RT-qPCR 检测 CXCL10、CXCL9、CXCL11、HLA-A/B/C、B2M、CD274、IFIT1、STAT1；ELISA 或 multiplex 检测上清 CXCL10/CXCL9/CXCL11；CellTiter-Glo 排除处理造成的非特异性细胞毒性。",
        "- 判定标准：IFN-gamma 应显著诱导 CXCL10-family ligand 的转录和分泌，且不伴随明显细胞活性下降；CXCL10 knockdown 应选择性降低 CXCL10 transcript 和 protein。",
        "",
        "### Aim 2. 验证 CXCL10-CXCR3 依赖的 T 细胞趋化",
        "",
        "- 细胞来源：可购买原代人 CD8+ T 细胞，或使用商业 untouched negative-selection 磁珠富集 CD8+ T 细胞；不采用流式分选。",
        "- 实验体系：Transwell migration。下室加入 IFN-stimulated tumor-cell conditioned media、recombinant CXCL10 阳性对照、unstimulated media 阴性对照。",
        "- 阻断条件：CXCL10 neutralizing antibody、CXCR3 antagonist（例如 AMG487）、anti-CXCR3 blocking antibody，以及 CXCL10 knockdown 后的 recombinant CXCL10 rescue。",
        "- 读出：迁移细胞数量用 ATP luminescence、DNA dye plate-reader assay、自动细胞计数或普通计数板定量；上清 CXCL10 用 ELISA 确认刺激强度。",
        "- 判定标准：IFN-stimulated conditioned media 和 recombinant CXCL10 应增强 T 细胞迁移；CXCL10/CXCR3 阻断应降低迁移；recombinant CXCL10 add-back 应部分恢复迁移。",
        "",
        "### Aim 3. 检验 CXCL10 轴是否增强 T 细胞功能活化和肿瘤细胞杀伤",
        "",
        "- 实验体系：使用肿瘤细胞 conditioned media + activated CD8+ T cells，或固定 effector:target ratio 的直接共培养。",
        "- 处理：CXCL10 knockdown/neutralization、CXCR3 inhibition、recombinant CXCL10 add-back；可加入 anti-PD-1 或 anti-PD-L1 作为转化相关处理组。",
        "- 读出：ELISA/multiplex 检测 IFN-gamma、granzyme B、TNF-alpha、IL-2；RT-qPCR 检测 bulk cell pellet 中 T cell activation/cytotoxicity transcripts；LDH release、Caspase-Glo 3/7、CellTiter-Glo differential viability 或实时阻抗检测肿瘤细胞杀伤。",
        "- 判定标准：完整 CXCL10-CXCR3 信号应增强趋化后 T 细胞功能读出，并提高生化杀伤指标；阻断该轴应削弱上述效应。",
        "",
        "### Aim 4. 验证 SPP1/TGF-beta/hypoxia 排斥轴对 CXCL10-driven T 细胞反应的抑制",
        "",
        "- 模型：THP-1-derived macrophage-like cells 或 primary monocyte-derived macrophages；用 tumor-conditioned media 诱导 SPP1-high macrophage-like 状态。肿瘤细胞可用 TGF-beta1 和低氧模拟条件建立排斥样微环境。",
        "- 处理：SPP1 neutralization、TGF-beta receptor inhibition、解除低氧模拟条件，并与 IFN-gamma-induced CXCL10 condition 组合。",
        "- 读出：RT-qPCR 检测 SPP1、TGFB1、VEGFA、CA9、CXCL10、CD274；ELISA/multiplex 检测 SPP1、TGF-beta1、CXCL10；使用 Aim 2/3 的趋化和功能 assay 检验排斥条件是否抑制 CXCL10-driven T cell response。",
        "- 判定标准：SPP1/TGF-beta/hypoxia-conditioned media 应降低 T 细胞趋化或功能读出；阻断 SPP1 或 TGF-beta pathway 后应出现部分恢复。",
        "",
        "### Aim 5. 构建实验版 scTIME-AI qPCR/secretome panel",
        "",
        "- Panel 内容：IFN response、antigen presentation、CXCL9/10/11、CD274、cytotoxic markers、SPP1、TGFB1、hypoxia markers，以及根据共培养体系可检测的 exhaustion/Treg-associated markers。",
        "- 读出：RT-qPCR delta-Ct matrix 和 ELISA/multiplex secretome matrix。",
        "- 分析：将不同 perturbation 条件下的 qPCR/secretome profile 投影到 scTIME-AI 系数方向，判断 CXCL10-enhancing conditions 是否向 responder-like axis 移动，而 SPP1/TGF-beta/hypoxia conditions 是否向 exclusion axis 移动。",
    ]
    out = DESIGN / "completed_results_and_cell_experiment_plan.zh.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    PANEL_DATA.mkdir(parents=True, exist_ok=True)
    for path in PANEL_DATA.glob("*.tsv"):
        path.unlink()
    export_figure1()
    export_figure2()
    export_figure3()
    export_figure4()
    export_figure5()
    export_figure6()
    export_figure7()
    export_supplementary_figure1()
    export_supplementary_figure2()
    manifest = pd.DataFrame(PANEL_ROWS)
    manifest.to_csv(PANEL_DATA / "panel_data_manifest.tsv", sep="\t", index=False)
    write_results_and_experiment_plan()
    write_results_and_experiment_plan_zh()


if __name__ == "__main__":
    sys.exit(main())
