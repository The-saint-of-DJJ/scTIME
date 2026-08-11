#!/usr/bin/env python
"""Generate publication-oriented multi-panel figures from CGZ analysis tables.

The figures are intentionally data/audit plots only. No mechanistic cartoon or
pathway schematic is generated here.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap, Normalize
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler

from figure6_heldout import (
    OOF_SCORE_PROVENANCE,
    build_combined_response_groups,
    build_feature_contrast,
    build_stratification_table,
    prepare_heldout_figure6_labeled,
)


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
TABLES = PACKAGE_ROOT / "source_data" / "results_tables"
FIGURES = PACKAGE_ROOT / "figures_jpg"
DESIGN = PACKAGE_ROOT / "source_data"
SUPPLEMENTARY = PACKAGE_ROOT / "supplementary"

JOURNAL_FIGURE_FILENAMES = {
    "figure2_single_cell_response_landscape": "Figure_2.jpg",
    "figure3_ai_reliability_sctime_modeling": "Figure_3.jpg",
    "figure4_cxcl10_ifn_antigen_axis": "Figure_4.jpg",
    "figure5_myeloid_tgf_hypoxia_exclusion_axis": "Figure_5.jpg",
    "figure6_cd274_tcr_immune_ecology": "Figure_6.jpg",
    "figure7_external_validation_cross_omics": "Figure_7.jpg",
    "supplementary_audit_model_details": "Supplementary_Figure_1.jpg",
    "supplementary_translation_lr_details": "Supplementary_Figure_2.jpg",
}

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

STATE_ORDER = [
    "Tumor/epithelial",
    "CD8 T",
    "CD4 T",
    "Treg",
    "NK",
    "B/plasma",
    "Myeloid/macrophage",
    "Dendritic",
    "Fibroblast",
    "Endothelial",
    "Mast",
]

PALETTE = {
    "positive": "#2f6f9f",
    "negative": "#b4473d",
    "neutral": "#6d6e71",
    "dark": "#2a2a2a",
    "light": "#f1f3f4",
}

FIGURE3A_RESEARCH_VALUE_THRESHOLD = 0.65
ACTIVATION_FEATURES = ["IFN_response", "Antigen_presentation", "Cytotoxic_CD8", "NK", "CXCL10_CXCR3_axis", "CD274_expr"]
# Contextual exclusion index: SPP1 remains a separate myeloid feature and is
# excluded from the composite to avoid partial self-correlation.
EXCLUSION_FEATURES = ["TGF_beta_EMT", "Hypoxia", "Treg", "Terminal_exhausted_CD8"]
DIVERGING_CMAP = LinearSegmentedColormap.from_list(
    "sctime_three_color_diverging",
    [PALETTE["negative"], "#f7f7f7", PALETTE["positive"]],
)
SEQUENTIAL_CMAP = LinearSegmentedColormap.from_list(
    "sctime_three_color_sequential",
    ["#f7f7f7", PALETTE["neutral"], PALETTE["positive"]],
)
NEGATIVE_CMAP = LinearSegmentedColormap.from_list(
    "sctime_three_color_negative",
    ["#f7f7f7", PALETTE["neutral"], PALETTE["negative"]],
)

MAIN_FIGURE_PLAN = [
    {
        "figure": "Figure 2",
        "file": "figure2_single_cell_response_landscape.png/pdf",
        "purpose": "Single-cell discovery figure: response-associated immune states without low-yield audit panels.",
        "panels": [
            "A GSE207422 single-cell immune landscape",
            "B GSE243013 response-associated cell-state shifts",
            "C Patient-level fractions of response-informative cell states",
            "D Patient-by-state fraction heatmap",
            "E Response-linked single-cell state network summary",
        ],
    },
    {
        "figure": "Figure 3",
        "file": "figure3_ai_reliability_sctime_modeling.png/pdf",
        "purpose": "Complex scTIME reliability, cross-cohort validation and interpretability figure.",
        "panels": [
            "A Bulk immune-state PCA with response and score overlay",
            "B scTIME reliability matrix across evaluation designs",
            "C ElasticNet ROC curves",
            "D ElasticNet precision-recall curves",
            "E Calibration with binomial confidence intervals",
            "F Decision-curve analysis",
            "G Bootstrap reliability intervals",
            "H Interpretable scTIME coefficients",
            "I Patient-level feature contribution heatmap",
        ],
    },
    {
        "figure": "Figure 4",
        "file": "figure4_cxcl10_ifn_antigen_axis.png/pdf",
        "purpose": "CXCL10-CXCR3, IFN response and antigen presentation evidence chain.",
        "panels": [
            "A CXCL10-CXCR3 axis by response",
            "B IFN-antigen index links to CXCL10-CXCR3",
            "C Single-cell CXCL9/10/11-CXCR3 LR score",
            "D Ligand activity against effector module",
            "E Cross-context CXCL10-CXCR3 correlations",
        ],
    },
    {
        "figure": "Figure 5",
        "file": "figure5_myeloid_tgf_hypoxia_exclusion_axis.png/pdf",
        "purpose": "Non-responder biology focused on myeloid, TGF-beta/EMT and hypoxia-associated exclusion.",
        "panels": [
            "A Exclusion-feature response effects",
            "B Cohort-stratified myeloid deconvolution and SPP1 macrophage consistency",
            "C Deconvolution-module correlation matrix",
            "D Project-stratified TCGA exclusion context vs scTIME",
            "E CPTAC exclusion index by stage group",
            "F TCGA exclusion co-program structure",
        ],
    },
    {
        "figure": "Figure 6",
        "file": "figure6_cd274_tcr_immune_ecology.png/pdf",
        "purpose": "PD-L1/scTIME complementarity and TCR ecology without low-density single-variable TCR panels.",
        "panels": [
            "A CD274 and scTIME co-state in ICI cohorts",
            "B Matched held-out scTIME and CD274 response stratification",
            "C CD274-high/OOF-scTIME-low immune-feature contrast",
            "D TCR clonality and immune-cell composition dot matrix",
            "E TCR clonality metrics by pathological response",
        ],
    },
    {
        "figure": "Figure 7",
        "file": "figure7_external_validation_cross_omics.png/pdf",
        "purpose": "Exploratory survival association and cross-omics projection kept compact and non-redundant.",
        "panels": [
            "A GSE135222 exploratory PFS association (Kaplan-Meier)",
            "B Survival association summary",
            "C TCGA-LUAD/LUSC scTIME distributions",
            "D CPTAC proteome/phosphoproteome projection by stage",
            "E CPTAC stage shifts in key immune programs",
        ],
    },
    {
        "figure": "Supplementary Figure 1",
        "file": "supplementary_audit_model_details.png/pdf",
        "purpose": "Compact cohort and model-performance audit material moved out of the main text.",
        "panels": [
            "A Cohort sample audit",
            "B Model performance across evaluation designs",
            "C Calibration",
            "D Decision curve",
        ],
    },
    {
        "figure": "Supplementary Figure 2",
        "file": "supplementary_translation_lr_details.png/pdf",
        "purpose": "Signature coverage, marker validation, deconvolution and LR detail panels retained as support.",
        "panels": [
            "A Cluster marker-program audit",
            "B Bulk/CPTAC signature coverage",
            "C LR axis activity matrix",
            "D Deconvolution-vs-module correlations",
            "E Reference signature separability",
            "F Matched scRNA-bulk concordance",
        ],
    },
]


def setup() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    DESIGN.mkdir(parents=True, exist_ok=True)
    sns.set_theme(
        context="paper",
        style="whitegrid",
        font="DejaVu Sans",
        rc={
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.edgecolor": "#333333",
            "axes.linewidth": 0.7,
            "grid.color": "#e6e6e6",
            "grid.linewidth": 0.45,
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
        },
    )


def read_table(name: str, **kwargs) -> pd.DataFrame:
    return pd.read_csv(TABLES / f"{name}.tsv", sep="\t", **kwargs)


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.18,
        1.13,
        label,
        transform=ax.transAxes,
        fontsize=12,
        fontweight="bold",
        va="top",
        ha="left",
    )


def panel_label_tight(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.28,
        1.18,
        label,
        transform=ax.transAxes,
        fontsize=12,
        fontweight="bold",
        va="top",
        ha="left",
    )


def panel_label_corner(ax: plt.Axes, label: str) -> None:
    ax.annotate(
        label,
        xy=(0, 1),
        xycoords="axes fraction",
        xytext=(-12, 7),
        textcoords="offset points",
        fontsize=12,
        fontweight="bold",
        va="bottom",
        ha="right",
        annotation_clip=False,
    )


def clean_label(value: object) -> str:
    text = str(value)
    return text.replace("_", " ").replace("GSE207422 bulk", "GSE207422").replace("TCGA-", "")


def savefig(fig: plt.Figure, filename: str) -> None:
    output_name = JOURNAL_FIGURE_FILENAMES.get(filename, f"{filename}.jpg")
    fig.savefig(
        FIGURES / output_name,
        format="jpeg",
        dpi=300,
        facecolor="white",
        bbox_inches="tight",
    )
    plt.close(fig)


def signed_log10_p(p: pd.Series, effect: pd.Series) -> pd.Series:
    p = pd.to_numeric(p, errors="coerce").clip(lower=1e-300)
    effect = pd.to_numeric(effect, errors="coerce").fillna(0)
    return -np.log10(p) * np.sign(effect)


def mannwhitney_rows(df: pd.DataFrame, group_col: str, value_cols: list[str], positive_value: object) -> pd.DataFrame:
    rows = []
    for col in value_cols:
        sub = df[[group_col, col]].dropna()
        pos = pd.to_numeric(sub.loc[sub[group_col] == positive_value, col], errors="coerce").dropna()
        neg = pd.to_numeric(sub.loc[sub[group_col] != positive_value, col], errors="coerce").dropna()
        if len(pos) < 2 or len(neg) < 2:
            p = np.nan
            u = np.nan
        else:
            from scipy.stats import mannwhitneyu

            u, p = mannwhitneyu(pos, neg, alternative="two-sided")
        rows.append(
            {
                "feature": col,
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
    out["rank"] = out["p_value"].rank(method="first")
    m = out["p_value"].notna().sum()
    out["fdr"] = (out["p_value"] * m / out["rank"]).clip(upper=1.0)
    return out.drop(columns=["rank"])


def benjamini_hochberg(p_values: pd.Series) -> pd.Series:
    p = pd.to_numeric(p_values, errors="coerce")
    out = pd.Series(np.nan, index=p.index, dtype=float)
    valid = p.dropna().sort_values()
    m = len(valid)
    if m == 0:
        return out
    adjusted = valid * m / np.arange(1, m + 1)
    adjusted = np.minimum.accumulate(adjusted.iloc[::-1]).iloc[::-1].clip(upper=1.0)
    out.loc[adjusted.index] = adjusted
    return out


def spearman_table(df: pd.DataFrame, x_cols: list[str], y_cols: list[str], prefix: str = "") -> pd.DataFrame:
    rows = []
    for x in x_cols:
        for y in y_cols:
            if x not in df.columns or y not in df.columns:
                rows.append({"x": x, "y": y, "rho": np.nan, "p_value": np.nan, "n": 0, "analysis": prefix})
                continue
            if x == y:
                n = pd.to_numeric(df[x], errors="coerce").dropna().shape[0]
                rows.append({"x": x, "y": y, "rho": 1.0, "p_value": 0.0, "n": n, "analysis": prefix})
                continue
            sub = df[[x, y]].apply(pd.to_numeric, errors="coerce").dropna()
            if len(sub) < 4 or sub[x].nunique() < 2 or sub[y].nunique() < 2:
                rho, p = np.nan, np.nan
            else:
                rho, p = stats.spearmanr(sub[x], sub[y])
            rows.append({"x": x, "y": y, "rho": rho, "p_value": p, "n": len(sub), "analysis": prefix})
    out = pd.DataFrame(rows)
    out["fdr"] = benjamini_hochberg(out["p_value"])
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
        if len(pos) < 2 or len(neg) < 2:
            u, p = np.nan, np.nan
        else:
            u, p = stats.mannwhitneyu(pos, neg, alternative="two-sided")
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


def remove_existing_figure_files() -> None:
    """Remove generated figure images so stale main-text panels cannot remain."""
    for path in FIGURES.iterdir():
        if path.is_file() and path.suffix.lower() in {".png", ".pdf"}:
            path.unlink()
        elif path.is_dir() and path.name == "main_figure_panels":
            for nested in path.iterdir():
                if nested.is_file() and nested.suffix.lower() in {".png", ".pdf"}:
                    nested.unlink()
    for pattern in ["figure*.tsv", "supplementary*.tsv"]:
        for path in DESIGN.glob(pattern):
            if path.is_file():
                path.unlink()


def neutral_palette(df: pd.DataFrame, hue_col: str, color: str) -> dict[str, str]:
    return {str(value): color for value in df[hue_col].dropna().astype(str).unique()}


def response_label(series: pd.Series, positive: str = "benefit/MPR", negative: str = "no benefit") -> pd.Series:
    benefit = pd.to_numeric(series, errors="coerce")
    return pd.Series(np.where(benefit.eq(1), positive, negative), index=series.index)


def response_palette(*labels: str) -> dict[str, str]:
    palette = {
        "benefit": PALETTE["positive"],
        "benefit/MPR": PALETTE["positive"],
        "MPR/pCR": PALETTE["positive"],
        "High": PALETTE["positive"],
        "Early": PALETTE["positive"],
        "complete": PALETTE["positive"],
        "no benefit": PALETTE["negative"],
        "non-MPR": PALETTE["negative"],
        "non-MPR/NE": PALETTE["negative"],
        "Low": PALETTE["negative"],
        "Advanced": PALETTE["negative"],
        "not complete": PALETTE["negative"],
        "unlabeled": PALETTE["neutral"],
    }
    if labels:
        return {label: palette.get(label, PALETTE["neutral"]) for label in labels}
    return palette


def add_figure3a_display_fields(df: pd.DataFrame, threshold: float = FIGURE3A_RESEARCH_VALUE_THRESHOLD) -> pd.DataFrame:
    out = df.copy()
    score = pd.to_numeric(out["scTIME_AI_score"], errors="coerce")
    out["research_value_threshold"] = threshold
    out["high_research_value"] = score.ge(threshold)
    out["display_alpha"] = np.where(out["high_research_value"], 0.88, 0.22)
    out["display_size"] = 30 + np.where(out["high_research_value"], 95 * score.fillna(0), 38 * score.fillna(0))
    out["display_marker"] = out["response"].map({"benefit": "o", "no benefit": "^", "unlabeled": "s"}).fillna("s")
    return out


def category_palette(values: pd.Series | list[object]) -> dict[str, str]:
    colors = [PALETTE["positive"], PALETTE["negative"], PALETTE["neutral"]]
    unique = list(dict.fromkeys(pd.Series(values).dropna().astype(str)))
    return {value: colors[i % len(colors)] for i, value in enumerate(unique)}


def model_label(value: object) -> str:
    return (
        str(value)
        .replace("_logistic", "")
        .replace("ElasticNet", "ElasticNet")
        .replace("RandomForest", "Random forest")
        .replace("RBF_SVM", "RBF SVM")
    )


def validation_label(row: pd.Series) -> str:
    if str(row.get("evaluation", "")).startswith("5-fold"):
        return "Pooled CV"
    train = clean_label(row.get("train_dataset", ""))
    test = clean_label(row.get("test_dataset", ""))
    return f"{train} -> {test}"


def metric_ci(values: list[float]) -> tuple[float, float]:
    clean = pd.Series(values, dtype=float).dropna()
    if clean.empty:
        return np.nan, np.nan
    return float(clean.quantile(0.025)), float(clean.quantile(0.975))


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
        estimates = {
            "ROC-AUC": roc_auc_score(y, p),
            "PR-AUC": average_precision_score(y, p),
            "Brier": np.mean((y - p) ** 2),
        }
        boot = {metric: [] for metric in estimates}
        for _ in range(n_boot):
            idx = rng.integers(0, len(sub), len(sub))
            if np.unique(y[idx]).size < 2:
                continue
            boot["ROC-AUC"].append(roc_auc_score(y[idx], p[idx]))
            boot["PR-AUC"].append(average_precision_score(y[idx], p[idx]))
            boot["Brier"].append(np.mean((y[idx] - p[idx]) ** 2))
        for metric, estimate in estimates.items():
            low, high = metric_ci(boot[metric])
            rows.append(
                {
                    "validation": label,
                    "metric": metric,
                    "estimate": estimate,
                    "ci_low": low,
                    "ci_high": high,
                    "n": len(sub),
                    "positives": int(y.sum()),
                }
            )
    return pd.DataFrame(rows)


def add_activation_exclusion_indices(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["immune_activation_index"] = out[ACTIVATION_FEATURES].apply(pd.to_numeric, errors="coerce").mean(axis=1)
    out["immune_exclusion_index"] = out[EXCLUSION_FEATURES].apply(pd.to_numeric, errors="coerce").mean(axis=1)
    out["activation_exclusion_balance"] = out["immune_activation_index"] - out["immune_exclusion_index"]
    return out


def lollipop(
    ax: plt.Axes,
    labels: pd.Series,
    values: pd.Series,
    title: str,
    xlabel: str,
    *,
    colors: list[str] | None = None,
) -> None:
    plot = pd.DataFrame({"label": labels.astype(str), "value": pd.to_numeric(values, errors="coerce")}).dropna()
    plot = plot.sort_values("value")
    y = np.arange(len(plot))
    if colors is None:
        colors = [PALETTE["negative"] if v < 0 else PALETTE["positive"] for v in plot["value"]]
    else:
        colors = [colors[i] for i in plot.index]
    ax.hlines(y=y, xmin=0, xmax=plot["value"], color=PALETTE["neutral"], linewidth=1.0, alpha=0.5, zorder=1)
    ax.scatter(plot["value"], y, color=colors, s=38, zorder=2)
    ax.axvline(0, color=PALETTE["dark"], linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(plot["label"])
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("")


def dot_matrix(
    ax: plt.Axes,
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    color_col: str,
    p_col: str | None,
    title: str,
    cbar_label: str,
    *,
    vmin: float = -1,
    vmax: float = 1,
    cbar_fraction: float = 0.046,
    cbar_pad: float = 0.03,
    cbar_shrink: float = 1.0,
) -> None:
    plot = df[[x_col, y_col, color_col] + ([p_col] if p_col else [])].dropna(subset=[x_col, y_col, color_col]).copy()
    if plot.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        ax.set_axis_off()
        return
    x_order = list(dict.fromkeys(plot[x_col].astype(str)))
    y_order = list(dict.fromkeys(plot[y_col].astype(str)))
    x_codes = plot[x_col].astype(str).map({v: i for i, v in enumerate(x_order)})
    y_codes = plot[y_col].astype(str).map({v: i for i, v in enumerate(y_order)})
    if p_col:
        p = pd.to_numeric(plot[p_col], errors="coerce").clip(lower=1e-300)
        sizes = 35 + np.clip(-np.log10(p).fillna(0), 0, 8) * 18
    else:
        sizes = np.repeat(80, len(plot))
    sc = ax.scatter(
        x_codes,
        y_codes,
        c=pd.to_numeric(plot[color_col], errors="coerce"),
        s=sizes,
        marker="s",
        cmap=DIVERGING_CMAP,
        vmin=vmin,
        vmax=vmax,
        edgecolor="white",
        linewidth=0.5,
    )
    ax.set_xticks(range(len(x_order)))
    ax.set_xticklabels(x_order, rotation=35, ha="right")
    ax.set_yticks(range(len(y_order)))
    ax.set_yticklabels(y_order)
    ax.set_title(title)
    ax.set_xlabel("")
    ax.set_ylabel("")
    cbar = ax.figure.colorbar(sc, ax=ax, fraction=cbar_fraction, pad=cbar_pad, shrink=cbar_shrink)
    cbar.set_label(cbar_label)


def grouped_dot_matrix(
    ax: plt.Axes,
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    group_col: str,
    color_col: str,
    p_col: str | None,
    title: str,
    cbar_label: str,
    group_label: str,
    *,
    vmin: float = -1,
    vmax: float = 1,
    legend_outside_top: bool = False,
) -> None:
    """Draw stratified correlation estimates without overplotting cohorts."""

    columns = [x_col, y_col, group_col, color_col] + ([p_col] if p_col else [])
    plot = df[columns].dropna(subset=[x_col, y_col, group_col, color_col]).copy()
    if plot.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        ax.set_axis_off()
        return

    x_order = list(dict.fromkeys(plot[x_col].astype(str)))
    y_order = list(dict.fromkeys(plot[y_col].astype(str)))
    group_order = list(dict.fromkeys(plot[group_col].astype(str)))
    x_map = {value: index for index, value in enumerate(x_order)}
    y_map = {value: index for index, value in enumerate(y_order)}
    markers = ["s", "o", "^", "D", "P", "X"]
    offsets = np.linspace(-0.27, 0.27, len(group_order)) if len(group_order) > 1 else np.array([0.0])
    mappable = None
    legend_handles = []

    for group_index, group in enumerate(group_order):
        sub = plot[plot[group_col].astype(str).eq(group)].copy()
        x_codes = sub[x_col].astype(str).map(x_map).astype(float) + offsets[group_index]
        y_codes = sub[y_col].astype(str).map(y_map).astype(float)
        if p_col:
            p_values = pd.to_numeric(sub[p_col], errors="coerce").clip(lower=1e-300)
            sizes = 24 + np.clip(-np.log10(p_values).fillna(0), 0, 8) * 11
        else:
            sizes = np.repeat(55, len(sub))
        marker = markers[group_index % len(markers)]
        mappable = ax.scatter(
            x_codes,
            y_codes,
            c=pd.to_numeric(sub[color_col], errors="coerce"),
            s=sizes,
            marker=marker,
            cmap=DIVERGING_CMAP,
            vmin=vmin,
            vmax=vmax,
            edgecolor="white",
            linewidth=0.45,
        )
        legend_handles.append(
            ax.scatter([], [], s=34, marker=marker, color=PALETTE["neutral"], edgecolor="white", linewidth=0.45, label=group)
        )

    ax.set_xticks(range(len(x_order)))
    ax.set_xticklabels(x_order, rotation=35, ha="right")
    ax.set_yticks(range(len(y_order)))
    ax.set_yticklabels(y_order)
    ax.set_xlim(-0.55, len(x_order) - 0.45)
    ax.set_title(title, pad=44 if legend_outside_top else None)
    ax.set_xlabel("")
    ax.set_ylabel("")
    legend_kwargs = {
        "handles": legend_handles,
        "title": group_label,
        "frameon": False,
        "fontsize": 5.5,
        "title_fontsize": 6,
    }
    if legend_outside_top:
        legend_kwargs.update(
            {
                "loc": "lower center",
                "bbox_to_anchor": (0.5, 1.01),
                "ncol": len(group_order),
                "columnspacing": 0.8,
                "handletextpad": 0.35,
            }
        )
    else:
        legend_kwargs["loc"] = "best"
    ax.legend(**legend_kwargs)
    if mappable is not None:
        cbar = ax.figure.colorbar(mappable, ax=ax, fraction=0.046, pad=0.03)
        cbar.set_label(cbar_label)


def wilson_interval(successes: pd.Series, totals: pd.Series, z: float = 1.96) -> tuple[pd.Series, pd.Series]:
    k = pd.to_numeric(successes, errors="coerce").astype(float)
    n = pd.to_numeric(totals, errors="coerce").astype(float)
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt((p * (1 - p) + z**2 / (4 * n)) / n) / denom
    return (center - half).clip(lower=0), (center + half).clip(upper=1)


def plot_main_single_cell_response_landscape() -> None:
    emb = read_table("gse207422_scanpy_cell_embeddings")
    comp = read_table("gse243013_sc_state_composition")
    diff = read_table("gse243013_sc_state_response_differential")

    fig, axes = plt.subplot_mosaic(
        [["A", "A", "B"], ["A", "A", "C"], ["D", "D", "E"]],
        figsize=(13.2, 9.0),
        constrained_layout=True,
    )
    ax1 = axes["A"]
    ax2 = axes["B"]
    ax3 = axes["C"]
    ax4 = axes["D"]
    ax5 = axes["E"]

    rng = np.random.default_rng(7)
    emb_plot = emb.iloc[rng.choice(len(emb), min(len(emb), 35000), replace=False)].copy()
    state_palette = dict(zip(sorted(emb["cell_state"].dropna().unique()), sns.color_palette("tab20", n_colors=emb["cell_state"].nunique())))
    sns.scatterplot(
        data=emb_plot,
        x="UMAP1",
        y="UMAP2",
        hue="cell_state",
        s=2.0,
        linewidth=0,
        alpha=0.72,
        palette=state_palette,
        ax=ax1,
    )
    ax1.set_title("GSE207422 single-cell immune landscape")
    ax1.set_xlabel("UMAP1")
    ax1.set_ylabel("UMAP2")
    ax1.legend(frameon=False, markerscale=4, bbox_to_anchor=(0.01, -0.08), loc="upper left", fontsize=6, ncol=3)
    panel_label(ax1, "A")

    diff_plot = diff.copy()
    diff_plot["signed_log10_p"] = signed_log10_p(diff_plot["p_value"], diff_plot["mean_difference"])
    lollipop(
        ax2,
        diff_plot["state_category"],
        diff_plot["signed_log10_p"],
        "Response-associated state shifts",
        "signed -log10(P), MPR minus non-MPR",
    )
    panel_label(ax2, "B")

    key_states = (
        diff_plot.assign(abs_signal=diff_plot["signed_log10_p"].abs())
        .sort_values("abs_signal", ascending=False)
        .head(6)["state_category"]
        .tolist()
    )
    comp_plot = comp[comp["state_category"].isin(key_states)].dropna(subset=["benefit"]).copy()
    comp_plot["response"] = response_label(comp_plot["benefit"], "MPR/pCR", "non-MPR")
    order = [s for s in key_states if s in comp_plot["state_category"].unique()]
    sns.boxplot(data=comp_plot, x="fraction", y="state_category", hue="response", order=order, fliersize=0, palette=response_palette("non-MPR", "MPR/pCR"), ax=ax3)
    sns.stripplot(data=comp_plot, x="fraction", y="state_category", hue="response", order=order, dodge=True, palette=neutral_palette(comp_plot, "response", PALETTE["neutral"]), size=1.6, alpha=0.35, legend=False, ax=ax3)
    ax3.set_title("Patient-level fractions of response-informative states")
    ax3.set_xlabel("Fraction of cells")
    ax3.set_ylabel("")
    ax3.legend(frameon=False, title="")
    panel_label(ax3, "C")

    heat = comp.pivot_table(index="sampleID", columns="state_category", values="fraction", fill_value=0)
    benefit = comp.groupby("sampleID", observed=True)["benefit"].first()
    state_order = [s for s in diff_plot.assign(abs_signal=diff_plot["signed_log10_p"].abs()).sort_values("abs_signal", ascending=False)["state_category"] if s in heat.columns]
    heat = heat.loc[benefit.sort_values(ascending=False).index, state_order[:10]]
    sns.heatmap(
        heat,
        cmap=SEQUENTIAL_CMAP,
        ax=ax4,
        cbar_kws={"label": "Fraction"},
        linewidths=0.2,
        linecolor="white",
    )
    ax4.set_title("Patient-by-state composition structure")
    ax4.set_xlabel("Cell state")
    ax4.set_ylabel("Patient")
    ax4.tick_params(axis="x", rotation=25)
    ax4.tick_params(axis="y", labelsize=5)
    panel_label(ax4, "D")

    network = diff_plot.copy()
    network["effect"] = pd.to_numeric(network["mean_difference"], errors="coerce")
    network["signal"] = pd.to_numeric(network["signed_log10_p"], errors="coerce").abs()
    network = network.dropna(subset=["effect", "signal"]).sort_values("signal", ascending=False).head(10)
    if not network.empty:
        theta = np.linspace(0, 2 * np.pi, len(network), endpoint=False)
        radius = 1.0 + 0.18 * np.sqrt(network["signal"].to_numpy())
        x = radius * np.cos(theta)
        y = radius * np.sin(theta)
        sizes = 100 + 90 * np.sqrt(network["signal"].to_numpy())
        colors = [PALETTE["positive"] if v >= 0 else PALETTE["negative"] for v in network["effect"]]
        for xi, yi, row in zip(x, y, network.itertuples(index=False)):
            ax5.plot([0, xi], [0, yi], color=PALETTE["neutral"], linewidth=0.8, alpha=0.55)
            ha = "left" if xi >= 0 else "right"
            ax5.annotate(
                row.state_category,
                (xi, yi),
                xytext=(7 if xi >= 0 else -7, 0),
                textcoords="offset points",
                ha=ha,
                va="center",
                fontsize=5.5,
            )
        ax5.scatter(x, y, s=sizes, c=colors, edgecolor="white", linewidth=0.7, zorder=3)
        ax5.scatter([0], [0], s=320, c=PALETTE["neutral"], edgecolor="white", linewidth=0.8, zorder=4)
        ax5.text(0, 0, "MPR\nshift", ha="center", va="center", color="white", fontsize=7)
    ax5.margins(x=0.42, y=0.25)
    ax5.set_title("Response-linked single-cell\nstate network", pad=12)
    ax5.set_aspect("equal")
    ax5.set_axis_off()
    panel_label_tight(ax5, "E")

    savefig(fig, "figure2_single_cell_response_landscape")


def plot_main_ai_reliability_sctime_modeling() -> None:
    bulk = read_table("scTIME_AI_scores_all_bulk_cohorts")
    pred = read_table("model_predictions")
    perf = read_table("model_performance")
    cal = read_table("elasticnet_calibration")
    dca = read_table("elasticnet_decision_curve")
    coef = read_table("elasticnet_coefficients")
    contrib = read_table("elasticnet_linear_shap_style_contributions")
    labeled = add_activation_exclusion_indices(bulk[bulk["dataset"].isin(["GSE126044", "GSE207422_bulk"])].dropna(subset=["benefit"]).copy())

    z_features = [f"z_{f}" for f in FEATURES if f"z_{f}" in bulk.columns]
    pca_input = bulk.dropna(subset=z_features).copy()
    pca = PCA(n_components=2, random_state=1)
    coords = pca.fit_transform(pca_input[z_features])
    pca_df = pca_input[["dataset", "sample", "benefit", "scTIME_AI_score"]].copy()
    pca_df["PC1"] = coords[:, 0]
    pca_df["PC2"] = coords[:, 1]
    pca_df["response"] = np.where(pca_df["benefit"].eq(1), "benefit", np.where(pca_df["benefit"].eq(0), "no benefit", "unlabeled"))
    pca_df = add_figure3a_display_fields(pca_df)
    pca_df.to_csv(DESIGN / "figure3_bulk_immune_state_pca.tsv", sep="\t", index=False)

    response_tests = binary_group_tests(labeled, "benefit", ["immune_activation_index", "immune_exclusion_index", "activation_exclusion_balance", "scTIME_AI_score"])
    response_tests.to_csv(DESIGN / "figure3_activation_exclusion_response_tests.tsv", sep="\t", index=False)

    cv_pred = pred[pred["model"].eq("ElasticNet_logistic")].dropna(subset=["benefit", "prediction"]).copy()
    boot = bootstrap_prediction_metrics(cv_pred)
    boot.to_csv(DESIGN / "figure3_elasticnet_bootstrap_reliability.tsv", sep="\t", index=False)

    cal = cal.copy()
    if not cal.empty:
        cal["ci_low"], cal["ci_high"] = wilson_interval(cal["observed_rate"] * cal["n"], cal["n"])
        cal.to_csv(DESIGN / "figure3_elasticnet_calibration_wilson.tsv", sep="\t", index=False)

    perf_plot = perf.copy()
    perf_plot["validation"] = perf_plot.apply(validation_label, axis=1)
    perf_plot["model_label"] = perf_plot["model"].map(model_label)
    perf_long = perf_plot.melt(
        id_vars=["validation", "model_label"],
        value_vars=["roc_auc", "pr_auc", "brier"],
        var_name="metric",
        value_name="value",
    )
    perf_long["metric"] = perf_long["metric"].map({"roc_auc": "ROC-AUC", "pr_auc": "PR-AUC", "brier": "Brier"})
    perf_long.to_csv(DESIGN / "figure3_model_benchmark_long.tsv", sep="\t", index=False)
    enet_long = perf_long[perf_long["model_label"].eq("ElasticNet")].copy()
    enet_long.to_csv(DESIGN / "figure3_elasticnet_validation_metrics.tsv", sep="\t", index=False)

    fig, axes = plt.subplot_mosaic(
        [["A", "B", "B"], ["C", "D", "E"], ["F", "G", "H"], ["I", "I", "I"]],
        figsize=(13.5, 12.5),
        constrained_layout=True,
    )

    score_norm = Normalize(
        vmin=float(pd.to_numeric(pca_df["scTIME_AI_score"], errors="coerce").min()),
        vmax=float(pd.to_numeric(pca_df["scTIME_AI_score"], errors="coerce").max()),
    )
    marker_map = {"benefit": "o", "no benefit": "^", "unlabeled": "s"}
    label_map = {"benefit": "benefit", "no benefit": "no benefit", "unlabeled": "unlabeled"}
    for response, marker in marker_map.items():
        response_has_label = False
        for high_value, alpha in [(False, 0.22), (True, 0.88)]:
            sub = pca_df[pca_df["response"].eq(response) & pca_df["high_research_value"].eq(high_value)]
            if sub.empty:
                continue
            label = None
            if high_value or not response_has_label:
                label = label_map[response]
                response_has_label = True
            axes["A"].scatter(
                sub["PC1"],
                sub["PC2"],
                c=sub["scTIME_AI_score"],
                cmap=SEQUENTIAL_CMAP,
                norm=score_norm,
                s=sub["display_size"],
                alpha=alpha,
                marker=marker,
                edgecolor="white",
                linewidth=0.4,
                label=label,
            )
    sc = plt.cm.ScalarMappable(cmap=SEQUENTIAL_CMAP, norm=score_norm)
    sc.set_array([])
    axes["A"].axvline(0, color=PALETTE["neutral"], linewidth=0.6, alpha=0.35)
    axes["A"].axhline(0, color=PALETTE["neutral"], linewidth=0.6, alpha=0.35)
    axes["A"].text(
        0.02,
        0.98,
        f"Highlighted: scTIME >= {FIGURE3A_RESEARCH_VALUE_THRESHOLD:.2f}",
        transform=axes["A"].transAxes,
        va="top",
        ha="left",
        fontsize=6,
        color=PALETTE["dark"],
    )
    cbar = fig.colorbar(sc, ax=axes["A"], fraction=0.046, pad=0.03)
    cbar.set_label("scTIME score")
    axes["A"].set_title("Bulk immune-state PCA")
    axes["A"].set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}%)")
    axes["A"].set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}%)")
    axes["A"].legend(frameon=False, fontsize=6, loc="lower right", title="")
    panel_label_corner(axes["A"], "A")

    benchmark = enet_long.pivot_table(index="metric", columns="validation", values="value")
    benchmark = benchmark.reindex(["ROC-AUC", "PR-AUC", "Brier"])
    sns.heatmap(
        benchmark,
        cmap=DIVERGING_CMAP,
        vmin=0,
        vmax=1,
        annot=True,
        fmt=".2f",
        linewidths=0.4,
        linecolor="white",
        cbar_kws={"label": "Metric value"},
        ax=axes["B"],
    )
    axes["B"].set_title("scTIME reliability across evaluation designs")
    axes["B"].set_xlabel("")
    axes["B"].set_ylabel("Metric")
    axes["B"].tick_params(axis="x", rotation=25)
    panel_label_corner(axes["B"], "B")

    curve_colors = category_palette(cv_pred["dataset"])
    for dataset, sub in cv_pred.groupby("dataset", observed=True):
        if sub["benefit"].nunique() < 2:
            continue
        y = sub["benefit"].astype(int)
        score = sub["prediction"].astype(float)
        fpr, tpr, _ = roc_curve(y, score)
        auc = roc_auc_score(y, score)
        axes["C"].plot(fpr, tpr, label=f"{clean_label(dataset)} AUC={auc:.2f}", linewidth=1.5, color=curve_colors[str(dataset)])
    if cv_pred["benefit"].nunique() == 2:
        y = cv_pred["benefit"].astype(int)
        score = cv_pred["prediction"].astype(float)
        fpr, tpr, _ = roc_curve(y, score)
        auc = roc_auc_score(y, score)
        axes["C"].plot(fpr, tpr, label=f"Pooled AUC={auc:.2f}", linewidth=2.0, color=PALETTE["dark"])
    axes["C"].plot([0, 1], [0, 1], linestyle="--", color=PALETTE["neutral"], linewidth=0.8)
    axes["C"].set_title("ElasticNet ROC stability")
    axes["C"].set_xlabel("False positive rate")
    axes["C"].set_ylabel("True positive rate")
    axes["C"].legend(frameon=False, fontsize=6)
    panel_label_corner(axes["C"], "C")

    for dataset, sub in cv_pred.groupby("dataset", observed=True):
        if sub["benefit"].nunique() < 2:
            continue
        y = sub["benefit"].astype(int)
        score = sub["prediction"].astype(float)
        precision, recall, _ = precision_recall_curve(y, score)
        ap = average_precision_score(y, score)
        axes["D"].plot(recall, precision, label=f"{clean_label(dataset)} AP={ap:.2f}", linewidth=1.5, color=curve_colors[str(dataset)])
    if cv_pred["benefit"].nunique() == 2:
        y = cv_pred["benefit"].astype(int)
        score = cv_pred["prediction"].astype(float)
        precision, recall, _ = precision_recall_curve(y, score)
        ap = average_precision_score(y, score)
        axes["D"].plot(recall, precision, label=f"Pooled AP={ap:.2f}", linewidth=2.0, color=PALETTE["dark"])
        axes["D"].axhline(y.mean(), linestyle="--", color=PALETTE["neutral"], linewidth=0.8)
    axes["D"].set_title("ElasticNet precision-recall stability")
    axes["D"].set_xlabel("Recall")
    axes["D"].set_ylabel("Precision")
    axes["D"].legend(frameon=False, fontsize=6)
    panel_label_corner(axes["D"], "D")

    if not cal.empty:
        xerr = None
        yerr = np.vstack([cal["observed_rate"] - cal["ci_low"], cal["ci_high"] - cal["observed_rate"]])
        axes["E"].errorbar(cal["mean_prediction"], cal["observed_rate"], yerr=yerr, xerr=xerr, fmt="o", color=PALETTE["positive"], ecolor=PALETTE["neutral"], capsize=3)
        for row in cal.itertuples(index=False):
            axes["E"].text(row.mean_prediction, min(row.ci_high + 0.05, 1), f"n={int(row.n)}", ha="center", fontsize=6)
    axes["E"].plot([0, 1], [0, 1], linestyle="--", color=PALETTE["neutral"], linewidth=0.8)
    axes["E"].set_xlim(0, 1)
    axes["E"].set_ylim(0, 1)
    axes["E"].set_title("Calibration with Wilson intervals")
    axes["E"].set_xlabel("Mean predicted probability")
    axes["E"].set_ylabel("Observed benefit rate")
    panel_label_corner(axes["E"], "E")

    sns.lineplot(data=dca, x="threshold", y="net_benefit_model", ax=axes["F"], label="scTIME", color=PALETTE["positive"], linewidth=1.6)
    sns.lineplot(data=dca, x="threshold", y="net_benefit_treat_all", ax=axes["F"], label="Treat all", color=PALETTE["negative"], linewidth=1.2)
    sns.lineplot(data=dca, x="threshold", y="net_benefit_treat_none", ax=axes["F"], label="Treat none", color=PALETTE["neutral"], linewidth=1.2)
    axes["F"].set_title("Decision-curve analysis")
    axes["F"].set_xlabel("Threshold probability")
    axes["F"].set_ylabel("Net benefit")
    axes["F"].legend(frameon=False, fontsize=6)
    panel_label_corner(axes["F"], "F")

    boot_plot = boot[boot["metric"].isin(["ROC-AUC", "PR-AUC", "Brier"])].copy()
    if not boot_plot.empty:
        boot_plot["label"] = boot_plot["validation"] + "\n" + boot_plot["metric"]
        boot_plot = boot_plot.sort_values(["metric", "validation"])
        y = np.arange(len(boot_plot))
        xerr = np.vstack([boot_plot["estimate"] - boot_plot["ci_low"], boot_plot["ci_high"] - boot_plot["estimate"]])
        colors = [PALETTE["negative"] if metric == "Brier" else PALETTE["positive"] for metric in boot_plot["metric"]]
        axes["G"].errorbar(boot_plot["estimate"], y, xerr=xerr, fmt="o", color=PALETTE["dark"], ecolor=PALETTE["neutral"], capsize=3)
        axes["G"].scatter(boot_plot["estimate"], y, c=colors, s=42, zorder=3)
        axes["G"].set_yticks(y)
        axes["G"].set_yticklabels(boot_plot["label"])
        axes["G"].set_xlim(0, 1.05)
    axes["G"].set_title("Bootstrap reliability intervals")
    axes["G"].set_xlabel("Metric estimate with 95% interval")
    axes["G"].set_ylabel("")
    panel_label_corner(axes["G"], "G")

    coef_plot = coef.copy()
    coef_plot["label"] = coef_plot["feature"].map(lambda x: FEATURE_LABELS.get(x, x))
    lollipop(axes["H"], coef_plot["label"], coef_plot["coefficient"], "Interpretable scTIME coefficients", "ElasticNet coefficient")
    panel_label_corner(axes["H"], "H")

    contrib_cols = [c for c in FEATURES if c in contrib.columns]
    contribution = contrib.merge(labeled[["dataset", "sample", "benefit"]], on=["dataset", "sample"], how="left")
    contribution["sort_key"] = contribution["prediction"]
    contribution = contribution.sort_values(["benefit", "sort_key"], ascending=[False, False])
    heat = contribution.set_index(["dataset", "sample"])[contrib_cols]
    heat.columns = [FEATURE_LABELS.get(c, c) for c in heat.columns]
    vmax = np.nanquantile(np.abs(heat.to_numpy()).ravel(), 0.98)
    sns.heatmap(
        heat.T,
        cmap=DIVERGING_CMAP,
        center=0,
        vmin=-vmax,
        vmax=vmax,
        ax=axes["I"],
        cbar_kws={"label": "Logit contribution"},
        xticklabels=False,
        linewidths=0.05,
        linecolor="white",
    )
    axes["I"].set_title("Patient-level feature contribution profile")
    axes["I"].set_xlabel("Labeled bulk patients ordered by response and scTIME score")
    axes["I"].set_ylabel("")
    panel_label_corner(axes["I"], "I")

    savefig(fig, "figure3_ai_reliability_sctime_modeling")


def plot_main_cxcl10_ifn_antigen_axis() -> None:
    bulk = read_table("scTIME_AI_scores_all_bulk_cohorts")
    pseudo = read_table("gse207422_sc_pseudobulk_signature_scores")
    lr = read_table("gse207422_lr_axis_scores")
    ligand = read_table("gse207422_nichenet_like_ligand_activity")
    tcga = read_table("tcga_sctime_ai_scores")
    cptac = read_table("cptac_sctime_ai_projection")

    labeled = bulk[bulk["dataset"].isin(["GSE126044", "GSE207422_bulk"])].dropna(subset=["benefit"]).copy()
    labeled["response"] = response_label(labeled["benefit"])
    labeled["IFN_antigen_index"] = labeled[["IFN_response", "Antigen_presentation"]].mean(axis=1)
    labeled["CXCL10_CXCR3_axis"] = labeled["z_CXCL10_CXCR3_axis"]
    test_features = ["CXCL10_CXCR3_axis", "IFN_response", "Antigen_presentation", "Cytotoxic_CD8", "CD274_expr"]
    tests = read_table("feature_response_associations")
    tests = tests[tests["feature"].isin(test_features)].rename(
        columns={"positive_mean_z": "positive_mean", "negative_mean_z": "negative_mean"}
    )
    tests.to_csv(DESIGN / "figure4_cxcl10_ifn_axis_response_tests.tsv", sep="\t", index=False)

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
    corr.to_csv(DESIGN / "figure4_cxcl10_cross_context_correlations.tsv", sep="\t", index=False)

    fig, axes = plt.subplot_mosaic(
        [["A", "B", "C"], ["D", "E", "E"]],
        figsize=(12, 7.6),
        constrained_layout=True,
    )

    sns.boxplot(data=labeled, x="dataset", y="CXCL10_CXCR3_axis", hue="response", fliersize=0, palette=response_palette("no benefit", "benefit/MPR"), ax=axes["A"])
    sns.stripplot(data=labeled, x="dataset", y="CXCL10_CXCR3_axis", hue="response", dodge=True, palette=neutral_palette(labeled, "response", PALETTE["neutral"]), size=2.1, alpha=0.45, legend=False, ax=axes["A"])
    axes["A"].set_title("CXCL10-CXCR3 axis by response")
    axes["A"].set_xlabel("")
    axes["A"].set_ylabel("Module score")
    axes["A"].legend(frameon=False, title="")
    panel_label(axes["A"], "A")

    sns.regplot(data=labeled, x="IFN_antigen_index", y="CXCL10_CXCR3_axis", scatter_kws={"s": 42, "alpha": 0.75}, line_kws={"color": PALETTE["dark"]}, ax=axes["B"])
    rho, p = stats.spearmanr(labeled["IFN_antigen_index"], labeled["CXCL10_CXCR3_axis"])
    axes["B"].text(0.05, 0.95, f"rho={rho:.2f}\nP={p:.2g}", transform=axes["B"].transAxes, va="top")
    axes["B"].set_title("IFN-antigen index links to CXCL10-CXCR3")
    axes["B"].set_xlabel("IFN-antigen index")
    axes["B"].set_ylabel("CXCL10-CXCR3 axis")
    panel_label(axes["B"], "B")

    lr_cxcl = lr[lr["axis"].eq("CXCL9/10/11-CXCR3")].dropna(subset=["benefit"]).copy()
    lr_cxcl["response"] = response_label(lr_cxcl["benefit"], "MPR/pCR", "NMPR")
    sns.boxplot(data=lr_cxcl, x="response", y="interaction_score", hue="response", fliersize=0, palette=response_palette("NMPR", "MPR/pCR"), legend=False, ax=axes["C"])
    sns.stripplot(data=lr_cxcl, x="response", y="interaction_score", color=PALETTE["neutral"], size=3, alpha=0.65, ax=axes["C"])
    axes["C"].set_title("Single-cell CXCL9/10/11-CXCR3 LR score")
    axes["C"].set_xlabel("")
    axes["C"].set_ylabel("Interaction score")
    panel_label(axes["C"], "C")

    lig_plot = ligand.copy()
    lig_plot["minus_log10_p"] = -np.log10(pd.to_numeric(lig_plot["spearman_p_value"], errors="coerce").clip(lower=1e-300))
    axes["D"].scatter(
        lig_plot["spearman_with_effector_score"],
        lig_plot["minus_log10_p"],
        c=lig_plot["mpr_mean_ligand_score"],
        s=45 + 175 * (lig_plot["mpr_mean_ligand_score"] - lig_plot["mpr_mean_ligand_score"].min()) / max(lig_plot["mpr_mean_ligand_score"].max() - lig_plot["mpr_mean_ligand_score"].min(), 1e-9),
        cmap=SEQUENTIAL_CMAP,
        edgecolor="white",
        linewidth=0.5,
    )
    for row in lig_plot.sort_values("minus_log10_p", ascending=False).head(4).itertuples(index=False):
        axes["D"].text(row.spearman_with_effector_score, row.minus_log10_p, str(row.axis), fontsize=6, ha="left", va="bottom")
    axes["D"].axvline(0, color=PALETTE["dark"], linewidth=0.8)
    axes["D"].set_title("Ligand activity against effector module")
    axes["D"].set_xlabel("Spearman rho")
    axes["D"].set_ylabel("-log10(P)")
    panel_label(axes["D"], "D")

    corr_plot = corr.copy()
    corr_plot["target"] = corr_plot["y"].map(lambda x: FEATURE_LABELS.get(x, x))
    dot_matrix(
        axes["E"],
        corr_plot,
        "analysis",
        "target",
        "rho",
        "p_value",
        "CXCL10-CXCR3 cross-context correlations",
        "Spearman rho",
    )
    panel_label(axes["E"], "E")

    savefig(fig, "figure4_cxcl10_ifn_antigen_axis")


def plot_main_myeloid_tgf_hypoxia_exclusion_axis() -> None:
    bulk = read_table("scTIME_AI_scores_all_bulk_cohorts")
    fractions = read_table("music_style_nnls_cell_fractions")
    tcga = read_table("tcga_sctime_ai_scores")
    cptac = read_table("cptac_sctime_ai_projection")
    labeled = bulk[bulk["dataset"].isin(["GSE126044", "GSE207422_bulk"])].dropna(subset=["benefit"]).copy()
    exclusion_features = ["SPP1_macrophage"] + EXCLUSION_FEATURES
    exclusion_tests = read_table("feature_response_associations")
    exclusion_tests = exclusion_tests[exclusion_tests["feature"].isin(exclusion_features)].rename(
        columns={"positive_mean_z": "positive_mean", "negative_mean_z": "negative_mean"}
    )
    exclusion_tests.to_csv(DESIGN / "figure5_exclusion_axis_response_tests.tsv", sep="\t", index=False)

    deconv = fractions.merge(
        bulk[["dataset", "sample", "benefit", "SPP1_macrophage", "TGF_beta_EMT", "Hypoxia", "scTIME_AI_score"]],
        on=["dataset", "sample"],
        how="inner",
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
    deconv_corr.to_csv(DESIGN / "figure5_deconvolution_exclusion_correlations.tsv", sep="\t", index=False)

    tcga = add_activation_exclusion_indices(tcga)
    cptac = add_activation_exclusion_indices(cptac)
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
    tcga_corr.to_csv(DESIGN / "figure5_tcga_exclusion_coprogram_correlations.tsv", sep="\t", index=False)

    fig, axes = plt.subplot_mosaic(
        [["A", "B", "C"], ["D", "E", "F"]],
        figsize=(13.0, 8.0),
        constrained_layout=True,
    )

    effects = exclusion_tests[exclusion_tests["feature"].isin(exclusion_features)].copy()
    effects["label"] = effects["feature"].map(lambda x: FEATURE_LABELS.get(x, x))
    lollipop(axes["A"], effects["label"], effects["mean_difference"], "Exclusion-feature response effects", "Responder minus non-responder mean score")
    panel_label(axes["A"], "A")

    if "Myeloid/macrophage" in deconv.columns:
        cohort_colors = category_palette(deconv["dataset"])
        cohort_stats = []
        for dataset, cohort in deconv.groupby("dataset", observed=True):
            sub = cohort[["Myeloid/macrophage", "SPP1_macrophage"]].dropna()
            color = cohort_colors[str(dataset)]
            axes["B"].scatter(
                sub["Myeloid/macrophage"],
                sub["SPP1_macrophage"],
                s=21,
                alpha=0.58,
                color=color,
                label=clean_label(dataset),
            )
            if len(sub) >= 4:
                rho, p = stats.spearmanr(sub["Myeloid/macrophage"], sub["SPP1_macrophage"])
                sns.regplot(
                    data=sub,
                    x="Myeloid/macrophage",
                    y="SPP1_macrophage",
                    scatter=False,
                    ci=None,
                    line_kws={"color": color, "linewidth": 1.2},
                    ax=axes["B"],
                )
                cohort_stats.append(f"{clean_label(dataset)}: rho={rho:.2f}, P={p:.2g}")
        axes["B"].text(
            0.03,
            0.97,
            "\n".join(cohort_stats),
            transform=axes["B"].transAxes,
            va="top",
            fontsize=6,
        )
        axes["B"].legend(frameon=False, fontsize=5.5, loc="lower right", title="")
    axes["B"].set_title("Cohort-stratified myeloid-SPP1 consistency")
    axes["B"].set_xlabel("NNLS myeloid/macrophage fraction")
    axes["B"].set_ylabel("SPP1 macrophage module")
    panel_label(axes["B"], "B")

    deconv_plot = deconv_corr.copy()
    deconv_plot["module"] = deconv_plot["y"].map(lambda x: FEATURE_LABELS.get(x, x))
    deconv_plot["cohort"] = deconv_plot["analysis"].str.replace(r"^bulk NNLS ", "", regex=True).map(clean_label)
    grouped_dot_matrix(
        axes["C"],
        deconv_plot,
        "x",
        "module",
        "cohort",
        "rho",
        "p_value",
        "Cohort-stratified deconvolution-module correlations",
        "Spearman rho",
        "Cohort",
        legend_outside_top=True,
    )
    panel_label(axes["C"], "C")

    tcga_plot = tcga.dropna(subset=["immune_exclusion_index", "scTIME_AI_score"])
    project_colors = category_palette(tcga_plot["project"])
    project_stats = []
    for project, cohort in tcga_plot.groupby("project", observed=True):
        color = project_colors[str(project)]
        axes["D"].scatter(
            cohort["immune_exclusion_index"],
            cohort["scTIME_AI_score"],
            s=9,
            alpha=0.18,
            color=color,
            linewidth=0,
            label=project,
        )
        rho, p = stats.spearmanr(cohort["immune_exclusion_index"], cohort["scTIME_AI_score"])
        sns.regplot(
            data=cohort,
            x="immune_exclusion_index",
            y="scTIME_AI_score",
            scatter=False,
            ci=None,
            line_kws={"color": color, "linewidth": 1.4},
            ax=axes["D"],
        )
        project_stats.append(f"{project}: rho={rho:.2f}, P={p:.2g}")
    axes["D"].text(
        0.03,
        0.97,
        "\n".join(project_stats),
        transform=axes["D"].transAxes,
        va="top",
        fontsize=6,
    )
    axes["D"].legend(frameon=False, fontsize=6, loc="lower right", title="")
    axes["D"].set_title("TCGA project-stratified exclusion context")
    axes["D"].set_xlabel("Exclusion index")
    axes["D"].set_ylabel("scTIME score")
    panel_label(axes["D"], "D")

    cptac_plot = cptac[cptac["stage_group"].isin(["Early", "Advanced"])].copy()
    cptac_plot["cohort_omics"] = (
        cptac_plot["cohort"].str.replace("CPTAC-", "", regex=False) + "\n" + cptac_plot["omics"]
    )
    sns.violinplot(data=cptac_plot, x="cohort_omics", y="immune_exclusion_index", hue="stage_group", cut=0, inner=None, palette=response_palette("Early", "Advanced"), ax=axes["E"])
    sns.stripplot(data=cptac_plot, x="cohort_omics", y="immune_exclusion_index", hue="stage_group", dodge=True, palette=neutral_palette(cptac_plot, "stage_group", PALETTE["neutral"]), size=1.4, alpha=0.35, legend=False, ax=axes["E"])
    axes["E"].set_title("CPTAC exclusion index by stage group")
    axes["E"].set_xlabel("")
    axes["E"].set_ylabel("Exclusion index")
    axes["E"].tick_params(axis="x", rotation=20)
    axes["E"].legend(frameon=False, title="")
    panel_label(axes["E"], "E")

    tcga_corr_plot = tcga_corr[tcga_corr["x"].isin(EXCLUSION_FEATURES) & tcga_corr["y"].isin(EXCLUSION_FEATURES + ["scTIME_AI_score"])].copy()
    tcga_corr_plot["program"] = tcga_corr_plot["x"].map(lambda x: FEATURE_LABELS.get(x, x))
    tcga_corr_plot["target"] = tcga_corr_plot["y"].map(lambda x: FEATURE_LABELS.get(x, x))
    tcga_corr_plot["project"] = tcga_corr_plot["analysis"].str.replace(r"^TCGA exclusion ", "", regex=True)
    grouped_dot_matrix(
        axes["F"],
        tcga_corr_plot,
        "program",
        "target",
        "project",
        "rho",
        "p_value",
        "Project-stratified TCGA exclusion co-programs",
        "Spearman rho",
        "Project",
        legend_outside_top=True,
    )
    panel_label(axes["F"], "F")

    savefig(fig, "figure5_myeloid_tgf_hypoxia_exclusion_axis")


def plot_main_cd274_tcr_immune_ecology() -> None:
    comp = read_table("gse243013_sc_state_composition")
    tcr = read_table("gse243013_tcr_clonality")
    labeled = prepare_heldout_figure6_labeled(TABLES)
    group_resp = build_combined_response_groups(labeled)
    group_resp.to_csv(DESIGN / "figure6_cd274_sctime_response_groups.tsv", sep="\t", index=False)
    stratified_resp = build_stratification_table(labeled)
    stratified_resp.to_csv(DESIGN / "figure6_sctime_cd274_stratified_response.tsv", sep="\t", index=False)
    contrast = build_feature_contrast(labeled)
    contrast.to_csv(DESIGN / "figure6_cd274_high_score_low_feature_contrast.tsv", sep="\t", index=False)

    comp_wide = comp.pivot_table(index="sampleID", columns="state_category", values="fraction", fill_value=0).reset_index()
    tcr_merged = tcr.merge(comp_wide, on="sampleID", how="left")
    tcr_metrics = ["top_clonotype_fraction", "expanded_fraction", "normalized_shannon"]
    state_cols = [c for c in comp_wide.columns if c != "sampleID"]
    all_tcr_corr = spearman_table(tcr_merged, tcr_metrics, state_cols, "GSE243013")
    state_mean = comp_wide[state_cols].mean().rename("mean_fraction")
    state_rank = (
        all_tcr_corr.assign(abs_rho=all_tcr_corr["rho"].abs())
        .groupby("y", observed=True)["abs_rho"]
        .max()
        .to_frame()
        .join(state_mean)
        .query("mean_fraction >= 0.005")
        .sort_values(["abs_rho", "mean_fraction"], ascending=False)
    )
    immune_cols = state_rank.head(8).index.tolist()
    tcr_corr = all_tcr_corr[all_tcr_corr["y"].isin(immune_cols)].copy()
    tcr_corr.to_csv(DESIGN / "figure6_tcr_cell_state_correlations.tsv", sep="\t", index=False)

    fig, axes = plt.subplot_mosaic(
        [["A", "B", "C"], ["D", "D", "E"]],
        figsize=(12, 7.7),
        constrained_layout=True,
    )

    dataset_marker_map = {dataset: marker for dataset, marker in zip(sorted(labeled["dataset"].dropna().unique()), ["s", "^", "o", "D"])}
    sns.scatterplot(
        data=labeled,
        x="CD274_expr",
        y="scTIME_oof_score",
        hue="response",
        style="dataset",
        markers=dataset_marker_map,
        s=54,
        palette=response_palette("no benefit", "benefit/MPR"),
        ax=axes["A"],
    )
    rho, p = stats.spearmanr(labeled["CD274_expr"], labeled["scTIME_oof_score"])
    axes["A"].text(
        0.04,
        0.96,
        f"rho={rho:.2f}\nP={p:.2g}\nheld-out score",
        transform=axes["A"].transAxes,
        va="top",
        fontsize=7,
    )
    axes["A"].set_title("CD274 and held-out scTIME scores")
    axes["A"].set_xlabel("CD274 expression/module")
    axes["A"].set_ylabel("scTIME OOF probability")
    axes["A"].legend(frameon=False, fontsize=6)
    panel_label(axes["A"], "A")

    order = [
        "scTIME OOF bottom quartile",
        "scTIME OOF intermediate",
        "scTIME OOF top quartile",
        "CD274 bottom quartile",
        "CD274 intermediate",
        "CD274 top quartile",
        "scTIME OOF low (median)",
        "scTIME OOF high (median)",
        "CD274 low (median)",
        "CD274 high (median)",
    ]
    resp_plot = stratified_resp[stratified_resp["group"].isin(order)].set_index("group").reindex(order).dropna(subset=["n"]).reset_index()
    y = np.arange(len(resp_plot))
    xerr = np.vstack([resp_plot["benefit_rate"] - resp_plot["ci_low"], resp_plot["ci_high"] - resp_plot["benefit_rate"]])
    point_colors = [
        PALETTE["negative"],
        PALETTE["neutral"],
        PALETTE["positive"],
        PALETTE["negative"],
        PALETTE["neutral"],
        PALETTE["positive"],
        PALETTE["negative"],
        PALETTE["positive"],
        PALETTE["negative"],
        PALETTE["positive"],
    ][: len(resp_plot)]
    axes["B"].errorbar(resp_plot["benefit_rate"], y, xerr=xerr, fmt="none", ecolor=PALETTE["neutral"], capsize=3)
    axes["B"].scatter(resp_plot["benefit_rate"], y, c=point_colors, s=52, zorder=3)
    for yi, row in zip(y, resp_plot.itertuples(index=False)):
        axes["B"].text(min(row.ci_high + 0.03, 0.98), yi, f"n={int(row.n)}", va="center", fontsize=7)
    axes["B"].set_yticks(y)
    axes["B"].set_yticklabels(resp_plot["group"])
    axes["B"].set_xlim(0, 1.05)
    axes["B"].set_title("Matched within-cohort scTIME OOF and CD274 splits")
    axes["B"].set_xlabel("Benefit/MPR fraction with 95% Wilson CI")
    axes["B"].set_ylabel("")
    panel_label(axes["B"], "B")

    contrast_plot = contrast.copy()
    contrast_plot["label"] = contrast_plot["feature"].map(lambda x: FEATURE_LABELS.get(x, x))
    lollipop(
        axes["C"],
        contrast_plot["label"],
        contrast_plot["difference"],
        "CD274-high / low OOF-scTIME feature contrast",
        "High OOF-scTIME minus CD274-high / low OOF-scTIME",
    )
    panel_label(axes["C"], "C")

    corr_plot = tcr_corr.copy()
    corr_plot["metric"] = corr_plot["x"].map(
        {
            "top_clonotype_fraction": "Top clonotype",
            "expanded_fraction": "Expanded fraction",
            "normalized_shannon": "TCR diversity",
        }
    )
    dot_matrix(axes["D"], corr_plot, "metric", "y", "rho", "p_value", "TCR clonality and immune-cell composition", "Spearman rho")
    panel_label(axes["D"], "D")

    tcr_plot = tcr.dropna(subset=["benefit"]).melt(
        id_vars=["sampleID", "benefit"],
        value_vars=tcr_metrics,
        var_name="metric",
        value_name="value",
    )
    tcr_plot["metric"] = tcr_plot["metric"].map(
        {
            "top_clonotype_fraction": "Top clonotype",
            "expanded_fraction": "Expanded fraction",
            "normalized_shannon": "TCR diversity",
        }
    )
    tcr_plot["response"] = response_label(tcr_plot["benefit"], "MPR/pCR", "non-MPR")
    sns.boxplot(data=tcr_plot, x="metric", y="value", hue="response", fliersize=0, palette=response_palette("non-MPR", "MPR/pCR"), ax=axes["E"])
    sns.stripplot(data=tcr_plot, x="metric", y="value", hue="response", dodge=True, palette=neutral_palette(tcr_plot, "response", PALETTE["neutral"]), size=1.8, alpha=0.35, legend=False, ax=axes["E"])
    axes["E"].set_title("TCR clonality metrics by pathological response")
    axes["E"].set_xlabel("")
    axes["E"].set_ylabel("Metric value")
    axes["E"].tick_params(axis="x", rotation=20)
    axes["E"].legend(frameon=False, title="")
    panel_label(axes["E"], "E")

    savefig(fig, "figure6_cd274_tcr_immune_ecology")


def plot_main_external_validation_cross_omics() -> None:
    tcga_status = read_table("tcga_secondary_validation")
    tcga = read_table("tcga_sctime_ai_scores")
    cptac = read_table("cptac_sctime_ai_projection")
    cptac_stage = read_table("cptac_stage_associations")
    gse135 = read_table("gse135222_survival_validation")
    gse_scores = read_table("scTIME_AI_scores_all_bulk_cohorts")

    fig, axes = plt.subplot_mosaic(
        [["A", "B", "C"], ["D", "E", "E"]],
        figsize=(12, 7.6),
        constrained_layout=True,
    )

    g135 = gse_scores[gse_scores["dataset"].eq("GSE135222")].dropna(subset=["pfs_time", "pfs_event", "scTIME_AI_score"]).copy()
    if not g135.empty:
        g135["score_group"] = np.where(g135["scTIME_AI_score"] >= g135["scTIME_AI_score"].median(), "High", "Low")
        for label, color in [("High", PALETTE["positive"]), ("Low", PALETTE["negative"])]:
            sub = g135[g135["score_group"].eq(label)]
            t, s = km_curve(sub["pfs_time"].astype(float).to_numpy(), sub["pfs_event"].astype(int).to_numpy())
            axes["A"].step(t, s, where="post", label=f"{label} (n={len(sub)})", color=color)
        p = gse135.loc[gse135["analysis"].str.contains("log-rank", na=False), "p_value"].iloc[0]
        axes["A"].text(0.05, 0.08, f"log-rank P={p:.3g}", transform=axes["A"].transAxes)
    axes["A"].set_title("GSE135222 exploratory PFS association")
    axes["A"].set_xlabel("PFS time")
    axes["A"].set_ylabel("PFS probability")
    axes["A"].legend(frameon=False)
    panel_label(axes["A"], "A")

    forest_rows = []
    row = gse135[gse135["analysis"].str.contains("Cox", na=False)].head(1)
    if not row.empty:
        forest_rows.append({"analysis": "GSE135222 exploratory PFS association", "hr": float(row["hazard_ratio"].iloc[0]), "p": float(row["p_value"].iloc[0])})
    for _, row in tcga_status.iterrows():
        forest_rows.append({"analysis": f"{row['project']} OS", "hr": row["cox_hr"], "p": row["cox_p"]})
    forest = pd.DataFrame(forest_rows).dropna(subset=["hr"]).sort_values("hr")
    y = np.arange(len(forest))
    axes["B"].scatter(forest["hr"], y, s=70, color=PALETTE["positive"])
    axes["B"].axvline(1, color=PALETTE["dark"], linestyle="--", linewidth=0.8)
    for yi, row in zip(y, forest.itertuples(index=False)):
        axes["B"].text(row.hr * 1.05, yi, f"P={row.p:.2g}", va="center", fontsize=7)
    axes["B"].set_yticks(y)
    axes["B"].set_yticklabels(forest["analysis"])
    axes["B"].set_xscale("log")
    axes["B"].set_xticks([0.25, 0.5, 1.0])
    axes["B"].set_xticklabels(["0.25", "0.50", "1.00"])
    axes["B"].tick_params(axis="x", which="minor", labelbottom=False)
    axes["B"].set_title("Survival association summary")
    axes["B"].set_xlabel("Hazard ratio per scTIME score")
    axes["B"].set_ylabel("")
    panel_label(axes["B"], "B")

    tcga_plot = tcga.dropna(subset=["project", "scTIME_AI_score"]).copy()
    sns.violinplot(data=tcga_plot, x="project", y="scTIME_AI_score", hue="project", inner=None, cut=0, palette=category_palette(tcga_plot["project"]), legend=False, ax=axes["C"])
    sns.boxplot(data=tcga_plot, x="project", y="scTIME_AI_score", width=0.23, color="white", fliersize=0, ax=axes["C"])
    axes["C"].set_title("TCGA-LUAD/LUSC scTIME distributions")
    axes["C"].set_xlabel("")
    axes["C"].set_ylabel("scTIME score")
    panel_label(axes["C"], "C")

    cptac_plot = cptac[cptac["stage_group"].isin(["Early", "Advanced"])].copy()
    cptac_plot["cohort_omics"] = (
        cptac_plot["cohort"].str.replace("CPTAC-", "", regex=False)
        + "\n"
        + cptac_plot["omics"].replace({"proteome": "Prot.", "phosphoproteome": "Phospho."})
    )
    sns.boxplot(data=cptac_plot, x="cohort_omics", y="scTIME_AI_score", hue="stage_group", fliersize=0, ax=axes["D"], palette=response_palette("Early", "Advanced"))
    sns.stripplot(data=cptac_plot, x="cohort_omics", y="scTIME_AI_score", hue="stage_group", dodge=True, palette=neutral_palette(cptac_plot, "stage_group", PALETTE["neutral"]), size=1.4, alpha=0.35, legend=False, ax=axes["D"])
    axes["D"].set_title("CPTAC proteome/phosphoproteome projection")
    axes["D"].set_xlabel("")
    axes["D"].set_ylabel("scTIME score")
    axes["D"].tick_params(axis="x", rotation=0, labelsize=6, pad=2)
    axes["D"].legend(frameon=False, title="")
    panel_label(axes["D"], "D")

    key = ["scTIME_AI_score", "IFN_response", "Antigen_presentation", "Cytotoxic_CD8", "CXCL10_CXCR3_axis", "SPP1_macrophage", "TGF_beta_EMT", "Hypoxia", "CD274_expr"]
    stage_plot = cptac_stage[cptac_stage["feature"].isin(key)].copy()
    stage_plot["feature_label"] = stage_plot["feature"].map(lambda x: FEATURE_LABELS.get(x, x))
    stage_plot["label"] = stage_plot["cohort"] + " " + stage_plot["omics"]
    feature_order = [FEATURE_LABELS.get(x, x) for x in key if FEATURE_LABELS.get(x, x) in stage_plot["feature_label"].unique()]
    palette = category_palette(stage_plot["label"])
    for label, sub in stage_plot.groupby("label", observed=True):
        y_pos = [feature_order.index(v) for v in sub["feature_label"] if v in feature_order]
        sub = sub[sub["feature_label"].isin(feature_order)]
        axes["E"].scatter(sub["advanced_minus_early"], y_pos, label=label, s=38, alpha=0.78, color=palette[label])
    axes["E"].axvline(0, color=PALETTE["dark"], linewidth=0.8)
    axes["E"].set_yticks(range(len(feature_order)))
    axes["E"].set_yticklabels(feature_order)
    axes["E"].set_title("CPTAC stage shifts in key immune programs")
    axes["E"].set_xlabel("Advanced minus early")
    axes["E"].set_ylabel("")
    axes["E"].legend(frameon=False, fontsize=6, bbox_to_anchor=(1.02, 1), loc="upper left")
    panel_label(axes["E"], "E")

    savefig(fig, "figure7_external_validation_cross_omics")


def plot_supplementary_audit_model_details() -> None:
    dataset_summary = read_table("dataset_summary")
    tcga = read_table("tcga_secondary_validation")
    cptac = read_table("cptac_secondary_validation_status")
    gse243 = read_table("gse243013_sc_patient_summary")
    model = read_table("model_performance")
    cal = read_table("elasticnet_calibration")
    dca = read_table("elasticnet_decision_curve")

    cohort_rows = []
    for _, row in dataset_summary.iterrows():
        sample_count = row["n_samples"] if "n_samples" in row.index else row["samples"]
        cohort_rows.append({"cohort": row["dataset"], "modality": "bulk RNA-seq", "samples": sample_count, "role": "training/validation" if row["dataset"] in {"GSE126044", "GSE207422_bulk"} else "projection/survival"})
    cohort_rows.append({"cohort": "GSE243013", "modality": "scRNA-seq/TCR", "samples": len(gse243), "role": "single-cell discovery"})
    for _, row in tcga.iterrows():
        cohort_rows.append({"cohort": row["project"], "modality": "bulk RNA-seq", "samples": row["n_samples"], "role": "secondary validation"})
    for (cohort, omics), sub in cptac.groupby(["cohort", "omics"], observed=True):
        cohort_rows.append({"cohort": f"{cohort} {omics}", "modality": omics, "samples": sub["n_samples"].max(), "role": "proteomic validation"})
    cohort_df = pd.DataFrame(cohort_rows)
    cohort_df.to_csv(DESIGN / "supplementary_audit_cohort_table.tsv", sep="\t", index=False)

    fig, axes = plt.subplot_mosaic(
        [["A", "B"], ["C", "D"]],
        figsize=(10.5, 7.4),
        constrained_layout=True,
    )

    modality_palette = {
        "bulk RNA-seq": "#4c78a8",
        "scRNA-seq/TCR": "#72b7b2",
        "proteome": "#f58518",
        "phosphoproteome": "#b279a2",
    }
    plot_df = cohort_df.sort_values(["role", "modality", "samples"])
    sns.barplot(data=plot_df, y="cohort", x="samples", hue="modality", palette=modality_palette, ax=axes["A"], dodge=False)
    axes["A"].set_title("Cohort sample audit")
    axes["A"].set_xlabel("Analyzed samples or patients")
    axes["A"].set_ylabel("")
    axes["A"].legend(frameon=False, loc="lower right")
    panel_label(axes["A"], "A")

    perf = model.copy()
    perf["label"] = perf["model"] + " | " + perf["evaluation"].str.replace(" validation", "", regex=False)
    top_perf = perf.sort_values("roc_auc", ascending=False).head(18)
    sns.scatterplot(data=top_perf, x="roc_auc", y="pr_auc", hue="model", size="brier", sizes=(40, 150), ax=axes["B"])
    axes["B"].set_title("Model performance across evaluation designs")
    axes["B"].set_xlabel("ROC-AUC")
    axes["B"].set_ylabel("PR-AUC")
    axes["B"].legend(frameon=False, fontsize=6, bbox_to_anchor=(1.02, 1), loc="upper left")
    panel_label(axes["B"], "B")

    sns.lineplot(data=cal, x="mean_prediction", y="observed_rate", marker="o", ax=axes["C"], color=PALETTE["positive"])
    axes["C"].plot([0, 1], [0, 1], linestyle="--", color=PALETTE["neutral"], linewidth=0.8)
    axes["C"].set_title("Calibration")
    axes["C"].set_xlabel("Mean predicted probability")
    axes["C"].set_ylabel("Observed response rate")
    axes["C"].set_xlim(0, 1)
    axes["C"].set_ylim(0, 1)
    panel_label(axes["C"], "C")

    sns.lineplot(data=dca, x="threshold", y="net_benefit_model", ax=axes["D"], label="scTIME", color=PALETTE["positive"])
    sns.lineplot(data=dca, x="threshold", y="net_benefit_treat_all", ax=axes["D"], label="Treat all", color=PALETTE["neutral"])
    sns.lineplot(data=dca, x="threshold", y="net_benefit_treat_none", ax=axes["D"], label="Treat none", color=PALETTE["dark"])
    axes["D"].set_title("Decision-curve analysis")
    axes["D"].set_xlabel("Threshold probability")
    axes["D"].set_ylabel("Net benefit")
    axes["D"].legend(frameon=False)
    panel_label(axes["D"], "D")

    savefig(fig, "supplementary_audit_model_details")


def plot_supplementary_translation_lr_details() -> None:
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
    cluster_heat = cluster_heat.join(cluster_summary.set_index("leiden")[["cell_state", "n_cells"]]).sort_values(["cell_state", "n_cells"], ascending=[True, False])

    bulk_cov["coverage"] = bulk_cov["available_genes"] / bulk_cov["total_genes"]
    cptac_cov["coverage"] = cptac_cov["available_genes"] / cptac_cov["total_genes"]
    cptac_cov["dataset"] = cptac_cov["cohort"] + " " + cptac_cov["omics"]
    coverage_long = pd.concat(
        [
            bulk_cov[["dataset", "module", "coverage"]].assign(source="bulk RNA-seq"),
            cptac_cov[["dataset", "module", "coverage"]].assign(source="CPTAC"),
        ],
        ignore_index=True,
    )
    coverage_long.to_csv(DESIGN / "supplementary_signature_coverage_combined.tsv", sep="\t", index=False)

    lr_mat = lr.pivot_table(index="axis", columns="sample", values="interaction_score", fill_value=0)

    frac_merged = fractions.merge(bulk[["dataset", "sample", "scTIME_AI_score", "benefit", "Cytotoxic_CD8", "SPP1_macrophage"]], on=["dataset", "sample"], how="left")
    frac_corr = spearman_table(frac_merged, ["CD8 T", "Myeloid/macrophage", "Tumor/epithelial", "Fibroblast"], ["Cytotoxic_CD8", "SPP1_macrophage", "scTIME_AI_score"], "NNLS-vs-bulk")
    frac_corr.to_csv(DESIGN / "supplementary_deconvolution_signature_correlations.tsv", sep="\t", index=False)

    overlap_patients = sorted(set(sc_pseudo["patient"].dropna()) & set(bulk_scores.loc[bulk_scores["dataset"].eq("GSE207422_bulk"), "patient"].dropna()))
    concord_rows = []
    if overlap_patients:
        sc_overlap = sc_pseudo[sc_pseudo["patient"].isin(overlap_patients)].set_index("patient")
        bulk_overlap = bulk_scores[(bulk_scores["dataset"].eq("GSE207422_bulk")) & (bulk_scores["patient"].isin(overlap_patients))].set_index("patient")
        for feature in FEATURES:
            sub = pd.concat([sc_overlap[feature].rename("single_cell"), bulk_overlap[feature].rename("bulk")], axis=1).dropna()
            if len(sub) >= 3:
                rho, p = stats.spearmanr(sub["single_cell"], sub["bulk"])
            else:
                rho, p = np.nan, np.nan
            concord_rows.append({"feature": feature, "rho": rho, "p_value": p, "n_overlap_patients": len(sub)})
    concord = pd.DataFrame(concord_rows)
    concord.to_csv(DESIGN / "supplementary_sc_bulk_overlapping_patient_concordance.tsv", sep="\t", index=False)

    fig, axes = plt.subplot_mosaic(
        [["A", "B", "B"], ["C", "D", "E"], ["C", "F", "F"]],
        figsize=(12.5, 10),
        constrained_layout=True,
    )

    heat = cluster_heat.drop(columns=["cell_state", "n_cells"])
    sns.heatmap(heat, cmap=DIVERGING_CMAP, center=0, ax=axes["A"], cbar_kws={"label": "marker z-score"})
    axes["A"].set_title("Cluster marker-program audit")
    axes["A"].set_xlabel("Marker program")
    axes["A"].set_ylabel("Leiden cluster")
    panel_label_tight(axes["A"], "A")

    cov_plot = coverage_long.copy()
    cov_plot["module_label"] = cov_plot["module"].map(lambda x: FEATURE_LABELS.get(x, x))
    cov_plot["dataset_label"] = cov_plot["dataset"].map(clean_label)
    cov_plot = cov_plot[cov_plot["module"].isin(FEATURES)]
    dot_matrix(axes["B"], cov_plot, "dataset_label", "module_label", "coverage", None, "Bulk/CPTAC signature coverage", "Gene coverage", vmin=0, vmax=1)
    panel_label_tight(axes["B"], "B")

    sns.heatmap(lr_mat, cmap=SEQUENTIAL_CMAP, ax=axes["C"], cbar_kws={"label": "LR score"}, xticklabels=False)
    axes["C"].set_title("LR axis activity matrix")
    axes["C"].set_xlabel("GSE207422 scRNA samples")
    axes["C"].set_ylabel("")
    panel_label_tight(axes["C"], "C")

    frac_corr_plot = frac_corr.copy()
    frac_corr_plot["module"] = frac_corr_plot["y"].map(lambda x: FEATURE_LABELS.get(x, x))
    dot_matrix(
        axes["D"],
        frac_corr_plot,
        "x",
        "module",
        "rho",
        "p_value",
        "Deconvolution-vs-module correlations",
        "Spearman rho",
        cbar_fraction=0.06,
        cbar_pad=0.006,
        cbar_shrink=0.72,
    )
    panel_label_tight(axes["D"], "D")

    ref_corr = ref.corr(method="spearman")
    cax_e = inset_axes(
        axes["E"],
        width="4%",
        height="52%",
        loc="center right",
        bbox_to_anchor=(0.04, 0, 1, 1),
        bbox_transform=axes["E"].transAxes,
        borderpad=0,
    )
    sns.heatmap(
        ref_corr,
        cmap=DIVERGING_CMAP,
        vmin=-1,
        vmax=1,
        center=0,
        square=True,
        ax=axes["E"],
        cbar_ax=cax_e,
        cbar_kws={"label": "Spearman rho"},
    )
    axes["E"].set_title("Reference signature separability")
    axes["E"].set_xlabel("")
    axes["E"].set_ylabel("")
    panel_label_tight(axes["E"], "E")

    if not concord.empty:
        concord_plot = concord.copy()
        concord_plot["label"] = concord_plot["feature"].map(lambda x: FEATURE_LABELS.get(x, x))
        lollipop(axes["F"], concord_plot["label"], concord_plot["rho"], "Matched scRNA-bulk signature concordance", "Spearman rho")
    else:
        axes["F"].text(0.5, 0.5, "No overlapping patients", ha="center", va="center")
        axes["F"].set_axis_off()
    panel_label_tight(axes["F"], "F")

    savefig(fig, "supplementary_translation_lr_details")


def write_figure_plan() -> None:
    plan = MAIN_FIGURE_PLAN
    with (SUPPLEMENTARY / "publication_figure_plan.json").open("w", encoding="utf-8") as handle:
        json.dump(plan, handle, ensure_ascii=False, indent=2)
    lines = ["# Publication Figure Design\n", "Main-text figures are organized as six generated data figures that follow the immunotherapy-response storyline.\n"]
    lines.extend(
        [
            "\n## Recommended Main-Text Strategy\n",
            "- Generated main-text figures are numbered Figure 2 to Figure 7: single-cell discovery, scTIME reliability, CXCL10/IFN axis, exclusion biology, CD274/TCR interpretation, and exploratory external projection.\n",
            "- Main-text non-single-cell data encodings use the three-color system: responder/benefit blue, non-responder/exclusion red, and neutral gray, including blue-red-gray gradients.\n",
            "- Broad model benchmarking, coverage, raw-data audit, LR matrices, and marker-validation panels are retained as supplementary support.\n",
            "- Low-information bar-only panels and duplicated heatmaps are not retained in the main-text set.\n",
            "- No mechanism cartoon, decorative circle plot, card plot, or AI-looking schematic is generated.\n",
        ]
    )
    for item in plan:
        lines.append(f"\n## {item['figure']}\n")
        lines.append(f"- File: `{item['file']}`\n")
        lines.append(f"- Purpose: {item['purpose']}\n")
        if item["panels"]:
            lines.append("- Panels:\n")
            for panel in item["panels"]:
                lines.append(f"  - {panel}\n")
    (SUPPLEMENTARY / "publication_figure_plan.md").write_text("".join(lines), encoding="utf-8")

    strategy = """# Figure Strategy for Frontiers in Artificial Intelligence

## Main-Text Priority

1. Figure 2: Single-cell response landscape.
2. Figure 3: scTIME reliability, validation and interpretability.
3. Figure 4: CXCL10-CXCR3 / IFN-antigen evidence chain.
4. Figure 5: Myeloid/TGF-beta/hypoxia exclusion biology.
5. Figure 6: CD274 complementarity and TCR immune ecology.
6. Figure 7: Exploratory PFS association and cross-omics projection.

## Supplementary Material

- Supplementary Figure 1: Cohort audit, broad model performance, calibration, and decision curve.
- Supplementary Figure 2: Marker-program audit, signature coverage, LR matrix, deconvolution correlations, reference separability, and matched scRNA-bulk concordance.

## Constraints Satisfied

- Old publication figures are deleted before regeneration, preventing stale duplicate images from persisting in `results/figures`.
- Main text no longer contains manifest/module audit tiles, marker heatmap audits, ecotype fraction bars, low-yield single-variable TCR panels, or duplicated coverage heatmaps.
- Non-single-cell main-text plots use the blue/red/gray three-color system or gradients derived from those colors.
- The model figure emphasizes final ElasticNet scTIME reliability rather than visually promoting weaker exploratory algorithms.
- Chart types are diversified with UMAP/PCA, KM, ROC, PR, calibration, decision curve, bootstrap reliability intervals, violin/box/strip plots, lollipop/forest-style estimates, regression, hexbin density, contribution heatmaps, and dot matrices.
- No mechanistic cartoon, circle plot, card plot, or decorative AI schematic is generated.
- Main-text individual panel files are regenerated independently from panel source data by `scripts/export_main_figure_individual_panels.py`; they are not cropped from composite figures and do not contain A/B/C panel-letter marks.
"""
    (SUPPLEMENTARY / "frontiers_figure_strategy.md").write_text(strategy, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--figure6",
        action="store_true",
        help="Regenerate only the held-out Figure 6 composite and its source summaries.",
    )
    parser.add_argument(
        "--figure5",
        action="store_true",
        help="Regenerate only the cohort-stratified Figure 5 composite and summaries.",
    )
    args = parser.parse_args()
    setup()
    if args.figure6:
        plot_main_cd274_tcr_immune_ecology()
        return
    if args.figure5:
        plot_main_myeloid_tgf_hypoxia_exclusion_axis()
        return
    remove_existing_figure_files()
    plot_main_single_cell_response_landscape()
    plot_main_ai_reliability_sctime_modeling()
    plot_main_cxcl10_ifn_antigen_axis()
    plot_main_myeloid_tgf_hypoxia_exclusion_axis()
    plot_main_cd274_tcr_immune_ecology()
    plot_main_external_validation_cross_omics()
    plot_supplementary_audit_model_details()
    plot_supplementary_translation_lr_details()
    write_figure_plan()


if __name__ == "__main__":
    main()
