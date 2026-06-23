#!/usr/bin/env python
"""Regenerate each main-text figure panel as an independent file.

This script redraws panels from source tables instead of cropping the composite
figures. Panel-letter labels are intentionally omitted from the saved images.
"""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import Normalize
from scipy import stats

from design_publication_figures import (
    DESIGN,
    DIVERGING_CMAP,
    FEATURE_LABELS,
    FIGURES,
    FIGURE3A_RESEARCH_VALUE_THRESHOLD,
    NEGATIVE_CMAP,
    PALETTE,
    SEQUENTIAL_CMAP,
    category_palette,
    clean_label,
    dot_matrix,
    km_curve,
    lollipop,
    response_label,
    response_palette,
    setup,
    signed_log10_p,
)


PANEL_DATA = DESIGN / "panel_data"
PANEL_FIGURES = FIGURES / "main_figure_panels"


def read_panel(name: str) -> pd.DataFrame:
    return pd.read_csv(PANEL_DATA / name, sep="\t")


def slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def save_panel(fig: plt.Figure, figure: int, panel: str, title: str) -> None:
    PANEL_FIGURES.mkdir(parents=True, exist_ok=True)
    stem = f"figure_{figure}_panel_{panel.lower()}_{slug(title)}"
    for ext in ["png", "pdf"]:
        fig.savefig(PANEL_FIGURES / f"{stem}.{ext}", bbox_inches="tight")
    plt.close(fig)


def new_fig(width: float = 4.2, height: float = 3.4) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=(width, height), constrained_layout=True)
    return fig, ax


def strip_duplicate_legend(ax: plt.Axes) -> None:
    handles, labels = ax.get_legend_handles_labels()
    if not handles:
        return
    seen: dict[str, object] = {}
    for handle, label in zip(handles, labels):
        if label and not label.startswith("_") and label not in seen:
            seen[label] = handle
    if seen:
        ax.legend(seen.values(), seen.keys(), frameon=False, fontsize=7)


def export_figure2_panels() -> None:
    emb = read_panel("figure_2_panela_gse207422_single_cell_immune_landscape.tsv")
    diff = read_panel("figure_2_panelb_response_associated_state_shifts.tsv")
    comp = read_panel("figure_2_panelc_patient_level_fractions_of_response_informative_states.tsv")
    heat_long = read_panel("figure_2_paneld_patient_by_state_composition_structure.tsv")
    network = read_panel("figure_2_panele_response_linked_single_cell_state_network.tsv")

    fig, ax = new_fig(5.2, 4.2)
    state_palette = dict(zip(sorted(emb["cell_state"].dropna().unique()), sns.color_palette("tab20", n_colors=emb["cell_state"].nunique())))
    sns.scatterplot(data=emb, x="UMAP1", y="UMAP2", hue="cell_state", s=2.0, linewidth=0, alpha=0.72, palette=state_palette, ax=ax)
    ax.set_title("GSE207422 single-cell immune landscape")
    ax.set_xlabel("UMAP1")
    ax.set_ylabel("UMAP2")
    ax.legend(frameon=False, markerscale=4, bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=6)
    save_panel(fig, 2, "A", "GSE207422 single-cell immune landscape")

    fig, ax = new_fig(4.2, 4.5)
    lollipop(ax, diff["state_category"], diff["signed_log10_p"], "Response-associated state shifts", "signed -log10(P), MPR minus non-MPR")
    save_panel(fig, 2, "B", "Response-associated state shifts")

    fig, ax = new_fig(4.6, 3.8)
    order = list(dict.fromkeys(comp["state_category"]))
    sns.boxplot(data=comp, x="fraction", y="state_category", hue="response", order=order, fliersize=0, palette=response_palette("non-MPR", "MPR/pCR"), ax=ax)
    sns.stripplot(data=comp, x="fraction", y="state_category", hue="response", order=order, dodge=True, palette={k: PALETTE["neutral"] for k in comp["response"].dropna().unique()}, size=1.6, alpha=0.35, legend=False, ax=ax)
    ax.set_title("Patient-level fractions of response-informative states")
    ax.set_xlabel("Fraction of cells")
    ax.set_ylabel("")
    strip_duplicate_legend(ax)
    save_panel(fig, 2, "C", "Patient-level fractions of response-informative states")

    fig, ax = new_fig(6.2, 4.0)
    heat = heat_long.pivot_table(index="sampleID", columns="state_category", values="fraction", fill_value=0)
    if "state_order" in heat_long.columns:
        state_order = heat_long.dropna(subset=["state_order"]).sort_values("state_order")["state_category"].drop_duplicates().tolist()
        heat = heat[[c for c in state_order if c in heat.columns]]
    benefit = heat_long.groupby("sampleID", observed=True)["benefit"].first()
    heat = heat.loc[[idx for idx in benefit.sort_values(ascending=False).index if idx in heat.index]]
    sns.heatmap(heat, cmap=SEQUENTIAL_CMAP, ax=ax, cbar_kws={"label": "Fraction"}, linewidths=0.2, linecolor="white")
    ax.set_title("Patient-by-state composition structure")
    ax.set_xlabel("Cell state")
    ax.set_ylabel("Patient")
    ax.tick_params(axis="x", rotation=25)
    ax.tick_params(axis="y", labelsize=5)
    save_panel(fig, 2, "D", "Patient-by-state composition structure")

    fig, ax = new_fig(4.6, 4.2)
    if not network.empty:
        sizes = 100 + 90 * np.sqrt(pd.to_numeric(network["signal"], errors="coerce").fillna(0))
        colors = [PALETTE["positive"] if v >= 0 else PALETTE["negative"] for v in network["effect"]]
        for row in network.itertuples(index=False):
            ax.plot([0, row.network_x], [0, row.network_y], color=PALETTE["neutral"], linewidth=0.8, alpha=0.55)
            ha = "left" if row.network_x >= 0 else "right"
            ax.text(row.network_x * 1.08, row.network_y * 1.08, row.state_category, ha=ha, va="center", fontsize=6)
        ax.scatter(network["network_x"], network["network_y"], s=sizes, c=colors, edgecolor="white", linewidth=0.7, zorder=3)
        ax.scatter([0], [0], s=320, c=PALETTE["neutral"], edgecolor="white", linewidth=0.8, zorder=4)
        ax.text(0, 0, "MPR\nshift", ha="center", va="center", color="white", fontsize=7)
    ax.set_title("Response-linked single-cell state network")
    ax.set_aspect("equal")
    ax.set_axis_off()
    save_panel(fig, 2, "E", "Response-linked single-cell state network")


def export_figure3_panels() -> None:
    pca = read_panel("figure_3_panela_bulk_immune_state_pca.tsv")
    perf = read_panel("figure_3_panelb_sctime_ai_reliability_matrix_across_validation_designs.tsv")
    roc = read_panel("figure_3_panelc_elasticnet_roc_curves.tsv")
    pr = read_panel("figure_3_paneld_elasticnet_precision_recall_curves.tsv")
    cal = read_panel("figure_3_panele_calibration_with_wilson_intervals.tsv")
    dca = read_panel("figure_3_panelf_decision_curve_analysis.tsv")
    boot = read_panel("figure_3_panelg_bootstrap_reliability_intervals.tsv")
    coef = read_panel("figure_3_panelh_interpretable_sctime_ai_coefficients.tsv")
    contribution = read_panel("figure_3_paneli_patient_level_feature_contribution_profile.tsv")

    fig, ax = new_fig(4.4, 3.8)
    score_norm = Normalize(vmin=float(pca["scTIME_AI_score"].min()), vmax=float(pca["scTIME_AI_score"].max()))
    marker_map = {"benefit": "o", "no benefit": "^", "unlabeled": "s"}
    for response, marker in marker_map.items():
        for high_value, alpha in [(False, 0.22), (True, 0.88)]:
            sub = pca[pca["response"].eq(response) & pca["high_research_value"].eq(high_value)]
            if sub.empty:
                continue
            ax.scatter(sub["PC1"], sub["PC2"], c=sub["scTIME_AI_score"], cmap=SEQUENTIAL_CMAP, norm=score_norm, s=sub["display_size"], alpha=alpha, marker=marker, edgecolor="white", linewidth=0.4, label=response if high_value else None)
    sc = plt.cm.ScalarMappable(cmap=SEQUENTIAL_CMAP, norm=score_norm)
    sc.set_array([])
    ax.axvline(0, color=PALETTE["neutral"], linewidth=0.6, alpha=0.35)
    ax.axhline(0, color=PALETTE["neutral"], linewidth=0.6, alpha=0.35)
    ax.text(0.02, 0.98, f"Highlighted: scTIME-AI >= {FIGURE3A_RESEARCH_VALUE_THRESHOLD:.2f}", transform=ax.transAxes, va="top", ha="left", fontsize=6, color=PALETTE["dark"])
    fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.03, label="scTIME-AI score")
    ax.set_title("Bulk immune-state PCA")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    strip_duplicate_legend(ax)
    save_panel(fig, 3, "A", "Bulk immune-state PCA")

    fig, ax = new_fig(5.0, 2.8)
    benchmark = perf.pivot_table(index="metric", columns="validation", values="value").reindex(["ROC-AUC", "PR-AUC", "Brier"])
    sns.heatmap(benchmark, cmap=DIVERGING_CMAP, vmin=0, vmax=1, annot=True, fmt=".2f", linewidths=0.4, linecolor="white", cbar_kws={"label": "Metric value"}, ax=ax)
    ax.set_title("scTIME-AI reliability across validation designs")
    ax.set_xlabel("")
    ax.set_ylabel("Metric")
    ax.tick_params(axis="x", rotation=25)
    save_panel(fig, 3, "B", "scTIME-AI reliability matrix across validation designs")

    fig, ax = new_fig(4.0, 3.4)
    for label, sub in roc.groupby("validation", observed=True):
        width = 2.0 if label == "Pooled" else 1.4
        color = PALETTE["dark"] if label == "Pooled" else category_palette(roc["validation"])[str(label)]
        auc = sub["auc"].dropna().iloc[0]
        ax.plot(sub["fpr"], sub["tpr"], label=f"{label} AUC={auc:.2f}", linewidth=width, color=color)
    ax.plot([0, 1], [0, 1], linestyle="--", color=PALETTE["neutral"], linewidth=0.8)
    ax.set_title("ElasticNet ROC stability")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.legend(frameon=False, fontsize=6)
    save_panel(fig, 3, "C", "ElasticNet ROC curves")

    fig, ax = new_fig(4.0, 3.4)
    for label, sub in pr.groupby("validation", observed=True):
        width = 2.0 if label == "Pooled" else 1.4
        color = PALETTE["dark"] if label == "Pooled" else category_palette(pr["validation"])[str(label)]
        ap = sub["average_precision"].dropna().iloc[0]
        ax.plot(sub["recall"], sub["precision"], label=f"{label} AP={ap:.2f}", linewidth=width, color=color)
    ax.set_title("ElasticNet precision-recall stability")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.legend(frameon=False, fontsize=6)
    save_panel(fig, 3, "D", "ElasticNet precision-recall curves")

    fig, ax = new_fig(3.8, 3.4)
    if not cal.empty:
        yerr = np.vstack([cal["observed_rate"] - cal["ci_low"], cal["ci_high"] - cal["observed_rate"]])
        ax.errorbar(cal["mean_prediction"], cal["observed_rate"], yerr=yerr, fmt="o", color=PALETTE["positive"], ecolor=PALETTE["neutral"], capsize=3)
        for row in cal.itertuples(index=False):
            ax.text(row.mean_prediction, min(row.ci_high + 0.05, 1), f"n={int(row.n)}", ha="center", fontsize=6)
    ax.plot([0, 1], [0, 1], linestyle="--", color=PALETTE["neutral"], linewidth=0.8)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("Calibration with Wilson intervals")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed benefit rate")
    save_panel(fig, 3, "E", "Calibration with Wilson intervals")

    fig, ax = new_fig(4.0, 3.4)
    sns.lineplot(data=dca, x="threshold", y="net_benefit_model", ax=ax, label="scTIME-AI", color=PALETTE["positive"], linewidth=1.6)
    sns.lineplot(data=dca, x="threshold", y="net_benefit_treat_all", ax=ax, label="Treat all", color=PALETTE["negative"], linewidth=1.2)
    sns.lineplot(data=dca, x="threshold", y="net_benefit_treat_none", ax=ax, label="Treat none", color=PALETTE["neutral"], linewidth=1.2)
    ax.set_title("Decision-curve analysis")
    ax.set_xlabel("Threshold probability")
    ax.set_ylabel("Net benefit")
    ax.legend(frameon=False, fontsize=7)
    save_panel(fig, 3, "F", "Decision-curve analysis")

    fig, ax = new_fig(4.5, 4.0)
    boot_plot = boot[boot["metric"].isin(["ROC-AUC", "PR-AUC", "Brier"])].copy().sort_values(["metric", "validation"])
    boot_plot["label"] = boot_plot["validation"] + "\n" + boot_plot["metric"]
    y = np.arange(len(boot_plot))
    xerr = np.vstack([boot_plot["estimate"] - boot_plot["ci_low"], boot_plot["ci_high"] - boot_plot["estimate"]])
    colors = [PALETTE["negative"] if metric == "Brier" else PALETTE["positive"] for metric in boot_plot["metric"]]
    ax.errorbar(boot_plot["estimate"], y, xerr=xerr, fmt="o", color=PALETTE["dark"], ecolor=PALETTE["neutral"], capsize=3)
    ax.scatter(boot_plot["estimate"], y, c=colors, s=42, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels(boot_plot["label"])
    ax.set_xlim(0, 1.05)
    ax.set_title("Bootstrap reliability intervals")
    ax.set_xlabel("Metric estimate with 95% interval")
    ax.set_ylabel("")
    save_panel(fig, 3, "G", "Bootstrap reliability intervals")

    fig, ax = new_fig(4.5, 3.4)
    coef_plot = coef.copy()
    coef_plot["label"] = coef_plot["feature"].map(lambda x: FEATURE_LABELS.get(x, x))
    lollipop(ax, coef_plot["label"], coef_plot["coefficient"], "Interpretable scTIME-AI coefficients", "ElasticNet coefficient")
    save_panel(fig, 3, "H", "Interpretable scTIME-AI coefficients")

    fig, ax = new_fig(9.5, 3.4)
    contrib_cols = [c for c in contribution.columns if c in FEATURE_LABELS]
    heat = contribution.set_index(["dataset", "sample"])[contrib_cols]
    heat.columns = [FEATURE_LABELS.get(c, c) for c in heat.columns]
    vmax = np.nanquantile(np.abs(heat.to_numpy()).ravel(), 0.98)
    sns.heatmap(heat.T, cmap=DIVERGING_CMAP, center=0, vmin=-vmax, vmax=vmax, ax=ax, cbar_kws={"label": "Logit contribution"}, xticklabels=False, linewidths=0.05, linecolor="white")
    ax.set_title("Patient-level feature contribution profile")
    ax.set_xlabel("Labeled bulk patients ordered by response and scTIME-AI score")
    ax.set_ylabel("")
    save_panel(fig, 3, "I", "Patient-level feature contribution profile")


def export_figure4_panels() -> None:
    f4a = read_panel("figure_4_panela_cxcl10_cxcr3_axis_by_response.tsv")
    f4b = read_panel("figure_4_panelb_ifn_antigen_index_links_to_cxcl10_cxcr3.tsv")
    f4c = read_panel("figure_4_panelc_single_cell_cxcl9_10_11_cxcr3_lr_score.tsv")
    f4d = read_panel("figure_4_paneld_ligand_activity_against_effector_module.tsv")
    f4e = read_panel("figure_4_panele_cxcl10_cxcr3_cross_context_correlations.tsv")

    fig, ax = new_fig(4.2, 3.4)
    sns.boxplot(data=f4a, x="dataset", y="CXCL10_CXCR3_axis", hue="response", fliersize=0, palette=response_palette("no benefit", "benefit/MPR"), ax=ax)
    sns.stripplot(data=f4a, x="dataset", y="CXCL10_CXCR3_axis", hue="response", dodge=True, palette={k: PALETTE["neutral"] for k in f4a["response"].dropna().unique()}, size=2.1, alpha=0.45, legend=False, ax=ax)
    ax.set_title("CXCL10-CXCR3 axis by response")
    ax.set_xlabel("")
    ax.set_ylabel("Module score")
    strip_duplicate_legend(ax)
    save_panel(fig, 4, "A", "CXCL10-CXCR3 axis by response")

    fig, ax = new_fig(4.0, 3.4)
    sns.regplot(data=f4b, x="IFN_antigen_index", y="CXCL10_CXCR3_axis", scatter_kws={"s": 42, "alpha": 0.75}, line_kws={"color": PALETTE["dark"]}, ax=ax)
    rho, p = stats.spearmanr(f4b["IFN_antigen_index"], f4b["CXCL10_CXCR3_axis"])
    ax.text(0.05, 0.95, f"rho={rho:.2f}\nP={p:.2g}", transform=ax.transAxes, va="top")
    ax.set_title("IFN-antigen index links to CXCL10-CXCR3")
    ax.set_xlabel("IFN-antigen index")
    ax.set_ylabel("CXCL10-CXCR3 axis")
    save_panel(fig, 4, "B", "IFN-antigen index links to CXCL10-CXCR3")

    fig, ax = new_fig(3.8, 3.4)
    f4c["response"] = response_label(f4c["benefit"], "MPR/pCR", "non-MPR/NE")
    sns.boxplot(data=f4c, x="response", y="interaction_score", hue="response", fliersize=0, palette=response_palette("non-MPR/NE", "MPR/pCR"), legend=False, ax=ax)
    sns.stripplot(data=f4c, x="response", y="interaction_score", color=PALETTE["neutral"], size=3, alpha=0.65, ax=ax)
    ax.set_title("Single-cell CXCL9/10/11-CXCR3 LR score")
    ax.set_xlabel("")
    ax.set_ylabel("Interaction score")
    save_panel(fig, 4, "C", "Single-cell CXCL9/10/11-CXCR3 LR score")

    fig, ax = new_fig(4.0, 3.4)
    sc = ax.scatter(f4d["spearman_with_effector_score"], f4d["minus_log10_p"], c=f4d["mpr_mean_ligand_score"], s=45 + 175 * (f4d["mpr_mean_ligand_score"] - f4d["mpr_mean_ligand_score"].min()) / max(f4d["mpr_mean_ligand_score"].max() - f4d["mpr_mean_ligand_score"].min(), 1e-9), cmap=SEQUENTIAL_CMAP, edgecolor="white", linewidth=0.5)
    fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.03, label="MPR mean ligand score")
    for row in f4d.sort_values("minus_log10_p", ascending=False).head(4).itertuples(index=False):
        ax.text(row.spearman_with_effector_score, row.minus_log10_p, str(row.axis), fontsize=6, ha="left", va="bottom")
    ax.axvline(0, color=PALETTE["dark"], linewidth=0.8)
    ax.set_title("Ligand activity against effector module")
    ax.set_xlabel("Spearman rho")
    ax.set_ylabel("-log10(P)")
    save_panel(fig, 4, "D", "Ligand activity against effector module")

    fig, ax = new_fig(5.0, 3.4)
    f4e["target"] = f4e["y"].map(lambda x: FEATURE_LABELS.get(x, x))
    dot_matrix(ax, f4e, "analysis", "target", "rho", "p_value", "CXCL10-CXCR3 cross-context correlations", "Spearman rho")
    save_panel(fig, 4, "E", "CXCL10-CXCR3 cross-context correlations")


def export_figure5_panels() -> None:
    f5a = read_panel("figure_5_panela_exclusion_feature_response_effects.tsv")
    f5b = read_panel("figure_5_panelb_myeloid_deconvolution_moderately_aligns_with_spp1_macrophage_score.tsv")
    f5c = read_panel("figure_5_panelc_deconvolution_module_correlation_matrix.tsv")
    f5d = read_panel("figure_5_paneld_tcga_exclusion_index_density_vs_sctime_ai.tsv")
    f5e = read_panel("figure_5_panele_cptac_exclusion_index_by_stage_group.tsv")
    f5f = read_panel("figure_5_panelf_tcga_exclusion_co_program_structure.tsv")

    fig, ax = new_fig(4.5, 3.4)
    plot = f5a[f5a["feature"].isin(["SPP1_macrophage", "TGF_beta_EMT", "Hypoxia", "Treg", "Terminal_exhausted_CD8"])].copy()
    plot["label"] = plot["feature"].map(lambda x: FEATURE_LABELS.get(x, x))
    lollipop(ax, plot["label"], plot["mean_difference"], "Exclusion-feature response effects", "Responder minus non-responder mean score")
    save_panel(fig, 5, "A", "Exclusion-feature response effects")

    fig, ax = new_fig(4.0, 3.4)
    sns.regplot(data=f5b, x="Myeloid/macrophage", y="SPP1_macrophage", scatter_kws={"s": 24, "alpha": 0.55}, line_kws={"color": PALETTE["dark"]}, ax=ax)
    sub = f5b[["Myeloid/macrophage", "SPP1_macrophage"]].dropna()
    rho, p = stats.spearmanr(sub["Myeloid/macrophage"], sub["SPP1_macrophage"]) if len(sub) >= 4 else (np.nan, np.nan)
    ax.text(0.05, 0.95, f"rho={rho:.2f}\nP={p:.2g}", transform=ax.transAxes, va="top")
    ax.set_title("Myeloid deconvolution moderately aligns with SPP1 score")
    ax.set_xlabel("NNLS myeloid/macrophage fraction")
    ax.set_ylabel("SPP1 macrophage module")
    save_panel(fig, 5, "B", "Myeloid deconvolution moderately aligns with SPP1 macrophage score")

    fig, ax = new_fig(4.8, 3.6)
    f5c["module"] = f5c["y"].map(lambda x: FEATURE_LABELS.get(x, x))
    dot_matrix(ax, f5c, "x", "module", "rho", "p_value", "Deconvolution-module correlation matrix", "Spearman rho")
    save_panel(fig, 5, "C", "Deconvolution-module correlation matrix")

    fig, ax = new_fig(4.0, 3.4)
    hb = ax.hexbin(f5d["immune_exclusion_index"], f5d["scTIME_AI_score"], gridsize=32, mincnt=1, cmap=SEQUENTIAL_CMAP, linewidths=0)
    fig.colorbar(hb, ax=ax, fraction=0.046, pad=0.03, label="Samples")
    rho, p = stats.spearmanr(f5d["immune_exclusion_index"], f5d["scTIME_AI_score"]) if len(f5d) >= 4 else (np.nan, np.nan)
    ax.text(0.05, 0.95, f"rho={rho:.2f}\nP={p:.2g}", transform=ax.transAxes, va="top")
    ax.set_title("TCGA exclusion-index density vs scTIME-AI")
    ax.set_xlabel("Exclusion index")
    ax.set_ylabel("scTIME-AI score")
    save_panel(fig, 5, "D", "TCGA exclusion-index density vs scTIME-AI")

    fig, ax = new_fig(4.6, 3.5)
    sns.violinplot(data=f5e, x="cohort_omics", y="immune_exclusion_index", hue="stage_group", cut=0, inner=None, palette=response_palette("Early", "Advanced"), ax=ax)
    sns.stripplot(data=f5e, x="cohort_omics", y="immune_exclusion_index", hue="stage_group", dodge=True, palette={k: PALETTE["neutral"] for k in f5e["stage_group"].dropna().unique()}, size=1.4, alpha=0.35, legend=False, ax=ax)
    ax.set_title("CPTAC exclusion index by stage group")
    ax.set_xlabel("")
    ax.set_ylabel("Exclusion index")
    ax.tick_params(axis="x", rotation=20)
    strip_duplicate_legend(ax)
    save_panel(fig, 5, "E", "CPTAC exclusion index by stage group")

    fig, ax = new_fig(5.0, 3.8)
    keep = ["SPP1_macrophage", "TGF_beta_EMT", "Hypoxia", "Treg", "Terminal_exhausted_CD8"]
    plot = f5f[f5f["x"].isin(keep) & f5f["y"].isin(keep + ["scTIME_AI_score"])].copy()
    plot["program"] = plot["x"].map(lambda x: FEATURE_LABELS.get(x, x))
    plot["target"] = plot["y"].map(lambda x: FEATURE_LABELS.get(x, x))
    dot_matrix(ax, plot, "program", "target", "rho", "p_value", "TCGA exclusion co-program structure", "Spearman rho")
    save_panel(fig, 5, "F", "TCGA exclusion co-program structure")


def export_figure6_panels() -> None:
    f6a = read_panel("figure_6_panela_cd274_and_sctime_ai_co_state_in_ici_cohorts.tsv")
    f6b = read_panel("figure_6_panelb_high_confidence_sctime_ai_enrichment_versus_cd274_stratification.tsv")
    f6c = read_panel("figure_6_panelc_cd274_high_sctime_ai_low_feature_contrast.tsv")
    f6d = read_panel("figure_6_paneld_tcr_clonality_and_immune_cell_composition.tsv")
    f6e = read_panel("figure_6_panele_tcr_clonality_metrics_by_pathological_response.tsv")

    fig, ax = new_fig(4.0, 3.4)
    marker_map = dict(zip(sorted(f6a["dataset"].dropna().unique()), ["s", "^", "o", "D"]))
    sns.scatterplot(data=f6a, x="CD274_expr", y="scTIME_AI_score", hue="response", style="dataset", markers=marker_map, s=54, palette=response_palette("no benefit", "benefit/MPR"), ax=ax)
    ax.axvline(f6a["CD274_expr"].median(), color=PALETTE["neutral"], linestyle="--", linewidth=0.8)
    ax.axhline(f6a["scTIME_AI_score"].median(), color=PALETTE["neutral"], linestyle="--", linewidth=0.8)
    rho, p = stats.spearmanr(f6a["CD274_expr"], f6a["scTIME_AI_score"])
    ax.text(0.05, 0.95, f"rho={rho:.2f}\nP={p:.2g}", transform=ax.transAxes, va="top")
    ax.set_title("CD274 and scTIME-AI co-state in ICI cohorts")
    ax.set_xlabel("CD274 expression/module")
    ax.set_ylabel("scTIME-AI score")
    ax.legend(frameon=False, fontsize=6)
    save_panel(fig, 6, "A", "CD274 and scTIME-AI co-state in ICI cohorts")

    fig, ax = new_fig(4.8, 3.4)
    order = ["scTIME-AI bottom quartile", "scTIME-AI intermediate", "scTIME-AI top quartile", "CD274 low", "CD274 high"]
    resp_plot = f6b[f6b["group"].isin(order)].set_index("group").reindex(order).dropna(subset=["n"]).reset_index()
    y = np.arange(len(resp_plot))
    xerr = np.vstack([resp_plot["benefit_rate"] - resp_plot["ci_low"], resp_plot["ci_high"] - resp_plot["benefit_rate"]])
    colors = [PALETTE["negative"], PALETTE["neutral"], PALETTE["positive"], PALETTE["neutral"], PALETTE["neutral"]][: len(resp_plot)]
    ax.errorbar(resp_plot["benefit_rate"], y, xerr=xerr, fmt="none", ecolor=PALETTE["neutral"], capsize=3)
    ax.scatter(resp_plot["benefit_rate"], y, c=colors, s=52, zorder=3)
    for yi, row in zip(y, resp_plot.itertuples(index=False)):
        ax.text(min(row.ci_high + 0.03, 0.98), yi, f"n={int(row.n)}", va="center", fontsize=7)
    ax.set_yticks(y)
    ax.set_yticklabels(resp_plot["group"])
    ax.set_xlim(0, 1.05)
    ax.set_title("High-confidence scTIME-AI enrichment vs CD274")
    ax.set_xlabel("Benefit/MPR fraction with 95% Wilson CI")
    ax.set_ylabel("")
    save_panel(fig, 6, "B", "High-confidence scTIME-AI enrichment versus CD274 stratification")

    fig, ax = new_fig(4.5, 3.4)
    f6c["label"] = f6c["feature"].map(lambda x: FEATURE_LABELS.get(x, x))
    lollipop(ax, f6c["label"], f6c["difference"], "CD274-high/scTIME-AI-low feature contrast", "High scTIME-AI minus CD274-high/scTIME-AI-low")
    save_panel(fig, 6, "C", "CD274-high scTIME-AI-low feature contrast")

    fig, ax = new_fig(5.0, 3.7)
    f6d["metric"] = f6d["x"].map({"top_clonotype_fraction": "Top clonotype", "expanded_fraction": "Expanded fraction", "normalized_shannon": "TCR diversity"})
    dot_matrix(ax, f6d, "metric", "y", "rho", "p_value", "TCR clonality and immune-cell composition", "Spearman rho")
    save_panel(fig, 6, "D", "TCR clonality and immune-cell composition")

    fig, ax = new_fig(4.4, 3.4)
    f6e["metric"] = f6e["metric"].map({"top_clonotype_fraction": "Top clonotype", "expanded_fraction": "Expanded fraction", "normalized_shannon": "TCR diversity"}).fillna(f6e["metric"])
    sns.boxplot(data=f6e, x="metric", y="value", hue="response", fliersize=0, palette=response_palette("non-MPR", "MPR/pCR"), ax=ax)
    sns.stripplot(data=f6e, x="metric", y="value", hue="response", dodge=True, palette={k: PALETTE["neutral"] for k in f6e["response"].dropna().unique()}, size=1.8, alpha=0.35, legend=False, ax=ax)
    ax.set_title("TCR clonality metrics by pathological response")
    ax.set_xlabel("")
    ax.set_ylabel("Metric value")
    ax.tick_params(axis="x", rotation=20)
    strip_duplicate_legend(ax)
    save_panel(fig, 6, "E", "TCR clonality metrics by pathological response")


def export_figure7_panels() -> None:
    f7a = read_panel("figure_7_panela_gse135222_pfs_validation.tsv")
    f7b = read_panel("figure_7_panelb_survival_association_summary.tsv")
    f7c = read_panel("figure_7_panelc_tcga_luad_lusc_sctime_ai_distributions.tsv")
    f7d = read_panel("figure_7_paneld_cptac_proteome_phosphoproteome_projection.tsv")
    f7e = read_panel("figure_7_panele_cptac_stage_shifts_in_key_immune_programs.tsv")

    fig, ax = new_fig(4.0, 3.4)
    for label, color in [("High", PALETTE["positive"]), ("Low", PALETTE["negative"])]:
        sub = f7a[f7a["score_group"].eq(label)]
        if not sub.empty:
            ax.step(sub["time"], sub["survival"], where="post", label=f"{label} (n={int(sub['n'].iloc[0])})", color=color)
    ax.set_title("GSE135222 PFS validation")
    ax.set_xlabel("PFS time")
    ax.set_ylabel("PFS probability")
    ax.legend(frameon=False)
    save_panel(fig, 7, "A", "GSE135222 PFS validation")

    fig, ax = new_fig(4.0, 3.2)
    forest = f7b.dropna(subset=["hr"]).sort_values("hr")
    y = np.arange(len(forest))
    ax.scatter(forest["hr"], y, s=70, color=PALETTE["positive"])
    ax.axvline(1, color=PALETTE["dark"], linestyle="--", linewidth=0.8)
    for yi, row in zip(y, forest.itertuples(index=False)):
        ax.text(row.hr * 1.05, yi, f"P={row.p:.2g}", va="center", fontsize=7)
    ax.set_yticks(y)
    ax.set_yticklabels(forest["analysis"])
    ax.set_xscale("log")
    ax.set_title("Survival association summary")
    ax.set_xlabel("Hazard ratio per scTIME-AI score")
    ax.set_ylabel("")
    save_panel(fig, 7, "B", "Survival association summary")

    fig, ax = new_fig(3.8, 3.4)
    sns.violinplot(data=f7c, x="project", y="scTIME_AI_score", hue="project", inner=None, cut=0, palette=category_palette(f7c["project"]), legend=False, ax=ax)
    sns.boxplot(data=f7c, x="project", y="scTIME_AI_score", width=0.23, color="white", fliersize=0, ax=ax)
    ax.set_title("TCGA-LUAD/LUSC scTIME-AI distributions")
    ax.set_xlabel("")
    ax.set_ylabel("scTIME-AI score")
    save_panel(fig, 7, "C", "TCGA-LUAD LUSC scTIME-AI distributions")

    fig, ax = new_fig(4.6, 3.4)
    sns.boxplot(data=f7d, x="cohort_omics", y="scTIME_AI_score", hue="stage_group", fliersize=0, ax=ax, palette=response_palette("Early", "Advanced"))
    sns.stripplot(data=f7d, x="cohort_omics", y="scTIME_AI_score", hue="stage_group", dodge=True, palette={k: PALETTE["neutral"] for k in f7d["stage_group"].dropna().unique()}, size=1.4, alpha=0.35, legend=False, ax=ax)
    ax.set_title("CPTAC proteome/phosphoproteome projection")
    ax.set_xlabel("")
    ax.set_ylabel("scTIME-AI score")
    ax.tick_params(axis="x", rotation=20)
    strip_duplicate_legend(ax)
    save_panel(fig, 7, "D", "CPTAC proteome phosphoproteome projection")

    fig, ax = new_fig(5.0, 3.6)
    key = ["scTIME_AI_score", "IFN_response", "Antigen_presentation", "Cytotoxic_CD8", "CXCL10_CXCR3_axis", "SPP1_macrophage", "TGF_beta_EMT", "Hypoxia", "CD274_expr"]
    plot = f7e[f7e["feature"].isin(key)].copy()
    plot["feature_label"] = plot["feature"].map(lambda x: FEATURE_LABELS.get(x, x))
    plot["label"] = plot["cohort"] + " " + plot["omics"]
    feature_order = [FEATURE_LABELS.get(x, x) for x in key if FEATURE_LABELS.get(x, x) in plot["feature_label"].unique()]
    palette = category_palette(plot["label"])
    for label, sub in plot.groupby("label", observed=True):
        sub = sub[sub["feature_label"].isin(feature_order)]
        y_pos = [feature_order.index(v) for v in sub["feature_label"]]
        ax.scatter(sub["advanced_minus_early"], y_pos, label=label, s=38, alpha=0.78, color=palette[label])
    ax.axvline(0, color=PALETTE["dark"], linewidth=0.8)
    ax.set_yticks(range(len(feature_order)))
    ax.set_yticklabels(feature_order)
    ax.set_title("CPTAC stage shifts in key immune programs")
    ax.set_xlabel("Advanced minus early")
    ax.set_ylabel("")
    ax.legend(frameon=False, fontsize=6, bbox_to_anchor=(1.02, 1), loc="upper left")
    save_panel(fig, 7, "E", "CPTAC stage shifts in key immune programs")


def main() -> None:
    setup()
    PANEL_FIGURES.mkdir(parents=True, exist_ok=True)
    for path in PANEL_FIGURES.glob("*"):
        if path.is_file() and path.suffix.lower() in {".png", ".pdf"}:
            path.unlink()
    export_figure2_panels()
    export_figure3_panels()
    export_figure4_panels()
    export_figure5_panels()
    export_figure6_panels()
    export_figure7_panels()


if __name__ == "__main__":
    main()
