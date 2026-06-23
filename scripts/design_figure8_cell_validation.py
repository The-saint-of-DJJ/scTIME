#!/usr/bin/env python
"""Generate Figure 8 from the experimental validation data in Biodata."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap


ROOT = Path(__file__).resolve().parents[1]
BIODATA = ROOT / "Biodata"
FIGURES = ROOT / "results" / "figures"
PANEL_DATA = ROOT / "results" / "figure_design" / "panel_data"
MANIFEST = PANEL_DATA / "panel_data_manifest.tsv"

PALETTE = {
    "positive": "#2f6f9f",
    "negative": "#b4473d",
    "neutral": "#6d6e71",
    "dark": "#2a2a2a",
    "light": "#f1f3f4",
}
DIVERGING_CMAP = LinearSegmentedColormap.from_list(
    "sctime_three_color_diverging",
    [PALETTE["negative"], "#f7f7f7", PALETTE["positive"]],
)

CONDITION_ORDER = ["G1", "G2", "G3", "G4", "G5"]
CONDITION_LABELS = {
    "G1": "G1\nControl",
    "G2": "G2\nIFN",
    "G3": "G3\nIFN+\nCXCL10 block",
    "G4": "G4\nIFN+\nexclusion",
    "G5": "G5\nRescue",
}
SHORT_CONDITION_LABELS = {
    "G1": "G1",
    "G2": "G2",
    "G3": "G3",
    "G4": "G4",
    "G5": "G5",
}
CONDITION_COLORS = {
    "G1": PALETTE["neutral"],
    "G2": PALETTE["positive"],
    "G3": PALETTE["neutral"],
    "G4": PALETTE["negative"],
    "G5": PALETTE["positive"],
}


def setup() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    PANEL_DATA.mkdir(parents=True, exist_ok=True)
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


def read_csv(name: str) -> pd.DataFrame:
    return pd.read_csv(BIODATA / name, encoding="utf-8-sig")


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.12,
        1.08,
        label,
        transform=ax.transAxes,
        fontsize=12,
        fontweight="bold",
        va="top",
        ha="left",
    )


def savefig(fig: plt.Figure, filename: str) -> None:
    for ext in ["png", "pdf"]:
        fig.savefig(FIGURES / f"{filename}.{ext}", bbox_inches="tight")
    plt.close(fig)


def zscore(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    sd = values.std(ddof=0)
    if not np.isfinite(sd) or sd == 0:
        return values * 0
    return (values - values.mean()) / sd


def mean_z_heatmap(
    df: pd.DataFrame,
    value_col: str,
    feature_col: str,
    feature_order: list[str],
) -> pd.DataFrame:
    out = df.copy()
    out["z_value"] = out.groupby(feature_col, group_keys=False)[value_col].apply(zscore)
    heat = (
        out.groupby(["condition", feature_col], observed=True)["z_value"]
        .mean()
        .unstack(feature_col)
        .reindex(index=CONDITION_ORDER, columns=feature_order)
    )
    return heat


def condition_tick_labels(ax: plt.Axes) -> None:
    ax.set_xticklabels([CONDITION_LABELS.get(t.get_text(), t.get_text()) for t in ax.get_xticklabels()], rotation=0)


def add_bar_points(
    ax: plt.Axes,
    data: pd.DataFrame,
    y_col: str,
    y_label: str,
    title: str,
) -> None:
    positions = np.arange(len(CONDITION_ORDER))
    means = data.groupby("condition")[y_col].mean().reindex(CONDITION_ORDER)
    stds = data.groupby("condition")[y_col].std().reindex(CONDITION_ORDER)
    colors = [CONDITION_COLORS[c] for c in CONDITION_ORDER]
    ax.bar(positions, means, yerr=stds, color=colors, alpha=0.78, edgecolor="#333333", linewidth=0.55, capsize=2)
    for i, condition in enumerate(CONDITION_ORDER):
        vals = data.loc[data["condition"].eq(condition), y_col].dropna().to_numpy()
        if len(vals):
            offsets = np.linspace(-0.12, 0.12, len(vals))
            ax.scatter(
                np.full(len(vals), i) + offsets,
                vals,
                s=16,
                facecolor="#f7f7f7",
                edgecolor="#333333",
                linewidth=0.45,
                zorder=3,
            )
    ax.axhline(0, color="#333333", linewidth=0.65, alpha=0.65)
    ax.set_xticks(positions)
    ax.set_xticklabels([SHORT_CONDITION_LABELS[c] for c in CONDITION_ORDER], rotation=0)
    ax.set_ylabel(y_label)
    ax.set_xlabel("")
    ax.set_title(title, loc="left", fontweight="bold")


def add_grouped_bar_points(
    ax: plt.Axes,
    data: pd.DataFrame,
    value_cols: list[str],
    labels: list[str],
    colors: list[str],
    y_label: str,
    title: str,
) -> None:
    positions = np.arange(len(CONDITION_ORDER))
    width = 0.34
    offsets = np.linspace(-width / 2, width / 2, len(value_cols))
    for col, label, color, offset in zip(value_cols, labels, colors, offsets, strict=True):
        means = data.groupby("condition")[col].mean().reindex(CONDITION_ORDER)
        stds = data.groupby("condition")[col].std().reindex(CONDITION_ORDER)
        x = positions + offset
        ax.bar(x, means, width=width, yerr=stds, color=color, alpha=0.78, edgecolor="#333333", linewidth=0.55, capsize=2, label=label)
        for i, condition in enumerate(CONDITION_ORDER):
            vals = data.loc[data["condition"].eq(condition), col].dropna().to_numpy()
            if len(vals):
                point_offsets = np.linspace(-0.06, 0.06, len(vals))
                ax.scatter(
                    np.full(len(vals), x[i]) + point_offsets,
                    vals,
                    s=14,
                    facecolor="#f7f7f7",
                    edgecolor="#333333",
                    linewidth=0.45,
                    zorder=3,
                )
    ax.axhline(0, color="#333333", linewidth=0.65, alpha=0.65)
    ax.set_xticks(positions)
    ax.set_xticklabels([SHORT_CONDITION_LABELS[c] for c in CONDITION_ORDER], rotation=0)
    ax.set_ylabel(y_label)
    ax.set_xlabel("")
    ax.set_title(title, loc="left", fontweight="bold")
    ax.legend(frameon=False, loc="upper left", ncol=1)


def plot_figure() -> None:
    tumor = read_csv("tumor_qPCR_long.csv")
    secretome = read_csv("secretome_viability_long.csv")
    migration = read_csv("CD8_migration_long.csv")
    scores = read_csv("score_by_replicate.csv")
    summary = read_csv("summary_indices_by_condition.csv")

    fig, axes = plt.subplot_mosaic(
        [["A", "B", "C"], ["D", "E", "F"]],
        figsize=(14.2, 8.4),
        gridspec_kw={"width_ratios": [1.35, 1.05, 0.9], "height_ratios": [1.05, 1.0]},
        constrained_layout=True,
    )

    # Panel A: tumor qPCR feature direction.
    genes = ["CXCL10", "CXCL9", "CXCL11", "HLA-A", "B2M", "STAT1", "IFIT1", "CD274", "SPP1", "TGFB1", "CA9", "VEGFA"]
    heat_a = mean_z_heatmap(tumor, "log2_relative_expression", "gene", genes)
    sns.heatmap(
        heat_a,
        ax=axes["A"],
        cmap=DIVERGING_CMAP,
        center=0,
        vmin=-2.2,
        vmax=2.2,
        linewidths=0.35,
        linecolor="#ffffff",
        cbar_kws={"label": "mean z-score"},
    )
    axes["A"].set_yticklabels([CONDITION_LABELS[c].replace("\n", " ") for c in heat_a.index], rotation=0)
    axes["A"].set_xticklabels(genes, rotation=45, ha="right")
    axes["A"].set_xlabel("")
    axes["A"].set_ylabel("")
    axes["A"].set_title("qPCR feature direction", loc="left", fontweight="bold")
    for x in [7, 8]:
        axes["A"].axvline(x, color="#333333", linewidth=0.8)
    panel_label(axes["A"], "A")

    # Panel B: secretome and viability.
    secretome_order = [
        "CXCL10_bioavailable_pg_ml",
        "CXCL9_pg_ml",
        "CXCL11_pg_ml",
        "SPP1_pg_ml",
        "TGFB1_pg_ml",
        "celltiter_glo_pct",
    ]
    secretome_labels = ["CXCL10", "CXCL9", "CXCL11", "SPP1", "TGF-beta1", "Viability"]
    secretome_z = secretome.loc[secretome["analyte"].isin(secretome_order)].copy()
    secretome_z["z_value"] = secretome_z.groupby("analyte", group_keys=False)["value"].apply(zscore)
    secretome_profile = (
        secretome_z.groupby(["condition", "analyte"], observed=True)["z_value"]
        .mean()
        .reset_index()
    )
    analyte_styles = {
        "CXCL10_bioavailable_pg_ml": ("CXCL10", PALETTE["positive"], "o", "-"),
        "CXCL9_pg_ml": ("CXCL9", PALETTE["positive"], "s", "--"),
        "CXCL11_pg_ml": ("CXCL11", PALETTE["positive"], "^", ":"),
        "SPP1_pg_ml": ("SPP1", PALETTE["negative"], "o", "-"),
        "TGFB1_pg_ml": ("TGF-beta1", PALETTE["negative"], "s", "--"),
        "celltiter_glo_pct": ("Viability", PALETTE["neutral"], "D", "-."),
    }
    x_positions = np.arange(len(CONDITION_ORDER))
    for analyte in secretome_order:
        label, color, marker, linestyle = analyte_styles[analyte]
        sub = secretome_profile.loc[secretome_profile["analyte"].eq(analyte)].set_index("condition").reindex(CONDITION_ORDER)
        axes["B"].plot(
            x_positions,
            sub["z_value"],
            marker=marker,
            markersize=4,
            linewidth=1.2,
            linestyle=linestyle,
            color=color,
            alpha=0.9,
            label=label,
        )
    axes["B"].axhline(0, color="#333333", linewidth=0.65, alpha=0.65)
    axes["B"].set_xticks(x_positions)
    axes["B"].set_xticklabels([SHORT_CONDITION_LABELS[c] for c in CONDITION_ORDER])
    axes["B"].set_ylabel("mean z-score")
    axes["B"].set_xlabel("")
    axes["B"].set_title("Secretome profile", loc="left", fontweight="bold")
    axes["B"].set_ylim(-2.25, 2.05)
    axes["B"].legend(frameon=False, loc="upper left", ncol=2, columnspacing=0.8, handlelength=1.6)
    panel_label(axes["B"], "B")

    # Panel C: CD8 migration.
    migration_plot = scores[["condition", "condition_name", "biological_replicate", "CD8_migration_index"]].copy()
    add_bar_points(axes["C"], migration_plot, "CD8_migration_index", "z-index", "CD8 migration")
    panel_label(axes["C"], "C")

    # Panel D: activation and cytotoxic function.
    function_plot = scores[
        [
            "condition",
            "condition_name",
            "biological_replicate",
            "T_cell_activation_index",
            "tumor_killing_index",
        ]
    ].copy()
    add_grouped_bar_points(
        axes["D"],
        function_plot,
        ["T_cell_activation_index", "tumor_killing_index"],
        ["T cell activation", "Tumor killing"],
        [PALETTE["positive"], PALETTE["neutral"]],
        "z-index",
        "T cell activation and tumor killing",
    )
    panel_label(axes["D"], "D")

    # Panel E: experimental scTIME-AI-like score.
    score_plot = scores[["condition", "condition_name", "biological_replicate", "experimental_scTIME_AI_like_score"]].copy()
    positions = {condition: i for i, condition in enumerate(CONDITION_ORDER)}
    for _, sub in score_plot.groupby("biological_replicate", sort=False):
        sub = sub.set_index("condition").reindex(CONDITION_ORDER).reset_index()
        axes["E"].plot(
            [positions[c] for c in sub["condition"]],
            sub["experimental_scTIME_AI_like_score"],
            color="#9a9a9a",
            linewidth=0.75,
            alpha=0.55,
            zorder=1,
        )
    add_bar_points(axes["E"], score_plot, "experimental_scTIME_AI_like_score", "score", "Experimental scTIME-AI-like score")
    axes["E"].set_ylim(
        min(-2.3, score_plot["experimental_scTIME_AI_like_score"].min() - 0.25),
        max(2.1, score_plot["experimental_scTIME_AI_like_score"].max() + 0.25),
    )
    panel_label(axes["E"], "E")

    # Panel F: model-function concordance summary.
    concordance = scores[
        [
            "condition",
            "condition_name",
            "biological_replicate",
            "experimental_scTIME_AI_like_score",
            "CD8_migration_index",
            "T_cell_activation_index",
            "tumor_killing_index",
        ]
    ].copy()
    concordance["immune_function_composite"] = concordance[
        ["CD8_migration_index", "T_cell_activation_index", "tumor_killing_index"]
    ].mean(axis=1)
    long_f = concordance.melt(
        id_vars=["condition", "biological_replicate"],
        value_vars=["experimental_scTIME_AI_like_score", "immune_function_composite"],
        var_name="metric",
        value_name="value",
    )
    metric_styles = {
        "experimental_scTIME_AI_like_score": ("scTIME-AI-like score", PALETTE["positive"], "o"),
        "immune_function_composite": ("Immune function composite", PALETTE["neutral"], "s"),
    }
    x_positions = np.arange(len(CONDITION_ORDER))
    for metric, (label, color, marker) in metric_styles.items():
        sub = long_f.loc[long_f["metric"].eq(metric)]
        mean = sub.groupby("condition")["value"].mean().reindex(CONDITION_ORDER)
        sd = sub.groupby("condition")["value"].std().reindex(CONDITION_ORDER)
        axes["F"].errorbar(
            x_positions,
            mean,
            yerr=sd,
            color=color,
            marker=marker,
            linewidth=1.5,
            markersize=5,
            capsize=2,
            label=label,
            zorder=3,
        )
        for i, condition in enumerate(CONDITION_ORDER):
            vals = sub.loc[sub["condition"].eq(condition), "value"].dropna().to_numpy()
            if len(vals):
                offsets = np.linspace(-0.06, 0.06, len(vals))
                axes["F"].scatter(
                    np.full(len(vals), i) + offsets,
                    vals,
                    s=11,
                    facecolor="#f7f7f7",
                    edgecolor=color,
                    linewidth=0.45,
                    alpha=0.85,
                    zorder=2,
                )
    axes["F"].axhline(0, color="#333333", linewidth=0.55, alpha=0.45)
    axes["F"].set_xticks(x_positions)
    axes["F"].set_xticklabels([SHORT_CONDITION_LABELS[c] for c in CONDITION_ORDER])
    axes["F"].set_xlabel("")
    axes["F"].set_ylabel("z-index / score")
    axes["F"].set_title("Model-function concordance trajectory", loc="left", fontweight="bold")
    axes["F"].legend(frameon=False, loc="upper left")
    panel_label(axes["F"], "F")

    fig.suptitle(
        "Figure 8. Experimental validation of the scTIME-AI responder and exclusion axes",
        fontsize=12,
        fontweight="bold",
        x=0.015,
        ha="left",
    )
    savefig(fig, "figure8_cell_validation_sctime_axes")


def export_panel_data() -> None:
    records = []
    function_plot = read_csv("score_by_replicate.csv")[
        [
            "condition",
            "condition_name",
            "model_interpretation",
            "biological_replicate",
            "T_cell_activation_index",
            "tumor_killing_index",
        ]
    ].copy()
    concordance = read_csv("score_by_replicate.csv")
    concordance["immune_function_composite"] = concordance[
        ["CD8_migration_index", "T_cell_activation_index", "tumor_killing_index"]
    ].mean(axis=1)
    exports = [
        (
            "A",
            "qPCR validates scTIME-AI feature direction",
            read_csv("tumor_qPCR_long.csv"),
            "Biodata/tumor_qPCR_long.csv",
            "Raw tumor-cell qPCR values used for the qPCR feature-direction heatmap.",
            "figure_8_panela_qpcr_validates_sctime_ai_feature_direction.tsv",
        ),
        (
            "B",
            "Secretome profile and viability",
            read_csv("secretome_viability_long.csv"),
            "Biodata/secretome_viability_long.csv",
            "Raw ELISA/multiplex and CellTiter-Glo values used for the secretome profile plot.",
            "figure_8_panelb_secretome_and_viability.tsv",
        ),
        (
            "C",
            "CD8 T cell transwell migration",
            read_csv("CD8_migration_long.csv"),
            "Biodata/CD8_migration_long.csv",
            "Raw CD8 T cell transwell migration measurements.",
            "figure_8_panelc_cd8_t_cell_transwell_migration.tsv",
        ),
        (
            "D",
            "T cell activation and tumor killing",
            function_plot,
            "Biodata/Tcell_activation_killing_long.csv;Biodata/score_by_replicate.csv",
            "Replicate-level activation and tumor-killing indices used for the grouped functional-index plot; raw cytokine and cytotoxicity measurements are in Biodata/Tcell_activation_killing_long.csv.",
            "figure_8_paneld_t_cell_activation_and_tumor_killing.tsv",
        ),
        (
            "E",
            "Experimental scTIME-AI-like score",
            read_csv("score_by_replicate.csv"),
            "Biodata/score_by_replicate.csv",
            "Replicate-level feature z-scores and experimental scTIME-AI-like score.",
            "figure_8_panele_experimental_sctime_ai_like_score.tsv",
        ),
        (
            "F",
            "Model-function concordance",
            concordance,
            "Biodata/score_by_replicate.csv",
            "Replicate-level experimental scTIME-AI-like score and immune-function composite used for the concordance trajectory plot.",
            "figure_8_panelf_model_function_concordance.tsv",
        ),
    ]
    for panel, title, df, source, description, filename in exports:
        out_path = PANEL_DATA / filename
        df.to_csv(out_path, sep="\t", index=False)
        records.append(
            {
                "figure": "Figure 8",
                "panel": panel,
                "title": title,
                "panel_data_file": str(out_path.relative_to(ROOT)),
                "n_rows": len(df),
                "n_columns": len(df.columns),
                "source_tables": source,
                "description": description,
            }
        )

    new_manifest = pd.DataFrame(records)
    if MANIFEST.exists():
        manifest = pd.read_csv(MANIFEST, sep="\t")
        manifest = manifest.loc[manifest["figure"].ne("Figure 8")].copy()
        manifest = pd.concat([manifest, new_manifest], ignore_index=True)
    else:
        manifest = new_manifest
    manifest.to_csv(MANIFEST, sep="\t", index=False)


def main() -> None:
    setup()
    plot_figure()
    export_panel_data()
    print("Generated Figure 8 and exported panel source data.")


if __name__ == "__main__":
    main()
