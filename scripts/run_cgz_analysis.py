#!/usr/bin/env python
"""Reproducible analysis pipeline for the CGZ lung cancer immunology project.

The script intentionally uses only the core scientific Python stack installed
in the BJMS conda environment. It performs the computational portions that can
be run from the downloaded public data: GEO metadata parsing, bulk signature
projection, response modelling, survival validation, single-cell annotation
summaries, TCR summaries, and figure/table generation.
"""

from __future__ import annotations

import csv
import gzip
import json
import logging
import pickle
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import seaborn as sns
from scipy import sparse, stats
from scipy.optimize import nnls
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from sklearn.svm import SVC
except Exception:  # pragma: no cover
    SVC = None

try:
    import scanpy as sc
except Exception:  # pragma: no cover
    sc = None

try:
    from xgboost import XGBClassifier
except Exception:  # pragma: no cover
    XGBClassifier = None

try:
    from lightgbm import LGBMClassifier
except Exception:  # pragma: no cover
    LGBMClassifier = None

try:
    from statsmodels.duration.hazard_regression import PHReg
except Exception:  # pragma: no cover
    PHReg = None


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
GEO = DATA / "geo"
RESULTS = ROOT / "results"
TABLES = RESULTS / "tables"
FIGURES = RESULTS / "figures"
MODELS = RESULTS / "models"
LOGS = ROOT / "logs"
REFERENCE = DATA / "reference"

# Legacy single-panel quicklooks are disabled so full analysis reruns do not
# repopulate results/figures with superseded main-text candidates.
GENERATE_LEGACY_QUICKLOOK_FIGURES = False


SIGNATURES: dict[str, list[str]] = {
    "IFN_response": [
        "IFIT1",
        "IFIT3",
        "ISG15",
        "STAT1",
        "IRF1",
        "GBP1",
        "CXCL9",
        "CXCL10",
        "CXCL11",
    ],
    "Antigen_presentation": [
        "HLA-A",
        "HLA-B",
        "HLA-C",
        "B2M",
        "TAP1",
        "TAP2",
        "NLRC5",
        "HLA-E",
        "PSMB8",
        "PSMB9",
    ],
    "Cytotoxic_CD8": [
        "CD8A",
        "CD8B",
        "GZMB",
        "PRF1",
        "GNLY",
        "NKG7",
        "IFNG",
        "CCL5",
        "CXCR3",
    ],
    "Progenitor_exhausted_CD8": ["TCF7", "IL7R", "CCR7", "CXCR5", "SELL", "LEF1"],
    "Terminal_exhausted_CD8": ["PDCD1", "LAG3", "HAVCR2", "TIGIT", "TOX", "CTLA4"],
    "Treg": ["FOXP3", "IL2RA", "CTLA4", "IKZF2", "CCR8", "TNFRSF18"],
    "SPP1_macrophage": ["SPP1", "APOE", "TREM2", "LIPA", "CD68", "MARCO"],
    "TGF_beta_EMT": [
        "TGFB1",
        "TGFBR1",
        "TGFBR2",
        "COL1A1",
        "ACTA2",
        "VIM",
        "FN1",
        "ZEB1",
        "SNAI1",
        "TWIST1",
    ],
    "Hypoxia": ["HIF1A", "CA9", "VEGFA", "LDHA", "SLC2A1", "ENO1"],
    "NK": ["NKG7", "GNLY", "KLRD1", "KLRF1", "GZMB", "PRF1"],
    "CXCL10_CXCR3_axis": ["CXCL9", "CXCL10", "CXCL11", "CXCR3"],
}

KNOWN_MARKERS = sorted(set(g for genes in SIGNATURES.values() for g in genes) | {"CD274"})

LR_AXES: list[dict[str, object]] = [
    {"axis": "CXCL9/10/11-CXCR3", "ligands": ["CXCL9", "CXCL10", "CXCL11"], "receptors": ["CXCR3"]},
    {"axis": "CCL5-CCR5", "ligands": ["CCL5"], "receptors": ["CCR5"]},
    {"axis": "CXCL16-CXCR6", "ligands": ["CXCL16"], "receptors": ["CXCR6"]},
    {"axis": "SPP1-CD44", "ligands": ["SPP1"], "receptors": ["CD44"]},
    {"axis": "SPP1-integrin", "ligands": ["SPP1"], "receptors": ["ITGAV", "ITGB1", "ITGB5"]},
    {"axis": "TGFB1-TGFBR", "ligands": ["TGFB1"], "receptors": ["TGFBR1", "TGFBR2"]},
    {"axis": "CD274-PDCD1", "ligands": ["CD274"], "receptors": ["PDCD1"]},
    {"axis": "NECTIN2-TIGIT", "ligands": ["NECTIN2"], "receptors": ["TIGIT"]},
]

LR_GENES = sorted(set(g for axis in LR_AXES for key in ["ligands", "receptors"] for g in axis[key]))
SCANPY_N_HVG = 2000

CELL_STATE_MARKERS: dict[str, list[str]] = {
    "Tumor/epithelial": ["EPCAM", "KRT8", "KRT18", "KRT19", "MUC1"],
    "CD8 T": ["CD3D", "CD3E", "CD8A", "CD8B", "GZMB", "NKG7"],
    "CD4 T": ["CD3D", "CD3E", "CD4", "IL7R", "CCR7"],
    "Treg": ["FOXP3", "IL2RA", "CTLA4", "CCR8"],
    "NK": ["NKG7", "GNLY", "KLRD1", "KLRF1", "NCR1"],
    "B/plasma": ["MS4A1", "CD79A", "CD79B", "MZB1", "JCHAIN"],
    "Myeloid/macrophage": ["LYZ", "CD68", "C1QA", "C1QB", "APOE", "SPP1"],
    "Dendritic": ["FCER1A", "CLEC9A", "XCR1", "LILRA4", "ITGAX"],
    "Fibroblast": ["COL1A1", "COL1A2", "DCN", "LUM", "ACTA2"],
    "Endothelial": ["PECAM1", "VWF", "KDR", "RAMP2"],
    "Mast": ["TPSAB1", "TPSB2", "KIT", "CPA3"],
}

CELL_STATE_MARKER_GENES = sorted(set(g for genes in CELL_STATE_MARKERS.values() for g in genes))

FALLBACK_ENSEMBL_TO_SYMBOL = {
    "ENSG00000169245": "CXCL10",
    "ENSG00000138755": "CXCL9",
    "ENSG00000169248": "CXCL11",
    "ENSG00000186810": "CXCR3",
    "ENSG00000185745": "IFIT1",
    "ENSG00000119917": "IFIT3",
    "ENSG00000187608": "ISG15",
    "ENSG00000115415": "STAT1",
    "ENSG00000125347": "IRF1",
    "ENSG00000117228": "GBP1",
    "ENSG00000206503": "HLA-A",
    "ENSG00000234745": "HLA-B",
    "ENSG00000204525": "HLA-C",
    "ENSG00000166710": "B2M",
    "ENSG00000168394": "TAP1",
    "ENSG00000204267": "TAP2",
    "ENSG00000140853": "NLRC5",
    "ENSG00000204592": "HLA-E",
    "ENSG00000204264": "PSMB8",
    "ENSG00000240065": "PSMB9",
    "ENSG00000153563": "CD8A",
    "ENSG00000172116": "CD8B",
    "ENSG00000100453": "GZMB",
    "ENSG00000180644": "PRF1",
    "ENSG00000115523": "GNLY",
    "ENSG00000105374": "NKG7",
    "ENSG00000111537": "IFNG",
    "ENSG00000271503": "CCL5",
    "ENSG00000081059": "TCF7",
    "ENSG00000168685": "IL7R",
    "ENSG00000126353": "CCR7",
    "ENSG00000160683": "CXCR5",
    "ENSG00000188404": "SELL",
    "ENSG00000138795": "LEF1",
    "ENSG00000188389": "PDCD1",
    "ENSG00000089692": "LAG3",
    "ENSG00000135077": "HAVCR2",
    "ENSG00000181847": "TIGIT",
    "ENSG00000198846": "TOX",
    "ENSG00000163599": "CTLA4",
    "ENSG00000049768": "FOXP3",
    "ENSG00000134460": "IL2RA",
    "ENSG00000030419": "IKZF2",
    "ENSG00000179934": "CCR8",
    "ENSG00000186891": "TNFRSF18",
    "ENSG00000118785": "SPP1",
    "ENSG00000130203": "APOE",
    "ENSG00000095970": "TREM2",
    "ENSG00000107798": "LIPA",
    "ENSG00000129226": "CD68",
    "ENSG00000019169": "MARCO",
    "ENSG00000105329": "TGFB1",
    "ENSG00000106799": "TGFBR1",
    "ENSG00000163513": "TGFBR2",
    "ENSG00000108821": "COL1A1",
    "ENSG00000107796": "ACTA2",
    "ENSG00000026025": "VIM",
    "ENSG00000115414": "FN1",
    "ENSG00000148516": "ZEB1",
    "ENSG00000124216": "SNAI1",
    "ENSG00000122691": "TWIST1",
    "ENSG00000100644": "HIF1A",
    "ENSG00000107159": "CA9",
    "ENSG00000112715": "VEGFA",
    "ENSG00000134333": "LDHA",
    "ENSG00000117394": "SLC2A1",
    "ENSG00000074800": "ENO1",
    "ENSG00000134539": "KLRD1",
    "ENSG00000150045": "KLRF1",
    "ENSG00000120217": "CD274",
}


@dataclass
class ExpressionDataset:
    dataset: str
    expression: pd.DataFrame
    metadata: pd.DataFrame
    endpoint: str | None
    unit: str


def setup_logging() -> None:
    for directory in [RESULTS, TABLES, FIGURES, MODELS, LOGS, REFERENCE]:
        directory.mkdir(parents=True, exist_ok=True)
    log_file = LOGS / "run_cgz_analysis.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(log_file, mode="w"), logging.StreamHandler(sys.stdout)],
    )
    logging.info("CGZ analysis started")
    logging.info("Python executable: %s", sys.executable)


def save_quicklook(fig: plt.Figure, filename: str, *, dpi: int = 220) -> None:
    if GENERATE_LEGACY_QUICKLOOK_FIGURES:
        fig.savefig(FIGURES / filename, dpi=dpi)


def clean_key(value: str) -> str:
    value = value.strip().strip('"').strip()
    value = re.sub(r"\s+", " ", value)
    return value


def normalize_column_name(value: str) -> str:
    value = clean_key(value).lower()
    value = value.replace("%", "pct")
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return value.strip("_")


def normalize_gene_symbol(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip().upper()


def strip_ensembl_version(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).split(".")[0].strip()


def safe_auc(y_true: Iterable[int], y_score: Iterable[float]) -> float:
    y_true = np.asarray(list(y_true), dtype=float)
    y_score = np.asarray(list(y_score), dtype=float)
    if len(np.unique(y_true[~np.isnan(y_true)])) < 2:
        return np.nan
    return float(roc_auc_score(y_true, y_score))


def safe_average_precision(y_true: Iterable[int], y_score: Iterable[float]) -> float:
    y_true = np.asarray(list(y_true), dtype=float)
    y_score = np.asarray(list(y_score), dtype=float)
    if len(np.unique(y_true[~np.isnan(y_true)])) < 2:
        return np.nan
    return float(average_precision_score(y_true, y_score))


def benjamini_hochberg(pvalues: Iterable[float]) -> list[float]:
    pvalues = np.asarray(list(pvalues), dtype=float)
    out = np.full_like(pvalues, np.nan, dtype=float)
    valid = np.where(~np.isnan(pvalues))[0]
    if len(valid) == 0:
        return out.tolist()
    order = valid[np.argsort(pvalues[valid])]
    ranked = pvalues[order]
    q = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out[order] = np.minimum(q, 1.0)
    return out.tolist()


def write_status(rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path = TABLES / "enhanced_analysis_status.tsv"
    new = pd.DataFrame(rows)
    if path.exists():
        old = pd.read_csv(path, sep="\t")
        new = pd.concat([old, new], ignore_index=True, sort=False)
    new.to_csv(path, sep="\t", index=False)


def check_manifest() -> pd.DataFrame:
    manifest = ROOT / "CGZ_public_data_manifest.tsv"
    if not manifest.exists():
        logging.warning("Manifest not found: %s", manifest)
        return pd.DataFrame()
    df = pd.read_csv(manifest, sep="\t")
    rows = []
    for _, row in df.iterrows():
        local_path_text = str(row["LocalPath"]).replace("\\", "/")
        local_path = Path(local_path_text)
        if not local_path.is_absolute():
            local_path = ROOT / local_path
        exists = local_path.exists()
        actual = local_path.stat().st_size if exists else 0
        expected = int(row.get("ExpectedBytes", 0))
        rows.append(
            {
                "dataset": row.get("Dataset"),
                "group": row.get("Group"),
                "file": local_path.name,
                "exists": exists,
                "expected_bytes": expected,
                "actual_bytes": actual,
                "size_matches_expected": bool(expected == 0 or actual == expected),
                "path": str(local_path.relative_to(ROOT)) if local_path.exists() else str(local_path),
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(TABLES / "download_manifest_check.tsv", sep="\t", index=False)
    logging.info("Manifest check: %d files, %d complete", len(out), int(out["size_matches_expected"].sum()))
    return out


def parse_geo_series_matrix(path: Path) -> pd.DataFrame:
    sample_fields: dict[str, list[str]] = {}
    char_fields: dict[str, list[str]] = {}
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as handle:
        reader = csv.reader(handle, delimiter="\t")
        for row in reader:
            if not row:
                continue
            key = clean_key(row[0])
            if key == "!series_matrix_table_begin":
                break
            if not key.startswith("!Sample_"):
                continue
            values = [clean_key(v) for v in row[1:]]
            key_clean = key[1:]
            if key_clean == "Sample_characteristics_ch1":
                for idx, value in enumerate(values):
                    if ":" in value:
                        field, val = value.split(":", 1)
                        field = normalize_column_name(field)
                        val = clean_key(val)
                    else:
                        field = "characteristics"
                        val = value
                    char_fields.setdefault(field, [])
                    while len(char_fields[field]) < idx:
                        char_fields[field].append("")
                    if len(char_fields[field]) == idx:
                        char_fields[field].append(val)
                    else:
                        prev = char_fields[field][idx]
                        char_fields[field][idx] = f"{prev}; {val}" if prev else val
            else:
                sample_fields[key_clean] = values

    n = max([len(v) for v in sample_fields.values()] + [len(v) for v in char_fields.values()])
    df = pd.DataFrame({"series_index": np.arange(n)})
    for key, vals in sample_fields.items():
        vals = vals + [""] * (n - len(vals))
        df[normalize_column_name(key.replace("Sample_", ""))] = vals
    for key, vals in char_fields.items():
        vals = vals + [""] * (n - len(vals))
        if key in df.columns:
            key = f"characteristic_{key}"
        df[key] = vals
    return df


def derive_gse126_sample_name(title: str) -> str:
    return str(title).replace("RNA-seq_", "").strip()


def derive_gse135_sample_name(title: str) -> str:
    return re.sub(r"\s+", "", str(title).strip())


def extract_luc_id(value: object) -> str:
    text = str(value)
    match = re.search(r"luc[_-]?(\d+)|luc(\d+)", text, flags=re.IGNORECASE)
    if not match:
        return ""
    return next(g for g in match.groups() if g)


def load_hgnc_mapping() -> tuple[dict[str, str], str]:
    REFERENCE.mkdir(parents=True, exist_ok=True)
    target = REFERENCE / "hgnc_complete_set.txt"
    if not target.exists():
        logging.info("No cached HGNC mapping found; using built-in marker Ensembl mapping")
        return dict(FALLBACK_ENSEMBL_TO_SYMBOL), "fallback_marker_set"
    status = "cached"
    try:
        df = pd.read_csv(target, sep="\t", dtype=str, low_memory=False)
        mapping = dict(FALLBACK_ENSEMBL_TO_SYMBOL)
        if "ensembl_gene_id" in df.columns and "symbol" in df.columns:
            tmp = df[["ensembl_gene_id", "symbol"]].dropna()
            tmp["ensembl_gene_id"] = tmp["ensembl_gene_id"].map(strip_ensembl_version)
            tmp["symbol"] = tmp["symbol"].map(normalize_gene_symbol)
            tmp = tmp[(tmp["ensembl_gene_id"] != "") & (tmp["symbol"] != "")]
            mapping.update(dict(zip(tmp["ensembl_gene_id"], tmp["symbol"])))
        return mapping, status
    except Exception as exc:
        logging.warning("HGNC mapping parse failed, using fallback marker mapping: %s", exc)
        return dict(FALLBACK_ENSEMBL_TO_SYMBOL), "fallback_only"


def collapse_expression_by_symbol(df: pd.DataFrame) -> pd.DataFrame:
    df.index = [normalize_gene_symbol(x) for x in df.index]
    df = df.loc[[idx != "" for idx in df.index]]
    df = df.apply(pd.to_numeric, errors="coerce")
    return df.groupby(df.index).mean()


def normalize_counts_to_log_cpm(df: pd.DataFrame) -> pd.DataFrame:
    df = df.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    lib_size = df.sum(axis=0).replace(0, np.nan)
    cpm = df.div(lib_size, axis=1) * 1_000_000.0
    return np.log2(cpm + 1.0)


def read_expression_table(
    path: Path,
    first_col: str | None,
    unit: str,
    ensembl_map: dict[str, str] | None = None,
) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", compression="infer", low_memory=False)
    if first_col is None:
        first_col = df.columns[0]
    genes = df[first_col].astype(str)
    values = df.drop(columns=[first_col])
    if ensembl_map is not None:
        mapped = genes.map(lambda x: ensembl_map.get(strip_ensembl_version(x), ""))
        values.index = mapped
    else:
        values.index = genes
    values = collapse_expression_by_symbol(values)
    if unit == "counts":
        values = normalize_counts_to_log_cpm(values)
    elif unit == "tpm":
        values = values.apply(pd.to_numeric, errors="coerce").fillna(0.0)
        values = np.log2(values + 1.0)
    elif unit == "log2tpm":
        values = values.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    else:
        raise ValueError(f"Unknown expression unit: {unit}")
    return values


def module_scores(expr: pd.DataFrame, dataset: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    coverage_rows = []
    for module, genes in SIGNATURES.items():
        available = [g for g in genes if g in expr.index]
        coverage_rows.append(
            {
                "dataset": dataset,
                "module": module,
                "available_genes": len(available),
                "total_genes": len(genes),
                "available_gene_list": ",".join(available),
                "missing_gene_list": ",".join([g for g in genes if g not in expr.index]),
            }
        )
        if available:
            score = expr.loc[available].mean(axis=0)
        else:
            score = pd.Series(np.nan, index=expr.columns)
        rows.append(score.rename(module))
    scores = pd.concat(rows, axis=1)
    if "CD274" in expr.index:
        scores["CD274_expr"] = expr.loc["CD274"]
    else:
        scores["CD274_expr"] = np.nan
    scores.index.name = "sample"
    scores = scores.reset_index()
    scores.insert(0, "dataset", dataset)
    return scores, pd.DataFrame(coverage_rows)


def load_gse126044() -> ExpressionDataset:
    meta = parse_geo_series_matrix(GEO / "GSE126044" / "GSE126044_series_matrix.txt.gz")
    meta["sample"] = meta["title"].map(derive_gse126_sample_name)
    meta["endpoint_text"] = meta["patient_response"].str.lower()
    meta["benefit"] = meta["endpoint_text"].map({"responder": 1, "non-responder": 0})
    expr = read_expression_table(
        GEO / "GSE126044" / "GSE126044_counts.txt.gz",
        first_col=None,
        unit="counts",
    )
    meta = meta[meta["sample"].isin(expr.columns)].copy()
    return ExpressionDataset("GSE126044", expr.loc[:, meta["sample"]], meta, "anti-PD-1 response", "log2CPM")


def load_gse135222(ensembl_map: dict[str, str]) -> ExpressionDataset:
    meta = parse_geo_series_matrix(GEO / "GSE135222" / "GSE135222_series_matrix.txt.gz")
    meta["sample"] = meta["title"].map(derive_gse135_sample_name)
    meta["pfs_event"] = pd.to_numeric(meta.get("progression_free_survival_pfs"), errors="coerce")
    meta["pfs_time"] = pd.to_numeric(meta.get("pfs_time"), errors="coerce")
    expr = read_expression_table(
        GEO / "GSE135222" / "GSE135222_GEO_RNA-seq_omicslab_exp.tsv.gz",
        first_col="gene_id",
        unit="tpm",
        ensembl_map=ensembl_map,
    )
    meta = meta[meta["sample"].isin(expr.columns)].copy()
    return ExpressionDataset("GSE135222", expr.loc[:, meta["sample"]], meta, "PFS", "log2TPM")


def load_gse207422_bulk() -> ExpressionDataset:
    meta = pd.read_excel(GEO / "GSE207422" / "GSE207422_NSCLC_bulk_RNAseq_metadata.xlsx")
    meta.columns = [normalize_column_name(c) for c in meta.columns]
    meta = meta.rename(columns={"sample": "sample"})
    meta["endpoint_text"] = meta["pathologic_response"].astype(str)
    meta["benefit"] = meta["endpoint_text"].str.contains("MPR", case=False, na=False).astype(float)
    meta.loc[meta["endpoint_text"].str.contains("NMPR", case=False, na=False), "benefit"] = 0.0
    expr = read_expression_table(
        GEO / "GSE207422" / "GSE207422_NSCLC_bulk_RNAseq_log2TPM.txt.gz",
        first_col="Gene",
        unit="log2tpm",
    )
    meta = meta[meta["sample"].isin(expr.columns)].copy()
    return ExpressionDataset("GSE207422_bulk", expr.loc[:, meta["sample"]], meta, "MPR", "log2TPM")


def load_gse274975(ensembl_map: dict[str, str]) -> ExpressionDataset:
    meta = parse_geo_series_matrix(GEO / "GSE274975" / "GSE274975_series_matrix.txt.gz")
    meta["luc_id"] = meta["title"].map(extract_luc_id)
    meta["histology"] = meta.get("tissue", meta.get("source_name_ch1", "")).astype(str)
    expr = read_expression_table(
        GEO / "GSE274975" / "GSE274975_raw_counts.tsv.gz",
        first_col=None,
        unit="counts",
        ensembl_map=ensembl_map,
    )
    col_to_lucid = {col: extract_luc_id(col) for col in expr.columns}
    expr_meta = pd.DataFrame({"sample": list(expr.columns), "luc_id": list(col_to_lucid.values())})
    meta = expr_meta.merge(meta, on="luc_id", how="left", suffixes=("", "_geo"))
    meta["sample"] = meta["sample"].astype(str)
    return ExpressionDataset("GSE274975", expr.loc[:, meta["sample"]], meta, "projection_only", "log2CPM")


def build_bulk_scores(datasets: list[ExpressionDataset]) -> tuple[pd.DataFrame, pd.DataFrame]:
    all_scores = []
    all_coverage = []
    metadata_tables = []
    for ds in datasets:
        logging.info("Scoring bulk dataset %s with %d genes x %d samples", ds.dataset, *ds.expression.shape)
        scores, coverage = module_scores(ds.expression, ds.dataset)
        meta_cols = [c for c in ds.metadata.columns if c not in scores.columns or c == "sample"]
        meta = ds.metadata[meta_cols].copy()
        meta.insert(0, "dataset", ds.dataset)
        merged = scores.merge(meta, on=["dataset", "sample"], how="left")
        merged["endpoint"] = ds.endpoint
        merged["expression_unit"] = ds.unit
        all_scores.append(merged)
        all_coverage.append(coverage)
        ds.metadata.to_csv(TABLES / f"{ds.dataset}_metadata.tsv", sep="\t", index=False)
        metadata_tables.append({"dataset": ds.dataset, "samples": len(ds.metadata), "endpoint": ds.endpoint})
    bulk_scores = pd.concat(all_scores, ignore_index=True, sort=False)
    coverage = pd.concat(all_coverage, ignore_index=True)
    bulk_scores.to_csv(TABLES / "bulk_signature_scores.tsv", sep="\t", index=False)
    coverage.to_csv(TABLES / "signature_gene_coverage.tsv", sep="\t", index=False)
    pd.DataFrame(metadata_tables).to_csv(TABLES / "dataset_summary.tsv", sep="\t", index=False)
    return bulk_scores, coverage


def zscore_within_dataset(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    out = df.copy()
    for feature in features:
        zname = f"z_{feature}"
        out[zname] = np.nan
        for dataset, idx in out.groupby("dataset").groups.items():
            values = pd.to_numeric(out.loc[idx, feature], errors="coerce")
            mean = values.mean()
            sd = values.std(ddof=0)
            if np.isnan(sd) or sd == 0:
                out.loc[idx, zname] = 0.0
            else:
                out.loc[idx, zname] = (values - mean) / sd
    return out


def make_models(random_state: int = 1) -> dict[str, Pipeline]:
    models: dict[str, Pipeline] = {
        "ElasticNet_logistic": Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        penalty="elasticnet",
                        solver="saga",
                        l1_ratio=0.5,
                        C=0.5,
                        class_weight="balanced",
                        max_iter=10000,
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        "Ridge_logistic": Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        penalty="l2",
                        C=1.0,
                        class_weight="balanced",
                        max_iter=5000,
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        "RandomForest": Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=500,
                        min_samples_leaf=2,
                        class_weight="balanced_subsample",
                        random_state=random_state,
                    ),
                ),
            ]
        ),
    }
    if SVC is not None:
        models["RBF_SVM"] = Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
                ("model", SVC(probability=True, class_weight="balanced", random_state=random_state)),
            ]
        )
    if XGBClassifier is not None:
        models["XGBoost"] = Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                (
                    "model",
                    XGBClassifier(
                        n_estimators=120,
                        max_depth=2,
                        learning_rate=0.05,
                        subsample=0.9,
                        colsample_bytree=0.9,
                        eval_metric="logloss",
                        random_state=random_state,
                        n_jobs=1,
                    ),
                ),
            ]
        )
    if LGBMClassifier is not None:
        models["LightGBM"] = Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                (
                    "model",
                    LGBMClassifier(
                        n_estimators=120,
                        learning_rate=0.05,
                        num_leaves=7,
                        min_child_samples=2,
                        subsample=0.9,
                        colsample_bytree=0.9,
                        class_weight="balanced",
                        random_state=random_state,
                        n_jobs=1,
                        verbose=-1,
                    ),
                ),
            ]
        )
    return models


def evaluate_response_models(bulk_scores: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, Pipeline, list[str]]:
    features = list(SIGNATURES.keys()) + ["CD274_expr"]
    scored = zscore_within_dataset(bulk_scores, features)
    feature_cols = [f"z_{f}" for f in features]
    labeled = scored[scored["dataset"].isin(["GSE126044", "GSE207422_bulk"])].copy()
    labeled = labeled[pd.to_numeric(labeled["benefit"], errors="coerce").notna()].copy()
    labeled["benefit"] = labeled["benefit"].astype(int)
    labeled.to_csv(TABLES / "labeled_bulk_model_matrix.tsv", sep="\t", index=False)

    X = labeled[feature_cols]
    y = labeled["benefit"].astype(int).to_numpy()
    min_class = int(pd.Series(y).value_counts().min())
    n_splits = max(2, min(5, min_class))
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=1)
    models = make_models()
    perf_rows = []
    prediction_frames = []

    for name, model in models.items():
        logging.info("Cross-validating model: %s", name)
        try:
            prob = cross_val_predict(model, X, y, cv=cv, method="predict_proba")[:, 1]
            perf_rows.append(
                {
                    "evaluation": f"{n_splits}-fold CV pooled labeled cohorts",
                    "model": name,
                    "train_dataset": "GSE126044+GSE207422_bulk",
                    "test_dataset": "GSE126044+GSE207422_bulk",
                    "n": len(y),
                    "positives": int(y.sum()),
                    "roc_auc": safe_auc(y, prob),
                    "pr_auc": safe_average_precision(y, prob),
                    "brier": float(brier_score_loss(y, prob)),
                }
            )
            pred = labeled[["dataset", "sample", "endpoint", "endpoint_text", "benefit"]].copy()
            pred["model"] = name
            pred["prediction"] = prob
            prediction_frames.append(pred)
        except Exception as exc:
            logging.warning("Model CV failed for %s: %s", name, exc)

    for train_ds, test_ds in [("GSE207422_bulk", "GSE126044"), ("GSE126044", "GSE207422_bulk")]:
        train = labeled[labeled["dataset"] == train_ds]
        test = labeled[labeled["dataset"] == test_ds]
        if train["benefit"].nunique() < 2 or test["benefit"].nunique() < 2:
            continue
        for name, model in models.items():
            try:
                fitted = clone(model).fit(train[feature_cols], train["benefit"].astype(int))
                prob = fitted.predict_proba(test[feature_cols])[:, 1]
                perf_rows.append(
                    {
                        "evaluation": "cross-cohort validation",
                        "model": name,
                        "train_dataset": train_ds,
                        "test_dataset": test_ds,
                        "n": len(test),
                        "positives": int(test["benefit"].sum()),
                        "roc_auc": safe_auc(test["benefit"], prob),
                        "pr_auc": safe_average_precision(test["benefit"], prob),
                        "brier": float(brier_score_loss(test["benefit"], prob)),
                    }
                )
            except Exception as exc:
                logging.warning("Cross-cohort validation failed for %s %s->%s: %s", name, train_ds, test_ds, exc)

    final_model = clone(models["ElasticNet_logistic"]).fit(X, y)
    final_prob = final_model.predict_proba(X)[:, 1]
    final_pred = labeled[["dataset", "sample", "endpoint", "endpoint_text", "benefit"]].copy()
    final_pred["model"] = "ElasticNet_logistic_final_fit"
    final_pred["prediction"] = final_prob
    prediction_frames.append(final_pred)

    performance = pd.DataFrame(perf_rows)
    predictions = pd.concat(prediction_frames, ignore_index=True, sort=False) if prediction_frames else pd.DataFrame()
    performance.to_csv(TABLES / "model_performance.tsv", sep="\t", index=False)
    predictions.to_csv(TABLES / "model_predictions.tsv", sep="\t", index=False)

    save_model_details(final_model, feature_cols, labeled, final_prob)
    plot_model_outputs(performance, predictions, final_model, feature_cols, labeled)
    score_all_cohorts(scored, final_model, feature_cols)
    return performance, predictions, final_model, feature_cols


def save_model_details(model: Pipeline, feature_cols: list[str], labeled: pd.DataFrame, final_prob: np.ndarray) -> None:
    estimator = model.named_steps["model"]
    coef = np.ravel(estimator.coef_)
    rows = []
    for feature, value in zip(feature_cols, coef):
        rows.append({"feature": feature.replace("z_", ""), "coefficient": float(value)})
    coef_df = pd.DataFrame(rows).sort_values("coefficient", key=lambda s: s.abs(), ascending=False)
    coef_df.to_csv(TABLES / "elasticnet_coefficients.tsv", sep="\t", index=False)

    imputed = model.named_steps["impute"].transform(labeled[feature_cols])
    scaled = model.named_steps["scale"].transform(imputed)
    contrib = pd.DataFrame(scaled * coef, columns=[c.replace("z_", "") for c in feature_cols])
    contrib.insert(0, "sample", labeled["sample"].to_numpy())
    contrib.insert(0, "dataset", labeled["dataset"].to_numpy())
    contrib["linear_logit"] = np.ravel(estimator.intercept_) + contrib[[c.replace("z_", "") for c in feature_cols]].sum(axis=1)
    contrib["prediction"] = final_prob
    contrib.to_csv(TABLES / "elasticnet_linear_shap_style_contributions.tsv", sep="\t", index=False)

    assoc_rows = []
    for feature in feature_cols:
        values = labeled[feature].astype(float)
        pos = values[labeled["benefit"] == 1]
        neg = values[labeled["benefit"] == 0]
        if len(pos) > 1 and len(neg) > 1:
            stat, p = stats.mannwhitneyu(pos, neg, alternative="two-sided")
            effect = pos.mean() - neg.mean()
        else:
            stat, p, effect = np.nan, np.nan, np.nan
        assoc_rows.append(
            {
                "feature": feature.replace("z_", ""),
                "positive_mean_z": pos.mean(),
                "negative_mean_z": neg.mean(),
                "mean_difference": effect,
                "mannwhitney_u": stat,
                "p_value": p,
            }
        )
    assoc = pd.DataFrame(assoc_rows)
    assoc["fdr"] = benjamini_hochberg(assoc["p_value"])
    assoc.to_csv(TABLES / "feature_response_associations.tsv", sep="\t", index=False)

    with (MODELS / "scTIME_AI_elasticnet_pipeline.pkl").open("wb") as handle:
        pickle.dump(model, handle)
    metadata = {
        "model_name": "scTIME-AI ElasticNet logistic regression",
        "model_file": "scTIME_AI_elasticnet_pipeline.pkl",
        "input_columns": feature_cols,
        "reported_features": [c.replace("z_", "") for c in feature_cols],
        "training_datasets": sorted(labeled["dataset"].dropna().unique().tolist()),
        "training_samples": int(len(labeled)),
        "positive_samples": int(labeled["benefit"].sum()),
        "generated": datetime.now().isoformat(timespec="seconds"),
        "python": sys.executable,
    }
    (MODELS / "scTIME_AI_elasticnet_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


def plot_model_outputs(
    performance: pd.DataFrame,
    predictions: pd.DataFrame,
    model: Pipeline,
    feature_cols: list[str],
    labeled: pd.DataFrame,
) -> None:
    sns.set_theme(style="whitegrid", font_scale=0.9)
    cv_pred = predictions[predictions["model"] == "ElasticNet_logistic"].copy()
    if not cv_pred.empty and cv_pred["benefit"].nunique() == 2:
        y = cv_pred["benefit"].astype(int)
        prob = cv_pred["prediction"].astype(float)
        fpr, tpr, _ = roc_curve(y, prob)
        precision, recall, _ = precision_recall_curve(y, prob)
        fig, axes = plt.subplots(1, 2, figsize=(9, 4))
        axes[0].plot(fpr, tpr, label=f"AUC={safe_auc(y, prob):.3f}", color="#2f6f9f")
        axes[0].plot([0, 1], [0, 1], "--", color="0.6", linewidth=1)
        axes[0].set_xlabel("False positive rate")
        axes[0].set_ylabel("True positive rate")
        axes[0].set_title("ElasticNet CV ROC")
        axes[0].legend(frameon=False)
        axes[1].plot(recall, precision, label=f"AP={safe_average_precision(y, prob):.3f}", color="#b4473d")
        axes[1].set_xlabel("Recall")
        axes[1].set_ylabel("Precision")
        axes[1].set_title("ElasticNet CV PR")
        axes[1].legend(frameon=False)
        fig.tight_layout()
        save_quicklook(fig, "model_roc_pr_curves.png", dpi=220)
        plt.close(fig)

        calibration_and_decision_tables(cv_pred)

    coef = pd.read_csv(TABLES / "elasticnet_coefficients.tsv", sep="\t")
    if not coef.empty:
        plot_df = coef.head(12).copy()
        fig, ax = plt.subplots(figsize=(7, max(4, 0.35 * len(plot_df))))
        colors = np.where(plot_df["coefficient"] >= 0, "#2f6f9f", "#b4473d")
        ax.barh(plot_df["feature"], plot_df["coefficient"], color=colors)
        ax.axvline(0, color="0.3", linewidth=1)
        ax.invert_yaxis()
        ax.set_xlabel("ElasticNet coefficient")
        ax.set_ylabel("")
        ax.set_title("scTIME-AI feature weights")
        fig.tight_layout()
        save_quicklook(fig, "elasticnet_feature_weights.png", dpi=220)
        plt.close(fig)

    features = [c for c in feature_cols if c in labeled.columns]
    if features:
        heat = labeled.set_index(["dataset", "sample"])[features].copy()
        heat.columns = [c.replace("z_", "") for c in heat.columns]
        fig, ax = plt.subplots(figsize=(8, max(5, 0.18 * len(heat))))
        sns.heatmap(heat, cmap="vlag", center=0, ax=ax, cbar_kws={"label": "Within-cohort z score"})
        ax.set_title("Bulk immune module projection")
        fig.tight_layout()
        save_quicklook(fig, "bulk_module_score_heatmap.png", dpi=220)
        plt.close(fig)

    assoc_path = TABLES / "feature_response_associations.tsv"
    if assoc_path.exists():
        assoc = pd.read_csv(assoc_path, sep="\t").sort_values("p_value").head(6)
        long = labeled.melt(
            id_vars=["dataset", "sample", "benefit"],
            value_vars=[f"z_{x}" for x in assoc["feature"].tolist() if f"z_{x}" in labeled.columns],
            var_name="feature",
            value_name="z_score",
        )
        if not long.empty:
            long["feature"] = long["feature"].str.replace("z_", "", regex=False)
            long["benefit_group"] = np.where(long["benefit"] == 1, "Benefit", "No benefit")
            fig, ax = plt.subplots(figsize=(10, 4.5))
            sns.boxplot(data=long, x="feature", y="z_score", hue="benefit_group", ax=ax, fliersize=2)
            sns.stripplot(data=long, x="feature", y="z_score", hue="benefit_group", dodge=True, ax=ax, color="0.2", size=2, alpha=0.45)
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(handles[:2], labels[:2], frameon=False, title="")
            ax.set_xlabel("")
            ax.set_ylabel("Within-cohort z score")
            ax.set_title("Top response-associated modules")
            ax.tick_params(axis="x", rotation=30)
            fig.tight_layout()
            save_quicklook(fig, "response_module_boxplots.png", dpi=220)
            plt.close(fig)


def calibration_and_decision_tables(cv_pred: pd.DataFrame) -> None:
    df = cv_pred.copy()
    df["benefit"] = df["benefit"].astype(int)
    try:
        df["bin"] = pd.qcut(df["prediction"], q=min(5, len(df)), duplicates="drop")
        cal = df.groupby("bin", observed=True).agg(
            n=("benefit", "size"),
            mean_prediction=("prediction", "mean"),
            observed_rate=("benefit", "mean"),
        ).reset_index()
        cal["bin"] = cal["bin"].astype(str)
    except Exception:
        cal = pd.DataFrame()
    cal.to_csv(TABLES / "elasticnet_calibration.tsv", sep="\t", index=False)

    rows = []
    y = df["benefit"].to_numpy()
    p = df["prediction"].to_numpy()
    n = len(df)
    prevalence = y.mean()
    for threshold in np.linspace(0.05, 0.95, 19):
        pred_pos = p >= threshold
        tp = int(((y == 1) & pred_pos).sum())
        fp = int(((y == 0) & pred_pos).sum())
        net_benefit = tp / n - fp / n * (threshold / (1 - threshold))
        treat_all = prevalence - (1 - prevalence) * (threshold / (1 - threshold))
        rows.append(
            {
                "threshold": threshold,
                "net_benefit_model": net_benefit,
                "net_benefit_treat_all": treat_all,
                "net_benefit_treat_none": 0.0,
            }
        )
    dca = pd.DataFrame(rows)
    dca.to_csv(TABLES / "elasticnet_decision_curve.tsv", sep="\t", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    if not cal.empty:
        axes[0].plot(cal["mean_prediction"], cal["observed_rate"], marker="o", color="#2f6f9f")
        axes[0].plot([0, 1], [0, 1], "--", color="0.6", linewidth=1)
    axes[0].set_xlabel("Mean predicted probability")
    axes[0].set_ylabel("Observed benefit rate")
    axes[0].set_title("Calibration")
    axes[1].plot(dca["threshold"], dca["net_benefit_model"], label="Model", color="#2f6f9f")
    axes[1].plot(dca["threshold"], dca["net_benefit_treat_all"], label="Treat all", color="0.5")
    axes[1].plot(dca["threshold"], dca["net_benefit_treat_none"], label="Treat none", color="0.2")
    axes[1].set_xlabel("Threshold probability")
    axes[1].set_ylabel("Net benefit")
    axes[1].set_title("Decision curve")
    axes[1].legend(frameon=False)
    fig.tight_layout()
    save_quicklook(fig, "calibration_decision_curve.png", dpi=220)
    plt.close(fig)


def score_all_cohorts(scored: pd.DataFrame, model: Pipeline, feature_cols: list[str]) -> pd.DataFrame:
    all_df = scored.copy()
    all_df["scTIME_AI_score"] = model.predict_proba(all_df[feature_cols])[:, 1]
    all_df.to_csv(TABLES / "scTIME_AI_scores_all_bulk_cohorts.tsv", sep="\t", index=False)
    return all_df


def km_curve(time: np.ndarray, event: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(time)
    time = time[order]
    event = event[order]
    unique_times = np.unique(time[event == 1])
    surv = [1.0]
    times = [0.0]
    current = 1.0
    for t in unique_times:
        at_risk = np.sum(time >= t)
        events = np.sum((time == t) & (event == 1))
        if at_risk > 0:
            current *= 1.0 - events / at_risk
        times.append(t)
        surv.append(current)
    return np.asarray(times), np.asarray(surv)


def logrank_test(time: np.ndarray, event: np.ndarray, group: np.ndarray) -> tuple[float, float]:
    time = np.asarray(time, dtype=float)
    event = np.asarray(event, dtype=int)
    group = np.asarray(group, dtype=int)
    event_times = np.unique(time[event == 1])
    observed = 0.0
    expected = 0.0
    variance = 0.0
    for t in event_times:
        at_risk_1 = np.sum((time >= t) & (group == 1))
        at_risk_0 = np.sum((time >= t) & (group == 0))
        events_1 = np.sum((time == t) & (event == 1) & (group == 1))
        events_total = np.sum((time == t) & (event == 1))
        n = at_risk_1 + at_risk_0
        if n <= 1:
            continue
        expected_1 = events_total * at_risk_1 / n
        var_1 = at_risk_1 * at_risk_0 * events_total * (n - events_total) / (n * n * (n - 1))
        observed += events_1
        expected += expected_1
        variance += var_1
    if variance <= 0:
        return np.nan, np.nan
    chisq = (observed - expected) ** 2 / variance
    return float(chisq), float(stats.chi2.sf(chisq, 1))


def survival_validation(scored_all: pd.DataFrame) -> pd.DataFrame:
    df = pd.read_csv(TABLES / "scTIME_AI_scores_all_bulk_cohorts.tsv", sep="\t")
    gse = df[df["dataset"] == "GSE135222"].copy()
    gse["pfs_time"] = pd.to_numeric(gse["pfs_time"], errors="coerce")
    gse["pfs_event"] = pd.to_numeric(gse["pfs_event"], errors="coerce")
    gse = gse.dropna(subset=["pfs_time", "pfs_event", "scTIME_AI_score"])
    rows = []
    if not gse.empty:
        median = gse["scTIME_AI_score"].median()
        gse["score_group"] = np.where(gse["scTIME_AI_score"] >= median, 1, 0)
        chisq, p = logrank_test(gse["pfs_time"].to_numpy(), gse["pfs_event"].astype(int).to_numpy(), gse["score_group"].to_numpy())
        rows.append(
            {
                "dataset": "GSE135222",
                "analysis": "median split log-rank",
                "n": len(gse),
                "events": int(gse["pfs_event"].sum()),
                "statistic": chisq,
                "p_value": p,
                "hazard_ratio": np.nan,
                "note": "High score group coded as 1",
            }
        )
        if PHReg is not None and gse["pfs_event"].nunique() > 1:
            try:
                x = gse[["scTIME_AI_score"]].astype(float)
                fit = PHReg(gse["pfs_time"].astype(float), x, status=gse["pfs_event"].astype(int)).fit(disp=0)
                coef = float(fit.params[0])
                rows.append(
                    {
                        "dataset": "GSE135222",
                        "analysis": "univariate Cox PH",
                        "n": len(gse),
                        "events": int(gse["pfs_event"].sum()),
                        "statistic": float(fit.tvalues[0]),
                        "p_value": float(fit.pvalues[0]),
                        "hazard_ratio": float(np.exp(coef)),
                        "note": "Continuous scTIME-AI score",
                    }
                )
            except Exception as exc:
                logging.warning("Cox PH failed for GSE135222: %s", exc)
        plot_km(gse)
    out = pd.DataFrame(rows)
    out.to_csv(TABLES / "gse135222_survival_validation.tsv", sep="\t", index=False)
    return out


def plot_km(gse: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    for label, group_value, color in [("High scTIME-AI", 1, "#2f6f9f"), ("Low scTIME-AI", 0, "#b4473d")]:
        sub = gse[gse["score_group"] == group_value]
        if sub.empty:
            continue
        t, s = km_curve(sub["pfs_time"].to_numpy(), sub["pfs_event"].astype(int).to_numpy())
        ax.step(t, s, where="post", label=f"{label} (n={len(sub)})", color=color)
    ax.set_xlabel("PFS time")
    ax.set_ylabel("Progression-free probability")
    ax.set_title("GSE135222 PFS by scTIME-AI score")
    ax.legend(frameon=False)
    fig.tight_layout()
    save_quicklook(fig, "gse135222_pfs_km.png", dpi=220)
    plt.close(fig)


def classify_sc_state(row: pd.Series) -> str:
    major = str(row.get("major_cell_type", "")).lower()
    sub = str(row.get("sub_cell_type", "")).lower()
    text = f"{major} {sub}"
    if "treg" in text or "foxp3" in text:
        return "Treg"
    if "cd8" in text and any(x in text for x in ["gzmk", "gzmh", "gzmb", "gnly", "nkg7", "tem"]):
        return "Cytotoxic/effector CD8 T"
    if "cd8" in text and any(x in text for x in ["tcf7", "il7r", "ccr7", "tm"]):
        return "Memory/progenitor CD8 T"
    if "cd4" in text:
        return "CD4 T"
    if "nk" in text:
        return "NK"
    if "spp1" in text and ("macro" in text or "myeloid" in text):
        return "SPP1 macrophage"
    if "macro" in text or "monocyte" in text or "myeloid" in text:
        return "Other myeloid"
    if "b cell" in text or "plasma" in text or re.search(r"\bb_", text):
        return "B/plasma"
    if "dc" in text or "dendritic" in text:
        return "Dendritic cell"
    if "mast" in text:
        return "Mast cell"
    return str(row.get("major_cell_type", "Other")) or "Other"


def classify_sc_states_vectorized(meta: pd.DataFrame) -> pd.Series:
    major = meta["major_cell_type"].fillna("").astype(str)
    sub = meta["sub_cell_type"].fillna("").astype(str)
    major_l = major.str.lower()
    sub_l = sub.str.lower()
    text = (major + " " + sub).str.lower()
    state = pd.Series("Other", index=meta.index, dtype=object)

    state.loc[major_l.str.contains("b cell", regex=False) | sub_l.str.contains("^b[nm]_|plasma", regex=True)] = "B/plasma"
    state.loc[major_l.str.contains("myeloid", regex=False) | sub_l.str.contains("mφ|macro|mono|neu_|cdc|dc", regex=True)] = "Other myeloid"
    state.loc[sub_l.str.contains("^cdc|\\bdc", regex=True)] = "Dendritic cell"
    state.loc[sub_l.str.contains("spp1", regex=False) & sub_l.str.contains("mφ|macro", regex=True)] = "SPP1 macrophage"
    state.loc[sub_l.str.contains("mast", regex=False)] = "Mast cell"

    state.loc[sub_l.str.contains("^nk_", regex=True)] = "NK"
    state.loc[sub_l.str.contains("^cd4", regex=True)] = "CD4 T"
    state.loc[sub_l.str.contains("treg|foxp3|ccr8", regex=True)] = "Treg"
    state.loc[
        sub_l.str.contains("^cd8", regex=True)
        & sub_l.str.contains("tcf7|il7r|ccr7|tm_|_tm", regex=True)
    ] = "Memory/progenitor CD8 T"
    state.loc[
        sub_l.str.contains("^cd8", regex=True)
        & sub_l.str.contains("gzmk|gzmh|gzmb|gnly|nkg7|tem|prf|trm|tex|mait|isg15", regex=True)
    ] = "Cytotoxic/effector CD8 T"
    state.loc[sub_l.str.contains("^cd8", regex=True) & (state == "Other")] = "Other CD8 T"
    state.loc[sub_l.str.contains("^t_gdt", regex=True)] = "Gamma-delta T"
    state.loc[sub_l.str.contains("^ilc", regex=True)] = "ILC"
    return state


def single_cell_gse243013() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    logging.info("Reading GSE243013 single-cell metadata")
    usecols = [
        "sampleID",
        "cellID",
        "n_genes",
        "total_counts",
        "pct_counts_mt",
        "major_cell_type",
        "sub_cell_type",
        "gender",
        "age",
        "smoking_history",
        "cancer_type",
        "pre_treatment_staging",
        "anti-PD1_therapy",
        "chemotherapy",
        "targeted_therapy",
        "cycles",
        "pathological_response",
        "pathological_response_rate",
        "radiological_response",
    ]
    meta = pd.read_csv(
        GEO / "GSE243013" / "GSE243013_NSCLC_immune_scRNA_metadata.csv.gz",
        usecols=usecols,
        low_memory=False,
    )
    meta["state_category"] = classify_sc_states_vectorized(meta)
    meta["is_mpr"] = meta["pathological_response"].astype(str).str.fullmatch("MPR", case=False, na=False).astype(int)
    meta["is_non_mpr"] = meta["pathological_response"].astype(str).str.contains("non-MPR", case=False, na=False).astype(int)

    sample_clin = (
        meta.groupby("sampleID")
        .agg(
            n_cells=("cellID", "size"),
            mean_n_genes=("n_genes", "mean"),
            median_total_counts=("total_counts", "median"),
            mean_pct_counts_mt=("pct_counts_mt", "mean"),
            pathological_response=("pathological_response", "first"),
            pathological_response_rate=("pathological_response_rate", "first"),
            radiological_response=("radiological_response", "first"),
            gender=("gender", "first"),
            age=("age", "first"),
            smoking_history=("smoking_history", "first"),
            cancer_type=("cancer_type", "first"),
            pre_treatment_staging=("pre_treatment_staging", "first"),
            anti_PD1_therapy=("anti-PD1_therapy", "first"),
            chemotherapy=("chemotherapy", "first"),
            targeted_therapy=("targeted_therapy", "first"),
            cycles=("cycles", "first"),
        )
        .reset_index()
    )
    sample_clin["benefit"] = sample_clin["pathological_response"].astype(str).str.fullmatch("MPR", case=False, na=False).astype(int)
    sample_clin.to_csv(TABLES / "gse243013_sc_patient_summary.tsv", sep="\t", index=False)

    comp = meta.groupby(["sampleID", "state_category"]).size().rename("n_cells").reset_index()
    totals = meta.groupby("sampleID").size().rename("sample_total").reset_index()
    comp = comp.merge(totals, on="sampleID", how="left")
    comp["fraction"] = comp["n_cells"] / comp["sample_total"]
    comp = comp.merge(sample_clin[["sampleID", "pathological_response", "benefit", "radiological_response"]], on="sampleID", how="left")
    comp.to_csv(TABLES / "gse243013_sc_state_composition.tsv", sep="\t", index=False)

    diff_rows = []
    pivot = comp.pivot_table(index="sampleID", columns="state_category", values="fraction", fill_value=0)
    pivot = pivot.merge(sample_clin[["sampleID", "benefit"]], left_index=True, right_on="sampleID", how="left").set_index("sampleID")
    for state in [c for c in pivot.columns if c != "benefit"]:
        pos = pivot.loc[pivot["benefit"] == 1, state]
        neg = pivot.loc[pivot["benefit"] == 0, state]
        if len(pos) > 2 and len(neg) > 2:
            stat, p = stats.mannwhitneyu(pos, neg, alternative="two-sided")
            diff = pos.mean() - neg.mean()
        else:
            stat, p, diff = np.nan, np.nan, np.nan
        diff_rows.append(
            {
                "state_category": state,
                "mpr_mean_fraction": pos.mean(),
                "non_mpr_mean_fraction": neg.mean(),
                "mean_difference": diff,
                "mannwhitney_u": stat,
                "p_value": p,
            }
        )
    diff = pd.DataFrame(diff_rows)
    diff["fdr"] = benjamini_hochberg(diff["p_value"])
    diff.to_csv(TABLES / "gse243013_sc_state_response_differential.tsv", sep="\t", index=False)

    subtype_comp = meta.groupby(["sampleID", "major_cell_type", "sub_cell_type"]).size().rename("n_cells").reset_index()
    subtype_comp = subtype_comp.merge(totals, on="sampleID", how="left")
    subtype_comp["fraction"] = subtype_comp["n_cells"] / subtype_comp["sample_total"]
    subtype_comp = subtype_comp.merge(sample_clin[["sampleID", "pathological_response", "benefit"]], on="sampleID", how="left")
    subtype_comp.to_csv(TABLES / "gse243013_sc_subtype_composition.tsv", sep="\t", index=False)

    plot_gse243013_sc(comp, diff)
    tcr = tcr_gse243013(sample_clin)
    nmf = nmf_gse243013(sample_clin)
    del meta
    return sample_clin, comp, tcr


def plot_gse243013_sc(comp: pd.DataFrame, diff: pd.DataFrame) -> None:
    top_states = diff.sort_values("p_value").head(8)["state_category"].tolist()
    plot = comp[comp["state_category"].isin(top_states)].copy()
    if not plot.empty:
        plot["benefit_group"] = np.where(plot["benefit"] == 1, "MPR", "non-MPR")
        fig, ax = plt.subplots(figsize=(10, 4.8))
        sns.boxplot(data=plot, x="state_category", y="fraction", hue="benefit_group", ax=ax, fliersize=2)
        sns.stripplot(data=plot, x="state_category", y="fraction", hue="benefit_group", dodge=True, ax=ax, color="0.2", size=2, alpha=0.5)
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles[:2], labels[:2], frameon=False, title="")
        ax.set_xlabel("")
        ax.set_ylabel("Fraction of immune cells")
        ax.set_title("GSE243013 immune-state composition by pathology response")
        ax.tick_params(axis="x", rotation=30)
        fig.tight_layout()
        save_quicklook(fig, "gse243013_state_response_boxplots.png", dpi=220)
        plt.close(fig)

    stack = comp.copy()
    top = stack.groupby("state_category")["n_cells"].sum().sort_values(ascending=False).head(10).index
    stack["state_plot"] = np.where(stack["state_category"].isin(top), stack["state_category"], "Other")
    stack = stack.groupby(["sampleID", "state_plot", "benefit"], as_index=False)["fraction"].sum()
    order = stack.groupby("sampleID")["benefit"].first().sort_values().index.tolist()
    table = stack.pivot_table(index="sampleID", columns="state_plot", values="fraction", fill_value=0).loc[order]
    fig, ax = plt.subplots(figsize=(12, 5))
    bottom = np.zeros(len(table))
    palette = sns.color_palette("tab20", n_colors=len(table.columns))
    for color, col in zip(palette, table.columns):
        ax.bar(table.index, table[col].to_numpy(), bottom=bottom, label=col, color=color)
        bottom += table[col].to_numpy()
    ax.set_ylabel("Fraction")
    ax.set_xlabel("Patient")
    ax.set_title("GSE243013 immune-cell composition")
    ax.tick_params(axis="x", rotation=90)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False, fontsize=8)
    fig.tight_layout()
    save_quicklook(fig, "gse243013_cell_composition_stacked.png", dpi=220)
    plt.close(fig)


def tcr_gse243013(sample_clin: pd.DataFrame) -> pd.DataFrame:
    logging.info("Reading GSE243013 TCR annotation")
    tcr = pd.read_csv(GEO / "GSE243013" / "GSE243013_T_with_TCR_annotation.csv.gz")
    rows = []
    for sample, sub in tcr.groupby("sampleID"):
        counts = sub["clonotype"].value_counts(dropna=True)
        total = counts.sum()
        if total == 0:
            continue
        p = counts / total
        shannon = float(-(p * np.log(p)).sum())
        norm_shannon = shannon / np.log(len(counts)) if len(counts) > 1 else 0.0
        rows.append(
            {
                "sampleID": sample,
                "t_cells_with_tcr": int(total),
                "unique_clonotypes": int(len(counts)),
                "top_clonotype_fraction": float(p.iloc[0]),
                "normalized_shannon": norm_shannon,
                "expanded_fraction": float((sub["expansion"].astype(str).str.lower() != "non-expanded").mean()),
                "mean_clonotype_number": float(pd.to_numeric(sub["clonotype_number"], errors="coerce").mean()),
            }
        )
    summary = pd.DataFrame(rows).merge(
        sample_clin[["sampleID", "pathological_response", "benefit", "radiological_response"]],
        on="sampleID",
        how="left",
    )
    summary.to_csv(TABLES / "gse243013_tcr_clonality.tsv", sep="\t", index=False)

    diff_rows = []
    for metric in ["top_clonotype_fraction", "normalized_shannon", "expanded_fraction", "mean_clonotype_number"]:
        pos = summary.loc[summary["benefit"] == 1, metric].dropna()
        neg = summary.loc[summary["benefit"] == 0, metric].dropna()
        if len(pos) > 2 and len(neg) > 2:
            stat, pval = stats.mannwhitneyu(pos, neg, alternative="two-sided")
        else:
            stat, pval = np.nan, np.nan
        diff_rows.append(
            {
                "metric": metric,
                "mpr_mean": pos.mean(),
                "non_mpr_mean": neg.mean(),
                "mean_difference": pos.mean() - neg.mean(),
                "mannwhitney_u": stat,
                "p_value": pval,
            }
        )
    diff = pd.DataFrame(diff_rows)
    diff["fdr"] = benjamini_hochberg(diff["p_value"])
    diff.to_csv(TABLES / "gse243013_tcr_response_differential.tsv", sep="\t", index=False)

    if not summary.empty:
        plot = summary.melt(
            id_vars=["sampleID", "benefit"],
            value_vars=["top_clonotype_fraction", "normalized_shannon", "expanded_fraction"],
            var_name="metric",
            value_name="value",
        )
        plot["benefit_group"] = np.where(plot["benefit"] == 1, "MPR", "non-MPR")
        fig, ax = plt.subplots(figsize=(8, 4.5))
        sns.boxplot(data=plot, x="metric", y="value", hue="benefit_group", ax=ax, fliersize=2)
        sns.stripplot(data=plot, x="metric", y="value", hue="benefit_group", dodge=True, ax=ax, color="0.2", size=2, alpha=0.5)
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles[:2], labels[:2], frameon=False, title="")
        ax.set_xlabel("")
        ax.set_ylabel("Metric value")
        ax.set_title("GSE243013 TCR clonality metrics")
        ax.tick_params(axis="x", rotation=20)
        fig.tight_layout()
        save_quicklook(fig, "gse243013_tcr_clonality.png", dpi=220)
        plt.close(fig)
    return summary


def nmf_gse243013(sample_clin: pd.DataFrame) -> pd.DataFrame:
    path = GEO / "GSE243013" / "GSE243013_NMF_all_group_5.csv.gz"
    nmf = pd.read_csv(path)
    nmf = nmf.merge(sample_clin[["sampleID", "pathological_response", "benefit"]], on="sampleID", how="left")
    crosstab = pd.crosstab(nmf["group"], nmf["pathological_response"], dropna=False)
    crosstab.to_csv(TABLES / "gse243013_nmf_group_response_crosstab.tsv", sep="\t")
    nmf.to_csv(TABLES / "gse243013_nmf_groups.tsv", sep="\t", index=False)
    return nmf


def gse207422_sc_pseudobulk() -> pd.DataFrame:
    path = GEO / "GSE207422" / "GSE207422_NSCLC_scRNAseq_UMI_matrix.txt.gz"
    meta = pd.read_excel(GEO / "GSE207422" / "GSE207422_NSCLC_scRNAseq_metadata.xlsx")
    meta.columns = [normalize_column_name(c) for c in meta.columns]
    selected = set(KNOWN_MARKERS) | set(LR_GENES)
    logging.info("Streaming selected marker genes from GSE207422 scRNA UMI matrix")
    gene_sample_means: dict[str, np.ndarray] = {}
    sample_names: list[str] = []
    sample_codes: np.ndarray | None = None
    sample_counts: np.ndarray | None = None
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as handle:
        header = handle.readline().rstrip("\n").split("\t")
        cells = header[1:]
        sample_names_per_cell = [c.rsplit("_", 1)[0] for c in cells]
        unique_samples = sorted(set(sample_names_per_cell))
        sample_to_code = {s: i for i, s in enumerate(unique_samples)}
        sample_codes = np.asarray([sample_to_code[s] for s in sample_names_per_cell], dtype=np.int32)
        sample_counts = np.bincount(sample_codes, minlength=len(unique_samples)).astype(float)
        sample_names = unique_samples
        for line_number, line in enumerate(handle, start=2):
            gene, rest = line.split("\t", 1)
            gene = normalize_gene_symbol(gene)
            if gene not in selected:
                continue
            values = np.fromstring(rest, sep="\t", dtype=float)
            if values.size != sample_codes.size:
                logging.warning("Skipping malformed line for %s at line %d", gene, line_number)
                continue
            sums = np.bincount(sample_codes, weights=values, minlength=len(sample_names))
            gene_sample_means[gene] = np.log1p(sums / np.maximum(sample_counts, 1.0))

    if not gene_sample_means:
        logging.warning("No selected genes recovered from GSE207422 scRNA matrix")
        return pd.DataFrame()
    expr = pd.DataFrame(gene_sample_means, index=sample_names).T
    scores, coverage = module_scores(expr, "GSE207422_sc_pseudobulk")
    scores = scores.merge(meta, on="sample", how="left")
    scores["benefit"] = scores["pathologic_response"].astype(str).str.contains("MPR", case=False, na=False).astype(float)
    scores.loc[scores["pathologic_response"].astype(str).str.contains("NMPR", case=False, na=False), "benefit"] = 0.0
    scores.to_csv(TABLES / "gse207422_sc_pseudobulk_signature_scores.tsv", sep="\t", index=False)
    coverage.to_csv(TABLES / "gse207422_sc_pseudobulk_signature_coverage.tsv", sep="\t", index=False)

    plot_cols = ["IFN_response", "Antigen_presentation", "Cytotoxic_CD8", "CXCL10_CXCR3_axis", "CD274_expr"]
    plot_cols = [c for c in plot_cols if c in scores.columns]
    plot = scores.melt(
        id_vars=["sample", "pathologic_response", "benefit"],
        value_vars=plot_cols,
        var_name="module",
        value_name="score",
    ).dropna(subset=["benefit"])
    if not plot.empty:
        plot["benefit_group"] = np.where(plot["benefit"] == 1, "MPR", "non-MPR")
        fig, ax = plt.subplots(figsize=(9, 4.5))
        sns.boxplot(data=plot, x="module", y="score", hue="benefit_group", ax=ax, fliersize=2)
        sns.stripplot(data=plot, x="module", y="score", hue="benefit_group", dodge=True, ax=ax, color="0.2", size=2, alpha=0.5)
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles[:2], labels[:2], frameon=False, title="")
        ax.set_xlabel("")
        ax.set_ylabel("log1p mean UMI")
        ax.set_title("GSE207422 scRNA sample-level marker projection")
        ax.tick_params(axis="x", rotation=25)
        fig.tight_layout()
        save_quicklook(fig, "gse207422_sc_pseudobulk_scores.png", dpi=220)
        plt.close(fig)
    return scores


def gse207422_sample_metadata() -> pd.DataFrame:
    meta = pd.read_excel(GEO / "GSE207422" / "GSE207422_NSCLC_scRNAseq_metadata.xlsx")
    meta.columns = [normalize_column_name(c) for c in meta.columns]
    meta = meta[meta["sample"].astype(str).str.match(r"^BD_immune\d+", na=False)].copy()
    meta["benefit"] = meta["pathologic_response"].astype(str).str.contains("MPR", case=False, na=False).astype(float)
    meta.loc[meta["pathologic_response"].astype(str).str.contains("NMPR", case=False, na=False), "benefit"] = 0.0
    return meta


def select_gse207422_scanpy_genes(path: Path, required_genes: set[str], n_hvg: int = SCANPY_N_HVG) -> tuple[list[str], pd.DataFrame, list[str]]:
    logging.info("Selecting HVGs from GSE207422 scRNA matrix")
    rows = []
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as handle:
        header = handle.readline().rstrip("\n").split("\t")
        cells = header[1:]
        for line_number, line in enumerate(handle, start=2):
            gene, rest = line.split("\t", 1)
            gene = normalize_gene_symbol(gene)
            if not gene:
                continue
            values = np.fromstring(rest, sep="\t", dtype=np.float32)
            if values.size != len(cells):
                logging.warning("Skipping malformed GSE207422 scRNA line %d for %s", line_number, gene)
                continue
            mean = float(values.mean())
            var = float(values.var())
            detected = int(np.count_nonzero(values))
            rows.append(
                {
                    "gene": gene,
                    "mean_count": mean,
                    "variance": var,
                    "detected_cells": detected,
                    "dispersion": var / (mean + 1e-6),
                    "required_marker": gene in required_genes,
                }
            )
    stats_df = pd.DataFrame(rows).drop_duplicates("gene", keep="first")
    hvg = (
        stats_df[(stats_df["detected_cells"] >= 20) & (stats_df["mean_count"] > 0)]
        .sort_values("dispersion", ascending=False)
        .head(n_hvg)["gene"]
        .tolist()
    )
    selected = sorted(set(hvg) | (set(stats_df["gene"]) & required_genes))
    stats_df["selected_for_scanpy"] = stats_df["gene"].isin(selected)
    stats_df.to_csv(TABLES / "gse207422_scanpy_gene_selection.tsv", sep="\t", index=False)
    return selected, stats_df, cells


def build_gse207422_anndata(path: Path, selected_genes: list[str], cells: list[str]) -> object | None:
    if sc is None:
        write_status([{"module": "GSE207422 Scanpy clustering", "status": "skipped", "reason": "scanpy is not importable"}])
        return None
    selected = set(selected_genes)
    rows = []
    row_names = []
    seen = set()
    logging.info("Building sparse AnnData for GSE207422 scRNA clustering with %d selected genes", len(selected_genes))
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as handle:
        header = handle.readline().rstrip("\n").split("\t")
        matrix_cells = header[1:]
        if matrix_cells != cells:
            cells = matrix_cells
        for line_number, line in enumerate(handle, start=2):
            gene, rest = line.split("\t", 1)
            gene = normalize_gene_symbol(gene)
            if gene not in selected or gene in seen:
                continue
            values = np.fromstring(rest, sep="\t", dtype=np.float32)
            if values.size != len(cells):
                logging.warning("Skipping malformed selected GSE207422 scRNA line %d for %s", line_number, gene)
                continue
            rows.append(sparse.csr_matrix(values.reshape(1, -1)))
            row_names.append(gene)
            seen.add(gene)
    if not rows:
        write_status([{"module": "GSE207422 Scanpy clustering", "status": "skipped", "reason": "no selected genes found"}])
        return None
    gene_by_cell = sparse.vstack(rows, format="csr")
    adata = sc.AnnData(X=gene_by_cell.T)
    adata.obs_names = cells
    adata.var_names = row_names
    adata.obs["sample"] = [c.rsplit("_", 1)[0] for c in cells]
    meta = gse207422_sample_metadata().set_index("sample")
    adata.obs = adata.obs.join(meta, on="sample")
    return adata


def annotate_scanpy_clusters(adata: object) -> tuple[pd.DataFrame, pd.DataFrame]:
    score_cols = []
    for state, genes in CELL_STATE_MARKERS.items():
        available = [g for g in genes if g in adata.var_names]
        col = f"score_{state}"
        if available:
            sc.tl.score_genes(adata, available, score_name=col, use_raw=False)
        else:
            adata.obs[col] = 0.0
        score_cols.append(col)
    cluster_scores = adata.obs.groupby("leiden", observed=True)[score_cols].mean()
    cluster_to_state = {}
    for cluster, row in cluster_scores.iterrows():
        best = row.astype(float).idxmax().replace("score_", "")
        cluster_to_state[str(cluster)] = best
    adata.obs["cell_state"] = adata.obs["leiden"].astype(str).map(cluster_to_state).fillna("Unassigned")
    cluster_summary = (
        adata.obs.groupby(["leiden", "cell_state"], observed=True)
        .agg(
            n_cells=("sample", "size"),
            n_samples=("sample", "nunique"),
            mpr_fraction=("benefit", "mean"),
        )
        .reset_index()
    )
    score_table = cluster_scores.reset_index()
    score_table["cell_state"] = score_table["leiden"].astype(str).map(cluster_to_state)
    cluster_summary.to_csv(TABLES / "gse207422_scanpy_cluster_summary.tsv", sep="\t", index=False)
    score_table.to_csv(TABLES / "gse207422_scanpy_cluster_marker_scores.tsv", sep="\t", index=False)
    return cluster_summary, score_table


def run_gse207422_scanpy_clustering() -> object | None:
    path = GEO / "GSE207422" / "GSE207422_NSCLC_scRNAseq_UMI_matrix.txt.gz"
    if sc is None:
        write_status([{"module": "GSE207422 Scanpy clustering", "status": "skipped", "reason": "scanpy is not installed"}])
        return None
    required = set(KNOWN_MARKERS) | set(LR_GENES) | set(CELL_STATE_MARKER_GENES)
    selected, _, cells = select_gse207422_scanpy_genes(path, required)
    adata = build_gse207422_anndata(path, selected, cells)
    if adata is None:
        return None

    logging.info("Running Scanpy normalization, PCA, Leiden clustering, and UMAP for GSE207422")
    adata.layers["counts"] = adata.X.copy()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    adata.raw = adata
    sc.pp.pca(adata, n_comps=30, zero_center=False, random_state=1)
    sc.pp.neighbors(adata, n_neighbors=15, n_pcs=30, random_state=1)
    sc.tl.leiden(adata, resolution=0.6, key_added="leiden", random_state=1)
    sc.tl.umap(adata, random_state=1)
    annotate_scanpy_clusters(adata)

    try:
        sc.tl.rank_genes_groups(adata, groupby="leiden", method="t-test", n_genes=25)
        markers = sc.get.rank_genes_groups_df(adata, group=None)
        markers.to_csv(TABLES / "gse207422_scanpy_cluster_markers.tsv", sep="\t", index=False)
    except Exception as exc:
        logging.warning("GSE207422 Scanpy marker test failed: %s", exc)

    emb = pd.DataFrame(adata.obsm["X_umap"], columns=["UMAP1", "UMAP2"], index=adata.obs_names)
    emb.insert(0, "cell", emb.index)
    emb["sample"] = adata.obs["sample"].to_numpy()
    emb["leiden"] = adata.obs["leiden"].astype(str).to_numpy()
    emb["cell_state"] = adata.obs["cell_state"].astype(str).to_numpy()
    emb["pathologic_response"] = adata.obs["pathologic_response"].astype(str).to_numpy()
    emb.to_csv(TABLES / "gse207422_scanpy_cell_embeddings.tsv", sep="\t", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    plot_df = emb.copy()
    sns.scatterplot(data=plot_df, x="UMAP1", y="UMAP2", hue="leiden", s=2, linewidth=0, ax=axes[0], palette="tab20", legend=False)
    axes[0].set_title("GSE207422 Scanpy Leiden clusters")
    sns.scatterplot(data=plot_df, x="UMAP1", y="UMAP2", hue="cell_state", s=2, linewidth=0, ax=axes[1], palette="tab20", legend=False)
    axes[1].set_title("Marker-based cluster states")
    for ax in axes:
        ax.set_xlabel("UMAP1")
        ax.set_ylabel("UMAP2")
    fig.tight_layout()
    save_quicklook(fig, "gse207422_scanpy_umap.png", dpi=220)
    plt.close(fig)

    for col in adata.obs.columns:
        if pd.api.types.is_object_dtype(adata.obs[col]) or isinstance(adata.obs[col].dtype, pd.CategoricalDtype):
            adata.obs[col] = adata.obs[col].astype(str).replace({"nan": "", "None": ""})
    adata.write_h5ad(MODELS / "gse207422_scanpy_clustered.h5ad", compression="gzip")
    write_status(
        [
            {
                "module": "GSE207422 Scanpy clustering",
                "status": "completed",
                "n_cells": int(adata.n_obs),
                "n_genes": int(adata.n_vars),
                "n_clusters": int(adata.obs["leiden"].nunique()),
            }
        ]
    )
    return adata


def sparse_to_dense_frame(adata: object, genes: list[str]) -> pd.DataFrame:
    genes = [g for g in genes if g in adata.var_names]
    if not genes:
        return pd.DataFrame(index=adata.obs_names)
    x = adata[:, genes].X
    if sparse.issparse(x):
        x = x.toarray()
    return pd.DataFrame(np.asarray(x), index=adata.obs_names, columns=genes)


def gse207422_lr_and_nichenet_like(adata: object | None) -> pd.DataFrame:
    if adata is None:
        write_status([{"module": "CellChat/NicheNet-like LR analysis", "status": "skipped", "reason": "Scanpy AnnData unavailable"}])
        return pd.DataFrame()
    genes = sorted(set(LR_GENES) | {"GZMB", "PRF1", "NKG7", "IFNG", "CD8A", "CD8B"})
    expr = sparse_to_dense_frame(adata, genes)
    if expr.empty:
        write_status([{"module": "CellChat/NicheNet-like LR analysis", "status": "skipped", "reason": "no LR genes found in AnnData"}])
        return pd.DataFrame()
    obs = adata.obs[["sample", "cell_state", "pathologic_response", "benefit"]].copy()
    long = pd.concat([obs.reset_index(drop=True), expr.reset_index(drop=True)], axis=1)
    state_means = long.groupby(["sample", "cell_state"], observed=True)[expr.columns.tolist()].mean().reset_index()
    lymphoid = {"CD8 T", "CD4 T", "Treg", "NK"}
    rows = []
    for sample, sample_df in state_means.groupby("sample", observed=True):
        meta = obs[obs["sample"] == sample].iloc[0]
        for axis in LR_AXES:
            ligand_genes = [g for g in axis["ligands"] if g in sample_df.columns]
            receptor_genes = [g for g in axis["receptors"] if g in sample_df.columns]
            if not ligand_genes or not receptor_genes:
                continue
            sender_df = sample_df[~sample_df["cell_state"].isin(lymphoid)]
            target_df = sample_df[sample_df["cell_state"].isin(lymphoid)]
            if sender_df.empty:
                sender_df = sample_df
            if target_df.empty:
                target_df = sample_df
            ligand_expr = float(sender_df[ligand_genes].mean(axis=1).max())
            receptor_expr = float(target_df[receptor_genes].mean(axis=1).max())
            rows.append(
                {
                    "sample": sample,
                    "axis": axis["axis"],
                    "ligand_genes": ",".join(ligand_genes),
                    "receptor_genes": ",".join(receptor_genes),
                    "sender_max_ligand_expr": ligand_expr,
                    "target_max_receptor_expr": receptor_expr,
                    "interaction_score": ligand_expr * receptor_expr,
                    "pathologic_response": meta.get("pathologic_response"),
                    "benefit": meta.get("benefit"),
                }
            )
    axis_scores = pd.DataFrame(rows)
    axis_scores.to_csv(TABLES / "gse207422_lr_axis_scores.tsv", sep="\t", index=False)

    diff_rows = []
    for axis, sub in axis_scores.groupby("axis", observed=True):
        pos = sub.loc[sub["benefit"] == 1, "interaction_score"].dropna()
        neg = sub.loc[sub["benefit"] == 0, "interaction_score"].dropna()
        if len(pos) > 1 and len(neg) > 1:
            stat, pval = stats.mannwhitneyu(pos, neg, alternative="two-sided")
        else:
            stat, pval = np.nan, np.nan
        diff_rows.append(
            {
                "axis": axis,
                "mpr_mean_score": pos.mean(),
                "non_mpr_mean_score": neg.mean(),
                "mean_difference": pos.mean() - neg.mean(),
                "mannwhitney_u": stat,
                "p_value": pval,
            }
        )
    diff = pd.DataFrame(diff_rows)
    diff["fdr"] = benjamini_hochberg(diff["p_value"])
    diff.to_csv(TABLES / "gse207422_lr_axis_response_differential.tsv", sep="\t", index=False)

    effector_genes = [g for g in ["GZMB", "PRF1", "NKG7", "IFNG", "CD8A", "CD8B"] if g in long.columns]
    ligand_rows = []
    if effector_genes:
        sample_effector = long.groupby("sample", observed=True)[effector_genes].mean().mean(axis=1)
        sample_benefit = obs.groupby("sample", observed=True)["benefit"].first()
        for axis in LR_AXES:
            ligand_genes = [g for g in axis["ligands"] if g in long.columns]
            if not ligand_genes:
                continue
            ligand_score = long.groupby("sample", observed=True)[ligand_genes].mean().mean(axis=1)
            common = ligand_score.index.intersection(sample_effector.index)
            rho, pval = stats.spearmanr(ligand_score.loc[common], sample_effector.loc[common]) if len(common) > 3 else (np.nan, np.nan)
            pos = ligand_score.loc[sample_benefit == 1]
            neg = ligand_score.loc[sample_benefit == 0]
            ligand_rows.append(
                {
                    "axis": axis["axis"],
                    "ligands": ",".join(ligand_genes),
                    "spearman_with_effector_score": rho,
                    "spearman_p_value": pval,
                    "mpr_mean_ligand_score": pos.mean(),
                    "non_mpr_mean_ligand_score": neg.mean(),
                }
            )
    ligand_activity = pd.DataFrame(ligand_rows)
    ligand_activity.to_csv(TABLES / "gse207422_nichenet_like_ligand_activity.tsv", sep="\t", index=False)

    if not axis_scores.empty:
        heat = axis_scores.pivot_table(index="axis", columns="sample", values="interaction_score", fill_value=0)
        fig, ax = plt.subplots(figsize=(8, max(4, 0.35 * len(heat))))
        sns.heatmap(heat, cmap="mako", ax=ax, cbar_kws={"label": "LR interaction score"})
        ax.set_title("GSE207422 CellChat-like ligand-receptor axes")
        fig.tight_layout()
        save_quicklook(fig, "gse207422_lr_axis_heatmap.png", dpi=220)
        plt.close(fig)
    write_status(
        [
            {
                "module": "CellChat/NicheNet-like LR analysis",
                "status": "completed",
                "n_axes": int(axis_scores["axis"].nunique()) if not axis_scores.empty else 0,
                "n_samples": int(axis_scores["sample"].nunique()) if not axis_scores.empty else 0,
            }
        ]
    )
    return axis_scores


def music_style_nnls_deconvolution(datasets: list[ExpressionDataset], adata: object | None) -> pd.DataFrame:
    if adata is None:
        write_status([{"module": "MuSiC/CIBERSORTx-style deconvolution", "status": "skipped", "reason": "Scanpy AnnData unavailable"}])
        return pd.DataFrame()
    states = adata.obs["cell_state"].astype(str)
    valid_states = states.value_counts()
    valid_states = valid_states[valid_states >= 50].index.tolist()
    genes = [g for g in adata.var_names if g in set().union(*[set(ds.expression.index) for ds in datasets])]
    if len(genes) < 20 or not valid_states:
        write_status([{"module": "MuSiC/CIBERSORTx-style deconvolution", "status": "skipped", "reason": "insufficient common genes or cell states"}])
        return pd.DataFrame()
    expr = sparse_to_dense_frame(adata, genes)
    expr["cell_state"] = states.to_numpy()
    ref = expr[expr["cell_state"].isin(valid_states)].groupby("cell_state", observed=True)[genes].mean().T
    ref = ref.loc[(ref.sum(axis=1) > 0)]
    ref.to_csv(TABLES / "music_style_reference_signature_matrix.tsv", sep="\t")
    ref.to_csv(TABLES / "cibersortx_signature_matrix.tsv", sep="\t")

    mixture_frames = []
    frac_rows = []
    for ds in datasets:
        common = [g for g in ref.index if g in ds.expression.index]
        if len(common) < 20:
            continue
        ref_mat = ref.loc[common].to_numpy(dtype=float)
        bulk = ds.expression.loc[common].apply(pd.to_numeric, errors="coerce").fillna(0.0)
        mixture = bulk.copy()
        mixture.columns = [f"{ds.dataset}|{c}" for c in mixture.columns]
        mixture_frames.append(mixture)
        for sample in bulk.columns:
            y = bulk[sample].to_numpy(dtype=float)
            coef, residual = nnls(ref_mat, y)
            frac = coef / coef.sum() if coef.sum() > 0 else np.full_like(coef, np.nan)
            row = {"dataset": ds.dataset, "sample": sample, "nnls_residual": float(residual), "n_genes": len(common)}
            row.update({state: float(value) for state, value in zip(ref.columns, frac)})
            frac_rows.append(row)
    if mixture_frames:
        mixture = pd.concat(mixture_frames, axis=1, join="inner")
        mixture.to_csv(TABLES / "cibersortx_mixture_matrix.tsv", sep="\t")
    fractions = pd.DataFrame(frac_rows)
    fractions.to_csv(TABLES / "music_style_nnls_cell_fractions.tsv", sep="\t", index=False)
    if not fractions.empty:
        plot_cols = [c for c in fractions.columns if c not in {"dataset", "sample", "nnls_residual", "n_genes"}]
        mean_frac = fractions.groupby("dataset", observed=True)[plot_cols].mean()
        fig, ax = plt.subplots(figsize=(8, 4.8))
        bottom = np.zeros(len(mean_frac))
        palette = sns.color_palette("tab20", n_colors=len(plot_cols))
        for color, col in zip(palette, plot_cols):
            ax.bar(mean_frac.index, mean_frac[col], bottom=bottom, label=col, color=color)
            bottom += mean_frac[col].to_numpy()
        ax.set_ylabel("Mean NNLS fraction")
        ax.set_xlabel("")
        ax.set_title("MuSiC-style NNLS deconvolution by cohort")
        ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False, fontsize=8)
        fig.tight_layout()
        save_quicklook(fig, "music_style_nnls_deconvolution.png", dpi=220)
        plt.close(fig)
    write_status(
        [
            {
                "module": "MuSiC/CIBERSORTx-style deconvolution",
                "status": "completed",
                "n_reference_genes": int(ref.shape[0]),
                "n_cell_states": int(ref.shape[1]),
                "n_bulk_samples": int(len(fractions)),
            }
        ]
    )
    return fractions


def tcga_numeric(value: object) -> float:
    text = str(value).strip().replace("'", "")
    if text in {"", "--", "nan", "None"}:
        return np.nan
    return float(pd.to_numeric(text, errors="coerce"))


def tcga_clinical(project: str) -> pd.DataFrame:
    path = DATA / "gdc_clinical" / project / "clinical" / "clinical.tsv"
    if not path.exists():
        return pd.DataFrame()
    clinical = pd.read_csv(path, sep="\t", dtype=str, low_memory=False)
    rows = []
    for patient, sub in clinical.groupby("cases.submitter_id", dropna=True):
        row = sub.iloc[0]
        death = tcga_numeric(row.get("demographic.days_to_death"))
        follow = pd.to_numeric(sub.get("diagnoses.days_to_last_follow_up", pd.Series(dtype=str)).map(tcga_numeric), errors="coerce").max()
        os_event = 1 if str(row.get("demographic.vital_status", "")).lower() == "dead" else 0
        os_time = death if not np.isnan(death) else follow
        rows.append(
            {
                "project": project,
                "patient": patient,
                "os_time": os_time,
                "os_event": os_event,
                "age_at_index": tcga_numeric(row.get("demographic.age_at_index")),
                "sex": row.get("demographic.gender"),
                "pathologic_stage": row.get("diagnoses.ajcc_pathologic_stage"),
            }
        )
    out = pd.DataFrame(rows)
    return out.dropna(subset=["patient"])


def find_tcga_expression_file(project: str) -> Path | None:
    candidates = [
        DATA / "xena" / f"{project}.star_counts.tsv.gz",
        DATA / "xena" / f"{project}.HiSeqV2.gz",
        DATA / "xena" / f"{project.replace('-', '.')}.HiSeqV2.gz",
        DATA / "xena" / f"{project.replace('-', '.')}.star_counts.tsv.gz",
    ]
    for path in candidates:
        if path.exists() and path.stat().st_size > 1024:
            return path
    return None


def read_tcga_expression(path: Path, ensembl_map: dict[str, str]) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", compression="infer", low_memory=False)
    first = df.columns[0]
    genes = df[first].astype(str)
    values = df.drop(columns=[first])
    mapped = genes.map(lambda x: ensembl_map.get(strip_ensembl_version(x), normalize_gene_symbol(x)))
    values.index = mapped
    values = collapse_expression_by_symbol(values)
    values = values.loc[[idx != "" for idx in values.index]]
    values = values.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    if values.max().max() > 1000:
        values = normalize_counts_to_log_cpm(values)
    return values


def tcga_secondary_validation(
    final_model: Pipeline,
    feature_cols: list[str],
    ensembl_map: dict[str, str],
) -> pd.DataFrame:
    rows = []
    score_frames = []
    features = [c.replace("z_", "") for c in feature_cols]
    for project in ["TCGA-LUAD", "TCGA-LUSC"]:
        clinical = tcga_clinical(project)
        clinical.to_csv(TABLES / f"{project}_clinical_survival.tsv", sep="\t", index=False)
        expr_path = find_tcga_expression_file(project)
        if expr_path is None:
            rows.append(
                {
                    "project": project,
                    "status": "skipped",
                    "reason": "TCGA expression matrix not found in data/xena; clinical tables and GDC manifests are present",
                    "clinical_patients": len(clinical),
                }
            )
            continue
        expr = read_tcga_expression(expr_path, ensembl_map)
        expr_patients = set(pd.Series(expr.columns.astype(str)).str.slice(0, 12))
        clinical_patients = set(clinical["patient"].astype(str)) if "patient" in clinical.columns else set()
        overlap_patients = expr_patients & clinical_patients
        if clinical_patients and not overlap_patients:
            rows.append(
                {
                    "project": project,
                    "status": "skipped_sample_id_mismatch",
                    "expression_file": str(expr_path.relative_to(ROOT)),
                    "n_expression_samples": expr.shape[1],
                    "clinical_patients": len(clinical_patients),
                    "clinical_overlap_patients": 0,
                    "reason": "no overlap between expression sample patient IDs and clinical table",
                }
            )
            continue
        scores, _ = module_scores(expr, project)
        scores["patient"] = scores["sample"].astype(str).str.slice(0, 12)
        scored = zscore_within_dataset(scores, features)
        scored["scTIME_AI_score"] = final_model.predict_proba(scored[feature_cols])[:, 1]
        merged = scored.merge(clinical, on="patient", how="left")
        score_frames.append(merged)
        surv = merged.dropna(subset=["os_time", "os_event", "scTIME_AI_score"]).copy()
        if not surv.empty and surv["os_event"].nunique() > 1:
            median = surv["scTIME_AI_score"].median()
            surv["score_group"] = (surv["scTIME_AI_score"] >= median).astype(int)
            chisq, p = logrank_test(surv["os_time"].to_numpy(), surv["os_event"].astype(int).to_numpy(), surv["score_group"].to_numpy())
            hr = np.nan
            cox_p = np.nan
            if PHReg is not None:
                try:
                    fit = PHReg(surv["os_time"].astype(float), surv[["scTIME_AI_score"]].astype(float), status=surv["os_event"].astype(int)).fit(disp=0)
                    hr = float(np.exp(fit.params[0]))
                    cox_p = float(fit.pvalues[0])
                except Exception as exc:
                    logging.warning("TCGA Cox failed for %s: %s", project, exc)
            rows.append(
                {
                    "project": project,
                    "status": "completed",
                    "expression_file": str(expr_path.relative_to(ROOT)),
                    "n_expression_samples": expr.shape[1],
                    "clinical_overlap_patients": len(overlap_patients),
                    "n_samples": len(surv),
                    "events": int(surv["os_event"].sum()),
                    "logrank_p": p,
                    "cox_p": cox_p,
                    "cox_hr": hr,
                }
            )
        else:
            rows.append(
                {
                    "project": project,
                    "status": "completed_no_survival_test",
                    "expression_file": str(expr_path.relative_to(ROOT)),
                    "n_expression_samples": expr.shape[1],
                    "clinical_overlap_patients": len(overlap_patients),
                    "n_samples": len(surv),
                    "reason": "insufficient OS events",
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(TABLES / "tcga_secondary_validation.tsv", sep="\t", index=False)
    if score_frames:
        all_scores = pd.concat(score_frames, ignore_index=True, sort=False)
        all_scores.to_csv(TABLES / "tcga_sctime_ai_scores.tsv", sep="\t", index=False)
        plot_tcga_survival(all_scores)
    write_status(
        [
            {
                "module": "TCGA LUAD/LUSC secondary validation",
                "status": "completed" if any(out["status"].astype(str).str.startswith("completed")) else "skipped",
                "details": "; ".join(out["project"].astype(str) + ":" + out["status"].astype(str)) if not out.empty else "no projects",
            }
        ]
    )
    return out


def plot_tcga_survival(scores: pd.DataFrame) -> None:
    plot = scores.dropna(subset=["os_time", "os_event", "scTIME_AI_score"]).copy()
    if plot.empty:
        return
    fig, axes = plt.subplots(1, plot["project"].nunique(), figsize=(5.2 * plot["project"].nunique(), 4.2), squeeze=False)
    for ax, (project, sub) in zip(axes.ravel(), plot.groupby("project", observed=True)):
        if sub["os_event"].nunique() < 2:
            ax.set_axis_off()
            continue
        sub = sub.copy()
        sub["score_group"] = (sub["scTIME_AI_score"] >= sub["scTIME_AI_score"].median()).astype(int)
        for label, value, color in [("High scTIME-AI", 1, "#2f6f9f"), ("Low scTIME-AI", 0, "#b4473d")]:
            group = sub[sub["score_group"] == value]
            t, s = km_curve(group["os_time"].to_numpy(), group["os_event"].astype(int).to_numpy())
            ax.step(t, s, where="post", label=f"{label} (n={len(group)})", color=color)
        ax.set_title(project)
        ax.set_xlabel("OS time")
        ax.set_ylabel("Overall survival probability")
        ax.legend(frameon=False)
    fig.tight_layout()
    save_quicklook(fig, "tcga_sctime_ai_os_km.png", dpi=220)
    plt.close(fig)


def parse_cptac_gene_symbol(value: object) -> str:
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return ""
    if ":" in text:
        text = text.split(":", 1)[0]
    text = re.sub(r"_(NP|XP|YP)_.*$", "", text)
    text = re.sub(r"[-_:][STY]\\d+[a-z]?$", "", text, flags=re.IGNORECASE)
    return normalize_gene_symbol(text)


def read_cptac_matrix(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", low_memory=False)
    first = df.columns[0]
    genes = df[first].map(parse_cptac_gene_symbol)
    values = df.drop(columns=[first]).replace({"NA": np.nan, "": np.nan, "-": np.nan})
    values.index = genes
    values = values.loc[[idx != "" for idx in values.index]]
    values = values.apply(pd.to_numeric, errors="coerce")
    return values.groupby(values.index).mean()


def read_cptac_clinical(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", dtype=str)
    if len(df) > 0:
        first_row = df.iloc[0].astype(str).str.upper()
        if set(first_row.unique()).intersection({"CAT", "CON", "ORD", "BIN"}):
            df = df.iloc[1:].copy()
    df = df.rename(columns={c: normalize_column_name(c) for c in df.columns})
    sample_col = "sample_id" if "sample_id" in df.columns else df.columns[0]
    df = df.rename(columns={sample_col: "sample"})
    return df


def cptac_stage_group(value: object) -> str:
    text = str(value).strip().upper()
    if text in {"", "NA", "NAN", "--"}:
        return "Unknown"
    if text.startswith(("III", "IV", "3", "4")):
        return "Advanced"
    if text.startswith(("I", "II", "1", "2")):
        return "Early"
    return "Unknown"


def cptac_secondary_validation(final_model: Pipeline, feature_cols: list[str]) -> pd.DataFrame:
    specs = [
        {
            "cohort": "CPTAC-LUAD",
            "omics": "proteome",
            "matrix": DATA / "cptac" / "LinkedOmics" / "HS_CPTAC_LUAD_proteome_ratio_NArm_TUMOR.cct",
            "clinical": DATA / "cptac" / "LinkedOmics" / "HS_CPTAC_LUAD_cli.tsi",
        },
        {
            "cohort": "CPTAC-LUAD",
            "omics": "phosphoproteome",
            "matrix": DATA / "cptac" / "LinkedOmics" / "HS_CPTAC_LUAD_phosphoproteome_ratio_norm_NArm_TUMOR.cct",
            "clinical": DATA / "cptac" / "LinkedOmics" / "HS_CPTAC_LUAD_cli.tsi",
        },
        {
            "cohort": "CPTAC-LSCC",
            "omics": "proteome",
            "matrix": DATA / "cptac" / "LinkedOmics" / "HS_CPTAC_LSCC_2020_proteome_ratio_NArm_TUMOR.cct",
            "clinical": DATA / "cptac" / "LinkedOmics" / "HS_CPTAC_LSCC_2020_clinical_phenotypes_TUMOR.tsi",
        },
        {
            "cohort": "CPTAC-LSCC",
            "omics": "phosphoproteome",
            "matrix": DATA / "cptac" / "LinkedOmics" / "HS_CPTAC_LSCC_2020_phospho_ratio_norm_NArm_TUMOR.cct",
            "clinical": DATA / "cptac" / "LinkedOmics" / "HS_CPTAC_LSCC_2020_clinical_phenotypes_TUMOR.tsi",
        },
    ]
    status_rows = []
    score_frames = []
    coverage_frames = []
    features = [c.replace("z_", "") for c in feature_cols]
    for spec in specs:
        matrix_path = spec["matrix"]
        clinical_path = spec["clinical"]
        if not matrix_path.exists() or not clinical_path.exists():
            status_rows.append(
                {
                    "cohort": spec["cohort"],
                    "omics": spec["omics"],
                    "status": "skipped",
                    "reason": "matrix or clinical file missing",
                    "matrix": str(matrix_path.relative_to(ROOT)),
                    "clinical": str(clinical_path.relative_to(ROOT)),
                }
            )
            continue
        expr = read_cptac_matrix(matrix_path)
        clinical = read_cptac_clinical(clinical_path)
        scores, coverage = module_scores(expr, f"{spec['cohort']}_{spec['omics']}")
        scores["cohort"] = spec["cohort"]
        scores["omics"] = spec["omics"]
        scores = scores.merge(clinical, on="sample", how="left")
        scores["stage_group"] = scores["stage"].map(cptac_stage_group) if "stage" in scores.columns else "Unknown"
        scored = zscore_within_dataset(scores, features)
        scored["scTIME_AI_score"] = final_model.predict_proba(scored[feature_cols])[:, 1]
        score_frames.append(scored)
        coverage["cohort"] = spec["cohort"]
        coverage["omics"] = spec["omics"]
        coverage_frames.append(coverage)
        status_rows.append(
            {
                "cohort": spec["cohort"],
                "omics": spec["omics"],
                "status": "completed",
                "n_genes": int(expr.shape[0]),
                "n_samples": int(expr.shape[1]),
                "matrix": str(matrix_path.relative_to(ROOT)),
            }
        )

    status = pd.DataFrame(status_rows)
    status.to_csv(TABLES / "cptac_secondary_validation_status.tsv", sep="\t", index=False)
    if coverage_frames:
        pd.concat(coverage_frames, ignore_index=True, sort=False).to_csv(TABLES / "cptac_signature_gene_coverage.tsv", sep="\t", index=False)
    if not score_frames:
        write_status(
            [
                {
                    "module": "CPTAC secondary validation",
                    "status": "skipped",
                    "reason": "No usable CPTAC LUAD/LSCC proteome or phosphoproteome matrix was available.",
                }
            ]
        )
        return status

    all_scores = pd.concat(score_frames, ignore_index=True, sort=False)
    all_scores.to_csv(TABLES / "cptac_sctime_ai_projection.tsv", sep="\t", index=False)

    assoc_rows = []
    test_features = ["scTIME_AI_score", "IFN_response", "Antigen_presentation", "Cytotoxic_CD8", "CXCL10_CXCR3_axis", "TGF_beta_EMT", "Hypoxia", "CD274_expr"]
    for (cohort, omics), sub in all_scores.groupby(["cohort", "omics"], observed=True):
        for feature in [f for f in test_features if f in sub.columns]:
            early = pd.to_numeric(sub.loc[sub["stage_group"] == "Early", feature], errors="coerce").dropna()
            advanced = pd.to_numeric(sub.loc[sub["stage_group"] == "Advanced", feature], errors="coerce").dropna()
            if len(early) > 2 and len(advanced) > 2:
                stat, pval = stats.mannwhitneyu(advanced, early, alternative="two-sided")
            else:
                stat, pval = np.nan, np.nan
            assoc_rows.append(
                {
                    "cohort": cohort,
                    "omics": omics,
                    "feature": feature,
                    "early_mean": early.mean(),
                    "advanced_mean": advanced.mean(),
                    "advanced_minus_early": advanced.mean() - early.mean(),
                    "mannwhitney_u": stat,
                    "p_value": pval,
                    "n_early": len(early),
                    "n_advanced": len(advanced),
                }
            )
    assoc = pd.DataFrame(assoc_rows)
    assoc["fdr"] = benjamini_hochberg(assoc["p_value"])
    assoc.to_csv(TABLES / "cptac_stage_associations.tsv", sep="\t", index=False)

    fig, ax = plt.subplots(figsize=(8, 4.8))
    plot = all_scores[all_scores["stage_group"].isin(["Early", "Advanced"])].copy()
    if not plot.empty:
        plot["cohort_omics"] = plot["cohort"] + " " + plot["omics"]
        sns.boxplot(data=plot, x="cohort_omics", y="scTIME_AI_score", hue="stage_group", ax=ax, fliersize=2)
        sns.stripplot(data=plot, x="cohort_omics", y="scTIME_AI_score", hue="stage_group", dodge=True, ax=ax, color="0.2", size=2, alpha=0.45)
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles[:2], labels[:2], frameon=False, title="")
        ax.set_xlabel("")
        ax.set_ylabel("Projected scTIME-AI score")
        ax.set_title("CPTAC proteome/phosphoproteome projection by stage")
        ax.tick_params(axis="x", rotation=25)
        fig.tight_layout()
        save_quicklook(fig, "cptac_sctime_ai_projection.png", dpi=220)
        plt.close(fig)

    write_status(
        [
            {
                "module": "CPTAC secondary validation",
                "status": "completed",
                "details": f"{status['status'].eq('completed').sum()} CPTAC omics matrices analyzed",
            }
        ]
    )
    return status


def projection_summary_by_histology() -> None:
    path = TABLES / "scTIME_AI_scores_all_bulk_cohorts.tsv"
    if not path.exists():
        return
    scores = pd.read_csv(path, sep="\t")
    gse = scores[scores["dataset"] == "GSE274975"].copy()
    if gse.empty:
        return
    def histology_group(x: object) -> str:
        text = str(x).lower()
        if "squamous" in text:
            return "Squamous"
        if "adeno" in text:
            return "Adenocarcinoma"
        if "small cell" in text:
            return "Small cell"
        return "Other/unspecified"
    source_col = "histology" if "histology" in gse.columns else "source_name_ch1"
    gse["histology_group"] = gse[source_col].map(histology_group)
    summary = gse.groupby("histology_group").agg(
        n=("sample", "size"),
        mean_scTIME_AI_score=("scTIME_AI_score", "mean"),
        median_scTIME_AI_score=("scTIME_AI_score", "median"),
        mean_IFN_response=("IFN_response", "mean"),
        mean_Cytotoxic_CD8=("Cytotoxic_CD8", "mean"),
    ).reset_index()
    summary.to_csv(TABLES / "gse274975_projection_by_histology.tsv", sep="\t", index=False)
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    sns.boxplot(data=gse, x="histology_group", y="scTIME_AI_score", ax=ax, color="#9fb7c9", fliersize=2)
    sns.stripplot(data=gse, x="histology_group", y="scTIME_AI_score", ax=ax, color="0.2", size=2.5, alpha=0.6)
    ax.set_xlabel("")
    ax.set_ylabel("scTIME-AI projected score")
    ax.set_title("GSE274975 projection by histology")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    save_quicklook(fig, "gse274975_projection_by_histology.png", dpi=220)
    plt.close(fig)


def write_report(
    manifest: pd.DataFrame,
    mapping_status: str,
    performance: pd.DataFrame,
    survival: pd.DataFrame,
    sc_patient: pd.DataFrame,
    sc_comp: pd.DataFrame,
    tcr: pd.DataFrame,
    gse207422_sc: pd.DataFrame,
) -> None:
    report = []
    report.append("# CGZ computational analysis summary")
    report.append("")
    report.append(f"Generated: {datetime.now().isoformat(timespec='seconds')}")
    report.append(f"Python: `{sys.executable}`")
    report.append(f"HGNC mapping status: `{mapping_status}`")
    report.append("")
    if not manifest.empty:
        complete = int(manifest["size_matches_expected"].sum())
        report.append(f"- Manifest check: {complete}/{len(manifest)} downloaded files match expected size.")
    if not performance.empty:
        best = performance.sort_values("roc_auc", ascending=False).head(5)
        report.append("- Top model evaluations:")
        for _, row in best.iterrows():
            report.append(
                f"  - {row['model']} {row['evaluation']} {row['train_dataset']} -> {row['test_dataset']}: "
                f"ROC-AUC={row['roc_auc']:.3f}, PR-AUC={row['pr_auc']:.3f}, n={int(row['n'])}."
            )
    if not survival.empty:
        report.append("- GSE135222 PFS validation:")
        for _, row in survival.iterrows():
            hr = row.get("hazard_ratio")
            hr_text = "" if pd.isna(hr) else f", HR={hr:.3f}"
            report.append(f"  - {row['analysis']}: p={row['p_value']:.4g}{hr_text}, n={int(row['n'])}.")
    if sc_patient is not None and not sc_patient.empty:
        mpr = int(sc_patient["benefit"].sum())
        report.append(f"- GSE243013 single-cell metadata: {len(sc_patient)} patients, {mpr} MPR patients.")
    if sc_comp is not None and not sc_comp.empty:
        total_cells = int(sc_comp.drop_duplicates("sampleID")["sample_total"].sum())
        report.append(f"- GSE243013 immune annotation composition summarized from {total_cells:,} cells.")
    if tcr is not None and not tcr.empty:
        report.append(f"- GSE243013 TCR clonality summarized for {len(tcr)} patients with TCR calls.")
    if gse207422_sc is not None and not gse207422_sc.empty:
        report.append(f"- GSE207422 scRNA marker pseudobulk scores generated for {len(gse207422_sc)} samples.")
    status_path = TABLES / "enhanced_analysis_status.tsv"
    if status_path.exists():
        status = pd.read_csv(status_path, sep="\t")
        if not status.empty:
            report.append("- Enhanced study-design modules:")
            for _, row in status.tail(8).iterrows():
                detail = row.get("details")
                detail_text = "" if pd.isna(detail) else f" ({detail})"
                report.append(f"  - {row.get('module')}: {row.get('status')}{detail_text}.")
    report.append("")
    report.append("Key output directories:")
    report.append("- `results/tables`: TSV matrices, statistics, model predictions, and summaries.")
    report.append("- `results/figures`: publication-oriented PNG figures.")
    report.append("- `results/models`: serialized final scTIME-AI model and metadata.")
    report.append("- `logs/run_cgz_analysis.log`: execution log.")
    report.append("")
    report.append("Notes:")
    report.append("- GSE274975 does not contain an ICI response endpoint in GEO metadata; it is used for projection and histology-level immune profiling only.")
    report.append("- GSE243013 includes a 7.1 GB MatrixMarket count matrix with 2.0 billion non-zero entries. The pipeline uses the downloaded official cell annotations and TCR table for full cohort-level single-cell calculations, and does not stream that matrix for marker expression.")
    report.append("- `elasticnet_linear_shap_style_contributions.tsv` contains exact linear logit feature contributions from the final ElasticNet model; no external SHAP package is required.")
    report.append("- GSE207422 full scRNA-seq clustering is implemented with Scanpy. CIBERSORTx/MuSiC functionality is implemented as local NNLS reference deconvolution and CIBERSORTx-ready signature/mixture exports.")
    report.append("- TCGA/CPTAC secondary validation is data-driven: local expression/proteome matrices are analyzed when present, otherwise status tables document the missing input.")
    (RESULTS / "analysis_summary.md").write_text("\n".join(report) + "\n", encoding="utf-8")


def main() -> None:
    setup_logging()
    manifest = check_manifest()
    ensembl_map, mapping_status = load_hgnc_mapping()
    logging.info("Gene mapping status: %s; mapped Ensembl IDs: %d", mapping_status, len(ensembl_map))

    datasets = [
        load_gse126044(),
        load_gse207422_bulk(),
        load_gse135222(ensembl_map),
        load_gse274975(ensembl_map),
    ]
    bulk_scores, coverage = build_bulk_scores(datasets)
    performance, predictions, final_model, feature_cols = evaluate_response_models(bulk_scores)
    scored_all = pd.read_csv(TABLES / "scTIME_AI_scores_all_bulk_cohorts.tsv", sep="\t")
    survival = survival_validation(scored_all)
    projection_summary_by_histology()
    sc_patient, sc_comp, tcr = single_cell_gse243013()
    gse207422_sc = gse207422_sc_pseudobulk()
    gse207422_adata = run_gse207422_scanpy_clustering()
    gse207422_lr_and_nichenet_like(gse207422_adata)
    music_style_nnls_deconvolution(datasets, gse207422_adata)
    tcga_secondary_validation(final_model, feature_cols, ensembl_map)
    cptac_secondary_validation(final_model, feature_cols)

    write_report(
        manifest=manifest,
        mapping_status=mapping_status,
        performance=performance,
        survival=survival,
        sc_patient=sc_patient,
        sc_comp=sc_comp,
        tcr=tcr,
        gse207422_sc=gse207422_sc,
    )
    logging.info("CGZ analysis completed")


if __name__ == "__main__":
    main()
