#!/usr/bin/env python
"""Build submission CSV tables from the canonical corrected TSV outputs."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
RESULTS = PACKAGE_ROOT / "source_data" / "results_tables"
SOURCE = PACKAGE_ROOT / "source_data"
TABLES = PACKAGE_ROOT / "tables"


def read_result(name: str) -> pd.DataFrame:
    return pd.read_csv(RESULTS / f"{name}.tsv", sep="\t")


def relative_source(name: str) -> str:
    return f"source_data/results_tables/{name}.tsv"


def build_table1() -> pd.DataFrame:
    tcga = read_result("tcga_secondary_validation").set_index("project")
    cptac = read_result("cptac_secondary_validation_status")
    sc243 = read_result("gse243013_sc_patient_summary")
    tcr243 = read_result("gse243013_tcr_clonality")
    sc207 = read_result("gse207422_sc_pseudobulk_signature_scores")

    def tcga_samples(project: str) -> str:
        row = tcga.loc[project]
        return f"{int(row['n_expression_patients'])} primary-tumor patients; {int(row['n_samples'])} valid OS records"

    def cptac_count(cohort: str) -> str:
        sub = cptac[cptac["cohort"].eq(cohort)]
        specimens = int(sub["n_samples"].max())
        return f"{specimens} specimens; {specimens * len(sub)} omics-layer records"

    included243 = sc243[sc243["analysis_included"].eq(True)]
    evaluable207 = sc207[sc207["benefit"].notna()]
    rows = [
        ("GSE126044", "bulk RNA-seq", "16", "response modeling; radiologic anti-PD-1 response"),
        ("GSE207422 bulk", "bulk RNA-seq", "24", "response modeling; pathologic response after neoadjuvant therapy"),
        ("GSE207422 single-cell", "scRNA-seq", f"{len(sc207)} samples; {len(evaluable207)} evaluable (4 MPR/pCR, 10 NMPR)", "single-cell ligand-receptor context; NE excluded from response tests"),
        ("GSE135222", "bulk RNA-seq", "27", "exploratory PFS projection only"),
        ("GSE274975", "bulk RNA-seq", "61", "histology-only projection; no ICI outcome"),
        ("GSE243013 single-cell", "scRNA-seq", f"243 total; {len(included243)} analyzed (125 MPR/pCR, 108 non-MPR)", "cell-state response analysis; no-anti-PD-1 and unknown-response records excluded"),
        ("GSE243013 TCR", "TCR-seq", f"{len(tcr243)} total; {int(tcr243['benefit'].notna().sum())} response-evaluable", "TCR immune-ecology analysis"),
        ("TCGA-LUAD", "bulk RNA-seq", tcga_samples("TCGA-LUAD"), "primary-tumor patient-level non-ICI projection and exploratory OS association"),
        ("TCGA-LUSC", "bulk RNA-seq", tcga_samples("TCGA-LUSC"), "primary-tumor patient-level non-ICI projection and exploratory OS association"),
        ("CPTAC-LSCC", "proteome and phosphoproteome", cptac_count("CPTAC-LSCC"), "non-ICI cross-omics projection; layers analyzed separately"),
        ("CPTAC-LUAD", "proteome and phosphoproteome", cptac_count("CPTAC-LUAD"), "non-ICI cross-omics projection; layers analyzed separately"),
    ]
    return pd.DataFrame(rows, columns=["cohort", "modality", "samples", "role"])


def build_table2() -> pd.DataFrame:
    performance = read_result("model_performance")
    per_cohort = read_result("per_cohort_oof_sensitivity")
    stratification = read_result("heldout_stratification")
    nested = read_result("nested_selection_sensitivity")
    rows: list[dict[str, object]] = []
    model_labels = {
        "ElasticNet_logistic": "ElasticNet logistic",
        "Ridge_logistic": "Ridge logistic",
        "RandomForest": "Random forest",
        "RBF_SVM": "RBF SVM",
        "XGBoost": "XGBoost",
        "LightGBM": "LightGBM",
    }
    pooled = performance[performance["evaluation"].str.startswith("5-fold CV")]
    for _, row in pooled.iterrows():
        rows.append(
            {
                "analysis": "candidate benchmark",
                "model_or_score": model_labels.get(row["model"], row["model"]),
                "evaluation_context": "Pooled five-fold CV with dataset-specific scaling estimated in each training fold",
                "n": int(row["n"]),
                "benefit_mpr": int(row["positives"]),
                "roc_auc": row["roc_auc"],
                "pr_auc": row["pr_auc"],
                "brier": row["brier"],
                "score_provenance": "one prediction per patient from a fold model and fold-fitted preprocessing",
                "interpretation": "Retained sparse interpretable model" if row["model"] == "ElasticNet_logistic" else "Candidate benchmark only",
            }
        )
    nested_summary = nested[nested["record_type"].eq("pooled_nested_summary")].iloc[0]
    rows.append(
        {
            "analysis": "nested selection sensitivity",
            "model_or_score": "Inner-selected candidate",
            "evaluation_context": "Five outer folds; four inner folds; preprocessing fit inside every training fold",
            "n": 40,
            "benefit_mpr": 14,
            "roc_auc": nested_summary["outer_roc_auc"],
            "pr_auc": nested_summary["outer_pr_auc"],
            "brier": nested_summary["outer_brier"],
            "score_provenance": "nested outer held-out",
            "interpretation": f"Selected model counts: {nested_summary['selected_model_counts']}",
        }
    )
    reciprocal = performance[
        performance["evaluation"].eq("cross-cohort validation") & performance["model"].eq("ElasticNet_logistic")
    ]
    for _, row in reciprocal.iterrows():
        rows.append(
            {
                "analysis": "reciprocal cohort sensitivity",
                "model_or_score": "ElasticNet logistic",
                "evaluation_context": f"{row['train_dataset']} training to {row['test_dataset']} test; preprocessing fit in training cohort",
                "n": int(row["n"]),
                "benefit_mpr": int(row["positives"]),
                "roc_auc": row["roc_auc"],
                "pr_auc": row["pr_auc"],
                "brier": row["brier"],
                "score_provenance": "held-out cohort",
                "interpretation": "Different response endpoint and treatment context from the training cohort",
            }
        )
    for _, row in per_cohort.iterrows():
        rows.append(
            {
                "analysis": "pooled-CV subgroup sensitivity",
                "model_or_score": "ElasticNet logistic",
                "evaluation_context": f"{row['dataset']} subset of pooled five-fold OOF predictions",
                "n": int(row["n"]),
                "benefit_mpr": int(row["benefit_mpr"]),
                "roc_auc": row["roc_auc"],
                "pr_auc": row["pr_auc"],
                "brier": row["brier"],
                "score_provenance": "pooled five-fold out-of-fold",
                "interpretation": "Subgroup sensitivity; not leave-one-cohort-out validation",
            }
        )
    for _, row in stratification.iterrows():
        score = row["score"]
        split = row["split"]
        direction = row["direction"]
        group_label = row["group_label"]
        rows.append(
            {
                "analysis": "held-out stratification",
                "model_or_score": score,
                "evaluation_context": f"Within-cohort {split} groups then pooled",
                "n": int(row["n"]),
                "benefit_mpr": int(row["benefit_mpr"]),
                "group_or_cutoff": group_label,
                "benefit_mpr_count": int(row["benefit_mpr"]),
                "benefit_mpr_rate": row["benefit_mpr_rate"],
                "two_sided_fisher_p": row["two_sided_fisher_p"],
                "score_provenance": row["score_provenance"],
                "interpretation": f"{direction}; identical split scheme used for scTIME and CD274",
            }
        )
    columns = [
        "analysis", "model_or_score", "evaluation_context", "n", "benefit_mpr", "roc_auc", "pr_auc", "brier",
        "group_or_cutoff", "benefit_mpr_count", "benefit_mpr_rate", "two_sided_fisher_p", "score_provenance", "interpretation",
    ]
    out = pd.DataFrame(rows).reindex(columns=columns)
    for column in ["roc_auc", "pr_auc", "brier", "benefit_mpr_rate", "two_sided_fisher_p"]:
        out[column] = pd.to_numeric(out[column], errors="coerce").round(6)
    return out


def section_frame(section: str, source_name: str, frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out.insert(0, "source_row", np.arange(1, len(out) + 1))
    out.insert(0, "source_file", relative_source(source_name))
    out.insert(0, "section", section)
    return out


def build_supplementary_table_s2() -> pd.DataFrame:
    frames = [
        section_frame("state_fraction_differential", "gse243013_sc_state_response_differential", read_result("gse243013_sc_state_response_differential")),
        section_frame("patient_level_state_fractions", "gse243013_sc_state_composition", read_result("gse243013_sc_state_composition")),
        section_frame("TCR_patient_metrics", "gse243013_tcr_clonality", read_result("gse243013_tcr_clonality")),
        section_frame("TCR_response_differential", "gse243013_tcr_response_differential", read_result("gse243013_tcr_response_differential")),
        section_frame("TCR_cell_state_correlations", "figure6_tcr_cell_state_correlations", pd.read_csv(SOURCE / "figure6_tcr_cell_state_correlations.tsv", sep="\t")),
        section_frame("response_state_network", "figure_2_panele_response_linked_single_cell_state_network", pd.read_csv(SOURCE / "panel_data" / "figure_2_panele_response_linked_single_cell_state_network.tsv", sep="\t")),
    ]
    return pd.concat(frames, ignore_index=True, sort=False)


def build_supplementary_table_s3() -> pd.DataFrame:
    rows = [
        ("Model benchmark", "Six-model pooled benchmark", relative_source("model_performance"), "40 labeled samples and 14 benefit/MPR labels", "Fold-local preprocessing; post hoc sparse ElasticNet retained", "ROC-AUC, PR-AUC and Brier results plus reciprocal cohort assessments."),
        ("Primary held-out audit", "OOF patient predictions", relative_source("heldout_predictions"), "40 labeled samples", "One pooled five-fold OOF ElasticNet score per sample", "Authoritative individual-level table for held-out response stratification."),
        ("Primary held-out audit", "Matched response stratification", relative_source("heldout_stratification"), "Matched within-cohort quartile and median splits", "OOF scTIME and standalone CD274 use identical split schemes", "Counts, rates and two-sided Fisher exact P values."),
        ("Endpoint sensitivity", "Per-cohort pooled-OOF metrics", relative_source("per_cohort_oof_sensitivity"), "GSE126044 and GSE207422 bulk", "Pooled-CV subgroup sensitivity", "Cohort-specific ROC-AUC, PR-AUC and Brier."),
        ("Selection sensitivity", "Fixed nested candidate selection", relative_source("nested_selection_sensitivity"), "Five outer and four inner folds", "Scaling and selection both fit inside training folds", "Inner candidate results and pooled outer-held-out metrics."),
        ("Coefficient audit", "Full final ElasticNet vector", relative_source("coefficient_audit"), "12 immune modules", "Full refit for coefficients and projection only", "All coefficients, including exact zeros."),
        ("Interpretability", "Patient-level linear contributions", relative_source("elasticnet_linear_shap_style_contributions"), "40 labeled samples", "Full-refit interpretation only", "Exact linear-logit feature contributions."),
        ("PFS sensitivity", "Age/sex-adjusted Cox sensitivity", relative_source("gse135222_multivariable_sensitivity"), "GSE135222 n=27; 21 events", "Exploratory full-refit projection", "Univariate and age/sex-adjusted sensitivity models."),
        ("Single-cell response", "Corrected GSE243013 labels", relative_source("gse243013_sc_patient_summary"), "243 total; 233 evaluable", "MPR+pCR versus non-MPR; no anti-PD-1 and unknown response excluded", "Patient-level inclusion and response fields."),
        ("Single-cell LR", "Corrected GSE207422 labels", relative_source("gse207422_lr_axis_response_differential"), "4 MPR/pCR versus 10 NMPR", "NE excluded", "Ligand-receptor response comparisons and FDR."),
        ("TCGA projection", "Primary-tumor patient-level OS", relative_source("tcga_secondary_validation"), "LUAD and LUSC", "Primary tumour only; duplicate aliquots aggregated; invalid OS times excluded", "Sample QC, survival denominators, HR and P values."),
    ]
    return pd.DataFrame(rows, columns=["section", "audit_artifact", "relative_path", "scope", "provenance_or_use", "description"])


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    outputs = {
        "Table_1_cohort_resource_summary.csv": build_table1(),
        "Table_2_main_model_validation_metrics.csv": build_table2(),
        "Supplementary_Table_S2_single_cell_state_and_TCR_summary.csv": build_supplementary_table_s2(),
        "Supplementary_Table_S3_model_assessment_and_interpretability.csv": build_supplementary_table_s3(),
    }
    for filename, frame in outputs.items():
        frame.to_csv(TABLES / filename, index=False, encoding="utf-8-sig")
        print(f"Wrote {filename}: {len(frame)} rows")


if __name__ == "__main__":
    main()
