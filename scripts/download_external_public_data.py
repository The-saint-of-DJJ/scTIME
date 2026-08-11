#!/usr/bin/env python3
"""Retrieve and stage public TCGA/Xena inputs used by the scTIME workflow.

This script intentionally does not guess direct URLs for the LinkedOmics CPTAC
exports.  Those files are selected through the public portal, then staged by
their documented filenames with ``--stage-cptac``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import requests


ROOT = Path(__file__).resolve().parents[1]
GDC_CASES_URL = "https://api.gdc.cancer.gov/cases"
XENA_DOWNLOAD_BASE = "https://gdc-hub.s3.us-east-1.amazonaws.com/download"
PROJECTS = ("TCGA-LUAD", "TCGA-LUSC")
GDC_FIELDS = (
    "submitter_id",
    "demographic.vital_status",
    "demographic.days_to_death",
    "demographic.age_at_index",
    "demographic.gender",
    "diagnoses.days_to_last_follow_up",
    "diagnoses.ajcc_pathologic_stage",
)
CPTAC_FILES = (
    "HS_CPTAC_LUAD_proteome_ratio_NArm_TUMOR.cct",
    "HS_CPTAC_LUAD_phosphoproteome_ratio_norm_NArm_TUMOR.cct",
    "HS_CPTAC_LUAD_cli.tsi",
    "HS_CPTAC_LSCC_2020_proteome_ratio_NArm_TUMOR.cct",
    "HS_CPTAC_LSCC_2020_phospho_ratio_norm_NArm_TUMOR.cct",
    "HS_CPTAC_LSCC_2020_clinical_phenotypes_TUMOR.tsi",
)
MANIFEST_FIELDS = (
    "retrieved_at_utc",
    "resource",
    "status",
    "destination",
    "source",
    "bytes",
    "sha256",
    "notes",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def sha256sum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def manifest_row(
    root: Path,
    resource: str,
    status: str,
    destination: Path,
    source: str,
    notes: str = "",
) -> dict[str, str]:
    exists = destination.exists()
    return {
        "retrieved_at_utc": utc_now(),
        "resource": resource,
        "status": status,
        "destination": str(destination.relative_to(root)),
        "source": source,
        "bytes": str(destination.stat().st_size if exists else 0),
        "sha256": sha256sum(destination) if exists else "",
        "notes": notes,
    }


def write_manifest(path: Path, rows: Iterable[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def gdc_filter(project: str) -> str:
    return json.dumps(
        {
            "op": "in",
            "content": {
                "field": "project.project_id",
                "value": [project],
            },
        },
        separators=(",", ":"),
    )


def first_nonempty(row: dict[str, str], field: str) -> str:
    """Return a scalar or first diagnosis value from a GDC TSV row."""
    candidates = [field]
    if field.startswith("diagnoses."):
        suffix = re.escape(field.removeprefix("diagnoses."))
        diagnoses = [
            key
            for key in row
            if re.fullmatch(rf"diagnoses\.\d+\.{suffix}", key)
        ]
        candidates.extend(sorted(diagnoses, key=lambda key: int(key.split(".")[1])))
    for candidate in candidates:
        value = str(row.get(candidate, "") or "").strip()
        if value and value.lower() not in {"nan", "none", "--"}:
            return value
    return ""


def download_gdc_clinical(
    session: requests.Session,
    root: Path,
    project: str,
    force: bool,
) -> dict[str, str]:
    destination = root / "data" / "gdc_clinical" / project / "clinical" / "clinical.tsv"
    if destination.exists() and not force:
        return manifest_row(
            root,
            f"{project} clinical",
            "exists",
            destination,
            GDC_CASES_URL,
            "Use --force to retrieve the current GDC case-level clinical table.",
        )

    params = {
        "filters": gdc_filter(project),
        "format": "TSV",
        "size": "10000",
        "fields": ",".join(GDC_FIELDS),
    }
    response = session.get(GDC_CASES_URL, params=params, timeout=120)
    response.raise_for_status()
    source_rows = list(csv.DictReader(io.StringIO(response.text), delimiter="\t"))
    if not source_rows:
        raise RuntimeError(f"GDC returned no clinical rows for {project}.")

    columns = (
        "cases.submitter_id",
        "demographic.days_to_death",
        "diagnoses.days_to_last_follow_up",
        "demographic.vital_status",
        "demographic.age_at_index",
        "demographic.gender",
        "diagnoses.ajcc_pathologic_stage",
        "gdc_case_id",
    )
    normalized_rows = []
    for row in source_rows:
        normalized_rows.append(
            {
                "cases.submitter_id": first_nonempty(row, "submitter_id"),
                "demographic.days_to_death": first_nonempty(row, "demographic.days_to_death"),
                "diagnoses.days_to_last_follow_up": first_nonempty(row, "diagnoses.days_to_last_follow_up"),
                "demographic.vital_status": first_nonempty(row, "demographic.vital_status"),
                "demographic.age_at_index": first_nonempty(row, "demographic.age_at_index"),
                "demographic.gender": first_nonempty(row, "demographic.gender"),
                "diagnoses.ajcc_pathologic_stage": first_nonempty(row, "diagnoses.ajcc_pathologic_stage"),
                "gdc_case_id": first_nonempty(row, "id"),
            }
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, delimiter="\t")
        writer.writeheader()
        writer.writerows(normalized_rows)
    return manifest_row(
        root,
        f"{project} clinical",
        "downloaded",
        destination,
        response.url,
        f"{len(normalized_rows)} GDC case-level rows normalized to the workflow schema.",
    )


def download_file(
    session: requests.Session,
    root: Path,
    resource: str,
    url: str,
    destination: Path,
    force: bool,
) -> dict[str, str]:
    if destination.exists() and not force:
        return manifest_row(
            root,
            resource,
            "exists",
            destination,
            url,
            "Use --force to retrieve the current published file.",
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f"{destination.name}.part")
    if temporary.exists():
        temporary.unlink()
    with session.get(url, stream=True, timeout=120) as response:
        response.raise_for_status()
        with temporary.open("wb") as handle:
            for block in response.iter_content(chunk_size=1024 * 1024):
                if block:
                    handle.write(block)
    temporary.replace(destination)
    return manifest_row(root, resource, "downloaded", destination, url)


def stage_cptac_exports(root: Path, source_dir: Path, force: bool) -> list[dict[str, str]]:
    if not source_dir.is_dir():
        raise FileNotFoundError(f"CPTAC source directory does not exist: {source_dir}")
    candidate_dirs = (source_dir, source_dir / "LinkedOmics")
    destination_dir = root / "data" / "cptac" / "LinkedOmics"
    rows = []
    for name in CPTAC_FILES:
        source = next((folder / name for folder in candidate_dirs if (folder / name).is_file()), None)
        destination = destination_dir / name
        if source is None:
            rows.append(
                manifest_row(
                    root,
                    f"CPTAC/LinkedOmics {name}",
                    "missing_source",
                    destination,
                    "https://linkedomics.org/",
                    "Obtain this exact portal export before staging it.",
                )
            )
            continue
        if destination.exists() and not force:
            rows.append(
                manifest_row(
                    root,
                    f"CPTAC/LinkedOmics {name}",
                    "exists",
                    destination,
                    str(source.resolve()),
                    "Use --force to replace the staged file.",
                )
            )
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        rows.append(
            manifest_row(
                root,
                f"CPTAC/LinkedOmics {name}",
                "staged",
                destination,
                str(source.resolve()),
                "Source export selected through the LinkedOmics public portal.",
            )
        )
    return rows


def verify_external_inputs(root: Path) -> list[dict[str, str]]:
    rows = []
    for project in PROJECTS:
        clinical = root / "data" / "gdc_clinical" / project / "clinical" / "clinical.tsv"
        expression = root / "data" / "xena" / f"{project}.star_counts.tsv.gz"
        rows.append(
            manifest_row(root, f"{project} clinical", "present" if clinical.exists() else "missing", clinical, GDC_CASES_URL)
        )
        rows.append(
            manifest_row(
                root,
                f"{project} Xena STAR counts",
                "present" if expression.exists() else "missing",
                expression,
                f"{XENA_DOWNLOAD_BASE}/{project}.star_counts.tsv.gz",
            )
        )
    for name in CPTAC_FILES:
        destination = root / "data" / "cptac" / "LinkedOmics" / name
        rows.append(
            manifest_row(
                root,
                f"CPTAC/LinkedOmics {name}",
                "present" if destination.exists() else "missing",
                destination,
                "https://linkedomics.org/",
            )
        )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT, help="Repository root (default: script parent).")
    parser.add_argument("--tcga", action="store_true", help="Download GDC clinical tables for LUAD and LUSC.")
    parser.add_argument("--xena", action="store_true", help="Download Xena GDC Hub STAR-count matrices for LUAD and LUSC.")
    parser.add_argument("--all", action="store_true", help="Run both --tcga and --xena.")
    parser.add_argument(
        "--stage-cptac",
        type=Path,
        metavar="DIRECTORY",
        help="Copy the six documented LinkedOmics exports from DIRECTORY into data/cptac/LinkedOmics.",
    )
    parser.add_argument("--verify", action="store_true", help="Check all TCGA/Xena/CPTAC expected paths without downloading.")
    parser.add_argument("--force", action="store_true", help="Replace existing downloaded or staged files.")
    args = parser.parse_args()
    if not (args.tcga or args.xena or args.all or args.stage_cptac or args.verify):
        parser.error("select --tcga, --xena, --all, --stage-cptac, or --verify")
    return args


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    rows: list[dict[str, str]] = []
    session = requests.Session()
    session.headers.update({"User-Agent": "scTIME-public-data-retrieval/1.0"})
    try:
        if args.all or args.tcga:
            for project in PROJECTS:
                rows.append(download_gdc_clinical(session, root, project, args.force))
        if args.all or args.xena:
            for project in PROJECTS:
                destination = root / "data" / "xena" / f"{project}.star_counts.tsv.gz"
                rows.append(
                    download_file(
                        session,
                        root,
                        f"{project} Xena STAR counts",
                        f"{XENA_DOWNLOAD_BASE}/{project}.star_counts.tsv.gz",
                        destination,
                        args.force,
                    )
                )
        if args.stage_cptac:
            rows.extend(stage_cptac_exports(root, args.stage_cptac.resolve(), args.force))
        if args.verify:
            rows.extend(verify_external_inputs(root))
    except (requests.RequestException, OSError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    manifest = root / "external_public_data_manifest.tsv"
    write_manifest(manifest, rows)
    for row in rows:
        print(f"{row['status']:14} {row['resource']} -> {row['destination']}")
    print(f"Manifest written to {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
