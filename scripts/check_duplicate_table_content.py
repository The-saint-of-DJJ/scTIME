#!/usr/bin/env python
"""Check the current final CSV tables for exact duplicate content."""

from __future__ import annotations

import argparse
import csv
import hashlib
from collections import Counter, defaultdict
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TABLES_DIR = PACKAGE_ROOT / "tables"
REPORT_NAME = "duplicate_content_check_report.csv"


def read_csv_rows(path: Path) -> tuple[list[str], list[list[str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.reader(handle))
    if not rows:
        return [], []
    return rows[0], rows[1:]


def duplicate_row_count(rows: list[list[str]]) -> int:
    counts = Counter(tuple(row) for row in rows)
    return sum(count - 1 for count in counts.values() if count > 1)


def status_for(count: int) -> str:
    return "no_duplicates" if count == 0 else "duplicates_detected"


def build_report(table_paths: list[Path]) -> list[list[object]]:
    report: list[list[object]] = [
        ["check_type", "scope", "item", "duplicate_count", "status", "details"]
    ]
    content_hashes: defaultdict[str, list[str]] = defaultdict(list)
    row_files: defaultdict[tuple[str, ...], set[str]] = defaultdict(set)
    summaries: list[tuple[Path, list[str], list[list[str]]]] = []

    for path in table_paths:
        header, rows = read_csv_rows(path)
        summaries.append((path, header, rows))
        content_hashes[hashlib.sha256(path.read_bytes()).hexdigest()].append(path.name)
        for row in rows:
            row_files[tuple(row)].add(path.name)

    duplicate_file_count = sum(len(paths) - 1 for paths in content_hashes.values() if len(paths) > 1)
    report.append(
        [
            "file_content_hash",
            "current_final_csv_tables",
            "all_final_csv_files",
            duplicate_file_count,
            status_for(duplicate_file_count),
            "Exact SHA-256 duplicate file-content check across current final CSV tables.",
        ]
    )

    for path, header, rows in summaries:
        duplicates = duplicate_row_count(rows)
        report.append(
            [
                "within_file_exact_duplicate_rows",
                path.name,
                "data_rows_excluding_header",
                duplicates,
                status_for(duplicates),
                f"{len(rows)} data rows checked; {len(header)} columns.",
            ]
        )

    cross_file_duplicates = sum(len(files) - 1 for files in row_files.values() if len(files) > 1)
    report.append(
        [
            "cross_file_exact_duplicate_rows",
            "current_final_csv_tables",
            "complete_data_rows",
            cross_file_duplicates,
            status_for(cross_file_duplicates),
            "Exact data-row values compared across distinct current final CSV tables.",
        ]
    )

    for path, header, rows in summaries:
        report.append(
            [
                "file_summary",
                path.name,
                "rows_columns",
                "",
                "checked",
                f"{len(rows)} data rows; {len(header)} columns; size={path.stat().st_size} bytes",
            ]
        )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tables-dir", type=Path, default=DEFAULT_TABLES_DIR)
    args = parser.parse_args()

    tables_dir = args.tables_dir.resolve()
    report_path = tables_dir / REPORT_NAME
    table_paths = sorted(path for path in tables_dir.glob("*.csv") if path.name != REPORT_NAME)
    if not table_paths:
        raise FileNotFoundError(f"No final CSV tables found in {tables_dir}")

    with report_path.open("w", encoding="utf-8", newline="") as handle:
        csv.writer(handle).writerows(build_report(table_paths))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
