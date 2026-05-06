"""Helpers for assembling JSON cleaning reports and terminal summaries."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def build_cleaning_report(
    *,
    original_rows: int,
    original_columns: int,
    columns: list[str],
    duplicate_rows: int,
    duplicates_removed: int,
    missing_values_before: dict[str, int],
    missing_values_after: dict[str, int],
    numeric_columns: list[str],
    categorical_columns: list[str],
    text_columns: list[str],
    datetime_columns: list[str],
    boolean_columns: list[str],
    id_like_columns: list[str],
    target_column: str | None,
    encoded_columns: list[str],
    skipped_encoding_columns: list[str],
    final_rows: int,
    final_columns: int,
) -> dict[str, Any]:
    """Create a standard report payload for cleaning runs."""
    return {
        "original_rows": original_rows,
        "original_columns": original_columns,
        "columns": columns,
        "duplicate_rows": duplicate_rows,
        "duplicates_removed": duplicates_removed,
        "missing_values_before": missing_values_before,
        "missing_values_after": missing_values_after,
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "text_columns": text_columns,
        "datetime_columns": datetime_columns,
        "boolean_columns": boolean_columns,
        "id_like_columns": id_like_columns,
        "target_column": target_column,
        "encoded_columns": encoded_columns,
        "skipped_encoding_columns": skipped_encoding_columns,
        "final_rows": final_rows,
        "final_columns": final_columns,
    }


def format_cleaning_summary(
    report_data: dict[str, Any],
    *,
    cleaned_file_path: str | Path,
    report_file_path: str | Path,
) -> str:
    """Create a terminal-friendly summary from a cleaning report."""
    summary_lines = [
        f"Duplicate rows: {report_data['duplicate_rows']}",
        f"Numeric columns: {report_data['numeric_columns']}",
        f"Categorical columns: {report_data['categorical_columns']}",
        f"Text/Object columns: {report_data['text_columns']}",
        f"Datetime columns: {report_data['datetime_columns']}",
        f"Boolean columns: {report_data['boolean_columns']}",
        f"ID-like columns: {report_data['id_like_columns']}",
        f"Target column: {report_data['target_column']}",
        f"Encoded columns: {report_data['encoded_columns']}",
        f"Skipped encoding columns: {report_data['skipped_encoding_columns']}",
        f"Cleaned file path: {cleaned_file_path}",
        f"Report file path: {report_file_path}",
    ]
    return "\n".join(summary_lines)


def generate_report(report_content: dict[str, Any] | str, output_path: str | Path) -> Path:
    """Write a report to disk, storing dictionaries as JSON."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(report_content, dict):
        path.write_text(json.dumps(report_content, indent=2), encoding="utf-8")
    else:
        path.write_text(report_content, encoding="utf-8")

    return path
