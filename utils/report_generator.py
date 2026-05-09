"""Helpers for assembling JSON cleaning reports and terminal summaries."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


def _make_json_serializable(value: Any) -> Any:
    """Recursively convert values into JSON-safe Python types."""
    if isinstance(value, dict):
        return {str(key): _make_json_serializable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_make_json_serializable(item) for item in value]
    if isinstance(value, tuple):
        return [_make_json_serializable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    try:
        json.dumps(value)
        return value
    except TypeError:
        return str(value)


def make_safe_stem(file_name: str) -> str:
    """Convert a user file name into a filesystem-safe stem."""
    raw_stem = Path(file_name).stem.strip().lower()
    safe_stem = re.sub(r"[^a-z0-9]+", "_", raw_stem).strip("_")
    return safe_stem or "dataset"


def generate_cleaning_report(
    profile: dict[str, Any],
    validation_result: dict[str, Any],
    cleaning_summary: dict[str, Any],
    ml_recommendation: dict[str, Any],
    original_file_name: str,
    cleaned_file_path: str | Path | None = None,
) -> tuple[dict[str, Any], str]:
    """Build and save a single JSON cleaning report for the current workflow."""
    report_name = f"cleaning_report_{make_safe_stem(original_file_name)}.json"
    report_path = Path("reports") / report_name

    report_data = {
        "original_file_name": original_file_name,
        "original_rows": profile.get("rows"),
        "original_columns": profile.get("columns"),
        "target_column": profile.get("target_column"),
        "final_rows": cleaning_summary.get("final_rows"),
        "final_columns": cleaning_summary.get("final_columns"),
        "missing_values_before_cleaning": cleaning_summary.get("missing_values_before", {}),
        "missing_values_after_cleaning": cleaning_summary.get("missing_values_after", {}),
        "duplicate_rows_removed": cleaning_summary.get("duplicate_rows_removed", 0),
        "numeric_columns_detected": profile.get("numeric_columns", []),
        "categorical_columns_detected": profile.get("categorical_columns", []),
        "text_columns_detected": profile.get("text_columns", []),
        "datetime_columns_detected": profile.get("datetime_columns", []),
        "boolean_columns_detected": profile.get("boolean_columns", []),
        "id_like_columns_detected": profile.get("id_like_columns", []),
        "validation_errors": validation_result.get("errors", []),
        "validation_warnings": validation_result.get("warnings", []),
        "columns_encoded": cleaning_summary.get("encoded_columns", []),
        "encoded_columns_generated_count": cleaning_summary.get(
            "encoded_columns_generated_count", 0
        ),
        "columns_scaled": cleaning_summary.get("scaled_columns", []),
        "scaler_used": cleaning_summary.get("scaler_used"),
        "outlier_handling_summary": cleaning_summary.get("outlier_summary", []),
        "target_encoding_recommendation": cleaning_summary.get("target_encoding_recommendation"),
        "options_used": cleaning_summary.get("options_used", {}),
        "nlp_cleaning_summary": {
            "cleaned_text_columns": cleaning_summary.get("cleaned_text_columns", []),
            "nlp_cleaning_actions": cleaning_summary.get("nlp_cleaning_actions", []),
            "backup_columns": cleaning_summary.get("nlp_original_backup_columns", []),
            "before_after_examples": cleaning_summary.get("nlp_before_after_examples", {}),
        },
        "ml_recommendation": {
            "problem_type": ml_recommendation.get("recommended_problem_type"),
            "reason": ml_recommendation.get("reason"),
            "target_column": ml_recommendation.get("target_column"),
            "detected_text_column": ml_recommendation.get("detected_text_column"),
            "algorithms": ml_recommendation.get("algorithms", []),
        },
        "recommended_ml_problem_type": ml_recommendation.get("recommended_problem_type"),
        "recommended_algorithms": ml_recommendation.get("algorithms", []),
        "before_vs_after_summary": cleaning_summary.get("before_vs_after_summary", {}),
        "cleaning_steps": cleaning_summary.get("cleaning_steps", []),
        "skipped_steps": cleaning_summary.get("skipped_steps", []),
        "output_files": {
            "cleaned_csv_path": cleaned_file_path,
            "report_path": report_path,
        },
    }

    safe_report_data = _make_json_serializable(report_data)
    generate_report(safe_report_data, report_path)

    return safe_report_data, str(report_path)


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
