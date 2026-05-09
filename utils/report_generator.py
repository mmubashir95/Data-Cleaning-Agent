"""Helpers for assembling JSON cleaning reports and terminal summaries."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from utils.library_usage import build_pandas_numpy_usage
from utils.viva_summary import build_viva_summary
from src.ai.flowise_client import build_default_flowise_metadata


def _make_json_serializable(value: Any) -> Any:
    """Recursively convert values into JSON-safe Python types.

    Reports aggregate pandas, NumPy, pathlib, and UI-derived values, so this
    conversion step prevents serialization failures late in the workflow.
    """
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
    flowise_metadata: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], str]:
    """Build and save a single JSON cleaning report for the current workflow.

    The report is intentionally structured for both machine-readable reuse and
    beginner-friendly explanation. It also records Flowise metadata so the
    project can show exactly what the AI layer did or did not receive.
    """
    report_name = f"cleaning_report_{make_safe_stem(original_file_name)}.json"
    report_path = Path("reports") / report_name
    options_used = cleaning_summary.get("options_used", {})
    missing_filled = cleaning_summary.get("missing_filled", {})
    converted_columns = cleaning_summary.get("converted_columns", {})
    outlier_summary = cleaning_summary.get("outlier_summary", [])
    encoded_columns = cleaning_summary.get("encoded_columns", [])
    scaled_columns = cleaning_summary.get("scaled_columns", [])
    cleaned_text_columns = cleaning_summary.get("cleaned_text_columns", [])
    pandas_numpy_usage = build_pandas_numpy_usage(
        original_file_name=original_file_name,
        profile=profile,
        cleaning_summary=cleaning_summary,
    )
    flowise_metadata = flowise_metadata or build_default_flowise_metadata()

    # Keep the action blocks explicit so the report can distinguish between a
    # step being selected, actually performed, or skipped for safety reasons.
    cleaning_actions = {
        "duplicates_removed": {
            "selected": options_used.get("remove_duplicates", False),
            "performed": cleaning_summary.get("duplicate_rows_removed", 0) > 0,
            "rows_removed": cleaning_summary.get("duplicate_rows_removed", 0),
        },
        "missing_values_handled": {
            "selected": options_used.get("handle_missing_values", False),
            "performed": bool(missing_filled),
            "columns_handled": sorted(missing_filled.keys()),
            "details": missing_filled,
        },
        "wrong_data_types_fixed": {
            "selected": options_used.get("fix_data_types", False),
            "performed": bool(converted_columns),
            "columns_fixed": sorted(converted_columns.keys()),
            "details": converted_columns,
        },
        "outliers_detected_and_handled": {
            "selected": options_used.get("handle_outliers", False),
            "performed": bool(outlier_summary),
            "columns_handled": [item.get("column_name") for item in outlier_summary],
            "details": outlier_summary,
        },
        "categorical_columns_encoded": {
            "selected": options_used.get("encode_categorical", False),
            "performed": bool(encoded_columns),
            "columns_encoded": encoded_columns,
            "generated_columns_count": cleaning_summary.get(
                "encoded_columns_generated_count", 0
            ),
        },
        "numeric_columns_scaled": {
            "selected": options_used.get("scale_numeric", False),
            "performed": bool(scaled_columns),
            "scaler_used": cleaning_summary.get("scaler_used"),
            "columns_scaled": scaled_columns,
        },
        "nlp_text_cleaning_applied": {
            "selected": options_used.get("nlp_cleaning", False),
            "performed": bool(cleaned_text_columns),
            "columns_cleaned": cleaned_text_columns,
            "details": cleaning_summary.get("nlp_cleaning_actions", []),
        },
    }
    viva_summary = build_viva_summary(
        original_file_name=original_file_name,
        cleaning_summary=cleaning_summary,
        ml_recommendation=ml_recommendation,
        cleaning_actions=cleaning_actions,
    )

    # The report stores Python-computed facts only. AI output is kept separate
    # so exact dataset statistics are never replaced by generated text.
    report_data = {
        "original_file_name": original_file_name,
        "original_rows": profile.get("rows"),
        "original_columns": profile.get("columns"),
        "target_column": profile.get("target_column"),
        "selected_target_column": ml_recommendation.get("selected_target_column"),
        "suggested_target_column": ml_recommendation.get("suggested_target_column"),
        "target_detection_confidence": ml_recommendation.get("target_detection_confidence"),
        "problem_type": ml_recommendation.get("recommended_problem_type"),
        "problem_type_reason": ml_recommendation.get("problem_type_reason"),
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
        "columns_encoded": encoded_columns,
        "encoded_columns_generated_count": cleaning_summary.get(
            "encoded_columns_generated_count", 0
        ),
        "columns_scaled": scaled_columns,
        "scaler_used": cleaning_summary.get("scaler_used"),
        "outlier_handling_summary": outlier_summary,
        "target_encoding_recommendation": cleaning_summary.get("target_encoding_recommendation"),
        "options_used": options_used,
        "cleaning_actions": cleaning_actions,
        "nlp_cleaning_summary": {
            "cleaned_text_columns": cleaned_text_columns,
            "nlp_cleaning_actions": cleaning_summary.get("nlp_cleaning_actions", []),
            "backup_columns": cleaning_summary.get("nlp_original_backup_columns", []),
            "before_after_examples": cleaning_summary.get("nlp_before_after_examples", {}),
        },
        "ml_recommendation": {
            "problem_type": ml_recommendation.get("recommended_problem_type"),
            "reason": ml_recommendation.get("reason"),
            "target_column": ml_recommendation.get("target_column"),
            "selected_target_column": ml_recommendation.get("selected_target_column"),
            "suggested_target_column": ml_recommendation.get("suggested_target_column"),
            "target_detection_confidence": ml_recommendation.get("target_detection_confidence"),
            "target_column_used_for_inference": ml_recommendation.get("target_column_used_for_inference"),
            "detected_text_column": ml_recommendation.get("detected_text_column"),
            "algorithms": ml_recommendation.get("algorithms", []),
            "algorithm_recommendation": ml_recommendation.get("algorithm_recommendation", {}),
            "target_detection_metadata": ml_recommendation.get("target_detection_metadata", {}),
        },
        "recommended_ml_problem_type": ml_recommendation.get("recommended_problem_type"),
        "recommended_algorithms": ml_recommendation.get("algorithms", []),
        "algorithm_recommendation": ml_recommendation.get("algorithm_recommendation", {}),
        "pandas_numpy_usage": pandas_numpy_usage,
        "project_summary_for_viva": viva_summary,
        "flowise_integration": flowise_metadata,
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
    """Write a report to disk, storing dictionaries as JSON.

    Disk output happens at the end so the app can show the same canonical
    report in the UI, downloads, and saved artifact.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(report_content, dict):
        path.write_text(json.dumps(report_content, indent=2), encoding="utf-8")
    else:
        path.write_text(report_content, encoding="utf-8")

    return path
