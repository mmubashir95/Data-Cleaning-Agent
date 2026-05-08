"""Flowise API client used only for AI explanations, not data cleaning."""

from __future__ import annotations

import json
from typing import Any

import pandas as pd
import requests

FLOWISE_PREDICTION_URL = (
    "https://cloud.flowiseai.com/api/v1/prediction/"
    "6a7b5277-b4bf-4a79-a785-8cde06dbf860"
)


def _to_json_safe(value: Any) -> Any:
    """Recursively convert Pandas/NumPy values into JSON-safe Python types."""
    if isinstance(value, dict):
        return {str(key): _to_json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_to_json_safe(item) for item in value]
    if isinstance(value, pd.DataFrame):
        return _to_json_safe(value.to_dict(orient="records"))
    if isinstance(value, pd.Series):
        return _to_json_safe(value.to_dict())
    if isinstance(value, pd.Index):
        return _to_json_safe(value.tolist())
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return str(value)
    return value


def extract_flowise_answer(response_json: dict) -> str:
    """Extract the clean answer text from a Flowise JSON response."""
    for key in ("text", "answer", "result", "output", "response"):
        value = response_json.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return json.dumps(response_json, indent=2, default=str)


def build_flowise_dataset_summary(
    dataframe,
    profile,
    ml_recommendation,
    target_column=None,
    cleaning_report=None,
) -> str:
    """Build a compact JSON summary for Flowise instead of sending full datasets.

    This protects the explanation layer from unnecessary token growth and helps
    avoid Flowise/OpenAI token overflow by sending only the most relevant,
    high-signal dataset summary and at most the first 5 rows.
    """
    summary = {
        "dataset_shape": {
            "rows": int(len(dataframe)),
            "columns": int(len(dataframe.columns)),
        },
        "column_names": _to_json_safe(profile.get("column_names", list(dataframe.columns))),
        "target_column": _to_json_safe(target_column if target_column is not None else profile.get("target_column")),
        "data_types": _to_json_safe(profile.get("data_types", {})),
        "missing_values": _to_json_safe(profile.get("missing_values", {})),
        "duplicate_rows": int(profile.get("duplicate_rows", int(dataframe.duplicated().sum()))),
        "numeric_columns": _to_json_safe(profile.get("numeric_columns", [])),
        "categorical_columns": _to_json_safe(profile.get("categorical_columns", [])),
        "text_columns": _to_json_safe(profile.get("text_columns", [])),
        "datetime_columns": _to_json_safe(profile.get("datetime_columns", [])),
        "boolean_columns": _to_json_safe(profile.get("boolean_columns", [])),
        "id_like_columns": _to_json_safe(profile.get("id_like_columns", [])),
        "recommended_problem_type": _to_json_safe(
            ml_recommendation.get("recommended_problem_type")
        ),
        "recommended_algorithms": _to_json_safe(
            [
                algorithm.get("name", str(algorithm))
                for algorithm in ml_recommendation.get("algorithms", [])
            ]
        ),
        "first_5_rows": _to_json_safe(dataframe.head(5).to_dict(orient="records")),
    }

    if cleaning_report is not None:
        summary["cleaning_report"] = _to_json_safe(
            {
                "final_rows": cleaning_report.get("final_rows"),
                "final_columns": cleaning_report.get("final_columns"),
                "duplicate_rows_removed": cleaning_report.get("duplicate_rows_removed"),
                "missing_values_after_cleaning": cleaning_report.get(
                    "missing_values_after_cleaning", {}
                ),
                "columns_encoded": cleaning_report.get("columns_encoded", []),
                "columns_scaled": cleaning_report.get("columns_scaled", []),
                "recommended_ml_problem_type": cleaning_report.get(
                    "recommended_ml_problem_type"
                ),
                "cleaning_steps": cleaning_report.get("cleaning_steps", []),
                "skipped_steps": cleaning_report.get("skipped_steps", []),
            }
        )

    return json.dumps(_to_json_safe(summary), indent=2, default=str)


def query_flowise_agent(question: str, file_summary: str | None = None) -> dict:
    """Query Flowise for explanations using only summarized dataset context.

    This helper must not be used for actual validation, profiling, cleaning,
    or report generation. Python remains responsible for all data processing.
    Flowise is used only as an AI explanation layer.
    """
    payload = {
        "question": question,
        "file": file_summary,
    }

    try:
        response = requests.post(FLOWISE_PREDICTION_URL, json=payload, timeout=60)
    except requests.Timeout:
        return {
            "success": False,
            "error": "The Flowise explanation service timed out. Please try again.",
        }
    except requests.ConnectionError:
        return {
            "success": False,
            "error": "The Flowise explanation service could not be reached. Please check your network connection.",
        }
    except requests.RequestException as exc:
        return {
            "success": False,
            "error": f"Flowise request failed: {exc}",
        }

    if response.status_code != 200:
        return {
            "success": False,
            "error": (
                "The Flowise explanation service returned an unexpected response "
                f"(HTTP {response.status_code})."
            ),
        }

    try:
        response_json = response.json()
    except ValueError:
        return {
            "success": False,
            "error": "The Flowise explanation service returned invalid JSON.",
        }

    return {
        "success": True,
        "answer": extract_flowise_answer(response_json),
        "raw_response": response_json,
    }
