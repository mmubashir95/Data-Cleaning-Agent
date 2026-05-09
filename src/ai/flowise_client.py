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
FLOWISE_UNAVAILABLE_MESSAGE = (
    "Flowise Agent is currently unavailable. The Python cleaning report is still available."
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
        "algorithm_recommendation": _to_json_safe(
            ml_recommendation.get("algorithm_recommendation", {})
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
                "algorithm_recommendation": cleaning_report.get(
                    "algorithm_recommendation", {}
                ),
                "cleaning_steps": cleaning_report.get("cleaning_steps", []),
                "skipped_steps": cleaning_report.get("skipped_steps", []),
            }
        )

    return json.dumps(_to_json_safe(summary), indent=2, default=str)


def build_flowise_file_preview(
    df,
    target_column=None,
    cleaning_report=None,
    *,
    profile=None,
    ml_recommendation=None,
    file_name: str | None = None,
    max_rows: int = 10,
) -> str:
    """Build a compact text preview for Flowise from a Python-read dataset.

    Python reads and summarizes the uploaded dataset first. Flowise is used only
    as an AI explanation layer, so the full dataset is never sent. This keeps
    prompts smaller and helps avoid token overflow on large files.
    """
    if df is None:
        raise ValueError("No dataset is available to summarize for the AI agent.")
    if df.empty:
        raise ValueError("The uploaded dataset is empty, so no AI preview can be created.")

    profile = profile or {}
    ml_recommendation = ml_recommendation or {}
    preview_rows = max(1, min(max_rows, 10))

    # Flowise appears to respond better to raw-looking tabular text than to
    # heavily structured JSON-like payloads. We therefore send a compact TSV
    # preview that resembles the manual copy/paste content, while still capping
    # it to a small number of rows to avoid token overflow.
    sample_rows = df.head(preview_rows).copy()
    sample_rows = sample_rows.fillna("")
    sample_rows_text = sample_rows.to_csv(sep="\t", index=False).strip()

    numeric_columns = profile.get("numeric_columns")
    if numeric_columns is None:
        numeric_columns = df.select_dtypes(include=["number"]).columns.tolist()

    categorical_columns = profile.get("categorical_columns", [])
    text_columns = profile.get("text_columns", [])
    datetime_columns = profile.get("datetime_columns", [])
    boolean_columns = profile.get("boolean_columns", [])
    id_like_columns = profile.get("id_like_columns", [])

    preview_sections = [
        "Dataset Preview For AI Agent",
        f"File Name: {file_name or 'Unknown'}",
        f"Dataset Shape: {len(df)} rows x {len(df.columns)} columns",
        f"Column Names: {list(profile.get('column_names', list(df.columns)))}",
        f"Target Column: {target_column if target_column is not None else 'None'}",
        f"Data Types: {_to_json_safe(profile.get('data_types', df.dtypes.astype(str).to_dict()))}",
        f"Missing Values Summary: {_to_json_safe(profile.get('missing_values', df.isna().sum().to_dict()))}",
        f"Duplicate Row Count: {int(profile.get('duplicate_rows', int(df.duplicated().sum())))}",
        f"Numeric Columns: {_to_json_safe(numeric_columns)}",
        f"Categorical Columns: {_to_json_safe(categorical_columns)}",
        f"Text/Object Columns: {_to_json_safe(text_columns)}",
        f"Datetime Columns: {_to_json_safe(datetime_columns)}",
        f"Boolean Columns: {_to_json_safe(boolean_columns)}",
        f"ID-like Columns: {_to_json_safe(id_like_columns)}",
    ]

    if ml_recommendation:
        preview_sections.append(
            "Recommended Problem Type: "
            f"{ml_recommendation.get('recommended_problem_type', 'Unknown')}"
        )
        preview_sections.append(
            "Recommended Algorithms: "
            + ", ".join(
                algorithm.get("name", str(algorithm))
                for algorithm in ml_recommendation.get("algorithms", [])
            )
        )
        algorithm_recommendation = ml_recommendation.get("algorithm_recommendation", {})
        first_choice = algorithm_recommendation.get("beginner_friendly_first_choice", {})
        if first_choice:
            preview_sections.append(
                "Beginner-Friendly First Choice: "
                f"{first_choice.get('name', 'Unknown')}"
            )
            preview_sections.append(
                "Why It Fits: "
                f"{first_choice.get('reason', 'No reason provided.')}"
            )
        if algorithm_recommendation.get("target_variable_type"):
            preview_sections.append(
                "Target Variable Type: "
                f"{algorithm_recommendation['target_variable_type']}"
            )

    preview_sections.extend(
        [
            "Dataset Content Preview (tab-separated):",
            sample_rows_text,
        ]
    )

    if cleaning_report is not None:
        preview_sections.extend(
            [
                "Cleaning Summary:",
                json.dumps(
                    _to_json_safe(
                        {
                            "final_rows": cleaning_report.get("final_rows"),
                            "final_columns": cleaning_report.get("final_columns"),
                            "duplicate_rows_removed": cleaning_report.get(
                                "duplicate_rows_removed"
                            ),
                            "columns_encoded": cleaning_report.get("columns_encoded", []),
                            "columns_scaled": cleaning_report.get("columns_scaled", []),
                            "algorithm_recommendation": cleaning_report.get(
                                "algorithm_recommendation", {}
                            ),
                            "cleaning_steps": cleaning_report.get("cleaning_steps", []),
                            "skipped_steps": cleaning_report.get("skipped_steps", []),
                        }
                    ),
                    indent=2,
                    default=str,
                ),
            ]
        )

    return "\n\n".join(str(section) for section in preview_sections if section is not None)


def build_flowise_combined_question(question: str, file_preview: str | None = None) -> str:
    """Build the actual prompt sent to Flowise.

    Some Flowise agents respond better when the dataset preview appears directly
    inside the question text rather than relying only on a separate file field.
    The Streamlit UI still keeps the user's question box clean; Python injects
    the preview only in the outbound API request.
    """
    cleaned_question = (question or "").strip()
    if not file_preview:
        return cleaned_question

    instruction_block = (
        "Use the dataset preview below to answer the question. "
        "Do not ask the user to paste the dataset again. "
        "Do not request the full CSV or Excel file.\n\n"
        "Dataset preview:\n"
        f"{file_preview}\n\n"
        "User question:\n"
    )
    return instruction_block + cleaned_question


def query_flowise_agent(question: str, file_summary: str | None = None) -> dict:
    """Query Flowise for explanations using only summarized dataset context.

    This helper must not be used for actual validation, profiling, cleaning,
    or report generation. Python remains responsible for all data processing.
    Flowise is used only as an optional AI explanation layer.
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
            "error": FLOWISE_UNAVAILABLE_MESSAGE,
        }
    except requests.ConnectionError:
        return {
            "success": False,
            "error": FLOWISE_UNAVAILABLE_MESSAGE,
        }
    except requests.RequestException as exc:
        return {
            "success": False,
            "error": FLOWISE_UNAVAILABLE_MESSAGE,
        }
    except Exception:
        return {
            "success": False,
            "error": FLOWISE_UNAVAILABLE_MESSAGE,
        }

    if response.status_code != 200:
        return {
            "success": False,
            "error": FLOWISE_UNAVAILABLE_MESSAGE,
        }

    try:
        response_json = response.json()
    except ValueError:
        return {
            "success": False,
            "error": FLOWISE_UNAVAILABLE_MESSAGE,
        }

    return {
        "success": True,
        "answer": extract_flowise_answer(response_json),
        "raw_response": response_json,
    }
