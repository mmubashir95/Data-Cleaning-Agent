"""Flowise integration used only for AI explanations, never for core processing.

Python remains the source of truth for validation, profiling, statistics,
cleaning, and report generation. The Flowise call receives only a compact
preview/profile so the app avoids token overflow, reduces privacy exposure,
and prevents the LLM from being treated as the executor of data operations.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from utils.library_usage import build_pandas_numpy_usage

logger = logging.getLogger(__name__)

FLOWISE_PREDICTION_URL = (
    "https://cloud.flowiseai.com/api/v1/prediction/"
    "6a7b5277-b4bf-4a79-a785-8cde06dbf860"
)
FLOWISE_UNAVAILABLE_MESSAGE = (
    "Flowise Agent is currently unavailable. The Python cleaning report is still available."
)
FLOWISE_PROFILE_NOTE = (
    "This AI explanation is based on a Python-generated dataset profile, cleaning report, "
    "and small preview only. The full raw dataset was not sent to the AI model. Do not "
    "invent exact statistics that are not present in the provided profile. Python performs "
    "the real cleaning and preprocessing; Flowise only explains the generated summary."
)
FLOWISE_RESPONSE_PREFIX = (
    "This explanation is based on the Python-generated dataset profile and cleaning report, "
    "not the full raw dataset."
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
            answer = value.strip()
            if FLOWISE_RESPONSE_PREFIX not in answer:
                return f"{FLOWISE_RESPONSE_PREFIX}\n\n{answer}"
            return answer
    fallback_answer = json.dumps(response_json, indent=2, default=str)
    return f"{FLOWISE_RESPONSE_PREFIX}\n\n{fallback_answer}"


def build_default_flowise_metadata() -> dict[str, Any]:
    """Return a default metadata block for report output before any AI call happens."""
    return {
        "flowise_called": False,
        "profile_sent_to_flowise": False,
        "profile_keys_sent": [],
        "full_dataset_sent_to_flowise": False,
        "preview_rows_sent": 0,
        "flowise_status": "skipped",
        "flowise_error_message": None,
    }


def _truncate_text_value(value: Any, max_length: int = 120) -> Any:
    """Trim long text values so Flowise receives a compact preview."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None

    text = str(value)
    if len(text) <= max_length:
        return text
    return text[: max_length - 3] + "..."


def _truncate_sample_rows(
    dataframe: pd.DataFrame,
    *,
    max_rows: int,
    max_columns: int = 25,
    max_text_length: int = 120,
) -> tuple[list[dict[str, Any]], list[str], int]:
    """Build a token-safe sample row preview.

    Only a small preview is sent to Flowise because large raw datasets are both
    expensive and unreliable in an LLM context. This keeps prompts bounded and
    ensures Python, not the model, performs any exact dataset computation.
    """
    preview_rows = max(1, min(max_rows, 10))
    preview_columns = list(dataframe.columns[:max_columns])
    # Cap both rows and columns so prompt size stays predictable even for wide
    # datasets that would otherwise exceed LLM context limits.
    preview_frame = dataframe.loc[:, preview_columns].head(preview_rows).copy()
    preview_frame = preview_frame.where(pd.notna(preview_frame), None)

    records: list[dict[str, Any]] = []
    for record in preview_frame.to_dict(orient="records"):
        records.append(
            {
                str(column): _truncate_text_value(value, max_length=max_text_length)
                for column, value in record.items()
            }
        )

    return records, preview_columns, max(0, len(dataframe.columns) - len(preview_columns))


def build_flowise_profile_object(
    dataframe: pd.DataFrame,
    profile: dict[str, Any],
    ml_recommendation: dict[str, Any],
    *,
    target_column: str | None = None,
    cleaning_report: dict[str, Any] | None = None,
    file_name: str | None = None,
    max_rows: int = 10,
) -> dict[str, Any]:
    """Build the compact structured profile object sent to Flowise.

    The object intentionally contains Python-computed summaries instead of the
    full dataset. That lets the AI explain verified facts while avoiding
    hallucinated exact counts, wide-table prompt explosions, and unnecessary
    transmission of raw records.
    """
    profile = profile or {}
    ml_recommendation = ml_recommendation or {}
    cleaning_report = cleaning_report or {}

    original_file_name = file_name or "dataset.csv"
    dataset_name = Path(original_file_name).stem or "dataset"
    pandas_numpy_usage = cleaning_report.get("pandas_numpy_usage") or build_pandas_numpy_usage(
        original_file_name=original_file_name,
        profile=profile,
        cleaning_summary={},
    )
    sample_rows, sample_row_columns, omitted_sample_columns = _truncate_sample_rows(
        dataframe,
        max_rows=max_rows,
    )
    missing_value_summary = {
        str(column): int(count)
        for column, count in (profile.get("missing_values", {}) or {}).items()
        if int(count) > 0
    }
    cleaning_actions_performed = []
    cleaning_actions = cleaning_report.get("cleaning_actions", {})
    for action_name, details in cleaning_actions.items():
        if details.get("performed"):
            # Send only compact action summaries. The report on disk keeps the
            # full detail, while the LLM only needs enough context to explain
            # what happened.
            cleaning_actions_performed.append(
                {
                    "action": action_name,
                    "summary": {
                        key: value
                        for key, value in details.items()
                        if key in {"rows_removed", "columns_handled", "columns_encoded", "columns_scaled", "scaler_used"}
                    },
                }
            )

    skipped_cleaning_steps = cleaning_report.get("skipped_steps", [])

    return {
        "note": FLOWISE_PROFILE_NOTE,
        "dataset_name": dataset_name,
        "original_file_name": original_file_name,
        "shape": {
            "rows": int(profile.get("rows", len(dataframe))),
            "columns": int(profile.get("columns", len(dataframe.columns))),
        },
        "rows": int(profile.get("rows", len(dataframe))),
        "columns": int(profile.get("columns", len(dataframe.columns))),
        "column_names": _to_json_safe(profile.get("column_names", list(dataframe.columns))),
        "dtypes": _to_json_safe(profile.get("data_types", dataframe.dtypes.astype(str).to_dict())),
        "numeric_columns": _to_json_safe(profile.get("numeric_columns", [])),
        "categorical_columns": _to_json_safe(profile.get("categorical_columns", [])),
        "text_columns": _to_json_safe(profile.get("text_columns", [])),
        "datetime_columns": _to_json_safe(profile.get("datetime_columns", [])),
        "missing_value_summary": _to_json_safe(missing_value_summary),
        "total_missing_values": int(sum((profile.get("missing_values", {}) or {}).values())),
        "duplicate_count": int(profile.get("duplicate_rows", int(dataframe.duplicated().sum()))),
        "sample_rows": _to_json_safe(sample_rows),
        "sample_row_columns_included": sample_row_columns,
        "sample_row_omitted_column_count": omitted_sample_columns,
        "cleaning_actions_performed": _to_json_safe(cleaning_actions_performed),
        "skipped_cleaning_steps": _to_json_safe(skipped_cleaning_steps),
        "before_vs_after_summary": _to_json_safe(
            cleaning_report.get("before_vs_after_summary", {})
        ),
        "problem_type": ml_recommendation.get("recommended_problem_type"),
        "problem_type_reason": ml_recommendation.get("problem_type_reason"),
        "selected_target_column": ml_recommendation.get("selected_target_column", target_column),
        "suggested_target_column": ml_recommendation.get("suggested_target_column"),
        "recommended_algorithms": _to_json_safe(
            [
                {
                    "name": algorithm.get("name"),
                    "reason": algorithm.get("reason"),
                }
                for algorithm in ml_recommendation.get("algorithms", [])
            ]
        ),
        "recommendation_ready": bool(
            cleaning_report.get("recommendation_ready")
            or ml_recommendation.get("recommendation_ready", False)
        ),
        "ecommerce_preprocessing_applied": bool(
            cleaning_report.get("ecommerce_preprocessing_applied", False)
        ),
        "dropped_reference_columns": _to_json_safe(
            cleaning_report.get("dropped_reference_columns", [])
        ),
        "ai_limitations": (
            "The dataset may support future recommendation or ranking systems after preprocessing, "
            "but no recommendation model has been trained in this stage."
        ),
        "pandas_numpy_usage": _to_json_safe(pandas_numpy_usage),
        "preview_rows_sent": len(sample_rows),
        "full_dataset_sent_to_flowise": False,
    }


def build_flowise_dataset_summary(
    dataframe,
    profile,
    ml_recommendation,
    target_column=None,
    cleaning_report=None,
    file_name: str | None = None,
) -> str:
    """Build the compact structured summary string sent to Flowise."""
    summary = build_flowise_profile_object(
        dataframe,
        profile or {},
        ml_recommendation or {},
        target_column=target_column,
        cleaning_report=cleaning_report,
        file_name=file_name,
        max_rows=10,
    )
    return json.dumps(_to_json_safe(summary), indent=2, default=str)


def validate_flowise_profile_text(profile_text: str | None) -> tuple[bool, dict[str, Any]]:
    """Check whether the profile contains the minimum fields required for Flowise."""
    if not profile_text or not profile_text.strip():
        return False, {}

    try:
        profile_data = json.loads(profile_text)
    except json.JSONDecodeError:
        return False, {}

    has_shape = isinstance(profile_data.get("shape"), dict) and {
        "rows",
        "columns",
    }.issubset(profile_data["shape"].keys())
    has_columns = bool(profile_data.get("column_names"))
    has_sample_rows = isinstance(profile_data.get("sample_rows"), list)
    if not (has_shape and has_columns and has_sample_rows):
        return False, profile_data

    return True, profile_data


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
    """Build a compact profile preview for Flowise without sending raw CSV data.

    This is the main safety boundary for the AI layer: Python summarizes first,
    and Flowise sees only the summary. Full datasets are deliberately excluded
    to avoid token overflow and to keep exact statistics grounded in code.
    """
    if df is None:
        raise ValueError("No dataset is available to summarize for the AI agent.")
    if df.empty:
        raise ValueError("The uploaded dataset is empty, so no AI preview can be created.")
    return build_flowise_dataset_summary(
        df,
        profile or {},
        ml_recommendation or {},
        target_column=target_column,
        file_name=file_name,
        cleaning_report=cleaning_report,
    )


def build_flowise_combined_question(question: str, profile_text: str | None = None) -> str:
    """Build the actual prompt sent to Flowise with the profile embedded in the question."""
    cleaned_question = (question or "").strip()
    if not profile_text:
        return cleaned_question

    instruction_block = (
        "If Python-generated dataset profile is provided, use it as the source of truth. "
        "Do not ask for uploaded file content. Only ask for manual details if BOTH uploaded "
        "file content and Python-generated profile are missing.\n\n"
        "User question:\n"
        f"{cleaned_question}\n\n"
        "Python-generated dataset profile:\n"
        f"{profile_text}\n\n"
        "Important: Use this profile as the dataset source. The full raw CSV was not sent. "
        "If the dataset looks recommendation-ready, explain readiness only and do not claim a trained model exists."
    )
    return instruction_block


def build_flowise_request_payload(
    question: str,
    profile_text: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build the exact payload sent to Flowise and derived metadata.

    Metadata is captured alongside the payload so reports can show whether the
    AI explanation used a validated preview, how many preview rows were sent,
    and whether the request stayed within the no-full-dataset contract.
    """
    is_valid_profile, profile_data = validate_flowise_profile_text(profile_text)
    combined_question = build_flowise_combined_question(question, profile_text if is_valid_profile else None)
    preview_rows_sent = len(profile_data.get("sample_rows", [])) if profile_data else 0
    profile_keys_sent = sorted(profile_data.keys()) if profile_data else []

    payload = {
        "question": combined_question,
        "overrideConfig": {},
    }
    metadata = {
        "flowise_called": True,
        "profile_sent_to_flowise": is_valid_profile,
        "profile_keys_sent": profile_keys_sent,
        "full_dataset_sent_to_flowise": False,
        "preview_rows_sent": preview_rows_sent,
    }
    return payload, metadata


def query_flowise_agent(question: str, file_summary: str | None = None) -> dict:
    """Query Flowise for explanations using only summarized dataset context.

    This helper must not be used for actual validation, profiling, cleaning,
    or report generation. Python remains responsible for all data processing.
    Flowise is used only as an optional AI explanation layer.
    """
    payload, request_metadata = build_flowise_request_payload(question, file_summary)

    try:
        response = requests.post(FLOWISE_PREDICTION_URL, json=payload, timeout=60)
    except requests.Timeout:
        logger.warning("Flowise request timed out.")
        # Fallback/manual mode: when Flowise is unavailable, the app still has
        # the Python-generated profile, cleaning summary, and downloadable
        # report, so explanation failure does not block the main workflow.
        return {
            "success": False,
            "error": FLOWISE_UNAVAILABLE_MESSAGE,
            "metadata": {
                **request_metadata,
                "flowise_status": "error",
                "flowise_error_message": "Request timed out.",
            },
            "payload": payload,
        }
    except requests.ConnectionError:
        logger.warning("Flowise connection error.")
        return {
            "success": False,
            "error": FLOWISE_UNAVAILABLE_MESSAGE,
            "metadata": {
                **request_metadata,
                "flowise_status": "error",
                "flowise_error_message": "Connection failed.",
            },
            "payload": payload,
        }
    except requests.RequestException as exc:
        logger.warning("Flowise request error: %s", exc.__class__.__name__)
        return {
            "success": False,
            "error": FLOWISE_UNAVAILABLE_MESSAGE,
            "metadata": {
                **request_metadata,
                "flowise_status": "error",
                "flowise_error_message": str(exc),
            },
            "payload": payload,
        }
    except Exception as exc:
        logger.exception("Unexpected Flowise request failure.")
        return {
            "success": False,
            "error": FLOWISE_UNAVAILABLE_MESSAGE,
            "metadata": {
                **request_metadata,
                "flowise_status": "error",
                "flowise_error_message": str(exc),
            },
            "payload": payload,
        }

    if response.status_code != 200:
        logger.warning("Flowise returned non-200 status: %s", response.status_code)
        return {
            "success": False,
            "error": FLOWISE_UNAVAILABLE_MESSAGE,
            "metadata": {
                **request_metadata,
                "flowise_status": "error",
                "flowise_error_message": f"HTTP {response.status_code}",
            },
            "payload": payload,
        }

    try:
        response_json = response.json()
    except ValueError:
        logger.warning("Flowise returned invalid JSON.")
        return {
            "success": False,
            "error": FLOWISE_UNAVAILABLE_MESSAGE,
            "metadata": {
                **request_metadata,
                "flowise_status": "error",
                "flowise_error_message": "Invalid JSON response.",
            },
            "payload": payload,
        }

    return {
        "success": True,
        # Answers are post-processed so the UI always reminds users that the
        # response is based on a compact Python summary, not the raw file.
        "answer": extract_flowise_answer(response_json),
        "raw_response": response_json,
        "metadata": {
            **request_metadata,
            "flowise_status": "success",
            "flowise_error_message": None,
        },
        "payload": payload,
    }
