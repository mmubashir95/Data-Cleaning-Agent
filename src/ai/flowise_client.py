"""Flowise API client used only for AI explanations, not data cleaning."""

from __future__ import annotations

from typing import Any

import requests

FLOWISE_PREDICTION_URL = (
    "https://cloud.flowiseai.com/api/v1/prediction/"
    "6a7b5277-b4bf-4a79-a785-8cde06dbf860"
)


def _extract_answer(response_json: Any) -> str:
    """Best-effort extraction of a readable answer from Flowise JSON."""
    if isinstance(response_json, dict):
        for key in ("text", "answer", "response", "output", "message"):
            value = response_json.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    if isinstance(response_json, str) and response_json.strip():
        return response_json.strip()
    return "Flowise returned a response, but no readable answer field was found."


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
        "answer": _extract_answer(response_json),
        "raw_response": response_json,
    }
