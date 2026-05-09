"""Profiling helpers for generating reusable dataset summaries."""

from __future__ import annotations

import io
from typing import Any

import pandas as pd
from pandas.api.types import (
    is_bool_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype,
    is_object_dtype,
    is_string_dtype,
)


def _normalize_boolean_token(value: Any) -> str:
    """Normalize values so common boolean-like tokens can be compared safely."""
    return str(value).strip().lower()


def _is_boolean_like(series: pd.Series) -> bool:
    """Detect native booleans and common two-value boolean-style columns."""
    if is_bool_dtype(series):
        return True

    non_null_values = series.dropna()
    if non_null_values.empty:
        return False

    unique_values = {_normalize_boolean_token(value) for value in non_null_values.unique()}
    if len(unique_values) != 2:
        return False

    boolean_token_sets = [
        {"0", "1"},
        {"true", "false"},
        {"yes", "no"},
        {"y", "n"},
        {"t", "f"},
    ]
    return any(unique_values == token_set for token_set in boolean_token_sets)


def _looks_datetime(series: pd.Series, column_name: str) -> bool:
    """Check whether a column should be treated as datetime-like.

    Detection combines naming hints with parsing success so the profiler avoids
    over-classifying arbitrary text as dates.
    """
    if is_datetime64_any_dtype(series):
        return True

    if not (is_object_dtype(series) or is_string_dtype(series)):
        return False

    non_null_values = series.dropna()
    if len(non_null_values) < 2:
        return False

    sample = non_null_values.astype(str).head(100)
    sample_text = " ".join(sample.tolist()).lower()
    column_name = column_name.lower()

    has_datetime_hint = any(
        token in column_name
        for token in ["date", "time", "timestamp", "created", "updated", "month", "year"]
    )
    has_date_format_hint = any(symbol in sample_text for symbol in ["-", "/", ":"])

    if not (has_datetime_hint or has_date_format_hint):
        return False

    parsed = pd.to_datetime(sample, format="mixed", errors="coerce")
    parse_ratio = parsed.notna().mean()
    return parse_ratio >= 0.7


def _looks_numeric_like(series: pd.Series) -> bool:
    """Check whether an object column is mostly numeric despite dirty values.

    This allows profiling and cleaning to recognize numeric intent in common
    exported datasets where numbers arrive as strings.
    """
    if not (is_object_dtype(series) or is_string_dtype(series)):
        return False

    non_null_values = series.dropna()
    if non_null_values.empty:
        return False

    converted = pd.to_numeric(non_null_values.astype(str).str.strip(), errors="coerce")
    success_ratio = converted.notna().mean()
    return success_ratio >= 0.7


def _is_id_like(series: pd.Series, column_name: str, row_count: int) -> bool:
    """Detect identifier-style columns using naming and uniqueness heuristics."""
    normalized_name = column_name.lower()
    name_suggests_id = any(
        token in normalized_name for token in ["id", "uuid", "key", "code", "identifier"]
    )
    if name_suggests_id:
        return True

    non_null_values = series.dropna()
    if non_null_values.empty or row_count == 0:
        return False

    unique_count = series.nunique(dropna=True)
    unique_ratio = unique_count / row_count

    # Avoid treating free-form text like names or comments as identifiers.
    text_values = non_null_values.astype(str)
    average_length = text_values.str.len().mean()
    whitespace_ratio = text_values.str.contains(r"\s", regex=True).mean()

    return (
        unique_ratio >= 0.95
        and unique_count >= min(20, row_count)
        and average_length <= 24
        and whitespace_ratio <= 0.2
    )


def classify_columns(df: pd.DataFrame, target_column: str | None = None) -> dict[str, Any]:
    """Classify dataset columns into reusable semantic groups.

    Thresholds are intentionally kept near the top of the function so they can
    be adjusted later as more datasets are tested.
    """
    # These heuristics are intentionally simple and explainable. The goal is to
    # support a broad classroom/project dataset mix without hiding decisions
    # behind opaque scoring models.
    text_unique_ratio_threshold = 0.5
    text_average_length_threshold = 25
    text_max_length_threshold = 60
    categorical_unique_ratio_threshold = 0.2
    categorical_unique_count_threshold = 30
    moderate_unique_count_threshold = 50

    row_count = len(df)

    classification: dict[str, Any] = {
        "numeric_columns": [],
        "categorical_columns": [],
        "text_columns": [],
        "datetime_columns": [],
        "boolean_columns": [],
        "id_like_columns": [],
        "target_column": target_column,
    }

    if df.empty or len(df.columns) == 0:
        return classification

    for column in df.columns:
        series = df[column]

        if _looks_datetime(series, column):
            classification["datetime_columns"].append(column)
            continue

        if _is_boolean_like(series):
            classification["boolean_columns"].append(column)
            if is_numeric_dtype(series):
                classification["numeric_columns"].append(column)
            continue

        if _looks_numeric_like(series):
            classification["numeric_columns"].append(column)
            continue

        if is_numeric_dtype(series):
            classification["numeric_columns"].append(column)
            if _is_id_like(series, column, row_count):
                classification["id_like_columns"].append(column)
            continue

        if is_object_dtype(series) or is_string_dtype(series) or isinstance(
            series.dtype, pd.CategoricalDtype
        ):
            non_null_values = series.dropna().astype(str)
            unique_count = series.nunique(dropna=True)
            unique_ratio = (unique_count / row_count) if row_count else 0
            average_length = non_null_values.str.len().mean() if not non_null_values.empty else 0
            max_length = non_null_values.str.len().max() if not non_null_values.empty else 0

            if _is_id_like(series, column, row_count):
                classification["id_like_columns"].append(column)

            # Treat repeated low-cardinality values as categorical features so
            # downstream cleaning and algorithm selection remain beginner-safe.
            if (
                unique_count <= categorical_unique_count_threshold
                or (
                    unique_ratio <= categorical_unique_ratio_threshold
                    and unique_count <= moderate_unique_count_threshold
                )
            ):
                classification["categorical_columns"].append(column)
                continue

            # Long or highly unique free-form values should be preserved as text
            # because collapsing them into categories usually destroys meaning.
            if (
                unique_ratio >= text_unique_ratio_threshold
                or average_length >= text_average_length_threshold
                or max_length >= text_max_length_threshold
            ):
                classification["text_columns"].append(column)
                continue

            classification["categorical_columns"].append(column)

    return classification


def profile_dataset(df: pd.DataFrame, target_column: str | None = None) -> dict[str, Any]:
    """Create a profile summary that includes shape, quality, and column classes.

    The resulting profile is the shared source of truth for the UI, reports,
    ML recommendation logic, and the compact Flowise preview.
    """
    classification = classify_columns(df, target_column=target_column)
    info_buffer = io.StringIO()
    df.info(buf=info_buffer)

    profile: dict[str, Any] = {
        "rows": len(df),
        "columns": len(df.columns),
        "column_names": list(df.columns),
        "data_types": df.dtypes.astype(str).to_dict(),
        "dataframe_info": info_buffer.getvalue(),
        "missing_values": df.isna().sum().to_dict(),
        "duplicate_rows": int(df.duplicated().sum()),
    }
    profile.update(classification)
    return profile
