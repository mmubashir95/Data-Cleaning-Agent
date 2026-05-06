"""Profiling helpers for generating quick dataset summaries."""

from __future__ import annotations

import pandas as pd
from pandas.api.types import is_numeric_dtype


def profile_dataset(df: pd.DataFrame) -> dict:
    """Create a beginner-friendly profile summary for an uploaded dataset."""
    numeric_columns: list[str] = []
    categorical_columns: list[str] = []
    text_columns: list[str] = []

    for column in df.columns:
        series = df[column]

        # Numeric columns are detected using pandas dtype helpers.
        if is_numeric_dtype(series):
            numeric_columns.append(column)
            continue

        # Object and category columns need a simple heuristic split between
        # categorical data and free-form text.
        if pd.api.types.is_object_dtype(series) or pd.api.types.is_categorical_dtype(series):
            non_null_series = series.dropna().astype(str)
            unique_count = series.nunique(dropna=True)
            row_count = len(df)
            average_text_length = non_null_series.str.len().mean() if not non_null_series.empty else 0
            unique_ratio = (unique_count / row_count) if row_count else 0

            # Treat columns as text when values are longer on average or when
            # most rows are unique, which usually signals free-form content.
            if average_text_length > 30 or unique_ratio > 0.5:
                text_columns.append(column)
            else:
                categorical_columns.append(column)

    return {
        "rows": len(df),
        "columns": len(df.columns),
        "column_names": list(df.columns),
        "data_types": df.dtypes.astype(str).to_dict(),
        "missing_values": df.isna().sum().to_dict(),
        "duplicate_rows": int(df.duplicated().sum()),
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "text_columns": text_columns,
    }
