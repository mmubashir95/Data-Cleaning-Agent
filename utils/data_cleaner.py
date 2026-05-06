"""Core cleaning helpers for the first dataset cleaning workflow."""

from __future__ import annotations

from typing import Any

import pandas as pd

from utils.data_profiler import classify_columns


def _get_mode_or_unknown(series: pd.Series) -> Any:
    """Return the most common value, or a safe fallback when no mode exists."""
    mode = series.mode(dropna=True)
    if not mode.empty:
        return mode.iloc[0]
    return "Unknown"


def clean_dataset(
    df: pd.DataFrame,
    options: dict[str, bool],
    target_column: str | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Apply the selected starter cleaning steps to a validated dataset."""
    cleaned_df = df.copy()
    cleaning_steps: list[str] = []
    filled_columns: list[str] = []

    original_rows = len(cleaned_df)
    original_columns = len(cleaned_df.columns)
    missing_values_before = cleaned_df.isna().sum().to_dict()

    # Pre-cleaning validation is assumed to have already passed upstream.
    if options.get("remove_duplicates", False):
        cleaned_df = cleaned_df.drop_duplicates()
        duplicates_removed = original_rows - len(cleaned_df)
        cleaning_steps.append(f"Removed {duplicates_removed} duplicate rows.")
    else:
        duplicates_removed = 0
        cleaning_steps.append("Skipped duplicate removal.")

    if options.get("handle_missing_values", False):
        column_groups = classify_columns(cleaned_df, target_column=target_column)

        for column in column_groups["numeric_columns"]:
            if cleaned_df[column].isna().any():
                cleaned_df[column] = cleaned_df[column].fillna(cleaned_df[column].median())
                filled_columns.append(column)

        for column in column_groups["categorical_columns"]:
            if cleaned_df[column].isna().any():
                cleaned_df[column] = cleaned_df[column].fillna(_get_mode_or_unknown(cleaned_df[column]))
                filled_columns.append(column)

        for column in column_groups["text_columns"]:
            if cleaned_df[column].isna().any():
                cleaned_df[column] = cleaned_df[column].fillna("")
                filled_columns.append(column)

        if filled_columns:
            cleaning_steps.append(
                "Filled missing values in these columns: "
                + ", ".join(sorted(set(filled_columns)))
                + "."
            )
        else:
            cleaning_steps.append("No missing values needed filling in supported column groups.")
    else:
        cleaning_steps.append("Skipped missing value handling.")

    missing_values_after = cleaned_df.isna().sum().to_dict()

    cleaning_summary = {
        "original_rows": original_rows,
        "original_columns": original_columns,
        "final_rows": len(cleaned_df),
        "final_columns": len(cleaned_df.columns),
        "missing_values_before": missing_values_before,
        "missing_values_after": missing_values_after,
        "duplicate_rows_removed": duplicates_removed,
        "filled_missing_columns": filled_columns,
        "cleaning_steps": cleaning_steps,
    }

    return cleaned_df, cleaning_summary
