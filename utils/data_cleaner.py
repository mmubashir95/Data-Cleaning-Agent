"""Core cleaning helpers for the first dataset cleaning workflow."""

from __future__ import annotations

from typing import Any

import pandas as pd


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
    cleaned_df = df.copy(deep=True)
    cleaning_steps: list[str] = []
    handle_missing_values_selected = options.get("handle_missing_values", False)
    missing_filled: dict[str, dict[str, str | int]] = {}

    original_rows = len(df)
    original_columns = len(df.columns)
    missing_values_before = df.isnull().sum().to_dict()

    # Pre-cleaning validation is assumed to have already passed upstream.
    if options.get("remove_duplicates", False):
        cleaned_df = cleaned_df.drop_duplicates()
        duplicates_removed = original_rows - len(cleaned_df)
        cleaning_steps.append(f"Removed {duplicates_removed} duplicate rows.")
    else:
        duplicates_removed = 0
        cleaning_steps.append("Skipped duplicate removal.")

    if handle_missing_values_selected:
        for column in cleaned_df.columns:
            if column == target_column:
                continue
            missing_count = int(cleaned_df[column].isna().sum())

            if missing_count == 0:
                continue

            if pd.api.types.is_numeric_dtype(cleaned_df[column]):
                median_value = cleaned_df[column].median()

                if pd.isna(median_value):
                    cleaned_df[column] = cleaned_df[column].fillna(0)
                    fill_value = 0
                else:
                    cleaned_df[column] = cleaned_df[column].fillna(median_value)
                    fill_value = median_value

                missing_filled[column] = {
                    "missing_count_before": missing_count,
                    "strategy": "median",
                    "fill_value": str(fill_value),
                }
            else:
                mode_values = cleaned_df[column].mode(dropna=True)

                if not mode_values.empty:
                    fill_value = mode_values.iloc[0]
                    cleaned_df[column] = cleaned_df[column].fillna(fill_value)
                    strategy = "mode"
                else:
                    fill_value = "Unknown"
                    cleaned_df[column] = cleaned_df[column].fillna(fill_value)
                    strategy = "Unknown"

                missing_filled[column] = {
                    "missing_count_before": missing_count,
                    "strategy": strategy,
                    "fill_value": str(fill_value),
                }

        if missing_filled:
            filled_columns = sorted(missing_filled.keys())
            cleaning_steps.append(
                "Filled missing values in these columns: " + ", ".join(filled_columns) + "."
            )
        else:
            filled_columns = []
            cleaning_steps.append("No missing values needed filling.")
    else:
        filled_columns = []
        cleaning_steps.append(
            "Missing value handling was not selected, so missing values were not changed."
        )

    missing_values_after = cleaned_df.isnull().sum().to_dict()

    cleaning_summary = {
        "original_rows": original_rows,
        "original_columns": original_columns,
        "final_rows": len(cleaned_df),
        "final_columns": len(cleaned_df.columns),
        "missing_values_before": missing_values_before,
        "missing_values_after": missing_values_after,
        "missing_filled": missing_filled,
        "duplicate_rows_removed": duplicates_removed,
        "columns_where_missing_values_were_filled": filled_columns,
        "options_used": options.copy(),
        "cleaning_steps": cleaning_steps,
    }

    return cleaned_df, cleaning_summary
