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


def _should_convert_to_numeric(series: pd.Series) -> bool:
    """Check whether an object column looks safely convertible to numeric."""
    non_null = series.dropna()
    if non_null.empty:
        return False

    as_text = non_null.astype(str).str.strip()
    converted = pd.to_numeric(as_text, errors="coerce")
    success_ratio = converted.notna().mean()
    return success_ratio >= 0.7


def _should_convert_to_datetime(series: pd.Series, column_name: str) -> bool:
    """Check whether an object column looks safely convertible to datetime."""
    non_null = series.dropna()
    if len(non_null) < 2:
        return False

    sample = non_null.astype(str).str.strip().head(100)
    column_name = column_name.lower()
    has_datetime_hint = any(
        token in column_name
        for token in ["date", "time", "timestamp", "created", "updated", "month", "year"]
    )
    has_format_hint = sample.str.contains(r"[-/:]", regex=True).mean() >= 0.5

    if not (has_datetime_hint or has_format_hint):
        return False

    converted = pd.to_datetime(sample, errors="coerce")
    success_ratio = converted.notna().mean()
    return success_ratio >= 0.7


def clean_dataset(
    df: pd.DataFrame,
    options: dict[str, bool],
    target_column: str | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Apply the selected starter cleaning steps to a validated dataset."""
    cleaned_df = df.copy(deep=True)
    cleaning_steps: list[str] = []
    handle_missing_values_selected = options.get("handle_missing_values", False)
    fix_data_types_selected = options.get("fix_data_types", False)
    handle_outliers_selected = options.get("handle_outliers", False)
    missing_filled: dict[str, dict[str, str | int]] = {}
    converted_columns: dict[str, dict[str, str]] = {}
    converted_numeric_columns: list[str] = []
    converted_date_columns: list[str] = []
    skipped_type_conversion_columns: dict[str, str] = {}
    type_conversion_notes: list[str] = []
    outlier_summary: list[dict[str, str | float | int]] = []

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

    if fix_data_types_selected:
        for column in cleaned_df.columns:
            if column == target_column:
                continue

            series = cleaned_df[column]
            if not pd.api.types.is_object_dtype(series):
                continue

            try:
                if _should_convert_to_numeric(series):
                    cleaned_df[column] = pd.to_numeric(series, errors="coerce")
                    converted_columns[column] = {
                        "converted_to": "numeric",
                        "reason": "Numeric types help machine learning models work with quantitative values.",
                    }
                    converted_numeric_columns.append(column)
                    continue

                if _should_convert_to_datetime(series, column):
                    cleaned_df[column] = pd.to_datetime(series, errors="coerce")
                    converted_columns[column] = {
                        "converted_to": "datetime",
                        "reason": "Datetime conversion makes time-based features easier to clean and engineer for ML.",
                    }
                    converted_date_columns.append(column)
                else:
                    skipped_type_conversion_columns[column] = (
                        "Skipped because the column does not look reliably numeric or date-like."
                    )
            except Exception:
                # Skip unsafe conversions so the app continues without crashing.
                skipped_type_conversion_columns[column] = (
                    "Skipped because safe type conversion failed for this column."
                )
                continue

        if converted_columns:
            converted_list = ", ".join(sorted(converted_columns.keys()))
            cleaning_steps.append(f"Converted column data types for: {converted_list}.")
            cleaning_steps.append(
                "Type conversion helps ML because numeric and datetime values are easier to validate, clean, and transform."
            )
            if converted_numeric_columns:
                type_conversion_notes.append(
                    "Converted numeric-like text columns to numeric so models can use them as quantitative features."
                )
            if converted_date_columns:
                type_conversion_notes.append(
                    "Converted date-like text columns to datetime so time information can be handled more consistently."
                )
        else:
            cleaning_steps.append("No safe data type conversions were detected.")
            type_conversion_notes.append("No columns met the safety threshold for type conversion.")
    else:
        cleaning_steps.append("Skipped wrong data type fixing.")
        type_conversion_notes.append("Wrong data type fixing was not selected.")

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

    if handle_outliers_selected:
        numeric_columns = cleaned_df.select_dtypes(include=["number"]).columns

        for column in numeric_columns:
            if column == target_column:
                continue

            series = cleaned_df[column].dropna()
            if len(series) < 4:
                continue

            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1

            if pd.isna(iqr) or iqr == 0:
                continue

            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            outlier_mask = (cleaned_df[column] < lower_bound) | (cleaned_df[column] > upper_bound)
            outlier_count = int(outlier_mask.sum())

            if outlier_count == 0:
                continue

            cleaned_df[column] = cleaned_df[column].clip(lower=lower_bound, upper=upper_bound)
            outlier_summary.append(
                {
                    "column_name": column,
                    "lower_bound": float(lower_bound),
                    "upper_bound": float(upper_bound),
                    "outliers_detected": outlier_count,
                    "action_taken": "capped/winsorized",
                }
            )

        if outlier_summary:
            capped_columns = ", ".join(item["column_name"] for item in outlier_summary)
            cleaning_steps.append(f"Capped outliers using the IQR method for: {capped_columns}.")
            cleaning_steps.append(
                "Outliers were capped instead of deleting rows, because extreme values should not be removed blindly."
            )
        else:
            cleaning_steps.append("No numeric outliers were capped with the IQR method.")
    else:
        cleaning_steps.append("Skipped outlier handling.")

    missing_values_after = cleaned_df.isnull().sum().to_dict()

    cleaning_summary = {
        "original_rows": original_rows,
        "original_columns": original_columns,
        "final_rows": len(cleaned_df),
        "final_columns": len(cleaned_df.columns),
        "missing_values_before": missing_values_before,
        "missing_values_after": missing_values_after,
        "missing_filled": missing_filled,
        "converted_columns": converted_columns,
        "converted_numeric_columns": sorted(converted_numeric_columns),
        "converted_date_columns": sorted(converted_date_columns),
        "skipped_type_conversion_columns": skipped_type_conversion_columns,
        "type_conversion_notes": type_conversion_notes,
        "duplicate_rows_removed": duplicates_removed,
        "columns_where_missing_values_were_filled": filled_columns,
        "outlier_summary": outlier_summary,
        "options_used": options.copy(),
        "cleaning_steps": cleaning_steps,
    }

    return cleaned_df, cleaning_summary
