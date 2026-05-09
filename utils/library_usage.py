"""Helpers for explaining Pandas and NumPy usage in beginner-friendly terms."""

from __future__ import annotations

from typing import Any


def build_pandas_numpy_usage(
    *,
    original_file_name: str,
    profile: dict[str, Any],
    cleaning_summary: dict[str, Any],
) -> dict[str, Any]:
    """Describe only the Pandas and NumPy functions that were actually relevant."""
    pandas_functions: list[dict[str, str]] = []
    numpy_functions: list[dict[str, str]] = []

    if original_file_name.lower().endswith(".csv"):
        pandas_functions.append(
            {
                "function": "read_csv()",
                "why_used": "Used to load the uploaded CSV file into a DataFrame so the app can inspect and clean it.",
            }
        )

    pandas_functions.extend(
        [
            {
                "function": "head()",
                "why_used": "Used to show the first few rows so a beginner can quickly preview the dataset before and after cleaning.",
            },
            {
                "function": "info()",
                "why_used": "Used to summarize column names, non-null counts, and data types before cleaning.",
            },
            {
                "function": "isnull().sum()",
                "why_used": "Used to count missing values in each column before and after cleaning.",
            },
            {
                "function": "duplicated().sum()",
                "why_used": "Used to count duplicate rows so the app can report whether duplicates exist.",
            },
        ]
    )

    if cleaning_summary.get("duplicate_rows_removed", 0) > 0:
        pandas_functions.append(
            {
                "function": "drop_duplicates()",
                "why_used": "Used to remove repeated rows and keep only one copy of each duplicate record.",
            }
        )

    missing_filled = cleaning_summary.get("missing_filled", {})
    if missing_filled:
        strategies = sorted(
            {
                details.get("strategy", "unknown")
                for details in missing_filled.values()
                if details.get("strategy")
            }
        )
        strategy_text = ", ".join(strategies)
        pandas_functions.append(
            {
                "function": "fillna()",
                "why_used": "Used to replace missing values so the dataset becomes easier for ML models to use.",
            }
        )
        pandas_functions.append(
            {
                "function": "median()/mode()",
                "why_used": "Used to choose simple fill values for missing data. Numeric columns used the median, while categorical columns used the mode.",
            }
        )
        if "median" in strategies:
            numpy_functions.append(
                {
                    "function": "np.median()",
                    "why_used": "Used to calculate a robust middle value for numeric columns before filling missing entries.",
                }
            )
        if "mode" in strategies:
            pandas_functions.append(
                {
                    "function": "mode()",
                    "why_used": "Used to find the most common category when filling missing values in non-numeric columns.",
                }
            )

    converted_columns = cleaning_summary.get("converted_columns", {})
    if converted_columns:
        converted_targets = {details.get("converted_to") for details in converted_columns.values()}
        pandas_functions.append(
            {
                "function": "astype()",
                "why_used": "Used while checking and standardizing column values as text before deciding whether a safe type conversion is possible.",
            }
        )
        if "numeric" in converted_targets:
            pandas_functions.append(
                {
                    "function": "to_numeric()",
                    "why_used": "Used to convert numeric-looking text columns into real numeric columns for analysis and modeling.",
                }
            )

    if cleaning_summary.get("encoded_columns"):
        pandas_functions.append(
            {
                "function": "get_dummies()",
                "why_used": "Used to turn categorical labels into 0/1 columns so machine learning models can work with them.",
            }
        )

    outlier_summary = cleaning_summary.get("outlier_summary", [])
    if outlier_summary:
        pandas_functions.append(
            {
                "function": "quantile()",
                "why_used": "Used to calculate Q1 and Q3 so the app can apply the IQR rule for outlier detection.",
            }
        )
        pandas_functions.append(
            {
                "function": "clip()",
                "why_used": "Used to cap extreme values at the IQR bounds instead of deleting whole rows.",
            }
        )
        numpy_functions.append(
            {
                "function": "np.where()",
                "why_used": "Used to locate values outside the IQR bounds so the app can count and cap outliers safely.",
            }
        )

    if cleaning_summary.get("scaled_columns"):
        scaler_used = cleaning_summary.get("scaler_used")
        if scaler_used == "StandardScaler" and cleaning_summary.get("scaling_reference_stats"):
            numpy_functions.extend(
                [
                    {
                        "function": "np.mean()",
                        "why_used": "Used to inspect the average value of numeric columns before standard scaling.",
                    },
                    {
                        "function": "np.std()",
                        "why_used": "Used to inspect the spread of numeric columns before standard scaling.",
                    },
                ]
            )

    if not pandas_functions:
        pandas_functions.append(
            {
                "function": "No major Pandas actions recorded",
                "why_used": "The app loaded and inspected the dataset, but no additional Pandas cleaning steps were performed.",
            }
        )

    summary_parts = [
        f"Pandas functions recorded: {len(pandas_functions)}",
        f"NumPy functions recorded: {len(numpy_functions)}",
    ]
    if profile.get("rows") is not None and profile.get("columns") is not None:
        summary_parts.append(
            f"Dataset size reviewed: {profile['rows']} rows x {profile['columns']} columns"
        )

    return {
        "summary": ". ".join(summary_parts) + ".",
        "pandas_functions": pandas_functions,
        "numpy_functions": numpy_functions,
    }
