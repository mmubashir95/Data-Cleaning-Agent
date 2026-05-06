"""Validation helpers for checking dataset structure before cleaning starts."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def validate_dataset(
    df: pd.DataFrame | None,
    uploaded_file_name: str,
    target_column: str | None = None,
) -> dict:
    """Validate an uploaded dataset before profiling or cleaning.

    This step is intentionally separate from profiling so the app can stop
    early when the dataset is invalid, while still allowing non-blocking
    warnings to pass through to later stages.
    """
    errors: list[str] = []
    warnings: list[str] = []

    supported_extensions = {".csv", ".xlsx", ".xls"}
    file_extension = Path(uploaded_file_name).suffix.lower()

    if file_extension not in supported_extensions:
        errors.append("Unsupported file type. Please upload a CSV or Excel file.")

    if df is None:
        errors.append("The dataset could not be loaded into a DataFrame.")
        return {"is_valid": False, "errors": errors, "warnings": warnings}

    if df.empty:
        errors.append("The uploaded dataset is empty.")

    if len(df.index) < 1:
        errors.append("The dataset must contain at least one row.")

    if len(df.columns) < 1:
        errors.append("The dataset must contain at least one column.")

    if df.columns.isna().any():
        errors.append("Some column names are missing.")

    normalized_column_names = [str(column).strip() for column in df.columns]
    if any(name == "" for name in normalized_column_names):
        errors.append("Some column names are blank.")

    if pd.Index(normalized_column_names).duplicated().any():
        errors.append("The dataset contains duplicate column names.")

    if target_column is not None and target_column not in df.columns:
        errors.append(f"Selected target column '{target_column}' was not found in the dataset.")

    # Warnings below do not block profiling or future cleaning steps.
    if len(df.index) == 1:
        warnings.append("The dataset contains only one row, which may be too small for analysis.")

    if len(df.columns) == 1:
        warnings.append("The dataset contains only one column.")

    missing_percentage = df.isna().mean() * 100
    all_missing_columns = missing_percentage[missing_percentage == 100].index.tolist()
    for column in all_missing_columns:
        warnings.append(f"Column '{column}' contains only missing values.")

    high_missing_columns = missing_percentage[
        (missing_percentage >= 50) & (missing_percentage < 100)
    ]
    for column, percentage in high_missing_columns.items():
        warnings.append(f"Column '{column}' has high missing values ({percentage:.1f}%).")

    row_count = len(df.index)
    if row_count > 0:
        object_columns = df.select_dtypes(include=["object", "category", "string"]).columns
        for column in object_columns:
            unique_ratio = df[column].nunique(dropna=True) / row_count
            if df[column].nunique(dropna=True) >= 50 or unique_ratio >= 0.8:
                warnings.append(
                    f"Column '{column}' has very high cardinality and may need special handling."
                )

    return {"is_valid": len(errors) == 0, "errors": errors, "warnings": warnings}
