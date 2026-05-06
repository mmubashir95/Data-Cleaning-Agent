"""Validation helpers for checking dataset structure and basic quality."""

import pandas as pd


def validate_dataframe(dataframe: pd.DataFrame) -> dict:
    """Return basic validation results for the given DataFrame."""
    return {
        "is_empty": dataframe.empty,
        "row_count": len(dataframe),
        "column_count": len(dataframe.columns),
        "duplicate_rows": int(dataframe.duplicated().sum()),
        "missing_values": dataframe.isna().sum().to_dict(),
    }
