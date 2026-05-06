"""Core cleaning helpers for common tabular data issues."""

import pandas as pd


def clean_dataframe(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Apply a minimal starter cleaning pipeline to a DataFrame."""
    cleaned = dataframe.copy()

    # Remove exact duplicate rows as a safe default operation.
    cleaned = cleaned.drop_duplicates()

    # Standardize column names for easier downstream processing.
    cleaned.columns = [column.strip().lower().replace(" ", "_") for column in cleaned.columns]

    return cleaned
