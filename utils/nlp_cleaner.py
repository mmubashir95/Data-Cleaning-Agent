"""Text-specific cleaning helpers for free-form string columns."""

import pandas as pd


def normalize_text_columns(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Trim and normalize whitespace for object columns."""
    cleaned = dataframe.copy()

    for column in cleaned.select_dtypes(include="object").columns:
        cleaned[column] = cleaned[column].astype(str).str.strip()
        cleaned[column] = cleaned[column].str.replace(r"\s+", " ", regex=True)

    return cleaned
