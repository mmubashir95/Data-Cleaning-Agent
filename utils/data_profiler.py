"""Profiling helpers for generating quick dataset summaries."""

import pandas as pd


def profile_dataframe(dataframe: pd.DataFrame) -> dict:
    """Build a lightweight profile for a DataFrame."""
    return {
        "columns": list(dataframe.columns),
        "dtypes": dataframe.dtypes.astype(str).to_dict(),
        "summary": dataframe.describe(include="all").fillna("").to_dict(),
    }
