"""Utilities for safely loading uploaded datasets into pandas DataFrames."""

from __future__ import annotations

import pandas as pd


def load_dataset(uploaded_file) -> tuple[pd.DataFrame | None, str | None]:
    """Load a CSV or Excel upload and return the data plus an error message.

    The function is intentionally defensive so the Streamlit app can continue
    running even when a user uploads an invalid or unsupported file.
    """
    if uploaded_file is None:
        return None, "Please upload a CSV or Excel file to begin cleaning."

    file_name = uploaded_file.name.lower()

    try:
        # Reset the in-memory file pointer so repeated reads behave consistently.
        uploaded_file.seek(0)

        if file_name.endswith(".csv"):
            dataframe = pd.read_csv(uploaded_file)
        elif file_name.endswith((".xlsx", ".xls")):
            dataframe = pd.read_excel(uploaded_file)
        else:
            return None, "Unsupported file format. Please upload a CSV or Excel file."
    except pd.errors.EmptyDataError:
        return None, "The uploaded file is empty. Please upload a file that contains data."
    except UnicodeDecodeError:
        return None, (
            "The file could not be read as text. Please check that it is a valid CSV or Excel file."
        )
    except Exception as exc:
        return None, f"Could not read the uploaded file. Details: {exc}"

    return dataframe, None
