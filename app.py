"""
Data Cleaning Agent for ML Dataset Preparation

Run instructions:
pip install -r requirements.txt
streamlit run app.py
"""

import streamlit as st

from utils.data_loader import load_dataset
from utils.data_profiler import profile_dataset


def build_sidebar():
    """Render the sidebar controls and return the uploaded file object."""
    st.sidebar.header("Dataset Settings")

    # Allow users to upload tabular datasets in common formats.
    uploaded_file = st.sidebar.file_uploader(
        "Upload a dataset",
        type=["csv", "xlsx", "xls"],
        help="Supported formats: CSV and Excel.",
    )

    # This dropdown will help guide future cleaning logic.
    st.sidebar.selectbox(
        "Problem type",
        options=[
            "Auto-detect",
            "Classification",
            "Regression",
            "Clustering",
            "NLP/Text Classification",
        ],
    )

    st.sidebar.subheader("Cleaning Options")

    # These checkboxes are UI placeholders for future cleaning steps.
    st.sidebar.checkbox("Remove duplicates")
    st.sidebar.checkbox("Handle missing values")
    st.sidebar.checkbox("Fix wrong data types")
    st.sidebar.checkbox("Encode categorical columns")
    st.sidebar.checkbox("Scale numeric columns")
    st.sidebar.checkbox("Handle outliers")
    st.sidebar.checkbox("NLP text cleaning")

    # The scaler choice is added now and can be used later during implementation.
    st.sidebar.subheader("Scaler Choice")
    st.sidebar.radio(
        "Choose a scaler",
        options=["StandardScaler", "MinMaxScaler"],
    )

    return uploaded_file


def render_uploaded_dataset(uploaded_file) -> None:
    """Display dataset details after a file has been uploaded."""
    dataframe, error_message = load_dataset(uploaded_file)

    # Show a readable error without breaking the app when loading fails.
    if error_message:
        st.error(error_message)
        return

    if dataframe is None:
        st.error("The dataset could not be loaded.")
        return

    st.success(f"Uploaded file: {uploaded_file.name}")
    st.write(f"Dataset shape: {dataframe.shape[0]} rows x {dataframe.shape[1]} columns")

    # Warn users about empty files while still keeping the app responsive.
    if dataframe.empty:
        st.warning("The uploaded dataset is empty. Please upload a file with data.")
        return

    # Let the user choose an optional target column before profiling so the
    # profiling summary can clearly mark it in the results.
    target_options = [None] + list(dataframe.columns)
    selected_target = st.selectbox(
        "Select target column (optional)",
        options=target_options,
        format_func=lambda value: "None" if value is None else value,
        help="Choose the target column you plan to predict or analyze.",
    )

    st.subheader("Dataset Preview")
    st.dataframe(dataframe.head())

    # Profiling currently runs immediately after loading. In a later phase,
    # this should run only after dataset validation passes successfully.
    profile = profile_dataset(dataframe, target_column=selected_target)

    st.subheader("Dataset Profiling")
    st.write(f"Rows: {profile['rows']}")
    st.write(f"Columns: {profile['columns']}")
    st.write(f"Target column: {profile['target_column']}")

    st.write("Column names:")
    st.write(profile["column_names"])

    st.write("Data types:")
    st.dataframe(
        {
            "Column": list(profile["data_types"].keys()),
            "Data Type": list(profile["data_types"].values()),
        },
        use_container_width=True,
    )

    st.write("Missing values:")
    st.dataframe(
        {
            "Column": list(profile["missing_values"].keys()),
            "Missing Values": list(profile["missing_values"].values()),
        },
        use_container_width=True,
    )

    st.write(f"Duplicate rows: {profile['duplicate_rows']}")
    st.write(f"Numeric columns: {profile['numeric_columns']}")
    st.write(f"Categorical columns: {profile['categorical_columns']}")
    st.write(f"Text/Object columns: {profile['text_columns']}")
    st.write(f"Datetime columns: {profile['datetime_columns']}")
    st.write(f"Boolean columns: {profile['boolean_columns']}")
    st.write(f"ID-like columns: {profile['id_like_columns']}")


def main() -> None:
    """Render the starter interface for the data cleaning app."""
    # Set the browser tab title and make the layout easier to read.
    st.set_page_config(
        page_title="Data Cleaning Agent for ML Dataset Preparation",
        layout="wide",
    )

    uploaded_file = build_sidebar()

    # Main page heading and short introduction.
    st.title("Data Cleaning Agent for ML Dataset Preparation")
    st.write(
        "Welcome to the Data Cleaning Agent. This app is designed to help you "
        "prepare machine learning datasets through a guided cleaning workflow."
    )

    # Simple explanation for new users.
    st.markdown(
        """
        This starter interface will eventually help you:
        - upload raw datasets
        - inspect common data quality issues
        - choose cleaning steps based on your ML problem
        - prepare data for modeling
        """
    )

    # Keep the landing page simple until the user uploads a dataset.
    if uploaded_file is None:
        st.info("Upload a CSV or Excel dataset from the sidebar to get started.")
        return

    render_uploaded_dataset(uploaded_file)


if __name__ == "__main__":
    main()
