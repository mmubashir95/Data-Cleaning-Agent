"""
Data Cleaning Agent for ML Dataset Preparation

Run instructions:
pip install -r requirements.txt
streamlit run app.py
"""

import streamlit as st


def build_sidebar() -> None:
    """Render the sidebar controls for future cleaning options."""
    st.sidebar.header("Dataset Settings")

    # Let the user upload either CSV or Excel files.
    st.sidebar.file_uploader(
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


def main() -> None:
    """Render the starter interface for the data cleaning app."""
    # Set the browser tab title and make the layout easier to read.
    st.set_page_config(
        page_title="Data Cleaning Agent for ML Dataset Preparation",
        layout="wide",
    )

    build_sidebar()

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

    # Clear prompt for the next user action.
    st.info("Upload a CSV or Excel dataset from the sidebar to get started.")


if __name__ == "__main__":
    main()
