"""
Data Cleaning Agent for ML Dataset Preparation

Run instructions:
pip install -r requirements.txt
streamlit run app.py
"""

import pandas as pd
import streamlit as st

from utils.data_cleaner import clean_dataset
from utils.data_loader import load_dataset
from utils.ml_recommender import recommend_ml_approach
from utils.data_profiler import profile_dataset
from utils.data_validator import validate_dataset


def build_sidebar():
    """Render the sidebar controls and return the selected inputs."""
    st.sidebar.header("Dataset Settings")

    # Allow users to upload tabular datasets in common formats.
    uploaded_file = st.sidebar.file_uploader(
        "Upload a dataset",
        type=["csv", "xlsx", "xls"],
        help="Supported formats: CSV and Excel.",
    )

    # This dropdown will help guide future cleaning logic.
    problem_type = st.sidebar.selectbox(
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

    # Store each checkbox in a named variable so the keys passed to the cleaner
    # are explicit and easy to debug.
    remove_duplicates = st.sidebar.checkbox("Remove duplicates")
    handle_missing_values = st.sidebar.checkbox("Handle missing values")
    fix_data_types = st.sidebar.checkbox("Fix wrong data types")
    encode_categorical = st.sidebar.checkbox("Encode categorical columns")
    scale_numeric = st.sidebar.checkbox("Scale numeric columns")
    handle_outliers = st.sidebar.checkbox("Handle outliers")
    nlp_cleaning = st.sidebar.checkbox("NLP text cleaning")

    scaler_choice = None
    if scale_numeric:
        st.sidebar.subheader("Scaler Choice")
        scaler_choice = st.sidebar.radio(
            "Choose a scaler",
            options=["StandardScaler", "MinMaxScaler"],
        )

    cleaning_options = {
        "remove_duplicates": remove_duplicates,
        "handle_missing_values": handle_missing_values,
        "fix_data_types": fix_data_types,
        "handle_outliers": handle_outliers,
        "encode_categorical": encode_categorical,
        "scale_numeric": scale_numeric,
        "scaler_choice": scaler_choice,
        "nlp_cleaning": nlp_cleaning,
    }

    return uploaded_file, problem_type, cleaning_options, scaler_choice


def render_uploaded_dataset(
    uploaded_file,
    selected_problem_type: str,
    cleaning_options: dict[str, bool],
) -> None:
    """Display dataset details after a file has been uploaded."""
    # Execution order for the workflow should remain:
    # load dataset -> validate -> profile -> clean
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

    # This is the dedicated pre-cleaning validation step. Profiling and any
    # later cleaning steps must stop when blocking validation errors exist.
    validation_result = validate_dataset(
        dataframe,
        uploaded_file_name=uploaded_file.name,
        target_column=selected_target,
    )

    st.subheader("Pre-Cleaning Validation")

    for warning_message in validation_result["warnings"]:
        st.warning(warning_message)

    if validation_result["errors"]:
        for error_message in validation_result["errors"]:
            st.error(error_message)
        return

    st.success("Validation passed. Profiling can continue.")

    # Profiling only runs after validation passes successfully.
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

    ml_recommendation = recommend_ml_approach(
        dataframe,
        target_column=selected_target,
        problem_type=selected_problem_type,
        text_columns=profile["text_columns"],
    )

    st.subheader("ML Recommendation")
    st.write(
        f"Recommended problem type: {ml_recommendation['recommended_problem_type']}"
    )
    st.write(f"Reason: {ml_recommendation['reason']}")

    for warning_message in ml_recommendation["warnings"]:
        st.warning(warning_message)

    st.write("Recommended algorithms:")
    for algorithm in ml_recommendation["algorithms"]:
        st.write(f"- {algorithm['name']}: {algorithm['reason']}")

    # Cleaning runs only after the dataset has passed pre-cleaning validation.
    if st.button("Clean Dataset"):
        cleaned_df, cleaning_summary = clean_dataset(
            dataframe,
            options=cleaning_options,
            target_column=selected_target,
        )

        st.subheader("Cleaned Dataset Preview")
        st.dataframe(cleaned_df.head())
        st.download_button(
            "Download Cleaned CSV",
            data=cleaned_df.to_csv(index=False).encode("utf-8"),
            file_name="cleaned_dataset.csv",
            mime="text/csv",
        )

        st.subheader("Cleaning Summary")
        st.write(f"Original rows: {cleaning_summary['original_rows']}")
        st.write(f"Original columns: {cleaning_summary['original_columns']}")
        st.write(f"Final rows: {cleaning_summary['final_rows']}")
        st.write(f"Final columns: {cleaning_summary['final_columns']}")
        st.write(f"Duplicate rows removed: {cleaning_summary['duplicate_rows_removed']}")
        st.write(
            "Missing value handling selected: "
            f"{cleaning_summary['options_used']['handle_missing_values']}"
        )
        st.write(
            "Fix wrong data types selected: "
            f"{cleaning_summary['options_used']['fix_data_types']}"
        )
        st.write(
            "Handle outliers selected: "
            f"{cleaning_summary['options_used']['handle_outliers']}"
        )
        st.write(
            "Encode categorical columns selected: "
            f"{cleaning_summary['options_used']['encode_categorical']}"
        )
        st.write(
            "Scale numeric columns selected: "
            f"{cleaning_summary['options_used']['scale_numeric']}"
        )
        st.write(
            "NLP text cleaning selected: "
            f"{cleaning_summary['options_used']['nlp_cleaning']}"
        )
        st.write("Scaler used:", cleaning_summary.get("scaler_used"))
        st.write(
            "Columns where missing values were filled: "
            f"{cleaning_summary['columns_where_missing_values_were_filled']}"
        )
        st.write("Converted columns:", cleaning_summary.get("converted_columns", {}))
        st.write(
            "Converted numeric columns:",
            cleaning_summary.get("converted_numeric_columns", []),
        )
        st.write(
            "Converted date columns:",
            cleaning_summary.get("converted_date_columns", []),
        )
        st.write(
            "Skipped type conversion columns:",
            cleaning_summary.get("skipped_type_conversion_columns", {}),
        )
        st.write("Outlier summary:", cleaning_summary.get("outlier_summary", []))
        st.write("Encoded source columns:", cleaning_summary.get("encoded_columns", []))
        st.write(
            "New encoded columns count:",
            cleaning_summary.get("encoded_columns_generated_count", 0),
        )
        st.write("Cleaned text columns:", cleaning_summary.get("cleaned_text_columns", []))
        st.write(
            "Original text backup columns:",
            cleaning_summary.get("nlp_original_backup_columns", []),
        )
        st.write("Scaled numeric columns:", cleaning_summary.get("scaled_columns", []))

        if cleaning_summary.get("encoded_columns"):
            st.info(
                "ML models need numeric input, so text categories are converted into 0/1 columns."
            )

        if cleaning_summary.get("target_encoding_recommendation"):
            st.info(cleaning_summary["target_encoding_recommendation"])

        if cleaning_summary.get("converted_columns"):
            st.info(
                "Type conversion helps ML because numeric and datetime columns are easier to clean, validate, and transform."
            )

        if cleaning_summary.get("outlier_summary"):
            st.info(
                "Outliers were capped with the IQR method. Extreme values should not be deleted blindly, because they may still contain useful signal."
            )

        if cleaning_summary.get("scaled_columns"):
            st.info(
                "Scaling helps algorithms like Logistic Regression, Linear Regression, KNN, SVM, and Neural Networks."
            )

        if cleaning_summary.get("cleaned_text_columns"):
            st.info(
                "Cleaned text can later be converted into numeric features using TF-IDF or Bag-of-Words."
            )
            st.write("NLP cleaning actions:", cleaning_summary.get("nlp_cleaning_actions", []))
            st.write("NLP before vs after examples:")
            for column_name, example in cleaning_summary.get("nlp_before_after_examples", {}).items():
                st.write(f"{column_name} before: {example['before']}")
                st.write(f"{column_name} after: {example['after']}")

        st.write("DEBUG - Options passed to cleaner:", cleaning_summary["options_used"])
        st.write("DEBUG - Missing filled columns:", cleaning_summary.get("missing_filled", {}))
        st.write("DEBUG - cleaned_df missing values:", cleaned_df.isna().sum().to_dict())

        if not cleaning_summary["options_used"]["handle_missing_values"]:
            st.info(
                "Missing value handling was not selected, so missing values were not changed."
            )

        st.write("Missing values before cleaning:")
        st.dataframe(
            pd.DataFrame(
                {
                "Column": list(cleaning_summary["missing_values_before"].keys()),
                "Missing Values": list(cleaning_summary["missing_values_before"].values()),
                }
            ),
            use_container_width=True,
        )

        st.write("Missing values after cleaning:")
        st.dataframe(
            pd.DataFrame(
                {
                "Column": list(cleaning_summary["missing_values_after"].keys()),
                "Missing Values": list(cleaning_summary["missing_values_after"].values()),
                }
            ),
            use_container_width=True,
        )

        st.write("Step-by-step explanation:")
        for step in cleaning_summary["cleaning_steps"]:
            st.write(f"- {step}")

        st.write("Type conversion notes:")
        for note in cleaning_summary.get("type_conversion_notes", []):
            st.write(f"- {note}")


def main() -> None:
    """Render the starter interface for the data cleaning app."""
    # Set the browser tab title and make the layout easier to read.
    st.set_page_config(
        page_title="Data Cleaning Agent for ML Dataset Preparation",
        layout="wide",
    )

    uploaded_file, selected_problem_type, cleaning_options, _ = build_sidebar()

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

    render_uploaded_dataset(uploaded_file, selected_problem_type, cleaning_options)


if __name__ == "__main__":
    main()
