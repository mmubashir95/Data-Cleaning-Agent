"""
Data Cleaning Agent for ML Dataset Preparation

Run instructions:
pip install -r requirements.txt
streamlit run app.py
"""

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from src.ai.flowise_client import (
    build_flowise_file_preview,
    query_flowise_agent,
)
from utils.data_cleaner import clean_dataset
from utils.data_loader import load_dataset
from utils.data_profiler import profile_dataset
from utils.data_validator import validate_dataset
from utils.ml_recommender import recommend_ml_approach
from utils.report_generator import generate_cleaning_report, make_safe_stem

FLOWISE_CONTEXT_INSTRUCTION = (
    "Important: the file field contains a compact text preview of the dataset "
    "produced by the Python app, not the raw CSV or Excel file. Do not ask "
    "for the dataset again. Do not request the full file. Use the provided "
    "preview directly to answer. If something is missing, make a reasonable "
    "inference and clearly label it as an inference."
)

PROMPT_TEMPLATES = {
    "Full 25/25 Midterm Answer": (
        "Please give me a full 25/25 midterm answer covering Part A: Agent "
        "Development in Flowise - 10 Marks and Part B: Case Study Analysis - "
        "15 Marks. Use the provided dataset profile."
    ),
    "Part A Only": (
        "Please answer only Part A: Flowise Agent Development - 10 Marks "
        "using this dataset profile."
    ),
    "Part B Only": (
        "Please answer only Part B: Case Study Analysis - 15 Marks. "
        "Answer all 5 case study questions in an exam-focused way."
    ),
    "Explain Cleaning Steps": (
        "Explain the required data cleaning steps for this dataset. For each "
        "step, include the issue, reason, method, Pandas/NumPy function, and "
        "expected result."
    ),
    "Recommend Algorithms": (
        "Recommend suitable ML algorithms for this dataset after cleaning. "
        "Explain why each algorithm matches the problem type and target variable."
    ),
    "Short Summary": (
        "Give me a short summary of dataset type, ML problem type, main "
        "cleaning issues, final ML-ready output, and recommended algorithms."
    ),
}

CUSTOM_PROMPT_OPTION = "Custom Prompt / Ask Anything"


def build_sidebar():
    """Render the sidebar controls and return the selected inputs."""
    st.sidebar.header("Dataset Settings")

    uploaded_file = st.sidebar.file_uploader(
        "Upload a dataset",
        type=["csv", "xlsx", "xls"],
        help="Supported formats: CSV and Excel.",
    )

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
    remove_duplicates = st.sidebar.checkbox("Remove duplicates")
    handle_missing_values = st.sidebar.checkbox("Handle missing values")
    fix_data_types = st.sidebar.checkbox("Fix wrong data types")
    encode_categorical = st.sidebar.checkbox("Encode categorical columns")
    scale_numeric = st.sidebar.checkbox("Scale numeric columns")
    handle_outliers = st.sidebar.checkbox("Handle outliers")
    nlp_cleaning = st.sidebar.checkbox("NLP text cleaning")

    scaler_choice = None
    if scale_numeric:
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

    return uploaded_file, problem_type, cleaning_options


def render_cleaning_options_summary(cleaning_options: dict[str, bool]) -> None:
    """Show the selected cleaning settings on the main page."""
    st.subheader("6. Cleaning Options")

    selected_options = [
        ("Remove duplicates", cleaning_options.get("remove_duplicates", False)),
        ("Handle missing values", cleaning_options.get("handle_missing_values", False)),
        ("Fix wrong data types", cleaning_options.get("fix_data_types", False)),
        ("Encode categorical columns", cleaning_options.get("encode_categorical", False)),
        ("Scale numeric columns", cleaning_options.get("scale_numeric", False)),
        ("Handle outliers", cleaning_options.get("handle_outliers", False)),
        ("NLP text cleaning", cleaning_options.get("nlp_cleaning", False)),
    ]

    enabled_options = [label for label, is_enabled in selected_options if is_enabled]
    if enabled_options:
        st.info("Selected cleaning steps: " + ", ".join(enabled_options))
    else:
        st.info("No cleaning steps selected yet. You can still review the dataset before cleaning.")

    if cleaning_options.get("scale_numeric"):
        st.write(f"Scaler selected: {cleaning_options.get('scaler_choice')}")


def render_dataset_profile(profile: dict) -> None:
    """Show a beginner-friendly dataset profiling section before cleaning."""
    st.subheader("3. Dataset Profile")

    metric_columns = st.columns(3)
    metric_columns[0].metric("Rows", profile["rows"])
    metric_columns[1].metric("Columns", profile["columns"])
    metric_columns[2].metric("Duplicate Rows", profile["duplicate_rows"])

    st.write("Column names:")
    st.write(profile["column_names"])

    st.write(f"Numeric columns: {profile['numeric_columns']}")
    st.write(f"Categorical columns: {profile['categorical_columns']}")
    st.write(f"Text/Object columns: {profile['text_columns']}")
    st.write(f"Datetime columns: {profile['datetime_columns']}")
    st.write(f"Boolean columns: {profile['boolean_columns']}")
    st.write(f"ID-like columns: {profile['id_like_columns']}")

    with st.expander("View detailed dataset profile"):
        st.write("Data types:")
        st.dataframe(
            {
                "Column": list(profile["data_types"].keys()),
                "Data Type": list(profile["data_types"].values()),
            },
            width="stretch",
        )

        st.write("Missing values:")
        st.dataframe(
            {
                "Column": list(profile["missing_values"].keys()),
                "Missing Values": list(profile["missing_values"].values()),
            },
            width="stretch",
        )

    if any(value > 0 for value in profile["missing_values"].values()):
        st.info("Some columns have missing values. Review them before cleaning.")
    else:
        st.info("No missing values were detected in this dataset.")


def _render_missing_values_chart(dataframe: pd.DataFrame) -> None:
    """Render a missing values bar chart or a friendly skip message."""
    missing_counts = dataframe.isnull().sum()
    missing_counts = missing_counts[missing_counts > 0].sort_values(ascending=False)

    with st.expander("Missing Values Bar Chart"):
        if missing_counts.empty:
            st.info("Skipped missing values chart because no missing values were detected before cleaning.")
            return

        missing_chart_data = (
            missing_counts.rename_axis("column")
            .reset_index(name="missing_values")
        )
        st.bar_chart(
            missing_chart_data.set_index("column")["missing_values"],
        )
        st.caption("This chart highlights which columns have the most missing values before cleaning.")


def _render_target_distribution_chart(
    dataframe: pd.DataFrame,
    target_column: str | None,
) -> None:
    """Render a target/class distribution chart when it is meaningful."""
    with st.expander("Target/Class Distribution Chart"):
        if target_column is None:
            st.info("Skipped target distribution chart because no target column was selected.")
            return

        target_series = dataframe[target_column]
        unique_count = int(target_series.dropna().nunique())
        if unique_count == 0:
            st.info("Skipped target distribution chart because the selected target column has no usable values.")
            return
        if unique_count > 20:
            st.info(
                "Skipped target distribution chart because the selected target has too many unique values and behaves more like a continuous variable."
            )
            return

        target_counts = (
            target_series.fillna("Missing")
            .astype(str)
            .value_counts(dropna=False)
            .rename_axis("target_value")
            .reset_index(name="count")
        )
        st.bar_chart(
            target_counts.set_index("target_value")["count"],
        )
        st.caption("This chart shows how the selected target or class values are distributed.")


def _render_numeric_boxplot(dataframe: pd.DataFrame) -> None:
    """Render a boxplot across numeric columns or a skip message."""
    numeric_df = dataframe.select_dtypes(include=["number"])

    with st.expander("Numeric Columns Boxplot"):
        if numeric_df.empty:
            st.info("Skipped numeric boxplot because this dataset has no numeric columns.")
            return

        numeric_long = numeric_df.melt(var_name="column", value_name="value").dropna()
        if numeric_long.empty:
            st.info("Skipped numeric boxplot because the numeric columns do not contain plottable values.")
            return

        st.vega_lite_chart(
            numeric_long,
            {
                "mark": {"type": "boxplot", "extent": 1.5},
                "encoding": {
                    "x": {"field": "column", "type": "nominal", "title": "Numeric Column"},
                    "y": {"field": "value", "type": "quantitative", "title": "Value"},
                    "color": {"field": "column", "type": "nominal", "legend": None},
                },
            },
        )
        st.caption("Boxplots help spot spread, median values, and possible outliers in numeric columns.")


def _render_correlation_heatmap(dataframe: pd.DataFrame) -> None:
    """Render a numeric correlation heatmap or a skip message."""
    numeric_df = dataframe.select_dtypes(include=["number"])

    with st.expander("Correlation Heatmap"):
        if numeric_df.shape[1] < 2:
            st.info("Skipped correlation heatmap because at least two numeric columns are required.")
            return

        correlation_matrix = numeric_df.corr().fillna(0.0)
        correlation_heatmap_data = (
            correlation_matrix.rename_axis(index="column_a", columns="column_b")
            .stack()
            .reset_index(name="correlation")
        )
        if correlation_heatmap_data.empty:
            st.info("Skipped correlation heatmap because no numeric correlation values were available.")
            return

        st.vega_lite_chart(
            correlation_heatmap_data,
            {
                "mark": "rect",
                "encoding": {
                    "x": {"field": "column_a", "type": "nominal", "title": ""},
                    "y": {"field": "column_b", "type": "nominal", "title": ""},
                    "color": {
                        "field": "correlation",
                        "type": "quantitative",
                        "scale": {"domain": [-1, 1], "scheme": "redblue"},
                        "title": "Correlation",
                    },
                    "tooltip": [
                        {"field": "column_a", "type": "nominal", "title": "Column A"},
                        {"field": "column_b", "type": "nominal", "title": "Column B"},
                        {"field": "correlation", "type": "quantitative", "format": ".2f"},
                    ],
                },
            },
        )
        st.caption("The heatmap shows how strongly numeric columns move together.")


def render_dataset_visualizations(
    dataframe: pd.DataFrame,
    target_column: str | None,
) -> None:
    """Render safe pre-cleaning dataset visualizations inside expanders."""
    st.subheader("Dataset Visualizations")
    _render_missing_values_chart(dataframe)
    _render_target_distribution_chart(dataframe, target_column)
    _render_numeric_boxplot(dataframe)
    _render_correlation_heatmap(dataframe)


def render_data_quality_report(profile: dict, ml_recommendation: dict) -> None:
    """Show a beginner-friendly data quality summary and ML recommendation."""
    st.subheader("4. Data Quality Report")

    st.write(f"Selected Target Column: {ml_recommendation['selected_target_column']}")
    st.write(f"Suggested Target Column: {ml_recommendation['suggested_target_column']}")
    st.write(f"Suggested Problem Type: {ml_recommendation['suggested_problem_type']}")
    st.write(f"Confidence Score: {ml_recommendation['target_detection_confidence']}")
    st.write(f"Problem Type Reason: {ml_recommendation['problem_type_reason']}")
    st.write(f"Duplicate rows detected: {profile['duplicate_rows']}")

    columns_with_missing_values = [
        column for column, count in profile["missing_values"].items() if count > 0
    ]
    st.write(f"Columns with missing values: {columns_with_missing_values}")

    st.subheader("5. Recommended ML Algorithm")
    st.success(f"Recommended problem type: {ml_recommendation['recommended_problem_type']}")
    st.write(f"Reason: {ml_recommendation['reason']}")
    st.write(f"Target Column Used For Inference: {ml_recommendation['target_column_used_for_inference']}")
    st.write(f"Detected Text Column: {ml_recommendation['detected_text_column']}")

    algorithm_recommendation = ml_recommendation.get("algorithm_recommendation", {})
    beginner_choice = algorithm_recommendation.get("beginner_friendly_first_choice")
    if algorithm_recommendation.get("summary"):
        st.info(algorithm_recommendation["summary"])
    if algorithm_recommendation.get("target_variable_type"):
        st.write(
            f"Target Variable Type: {algorithm_recommendation['target_variable_type']}"
        )
    if beginner_choice:
        st.success(
            "Beginner-friendly first choice: "
            f"{beginner_choice['name']}"
        )
        st.write(f"Why it is suitable: {beginner_choice['reason']}")

    for warning_message in ml_recommendation["warnings"]:
        st.warning(warning_message)

    st.write("Recommended algorithms:")
    for algorithm in algorithm_recommendation.get(
        "recommended_algorithms",
        ml_recommendation["algorithms"],
    ):
        st.markdown(f"- **{algorithm['name']}**: {algorithm['reason']}")

    with st.expander("View detailed ML recommendation metadata"):
        st.write(f"Numeric Columns: {ml_recommendation['numeric_columns']}")
        st.write(f"Categorical Columns: {ml_recommendation['categorical_columns']}")
        st.write(f"Text Columns: {ml_recommendation['text_columns']}")
        st.write(f"Boolean Columns: {ml_recommendation['boolean_columns']}")
        st.write(f"Datetime Columns: {ml_recommendation['datetime_columns']}")
        st.write(f"ID-like Columns: {ml_recommendation['id_like_columns']}")
        st.write("Top target suggestions:", ml_recommendation["target_detection_metadata"].get("top_suggestions", []))


def render_pandas_numpy_usage_section(pandas_numpy_usage: dict) -> None:
    """Show a beginner-friendly explanation of relevant Pandas and NumPy usage."""
    st.subheader("Pandas and NumPy Usage")
    st.info(pandas_numpy_usage.get("summary", "No Pandas or NumPy usage details were recorded."))

    st.write("Pandas functions used:")
    for entry in pandas_numpy_usage.get("pandas_functions", []):
        st.markdown(f"- **{entry['function']}**: {entry['why_used']}")

    numpy_functions = pandas_numpy_usage.get("numpy_functions", [])
    if numpy_functions:
        st.write("NumPy functions used or directly relevant:")
        for entry in numpy_functions:
            st.markdown(f"- **{entry['function']}**: {entry['why_used']}")


def build_flowise_prompt(
    selected_prompt_type: str,
    dataset_summary: str,
    *,
    custom_prompt: str | None = None,
) -> str | None:
    """Build the final AI prompt from reusable templates and dataset summary."""
    if selected_prompt_type == CUSTOM_PROMPT_OPTION:
        cleaned_custom_prompt = (custom_prompt or "").strip()
        if not cleaned_custom_prompt:
            return None
        return (
            f"{FLOWISE_CONTEXT_INSTRUCTION}\n\n"
            "User Custom Question:\n"
            f"{cleaned_custom_prompt}\n\n"
            "Dataset Summary:\n"
            f"{dataset_summary}"
        )

    selected_template = PROMPT_TEMPLATES[selected_prompt_type]
    return (
        f"{FLOWISE_CONTEXT_INSTRUCTION}\n\n"
        f"{selected_template}\n\n"
        "Dataset Summary:\n"
        f"{dataset_summary}"
    )


def render_flowise_explanation_section(
    dataframe: pd.DataFrame,
    profile: dict,
    ml_recommendation: dict,
    target_column: str | None = None,
    cleaning_report: dict | None = None,
    key_prefix: str = "profile",
    dataset_identity: str | None = None,
    file_name: str | None = None,
) -> None:
    """Render the Flowise explanation UI using only summarized dataset context."""
    st.subheader("AI Agent Explanation")

    dataset_state_key = f"{key_prefix}_flowise_dataset_identity"
    answer_state_key = "last_flowise_answer"
    raw_response_state_key = "last_flowise_raw_response"

    # Clear stale Flowise output when the active dataset changes so a previous
    # explanation is not shown for a different upload.
    if st.session_state.get(dataset_state_key) != dataset_identity:
        st.session_state[dataset_state_key] = dataset_identity
        st.session_state.pop(answer_state_key, None)
        st.session_state.pop(raw_response_state_key, None)

    selected_prompt = st.selectbox(
        "Choose AI Prompt Type",
        options=[*PROMPT_TEMPLATES.keys(), CUSTOM_PROMPT_OPTION],
        key=f"{key_prefix}_flowise_prompt",
    )

    custom_prompt_state_key = f"{key_prefix}_flowise_custom_question"
    custom_prompt = None
    if selected_prompt == CUSTOM_PROMPT_OPTION:
        custom_prompt = st.text_area(
            "Custom AI Question",
            height=140,
            placeholder=(
                "Explain why Random Forest is suitable\n"
                "Explain outlier handling in simple words\n"
                "Generate viva questions\n"
                "Compare Logistic Regression vs Decision Tree\n"
                "Suggest feature engineering ideas"
            ),
            key=custom_prompt_state_key,
        )

    try:
        # Python reads and summarizes the dataset first. Flowise receives only
        # this compact preview text so the explanation layer never gets the
        # full dataset content.
        file_preview = build_flowise_file_preview(
            dataframe,
            target_column=target_column,
            cleaning_report=cleaning_report,
            profile=profile,
            ml_recommendation=ml_recommendation,
            file_name=file_name,
            max_rows=10,
        )
    except ValueError as exc:
        st.error(str(exc))
        return
    except Exception:
        st.error("The uploaded dataset could not be summarized for the AI agent.")
        return

    with st.expander("Preview sent to AI Agent"):
        st.text(file_preview)

    if st.button("Ask Flowise Agent", key=f"{key_prefix}_ask_flowise"):
        if selected_prompt == CUSTOM_PROMPT_OPTION and not (custom_prompt or "").strip():
            st.warning("Please enter a custom AI question.")
            return

        outbound_question = build_flowise_prompt(
            selected_prompt,
            file_preview,
            custom_prompt=custom_prompt,
        )
        if outbound_question is None:
            st.warning("Please enter a custom AI question.")
            return

        # Flowise is optional. If it fails, the Streamlit app should continue
        # to provide the Python-generated profiling, cleaning summary, and downloads.
        with st.spinner("Asking Flowise Agent..."):
            result = query_flowise_agent(outbound_question, file_summary=file_preview)

        if not result["success"]:
            st.error(result["error"])
            return

        st.session_state[answer_state_key] = result["answer"]
        st.session_state[raw_response_state_key] = result.get("raw_response")

    last_answer = st.session_state.get(answer_state_key)
    last_raw_response = st.session_state.get(raw_response_state_key)

    if last_answer:
        st.markdown(last_answer)
        st.download_button(
            "Download AI Explanation",
            data=last_answer,
            file_name="flowise_ai_explanation.md",
            mime="text/markdown",
            key=f"{key_prefix}_download_flowise_answer",
        )

    if last_raw_response:
        with st.expander("Raw Flowise Response"):
            st.json(last_raw_response)


def render_cleaning_results(
    cleaned_df: pd.DataFrame,
    cleaning_summary: dict,
    cleaning_report: dict,
    cleaned_csv_name: str,
    cleaned_csv_path: Path,
    cleaning_report_path: str,
) -> None:
    """Render the post-cleaning results in clear demo-friendly sections."""
    st.subheader("7. Cleaning Summary")
    st.success("Cleaning completed successfully.")

    st.write(f"Original rows: {cleaning_summary['original_rows']}")
    st.write(f"Original columns: {cleaning_summary['original_columns']}")
    st.write(f"Final rows: {cleaning_summary['final_rows']}")
    st.write(f"Final columns: {cleaning_summary['final_columns']}")
    st.write(f"Duplicate rows removed: {cleaning_summary['duplicate_rows_removed']}")
    st.write(
        f"Columns with missing values filled: {cleaning_summary['columns_where_missing_values_were_filled']}"
    )
    st.write(f"Encoded source columns: {cleaning_summary.get('encoded_columns', [])}")
    st.write(f"Scaled numeric columns: {cleaning_summary.get('scaled_columns', [])}")
    st.write(f"Cleaned text columns: {cleaning_summary.get('cleaned_text_columns', [])}")

    if cleaning_summary.get("target_encoding_recommendation"):
        st.info(cleaning_summary["target_encoding_recommendation"])

    if cleaning_summary.get("encoded_columns"):
        st.info(
            "ML models need numeric input, so text categories are converted into 0/1 columns."
        )

    if cleaning_summary.get("outlier_summary"):
        st.info("Outliers were capped with the IQR method instead of removing rows.")

    if cleaning_summary.get("scaled_columns"):
        st.info(
            "Scaling helps distance-based and gradient-based machine learning algorithms."
        )

    if cleaning_summary.get("cleaned_text_columns"):
        st.info(
            "Cleaned text can later be converted into numeric features using TF-IDF or Bag-of-Words."
        )

    if not cleaning_summary["options_used"]["handle_missing_values"]:
        st.info("Missing value handling was not selected, so missing values were not changed.")

    st.write("Step-by-step explanation:")
    for step in cleaning_summary["cleaning_steps"]:
        st.write(f"- {step}")

    if cleaning_summary.get("skipped_steps"):
        for skipped_step in cleaning_summary["skipped_steps"]:
            st.warning(skipped_step)

    st.subheader("Before vs After Cleaning")
    comparison_metrics = cleaning_summary.get("before_vs_after_summary", {}).get("metrics", [])
    if comparison_metrics:
        st.dataframe(comparison_metrics, width="stretch")

    with st.expander("View cleaned dataset preview"):
        st.dataframe(cleaned_df.head(), width="stretch")

    with st.expander("View detailed cleaning metadata"):
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
        st.write(
            "New encoded columns count:",
            cleaning_summary.get("encoded_columns_generated_count", 0),
        )
        st.write(
            "Original text backup columns:",
            cleaning_summary.get("nlp_original_backup_columns", []),
        )
        st.write("Scaler used:", cleaning_summary.get("scaler_used"))
        st.write("Type conversion notes:", cleaning_summary.get("type_conversion_notes", []))

        st.write("Missing values before cleaning:")
        st.dataframe(
            pd.DataFrame(
                {
                    "Column": list(cleaning_summary["missing_values_before"].keys()),
                    "Missing Values": list(cleaning_summary["missing_values_before"].values()),
                }
            ),
            width="stretch",
        )

        st.write("Missing values after cleaning:")
        st.dataframe(
            pd.DataFrame(
                {
                    "Column": list(cleaning_summary["missing_values_after"].keys()),
                    "Missing Values": list(cleaning_summary["missing_values_after"].values()),
                }
            ),
            width="stretch",
        )

        if cleaning_summary.get("nlp_before_after_examples"):
            st.write("NLP before vs after examples:")
            for column_name, example in cleaning_summary["nlp_before_after_examples"].items():
                st.write(f"{column_name} before: {example['before']}")
                st.write(f"{column_name} after: {example['after']}")

    render_pandas_numpy_usage_section(
        cleaning_report.get("pandas_numpy_usage", {})
    )

    st.subheader("8. Download Output Files")
    st.download_button(
        "Download Cleaned CSV",
        data=cleaned_df.to_csv(index=False).encode("utf-8"),
        file_name=cleaned_csv_name,
        mime="text/csv",
    )
    st.download_button(
        "Download Cleaning Report",
        data=json.dumps(cleaning_report, indent=2),
        file_name=Path(cleaning_report_path).name,
        mime="application/json",
    )
    st.success(f"Cleaned CSV saved to: {cleaned_csv_path}")
    st.success(f"Cleaning report saved to: {cleaning_report_path}")

    with st.expander("View JSON report output"):
        st.json(cleaning_report)


def render_uploaded_dataset(
    uploaded_file,
    selected_problem_type: str,
    cleaning_options: dict[str, bool],
) -> None:
    """Display dataset details after a file has been uploaded."""
    dataframe, error_message = load_dataset(uploaded_file)

    if error_message:
        st.error(error_message)
        return

    if dataframe is None:
        st.error("The dataset could not be loaded.")
        return

    if dataframe.empty:
        st.warning("The uploaded dataset is empty, so there is nothing to profile or clean yet.")
        return

    st.success(f"Uploaded file: {uploaded_file.name}")
    st.write(f"Dataset shape: {dataframe.shape[0]} rows x {dataframe.shape[1]} columns")

    target_options = [None] + list(dataframe.columns)
    selected_target = st.selectbox(
        "Select target column (optional)",
        options=target_options,
        format_func=lambda value: "None" if value is None else value,
        help="Choose the target column you plan to predict or analyze.",
    )

    if selected_target is None:
        st.info(
            "No target column selected. Cleaning can still continue, and clustering will be recommended by default."
        )

    st.subheader("1. Dataset Preview")
    st.dataframe(dataframe.head(), width="stretch")

    validation_result = validate_dataset(
        dataframe,
        uploaded_file_name=uploaded_file.name,
        target_column=selected_target,
    )

    st.subheader("2. Pre-cleaning Validation")
    for warning_message in validation_result["warnings"]:
        st.warning(warning_message)

    if validation_result["errors"]:
        for error_message in validation_result["errors"]:
            st.error(error_message)
        return

    if not validation_result["is_valid"]:
        return

    st.success("Validation passed. Profiling can continue.")

    profile = profile_dataset(dataframe, target_column=selected_target)
    ml_recommendation = recommend_ml_approach(
        dataframe,
        target_column=selected_target,
        problem_type=selected_problem_type,
        text_columns=profile["text_columns"],
    )

    render_dataset_profile(profile)
    render_dataset_visualizations(dataframe, selected_target)
    render_data_quality_report(profile, ml_recommendation)
    render_cleaning_options_summary(cleaning_options)
    render_flowise_explanation_section(
        dataframe,
        profile,
        ml_recommendation,
        target_column=selected_target,
        key_prefix="profile",
        dataset_identity=f"{uploaded_file.name}:{selected_target}:{dataframe.shape}",
        file_name=uploaded_file.name,
    )

    if st.button("Clean Dataset"):
        try:
            cleaned_df, cleaning_summary = clean_dataset(
                dataframe,
                options=cleaning_options,
                target_column=selected_target,
            )
            cleaned_csv_name = f"cleaned_{make_safe_stem(uploaded_file.name)}.csv"
            cleaned_csv_path = Path("output") / cleaned_csv_name
            cleaned_csv_path.parent.mkdir(parents=True, exist_ok=True)
            cleaned_df.to_csv(cleaned_csv_path, index=False)

            cleaning_report, cleaning_report_path = generate_cleaning_report(
                profile,
                validation_result,
                cleaning_summary,
                ml_recommendation,
                uploaded_file.name,
                cleaned_file_path=cleaned_csv_path,
            )
        except Exception as exc:
            st.error(f"Cleaning could not be completed safely. Details: {exc}")
            return

        render_cleaning_results(
            cleaned_df,
            cleaning_summary,
            cleaning_report,
            cleaned_csv_name,
            cleaned_csv_path,
            cleaning_report_path,
        )


def main() -> None:
    """Render the starter interface for the data cleaning app."""
    st.set_page_config(
        page_title="Data Cleaning Agent for ML Dataset Preparation",
        layout="wide",
    )

    uploaded_file, selected_problem_type, cleaning_options = build_sidebar()

    st.title("Data Cleaning Agent for ML Dataset Preparation")
    st.write(
        "Welcome to the Data Cleaning Agent. This app is designed to help you "
        "prepare machine learning datasets through a guided cleaning workflow."
    )

    st.markdown(
        """
        This starter interface will help you:
        - upload a dataset
        - validate common issues before cleaning
        - review a simple data quality report
        - choose beginner-friendly cleaning steps
        - prepare data for machine learning
        """
    )

    if uploaded_file is None:
        st.info("No file uploaded yet. Upload a CSV or Excel dataset from the sidebar to get started.")
        return

    render_uploaded_dataset(uploaded_file, selected_problem_type, cleaning_options)


if __name__ == "__main__":
    main()
