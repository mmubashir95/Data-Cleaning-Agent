"""
Data Cleaning Agent for ML Dataset Preparation

Run instructions:
pip install -r requirements.txt
streamlit run app.py
"""

import json
from collections import Counter
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from src.ai.flowise_client import (
    FLOWISE_PROFILE_NOTE,
    build_flowise_request_payload,
    build_default_flowise_metadata,
    build_flowise_file_preview,
    query_flowise_agent,
    validate_flowise_profile_text,
)
from utils.data_cleaner import clean_dataset
from utils.data_loader import load_dataset
from utils.data_profiler import profile_dataset
from utils.data_validator import validate_dataset
from utils.ecommerce_preprocessing import (
    build_ecommerce_output_datasets,
    build_ecommerce_preprocessed_view,
    detect_mobile_ecommerce_dataset,
)
from utils.ml_recommender import recommend_ml_approach
from utils.report_generator import generate_cleaning_report, make_safe_stem
from utils.smartphone_preprocessing import validate_smartphone_outputs

# The AI prompt explicitly tells Flowise to rely on the Python-generated
# profile. This keeps exact dataset facts grounded in code instead of model
# guesses and prevents the assistant from asking for the full file again.
FLOWISE_CONTEXT_INSTRUCTION = (
    "Important: a compact Python-generated dataset profile, cleaning report, and small "
    "preview will be provided by the Python app. If Python-generated dataset profile is "
    "provided, use it as the source of truth. Do not ask for uploaded file content. "
    "Only ask for manual details if BOTH uploaded file content and Python-generated profile "
    "are missing. Do not request the full file. Use only the provided profile and preview to answer. "
    "If something is missing, make a reasonable inference and clearly label it as an inference. "
    "Your explanation must clearly cover the dataset type, problem type, key data quality issues, "
    "cleaning actions performed, skipped actions and why, the recommended algorithm and reason, "
    "and the limitation that the answer is based on the profile and preview only. "
    "If the dataset looks like a scraped e-commerce mobile-phone catalog, your explanation must "
    "state clearly: (1) Python and Pandas performed all actual cleaning — Flowise only explains "
    "the profile and report. (2) No target column was selected and no model was trained at this "
    "stage. (3) The system generates two output files: a human-readable cleaned dataset that "
    "preserves important readable values and also includes processed and scaled columns for "
    "transparency, and a fully ML-ready dataset that contains only numeric scaled and encoded "
    "features suitable for future recommendation or ranking models. (4) The dataset is now "
    "prepared for a future product recommendation or ranking workflow only. "
    "If the dataset looks suitable for recommendation or ranking later, explain that it is only "
    "preparation-readiness and that no recommendation model is trained yet.\n\n"
    f"{FLOWISE_PROFILE_NOTE}"
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
MAX_VISUALIZATION_ROWS = 5000
MAX_HEATMAP_COLUMNS = 12
MAX_BOXPLOT_COLUMNS = 12
MAX_WORDS_IN_CHART = 10


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
    if profile.get("reference_columns"):
        st.write(f"Reference columns: {profile['reference_columns']}")

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
    """Render a missing values chart with counts and percentages."""
    missing_counts = dataframe.isnull().sum()
    missing_counts = missing_counts[missing_counts > 0].sort_values(ascending=False)
    total_rows = max(len(dataframe), 1)

    with st.expander("Missing Values Bar Chart", expanded=False):
        if missing_counts.empty:
            st.info("Skipped missing values chart because no missing values were detected before cleaning.")
            return

        missing_chart_data = missing_counts.rename_axis("column").reset_index(name="missing_values")
        missing_chart_data["missing_percentage"] = (
            (missing_chart_data["missing_values"] / total_rows) * 100
        )
        chart = (
            alt.Chart(missing_chart_data)
            .mark_bar(cornerRadiusTopRight=4, cornerRadiusBottomRight=4, color="#d97706")
            .encode(
                x=alt.X("missing_values:Q", title="Missing Value Count"),
                y=alt.Y("column:N", sort="-x", title="Column"),
                tooltip=[
                    alt.Tooltip("column:N", title="Column"),
                    alt.Tooltip("missing_values:Q", title="Missing Values"),
                    alt.Tooltip("missing_percentage:Q", title="Missing %", format=".1f"),
                ],
            )
            .properties(height=max(220, len(missing_chart_data) * 32), title="Columns With Missing Values")
        )
        labels = (
            alt.Chart(missing_chart_data)
            .mark_text(align="left", baseline="middle", dx=6, color="#374151")
            .encode(
                x=alt.X("missing_values:Q"),
                y=alt.Y("column:N", sort="-x"),
                text=alt.Text("missing_percentage:Q", format=".1f"),
            )
        )
        st.altair_chart(chart + labels, use_container_width=True)
        st.caption("Only columns with missing values are shown. Labels indicate the percentage of missing entries.")


def _render_target_distribution_chart(
    dataframe: pd.DataFrame,
    target_column: str | None,
) -> None:
    """Render a target/class distribution chart when it is meaningful."""
    with st.expander("Target/Class Distribution Chart", expanded=False):
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
        chart = (
            alt.Chart(target_counts)
            .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4, color="#0f766e")
            .encode(
                x=alt.X("target_value:N", title="Target / Class Value", sort="-y"),
                y=alt.Y("count:Q", title="Record Count"),
                tooltip=[
                    alt.Tooltip("target_value:N", title="Value"),
                    alt.Tooltip("count:Q", title="Count"),
                ],
            )
            .properties(height=320, title="Target or Class Distribution")
        )
        st.altair_chart(chart, use_container_width=True)
        st.caption("This chart shows how the selected target or class values are distributed.")


def _render_numeric_boxplot(dataframe: pd.DataFrame) -> None:
    """Render a boxplot across numeric columns or a skip message."""
    numeric_df = dataframe.select_dtypes(include=["number"])

    with st.expander("Numeric Columns Boxplot"):
        if numeric_df.empty:
            st.info("Skipped numeric boxplot because this dataset has no numeric columns.")
            return

        sampled_numeric_df, sample_note = _sample_dataframe_for_visuals(
            numeric_df,
            max_rows=MAX_VISUALIZATION_ROWS,
        )
        plot_columns = list(sampled_numeric_df.columns[:MAX_BOXPLOT_COLUMNS])
        numeric_long = (
            sampled_numeric_df.loc[:, plot_columns]
            .melt(var_name="column", value_name="value")
            .dropna()
        )
        if numeric_long.empty:
            st.info("Skipped numeric boxplot because the numeric columns do not contain plottable values.")
            return

        chart = (
            alt.Chart(numeric_long)
            .mark_boxplot(extent=1.5, size=28)
            .encode(
                x=alt.X("column:N", title="Numeric Column", axis=alt.Axis(labelAngle=-35)),
                y=alt.Y("value:Q", title="Value"),
                color=alt.Color("column:N", legend=None),
                tooltip=[
                    alt.Tooltip("column:N", title="Column"),
                    alt.Tooltip("value:Q", title="Value"),
                ],
            )
            .properties(height=420, title="Numeric Feature Spread and Potential Outliers")
        )
        st.altair_chart(chart, use_container_width=True)
        if len(numeric_df.columns) > len(plot_columns):
            st.caption(
                f"Showing the first {len(plot_columns)} numeric columns to keep the view readable."
            )
        if sample_note:
            st.caption(sample_note)
        st.caption("Boxplots help spot spread, median values, and possible outliers in numeric columns.")


def _render_correlation_heatmap(dataframe: pd.DataFrame) -> None:
    """Render a numeric correlation heatmap or a skip message."""
    numeric_df = dataframe.select_dtypes(include=["number"])

    with st.expander("Correlation Heatmap", expanded=False):
        if numeric_df.shape[1] < 2:
            st.info("Skipped correlation heatmap because at least two numeric columns are required.")
            return

        sampled_numeric_df, sample_note = _sample_dataframe_for_visuals(
            numeric_df,
            max_rows=MAX_VISUALIZATION_ROWS,
        )
        limited_numeric_df = sampled_numeric_df.iloc[:, :MAX_HEATMAP_COLUMNS]
        correlation_matrix = limited_numeric_df.corr().fillna(0.0)
        correlation_heatmap_data = (
            correlation_matrix.rename_axis(index="column_a", columns="column_b")
            .stack()
            .reset_index(name="correlation")
        )
        if correlation_heatmap_data.empty:
            st.info("Skipped correlation heatmap because no numeric correlation values were available.")
            return

        base_chart = alt.Chart(correlation_heatmap_data).encode(
            x=alt.X("column_a:N", title="", axis=alt.Axis(labelAngle=-35)),
            y=alt.Y("column_b:N", title=""),
        )
        heatmap = base_chart.mark_rect().encode(
            color=alt.Color(
                "correlation:Q",
                scale=alt.Scale(domain=[-1, 1], scheme="teals"),
                title="Correlation",
            ),
            tooltip=[
                alt.Tooltip("column_a:N", title="Column A"),
                alt.Tooltip("column_b:N", title="Column B"),
                alt.Tooltip("correlation:Q", title="Correlation", format=".2f"),
            ],
        )
        labels = base_chart.mark_text(baseline="middle", fontSize=11).encode(
            text=alt.Text("correlation:Q", format=".2f"),
            color=alt.condition(
                "abs(datum.correlation) > 0.45",
                alt.value("white"),
                alt.value("#111827"),
            ),
        )
        st.altair_chart((heatmap + labels).properties(height=460, title="Numeric Correlation Heatmap"), use_container_width=True)
        if numeric_df.shape[1] > limited_numeric_df.shape[1]:
            st.caption(
                f"Showing the first {limited_numeric_df.shape[1]} numeric columns to keep the heatmap readable."
            )
        if sample_note:
            st.caption(sample_note)
        st.caption("The heatmap shows how strongly numeric columns move together.")


def _sample_dataframe_for_visuals(
    dataframe: pd.DataFrame,
    *,
    max_rows: int = MAX_VISUALIZATION_ROWS,
) -> tuple[pd.DataFrame, str | None]:
    """Sample very large datasets so the dashboard stays responsive."""
    if len(dataframe) <= max_rows:
        return dataframe, None

    sampled_df = dataframe.sample(n=max_rows, random_state=42)
    return (
        sampled_df,
        f"Showing a random sample of {max_rows:,} rows for visualization performance.",
    )


def _render_visualization_metric_cards(dataframe: pd.DataFrame, profile: dict) -> None:
    """Render a compact dataset quality summary above the charts."""
    total_missing_values = int(sum(profile.get("missing_values", {}).values()))
    metric_columns = st.columns(6)
    metric_columns[0].metric("Total Rows", f"{len(dataframe):,}")
    metric_columns[1].metric("Total Columns", f"{len(dataframe.columns):,}")
    metric_columns[2].metric("Duplicate Rows", f"{int(profile.get('duplicate_rows', 0)):,}")
    metric_columns[3].metric("Missing Values", f"{total_missing_values:,}")
    metric_columns[4].metric("Numeric Columns", f"{len(profile.get('numeric_columns', [])):,}")
    metric_columns[5].metric("Categorical Columns", f"{len(profile.get('categorical_columns', [])):,}")


def _render_ecommerce_visualizations(dataframe: pd.DataFrame) -> None:
    """Render mobile-commerce specific charts when parsed numeric features exist."""
    with st.expander("E-commerce Product Insights", expanded=False):
        ecommerce_view, ecommerce_metadata = build_ecommerce_preprocessed_view(dataframe)
        if not ecommerce_metadata.get("ecommerce_preprocessing_applied", False):
            st.info("Skipped e-commerce visualizations because this dataset does not look like a mobile product catalog.")
            return

        sampled_df, sample_note = _sample_dataframe_for_visuals(
            ecommerce_view,
            max_rows=MAX_VISUALIZATION_ROWS,
        )

        chart_rendered = False
        if "price" in sampled_df.columns:
            price_series = sampled_df["price"].dropna()
            if not price_series.empty:
                price_chart = (
                    alt.Chart(pd.DataFrame({"price": price_series}))
                    .mark_bar(color="#2563eb")
                    .encode(
                        x=alt.X("price:Q", bin=alt.Bin(maxbins=25), title="Price"),
                        y=alt.Y("count():Q", title="Product Count"),
                    )
                    .properties(height=280, title="Price Distribution")
                )
                st.altair_chart(price_chart, use_container_width=True)
                chart_rendered = True

        if "rating" in sampled_df.columns:
            rating_series = sampled_df["rating"].dropna()
            if not rating_series.empty:
                rating_chart = (
                    alt.Chart(pd.DataFrame({"rating": rating_series}))
                    .mark_bar(color="#0f766e")
                    .encode(
                        x=alt.X("rating:Q", bin=alt.Bin(maxbins=20), title="Rating"),
                        y=alt.Y("count():Q", title="Product Count"),
                    )
                    .properties(height=280, title="Rating Distribution")
                )
                st.altair_chart(rating_chart, use_container_width=True)
                chart_rendered = True

        if {"ram_gb", "price"}.issubset(sampled_df.columns):
            ram_price_df = sampled_df[["ram_gb", "price"]].dropna()
            if not ram_price_df.empty:
                ram_price_chart = (
                    alt.Chart(ram_price_df)
                    .mark_circle(size=70, color="#7c3aed")
                    .encode(
                        x=alt.X("ram_gb:Q", title="RAM (GB)"),
                        y=alt.Y("price:Q", title="Price"),
                        tooltip=["ram_gb", "price"],
                    )
                    .properties(height=320, title="RAM vs Price")
                )
                st.altair_chart(ram_price_chart, use_container_width=True)
                chart_rendered = True

        if {"storage_gb", "price"}.issubset(sampled_df.columns):
            storage_price_df = sampled_df[["storage_gb", "price"]].dropna()
            if not storage_price_df.empty:
                storage_price_chart = (
                    alt.Chart(storage_price_df)
                    .mark_circle(size=70, color="#db2777")
                    .encode(
                        x=alt.X("storage_gb:Q", title="Storage (GB)"),
                        y=alt.Y("price:Q", title="Price"),
                        tooltip=["storage_gb", "price"],
                    )
                    .properties(height=320, title="Storage vs Price")
                )
                st.altair_chart(storage_price_chart, use_container_width=True)
                chart_rendered = True

        if "brand" in sampled_df.columns:
            brand_counts = (
                sampled_df["brand"]
                .dropna()
                .astype(str)
                .value_counts()
                .head(10)
                .rename_axis("brand")
                .reset_index(name="count")
            )
            if not brand_counts.empty:
                brand_chart = (
                    alt.Chart(brand_counts)
                    .mark_bar(cornerRadiusTopRight=4, cornerRadiusBottomRight=4, color="#f59e0b")
                    .encode(
                        x=alt.X("count:Q", title="Product Count"),
                        y=alt.Y("brand:N", title="Brand", sort="-x"),
                    )
                    .properties(height=320, title="Top Brand Frequency")
                )
                st.altair_chart(brand_chart, use_container_width=True)
                chart_rendered = True

        if sample_note:
            st.caption(sample_note)
        if not chart_rendered:
            st.info("Skipped e-commerce visualizations because the parsed numeric product fields were not available yet.")


def _render_data_type_distribution(profile: dict) -> None:
    """Show the distribution of major data types detected in the dataset."""
    with st.expander("Data Type Distribution", expanded=False):
        distribution_data = pd.DataFrame(
            {
                "data_type": [
                    "Numeric",
                    "Categorical",
                    "Text",
                    "Datetime",
                ],
                "count": [
                    len(profile.get("numeric_columns", [])),
                    len(profile.get("categorical_columns", [])),
                    len(profile.get("text_columns", [])),
                    len(profile.get("datetime_columns", [])),
                ],
            }
        )
        non_zero_distribution_data = distribution_data[distribution_data["count"] > 0]
        if non_zero_distribution_data.empty:
            st.info("Skipped data type distribution because no columns were available to classify.")
            return

        chart = (
            alt.Chart(non_zero_distribution_data)
            .mark_arc(innerRadius=55)
            .encode(
                theta=alt.Theta("count:Q", title="Column Count"),
                color=alt.Color(
                    "data_type:N",
                    scale=alt.Scale(
                        range=["#2563eb", "#16a34a", "#f59e0b", "#7c3aed"]
                    ),
                    title="Data Type",
                ),
                tooltip=[
                    alt.Tooltip("data_type:N", title="Data Type"),
                    alt.Tooltip("count:Q", title="Column Count"),
                ],
            )
            .properties(height=320, title="Detected Data Type Distribution")
        )
        st.altair_chart(chart, use_container_width=True)


def _render_nlp_visualizations(
    dataframe: pd.DataFrame,
    text_columns: list[str],
    target_column: str | None,
) -> None:
    """Render lightweight text visualizations only when free-text columns exist."""
    with st.expander("NLP/Text Insights"):
        if not text_columns:
            st.info("Skipped NLP visualizations because no text columns were detected.")
            return

        text_column = text_columns[0]
        text_series = dataframe[text_column].fillna("").astype(str).str.strip()
        text_series = text_series[text_series != ""]
        if text_series.empty:
            st.info("Skipped NLP visualizations because the detected text column has no usable text values.")
            return

        sampled_text_series, sample_note = _sample_dataframe_for_visuals(
            text_series.to_frame(name=text_column),
            max_rows=MAX_VISUALIZATION_ROWS,
        )
        sampled_text_series = sampled_text_series[text_column]
        length_data = pd.DataFrame(
            {
                "text_length": sampled_text_series.str.len(),
            }
        )
        length_chart = (
            alt.Chart(length_data)
            .mark_bar(color="#6366f1")
            .encode(
                x=alt.X("text_length:Q", bin=alt.Bin(maxbins=25), title="Text Length (Characters)"),
                y=alt.Y("count():Q", title="Record Count"),
                tooltip=[alt.Tooltip("count():Q", title="Records")],
            )
            .properties(height=300, title=f"Length Distribution for '{text_column}'")
        )
        st.altair_chart(length_chart, use_container_width=True)

        token_counter = Counter()
        for value in sampled_text_series:
            tokens = [token.lower() for token in value.split() if len(token) > 2]
            token_counter.update(tokens)

        if token_counter:
            common_words_data = pd.DataFrame(
                token_counter.most_common(MAX_WORDS_IN_CHART),
                columns=["word", "count"],
            )
            common_words_chart = (
                alt.Chart(common_words_data)
                .mark_bar(cornerRadiusTopRight=4, cornerRadiusBottomRight=4, color="#8b5cf6")
                .encode(
                    x=alt.X("count:Q", title="Frequency"),
                    y=alt.Y("word:N", sort="-x", title="Word"),
                    tooltip=[
                        alt.Tooltip("word:N", title="Word"),
                        alt.Tooltip("count:Q", title="Frequency"),
                    ],
                )
                .properties(height=320, title="Most Common Words in Sampled Text")
            )
            st.altair_chart(common_words_chart, use_container_width=True)
        else:
            st.info("Skipped common-word visualization because the sampled text did not contain enough tokens.")

        if target_column is not None:
            target_series = dataframe[target_column]
            unique_count = int(target_series.dropna().nunique())
            if 0 < unique_count <= 20:
                target_counts = (
                    target_series.fillna("Missing")
                    .astype(str)
                    .value_counts(dropna=False)
                    .rename_axis("target_value")
                    .reset_index(name="count")
                )
                target_chart = (
                    alt.Chart(target_counts)
                    .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4, color="#db2777")
                    .encode(
                        x=alt.X("target_value:N", title="Label", sort="-y"),
                        y=alt.Y("count:Q", title="Record Count"),
                    )
                    .properties(height=280, title="Text Dataset Label Distribution")
                )
                st.altair_chart(target_chart, use_container_width=True)

        if sample_note:
            st.caption(sample_note)


def _render_outlier_visualizations(dataframe: pd.DataFrame) -> None:
    """Render outlier-focused plots without mutating the dataset."""
    with st.expander("Outlier Overview"):
        numeric_columns = dataframe.select_dtypes(include=["number"]).columns.tolist()
        if not numeric_columns:
            st.info("Skipped outlier visualization because this dataset has no numeric columns.")
            return

        selected_column = st.selectbox(
            "Select numeric column for outlier review",
            options=numeric_columns,
            key="visualization_outlier_column",
        )
        sampled_df, sample_note = _sample_dataframe_for_visuals(
            dataframe[[selected_column]].dropna(),
            max_rows=MAX_VISUALIZATION_ROWS,
        )
        if sampled_df.empty:
            st.info("Skipped outlier visualization because the selected column has no usable numeric values.")
            return

        series = sampled_df[selected_column]
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outlier_count = int(((series < lower_bound) | (series > upper_bound)).sum()) if pd.notna(iqr) else 0

        metric_columns = st.columns(3)
        metric_columns[0].metric("IQR Lower Bound", f"{lower_bound:,.2f}")
        metric_columns[1].metric("IQR Upper Bound", f"{upper_bound:,.2f}")
        metric_columns[2].metric("Potential Outliers", f"{outlier_count:,}")

        histogram_data = sampled_df.rename(columns={selected_column: "value"})
        histogram_chart = (
            alt.Chart(histogram_data)
            .mark_bar(color="#0284c7")
            .encode(
                x=alt.X("value:Q", bin=alt.Bin(maxbins=30), title=selected_column),
                y=alt.Y("count():Q", title="Record Count"),
            )
            .properties(height=280, title=f"Distribution of '{selected_column}'")
        )
        st.altair_chart(histogram_chart, use_container_width=True)

        boxplot_chart = (
            alt.Chart(histogram_data)
            .mark_boxplot(extent=1.5, size=45, color="#0f766e")
            .encode(
                x=alt.X("value:Q", title=selected_column),
            )
            .properties(height=140, title=f"Outlier Review Boxplot for '{selected_column}'")
        )
        st.altair_chart(boxplot_chart, use_container_width=True)

        if sample_note:
            st.caption(sample_note)


def render_dataset_visualizations(
    dataframe: pd.DataFrame,
    profile: dict,
    target_column: str | None,
) -> None:
    """Render safe pre-cleaning dataset visualizations inside expanders."""
    visualization_df = dataframe
    if detect_mobile_ecommerce_dataset(dataframe.columns):
        visualization_df, _ = build_ecommerce_preprocessed_view(dataframe)

    st.subheader("Dataset Visualizations")
    st.caption("Use these charts to inspect quality issues before applying any cleaning step.")
    _render_visualization_metric_cards(dataframe, profile)
    st.markdown("")
    _render_missing_values_chart(dataframe)
    _render_data_type_distribution(profile)
    _render_target_distribution_chart(dataframe, target_column)
    _render_numeric_boxplot(visualization_df)
    _render_correlation_heatmap(visualization_df)
    _render_outlier_visualizations(visualization_df)
    _render_ecommerce_visualizations(dataframe)
    _render_nlp_visualizations(
        dataframe,
        profile.get("text_columns", []),
        target_column,
    )


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
    if profile.get("reference_columns"):
        st.write(f"Reference/URL columns: {profile['reference_columns']}")
    if detect_mobile_ecommerce_dataset(profile.get("column_names", [])):
        st.info(
            "This dataset looks like a mobile-product catalog. The app will keep Python responsible for preprocessing and treat the result as future recommendation or ranking readiness unless you explicitly select a target."
        )
    if ml_recommendation.get("smartphone_dataset_detected"):
        st.info(
            "This upload matches the complex smartphone recommendation dataset. Generic cleaning alone is not enough, so smartphone-specific feature extraction and shifted-column handling will be applied."
        )

    st.subheader("5. Recommended ML Algorithm")
    st.success(f"Recommended problem type: {ml_recommendation['recommended_problem_type']}")
    st.write(f"Reason: {ml_recommendation['reason']}")
    st.write(f"Target Column Used For Inference: {ml_recommendation['target_column_used_for_inference']}")
    if ml_recommendation.get("smartphone_dataset_detected"):
        detected_text_columns = ml_recommendation.get("detected_text_columns", [])
        st.write(
            "Detected Text Columns: "
            + (", ".join(detected_text_columns) if detected_text_columns else "None")
        )
        ignored_columns = ml_recommendation.get("ignored_columns", [])
        if ignored_columns:
            ignored_column = ignored_columns[0]
            st.write(
                f"Dropped/Ignored Column: {ignored_column['column']}, because {ignored_column['reason'].rstrip('.')}"
            )
    else:
        st.write(f"Detected Text Column: {ml_recommendation['detected_text_column']}")
    if ml_recommendation.get("recommendation_ready"):
        if ml_recommendation.get("smartphone_dataset_detected"):
            st.info(
                "The current stage prepares this smartphone dataset for content-based recommendation using cosine similarity. No recommendation model is trained yet."
            )
            st.info(
                "The system can generate both a cleaned readable smartphone dataset and a fully ML-ready smartphone recommendation dataset with numeric, boolean, and encoded categorical features."
            )
        else:
            st.info(
                "The current stage prepares the product data for future recommendation or ranking workflows. No recommendation model is trained yet."
            )
            st.info(
                "The system can generate two outputs: a human-readable cleaned dataset with readable numeric values plus separate scaled columns, and a fully ML-ready dataset containing only numeric scaled and encoded features."
            )

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

    if ml_recommendation.get("smartphone_dataset_detected"):
        secondary_algorithms = algorithm_recommendation.get("recommended_algorithms", [])[1:2]
        if secondary_algorithms:
            st.write(f"Secondary option: {secondary_algorithms[0]['name']}")
        not_suitable_currently = ml_recommendation.get("not_suitable_currently", [])
        if not_suitable_currently:
            st.write("Not suitable currently:")
            for item in not_suitable_currently:
                st.markdown(f"- **{item['approach']}**: {item['reason']}")

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
        ignored_columns = profile.get("ignored_columns", [])
        if ignored_columns:
            st.write(
                "Dropped/Ignored Columns:",
                [entry["column"] for entry in ignored_columns],
            )
            for entry in ignored_columns:
                st.write(f"Dropped/Ignored Reason: {entry['reason']}")
        st.write(f"Recommendation Ready: {ml_recommendation.get('recommendation_ready', False)}")
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


def render_project_summary_for_viva(viva_summary: dict) -> None:
    """Render a short presentation-friendly project summary."""
    st.subheader("Project Summary for Viva")
    st.markdown(viva_summary.get("markdown", "No viva summary is available."))


def build_flowise_prompt(
    selected_prompt_type: str,
    dataset_summary: str,
    *,
    custom_prompt: str | None = None,
) -> str | None:
    """Build the final AI prompt from reusable templates and dataset summary.

    The dataset summary is not interpolated here because it is embedded later
    inside the Flowise request payload. Keeping prompt selection separate from
    profile injection makes the safety boundary easier to audit.
    """
    if selected_prompt_type == CUSTOM_PROMPT_OPTION:
        cleaned_custom_prompt = (custom_prompt or "").strip()
        if not cleaned_custom_prompt:
            return None
        return (
            f"{FLOWISE_CONTEXT_INSTRUCTION}\n\n"
            "User Custom Question:\n"
            f"{cleaned_custom_prompt}\n\n"
            "Use the embedded Python-generated dataset profile included in this request."
        )

    selected_template = PROMPT_TEMPLATES[selected_prompt_type]
    return (
        f"{FLOWISE_CONTEXT_INSTRUCTION}\n\n"
        f"{selected_template}\n\n"
        "Use the embedded Python-generated dataset profile included in this request."
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
    """Render the Flowise explanation UI using only summarized dataset context.

    This section is intentionally isolated from the cleaning pipeline so AI
    availability never determines whether profiling, cleaning, or reporting can
    complete.
    """
    st.subheader("AI Agent Explanation")

    dataset_state_key = f"{key_prefix}_flowise_dataset_identity"
    answer_state_key = "last_flowise_answer"
    raw_response_state_key = "last_flowise_raw_response"
    metadata_state_key = f"{key_prefix}_flowise_metadata"
    payload_state_key = f"{key_prefix}_flowise_payload"

    # Clear stale Flowise output when the active dataset changes so a previous
    # explanation is not shown for a different upload.
    if st.session_state.get(dataset_state_key) != dataset_identity:
        st.session_state[dataset_state_key] = dataset_identity
        st.session_state.pop(answer_state_key, None)
        st.session_state.pop(raw_response_state_key, None)
        st.session_state[metadata_state_key] = build_default_flowise_metadata()
        st.session_state.pop(payload_state_key, None)

    if metadata_state_key not in st.session_state:
        st.session_state[metadata_state_key] = build_default_flowise_metadata()

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

    is_valid_profile, profile_data = validate_flowise_profile_text(file_preview)
    if not is_valid_profile:
        st.error("The Python-generated dataset profile is empty or incomplete, so Flowise cannot be called safely.")
        return

    preview_payload, preview_metadata = build_flowise_request_payload(
        build_flowise_prompt(selected_prompt, file_preview, custom_prompt=custom_prompt)
        or "",
        file_preview,
    )

    with st.expander("Preview sent to AI Agent"):
        st.json(
            {
                "payload": preview_payload,
                "profile_debug": {
                    "profile_sent_to_flowise": preview_metadata["profile_sent_to_flowise"],
                    "profile_keys_sent": preview_metadata["profile_keys_sent"],
                    "preview_rows_sent": preview_metadata["preview_rows_sent"],
                    "full_dataset_sent_to_flowise": preview_metadata["full_dataset_sent_to_flowise"],
                },
            }
        )

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

        is_valid_profile, profile_data = validate_flowise_profile_text(file_preview)
        if not is_valid_profile:
            st.error("The Python-generated dataset profile is empty or incomplete, so Flowise cannot be called safely.")
            return
        if not profile_data.get("shape") or "column_names" not in profile_data or "sample_rows" not in profile_data:
            st.error("The Python-generated dataset profile is missing required keys for Flowise.")
            return

        # Flowise is optional. If it fails, the Streamlit app should continue
        # to provide the Python-generated profiling, cleaning summary, and downloads.
        with st.spinner("Asking Flowise Agent..."):
            result = query_flowise_agent(outbound_question, file_summary=file_preview)

        st.session_state[metadata_state_key] = result.get(
            "metadata",
            build_default_flowise_metadata(),
        )
        st.session_state[payload_state_key] = result.get("payload")

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
    if cleaning_summary.get("ecommerce_preprocessing_applied"):
        st.write(
            f"Extracted numeric feature columns: {cleaning_summary.get('extracted_feature_columns', [])}"
        )
        st.write(
            f"Dropped reference columns: {cleaning_summary.get('dropped_reference_columns', [])}"
        )
        st.write(
            f"Normalized categorical columns: {cleaning_summary.get('normalized_categorical_columns', [])}"
        )
        st.write(
            f"Scaled columns created: {cleaning_summary.get('scaled_columns_created', [])}"
        )
    if cleaning_summary.get("smartphone_preprocessing_applied"):
        st.write(
            f"Shifted/misaligned smartphone fixes applied: {len(cleaning_summary.get('shifted_column_fixes', []))}"
        )
        st.write(
            f"Noise fixes applied: {cleaning_summary.get('noise_fixes', [])}"
        )

    st.subheader("Cleaning Impact Summary")
    comparison_columns = st.columns(4)
    before_missing = int(sum(cleaning_summary["missing_values_before"].values()))
    after_missing = int(sum(cleaning_summary["missing_values_after"].values()))
    comparison_columns[0].metric(
        "Missing Values",
        f"{after_missing:,}",
        delta=f"{after_missing - before_missing:,}",
        delta_color="inverse",
    )
    comparison_columns[1].metric(
        "Duplicate Rows",
        f"{int(cleaned_df.duplicated().sum()):,}",
        delta=f"{int(cleaned_df.duplicated().sum()) - int(cleaning_summary['duplicate_rows_removed'] + cleaned_df.duplicated().sum()):,}",
        delta_color="inverse",
    )
    comparison_columns[2].metric(
        "Row Count",
        f"{cleaning_summary['final_rows']:,}",
        delta=f"{cleaning_summary['final_rows'] - cleaning_summary['original_rows']:,}",
    )
    comparison_columns[3].metric(
        "Column Count",
        f"{cleaning_summary['final_columns']:,}",
        delta=f"{cleaning_summary['final_columns'] - cleaning_summary['original_columns']:,}",
    )

    impact_table = pd.DataFrame(
        [
            {
                "Metric": "Missing Values",
                "Before Cleaning": before_missing,
                "After Cleaning": after_missing,
            },
            {
                "Metric": "Duplicate Rows",
                "Before Cleaning": cleaning_summary["duplicate_rows_removed"]
                + int(cleaned_df.duplicated().sum()),
                "After Cleaning": int(cleaned_df.duplicated().sum()),
            },
            {
                "Metric": "Column Count",
                "Before Cleaning": cleaning_summary["original_columns"],
                "After Cleaning": cleaning_summary["final_columns"],
            },
            {
                "Metric": "Row Count",
                "Before Cleaning": cleaning_summary["original_rows"],
                "After Cleaning": cleaning_summary["final_rows"],
            },
        ]
    )
    st.dataframe(impact_table, width="stretch")

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
    if cleaning_summary.get("recommendation_ready"):
        if cleaning_summary.get("smartphone_preprocessing_applied"):
            st.info(
                "The cleaned dataset is now prepared for a content-based smartphone recommendation system using cosine similarity, but no recommendation model has been trained in this stage."
            )
            st.info(
                "Two outputs are available for this dataset: `cleaned_readable_smartphone_dataset.csv` and `ml_ready_smartphone_recommendation_dataset.csv`."
            )
        else:
            st.info(
                "The cleaned dataset is more suitable for future product recommendation or ranking workflows, but no model has been trained in this stage."
            )
            st.info(
                "Two outputs are available for this dataset: a human-readable cleaned CSV for viva and review, and a fully ML-ready CSV for future recommendation or ranking models."
            )

    if cleaning_summary.get("smartphone_preprocessing_applied"):
        st.subheader("Data Validity & Suspicious Records Check")
        smartphone_quality = cleaning_summary.get("smartphone_dataset_quality", {})
        quality_columns = st.columns(4)
        quality_columns[0].metric("Quality Mode", str(smartphone_quality.get("mode", "safe")).title())
        quality_columns[1].metric("Suspicious Records", int(smartphone_quality.get("suspicious_records_count", 0)))
        quality_columns[2].metric("Critical Records", int(smartphone_quality.get("critical_suspicious_records_count", 0)))
        quality_columns[3].metric(
            "Invalid ML Brand Columns",
            len(cleaning_summary.get("invalid_ml_ready_brand_columns", [])),
        )

        invalid_brand_columns = cleaning_summary.get("invalid_ml_ready_brand_columns", [])
        if invalid_brand_columns:
            st.warning(
                "Suspicious brand-derived ML columns detected: "
                + ", ".join(invalid_brand_columns)
                + ". Safe mode keeps them for review; strict mode removes the critical source rows."
            )

        suspicious_records = smartphone_quality.get("suspicious_records_details", [])
        if suspicious_records:
            st.dataframe(pd.DataFrame(suspicious_records), width="stretch")
        else:
            st.success("No suspicious smartphone records were detected.")

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
        st.write("Deduplication strategy:", cleaning_summary.get("deduplication_strategy"))
        st.write("Domain outlier adjustments:", cleaning_summary.get("domain_outlier_adjustments", []))
        if cleaning_summary.get("smartphone_preprocessing_applied"):
            st.write("Shifted column fixes:", cleaning_summary.get("shifted_column_fixes", []))
            st.write("Noise fixes:", cleaning_summary.get("noise_fixes", []))
            st.write("Smartphone validation checks:", cleaning_summary.get("smartphone_validation_checks", []))
            st.write("Smartphone dataset quality:", cleaning_summary.get("smartphone_dataset_quality", {}))

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
    render_project_summary_for_viva(
        cleaning_report.get("project_summary_for_viva", {})
    )

    st.subheader("8. Download Output Files")
    if cleaning_summary.get("ecommerce_preprocessing_applied"):
        readable_dataset = cleaning_summary.get("readable_cleaned_dataset")
        ml_ready_dataset = cleaning_summary.get("ml_ready_dataset")
        readable_csv_name = cleaning_summary.get("readable_cleaned_csv_name", cleaned_csv_name)
        ml_ready_csv_name = cleaning_summary.get("ml_ready_csv_name", f"ml_ready_{cleaned_csv_path.stem}.csv")

        if isinstance(readable_dataset, pd.DataFrame):
            st.download_button(
                "Download Human-Readable Cleaned CSV",
                data=readable_dataset.to_csv(index=False).encode("utf-8"),
                file_name=readable_csv_name,
                mime="text/csv",
            )
        if isinstance(ml_ready_dataset, pd.DataFrame):
            st.download_button(
                "Download Fully ML-Ready CSV",
                data=ml_ready_dataset.to_csv(index=False).encode("utf-8"),
                file_name=ml_ready_csv_name,
                mime="text/csv",
            )
        st.download_button(
            "Download Cleaned CSV",
            data=cleaned_df.to_csv(index=False).encode("utf-8"),
            file_name=cleaned_csv_name,
            mime="text/csv",
        )
    else:
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
    viva_summary = cleaning_report.get("project_summary_for_viva", {})
    if viva_summary.get("markdown"):
        st.download_button(
            "Download Viva Summary",
            data=viva_summary["markdown"],
            file_name=f"project_summary_{cleaned_csv_path.stem}.md",
            mime="text/markdown",
        )
    st.success(f"Cleaned CSV saved to: {cleaned_csv_path}")
    if cleaning_summary.get("readable_cleaned_csv_path"):
        st.success(f"Human-readable cleaned CSV saved to: {cleaning_summary['readable_cleaned_csv_path']}")
    if cleaning_summary.get("ml_ready_csv_path"):
        st.success(f"ML-ready CSV saved to: {cleaning_summary['ml_ready_csv_path']}")
    st.success(f"Cleaning report saved to: {cleaning_report_path}")

    with st.expander("View JSON report output"):
        st.json(cleaning_report)


def _clear_cleaning_state() -> None:
    """Remove all cleaning-output keys from session state."""
    for key in (
        "cleaned_df", "readable_cleaned_df", "ml_ready_df",
        "cleaning_report", "cleaning_summary", "cleaned_csv_name",
        "cleaned_csv_path", "cleaning_report_path", "cleaning_completed",
    ):
        st.session_state.pop(key, None)


def render_uploaded_dataset(
    uploaded_file,
    selected_problem_type: str,
    cleaning_options: dict[str, bool],
) -> None:
    """Display dataset details after a file has been uploaded.

    The flow deliberately runs as validate -> profile -> recommend -> explain
    -> clean so users can inspect issues before applying transformations.
    """
    # Detect a new upload and clear stale cleaning results so the user is never
    # shown outputs from a previously cleaned file.
    file_id = f"{uploaded_file.name}:{uploaded_file.size}"
    if st.session_state.get("_uploaded_file_id") != file_id:
        _clear_cleaning_state()
        st.session_state["_uploaded_file_id"] = file_id

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
    render_dataset_visualizations(dataframe, profile, selected_target)
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
            # Heavy processing stays in Python because exact row/column counts,
            # missing-value handling, scaling, and file generation must be
            # deterministic and should not depend on LLM output.
            cleaned_df, cleaning_summary = clean_dataset(
                dataframe,
                options=cleaning_options,
                target_column=selected_target,
            )
            cleaned_csv_name = f"cleaned_{make_safe_stem(uploaded_file.name)}.csv"
            cleaned_csv_path = Path("output") / cleaned_csv_name
            cleaned_csv_path.parent.mkdir(parents=True, exist_ok=True)
            dataset_to_save_as_cleaned = cleaned_df

            if cleaning_summary.get("ecommerce_preprocessing_applied"):
                readable_dataset, ml_ready_dataset = build_ecommerce_output_datasets(cleaned_df)
                if cleaning_summary.get("smartphone_preprocessing_applied"):
                    readable_csv_name = "cleaned_readable_smartphone_dataset.csv"
                    ml_ready_csv_name = "ml_ready_smartphone_recommendation_dataset.csv"
                    dataset_to_save_as_cleaned = readable_dataset
                else:
                    readable_csv_name = f"cleaned_readable_{make_safe_stem(uploaded_file.name)}.csv"
                    ml_ready_csv_name = f"ml_ready_{make_safe_stem(uploaded_file.name)}.csv"
                readable_csv_path = Path("output") / readable_csv_name
                ml_ready_csv_path = Path("output") / ml_ready_csv_name
                readable_dataset.to_csv(readable_csv_path, index=False)
                ml_ready_dataset.to_csv(ml_ready_csv_path, index=False)
                if cleaning_summary.get("smartphone_preprocessing_applied"):
                    cleaning_summary["smartphone_validation_checks"] = validate_smartphone_outputs(
                        cleaned_df,
                        ml_ready_dataset,
                    )
                cleaning_summary["readable_cleaned_dataset"] = readable_dataset
                cleaning_summary["ml_ready_dataset"] = ml_ready_dataset
                cleaning_summary["readable_cleaned_csv_name"] = readable_csv_name
                cleaning_summary["ml_ready_csv_name"] = ml_ready_csv_name
                cleaning_summary["readable_cleaned_csv_path"] = str(readable_csv_path)
                cleaning_summary["ml_ready_csv_path"] = str(ml_ready_csv_path)
                cleaning_summary["readable_dataset_columns"] = list(readable_dataset.columns)
                cleaning_summary["ml_ready_dataset_columns"] = list(ml_ready_dataset.columns)
                cleaning_summary["readable_dataset_rows"] = len(readable_dataset)
                cleaning_summary["ml_ready_dataset_rows"] = len(ml_ready_dataset)
                cleaning_summary["constant_features_dropped_from_ml_ready"] = list(
                    ml_ready_dataset.attrs.get("constant_features_dropped_from_ml_ready", [])
                )
                cleaning_summary["rows_removed_total"] = cleaning_summary["original_rows"] - len(cleaned_df)
                if not cleaning_summary.get("row_removal_reasons"):
                    cleaning_summary["row_removal_reasons"] = [
                        {
                            "reason": "No rows were removed because no exact duplicates or unusable records were found.",
                            "rows_removed": 0,
                        }
                    ]

            dataset_to_save_as_cleaned.to_csv(cleaned_csv_path, index=False)

            cleaning_report, cleaning_report_path = generate_cleaning_report(
                profile,
                validation_result,
                cleaning_summary,
                ml_recommendation,
                uploaded_file.name,
                cleaned_file_path=cleaned_csv_path,
                flowise_metadata=st.session_state.get(
                    "profile_flowise_metadata",
                    build_default_flowise_metadata(),
                ),
            )
        except Exception as exc:
            st.error(f"Cleaning could not be completed safely. Details: {exc}")
            return

        # Persist all cleaning outputs so that download-button reruns do not
        # reset the UI. Everything written here survives subsequent reruns.
        st.session_state["cleaned_df"] = cleaned_df
        st.session_state["cleaning_summary"] = cleaning_summary
        st.session_state["cleaning_report"] = cleaning_report
        st.session_state["cleaned_csv_name"] = cleaned_csv_name
        st.session_state["cleaned_csv_path"] = cleaned_csv_path
        st.session_state["cleaning_report_path"] = cleaning_report_path
        st.session_state["cleaning_completed"] = True

    # Render cleaning results on every rerun as long as cleaning has been done
    # for the current file. Download buttons trigger reruns, so reading from
    # session_state here keeps the UI visible without re-running the pipeline.
    if st.session_state.get("cleaning_completed"):
        render_cleaning_results(
            st.session_state["cleaned_df"],
            st.session_state["cleaning_summary"],
            st.session_state["cleaning_report"],
            st.session_state["cleaned_csv_name"],
            st.session_state["cleaned_csv_path"],
            st.session_state["cleaning_report_path"],
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
