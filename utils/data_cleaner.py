"""Core cleaning helpers for the first dataset cleaning workflow."""

from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from utils.nlp_cleaner import clean_text_columns, detect_text_columns
from utils.data_profiler import classify_columns


def _get_mode_or_unknown(series: pd.Series) -> Any:
    """Return the most common value, or a safe fallback when no mode exists."""
    mode = series.mode(dropna=True)
    if not mode.empty:
        return mode.iloc[0]
    return "Unknown"


def _should_convert_to_numeric(series: pd.Series) -> bool:
    """Check whether an object column looks safely convertible to numeric."""
    non_null = series.dropna()
    if non_null.empty:
        return False

    as_text = non_null.astype(str).str.strip()
    converted = pd.to_numeric(as_text, errors="coerce")
    success_ratio = converted.notna().mean()
    return success_ratio >= 0.7


def _should_convert_to_datetime(series: pd.Series, column_name: str) -> bool:
    """Check whether an object column looks safely convertible to datetime."""
    non_null = series.dropna()
    if len(non_null) < 2:
        return False

    sample = non_null.astype(str).str.strip().head(100)
    column_name = column_name.lower()
    has_datetime_hint = any(
        token in column_name
        for token in ["date", "time", "timestamp", "created", "updated", "month", "year"]
    )
    has_format_hint = sample.str.contains(r"[-/:]", regex=True).mean() >= 0.5

    if not (has_datetime_hint or has_format_hint):
        return False

    converted = pd.to_datetime(sample, errors="coerce")
    success_ratio = converted.notna().mean()
    return success_ratio >= 0.7


def clean_dataset(
    df: pd.DataFrame,
    options: dict[str, bool],
    target_column: str | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Apply the selected starter cleaning steps to a validated dataset."""
    cleaned_df = df.copy(deep=True)
    cleaning_steps: list[str] = []
    skipped_steps: list[str] = []
    handle_missing_values_selected = options.get("handle_missing_values", False)
    fix_data_types_selected = options.get("fix_data_types", False)
    handle_outliers_selected = options.get("handle_outliers", False)
    encode_categorical_selected = options.get("encode_categorical", False)
    scale_numeric_selected = options.get("scale_numeric", False)
    nlp_cleaning_selected = options.get("nlp_cleaning", False)
    scaler_choice = options.get("scaler_choice", "StandardScaler")
    missing_filled: dict[str, dict[str, str | int]] = {}
    converted_columns: dict[str, dict[str, str]] = {}
    converted_numeric_columns: list[str] = []
    converted_date_columns: list[str] = []
    skipped_type_conversion_columns: dict[str, str] = {}
    type_conversion_notes: list[str] = []
    outlier_summary: list[dict[str, str | float | int]] = []
    encoded_columns: list[str] = []
    encoded_columns_generated_count = 0
    target_encoding_recommendation: str | None = None
    scaled_columns: list[str] = []
    scaler_used: str | None = None
    cleaned_text_columns: list[str] = []
    nlp_cleaning_actions: list[str] = []
    nlp_original_backup_columns: list[str] = []
    nlp_before_after_examples: dict[str, dict[str, str]] = {}

    original_rows = len(df)
    original_columns = len(df.columns)
    missing_values_before = df.isnull().sum().to_dict()

    def record_skipped_step(message: str) -> None:
        cleaning_steps.append(message)
        skipped_steps.append(message)

    # Pre-cleaning validation is assumed to have already passed upstream.
    if options.get("remove_duplicates", False):
        cleaned_df = cleaned_df.drop_duplicates()
        duplicates_removed = original_rows - len(cleaned_df)
        cleaning_steps.append(f"Removed {duplicates_removed} duplicate rows.")
    else:
        duplicates_removed = 0
        cleaning_steps.append("Skipped duplicate removal.")

    if fix_data_types_selected:
        for column in cleaned_df.columns:
            if column == target_column:
                continue

            series = cleaned_df[column]
            if not pd.api.types.is_object_dtype(series):
                continue

            try:
                if _should_convert_to_numeric(series):
                    cleaned_df[column] = pd.to_numeric(series, errors="coerce")
                    converted_columns[column] = {
                        "converted_to": "numeric",
                        "reason": "Numeric types help machine learning models work with quantitative values.",
                    }
                    converted_numeric_columns.append(column)
                    continue

                if _should_convert_to_datetime(series, column):
                    cleaned_df[column] = pd.to_datetime(series, errors="coerce")
                    converted_columns[column] = {
                        "converted_to": "datetime",
                        "reason": "Datetime conversion makes time-based features easier to clean and engineer for ML.",
                    }
                    converted_date_columns.append(column)
                else:
                    skipped_type_conversion_columns[column] = (
                        "Skipped because the column does not look reliably numeric or date-like."
                    )
            except Exception:
                # Skip unsafe conversions so the app continues without crashing.
                skipped_type_conversion_columns[column] = (
                    "Skipped because safe type conversion failed for this column."
                )
                continue

        if converted_columns:
            converted_list = ", ".join(sorted(converted_columns.keys()))
            cleaning_steps.append(f"Converted column data types for: {converted_list}.")
            cleaning_steps.append(
                "Type conversion helps ML because numeric and datetime values are easier to validate, clean, and transform."
            )
            if converted_numeric_columns:
                type_conversion_notes.append(
                    "Converted numeric-like text columns to numeric so models can use them as quantitative features."
                )
            if converted_date_columns:
                type_conversion_notes.append(
                    "Converted date-like text columns to datetime so time information can be handled more consistently."
                )
        else:
            record_skipped_step("Data type fixing was selected, but no safe column conversions were found, so this step was skipped.")
            type_conversion_notes.append("No columns met the safety threshold for type conversion.")
    else:
        cleaning_steps.append("Skipped wrong data type fixing.")
        type_conversion_notes.append("Wrong data type fixing was not selected.")

    if handle_missing_values_selected:
        for column in cleaned_df.columns:
            if column == target_column:
                continue
            missing_count = int(cleaned_df[column].isna().sum())

            if missing_count == 0:
                continue

            if pd.api.types.is_numeric_dtype(cleaned_df[column]):
                median_value = cleaned_df[column].median()

                if pd.isna(median_value):
                    cleaned_df[column] = cleaned_df[column].fillna(0)
                    fill_value = 0
                else:
                    cleaned_df[column] = cleaned_df[column].fillna(median_value)
                    fill_value = median_value

                missing_filled[column] = {
                    "missing_count_before": missing_count,
                    "strategy": "median",
                    "fill_value": str(fill_value),
                }
            else:
                mode_values = cleaned_df[column].mode(dropna=True)

                if not mode_values.empty:
                    fill_value = mode_values.iloc[0]
                    cleaned_df[column] = cleaned_df[column].fillna(fill_value)
                    strategy = "mode"
                else:
                    fill_value = "Unknown"
                    cleaned_df[column] = cleaned_df[column].fillna(fill_value)
                    strategy = "Unknown"

                missing_filled[column] = {
                    "missing_count_before": missing_count,
                    "strategy": strategy,
                    "fill_value": str(fill_value),
                }

        if missing_filled:
            filled_columns = sorted(missing_filled.keys())
            cleaning_steps.append(
                "Filled missing values in these columns: " + ", ".join(filled_columns) + "."
            )
        else:
            filled_columns = []
            record_skipped_step("Missing value handling was selected, but no missing values were found, so this step was skipped.")
    else:
        filled_columns = []
        cleaning_steps.append(
            "Missing value handling was not selected, so missing values were not changed."
        )

    if handle_outliers_selected:
        numeric_columns = cleaned_df.select_dtypes(include=["number"]).columns

        for column in numeric_columns:
            if column == target_column:
                continue

            series = cleaned_df[column].dropna()
            if len(series) < 4:
                continue

            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1

            if pd.isna(iqr) or iqr == 0:
                continue

            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            outlier_mask = (cleaned_df[column] < lower_bound) | (cleaned_df[column] > upper_bound)
            outlier_count = int(outlier_mask.sum())

            if outlier_count == 0:
                continue

            cleaned_df[column] = cleaned_df[column].clip(lower=lower_bound, upper=upper_bound)
            outlier_summary.append(
                {
                    "column_name": column,
                    "lower_bound": float(lower_bound),
                    "upper_bound": float(upper_bound),
                    "outliers_detected": outlier_count,
                    "action_taken": "capped/winsorized",
                }
            )

        if outlier_summary:
            capped_columns = ", ".join(item["column_name"] for item in outlier_summary)
            cleaning_steps.append(f"Capped outliers using the IQR method for: {capped_columns}.")
            cleaning_steps.append(
                "Outliers were capped instead of deleting rows, because extreme values should not be removed blindly."
            )
        else:
            record_skipped_step("Outlier handling was selected, but no numeric feature columns with detectable outliers were found, so this step was skipped.")
    else:
        cleaning_steps.append("Skipped outlier handling.")

    if encode_categorical_selected:
        classification = classify_columns(cleaned_df, target_column=target_column)
        detected_text_columns = set(detect_text_columns(cleaned_df, target_column=target_column))
        candidate_columns = [
            column
            for column in classification["categorical_columns"]
            if column != target_column
            and column in cleaned_df.columns
            and column not in detected_text_columns
            and not str(column).endswith("_original")
        ]

        if candidate_columns:
            original_column_count = len(cleaned_df.columns)
            cleaned_df = pd.get_dummies(
                cleaned_df,
                columns=candidate_columns,
                drop_first=False,
            )
            encoded_columns = sorted(candidate_columns)
            encoded_columns_generated_count = len(cleaned_df.columns) - original_column_count
            cleaning_steps.append(
                "Encoded categorical columns with one-hot encoding: "
                + ", ".join(encoded_columns)
                + "."
            )
            cleaning_steps.append(
                "ML models need numeric input, so text categories are converted into 0/1 columns."
            )
        else:
            record_skipped_step(
                "Encoding was selected, but no categorical feature columns were found, so this step was skipped."
            )

        if target_column and target_column in classification["categorical_columns"]:
            target_encoding_recommendation = (
                f"Target column '{target_column}' looks categorical. Keep it separate from feature "
                "encoding and only encode it later if the chosen ML workflow requires it."
            )
    else:
        cleaning_steps.append("Skipped categorical encoding.")

    if nlp_cleaning_selected:
        candidate_text_columns = detect_text_columns(cleaned_df, target_column=target_column)

        if candidate_text_columns:
            (
                cleaned_df,
                cleaned_text_columns,
                nlp_original_backup_columns,
                nlp_before_after_examples,
            ) = clean_text_columns(
                cleaned_df,
                candidate_text_columns,
                remove_numbers=True,
                remove_stopwords=True,
                use_stemming=False,
            )
            nlp_cleaning_actions = [
                "Converted text to lowercase.",
                "Removed URLs.",
                "Removed emails.",
                "Removed HTML tags.",
                "Removed punctuation.",
                "Removed special characters.",
                "Removed numbers.",
                "Removed extra whitespace.",
                "Removed common English stopwords.",
                "Tokenized text internally and joined cleaned tokens back into strings.",
            ]
            if cleaned_text_columns:
                cleaning_steps.append(
                    "Applied NLP cleaning to text columns: "
                    + ", ".join(cleaned_text_columns)
                    + "."
                )
                cleaning_steps.append(
                    "Cleaned text can later be converted into numeric features using TF-IDF or Bag-of-Words."
                )
            else:
                record_skipped_step(
                    "NLP cleaning was selected, but no text columns were cleaned safely, so this step was skipped."
                )
        else:
            record_skipped_step(
                "NLP cleaning was selected, but no text columns were found, so this step was skipped."
            )
    else:
        cleaning_steps.append("Skipped NLP text cleaning.")

    if scale_numeric_selected:
        numeric_columns = cleaned_df.select_dtypes(include=["number"]).columns.tolist()
        feature_numeric_columns = [column for column in numeric_columns if column != target_column]

        if feature_numeric_columns:
            try:
                if scaler_choice == "MinMaxScaler":
                    scaler = MinMaxScaler()
                    scaler_used = "MinMaxScaler"
                    cleaning_steps.append(
                        "MinMaxScaler scales values into a fixed range, usually 0 to 1."
                    )
                else:
                    scaler = StandardScaler()
                    scaler_used = "StandardScaler"
                    cleaning_steps.append(
                        "StandardScaler standardizes values around mean 0 and standard deviation 1."
                    )

                cleaned_df[feature_numeric_columns] = scaler.fit_transform(
                    cleaned_df[feature_numeric_columns]
                )
                scaled_columns = feature_numeric_columns
                cleaning_steps.append(
                    "Scaled numeric feature columns: " + ", ".join(scaled_columns) + "."
                )
                cleaning_steps.append(
                    "Scaling helps algorithms like Logistic Regression, Linear Regression, KNN, SVM, and Neural Networks."
                )
            except Exception:
                scaler_used = scaler_choice
                record_skipped_step(
                    "Scaling was selected, but numeric scaling could not be applied safely, so this step was skipped."
                )
        else:
            scaler_used = scaler_choice
            record_skipped_step(
                "Scaling was selected, but no numeric feature columns were found, so this step was skipped."
            )
    else:
        cleaning_steps.append("Skipped numeric scaling.")

    missing_values_after = cleaned_df.isnull().sum().to_dict()

    cleaning_summary = {
        "original_rows": original_rows,
        "original_columns": original_columns,
        "final_rows": len(cleaned_df),
        "final_columns": len(cleaned_df.columns),
        "missing_values_before": missing_values_before,
        "missing_values_after": missing_values_after,
        "missing_filled": missing_filled,
        "converted_columns": converted_columns,
        "converted_numeric_columns": sorted(converted_numeric_columns),
        "converted_date_columns": sorted(converted_date_columns),
        "skipped_type_conversion_columns": skipped_type_conversion_columns,
        "type_conversion_notes": type_conversion_notes,
        "duplicate_rows_removed": duplicates_removed,
        "columns_where_missing_values_were_filled": filled_columns,
        "outlier_summary": outlier_summary,
        "encoded_columns": encoded_columns,
        "encoded_columns_generated_count": encoded_columns_generated_count,
        "target_encoding_recommendation": target_encoding_recommendation,
        "cleaned_text_columns": cleaned_text_columns,
        "nlp_cleaning_actions": nlp_cleaning_actions,
        "nlp_original_backup_columns": nlp_original_backup_columns,
        "nlp_before_after_examples": nlp_before_after_examples,
        "scaled_columns": scaled_columns,
        "scaler_used": scaler_used,
        "options_used": options.copy(),
        "cleaning_steps": cleaning_steps,
        "skipped_steps": skipped_steps,
    }

    return cleaned_df, cleaning_summary
