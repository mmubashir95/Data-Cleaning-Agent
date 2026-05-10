"""Core cleaning helpers for the first dataset cleaning workflow."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from utils.data_profiler import classify_columns
from utils.ecommerce_preprocessing import (
    apply_mobile_domain_outlier_rules,
    build_ecommerce_preprocessed_view,
    build_semantic_product_key,
    detect_mobile_ecommerce_dataset,
)
from utils.nlp_cleaner import clean_text_columns, detect_text_columns
from utils.smartphone_preprocessing import (
    BOOLEAN_FEATURE_COLUMNS,
    apply_smartphone_domain_outlier_rules,
    detect_smartphone_dataset,
    validate_smartphone_dataset_quality,
    validate_smartphone_outputs,
)


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
    return converted.notna().mean() >= 0.7


def _should_convert_to_datetime(series: pd.Series, column_name: str) -> bool:
    """Check whether an object column looks safely convertible to datetime."""
    non_null = series.dropna()
    if len(non_null) < 2:
        return False

    sample = non_null.astype(str).str.strip().head(100)
    lowered_name = column_name.lower()
    has_datetime_hint = any(
        token in lowered_name
        for token in ["date", "time", "timestamp", "created", "updated", "month", "year"]
    )
    has_format_hint = sample.str.contains(r"[-/:]", regex=True).mean() >= 0.5
    if not (has_datetime_hint or has_format_hint):
        return False

    converted = pd.to_datetime(sample, format="mixed", errors="coerce")
    return converted.notna().mean() >= 0.7


def clean_dataset(
    df: pd.DataFrame,
    options: dict[str, bool],
    target_column: str | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Apply the selected cleaning steps with safe smartphone-specific branching."""
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
    scaling_reference_stats: dict[str, dict[str, float]] = {}
    cleaned_text_columns: list[str] = []
    nlp_cleaning_actions: list[str] = []
    nlp_original_backup_columns: list[str] = []
    nlp_before_after_examples: dict[str, dict[str, str]] = {}
    ecommerce_preprocessing_applied = False
    smartphone_preprocessing_applied = False
    cleaned_numeric_columns: list[str] = []
    extracted_feature_columns: list[str] = []
    normalized_categorical_columns: list[str] = []
    dropped_reference_columns: list[str] = []
    recommendation_ready = False
    columns_excluded_from_ml: list[str] = []
    semantic_duplicate_rows_removed = 0
    exact_duplicate_rows_removed = 0
    deduplication_strategy = "exact row matching"
    scaled_columns_created: list[str] = []
    original_numeric_columns_preserved = False
    domain_outlier_rules_applied = False
    domain_outlier_adjustments: list[dict[str, Any]] = []
    shifted_column_fixes: list[dict[str, Any]] = []
    noise_fixes: list[dict[str, Any]] = []
    smartphone_columns_dropped: list[str] = []
    scalable_numeric_feature_columns: list[str] = []
    smartphone_validation_checks: list[dict[str, Any]] = []
    smartphone_dataset_quality: dict[str, Any] = {}
    row_removal_reasons: list[dict[str, Any]] = []

    original_rows = len(df)
    original_columns = len(df.columns)
    missing_values_before = df.isnull().sum().to_dict()

    effective_target_column = target_column
    if (
        detect_smartphone_dataset(df.columns)
        and target_column is not None
        and str(target_column).strip().lower() == "segment"
        and target_column in df.columns
        and df[target_column].isna().all()
    ):
        effective_target_column = None
        cleaning_steps.append(
            "Ignored the empty Segment column as a target because this smartphone dataset is prepared for content-based recommendation, not classification."
        )

    original_classification = classify_columns(df, target_column=effective_target_column)
    is_ecommerce_dataset = detect_mobile_ecommerce_dataset(df.columns)

    def record_skipped_step(message: str) -> None:
        cleaning_steps.append(message)
        skipped_steps.append(message)

    if options.get("remove_duplicates", False):
        cleaned_df = cleaned_df.drop_duplicates()
        duplicates_removed = original_rows - len(cleaned_df)
        exact_duplicate_rows_removed = duplicates_removed
        cleaning_steps.append(f"Removed {duplicates_removed} duplicate rows.")
        if duplicates_removed > 0:
            row_removal_reasons.append(
                {
                    "reason": "Exact duplicate rows removed",
                    "rows_removed": duplicates_removed,
                }
            )
    else:
        duplicates_removed = 0
        cleaning_steps.append("Skipped duplicate removal.")

    if is_ecommerce_dataset:
        ecommerce_view, ecommerce_metadata = build_ecommerce_preprocessed_view(
            cleaned_df,
            drop_reference_columns=True,
        )
        cleaned_df = ecommerce_view
        ecommerce_preprocessing_applied = ecommerce_metadata.get("ecommerce_preprocessing_applied", False)
        smartphone_preprocessing_applied = ecommerce_metadata.get("smartphone_preprocessing_applied", False)
        cleaned_numeric_columns = ecommerce_metadata.get("cleaned_numeric_columns", [])
        extracted_feature_columns = ecommerce_metadata.get("extracted_feature_columns", [])
        normalized_categorical_columns = ecommerce_metadata.get("normalized_categorical_columns", [])
        dropped_reference_columns = ecommerce_metadata.get("dropped_reference_columns", [])
        recommendation_ready = ecommerce_metadata.get("recommendation_ready", False)
        columns_excluded_from_ml = ecommerce_metadata.get("raw_source_columns_excluded_from_ml", [])
        shifted_column_fixes = ecommerce_metadata.get("shifted_column_fixes", [])
        noise_fixes = ecommerce_metadata.get("noise_fixes", [])
        smartphone_columns_dropped = ecommerce_metadata.get("columns_dropped", [])
        scalable_numeric_feature_columns = ecommerce_metadata.get("scalable_numeric_feature_columns", [])
        smartphone_quality_mode = options.get("smartphone_quality_mode", "safe")

        if smartphone_preprocessing_applied:
            cleaned_df, smartphone_dataset_quality = validate_smartphone_dataset_quality(
                cleaned_df,
                mode=smartphone_quality_mode,
            )
            suspicious_records = smartphone_dataset_quality.get("suspicious_records_details", [])
            if suspicious_records:
                cleaning_steps.append(
                    f"Ran smartphone data-validity checks in {smartphone_quality_mode} mode and flagged {len(suspicious_records)} suspicious record(s)."
                )
            if smartphone_dataset_quality.get("rows_removed_in_strict_mode", 0) > 0:
                removed_count = smartphone_dataset_quality["rows_removed_in_strict_mode"]
                cleaning_steps.append(
                    f"Strict smartphone quality mode removed {removed_count} critical invalid row(s) before ML preparation."
                )
                row_removal_reasons.append(
                    {
                        "reason": "Critical suspicious smartphone rows removed in strict quality mode",
                        "rows_removed": removed_count,
                    }
                )

        if smartphone_preprocessing_applied:
            cleaning_steps.append(
                "Detected the complex smartphone recommendation dataset and applied smartphone-specific feature extraction, noisy text repair, and shifted-column handling before the generic cleaning steps."
            )
            if smartphone_columns_dropped:
                cleaning_steps.append(
                    "Dropped unusable smartphone columns: "
                    + ", ".join(sorted(set(smartphone_columns_dropped)))
                    + "."
                )
            deduplication_strategy = "exact row matching plus smartphone product matching"
        elif ecommerce_preprocessing_applied:
            cleaning_steps.append(
                "Applied mobile e-commerce preprocessing to parse scraped numeric fields, normalize brand names, and remove reference-only URL columns from the ML-ready output."
            )
            deduplication_strategy = "exact row matching plus normalized product-name, brand, and numeric-price matching"

        if options.get("remove_duplicates", False) and {"product_name", "brand", "price"}.intersection(cleaned_df.columns):
            if smartphone_preprocessing_applied:
                cleaning_steps.append(
                    "Skipped smartphone semantic duplicate removal to preserve recommendation candidates. Only exact duplicate rows are removed for this dataset."
                )
            else:
                semantic_keys = build_semantic_product_key(cleaned_df)
                before_semantic_dedup_rows = len(cleaned_df)
                cleaned_df = cleaned_df.loc[~semantic_keys.duplicated()].copy()
                semantic_duplicate_rows_removed = before_semantic_dedup_rows - len(cleaned_df)
                duplicates_removed += semantic_duplicate_rows_removed
                if semantic_duplicate_rows_removed > 0:
                    cleaning_steps.append(
                        f"Removed {semantic_duplicate_rows_removed} near-duplicate product rows using normalized product name, brand, and price."
                    )
                    row_removal_reasons.append(
                        {
                            "reason": "Semantic near-duplicate product rows removed",
                            "rows_removed": semantic_duplicate_rows_removed,
                        }
                    )

    if fix_data_types_selected:
        for column in cleaned_df.columns:
            if column == effective_target_column:
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
                skipped_type_conversion_columns[column] = (
                    "Skipped because safe type conversion failed for this column."
                )

        if ecommerce_preprocessing_applied and (cleaned_numeric_columns or extracted_feature_columns):
            converted_numeric_columns = sorted(
                set(converted_numeric_columns + cleaned_numeric_columns + extracted_feature_columns)
            )
            for column in cleaned_numeric_columns:
                converted_columns.setdefault(
                    column,
                    {
                        "converted_to": "numeric",
                        "reason": "Scraped numeric text was converted into real numeric values for ML-ready preprocessing.",
                    },
                )
            for column in extracted_feature_columns:
                converted_columns.setdefault(
                    column,
                    {
                        "converted_to": "numeric",
                        "reason": "A numeric feature was extracted from a scraped device specification field.",
                    },
                )

        if converted_columns:
            cleaning_steps.append(
                "Converted column data types for: " + ", ".join(sorted(converted_columns.keys())) + "."
            )
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
            record_skipped_step(
                "Data type fixing was selected, but no safe column conversions were found, so this step was skipped."
            )
            type_conversion_notes.append("No columns met the safety threshold for type conversion.")
    else:
        cleaning_steps.append("Skipped wrong data type fixing.")
        type_conversion_notes.append("Wrong data type fixing was not selected.")

    if handle_missing_values_selected:
        for column in cleaned_df.columns:
            if column == effective_target_column:
                continue
            missing_count = int(cleaned_df[column].isna().sum())
            if missing_count == 0:
                continue

            if smartphone_preprocessing_applied and column in normalized_categorical_columns:
                cleaned_df[column] = cleaned_df[column].fillna("Unknown")
                missing_filled[column] = {
                    "missing_count_before": missing_count,
                    "strategy": "Unknown",
                    "fill_value": "Unknown",
                }
            elif smartphone_preprocessing_applied and column in BOOLEAN_FEATURE_COLUMNS:
                cleaned_df[column] = cleaned_df[column].fillna(0).astype(int)
                missing_filled[column] = {
                    "missing_count_before": missing_count,
                    "strategy": "false/0",
                    "fill_value": "0",
                }
            elif pd.api.types.is_numeric_dtype(cleaned_df[column]):
                median_value = np.median(cleaned_df[column].dropna())
                fill_value = 0 if pd.isna(median_value) else median_value
                cleaned_df[column] = cleaned_df[column].fillna(fill_value)
                missing_filled[column] = {
                    "missing_count_before": missing_count,
                    "strategy": "median",
                    "fill_value": str(fill_value),
                }
            else:
                fill_value = _get_mode_or_unknown(cleaned_df[column])
                cleaned_df[column] = cleaned_df[column].fillna(fill_value)
                missing_filled[column] = {
                    "missing_count_before": missing_count,
                    "strategy": "mode" if fill_value != "Unknown" else "Unknown",
                    "fill_value": str(fill_value),
                }

        if missing_filled:
            filled_columns = sorted(missing_filled.keys())
            cleaning_steps.append(
                "Filled missing values in these columns: " + ", ".join(filled_columns) + "."
            )
        else:
            filled_columns = []
            record_skipped_step(
                "Missing value handling was selected, but no missing values were found, so this step was skipped."
            )
    else:
        filled_columns = []
        cleaning_steps.append("Missing value handling was not selected, so missing values were not changed.")

    if handle_outliers_selected:
        if smartphone_preprocessing_applied:
            cleaned_df, domain_outlier_adjustments = apply_smartphone_domain_outlier_rules(cleaned_df)
            domain_outlier_rules_applied = True
            if domain_outlier_adjustments:
                cleaning_steps.append(
                    "Applied smartphone-specific validation bounds for price, rating, battery, display, refresh rate, charging wattage, and processor speed."
                )
        elif ecommerce_preprocessing_applied:
            cleaned_df, domain_outlier_adjustments = apply_mobile_domain_outlier_rules(cleaned_df)
            domain_outlier_rules_applied = True
            if domain_outlier_adjustments:
                cleaning_steps.append(
                    "Applied domain-safe outlier validation rules for mobile phone fields such as rating, battery capacity, and screen size."
                )

        skip_iqr_columns = set()
        if smartphone_preprocessing_applied:
            skip_iqr_columns = {
                "price",
                "rating",
                "battery_mah",
                "screen_size_inches",
                "refresh_rate_hz",
                "charging_watt",
                "processor_speed_ghz",
                "memory_card_max_gb",
            }
        elif ecommerce_preprocessing_applied:
            skip_iqr_columns = {"rating", "battery_mah", "screen_size_inches"}

        for column in cleaned_df.select_dtypes(include=["number"]).columns:
            if column == effective_target_column or column in skip_iqr_columns:
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
            outlier_count = int(((cleaned_df[column] < lower_bound) | (cleaned_df[column] > upper_bound)).sum())
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
            cleaning_steps.append(
                "Capped outliers using the IQR method for: "
                + ", ".join(item["column_name"] for item in outlier_summary)
                + "."
            )
            cleaning_steps.append(
                "Outliers were capped instead of deleting rows, because extreme values should not be removed blindly."
            )
        elif not domain_outlier_adjustments:
            record_skipped_step(
                "Outlier handling was selected, but no numeric feature columns with detectable outliers were found, so this step was skipped."
            )
    else:
        cleaning_steps.append("Skipped outlier handling.")

    if smartphone_preprocessing_applied:
        repaired_columns: list[str] = []
        numeric_repair_columns = sorted(
            {
                *cleaned_numeric_columns,
                *extracted_feature_columns,
                "price_scaled",
                "rating_scaled",
                "screen_size_inches_scaled",
            }
        )
        for column in normalized_categorical_columns:
            if column in cleaned_df.columns and cleaned_df[column].isna().any():
                cleaned_df[column] = cleaned_df[column].fillna("Unknown")
                repaired_columns.append(column)
        for column in BOOLEAN_FEATURE_COLUMNS:
            if column in cleaned_df.columns and cleaned_df[column].isna().any():
                cleaned_df[column] = cleaned_df[column].fillna(0).astype(int)
                repaired_columns.append(column)
        for column in numeric_repair_columns:
            if column not in cleaned_df.columns or not cleaned_df[column].isna().any():
                continue
            series = pd.to_numeric(cleaned_df[column], errors="coerce")
            median_value = series.median(skipna=True)
            cleaned_df[column] = series.fillna(0 if pd.isna(median_value) else median_value)
            repaired_columns.append(column)
        if repaired_columns:
            cleaning_steps.append(
                "Repaired remaining smartphone-specific missing values after validation and outlier checks for: "
                + ", ".join(sorted(set(repaired_columns)))
                + "."
            )

    if encode_categorical_selected:
        classification = classify_columns(cleaned_df, target_column=effective_target_column)
        detected_text_columns = set(detect_text_columns(cleaned_df, target_column=effective_target_column))
        if smartphone_preprocessing_applied:
            categorical_override = {
                column
                for column in {"brand", "processor_brand", "os_family", "price_segment"}
                if column in classification["categorical_columns"]
            }
        elif ecommerce_preprocessing_applied:
            categorical_override = {
                column
                for column in {"brand", "availability"}
                if column in classification["categorical_columns"]
            }
        else:
            categorical_override = set()

        candidate_columns = [
            column
            for column in classification["categorical_columns"]
            if column != effective_target_column
            and column in cleaned_df.columns
            and (column not in detected_text_columns or column in categorical_override)
            and not str(column).endswith("_original")
            and column not in dropped_reference_columns
            and column not in columns_excluded_from_ml
            and column not in {"product_name", "model", "phone_id"}
        ]

        if candidate_columns:
            original_column_count = len(cleaned_df.columns)
            cleaned_df = pd.get_dummies(cleaned_df, columns=candidate_columns, drop_first=False)
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

        if effective_target_column and effective_target_column in classification["categorical_columns"]:
            target_encoding_recommendation = (
                f"Target column '{effective_target_column}' looks categorical. Keep it separate from feature "
                "encoding and only encode it later if the chosen ML workflow requires it."
            )
    else:
        cleaning_steps.append("Skipped categorical encoding.")

    if nlp_cleaning_selected:
        candidate_text_columns = detect_text_columns(cleaned_df, target_column=effective_target_column)
        if ecommerce_preprocessing_applied:
            protected_text_columns = {
                "product_name",
                "brand",
                "price",
                "rating",
                "review_count",
                "ram",
                "storage",
                "battery",
                "screen_size",
                *dropped_reference_columns,
            }
            if smartphone_preprocessing_applied:
                protected_text_columns.update(
                    {
                        "model",
                        "sim",
                        "processor",
                        "display",
                        "camera",
                        "card",
                        "os",
                        "combined_text_features",
                        "extra_features",
                    }
                )
            candidate_text_columns = [
                column for column in candidate_text_columns if column not in protected_text_columns
            ]

        if candidate_text_columns:
            cleaned_df, cleaned_text_columns, nlp_original_backup_columns, nlp_before_after_examples = clean_text_columns(
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
                    "Applied NLP cleaning to text columns: " + ", ".join(cleaned_text_columns) + "."
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
        if smartphone_preprocessing_applied and scalable_numeric_feature_columns:
            feature_numeric_columns = [
                column
                for column in scalable_numeric_feature_columns
                if column in numeric_columns and column != effective_target_column
            ]
        else:
            feature_numeric_columns = [column for column in numeric_columns if column != effective_target_column]

        if feature_numeric_columns:
            try:
                effective_scaler_choice = (
                    "MinMaxScaler" if smartphone_preprocessing_applied else scaler_choice
                )
                if smartphone_preprocessing_applied and scaler_choice != "MinMaxScaler":
                    cleaning_steps.append(
                        "For the smartphone recommendation dataset, numeric recommendation features are scaled with MinMaxScaler so cosine similarity compares them on a consistent 0 to 1 range."
                    )
                if effective_scaler_choice == "MinMaxScaler":
                    scaler = MinMaxScaler()
                    scaler_used = "MinMaxScaler"
                    cleaning_steps.append("MinMaxScaler scales values into a fixed range, usually 0 to 1.")
                else:
                    scaler = StandardScaler()
                    scaler_used = "StandardScaler"
                    scaling_reference_stats = {
                        column: {
                            "mean_before_scaling": float(np.mean(cleaned_df[column])),
                            "std_before_scaling": float(np.std(cleaned_df[column])),
                        }
                        for column in feature_numeric_columns
                    }
                    cleaning_steps.append(
                        "StandardScaler standardizes values around mean 0 and standard deviation 1."
                    )
                scaled_result = scaler.fit_transform(cleaned_df[feature_numeric_columns])
                scaled_columns = feature_numeric_columns
                if ecommerce_preprocessing_applied:
                    original_numeric_columns_preserved = True
                    scaled_columns_created = [f"{column}_scaled" for column in feature_numeric_columns]
                    cleaned_df[scaled_columns_created] = scaled_result
                else:
                    cleaned_df[feature_numeric_columns] = scaled_result
                cleaning_steps.append(
                    "Scaled numeric feature columns: " + ", ".join(scaled_columns) + "."
                )
                if ecommerce_preprocessing_applied and scaled_columns_created:
                    cleaning_steps.append(
                        "Created separate scaled feature columns while preserving readable numeric values for viva and review."
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
    cleaned_classification = classify_columns(cleaned_df, target_column=effective_target_column)

    before_vs_after_summary = {
        "metrics": [
            {"metric": "Row count", "before": original_rows, "after": len(cleaned_df)},
            {"metric": "Column count", "before": original_columns, "after": len(cleaned_df.columns)},
            {
                "metric": "Total missing values",
                "before": int(sum(missing_values_before.values())),
                "after": int(sum(missing_values_after.values())),
            },
            {
                "metric": "Duplicate rows",
                "before": int(df.duplicated().sum()),
                "after": int(cleaned_df.duplicated().sum()),
            },
            {
                "metric": "Categorical columns count",
                "before": len(original_classification["categorical_columns"]),
                "after": len(cleaned_classification["categorical_columns"]),
            },
            {
                "metric": "Numeric columns count",
                "before": len(original_classification["numeric_columns"]),
                "after": len(cleaned_classification["numeric_columns"]),
            },
        ]
    }

    if smartphone_preprocessing_applied:
        smartphone_validation_checks = validate_smartphone_outputs(cleaned_df)

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
        "exact_duplicate_rows_removed": exact_duplicate_rows_removed,
        "semantic_duplicate_rows_removed": semantic_duplicate_rows_removed,
        "near_duplicates_removed": semantic_duplicate_rows_removed,
        "deduplication_strategy": deduplication_strategy,
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
        "scaled_columns_created": scaled_columns_created,
        "scaler_used": scaler_used,
        "scaling_reference_stats": scaling_reference_stats,
        "ecommerce_preprocessing_applied": ecommerce_preprocessing_applied,
        "smartphone_preprocessing_applied": smartphone_preprocessing_applied,
        "cleaned_numeric_columns": sorted(set(cleaned_numeric_columns)),
        "extracted_feature_columns": sorted(set(extracted_feature_columns)),
        "normalized_categorical_columns": sorted(set(normalized_categorical_columns)),
        "dropped_reference_columns": sorted(set(dropped_reference_columns)),
        "columns_excluded_from_ml": sorted(set(columns_excluded_from_ml)),
        "shifted_column_fixes": shifted_column_fixes,
        "noise_fixes": noise_fixes,
        "smartphone_columns_dropped": sorted(set(smartphone_columns_dropped)),
        "scalable_numeric_feature_columns": scalable_numeric_feature_columns,
        "smartphone_validation_checks": smartphone_validation_checks,
        "smartphone_dataset_quality": smartphone_dataset_quality,
        "suspicious_records_details": smartphone_dataset_quality.get("suspicious_records_details", []),
        "invalid_ml_ready_brand_columns": smartphone_dataset_quality.get("invalid_ml_ready_brand_columns", []),
        "row_removal_reasons": row_removal_reasons,
        "recommendation_ready": recommendation_ready,
        "original_numeric_columns_preserved": original_numeric_columns_preserved,
        "domain_outlier_rules_applied": domain_outlier_rules_applied,
        "domain_outlier_adjustments": domain_outlier_adjustments,
        "before_vs_after_summary": before_vs_after_summary,
        "options_used": options.copy(),
        "cleaning_steps": cleaning_steps,
        "skipped_steps": skipped_steps,
    }

    return cleaned_df, cleaning_summary
