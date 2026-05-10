"""EDA report data extraction: findings, data issues, and cleaning methods."""

from __future__ import annotations

from typing import Any

import pandas as pd


def generate_eda_findings(df: pd.DataFrame, cleaning_summary: dict[str, Any]) -> dict[str, Any]:
    """Extract written EDA insights from the cleaned dataset."""
    findings: dict[str, Any] = {}

    if "brand" in df.columns:
        brand_counts = df["brand"].dropna().astype(str).value_counts()
        if not brand_counts.empty:
            findings["top_brands"] = {str(k): int(v) for k, v in brand_counts.head(5).items()}
            findings["top_brand"] = str(brand_counts.index[0])
            findings["unique_brand_count"] = int(brand_counts.nunique())

    if "price" in df.columns:
        price_series = pd.to_numeric(df["price"], errors="coerce").dropna()
        if not price_series.empty:
            findings["price_min"] = round(float(price_series.min()), 2)
            findings["price_max"] = round(float(price_series.max()), 2)
            findings["price_median"] = round(float(price_series.median()), 2)
            findings["price_mean"] = round(float(price_series.mean()), 2)

    if "rating" in df.columns:
        rating_series = pd.to_numeric(df["rating"], errors="coerce").dropna()
        if not rating_series.empty:
            findings["rating_mean"] = round(float(rating_series.mean()), 2)
            findings["rating_median"] = round(float(rating_series.median()), 2)
            findings["rating_min"] = round(float(rating_series.min()), 2)
            findings["rating_max"] = round(float(rating_series.max()), 2)

    if "ram_gb" in df.columns:
        ram_series = pd.to_numeric(df["ram_gb"], errors="coerce").dropna()
        if not ram_series.empty:
            findings["common_ram_gb_values"] = {str(k): int(v) for k, v in ram_series.value_counts().head(3).items()}
            findings["ram_gb_median"] = round(float(ram_series.median()), 1)

    if "storage_gb" in df.columns:
        storage_series = pd.to_numeric(df["storage_gb"], errors="coerce").dropna()
        if not storage_series.empty:
            findings["common_storage_gb_values"] = {str(k): int(v) for k, v in storage_series.value_counts().head(3).items()}
            findings["storage_gb_median"] = round(float(storage_series.median()), 1)

    if "battery_mah" in df.columns:
        bat_series = pd.to_numeric(df["battery_mah"], errors="coerce").dropna()
        if not bat_series.empty:
            findings["battery_mean_mah"] = round(float(bat_series.mean()), 0)
            findings["battery_median_mah"] = round(float(bat_series.median()), 0)
            findings["battery_min_mah"] = round(float(bat_series.min()), 0)
            findings["battery_max_mah"] = round(float(bat_series.max()), 0)

    if "os_family" in df.columns:
        os_counts = df["os_family"].dropna().astype(str).value_counts()
        if not os_counts.empty:
            findings["os_distribution"] = {str(k): int(v) for k, v in os_counts.items()}
            findings["dominant_os"] = str(os_counts.index[0])

    if "has_5g" in df.columns:
        total = len(df)
        five_g_count = int(pd.to_numeric(df["has_5g"], errors="coerce").fillna(0).sum())
        findings["smartphones_with_5g_percent"] = round(five_g_count / total * 100, 1) if total > 0 else 0.0

    filtering = cleaning_summary.get("smartphone_filtering_summary", {})
    if filtering:
        findings["non_smartphones_removed"] = int(filtering.get("total_removed", 0))
        findings["total_rows_after_filter"] = int(filtering.get("total_rows_after", len(df)))

    return findings


def generate_data_issues_summary(
    cleaning_summary: dict[str, Any],
    profile: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Consolidate all detected data quality issues into a single summary."""
    issues: dict[str, Any] = {}

    missing_before = cleaning_summary.get("missing_values_before", {})
    issues["missing_values_columns"] = {col: int(count) for col, count in missing_before.items() if count > 0}
    issues["total_missing_values"] = int(sum(issues["missing_values_columns"].values()))

    issues["duplicate_rows"] = int(cleaning_summary.get("duplicate_rows_removed", 0))

    shifted = cleaning_summary.get("shifted_column_fixes", [])
    issues["column_shift_issues_fixed"] = len(shifted)
    issues["column_shift_examples"] = [str(f) for f in shifted[:3]]

    issues["noisy_text_cells_fixed"] = len(cleaning_summary.get("noise_fixes", []))

    filtering = cleaning_summary.get("smartphone_filtering_summary", {})
    issues["non_smartphone_records_removed"] = int(filtering.get("total_removed", 0))

    semantic = cleaning_summary.get("semantic_validation_summary", {})
    contamination_count = int(sum(
        v for k, v in semantic.items()
        if isinstance(v, int) and "contamination" in k
    ))
    issues["display_contamination_rows"] = contamination_count
    issues["display_contamination_breakdown"] = {
        k.replace("display_contamination_", ""): int(v)
        for k, v in semantic.items()
        if isinstance(v, int) and "contamination" in k
    }

    smartphone_quality = cleaning_summary.get("smartphone_dataset_quality", {})
    issues["suspicious_records_count"] = int(smartphone_quality.get("suspicious_records_count", 0))
    issues["critical_records_count"] = int(smartphone_quality.get("critical_suspicious_records_count", 0))

    outlier_summary = cleaning_summary.get("outlier_summary", [])
    issues["outlier_columns"] = [item.get("column_name", "") for item in outlier_summary]
    issues["outlier_column_count"] = len(issues["outlier_columns"])

    issues["wrong_type_columns_fixed"] = list(cleaning_summary.get("converted_columns", {}).keys())
    issues["columns_requiring_feature_extraction"] = cleaning_summary.get("extracted_feature_columns", [])

    return issues


def generate_cleaning_methods_summary(cleaning_summary: dict[str, Any]) -> dict[str, Any]:
    """Describe each cleaning step that was applied to the dataset."""
    methods: dict[str, Any] = {}

    missing_filled = cleaning_summary.get("missing_filled", {})
    if missing_filled:
        sample = list(missing_filled.keys())[:4]
        rest = max(0, len(missing_filled) - 4)
        methods["missing_value_handling"] = (
            "Filled: " + ", ".join(sample) + (f" and {rest} more" if rest else "") + "."
        )
    else:
        methods["missing_value_handling"] = "Not selected."

    dup_removed = int(cleaning_summary.get("duplicate_rows_removed", 0))
    methods["duplicate_removal"] = (
        f"Removed {dup_removed} exact duplicate rows." if dup_removed else "No duplicates found."
    )

    noise_count = len(cleaning_summary.get("noise_fixes", []))
    shift_count = len(cleaning_summary.get("shifted_column_fixes", []))
    methods["text_standardization"] = f"{noise_count} noise fixes, {shift_count} column-shift corrections applied."

    extracted = cleaning_summary.get("extracted_feature_columns", [])
    methods["numeric_extraction_from_text"] = (
        "Extracted from: " + ", ".join(extracted[:6]) + ("…" if len(extracted) > 6 else "") + "."
        if extracted else "Not applicable."
    )

    if cleaning_summary.get("smartphone_preprocessing_applied"):
        filtering = cleaning_summary.get("smartphone_filtering_summary", {})
        removed = int(filtering.get("total_removed", 0))
        methods["smartphone_only_filtering"] = (
            f"Removed {removed} non-smartphone records (feature phones, iPods, Karbonn, JioPhone, etc.)."
        )
        methods["price_parsing"] = "Numeric price extracted from formatted strings (Rs. prefix, commas, price ranges)."
        methods["ram_storage_extraction"] = "RAM (GB) and internal storage (GB) parsed from combined spec column."
        methods["battery_extraction"] = "Battery capacity (mAh) and fast-charging speed (W) extracted from battery column."
        methods["display_feature_extraction"] = (
            "Screen size (inches), resolution width/height, and refresh rate extracted from display column."
        )
        methods["camera_feature_extraction"] = (
            "Rear main camera (MP), front camera (MP), and rear camera count extracted from camera column."
        )
        methods["os_standardization"] = "OS family (Android / iOS / Other) and version number normalised from raw OS column."
        methods["display_contamination_handling"] = (
            "Rows where the display column contained battery/camera/storage/OS text were detected and flagged."
        )
        methods["price_outlier_flagging"] = (
            "Luxury devices (e.g. Vertu Signature Touch) flagged with is_price_outlier=1 and "
            "is_high_end_luxury_phone=1; original prices preserved."
        )

    encoded = cleaning_summary.get("encoded_columns", [])
    if encoded:
        methods["categorical_encoding"] = "One-hot encoding: " + ", ".join(encoded[:4]) + ("…" if len(encoded) > 4 else "") + "."
    elif cleaning_summary.get("smartphone_preprocessing_applied"):
        methods["categorical_encoding"] = (
            "One-hot encoding applied to brand, processor_brand, os_family, price_segment in ML-ready output."
        )
    else:
        methods["categorical_encoding"] = "Not applied."

    scaler = cleaning_summary.get("scaler_used")
    scaled = cleaning_summary.get("scaled_columns_created") or cleaning_summary.get("scaled_columns", [])
    if scaler and scaled:
        methods["feature_scaling"] = f"{scaler} applied to {len(scaled)} numeric columns."
    elif cleaning_summary.get("smartphone_preprocessing_applied"):
        methods["feature_scaling"] = "MinMaxScaler applied to numeric specification columns in ML-ready output."
    else:
        methods["feature_scaling"] = "Not applied."

    nlp_cols = cleaning_summary.get("cleaned_text_columns", [])
    methods["nlp_text_cleaning"] = (
        "Applied to: " + ", ".join(nlp_cols) + "." if nlp_cols else "Not selected."
    )

    return methods


def build_eda_charts_summary(df: pd.DataFrame) -> list[dict[str, str]]:
    """Return metadata for each EDA chart that will be rendered in the EDA Report section."""
    entries: list[dict[str, str]] = []

    for col, title in [
        ("price", "Price Distribution"),
        ("rating", "Rating Distribution"),
        ("battery_mah", "Battery Capacity Distribution (mAh)"),
        ("ram_gb", "RAM Comparison (GB)"),
        ("storage_gb", "Storage Comparison (GB)"),
        ("screen_size_inches", "Screen Size Distribution (inches)"),
    ]:
        entries.append({
            "chart": title,
            "column": col,
            "type": "histogram",
            "available": str(col in df.columns),
        })

    for col, title in [("brand", "Brand Analysis"), ("os_family", "OS Distribution")]:
        entries.append({
            "chart": title,
            "column": col,
            "type": "bar",
            "available": str(col in df.columns),
        })

    numeric_count = len(df.select_dtypes(include="number").columns)
    entries.append({
        "chart": "Correlation Heatmap",
        "column": "all_numeric",
        "type": "heatmap",
        "available": str(numeric_count >= 2),
    })

    return entries
