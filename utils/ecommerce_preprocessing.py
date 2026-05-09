"""Domain-specific helpers for scraped e-commerce mobile-phone datasets.

These helpers extend the generic cleaning workflow without replacing it.
Callers should always gate usage behind dataset detection so unrelated
datasets continue to use the existing generic logic unchanged.
"""

from __future__ import annotations

import re
from typing import Any

import pandas as pd

MOBILE_ECOMMERCE_COLUMN_HINTS = {
    "product_name",
    "brand",
    "price",
    "rating",
    "review_count",
    "ram",
    "storage",
    "battery",
    "camera",
    "processor",
    "screen_size",
    "availability",
    "product_url",
}

REFERENCE_COLUMN_TOKENS = {
    "url",
    "link",
    "image",
    "tracking",
}

E_COMMERCE_NUMERIC_COLUMN_HINTS = {
    "price",
    "rating",
    "review_count",
    "ram",
    "storage",
    "battery",
    "screen_size",
}

REFERENCE_TEXT_COLUMN_HINTS = {
    "product_name",
    "processor",
    "camera",
}

CATEGORICAL_ECOMMERCE_COLUMN_HINTS = {
    "brand",
    "availability",
}

AVAILABILITY_NORMALIZATION_MAP = {
    "in stock": "In Stock",
    "instock": "In Stock",
    "in-stock": "In Stock",
    "available": "In Stock",
    "out of stock": "Out of Stock",
    "outofstock": "Out of Stock",
    "out-of-stock": "Out of Stock",
    "unavailable": "Out of Stock",
    "not available": "Out of Stock",
    "pre order": "Pre Order",
    "pre-order": "Pre Order",
    "preorder": "Pre Order",
    "low stock": "Low Stock",
    "limited stock": "Low Stock",
}


def normalize_column_name(name: Any) -> str:
    """Normalize a column name for resilient domain matching."""
    return str(name).strip().lower().replace(" ", "_").replace("-", "_")


def detect_mobile_ecommerce_dataset(columns: list[str] | pd.Index) -> bool:
    """Detect whether the dataset looks like a scraped mobile-phone catalog."""
    normalized_columns = {normalize_column_name(column) for column in columns}
    matched_columns = normalized_columns & MOBILE_ECOMMERCE_COLUMN_HINTS

    if len(matched_columns) >= 5:
        return True

    core_columns = {"product_name", "brand", "price"}
    return len(matched_columns & core_columns) >= 2 and len(matched_columns) >= 4


def is_reference_column(column_name: str) -> bool:
    """Return whether a column appears to be a URL or other reference-only field."""
    normalized_name = normalize_column_name(column_name)
    return any(token in normalized_name for token in REFERENCE_COLUMN_TOKENS)


def get_reference_columns(columns: list[str] | pd.Index) -> list[str]:
    """Return reference-only columns that should not become ML features."""
    return [str(column) for column in columns if is_reference_column(str(column))]


def _extract_first_number(value: Any) -> float | None:
    """Extract the first numeric token from a dirty scraped value."""
    if value is None or pd.isna(value):
        return None

    text = str(value).strip()
    if not text:
        return None

    match = re.search(r"(\d+(?:\.\d+)?)", text.replace(",", ""))
    if not match:
        return None

    try:
        return float(match.group(1))
    except ValueError:
        return None


def clean_price_column(series: pd.Series) -> pd.Series:
    """Convert scraped price strings into numeric values.

    Non-price values such as stock-status text are converted to ``NaN`` so the
    generic missing-value logic can decide how to handle them safely.
    """
    cleaned_values: list[float | None] = []
    for value in series:
        if value is None or pd.isna(value):
            cleaned_values.append(None)
            continue

        text = str(value).strip()
        if not text:
            cleaned_values.append(None)
            continue

        normalized_text = text.lower()
        if any(token in normalized_text for token in ["out of stock", "unavailable", "coming soon"]):
            cleaned_values.append(None)
            continue

        compact_text = text.replace(",", "")
        matches = re.findall(r"\d+(?:\.\d+)?", compact_text)
        if not matches:
            cleaned_values.append(None)
            continue

        try:
            numeric_text = max(matches, key=len)
            cleaned_values.append(float(numeric_text))
        except ValueError:
            cleaned_values.append(None)

    return pd.to_numeric(pd.Series(cleaned_values, index=series.index), errors="coerce")


def clean_rating_column(series: pd.Series) -> pd.Series:
    """Convert scraped rating strings into floats."""
    cleaned_values = [_extract_first_number(value) for value in series]
    return pd.to_numeric(pd.Series(cleaned_values, index=series.index), errors="coerce")


def clean_review_count_column(series: pd.Series) -> pd.Series:
    """Convert scraped review-count strings into integers."""
    cleaned_values: list[float | None] = []
    for value in series:
        if value is None or pd.isna(value):
            cleaned_values.append(None)
            continue

        text = str(value).strip().lower()
        if not text or "no review" in text:
            cleaned_values.append(None)
            continue

        cleaned_values.append(_extract_first_number(text))

    return pd.to_numeric(pd.Series(cleaned_values, index=series.index), errors="coerce")


def _extract_numeric_gb(value: Any) -> list[int]:
    """Extract all GB-sized numeric tokens from a RAM/storage value."""
    if value is None or pd.isna(value):
        return []

    text = str(value).lower().replace(" ", "")
    return [int(number) for number in re.findall(r"(\d+)(?:gb)", text)]


def extract_ram_storage_features(dataframe: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Extract ``ram_gb`` and ``storage_gb`` numeric features when possible."""
    ram_values: list[float | None] = []
    storage_values: list[float | None] = []

    ram_series = dataframe["ram"] if "ram" in dataframe.columns else pd.Series(index=dataframe.index, dtype="object")
    storage_series = dataframe["storage"] if "storage" in dataframe.columns else pd.Series(index=dataframe.index, dtype="object")

    for index in dataframe.index:
        ram_text = ram_series.get(index) if not ram_series.empty else None
        storage_text = storage_series.get(index) if not storage_series.empty else None

        extracted_from_ram = _extract_numeric_gb(ram_text)
        extracted_from_storage = _extract_numeric_gb(storage_text)

        if len(extracted_from_ram) >= 2:
            ram_values.append(float(extracted_from_ram[0]))
            storage_values.append(float(extracted_from_ram[1]))
            continue

        ram_values.append(float(extracted_from_ram[0]) if extracted_from_ram else None)

        if extracted_from_storage:
            storage_values.append(float(extracted_from_storage[0]))
        elif len(extracted_from_ram) == 1 and "storage" not in dataframe.columns:
            storage_values.append(None)
        else:
            storage_values.append(None)

    return (
        pd.to_numeric(pd.Series(ram_values, index=dataframe.index), errors="coerce"),
        pd.to_numeric(pd.Series(storage_values, index=dataframe.index), errors="coerce"),
    )


def extract_battery_capacity(series: pd.Series) -> pd.Series:
    """Extract numeric battery capacity in mAh."""
    cleaned_values = [_extract_first_number(value) for value in series]
    return pd.to_numeric(pd.Series(cleaned_values, index=series.index), errors="coerce")


def extract_screen_size(series: pd.Series) -> pd.Series:
    """Extract numeric screen size in inches."""
    cleaned_values = [_extract_first_number(value) for value in series]
    return pd.to_numeric(pd.Series(cleaned_values, index=series.index), errors="coerce")


def normalize_brand_column(series: pd.Series) -> pd.Series:
    """Normalize brand values for readability without lowercasing all categories."""
    normalized_values: list[str | None] = []
    for value in series:
        if value is None or pd.isna(value):
            normalized_values.append(None)
            continue

        cleaned_text = re.sub(r"\s+", " ", str(value).strip())
        normalized_values.append(cleaned_text.title() if cleaned_text else None)

    return pd.Series(normalized_values, index=series.index, dtype="object")


def normalize_product_name_column(series: pd.Series) -> pd.Series:
    """Preserve product names as readable reference text."""
    normalized_values: list[str | None] = []
    for value in series:
        if value is None or pd.isna(value):
            normalized_values.append(None)
            continue

        cleaned_text = re.sub(r"\s+", " ", str(value).strip())
        normalized_values.append(cleaned_text or None)

    return pd.Series(normalized_values, index=series.index, dtype="object")


def normalize_availability_column(series: pd.Series) -> pd.Series:
    """Normalize stock-status values into a small set of readable labels."""
    normalized_values: list[str | None] = []
    for value in series:
        if value is None or pd.isna(value):
            normalized_values.append(None)
            continue

        cleaned_text = re.sub(r"[\-_]+", " ", str(value).strip().lower())
        cleaned_text = re.sub(r"\s+", " ", cleaned_text).strip()
        normalized_values.append(
            AVAILABILITY_NORMALIZATION_MAP.get(cleaned_text, cleaned_text.title() if cleaned_text else None)
        )

    return pd.Series(normalized_values, index=series.index, dtype="object")


def _normalize_product_name_for_matching(value: Any) -> str:
    """Normalize product names for near-duplicate comparison only."""
    if value is None or pd.isna(value):
        return ""

    text = normalize_product_name_column(pd.Series([value])).iloc[0] or ""
    text = text.lower()
    text = text.replace("/", " ").replace("+", " ")
    text = re.sub(r"(\d+)\s*gb", r"\1gb", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def create_product_dedup_key(dataframe: pd.DataFrame) -> pd.Series:
    """Create a normalized key for product-level near-duplicate detection."""
    normalized_name = (
        dataframe["product_name"].apply(_normalize_product_name_for_matching)
        if "product_name" in dataframe.columns
        else pd.Series("", index=dataframe.index)
    )
    normalized_brand = (
        normalize_brand_column(dataframe["brand"]).fillna("").astype(str).str.lower().str.strip()
        if "brand" in dataframe.columns
        else pd.Series("", index=dataframe.index)
    )
    normalized_price = (
        clean_price_column(dataframe["price"]).fillna(-1).astype(float).round(0).astype(int).astype(str)
        if "price" in dataframe.columns
        else pd.Series("", index=dataframe.index)
    )
    return normalized_name + "|" + normalized_brand + "|" + normalized_price


def apply_mobile_domain_outlier_rules(
    dataframe: pd.DataFrame,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Apply domain-safe bounds for common mobile-phone numeric fields."""
    adjusted_df = dataframe.copy()
    adjustments: list[dict[str, Any]] = []

    def _record_adjustment(column: str, action: str, count: int, rule: str) -> None:
        if count > 0:
            adjustments.append(
                {
                    "column_name": column,
                    "action_taken": action,
                    "affected_rows": int(count),
                    "rule": rule,
                }
            )

    if "rating" in adjusted_df.columns and pd.api.types.is_numeric_dtype(adjusted_df["rating"]):
        below_zero = adjusted_df["rating"] < 0
        above_five = adjusted_df["rating"] > 5
        adjusted_df.loc[below_zero, "rating"] = 0
        adjusted_df.loc[above_five, "rating"] = 5
        _record_adjustment("rating", "capped_to_valid_range", int(below_zero.sum() + above_five.sum()), "0 to 5")

    if "battery_mah" in adjusted_df.columns and pd.api.types.is_numeric_dtype(adjusted_df["battery_mah"]):
        invalid_battery = (adjusted_df["battery_mah"] < 1000) | (adjusted_df["battery_mah"] > 10000)
        adjusted_df.loc[invalid_battery, "battery_mah"] = pd.NA
        _record_adjustment("battery_mah", "flagged_as_invalid", int(invalid_battery.sum()), "1000 to 10000 mAh")

    if "screen_size_inches" in adjusted_df.columns and pd.api.types.is_numeric_dtype(adjusted_df["screen_size_inches"]):
        invalid_screen = (adjusted_df["screen_size_inches"] < 3) | (adjusted_df["screen_size_inches"] > 10)
        adjusted_df.loc[invalid_screen, "screen_size_inches"] = pd.NA
        _record_adjustment("screen_size_inches", "flagged_as_invalid", int(invalid_screen.sum()), "3 to 10 inches")

    return adjusted_df, adjustments


def build_ecommerce_output_datasets(
    dataframe: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create human-readable and ML-ready e-commerce output datasets."""
    readable_df = dataframe.copy(deep=True)

    readable_priority_columns = [
        "product_name",
        "brand",
        "availability",
        "price",
        "rating",
        "review_count",
        "ram_gb",
        "storage_gb",
        "battery_mah",
        "screen_size_inches",
        "price_scaled",
        "rating_scaled",
        "review_count_scaled",
        "ram_gb_scaled",
        "storage_gb_scaled",
        "battery_mah_scaled",
        "screen_size_inches_scaled",
        "camera",
        "processor",
    ]
    readable_columns = [column for column in readable_priority_columns if column in readable_df.columns]
    readable_df = readable_df.loc[:, readable_columns]

    ml_ready_base = readable_df.copy(deep=True)
    categorical_columns = [
        column for column in ["brand", "availability"] if column in ml_ready_base.columns
    ]
    if categorical_columns:
        ml_ready_base = pd.get_dummies(
            ml_ready_base,
            columns=categorical_columns,
            drop_first=False,
        )

    ml_ready_columns = [
        column
        for column in ml_ready_base.columns
        if column.endswith("_scaled") or column.startswith("brand_") or column.startswith("availability_")
    ]
    ml_ready_df = ml_ready_base.loc[:, ml_ready_columns]

    return readable_df, ml_ready_df


def build_ecommerce_preprocessed_view(
    dataframe: pd.DataFrame,
    *,
    drop_reference_columns: bool = False,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Return a cleaned preview copy for profiling, visualization, or ML inference.

    The returned frame is a non-destructive view of the original data. This is
    useful when the app needs better numeric and semantic detection before the
    main cleaning pipeline is executed.
    """
    preprocessed_df = dataframe.copy(deep=True)
    metadata: dict[str, Any] = {
        "ecommerce_preprocessing_applied": False,
        "cleaned_numeric_columns": [],
        "extracted_feature_columns": [],
        "normalized_categorical_columns": [],
        "dropped_reference_columns": [],
        "reference_columns": get_reference_columns(preprocessed_df.columns),
        "raw_source_columns_excluded_from_ml": [],
        "recommendation_ready": False,
    }

    if not detect_mobile_ecommerce_dataset(preprocessed_df.columns):
        return preprocessed_df, metadata

    metadata["ecommerce_preprocessing_applied"] = True
    metadata["recommendation_ready"] = True

    if "product_name" in preprocessed_df.columns:
        preprocessed_df["product_name"] = normalize_product_name_column(preprocessed_df["product_name"])

    if "brand" in preprocessed_df.columns:
        preprocessed_df["brand"] = normalize_brand_column(preprocessed_df["brand"])
        metadata["normalized_categorical_columns"].append("brand")

    if "availability" in preprocessed_df.columns:
        preprocessed_df["availability"] = normalize_availability_column(preprocessed_df["availability"])
        metadata["normalized_categorical_columns"].append("availability")

    if "price" in preprocessed_df.columns:
        preprocessed_df["price"] = clean_price_column(preprocessed_df["price"])
        metadata["cleaned_numeric_columns"].append("price")

    if "rating" in preprocessed_df.columns:
        preprocessed_df["rating"] = clean_rating_column(preprocessed_df["rating"])
        metadata["cleaned_numeric_columns"].append("rating")

    if "review_count" in preprocessed_df.columns:
        preprocessed_df["review_count"] = clean_review_count_column(preprocessed_df["review_count"])
        metadata["cleaned_numeric_columns"].append("review_count")

    if "ram" in preprocessed_df.columns or "storage" in preprocessed_df.columns:
        ram_gb, storage_gb = extract_ram_storage_features(preprocessed_df)
        preprocessed_df["ram_gb"] = ram_gb
        preprocessed_df["storage_gb"] = storage_gb
        metadata["extracted_feature_columns"].extend(["ram_gb", "storage_gb"])
        metadata["raw_source_columns_excluded_from_ml"].extend(
            [column for column in ["ram", "storage"] if column in preprocessed_df.columns]
        )

    if "battery" in preprocessed_df.columns:
        preprocessed_df["battery_mah"] = extract_battery_capacity(preprocessed_df["battery"])
        metadata["extracted_feature_columns"].append("battery_mah")
        metadata["raw_source_columns_excluded_from_ml"].append("battery")

    if "screen_size" in preprocessed_df.columns:
        preprocessed_df["screen_size_inches"] = extract_screen_size(preprocessed_df["screen_size"])
        metadata["extracted_feature_columns"].append("screen_size_inches")
        metadata["raw_source_columns_excluded_from_ml"].append("screen_size")

    if drop_reference_columns and metadata["reference_columns"]:
        preprocessed_df = preprocessed_df.drop(columns=metadata["reference_columns"], errors="ignore")
        metadata["dropped_reference_columns"] = list(metadata["reference_columns"])

    # Keep metadata stable and deduplicated for downstream reports.
    for key in [
        "cleaned_numeric_columns",
        "extracted_feature_columns",
        "normalized_categorical_columns",
        "dropped_reference_columns",
        "reference_columns",
        "raw_source_columns_excluded_from_ml",
    ]:
        metadata[key] = sorted(set(metadata[key]))

    return preprocessed_df, metadata


def build_semantic_product_key(dataframe: pd.DataFrame) -> pd.Series:
    """Create a normalized deduplication key for product-style rows."""
    return create_product_dedup_key(dataframe)
