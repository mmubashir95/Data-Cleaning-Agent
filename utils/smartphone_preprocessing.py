"""Dataset-specific preprocessing for the tricky smartphone recommendation CSV.

This module is intentionally separate from the generic cleaning pipeline.
Callers should detect the smartphone dataset first, then use these helpers to
extract structured recommendation features while preserving the existing app
flow for all unrelated datasets.
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

SMARTPHONE_COLUMN_HINTS = {
    "model",
    "price",
    "rating",
    "sim",
    "processor",
    "ram",
    "battery",
    "display",
    "camera",
    "card",
    "os",
    "segment",
}

SMARTPHONE_BRANDS = [
    "blackzone",
    "apple",
    "asus",
    "blackview",
    "dizo",
    "eunity",
    "google",
    "honor",
    "huawei",
    "infinix",
    "iqoo",
    "itel",
    "jio",
    "lava",
    "lenovo",
    "meizu",
    "motorola",
    "nothing",
    "nokia",
    "oneplus",
    "oppo",
    "poco",
    "realme",
    "redmi",
    "samsung",
    "sony",
    "tecno",
    "vertu",
    "vivo",
    "xiaomi",
]

PROCESSOR_BRAND_PATTERNS = {
    "snapdragon": "snapdragon",
    "dimensity": "dimensity",
    "exynos": "exynos",
    "bionic": "bionic",
    "tensor": "tensor",
    "helio": "helio",
    "unisoc": "unisoc",
    "kirin": "kirin",
}

OS_FAMILY_PATTERNS = {
    "android": "Android",
    "ios": "iOS",
    "harmony": "HarmonyOS",
    "kaios": "KaiOS",
    "rtos": "RTOS",
}

OS_POLLUTION_TOKENS = {
    "no fm radio",
    "bluetooth",
    "browser",
    "memory card",
    "supported",
    "hybrid",
    "upto",
    "up to",
}

BOOLEAN_FEATURE_COLUMNS = [
    "is_5g_model",
    "has_5g",
    "has_4g",
    "has_volte",
    "has_wifi",
    "has_nfc",
    "has_ir_blaster",
    "is_octa_core",
    "has_fast_charging",
    "is_foldable_display",
    "has_punch_hole",
    "has_waterdrop_notch",
    "memory_card_supported",
    "memory_card_hybrid",
    "has_front_camera",
]

SCALABLE_NUMERIC_COLUMNS = [
    "price",
    "rating",
    "ram_gb",
    "storage_gb",
    "battery_mah",
    "charging_watt",
    "screen_size_inches",
    "resolution_width",
    "resolution_height",
    "refresh_rate_hz",
    "rear_main_camera_mp",
    "front_camera_mp",
    "rear_camera_count",
    "processor_speed_ghz",
    "memory_card_max_gb",
]


def normalize_smartphone_column_name(name: Any) -> str:
    """Normalize column names while preserving semantic intent."""
    text = str(name).strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def detect_smartphone_dataset(columns: list[str] | pd.Index) -> bool:
    """Return True when the uploaded dataset matches the smartphone schema."""
    return count_smartphone_dataset_column_matches(columns) >= 8


def count_smartphone_dataset_column_matches(columns: list[str] | pd.Index) -> int:
    """Count smartphone-schema columns present in the uploaded dataset."""
    normalized_columns = {normalize_smartphone_column_name(column) for column in columns}
    matched_columns = normalized_columns & SMARTPHONE_COLUMN_HINTS
    return len(matched_columns)


def get_smartphone_dataset_matched_columns(columns: list[str] | pd.Index) -> list[str]:
    """Return the sorted smartphone-schema columns present in the dataset."""
    normalized_columns = {normalize_smartphone_column_name(column) for column in columns}
    return sorted(normalized_columns & SMARTPHONE_COLUMN_HINTS)


def _text_or_none(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    return text or None


def _normalize_noise_text(value: Any) -> str | None:
    """Repair noisy scraped separators without removing useful numeric tokens."""
    text = _text_or_none(value)
    if text is None:
        return None

    text = text.replace("\u00a0", " ")
    text = re.sub(r"(?<=\d)\?(?=\d)", " ", text)
    text = re.sub(r"(?<=\d)\?(?=[a-zA-Z])", " ", text)
    text = re.sub(r"(?<=[a-zA-Z])\?(?=\d)", " ", text)
    text = re.sub(r"(?<=\d)\s*\?\s*x\s*\?\s*(?=\d)", " x ", text, flags=re.IGNORECASE)
    text = re.sub(r"(?<=\d)\s*x\s*(?=\d)", " x ", text, flags=re.IGNORECASE)
    text = re.sub(r"\?", " ", text)
    text = re.sub(r"(?i)\bup\s*to\b", "upto", text)
    text = re.sub(r"\s+", " ", text).strip(" ,")
    return text or None


def _clean_all_known_text_columns(dataframe: pd.DataFrame) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Apply question-mark/noise cleanup to the core smartphone text columns."""
    cleaned_df = dataframe.copy(deep=True)
    noise_fixes: list[dict[str, Any]] = []
    text_columns = ["model", "sim", "processor", "ram", "battery", "display", "camera", "card", "os"]

    for column in text_columns:
        if column not in cleaned_df.columns:
            continue

        before_series = cleaned_df[column]
        after_series = before_series.map(_normalize_noise_text)
        changed_rows = int((before_series.fillna("") != after_series.fillna("")).sum())
        cleaned_df[column] = after_series
        if changed_rows > 0:
            noise_fixes.append(
                {
                    "column": column,
                    "rows_changed": changed_rows,
                    "fix": "replaced noisy question-mark separators while preserving smartphone specification tokens",
                }
            )

    return cleaned_df, noise_fixes


def _looks_like_os(text: str | None) -> bool:
    lowered = (text or "").strip().lower()
    if not lowered:
        return False
    return any(token in lowered for token in OS_FAMILY_PATTERNS)


def _looks_like_os_pollution(text: str | None) -> bool:
    lowered = (text or "").strip().lower()
    if not lowered:
        return False
    return any(token in lowered for token in OS_POLLUTION_TOKENS)


def _looks_like_card_text(text: str | None) -> bool:
    lowered = (text or "").strip().lower()
    if not lowered:
        return False
    return "memory card" in lowered or "not supported" in lowered or "upto" in lowered or "up to" in lowered


def _looks_like_camera_text(text: str | None) -> bool:
    lowered = (text or "").strip().lower()
    return "camera" in lowered or "mp" in lowered


def _looks_like_display_text(text: str | None) -> bool:
    lowered = (text or "").strip().lower()
    return (
        "display" in lowered
        or "inches" in lowered
        or " px" in lowered
        or "punch hole" in lowered
        or "notch" in lowered
        or "foldable" in lowered
    )


def _append_extra_feature(existing: str | None, new_value: str | None) -> str | None:
    if not new_value:
        return existing
    if not existing:
        return new_value
    return f"{existing} | {new_value}"


def _fix_shifted_columns(dataframe: pd.DataFrame) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Repair common scraped-column shifts between card, os, camera, and display."""
    fixed_df = dataframe.copy(deep=True)
    fixes: list[dict[str, Any]] = []
    fixed_df["extra_features"] = None

    for index, row in fixed_df.iterrows():
        card = _text_or_none(row.get("card"))
        os_value = _text_or_none(row.get("os"))
        camera = _text_or_none(row.get("camera"))
        display = _text_or_none(row.get("display"))

        if _looks_like_os(card) and (os_value is None or _looks_like_os_pollution(os_value)):
            if os_value and _looks_like_os_pollution(os_value):
                fixed_df.at[index, "extra_features"] = _append_extra_feature(
                    fixed_df.at[index, "extra_features"],
                    os_value,
                )
            fixed_df.at[index, "os"] = card
            fixed_df.at[index, "card"] = None
            fixes.append(
                {
                    "row_index": int(index),
                    "from_column": "card",
                    "to_column": "os",
                    "value": card,
                    "reason": "card contained an OS value while os was missing or polluted",
                }
            )
            card = None
            os_value = _text_or_none(fixed_df.at[index, "os"])

        if _looks_like_card_text(os_value) and (card is None or not _looks_like_card_text(card)):
            fixed_df.at[index, "card"] = os_value
            fixed_df.at[index, "os"] = None
            fixes.append(
                {
                    "row_index": int(index),
                    "from_column": "os",
                    "to_column": "card",
                    "value": os_value,
                    "reason": "os contained memory-card text instead of an operating system",
                }
            )
            card = _text_or_none(fixed_df.at[index, "card"])
            os_value = None

        if _looks_like_os_pollution(os_value):
            fixed_df.at[index, "extra_features"] = _append_extra_feature(
                fixed_df.at[index, "extra_features"],
                os_value,
            )
            fixed_df.at[index, "os"] = None
            fixes.append(
                {
                    "row_index": int(index),
                    "from_column": "os",
                    "to_column": "extra_features",
                    "value": os_value,
                    "reason": "os contained polluted non-OS metadata",
                }
            )

        if _text_or_none(camera):
            lowered_camera = camera.lower()
            if "memory card" in lowered_camera and (card is None or not _looks_like_card_text(card)):
                fixed_df.at[index, "card"] = camera
                fixed_df.at[index, "camera"] = None
                fixes.append(
                    {
                        "row_index": int(index),
                        "from_column": "camera",
                        "to_column": "card",
                        "value": camera,
                        "reason": "camera contained memory-card text instead of camera specifications",
                    }
                )
            elif "foldable display" in lowered_camera and not _looks_like_camera_text(camera):
                updated_display = _append_extra_feature(display, camera)
                fixed_df.at[index, "display"] = updated_display
                fixed_df.at[index, "camera"] = None
                fixes.append(
                    {
                        "row_index": int(index),
                        "from_column": "camera",
                        "to_column": "display",
                        "value": camera,
                        "reason": "camera contained display text instead of camera specifications",
                    }
                )

    return fixed_df, fixes


def _extract_numeric_token(text: str | None, pattern: str, *, flags: int = re.IGNORECASE) -> float | None:
    if not text:
        return None
    match = re.search(pattern, text, flags)
    if not match:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


def _parse_storage_token_to_gb(text: str | None) -> float | None:
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return None
    text = str(text)
    lowered = text.lower()
    if "not supported" in lowered:
        return 0.0
    match = re.search(r"(\d+(?:\.\d+)?)\s*(tb|gb)", text, re.IGNORECASE)
    if not match:
        return None
    value = float(match.group(1))
    unit = match.group(2).lower()
    return value * 1024 if unit == "tb" else value


def _extract_brand(model: str | None) -> str:
    text = _text_or_none(model) or ""
    lowered = text.lower()
    for brand in SMARTPHONE_BRANDS:
        if lowered.startswith(brand):
            return brand.title() if brand != "iqoo" else "iQOO"
    first_token = (text or "Unknown").split()
    if not first_token:
        return "Unknown"
    return first_token[0].title()


def _is_known_brand_label(brand: str | None) -> bool:
    normalized = (_text_or_none(brand) or "").lower()
    return normalized in set(SMARTPHONE_BRANDS)


def _extract_processor_brand(processor: str | None) -> str:
    lowered = (_text_or_none(processor) or "").lower()
    for token, label in PROCESSOR_BRAND_PATTERNS.items():
        if token in lowered:
            return label
    return "unknown"


def _extract_processor_speed(processor: str | None) -> float | None:
    return _extract_numeric_token(processor, r"(\d+(?:\.\d+)?)\s*ghz")


def _extract_ram_storage(ram_text: str | None) -> tuple[float | None, float | None]:
    ram_text = _text_or_none(ram_text)
    if not ram_text:
        return None, None
    physical_ram_match = re.search(r"\b(\d+(?:\.\d+)?)\s*gb\s*ram\b", ram_text, re.IGNORECASE)
    storage_match = re.search(r"\b(\d+(?:\.\d+)?)\s*(tb|gb)\s*inbuilt\b", ram_text, re.IGNORECASE)

    ram_gb = float(physical_ram_match.group(1)) if physical_ram_match else None
    storage_gb = None
    if storage_match:
        storage_value = float(storage_match.group(1))
        storage_gb = storage_value * 1024 if storage_match.group(2).lower() == "tb" else storage_value
    return ram_gb, storage_gb


def _extract_price(series: pd.Series) -> pd.Series:
    cleaned_values: list[float | None] = []
    for value in series:
        text = _text_or_none(value)
        if not text:
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


def _extract_rating(series: pd.Series) -> pd.Series:
    cleaned_values: list[float | None] = []
    for value in series:
        text = _text_or_none(value)
        if not text:
            cleaned_values.append(None)
            continue
        match = re.search(r"(\d+(?:\.\d+)?)", text)
        cleaned_values.append(float(match.group(1)) if match else None)
    return pd.to_numeric(pd.Series(cleaned_values, index=series.index), errors="coerce")


def _price_segment(price: float | None) -> str:
    if price is None or pd.isna(price):
        return "Unknown"
    if price < 20000:
        return "Budget"
    if price <= 49999:
        return "Mid Range"
    if price <= 99999:
        return "Premium"
    return "Flagship"


def _extract_camera_counts(camera: str | None) -> tuple[float | None, float | None, float | None, int]:
    text = _text_or_none(camera) or ""
    lowered = text.lower()
    mp_values = [float(value) for value in re.findall(r"(\d+(?:\.\d+)?)\s*mp", text, re.IGNORECASE)]
    rear_main = mp_values[0] if mp_values else None

    front_camera = None
    front_match = re.search(r"(\d+(?:\.\d+)?)\s*mp\s*front", lowered, re.IGNORECASE)
    if front_match:
        front_camera = float(front_match.group(1))

    rear_count = 0
    if "triple rear" in lowered:
        rear_count = 3
    elif "quad rear" in lowered:
        rear_count = 4
    elif "dual rear" in lowered:
        rear_count = 2
    elif "rear" in lowered:
        rear_prefix = re.split(r"rear", lowered, maxsplit=1)[0]
        rear_count = len(re.findall(r"\d+(?:\.\d+)?\s*mp", rear_prefix, re.IGNORECASE))
    if rear_count == 0 and rear_main is not None:
        rear_count = 1

    return rear_main, front_camera, float(rear_count) if rear_count else None, int(front_camera is not None or "front camera" in lowered)


def _normalize_os_fields(os_text: str | None) -> tuple[str, str | None]:
    lowered = (_text_or_none(os_text) or "").lower().strip()
    if not lowered:
        return "Other", None

    os_family = "Other"
    for token, family in OS_FAMILY_PATTERNS.items():
        if token in lowered:
            os_family = family
            break

    version_match = re.search(r"v?\s*(\d+(?:\.\d+)?)", lowered, re.IGNORECASE)
    os_version = version_match.group(1) if version_match else None
    return os_family, os_version


def _normalize_combined_text(text: str) -> str:
    cleaned = text.lower()
    cleaned = re.sub(r"[^a-z0-9.+\s]", " ", cleaned)
    cleaned = re.sub(r"(\d+(?:\.\d+)?)\s+(ghz|mp|hz|mah|gb|tb)", r"\1\2", cleaned)
    cleaned = re.sub(r"(\d)\s+g\b", r"\1g", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def build_smartphone_preprocessed_view(
    dataframe: pd.DataFrame,
    *,
    drop_reference_columns: bool = False,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Return a smartphone-specific cleaned view plus rich preprocessing metadata."""
    smartphone_df = dataframe.copy(deep=True)
    metadata: dict[str, Any] = {
        "smartphone_preprocessing_applied": False,
        "ecommerce_preprocessing_applied": False,
        "cleaned_numeric_columns": [],
        "extracted_feature_columns": [],
        "normalized_categorical_columns": [],
        "dropped_reference_columns": [],
        "raw_source_columns_excluded_from_ml": [],
        "recommendation_ready": False,
        "shifted_column_fixes": [],
        "noise_fixes": [],
        "columns_dropped": [],
        "validation_checks": [],
        "scalable_numeric_feature_columns": list(SCALABLE_NUMERIC_COLUMNS),
        "smartphone_feature_columns": [],
    }

    if not detect_smartphone_dataset(smartphone_df.columns):
        return smartphone_df, metadata

    rename_map = {column: normalize_smartphone_column_name(column) for column in smartphone_df.columns}
    smartphone_df = smartphone_df.rename(columns=rename_map)
    for required_column in ["model", "price", "rating", "sim", "processor", "ram", "battery", "display", "camera", "card", "os"]:
        if required_column not in smartphone_df.columns:
            smartphone_df[required_column] = pd.NA
    smartphone_df, noise_fixes = _clean_all_known_text_columns(smartphone_df)
    smartphone_df, shifted_fixes = _fix_shifted_columns(smartphone_df)

    if "segment" in smartphone_df.columns:
        if smartphone_df["segment"].isna().all() or smartphone_df["segment"].astype(str).str.strip().eq("").all():
            smartphone_df = smartphone_df.drop(columns=["segment"])
            metadata["columns_dropped"].append("segment")

    smartphone_df["price"] = _extract_price(smartphone_df["price"])
    smartphone_df["rating"] = _extract_rating(smartphone_df["rating"])
    smartphone_df["brand"] = smartphone_df["model"].map(_extract_brand)
    smartphone_df["is_5g_model"] = smartphone_df["model"].fillna("").str.contains(r"\b5g\b", case=False, regex=True).astype(int)

    sim_lower = smartphone_df["sim"].fillna("").str.lower()
    smartphone_df["has_5g"] = sim_lower.str.contains(r"\b5g\b", regex=True).astype(int)
    smartphone_df["has_4g"] = sim_lower.str.contains(r"\b4g\b", regex=True).astype(int)
    smartphone_df["has_volte"] = sim_lower.str.contains("volte|vo5g", regex=True).astype(int)
    smartphone_df["has_wifi"] = sim_lower.str.contains("wi-fi|wifi", regex=True).astype(int)
    smartphone_df["has_nfc"] = sim_lower.str.contains(r"\bnfc\b", regex=True).astype(int)
    smartphone_df["has_ir_blaster"] = sim_lower.str.contains("ir blaster", regex=True).astype(int)

    smartphone_df["processor_brand"] = smartphone_df["processor"].map(_extract_processor_brand)
    smartphone_df["processor_speed_ghz"] = smartphone_df["processor"].map(_extract_processor_speed)
    smartphone_df["is_octa_core"] = smartphone_df["processor"].fillna("").str.contains("octa core", case=False, regex=False).astype(int)

    ram_storage = smartphone_df["ram"].map(_extract_ram_storage)
    smartphone_df["ram_gb"] = ram_storage.map(lambda item: item[0])
    smartphone_df["storage_gb"] = ram_storage.map(lambda item: item[1])

    smartphone_df["battery_mah"] = smartphone_df["battery"].map(lambda value: _extract_numeric_token(value, r"(\d+(?:\.\d+)?)\s*mah"))
    smartphone_df["charging_watt"] = smartphone_df["battery"].map(lambda value: _extract_numeric_token(value, r"(\d+(?:\.\d+)?)\s*w"))
    smartphone_df["has_fast_charging"] = smartphone_df["battery"].fillna("").str.contains("fast charging|charging|turbo charge", case=False, regex=True).astype(int)

    smartphone_df["screen_size_inches"] = smartphone_df["display"].map(lambda value: _extract_numeric_token(value, r"(\d+(?:\.\d+)?)\s*inches"))
    smartphone_df["resolution_width"] = smartphone_df["display"].map(lambda value: _extract_numeric_token(value, r"(\d+)\s*x\s*\d+\s*px"))
    smartphone_df["resolution_height"] = smartphone_df["display"].map(lambda value: _extract_numeric_token(value, r"\d+\s*x\s*(\d+)\s*px"))
    smartphone_df["refresh_rate_hz"] = smartphone_df["display"].map(lambda value: _extract_numeric_token(value, r"(\d+(?:\.\d+)?)\s*hz"))
    display_lower = smartphone_df["display"].fillna("").str.lower()
    smartphone_df["is_foldable_display"] = display_lower.str.contains("foldable", regex=False).astype(int)
    smartphone_df["has_punch_hole"] = display_lower.str.contains("punch hole", regex=False).astype(int)
    smartphone_df["has_waterdrop_notch"] = display_lower.str.contains("water drop notch", regex=False).astype(int)

    camera_features = smartphone_df["camera"].map(_extract_camera_counts)
    smartphone_df["rear_main_camera_mp"] = camera_features.map(lambda item: item[0])
    smartphone_df["front_camera_mp"] = camera_features.map(lambda item: item[1])
    smartphone_df["rear_camera_count"] = camera_features.map(lambda item: item[2])
    smartphone_df["has_front_camera"] = camera_features.map(lambda item: item[3]).fillna(0).astype(int)

    card_lower = smartphone_df["card"].fillna("").str.lower()
    smartphone_df["memory_card_supported"] = (
        card_lower.str.contains("memory card", regex=False) & ~card_lower.str.contains("not supported", regex=False)
    ).astype(int)
    smartphone_df["memory_card_hybrid"] = card_lower.str.contains("hybrid", regex=False).astype(int)
    smartphone_df["memory_card_max_gb"] = smartphone_df["card"].map(_parse_storage_token_to_gb)

    os_fields = smartphone_df["os"].map(_normalize_os_fields)
    smartphone_df["os_family"] = os_fields.map(lambda item: item[0])
    smartphone_df["os_version"] = os_fields.map(lambda item: item[1])
    smartphone_df["price_segment"] = smartphone_df["price"].map(_price_segment)
    smartphone_df["phone_id"] = np.arange(1, len(smartphone_df) + 1)

    combined_parts = []
    for column in ["model", "brand", "processor", "display", "camera", "card", "os", "price_segment"]:
        if column in smartphone_df.columns:
            combined_parts.append(smartphone_df[column].fillna(""))
    smartphone_df["combined_text_features"] = (
        pd.Series([" ".join(values) for values in zip(*combined_parts)], index=smartphone_df.index)
        .map(_normalize_combined_text)
    )

    # Numeric feature columns are left as numeric so the generic median-filling
    # step can impute them safely after this dataset-specific parsing stage.
    extracted_features = [
        "ram_gb",
        "storage_gb",
        "battery_mah",
        "charging_watt",
        "screen_size_inches",
        "resolution_width",
        "resolution_height",
        "refresh_rate_hz",
        "rear_main_camera_mp",
        "front_camera_mp",
        "rear_camera_count",
        "processor_speed_ghz",
        "memory_card_max_gb",
    ]
    boolean_columns = list(BOOLEAN_FEATURE_COLUMNS)
    categorical_columns = ["brand", "processor_brand", "os_family", "price_segment"]

    for column in boolean_columns:
        smartphone_df[column] = smartphone_df[column].fillna(0).astype(int)

    if drop_reference_columns:
        reference_columns = [column for column in smartphone_df.columns if "url" in column or "link" in column]
        if reference_columns:
            smartphone_df = smartphone_df.drop(columns=reference_columns, errors="ignore")
            metadata["dropped_reference_columns"] = sorted(reference_columns)

    metadata.update(
        {
            "smartphone_preprocessing_applied": True,
            "ecommerce_preprocessing_applied": True,
            "recommendation_ready": True,
            "noise_fixes": noise_fixes,
            "shifted_column_fixes": shifted_fixes,
            "cleaned_numeric_columns": ["price", "rating"],
            "extracted_feature_columns": extracted_features,
            "normalized_categorical_columns": categorical_columns,
            "raw_source_columns_excluded_from_ml": [
                "model",
                "sim",
                "processor",
                "ram",
                "battery",
                "display",
                "camera",
                "card",
                "os",
                "extra_features",
                "combined_text_features",
                "os_version",
                "phone_id",
            ],
            "smartphone_feature_columns": extracted_features + boolean_columns + categorical_columns + ["combined_text_features", "phone_id"],
        }
    )

    return smartphone_df, metadata


def apply_smartphone_domain_outlier_rules(
    dataframe: pd.DataFrame,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Apply domain-safe bounds for smartphone specifications."""
    adjusted_df = dataframe.copy(deep=True)
    adjustments: list[dict[str, Any]] = []

    def _invalidate_by_rule(column: str, rule: str, mask: pd.Series) -> None:
        affected = int(mask.fillna(False).sum())
        if affected <= 0:
            return
        adjusted_df.loc[mask, column] = pd.NA
        adjustments.append(
            {
                "column_name": column,
                "action_taken": "flagged_as_invalid",
                "affected_rows": affected,
                "rule": rule,
            }
        )

    if "price" in adjusted_df.columns:
        _invalidate_by_rule("price", "price must be positive", adjusted_df["price"] <= 0)
        premium_price_mask = adjusted_df["price"] > 500000
        premium_price_rows = int(premium_price_mask.fillna(False).sum())
        if premium_price_rows > 0:
            adjustments.append(
                {
                    "column_name": "price",
                    "action_taken": "flagged_as_high_end_outlier_kept",
                    "affected_rows": premium_price_rows,
                    "rule": "price above 500000 was preserved and flagged as a high-end/luxury outlier instead of being dropped",
                }
            )
    if "rating" in adjusted_df.columns:
        _invalidate_by_rule("rating", "rating must stay within the source scale 0 to 100", (adjusted_df["rating"] < 0) | (adjusted_df["rating"] > 100))
    if "battery_mah" in adjusted_df.columns:
        _invalidate_by_rule("battery_mah", "battery capacity must be 1000 to 10000 mAh", (adjusted_df["battery_mah"] < 1000) | (adjusted_df["battery_mah"] > 10000))
    if "screen_size_inches" in adjusted_df.columns:
        _invalidate_by_rule("screen_size_inches", "screen size must be 1.5 to 10 inches", (adjusted_df["screen_size_inches"] < 1.5) | (adjusted_df["screen_size_inches"] > 10))
    if "refresh_rate_hz" in adjusted_df.columns:
        _invalidate_by_rule("refresh_rate_hz", "refresh rate must be 30 to 240 Hz", (adjusted_df["refresh_rate_hz"] < 30) | (adjusted_df["refresh_rate_hz"] > 240))
    if "charging_watt" in adjusted_df.columns:
        _invalidate_by_rule("charging_watt", "charging wattage must be 1 to 240 W", (adjusted_df["charging_watt"] < 1) | (adjusted_df["charging_watt"] > 240))
    if "processor_speed_ghz" in adjusted_df.columns:
        _invalidate_by_rule("processor_speed_ghz", "processor speed must be 0.5 to 5 GHz", (adjusted_df["processor_speed_ghz"] < 0.5) | (adjusted_df["processor_speed_ghz"] > 5))
    if "memory_card_max_gb" in adjusted_df.columns:
        _invalidate_by_rule(
            "memory_card_max_gb",
            "memory card capacity must be 0 to 2048 GB",
            (adjusted_df["memory_card_max_gb"] < 0) | (adjusted_df["memory_card_max_gb"] > 2048),
        )

    return adjusted_df, adjustments


_PROCESSOR_POLLUTION_PATTERNS = re.compile(
    r"\b\d+\s*gb\s*ram\b|\b\d+\s*mb\s*ram\b|\b\d+\s*gb\s*inbuilt\b"
    r"|\bno\s+3g\b|\bno\s+4g\b|\bno\s+wifi\b|\bno\s+volte\b",
    re.IGNORECASE,
)
_MB_RAM_PATTERN = re.compile(r"\d+\s*mb\s*ram", re.IGNORECASE)

_KNOWN_BRAND_SET = frozenset(b.lower() for b in SMARTPHONE_BRANDS)

_OUTLIER_REASON_TAG = "parsed price is above 500000 and should be treated as a preserved high-end/luxury outlier"


def _count_column_shift_signals(
    processor: str,
    display: str,
    camera: str,
    card: str,
    os_value: str | None,
) -> int:
    """Count how many core columns look like they contain misaligned data."""
    score = 0
    if _PROCESSOR_POLLUTION_PATTERNS.search(processor):
        score += 1
    if _looks_like_camera_text(display) and "inch" not in display.lower():
        score += 1
    if _looks_like_os(camera):
        score += 1
    if any(token in card.lower() for token in ["bluetooth", "browser", "fm radio"]):
        score += 1
    if os_value is None:
        score += 1
    return score


def validate_smartphone_dataset_quality(
    dataframe: pd.DataFrame,
    *,
    mode: str = "safe",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Detect suspicious smartphone rows and optionally drop critical ones.

    Safe mode preserves every row and only reports suspicious records.
    Strict mode removes critical rows whose fields are badly misaligned or
    clearly invalid for recommendation modelling.

    The returned quality report contains a ``data_validity_report`` section
    with per-category validation summaries and the full suspicious record list.
    """
    validated_df = dataframe.copy(deep=True)
    normalized_mode = "strict" if str(mode).strip().lower() == "strict" else "safe"
    suspicious_records_details: list[dict[str, Any]] = []
    critical_row_indices: list[int] = []
    invalid_ml_ready_brand_columns: list[str] = []

    for index, row in validated_df.iterrows():
        model = _text_or_none(row.get("model")) or ""
        brand = _text_or_none(row.get("brand")) or _extract_brand(model)
        processor = _text_or_none(row.get("processor")) or ""
        display = _text_or_none(row.get("display")) or ""
        camera = _text_or_none(row.get("camera")) or ""
        card = _text_or_none(row.get("card")) or ""
        os_value = _text_or_none(row.get("os"))
        ram_text = _text_or_none(row.get("ram")) or ""
        price = pd.to_numeric(pd.Series([row.get("price")]), errors="coerce").iloc[0]
        screen_size = pd.to_numeric(pd.Series([row.get("screen_size_inches")]), errors="coerce").iloc[0]

        reasons: list[str] = []
        severity = "warning"

        # --- brand / model validity ---
        if brand.lower() == "achhe" or "achhe din mobile" in model.lower():
            reasons.append("invalid or suspicious smartphone model name")
        elif brand.lower() not in _KNOWN_BRAND_SET:
            reasons.append(f"brand '{brand}' is not in the known smartphone brand list — verify manually")

        # --- price range ---
        if pd.notna(price) and price < 500:
            reasons.append(f"unrealistic price {price:.0f} PKR — suspiciously low for a smartphone")
        if pd.notna(price) and price > 500000:
            reasons.append(_OUTLIER_REASON_TAG)

        # --- display column sanity ---
        if display and _looks_like_camera_text(display) and "inch" not in display.lower():
            reasons.append("display column contains camera specs instead of screen information")

        # --- camera column sanity ---
        if _looks_like_os(camera):
            reasons.append("camera column contains operating-system text instead of camera specifications")

        # --- card column sanity ---
        if any(token in card.lower() for token in ["bluetooth", "browser", "fm radio"]):
            reasons.append("card column contains connectivity/browser metadata instead of memory-card details")

        # --- screen size extraction ---
        if pd.isna(screen_size):
            reasons.append("screen_size_inches could not be extracted from the display text")

        # --- OS validation ---
        if os_value is None:
            reasons.append("OS field is missing — no operating system could be identified")
        elif not _looks_like_os(os_value) and _looks_like_os_pollution(os_value):
            reasons.append("OS field contains non-OS metadata instead of an operating system name")

        # --- processor column validation (F) ---
        if _PROCESSOR_POLLUTION_PATTERNS.search(processor):
            reasons.append(
                "processor column contains RAM/storage/network information instead of processor details"
            )

        # --- column-shift / corruption scoring (G) ---
        shift_score = _count_column_shift_signals(processor, display, camera, card, os_value)
        if shift_score >= 5:
            reasons.append(
                f"critical column shift / corrupted row: {shift_score} of 5 core fields contain misaligned data"
            )
        elif shift_score >= 3:
            reasons.append(
                f"possible column shift / corrupted row: {shift_score} fields appear to contain misaligned data"
            )

        # --- feature-phone-like detection (I) ---
        has_mb_ram = bool(_MB_RAM_PATTERN.search(ram_text))
        feature_phone_flags: list[str] = []
        if pd.notna(screen_size) and screen_size < 3.0:
            feature_phone_flags.append("very small screen (< 3 inches)")
        if pd.notna(price) and price < 1000:
            feature_phone_flags.append("very low price (< 1000 PKR)")
        if has_mb_ram:
            feature_phone_flags.append("RAM measured in MB instead of GB")
        if os_value is None or not _looks_like_os(os_value):
            feature_phone_flags.append("no smartphone OS detected")
        if len(feature_phone_flags) >= 2:
            reasons.append("feature-phone-like record: " + ", ".join(feature_phone_flags))

        # --- severity classification ---
        non_outlier_reasons = [r for r in reasons if _OUTLIER_REASON_TAG not in r]
        if non_outlier_reasons:
            if (
                brand.lower() == "achhe"
                or "achhe din mobile" in model.lower()
                or (pd.notna(price) and price < 500 and len(non_outlier_reasons) >= 3)
                or shift_score >= 5
            ):
                severity = "critical"

        if reasons:
            suspicious_records_details.append(
                {
                    "row_index": int(index),
                    "model": model or "Unknown",
                    "brand": brand or "Unknown",
                    "severity": severity,
                    "reasons": reasons,
                }
            )
            if severity == "critical":
                critical_row_indices.append(int(index))
                invalid_ml_ready_brand_columns.append(f"brand_{brand}")

    invalid_ml_ready_brand_columns = sorted(set(invalid_ml_ready_brand_columns))
    rows_removed = len(set(critical_row_indices)) if normalized_mode == "strict" else 0

    if normalized_mode == "strict" and critical_row_indices:
        validated_df = validated_df.drop(index=critical_row_indices).reset_index(drop=True)

    critical_count = sum(1 for item in suspicious_records_details if item["severity"] == "critical")

    data_validity_report: dict[str, Any] = {
        "input_row_count": len(dataframe),
        "final_row_count": len(dataframe) - rows_removed,
        "duplicate_count": int(dataframe.duplicated().sum()),
        "invalid_model_names_detected": sorted(
            {item["model"] for item in suspicious_records_details if item["severity"] == "critical"}
        ),
        "suspicious_records_count": len(suspicious_records_details),
        "critical_records_count": critical_count,
        "suspicious_records_details": suspicious_records_details,
        "price_validation": {
            "rows_with_suspicious_low_price": sum(
                1 for item in suspicious_records_details
                if any("unrealistic price" in r or "suspiciously low" in r for r in item["reasons"])
            ),
            "rows_with_high_end_price": sum(
                1 for item in suspicious_records_details
                if any("high-end/luxury outlier" in r for r in item["reasons"])
            ),
        },
        "ram_storage_validation": {
            "rows_with_mb_ram": sum(
                1 for item in suspicious_records_details
                if any("mb instead of gb" in r.lower() or "measured in mb" in r.lower() for r in item["reasons"])
            ),
        },
        "processor_validation": {
            "rows_with_processor_pollution": sum(
                1 for item in suspicious_records_details
                if any("processor column contains" in r for r in item["reasons"])
            ),
        },
        "display_validation": {
            "rows_with_display_column_issues": sum(
                1 for item in suspicious_records_details
                if any("display column contains" in r for r in item["reasons"])
            ),
        },
        "camera_validation": {
            "rows_with_camera_column_issues": sum(
                1 for item in suspicious_records_details
                if any("camera column contains" in r for r in item["reasons"])
            ),
        },
        "os_validation": {
            "rows_with_missing_os": sum(
                1 for item in suspicious_records_details
                if any("os field is missing" in r.lower() for r in item["reasons"])
            ),
            "rows_with_invalid_os": sum(
                1 for item in suspicious_records_details
                if any("os field contains non-os" in r.lower() for r in item["reasons"])
            ),
        },
        "column_shift_validation": {
            "rows_with_critical_column_shift": sum(
                1 for item in suspicious_records_details
                if any("critical column shift" in r for r in item["reasons"])
            ),
            "rows_with_possible_column_shift": sum(
                1 for item in suspicious_records_details
                if any("possible column shift" in r for r in item["reasons"])
            ),
        },
        "feature_phone_validation": {
            "feature_phone_like_rows": sum(
                1 for item in suspicious_records_details
                if any("feature-phone-like" in r for r in item["reasons"])
            ),
        },
        "strict_mode_enabled": normalized_mode == "strict",
        "rows_removed_in_strict_mode": rows_removed,
        "final_dataset_status": (
            "critical_issues_present" if critical_count > 0
            else "warnings_only" if suspicious_records_details
            else "passed"
        ),
    }

    quality_report = {
        "mode": normalized_mode,
        "suspicious_records_count": len(suspicious_records_details),
        "critical_suspicious_records_count": critical_count,
        "suspicious_records_details": suspicious_records_details,
        "critical_row_indices": sorted(set(critical_row_indices)),
        "invalid_ml_ready_brand_columns": invalid_ml_ready_brand_columns,
        "rows_removed_in_strict_mode": rows_removed,
        "strict_mode_applied": normalized_mode == "strict",
        "data_validity_report": data_validity_report,
    }
    return validated_df, quality_report


def build_smartphone_output_datasets(dataframe: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create the readable and ML-ready smartphone recommendation datasets."""
    readable_priority_columns = [
        "phone_id",
        "model",
        "brand",
        "price",
        "rating",
        "sim",
        "processor",
        "ram",
        "battery",
        "display",
        "camera",
        "card",
        "os",
        "price_segment",
        "ram_gb",
        "storage_gb",
        "battery_mah",
        "charging_watt",
        "screen_size_inches",
        "resolution_width",
        "resolution_height",
        "refresh_rate_hz",
        "rear_main_camera_mp",
        "front_camera_mp",
        "rear_camera_count",
        "processor_brand",
        "processor_speed_ghz",
        "has_5g",
        "has_4g",
        "has_volte",
        "has_wifi",
        "has_nfc",
        "has_ir_blaster",
        "memory_card_supported",
        "memory_card_hybrid",
        "memory_card_max_gb",
        "os_family",
        "os_version",
        "combined_text_features",
        "extra_features",
    ]
    readable_columns = [column for column in readable_priority_columns if column in dataframe.columns]
    readable_df = dataframe[readable_columns].copy(deep=True)

    def _restore_readable_category(
        target_column: str,
        encoded_prefix: str,
        default_value: str = "Unknown",
    ) -> None:
        if target_column in readable_df.columns:
            return
        encoded_columns = sorted(column for column in dataframe.columns if column.startswith(encoded_prefix))
        if not encoded_columns:
            return

        encoded_frame = dataframe[encoded_columns].apply(pd.to_numeric, errors="coerce").fillna(0)
        restored_values = encoded_frame.idxmax(axis=1).str.replace(encoded_prefix, "", regex=False)
        insert_position = min(
            readable_priority_columns.index(target_column),
            len(readable_df.columns),
        )
        readable_df.insert(insert_position, target_column, restored_values.where(restored_values != "", default_value))

    _restore_readable_category("brand", "brand_")
    _restore_readable_category("processor_brand", "processor_brand_")
    _restore_readable_category("os_family", "os_family_")
    _restore_readable_category("price_segment", "price_segment_")

    for column in BOOLEAN_FEATURE_COLUMNS:
        if column in readable_df.columns:
            readable_df[column] = (
                pd.to_numeric(readable_df[column], errors="coerce").fillna(0).astype(int).astype(bool)
            )

    ml_ready_base = dataframe.copy(deep=True)
    scalable_columns_present = [column for column in SCALABLE_NUMERIC_COLUMNS if column in ml_ready_base.columns]

    # The readable smartphone dataset preserves human-friendly numeric values,
    # while the ML-ready export recomputes its scaled columns with MinMaxScaler
    # so cosine similarity can compare features on a common range.
    if scalable_columns_present:
        numeric_frame = ml_ready_base[scalable_columns_present].apply(pd.to_numeric, errors="coerce")
        for column in scalable_columns_present:
            column_series = numeric_frame[column]
            median_value = column_series.median(skipna=True)
            numeric_frame[column] = column_series.fillna(0 if pd.isna(median_value) else median_value)

        scaler = MinMaxScaler()
        scaled_values = scaler.fit_transform(numeric_frame[scalable_columns_present])
        for index, column in enumerate(scalable_columns_present):
            ml_ready_base[f"{column}_scaled"] = scaled_values[:, index]

    for column in BOOLEAN_FEATURE_COLUMNS:
        if column in ml_ready_base.columns:
            ml_ready_base[column] = pd.to_numeric(ml_ready_base[column], errors="coerce").fillna(0).astype(int)

    for column in ["brand", "processor_brand", "os_family", "price_segment"]:
        if column in ml_ready_base.columns:
            ml_ready_base[column] = ml_ready_base[column].fillna("Unknown")

    raw_categoricals = [column for column in ["brand", "processor_brand", "os_family", "price_segment"] if column in ml_ready_base.columns]
    if raw_categoricals:
        ml_ready_base = pd.get_dummies(
            ml_ready_base,
            columns=raw_categoricals,
            drop_first=False,
            dtype=int,
        )

    encoded_prefixes = ("brand_", "processor_brand_", "os_family_", "price_segment_")
    required_base_columns = ["phone_id", "model"]
    scaled_columns = [f"{column}_scaled" for column in SCALABLE_NUMERIC_COLUMNS if f"{column}_scaled" in ml_ready_base.columns]
    boolean_columns = [column for column in BOOLEAN_FEATURE_COLUMNS if column in ml_ready_base.columns]
    categorical_encoded_columns = [
        column for column in ml_ready_base.columns if column.startswith(encoded_prefixes)
    ]
    ml_ready_columns = required_base_columns + scaled_columns + boolean_columns + sorted(categorical_encoded_columns)
    ml_ready_columns = [column for column in ml_ready_columns if column in ml_ready_base.columns]
    ml_ready_df = ml_ready_base[ml_ready_columns].copy(deep=True)
    for column in ml_ready_df.columns:
        if str(ml_ready_df[column].dtype) == "bool":
            ml_ready_df[column] = ml_ready_df[column].astype(int)

    boolean_string_columns = [
        column
        for column in ml_ready_df.columns
        if ml_ready_df[column].astype(str).str.lower().isin(["true", "false"]).all()
    ]
    for column in boolean_string_columns:
        ml_ready_df[column] = (
            ml_ready_df[column]
            .astype(str)
            .str.lower()
            .map({"true": 1, "false": 0})
            .astype(int)
        )

    if len(ml_ready_df.columns) > 2:
        feature_columns = [column for column in ml_ready_df.columns if column not in {"phone_id", "model"}]
        ml_ready_df[feature_columns] = ml_ready_df[feature_columns].apply(pd.to_numeric, errors="coerce").fillna(0)

    # Constant recommendation features add no signal for cosine similarity, so
    # drop them only from the ML-ready export while keeping the readable file
    # intact for EDA and viva explanations.
    constant_feature_columns = [
        column
        for column in ml_ready_df.columns
        if column not in {"phone_id", "model"} and ml_ready_df[column].nunique(dropna=False) <= 1
    ]
    if constant_feature_columns:
        ml_ready_df = ml_ready_df.drop(columns=constant_feature_columns)
    ml_ready_df.attrs["constant_features_dropped_from_ml_ready"] = constant_feature_columns
    return readable_df, ml_ready_df


def validate_smartphone_outputs(dataframe: pd.DataFrame, ml_ready_df: pd.DataFrame | None = None) -> list[dict[str, Any]]:
    """Run dataset-specific validation checks for the smartphone workflow."""
    checks: list[dict[str, Any]] = []
    _, quality_report = validate_smartphone_dataset_quality(dataframe, mode="safe")

    def _record(name: str, passed: bool, details: str) -> None:
        checks.append({"check": name, "passed": bool(passed), "details": details})

    _record("price_numeric", pd.api.types.is_numeric_dtype(dataframe.get("price")), "price should be numeric")
    price_has_rs = dataframe["price"].astype(str).str.contains("rs", case=False, na=False).any() if "price" in dataframe.columns else True
    _record("price_without_rs_text", not price_has_rs, "cleaned price should not contain Rs text")

    if "rating" in dataframe.columns:
        rating_series = pd.to_numeric(dataframe["rating"], errors="coerce").dropna()
        _record(
            "rating_not_constant_5",
            not (not rating_series.empty and rating_series.nunique() == 1 and float(rating_series.iloc[0]) == 5.0),
            "rating should not collapse into a constant 5.0 column",
        )
    else:
        _record("rating_not_constant_5", False, "rating column missing")

    if "rating_scaled" in dataframe.columns:
        rating_scaled = pd.to_numeric(dataframe["rating_scaled"], errors="coerce").fillna(0)
        _record("rating_scaled_not_all_zero", not np.isclose(rating_scaled.abs().sum(), 0.0), "rating_scaled should not be all zeros")

    _record("segment_dropped", "segment" not in dataframe.columns, "segment should be dropped because it is fully empty and not a target")

    for column in [
        "processor_speed_ghz",
        "screen_size_inches",
        "rear_main_camera_mp",
        "refresh_rate_hz",
        "battery_mah",
        "charging_watt",
    ]:
        passed = column in dataframe.columns and pd.to_numeric(dataframe[column], errors="coerce").notna().any()
        _record(f"{column}_extracted", passed, f"{column} should be extracted where available")

    text_columns = [column for column in ["processor", "display", "camera", "card", "os", "combined_text_features"] if column in dataframe.columns]
    no_noise = True
    for column in text_columns:
        if dataframe[column].astype(str).str.contains(r"\?", regex=True, na=False).any():
            no_noise = False
            break
    _record("no_question_mark_noise", no_noise, "question-mark noise should be removed from cleaned text fields")

    if "combined_text_features" in dataframe.columns:
        combined_text = " ".join(dataframe["combined_text_features"].dropna().astype(str).head(50).tolist())
        expected_token_hits = sum(token in combined_text for token in ["5g", "mp", "hz", "mah", "snapdragon", "dimensity", "android", "ios"])
        _record("combined_text_preserves_useful_tokens", expected_token_hits >= 4, "combined_text_features should preserve important smartphone tokens")
    else:
        _record("combined_text_preserves_useful_tokens", False, "combined_text_features column missing")

    _record(
        "critical_suspicious_records_flagged",
        quality_report.get("critical_suspicious_records_count", 0) >= 0,
        "suspicious smartphone records should be identified and reported for manual review or strict-mode removal",
    )

    if ml_ready_df is not None:
        feature_count = max(0, len(ml_ready_df.columns) - 2)
        _record("ml_ready_feature_count", feature_count > 20, "ML-ready dataset should contain more than 20 useful recommendation features")
        dropped_constant_features = set(ml_ready_df.attrs.get("constant_features_dropped_from_ml_ready", []))
        required_scaled_columns = [
            "price_scaled",
            "rating_scaled",
            "ram_gb_scaled",
            "storage_gb_scaled",
            "battery_mah_scaled",
            "charging_watt_scaled",
            "screen_size_inches_scaled",
            "refresh_rate_hz_scaled",
            "rear_main_camera_mp_scaled",
            "front_camera_mp_scaled",
            "processor_speed_ghz_scaled",
        ]
        missing_scaled_columns = [
            column
            for column in required_scaled_columns
            if column not in ml_ready_df.columns and column not in dropped_constant_features
        ]
        _record(
            "required_scaled_features_present",
            not missing_scaled_columns,
            "ML-ready dataset should include the required scaled numeric smartphone features",
        )
        polluted_os_columns = [
            column for column in ml_ready_df.columns
            if column.startswith("os_family_") and any(token in column.lower() for token in ["bluetooth", "browser", "memory_card", "fm_radio"])
        ]
        _record("os_encoding_not_polluted", not polluted_os_columns, "polluted OS values should not become OS encoded columns")
        _record(
            "ml_ready_has_no_missing_values",
            not ml_ready_df.isna().any().any(),
            "ML-ready dataset should not contain missing values",
        )
        constant_features = [
            column
            for column in ml_ready_df.columns
            if column not in {"phone_id", "model"} and ml_ready_df[column].nunique(dropna=False) <= 1
        ]
        _record(
            "ml_ready_constant_features_dropped",
            not constant_features,
            "constant features should be removed from the ML-ready dataset because they add no similarity signal",
        )
        invalid_brand_columns = [
            column
            for column in ml_ready_df.columns
            if column.startswith("brand_") and column in quality_report.get("invalid_ml_ready_brand_columns", [])
        ]
        _record(
            "invalid_brand_columns_removed_or_flagged",
            not invalid_brand_columns,
            "invalid suspicious brand columns such as brand_Achhe should not survive in a strict ML-ready export",
        )

    _record("records_preserved", len(dataframe) > 0, "cleaned dataset should preserve smartphone records")
    return checks
