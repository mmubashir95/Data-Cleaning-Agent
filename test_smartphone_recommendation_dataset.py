"""Tests for the smartphone-specific recommendation preprocessing path."""

from __future__ import annotations

import unittest
from pathlib import Path

import pandas as pd

from utils.data_cleaner import clean_dataset
from utils.data_profiler import profile_dataset
from utils.ecommerce_preprocessing import build_ecommerce_output_datasets, build_ecommerce_preprocessed_view
from utils.ml_recommender import recommend_ml_approach
from utils.report_generator import generate_cleaning_report
from utils.smartphone_preprocessing import (
    detect_smartphone_dataset,
    validate_smartphone_dataset_quality,
    validate_smartphone_outputs,
)


def _smartphone_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "model": [
                "OnePlus 11 5G",
                "Samsung Galaxy S23 Ultra 5G",
                "Nothing Phone 1",
                "Apple iPhone 14",
            ],
            "Price": ["Rs.54999", "Rs.1,14,990", "Rs.26749", "Rs.65999"],
            "rating": [89, None, 85, 81],
            "sim": [
                "Dual Sim, 3G, 4G, 5G, VoLTE, Wi-Fi, NFC",
                "Dual Sim, 3G, 4G, 5G, VoLTE, Vo5G, Wi-Fi, NFC",
                "Dual Sim, 3G, 4G, 5G, VoLTE, Wi-Fi, NFC",
                "Dual Sim, 3G, 4G, 5G, VoLTE, Wi-Fi, NFC",
            ],
            "processor": [
                "snapdragon 8 gen2, octa core, 3.2?ghz processor",
                "snapdragon 8 gen2, octa core, 3.2?ghz processor",
                "snapdragon 778g plus, octa core, 2.5?ghz processor",
                "bionic a15, hexa core, 3.22?ghz processor",
            ],
            "ram": [
                "12GB RAM, 512GB inbuilt",
                "8GB RAM, 256GB inbuilt",
                "12GB RAM, 512GB inbuilt",
                "8GB RAM, 256GB inbuilt",
            ],
            "battery": [
                "6000mAh Battery with Fast Charging",
                "5000mAh Battery with 67W Fast Charging",
                "6000mAh Battery with Fast Charging",
                "5000mAh Battery with 67W Fast Charging",
            ],
            "display": [
                "6.7 inches, 1440?x?3216?px, 120 Hz Display with Punch Hole",
                "6.8 inches, 1440?x?3088?px, 120 Hz Display with Punch Hole",
                "6.55 inches, 1080?x?2400?px, 120 Hz Display with Punch Hole",
                "6.1 inches, 1170?x?2532?px Display with Small Notch",
            ],
            "camera": [
                "50?MP + 48?MP + 32?MP Triple Rear & 16?MP Front Camera",
                "200?MP Quad Rear & 12?MP Front Camera",
                "50?MP + 50?MP Dual Rear & 16?MP Front Camera",
                "12?MP + 12?MP Dual Rear & 12?MP Front Camera",
            ],
            "card": [
                "memory card supported, upto 1?TB",
                "Memory Card Not Supported",
                "Android v12",
                "Memory Card Not Supported",
            ],
            "os": [
                "android V12",
                "Android v13",
                "No FM Radio",
                "iOS v16",
            ],
            "Segment": [None, None, None, None],
        }
    )


class TestSmartphoneRecommendationDataset(unittest.TestCase):
    def test_quality_validator_flags_critical_suspicious_rows_and_supports_strict_mode(self):
        df = pd.DataFrame(
            {
                "model": ["Achhe Din Mobile", "Vertu Signature Touch", "Nokia 105 (2019)"],
                "Price": ["Rs.99", "Rs.6,50,000", "Rs.1299"],
                "rating": [None, 62, 54],
                "sim": [
                    "Dual Sim, 3G, Wi-Fi",
                    "Single Sim, 3G, 4G, Wi-Fi, NFC",
                    "Dual Sim",
                ],
                "processor": [
                    "1?gb ram, 4?gb inbuilt",
                    "snapdragon 801, octa core, 1.5?ghz processor",
                    "No",
                ],
                "ram": [
                    "12GB RAM, 512GB inbuilt",
                    "6GB RAM, 128GB inbuilt",
                    "4MB RAM",
                ],
                "battery": [
                    "6000mAh Battery with Fast Charging",
                    "5000mAh Battery with 33W Turbo Charge",
                    "800mAh Battery",
                ],
                "display": [
                    "2?MP Rear & 0.3?MP Front Camera",
                    "4.7 inches, 1080?x?1920?px Display",
                    "1.77 inches, 120?x?160?px Display",
                ],
                "camera": [
                    "Android v5.0 (Lollipop)",
                    "13?MP Rear & 2.1?MP Front Camera",
                    "No",
                ],
                "card": ["Bluetooth", "Memory Card Not Supported", "Memory Card Not Supported"],
                "os": [None, "Android v4.4.2 (KitKat)", None],
                "Segment": [None, None, None],
            }
        )
        preview_df, _ = build_ecommerce_preprocessed_view(df, drop_reference_columns=True)

        safe_df, safe_report = validate_smartphone_dataset_quality(preview_df, mode="safe")
        self.assertEqual(len(safe_df), 3)
        self.assertTrue(any(item["model"] == "Achhe Din Mobile" and item["severity"] == "critical" for item in safe_report["suspicious_records_details"]))
        self.assertIn("brand_Achhe", safe_report["invalid_ml_ready_brand_columns"])

        strict_df, strict_report = validate_smartphone_dataset_quality(preview_df, mode="strict")
        self.assertEqual(len(strict_df), 2)
        self.assertFalse(strict_df["model"].astype(str).str.contains("Achhe Din Mobile", case=False, na=False).any())
        self.assertEqual(strict_report["rows_removed_in_strict_mode"], 1)

    def test_duplicate_removal_keeps_distinct_smartphones_with_similar_names(self):
        df = pd.DataFrame(
            {
                "model": ["Motorola Moto G82 5G", "Motorola Moto G82 5G"],
                "Price": ["Rs.19999", "Rs.20999"],
                "rating": [81, 82],
                "sim": ["Dual Sim, 4G, 5G, VoLTE, Wi-Fi", "Dual Sim, 4G, 5G, VoLTE, Wi-Fi"],
                "processor": ["snapdragon 695, octa core, 2.2?ghz processor", "snapdragon 695, octa core, 2.2?ghz processor"],
                "ram": ["6GB RAM, 128GB inbuilt", "8GB RAM, 128GB inbuilt"],
                "battery": ["5000mAh Battery with 30W Fast Charging", "5000mAh Battery with 30W Fast Charging"],
                "display": ["6.6 inches, 1080?x?2400?px, 120 Hz Display", "6.6 inches, 1080?x?2400?px, 120 Hz Display"],
                "camera": ["50?MP + 8?MP + 2?MP Triple Rear & 16?MP Front Camera", "50?MP + 8?MP + 2?MP Triple Rear & 16?MP Front Camera"],
                "card": ["memory card supported, upto 1TB", "memory card supported, upto 1TB"],
                "os": ["Android v12", "Android v12"],
                "Segment": [None, None],
            }
        )
        cleaned_df, summary = clean_dataset(
            df,
            {
                "remove_duplicates": True,
                "handle_missing_values": True,
                "fix_data_types": True,
                "handle_outliers": True,
                "encode_categorical": True,
                "scale_numeric": True,
                "scaler_choice": "StandardScaler",
                "nlp_cleaning": False,
            },
            target_column="Segment",
        )

        self.assertEqual(len(cleaned_df), 2)
        self.assertEqual(summary["semantic_duplicate_rows_removed"], 0)

    def test_detection_and_preprocessing_extract_recommendation_features(self):
        df = _smartphone_df()
        self.assertTrue(detect_smartphone_dataset(df.columns))

        preview_df, metadata = build_ecommerce_preprocessed_view(df, drop_reference_columns=True)
        self.assertTrue(metadata["smartphone_preprocessing_applied"])
        self.assertTrue(metadata["ecommerce_preprocessing_applied"])
        self.assertIn("price_segment", preview_df.columns)
        self.assertIn("processor_speed_ghz", preview_df.columns)
        self.assertIn("screen_size_inches", preview_df.columns)
        self.assertIn("rear_main_camera_mp", preview_df.columns)
        self.assertIn("combined_text_features", preview_df.columns)
        self.assertNotIn("segment", preview_df.columns)
        self.assertTrue(pd.api.types.is_numeric_dtype(preview_df["price"]))
        self.assertTrue(pd.api.types.is_numeric_dtype(preview_df["rating"]))
        self.assertTrue(preview_df["combined_text_features"].str.contains("5g").any())

    def test_cleaning_pipeline_produces_strong_ml_ready_smartphone_features(self):
        df = _smartphone_df()
        cleaned_df, summary = clean_dataset(
            df,
            {
                "remove_duplicates": True,
                "handle_missing_values": True,
                "fix_data_types": True,
                "handle_outliers": True,
                "encode_categorical": True,
                "scale_numeric": True,
                "scaler_choice": "StandardScaler",
                "nlp_cleaning": True,
            },
            target_column="Segment",
        )

        self.assertTrue(summary["smartphone_preprocessing_applied"])
        self.assertIn("processor_speed_ghz", cleaned_df.columns)
        self.assertIn("screen_size_inches", cleaned_df.columns)
        self.assertIn("rear_main_camera_mp", cleaned_df.columns)
        self.assertIn("refresh_rate_hz", cleaned_df.columns)
        self.assertIn("battery_mah", cleaned_df.columns)
        self.assertIn("charging_watt", cleaned_df.columns)
        self.assertIn("price_scaled", cleaned_df.columns)
        self.assertIn("rating_scaled", cleaned_df.columns)
        self.assertEqual(summary["scaler_used"], "MinMaxScaler")
        self.assertEqual(float(cleaned_df.loc[0, "price"]), 54999.0)
        self.assertEqual(float(cleaned_df.loc[1, "price"]), 114990.0)
        self.assertEqual(float(cleaned_df.loc[0, "ram_gb"]), 12.0)
        self.assertEqual(float(cleaned_df.loc[0, "storage_gb"]), 512.0)
        self.assertEqual(float(cleaned_df.loc[0, "battery_mah"]), 6000.0)
        self.assertFalse((cleaned_df["price"] > 0).all() and (cleaned_df["price"] < 1).any())
        self.assertFalse(cleaned_df["rating"].nunique(dropna=True) == 1 and float(cleaned_df["rating"].dropna().iloc[0]) == 5.0)
        self.assertTrue(cleaned_df["price"].astype(str).str.contains("Rs", case=False, na=False).sum() == 0)
        self.assertFalse(cleaned_df["processor"].astype(str).str.contains(r"\?", regex=True, na=False).any())
        self.assertIn("brand", summary["encoded_columns"])
        self.assertIn("processor_brand", summary["encoded_columns"])
        self.assertIn("os_family", summary["encoded_columns"])
        self.assertIn("price_segment", summary["encoded_columns"])

        readable_df, ml_ready_df = build_ecommerce_output_datasets(cleaned_df)
        validation_checks = validate_smartphone_outputs(cleaned_df, ml_ready_df)
        failed_checks = [check for check in validation_checks if not check["passed"]]
        self.assertEqual(failed_checks, [], failed_checks)
        self.assertGreater(len(ml_ready_df.columns) - 2, 20)
        self.assertFalse(any(column.endswith("_scaled") for column in readable_df.columns))
        self.assertFalse(any(column.startswith(("brand_", "os_family_", "processor_brand_", "price_segment_")) for column in readable_df.columns))
        self.assertIn("brand", readable_df.columns)
        self.assertIn("os_family", readable_df.columns)
        self.assertIn("processor_brand", readable_df.columns)
        self.assertIn("price_segment", readable_df.columns)
        self.assertIn(1024.0, readable_df["memory_card_max_gb"].tolist())
        self.assertTrue(pd.api.types.is_bool_dtype(readable_df["has_5g"]))
        if "has_5g" in ml_ready_df.columns:
            self.assertTrue(pd.api.types.is_integer_dtype(ml_ready_df["has_5g"]))
        else:
            self.assertIn("has_5g", ml_ready_df.attrs.get("constant_features_dropped_from_ml_ready", []))
        self.assertFalse(any(str(ml_ready_df[column].dtype) == "bool" for column in ml_ready_df.columns))
        self.assertFalse(
            ml_ready_df.astype(str).apply(
                lambda series: series.str.contains(r"^(?:True|False)$", regex=True).any()
            ).any()
        )
        self.assertNotIn("has_fast_charging", ml_ready_df.columns)
        self.assertIn(
            "has_fast_charging",
            ml_ready_df.attrs.get("constant_features_dropped_from_ml_ready", []),
        )
        remaining_constant_columns = [
            column
            for column in ml_ready_df.columns
            if column not in {"phone_id", "model"} and ml_ready_df[column].nunique(dropna=False) <= 1
        ]
        self.assertEqual(remaining_constant_columns, [])
        self.assertIn("Premium", readable_df["price_segment"].tolist())
        self.assertIn("Flagship", readable_df["price_segment"].tolist())
        for column in [
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
        ]:
            if column not in ml_ready_df.columns:
                self.assertIn(
                    column,
                    ml_ready_df.attrs.get("constant_features_dropped_from_ml_ready", []),
                )
            else:
                self.assertIn(column, ml_ready_df.columns)
        self.assertFalse(ml_ready_df.isna().any().any())
        self.assertNotIn("segment", [column.lower() for column in ml_ready_df.columns])
        self.assertTrue(readable_df["combined_text_features"].str.contains("1tb", na=False).any())
        self.assertTrue(all("bluetooth" not in column.lower() for column in ml_ready_df.columns if column.startswith("os_family_")))
        self.assertEqual(readable_df.columns[1], "model")

    def test_recommendation_and_report_switch_to_smartphone_explanation(self):
        df = _smartphone_df()
        profile = profile_dataset(df)
        self.assertNotIn("Segment", profile["numeric_columns"])
        self.assertNotIn("Segment", profile["categorical_columns"])
        self.assertNotIn("Segment", profile["text_columns"])
        self.assertEqual(profile["ignored_columns"][0]["column"], "Segment")
        self.assertIn("not usable as a target column", profile["ignored_columns"][0]["reason"])

        recommendation = recommend_ml_approach(df, None, "Auto-detect", text_columns=[])
        self.assertEqual(recommendation["recommended_problem_type"], "Smartphone Content-Based Recommendation")
        self.assertTrue(recommendation["smartphone_dataset_detected"])
        self.assertEqual(recommendation["target_detection_metadata"]["top_suggestions"], [])
        self.assertIsNone(recommendation["target_column_used_for_inference"])
        self.assertEqual(
            recommendation["algorithm_recommendation"]["target_variable_type"],
            "Not applicable for content-based recommendation",
        )
        self.assertEqual(
            recommendation["detected_text_columns"],
            ["model", "processor", "display", "camera"],
        )
        self.assertEqual(recommendation["ignored_columns"][0]["column"], "Segment")
        self.assertEqual(
            recommendation["algorithm_recommendation"]["beginner_friendly_first_choice"]["name"],
            "Content-Based Recommendation with Cosine Similarity",
        )
        self.assertEqual(
            recommendation["algorithm_recommendation"]["recommended_algorithms"][1]["name"],
            "Clustering for Similar Phone Groups",
        )
        not_suitable = {item["approach"]: item["reason"] for item in recommendation["not_suitable_currently"]}
        self.assertIn("Classification", not_suitable)
        self.assertIn("Regression", not_suitable)
        self.assertIn("Collaborative filtering", not_suitable)

        cleaned_df, summary = clean_dataset(
            df,
            {
                "remove_duplicates": False,
                "handle_missing_values": True,
                "fix_data_types": True,
                "handle_outliers": False,
                "encode_categorical": True,
                "scale_numeric": True,
                "scaler_choice": "StandardScaler",
                "nlp_cleaning": False,
            },
            target_column=None,
        )
        _, ml_ready_df = build_ecommerce_output_datasets(cleaned_df)
        summary["smartphone_validation_checks"] = validate_smartphone_outputs(cleaned_df, ml_ready_df)
        summary["constant_features_dropped_from_ml_ready"] = list(
            ml_ready_df.attrs.get("constant_features_dropped_from_ml_ready", [])
        )

        report, report_path = generate_cleaning_report(
            {
                "rows": len(df),
                "columns": len(df.columns),
                "target_column": None,
                "numeric_columns": [],
                "categorical_columns": [],
                "text_columns": list(df.columns),
                "datetime_columns": [],
                "boolean_columns": [],
                "id_like_columns": [],
                "reference_columns": [],
            },
            {"errors": [], "warnings": []},
            summary,
            recommendation,
            "Tools-and-Techniques-MidTerm-Spring-2026.csv",
            cleaned_file_path="output/cleaned_tools_and_techniques_midterm_spring_2026.csv",
        )

        self.assertTrue(report["smartphone_preprocessing_applied"])
        self.assertEqual(report["recommendation_algorithm_suggested"], "Content-Based Recommendation using Cosine Similarity")
        self.assertEqual(report_path.endswith("cleaning_report_smartphone_dataset.json"), True)
        self.assertIn("complex and tricky", report["smartphone_preprocessing"]["complexity_note"].lower())
        self.assertIn("has_fast_charging", report["constant_features_dropped_from_ml_ready"])
        self.assertIn(
            "has_fast_charging",
            report["smartphone_preprocessing"]["constant_features_dropped_from_ml_ready"],
        )

    @unittest.skipUnless(
        Path("/Users/mohammadmubashir/Downloads/Tools-and-Techniques-MidTerm-Spring-2026.csv").exists(),
        "Real smartphone dataset not available on this machine.",
    )
    def test_real_dataset_acceptance_checks(self):
        real_path = Path("/Users/mohammadmubashir/Downloads/Tools-and-Techniques-MidTerm-Spring-2026.csv")
        df = pd.read_csv(real_path)

        cleaned_df, summary = clean_dataset(
            df,
            {
                "remove_duplicates": True,
                "handle_missing_values": True,
                "fix_data_types": True,
                "handle_outliers": True,
                "encode_categorical": True,
                "scale_numeric": True,
                "scaler_choice": "StandardScaler",
                "nlp_cleaning": False,
                "smartphone_quality_mode": "safe",
            },
            target_column="Segment",
        )
        readable_df, ml_ready_df = build_ecommerce_output_datasets(cleaned_df)
        summary["smartphone_validation_checks"] = validate_smartphone_outputs(cleaned_df, ml_ready_df)
        summary["constant_features_dropped_from_ml_ready"] = list(
            ml_ready_df.attrs.get("constant_features_dropped_from_ml_ready", [])
        )

        self.assertEqual(len(cleaned_df), 1023)
        self.assertEqual(len(ml_ready_df), 1023)
        self.assertEqual(int(ml_ready_df.isna().sum().sum()), 0)
        self.assertEqual(int(cleaned_df["price"].isna().sum()), 0)
        self.assertEqual(int(cleaned_df["screen_size_inches"].isna().sum()), 0)
        self.assertEqual(int(cleaned_df["price_scaled"].isna().sum()), 0)
        self.assertEqual(int(cleaned_df["screen_size_inches_scaled"].isna().sum()), 0)
        self.assertTrue(readable_df["model"].astype(str).str.contains("Achhe Din Mobile", case=False, na=False).any())

        quality_report = summary["smartphone_dataset_quality"]
        self.assertTrue(any(item["model"] == "Achhe Din Mobile" and item["severity"] == "critical" for item in quality_report["suspicious_records_details"]))
        self.assertIn("brand_Achhe", quality_report["invalid_ml_ready_brand_columns"])

        invalid_brand_check = next(
            check for check in summary["smartphone_validation_checks"]
            if check["check"] == "invalid_brand_columns_removed_or_flagged"
        )
        self.assertFalse(invalid_brand_check["passed"])
        self.assertIn("brand_Achhe", ml_ready_df.columns)

        vertu_rows = cleaned_df.loc[
            cleaned_df["model"].astype(str).str.contains("Vertu Signature Touch", case=False, na=False),
            ["model", "price"],
        ]
        self.assertEqual(float(vertu_rows.iloc[0]["price"]), 650000.0)

        for expected_value in [1.77, 1.8, 2.0, 2.4, 2.7, 2.8]:
            self.assertTrue((cleaned_df["screen_size_inches"] == expected_value).any(), expected_value)

        strict_cleaned_df, strict_summary = clean_dataset(
            df,
            {
                "remove_duplicates": True,
                "handle_missing_values": True,
                "fix_data_types": True,
                "handle_outliers": True,
                "encode_categorical": True,
                "scale_numeric": True,
                "scaler_choice": "StandardScaler",
                "nlp_cleaning": False,
                "smartphone_quality_mode": "strict",
            },
            target_column="Segment",
        )
        self.assertLess(len(strict_cleaned_df), 1023)
        self.assertFalse(strict_cleaned_df["model"].astype(str).str.contains("Achhe Din Mobile", case=False, na=False).any())
        self.assertTrue(strict_summary["suspicious_records_details"])


class TestEnhancedValidation(unittest.TestCase):
    """Tests for processor validation, column-shift scoring, feature-phone detection, and data_validity_report."""

    def _make_corrupted_df(self) -> pd.DataFrame:
        """Synthetic dataframe that exercises all validation paths."""
        return pd.DataFrame(
            {
                "model": [
                    "Achhe Din Mobile",       # critical: brand + price + column shift
                    "Vertu Signature Touch",  # warning: high-end price only
                    "Nokia 105 (2019)",       # warning: feature phone
                    "Samsung Galaxy S23",     # clean: no issues
                ],
                "Price": ["Rs.99", "Rs.6,50,000", "Rs.1299", "Rs.79999"],
                "rating": [None, 62, 54, 88],
                "sim": [
                    "Dual Sim, 3G, Wi-Fi",
                    "Single Sim, 4G, Wi-Fi, NFC",
                    "Dual Sim",
                    "Dual Sim, 4G, 5G, VoLTE, Wi-Fi, NFC",
                ],
                "processor": [
                    "1?gb ram, 4?gb inbuilt",                     # pollution in processor
                    "snapdragon 801, octa core, 1.5?ghz processor",
                    "No",
                    "snapdragon 8 gen2, octa core, 3.2?ghz processor",
                ],
                "ram": [
                    "12GB RAM, 512GB inbuilt",
                    "6GB RAM, 128GB inbuilt",
                    "4MB RAM",                # MB RAM feature-phone signal
                    "8GB RAM, 256GB inbuilt",
                ],
                "battery": [
                    "6000mAh Battery with Fast Charging",
                    "5000mAh Battery with 33W Turbo Charge",
                    "800mAh Battery",
                    "5000mAh Battery with 45W Fast Charging",
                ],
                "display": [
                    "2?MP Rear & 0.3?MP Front Camera",   # camera data in display
                    "4.7 inches, 1080?x?1920?px Display",
                    "1.77 inches, 120?x?160?px Display",
                    "6.1 inches, 1080?x?2340?px, 120 Hz Display",
                ],
                "camera": [
                    "Android v5.0 (Lollipop)",    # OS in camera
                    "13?MP Rear & 2.1?MP Front Camera",
                    "No",
                    "50?MP + 12?MP Dual Rear & 12?MP Front Camera",
                ],
                "card": [
                    "Bluetooth",                 # connectivity in card
                    "Memory Card Not Supported",
                    "Memory Card Not Supported",
                    "Memory Card Not Supported",
                ],
                "os": [None, "Android v4.4.2 (KitKat)", None, "Android v13"],
                "Segment": [None, None, None, None],
            }
        )

    def _preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        preview_df, _ = build_ecommerce_preprocessed_view(df, drop_reference_columns=True)
        return preview_df

    # --- Minimum test case 1: Achhe Din Mobile flagged as critical ---
    def test_achhe_din_mobile_is_flagged_as_critical(self):
        preview = self._preprocess(self._make_corrupted_df())
        _, report = validate_smartphone_dataset_quality(preview, mode="safe")
        achhe_entries = [
            item for item in report["suspicious_records_details"]
            if item["model"] == "Achhe Din Mobile"
        ]
        self.assertTrue(achhe_entries, "Achhe Din Mobile must appear in suspicious records")
        self.assertEqual(achhe_entries[0]["severity"], "critical")

    # --- Minimum test case 2: brand_Achhe causes validate_smartphone_outputs to fail ---
    def test_brand_achhe_fails_validate_smartphone_outputs(self):
        df = self._make_corrupted_df()
        cleaned_df, _ = clean_dataset(
            df,
            {
                "remove_duplicates": False,
                "handle_missing_values": True,
                "fix_data_types": True,
                "handle_outliers": False,
                "encode_categorical": True,
                "scale_numeric": True,
                "scaler_choice": "StandardScaler",
                "nlp_cleaning": False,
                "smartphone_quality_mode": "safe",
            },
        )
        _, ml_ready_df = build_ecommerce_output_datasets(cleaned_df)
        checks = validate_smartphone_outputs(cleaned_df, ml_ready_df)
        brand_check = next(
            (c for c in checks if c["check"] == "invalid_brand_columns_removed_or_flagged"),
            None,
        )
        self.assertIsNotNone(brand_check, "invalid_brand_columns_removed_or_flagged check must exist")
        self.assertFalse(brand_check["passed"], "check must FAIL because brand_Achhe is in the ML-ready file")

    # --- Minimum test case 3: Vertu price parses as 650000 ---
    def test_vertu_signature_touch_price_parses_as_650000(self):
        preview = self._preprocess(self._make_corrupted_df())
        vertu_rows = preview[preview["model"].astype(str).str.contains("Vertu", case=False, na=False)]
        self.assertEqual(len(vertu_rows), 1)
        self.assertEqual(float(vertu_rows.iloc[0]["price"]), 650000.0)

    # --- Minimum test case 4: Rs.6,50,000 does not become NaN ---
    def test_rs_6_50_000_does_not_become_nan(self):
        preview = self._preprocess(self._make_corrupted_df())
        vertu_price = preview.loc[
            preview["model"].astype(str).str.contains("Vertu", case=False, na=False), "price"
        ]
        self.assertFalse(vertu_price.isna().any(), "Vertu price must not be NaN")

    # --- Minimum test case 5: small screen sizes are extracted ---
    def test_small_screen_sizes_are_extracted(self):
        for inch_value, display_text in [
            (1.77, "1.77 inches, 120x160px Display"),
            (1.8,  "1.8 inches, 260x240px Display"),
            (2.0,  "2 inches, 128x160px Display"),
            (2.4,  "2.4 inches, 320x240px Display"),
            (2.7,  "2.7 inches, 240x320px Display"),
            (2.8,  "2.8 inches, 240x320px Display"),
        ]:
            sample = pd.DataFrame(
                {
                    "model": ["TestPhone"],
                    "Price": ["Rs.999"],
                    "rating": [50],
                    "sim": ["Dual Sim"],
                    "processor": ["No"],
                    "ram": ["4MB RAM"],
                    "battery": ["800mAh Battery"],
                    "display": [display_text],
                    "camera": ["No"],
                    "card": ["Memory Card Not Supported"],
                    "os": [None],
                    "Segment": [None],
                }
            )
            preview, _ = build_ecommerce_preprocessed_view(sample, drop_reference_columns=True)
            self.assertFalse(
                preview["screen_size_inches"].isna().any(),
                f"screen_size_inches must be extracted from '{display_text}'",
            )
            self.assertEqual(
                float(preview["screen_size_inches"].iloc[0]),
                inch_value,
                f"Expected {inch_value} from '{display_text}'",
            )

    # --- Minimum test case 6: safe mode preserves all rows ---
    def test_safe_mode_preserves_all_rows(self):
        preview = self._preprocess(self._make_corrupted_df())
        result_df, report = validate_smartphone_dataset_quality(preview, mode="safe")
        self.assertEqual(len(result_df), len(preview))
        self.assertEqual(report["rows_removed_in_strict_mode"], 0)

    # --- Minimum test case 7: strict mode removes Achhe Din Mobile ---
    def test_strict_mode_removes_achhe_din_mobile(self):
        preview = self._preprocess(self._make_corrupted_df())
        result_df, report = validate_smartphone_dataset_quality(preview, mode="strict")
        self.assertFalse(
            result_df["model"].astype(str).str.contains("Achhe Din Mobile", case=False, na=False).any(),
            "Achhe Din Mobile must be removed in strict mode",
        )
        self.assertGreater(report["rows_removed_in_strict_mode"], 0)

    # --- Minimum test case 8: cleaned file has no missing in key columns ---
    def test_cleaned_file_has_no_missing_in_key_columns(self):
        df = _smartphone_df()
        cleaned_df, _ = clean_dataset(
            df,
            {
                "remove_duplicates": True,
                "handle_missing_values": True,
                "fix_data_types": True,
                "handle_outliers": True,
                "encode_categorical": True,
                "scale_numeric": True,
                "scaler_choice": "StandardScaler",
                "nlp_cleaning": False,
            },
        )
        self.assertEqual(int(cleaned_df["price"].isna().sum()), 0)
        self.assertEqual(int(cleaned_df["screen_size_inches"].isna().sum()), 0)
        self.assertEqual(int(cleaned_df["price_scaled"].isna().sum()), 0)
        self.assertEqual(int(cleaned_df["screen_size_inches_scaled"].isna().sum()), 0)

    # --- Minimum test case 9: ML-ready file has 0 missing values ---
    def test_ml_ready_has_zero_missing_values(self):
        df = _smartphone_df()
        cleaned_df, _ = clean_dataset(
            df,
            {
                "remove_duplicates": False,
                "handle_missing_values": True,
                "fix_data_types": True,
                "handle_outliers": True,
                "encode_categorical": True,
                "scale_numeric": True,
                "scaler_choice": "StandardScaler",
                "nlp_cleaning": False,
            },
        )
        _, ml_ready_df = build_ecommerce_output_datasets(cleaned_df)
        self.assertEqual(int(ml_ready_df.isna().sum().sum()), 0)

    # --- Minimum test case 10: suspicious_records_details is in validation report ---
    def test_validation_report_includes_suspicious_records_details(self):
        preview = self._preprocess(self._make_corrupted_df())
        _, report = validate_smartphone_dataset_quality(preview, mode="safe")
        self.assertIn("suspicious_records_details", report)
        self.assertIsInstance(report["suspicious_records_details"], list)

    # --- Processor validation ---
    def test_processor_column_pollution_is_detected(self):
        preview = self._preprocess(self._make_corrupted_df())
        _, report = validate_smartphone_dataset_quality(preview, mode="safe")
        achhe_entry = next(
            (item for item in report["suspicious_records_details"] if item["model"] == "Achhe Din Mobile"),
            None,
        )
        self.assertIsNotNone(achhe_entry)
        processor_reasons = [r for r in achhe_entry["reasons"] if "processor column" in r.lower()]
        self.assertTrue(processor_reasons, "Processor pollution must be flagged for Achhe Din Mobile")

    # --- Column-shift scoring ---
    def test_critical_column_shift_is_reported_for_achhe(self):
        preview = self._preprocess(self._make_corrupted_df())
        _, report = validate_smartphone_dataset_quality(preview, mode="safe")
        achhe_entry = next(
            (item for item in report["suspicious_records_details"] if item["model"] == "Achhe Din Mobile"),
            None,
        )
        self.assertIsNotNone(achhe_entry)
        shift_reasons = [r for r in achhe_entry["reasons"] if "column shift" in r.lower()]
        self.assertTrue(shift_reasons, "Column shift must be flagged for Achhe Din Mobile")
        self.assertTrue(any("critical" in r.lower() for r in shift_reasons))

    # --- Feature-phone detection ---
    def test_feature_phone_like_records_are_flagged_as_warnings(self):
        preview = self._preprocess(self._make_corrupted_df())
        _, report = validate_smartphone_dataset_quality(preview, mode="safe")
        nokia_entry = next(
            (item for item in report["suspicious_records_details"] if "Nokia 105" in item["model"]),
            None,
        )
        self.assertIsNotNone(nokia_entry, "Nokia 105 must appear in suspicious records as a feature-phone-like entry")
        self.assertEqual(nokia_entry["severity"], "warning", "Feature-phone-like Nokia 105 must be warning, not critical")
        fp_reasons = [r for r in nokia_entry["reasons"] if "feature-phone" in r.lower()]
        self.assertTrue(fp_reasons, "Feature-phone-like reason must be reported for Nokia 105")

    # --- data_validity_report structure ---
    def test_data_validity_report_structure_is_present(self):
        preview = self._preprocess(self._make_corrupted_df())
        _, report = validate_smartphone_dataset_quality(preview, mode="safe")
        dvr = report.get("data_validity_report")
        self.assertIsNotNone(dvr, "data_validity_report must be present in quality report")
        for required_key in [
            "input_row_count",
            "final_row_count",
            "suspicious_records_count",
            "critical_records_count",
            "suspicious_records_details",
            "price_validation",
            "processor_validation",
            "display_validation",
            "camera_validation",
            "os_validation",
            "column_shift_validation",
            "feature_phone_validation",
            "strict_mode_enabled",
            "rows_removed_in_strict_mode",
            "final_dataset_status",
        ]:
            self.assertIn(required_key, dvr, f"data_validity_report must contain '{required_key}'")

    def test_data_validity_report_counts_are_accurate(self):
        preview = self._preprocess(self._make_corrupted_df())
        _, report = validate_smartphone_dataset_quality(preview, mode="safe")
        dvr = report["data_validity_report"]
        self.assertEqual(dvr["input_row_count"], len(preview))
        self.assertGreaterEqual(dvr["suspicious_records_count"], 1)
        self.assertGreaterEqual(dvr["critical_records_count"], 1)
        self.assertEqual(dvr["final_dataset_status"], "critical_issues_present")
        self.assertGreater(dvr["column_shift_validation"]["rows_with_critical_column_shift"], 0)
        self.assertGreater(dvr["processor_validation"]["rows_with_processor_pollution"], 0)
        self.assertGreater(dvr["feature_phone_validation"]["feature_phone_like_rows"], 0)
        self.assertGreater(dvr["os_validation"]["rows_with_missing_os"], 0)

    def test_data_validity_report_strict_mode_counts_rows_removed(self):
        preview = self._preprocess(self._make_corrupted_df())
        _, report = validate_smartphone_dataset_quality(preview, mode="strict")
        dvr = report["data_validity_report"]
        self.assertTrue(dvr["strict_mode_enabled"])
        self.assertGreater(dvr["rows_removed_in_strict_mode"], 0)
        self.assertLess(dvr["final_row_count"], dvr["input_row_count"])

    def test_clean_row_has_no_suspicious_flags(self):
        preview = self._preprocess(self._make_corrupted_df())
        _, report = validate_smartphone_dataset_quality(preview, mode="safe")
        samsung_entries = [
            item for item in report["suspicious_records_details"]
            if "Samsung Galaxy S23" in item["model"]
        ]
        # Samsung Galaxy S23 is a clean well-known brand — it may have an unknown-brand
        # warning but must not be critical and must not have column-shift or processor flags.
        for entry in samsung_entries:
            self.assertNotEqual(entry["severity"], "critical")
            critical_reasons = [r for r in entry["reasons"] if "column shift" in r.lower() or "processor column" in r.lower()]
            self.assertEqual(critical_reasons, [])


if __name__ == "__main__":
    unittest.main()
