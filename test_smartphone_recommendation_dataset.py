"""Tests for the smartphone-specific recommendation preprocessing path."""

from __future__ import annotations

import unittest

import pandas as pd

from utils.data_cleaner import clean_dataset
from utils.ecommerce_preprocessing import build_ecommerce_output_datasets, build_ecommerce_preprocessed_view
from utils.ml_recommender import recommend_ml_approach
from utils.report_generator import generate_cleaning_report
from utils.smartphone_preprocessing import detect_smartphone_dataset, validate_smartphone_outputs


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
        self.assertTrue(all("bluetooth" not in column.lower() for column in ml_ready_df.columns if column.startswith("os_family_")))
        self.assertEqual(readable_df.columns[1], "model")

    def test_recommendation_and_report_switch_to_smartphone_explanation(self):
        df = _smartphone_df()
        recommendation = recommend_ml_approach(df, None, "Auto-detect", text_columns=[])
        self.assertEqual(recommendation["recommended_problem_type"], "Smartphone Content-Based Recommendation")
        self.assertTrue(recommendation["smartphone_dataset_detected"])
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


if __name__ == "__main__":
    unittest.main()
