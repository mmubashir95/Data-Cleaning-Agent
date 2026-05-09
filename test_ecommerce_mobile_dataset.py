"""Tests for the mobile e-commerce extension layer."""

from __future__ import annotations

import io
import unittest

import pandas as pd
from streamlit.testing.v1 import AppTest

from utils.data_cleaner import clean_dataset
from utils.data_profiler import profile_dataset
from utils.ecommerce_preprocessing import (
    build_ecommerce_output_datasets,
    build_ecommerce_preprocessed_view,
    clean_price_column,
    clean_rating_column,
    clean_review_count_column,
    detect_mobile_ecommerce_dataset,
    normalize_availability_column,
)
from utils.ml_recommender import recommend_ml_approach
from utils.report_generator import generate_cleaning_report


def _to_csv_bytes(df: pd.DataFrame) -> bytes:
    buffer = io.BytesIO()
    df.to_csv(buffer, index=False)
    return buffer.getvalue()


def _mobile_products_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "product_name": [
                "Samsung Galaxy A54",
                "Samsung Galaxy A54",
                "iPhone 13",
                "Xiaomi Redmi Note 12",
                "OnePlus 12R",
                "OnePlus 12R",
            ],
            "brand": ["Samsung", "samsung", "APPLE", "xiaomi", "OnePlus", "OnePlus"],
            "price": [
                "Rs. 129,999",
                "PKR 129,999",
                "PKR 224,999",
                "PKR 59,999",
                "PKR 159999",
                "PKR 159999",
            ],
            "rating": [
                "4.5 out of 5",
                "4.5 out of 5",
                "4.8 out of 5",
                "4.2/5",
                "No rating",
                "No rating",
            ],
            "review_count": [
                "1,245 reviews",
                "1245 reviews",
                "2,501 reviews",
                "845 reviews",
                "No reviews",
                "No reviews",
            ],
            "ram": ["8GB RAM", "8GB RAM", "4GB", "8/256GB", "16GB", "16GB"],
            "storage": ["128 GB", "128GB", "128 GB", "256 GB", "256 GB", "256 GB"],
            "battery": ["5000mAh", "5000 mAh", None, "5000mAh", "5500mAh", "5500mAh"],
            "camera": ["50MP + 12MP + 5MP", "50MP + 12MP + 5MP", "12MP dual", None, "50MP", "50MP"],
            "processor": ["Exynos 1380", "Exynos 1380", "A15 Bionic", "Snapdragon 685", "Snapdragon 8 Gen 2", "Snapdragon 8 Gen 2"],
            "screen_size": ["6.4 inch", "6.4 inches", "6.1 inch", "6.67 inch", '6.78"', '6.78"'],
            "availability": ["In stock", "Available", "In stock", "Out of stock", "In stock", "In stock"],
            "product_url": [
                "https://shop.example.com/p/samsung-galaxy-a54",
                "https://shop.example.com/p/samsung-galaxy-a54",
                "https://shop.example.com/p/apple-iphone-13",
                "https://shop.example.com/p/redmi-note-12",
                "https://shop.example.com/p/oneplus-12r",
                "https://shop.example.com/p/oneplus-12r",
            ],
        }
    )


def _generic_house_prices_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "LotArea": [8450, 9600, 11250, 9550, 14260],
            "OverallQual": [7, 6, 7, 7, 8],
            "YearBuilt": [2003, 1976, 2001, 1915, 2000],
            "SalePrice": [208500, 181500, 223500, 140000, 250000],
        }
    )


class TestEcommerceMobileDataset(unittest.TestCase):
    def test_scraped_numeric_helpers_parse_expected_values(self):
        price = clean_price_column(pd.Series(["Rs. 129,999", "PKR 85,000", "$799", "Out of stock"]))
        rating = clean_rating_column(pd.Series(["4.5 out of 5", "4.2/5", "No rating"]))
        reviews = clean_review_count_column(pd.Series(["1,245 reviews", "999 Reviews", "No reviews"]))

        self.assertEqual(price.tolist()[:3], [129999.0, 85000.0, 799.0])
        self.assertTrue(pd.isna(price.iloc[3]))
        self.assertEqual(rating.tolist()[:2], [4.5, 4.2])
        self.assertTrue(pd.isna(rating.iloc[2]))
        self.assertEqual(reviews.tolist()[:2], [1245.0, 999.0])
        self.assertTrue(pd.isna(reviews.iloc[2]))
        normalized_availability = normalize_availability_column(
            pd.Series(["in stock", "Available", "Out-of-stock", "Pre Order", "Limited Stock"])
        )
        self.assertEqual(
            normalized_availability.tolist(),
            ["In Stock", "In Stock", "Out of Stock", "Pre Order", "Low Stock"],
        )

    def test_ecommerce_dataset_gets_domain_specific_preprocessing(self):
        df = _mobile_products_df()
        self.assertTrue(detect_mobile_ecommerce_dataset(df.columns))

        preview_df, preview_metadata = build_ecommerce_preprocessed_view(df, drop_reference_columns=True)
        self.assertTrue(preview_metadata["ecommerce_preprocessing_applied"])
        self.assertIn("price", preview_metadata["cleaned_numeric_columns"])
        self.assertIn("ram_gb", preview_metadata["extracted_feature_columns"])
        self.assertIn("product_url", preview_metadata["dropped_reference_columns"])
        self.assertIn("availability", preview_metadata["normalized_categorical_columns"])
        self.assertTrue(pd.api.types.is_numeric_dtype(preview_df["price"]))
        self.assertIn("ram_gb", preview_df.columns)
        self.assertIn("storage_gb", preview_df.columns)
        self.assertNotIn("product_url", preview_df.columns)

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
            target_column=None,
        )
        profile = profile_dataset(df)
        recommendation = recommend_ml_approach(df, None, "Auto-detect", profile["text_columns"])
        report, _ = generate_cleaning_report(
            profile,
            {"errors": [], "warnings": []},
            summary,
            recommendation,
            "mobile_products.csv",
            cleaned_file_path="output/cleaned_mobile_products.csv",
        )

        self.assertTrue(summary["ecommerce_preprocessing_applied"])
        self.assertGreaterEqual(summary["duplicate_rows_removed"], 1)
        self.assertIn("ram_gb", cleaned_df.columns)
        self.assertIn("storage_gb", cleaned_df.columns)
        self.assertIn("battery_mah", cleaned_df.columns)
        self.assertIn("screen_size_inches", cleaned_df.columns)
        self.assertNotIn("product_url", cleaned_df.columns)
        self.assertIn("brand", summary["normalized_categorical_columns"])
        self.assertIn("availability", summary["normalized_categorical_columns"])
        self.assertIn("product_url", summary["dropped_reference_columns"])
        self.assertTrue(summary["original_numeric_columns_preserved"] or not summary["scaled_columns_created"])
        self.assertEqual(recommendation["recommended_problem_type"], "Recommendation / Ranking Readiness")
        self.assertTrue(recommendation["recommendation_ready"])
        self.assertTrue(report["ecommerce_preprocessing_applied"])
        self.assertTrue(report["recommendation_ready"])
        self.assertIn("ram_gb", report["extracted_feature_columns"])

        readable_df, ml_ready_df = build_ecommerce_output_datasets(cleaned_df)
        self.assertIn("brand", readable_df.columns)
        self.assertIn("availability", readable_df.columns)
        self.assertNotIn("product_url", readable_df.columns)
        self.assertTrue(all(not column.endswith("_scaled") or column in readable_df.columns for column in readable_df.columns))
        self.assertTrue(all(column.endswith("_scaled") or column.startswith("brand_") or column.startswith("availability_") for column in ml_ready_df.columns))

        if summary["scaled_columns_created"]:
            self.assertIn("price_scaled", summary["scaled_columns_created"])
            self.assertIn("price", cleaned_df.columns)
            self.assertIn("price_scaled", cleaned_df.columns)
        self.assertTrue(report["ecommerce_preprocessing"]["availability_normalized"])
        self.assertIn("readable_cleaned_csv_path", report["output_files"])
        self.assertIn("ml_ready_csv_path", report["output_files"])

    def test_generic_dataset_path_remains_unchanged(self):
        df = _generic_house_prices_df()
        cleaned_df, summary = clean_dataset(
            df,
            {
                "remove_duplicates": False,
                "handle_missing_values": False,
                "fix_data_types": False,
                "handle_outliers": False,
                "encode_categorical": False,
                "scale_numeric": False,
                "scaler_choice": None,
                "nlp_cleaning": False,
            },
            target_column="SalePrice",
        )

        self.assertFalse(summary["ecommerce_preprocessing_applied"])
        self.assertNotIn("ram_gb", cleaned_df.columns)
        self.assertEqual(cleaned_df.columns.tolist(), df.columns.tolist())

    def test_app_renders_ecommerce_flow_and_cleaning_results(self):
        df = _mobile_products_df()
        at = AppTest.from_file("app.py").run()
        at.sidebar.file_uploader[0].upload(
            "mobile_products.csv",
            _to_csv_bytes(df),
            "text/csv",
        ).run()
        at.sidebar.checkbox[0].set_value(True).run()
        at.sidebar.checkbox[1].set_value(True).run()
        at.sidebar.checkbox[2].set_value(True).run()
        at.sidebar.checkbox[3].set_value(True).run()
        at.sidebar.checkbox[5].set_value(True).run()
        at.sidebar.checkbox[6].set_value(True).run()
        at.button[1].click().run()

        subheaders = [element.value for element in at.subheader]
        page_text = " ".join(
            [element.value for element in at.info]
            + [element.value for element in at.success]
            + [element.value for element in at.warning]
        )

        self.assertEqual(len(at.exception), 0)
        self.assertIn("Dataset Visualizations", subheaders)
        self.assertIn("Cleaning Impact Summary", subheaders)
        self.assertIn("Pandas and NumPy Usage", subheaders)
        self.assertIn("Project Summary for Viva", subheaders)
        self.assertIn("future recommendation or ranking", page_text.lower())


if __name__ == "__main__":
    unittest.main()
