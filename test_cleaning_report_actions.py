"""Tests for structured cleaning action reporting in JSON output."""

from __future__ import annotations

import unittest

import pandas as pd

from utils.data_cleaner import clean_dataset
from utils.data_profiler import profile_dataset
from utils.ml_recommender import recommend_ml_approach
from utils.report_generator import generate_cleaning_report


class TestCleaningReportActions(unittest.TestCase):
    def test_report_clearly_marks_performed_and_skipped_actions(self):
        dataframe = pd.DataFrame(
            {
                "category": ["a", "b", "a", "b"],
                "numeric_value": [1.0, 2.0, 3.0, 4.0],
                "text_note": ["short", "short", "short", "short"],
                "target": [0, 1, 0, 1],
            }
        )

        cleaning_options = {
            "remove_duplicates": True,
            "handle_missing_values": True,
            "fix_data_types": False,
            "handle_outliers": True,
            "encode_categorical": True,
            "scale_numeric": True,
            "scaler_choice": "StandardScaler",
            "nlp_cleaning": True,
        }

        cleaned_df, cleaning_summary = clean_dataset(
            dataframe,
            cleaning_options,
            target_column="target",
        )
        profile = profile_dataset(dataframe, target_column="target")
        recommendation = recommend_ml_approach(
            dataframe,
            target_column="target",
            problem_type="Auto-detect",
            text_columns=profile["text_columns"],
        )
        report, _ = generate_cleaning_report(
            profile,
            {"errors": [], "warnings": [], "is_valid": True},
            cleaning_summary,
            recommendation,
            "sample.csv",
        )

        actions = report["cleaning_actions"]

        self.assertTrue(actions["duplicates_removed"]["selected"])
        self.assertFalse(actions["duplicates_removed"]["performed"])

        self.assertTrue(actions["missing_values_handled"]["selected"])
        self.assertFalse(actions["missing_values_handled"]["performed"])

        self.assertTrue(actions["outliers_detected_and_handled"]["selected"])
        self.assertFalse(actions["outliers_detected_and_handled"]["performed"])

        self.assertTrue(actions["categorical_columns_encoded"]["selected"])
        self.assertTrue(actions["categorical_columns_encoded"]["performed"])
        self.assertEqual(actions["categorical_columns_encoded"]["columns_encoded"], ["category"])

        self.assertTrue(actions["numeric_columns_scaled"]["selected"])
        self.assertTrue(actions["numeric_columns_scaled"]["performed"])
        self.assertEqual(actions["numeric_columns_scaled"]["scaler_used"], "StandardScaler")

        self.assertTrue(actions["nlp_text_cleaning_applied"]["selected"])
        self.assertFalse(actions["nlp_text_cleaning_applied"]["performed"])

        self.assertIn("skipped_steps", report)
        self.assertIn("before_vs_after_summary", report)
        self.assertIn("algorithm_recommendation", report)
        self.assertEqual(report["recommended_ml_problem_type"], "Classification")
        self.assertEqual(report["scaler_used"], "StandardScaler")
        self.assertEqual(
            report["algorithm_recommendation"]["beginner_friendly_first_choice"]["name"],
            "Logistic Regression",
        )
        self.assertEqual(len(cleaned_df), len(dataframe))


if __name__ == "__main__":
    unittest.main()
