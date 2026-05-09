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
        self.assertIn("pandas_numpy_usage", report)
        numpy_functions = {
            entry["function"] for entry in report["pandas_numpy_usage"]["numpy_functions"]
        }
        self.assertIn("np.mean()", numpy_functions)
        self.assertIn("np.std()", numpy_functions)
        self.assertEqual(len(cleaned_df), len(dataframe))

    def test_report_includes_only_relevant_pandas_numpy_usage_for_performed_actions(self):
        dataframe = pd.DataFrame(
            {
                "category": ["a", "a", "b", "b", "c", "c"],
                "numeric_value": [1.0, None, 2.0, 3.0, 4.0, 1000.0],
                "target": [0, 0, 1, 1, 1, 1],
            }
        )
        dataframe = pd.concat([dataframe, dataframe.iloc[[4]]], ignore_index=True)

        cleaning_options = {
            "remove_duplicates": True,
            "handle_missing_values": True,
            "fix_data_types": False,
            "handle_outliers": True,
            "encode_categorical": True,
            "scale_numeric": False,
            "scaler_choice": None,
            "nlp_cleaning": False,
        }

        _, cleaning_summary = clean_dataset(
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

        pandas_functions = {
            entry["function"] for entry in report["pandas_numpy_usage"]["pandas_functions"]
        }
        numpy_functions = {
            entry["function"] for entry in report["pandas_numpy_usage"]["numpy_functions"]
        }

        self.assertIn("fillna()", pandas_functions)
        self.assertIn("median()/mode()", pandas_functions)
        self.assertIn("duplicated().sum()", pandas_functions)
        self.assertIn("drop_duplicates()", pandas_functions)
        self.assertIn("get_dummies()", pandas_functions)
        self.assertIn("quantile()", pandas_functions)
        self.assertIn("np.median()", numpy_functions)


if __name__ == "__main__":
    unittest.main()
