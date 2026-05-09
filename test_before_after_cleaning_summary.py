"""Tests for before-vs-after cleaning summary metrics."""

from __future__ import annotations

import unittest

import pandas as pd

from utils.data_cleaner import clean_dataset
from utils.data_profiler import profile_dataset
from utils.ml_recommender import recommend_ml_approach
from utils.report_generator import generate_cleaning_report


class TestBeforeAfterCleaningSummary(unittest.TestCase):
    def test_summary_metrics_and_report_payload_are_created(self):
        dataframe = pd.DataFrame(
            {
                "category": ["a", "a", "b", "b"],
                "value": [1.0, 1.0, None, 4.0],
                "target": [0, 0, 1, 1],
            }
        )
        dataframe = pd.concat([dataframe, dataframe.iloc[[0]]], ignore_index=True)

        cleaned_df, cleaning_summary = clean_dataset(
            dataframe,
            {
                "remove_duplicates": True,
                "handle_missing_values": True,
                "fix_data_types": False,
                "handle_outliers": False,
                "encode_categorical": True,
                "scale_numeric": False,
                "scaler_choice": None,
                "nlp_cleaning": False,
            },
            target_column="target",
        )

        metrics = cleaning_summary["before_vs_after_summary"]["metrics"]
        metric_map = {item["metric"]: item for item in metrics}

        self.assertEqual(metric_map["Row count"]["before"], 5)
        self.assertEqual(metric_map["Row count"]["after"], len(cleaned_df))
        self.assertEqual(metric_map["Total missing values"]["before"], 1)
        self.assertEqual(metric_map["Total missing values"]["after"], 0)
        self.assertEqual(metric_map["Duplicate rows"]["before"], 1)
        self.assertEqual(metric_map["Duplicate rows"]["after"], 0)

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

        self.assertIn("before_vs_after_summary", report)
        self.assertEqual(
            report["before_vs_after_summary"]["metrics"][0]["metric"],
            "Row count",
        )


if __name__ == "__main__":
    unittest.main()
