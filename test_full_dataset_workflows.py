"""End-to-end dataset workflow checks for recommendation, cleaning, and reports."""

from __future__ import annotations

import io
import unittest
from pathlib import Path

import pandas as pd
from streamlit.testing.v1 import AppTest

from utils.data_cleaner import clean_dataset
from utils.data_profiler import profile_dataset
from utils.ml_recommender import recommend_ml_approach
from utils.report_generator import generate_cleaning_report


def _to_csv_bytes(df: pd.DataFrame) -> bytes:
    buffer = io.BytesIO()
    df.to_csv(buffer, index=False)
    return buffer.getvalue()


def _upload(filename: str, df: pd.DataFrame) -> AppTest:
    app = AppTest.from_file("app.py").run()
    app.sidebar.file_uploader[0].upload(filename, _to_csv_bytes(df), "text/csv").run()
    return app


def _titanic_df() -> pd.DataFrame:
    n = 50
    return pd.DataFrame(
        {
            "PassengerId": range(1, n + 1),
            "Survived": [0, 1, 1, 1, 0] * 10,
            "Pclass": [3, 1, 3, 1, 2] * 10,
            "Name": [f"Smith Mr. John {i}" for i in range(n)],
            "Sex": ["male", "female"] * 25,
            "Age": [22.0, 38.0, None, 35.0, 27.0] * 10,
            "SibSp": [1, 0, 0, 1, 0] * 10,
            "Parch": [0, 0, 0, 0, 2] * 10,
            "Ticket": [f"T{i:04d}" for i in range(n)],
            "Fare": [7.25, 71.28, 7.93, 53.10, 11.13] * 10,
            "Cabin": [None, "C85", None, "C123", None] * 10,
            "Embarked": ["S", "C", "S", "S", "Q"] * 10,
        }
    )


def _house_prices_df() -> pd.DataFrame:
    n = 50
    return pd.DataFrame(
        {
            "Id": range(1, n + 1),
            "MSSubClass": [60, 20, 70, 60, 50] * 10,
            "MSZoning": ["RL", "RM", "C(all)", "FV", "RH"] * 10,
            "LotArea": [8450, 9600, 11250, 9550, 14260] * 10,
            "Street": ["Pave", "Grvl"] * 25,
            "BldgType": ["1Fam", "2FmCon", "Duplx", "TwnhsE", "Twnhs"] * 10,
            "OverallQual": [7, 6, 7, 7, 8] * 10,
            "OverallCond": [5, 8, 5, 5, 5] * 10,
            "YearBuilt": [2003, 1976, 2001, 1915, 2000] * 10,
            "SalePrice": [208500, 181500, 223500, 140000, 250000] * 10,
        }
    )


def _spam_sms_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "label": ["ham", "spam"] * 20,
            "message": [
                (
                    f"Message {i}: Congratulations! You have won a free prize, call now."
                    if i % 3 == 0
                    else f"Message {i}: Hey, just checking our weekend plans and timing."
                )
                for i in range(40)
            ],
        }
    )


def _customer_segmentation_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "CustomerID": range(1, 21),
            "Age": [20, 21, 22, 23, 24] * 4,
            "AnnualIncome": [30, 40, 50, 60, 70] * 4,
            "SpendingScore": [15, 35, 55, 75, 95] * 4,
            "Region": ["North", "South", "East", "West", "North"] * 4,
        }
    )


def _text_no_target_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "message": [
                f"Customer message {i} about service quality, billing confusion, and delayed delivery."
                for i in range(20)
            ],
            "channel": ["email", "chat"] * 10,
        }
    )


class TestFullDatasetWorkflows(unittest.TestCase):
    def test_titanic_recommended_as_classification(self):
        df = _titanic_df()
        profile = profile_dataset(df, target_column="Survived")
        recommendation = recommend_ml_approach(
            df, "Survived", "Auto-detect", profile["text_columns"]
        )
        self.assertEqual(recommendation["recommended_problem_type"], "Classification")

    def test_house_prices_recommended_as_regression(self):
        df = _house_prices_df()
        profile = profile_dataset(df, target_column="SalePrice")
        recommendation = recommend_ml_approach(
            df, "SalePrice", "Auto-detect", profile["text_columns"]
        )
        self.assertEqual(recommendation["recommended_problem_type"], "Regression")

    def test_spam_sms_recommended_as_nlp_classification(self):
        df = _spam_sms_df()
        profile = profile_dataset(df, target_column="label")
        recommendation = recommend_ml_approach(
            df, "label", "Auto-detect", profile["text_columns"]
        )
        self.assertEqual(recommendation["recommended_problem_type"], "NLP/Text Classification")
        self.assertEqual(recommendation["detected_text_column"], "message")

    def test_customer_segmentation_without_target_defaults_to_clustering(self):
        df = _customer_segmentation_df()
        profile = profile_dataset(df)
        recommendation = recommend_ml_approach(df, None, "Auto-detect", profile["text_columns"])
        self.assertEqual(recommendation["recommended_problem_type"], "Clustering")

    def test_text_dataset_without_target_stays_clustering_not_nlp_classification(self):
        df = _text_no_target_df()
        profile = profile_dataset(df)
        recommendation = recommend_ml_approach(df, None, "Auto-detect", profile["text_columns"])
        self.assertEqual(recommendation["recommended_problem_type"], "Clustering")
        self.assertEqual(recommendation["detected_text_column"], "message")

    def test_nlp_cleaning_does_not_modify_label_column(self):
        df = _spam_sms_df()
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
                "nlp_cleaning": True,
            },
            target_column="label",
        )
        self.assertTrue(cleaned_df["label"].equals(df["label"]))
        self.assertEqual(summary["cleaned_text_columns"], ["message"])
        self.assertIn("message_original", cleaned_df.columns)

    def test_scaling_does_not_modify_target_column(self):
        df = _house_prices_df()
        original_target = df["SalePrice"].copy()
        cleaned_df, summary = clean_dataset(
            df,
            {
                "remove_duplicates": False,
                "handle_missing_values": False,
                "fix_data_types": False,
                "handle_outliers": False,
                "encode_categorical": False,
                "scale_numeric": True,
                "scaler_choice": "StandardScaler",
                "nlp_cleaning": False,
            },
            target_column="SalePrice",
        )
        self.assertTrue(cleaned_df["SalePrice"].equals(original_target))
        self.assertNotIn("SalePrice", summary["scaled_columns"])

    def test_encoding_does_not_modify_target_column(self):
        df = _spam_sms_df()
        original_target = df["label"].copy()
        cleaned_df, summary = clean_dataset(
            df,
            {
                "remove_duplicates": False,
                "handle_missing_values": False,
                "fix_data_types": False,
                "handle_outliers": False,
                "encode_categorical": True,
                "scale_numeric": False,
                "scaler_choice": None,
                "nlp_cleaning": False,
            },
            target_column="label",
        )
        self.assertTrue(cleaned_df["label"].equals(original_target))
        self.assertNotIn("label", summary["encoded_columns"])

    def test_report_includes_target_problem_and_output_fields(self):
        df = _spam_sms_df()
        profile = profile_dataset(df, target_column="label")
        recommendation = recommend_ml_approach(
            df, "label", "Auto-detect", profile["text_columns"]
        )
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
                "nlp_cleaning": True,
            },
            target_column="label",
        )
        report, report_path = generate_cleaning_report(
            profile,
            {"errors": [], "warnings": []},
            summary,
            recommendation,
            "spam_full_workflow.csv",
            cleaned_file_path="output/cleaned_spam_full_workflow.csv",
        )
        self.assertEqual(report["target_column"], "label")
        self.assertEqual(report["ml_recommendation"]["problem_type"], "NLP/Text Classification")
        self.assertEqual(report["ml_recommendation"]["detected_text_column"], "message")
        self.assertEqual(
            report["algorithm_recommendation"]["beginner_friendly_first_choice"]["name"],
            "Naive Bayes",
        )
        self.assertEqual(report["flowise_integration"]["flowise_called"], False)
        self.assertFalse(report["flowise_integration"]["full_dataset_sent_to_flowise"])
        self.assertIn("project_summary_for_viva", report)
        self.assertIn("spam_full_workflow", report["project_summary_for_viva"]["plain_text"])
        self.assertIn("output_files", report)
        self.assertTrue(Path(report_path).exists())
        self.assertTrue(cleaned_df["label"].equals(df["label"]))

    def test_app_generates_output_files_for_cleaning_run(self):
        df = _house_prices_df()
        cleaned_output = Path("output/cleaned_house_prices_full_app.csv")
        report_output = Path("reports/cleaning_report_house_prices_full_app.json")
        if cleaned_output.exists():
            cleaned_output.unlink()
        if report_output.exists():
            report_output.unlink()

        app = _upload("house_prices_full_app.csv", df)
        target_box = next(
            element for element in app.selectbox if element.label == "Select target column (optional)"
        )
        target_box.set_value("SalePrice").run()
        app.sidebar.checkbox[3].set_value(True).run()
        app.sidebar.checkbox[4].set_value(True).run()
        app.button[0].click().run()

        self.assertEqual(len(app.exception), 0)
        self.assertTrue(cleaned_output.exists())
        self.assertTrue(report_output.exists())
        self.assertIn("Pandas and NumPy Usage", [element.value for element in app.subheader])
        self.assertIn("Project Summary for Viva", [element.value for element in app.subheader])


if __name__ == "__main__":
    unittest.main()
