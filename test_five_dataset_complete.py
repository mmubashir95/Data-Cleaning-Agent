"""Comprehensive end-to-end tests verifying all 13 requirements across 5 datasets.

Datasets exercised:
  1. Titanic           — Classification with numeric/text/categorical columns and missing values
  2. House Prices      — Regression with a continuous numeric target
  3. Spam SMS          — NLP/Text Classification with a text feature and categorical label
  4. Customer Segment  — Clustering (no target column provided)
  5. Text No Target    — Text columns present but no target → must NOT become NLP Classification

Requirements verified per dataset:
  R1.  Upload succeeds (success message, no exception)
  R2.  Pre-cleaning validation subheader appears before profiling subheader
  R3.  Dataset Profiling subheader appears only after validation passes
  R4.  Cleaning runs only after validation passes (Clean button produces a success message)
  R5.  Cleaned CSV is written to disk
  R6.  JSON cleaning report is written to disk
  R7.  Paths for cleaned CSV and report are shown in UI (proxy for download availability)
  R8.  ML problem type is correct for the dataset
  R9.  Recommended algorithm names are correct for the problem type
  R10. Profiling output does not display unsupported statistics (skewness, kurtosis, correlation …)
  R11. Target column is not modified by scaling or encoding
  R12. Text dataset with no target → Clustering, not NLP Classification (Dataset 5 only)
  R13. Scaler radio button appears only when "Scale numeric columns" is checked
"""

from __future__ import annotations

import io
import unittest
from pathlib import Path

import pandas as pd
from streamlit.testing.v1 import AppTest

from utils.data_cleaner import clean_dataset
from utils.data_profiler import profile_dataset
from utils.ml_recommender import recommend_ml_approach

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_UNSUPPORTED_STAT_TERMS = (
    "skewness",
    "kurtosis",
    "correlation",
    "percentile",
    "variance",
    "covariance",
    "standard deviation",
)


def _to_csv_bytes(df: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    df.to_csv(buf, index=False)
    return buf.getvalue()


def _upload(filename: str, df: pd.DataFrame) -> AppTest:
    app = AppTest.from_file("app.py").run()
    app.sidebar.file_uploader[0].upload(filename, _to_csv_bytes(df), "text/csv").run()
    return app


def _set_target(app: AppTest, target: str) -> AppTest:
    box = next(e for e in app.selectbox if e.label == "Select target column (optional)")
    return box.set_value(target).run()


def _click_clean(app: AppTest) -> AppTest:
    return app.button[0].click().run()


def _subheader_values(app: AppTest) -> list[str]:
    return [s.value for s in app.subheader]


def _markdown_values(app: AppTest) -> list[str]:
    return [m.value for m in app.markdown]


def _success_values(app: AppTest) -> list[str]:
    return [s.value for s in app.success]


def _warning_values(app: AppTest) -> list[str]:
    return [w.value for w in app.warning]


# ---------------------------------------------------------------------------
# Dataset factories
# ---------------------------------------------------------------------------

def _titanic_df() -> pd.DataFrame:
    n = 60
    return pd.DataFrame({
        "PassengerId": range(1, n + 1),
        "Survived":    [0, 1, 1, 0, 1, 0] * 10,
        "Pclass":      [3, 1, 2, 3, 1, 2] * 10,
        "Name":        [f"Doe Mr. Test {i}" for i in range(n)],
        "Sex":         ["male", "female"] * 30,
        "Age":         [22.0, 38.0, None, 35.0, 27.0, None] * 10,
        "SibSp":       [1, 0, 0, 1, 0, 1] * 10,
        "Parch":       [0, 0, 0, 0, 2, 1] * 10,
        "Ticket":      [f"T{i:04d}" for i in range(n)],
        "Fare":        [7.25, 71.28, 7.93, 53.10, 11.13, 30.0] * 10,
        "Cabin":       [None, "C85", None, "C123", None, "B20"] * 10,
        "Embarked":    ["S", "C", "S", "S", "Q", "C"] * 10,
    })


def _house_prices_df() -> pd.DataFrame:
    n = 60
    return pd.DataFrame({
        "Id":           range(1, n + 1),
        "MSSubClass":   [60, 20, 70, 60, 50, 120] * 10,
        "MSZoning":     ["RL", "RM", "C(all)", "FV", "RH", "RL"] * 10,
        "LotArea":      [8450, 9600, 11250, 9550, 14260, 7200] * 10,
        "Street":       ["Pave", "Grvl"] * 30,
        "BldgType":     ["1Fam", "2FmCon", "Duplx", "TwnhsE", "Twnhs", "1Fam"] * 10,
        "OverallQual":  [7, 6, 7, 7, 8, 5] * 10,
        "OverallCond":  [5, 8, 5, 5, 5, 6] * 10,
        "YearBuilt":    [2003, 1976, 2001, 1915, 2000, 1990] * 10,
        "SalePrice":    [208500, 181500, 223500, 140000, 250000, 195000] * 10,
    })


def _spam_sms_df() -> pd.DataFrame:
    return pd.DataFrame({
        "label": ["ham", "spam"] * 25,
        "message": [
            (
                f"Congratulations! You have won a free prize, call now to claim #{i}!"
                if i % 3 == 0
                else f"Hey, are we still meeting this weekend? Just checking the plan for {i}."
            )
            for i in range(50)
        ],
    })


def _customer_segmentation_df() -> pd.DataFrame:
    return pd.DataFrame({
        "CustomerID":    range(1, 31),
        "Age":           [20, 21, 22, 23, 24, 25] * 5,
        "AnnualIncome":  [30, 40, 50, 60, 70, 80] * 5,
        "SpendingScore": [15, 35, 55, 75, 95, 10] * 5,
        "Region":        ["North", "South", "East", "West", "North", "East"] * 5,
    })


def _text_no_target_df() -> pd.DataFrame:
    return pd.DataFrame({
        "message": [
            f"Customer complaint #{i}: billing confusion, delayed delivery, and unresponsive support."
            for i in range(25)
        ],
        "channel": ["email", "chat", "phone"] * 8 + ["email"],
    })


# ---------------------------------------------------------------------------
# Dataset 1 — Titanic (Classification)
# ---------------------------------------------------------------------------

class TestTitanicComplete(unittest.TestCase):
    FILENAME = "titanic_complete.csv"
    TARGET = "Survived"
    EXPECTED_PROBLEM_TYPE = "Classification"
    EXPECTED_ALGORITHMS = ["Logistic Regression", "Decision Tree", "Random Forest"]

    def setUp(self):
        self.df = _titanic_df()
        self.app = _upload(self.FILENAME, self.df)
        self.app = _set_target(self.app, self.TARGET)

    # R1 — upload works
    def test_r1_upload_succeeds(self):
        self.assertEqual(len(self.app.exception), 0)
        self.assertTrue(
            any(f"Uploaded file: {self.FILENAME}" in v for v in _success_values(self.app))
        )

    # R2 — validation before profiling
    def test_r2_validation_subheader_before_profiling(self):
        headers = _subheader_values(self.app)
        self.assertIn("Pre-Cleaning Validation", headers)
        self.assertIn("Dataset Profiling", headers)
        self.assertLess(
            headers.index("Pre-Cleaning Validation"),
            headers.index("Dataset Profiling"),
        )

    # R3 — profiling runs after validation passes
    def test_r3_profiling_subheader_visible(self):
        self.assertIn("Dataset Profiling", _subheader_values(self.app))
        self.assertTrue(
            any("Validation passed" in v for v in _success_values(self.app))
        )

    # R4 — cleaning runs after validation passes
    def test_r4_cleaning_produces_success_message(self):
        self.app.sidebar.checkbox[0].set_value(True).run()
        app = _click_clean(self.app)
        self.assertEqual(len(app.exception), 0)
        self.assertTrue(
            any("Cleaning completed successfully" in v for v in _success_values(app))
        )

    # R5 — cleaned CSV written to disk
    def test_r5_cleaned_csv_generated_on_disk(self):
        app = _click_clean(self.app)
        self.assertEqual(len(app.exception), 0)
        cleaned_path = Path("output") / f"cleaned_{Path(self.FILENAME).stem}.csv"
        self.assertTrue(cleaned_path.exists(), f"Missing: {cleaned_path}")

    # R6 — JSON report written to disk
    def test_r6_json_report_generated_on_disk(self):
        app = _click_clean(self.app)
        self.assertEqual(len(app.exception), 0)
        report_path = Path("reports") / f"cleaning_report_{Path(self.FILENAME).stem}.json"
        self.assertTrue(report_path.exists(), f"Missing: {report_path}")

    # R7 — file paths appear in UI (proxy for download buttons being ready)
    def test_r7_download_paths_shown_in_ui(self):
        app = _click_clean(self.app)
        markdowns = " ".join(_markdown_values(app))
        self.assertIn("Cleaned CSV saved to:", markdowns)
        self.assertIn("Cleaning report saved to:", markdowns)

    # R8 — correct problem type
    def test_r8_problem_type_is_classification(self):
        markdowns = _markdown_values(self.app)
        self.assertTrue(
            any(f"Problem Type: {self.EXPECTED_PROBLEM_TYPE}" in m for m in markdowns)
        )

    # R9 — correct algorithm recommendations
    def test_r9_correct_algorithms_recommended(self):
        markdowns = " ".join(_markdown_values(self.app))
        for algo in self.EXPECTED_ALGORITHMS:
            self.assertIn(algo, markdowns, f"Algorithm '{algo}' not found in recommendations")

    # R10 — no hallucinated unsupported statistics in profiling output
    def test_r10_no_hallucinated_statistics(self):
        markdowns = " ".join(m.lower() for m in _markdown_values(self.app))
        for term in _UNSUPPORTED_STAT_TERMS:
            self.assertNotIn(term, markdowns, f"Unsupported stat term '{term}' found in output")

    # R11 — target column not modified by scaling
    def test_r11_target_column_protected_from_scaling(self):
        original_target = self.df[self.TARGET].copy()
        cleaned_df, summary = clean_dataset(
            self.df,
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
            target_column=self.TARGET,
        )
        self.assertTrue(
            cleaned_df[self.TARGET].equals(original_target),
            "Target column was modified by scaling",
        )
        self.assertNotIn(self.TARGET, summary["scaled_columns"])

    # R13 — scaler radio hidden until scale checkbox checked
    def test_r13_scaler_radio_only_when_scale_checked(self):
        self.assertEqual(len(self.app.sidebar.radio), 0)
        self.app.sidebar.checkbox[4].set_value(True).run()
        self.assertEqual(len(self.app.sidebar.radio), 1)
        self.assertEqual(self.app.sidebar.radio[0].label, "Choose a scaler")


# ---------------------------------------------------------------------------
# Dataset 2 — House Prices (Regression)
# ---------------------------------------------------------------------------

class TestHousePricesComplete(unittest.TestCase):
    FILENAME = "house_prices_complete.csv"
    TARGET = "SalePrice"
    EXPECTED_PROBLEM_TYPE = "Regression"
    EXPECTED_ALGORITHMS = [
        "Linear Regression",
        "Random Forest Regressor",
        "Gradient Boosting Regressor",
    ]

    def setUp(self):
        self.df = _house_prices_df()
        self.app = _upload(self.FILENAME, self.df)
        self.app = _set_target(self.app, self.TARGET)

    def test_r1_upload_succeeds(self):
        self.assertEqual(len(self.app.exception), 0)
        self.assertTrue(
            any(f"Uploaded file: {self.FILENAME}" in v for v in _success_values(self.app))
        )

    def test_r2_validation_subheader_before_profiling(self):
        headers = _subheader_values(self.app)
        self.assertLess(
            headers.index("Pre-Cleaning Validation"),
            headers.index("Dataset Profiling"),
        )

    def test_r3_profiling_subheader_visible(self):
        self.assertIn("Dataset Profiling", _subheader_values(self.app))

    def test_r4_cleaning_produces_success_message(self):
        self.app.sidebar.checkbox[1].set_value(True).run()
        app = _click_clean(self.app)
        self.assertEqual(len(app.exception), 0)
        self.assertTrue(
            any("Cleaning completed successfully" in v for v in _success_values(app))
        )

    def test_r5_cleaned_csv_generated_on_disk(self):
        app = _click_clean(self.app)
        cleaned_path = Path("output") / f"cleaned_{Path(self.FILENAME).stem}.csv"
        self.assertTrue(cleaned_path.exists(), f"Missing: {cleaned_path}")

    def test_r6_json_report_generated_on_disk(self):
        app = _click_clean(self.app)
        report_path = Path("reports") / f"cleaning_report_{Path(self.FILENAME).stem}.json"
        self.assertTrue(report_path.exists(), f"Missing: {report_path}")

    def test_r7_download_paths_shown_in_ui(self):
        app = _click_clean(self.app)
        markdowns = " ".join(_markdown_values(app))
        self.assertIn("Cleaned CSV saved to:", markdowns)
        self.assertIn("Cleaning report saved to:", markdowns)

    def test_r8_problem_type_is_regression(self):
        markdowns = _markdown_values(self.app)
        self.assertTrue(
            any(f"Problem Type: {self.EXPECTED_PROBLEM_TYPE}" in m for m in markdowns)
        )

    def test_r9_correct_algorithms_recommended(self):
        markdowns = " ".join(_markdown_values(self.app))
        for algo in self.EXPECTED_ALGORITHMS:
            self.assertIn(algo, markdowns, f"Algorithm '{algo}' not found in recommendations")

    def test_r10_no_hallucinated_statistics(self):
        markdowns = " ".join(m.lower() for m in _markdown_values(self.app))
        for term in _UNSUPPORTED_STAT_TERMS:
            self.assertNotIn(term, markdowns, f"Unsupported stat term '{term}' found")

    def test_r11_target_column_protected_from_encoding(self):
        original_target = self.df[self.TARGET].copy()
        cleaned_df, summary = clean_dataset(
            self.df,
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
            target_column=self.TARGET,
        )
        self.assertIn(self.TARGET, cleaned_df.columns)
        self.assertTrue(cleaned_df[self.TARGET].equals(original_target))
        self.assertNotIn(self.TARGET, summary["encoded_columns"])

    def test_r13_scaler_radio_only_when_scale_checked(self):
        self.assertEqual(len(self.app.sidebar.radio), 0)
        self.app.sidebar.checkbox[4].set_value(True).run()
        self.assertEqual(len(self.app.sidebar.radio), 1)


# ---------------------------------------------------------------------------
# Dataset 3 — Spam SMS (NLP/Text Classification)
# ---------------------------------------------------------------------------

class TestSpamSMSComplete(unittest.TestCase):
    FILENAME = "spam_sms_complete.csv"
    TARGET = "label"
    EXPECTED_PROBLEM_TYPE = "NLP/Text Classification"
    EXPECTED_ALGORITHMS = ["Naive Bayes", "Logistic Regression", "Linear SVM"]

    def setUp(self):
        self.df = _spam_sms_df()
        self.app = _upload(self.FILENAME, self.df)
        self.app = _set_target(self.app, self.TARGET)

    def test_r1_upload_succeeds(self):
        self.assertEqual(len(self.app.exception), 0)
        self.assertTrue(
            any(f"Uploaded file: {self.FILENAME}" in v for v in _success_values(self.app))
        )

    def test_r2_validation_subheader_before_profiling(self):
        headers = _subheader_values(self.app)
        self.assertLess(
            headers.index("Pre-Cleaning Validation"),
            headers.index("Dataset Profiling"),
        )

    def test_r3_profiling_subheader_visible(self):
        self.assertIn("Dataset Profiling", _subheader_values(self.app))

    def test_r4_cleaning_produces_success_message(self):
        self.app.sidebar.checkbox[6].set_value(True).run()
        app = _click_clean(self.app)
        self.assertEqual(len(app.exception), 0)
        self.assertTrue(
            any("Cleaning completed successfully" in v for v in _success_values(app))
        )

    def test_r5_cleaned_csv_generated_on_disk(self):
        app = _click_clean(self.app)
        cleaned_path = Path("output") / f"cleaned_{Path(self.FILENAME).stem}.csv"
        self.assertTrue(cleaned_path.exists(), f"Missing: {cleaned_path}")

    def test_r6_json_report_generated_on_disk(self):
        app = _click_clean(self.app)
        report_path = Path("reports") / f"cleaning_report_{Path(self.FILENAME).stem}.json"
        self.assertTrue(report_path.exists(), f"Missing: {report_path}")

    def test_r7_download_paths_shown_in_ui(self):
        app = _click_clean(self.app)
        markdowns = " ".join(_markdown_values(app))
        self.assertIn("Cleaned CSV saved to:", markdowns)
        self.assertIn("Cleaning report saved to:", markdowns)

    def test_r8_problem_type_is_nlp_classification(self):
        markdowns = _markdown_values(self.app)
        self.assertTrue(
            any(f"Problem Type: {self.EXPECTED_PROBLEM_TYPE}" in m for m in markdowns)
        )

    def test_r9_correct_algorithms_recommended(self):
        markdowns = " ".join(_markdown_values(self.app))
        for algo in self.EXPECTED_ALGORITHMS:
            self.assertIn(algo, markdowns, f"Algorithm '{algo}' not found in recommendations")

    def test_r10_no_hallucinated_statistics(self):
        markdowns = " ".join(m.lower() for m in _markdown_values(self.app))
        for term in _UNSUPPORTED_STAT_TERMS:
            self.assertNotIn(term, markdowns, f"Unsupported stat term '{term}' found")

    # R11 — NLP cleaning must not touch the target label column
    def test_r11_target_column_protected_from_nlp_cleaning(self):
        original_label = self.df[self.TARGET].copy()
        cleaned_df, summary = clean_dataset(
            self.df,
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
            target_column=self.TARGET,
        )
        self.assertTrue(
            cleaned_df[self.TARGET].equals(original_label),
            "NLP cleaning modified the target label column",
        )
        self.assertNotIn(self.TARGET, summary["cleaned_text_columns"])

    def test_r13_scaler_radio_only_when_scale_checked(self):
        self.assertEqual(len(self.app.sidebar.radio), 0)
        self.app.sidebar.checkbox[4].set_value(True).run()
        self.assertEqual(len(self.app.sidebar.radio), 1)


# ---------------------------------------------------------------------------
# Dataset 4 — Customer Segmentation (Clustering, no target)
# ---------------------------------------------------------------------------

class TestCustomerSegmentationComplete(unittest.TestCase):
    FILENAME = "customer_segmentation_complete.csv"
    EXPECTED_PROBLEM_TYPE = "Clustering"
    EXPECTED_ALGORITHMS = ["K-Means", "DBSCAN"]

    def setUp(self):
        self.df = _customer_segmentation_df()
        # No target column selected — clustering path
        self.app = _upload(self.FILENAME, self.df)

    def test_r1_upload_succeeds(self):
        self.assertEqual(len(self.app.exception), 0)
        self.assertTrue(
            any(f"Uploaded file: {self.FILENAME}" in v for v in _success_values(self.app))
        )

    def test_r2_validation_subheader_before_profiling(self):
        headers = _subheader_values(self.app)
        self.assertLess(
            headers.index("Pre-Cleaning Validation"),
            headers.index("Dataset Profiling"),
        )

    def test_r3_profiling_subheader_visible(self):
        self.assertIn("Dataset Profiling", _subheader_values(self.app))

    def test_r4_cleaning_produces_success_message(self):
        self.app.sidebar.checkbox[0].set_value(True).run()
        app = _click_clean(self.app)
        self.assertEqual(len(app.exception), 0)
        self.assertTrue(
            any("Cleaning completed successfully" in v for v in _success_values(app))
        )

    def test_r5_cleaned_csv_generated_on_disk(self):
        app = _click_clean(self.app)
        cleaned_path = Path("output") / f"cleaned_{Path(self.FILENAME).stem}.csv"
        self.assertTrue(cleaned_path.exists(), f"Missing: {cleaned_path}")

    def test_r6_json_report_generated_on_disk(self):
        app = _click_clean(self.app)
        report_path = Path("reports") / f"cleaning_report_{Path(self.FILENAME).stem}.json"
        self.assertTrue(report_path.exists(), f"Missing: {report_path}")

    def test_r7_download_paths_shown_in_ui(self):
        app = _click_clean(self.app)
        markdowns = " ".join(_markdown_values(app))
        self.assertIn("Cleaned CSV saved to:", markdowns)
        self.assertIn("Cleaning report saved to:", markdowns)

    def test_r8_problem_type_is_clustering(self):
        markdowns = _markdown_values(self.app)
        self.assertTrue(
            any(f"Problem Type: {self.EXPECTED_PROBLEM_TYPE}" in m for m in markdowns)
        )

    def test_r9_correct_algorithms_recommended(self):
        markdowns = " ".join(_markdown_values(self.app))
        for algo in self.EXPECTED_ALGORITHMS:
            self.assertIn(algo, markdowns, f"Algorithm '{algo}' not found in recommendations")

    def test_r10_no_hallucinated_statistics(self):
        markdowns = " ".join(m.lower() for m in _markdown_values(self.app))
        for term in _UNSUPPORTED_STAT_TERMS:
            self.assertNotIn(term, markdowns, f"Unsupported stat term '{term}' found")

    def test_r11_target_column_not_applicable_no_target_selected(self):
        """With no target, all numeric columns are eligible for scaling — none should crash."""
        cleaned_df, summary = clean_dataset(
            self.df,
            {
                "remove_duplicates": False,
                "handle_missing_values": False,
                "fix_data_types": False,
                "handle_outliers": False,
                "encode_categorical": False,
                "scale_numeric": True,
                "scaler_choice": "MinMaxScaler",
                "nlp_cleaning": False,
            },
            target_column=None,
        )
        self.assertFalse(cleaned_df.empty)
        self.assertGreater(len(summary["scaled_columns"]), 0)

    def test_r13_scaler_radio_only_when_scale_checked(self):
        self.assertEqual(len(self.app.sidebar.radio), 0)
        self.app.sidebar.checkbox[4].set_value(True).run()
        self.assertEqual(len(self.app.sidebar.radio), 1)


# ---------------------------------------------------------------------------
# Dataset 5 — Text with no target (must stay Clustering, not NLP Classification)
# ---------------------------------------------------------------------------

class TestTextNoTargetComplete(unittest.TestCase):
    FILENAME = "text_no_target_complete.csv"
    EXPECTED_PROBLEM_TYPE = "Clustering"
    EXPECTED_ALGORITHMS = ["K-Means", "DBSCAN"]

    def setUp(self):
        self.df = _text_no_target_df()
        # No target selected — even though text columns exist, it must NOT be NLP Classification
        self.app = _upload(self.FILENAME, self.df)

    def test_r1_upload_succeeds(self):
        self.assertEqual(len(self.app.exception), 0)
        self.assertTrue(
            any(f"Uploaded file: {self.FILENAME}" in v for v in _success_values(self.app))
        )

    def test_r2_validation_subheader_before_profiling(self):
        headers = _subheader_values(self.app)
        self.assertLess(
            headers.index("Pre-Cleaning Validation"),
            headers.index("Dataset Profiling"),
        )

    def test_r3_profiling_subheader_visible(self):
        self.assertIn("Dataset Profiling", _subheader_values(self.app))

    def test_r4_cleaning_produces_success_message(self):
        self.app.sidebar.checkbox[6].set_value(True).run()
        app = _click_clean(self.app)
        self.assertEqual(len(app.exception), 0)
        self.assertTrue(
            any("Cleaning completed successfully" in v for v in _success_values(app))
        )

    def test_r5_cleaned_csv_generated_on_disk(self):
        app = _click_clean(self.app)
        cleaned_path = Path("output") / f"cleaned_{Path(self.FILENAME).stem}.csv"
        self.assertTrue(cleaned_path.exists(), f"Missing: {cleaned_path}")

    def test_r6_json_report_generated_on_disk(self):
        app = _click_clean(self.app)
        report_path = Path("reports") / f"cleaning_report_{Path(self.FILENAME).stem}.json"
        self.assertTrue(report_path.exists(), f"Missing: {report_path}")

    def test_r7_download_paths_shown_in_ui(self):
        app = _click_clean(self.app)
        markdowns = " ".join(_markdown_values(app))
        self.assertIn("Cleaned CSV saved to:", markdowns)
        self.assertIn("Cleaning report saved to:", markdowns)

    # R8 + R12 — problem type must be Clustering, NOT NLP Classification
    def test_r8_and_r12_problem_type_is_clustering_not_nlp_classification(self):
        markdowns = _markdown_values(self.app)
        self.assertTrue(
            any(f"Problem Type: {self.EXPECTED_PROBLEM_TYPE}" in m for m in markdowns),
            "Expected 'Clustering' but did not find it in the problem type output",
        )
        self.assertFalse(
            any("Problem Type: NLP/Text Classification" in m for m in markdowns),
            "Text-only dataset with no target was incorrectly classified as NLP/Text Classification",
        )

    def test_r9_correct_algorithms_recommended(self):
        markdowns = " ".join(_markdown_values(self.app))
        for algo in self.EXPECTED_ALGORITHMS:
            self.assertIn(algo, markdowns, f"Algorithm '{algo}' not found in recommendations")

    def test_r10_no_hallucinated_statistics(self):
        markdowns = " ".join(m.lower() for m in _markdown_values(self.app))
        for term in _UNSUPPORTED_STAT_TERMS:
            self.assertNotIn(term, markdowns, f"Unsupported stat term '{term}' found")

    def test_r11_nlp_cleaning_skips_target_safely_when_no_target(self):
        """NLP cleaning with no target must not crash and must clean the text column."""
        cleaned_df, summary = clean_dataset(
            self.df,
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
            target_column=None,
        )
        self.assertFalse(cleaned_df.empty)
        self.assertIn("message", summary["cleaned_text_columns"])

    # R12 — unit-level check via recommend_ml_approach directly
    def test_r12_unit_text_no_target_stays_clustering(self):
        profile = profile_dataset(self.df)
        recommendation = recommend_ml_approach(
            self.df, None, "Auto-detect", profile["text_columns"]
        )
        self.assertEqual(recommendation["recommended_problem_type"], "Clustering")
        # Detected text column should still be identified even when recommendation is Clustering
        self.assertIsNotNone(recommendation["detected_text_column"])

    def test_r13_scaler_radio_only_when_scale_checked(self):
        self.assertEqual(len(self.app.sidebar.radio), 0)
        self.app.sidebar.checkbox[4].set_value(True).run()
        self.assertEqual(len(self.app.sidebar.radio), 1)
        self.app.sidebar.checkbox[4].set_value(False).run()
        self.assertEqual(len(self.app.sidebar.radio), 0)


# ---------------------------------------------------------------------------
# Cross-dataset: scaler radio button toggle (dedicated class)
# ---------------------------------------------------------------------------

class TestScalerRadioToggle(unittest.TestCase):
    """R13 — radio button must appear only when the Scale numeric columns checkbox is on."""

    def test_radio_hidden_on_empty_sidebar(self):
        app = AppTest.from_file("app.py").run()
        self.assertEqual(len(app.sidebar.radio), 0)

    def test_radio_hidden_after_upload_before_scale_checked(self):
        df = _house_prices_df()
        app = _upload("hp_radio_test.csv", df)
        self.assertEqual(len(app.sidebar.radio), 0)

    def test_radio_visible_after_scale_checked(self):
        df = _house_prices_df()
        app = _upload("hp_radio_test.csv", df)
        app.sidebar.checkbox[4].set_value(True).run()
        self.assertEqual(len(app.sidebar.radio), 1)
        self.assertIn("StandardScaler", list(app.sidebar.radio[0].options))
        self.assertIn("MinMaxScaler", list(app.sidebar.radio[0].options))

    def test_radio_hidden_again_after_scale_unchecked(self):
        df = _house_prices_df()
        app = _upload("hp_radio_test.csv", df)
        app.sidebar.checkbox[4].set_value(True).run()
        app.sidebar.checkbox[4].set_value(False).run()
        self.assertEqual(len(app.sidebar.radio), 0)

    def test_scaler_choice_reflected_in_cleaning_summary(self):
        df = _house_prices_df()
        for scaler_name in ("StandardScaler", "MinMaxScaler"):
            cleaned_df, summary = clean_dataset(
                df,
                {
                    "remove_duplicates": False,
                    "handle_missing_values": False,
                    "fix_data_types": False,
                    "handle_outliers": False,
                    "encode_categorical": False,
                    "scale_numeric": True,
                    "scaler_choice": scaler_name,
                    "nlp_cleaning": False,
                },
                target_column="SalePrice",
            )
            self.assertEqual(summary["scaler_used"], scaler_name)
            self.assertGreater(len(summary["scaled_columns"]), 0)
            self.assertNotIn("SalePrice", summary["scaled_columns"])


if __name__ == "__main__":
    unittest.main()
