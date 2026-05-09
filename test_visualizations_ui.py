"""UI tests for the dataset visualization section."""

from __future__ import annotations

import io
import unittest

import pandas as pd
from streamlit.testing.v1 import AppTest

import app


def _to_csv_bytes(df: pd.DataFrame) -> bytes:
    buffer = io.BytesIO()
    df.to_csv(buffer, index=False)
    return buffer.getvalue()


def _upload_app(filename: str, df: pd.DataFrame) -> AppTest:
    at = AppTest.from_function(app.main).run()
    at.sidebar.file_uploader[0].upload(filename, _to_csv_bytes(df), "text/csv").run()
    return at


def _subheaders(at: AppTest) -> list[str]:
    return [element.value for element in at.subheader]


def _infos(at: AppTest) -> list[str]:
    return [element.value for element in at.info]


def _expander_labels(at: AppTest) -> list[str]:
    return [element.label for element in at.expander]


class TestDatasetVisualizationsUI(unittest.TestCase):
    def test_titanic_shows_visualization_section_and_expanders(self):
        df = pd.DataFrame(
            {
                "PassengerId": [1, 2, 3, 4, 5],
                "Survived": [0, 1, 1, 0, 1],
                "Pclass": [3, 1, 3, 2, 1],
                "Age": [22.0, 38.0, None, 35.0, 27.0],
                "Fare": [7.25, 71.28, 7.93, 53.10, 11.13],
                "Embarked": ["S", "C", "S", "Q", "S"],
            }
        )
        at = _upload_app("titanic.csv", df)
        target_box = next(
            element for element in at.selectbox if element.label == "Select target column (optional)"
        )
        at = target_box.set_value("Survived").run()

        self.assertEqual(len(at.exception), 0)
        self.assertIn("Dataset Visualizations", _subheaders(at))
        expander_labels = _expander_labels(at)
        self.assertIn("Missing Values Bar Chart", expander_labels)
        self.assertIn("Target/Class Distribution Chart", expander_labels)
        self.assertIn("Numeric Columns Boxplot", expander_labels)
        self.assertIn("Correlation Heatmap", expander_labels)

    def test_house_prices_keeps_numeric_visualizations_available(self):
        df = pd.DataFrame(
            {
                "LotArea": [8450, 9600, 11250, 9550, 14260],
                "OverallQual": [7, 6, 7, 7, 8],
                "YearBuilt": [2003, 1976, 2001, 1915, 2000],
                "SalePrice": [208500, 181500, 223500, 140000, 250000],
            }
        )
        at = _upload_app("house_prices.csv", df)
        target_box = next(
            element for element in at.selectbox if element.label == "Select target column (optional)"
        )
        at = target_box.set_value("SalePrice").run()

        info_messages = " ".join(_infos(at))
        self.assertEqual(len(at.exception), 0)
        self.assertNotIn("Skipped numeric boxplot", info_messages)
        self.assertNotIn("Skipped correlation heatmap", info_messages)

    def test_spam_sms_skips_numeric_visualizations_but_keeps_target_distribution(self):
        df = pd.DataFrame(
            {
                "label": ["ham", "spam", "ham", "spam"],
                "message": [
                    "Hey are we still meeting later tonight?",
                    "Win a free prize now by calling this number immediately",
                    "Please review the document I sent earlier today",
                    "Claim your free reward by replying yes right now",
                ],
            }
        )
        at = _upload_app("spam_sms.csv", df)
        target_box = next(
            element for element in at.selectbox if element.label == "Select target column (optional)"
        )
        at = target_box.set_value("label").run()

        info_messages = " ".join(_infos(at))
        self.assertEqual(len(at.exception), 0)
        self.assertNotIn("Skipped target distribution chart", info_messages)
        self.assertIn("Skipped numeric boxplot because this dataset has no numeric columns.", info_messages)
        self.assertIn("Skipped correlation heatmap because at least two numeric columns are required.", info_messages)

    def test_dataset_with_no_numeric_columns_skips_numeric_charts_safely(self):
        df = pd.DataFrame(
            {
                "city": ["Lahore", "Karachi", "Islamabad", "Lahore"],
                "department": ["Sales", "HR", "IT", "Finance"],
            }
        )
        at = _upload_app("text_only.csv", df)

        info_messages = " ".join(_infos(at))
        self.assertEqual(len(at.exception), 0)
        self.assertIn("Skipped numeric boxplot because this dataset has no numeric columns.", info_messages)
        self.assertIn("Skipped correlation heatmap because at least two numeric columns are required.", info_messages)

    def test_dataset_with_no_target_skips_target_distribution_safely(self):
        df = pd.DataFrame(
            {
                "Age": [21, 35, 45, 29, 39],
                "Income": [30, 80, 55, 42, 71],
                "SpendingScore": [81, 12, 55, 72, 34],
            }
        )
        at = _upload_app("segmentation.csv", df)

        info_messages = " ".join(_infos(at))
        self.assertEqual(len(at.exception), 0)
        self.assertIn("Skipped target distribution chart because no target column was selected.", info_messages)


if __name__ == "__main__":
    unittest.main()
