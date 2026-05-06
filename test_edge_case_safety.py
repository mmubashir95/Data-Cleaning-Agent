"""Edge-case safety checks for the Streamlit cleaning workflow."""

from __future__ import annotations

import io
import unittest

import pandas as pd
from streamlit.testing.v1 import AppTest


INVALID_CSV_BYTES = b"\x00\x01\x02\x03\xff\xfe"


def _to_csv_bytes(df: pd.DataFrame) -> bytes:
    buffer = io.BytesIO()
    df.to_csv(buffer, index=False)
    return buffer.getvalue()


def _upload_csv(filename: str, csv_bytes: bytes) -> AppTest:
    app = AppTest.from_file("app.py").run()
    app.sidebar.file_uploader[0].upload(filename, csv_bytes, "text/csv").run()
    return app


def _warning_texts(app: AppTest) -> list[str]:
    return [element.value for element in app.warning]


def _error_texts(app: AppTest) -> list[str]:
    return [element.value for element in app.error]


def _info_texts(app: AppTest) -> list[str]:
    return [element.value for element in app.info]


def _markdown_texts(app: AppTest) -> list[str]:
    return [element.value for element in app.markdown]


def _click_clean_button(app: AppTest) -> AppTest:
    return app.button[0].click().run()


class TestEdgeCaseSafety(unittest.TestCase):
    def test_no_file_uploaded(self):
        app = AppTest.from_file("app.py").run()

        self.assertEqual(len(app.exception), 0)
        self.assertTrue(
            any("No file uploaded yet" in message for message in _info_texts(app))
        )

    def test_empty_csv(self):
        app = _upload_csv("empty.csv", b"col1,col2\n")

        self.assertEqual(len(app.exception), 0)
        self.assertTrue(
            any("empty" in message.lower() for message in _warning_texts(app))
        )

    def test_invalid_csv(self):
        app = _upload_csv("invalid.csv", INVALID_CSV_BYTES)

        self.assertEqual(len(app.exception), 0)
        self.assertTrue(
            any(
                "could not be read as text" in message.lower()
                for message in _error_texts(app)
            )
        )

    def test_numeric_only_dataset(self):
        df = pd.DataFrame(
            {
                "age": [20, 25, 30, 35, 40],
                "income": [30000, 45000, 50000, 60000, 70000],
                "score": [1.0, 2.5, 3.0, 3.5, 4.0],
            }
        )
        app = _upload_csv("numeric_only.csv", _to_csv_bytes(df))

        self.assertEqual(len(app.exception), 0)
        self.assertIn("Dataset Profiling", [element.value for element in app.subheader])

    def test_text_only_dataset(self):
        df = pd.DataFrame(
            {
                "review": [
                    "This product worked very well for me.",
                    "Shipping was slow but the packaging was good.",
                    "Support responded quickly and solved the issue.",
                    "I would probably buy this again next month.",
                    "The color and finish matched the photos online.",
                ],
                "comment": [
                    "Helpful staff and clear instructions.",
                    "The box arrived slightly damaged.",
                    "Quality felt better than expected.",
                    "The manual was short but enough.",
                    "Setup was easy and quick.",
                ],
            }
        )
        app = _upload_csv("text_only.csv", _to_csv_bytes(df))

        self.assertEqual(len(app.exception), 0)
        self.assertIn("Dataset Profiling", [element.value for element in app.subheader])

    def test_dataset_with_no_target_selected(self):
        df = pd.DataFrame(
            {
                "age": [20, 25, 30, 35, 40],
                "city": ["A", "B", "C", "A", "B"],
            }
        )
        app = _upload_csv("no_target.csv", _to_csv_bytes(df))

        self.assertEqual(len(app.exception), 0)
        self.assertTrue(
            any("No target column selected" in message for message in _info_texts(app))
        )
        self.assertTrue(
            any("Problem Type: Clustering" in message for message in _markdown_texts(app))
        )

    def test_dataset_with_no_missing_values(self):
        df = pd.DataFrame(
            {
                "age": [20, 25, 30, 35, 40],
                "city": ["A", "B", "C", "A", "B"],
            }
        )
        app = _upload_csv("no_missing.csv", _to_csv_bytes(df))
        app.sidebar.checkbox[1].set_value(True).run()
        app = _click_clean_button(app)

        self.assertEqual(len(app.exception), 0)
        self.assertTrue(
            any(
                "no missing values were found" in message.lower()
                for message in _warning_texts(app)
            )
        )

    def test_dataset_with_no_categorical_columns(self):
        df = pd.DataFrame(
            {
                "x1": [1, 2, 3, 4, 5],
                "x2": [10, 20, 30, 40, 50],
                "x3": [100, 200, 300, 400, 500],
            }
        )
        app = _upload_csv("no_categorical.csv", _to_csv_bytes(df))
        app.sidebar.checkbox[3].set_value(True).run()
        app = _click_clean_button(app)

        self.assertEqual(len(app.exception), 0)
        self.assertTrue(
            any(
                "no categorical feature columns were found" in message.lower()
                for message in _warning_texts(app)
            )
        )

    def test_dataset_with_no_numeric_columns(self):
        df = pd.DataFrame(
            {
                "city": ["Lahore", "Karachi", "Islamabad", "Lahore", "Karachi"],
                "department": ["Sales", "HR", "IT", "IT", "Sales"],
            }
        )
        app = _upload_csv("no_numeric.csv", _to_csv_bytes(df))
        app.sidebar.checkbox[4].set_value(True).run()
        app = _click_clean_button(app)

        self.assertEqual(len(app.exception), 0)
        self.assertTrue(
            any(
                "no numeric feature columns were found" in message.lower()
                for message in _warning_texts(app)
            )
        )

    def test_nlp_cleaning_when_no_text_columns_exist(self):
        df = pd.DataFrame(
            {
                "age": [20, 25, 30, 35, 40],
                "income": [30000, 45000, 50000, 60000, 70000],
            }
        )
        app = _upload_csv("no_text.csv", _to_csv_bytes(df))
        app.sidebar.checkbox[6].set_value(True).run()
        app = _click_clean_button(app)

        self.assertEqual(len(app.exception), 0)
        self.assertTrue(
            any(
                "nlp cleaning was selected, but no text columns were found" in message.lower()
                for message in _warning_texts(app)
            )
        )

    def test_scaling_when_no_numeric_feature_columns_exist(self):
        df = pd.DataFrame({"target": [1, 0, 1, 0, 1]})
        app = _upload_csv("target_only.csv", _to_csv_bytes(df))
        target_selectbox = next(
            element for element in app.selectbox if element.label == "Select target column (optional)"
        )
        target_selectbox.set_value("target").run()
        app.sidebar.checkbox[4].set_value(True).run()
        app = _click_clean_button(app)

        self.assertEqual(len(app.exception), 0)
        self.assertTrue(
            any(
                "no numeric feature columns were found" in message.lower()
                for message in _warning_texts(app)
            )
        )

    def test_encoding_when_no_categorical_feature_columns_exist(self):
        df = pd.DataFrame({"target": ["yes", "no", "yes", "no", "yes"]})
        app = _upload_csv("categorical_target_only.csv", _to_csv_bytes(df))
        target_selectbox = next(
            element for element in app.selectbox if element.label == "Select target column (optional)"
        )
        target_selectbox.set_value("target").run()
        app.sidebar.checkbox[3].set_value(True).run()
        app = _click_clean_button(app)

        self.assertEqual(len(app.exception), 0)
        self.assertTrue(
            any(
                "no categorical feature columns were found" in message.lower()
                for message in _warning_texts(app)
            )
        )


if __name__ == "__main__":
    unittest.main()
