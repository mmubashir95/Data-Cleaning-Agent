"""UI tests for the Dataset Profile section."""

from __future__ import annotations

import io
import unittest

import pandas as pd
from streamlit.testing.v1 import AppTest


def _to_csv_bytes(df: pd.DataFrame) -> bytes:
    buffer = io.BytesIO()
    df.to_csv(buffer, index=False)
    return buffer.getvalue()


def _titanic_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "PassengerId": [1, 2, 3, 4, 5],
            "Survived": [0, 1, 1, 1, 0],
            "Pclass": [3, 1, 3, 1, 3],
            "Name": [
                "Braund, Mr. Owen Harris",
                "Cumings, Mrs. John Bradley",
                "Heikkinen, Miss. Laina",
                "Futrelle, Mrs. Jacques Heath",
                "Allen, Mr. William Henry",
            ],
            "Sex": ["male", "female", "female", "female", "male"],
            "Age": [22.0, 38.0, None, 35.0, 35.0],
            "Cabin": [None, "C85", None, "C123", None],
            "Embarked": ["S", "C", "S", "S", None],
        }
    )


class TestDatasetProfileUI(unittest.TestCase):
    def test_dataset_profile_section_appears_before_cleaning(self):
        at = AppTest.from_file("app.py").run()
        at.sidebar.file_uploader[0].upload(
            "titanic.csv",
            _to_csv_bytes(_titanic_df()),
            "text/csv",
        ).run()

        labels = [subheader.value for subheader in at.subheader]
        self.assertIn("3. Dataset Profile", labels)
        self.assertIn("6. Cleaning Options", labels)
        self.assertLess(labels.index("3. Dataset Profile"), labels.index("6. Cleaning Options"))

    def test_dataset_profile_shows_profile_content(self):
        at = AppTest.from_file("app.py").run()
        at.sidebar.file_uploader[0].upload(
            "titanic.csv",
            _to_csv_bytes(_titanic_df()),
            "text/csv",
        ).run()

        page_text = " ".join(
            [element.value for element in at.markdown]
            + [element.value for element in at.info]
            + [element.value for element in at.success]
        )
        self.assertIn("Some columns have missing values", page_text)


if __name__ == "__main__":
    unittest.main()
