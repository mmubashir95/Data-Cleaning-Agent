"""UI tests for the Flowise prompt selection behavior."""

from __future__ import annotations

import io
import unittest
from unittest.mock import patch

import pandas as pd
from streamlit.testing.v1 import AppTest

import app


def _sample_titanic_df() -> pd.DataFrame:
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
            "Age": [22, 38, 26, 35, 35],
            "Fare": [7.25, 71.2833, 7.925, 53.1, 8.05],
            "Embarked": ["S", "C", "S", "S", "S"],
        }
    )


def _to_csv_bytes(df: pd.DataFrame) -> bytes:
    buffer = io.BytesIO()
    df.to_csv(buffer, index=False)
    return buffer.getvalue()


def _upload_app() -> AppTest:
    at = AppTest.from_function(app.main).run()
    at.sidebar.file_uploader[0].upload(
        "titanic.csv",
        _to_csv_bytes(_sample_titanic_df()),
        "text/csv",
    ).run()
    return at


class TestFlowisePromptUI(unittest.TestCase):
    def test_predefined_prompt_hides_custom_textarea(self):
        at = _upload_app()
        at.selectbox(key="profile_flowise_prompt").set_value("Explain Cleaning Steps").run()

        labels = [element.label for element in at.text_area]
        self.assertNotIn("Custom AI Question", labels)

    def test_custom_prompt_shows_custom_textarea(self):
        at = _upload_app()
        at.selectbox(key="profile_flowise_prompt").set_value(app.CUSTOM_PROMPT_OPTION).run()

        labels = [element.label for element in at.text_area]
        self.assertIn("Custom AI Question", labels)

    def test_empty_custom_prompt_shows_validation_warning(self):
        at = _upload_app()
        at.selectbox(key="profile_flowise_prompt").set_value(app.CUSTOM_PROMPT_OPTION).run()
        at.text_area(key="profile_flowise_custom_question").set_value("").run()
        at.button(key="profile_ask_flowise").click().run()

        warning_bodies = [warning.value for warning in at.warning]
        self.assertIn("Please enter a custom AI question.", warning_bodies)

    def test_custom_prompt_is_sent_with_dataset_summary(self):
        at = _upload_app()

        with patch.object(
            app,
            "query_flowise_agent",
            return_value={
                "success": True,
                "answer": "Random Forest is suitable.",
                "raw_response": {"text": "Random Forest is suitable."},
            },
        ) as mocked_query:
            at.selectbox(key="profile_flowise_prompt").set_value(app.CUSTOM_PROMPT_OPTION).run()
            at.text_area(key="profile_flowise_custom_question").set_value(
                "Explain why Random Forest is suitable for Titanic dataset"
            ).run()
            at.button(key="profile_ask_flowise").click().run()

        self.assertTrue(mocked_query.called)
        sent_question = mocked_query.call_args.args[0]
        sent_preview = mocked_query.call_args.kwargs["file_summary"]

        self.assertIn("User Custom Question:", sent_question)
        self.assertIn("Explain why Random Forest is suitable for Titanic dataset", sent_question)
        self.assertIn("Dataset Summary:", sent_question)
        self.assertIn("PassengerId", sent_question)
        self.assertIn("PassengerId", sent_preview)
        self.assertNotIn("Explain why Random Forest is suitable for Titanic dataset", sent_preview)


if __name__ == "__main__":
    unittest.main()
