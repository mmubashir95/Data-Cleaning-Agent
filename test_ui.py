"""UI-only tests for app.py — no cleaning logic exercised."""

import unittest
from streamlit.testing.v1 import AppTest


class TestSidebar(unittest.TestCase):
    def setUp(self):
        self.at = AppTest.from_file("app.py").run()

    # ── file uploader ──────────────────────────────────────────────────────────
    def test_file_uploader_present(self):
        uploaders = self.at.sidebar.file_uploader
        self.assertEqual(len(uploaders), 1)
        self.assertEqual(uploaders[0].label, "Upload a dataset")

    def test_file_uploader_accepts_csv_and_excel(self):
        # AppTest exposes accepted extensions via .allowed_type (with leading dots)
        accepted = self.at.sidebar.file_uploader[0].allowed_type
        self.assertIn(".csv", accepted)
        self.assertIn(".xlsx", accepted)
        self.assertIn(".xls", accepted)

    # ── problem-type selectbox ─────────────────────────────────────────────────
    def test_problem_type_selector_present(self):
        selects = self.at.sidebar.selectbox
        self.assertEqual(len(selects), 1)
        self.assertEqual(selects[0].label, "Problem type")

    def test_problem_type_options(self):
        options = list(self.at.sidebar.selectbox[0].options)
        expected = [
            "Auto-detect",
            "Classification",
            "Regression",
            "Clustering",
            "NLP/Text Classification",
        ]
        self.assertEqual(options, expected)

    # ── cleaning-option checkboxes ─────────────────────────────────────────────
    def _checkbox_labels(self):
        return [cb.label for cb in self.at.sidebar.checkbox]

    def test_cleaning_checkboxes_count(self):
        self.assertEqual(len(self.at.sidebar.checkbox), 7)

    def test_remove_duplicates_checkbox(self):
        self.assertIn("Remove duplicates", self._checkbox_labels())

    def test_handle_missing_values_checkbox(self):
        self.assertIn("Handle missing values", self._checkbox_labels())

    def test_fix_wrong_data_types_checkbox(self):
        self.assertIn("Fix wrong data types", self._checkbox_labels())

    def test_encode_categorical_columns_checkbox(self):
        self.assertIn("Encode categorical columns", self._checkbox_labels())

    def test_scale_numeric_columns_checkbox(self):
        self.assertIn("Scale numeric columns", self._checkbox_labels())

    def test_handle_outliers_checkbox(self):
        self.assertIn("Handle outliers", self._checkbox_labels())

    def test_nlp_text_cleaning_checkbox(self):
        self.assertIn("NLP text cleaning", self._checkbox_labels())

    # ── scaler radio ───────────────────────────────────────────────────────────
    def test_scaler_radio_present(self):
        radios = self.at.sidebar.radio
        self.assertEqual(len(radios), 1)
        self.assertEqual(radios[0].label, "Choose a scaler")

    def test_scaler_radio_options(self):
        options = list(self.at.sidebar.radio[0].options)
        self.assertEqual(options, ["StandardScaler", "MinMaxScaler"])


class TestMainPage(unittest.TestCase):
    def setUp(self):
        self.at = AppTest.from_file("app.py").run()

    def test_page_title(self):
        titles = self.at.title
        self.assertEqual(len(titles), 1)
        self.assertEqual(
            titles[0].value,
            "Data Cleaning Agent for ML Dataset Preparation",
        )

    def test_upload_prompt_info_message(self):
        infos = self.at.info
        self.assertEqual(len(infos), 1)
        self.assertIn(
            "Upload a CSV or Excel dataset from the sidebar to get started.",
            infos[0].value,
        )


if __name__ == "__main__":
    unittest.main()
