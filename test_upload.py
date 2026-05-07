"""Tests for the dataset upload and loading feature in app.py."""

import io
import unittest

import openpyxl
from streamlit.testing.v1 import AppTest

# ── fixture helpers ────────────────────────────────────────────────────────────

VALID_CSV_BYTES = (
    b"name,age,score\n"
    b"Alice,30,95.5\n"
    b"Bob,25,88.0\n"
    b"Carol,35,72.3\n"
    b"Dave,28,61.0\n"
    b"Eve,22,99.1\n"
    b"Frank,45,55.7\n"
)

EMPTY_CSV_BYTES = b"col1,col2\n"  # header only → dataframe.empty is True

# Binary content that pandas cannot decode as UTF-8, producing a friendly error.
INVALID_CSV_BYTES = b"\x00\x01\x02\x03\xff\xfe"


def _make_excel_bytes() -> bytes:
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.append(["name", "age", "score"])
    ws.append(["Alice", 30, 95.5])
    ws.append(["Bob", 25, 88.0])
    ws.append(["Carol", 35, 72.3])
    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


VALID_EXCEL_BYTES = _make_excel_bytes()

EXCEL_MIME = (
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)


# ── helper ─────────────────────────────────────────────────────────────────────

def _upload(filename: str, content: bytes, mime: str = "text/csv") -> AppTest:
    at = AppTest.from_file("app.py").run()
    at.sidebar.file_uploader[0].upload(filename, content, mime).run()
    return at


def _shape_markdown(at: AppTest) -> str | None:
    for elem in at.markdown:
        if "Dataset shape:" in elem.value:
            return elem.value
    return None


# ── valid CSV ──────────────────────────────────────────────────────────────────

class TestValidCSV(unittest.TestCase):
    def setUp(self):
        self.at = _upload("sales.csv", VALID_CSV_BYTES)

    def test_no_error(self):
        self.assertEqual(len(self.at.error), 0)

    def test_filename_displayed(self):
        bodies = [e.body for e in self.at.success]
        self.assertTrue(
            any("sales.csv" in b for b in bodies),
            f"Expected filename in success messages; got {bodies}",
        )

    def test_shape_displayed(self):
        shape_text = _shape_markdown(self.at)
        self.assertIsNotNone(shape_text, "No 'Dataset shape:' markdown found")
        self.assertIn("6 rows", shape_text)
        self.assertIn("3 columns", shape_text)

    def test_first_five_rows_displayed(self):
        self.assertGreaterEqual(len(self.at.dataframe), 1)
        df = self.at.dataframe[0].value
        self.assertLessEqual(len(df), 5)
        self.assertEqual(len(df), 5)

    def test_expected_columns(self):
        df = self.at.dataframe[0].value
        self.assertListEqual(list(df.columns), ["name", "age", "score"])


# ── valid Excel ────────────────────────────────────────────────────────────────

class TestValidExcel(unittest.TestCase):
    def setUp(self):
        self.at = _upload("report.xlsx", VALID_EXCEL_BYTES, EXCEL_MIME)

    def test_no_error(self):
        self.assertEqual(len(self.at.error), 0)

    def test_filename_displayed(self):
        bodies = [e.body for e in self.at.success]
        self.assertTrue(
            any("report.xlsx" in b for b in bodies),
            f"Expected filename in success messages; got {bodies}",
        )

    def test_shape_displayed(self):
        shape_text = _shape_markdown(self.at)
        self.assertIsNotNone(shape_text, "No 'Dataset shape:' markdown found")
        self.assertIn("3 rows", shape_text)
        self.assertIn("3 columns", shape_text)

    def test_first_five_rows_displayed(self):
        self.assertGreaterEqual(len(self.at.dataframe), 1)
        df = self.at.dataframe[0].value
        self.assertLessEqual(len(df), 5)

    def test_expected_columns(self):
        df = self.at.dataframe[0].value
        self.assertListEqual(list(df.columns), ["name", "age", "score"])


# ── empty CSV ──────────────────────────────────────────────────────────────────

class TestEmptyCSV(unittest.TestCase):
    def setUp(self):
        self.at = _upload("empty.csv", EMPTY_CSV_BYTES)

    def test_no_crash(self):
        self.assertEqual(len(self.at.exception), 0)

    def test_no_hard_error(self):
        self.assertEqual(len(self.at.error), 0)

    def test_empty_warning_shown(self):
        bodies = [e.body for e in self.at.warning]
        self.assertTrue(
            any("empty" in b.lower() for b in bodies),
            f"Expected empty-dataset warning; got {bodies}",
        )

    def test_no_dataframe_rendered(self):
        self.assertEqual(len(self.at.dataframe), 0)


# ── invalid CSV (binary content) ───────────────────────────────────────────────

class TestInvalidCSV(unittest.TestCase):
    def setUp(self):
        self.at = _upload("corrupt.csv", INVALID_CSV_BYTES)

    def test_no_crash(self):
        self.assertEqual(len(self.at.exception), 0)

    def test_friendly_error_shown(self):
        bodies = [e.body for e in self.at.error]
        self.assertTrue(
            any("could not be read as text" in b.lower() for b in bodies),
            f"Expected friendly error message; got {bodies}",
        )

    def test_no_dataframe_rendered(self):
        self.assertEqual(len(self.at.dataframe), 0)

    def test_no_success_message(self):
        self.assertEqual(len(self.at.success), 0)


if __name__ == "__main__":
    unittest.main()
