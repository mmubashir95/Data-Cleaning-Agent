"""Tests for the pre-cleaning validation step.

Unit tests call validate_dataset() directly (all 6 input cases).
UI tests use AppTest to verify what the app renders (items 1-6 from the spec).

Notes on cases that are unit-only:
- Duplicate column names: pandas auto-renames duplicates during CSV loading,
  so the app's load → validate flow can never receive a duplicate-column
  DataFrame from a CSV upload.  The validator logic is still tested directly.
- Non-existent target: the selectbox is populated from dataframe.columns,
  so the app UI prevents selecting a column that doesn't exist.  Tested
  directly against the validator.
"""

import io
import unittest

import pandas as pd
from streamlit.testing.v1 import AppTest

from utils.data_validator import validate_dataset

# ── CSV fixtures ───────────────────────────────────────────────────────────────

VALID_TITANIC_BYTES = (
    b"PassengerId,Survived,Pclass,Name,Sex,Age\n"
    b"1,0,3,Smith Mr. John,male,22.0\n"
    b"2,1,1,Jones Mrs. Alice,female,38.0\n"
    b"3,1,3,Brown Miss Lily,female,26.0\n"
    b"4,1,1,White Mr. James,male,35.0\n"
    b"5,0,3,Green Mr. Henry,male,27.0\n"
)

EMPTY_CSV_BYTES = b"col1,col2\n"                      # header only → df.empty

ONE_COLUMN_CSV_BYTES = b"age\n22\n38\n26\n"           # validator warning, not error

FULLY_MISSING_COLUMN_BYTES = b"name,score,blank\nAlice,1.0,\nBob,2.0,\n"


def _load(csv_bytes: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(csv_bytes))


def _duplicate_columns_df() -> pd.DataFrame:
    """Return a DataFrame with two columns both named 'col'.

    This cannot arise from pd.read_csv because pandas auto-renames duplicates,
    so this DataFrame is constructed manually to exercise the validator rule.
    """
    df = pd.DataFrame([[1, 2, 3]])
    df.columns = pd.Index(["col", "value", "col"])
    return df


# ── helpers ────────────────────────────────────────────────────────────────────

def _upload(filename: str, csv_bytes: bytes) -> AppTest:
    at = AppTest.from_file("app.py").run()
    at.sidebar.file_uploader[0].upload(filename, csv_bytes, "text/csv").run()
    return at


def _subheader_labels(at: AppTest) -> list[str]:
    return [sh.value for sh in at.subheader]


# ══════════════════════════════════════════════════════════════════════════════
# Unit tests — validate_dataset() called directly
# ══════════════════════════════════════════════════════════════════════════════

class TestValidateDatasetUnit(unittest.TestCase):

    # Case 1 — valid Titanic CSV ───────────────────────────────────────────────
    def test_valid_titanic_is_valid(self):
        df = _load(VALID_TITANIC_BYTES)
        result = validate_dataset(df, "titanic.csv")
        self.assertTrue(result["is_valid"])

    def test_valid_titanic_has_no_errors(self):
        df = _load(VALID_TITANIC_BYTES)
        result = validate_dataset(df, "titanic.csv")
        self.assertEqual(result["errors"], [])

    def test_valid_titanic_result_has_required_keys(self):
        df = _load(VALID_TITANIC_BYTES)
        result = validate_dataset(df, "titanic.csv")
        self.assertIn("is_valid", result)
        self.assertIn("errors", result)
        self.assertIn("warnings", result)

    # Case 2 — empty CSV ───────────────────────────────────────────────────────
    def test_empty_dataset_is_invalid(self):
        df = _load(EMPTY_CSV_BYTES)
        result = validate_dataset(df, "empty.csv")
        self.assertFalse(result["is_valid"])

    def test_empty_dataset_reports_error(self):
        df = _load(EMPTY_CSV_BYTES)
        result = validate_dataset(df, "empty.csv")
        combined = " ".join(result["warnings"]).lower()
        self.assertIn("empty", combined)

    def test_empty_dataset_error_is_readable_string(self):
        df = _load(EMPTY_CSV_BYTES)
        result = validate_dataset(df, "empty.csv")
        for msg in result["warnings"]:
            self.assertIsInstance(msg, str)
            self.assertGreater(len(msg), 5)

    # Case 3 — duplicate column names (unit-only; see module docstring) ────────
    def test_duplicate_columns_is_invalid(self):
        df = _duplicate_columns_df()
        result = validate_dataset(df, "data.csv")
        self.assertFalse(result["is_valid"])

    def test_duplicate_columns_reports_error(self):
        df = _duplicate_columns_df()
        result = validate_dataset(df, "data.csv")
        combined = " ".join(result["errors"]).lower()
        self.assertIn("duplicate", combined)

    def test_duplicate_columns_does_not_raise(self):
        df = _duplicate_columns_df()
        try:
            validate_dataset(df, "data.csv")
        except Exception as exc:
            self.fail(f"validate_dataset raised unexpectedly: {exc}")

    # Case 4 — one column (warning, not error) ─────────────────────────────────
    def test_one_column_is_valid(self):
        df = _load(ONE_COLUMN_CSV_BYTES)
        result = validate_dataset(df, "single.csv")
        self.assertTrue(result["is_valid"])

    def test_one_column_raises_warning(self):
        df = _load(ONE_COLUMN_CSV_BYTES)
        result = validate_dataset(df, "single.csv")
        combined = " ".join(result["warnings"]).lower()
        self.assertIn("one column", combined)

    def test_one_column_has_no_errors(self):
        df = _load(ONE_COLUMN_CSV_BYTES)
        result = validate_dataset(df, "single.csv")
        self.assertEqual(result["errors"], [])

    # Case 5 — fully missing column (warning, not error) ──────────────────────
    def test_fully_missing_column_is_valid(self):
        df = _load(FULLY_MISSING_COLUMN_BYTES)
        result = validate_dataset(df, "data.csv")
        self.assertTrue(result["is_valid"])

    def test_fully_missing_column_raises_warning(self):
        df = _load(FULLY_MISSING_COLUMN_BYTES)
        result = validate_dataset(df, "data.csv")
        combined = " ".join(result["warnings"]).lower()
        self.assertIn("missing values", combined)

    def test_fully_missing_column_has_no_errors(self):
        df = _load(FULLY_MISSING_COLUMN_BYTES)
        result = validate_dataset(df, "data.csv")
        self.assertEqual(result["errors"], [])

    # Case 6 — target column not in dataset (unit-only; see module docstring) ──
    def test_nonexistent_target_is_invalid(self):
        df = _load(VALID_TITANIC_BYTES)
        result = validate_dataset(df, "titanic.csv", target_column="ghost_column")
        self.assertFalse(result["is_valid"])

    def test_nonexistent_target_reports_error(self):
        df = _load(VALID_TITANIC_BYTES)
        result = validate_dataset(df, "titanic.csv", target_column="ghost_column")
        combined = " ".join(result["errors"]).lower()
        self.assertTrue("not found" in combined or "ghost_column" in combined)

    def test_nonexistent_target_error_names_the_column(self):
        df = _load(VALID_TITANIC_BYTES)
        result = validate_dataset(df, "titanic.csv", target_column="ghost_column")
        combined = " ".join(result["errors"])
        self.assertIn("ghost_column", combined)


# ══════════════════════════════════════════════════════════════════════════════
# UI tests — what the app renders (items 1-6 from the spec)
# ══════════════════════════════════════════════════════════════════════════════

class TestValidationUI(unittest.TestCase):

    # ── item 1 + item 6: validation runs before profiling (execution order) ───
    def test_validation_subheader_precedes_profiling_subheader(self):
        """Pre-cleaning Validation must appear before Data Quality Report."""
        at = _upload("titanic.csv", VALID_TITANIC_BYTES)
        labels = _subheader_labels(at)
        self.assertIn("2. Pre-cleaning Validation", labels)
        self.assertIn("3. Data Quality Report", labels)
        self.assertLess(
            labels.index("2. Pre-cleaning Validation"),
            labels.index("3. Data Quality Report"),
        )

    # ── item 2: errors stop profiling ─────────────────────────────────────────
    def test_empty_csv_error_blocks_profiling_subheader(self):
        at = _upload("empty.csv", EMPTY_CSV_BYTES)
        self.assertNotIn("3. Data Quality Report", _subheader_labels(at))

    def test_empty_csv_shows_error_messages(self):
        at = _upload("empty.csv", EMPTY_CSV_BYTES)
        self.assertGreater(len(at.warning), 0)

    def test_empty_csv_does_not_crash(self):
        at = _upload("empty.csv", EMPTY_CSV_BYTES)
        self.assertEqual(len(at.exception), 0)

    # ── item 3: warnings do NOT stop profiling ────────────────────────────────
    def test_one_column_warning_still_shows_profiling(self):
        at = _upload("single.csv", ONE_COLUMN_CSV_BYTES)
        self.assertIn("3. Data Quality Report", _subheader_labels(at))

    def test_one_column_shows_warning_message(self):
        at = _upload("single.csv", ONE_COLUMN_CSV_BYTES)
        bodies = " ".join(e.body for e in at.warning).lower()
        self.assertIn("one column", bodies)

    def test_fully_missing_column_warning_still_shows_profiling(self):
        at = _upload("data.csv", FULLY_MISSING_COLUMN_BYTES)
        self.assertIn("3. Data Quality Report", _subheader_labels(at))

    def test_fully_missing_column_shows_warning_message(self):
        at = _upload("data.csv", FULLY_MISSING_COLUMN_BYTES)
        bodies = " ".join(e.body for e in at.warning).lower()
        self.assertIn("missing values", bodies)

    # ── item 4: user sees readable validation messages ────────────────────────
    def test_validation_success_message_is_readable(self):
        at = _upload("titanic.csv", VALID_TITANIC_BYTES)
        success_texts = [e.body for e in at.success]
        self.assertTrue(
            any("Validation passed" in t for t in success_texts),
            f"Expected 'Validation passed' in successes; got {success_texts}",
        )

    def test_validation_error_message_is_readable_string(self):
        at = _upload("empty.csv", EMPTY_CSV_BYTES)
        for elem in at.warning:
            self.assertIsInstance(elem.body, str)
            self.assertGreater(len(elem.body), 5)
            self.assertNotIn("Traceback", elem.body)

    def test_validation_warning_message_is_readable_string(self):
        at = _upload("single.csv", ONE_COLUMN_CSV_BYTES)
        for elem in at.warning:
            self.assertIsInstance(elem.body, str)
            self.assertGreater(len(elem.body), 5)
            self.assertNotIn("Traceback", elem.body)

    # ── item 5: cleaning does not run when validation fails ───────────────────
    def test_validation_failure_no_further_processing(self):
        """When validation errors exist, profiling (and cleaning) are absent."""
        at = _upload("empty.csv", EMPTY_CSV_BYTES)
        labels = _subheader_labels(at)
        self.assertNotIn("3. Data Quality Report", labels)
        self.assertEqual(len(at.exception), 0)

    def test_validation_failure_no_exception_raised(self):
        at = _upload("empty.csv", EMPTY_CSV_BYTES)
        self.assertEqual(len(at.exception), 0)

    # ── item 6: execution order load → validate → profile ─────────────────────
    def test_execution_order_all_three_stages_present_for_valid_data(self):
        """Load (success message), validate (section), profile (section) all appear."""
        at = _upload("titanic.csv", VALID_TITANIC_BYTES)
        labels = _subheader_labels(at)
        successes = [e.body for e in at.success]
        self.assertTrue(any("Uploaded file" in s for s in successes), "Load stage missing")
        self.assertIn("2. Pre-cleaning Validation", labels, "Validate stage missing")
        self.assertIn("3. Data Quality Report", labels, "Profile stage missing")

    def test_execution_order_validate_before_profile_index(self):
        """Index of validate subheader must be strictly less than profile subheader."""
        at = _upload("titanic.csv", VALID_TITANIC_BYTES)
        labels = _subheader_labels(at)
        self.assertLess(
            labels.index("2. Pre-cleaning Validation"),
            labels.index("3. Data Quality Report"),
        )

    def test_execution_order_load_stage_present_even_when_validation_fails(self):
        """Validation still runs, but profiling does not, when the dataset is empty."""
        at = _upload("empty.csv", EMPTY_CSV_BYTES)
        labels = _subheader_labels(at)
        self.assertNotIn("2. Pre-cleaning Validation", labels)
        self.assertNotIn("3. Data Quality Report", labels)


if __name__ == "__main__":
    unittest.main()
