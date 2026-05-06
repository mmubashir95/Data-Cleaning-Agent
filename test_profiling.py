"""Profiling tests for Titanic, House Prices, and Spam SMS CSV fixtures.

Items 1-8 test profile_dataset() directly.
Item 9 tests the target-column selectbox via AppTest.
"""

import io
import unittest

import pandas as pd
from streamlit.testing.v1 import AppTest

from utils.data_profiler import profile_dataset

# ── fixture builders ───────────────────────────────────────────────────────────

def _to_csv_bytes(df: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    df.to_csv(buf, index=False)
    return buf.getvalue()


def _titanic_df() -> pd.DataFrame:
    n = 50
    return pd.DataFrame({
        "PassengerId": range(1, n + 1),
        "Survived":   [0, 1, 1, 1, 0] * 10,
        "Pclass":     [3, 1, 3, 1, 2] * 10,
        "Name":       [f"Smith Mr. John {i}" for i in range(n)],
        "Sex":        ["male", "female"] * 25,
        "Age":        [22.0, 38.0, None, 35.0, 27.0] * 10,   # 10 missing
        "SibSp":      [1, 0, 0, 1, 0] * 10,
        "Parch":      [0, 0, 0, 0, 2] * 10,
        "Ticket":     [f"T{i:04d}" for i in range(n)],
        "Fare":       [7.25, 71.28, 7.93, 53.10, 11.13] * 10,
        "Cabin":      [None, "C85", None, "C123", None] * 10,  # 30 missing
        "Embarked":   ["S", "C", "S", "S", "Q"] * 10,
    })


def _house_prices_df() -> pd.DataFrame:
    n = 50
    mszoning  = ["RL", "RM", "C(all)", "FV", "RH"] * 10
    street    = ["Pave", "Grvl"] * 25
    bldgtype  = ["1Fam", "2FmCon", "Duplx", "TwnhsE", "Twnhs"] * 10
    return pd.DataFrame({
        "Id":          range(1, n + 1),
        "MSSubClass":  [60, 20, 70, 60, 50] * 10,
        "MSZoning":    mszoning,
        "LotArea":     [8450, 9600, 11250, 9550, 14260] * 10,
        "Street":      street,
        "BldgType":    bldgtype,
        "OverallQual": [7, 6, 7, 7, 8] * 10,
        "OverallCond": [5, 8, 5, 5, 5] * 10,
        "YearBuilt":   [2003, 1976, 2001, 1915, 2000] * 10,
        "SalePrice":   [208500, 181500, 223500, 140000, 250000] * 10,
    })


def _spam_sms_df() -> pd.DataFrame:
    # 40 rows are required so that unique_count > 30 and the profiler
    # routes the message column to text rather than categorical.
    labels = (["ham", "spam"] * 20)
    messages = [
        f"Message {i}: " + (
            "Congratulations! You have won a free prize. Call 555-0100 now to claim!"
            if i % 3 == 0
            else f"Hey, just wanted to check in about our plans for the weekend on day {i}."
        )
        for i in range(40)
    ]
    return pd.DataFrame({"label": labels, "message": messages})


# Pre-built DataFrames (CSV round-trip mirrors what data_loader does)
_TITANIC_DF       = pd.read_csv(io.BytesIO(_to_csv_bytes(_titanic_df())))
_HOUSE_PRICES_DF  = pd.read_csv(io.BytesIO(_to_csv_bytes(_house_prices_df())))
_SPAM_SMS_DF      = pd.read_csv(io.BytesIO(_to_csv_bytes(_spam_sms_df())))

_TITANIC_PROFILE      = profile_dataset(_TITANIC_DF)
_HOUSE_PRICES_PROFILE = profile_dataset(_HOUSE_PRICES_DF)
_SPAM_SMS_PROFILE     = profile_dataset(_SPAM_SMS_DF)

_TITANIC_CSV_BYTES      = _to_csv_bytes(_titanic_df())
_HOUSE_PRICES_CSV_BYTES = _to_csv_bytes(_house_prices_df())
_SPAM_SMS_CSV_BYTES     = _to_csv_bytes(_spam_sms_df())


# ── shared profiling mixin ─────────────────────────────────────────────────────

class _ProfilingChecks:
    """Shared assertions mixed into each dataset test class.

    Subclasses must define:
        profile        – result of profile_dataset(df)
        expected_rows  – int
        expected_cols  – int
        expected_columns      – list[str]
        known_numeric         – list[str]  (must be subset of numeric_columns)
        known_categorical     – list[str]  (must be subset of categorical_columns)
        known_text            – list[str]  (must be subset of text_columns)
        expected_missing      – dict[str, int]  (spot-check: col → count)
    """

    # 1 ── shape
    def test_shape_rows(self):
        self.assertEqual(self.profile["rows"], self.expected_rows)

    def test_shape_columns(self):
        self.assertEqual(self.profile["columns"], self.expected_cols)

    # 2 ── column names
    def test_column_names_complete(self):
        self.assertEqual(self.profile["column_names"], self.expected_columns)

    # 3 ── data types
    def test_data_types_has_entry_for_every_column(self):
        dtypes = self.profile["data_types"]
        self.assertEqual(set(dtypes.keys()), set(self.expected_columns))

    def test_data_types_are_strings(self):
        for col, dtype in self.profile["data_types"].items():
            self.assertIsInstance(dtype, str, f"dtype for '{col}' should be a string")

    # 4 ── missing values
    def test_missing_values_has_entry_for_every_column(self):
        mv = self.profile["missing_values"]
        self.assertEqual(set(mv.keys()), set(self.expected_columns))

    def test_missing_values_are_non_negative(self):
        for col, count in self.profile["missing_values"].items():
            self.assertGreaterEqual(count, 0, f"negative missing count for '{col}'")

    def test_known_missing_counts(self):
        mv = self.profile["missing_values"]
        for col, expected_count in self.expected_missing.items():
            self.assertEqual(mv[col], expected_count, f"wrong missing count for '{col}'")

    # 5 ── duplicate count
    def test_duplicate_count_is_integer(self):
        self.assertIsInstance(self.profile["duplicate_rows"], int)

    def test_duplicate_count_correct(self):
        self.assertEqual(self.profile["duplicate_rows"], self.expected_duplicates)

    # 6 ── numeric columns
    def test_numeric_columns_detected(self):
        numeric = self.profile["numeric_columns"]
        for col in self.known_numeric:
            self.assertIn(col, numeric, f"'{col}' should be numeric")

    # 7 ── categorical columns
    def test_categorical_columns_detected(self):
        categorical = self.profile["categorical_columns"]
        for col in self.known_categorical:
            self.assertIn(col, categorical, f"'{col}' should be categorical")

    # 8 ── text columns
    def test_text_columns_detected(self):
        text = self.profile["text_columns"]
        for col in self.known_text:
            self.assertIn(col, text, f"'{col}' should be text")


# ── Titanic ────────────────────────────────────────────────────────────────────

class TestTitanicProfiling(_ProfilingChecks, unittest.TestCase):
    profile     = _TITANIC_PROFILE
    expected_rows    = 50
    expected_cols    = 12
    expected_columns = [
        "PassengerId", "Survived", "Pclass", "Name", "Sex", "Age",
        "SibSp", "Parch", "Ticket", "Fare", "Cabin", "Embarked",
    ]
    known_numeric     = ["Age", "Fare", "Pclass"]
    known_categorical = ["Sex", "Embarked"]
    known_text        = ["Name"]
    expected_missing  = {"Age": 10, "Cabin": 30}
    expected_duplicates = 0


# ── House Prices ───────────────────────────────────────────────────────────────

class TestHousePricesProfiling(_ProfilingChecks, unittest.TestCase):
    profile     = _HOUSE_PRICES_PROFILE
    expected_rows    = 50
    expected_cols    = 10
    expected_columns = [
        "Id", "MSSubClass", "MSZoning", "LotArea", "Street",
        "BldgType", "OverallQual", "OverallCond", "YearBuilt", "SalePrice",
    ]
    known_numeric     = ["LotArea", "YearBuilt", "SalePrice"]
    known_categorical = ["MSZoning", "Street", "BldgType"]
    known_text        = []   # no free-form text columns in this dataset
    expected_missing  = {col: 0 for col in expected_columns}
    expected_duplicates = 0

    def test_text_columns_detected(self):
        # House Prices has no free-form text columns; assert list is empty.
        self.assertEqual(self.profile["text_columns"], [])


# ── Spam SMS ───────────────────────────────────────────────────────────────────

class TestSpamSMSProfiling(_ProfilingChecks, unittest.TestCase):
    profile     = _SPAM_SMS_PROFILE
    expected_rows    = 40
    expected_cols    = 2
    expected_columns = ["label", "message"]
    known_numeric     = []   # no numeric columns
    known_categorical = ["label"]
    known_text        = ["message"]
    expected_missing  = {"label": 0, "message": 0}
    expected_duplicates = 0

    def test_numeric_columns_detected(self):
        # Spam SMS has no numeric columns.
        self.assertEqual(self.profile["numeric_columns"], [])


# ── Target column selector (UI — item 9) ──────────────────────────────────────

class TestTargetColumnSelectorUI(unittest.TestCase):
    """The target-column selectbox must list every column in the uploaded dataset."""

    def _run(self, filename: str, csv_bytes: bytes, expected_columns: list) -> None:
        at = AppTest.from_file("app.py").run()
        at.sidebar.file_uploader[0].upload(filename, csv_bytes, "text/csv").run()

        target_box = next(
            (sb for sb in at.selectbox if sb.label == "Select target column (optional)"),
            None,
        )
        self.assertIsNotNone(target_box, "Target column selectbox not found on main page")

        # AppTest serialises Python None to the string 'None' in options.
        column_options = [o for o in target_box.options if o not in (None, "None")]
        self.assertEqual(column_options, expected_columns)

    def test_titanic_target_selector(self):
        cols = [
            "PassengerId", "Survived", "Pclass", "Name", "Sex", "Age",
            "SibSp", "Parch", "Ticket", "Fare", "Cabin", "Embarked",
        ]
        self._run("titanic.csv", _TITANIC_CSV_BYTES, cols)

    def test_house_prices_target_selector(self):
        cols = [
            "Id", "MSSubClass", "MSZoning", "LotArea", "Street",
            "BldgType", "OverallQual", "OverallCond", "YearBuilt", "SalePrice",
        ]
        self._run("house_prices.csv", _HOUSE_PRICES_CSV_BYTES, cols)

    def test_spam_sms_target_selector(self):
        self._run("spam_sms.csv", _SPAM_SMS_CSV_BYTES, ["label", "message"])


if __name__ == "__main__":
    unittest.main()
