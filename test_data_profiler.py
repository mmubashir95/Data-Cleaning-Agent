"""Tests for reusable dataset profiling and column classification."""

import unittest

import pandas as pd

from utils.data_profiler import classify_columns, profile_dataset


class TestClassifyColumns(unittest.TestCase):
    def test_titanic_like_columns_are_classified_generically(self):
        df = pd.DataFrame(
            {
                "PassengerId": [1, 2, 3, 4, 5, 6],
                "Survived": [0, 1, 1, 0, 1, 0],
                "Pclass": [3, 1, 3, 1, 2, 3],
                "Name": [
                    "Allen, Mr. William Henry",
                    "Bonnell, Miss. Elizabeth",
                    "Moran, Mr. James",
                    "Nasser, Mrs. Nicholas",
                    "Palsson, Master. Gosta",
                    "Rice, Master. Eugene",
                ],
                "Sex": ["male", "female", "male", "female", "male", "male"],
                "Age": [35.0, 58.0, None, 14.0, 2.0, 7.0],
                "Ticket": ["373450", "113783", "330877", "237736", "349909", "382652"],
                "Fare": [8.05, 71.2833, 8.4583, 13.0, 21.075, 11.1333],
                "Cabin": ["C85", "C123", None, "E46", None, None],
                "Embarked": ["S", "C", "Q", "S", "S", "S"],
            }
        )

        classified = classify_columns(df, target_column="Survived")

        self.assertIn("PassengerId", classified["numeric_columns"])
        self.assertIn("PassengerId", classified["id_like_columns"])
        self.assertIn("Survived", classified["numeric_columns"])
        self.assertIn("Survived", classified["boolean_columns"])
        self.assertIn("Sex", classified["categorical_columns"])
        self.assertIn("Embarked", classified["categorical_columns"])
        self.assertIn("Name", classified["text_columns"])
        self.assertIn("Ticket", classified["text_columns"])
        self.assertIn("Cabin", classified["text_columns"])
        self.assertEqual(classified["target_column"], "Survived")

    def test_datetime_and_boolean_string_detection(self):
        df = pd.DataFrame(
            {
                "created_at": ["2024-01-01", "2024-01-02", "2024-01-03"],
                "is_active": ["yes", "no", "yes"],
                "message": [
                    "First long support message",
                    "Second long support message",
                    "Third long support message",
                ],
                "city": ["Lahore", "Karachi", "Lahore"],
            }
        )

        classified = classify_columns(df)

        self.assertIn("created_at", classified["datetime_columns"])
        self.assertIn("is_active", classified["boolean_columns"])
        self.assertIn("message", classified["text_columns"])
        self.assertIn("city", classified["categorical_columns"])

    def test_profile_dataset_contains_extended_groups(self):
        df = pd.DataFrame(
            {
                "Id": [101, 102, 103],
                "SalePrice": [200000, 185000, 230000],
                "Neighborhood": ["NAmes", "CollgCr", "NAmes"],
            }
        )

        profile = profile_dataset(df, target_column="SalePrice")

        self.assertEqual(profile["rows"], 3)
        self.assertEqual(profile["columns"], 3)
        self.assertIn("numeric_columns", profile)
        self.assertIn("categorical_columns", profile)
        self.assertIn("text_columns", profile)
        self.assertIn("datetime_columns", profile)
        self.assertIn("boolean_columns", profile)
        self.assertIn("id_like_columns", profile)
        self.assertEqual(profile["target_column"], "SalePrice")


if __name__ == "__main__":
    unittest.main()
