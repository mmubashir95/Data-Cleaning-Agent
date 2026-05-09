"""Tests for target suggestion and ML problem type inference."""

from __future__ import annotations

import unittest

import pandas as pd

from utils.data_profiler import profile_dataset
from utils.ml_recommender import recommend_ml_approach


class TestMlRecommenderIntelligence(unittest.TestCase):
    def test_titanic_without_manual_target_suggests_survived_classification(self):
        dataframe = pd.DataFrame(
            {
                "PassengerId": [1, 2, 3, 4, 5],
                "Survived": [0, 1, 1, 0, 1],
                "Pclass": [3, 1, 3, 1, 2],
                "Age": [22.0, 38.0, None, 35.0, 27.0],
                "Fare": [7.25, 71.28, 7.93, 53.10, 11.13],
                "Embarked": ["S", "C", "S", "S", "Q"],
            }
        )
        profile = profile_dataset(dataframe)
        recommendation = recommend_ml_approach(
            dataframe,
            target_column=None,
            problem_type="Auto-detect",
            text_columns=profile["text_columns"],
        )

        self.assertEqual(recommendation["suggested_target_column"], "Survived")
        self.assertEqual(recommendation["recommended_problem_type"], "Binary Classification")
        self.assertEqual(recommendation["target_detection_confidence"], "High")
        self.assertEqual(
            recommendation["algorithm_recommendation"]["beginner_friendly_first_choice"]["name"],
            "Logistic Regression",
        )
        self.assertEqual(
            recommendation["algorithm_recommendation"]["target_variable_type"],
            "Binary categorical target",
        )

    def test_house_prices_detects_regression(self):
        dataframe = pd.DataFrame(
            {
                "LotArea": [8450, 9600, 11250, 9550, 14260],
                "OverallQual": [7, 6, 7, 7, 8],
                "Neighborhood": ["NAmes", "CollgCr", "OldTown", "Edwards", "Somerst"],
                "SalePrice": [208500, 181500, 223500, 140000, 250000],
            }
        )
        profile = profile_dataset(dataframe, target_column="SalePrice")
        recommendation = recommend_ml_approach(
            dataframe,
            target_column="SalePrice",
            problem_type="Auto-detect",
            text_columns=profile["text_columns"],
        )

        self.assertEqual(recommendation["recommended_problem_type"], "Regression")
        self.assertEqual(
            recommendation["algorithm_recommendation"]["beginner_friendly_first_choice"]["name"],
            "Linear Regression",
        )
        self.assertEqual(
            recommendation["algorithm_recommendation"]["target_variable_type"],
            "Continuous numeric target",
        )

    def test_spam_sms_detects_nlp_text_classification(self):
        dataframe = pd.DataFrame(
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
        profile = profile_dataset(dataframe, target_column="label")
        recommendation = recommend_ml_approach(
            dataframe,
            target_column="label",
            problem_type="Auto-detect",
            text_columns=profile["text_columns"],
        )

        self.assertEqual(recommendation["recommended_problem_type"], "NLP/Text Classification")
        self.assertEqual(
            recommendation["algorithm_recommendation"]["beginner_friendly_first_choice"]["name"],
            "Naive Bayes",
        )
        self.assertEqual(
            recommendation["algorithm_recommendation"]["target_variable_type"],
            "Binary text label target",
        )

    def test_customer_segmentation_without_target_stays_clustering(self):
        dataframe = pd.DataFrame(
            {
                "CustomerID": [101, 102, 103, 104, 105],
                "Age": [21, 35, 45, 29, 39],
                "Income": [30, 80, 55, 42, 71],
                "SpendingScore": [81, 12, 55, 72, 34],
                "PurchaseFrequency": [9, 2, 5, 7, 4],
            }
        )
        profile = profile_dataset(dataframe)
        recommendation = recommend_ml_approach(
            dataframe,
            target_column=None,
            problem_type="Auto-detect",
            text_columns=profile["text_columns"],
        )

        self.assertIsNone(recommendation["suggested_target_column"])
        self.assertEqual(recommendation["recommended_problem_type"], "Clustering")
        self.assertEqual(
            recommendation["algorithm_recommendation"]["beginner_friendly_first_choice"]["name"],
            "K-Means",
        )
        self.assertEqual(
            recommendation["algorithm_recommendation"]["target_variable_type"],
            "No target variable provided (unsupervised learning)",
        )

    def test_dataset_with_only_ids_returns_unknown(self):
        dataframe = pd.DataFrame(
            {
                "customer_id": [1, 2, 3, 4, 5],
                "order_id": [101, 102, 103, 104, 105],
                "email": [f"user{i}@example.com" for i in range(5)],
            }
        )
        profile = profile_dataset(dataframe)
        recommendation = recommend_ml_approach(
            dataframe,
            target_column=None,
            problem_type="Auto-detect",
            text_columns=profile["text_columns"],
        )

        self.assertEqual(
            recommendation["recommended_problem_type"],
            "Unknown / Needs More Information",
        )
        self.assertEqual(
            recommendation["algorithm_recommendation"]["beginner_friendly_first_choice"]["name"],
            "Ask user for more information",
        )


if __name__ == "__main__":
    unittest.main()
