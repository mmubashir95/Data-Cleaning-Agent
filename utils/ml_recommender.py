"""Helpers for selecting a likely ML problem type and starter algorithms."""

from __future__ import annotations

from typing import Any

import pandas as pd
from pandas.api.types import is_numeric_dtype


def recommend_ml_approach(
    df: pd.DataFrame,
    target_column: str | None,
    problem_type: str,
    text_columns: list[str],
) -> dict[str, Any]:
    """Recommend a beginner-friendly ML approach from simple dataset signals."""
    if target_column is None:
        warnings: list[str] = []
        if problem_type in {"Classification", "Regression", "NLP/Text Classification"}:
            warnings.append(
                f"You selected '{problem_type}', but no target column is available. Clustering is recommended until a supervised target is chosen."
            )
        return {
            "recommended_problem_type": "Clustering",
            "reason": "No target column was selected, so there is no supervised label available.",
            "warnings": warnings,
            "algorithms": [
                {
                    "name": "K-Means",
                    "reason": "K-Means is a simple beginner-friendly clustering method for grouping similar records.",
                },
                {
                    "name": "DBSCAN",
                    "reason": "DBSCAN can discover clusters with unusual shapes and can also flag noise points.",
                },
            ],
        }

    warnings: list[str] = []
    target_series = df[target_column]

    if problem_type != "Auto-detect":
        recommended_problem_type = problem_type
        reason = f"The user selected '{problem_type}', so that choice is being respected."
    else:
        if text_columns and target_column is not None:
            recommended_problem_type = "NLP/Text Classification"
            reason = "Text columns and a target column are both present, which strongly suggests a supervised NLP task."
        elif is_numeric_dtype(target_series) and target_series.nunique(dropna=True) > 10:
            recommended_problem_type = "Regression"
            reason = "The target column is numeric and looks continuous, which fits regression."
        else:
            recommended_problem_type = "Classification"
            reason = "The target column has a small number of distinct values, which fits classification."

    if recommended_problem_type == "Clustering":
        algorithms = [
            {
                "name": "K-Means",
                "reason": "K-Means is simple to start with when you need to group data without labels.",
            },
            {
                "name": "DBSCAN",
                "reason": "DBSCAN is useful when clusters are irregular and when you want noise detection.",
            },
        ]
    elif recommended_problem_type == "NLP/Text Classification":
        algorithms = [
            {
                "name": "Naive Bayes",
                "reason": "Naive Bayes is a classic beginner-friendly baseline for TF-IDF or Bag-of-Words text features.",
            },
            {
                "name": "Logistic Regression",
                "reason": "Logistic Regression performs strongly on sparse text vectors and is easy to interpret.",
            },
            {
                "name": "Linear SVM",
                "reason": "Linear SVM often works very well for text classification with high-dimensional sparse features.",
            },
        ]
    elif recommended_problem_type == "Regression":
        algorithms = [
            {
                "name": "Linear Regression",
                "reason": "Linear Regression is the most beginner-friendly regression baseline.",
            },
            {
                "name": "Random Forest Regressor",
                "reason": "Random Forest Regressor handles non-linear relationships with less manual tuning.",
            },
            {
                "name": "Gradient Boosting Regressor",
                "reason": "Gradient Boosting Regressor can model complex patterns and often improves accuracy.",
            },
        ]
    else:
        algorithms = [
            {
                "name": "Logistic Regression",
                "reason": "Logistic Regression is a strong and simple baseline for binary or multiclass classification.",
            },
            {
                "name": "Decision Tree",
                "reason": "Decision Trees are easy to understand and explain to beginners.",
            },
            {
                "name": "Random Forest",
                "reason": "Random Forest usually improves stability and accuracy over a single decision tree.",
            },
        ]

    return {
        "recommended_problem_type": recommended_problem_type,
        "reason": reason,
        "warnings": warnings,
        "algorithms": algorithms,
    }
