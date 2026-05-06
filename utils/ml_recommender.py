"""Helpers for selecting a likely ML problem type and starter algorithms."""

from __future__ import annotations

from typing import Any

import pandas as pd
from pandas.api.types import is_numeric_dtype

from utils.nlp_cleaner import detect_text_columns
from utils.data_profiler import classify_columns

NLP_TEXT_HINTS = {
    "message",
    "review",
    "comment",
    "text",
    "sentence",
    "feedback",
    "tweet",
    "description",
    "content",
    "body",
}

NON_NLP_TEXT_HINTS = {
    "name",
    "ticket",
    "cabin",
    "productcode",
    "product_code",
    "orderid",
    "order_id",
    "customername",
    "customer_name",
    "passengername",
    "passenger_name",
    "address",
    "id",
    "code",
    "sku",
}


def _is_classification_target(series: pd.Series) -> bool:
    """Detect common classification-style targets from value patterns."""
    non_null = series.dropna()
    if non_null.empty:
        return False

    unique_count = non_null.nunique(dropna=True)
    normalized_values = {str(value).strip().lower() for value in non_null.unique()}

    if normalized_values in [
        {"0", "1"},
        {"true", "false"},
        {"yes", "no"},
        {"approved", "rejected"},
        {"spam", "ham"},
    ]:
        return True

    if not is_numeric_dtype(series):
        return True

    return unique_count <= max(10, int(len(non_null) * 0.05))


def _is_regression_target(series: pd.Series) -> bool:
    """Detect continuous numeric targets for regression."""
    non_null = series.dropna()
    if non_null.empty or not is_numeric_dtype(series):
        return False

    unique_count = non_null.nunique(dropna=True)
    unique_ratio = unique_count / len(non_null)
    return unique_count > 4 and unique_ratio >= 0.5


def _looks_like_main_nlp_text_feature(df: pd.DataFrame, text_columns: list[str]) -> str | None:
    """Return the strongest NLP-style text feature, or None if text is metadata-like."""
    for column in text_columns:
        normalized_name = column.strip().lower()
        collapsed_name = normalized_name.replace(" ", "").replace("-", "").replace(".", "")

        if normalized_name in NLP_TEXT_HINTS or collapsed_name in NLP_TEXT_HINTS:
            return column

        if normalized_name in NON_NLP_TEXT_HINTS or collapsed_name in NON_NLP_TEXT_HINTS:
            continue

        series = df[column].dropna().astype(str).str.strip()
        if series.empty:
            continue

        average_length = series.str.len().mean()
        average_word_count = series.str.split().str.len().mean()

        if average_length > 20 and average_word_count >= 4:
            return column

    return None


def recommend_ml_approach(
    df: pd.DataFrame,
    target_column: str | None,
    problem_type: str,
    text_columns: list[str],
) -> dict[str, Any]:
    """Recommend a generic ML approach using target behavior and column heuristics."""
    column_types = classify_columns(df, target_column=target_column)
    warnings: list[str] = []
    candidate_text_columns = detect_text_columns(df, target_column=target_column)
    detected_text_column = _looks_like_main_nlp_text_feature(df, candidate_text_columns)

    if target_column is None:
        if problem_type in {"Classification", "Regression", "NLP/Text Classification"}:
            warnings.append(
                f"You selected '{problem_type}', but no target column is available. Unsupervised clustering is recommended by default."
            )

        algorithms = [
            {
                "name": "K-Means",
                "reason": "K-Means is a simple beginner-friendly clustering algorithm for grouping similar records.",
            },
            {
                "name": "DBSCAN",
                "reason": "DBSCAN is useful when clusters are irregular and when you want to detect noise points.",
            },
        ]
        return {
            "recommended_problem_type": "Clustering",
            "reason": "No target column was selected, so there is no supervised label available.",
            "warnings": warnings,
            "target_column": None,
            "detected_text_column": detected_text_column,
            "numeric_columns": column_types["numeric_columns"],
            "categorical_columns": column_types["categorical_columns"],
            "text_columns": column_types["text_columns"],
            "boolean_columns": column_types["boolean_columns"],
            "datetime_columns": column_types["datetime_columns"],
            "id_like_columns": column_types["id_like_columns"],
            "algorithms": algorithms,
        }

    target_series = df[target_column]

    if problem_type != "Auto-detect":
        recommended_problem_type = problem_type
        reason = f"The user selected '{problem_type}', so that choice is being respected."
    elif detected_text_column is not None:
        recommended_problem_type = "NLP/Text Classification"
        reason = (
            f"A text column was detected ({detected_text_column}) and a target column exists, "
            "so this looks like a supervised NLP task."
        )
    elif _is_regression_target(target_series):
        recommended_problem_type = "Regression"
        reason = "The target column is numeric and behaves like continuous data with many distinct values."
    else:
        recommended_problem_type = "Classification"
        reason = "The target column behaves like categories or has only a small number of distinct label values."

    if recommended_problem_type == "NLP/Text Classification":
        algorithms = [
            {
                "name": "Naive Bayes",
                "reason": "Naive Bayes is a strong beginner-friendly baseline for TF-IDF or Bag-of-Words text features.",
            },
            {
                "name": "Logistic Regression",
                "reason": "Logistic Regression performs very well on sparse text vectors and is easy to interpret.",
            },
            {
                "name": "Linear SVM",
                "reason": "Linear SVM often performs strongly for high-dimensional text classification.",
            },
        ]
        warnings.append(
            "Recommended NLP preprocessing: lowercase conversion, punctuation removal, stopword removal, tokenization, and TF-IDF or Bag-of-Words vectorization."
        )
    elif recommended_problem_type == "Regression":
        algorithms = [
            {
                "name": "Linear Regression",
                "reason": "Linear Regression is the most beginner-friendly baseline for continuous numeric targets.",
            },
            {
                "name": "Random Forest Regressor",
                "reason": "Random Forest Regressor handles non-linear patterns with less manual tuning.",
            },
            {
                "name": "Gradient Boosting Regressor",
                "reason": "Gradient Boosting Regressor can capture complex relationships and often improves accuracy.",
            },
        ]
    elif recommended_problem_type == "Clustering":
        algorithms = [
            {
                "name": "K-Means",
                "reason": "K-Means is simple to start with when grouping data without labels.",
            },
            {
                "name": "DBSCAN",
                "reason": "DBSCAN can discover irregular clusters and isolate noise points.",
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
        "target_column": target_column,
        "detected_text_column": detected_text_column,
        "numeric_columns": column_types["numeric_columns"],
        "categorical_columns": column_types["categorical_columns"],
        "text_columns": column_types["text_columns"],
        "boolean_columns": column_types["boolean_columns"],
        "datetime_columns": column_types["datetime_columns"],
        "id_like_columns": column_types["id_like_columns"],
        "algorithms": algorithms,
    }
