"""Helpers for suggesting target columns and ML problem types."""

from __future__ import annotations

from typing import Any

import pandas as pd
from pandas.api.types import is_numeric_dtype

from utils.data_profiler import classify_columns
from utils.nlp_cleaner import detect_text_columns

CLASSIFICATION_TARGET_HINTS = {
    "target",
    "label",
    "class",
    "category",
    "outcome",
    "result",
    "status",
    "survived",
    "churn",
    "exited",
    "default",
    "approved",
    "rejected",
    "spam",
    "sentiment",
    "diagnosis",
    "disease",
    "attrition",
    "fraud",
    "isfraud",
    "converted",
    "clicked",
    "subscribed",
    "cancelled",
    "passfail",
    "placement",
    "hired",
}

REGRESSION_TARGET_HINTS = {
    "price",
    "saleprice",
    "salary",
    "income",
    "score",
    "rating",
    "marks",
    "sales",
    "revenue",
    "amount",
    "demand",
    "quantity",
    "fare",
    "cost",
    "consumption",
    "duration",
    "value",
}

NLP_TARGET_HINTS = {
    "spam",
    "sentiment",
    "category",
    "label",
    "emotion",
    "topic",
    "toxicity",
    "intent",
}

INVALID_TARGET_HINTS = {
    "id",
    "userid",
    "customerid",
    "passengerid",
    "transactionid",
    "orderid",
    "accountid",
    "uuid",
    "email",
    "phone",
    "address",
    "name",
    "ticket",
    "cabin",
}

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

ALGORITHM_MAP = {
    "Classification": [
        {
            "name": "Logistic Regression",
            "reason": "Logistic Regression is a strong baseline for classification and is easy to explain.",
        },
        {
            "name": "Decision Tree",
            "reason": "Decision Trees are simple to interpret and work well for structured classification tasks.",
        },
        {
            "name": "Random Forest",
            "reason": "Random Forest is a strong default model for many tabular classification problems.",
        },
    ],
    "Binary Classification": [
        {
            "name": "Logistic Regression",
            "reason": "Logistic Regression is a strong baseline for binary targets and is easy to explain.",
        },
        {
            "name": "Decision Tree",
            "reason": "Decision Trees are simple to interpret and work well for structured classification tasks.",
        },
        {
            "name": "Random Forest",
            "reason": "Random Forest improves stability and accuracy over a single tree for binary classification.",
        },
    ],
    "Multi-class Classification": [
        {
            "name": "Logistic Regression",
            "reason": "Multinomial Logistic Regression is a clear baseline for multi-class targets.",
        },
        {
            "name": "Decision Tree",
            "reason": "Decision Trees naturally support multi-class prediction and are easy to explain.",
        },
        {
            "name": "Random Forest",
            "reason": "Random Forest is a strong default model for many tabular multi-class problems.",
        },
    ],
    "Regression": [
        {
            "name": "Linear Regression",
            "reason": "Linear Regression is the simplest baseline for continuous numeric targets.",
        },
        {
            "name": "Random Forest Regressor",
            "reason": "Random Forest Regressor handles non-linear structure with little manual tuning.",
        },
    ],
    "NLP/Text Classification": [
        {
            "name": "Naive Bayes",
            "reason": "Naive Bayes is a strong beginner-friendly baseline for TF-IDF or Bag-of-Words text features.",
        },
        {
            "name": "Logistic Regression",
            "reason": "Logistic Regression performs very well on sparse text vectors.",
        },
        {
            "name": "Linear SVM",
            "reason": "Linear SVM often performs strongly for high-dimensional text classification.",
        },
    ],
    "Clustering": [
        {
            "name": "K-Means",
            "reason": "K-Means is a simple starting point for grouping similar records without labels.",
        },
        {
            "name": "DBSCAN",
            "reason": "DBSCAN can discover irregular clusters and isolate noise points.",
        },
    ],
    "Unknown / Needs More Information": [
        {
            "name": "Ask user for more information",
            "reason": "A reliable supervised or unsupervised modeling choice needs a clearer objective or target column.",
        }
    ],
}


def _normalize_name(name: str) -> str:
    return name.strip().lower().replace(" ", "").replace("_", "").replace("-", "")


def _looks_like_main_nlp_text_feature(df: pd.DataFrame, text_columns: list[str]) -> str | None:
    """Return the strongest NLP-style text feature, or None if text is metadata-like."""
    for column in text_columns:
        normalized_name = column.strip().lower()
        collapsed_name = _normalize_name(column)

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


def _binary_token_set(series: pd.Series) -> set[str]:
    non_null = series.dropna()
    return {str(value).strip().lower() for value in non_null.unique()}


def _is_binary_target(series: pd.Series) -> bool:
    non_null = series.dropna()
    if non_null.empty:
        return False

    unique_count = non_null.nunique(dropna=True)
    if unique_count != 2:
        return False

    normalized_values = _binary_token_set(series)
    known_binary_sets = [
        {"0", "1"},
        {"true", "false"},
        {"yes", "no"},
        {"approved", "rejected"},
        {"spam", "ham"},
        {"pass", "fail"},
    ]
    return normalized_values in known_binary_sets or unique_count == 2


def _is_low_cardinality_numeric_classification(series: pd.Series) -> bool:
    non_null = series.dropna()
    if non_null.empty or not is_numeric_dtype(series):
        return False

    unique_count = non_null.nunique(dropna=True)
    if unique_count <= 1:
        return False

    return unique_count <= max(10, int(len(non_null) * 0.05))


def _is_continuous_numeric_target(series: pd.Series) -> bool:
    non_null = series.dropna()
    if non_null.empty or not is_numeric_dtype(series):
        return False

    unique_count = non_null.nunique(dropna=True)
    if unique_count <= 4:
        return False

    unique_ratio = unique_count / len(non_null)
    numeric_range = non_null.max() - non_null.min()
    looks_like_small_class_codes = (
        unique_count <= 10 and float(numeric_range) <= max(10, unique_count * 2)
    )
    if looks_like_small_class_codes:
        return False

    return unique_ratio >= 0.1 or float(numeric_range) > unique_count * 5


def _confidence_from_score(score: int) -> str:
    if score >= 8:
        return "High"
    if score >= 5:
        return "Medium"
    return "Low"


def _is_unknown_dataset(column_types: dict[str, Any], row_count: int) -> bool:
    feature_columns = (
        len(column_types["numeric_columns"])
        + len(column_types["categorical_columns"])
        + len(column_types["text_columns"])
        + len(column_types["datetime_columns"])
        + len(column_types["boolean_columns"])
    )
    id_columns = len(column_types["id_like_columns"])
    return row_count == 0 or feature_columns == 0 or (id_columns >= feature_columns and feature_columns <= 2)


def _infer_problem_type_for_target(
    df: pd.DataFrame,
    target_column: str,
    detected_text_column: str | None,
) -> tuple[str, str]:
    series = df[target_column]
    non_null = series.dropna()
    unique_count = non_null.nunique(dropna=True)
    normalized_name = _normalize_name(target_column)

    if detected_text_column is not None and (
        not is_numeric_dtype(series)
        or _is_binary_target(series)
        or _is_low_cardinality_numeric_classification(series)
        or normalized_name in NLP_TARGET_HINTS
    ):
        return (
            "NLP/Text Classification",
            f"A meaningful text column ({detected_text_column}) exists and '{target_column}' behaves like a label column.",
        )

    if _is_binary_target(series):
        return (
            "Binary Classification",
            f"The target column '{target_column}' contains binary or two-class values.",
        )

    if not is_numeric_dtype(series):
        return (
            "Multi-class Classification",
            f"The target column '{target_column}' is categorical and suitable for supervised classification.",
        )

    if _is_low_cardinality_numeric_classification(series):
        classification_type = "Binary Classification" if unique_count == 2 else "Multi-class Classification"
        return (
            classification_type,
            f"The numeric target column '{target_column}' has only {unique_count} distinct values, which looks like class labels.",
        )

    if _is_continuous_numeric_target(series):
        return (
            "Regression",
            f"The target column '{target_column}' is numeric with many distinct values, which suggests continuous prediction.",
        )

    return (
        "Unknown / Needs More Information",
        f"The target column '{target_column}' does not clearly behave like a standard classification or regression target.",
    )


def suggest_target_columns(
    df: pd.DataFrame,
    *,
    text_columns: list[str],
    max_suggestions: int = 3,
) -> list[dict[str, Any]]:
    """Score columns and return the strongest likely target suggestions."""
    suggestions: list[dict[str, Any]] = []
    row_count = len(df)

    if df.empty or len(df.columns) == 0:
        return suggestions

    last_column_name = str(df.columns[-1]) if len(df.columns) else None

    for column in df.columns:
        series = df[column]
        normalized_name = _normalize_name(column)
        non_null = series.dropna()
        unique_count = int(non_null.nunique(dropna=True)) if not non_null.empty else 0
        unique_ratio = (unique_count / len(non_null)) if len(non_null) else 0.0
        score = 0
        reasons: list[str] = []
        candidate_problem_type = "Unknown / Needs More Information"

        if non_null.empty:
            continue

        if normalized_name in INVALID_TARGET_HINTS:
            score -= 8
            reasons.append("Column name looks like an identifier or contact field, so it should usually not be a target.")

        if normalized_name in CLASSIFICATION_TARGET_HINTS:
            score += 6
            reasons.append("Column name strongly matches a common classification target.")

        if normalized_name in REGRESSION_TARGET_HINTS:
            score += 4 if str(column) == last_column_name else 2
            reasons.append("Column name strongly matches a common regression target.")

        if normalized_name in NLP_TARGET_HINTS:
            score += 5
            reasons.append("Column name matches a common NLP label field.")

        if _is_binary_target(series):
            score += 5
            candidate_problem_type = "Binary Classification"
            reasons.append("Column values look binary or two-class.")
        elif not is_numeric_dtype(series) and 2 <= unique_count <= max(20, int(row_count * 0.2)):
            score += 3
            candidate_problem_type = "Multi-class Classification" if unique_count > 2 else "Binary Classification"
            reasons.append("Column is categorical with manageable label cardinality.")
        elif _is_low_cardinality_numeric_classification(series):
            score += 3
            candidate_problem_type = "Multi-class Classification" if unique_count > 2 else "Binary Classification"
            reasons.append("Numeric column has low unique values, which looks like class labels.")
        elif _is_continuous_numeric_target(series):
            score += 2
            candidate_problem_type = "Regression"
            reasons.append("Numeric column has many distinct values and looks continuous.")

        if column in text_columns:
            score -= 2
            reasons.append("This column itself looks like free-form text, which is usually a feature instead of the target.")

        if str(column) == last_column_name and normalized_name not in INVALID_TARGET_HINTS:
            score += 2
            reasons.append("Last-column position slightly increases the chance that this is the intended target.")

        if detected_text_columns := text_columns:
            if (
                candidate_problem_type in {"Binary Classification", "Multi-class Classification"}
                and normalized_name not in CLASSIFICATION_TARGET_HINTS
                and normalized_name not in NLP_TARGET_HINTS
                and column not in detected_text_columns
            ):
                score -= 2
                reasons.append("A text-feature dataset without a label-like column name makes this target guess less certain.")

        if unique_ratio >= 0.95 and unique_count >= min(20, row_count):
            score -= 4
            reasons.append("Column values are almost all unique, which makes it look identifier-like instead of target-like.")

        if score <= 0:
            continue

        suggestions.append(
            {
                "column": column,
                "score": score,
                "confidence": _confidence_from_score(score),
                "suggested_problem_type": candidate_problem_type,
                "reason": " ".join(reasons),
                "metadata": {
                    "unique_count": unique_count,
                    "unique_ratio": round(unique_ratio, 3),
                    "is_numeric": bool(is_numeric_dtype(series)),
                    "binary_like": _is_binary_target(series),
                },
            }
        )

    suggestions.sort(key=lambda item: (-item["score"], item["column"].lower()))
    return suggestions[:max_suggestions]


def recommend_ml_approach(
    df: pd.DataFrame,
    target_column: str | None,
    problem_type: str,
    text_columns: list[str],
) -> dict[str, Any]:
    """Recommend a likely ML problem type, target, and starter algorithms."""
    column_types = classify_columns(df, target_column=target_column)
    warnings: list[str] = []
    candidate_text_columns = detect_text_columns(df, target_column=target_column)
    detected_text_column = _looks_like_main_nlp_text_feature(df, candidate_text_columns)
    target_suggestions = suggest_target_columns(df, text_columns=text_columns, max_suggestions=3)
    top_suggestion = target_suggestions[0] if target_suggestions else None
    suggested_target = top_suggestion["column"] if top_suggestion and top_suggestion["score"] >= 5 else None
    suggestion_confidence = top_suggestion["confidence"] if suggested_target else "Low"

    if df.empty or len(df.columns) == 0:
        recommended_problem_type = "Unknown / Needs More Information"
        reason = "The dataset is empty or has no columns, so a reliable ML problem type cannot be inferred."
        algorithms = ALGORITHM_MAP[recommended_problem_type]
        return {
            "recommended_problem_type": recommended_problem_type,
            "suggested_problem_type": recommended_problem_type,
            "reason": reason,
            "problem_type_reason": reason,
            "warnings": warnings,
            "selected_target_column": target_column,
            "suggested_target_column": None,
            "target_detection_confidence": "Low",
            "target_detection_metadata": {"top_suggestions": target_suggestions},
            "target_column": target_column,
            "target_column_used_for_inference": None,
            "detected_text_column": detected_text_column,
            "numeric_columns": column_types["numeric_columns"],
            "categorical_columns": column_types["categorical_columns"],
            "text_columns": column_types["text_columns"],
            "boolean_columns": column_types["boolean_columns"],
            "datetime_columns": column_types["datetime_columns"],
            "id_like_columns": column_types["id_like_columns"],
            "algorithms": algorithms,
        }

    if target_column is not None:
        target_used = target_column
        suggested_target_column = suggested_target
    else:
        target_used = suggested_target
        suggested_target_column = suggested_target

    if problem_type != "Auto-detect":
        recommended_problem_type = problem_type
        reason = f"The user selected '{problem_type}', so that manual override is being respected."
    elif target_used is not None:
        recommended_problem_type, reason = _infer_problem_type_for_target(
            df,
            target_used,
            detected_text_column,
        )
    elif _is_unknown_dataset(column_types, len(df)):
        recommended_problem_type = "Unknown / Needs More Information"
        reason = "The dataset appears to contain mostly identifiers or insufficient predictive features, so a reliable ML problem type cannot be inferred yet."
    else:
        recommended_problem_type = "Clustering"
        reason = "No target column was selected or confidently detected, so unsupervised grouping is the safest default."

    if target_column is None and suggested_target_column is not None:
        warnings.append(
            f"No target column was selected manually. Suggested target column: '{suggested_target_column}'."
        )
    elif target_column is None and recommended_problem_type in {"Clustering", "Unknown / Needs More Information"}:
        warnings.append(
            "No target column was selected manually, so the app could not confirm a supervised label."
        )

    algorithms = ALGORITHM_MAP.get(
        recommended_problem_type,
        ALGORITHM_MAP["Unknown / Needs More Information"],
    )

    return {
        "recommended_problem_type": recommended_problem_type,
        "suggested_problem_type": recommended_problem_type,
        "reason": reason,
        "problem_type_reason": reason,
        "warnings": warnings,
        "selected_target_column": target_column,
        "suggested_target_column": suggested_target_column,
        "target_detection_confidence": suggestion_confidence if suggested_target_column else "Low",
        "target_detection_metadata": {
            "top_suggestions": target_suggestions,
            "manual_override_used": target_column is not None or problem_type != "Auto-detect",
        },
        "target_column": target_column,
        "target_column_used_for_inference": target_used,
        "detected_text_column": detected_text_column,
        "numeric_columns": column_types["numeric_columns"],
        "categorical_columns": column_types["categorical_columns"],
        "text_columns": column_types["text_columns"],
        "boolean_columns": column_types["boolean_columns"],
        "datetime_columns": column_types["datetime_columns"],
        "id_like_columns": column_types["id_like_columns"],
        "algorithms": algorithms,
    }
