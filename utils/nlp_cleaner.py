"""Beginner-friendly helpers for lightweight NLP text cleaning."""

from __future__ import annotations

import re
import string

import pandas as pd

try:
    from nltk.corpus import stopwords
    from nltk.stem import PorterStemmer
except Exception:  # pragma: no cover - fallback when nltk is unavailable
    stopwords = None
    PorterStemmer = None


BUILT_IN_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "has",
    "he",
    "in",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "was",
    "were",
    "will",
    "with",
    "you",
    "your",
    "we",
    "they",
    "them",
    "i",
    "me",
    "my",
    "now",
    "our",
}

TEXT_COLUMN_HINTS = {
    "message",
    "text",
    "review",
    "comment",
    "content",
    "description",
    "sentence",
    "tweet",
    "sms",
}


def _get_stopwords() -> set[str]:
    """Load NLTK stopwords when available, otherwise use a built-in fallback."""
    if stopwords is not None:
        try:
            return set(stopwords.words("english"))
        except Exception:
            return BUILT_IN_STOPWORDS
    return BUILT_IN_STOPWORDS


def detect_text_columns(dataframe: pd.DataFrame, target_column: str | None = None) -> list[str]:
    """Detect likely NLP/free-text columns using names and content heuristics."""
    detected_columns: list[str] = []

    for column in dataframe.columns:
        if column == target_column or str(column).endswith("_original"):
            continue

        series = dataframe[column]
        if not (pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series)):
            continue

        non_null_values = series.dropna().astype(str).str.strip()
        if non_null_values.empty:
            continue

        normalized_name = str(column).lower()
        average_length = non_null_values.str.len().mean()
        max_length = non_null_values.str.len().max()
        unique_ratio = non_null_values.nunique(dropna=True) / max(len(non_null_values), 1)

        name_match = any(hint in normalized_name for hint in TEXT_COLUMN_HINTS)
        content_match = average_length >= 20 or max_length >= 40 or unique_ratio >= 0.5

        if name_match or content_match:
            detected_columns.append(column)

    return detected_columns


def preprocess_text(
    text: str,
    *,
    remove_numbers: bool = True,
    remove_stopwords: bool = True,
    use_stemming: bool = False,
) -> str:
    """Apply reusable NLP preprocessing in a clear beginner-friendly order."""
    cleaned_text = text.lower()

    # Remove URLs so links do not pollute the vocabulary.
    cleaned_text = re.sub(r"https?://\S+|www\.\S+", " ", cleaned_text)

    # Remove email addresses because they are usually identifiers, not content.
    cleaned_text = re.sub(r"\b[\w\.-]+@[\w\.-]+\.\w+\b", " ", cleaned_text)

    # Remove HTML tags that sometimes appear in scraped or exported text.
    cleaned_text = re.sub(r"<.*?>", " ", cleaned_text)

    # Remove punctuation and special symbols to keep the token space cleaner.
    cleaned_text = cleaned_text.translate(str.maketrans("", "", string.punctuation))
    cleaned_text = re.sub(r"[^a-zA-Z0-9\s]", " ", cleaned_text)

    if remove_numbers:
        cleaned_text = re.sub(r"\d+", " ", cleaned_text)

    # Normalize whitespace before token processing.
    cleaned_text = re.sub(r"\s+", " ", cleaned_text).strip()

    tokens = cleaned_text.split()

    if remove_stopwords and tokens:
        stopword_set = _get_stopwords()
        tokens = [token for token in tokens if token not in stopword_set]

    if use_stemming and PorterStemmer is not None:
        stemmer = PorterStemmer()
        tokens = [stemmer.stem(token) for token in tokens]

    return " ".join(tokens)


def clean_text_columns(
    dataframe: pd.DataFrame,
    text_columns: list[str],
    *,
    remove_numbers: bool = True,
    remove_stopwords: bool = True,
    use_stemming: bool = False,
) -> tuple[pd.DataFrame, list[str], list[str], dict[str, dict[str, str]]]:
    """Overwrite text columns with cleaned text and keep original backups."""
    cleaned = dataframe.copy()
    cleaned_columns: list[str] = []
    original_backup_columns: list[str] = []
    before_after_examples: dict[str, dict[str, str]] = {}

    for column in text_columns:
        if column not in cleaned.columns:
            continue

        original_series = cleaned[column].fillna("").astype(str)
        cleaned[f"{column}_original"] = original_series
        cleaned[column] = original_series.apply(
            lambda value: preprocess_text(
                value,
                remove_numbers=remove_numbers,
                remove_stopwords=remove_stopwords,
                use_stemming=use_stemming,
            )
        )

        cleaned_columns.append(column)
        original_backup_columns.append(f"{column}_original")

        for raw_text, cleaned_text in zip(original_series.tolist(), cleaned[column].tolist()):
            if raw_text.strip():
                before_after_examples[column] = {
                    "before": raw_text,
                    "after": cleaned_text,
                }
                break

    return cleaned, cleaned_columns, original_backup_columns, before_after_examples
