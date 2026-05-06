"""Beginner-friendly helpers for lightweight NLP text cleaning."""

from __future__ import annotations

import re

import pandas as pd

# Small built-in stopword list to avoid external downloads during demos.
ENGLISH_STOPWORDS = {
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
    "that",
    "the",
    "to",
    "was",
    "were",
    "will",
    "with",
}


def clean_text_value(text: str, remove_stopwords: bool = True) -> str:
    """Apply a simple text-cleaning pipeline to one string value."""
    cleaned_text = text.lower()
    cleaned_text = re.sub(r"[^\w\s]", " ", cleaned_text)
    cleaned_text = re.sub(r"\s+", " ", cleaned_text).strip()

    if remove_stopwords and cleaned_text:
        tokens = [token for token in cleaned_text.split() if token not in ENGLISH_STOPWORDS]
        cleaned_text = " ".join(tokens)

    return cleaned_text


def clean_text_columns(
    dataframe: pd.DataFrame,
    text_columns: list[str],
    *,
    remove_stopwords: bool = True,
) -> tuple[pd.DataFrame, list[str]]:
    """Clean the selected text columns and return the updated DataFrame."""
    cleaned = dataframe.copy()
    cleaned_columns: list[str] = []

    for column in text_columns:
        if column not in cleaned.columns:
            continue

        cleaned[column] = cleaned[column].fillna("")
        cleaned[column] = cleaned[column].astype(str).apply(
            lambda value: clean_text_value(value, remove_stopwords=remove_stopwords)
        )
        cleaned_columns.append(column)

    return cleaned, cleaned_columns
