"""
text_preprocessing.py

Reusable text preprocessing utilities for the Quora Question Pairs project.

Design goals:
- Keep functions small and composable (EDA / feature engineering can mix-and-match).
- Be explicit with flags (classic ML vs transformer-style minimal cleaning).
- Safe defaults for missing values.

Notes:
- For TF-IDF / BoW, you may enable stopword removal and (optionally) stemming/lemmatization.
- For BERT / transformer embeddings, usually use only normalize_text(minimal) and skip stopwords/stemming.

Dependencies:
- bs4 (BeautifulSoup) is optional; if not available, HTML stripping falls back to regex.
- nltk is used for stopwords, stemming, and lemmatization.
"""

from __future__ import annotations

import re
from typing import Iterable, List, Optional

try:
    from bs4 import BeautifulSoup
except Exception:
    BeautifulSoup = None

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer


import contractions


def strip_html(text: str) -> str:
    """Remove HTML markup while keeping visible text."""
    if not text:
        return ""
    if BeautifulSoup is not None:
        return BeautifulSoup(text, "html.parser").get_text(" ", strip=True)
    return re.sub(r"<[^>]+>", " ", text)


def normalize_text(
    text: Optional[str],
    *,
    lowercase: bool = True,
    expand_contractions: bool = True,
    replace_currency_and_units: bool = True,
    normalize_numbers: bool = True,
    remove_non_alnum: bool = True,
    strip_html_markup: bool = True,
    normalize_whitespace: bool = True,
) -> str:
    """
    Normalize raw text into a cleaned string.

    This function does NOT remove stopwords and does NOT stem/lemmatize by default.
    """
    x = "" if text is None else str(text)

    if lowercase:
        x = x.lower()

    x = x.replace("′", "'").replace("’", "'")

    if expand_contractions:
        x = contractions.fix(x)

    if replace_currency_and_units:
        x = (
            x.replace("%", " percent ")
             .replace("₹", " rupee ")
             .replace("$", " dollar ")
             .replace("€", " euro ")
        )

    if normalize_numbers:
        x = x.replace(",000,000", "m").replace(",000", "k")
        x = re.sub(r"([0-9]+)000000", r"\1m", x)
        x = re.sub(r"([0-9]+)000", r"\1k", x)

    if remove_non_alnum:
        # Keep letters, digits, and whitespace
        x = re.sub(r"[^a-z0-9\s]", " ", x)

    if strip_html_markup:
        x = strip_html(x)

    if normalize_whitespace:
        x = re.sub(r"\s+", " ", x).strip()

    return x


def simple_tokenize(text: str) -> List[str]:
    """Whitespace tokenization. Assumes text already normalized."""
    if not text:
        return []
    return text.split()


def get_stopwords(language: str = "english") -> set:
    """Load stopword set for a language (requires nltk stopwords corpus)."""
    return set(stopwords.words(language))


def remove_stopwords(tokens: Iterable[str], *, language: str = "english") -> List[str]:
    """Remove stopwords from token sequence."""
    sw = get_stopwords(language)
    return [t for t in tokens if t and t not in sw]


def stem_tokens(tokens: Iterable[str]) -> List[str]:
    """Stem tokens using Porter stemmer."""
    stemmer = PorterStemmer()
    return [stemmer.stem(t) for t in tokens if t]


def lemmatize_tokens(tokens: Iterable[str]) -> List[str]:
    """Lemmatize tokens using WordNet lemmatizer."""
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(t) for t in tokens if t]


def preprocess_classic_ml(
    text: Optional[str],
    *,
    lowercase: bool = True,
    remove_stop_words: bool = True,
    use_stemming: bool = False,
    use_lemmatization: bool = False,
    language: str = "english",
) -> str:
    """
    A convenient preprocessing pipeline for classic ML vectorizers (BoW/TF-IDF).

    Returns a single string (tokens joined with spaces).
    """
    x = normalize_text(
        text,
        lowercase=lowercase,
        expand_contractions=True,
        replace_currency_and_units=True,
        normalize_numbers=True,
        remove_non_alnum=True,
        strip_html_markup=True,
        normalize_whitespace=True,
    )

    tokens = simple_tokenize(x)

    if remove_stop_words:
        tokens = remove_stopwords(tokens, language=language)

    if use_lemmatization:
        tokens = lemmatize_tokens(tokens)

    if use_stemming:
        tokens = stem_tokens(tokens)

    return " ".join(tokens)
