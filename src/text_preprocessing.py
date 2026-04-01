"""
text_preprocessing.py

Reusable text preprocessing utilities for the Quora Question Pairs project.

This module is intentionally limited to text cleanup and normalization.
It should not contain pairwise similarity features or other feature engineering logic.

Main use cases:
- minimal normalization for general text cleanup;
- classic-ML preprocessing for sparse vectorizers such as BoW / TF-IDF;
- optional comparison of stopword removal, lemmatization, and stemming.

Notes:
- BeautifulSoup is optional; HTML stripping falls back to a regex when unavailable.
- NLTK resources can be prepared with ensure_nltk_resources().
"""

from __future__ import annotations

import html
import re
from functools import lru_cache
from typing import Iterable, List, Optional, Sequence

try:
    from bs4 import BeautifulSoup
except Exception:
    BeautifulSoup = None

import contractions
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer


DEFAULT_NLTK_RESOURCES: Sequence[str] = (
    "stopwords",
    "wordnet",
    "omw-1.4",
)


def ensure_nltk_resources(resources: Sequence[str] = DEFAULT_NLTK_RESOURCES, *, quiet: bool = True) -> None:
    """
    Download required NLTK resources if they are missing.

    This helper is convenient for notebooks where the environment may be fresh.
    """
    lookup_map = {
        "stopwords": "corpora/stopwords",
        "wordnet": "corpora/wordnet",
        "omw-1.4": "corpora/omw-1.4",
    }

    for resource in resources:
        lookup_name = lookup_map.get(resource, resource)
        try:
            nltk.data.find(lookup_name)
        except LookupError:
            nltk.download(resource, quiet=quiet)


def strip_html(text: str) -> str:
    """Remove HTML markup while keeping visible text."""
    if not text:
        return ""

    # Avoid passing plain filenames or ordinary text into BeautifulSoup.
    # If the string does not look like HTML/XML markup, return it as is.
    if "<" not in text and ">" not in text:
        return text

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

    This function keeps word forms intact and does not remove stopwords.
    It is a good default for a lightweight, non-aggressive text version.
    """
    x = "" if text is None else str(text)
    x = html.unescape(x)

    if strip_html_markup:
        x = strip_html(x)

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
        x = re.sub(r"([0-9]+)000000\b", r"\1m", x)
        x = re.sub(r"([0-9]+)000\b", r"\1k", x)

    if remove_non_alnum:
        x = re.sub(r"[^a-z0-9\s]", " ", x)

    if normalize_whitespace:
        x = re.sub(r"\s+", " ", x).strip()

    return x


def simple_tokenize(text: str) -> List[str]:
    """Whitespace tokenization. Assumes text is already normalized."""
    if not text:
        return []
    return text.split()


@lru_cache(maxsize=None)
def get_stopwords(language: str = "english") -> set[str]:
    """Load stopword set for a language and cache the result."""
    ensure_nltk_resources(("stopwords",))
    return set(stopwords.words(language))


def remove_stopwords(tokens: Iterable[str], *, language: str = "english") -> List[str]:
    """Remove stopwords from a token sequence."""
    sw = get_stopwords(language)
    return [token for token in tokens if token and token not in sw]


def stem_tokens(tokens: Iterable[str]) -> List[str]:
    """Stem tokens using the Porter stemmer."""
    stemmer = PorterStemmer()
    return [stemmer.stem(token) for token in tokens if token]


def lemmatize_tokens(tokens: Iterable[str]) -> List[str]:
    """Lemmatize tokens using the WordNet lemmatizer."""
    ensure_nltk_resources(("wordnet", "omw-1.4"))
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(token) for token in tokens if token]


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
    Preprocess text for classic ML vectorizers such as BoW or TF-IDF.

    The function returns a single string with tokens joined by spaces.

    Rules:
    - stopword removal is optional;
    - lemmatization and stemming are optional;
    - stemming and lemmatization cannot be enabled at the same time.
    """
    if use_stemming and use_lemmatization:
        raise ValueError("Choose either stemming or lemmatization, not both.")

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
