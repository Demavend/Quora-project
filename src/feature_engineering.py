
"""
feature_engineering.py

Reusable feature engineering utilities for the Quora Question Pairs project.

This module assumes that text preprocessing has already been completed.
Its job is to convert cleaned question pairs into numeric pair features.

Main feature groups:
- basic lexical and length features;
- fuzzy string-matching features;
- sparse-vector similarity features based on BoW / TF-IDF.

Notes:
- the preferred input is the saved dataset from 02_preprocessing.ipynb;
- q1_norm / q2_norm are used for lexical and fuzzy features;
- q1_classic / q2_classic are used for sparse vectorizers.
"""

from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from text_preprocessing import get_stopwords

try:
    from rapidfuzz import fuzz
except Exception:  # pragma: no cover - optional dependency
    fuzz = None


@dataclass
class VectorizerBundle:
    vectorizer: Union[CountVectorizer, TfidfVectorizer]
    q1: sparse.csr_matrix
    q2: sparse.csr_matrix


def _as_text_list(texts: Iterable[str]) -> List[str]:
    return ["" if pd.isna(x) else str(x) for x in texts]


def _safe_ratio(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return float(numerator) / float(denominator)


def _tokenize(text: str) -> List[str]:
    if not text:
        return []
    return text.split()


def _bigrams(tokens: Sequence[str]) -> set[tuple[str, str]]:
    return set(zip(tokens, tokens[1:])) if len(tokens) >= 2 else set()


def _longest_common_substring_ratio(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    match = SequenceMatcher(None, a, b).find_longest_match(0, len(a), 0, len(b))
    return _safe_ratio(match.size, min(len(a), len(b)))


def load_preprocessed_dataset(project_root) -> pd.DataFrame:
    """
    Load the preprocessed dataset saved in the previous notebook.

    Expected files:
    - data/processed/quora_preprocessed.parquet
    - or data/processed/quora_preprocessed.csv
    """
    processed_dir = project_root / "data" / "processed"
    parquet_path = processed_dir / "quora_preprocessed.parquet"
    csv_path = processed_dir / "quora_preprocessed.csv"

    if parquet_path.exists():
        df = pd.read_parquet(parquet_path)
    elif csv_path.exists():
        df = pd.read_csv(csv_path)
    else:
        raise FileNotFoundError(
            "Preprocessed dataset was not found. Run 02_preprocessing.ipynb first."
        )

    required = ["q1_norm", "q2_norm", "q1_classic", "q2_classic"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise KeyError(f"Missing columns in the preprocessed dataset: {missing}")

    return df


def build_basic_pair_features(
    df: pd.DataFrame,
    *,
    q1_col: str = "q1_norm",
    q2_col: str = "q2_norm",
    language: str = "english",
) -> pd.DataFrame:
    """
    Build basic lexical and length-based pair features.

    These features stay close to the cleaned text and do not rely on vectorizers.
    """
    stop_words = get_stopwords(language)

    q1_series = df[q1_col].fillna("").astype(str)
    q2_series = df[q2_col].fillna("").astype(str)

    rows = []
    for q1, q2 in zip(q1_series, q2_series):
        q1_tokens = _tokenize(q1)
        q2_tokens = _tokenize(q2)

        q1_set = set(q1_tokens)
        q2_set = set(q2_tokens)

        q1_non_stop = {token for token in q1_tokens if token not in stop_words}
        q2_non_stop = {token for token in q2_tokens if token not in stop_words}

        token_union = q1_set | q2_set
        token_intersection = q1_set & q2_set

        non_stop_union = q1_non_stop | q2_non_stop
        non_stop_intersection = q1_non_stop & q2_non_stop

        q1_bigrams = _bigrams(q1_tokens)
        q2_bigrams = _bigrams(q2_tokens)
        bigram_union = q1_bigrams | q2_bigrams
        bigram_intersection = q1_bigrams & q2_bigrams

        q1_word_count = len(q1_tokens)
        q2_word_count = len(q2_tokens)
        q1_char_count = len(q1)
        q2_char_count = len(q2)

        rows.append(
            {
                "exact_match": int(q1 == q2),
                "q1_word_count": q1_word_count,
                "q2_word_count": q2_word_count,
                "q1_char_count": q1_char_count,
                "q2_char_count": q2_char_count,
                "word_count_diff": abs(q1_word_count - q2_word_count),
                "char_count_diff": abs(q1_char_count - q2_char_count),
                "mean_word_count": (q1_word_count + q2_word_count) / 2.0,
                "mean_char_count": (q1_char_count + q2_char_count) / 2.0,
                "first_word_eq": int(
                    bool(q1_tokens and q2_tokens and q1_tokens[0] == q2_tokens[0])
                ),
                "last_word_eq": int(
                    bool(q1_tokens and q2_tokens and q1_tokens[-1] == q2_tokens[-1])
                ),
                "common_word_count": len(token_intersection),
                "common_non_stopword_count": len(non_stop_intersection),
                "total_unique_word_count": len(token_union),
                "word_jaccard": _safe_ratio(len(token_intersection), len(token_union)),
                "non_stopword_jaccard": _safe_ratio(
                    len(non_stop_intersection), len(non_stop_union)
                ),
                "word_overlap_min": _safe_ratio(
                    len(token_intersection), min(len(q1_set), len(q2_set))
                ),
                "word_overlap_max": _safe_ratio(
                    len(token_intersection), max(len(q1_set), len(q2_set))
                ),
                "bigram_jaccard": _safe_ratio(
                    len(bigram_intersection), len(bigram_union)
                ),
                "word_count_ratio": _safe_ratio(
                    min(q1_word_count, q2_word_count),
                    max(q1_word_count, q2_word_count),
                ),
                "char_count_ratio": _safe_ratio(
                    min(q1_char_count, q2_char_count),
                    max(q1_char_count, q2_char_count),
                ),
            }
        )

    return pd.DataFrame(rows)


def build_fuzzy_features(
    df: pd.DataFrame,
    *,
    q1_col: str = "q1_norm",
    q2_col: str = "q2_norm",
) -> pd.DataFrame:
    """
    Build fuzzy-matching features with rapidfuzz.

    Returned features are scaled in the familiar 0-100 range.
    """
    if fuzz is None:
        raise ImportError(
            "rapidfuzz is required for fuzzy features. Install it before running this step."
        )

    q1_series = df[q1_col].fillna("").astype(str)
    q2_series = df[q2_col].fillna("").astype(str)

    rows = []
    for q1, q2 in zip(q1_series, q2_series):
        rows.append(
            {
                "fuzz_ratio": fuzz.ratio(q1, q2),
                "fuzz_partial_ratio": fuzz.partial_ratio(q1, q2),
                "token_sort_ratio": fuzz.token_sort_ratio(q1, q2),
                "token_set_ratio": fuzz.token_set_ratio(q1, q2),
                "longest_substr_ratio": _longest_common_substring_ratio(q1, q2),
            }
        )

    return pd.DataFrame(rows)


def build_bow(
    q1: Iterable[str],
    q2: Iterable[str],
    *,
    max_features: int = 50000,
    ngram_range: Tuple[int, int] = (1, 2),
    min_df: int = 2,
) -> VectorizerBundle:
    vec = CountVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=min_df,
    )
    q1_list = _as_text_list(q1)
    q2_list = _as_text_list(q2)
    vec.fit(q1_list + q2_list)
    q1_x = vec.transform(q1_list)
    q2_x = vec.transform(q2_list)
    return VectorizerBundle(vec, q1_x.tocsr(), q2_x.tocsr())


def build_tfidf(
    q1: Iterable[str],
    q2: Iterable[str],
    *,
    max_features: int = 50000,
    ngram_range: Tuple[int, int] = (1, 2),
    min_df: int = 2,
    sublinear_tf: bool = True,
) -> VectorizerBundle:
    vec = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=min_df,
        sublinear_tf=sublinear_tf,
    )
    q1_list = _as_text_list(q1)
    q2_list = _as_text_list(q2)
    vec.fit(q1_list + q2_list)
    q1_x = vec.transform(q1_list)
    q2_x = vec.transform(q2_list)
    return VectorizerBundle(vec, q1_x.tocsr(), q2_x.tocsr())


def cosine_sim_sparse(q1: sparse.csr_matrix, q2: sparse.csr_matrix) -> np.ndarray:
    from sklearn.preprocessing import normalize

    q1n = normalize(q1, norm="l2", axis=1, copy=True)
    q2n = normalize(q2, norm="l2", axis=1, copy=True)
    return q1n.multiply(q2n).sum(axis=1).A1


def l2_distance_sparse(q1: sparse.csr_matrix, q2: sparse.csr_matrix) -> np.ndarray:
    diff = q1 - q2
    return np.sqrt(diff.multiply(diff).sum(axis=1)).A1


def l1_distance_sparse(q1: sparse.csr_matrix, q2: sparse.csr_matrix) -> np.ndarray:
    diff = q1 - q2
    return np.abs(diff).sum(axis=1).A1


def cosine_sim_dense(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_norm = np.linalg.norm(a, axis=1)
    b_norm = np.linalg.norm(b, axis=1)
    denominator = (a_norm * b_norm) + 1e-12
    numerator = (a * b).sum(axis=1)
    return numerator / denominator


def l2_distance_dense(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.linalg.norm(a - b, axis=1)


def l1_distance_dense(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.abs(a - b).sum(axis=1)


def build_similarity_frame(
    *,
    cosine: np.ndarray,
    euclidean: np.ndarray,
    manhattan: np.ndarray,
    prefix: str,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            f"{prefix}_cosine": cosine,
            f"{prefix}_euclidean": euclidean,
            f"{prefix}_manhattan": manhattan,
        }
    )


def build_sparse_similarity_features(
    df: pd.DataFrame,
    *,
    q1_col: str = "q1_classic",
    q2_col: str = "q2_classic",
    method: str = "tfidf",
    prefix: Optional[str] = None,
    max_features: int = 50000,
    ngram_range: Tuple[int, int] = (1, 2),
    min_df: int = 2,
    sublinear_tf: bool = True,
) -> pd.DataFrame:
    """
    Build pairwise similarity features from sparse vector representations.
    """
    q1 = df[q1_col].fillna("").astype(str).tolist()
    q2 = df[q2_col].fillna("").astype(str).tolist()

    if method == "bow":
        bundle = build_bow(
            q1,
            q2,
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=min_df,
        )
        prefix = prefix or "bow"
    elif method == "tfidf":
        bundle = build_tfidf(
            q1,
            q2,
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=min_df,
            sublinear_tf=sublinear_tf,
        )
        prefix = prefix or "tfidf"
    else:
        raise ValueError("method must be either 'bow' or 'tfidf'.")

    cosine = cosine_sim_sparse(bundle.q1, bundle.q2)
    euclidean = l2_distance_sparse(bundle.q1, bundle.q2)
    manhattan = l1_distance_sparse(bundle.q1, bundle.q2)

    return build_similarity_frame(
        cosine=cosine,
        euclidean=euclidean,
        manhattan=manhattan,
        prefix=prefix,
    )


def build_feature_table(
    df: pd.DataFrame,
    *,
    include_basic: bool = True,
    include_fuzzy: bool = True,
    include_bow: bool = True,
    include_tfidf: bool = True,
    language: str = "english",
) -> pd.DataFrame:
    """
    Build the main feature table used in downstream modeling.
    """
    pieces = []

    base_cols = []
    if "id" in df.columns:
        base_cols.append("id")
    if "is_duplicate" in df.columns:
        base_cols.append("is_duplicate")

    if base_cols:
        pieces.append(df[base_cols].reset_index(drop=True))

    if include_basic:
        pieces.append(
            build_basic_pair_features(
                df,
                q1_col="q1_norm",
                q2_col="q2_norm",
                language=language,
            ).reset_index(drop=True)
        )

    if include_fuzzy:
        pieces.append(
            build_fuzzy_features(
                df,
                q1_col="q1_norm",
                q2_col="q2_norm",
            ).reset_index(drop=True)
        )

    if include_bow:
        pieces.append(
            build_sparse_similarity_features(
                df,
                q1_col="q1_classic",
                q2_col="q2_classic",
                method="bow",
                prefix="bow",
            ).reset_index(drop=True)
        )

    if include_tfidf:
        pieces.append(
            build_sparse_similarity_features(
                df,
                q1_col="q1_classic",
                q2_col="q2_classic",
                method="tfidf",
                prefix="tfidf",
            ).reset_index(drop=True)
        )

    return pd.concat(pieces, axis=1)


def get_feature_groups() -> dict[str, list[str]]:
    """
    Return convenient feature groups for notebook visualizations.
    """
    return {
        "basic_length": [
            "q1_word_count",
            "q2_word_count",
            "word_count_diff",
            "q1_char_count",
            "q2_char_count",
            "char_count_diff",
        ],
        "basic_overlap": [
            "exact_match",
            "first_word_eq",
            "last_word_eq",
            "common_word_count",
            "word_jaccard",
            "non_stopword_jaccard",
            "bigram_jaccard",
            "word_count_ratio",
            "char_count_ratio",
        ],
        "fuzzy": [
            "fuzz_ratio",
            "fuzz_partial_ratio",
            "token_sort_ratio",
            "token_set_ratio",
            "longest_substr_ratio",
        ],
        "sparse": [
            "bow_cosine",
            "bow_euclidean",
            "bow_manhattan",
            "tfidf_cosine",
            "tfidf_euclidean",
            "tfidf_manhattan",
        ],
    }


def get_numeric_feature_columns(df: pd.DataFrame) -> list[str]:
    return [
        col
        for col in df.columns
        if col not in {"id", "is_duplicate"} and pd.api.types.is_numeric_dtype(df[col])
    ]
