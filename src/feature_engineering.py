from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse


@dataclass
class VectorizerBundle:
    vectorizer: Union[CountVectorizer, TfidfVectorizer]
    q1: sparse.csr_matrix
    q2: sparse.csr_matrix


def build_bow(
    q1: Iterable[str],
    q2: Iterable[str],
    *,
    max_features: int = 50000,
    ngram_range: Tuple[int, int] = (1, 2),
    min_df: int = 2,
) -> VectorizerBundle:
    vec = CountVectorizer(max_features=max_features, ngram_range=ngram_range, min_df=min_df)
    all_text = list(q1) + list(q2)
    vec.fit(all_text)
    q1_x = vec.transform(list(q1))
    q2_x = vec.transform(list(q2))
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
    all_text = list(q1) + list(q2)
    vec.fit(all_text)
    q1_x = vec.transform(list(q1))
    q2_x = vec.transform(list(q2))
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
    denom = (a_norm * b_norm) + 1e-12
    num = (a * b).sum(axis=1)
    return num / denom


def l2_distance_dense(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.linalg.norm(a - b, axis=1)


def l1_distance_dense(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.abs(a - b).sum(axis=1)


def _avg_word_vectors(
    texts: Iterable[str],
    *,
    keyed_vectors,
    dim: int,
) -> np.ndarray:
    out = np.zeros((len(list(texts)), dim), dtype=np.float32)
    for i, s in enumerate(list(texts)):
        toks = s.split()
        vecs = [keyed_vectors[w] for w in toks if w in keyed_vectors]
        if vecs:
            out[i] = np.mean(np.vstack(vecs), axis=0)
    return out


def build_glove_avg(
    q1: Iterable[str],
    q2: Iterable[str],
    *,
    glove_name: str = "glove-wiki-gigaword-100",
):
    import gensim.downloader as api
    kv = api.load(glove_name)
    dim = kv.vector_size
    q1_v = _avg_word_vectors(list(q1), keyed_vectors=kv, dim=dim)
    q2_v = _avg_word_vectors(list(q2), keyed_vectors=kv, dim=dim)
    return q1_v, q2_v


def build_word2vec_avg(
    q1: Iterable[str],
    q2: Iterable[str],
    *,
    w2v_name: str = "word2vec-google-news-300",
):
    import gensim.downloader as api
    kv = api.load(w2v_name)
    dim = kv.vector_size
    q1_v = _avg_word_vectors(list(q1), keyed_vectors=kv, dim=dim)
    q2_v = _avg_word_vectors(list(q2), keyed_vectors=kv, dim=dim)
    return q1_v, q2_v


def build_bert_embeddings(
    q1: List[str],
    q2: List[str],
    *,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    batch_size: int = 128,
    normalize: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name)
    q1_e = model.encode(q1, batch_size=batch_size, show_progress_bar=True, normalize_embeddings=normalize)
    q2_e = model.encode(q2, batch_size=batch_size, show_progress_bar=True, normalize_embeddings=normalize)
    return np.asarray(q1_e), np.asarray(q2_e)


def build_similarity_frame(
    ids: Optional[pd.Series],
    *,
    cosine: np.ndarray,
    euclidean: np.ndarray,
    manhattan: np.ndarray,
    prefix: str,
) -> pd.DataFrame:
    df = pd.DataFrame({
        f"{prefix}_cosine": cosine,
        f"{prefix}_euclidean": euclidean,
        f"{prefix}_manhattan": manhattan,
    })
    if ids is not None:
        df.insert(0, "id", ids.values)
    return df
