from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split


@dataclass
class Split:
    idx_train: np.ndarray
    idx_val: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray


def clip_probs(proba: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    return np.clip(np.asarray(proba, dtype=float), eps, 1.0 - eps)


def evaluate_predictions(y_true: np.ndarray, proba: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true)
    proba = clip_probs(proba)
    return {
        "log_loss": float(log_loss(y_true, proba)),
        "roc_auc": float(roc_auc_score(y_true, proba)),
    }


def make_split(
    y: np.ndarray,
    *,
    test_size: float = 0.2,
    random_state: int = 42,
    stratify: bool = True,
) -> Split:
    idx = np.arange(len(y))
    strat = y if stratify else None
    idx_train, idx_val = train_test_split(
        idx,
        test_size=test_size,
        random_state=random_state,
        stratify=strat,
    )
    y = np.asarray(y)
    return Split(
        idx_train=idx_train,
        idx_val=idx_val,
        y_train=y[idx_train],
        y_val=y[idx_val],
    )


def baseline_prior(y_train: np.ndarray, y_val: np.ndarray) -> Dict[str, float]:
    p = float(np.clip(np.mean(y_train), 1e-6, 1.0 - 1e-6))
    pred = np.full(shape=len(y_val), fill_value=p, dtype=float)
    return evaluate_predictions(y_val, pred)


def _read_table_with_fallback(csv_path: Path, parquet_path: Optional[Path] = None) -> pd.DataFrame:
    if parquet_path is not None and parquet_path.exists():
        try:
            return pd.read_parquet(parquet_path)
        except Exception:
            pass
    if csv_path.exists():
        return pd.read_csv(csv_path)
    raise FileNotFoundError(f"Could not load dataset from {csv_path} or {parquet_path}.")


def load_preprocessed_dataset(project_root: Path) -> pd.DataFrame:
    processed_dir = Path(project_root) / "data" / "processed"
    return _read_table_with_fallback(
        processed_dir / "quora_preprocessed.csv",
        processed_dir / "quora_preprocessed.parquet",
    )


def load_feature_table(project_root: Path) -> pd.DataFrame:
    processed_dir = Path(project_root) / "data" / "processed"
    feature_df = _read_table_with_fallback(
        processed_dir / "pair_similarity_features.csv",
        processed_dir / "pair_similarity_features.parquet",
    )

    has_bert = any(col.startswith("bert_") for col in feature_df.columns)
    bert_csv = processed_dir / "bert_similarity_features.csv"
    bert_parquet = processed_dir / "bert_similarity_features.parquet"

    if (not has_bert) and (bert_csv.exists() or bert_parquet.exists()):
        bert_df = _read_table_with_fallback(bert_csv, bert_parquet)
        if len(bert_df) != len(feature_df):
            raise ValueError("BERT feature table length does not match pair feature table length.")
        feature_df = pd.concat([feature_df.reset_index(drop=True), bert_df.reset_index(drop=True)], axis=1)
        feature_df = feature_df.loc[:, ~feature_df.columns.duplicated()]

    return feature_df


def select_numeric_feature_columns(
    df: pd.DataFrame,
    *,
    exclude: Sequence[str] = ("id", "is_duplicate"),
) -> list[str]:
    numeric_cols = df.select_dtypes(include=[np.number, "bool"]).columns.tolist()
    return [col for col in numeric_cols if col not in set(exclude)]


def fit_tfidf_concat(
    q1_train: pd.Series,
    q2_train: pd.Series,
    q1_val: pd.Series,
    q2_val: pd.Series,
    *,
    max_features: int = 200000,
    ngram_range: Tuple[int, int] = (1, 2),
    min_df: int = 2,
    sublinear_tf: bool = True,
):
    from scipy import sparse
    from sklearn.feature_extraction.text import TfidfVectorizer

    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=min_df,
        sublinear_tf=sublinear_tf,
    )

    train_corpus = pd.concat([q1_train, q2_train], axis=0).astype(str).tolist()
    vectorizer.fit(train_corpus)

    q1_train_x = vectorizer.transform(q1_train.astype(str).tolist())
    q2_train_x = vectorizer.transform(q2_train.astype(str).tolist())
    q1_val_x = vectorizer.transform(q1_val.astype(str).tolist())
    q2_val_x = vectorizer.transform(q2_val.astype(str).tolist())

    X_train = sparse.hstack([q1_train_x, q2_train_x]).tocsr()
    X_val = sparse.hstack([q1_val_x, q2_val_x]).tocsr()

    return vectorizer, X_train, X_val


def train_logreg(
    X_train,
    y_train: np.ndarray,
    X_val,
    y_val: np.ndarray,
    *,
    C: float = 2.0,
    max_iter: int = 400,
):
    from sklearn.linear_model import LogisticRegression

    model = LogisticRegression(
        C=C,
        max_iter=max_iter,
        solver="lbfgs",
    )
    model.fit(X_train, y_train)
    pred = model.predict_proba(X_val)[:, 1]
    metrics = evaluate_predictions(y_val, pred)
    return model, metrics, clip_probs(pred)


def train_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    *,
    params: Optional[Dict] = None,
    use_gpu: bool = False,
):
    import xgboost as xgb

    base_params = {
        "n_estimators": 700,
        "learning_rate": 0.05,
        "max_depth": 6,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "reg_lambda": 1.0,
        "min_child_weight": 1.0,
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "random_state": 42,
        "tree_method": "hist",
        "n_jobs": -1,
    }
    if params:
        base_params.update(params)

    candidate_params: list[dict] = []
    if use_gpu:
        gpu_new = dict(base_params)
        gpu_new["device"] = "cuda"
        candidate_params.append(gpu_new)

        gpu_old = dict(base_params)
        gpu_old["tree_method"] = "gpu_hist"
        candidate_params.append(gpu_old)

    candidate_params.append(dict(base_params))

    last_error = None
    for candidate in candidate_params:
        try:
            model = xgb.XGBClassifier(**candidate, early_stopping_rounds=50)
            model.fit(
                X_train,
                y_train,
                eval_set=[(X_val, y_val)],
                verbose=False,
            )
            pred = model.predict_proba(X_val)[:, 1]
            metrics = evaluate_predictions(y_val, pred)
            return model, metrics, clip_probs(pred)
        except Exception as exc:
            last_error = exc

    raise RuntimeError(f"Could not train XGBoost model. Last error: {last_error}")


def build_model_comparison_frame(results: Iterable[Tuple[str, Dict[str, float]]]) -> pd.DataFrame:
    rows = []
    for model_name, metrics in results:
        rows.append(
            {
                "model": model_name,
                "log_loss": float(metrics["log_loss"]),
                "roc_auc": float(metrics["roc_auc"]),
            }
        )
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["log_loss", "roc_auc"], ascending=[True, False]).reset_index(drop=True)
    return out
