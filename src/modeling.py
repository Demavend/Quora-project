from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split


@dataclass
class Split:
    X_train: any
    X_val: any
    y_train: np.ndarray
    y_val: np.ndarray
    idx_train: np.ndarray
    idx_val: np.ndarray


def make_split(n: int, y: np.ndarray, *, test_size: float = 0.2, random_state: int = 42, stratify: bool = True) -> Split:
    idx = np.arange(n)
    strat = y if stratify else None
    idx_train, idx_val = train_test_split(idx, test_size=test_size, random_state=random_state, stratify=strat)
    return Split(None, None, y[idx_train], y[idx_val], idx_train, idx_val)


def baseline_prior(y_train: np.ndarray, y_val: np.ndarray) -> float:
    p = float(np.clip(y_train.mean(), 1e-6, 1 - 1e-6))
    pred = np.full_like(y_val, p, dtype=float)
    return float(log_loss(y_val, pred))


def build_tfidf_concat(q1: pd.Series, q2: pd.Series, *, max_features: int = 200000, ngram_range: Tuple[int, int] = (1, 2), min_df: int = 2, sublinear_tf: bool = True):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from scipy import sparse
    vec = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range, min_df=min_df, sublinear_tf=sublinear_tf)
    all_text = pd.concat([q1, q2], axis=0).astype(str).tolist()
    vec.fit(all_text)
    q1x = vec.transform(q1.astype(str).tolist())
    q2x = vec.transform(q2.astype(str).tolist())
    X = sparse.hstack([q1x, q2x]).tocsr()
    return vec, X


def train_logreg(X_train, y_train: np.ndarray, X_val, y_val: np.ndarray, *, C: float = 2.0, max_iter: int = 200, n_jobs: int = -1):
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(C=C, max_iter=max_iter, n_jobs=n_jobs, solver="lbfgs")
    clf.fit(X_train, y_train)
    pred = np.clip(clf.predict_proba(X_val)[:, 1], 1e-6, 1 - 1e-6)
    return clf, float(log_loss(y_val, pred))


def train_xgboost(X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray, *, params: Optional[Dict] = None):
    import xgboost as xgb
    default = {
        "n_estimators": 2000,
        "learning_rate": 0.05,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_lambda": 1.0,
        "min_child_weight": 1.0,
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "tree_method": "hist",
    }
    if params:
        default.update(params)
    try:
        model = xgb.XGBClassifier(**default)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False, early_stopping_rounds=100)
    except TypeError:
        model = xgb.XGBClassifier(**default, early_stopping_rounds=100)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    pred = np.clip(model.predict_proba(X_val)[:, 1], 1e-6, 1 - 1e-6)
    return model, float(log_loss(y_val, pred))


def train_lstm_siamese(q1_train: np.ndarray, q2_train: np.ndarray, y_train: np.ndarray, q1_val: np.ndarray, q2_val: np.ndarray, y_val: np.ndarray, *, max_words: int = 100000, max_len: int = 40, emb_dim: int = 128, rnn_units: int = 64, epochs: int = 2, batch_size: int = 512, seed: int = 42):
    import tensorflow as tf
    tf.random.set_seed(seed)
    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    from tensorflow.keras import layers, Model

    tok = Tokenizer(num_words=max_words, oov_token="<unk>")
    tok.fit_on_texts(np.concatenate([q1_train, q2_train]).tolist())

    def seq(x):
        return pad_sequences(tok.texts_to_sequences(x.tolist()), maxlen=max_len, padding="post", truncating="post")

    q1_tr = seq(q1_train)
    q2_tr = seq(q2_train)
    q1_va = seq(q1_val)
    q2_va = seq(q2_val)

    inp1 = layers.Input(shape=(max_len,), dtype="int32")
    inp2 = layers.Input(shape=(max_len,), dtype="int32")

    emb = layers.Embedding(input_dim=max_words, output_dim=emb_dim, mask_zero=True)
    rnn = layers.Bidirectional(layers.GRU(rnn_units))

    v1 = rnn(emb(inp1))
    v2 = rnn(emb(inp2))

    x = layers.Concatenate()([v1, v2, layers.Lambda(lambda t: tf.abs(t[0] - t[1]))([v1, v2])])
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    out = layers.Dense(1, activation="sigmoid")(x)

    model = Model([inp1, inp2], out)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="binary_crossentropy")

    model.fit([q1_tr, q2_tr], y_train, validation_data=([q1_va, q2_va], y_val), epochs=epochs, batch_size=batch_size, verbose=0)

    pred = np.clip(model.predict([q1_va, q2_va], batch_size=batch_size, verbose=0).ravel(), 1e-6, 1 - 1e-6)
    return model, tok, float(log_loss(y_val, pred))


def train_bert_embeddings_logreg(q1: pd.Series, q2: pd.Series, y: np.ndarray, idx_train: np.ndarray, idx_val: np.ndarray, *, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", batch_size: int = 256):
    from sentence_transformers import SentenceTransformer
    from sklearn.linear_model import LogisticRegression

    m = SentenceTransformer(model_name)
    e1 = m.encode(q1.astype(str).tolist(), batch_size=batch_size, show_progress_bar=False, normalize_embeddings=True)
    e2 = m.encode(q2.astype(str).tolist(), batch_size=batch_size, show_progress_bar=False, normalize_embeddings=True)
    X = np.hstack([e1, e2]).astype(np.float32)

    clf = LogisticRegression(max_iter=300, n_jobs=-1, solver="lbfgs")
    clf.fit(X[idx_train], y[idx_train])
    pred = np.clip(clf.predict_proba(X[idx_val])[:, 1], 1e-6, 1 - 1e-6)
    return clf, float(log_loss(y[idx_val], pred))


def train_hf_bert_finetune(q1_train: np.ndarray, q2_train: np.ndarray, y_train: np.ndarray, q1_val: np.ndarray, q2_val: np.ndarray, y_val: np.ndarray, *, model_name: str = "distilbert-base-uncased", max_len: int = 96, batch_size: int = 16, epochs: int = 1, lr: float = 2e-5, seed: int = 42):
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    from torch.utils.data import Dataset, DataLoader

    torch.manual_seed(seed)
    tok = AutoTokenizer.from_pretrained(model_name)

    class PairDS(Dataset):
        def __init__(self, a, b, y):
            self.a = a
            self.b = b
            self.y = y

        def __len__(self):
            return len(self.y)

        def __getitem__(self, i):
            enc = tok(self.a[i], self.b[i], truncation=True, padding="max_length", max_length=max_len, return_tensors="pt")
            item = {k: v.squeeze(0) for k, v in enc.items()}
            item["labels"] = torch.tensor(int(self.y[i]), dtype=torch.long)
            return item

    tr = PairDS(q1_train.tolist(), q2_train.tolist(), y_train)
    va = PairDS(q1_val.tolist(), q2_val.tolist(), y_val)

    dl_tr = DataLoader(tr, batch_size=batch_size, shuffle=True)
    dl_va = DataLoader(va, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = torch.nn.CrossEntropyLoss()

    model.train()
    for _ in range(epochs):
        for batch in dl_tr:
            batch = {k: v.to(device) for k, v in batch.items()}
            opt.zero_grad()
            out = model(**{k: v for k, v in batch.items() if k != "labels"})
            loss = loss_fn(out.logits, batch["labels"])
            loss.backward()
            opt.step()

    model.eval()
    probs = []
    with torch.no_grad():
        for batch in dl_va:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**{k: v for k, v in batch.items() if k != "labels"})
            p = torch.softmax(out.logits, dim=1)[:, 1].detach().cpu().numpy()
            probs.append(p)

    pred = np.clip(np.concatenate(probs), 1e-6, 1 - 1e-6)
    return model, tok, float(log_loss(y_val, pred))
