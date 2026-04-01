
"""
advanced_feature_extraction.py

Legacy compatibility wrapper.

This file used to own both preprocessing and pair-feature logic.
That caused duplicated cleanup rules and mixed responsibilities.

The recommended workflow is now:
1. run 02_preprocessing.ipynb;
2. use feature_engineering.py on the saved preprocessed dataset.

These wrappers keep older imports working while delegating the real work
to the new feature_engineering module.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from feature_engineering import (
    build_basic_pair_features,
    build_fuzzy_features,
    load_preprocessed_dataset,
)


def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Backward-compatible wrapper that appends lexical and fuzzy pair features
    to an already preprocessed dataframe.
    """
    df = df.copy().reset_index(drop=True)

    if "q1_norm" not in df.columns or "q2_norm" not in df.columns:
        raise KeyError(
            "This wrapper now expects a preprocessed dataframe with 'q1_norm' and 'q2_norm'."
        )

    basic = build_basic_pair_features(df, q1_col="q1_norm", q2_col="q2_norm")
    fuzzy = build_fuzzy_features(df, q1_col="q1_norm", q2_col="q2_norm")

    return pd.concat([df, basic, fuzzy], axis=1)


def build_nlp_features_train(project_root: Path) -> pd.DataFrame:
    """
    Legacy wrapper that loads the saved preprocessed training set and builds
    lexical + fuzzy pair features.

    Output:
    - data/processed/nlp_features_train.csv
    """
    processed_path = project_root / "data" / "processed" / "nlp_features_train.csv"
    processed_path.parent.mkdir(parents=True, exist_ok=True)

    if processed_path.is_file():
        return pd.read_csv(processed_path)

    df = load_preprocessed_dataset(project_root)

    base_cols = [col for col in ["id", "is_duplicate"] if col in df.columns]
    out = df[base_cols].copy() if base_cols else pd.DataFrame(index=df.index)

    basic = build_basic_pair_features(df, q1_col="q1_norm", q2_col="q2_norm")
    fuzzy = build_fuzzy_features(df, q1_col="q1_norm", q2_col="q2_norm")

    out = pd.concat([out.reset_index(drop=True), basic, fuzzy], axis=1)
    out.to_csv(processed_path, index=False)

    return out
