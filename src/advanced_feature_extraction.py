import re
from pathlib import Path
from typing import List

import pandas as pd

from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from fuzzywuzzy import fuzz
import distance


SAFE_DIV = 0.0001
STOP_WORDS = set(stopwords.words("english"))


def preprocess(text: str) -> str:
    """
    Basic text normalization used before advanced feature extraction.

    Steps:
    - Lowercase conversion
    - Normalization of common contractions and symbols
    - Replacing large numeric groups with 'k' / 'm' suffixes
    - Removing non-word characters
    - Porter stemming
    - Removing any HTML artifacts via BeautifulSoup

    Notes:
    - This is a lightweight preprocessing pipeline. It does not perform tokenization,
      lemmatization, or stopword removal here (stopwords are handled later in token features).
    """
    x = str(text).lower()

    x = (
        x.replace(",000,000", "m")
         .replace(",000", "k")
         .replace("′", "'")
         .replace("’", "'")
         .replace("won't", "will not")
         .replace("cannot", "can not")
         .replace("can't", "can not")
         .replace("n't", " not")
         .replace("what's", "what is")
         .replace("it's", "it is")
         .replace("'ve", " have")
         .replace("i'm", "i am")
         .replace("'re", " are")
         .replace("he's", "he is")
         .replace("she's", "she is")
         .replace("'s", " own")
         .replace("%", " percent ")
         .replace("₹", " rupee ")
         .replace("$", " dollar ")
         .replace("€", " euro ")
         .replace("'ll", " will")
    )

    x = re.sub(r"([0-9]+)000000", r"\1m", x)
    x = re.sub(r"([0-9]+)000", r"\1k", x)

    pattern = re.compile(r"\W")
    x = re.sub(pattern, " ", x)

    porter = PorterStemmer()
    x = porter.stem(x)

    x = BeautifulSoup(x, "html.parser").get_text()

    return x


def get_token_features(q1: str, q2: str) -> List[float]:
    """
    Compute token-level overlap and length features for a pair of questions.

    The function returns a list of 10 numeric features:

    [0] cwc_min:
        Common non-stopword count normalized by the minimum non-stopword set size.
        High values indicate strong overlap relative to the shorter question.

    [1] cwc_max:
        Common non-stopword count normalized by the maximum non-stopword set size.
        High values indicate strong overlap even relative to the longer question.

    [2] csc_min:
        Common stopword count normalized by the minimum stopword set size.

    [3] csc_max:
        Common stopword count normalized by the maximum stopword set size.

    [4] ctc_min:
        Common token count normalized by the minimum token count.

    [5] ctc_max:
        Common token count normalized by the maximum token count.

    [6] last_word_eq:
        1 if the last tokens are identical, else 0.

    [7] first_word_eq:
        1 if the first tokens are identical, else 0.

    [8] abs_len_diff:
        Absolute difference in the number of tokens.

    [9] mean_len:
        Average number of tokens across the two questions.

    SAFE_DIV is used to avoid division by zero when one side is empty.
    """
    token_features = [0.0] * 10

    q1_tokens = q1.split()
    q2_tokens = q2.split()

    if len(q1_tokens) == 0 or len(q2_tokens) == 0:
        return token_features

    q1_words = {w for w in q1_tokens if w not in STOP_WORDS}
    q2_words = {w for w in q2_tokens if w not in STOP_WORDS}

    q1_stops = {w for w in q1_tokens if w in STOP_WORDS}
    q2_stops = {w for w in q2_tokens if w in STOP_WORDS}

    common_word_count = len(q1_words.intersection(q2_words))
    common_stop_count = len(q1_stops.intersection(q2_stops))
    common_token_count = len(set(q1_tokens).intersection(set(q2_tokens)))

    token_features[0] = common_word_count / (min(len(q1_words), len(q2_words)) + SAFE_DIV)
    token_features[1] = common_word_count / (max(len(q1_words), len(q2_words)) + SAFE_DIV)
    token_features[2] = common_stop_count / (min(len(q1_stops), len(q2_stops)) + SAFE_DIV)
    token_features[3] = common_stop_count / (max(len(q1_stops), len(q2_stops)) + SAFE_DIV)
    token_features[4] = common_token_count / (min(len(q1_tokens), len(q2_tokens)) + SAFE_DIV)
    token_features[5] = common_token_count / (max(len(q1_tokens), len(q2_tokens)) + SAFE_DIV)

    token_features[6] = int(q1_tokens[-1] == q2_tokens[-1])
    token_features[7] = int(q1_tokens[0] == q2_tokens[0])

    token_features[8] = abs(len(q1_tokens) - len(q2_tokens))
    token_features[9] = (len(q1_tokens) + len(q2_tokens)) / 2.0

    return token_features


def get_longest_substr_ratio(a: str, b: str) -> float:
    """
    Compute the ratio of the longest common substring length to the length of the shorter string.

    - If no common substring is found, return 0.
    - Otherwise, return: len(longest_common_substring) / (min(len(a), len(b)) + 1)

    The '+1' prevents division by zero and slightly regularizes very short strings.
    """
    substrings = list(distance.lcsubstrings(a, b))
    if len(substrings) == 0:
        return 0.0
    return len(substrings[0]) / (min(len(a), len(b)) + 1.0)


def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract advanced NLP features and append them to the input DataFrame.

    The pipeline includes:
    1) Preprocessing: normalize and clean `question1` and `question2`.
    2) Token overlap features (10 features from `get_token_features`).
    3) Fuzzy string-matching features via fuzzywuzzy:
       - token_set_ratio: overlap-oriented matching robust to token order
       - token_sort_ratio: matching after sorting tokens alphabetically
       - fuzz_ratio (QRatio): basic normalized similarity ratio
       - partial_ratio: similarity for best matching substring segment
       - longest_substr_ratio: custom feature based on longest common substring

    Expected columns in df:
    - question1, question2
    """
    df = df.copy()

    df["question1"] = df["question1"].fillna("").apply(preprocess)
    df["question2"] = df["question2"].fillna("").apply(preprocess)

    token_features = df.apply(lambda x: get_token_features(x["question1"], x["question2"]), axis=1)

    df["cwc_min"] = list(map(lambda x: x[0], token_features))
    df["cwc_max"] = list(map(lambda x: x[1], token_features))
    df["csc_min"] = list(map(lambda x: x[2], token_features))
    df["csc_max"] = list(map(lambda x: x[3], token_features))
    df["ctc_min"] = list(map(lambda x: x[4], token_features))
    df["ctc_max"] = list(map(lambda x: x[5], token_features))
    df["last_word_eq"] = list(map(lambda x: x[6], token_features))
    df["first_word_eq"] = list(map(lambda x: x[7], token_features))
    df["abs_len_diff"] = list(map(lambda x: x[8], token_features))
    df["mean_len"] = list(map(lambda x: x[9], token_features))

    df["token_set_ratio"] = df.apply(lambda x: fuzz.token_set_ratio(x["question1"], x["question2"]), axis=1)
    df["token_sort_ratio"] = df.apply(lambda x: fuzz.token_sort_ratio(x["question1"], x["question2"]), axis=1)
    df["fuzz_ratio"] = df.apply(lambda x: fuzz.QRatio(x["question1"], x["question2"]), axis=1)
    df["fuzz_partial_ratio"] = df.apply(lambda x: fuzz.partial_ratio(x["question1"], x["question2"]), axis=1)
    df["longest_substr_ratio"] = df.apply(lambda x: get_longest_substr_ratio(x["question1"], x["question2"]), axis=1)

    return df


def build_nlp_features_train(project_root: Path) -> pd.DataFrame:
    """
    Load cached NLP features if available; otherwise build them from the raw training file.

    Project structure assumed:
    - data/raw/quora_question_pairs_train.csv.zip
    - data/processed/nlp_features_train.csv

    The function:
    1) Checks for the processed cache.
    2) If missing, reads the raw zip CSV, extracts features, and saves the processed file.
    3) Returns the resulting DataFrame.
    """
    processed_path = project_root / "data" / "processed" / "nlp_features_train.csv"
    processed_path.parent.mkdir(parents=True, exist_ok=True)

    if processed_path.is_file():
        return pd.read_csv(processed_path, encoding="latin-1")

    raw_zip_path = project_root / "data" / "raw" / "quora_question_pairs_train.csv.zip"
    df_train = pd.read_csv(raw_zip_path, compression="zip", encoding="latin-1")

    df_train = extract_features(df_train)
    df_train.to_csv(processed_path, index=False)

    return df_train
