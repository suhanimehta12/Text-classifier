"""
preprocess.py
-------------
Text cleaning, normalization, and TF-IDF feature extraction.
"""

import re
import string
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

# Download required NLTK assets (safe to call multiple times)
for resource in ["punkt", "stopwords", "wordnet", "punkt_tab"]:
    nltk.download(resource, quiet=True)

STOP_WORDS = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()


# ---------------------------------------------------------------------------
# Text Cleaning
# ---------------------------------------------------------------------------

def clean_text(text: str) -> str:
    """
    Normalize a raw text string:
      1. Lowercase
      2. Remove URLs, mentions, hashtags
      3. Remove punctuation & digits
      4. Tokenize → remove stopwords → lemmatize
      5. Rejoin tokens
    """
    if not isinstance(text, str):
        return ""

    # Lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r"http\S+|www\.\S+", "", text)

    # Remove mentions / hashtags
    text = re.sub(r"[@#]\w+", "", text)

    # Remove punctuation and digits
    text = text.translate(str.maketrans("", "", string.punctuation + string.digits))

    # Tokenize
    tokens = word_tokenize(text)

    # Remove stopwords and short tokens, then lemmatize
    tokens = [
        lemmatizer.lemmatize(tok)
        for tok in tokens
        if tok not in STOP_WORDS and len(tok) > 2
    ]

    return " ".join(tokens)


def preprocess_dataframe(df: pd.DataFrame, text_col: str = "text", label_col: str = "label") -> pd.DataFrame:
    """Apply clean_text to a DataFrame and drop nulls."""
    df = df[[text_col, label_col]].dropna().copy()
    df["clean_text"] = df[text_col].apply(clean_text)
    df = df[df["clean_text"].str.strip() != ""]  # drop empty strings
    return df


# ---------------------------------------------------------------------------
# Feature Extraction
# ---------------------------------------------------------------------------

def build_vectorizer(
    max_features: int = 10_000,
    ngram_range: tuple = (1, 2),
    min_df: int = 2,
) -> TfidfVectorizer:
    """
    Return a configured TF-IDF vectorizer.
    Bigrams (1,2) capture short phrases like 'machine learning'.
    """
    return TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=min_df,
        sublinear_tf=True,   # log-scale TF dampens very frequent terms
    )


# ---------------------------------------------------------------------------
# Train / Test Split Helper
# ---------------------------------------------------------------------------

def split_data(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """Split into train/test sets and return X_train, X_test, y_train, y_test."""
    X = df["clean_text"]
    y = df["label"]
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
