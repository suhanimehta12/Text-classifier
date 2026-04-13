"""
train.py
--------
Train a text classification pipeline and save it to disk.
Supports: logistic regression, naive bayes, svm, random forest.
"""

import os
import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier

from src.preprocess import build_vectorizer, preprocess_dataframe, split_data

# ---------------------------------------------------------------------------
# Model Registry
# ---------------------------------------------------------------------------

MODEL_REGISTRY = {
    "logistic": LogisticRegression(max_iter=1000, C=5.0, solver="lbfgs", multi_class="auto"),
    "naive_bayes": MultinomialNB(alpha=0.1),
    "svm": LinearSVC(max_iter=2000, C=1.0),
    "random_forest": RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=42),
}


def build_pipeline(model_type: str = "logistic") -> Pipeline:
    """
    Create a sklearn Pipeline:
      TF-IDF Vectorizer → Classifier
    Using a Pipeline ensures the vectorizer is only fit on training data,
    preventing data leakage.
    """
    if model_type not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model type '{model_type}'. Choose from: {list(MODEL_REGISTRY)}")

    return Pipeline([
        ("tfidf", build_vectorizer()),
        ("clf",   MODEL_REGISTRY[model_type]),
    ])


def train(
    data_path: str,
    model_out: str,
    model_type: str = "logistic",
    text_col: str = "text",
    label_col: str = "label",
    test_size: float = 0.2,
) -> dict:
    """
    Full training run:
      1. Load CSV
      2. Preprocess text
      3. Train/test split
      4. Fit pipeline
      5. Save model
      6. Return train accuracy

    Returns a dict with keys: model_type, train_acc, n_classes, n_train, n_test
    """
    print(f"[train] Loading data from: {data_path}")
    df = pd.read_csv(data_path)

    print(f"[train] Preprocessing {len(df)} samples...")
    df = preprocess_dataframe(df, text_col=text_col, label_col=label_col)

    X_train, X_test, y_train, y_test = split_data(df, test_size=test_size)
    print(f"[train] Train: {len(X_train)} | Test: {len(X_test)}")

    print(f"[train] Building pipeline with model: {model_type}")
    pipeline = build_pipeline(model_type)

    print("[train] Fitting pipeline...")
    pipeline.fit(X_train, y_train)

    train_acc = pipeline.score(X_train, y_train)
    print(f"[train] Training accuracy: {train_acc:.4f}")

    # Save pipeline to disk
    os.makedirs(os.path.dirname(model_out) or ".", exist_ok=True)
    joblib.dump(pipeline, model_out)
    print(f"[train] Model saved → {model_out}")

    return {
        "model_type": model_type,
        "train_acc": train_acc,
        "n_classes": len(y_train.unique()),
        "n_train": len(X_train),
        "n_test": len(X_test),
    }
