"""
evaluate.py
-----------
Load a trained pipeline and evaluate it on a dataset.
Prints a full classification report and optionally saves a confusion matrix plot.
"""

import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from src.preprocess import preprocess_dataframe, split_data


def evaluate(
    model_path: str,
    data_path: str,
    text_col: str = "text",
    label_col: str = "label",
    test_size: float = 0.2,
    plot_cm: bool = True,
    cm_out: str = "models/confusion_matrix.png",
) -> dict:
    """
    Evaluate a saved pipeline on the test split of a dataset.

    Returns a dict with accuracy and the full classification report string.
    """
    print(f"[evaluate] Loading model from: {model_path}")
    pipeline = joblib.load(model_path)

    print(f"[evaluate] Loading data from: {data_path}")
    df = pd.read_csv(data_path)
    df = preprocess_dataframe(df, text_col=text_col, label_col=label_col)

    _, X_test, _, y_test = split_data(df, test_size=test_size)

    print(f"[evaluate] Predicting on {len(X_test)} samples...")
    y_pred = pipeline.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print(f"\n{'='*50}")
    print(f"  Accuracy: {acc:.4f}")
    print(f"{'='*50}")
    print("\nClassification Report:")
    print(report)

    if plot_cm:
        _plot_confusion_matrix(y_test, y_pred, pipeline.classes_, cm_out)

    return {"accuracy": acc, "report": report}


def _plot_confusion_matrix(y_true, y_pred, labels, out_path: str):
    """Generate and save a heatmap confusion matrix."""
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
    )
    ax.set_title("Confusion Matrix", fontsize=14, fontweight="bold")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    plt.tight_layout()

    import os
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"[evaluate] Confusion matrix saved → {out_path}")
    plt.close(fig)
