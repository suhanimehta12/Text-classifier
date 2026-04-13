"""
predict.py
----------
Run inference using a saved pipeline.
Supports single-text and batch (file) prediction.
"""

import joblib
from src.preprocess import clean_text


def load_model(model_path: str):
    """Load a serialized sklearn pipeline from disk."""
    return joblib.load(model_path)


def predict_text(text: str, pipeline) -> dict:
    """
    Classify a single raw text string.

    Returns:
        {
          "text":        original text,
          "clean":       preprocessed text,
          "label":       predicted class,
          "confidence":  probability of top class (if available),
          "all_probs":   {class: prob, ...} (if available)
        }
    """
    clean = clean_text(text)

    label = pipeline.predict([clean])[0]
    result = {"text": text, "clean": clean, "label": label}

    # Probabilities are only available for soft classifiers (not LinearSVC)
    if hasattr(pipeline, "predict_proba"):
        probs = pipeline.predict_proba([clean])[0]
        classes = pipeline.classes_
        prob_map = dict(zip(classes, probs.round(4)))
        result["confidence"] = float(max(probs))
        result["all_probs"] = prob_map
    else:
        result["confidence"] = None
        result["all_probs"] = None

    return result


def predict_batch(texts: list[str], pipeline) -> list[dict]:
    """Classify a list of texts and return results for each."""
    return [predict_text(t, pipeline) for t in texts]


def predict_from_file(file_path: str, pipeline) -> list[dict]:
    """
    Read texts from a plain-text file (one per line) and classify each.
    Lines starting with '#' and blank lines are skipped.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        texts = [
            line.strip()
            for line in f
            if line.strip() and not line.startswith("#")
        ]

    print(f"[predict] Loaded {len(texts)} texts from {file_path}")
    return predict_batch(texts, pipeline)


def format_result(result: dict) -> str:
    """Pretty-print a single prediction result."""
    lines = [
        f"  Text   : {result['text'][:80]}{'...' if len(result['text']) > 80 else ''}",
        f"  Label  : {result['label']}",
    ]
    if result["confidence"] is not None:
        lines.append(f"  Confidence: {result['confidence']:.2%}")
    if result["all_probs"]:
        prob_str = "  | ".join(f"{k}: {v:.2%}" for k, v in result["all_probs"].items())
        lines.append(f"  Probs  : {prob_str}")
    return "\n".join(lines)
