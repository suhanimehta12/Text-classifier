"""
tests/test_pipeline.py
----------------------
Unit tests for preprocessing, training, and prediction modules.
Run with: pytest tests/ -v
"""

import os
import pytest
import pandas as pd
import joblib

from src.preprocess import clean_text, preprocess_dataframe, build_vectorizer, split_data
from src.train import build_pipeline, train
from src.predict import load_model, predict_text, predict_batch


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_DATA = {
    "text": [
        "Scientists discover new planet using advanced telescope",
        "Team wins championship in overtime thriller",
        "Government passes new tax reform bill",
        "Movie star wins award at film ceremony",
        "New AI chip outperforms all competitors",
        "Athlete breaks world record at Olympics",
        "Parliament debates immigration law changes",
        "Streaming service launches original series",
    ],
    "label": ["tech", "sport", "politics", "entertainment", "tech", "sport", "politics", "entertainment"],
}


@pytest.fixture
def sample_df():
    return pd.DataFrame(SAMPLE_DATA)


@pytest.fixture
def sample_csv(tmp_path, sample_df):
    path = tmp_path / "news.csv"
    sample_df.to_csv(path, index=False)
    return str(path)


@pytest.fixture
def trained_model_path(tmp_path, sample_csv):
    out = str(tmp_path / "model.pkl")
    train(data_path=sample_csv, model_out=out, model_type="logistic")
    return out


# ---------------------------------------------------------------------------
# Preprocessing Tests
# ---------------------------------------------------------------------------

class TestCleanText:
    def test_lowercase(self):
        assert clean_text("HELLO WORLD") == clean_text("hello world")

    def test_removes_url(self):
        result = clean_text("Visit https://example.com for more")
        assert "http" not in result
        assert "example" not in result

    def test_removes_punctuation(self):
        result = clean_text("Hello, world! How are you?")
        assert "," not in result
        assert "!" not in result

    def test_removes_stopwords(self):
        result = clean_text("the cat sat on the mat")
        assert "the" not in result.split()
        assert "on" not in result.split()

    def test_empty_string(self):
        assert clean_text("") == ""

    def test_none_input(self):
        assert clean_text(None) == ""

    def test_returns_string(self):
        assert isinstance(clean_text("Some text here"), str)


class TestPreprocessDataframe:
    def test_adds_clean_text_column(self, sample_df):
        result = preprocess_dataframe(sample_df)
        assert "clean_text" in result.columns

    def test_drops_nulls(self):
        df = pd.DataFrame({"text": ["hello world", None, "foo bar"], "label": ["a", "b", "c"]})
        result = preprocess_dataframe(df)
        assert len(result) == 2

    def test_output_length(self, sample_df):
        result = preprocess_dataframe(sample_df)
        assert len(result) == len(sample_df)


# ---------------------------------------------------------------------------
# Vectorizer Tests
# ---------------------------------------------------------------------------

class TestBuildVectorizer:
    def test_returns_tfidf(self):
        from sklearn.feature_extraction.text import TfidfVectorizer
        vec = build_vectorizer()
        assert isinstance(vec, TfidfVectorizer)

    def test_custom_max_features(self):
        vec = build_vectorizer(max_features=500)
        assert vec.max_features == 500

    def test_fit_transform(self, sample_df):
        df = preprocess_dataframe(sample_df)
        vec = build_vectorizer(min_df=1)
        X = vec.fit_transform(df["clean_text"])
        assert X.shape[0] == len(df)


# ---------------------------------------------------------------------------
# Training Tests
# ---------------------------------------------------------------------------

class TestBuildPipeline:
    @pytest.mark.parametrize("model_type", ["logistic", "naive_bayes", "svm", "random_forest"])
    def test_all_model_types(self, model_type):
        pipeline = build_pipeline(model_type)
        assert pipeline is not None

    def test_invalid_model_raises(self):
        with pytest.raises(ValueError):
            build_pipeline("unknown_model")


class TestTrain:
    def test_creates_model_file(self, sample_csv, tmp_path):
        out = str(tmp_path / "model.pkl")
        train(data_path=sample_csv, model_out=out)
        assert os.path.exists(out)

    def test_returns_dict_keys(self, sample_csv, tmp_path):
        out = str(tmp_path / "model.pkl")
        result = train(data_path=sample_csv, model_out=out)
        for key in ["model_type", "train_acc", "n_classes", "n_train", "n_test"]:
            assert key in result

    def test_accuracy_between_0_and_1(self, sample_csv, tmp_path):
        out = str(tmp_path / "model.pkl")
        result = train(data_path=sample_csv, model_out=out)
        assert 0.0 <= result["train_acc"] <= 1.0


# ---------------------------------------------------------------------------
# Prediction Tests
# ---------------------------------------------------------------------------

class TestPredict:
    def test_predict_returns_label(self, trained_model_path):
        pipeline = load_model(trained_model_path)
        result = predict_text("Scientists build a powerful new AI chip", pipeline)
        assert "label" in result
        assert isinstance(result["label"], str)

    def test_predict_has_text_field(self, trained_model_path):
        pipeline = load_model(trained_model_path)
        text = "The team won the final match"
        result = predict_text(text, pipeline)
        assert result["text"] == text

    def test_predict_batch_length(self, trained_model_path):
        pipeline = load_model(trained_model_path)
        texts = ["Tech news here", "Sports update", "Political debate"]
        results = predict_batch(texts, pipeline)
        assert len(results) == 3

    def test_label_is_known_class(self, trained_model_path):
        pipeline = load_model(trained_model_path)
        result = predict_text("A new film breaks box office records", pipeline)
        assert result["label"] in ["tech", "sport", "politics", "entertainment"]
