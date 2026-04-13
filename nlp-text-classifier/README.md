#  NLP Text Classifier

A modular, CLI-driven text classification pipeline built with **NLTK** and **scikit-learn**. Classifies news headlines/articles into categories like Sports, Tech, Politics, and Entertainment.

---

## Project Structure

```
nlp-text-classifier/
├── data/
│   ├── raw/                  # Raw CSV datasets
│   └── processed/            # Tokenized/cleaned data
├── models/                   # Saved trained models (.pkl)
├── src/
│   ├── __init__.py
│   ├── preprocess.py         # Text cleaning & feature extraction
│   ├── train.py              # Model training
│   ├── evaluate.py           # Metrics & evaluation
│   └── predict.py            # Single/batch prediction
├── tests/
│   └── test_pipeline.py      # Unit tests
├── notebooks/
│   └── exploration.ipynb     # EDA notebook
├── cli.py                    # CLI entry point
├── requirements.txt
└── README.md
```

---

##  Setup

```bash
# 1. Clone the repo
git clone https://github.com/your-username/nlp-text-classifier.git
cd nlp-text-classifier

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"
```

---

##  Usage (CLI)

### Train a model
```bash
python cli.py train --data data/raw/news.csv --model-out models/classifier.pkl
```

### Evaluate on test set
```bash
python cli.py evaluate --data data/raw/news.csv --model models/classifier.pkl
```

### Predict a single text
```bash
python cli.py predict --text "Scientists discover new exoplanet using AI" --model models/classifier.pkl
```

### Predict from a file (batch)
```bash
python cli.py predict --file data/raw/sample_texts.txt --model models/classifier.pkl
```

---

##  Models Available

| Model | Notes |
|-------|-------|
| `logistic` | Fast, strong baseline (default) |
| `naive_bayes` | Classic NLP classifier |
| `svm` | High accuracy, slower training |
| `random_forest` | Ensemble, interpretable |

Select with `--model-type`:
```bash
python cli.py train --data data/raw/news.csv --model-type svm --model-out models/svm.pkl
```

---

##  Evaluation Output

```
Classification Report:
              precision    recall  f1-score   support
   politics       0.91      0.89      0.90       120
      sport       0.95      0.96      0.95       130
       tech       0.88      0.90      0.89       115
entertainment    0.92      0.91      0.91       105

    accuracy                           0.92       470
```

---

## Running Tests

```bash
pytest tests/ -v
```

---

##  Requirements

- Python 3.8+
- scikit-learn
- nltk
- pandas
- numpy
- joblib
- click

---

## 📄 License

MIT
