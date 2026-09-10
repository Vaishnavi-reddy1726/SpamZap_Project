# SpamZap — Spam Email Classifier (Python)

An end-to-end NLP pipeline that classifies emails/messages as **spam** or
**ham** (not spam), built with NLTK for text preprocessing, scikit-learn's
TF-IDF vectorizer for feature extraction, and a Multinomial Naive Bayes
classifier with Laplace smoothing.

**Test-set performance:** ~98% accuracy, 97% precision, 87% recall on spam.

## How it works

```
raw message
   │
   ▼
clean_text()        lowercase, strip URLs/emails/punctuation/digits
   │
   ▼
tokenize_and_stem()  NLTK word_tokenize → remove stopwords → Porter stemming
   │
   ▼
TF-IDF vectorizer    unigrams + bigrams, up to 5000 features
   │
   ▼
Multinomial Naive Bayes (alpha=0.1 Laplace/additive smoothing)
   │
   ▼
spam / ham + probability
```

Laplace (additive) smoothing is what lets the model handle words it never
saw during training — without it, any unseen word would zero out the entire
probability for a class.

## Project structure

```
spamzap/
├── data/
│   └── sms.tsv                  # labeled dataset (5,572 messages, spam/ham)
├── src/
│   ├── preprocess.py            # text cleaning, tokenization, stemming
│   ├── train.py                 # training + evaluation pipeline
│   └── predict.py               # CLI to classify new messages
├── models/                      # saved model + vectorizer (created by train.py)
│   ├── spamzap_model.joblib
│   └── tfidf_vectorizer.joblib
├── outputs/                     # metrics + confusion matrix (created by train.py)
│   ├── metrics.json
│   ├── classification_report.txt
│   └── confusion_matrix.png
├── main.py                      # convenience CLI entry point
├── requirements.txt
└── README.md
```

## Setup

```bash
pip install -r requirements.txt
```

The first run of `train.py` (or `preprocess.py`) will automatically download
the small NLTK `punkt` tokenizer and `stopwords` corpus if they aren't
already present on your machine.

## Usage

**1. Train the model** (reads `data/sms.tsv`, prints metrics, saves the model):

```bash
python main.py train
# or directly:
python src/train.py
```

**2. Classify a message:**

```bash
python main.py predict "Congratulations! You've won a free cruise, call now!"
# Label   : SPAM
# Spam prob: 96.4%
```

**3. Interactive mode:**

```bash
python main.py predict -i
```

## About the dataset

The included `data/sms.tsv` is the classic **SMS Spam Collection** dataset
(5,572 labeled English SMS/text messages, ~13% spam) — widely used as a
stand-in for email spam classification because the spam/ham language
patterns (promotions, urgency, links, "free", "win", "call now", etc.)
are nearly identical to spam email. To use your own dataset, replace
`data/sms.tsv` with a tab-separated file of `label<TAB>message` rows using
`ham`/`spam` labels, or edit `load_data()` in `src/train.py`.

## Evaluation methodology

- **80/20 stratified train/test split** so the spam/ham ratio is preserved
  in both sets.
- **Accuracy** — overall percentage of correct predictions.
- **Precision** — of messages predicted spam, how many actually were spam
  (controls false positives — i.e., real emails wrongly blocked).
- **Recall** — of actual spam messages, how many were caught (controls
  false negatives — i.e., spam that slips through).
- **Confusion matrix** — full breakdown of TP/FP/TN/FN, saved as
  `outputs/confusion_matrix.png`.

## Notes / possible extensions

- Swap `MultinomialNB` for `ComplementNB` (often stronger on imbalanced
  text data) or a linear SVM for comparison.
- Add `class_weight`-style rebalancing (e.g. oversample spam) to push
  recall higher without sacrificing precision.
- Wrap `predict.py` in a small Flask/FastAPI service for a real-time
  spam-filtering API.
