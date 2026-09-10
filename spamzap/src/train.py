"""
train.py
--------
Trains SpamZap: an end-to-end NLP spam classifier.

Steps:
    1. Load labeled email/SMS data
    2. Clean + tokenize + stem every message (preprocess.py)
    3. Vectorize with TF-IDF
    4. Train a Multinomial Naive Bayes classifier (Laplace/additive smoothing)
    5. Evaluate on a held-out test set (accuracy, precision, recall, F1,
       confusion matrix) and save all artifacts to disk.
"""

import json
import sys
import time
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
)

sys.path.append(str(Path(__file__).resolve().parent))
from preprocess import preprocess  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "sms.tsv"
MODEL_DIR = ROOT / "models"
OUTPUT_DIR = ROOT / "outputs"
MODEL_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)


def load_data() -> pd.DataFrame:
    """Load the labeled dataset (label<TAB>message, one per line)."""
    df = pd.read_csv(DATA_PATH, sep="\t", header=None, names=["label", "message"])
    df = df.dropna(subset=["message"])
    df["label_num"] = df["label"].map({"ham": 0, "spam": 1})
    return df


def main():
    t0 = time.time()

    print("[1/6] Loading dataset...")
    df = load_data()
    print(f"      {len(df)} messages loaded "
          f"({(df.label == 'spam').sum()} spam / {(df.label == 'ham').sum()} ham)")

    print("[2/6] Cleaning + tokenizing + stemming (NLTK)...")
    df["clean_message"] = df["message"].apply(preprocess)

    print("[3/6] Splitting train/test (80/20, stratified)...")
    X_train, X_test, y_train, y_test = train_test_split(
        df["clean_message"],
        df["label_num"],
        test_size=0.2,
        random_state=42,
        stratify=df["label_num"],
    )
    print(f"      Train: {len(X_train)}  |  Test: {len(X_test)}")

    print("[4/6] Vectorizing with TF-IDF...")
    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        min_df=1,
    )
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    print(f"      Vocabulary size: {len(vectorizer.vocabulary_)}")

    print("[5/6] Training Multinomial Naive Bayes (Laplace/additive smoothing, alpha=0.1)...")
    # alpha controls Laplace (additive) smoothing strength: it adds a fictitious
    # count to every word so unseen vocabulary never produces a zero probability.
    model = MultinomialNB(alpha=0.1)
    model.fit(X_train_tfidf, y_train)

    print("[6/6] Evaluating on held-out test set...")
    y_pred = model.predict(X_test_tfidf)
    y_proba = model.predict_proba(X_test_tfidf)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=["ham", "spam"])

    print("\n================ RESULTS ================")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1 score : {f1:.4f}")
    print(f"ROC AUC  : {auc:.4f}")
    print("\nConfusion matrix ([[TN, FP], [FN, TP]]):")
    print(cm)
    print("\nClassification report:")
    print(report)
    print(f"\nDone in {time.time() - t0:.1f}s")

    # --- Persist artifacts -------------------------------------------------
    joblib.dump(model, MODEL_DIR / "spamzap_model.joblib")
    joblib.dump(vectorizer, MODEL_DIR / "tfidf_vectorizer.joblib")

    metrics = {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "roc_auc": auc,
        "confusion_matrix": cm.tolist(),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "vocab_size": len(vectorizer.vocabulary_),
    }
    with open(OUTPUT_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    with open(OUTPUT_DIR / "classification_report.txt", "w") as f:
        f.write(report)

    # --- Confusion matrix plot ---------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns

        plt.figure(figsize=(5, 4))
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["ham", "spam"], yticklabels=["ham", "spam"],
        )
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title(f"SpamZap Confusion Matrix (acc={acc:.2%})")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "confusion_matrix.png", dpi=150)
        plt.close()
        print(f"Saved confusion matrix plot to {OUTPUT_DIR / 'confusion_matrix.png'}")
    except Exception as e:
        print(f"(Skipped plot generation: {e})")

    print(f"\nModel + vectorizer saved to {MODEL_DIR}/")
    print(f"Metrics + report saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
