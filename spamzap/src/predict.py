"""
predict.py
----------
Load the trained SpamZap model and classify new email/message text.

Usage:
    python src/predict.py "Congratulations! You've won a free cruise, call now!"
    python src/predict.py --interactive
"""

import sys
import argparse
from pathlib import Path

import joblib

sys.path.append(str(Path(__file__).resolve().parent))
from preprocess import preprocess  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models"


def load_artifacts():
    model_path = MODEL_DIR / "spamzap_model.joblib"
    vec_path = MODEL_DIR / "tfidf_vectorizer.joblib"
    if not model_path.exists() or not vec_path.exists():
        raise FileNotFoundError(
            "Trained model not found. Run `python src/train.py` first."
        )
    model = joblib.load(model_path)
    vectorizer = joblib.load(vec_path)
    return model, vectorizer


def classify(text: str, model, vectorizer) -> dict:
    cleaned = preprocess(text)
    X = vectorizer.transform([cleaned])
    pred = model.predict(X)[0]
    proba = model.predict_proba(X)[0]
    return {
        "label": "SPAM" if pred == 1 else "HAM",
        "spam_probability": round(float(proba[1]), 4),
        "ham_probability": round(float(proba[0]), 4),
    }


def main():
    parser = argparse.ArgumentParser(description="SpamZap email classifier")
    parser.add_argument("text", nargs="?", help="Email/message text to classify")
    parser.add_argument(
        "--interactive", "-i", action="store_true", help="Interactive prompt mode"
    )
    args = parser.parse_args()

    model, vectorizer = load_artifacts()

    if args.interactive or not args.text:
        print("SpamZap interactive mode. Type a message and press Enter.")
        print("(Ctrl+C or empty line to quit)\n")
        while True:
            try:
                text = input(">> ").strip()
            except (KeyboardInterrupt, EOFError):
                print("\nGoodbye!")
                break
            if not text:
                break
            result = classify(text, model, vectorizer)
            print(f"  -> {result['label']}  "
                  f"(spam prob: {result['spam_probability']:.1%}, "
                  f"ham prob: {result['ham_probability']:.1%})\n")
    else:
        result = classify(args.text, model, vectorizer)
        print(f"Message : {args.text}")
        print(f"Label   : {result['label']}")
        print(f"Spam prob: {result['spam_probability']:.1%}")
        print(f"Ham prob : {result['ham_probability']:.1%}")


if __name__ == "__main__":
    main()
