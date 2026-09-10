"""
preprocess.py
-------------
Text cleaning and tokenization utilities for SpamZap.

Pipeline per message:
    1. Lowercase
    2. Strip URLs, email addresses, and non-alphabetic characters
    3. Tokenize with NLTK's word_tokenize
    4. Remove stopwords (NLTK's English stopword list)
    5. Stem remaining tokens with NLTK's PorterStemmer
    6. Rejoin into a cleaned string ready for TF-IDF vectorization
"""

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

# ---------------------------------------------------------------------------
# One-time NLTK resource setup. If the required corpora aren't present on
# this machine yet, download them (only needs to happen once).
# ---------------------------------------------------------------------------
_REQUIRED_PACKAGES = {
    "tokenizers/punkt": "punkt",
    "tokenizers/punkt_tab": "punkt_tab",
    "corpora/stopwords": "stopwords",
}

for path, pkg_id in _REQUIRED_PACKAGES.items():
    try:
        nltk.data.find(path)
    except LookupError:
        nltk.download(pkg_id, quiet=True)

STOPWORDS = set(stopwords.words("english"))
STEMMER = PorterStemmer()

URL_RE = re.compile(r"http\S+|www\.\S+")
EMAIL_RE = re.compile(r"\S+@\S+")
NON_ALPHA_RE = re.compile(r"[^a-zA-Z\s]")


def clean_text(text: str) -> str:
    """Lowercase + strip URLs/emails/punctuation/numbers."""
    text = text.lower()
    text = URL_RE.sub(" ", text)
    text = EMAIL_RE.sub(" ", text)
    text = NON_ALPHA_RE.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize_and_stem(text: str) -> list[str]:
    """Tokenize, drop stopwords/short tokens, and stem."""
    tokens = word_tokenize(text)
    tokens = [
        STEMMER.stem(tok)
        for tok in tokens
        if tok not in STOPWORDS and len(tok) > 2
    ]
    return tokens


def preprocess(text: str) -> str:
    """Full pipeline: clean -> tokenize -> stem -> rejoin to a string."""
    cleaned = clean_text(text)
    tokens = tokenize_and_stem(cleaned)
    return " ".join(tokens)


if __name__ == "__main__":
    sample = "WINNER!! You have been selected to receive a $900 prize. Call 555-0199 now!! http://claim-prize.biz"
    print("Original :", sample)
    print("Processed:", preprocess(sample))
