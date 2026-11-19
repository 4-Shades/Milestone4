"""NLTK preprocessing for the Tier 2 (TF-IDF + Naive Bayes) model."""
import re
from functools import lru_cache

import nltk
from nltk.sentiment.util import mark_negation
from nltk.tokenize import TweetTokenizer

_TOKENIZER = TweetTokenizer(preserve_case=False, reduce_len=True)
_NEGATION_RE = re.compile(r"^(?:no|nor|not|never|none|nothing|nobody|nowhere|neither)$|n't$")
_KEEP_PUNCT = {"!", "?"}


@lru_cache(maxsize=1)
def stopwords() -> frozenset:
    """NLTK English stopwords minus negations ("not", "no", "don't", ...), which carry sentiment."""
    try:
        words = nltk.corpus.stopwords.words("english")
    except LookupError:
        nltk.download("stopwords", quiet=True)
        words = nltk.corpus.stopwords.words("english")
    return frozenset(w for w in words if not _NEGATION_RE.search(w))


def _is_kept_token(tok: str) -> bool:
    """Keep words, numbers, emoji and !/?; drop other pure punctuation."""
    return tok in _KEEP_PUNCT or any(ch.isalnum() or ord(ch) > 0x2000 for ch in tok)


def preprocess(text: str, negation: bool = True) -> str:
    """Tokenise -> (optional) mark negation scope -> drop stopwords/punctuation -> join.

    With ``negation=True``, tokens after a negation word up to the next clause
    punctuation get a ``_NEG`` suffix ("not good ." -> "not good_NEG ."), so
    "good" and "good_NEG" become different features.
    """
    tokens = _TOKENIZER.tokenize(text)
    if negation:
        tokens = mark_negation(tokens)
    stop = stopwords()
    return " ".join(
        t for t in tokens
        if t.removesuffix("_NEG") not in stop and _is_kept_token(t.removesuffix("_NEG"))
    )


def preprocess_many(texts, negation: bool = True) -> list[str]:
    return [preprocess(t, negation) for t in texts]
