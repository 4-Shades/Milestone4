"""Dataset loading and cleaning helpers shared by all notebooks."""
import hashlib
import html
import json
import re
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_PATH = DATA_DIR / "raw" / "IMDB-Dataset.csv"

# Frozen splits written once by notebooks/01_data.ipynb
SPLIT_FILES = {
    "train": DATA_DIR / "train.csv.gz",  # gzip keeps the ~50 MB CSV GitHub-friendly
    "test300": DATA_DIR / "test300.csv",
}
MANIFEST_PATH = DATA_DIR / "split_manifest.json"
HARD_CASES_PATH = DATA_DIR / "hard_cases.csv"

LABELS = ["negative", "positive"]


def load_raw(path: Path = RAW_PATH) -> pd.DataFrame:
    """Load the raw IMDB file as a DataFrame with columns ``text`` and ``label``.

    Labels are lowercase strings from ``LABELS``. No cleaning is applied here;
    duplicates and HTML tags are handled in notebooks/01_data.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Download the IMDB 50K dataset "
            "(see README, Dataset section) and save it there."
        )
    df = pd.read_csv(path).rename(columns={"review": "text", "sentiment": "label"})
    df["label"] = df["label"].str.strip().str.lower()

    unexpected = set(df["label"]) - set(LABELS)
    if unexpected:
        raise ValueError(f"Unexpected labels in raw data: {unexpected}")
    return df[["text", "label"]]


_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """Remove HTML tags (e.g. ``<br />``), decode HTML entities, collapse whitespace.

    Deliberately light: casing, punctuation and emoji are kept because VADER and
    the LLM use them as sentiment signals.
    """
    text = _TAG_RE.sub(" ", text)
    text = html.unescape(text)
    return _WS_RE.sub(" ", text).strip()


def frame_hash(df: pd.DataFrame) -> str:
    """SHA-256 of the ``text``/``label`` content.

    Hashes a canonical CSV string rather than file bytes, so it is unaffected
    by gzip settings or git line-ending conversion.
    """
    canonical = df[["text", "label"]].to_csv(index=False, lineterminator="\n")
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def load_split(name: str) -> pd.DataFrame:
    """Load a frozen split (``"train"`` or ``"test300"``) and verify it against the manifest."""
    if name not in SPLIT_FILES:
        raise KeyError(f"Unknown split {name!r}; choose from {list(SPLIT_FILES)}")
    df = pd.read_csv(SPLIT_FILES[name], keep_default_na=False)
    expected = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))["splits"][name]["sha256"]
    if frame_hash(df) != expected:
        raise ValueError(
            f"{SPLIT_FILES[name].name} does not match split_manifest.json. "
            "The frozen split was modified; restore it instead of regenerating."
        )
    return df


def load_hard_cases(path: Path = HARD_CASES_PATH) -> pd.DataFrame:
    """Load the 30 hand-written hard cases (columns: id, text, label, category, why_hard)."""
    df = pd.read_csv(path, encoding="utf-8", keep_default_na=False)
    if len(df) != 30 or df["id"].duplicated().any():
        raise ValueError(f"Expected 30 unique hard cases, found {len(df)}")
    if not set(df["label"]) <= set(LABELS):
        raise ValueError(f"Unexpected labels: {set(df['label']) - set(LABELS)}")
    return df
