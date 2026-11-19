# Milestone 4: Sentiment Analysis from Classic NLP to LLMs

This project classifies IMDB movie reviews as **positive** or **negative** in three ways and compares them on accuracy, macro-F1, a hard-case set, speed and cost:

| Tier | Approach | Tools |
|---|---|---|
| 1. Rule-based | Word-score lexicon, no training | VADER (TextBlob as a second lexicon) |
| 2. Classic ML | TF-IDF features + Naive Bayes | NLTK + scikit-learn |
| 3. LLM | Zero-shot prompt, no training | Claude API (Claude Haiku 5.5) |

**The question:** when is a simple model good enough, and when does an LLM earn its cost?

## Results

Every model is scored on the same frozen, stratified 300-review test subset and the same 30 hand-written hard cases.

| Model | Accuracy | Macro-F1 | Hard-cases acc. | Sec / 1k preds | Cost / 1k preds |
|---|---|---|---|---|---|
| Majority baseline | 0.503 | 0.335 | 0.50 | <0.1 | $0 |
| VADER | 0.727 | 0.715 | 0.33 | 4.7 | $0 |
| TextBlob | 0.710 | 0.687 | 0.40 | 2.1 | $0 |
| NLTK + Naive Bayes | 0.887 | 0.887 | 0.70 | 4.2 | ~$0 |
| LLM zero-shot (Haiku 5.5) | **0.957** | **0.957** | **1.00** | 1,560 | $0.075 |

Speed is measured on a laptop CPU (local timings vary by about ±30% between runs); for the LLM it is sequential API latency. The full analysis is in [`report/report.md`](report/report.md) and [`notebooks/05_compare.ipynb`](notebooks/05_compare.ipynb).

## Setup

Requires Python 3.10+ (developed on 3.14).

```powershell
python -m venv .venv
.venv\Scripts\activate          # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

**Data.** Download the IMDB 50K CSV (see [Dataset](#dataset)) and save it as `data/raw/IMDB-Dataset.csv`. The frozen splits in `data/` are committed. The raw file is needed only to re-verify them in `01_data`.

**API key** (only for new LLM calls). Set `ANTHROPIC_API_KEY` as an environment variable, or copy `.env.example` to `.env` and fill it in. `.env` is git-ignored. Never commit a key.

```powershell
setx ANTHROPIC_API_KEY "your-key-here"   # then open a new terminal
```

## How to run

Run the notebooks in order, in Jupyter / VS Code or headless:

```powershell
python -m nbconvert --to notebook --execute --inplace notebooks/01_data.ipynb
```

| Notebook | What it does | Runtime |
|---|---|---|
| `01_data` | Cleans the data, holds out the test pool, runs EDA on train only, freezes `train` / `test300` (re-runs verify, never overwrite) | ~1 min |
| `02_vader` | Majority baseline, VADER, TextBlob, train-only threshold check | ~1 min |
| `03_classic_ml` | NLTK preprocessing, 5-fold CV grid search on train, Naive Bayes | ~12–15 min |
| `04_llm` | Prompt trial on train reviews, then the test run with Claude Haiku 5.5 | seconds (cached) |
| `05_compare` | Results table, charts, error analysis, significance tests | seconds |

**LLM cache.** Every API response is stored in `results/llm_cache.jsonl`, keyed by model, prompt, schema, thinking setting and review text. Re-running `04_llm` makes **no API calls** and reproduces the scored labels exactly. Delete the cache to query the API again; a full run costs about $0.03. Labels may differ slightly because sampling can't be fixed on current models.

## Layout

```
Milestone4/
  README.md, requirements.txt, prompts.md, .env.example
  src/
    data_utils.py   load / clean data, frozen splits with hash check, hard cases
    text_prep.py    NLTK preprocessing for Naive Bayes
    eval_utils.py   shared scoring: metrics.csv row, confusion matrix, predictions
    llm.py          prompt, Claude call, response cache
  data/        train.csv.gz, test300.csv, split_manifest.json, hard_cases.csv (raw/ git-ignored)
  notebooks/   01_data, 02_vader, 03_classic_ml, 04_llm, 05_compare
  results/     metrics.csv, results_table.md, llm_cache.jsonl, preds/, charts/
  models/      trained Naive Bayes pipeline (git-ignored; rebuilt by 03_classic_ml)
  report/      report.md (+ exported copy)
```

## Dataset

**IMDB Large Movie Review Dataset (50K)**, a binary sentiment task.

| | |
|---|---|
| Source | Maas et al. (2011), *Learning Word Vectors for Sentiment Analysis*, ACL. Original: https://ai.stanford.edu/~amaas/data/sentiment/ ; CSV version listed in the GeeksforGeeks dataset guide (Kaggle "IMDB Dataset of 50K Movie Reviews") |
| License | Released for research use; cite the paper above |
| Size | 50,000 reviews; `positive` 25,000 / `negative` 25,000 |
| Raw columns | `review` -> `text`, `sentiment` -> `label` |
| Known issues | 418 duplicate rows; HTML `<br />` tags (both handled in `01_data`) |

**Frozen splits.** After cleaning and de-duplication (49,578 reviews), `01_data` makes a stratified 80/20 split with seed 42, then draws a stratified 300-review scoring subset from the 20% hold-out.

| Split | File | Rows | negative / positive |
|---|---|---|---|
| Train | `data/train.csv.gz` | 39,662 | 19,757 / 19,905 |
| Scoring subset | `data/test300.csv` | 300 | 149 / 151 |

`data/split_manifest.json` stores a content hash for each split. `src.data_utils.load_split()` refuses a modified file.

**Hard cases.** `data/hard_cases.csv` holds 30 hand-written sentences (15 positive / 15 negative) across 12 failure modes: negation, double negatives, sarcasm, contrast, dark plot summary, domain polarity, comparisons, faint praise, idioms, slang and emoji, rhetorical questions, and ratings. They were written before any model ran. Note: they were written by an LLM (Claude), which may favour Tier 3 (see the report).
