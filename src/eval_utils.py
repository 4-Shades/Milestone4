"""Shared evaluation harness: every model is scored the same way.

Usage in a notebook::

    from src.eval_utils import evaluate_model
    result = evaluate_model("VADER", predict_fn)            # times the run itself
    result = evaluate_model("LLM zero-shot", predict_fn,
                            sec_per_1k=..., cost_per_1k=...)  # LLM passes logged values

``predict_fn`` takes a list of texts and returns a list of labels from ``LABELS``.
"""
import re
import statistics
import time
from datetime import datetime, timezone

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from src.data_utils import LABELS, PROJECT_ROOT, load_hard_cases, load_split

RESULTS_DIR = PROJECT_ROOT / "results"
CHART_DIR = RESULTS_DIR / "charts"
PREDS_DIR = RESULTS_DIR / "preds"
METRICS_PATH = RESULTS_DIR / "metrics.csv"

METRIC_COLUMNS = [
    "model", "accuracy", "macro_f1", "hard_cases_acc", "sec_per_1k", "cost_per_1k",
    "n_test", "n_hard", "evaluated_at", "notes",
]

# Chart tokens (dataviz reference palette): one-hue blue sequential ramp + ink colours
_INK_PRIMARY, _INK_SECONDARY = "#0b0b0b", "#52514e"
_SEQ_BLUE = LinearSegmentedColormap.from_list(
    "seq_blue", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
)


def slugify(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def score(y_true, y_pred) -> dict:
    """Accuracy and macro-F1 (the main metric)."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", labels=LABELS, zero_division=0),
    }


def _check_preds(preds, n_expected: int, where: str) -> list:
    preds = list(preds)
    if len(preds) != n_expected:
        raise ValueError(f"{where}: got {len(preds)} predictions for {n_expected} texts")
    invalid = sorted(set(preds) - set(LABELS))
    if invalid:
        raise ValueError(
            f"{where}: predictions outside {LABELS}: {invalid}. "
            "Map unparseable outputs to a label inside predict_fn and report the rate."
        )
    return preds


def time_predictions(predict_fn, texts, repeats: int = 3):
    """Run ``predict_fn`` ``repeats`` times. Returns (preds, median seconds per 1,000 texts)."""
    timings, preds = [], None
    for _ in range(repeats):
        start = time.perf_counter()
        preds = predict_fn(texts)
        timings.append(time.perf_counter() - start)
    return preds, statistics.median(timings) / len(texts) * 1000


def plot_confusion_matrix(y_true, y_pred, model_name: str, save: bool = True):
    """Count heatmap with row percentages. Rows are true labels, columns are predictions."""
    cm = confusion_matrix(y_true, y_pred, labels=LABELS)
    row_pct = cm / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(4.6, 4.0))
    ax.imshow(cm, cmap=_SEQ_BLUE, vmin=0, vmax=max(cm.sum(axis=1)))
    for i in range(len(LABELS)):
        for j in range(len(LABELS)):
            dark_cell = cm[i, j] > 0.55 * max(cm.sum(axis=1))
            ax.text(j, i, f"{cm[i, j]}\n({row_pct[i, j]:.0%})", ha="center", va="center",
                    fontsize=11, color="white" if dark_cell else _INK_PRIMARY)
    ax.set_xticks(range(len(LABELS)), LABELS)
    ax.set_yticks(range(len(LABELS)), LABELS)
    ax.set_xlabel("Predicted", color=_INK_SECONDARY)
    ax.set_ylabel("True", color=_INK_SECONDARY)
    ax.set_title(f"{model_name}: confusion matrix (test300)", loc="left",
                 fontsize=11, color=_INK_PRIMARY)
    ax.tick_params(colors=_INK_SECONDARY, length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout()
    if save:
        CHART_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(CHART_DIR / f"cm_{slugify(model_name)}.png", dpi=150)
    return fig


def record_metrics(row: dict) -> pd.DataFrame:
    """Insert or replace the row for ``row['model']`` in results/metrics.csv."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if METRICS_PATH.exists():
        table = pd.read_csv(METRICS_PATH, keep_default_na=False)
        table = table[table["model"] != row["model"]]
    else:
        table = pd.DataFrame(columns=METRIC_COLUMNS)
    new = pd.DataFrame([{col: row.get(col, "") for col in METRIC_COLUMNS}])
    table = new if table.empty else pd.concat([table, new], ignore_index=True)
    table.to_csv(METRICS_PATH, index=False, lineterminator="\n")
    return table


def evaluate_model(model_name: str, predict_fn, *, sec_per_1k: float | None = None,
                   cost_per_1k: float = 0.0, notes: str = "", repeats: int = 3) -> dict:
    """Score a model on test300 and the hard cases, then save everything.

    Outputs: a row in results/metrics.csv, results/charts/cm_<slug>.png, and
    results/preds/<slug>.csv (per-example predictions for error analysis).
    If ``sec_per_1k`` is None the test300 run is timed here (median of ``repeats``);
    pass a measured value for models whose calls are cached (the LLM).
    """
    test = load_split("test300")
    hard = load_hard_cases()
    test_texts, hard_texts = test["text"].tolist(), hard["text"].tolist()

    if sec_per_1k is None:
        test_preds, sec_per_1k = time_predictions(predict_fn, test_texts, repeats)
    else:
        test_preds = predict_fn(test_texts)
    test_preds = _check_preds(test_preds, len(test), "test300")
    hard_preds = _check_preds(predict_fn(hard_texts), len(hard), "hard cases")

    test_scores = score(test["label"], test_preds)
    hard_acc = accuracy_score(hard["label"], hard_preds)

    row = {
        "model": model_name,
        "accuracy": round(test_scores["accuracy"], 4),
        "macro_f1": round(test_scores["macro_f1"], 4),
        "hard_cases_acc": round(hard_acc, 4),
        "sec_per_1k": round(sec_per_1k, 6),  # fast models take well under 1 ms per 1k
        "cost_per_1k": round(cost_per_1k, 4),
        "n_test": len(test),
        "n_hard": len(hard),
        "evaluated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "notes": notes,
    }
    record_metrics(row)

    PREDS_DIR.mkdir(parents=True, exist_ok=True)
    preds = pd.concat([
        pd.DataFrame({"split": "test300", "id": test.index, "label": test["label"],
                      "pred": test_preds, "category": ""}),
        pd.DataFrame({"split": "hard", "id": hard["id"], "label": hard["label"],
                      "pred": hard_preds, "category": hard["category"]}),
    ], ignore_index=True)
    preds["correct"] = preds["label"] == preds["pred"]
    preds.to_csv(PREDS_DIR / f"{slugify(model_name)}.csv", index=False, lineterminator="\n")

    plot_confusion_matrix(test["label"], test_preds, model_name)
    return {**row, "confusion_matrix": confusion_matrix(test["label"], test_preds, labels=LABELS)}


def hard_case_breakdown(model_name: str) -> pd.DataFrame:
    """Hard-case accuracy per category for a model that has already been evaluated."""
    preds = pd.read_csv(PREDS_DIR / f"{slugify(model_name)}.csv", keep_default_na=False)
    hard = preds[preds["split"] == "hard"]
    return (hard.groupby("category")["correct"].agg(["mean", "sum", "count"])
                .rename(columns={"mean": "accuracy", "sum": "correct", "count": "n"})
                .sort_values("accuracy"))
