"""Tier 3: zero-shot sentiment classification with the Claude API.

The prompt text lives here (single source of truth) and is mirrored in prompts.md;
notebooks/04_llm checks that the two match.
"""
import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

import anthropic
from dotenv import load_dotenv

from src.data_utils import LABELS, PROJECT_ROOT

# Reads ANTHROPIC_API_KEY from a local .env if present; never overrides a variable
# that is already set, and the key is never printed or written anywhere.
load_dotenv(PROJECT_ROOT / ".env")

CACHE_PATH = PROJECT_ROOT / "results" / "llm_cache.jsonl"

MODEL = "claude-haiku-5-5"
# USD per 1M tokens, Anthropic first-party API, prompts up to 100K tokens (cached 2026-11-03)
PRICE_PER_MTOK = {"claude-haiku-5-5": {"input": 0.10, "output": 0.50}}

PROMPT_VERSION = "v1"
SYSTEM_PROMPTS = {
    "v1": (
        "You are a sentiment classifier for movie reviews.\n"
        "Read the review and decide whether the reviewer's overall opinion of the film "
        "is positive or negative.\n"
        "Every review gets exactly one of these two labels. Reply only with the JSON "
        "object required by the output schema."
    ),
}

# Structured output: the response must be {"label": "positive" | "negative"}
OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {"label": {"type": "string", "enum": LABELS}},
    "required": ["label"],
    "additionalProperties": False,
}

# Used only if a response can't be parsed (refusal, truncation); the rate is reported
FALLBACK_LABEL = "positive"  # majority class in train


def get_client() -> anthropic.Anthropic:
    """Client with extra retries for rate limits / transient server errors."""
    return anthropic.Anthropic(max_retries=5)


def user_message(text: str) -> str:
    return f"<review>\n{text}\n</review>"


def cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    price = PRICE_PER_MTOK[model]
    return (input_tokens * price["input"] + output_tokens * price["output"]) / 1_000_000


def classify(client: anthropic.Anthropic, text: str, *, model: str = MODEL,
             prompt_version: str = PROMPT_VERSION, thinking: bool = False) -> dict:
    """Classify one review. Returns the label plus everything needed for the cost/latency log."""
    params = {
        "model": model,
        "system": SYSTEM_PROMPTS[prompt_version],
        "messages": [{"role": "user", "content": user_message(text)}],
        "output_config": {"format": {"type": "json_schema", "schema": OUTPUT_SCHEMA}},
    }
    if thinking:
        params.update(max_tokens=4000, thinking={"type": "adaptive"})
    else:
        params.update(max_tokens=256, thinking={"type": "disabled"})

    start = time.perf_counter()
    response = client.messages.create(**params)
    latency = time.perf_counter() - start

    raw = next((b.text for b in response.content if b.type == "text"), "")
    label = None
    if response.stop_reason == "end_turn":
        try:
            label = json.loads(raw).get("label")
        except (json.JSONDecodeError, AttributeError):
            label = None
    parsed = label in LABELS

    usage = response.usage
    return {
        "label": label if parsed else FALLBACK_LABEL,
        "parsed": parsed,
        "raw": raw,
        "stop_reason": response.stop_reason,
        "model": response.model,
        "prompt_version": prompt_version,
        "thinking": thinking,
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "latency_s": round(latency, 4),
        "cost_usd": cost_usd(model, usage.input_tokens, usage.output_tokens),
        "request_id": response._request_id,
    }


# ---------------------------------------------------------------------------
# Response cache (results/llm_cache.jsonl): one JSON line per API call.
# The key covers everything that changes the answer, so editing the prompt,
# model or thinking setting automatically triggers fresh calls. Review text is
# stored only as a hash.
# ---------------------------------------------------------------------------

def _sha256(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def cache_key(text: str, *, model: str = MODEL, prompt_version: str = PROMPT_VERSION,
              thinking: bool = False) -> str:
    payload = json.dumps({
        "model": model,
        "system": SYSTEM_PROMPTS[prompt_version],
        "schema": OUTPUT_SCHEMA,
        "thinking": thinking,
        "user": user_message(text),
    }, sort_keys=True)
    return _sha256(payload)


def load_cache(path=CACHE_PATH) -> dict:
    """key -> record. Later lines win if a key appears twice."""
    cache = {}
    if path.exists():
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    record = json.loads(line)
                    cache[record["key"]] = record
    return cache


def classify_cached(client, texts, cache: dict, *, max_workers: int = 4,
                    path=CACHE_PATH, **kwargs) -> list[dict]:
    """Classify ``texts`` using the cache; only misses hit the API.

    Misses run ``max_workers`` at a time. Each new record is appended to the cache
    file as soon as it arrives (from the main thread), so an interrupted run
    resumes where it stopped. Returned records carry ``cached=True/False``.
    """
    keys = [cache_key(t, **kwargs) for t in texts]
    misses = {k: t for k, t in zip(keys, texts) if k not in cache}

    if misses:
        path.parent.mkdir(parents=True, exist_ok=True)
        with ThreadPoolExecutor(max_workers=max_workers) as pool, \
                open(path, "a", encoding="utf-8") as f:
            futures = {pool.submit(classify, client, t, **kwargs): k for k, t in misses.items()}
            for future in as_completed(futures):
                key = futures[future]
                record = {
                    "key": key,
                    "text_sha256": _sha256(misses[key]),
                    "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    **future.result(),
                }
                f.write(json.dumps(record) + "\n")
                f.flush()
                cache[key] = record

    return [{**cache[k], "cached": k not in misses} for k in keys]
