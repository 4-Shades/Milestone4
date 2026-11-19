# Prompts

Prompts for the Tier 3 classifier, a zero-shot LLM. The source of truth is `SYSTEM_PROMPTS` in `src/llm.py`. `notebooks/04_llm` checks that the active prompt below matches it word for word.

## Setup

| Setting | Value |
|---|---|
| Model | `claude-haiku-5-5` (Claude Haiku 5.5), $0.10 / $0.50 per 1M input / output tokens |
| Examples in prompt | None (zero-shot) |
| Output | Structured output via JSON schema: `{"label": "positive" \| "negative"}`. The schema allows only these two values. |
| Thinking | Off (`thinking: {"type": "disabled"}`), `max_tokens` 256. In the trial, adaptive thinking gave the same accuracy and Haiku didn't actually think. |
| Sampling | API defaults. `temperature` can't be set on current models, so outputs are not guaranteed to be identical between runs. That is why responses are cached in `results/llm_cache.jsonl`. |
| Unparseable output | Mapped to `positive` (the majority training label). The rate is reported. |

The review is sent as the user message, wrapped in `<review>…</review>` tags.

## v1 (active)

System prompt:

```
You are a sentiment classifier for movie reviews.
Read the review and decide whether the reviewer's overall opinion of the film is positive or negative.
Every review gets exactly one of these two labels. Reply only with the JSON object required by the output schema.
```

Design choices:
- **Deliberately minimal.** There are no hints about sarcasm, negation, plot summaries or comparisons, even though the EDA and the hard-case set name those patterns. Tiers 1 and 2 got no such help, and naming the hard-case categories in the prompt would inflate the hard-case score.
- **The "overall opinion of the film" wording.** It asks for the reviewer's verdict, not the tone of the text. That is the definition of the task, not a hint.
- **No few-shot examples,** which keeps it zero-shot as the guidelines specify.

Prompt development used only a stratified sample of 40 **training** reviews. The test subset and the hard cases were not used.

## Change log

| Version | Date | Change | Train-sample result |
|---|---|---|---|
| v1 | 2026-11-19 | First version, frozen for the test run | 39/40 correct (0.975), 100% parsed, 757 input / 10 output tokens on average, about $0.08 per 1k (`notebooks/04_llm.ipynb` §3–4) |
