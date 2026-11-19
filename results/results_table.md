| Model | Tier | Accuracy | Macro-F1 | Hard-cases acc. | Sec / 1k preds | Cost / 1k preds |
|---|---|---|---|---|---|---|
| Majority baseline | - | 0.503 | 0.335 | 0.50 | <0.1 | $0 |
| VADER | 1 Rule-based | 0.727 | 0.715 | 0.33 | 4.7 | $0 |
| TextBlob | 1 Rule-based | 0.710 | 0.687 | 0.40 | 2.1 | $0 |
| NLTK + Naive Bayes | 2 Classic ML | 0.887 | 0.887 | 0.70 | 4.2 | $0 |
| LLM zero-shot (Haiku 5.5) | 3 LLM | 0.957 | 0.957 | 1.00 | 1,560.3 | $0.075 |
