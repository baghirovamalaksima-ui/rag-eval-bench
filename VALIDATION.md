# Local validation

Executed 2026-10-28 on Windows with Python 3.12 and pytest 8.3.5.

`pytest -q`: **7 passed**. `python benchmark.py --k 3` generated `results.json`.

| Metric | Observed value |
|---|---:|
| Cases | 12 |
| Answerable cases | 7 |
| Correct answer/abstention decisions | 9/12 (75%) |
| Answered cases | 8 |
| Precision@3, answerable cases | 0.333333 |
| Recall@3, answerable cases | 1.000000 |
| MRR@3, answerable cases | 0.928571 |
| Quote faithfulness, answered cases | 1.000000 |

Failures: paraphrase, false premise and negation. Recall is artificially forgiving on this five-paragraph toy corpus; even the missed paraphrase has its relevant paragraph in the top three through deterministic zero-score tie ordering. The answering threshold still abstains on that paraphrase. Perfect quotation attribution does not imply perfect relevance or factual correctness. This is an offline extractive baseline, not measured LLM performance.
