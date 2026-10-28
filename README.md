# Reproducible RAG Evaluation Benchmark

A junior portfolio benchmark for retrieval and grounded-answer evaluation. Includes 12 manually authored fictional-data cases, deterministic execution, case-level failures, input hashes, and pytest checks. Created with AI assistance; understand and extend it before presenting it as your own experience.

## Run on Windows (Python 3.11+)

```powershell
py -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
.\.venv\Scripts\python benchmark.py --k 3
.\.venv\Scripts\python -m pytest -q
```

`results.json` contains the executed report. Run with `--k 1 --output results-k1.json` to compare retrieval depth. There is no paid service or API key.

## Metrics

- Precision@k: relevant retrieved paragraphs / returned paragraphs.
- Recall@k: relevant retrieved paragraphs / labeled relevant paragraphs.
- MRR@k: reciprocal rank of the first relevant result, else zero.
- Quote faithfulness: fraction of answered cases whose quotes match cited source text. Abstentions are excluded; a fully abstaining system gets null, not a perfect score.
- Accuracy: answer contains the labeled expected string and a relevant citation, or correctly abstains for an unanswerable question. This is a deliberately simple proxy, not semantic grading.
- Coverage: answered cases / all cases. Read together with faithfulness to expose excessive refusal.

Retrieval means cover only answerable cases. Ranking is deterministic, including ties. The SHA-256 records dataset, source data and baseline contents; Python version and k are recorded. No random sampling or model judge is used.

## Baseline and honest limitations

`baseline.py` is an explicitly vendored copy of the companion Grounded Document Q&A engine so this repository runs independently. Default benchmark uses extractive answers and does NOT evaluate a running LLM. The optional LLM adapter is present in the baseline but unused by this runner. Mocked adapter tests in the companion project do not prove real-model quality.

The dataset includes direct questions, unknown information, paraphrase, false premises, negation and one query injection attempt. Paraphrase and false-premise failures are expected and reported; do not hide them. A quoted answer can be faithful yet irrelevant or misleading. This tiny synthetic development set is not held out, statistically representative, or evidence of generalization. The injection case is not a security certification. No published paper has been reproduced.

## Next experiments

Add independently labeled questions and document-side malicious instructions; keep a held-out split before tuning. Compare lexical retrieval to embeddings on the same frozen data. Add real LLM runs only with model/version, prompt, latency and raw output records; distinguish quote attribution from semantic faithfulness.

See `VALIDATION.md` for actual local checks.
