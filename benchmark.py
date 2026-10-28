"""Deterministic toy benchmark; no external model calls."""
from pathlib import Path
import argparse
import hashlib
import json
import platform
from baseline import ask, grounded, load_documents, retrieve

ROOT = Path(__file__).parent

def retrieval_metrics(ranked_ids, relevant):
    relevant = set(relevant)
    if not relevant:
        return None  # Unanswerable cases have no retrieval relevance denominator.
    hits = len(set(ranked_ids) & relevant)
    return {'precision': hits / len(ranked_ids) if ranked_ids else 0.0,
            'recall': hits / len(relevant),
            'mrr': next((1 / i for i, x in enumerate(ranked_ids, 1) if x in relevant), 0.0)}

def run(k=3):
    if k < 1:
        raise ValueError('k must be positive')
    docs = load_documents(ROOT / 'data')
    dataset = ROOT / 'cases.jsonl'
    cases = [json.loads(line) for line in dataset.read_text(encoding='utf-8').splitlines() if line]
    rows = []
    for case in cases:
        ranked = [c.id for _, c in retrieve(case['question'], docs, k)]
        result = ask(case['question'], docs, k)
        answered = result['status'] == 'answered'
        faithful = grounded(result['claims'], docs) if answered else None
        correct = (answered and case['expected'] in ' '.join(c['quote'] for c in result['claims'])
                   and any(c['source'] in case['relevant'] for c in result['claims'])) if case['relevant'] else not answered
        rows.append({'id': case['id'], 'slice': case['slice'],
                     'retrieval': retrieval_metrics(ranked, case['relevant']),
                     'answered': answered, 'faithful': faithful, 'correct': correct})
    retrieval = [row['retrieval'] for row in rows if row['retrieval'] is not None]
    answered_rows = [row for row in rows if row['answered']]
    summary = {name + f'@{k}': sum(row[name] for row in retrieval) / len(retrieval)
               for name in ('precision', 'recall', 'mrr')}
    summary.update({'cases': len(rows), 'answerable_cases': len(retrieval),
        'answered_cases': len(answered_rows),
        'accuracy': sum(r['correct'] for r in rows) / len(rows),
        'coverage': len(answered_rows) / len(rows),
        'quote_faithfulness': sum(r['faithful'] for r in answered_rows) / len(answered_rows) if answered_rows else None})
    digest = hashlib.sha256()
    for path in [dataset, ROOT / 'baseline.py', *sorted((ROOT / 'data').glob('*.md'))]:
        digest.update(path.name.encode()); digest.update(path.read_bytes())
    return {'metadata': {'python': platform.python_version(), 'k': k,
            'input_sha256': digest.hexdigest(), 'mode': 'offline-extractive-baseline'},
            'summary': summary, 'rows': rows}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--k', type=int, default=3)
    parser.add_argument('--output', type=Path, default=ROOT / 'results.json')
    args = parser.parse_args()
    report = run(args.k)
    args.output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report['summary'], indent=2))
