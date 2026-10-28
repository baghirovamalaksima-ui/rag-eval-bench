import pytest
from benchmark import retrieval_metrics, run

def test_known_metrics():
    metrics = retrieval_metrics(['b', 'a', 'c'], ['a', 'd'])
    assert metrics == {'precision': 1/3, 'recall': .5, 'mrr': .5}

def test_no_relevant():
    assert retrieval_metrics(['a'], []) is None

def test_empty_ranking():
    assert retrieval_metrics([], ['a']) == {'precision': 0, 'recall': 0, 'mrr': 0}

def test_reproducible():
    assert run() == run()

def test_failure_cases_remain_visible():
    rows = {r['id']: r for r in run()['rows']}
    assert not rows['paraphrase']['correct']
    assert not rows['false-premise']['correct']
    assert rows['unknown']['correct']

def test_counts_and_faithfulness():
    result = run()
    assert result['summary']['cases'] == 12
    assert result['summary']['quote_faithfulness'] == 1
    assert result['summary']['accuracy'] < 1

def test_invalid_k():
    with pytest.raises(ValueError):
        run(0)
