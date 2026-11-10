import pytest

from src.metrics import hit_at_k, reciprocal_rank, summarize


def test_hit_at_k():
    relevance = [False, True, False]
    assert hit_at_k(relevance, 1) == 0.0
    assert hit_at_k(relevance, 2) == 1.0


def test_reciprocal_rank():
    assert reciprocal_rank([True]) == 1.0
    assert reciprocal_rank([False, False, True]) == pytest.approx(1 / 3)
    assert reciprocal_rank([False, False]) == 0.0


def test_summarize_averages_over_questions():
    out = summarize([[True, False], [False, True], [False, False]], ks=(1, 2))
    assert out["hit_at_1"] == pytest.approx(1 / 3)
    assert out["hit_at_2"] == pytest.approx(2 / 3)
    assert out["mrr"] == pytest.approx((1 + 0.5 + 0) / 3)


def test_summarize_rejects_empty():
    with pytest.raises(ValueError):
        summarize([])
