"""Unit tests for QA and retrieval metrics."""

import pytest
from evaluation.qa_metrics import exact_match_score, f1_score, normalize_answer, compute_qa_metrics
from evaluation.retrieval_metrics import recall_at_k, precision_at_k, mean_reciprocal_rank, hit_at_k


def test_normalization():
    assert normalize_answer("The United States of America.") == "united states of america"
    assert normalize_answer("  An Apple, ") == "apple"
    assert normalize_answer("Edinburgh!") == "edinburgh"


def test_exact_match():
    assert exact_match_score("Edinburgh", "edinburgh") == 1.0
    assert exact_match_score("The City of Edinburgh", "City of Edinburgh") == 1.0
    assert exact_match_score("Glasgow", "Edinburgh") == 0.0


def test_f1_score():
    assert f1_score("Alexander Graham Bell", "Alexander Graham Bell") == 1.0
    assert f1_score("Graham Bell", "Alexander Graham Bell") == pytest.approx(0.8, 0.01)
    assert f1_score("Thomas Edison", "Alexander Graham Bell") == 0.0


def test_compute_qa_metrics_with_aliases():
    res = compute_qa_metrics("City of Edinburgh", "Edinburgh", ["City of Edinburgh"])
    assert res["exact_match"] == 1.0
    assert res["f1"] == 1.0


def test_retrieval_metrics():
    gold = {"p1", "p2"}
    retrieved = ["p1", "p3", "p2", "p4"]

    assert recall_at_k(retrieved, gold, k=1) == 0.5
    assert recall_at_k(retrieved, gold, k=3) == 1.0

    assert precision_at_k(retrieved, gold, k=1) == 1.0
    assert precision_at_k(retrieved, gold, k=2) == 0.5

    assert mean_reciprocal_rank(retrieved, gold) == 1.0
    assert mean_reciprocal_rank(["p3", "p1"], gold) == 0.5

    assert hit_at_k(retrieved, gold, k=1) == 1.0
    assert hit_at_k(["p3", "p4"], gold, k=2) == 0.0
