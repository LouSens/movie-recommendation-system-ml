"""Tests for rating and ranking metrics."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from src.evaluation.metrics import (
    catalog_coverage,
    evaluate_ranking,
    mae,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    relevant_items_by_user,
    rmse,
    top_k_indices,
)
from src.features.collaborative_features import IdEncoder


def test_rmse_and_mae():
    assert rmse([1, 2, 3], [1, 2, 3]) == 0.0
    assert rmse([0, 0], [3, 4]) == pytest.approx(math.sqrt(12.5))
    assert mae([0, 0], [3, -4]) == pytest.approx(3.5)
    with pytest.raises(ValueError):
        rmse([1], [1, 2])


def test_precision_and_recall():
    recommended = [10, 20, 30, 40]
    relevant = {20, 40, 99}
    assert precision_at_k(recommended, relevant, 2) == 0.5
    assert precision_at_k(recommended, relevant, 4) == 0.5
    assert recall_at_k(recommended, relevant, 4) == pytest.approx(2 / 3)
    assert recall_at_k(recommended, set(), 4) == 0.0
    with pytest.raises(ValueError):
        precision_at_k(recommended, relevant, 0)


def test_ndcg_perfect_and_partial():
    assert ndcg_at_k([1, 2, 3], {1, 2}, 3) == pytest.approx(1.0)
    # Single relevant item at rank 2: DCG = 1/log2(3), IDCG = 1.
    assert ndcg_at_k([5, 1, 6], {1}, 3) == pytest.approx(1 / math.log2(3))
    assert ndcg_at_k([5, 6], {1}, 2) == 0.0


def test_top_k_indices_sorted():
    scores = np.array([[0.1, 0.9, 0.5, 0.7]])
    assert top_k_indices(scores, 3).tolist() == [[1, 3, 2]]


def test_relevant_items_threshold():
    test = pd.DataFrame({"userId": [1, 1, 2], "movieId": [10, 11, 10], "rating": [4.0, 3.5, 5.0]})
    assert relevant_items_by_user(test, 4.0) == {1: {10}, 2: {10}}


def test_evaluate_ranking_masks_training_items():
    users, items = IdEncoder.fit([1]), IdEncoder.fit([10, 20, 30])
    train = pd.DataFrame({"userId": [1], "movieId": [10], "rating": [5.0]})
    test = pd.DataFrame({"userId": [1], "movieId": [30], "rating": [4.5]})
    # Item 10 has the highest score but was seen in training, so item 30 should rank first.
    scores = np.array([[0.9, 0.1, 0.5]])
    result = evaluate_ranking(scores, train, test, users, items, k_values=(1,))
    assert result["Precision@1"] == 1.0
    assert result["NDCG@1"] == 1.0
    assert result["evaluated_users"] == 1.0


def test_catalog_coverage():
    scores = np.array([[3.0, 2.0, 1.0], [3.0, 2.0, 1.0]])
    assert catalog_coverage(scores, k=1) == pytest.approx(1 / 3)
    mask = np.array([[True, False, False], [False, False, False]])
    assert catalog_coverage(scores, mask, k=1) == pytest.approx(2 / 3)
