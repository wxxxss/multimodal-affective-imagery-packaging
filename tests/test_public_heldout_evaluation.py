from __future__ import annotations

import math

import numpy as np

from scripts.modeling.public_heldout_evaluation import (
    average_precision,
    evaluate_ranking,
    roc_auc,
    score_frozen_logistic,
)


def test_frozen_logistic_scoring_uses_supplied_parameters_only():
    x = np.array([2.0, 4.0])
    mean = np.array([1.0, 2.0])
    scale = np.array([1.0, 2.0])
    coef = np.array([0.5, -0.25])
    expected_logit = 0.1 + 0.5 * 1.0 - 0.25 * 1.0
    expected = 1.0 / (1.0 + math.exp(-expected_logit))
    assert abs(score_frozen_logistic(x, mean, scale, coef, 0.1) - expected) < 1e-12


def test_rank_metrics_use_descending_scores_without_orientation_flip():
    labels = np.array([1, 0, 1, 0], dtype=int)
    scores = np.array([0.9, 0.8, 0.2, 0.1], dtype=float)
    keys = ["a", "b", "c", "d"]

    assert abs(average_precision(labels, scores, keys) - (1.0 + 2.0 / 3.0) / 2.0) < 1e-12
    assert abs(roc_auc(labels, scores, keys) - 0.75) < 1e-12

    metrics = evaluate_ranking(labels, scores, keys, fractions=(0.25, 0.50))
    assert metrics["n"] == 4
    assert metrics["positive"] == 2
    assert metrics["recall_at_top25"] == 0.5
    assert metrics["lift_at_top25"] == 2.0
    assert metrics["recall_at_top50"] == 0.5
    assert metrics["lift_at_top50"] == 1.0


def test_ties_are_broken_by_stable_product_key():
    labels = np.array([0, 1, 1], dtype=int)
    scores = np.array([0.5, 0.5, 0.4], dtype=float)
    keys = ["b", "a", "c"]

    # At the tied top score, key 'a' must rank before key 'b'.
    assert abs(average_precision(labels, scores, keys) - (1.0 + 2.0 / 3.0) / 2.0) < 1e-12
