"""Publication-facing held-out evaluation utilities for the PeerJ release.

This module intentionally contains no model fitting. It evaluates frozen
observed-positive scores under the study's positive-unlabeled (PU) coding.
A label of 0 is therefore *unlabeled/unobserved*, not a confirmed negative.

Important metric note
---------------------
AUROC is computed with scikit-learn's conventional ``roc_auc_score`` so that
its definition is the same as the P10 development-stage AUROC. Ranking-based
AP and top-k metrics use a deterministic score-descending/product-key order,
as specified for the held-out evaluation. Scores are never inverted after
observing held-out performance.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from sklearn.metrics import roc_auc_score

OUTCOMES = (
    "has_any_outer_imagery_observed",
    "general_visual_appeal_observed_positive_core",
    "cute_friendly_observed_positive_core",
)
TRACKS = ("openclip_512_logistic", "interpretable_36_logistic")
DEFAULT_FRACTIONS = (0.05, 0.10, 0.20)
DEFAULT_BOOTSTRAP_ITERATIONS = 5000
DEFAULT_BOOTSTRAP_SEED = 20260818


class EvaluationError(RuntimeError):
    pass


def _as_binary(labels: Iterable[int]) -> np.ndarray:
    y = np.asarray(list(labels), dtype=np.int64)
    if y.ndim != 1 or len(y) == 0 or not np.isin(y, [0, 1]).all():
        raise EvaluationError("labels must be a non-empty binary vector")
    if int(y.sum()) in {0, len(y)}:
        raise EvaluationError("both PU label states are required")
    return y


def _as_scores(scores: Iterable[float], n: int) -> np.ndarray:
    values = np.asarray(list(scores), dtype=np.float64)
    if values.ndim != 1 or len(values) != n or not np.isfinite(values).all():
        raise EvaluationError("scores must be finite and match label length")
    return values


def stable_rank_order(
    scores: Iterable[float],
    keys: Sequence[str],
    occurrences: Sequence[int] | None = None,
) -> np.ndarray:
    values = np.asarray(list(scores), dtype=np.float64)
    names = [str(value) for value in keys]
    if len(values) != len(names):
        raise EvaluationError("score/key length mismatch")
    occ = list(occurrences) if occurrences is not None else [0] * len(values)
    if len(occ) != len(values):
        raise EvaluationError("score/occurrence length mismatch")
    if not np.isfinite(values).all():
        raise EvaluationError("non-finite score")
    return np.asarray(
        sorted(range(len(values)), key=lambda i: (-float(values[i]), names[i], int(occ[i]))),
        dtype=np.int64,
    )


def average_precision(
    labels: Iterable[int],
    scores: Iterable[float],
    keys: Sequence[str],
    occurrences: Sequence[int] | None = None,
) -> float:
    y = _as_binary(labels)
    values = _as_scores(scores, len(y))
    order = stable_rank_order(values, keys, occurrences)
    ranked = y[order]
    positions = np.flatnonzero(ranked == 1)
    return float(np.sum(np.cumsum(ranked)[positions] / (positions + 1)) / int(y.sum()))


def roc_auc(labels: Iterable[int], scores: Iterable[float], keys: Sequence[str] | None = None) -> float:
    """Conventional AUROC; ``keys`` is accepted for a symmetric public API."""
    y = _as_binary(labels)
    values = _as_scores(scores, len(y))
    return float(roc_auc_score(y, values))


def _suffix(fraction: float) -> str:
    percentage = 100.0 * float(fraction)
    if abs(percentage - round(percentage)) > 1e-12:
        raise EvaluationError("top-k fractions must map to whole percentages")
    return str(int(round(percentage)))


def evaluate_ranking(
    labels: Iterable[int],
    scores: Iterable[float],
    keys: Sequence[str],
    *,
    fractions: Sequence[float] = DEFAULT_FRACTIONS,
    occurrences: Sequence[int] | None = None,
) -> dict[str, float | int]:
    y = _as_binary(labels)
    values = _as_scores(scores, len(y))
    order = stable_rank_order(values, keys, occurrences)
    ranked = y[order]
    positive = int(y.sum())
    prevalence = positive / len(y)
    result: dict[str, float | int] = {
        "n": int(len(y)),
        "positive": positive,
        "average_precision": average_precision(y, values, keys, occurrences),
        "roc_auc": roc_auc(y, values),
    }
    for fraction in fractions:
        if not (0.0 < float(fraction) <= 1.0):
            raise EvaluationError("top-k fraction outside (0, 1]")
        suffix = _suffix(float(fraction))
        k = max(1, int(math.ceil(len(y) * float(fraction))))
        found = int(ranked[:k].sum())
        result[f"recall_at_top{suffix}"] = found / positive
        result[f"lift_at_top{suffix}"] = (found / k) / prevalence
    return result


def score_frozen_logistic(
    raw_features: Iterable[float],
    scaler_mean: Iterable[float],
    scaler_scale: Iterable[float],
    coef: Iterable[float],
    intercept: float,
) -> float:
    x = np.asarray(list(raw_features), dtype=np.float64)
    mean = np.asarray(list(scaler_mean), dtype=np.float64)
    scale = np.asarray(list(scaler_scale), dtype=np.float64)
    weights = np.asarray(list(coef), dtype=np.float64)
    if not (x.ndim == mean.ndim == scale.ndim == weights.ndim == 1):
        raise EvaluationError("frozen logistic inputs must be one-dimensional")
    if not (len(x) == len(mean) == len(scale) == len(weights)) or np.any(scale <= 0):
        raise EvaluationError("frozen logistic parameter dimension/scale mismatch")
    if not all(np.isfinite(array).all() for array in (x, mean, scale, weights)):
        raise EvaluationError("non-finite frozen logistic input")
    logit = float(intercept) + float(np.dot((x - mean) / scale, weights))
    return float(1.0 / (1.0 + math.exp(-min(700.0, max(-700.0, logit)))))


def cluster_bootstrap_intervals(
    labels: Iterable[int],
    scores: Iterable[float],
    parent_asins: Sequence[str],
    group_keys: Sequence[str],
    *,
    iterations: int = DEFAULT_BOOTSTRAP_ITERATIONS,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> dict[str, dict[str, float]]:
    y = _as_binary(labels)
    values = _as_scores(scores, len(y))
    parents = [str(value) for value in parent_asins]
    groups = [str(value) for value in group_keys]
    if not (len(parents) == len(groups) == len(y)):
        raise EvaluationError("bootstrap identity length mismatch")
    if iterations <= 0:
        raise EvaluationError("bootstrap iterations must be positive")

    group_to_rows: dict[str, list[int]] = defaultdict(list)
    for index, group in enumerate(groups):
        group_to_rows[group].append(index)
    ordered_groups = sorted(group_to_rows)
    rng = np.random.Generator(np.random.PCG64(int(seed)))
    draws = rng.integers(0, len(ordered_groups), size=(int(iterations), len(ordered_groups)), dtype=np.int64)

    metric_names = ("average_precision", "roc_auc", "recall_at_top10", "lift_at_top10")
    distributions = {name: [] for name in metric_names}
    for draw in draws:
        selected: list[int] = []
        occurrences: list[int] = []
        for occurrence, group_index in enumerate(draw.tolist()):
            members = group_to_rows[ordered_groups[int(group_index)]]
            selected.extend(members)
            occurrences.extend([occurrence] * len(members))
        indices = np.asarray(selected, dtype=np.int64)
        bootstrap_y = y[indices]
        if int(bootstrap_y.sum()) in {0, len(bootstrap_y)}:
            raise EvaluationError("bootstrap replicate has insufficient class support")
        metrics = evaluate_ranking(
            bootstrap_y,
            values[indices],
            [parents[index] for index in indices],
            occurrences=occurrences,
        )
        for name in metric_names:
            distributions[name].append(float(metrics[name]))

    point = evaluate_ranking(y, values, parents)
    result: dict[str, dict[str, float]] = {}
    for name in metric_names:
        distribution = np.asarray(distributions[name], dtype=np.float64)
        result[name] = {
            "point_estimate": float(point[name]),
            "ci_lower": float(np.quantile(distribution, 0.025, method="linear")),
            "ci_upper": float(np.quantile(distribution, 0.975, method="linear")),
        }
    return result


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def evaluate_frozen_prediction_files(
    modeling_manifest_path: Path,
    predictions_path: Path,
    *,
    iterations: int = DEFAULT_BOOTSTRAP_ITERATIONS,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> dict[str, object]:
    manifest_rows = _read_csv(modeling_manifest_path)
    locked = [
        row for row in manifest_rows
        if str(row.get("main_analysis_included", "")).lower() == "true"
        and row.get("split_partition") == "locked_test"
    ]
    if len(locked) != 1036:
        raise EvaluationError(f"expected 1,036 locked products, found {len(locked)}")
    by_parent = {row["parent_asin"]: row for row in locked}
    if len(by_parent) != len(locked):
        raise EvaluationError("duplicate parent_asin in locked manifest")

    prediction_rows = _read_csv(predictions_path)
    result_rows = []
    bootstrap_rows = []
    expected_models = {f"{outcome}__{track}" for outcome in OUTCOMES for track in TRACKS}
    seen_models = {row.get("model_id", "") for row in prediction_rows}
    if seen_models != expected_models:
        raise EvaluationError("prediction model universe does not match the six frozen models")

    for model_id in sorted(expected_models):
        outcome, track = model_id.split("__", 1)
        model_predictions = [row for row in prediction_rows if row.get("model_id") == model_id]
        score_by_parent = {}
        for row in model_predictions:
            parent = row.get("parent_asin", "")
            if parent in score_by_parent:
                raise EvaluationError(f"duplicate prediction for {parent}/{model_id}")
            score_by_parent[parent] = float(row["score"])
        if set(score_by_parent) != set(by_parent):
            raise EvaluationError(f"prediction coverage mismatch for {model_id}")

        ordered_parents = [row["parent_asin"] for row in locked]
        labels = np.asarray([int(by_parent[parent][outcome]) for parent in ordered_parents], dtype=np.int64)
        scores = np.asarray([score_by_parent[parent] for parent in ordered_parents], dtype=np.float64)
        groups = [by_parent[parent]["primary_response_sha256"] for parent in ordered_parents]
        metrics = evaluate_ranking(labels, scores, ordered_parents)
        result_rows.append({"model_id": model_id, "outcome": outcome, "track": track, **metrics})
        intervals = cluster_bootstrap_intervals(
            labels,
            scores,
            ordered_parents,
            groups,
            iterations=iterations,
            seed=seed,
        )
        for metric, values in intervals.items():
            bootstrap_rows.append({"model_id": model_id, "outcome": outcome, "track": track, "metric": metric, **values})

    return {
        "pu_semantics": "1=observed-positive; 0=unobserved/unlabeled, not confirmed negative",
        "score_orientation": "unchanged; no post-hoc inversion",
        "auroc_definition": "sklearn.metrics.roc_auc_score",
        "bootstrap": {"unit": "primary_response_sha256", "iterations": int(iterations), "seed": int(seed)},
        "product_metrics": result_rows,
        "cluster_bootstrap_intervals": bootstrap_rows,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modeling-manifest", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-iterations", type=int, default=DEFAULT_BOOTSTRAP_ITERATIONS)
    parser.add_argument("--seed", type=int, default=DEFAULT_BOOTSTRAP_SEED)
    args = parser.parse_args(argv)

    result = evaluate_frozen_prediction_files(
        args.modeling_manifest,
        args.predictions,
        iterations=args.bootstrap_iterations,
        seed=args.seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
