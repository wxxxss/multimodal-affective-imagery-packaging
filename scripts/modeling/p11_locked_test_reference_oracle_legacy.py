"""Independent, synthetic-fixture-safe reference oracle for P11.

This module intentionally owns its ranking, grouping, bootstrap, and robustness
implementations.  It has no dependency on the P11 producer module; callers pass
already-materialized rows, scores, and frozen parameters explicitly.
"""
from __future__ import annotations

import hashlib
import math
from typing import Any, Iterable

import numpy as np


METRICS = (
    "average_precision",
    "roc_auc",
    "recall_at_top5",
    "recall_at_top10",
    "recall_at_top20",
    "lift_at_top5",
    "lift_at_top10",
    "lift_at_top20",
)
BOOTSTRAP_METRICS = ("average_precision", "roc_auc", "recall_at_top10", "lift_at_top10")
OUTCOMES = (
    "has_any_outer_imagery_observed",
    "general_visual_appeal_observed_positive_core",
    "cute_friendly_observed_positive_core",
)
TRACKS = ("openclip_512_logistic", "interpretable_36_logistic")
VARIANTS = (
    ("primary", "core", "has_any_outer_imagery_observed", True),
    ("primary", "all_level", "has_any_all_level_imagery_evidence", False),
    ("general_visual_appeal", "core", "general_visual_appeal_observed_positive_core", True),
    ("general_visual_appeal", "pilot", "general_visual_appeal_observed_positive_pilot", False),
    ("general_visual_appeal", "robust", "general_visual_appeal_observed_positive_robust", False),
    ("cute_friendly", "core", "cute_friendly_observed_positive_core", True),
    ("cute_friendly", "pilot", "cute_friendly_observed_positive_pilot", False),
    ("cute_friendly", "robust", "cute_friendly_observed_positive_robust", False),
)


class OracleError(RuntimeError):
    pass


def _required(row: dict[str, Any], field: str) -> Any:
    if field not in row or row[field] is None or (isinstance(row[field], str) and not row[field].strip()):
        raise OracleError("missing required field " + field)
    return row[field]


def _binary(rows: list[dict[str, Any]], field: str) -> np.ndarray:
    values = []
    for row in rows:
        value = str(_required(row, field)).strip()
        if value not in {"0", "1"}:
            raise OracleError("invalid binary field " + field)
        values.append(int(value))
    return np.asarray(values, dtype=np.int64)


def total_order(scores: Iterable[float], keys: Iterable[str], occurrences: Iterable[int] | None = None) -> np.ndarray:
    values = [float(value) for value in scores]
    names = [str(value) for value in keys]
    if len(values) != len(names):
        raise OracleError("score/key length mismatch")
    occurrence_values = list(occurrences) if occurrences is not None else [0] * len(values)
    if len(occurrence_values) != len(values):
        raise OracleError("score/occurrence length mismatch")
    if not all(math.isfinite(value) for value in values):
        raise OracleError("non-finite score")
    return np.asarray(sorted(range(len(values)), key=lambda index: (-values[index], names[index], int(occurrence_values[index]))), dtype=np.int64)


def _rank_metrics(labels: np.ndarray, order: np.ndarray, allow_insufficient: bool, context: str) -> dict[str, Any]:
    y = np.asarray(labels, dtype=np.int64)
    if len(y) == 0:
        raise OracleError("empty metric input " + context)
    positive = int(np.sum(y == 1))
    if positive == 0 or positive == len(y):
        if allow_insufficient:
            return {"metric_status": "insufficient_class_support", **{name: None for name in METRICS}, "n": len(y), "positive": positive}
        raise OracleError("invalid class support " + context)
    ranked = y[order]
    positions = np.flatnonzero(ranked == 1)
    average_precision = float(np.sum(np.cumsum(ranked)[positions] / (positions + 1)) / positive)
    ranks = np.arange(1, len(y) + 1, dtype=np.float64)
    positive_ranks = float(np.sum(ranks[ranked == 1]))
    negative = len(y) - positive
    roc_auc = (positive_ranks - positive * (positive + 1) / 2.0) / (positive * negative)
    result: dict[str, Any] = {
        "metric_status": "ok",
        "average_precision": average_precision,
        "roc_auc": float(roc_auc),
        "n": len(y),
        "positive": positive,
    }
    prevalence = positive / len(y)
    for fraction, suffix in ((0.05, "5"), (0.10, "10"), (0.20, "20")):
        k = max(1, int(math.ceil(len(y) * fraction)))
        found = int(np.sum(ranked[:k]))
        result["recall_at_top" + suffix] = found / positive
        result["lift_at_top" + suffix] = (found / k) / prevalence
    return result


def metric_bundle(labels: Iterable[int], scores: Iterable[float], keys: Iterable[str], *, group_keys: Iterable[str] | None = None, occurrences: Iterable[int] | None = None, allow_insufficient: bool = False, context: str = "") -> dict[str, Any]:
    y = np.asarray(list(labels), dtype=np.int64)
    values = np.asarray(list(scores), dtype=np.float64)
    names = [str(value) for value in keys]
    tie_keys = [str(value) for value in group_keys] if group_keys is not None else names
    order = total_order(values, tie_keys, occurrences)
    return _rank_metrics(y, order, allow_insufficient, context)


def group_records(rows: list[dict[str, Any]], scores: Iterable[float], label_field: str) -> list[dict[str, Any]]:
    values = [float(value) for value in scores]
    if len(values) != len(rows):
        raise OracleError("group score length mismatch")
    grouped: dict[str, dict[str, Any]] = {}
    for row, score in zip(rows, values):
        group = str(_required(row, "primary_response_sha256"))
        asin = str(_required(row, "parent_asin"))
        label = str(_required(row, label_field))
        if label not in {"0", "1"}:
            raise OracleError("invalid group label")
        current = grouped.setdefault(group, {"group_key": group, "observed_positive": 0, "score": score, "parent_asins": []})
        if abs(float(current["score"]) - score) > 1e-12:
            raise OracleError("unequal frozen scores within primary_response_sha256 group")
        current["observed_positive"] = max(int(current["observed_positive"]), int(label))
        current["parent_asins"].append(asin)
    return [grouped[key] for key in sorted(grouped)]


def bootstrap_plan(keys: Iterable[str], iterations: int = 5000, seed: int = 20260818) -> tuple[np.ndarray, str]:
    ordered = sorted(str(key) for key in keys)
    if not ordered or iterations <= 0:
        raise OracleError("invalid bootstrap universe")
    draws = np.random.Generator(np.random.PCG64(seed)).integers(0, len(ordered), size=(iterations, len(ordered)), dtype=np.int64)
    digest = hashlib.sha256(draws.astype("<i8", copy=False).tobytes()).hexdigest().upper()
    return draws, digest


def expand_bootstrap_rows(rows: list[dict[str, Any]], draw: Iterable[int], group_key: str = "primary_response_sha256") -> tuple[list[int], list[int]]:
    groups: dict[str, list[int]] = {}
    for index, row in enumerate(rows):
        groups.setdefault(str(_required(row, group_key)), []).append(index)
    ordered_groups = sorted(groups)
    selected: list[int] = []
    occurrences: list[int] = []
    for occurrence, group_index in enumerate(draw):
        if int(group_index) < 0 or int(group_index) >= len(ordered_groups):
            raise OracleError("bootstrap draw index out of range")
        members = groups[ordered_groups[int(group_index)]]
        selected.extend(members)
        occurrences.extend([occurrence] * len(members))
    return selected, occurrences


def product_metrics(rows: list[dict[str, Any]], scores: dict[str, np.ndarray], models: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for model_id, model in models.items():
        labels = _binary(rows, model["outcome"])
        values = np.asarray(scores[model_id], dtype=np.float64)
        metrics = metric_bundle(labels, values, [str(_required(row, "parent_asin")) for row in rows])
        result.append({"model_id": model_id, "outcome": model["outcome"], "role": model["outcome_role"], "track": model["track"], **metrics})
    return result


def group_metrics(rows: list[dict[str, Any]], scores: dict[str, np.ndarray], models: dict[str, dict[str, Any]], plan_sha: str) -> list[dict[str, Any]]:
    result = []
    for model_id, model in models.items():
        groups = group_records(rows, scores[model_id], model["outcome"])
        labels = np.asarray([record["observed_positive"] for record in groups], dtype=np.int64)
        values = np.asarray([record["score"] for record in groups], dtype=np.float64)
        keys = [record["group_key"] for record in groups]
        metrics = metric_bundle(labels, values, keys, group_keys=keys)
        result.append({"model_id": model_id, "outcome": model["outcome"], "role": model["outcome_role"], "track": model["track"], "group_count": len(groups), "multi_product_group_count": sum(len(record["parent_asins"]) > 1 for record in groups), "bootstrap_plan_sha256": plan_sha, **metrics})
    return result


def bootstrap_metrics(rows: list[dict[str, Any]], scores: dict[str, np.ndarray], models: dict[str, dict[str, Any]], iterations: int = 5000, seed: int = 20260818, product: list[dict[str, Any]] | None = None) -> tuple[list[dict[str, Any]], str]:
    keys = sorted({str(_required(row, "primary_response_sha256")) for row in rows})
    draws, plan_sha = bootstrap_plan(keys, iterations, seed)
    point = {(record["model_id"], metric): float(record[metric]) for record in (product or product_metrics(rows, scores, models)) for metric in BOOTSTRAP_METRICS}
    result = []
    for model_id, model in models.items():
        labels = _binary(rows, model["outcome"])
        values = np.asarray(scores[model_id], dtype=np.float64)
        names = [str(_required(row, "parent_asin")) for row in rows]
        distributions = {metric: [] for metric in BOOTSTRAP_METRICS}
        for replicate_index, draw in enumerate(draws):
            selected, occurrences = expand_bootstrap_rows(rows, draw)
            selected_array = np.asarray(selected, dtype=np.int64)
            metrics = metric_bundle(labels[selected_array], values[selected_array], [names[index] for index in selected], occurrences=occurrences, context=f"{model_id}/replicate={replicate_index}")
            for metric in BOOTSTRAP_METRICS:
                distributions[metric].append(metrics[metric])
        for metric in BOOTSTRAP_METRICS:
            dist = np.asarray(distributions[metric], dtype=np.float64)
            result.append({"model_id": model_id, "outcome": model["outcome"], "track": model["track"], "metric": metric, "point_estimate": point[(model_id, metric)], "ci_lower": float(np.quantile(dist, 0.025, method="linear")), "ci_upper": float(np.quantile(dist, 0.975, method="linear")), "iterations": iterations, "cluster_count": len(keys), "seed": seed, "bootstrap_plan_sha256": plan_sha, "status": "ok"})
    return result, plan_sha


def r1_metrics(rows: list[dict[str, Any]], scores: dict[str, np.ndarray], models: dict[str, dict[str, Any]], excluded: set[str]) -> list[dict[str, Any]]:
    keep = [index for index, row in enumerate(rows) if str(_required(row, "parent_asin")) not in excluded]
    return [{"model_id": model_id, "outcome": model["outcome"], "track": model["track"], "excluded_rows": len(rows) - len(keep), "retained_rows": len(keep), **metric_bundle(_binary(rows, model["outcome"])[keep], np.asarray(scores[model_id])[keep], [str(_required(rows[index], "parent_asin")) for index in keep])} for model_id, model in models.items()]


def r2_metrics(
    rows: list[dict[str, Any]],
    primary_scores: dict[str, np.ndarray],
    sensitivity_features: dict[str, np.ndarray],
    models: dict[str, dict[str, Any]],
    parameters: dict[str, dict[str, Iterable[float] | float]],
 ) -> list[dict[str, Any]]:
    """Independently derive the paired, no-retrain sensitivity ledger."""
    result = []
    available = []
    for index, row in enumerate(rows):
        value = str(_required(row, "sensitivity_feature_available")).strip()
        if value not in {"0", "1"}:
            raise OracleError("invalid sensitivity_feature_available")
        if value == "1":
            available.append(index)
    for model_id, model in models.items():
        if model_id not in primary_scores or model_id not in sensitivity_features or model_id not in parameters:
            raise OracleError("missing R2 model binding " + model_id)
        primary = np.asarray(primary_scores[model_id], dtype=np.float64)
        sensitivity = np.asarray(sensitivity_features[model_id], dtype=np.float64)
        if primary.ndim != 1 or len(primary) != len(rows) or sensitivity.ndim != 2 or len(sensitivity) != len(rows):
            raise OracleError("R2 score/feature shape mismatch")
        parameter = parameters[model_id]
        for name in ("mean", "scale", "coef", "intercept"):
            _required(parameter, name)
        for index in available:
            raw = sensitivity[index]
            sensitivity_score = score(raw, parameter["mean"], parameter["scale"], parameter["coef"], float(parameter["intercept"]))
            primary_score = float(primary[index])
            result.append({
                "parent_asin": str(_required(rows[index], "parent_asin")),
                "primary_response_sha256": str(_required(rows[index], "primary_response_sha256")),
                "primary_feature_row_index": str(_required(rows[index], "primary_feature_row_index")),
                "sensitivity_feature_row_index": str(_required(rows[index], "sensitivity_feature_row_index")),
                "outcome": str(_required(model, "outcome")),
                "track": str(_required(model, "track")),
                "model_id": model_id,
                "primary_score": primary_score,
                "sensitivity_score": sensitivity_score,
                "score_delta": sensitivity_score - primary_score,
                "no_retrain": True,
                "same_split_grouping": True,
            })
    return result


def r3_metrics(rows: list[dict[str, Any]], exposure: list[float], scores: dict[str, np.ndarray], models: dict[str, dict[str, Any]], thresholds: Iterable[int], strata: Iterable[str]) -> list[dict[str, Any]]:
    if len(rows) != len(exposure):
        raise OracleError("R3 exposure length mismatch")
    def stratum(value: float) -> str:
        return "0" if value == 0 else "1-2" if value <= 2 else "3-4" if value <= 4 else "5-9" if value <= 9 else "10-19" if value <= 19 else "20-49" if value <= 49 else "50-99" if value <= 99 else "100+"
    result = []
    blocks = [("threshold", str(level), [index for index, value in enumerate(exposure) if value >= level]) for level in thresholds]
    blocks += [("stratum", str(level), [index for index, value in enumerate(exposure) if stratum(value) == level]) for level in strata]
    for kind, level, selected in blocks:
        for model_id, model in models.items():
            labels = _binary(rows, model["outcome"])[selected]
            values = np.asarray(scores[model_id])[selected]
            names = [str(_required(rows[index], "parent_asin")) for index in selected]
            metrics = metric_bundle(labels, values, names, allow_insufficient=True, context=f"{kind}={level}/{model_id}") if selected else {"metric_status": "empty", **{name: None for name in METRICS}, "n": 0, "positive": 0}
            result.append({"model_id": model_id, "outcome": model["outcome"], "track": model["track"], "threshold": level if kind == "threshold" else "", "stratum": level if kind == "stratum" else "", "rows": len(selected), "positive": int(np.sum(labels)), "prediction_use": False, "performance_best_threshold_selection": False, **metrics})
    return result


def r4_metrics(rows: list[dict[str, Any]], scores: dict[str, np.ndarray], models: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    mapping = {"primary": OUTCOMES[0], "general_visual_appeal": OUTCOMES[1], "cute_friendly": OUTCOMES[2]}
    result = []
    names = [str(_required(row, "parent_asin")) for row in rows]
    for variant, level, field, core in VARIANTS:
        labels = _binary(rows, field)
        model_outcome = mapping[variant]
        for track in TRACKS:
            model_id = model_outcome + "__" + track
            metrics = metric_bundle(labels, scores[model_id], names)
            result.append({"variant": variant, "level": level, "label_field": field, "track": track, "model_id": model_id, "core_baseline": core, "promotion_or_relabeling": False, **metrics})
    return result


def descriptive(values: Iterable[float]) -> dict[str, float]:
    array = np.asarray(list(values), dtype=np.float64)
    if len(array) == 0 or not np.all(np.isfinite(array)):
        raise OracleError("invalid descriptive score input")
    return {"score_mean": float(np.mean(array)), "score_std": float(np.std(array)), "score_median": float(np.quantile(array, 0.5)), "score_p10": float(np.quantile(array, 0.1)), "score_p90": float(np.quantile(array, 0.9))}


def r5_metrics(rows: list[dict[str, Any]], scores: dict[str, np.ndarray], models: dict[str, dict[str, Any]], categories: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for category in categories:
        field = str(_required(category, "source_field"))
        value = str(_required(category, "value"))
        selected = []

        for index, row in enumerate(rows):
            if field not in row:
                raise OracleError(
                    "missing required field "
                    + field
                )

            if str(row[field]) == value:
                selected.append(index)
        for model_id, model in models.items():
            stats = descriptive(np.asarray(scores[model_id])[selected]) if selected else {key: None for key in ("score_mean", "score_std", "score_median", "score_p10", "score_p90")}
            result.append({"category": str(_required(category, "category")), "source_field": field, "value": value, "model_id": model_id, "outcome": model["outcome"], "track": model["track"], "rows": len(selected), "inferential_claim": False, **stats})
    return result


def score(raw: Iterable[float], mean: Iterable[float], scale: Iterable[float], coef: Iterable[float], intercept: float) -> float:
    x = np.asarray(list(raw), dtype=np.float64)
    mu = np.asarray(list(mean), dtype=np.float64)
    sigma = np.asarray(list(scale), dtype=np.float64)
    beta = np.asarray(list(coef), dtype=np.float64)
    if x.shape != mu.shape or x.shape != sigma.shape or x.shape != beta.shape or np.any(sigma == 0):
        raise OracleError("frozen score parameter shape/domain mismatch")
    linear = float(intercept + np.dot((x - mu) / sigma, beta))
    return float(1.0 / (1.0 + math.exp(-min(700.0, max(-700.0, linear)))))
