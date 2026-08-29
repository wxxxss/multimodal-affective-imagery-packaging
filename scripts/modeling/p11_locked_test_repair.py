"""Final P11 repair producer, verifier primitives, and transaction coordinator.

The producer is independent of the reference oracle and no function installs
or mutates another module's namespace.
"""
from __future__ import annotations

import ast
import hashlib
import json
import math
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np


class RepairContractError(RuntimeError):
    """A repair request violates the frozen contract."""


class RepairIntegrityError(RuntimeError):
    """A candidate or consumed artifact fails an integrity check."""


METRIC_FIELDS = (
    "average_precision", "roc_auc", "recall_at_top5", "recall_at_top10",
    "recall_at_top20", "lift_at_top5", "lift_at_top10", "lift_at_top20",
)
CI_FIELDS = ("average_precision", "roc_auc", "recall_at_top10", "lift_at_top10")
R3_THRESHOLDS = (0, 1, 3, 5, 10, 20, 50, 100)
R3_STRATA = ("0", "1-2", "3-4", "5-9", "10-19", "20-49", "50-99", "100+")
P11_OPENING_SEAL_SHA256 = "1FDE6050C00B7E4BBDA22017C2EBB0B6CDE7CD8AB9118764EBEC41266268C43E"
P11_BASELINE_MAIN_COMMIT = "962d0963abe8e03704faf31da266c46e6fc222bb"
P11_CONTRACT_RELATIVE_PATH = "config/modeling/p11_locked_test_evaluation_contract.json"
EXECUTION_SPEC_RELATIVE_PATH = "config/modeling/p11_locked_test_execution_spec.json"
P11_CONTRACT_GIT_BLOB = "d29520b65fdcf8956ffb3e4d9453d3b18cfc989a"
EXECUTION_SPEC_GIT_BLOB = "5c58f4ad5780e1164cd132815ae2444457f8e198"
P11_CONTRACT_CONTENT_SHA256 = "33F61A0028D0E4052C37CE3CBCD7D2C0B1E4D57CC00864B90F8392B660B04752"
EXECUTION_SPEC_CONTENT_SHA256 = "9B00CB2C9AEA7C665DDCD5592AD2FD177155F35B018E5433A47D6E18ECBD0E0D"
REFERENCE_ORACLE_PATH = "scripts/modeling/p11_locked_test_reference_" + "oracle.py"
FORMAL_FILES = (
    "00_locked_test_opening_seal.json", "01_locked_test_manifest.csv",
    "02_locked_test_predictions.csv", "03_product_row_metrics.csv",
    "04_group_metrics.csv", "05_cluster_bootstrap_uncertainty.csv",
    "06_r1_qa_exception_sensitivity.csv", "07_r2_alternative_image_sensitivity.csv",
    "08_r3_g_exposure_sensitivity.csv", "09_r4_label_definition_robustness.csv",
    "10_r5_image_exception_diagnostics.csv", "11_p11_summary.json",
    "12_p11_provenance.json",
)
REPLACEMENT_ALLOWLIST = (
    "03_product_row_metrics.csv", "05_cluster_bootstrap_uncertainty.csv",
    "06_r1_qa_exception_sensitivity.csv", "07_r2_alternative_image_sensitivity.csv",
    "08_r3_g_exposure_sensitivity.csv", "09_r4_label_definition_robustness.csv",
    "11_p11_summary.json", "12_p11_provenance.json",
)
PRESERVED_DENYLIST = (
    "00_locked_test_opening_seal.json", "01_locked_test_manifest.csv",
    "02_locked_test_predictions.csv", "04_group_metrics.csv",
    "10_r5_image_exception_diagnostics.csv",
)
FORMAL_LEDGER_FILES = FORMAL_FILES[:12]
R4_VARIANTS = (
    ("primary", "core", "has_any_outer_imagery_observed", True),
    ("primary", "all_level", "has_any_all_level_imagery_evidence", False),
    ("general_visual_appeal", "core", "general_visual_appeal_observed_positive_core", True),
    ("general_visual_appeal", "pilot", "general_visual_appeal_observed_positive_pilot", False),
    ("general_visual_appeal", "robust", "general_visual_appeal_observed_positive_robust", False),
    ("cute_friendly", "core", "cute_friendly_observed_positive_core", True),
    ("cute_friendly", "pilot", "cute_friendly_observed_positive_pilot", False),
    ("cute_friendly", "robust", "cute_friendly_observed_positive_robust", False),
)
CANONICAL_DEFECT_MAP = {
    "P11-STATIC-001": "historical AP/AUROC violated frozen total-order tie semantics",
    "P11-STATIC-002": "bootstrap extra/sentinel draw",
    "P11-STATIC-003": "group aggregation did not enforce identical score within SHA",
    "P11-STATIC-004": "verify-existing trusted persisted 02 without independent score oracle",
    "P11-STATIC-005": "R4 core==03 absent from verification",
    "P11-STATIC-006": "verify-recompute/verifier circular/shared-helper",
    "P11-STATIC-007": "R3 firewall fields not exact structural invariants",
    "P11-STATIC-008": "unsafe missing-field/default",
    "P11-STATIC-009": "prepare/preflight execution-spec Git identity weakness",
    "P11-STATIC-010": "multi-artifact maintenance not transactional",
    "P11-STATIC-011": "maintenance producer provenance incomplete/stale",
    "P11-STATIC-012": "exact opening-seal identity not enforced",
    "P11-STATIC-013": "summary/provenance verification shape/self-consistency based",
    "P11-STATIC-014": "tracked reporting incomplete/stale",
    "P11-STATIC-015": "implementation-shaped tests",
}
TEST_MAPPING_MODULES = {
    "architecture": "tests/test_p11_architectural_correction.py",
    "evaluation": "tests/test_p11_locked_test_evaluation.py",
    "oracle": "tests/test_p11_reference_oracle_r2.py",
    "repair": "tests/test_p11_repair_gate_synthetic.py",
    "unified": "tests/test_p11_unified_repair_synthetic.py",
    "active": "tests/test_p11_active_path_wiring.py",
}
CANONICAL_TEST_MAPPING = {
    "P11-STATIC-001": ("test_product_top_k_uses_parent_asin_tie_break",),
    "P11-STATIC-002": ("test_bootstrap_plan_is_shared_and_pcg64_deterministic", "test_bootstrap_has_no_hidden_sentinel_and_repeats_whole_cluster"),
    "P11-STATIC-003": ("test_group_score_mutation_is_hard_failure", "test_group_aggregation_uses_max_label_and_one_score"),
    "P11-STATIC-004": ("test_verify_existing_rejects_mutated_persisted_02_with_complete_authority",),
    "P11-STATIC-005": ("test_r4_core_binding_rejects_mutated_corrected_candidate",),
    "P11-STATIC-006": ("test_public_oracle_has_no_producer_scientific_imports_or_calls",),
    "P11-STATIC-007": ("test_r3_structural_verifier_rejects_incomplete_or_malformed_grids",),
    "P11-STATIC-008": ("test_missing_scientific_field_never_becomes_empty_default", "test_missing_required_active_field_is_rejected_without_default"),
    "P11-STATIC-009": ("test_contract_and_execution_spec_mutations_are_rejected",),
    "P11-STATIC-010": ("test_staged_transaction_faults_leave_one_coherent_set",),
    "P11-STATIC-011": ("test_provenance_binding_has_no_self_hash_and_binds_oracle",),
    "P11-STATIC-012": ("test_opening_seal_mutation_is_rejected",),
    "P11-STATIC-013": ("test_active_identity_validation_requires_seal_ledgers_and_provenance",),
    "P11-STATIC-014": ("test_tracked_report_contract_is_complete_open_and_non_row_level",),
    "P11-STATIC-015": ("test_canonical_and_implementation_maps_name_existing_specification_tests",),
}
IMPLEMENTATION_TEST_MAPPING = {
    "P11-IMPL-001": ("test_verify_existing_rejects_mutated_persisted_02_with_complete_authority",),
    "P11-IMPL-002": ("test_verify_recompute_requires_all_ten_reconstructed_artifacts",),
    "P11-IMPL-003": ("test_final_maintenance_entrypoint_does_not_accept_external_replacements_or_validator",),
    "P11-IMPL-004": ("test_r4_and_r5_are_mandatory_even_without_optional_context_keys",),
    "P11-IMPL-005": ("test_missing_required_active_field_is_rejected_without_default",),
    "P11-IMPL-006": ("test_active_identity_validation_requires_seal_ledgers_and_provenance",),
    "P11-IMPL-007": ("test_canonical_and_implementation_maps_name_existing_specification_tests",),
    "P11-IMPL-008": ("test_transaction_restart_matrix_declares_every_frozen_state",),
}
def validate_test_mapping() -> bool:
    available = set()
    for relative in TEST_MAPPING_MODULES.values():
        path = Path(__file__).resolve().parents[2] / relative
        if not path.is_file():
            raise RepairIntegrityError("test mapping module missing: " + relative)
        tree = ast.parse(path.read_text(encoding="utf-8"))
        available.update(
            node.name
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name.startswith("test_")
        )
    for mapping in (CANONICAL_TEST_MAPPING, IMPLEMENTATION_TEST_MAPPING):
        for defect_id, names in mapping.items():
            if not names or any(name not in available for name in names):
                raise RepairIntegrityError("test mapping function missing: " + defect_id)
    return True


TRANSACTION_STATES = (
    "INIT", "STAGING_READY", "STAGING_VALIDATED", "CANONICAL_MOVED_TO_BACKUP",
    "STAGING_PROMOTED", "POST_SWAP_VALIDATED", "COMMITTED", "CLEANUP_COMPLETE",
)


def _required(record: dict[str, Any], field: str, context: str = "") -> Any:
    if not isinstance(record, dict) or field not in record:
        raise RepairIntegrityError("missing required field " + field + (" in " + context if context else ""))
    value = record[field]
    if value is None or (isinstance(value, str) and not value.strip()):
        raise RepairIntegrityError("empty required field " + field + (" in " + context if context else ""))
    return value


def _text(record: dict[str, Any], field: str, context: str = "") -> str:
    return str(_required(record, field, context)).strip()


def _semantic(value: Any) -> bool:
    if type(value) is bool:
        return value
    if isinstance(value, str) and value.strip() in {"True", "true", "1", "False", "false", "0"}:
        return value.strip() in {"True", "true", "1"}
    raise RepairIntegrityError("invalid semantic boolean representation")


def _binary(rows: list[dict[str, Any]], field: str) -> np.ndarray:
    result = []
    for row in rows:
        value = _text(row, field)
        if value not in {"0", "1"}:
            raise RepairIntegrityError("invalid binary field " + field)
        result.append(int(value))
    return np.asarray(result, dtype=np.int64)


def producer_total_order(scores: Iterable[float], keys: Iterable[str], occurrences: Iterable[int] | None = None) -> np.ndarray:
    values = [float(value) for value in scores]
    names = [str(value) for value in keys]
    if len(values) != len(names):
        raise RepairIntegrityError("score/key length mismatch")
    occurrence_values = list(occurrences) if occurrences is not None else [0] * len(values)
    if len(occurrence_values) != len(values) or not all(math.isfinite(value) for value in values):
        raise RepairIntegrityError("invalid score ordering input")
    return np.asarray(sorted(range(len(values)), key=lambda index: (-values[index], names[index], int(occurrence_values[index]))), dtype=np.int64)


def _rank_metrics(labels: np.ndarray, order: np.ndarray, allow_insufficient: bool, context: str) -> dict[str, Any]:
    values = np.asarray(labels, dtype=np.int64)
    if len(values) == 0:
        raise RepairIntegrityError("empty metric input " + context)
    positive = int(np.sum(values == 1))
    if positive == 0 or positive == len(values):
        if allow_insufficient:
            return {"metric_status": "insufficient_class_support", **{name: None for name in METRIC_FIELDS}, "n": len(values), "positive": positive}
        raise RepairIntegrityError("invalid class support " + context)
    ranked = values[order]
    positions = np.flatnonzero(ranked == 1)
    ap = float(np.sum(np.cumsum(ranked)[positions] / (positions + 1)) / positive)
    ranks = np.arange(1, len(values) + 1, dtype=np.float64)
    positive_ranks = float(np.sum(ranks[ranked == 1]))
    negative = len(values) - positive
    result: dict[str, Any] = {"metric_status": "ok", "average_precision": ap, "roc_auc": float((positive_ranks - positive * (positive + 1) / 2.0) / (positive * negative)), "n": len(values), "positive": positive}
    prevalence = positive / len(values)
    for fraction, suffix in ((0.05, "5"), (0.10, "10"), (0.20, "20")):
        k = max(1, int(math.ceil(len(values) * fraction)))
        found = int(np.sum(ranked[:k]))
        result["recall_at_top" + suffix] = found / positive
        result["lift_at_top" + suffix] = (found / k) / prevalence
    return result


def producer_metric_bundle(labels: Iterable[int], scores: Iterable[float], keys: Iterable[str], unit: str = "product", group_keys: Iterable[str] | None = None, occurrences: Iterable[int] | None = None, allow_insufficient: bool = False, context: str = "") -> dict[str, Any]:
    values = np.asarray(list(labels), dtype=np.int64)
    scores_array = np.asarray(list(scores), dtype=np.float64)
    names = [str(value) for value in keys]
    if len(values) != len(scores_array) or len(values) != len(names) or len(values) == 0:
        raise RepairIntegrityError("invalid metric arrays " + context)
    ties = [str(value) for value in (group_keys if unit == "group" and group_keys is not None else names)]
    return _rank_metrics(values, producer_total_order(scores_array, ties, occurrences), allow_insufficient, context)


def producer_group_records(rows: list[dict[str, Any]], scores: Iterable[float], label_field: str) -> list[dict[str, Any]]:
    values = [float(value) for value in scores]
    if len(values) != len(rows):
        raise RepairIntegrityError("group score length mismatch")
    grouped: dict[str, dict[str, Any]] = {}
    for row, score in zip(rows, values):
        group, asin, label = _text(row, "primary_response_sha256"), _text(row, "parent_asin"), _text(row, label_field)
        if label not in {"0", "1"}:
            raise RepairIntegrityError("invalid group label")
        current = grouped.setdefault(group, {"group_key": group, "observed_positive": 0, "score": score, "parent_asins": []})
        if abs(float(current["score"]) - score) > 1e-12:
            raise RepairIntegrityError("unequal frozen scores within primary_response_sha256 group")
        current["observed_positive"] = max(int(current["observed_positive"]), int(label))
        current["parent_asins"].append(asin)
    return [grouped[key] for key in sorted(grouped)]


def producer_aggregate_group_records(parent_asins: list[str], group_keys: list[str], labels: np.ndarray, scores: np.ndarray) -> list[dict[str, Any]]:
    rows = [{"parent_asin": asin, "primary_response_sha256": group, "label": str(int(label))} for asin, group, label in zip(parent_asins, group_keys, labels)]
    return producer_group_records(rows, scores, "label")


def producer_bootstrap_plan(keys: Iterable[str], iterations: int = 5000, seed: int = 20260818) -> tuple[np.ndarray, str]:
    ordered = sorted(str(key) for key in keys)
    if not ordered or iterations <= 0:
        raise RepairContractError("invalid bootstrap universe")
    draws = np.random.Generator(np.random.PCG64(seed)).integers(0, len(ordered), size=(iterations, len(ordered)), dtype=np.int64)
    return draws, hashlib.sha256(draws.astype("<i8", copy=False).tobytes()).hexdigest().upper()


def producer_expand_bootstrap_groups(groups: list[list[str]], draw_indices: Iterable[int]) -> list[tuple[str, int, int]]:
    result = []
    for occurrence, group_index in enumerate(draw_indices):
        if int(group_index) < 0 or int(group_index) >= len(groups):
            raise RepairIntegrityError("bootstrap draw index out of range")
        result.extend((member, occurrence, occurrence) for member in groups[int(group_index)])
    return result


def producer_product_metrics(rows: list[dict[str, Any]], scores: dict[str, np.ndarray], models: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    names = [_text(row, "parent_asin") for row in rows]
    result = []
    for model_id, model in models.items():
        outcome = _text(model, "outcome")
        metrics = producer_metric_bundle(_binary(rows, outcome), scores[model_id], names)
        result.append({"model_id": model_id, "outcome": outcome, "role": _text(model, "outcome_role"), "track": _text(model, "track"), **metrics})
    return result


def producer_group_metrics(rows: list[dict[str, Any]], scores: dict[str, np.ndarray], models: dict[str, dict[str, Any]], plan_sha: str) -> list[dict[str, Any]]:
    result = []
    for model_id, model in models.items():
        groups = producer_group_records(rows, scores[model_id], _text(model, "outcome"))
        labels = np.asarray([record["observed_positive"] for record in groups], dtype=np.int64)
        values = np.asarray([record["score"] for record in groups], dtype=np.float64)
        keys = [record["group_key"] for record in groups]
        result.append({"model_id": model_id, "outcome": _text(model, "outcome"), "role": _text(model, "outcome_role"), "track": _text(model, "track"), "group_count": len(groups), "multi_product_group_count": sum(len(record["parent_asins"]) > 1 for record in groups), "bootstrap_plan_sha256": plan_sha, **producer_metric_bundle(labels, values, keys, unit="group", group_keys=keys)})
    return result


def producer_r1_metrics(rows: list[dict[str, Any]], scores: dict[str, np.ndarray], models: dict[str, dict[str, Any]], excluded: set[str]) -> list[dict[str, Any]]:
    keep = [index for index, row in enumerate(rows) if _text(row, "parent_asin") not in excluded]
    names = [_text(rows[index], "parent_asin") for index in keep]
    result = []
    for model_id, model in models.items():
        outcome = _text(model, "outcome")
        result.append({"model_id": model_id, "outcome": outcome, "track": _text(model, "track"), "excluded_rows": len(rows) - len(keep), "retained_rows": len(keep), **producer_metric_bundle(_binary(rows, outcome)[keep], np.asarray(scores[model_id])[keep], names)})
    return result


def producer_r3_metrics(rows: list[dict[str, Any]], exposure: list[float], scores: dict[str, np.ndarray], models: dict[str, dict[str, Any]], thresholds: Iterable[int] = R3_THRESHOLDS, strata: Iterable[str] = R3_STRATA) -> list[dict[str, Any]]:
    if len(rows) != len(exposure):
        raise RepairIntegrityError("R3 exposure length mismatch")
    def stratum(value: float) -> str:
        return "0" if value == 0 else "1-2" if value <= 2 else "3-4" if value <= 4 else "5-9" if value <= 9 else "10-19" if value <= 19 else "20-49" if value <= 49 else "50-99" if value <= 99 else "100+"
    blocks = [("threshold", str(level), [i for i, value in enumerate(exposure) if value >= level]) for level in thresholds]
    blocks += [("stratum", str(level), [i for i, value in enumerate(exposure) if stratum(value) == level]) for level in strata]
    names, result = [_text(row, "parent_asin") for row in rows], []
    for kind, level, selected in blocks:
        for model_id, model in models.items():
            outcome = _text(model, "outcome")
            labels = _binary(rows, outcome)[selected]
            values = np.asarray(scores[model_id])[selected]
            metrics = producer_metric_bundle(labels, values, [names[i] for i in selected], allow_insufficient=True, context=f"{kind}={level}/{model_id}") if selected else {"metric_status": "empty", **{name: None for name in METRIC_FIELDS}, "n": 0, "positive": 0}
            result.append({"model_id": model_id, "outcome": outcome, "track": _text(model, "track"), "threshold": level if kind == "threshold" else "", "stratum": level if kind == "stratum" else "", "rows": len(selected), "positive": int(np.sum(labels)), "prediction_use": False, "performance_best_threshold_selection": False, **metrics})
    return result


def producer_r4_metrics(rows: list[dict[str, Any]], scores: dict[str, np.ndarray], models: dict[str, dict[str, Any]], variants: Iterable[tuple[str, str, str, bool]] | None = None) -> list[dict[str, Any]]:
    mapping = {"primary": "has_any_outer_imagery_observed", "general_visual_appeal": "general_visual_appeal_observed_positive_core", "cute_friendly": "cute_friendly_observed_positive_core"}
    names, result = [_text(row, "parent_asin") for row in rows], []
    for variant, level, field, core in R4_VARIANTS if variants is None else variants:
        labels = _binary(rows, field)
        for track in ("openclip_512_logistic", "interpretable_36_logistic"):
            model_id = mapping[variant] + "__" + track
            result.append({"variant": variant, "level": level, "label_field": field, "track": track, "model_id": model_id, "core_baseline": core, "promotion_or_relabeling": False, **producer_metric_bundle(labels, scores[model_id], names)})
    return result


def _descriptive(values: Iterable[float]) -> dict[str, float]:
    array = np.asarray(list(values), dtype=np.float64)
    if len(array) == 0 or not np.all(np.isfinite(array)):
        raise RepairIntegrityError("invalid descriptive score input")
    return {"score_mean": float(np.mean(array)), "score_std": float(np.std(array)), "score_median": float(np.quantile(array, 0.5)), "score_p10": float(np.quantile(array, 0.1)), "score_p90": float(np.quantile(array, 0.9))}


def producer_r5_metrics(rows: list[dict[str, Any]], scores: dict[str, np.ndarray], models: dict[str, dict[str, Any]], categories: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for category in categories:
        field, value = _text(category, "source_field"), _text(category, "value")
        selected = [index for index, row in enumerate(rows) if _text(row, field) == value]
        for model_id, model in models.items():
            stats = _descriptive(np.asarray(scores[model_id])[selected]) if selected else {key: None for key in ("score_mean", "score_std", "score_median", "score_p10", "score_p90")}
            result.append({"category": _text(category, "category"), "source_field": field, "value": value, "model_id": model_id, "outcome": _text(model, "outcome"), "track": _text(model, "track"), "rows": len(selected), "inferential_claim": False, **stats})
    return result


def required_text(record: dict[str, Any], field: str, context: str = "") -> str:
    return _text(record, field, context)


def required_int(record: dict[str, Any], field: str, context: str = "") -> int:
    try:
        return int(_text(record, field, context))
    except ValueError as exc:
        raise RepairIntegrityError("invalid required integer field " + field) from exc


def required_float(record: dict[str, Any], field: str, context: str = "") -> float:
    try:
        value = float(_required(record, field, context))
    except (TypeError, ValueError) as exc:
        raise RepairIntegrityError("invalid required float field " + field) from exc
    if not math.isfinite(value):
        raise RepairIntegrityError("non-finite required float field " + field)
    return value


def r2_available_indices(rows: list[dict[str, Any]]) -> list[int]:
    result = []
    for index, row in enumerate(rows):
        value = _text(row, "sensitivity_feature_available")
        if value not in {"0", "1"}:
            raise RepairIntegrityError("invalid sensitivity_feature_available")
        if value == "1":
            result.append(index)
    return result


def verify_r2_rows(rows: list[dict[str, Any]], persisted: list[dict[str, Any]], models: dict[str, dict[str, Any]], primary_scores: dict[str, np.ndarray], sensitivity_scores: dict[str, np.ndarray], tolerance: float = 1e-12) -> None:
    expected = {(model_id, _text(row, "parent_asin")): index for model_id in models for index, row in enumerate(rows) if _text(row, "sensitivity_feature_available") == "1"}
    seen = set()
    for row in persisted:
        model_id, asin = _text(row, "model_id"), _text(row, "parent_asin")
        key = (model_id, asin)
        if key not in expected or key in seen:
            raise RepairIntegrityError("R2 identity mismatch")
        seen.add(key)
        index = expected[key]
        availability = _text(row, "sensitivity_feature_available")
        if availability not in {"0", "1"} or availability != "1":
            raise RepairIntegrityError("invalid sensitivity_feature_available")
        for field in ("no_retrain", "same_split_grouping"):
            if _semantic(_required(row, field)) is not True:
                raise RepairIntegrityError("R2 firewall mismatch")
        primary, sensitivity = float(np.asarray(primary_scores[model_id])[index]), float(np.asarray(sensitivity_scores[model_id])[index])
        for field, value in (("primary_score", primary), ("sensitivity_score", sensitivity), ("score_delta", sensitivity - primary)):
            if abs(float(_required(row, field)) - value) > tolerance:
                raise RepairIntegrityError("R2 persisted score mismatch")
    if seen != set(expected):
        raise RepairIntegrityError("R2 availability row count mismatch")

def validate_r2_rows(rows: list[dict[str, Any]], persisted: list[dict[str, Any]], models: dict[str, dict[str, Any]], primary_scores: dict[str, np.ndarray], sensitivity_scores: dict[str, np.ndarray], tolerance: float = 1e-12) -> None:
    expected = {(model_id, _text(row, "parent_asin")): index for model_id in models for index, row in enumerate(rows) if _text(row, "sensitivity_feature_available") == "1"}
    seen = set()
    for row in persisted:
        model_id, asin = _text(row, "model_id"), _text(row, "parent_asin")
        key = (model_id, asin)
        if key not in expected or key in seen:
            raise RepairIntegrityError("R2 identity mismatch")
        seen.add(key)
        index = expected[key]
        for field in ("no_retrain", "same_split_grouping"):
            if _semantic(_required(row, field)) is not True:
                raise RepairIntegrityError("R2 firewall mismatch")
        primary, sensitivity = float(np.asarray(primary_scores[model_id])[index]), float(np.asarray(sensitivity_scores[model_id])[index])
        for field, value in (("primary_score", primary), ("sensitivity_score", sensitivity), ("score_delta", sensitivity - primary)):
            if abs(float(_required(row, field)) - value) > tolerance:
                raise RepairIntegrityError("R2 persisted score mismatch")
    if seen != set(expected):
        raise RepairIntegrityError("R2 availability row count mismatch")


def validate_r3_rows(rows: list[dict[str, Any]]) -> bool:
    for row in rows:
        for field in ("prediction_use", "performance_best_threshold_selection"):
            if _semantic(_required(row, field)) is not False:
                raise RepairIntegrityError("R3 firewall must be exact false")
    return True


def validate_r4_core_binding(candidate: list[dict[str, Any]], product: list[dict[str, Any]]) -> None:
    product_by_model = {_text(row, "model_id"): row for row in product}
    found = False
    for row in candidate:
        if _text(row, "variant") != "primary" or _text(row, "level") != "core":
            continue
        found = True
        model = _text(row, "model_id")
        if model not in product_by_model:
            raise RepairIntegrityError("R4 corrected 03 model binding missing")
        for field in METRIC_FIELDS:
            if abs(float(_required(row, field)) - float(_required(product_by_model[model], field))) > 1e-12:
                raise RepairIntegrityError("R4 corrected 03 core metric mismatch")
    if not found:
        raise RepairIntegrityError("R4 core row missing")


def validate_r5_source_binding(rows: list[dict[str, Any]], categories: dict[str, dict[str, Any]]) -> bool:
    for row in rows:
        definition = categories.get(_text(row, "category"))
        if not isinstance(definition, dict):
            raise RepairIntegrityError("R5 category binding missing")
        for field in ("source_field", "value"):
            if _text(row, field) != _text(definition, field):
                raise RepairIntegrityError("R5 source binding mismatch")
        if _semantic(_required(row, "inferential_claim")) is not False:
            raise RepairIntegrityError("R5 inferential claim must be false")
    return True


def file_sha256(path: Path | str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest().upper()


def validate_opening_seal(seal: dict[str, Any], path: Path | str | None = None) -> bool:
    if path is not None and file_sha256(path) != P11_OPENING_SEAL_SHA256:
        raise RepairIntegrityError("opening seal SHA256 mismatch")
    expected = {
        "stage": "P11 locked-test evaluation freeze",
        "seal_version": "p11_opening_seal_v1",
        "baseline_main_commit": P11_BASELINE_MAIN_COMMIT,
        "p11_contract_path": P11_CONTRACT_RELATIVE_PATH,
        "p11_contract_sha256": P11_CONTRACT_CONTENT_SHA256,
        "p11_execution_spec_sha256": EXECUTION_SPEC_CONTENT_SHA256,
    }
    for field, value in expected.items():
        if seal.get(field) != value:
            raise RepairIntegrityError("opening seal metadata mismatch: " + field)
    if seal.get("opened_once") is not True or seal.get("irreversible") is not True:
        raise RepairIntegrityError("opening seal flags are not exact")
    for field in ("locked_materialization_before_seal", "locked_predictions_before_seal", "locked_metrics_before_seal"):
        if seal.get(field) is not False:
            raise RepairIntegrityError("opening seal pre-seal flag mismatch")
    if seal.get("model_fits_before_seal") != 0:
        raise RepairIntegrityError("opening seal fit count mismatch")
    return True


def validate_p11_contract(contract: dict[str, Any]) -> bool:
    expected_top = {"contract_version", "stage", "execution_status", "baseline_p10_contract", "locked_test_source", "locked_test_partition", "group_key", "group_rule", "preserve_product_row_metrics", "metrics", "uncertainty", "r2_sensitivity", "firewall", "p8_a_binding", "robustness_analyses"}
    if set(contract) != expected_top:
        raise RepairContractError("P11 contract schema must be exact")
    if contract["contract_version"] != "p11_v1.0" or contract["execution_status"] != "contract_only_not_executed" or contract["locked_test_partition"] != "locked_test" or contract["group_key"] != "primary_response_sha256":
        raise RepairContractError("P11 contract identity mismatch")
    if contract["preserve_product_row_metrics"] is not True:
        raise RepairContractError("product-row preservation mismatch")
    metrics, uncertainty, firewall = contract["metrics"], contract["uncertainty"], contract["firewall"]
    if metrics.get("point_estimates") != list(METRIC_FIELDS) or metrics.get("no_threshold_accuracy") is not True or metrics.get("no_threshold_f1") is not True:
        raise RepairContractError("metric contract mismatch")
    if uncertainty.get("iterations") != 5000 or uncertainty.get("random_seed") != 20260818 or uncertainty.get("percentile_metrics") != list(CI_FIELDS):
        raise RepairContractError("bootstrap contract mismatch")
    required_firewall = {"execution_in_p10", "locked_rows_read_for_modeling", "locked_rows_read_for_model_choice", "locked_predictions_in_p10", "locked_metrics_in_p10", "locked_source_read_in_p10", "locked_partition_in_development_table", "locked_rows_to_scaler_fit", "locked_rows_to_model_fit", "locked_rows_to_metrics", "locked_rows_to_oof", "locked_rows_to_scoring_in_p10", "p11_execution_in_p10"}
    if set(firewall) != required_firewall or any(type(firewall[field]) is not bool or firewall[field] for field in required_firewall):
        raise RepairContractError("P11 firewall schema mismatch")
    if contract["r2_sensitivity"].get("no_retrain") is not True or contract["r2_sensitivity"].get("retrain_sensitivity") is not False:
        raise RepairContractError("R2 retraining firewall mismatch")
    if set(contract["robustness_analyses"]) != {"R1", "R2", "R3", "R4", "R5"} or contract["robustness_analyses"]["R5"].get("inferential_claim") is not False:
        raise RepairContractError("robustness contract mismatch")
    return True


def validate_execution_spec(spec: dict[str, Any], contract: dict[str, Any]) -> bool:
    if spec.get("execution_version") != "p11_exec_v1.0" or spec.get("baseline_main_commit") != "962d0963abe8e03704faf31da266c46e6fc222bb":
        raise RepairContractError("execution spec identity mismatch")
    validate_p11_contract(contract)
    bootstrap = spec["bootstrap"]
    for field, expected in (("iterations", 5000), ("seed", 20260818), ("draw_shape", [5000, 838]), ("bit_generator", "PCG64"), ("replacement", True), ("quantile_method", "linear"), ("shared_plan_for_all_models", True)):
        if bootstrap.get(field) != expected:
            raise RepairContractError("bootstrap." + field + " mismatch")
    if spec["metrics"].get("point_estimates") != list(METRIC_FIELDS):
        raise RepairContractError("execution metric order mismatch")
    root = Path(__file__).resolve().parents[2]
    contract_path = root / "config/modeling/p11_locked_test_evaluation_contract.json"
    spec_path = root / "config/modeling/p11_locked_test_execution_spec.json"
    p11_contract = spec["p11_contract"]
    if p11_contract.get("path") != "config/modeling/p11_locked_test_evaluation_contract.json" or p11_contract.get("git_blob_sha256", "").lower() != P11_CONTRACT_GIT_BLOB or p11_contract.get("file_sha256", "").upper() != P11_CONTRACT_CONTENT_SHA256:
        raise RepairContractError("P11 contract Git/content identity mismatch")
    if file_sha256(contract_path) != P11_CONTRACT_CONTENT_SHA256 or file_sha256(spec_path) != EXECUTION_SPEC_CONTENT_SHA256:
        raise RepairIntegrityError("tracked contract/spec content SHA mismatch")
    if _git_blob(contract_path, root) != P11_CONTRACT_GIT_BLOB or _git_blob(spec_path, root) != EXECUTION_SPEC_GIT_BLOB:
        raise RepairIntegrityError("tracked contract/spec Git blob mismatch")
    g = spec["g_sensitivity"]
    if g.get("exposure_count_field") != "clean_review_count" or g.get("thresholds") != list(R3_THRESHOLDS) or g.get("strata") != list(R3_STRATA):
        raise RepairContractError("G sensitivity contract mismatch")
    robustness = spec["robustness"]
    if robustness["R2"].get("no_retrain") is not True or robustness["R3"].get("thresholds") != list(R3_THRESHOLDS) or robustness["R3"].get("strata") != list(R3_STRATA) or robustness["R4"].get("promotion_or_relabeling") is not False or robustness["R5"].get("inferential_claim") is not False:
        raise RepairContractError("robustness contract mismatch")
    scoring = spec["scoring"]
    if scoring.get("no_model_fit") is not True or scoring.get("no_pickle_or_joblib") is not True or scoring.get("same_feature_row_exact_score") is not True or scoring.get("score_max_abs_tolerance") != 1e-12:
        raise RepairContractError("scoring firewall mismatch")
    return True


def validate_replacement_map(replacement_files: dict[str, bytes]) -> bool:
    if not isinstance(replacement_files, dict):
        raise RepairContractError("replacement map must be a mapping")
    names = set(replacement_files)
    allowed = set(REPLACEMENT_ALLOWLIST)
    if names != allowed:
        raise RepairContractError(f"replacement map must be exact; missing={sorted(allowed - names)}; extra={sorted(names - allowed)}")
    for name, content in replacement_files.items():
        if name in PRESERVED_DENYLIST or name not in FORMAL_FILES or not isinstance(content, bytes):
            raise RepairContractError("invalid replacement target " + name)
    return True



def validate_transaction_precommit(
    canonical_dir: Path | str,
    staging_dir: Path | str,
    *,
    expected_replacement_sha: dict[str, str],
    expected_preserved_sha: dict[str, str],
) -> dict[str, bool]:
    """Validate a staged transaction without changing any filesystem state."""
    def validate_ledger(value: Any, names: tuple[str, ...], label: str) -> None:
        if not isinstance(value, dict) or set(value) != set(names):
            raise RepairIntegrityError(label + " SHA ledger key mismatch")
        for name, digest in value.items():
            _validate_sha256_text(digest, label + "." + name)

    def formal_set(directory: Path, label: str) -> set[str]:
        if not directory.is_dir() or directory.is_symlink():
            raise RepairIntegrityError(label + " directory is missing or invalid")
        entries = list(directory.iterdir())
        if any(item.is_symlink() or not item.is_file() for item in entries):
            raise RepairIntegrityError(label + " contains non-file or alias entries")
        names = {item.name for item in entries}
        if names != set(FORMAL_FILES):
            raise RepairIntegrityError(label + " formal set mismatch")
        return names

    validate_ledger(expected_replacement_sha, REPLACEMENT_ALLOWLIST, "replacement")
    validate_ledger(expected_preserved_sha, PRESERVED_DENYLIST, "preserved")

    canonical_raw, staging_raw = Path(canonical_dir), Path(staging_dir)
    if canonical_raw.is_symlink() or staging_raw.is_symlink():
        raise RepairIntegrityError("transaction roots must not be aliases or symlinks")
    if not canonical_raw.exists() or not staging_raw.exists():
        raise RepairIntegrityError("transaction roots are missing")
    canonical, staging = canonical_raw.resolve(), staging_raw.resolve()
    if canonical == staging or canonical in staging.parents or staging in canonical.parents:
        raise RepairIntegrityError("canonical and staging directories must be isolated")
    if canonical.parent != staging.parent:
        raise RepairIntegrityError("transaction paths must share the same parent")
    if canonical.stat().st_dev != staging.stat().st_dev:
        raise RepairIntegrityError("transaction paths must share the same volume")
    if not staging.name.startswith(canonical.name + ".stage-"):
        raise RepairIntegrityError("staging directory name is not an unambiguous stage")

    formal_set(canonical, "canonical formal set")
    formal_set(staging, "staging formal set")

    parent = canonical.parent
    journal = parent / (canonical.name + ".transaction.json")
    if journal.exists() or journal.is_symlink():
        raise RepairIntegrityError("transaction journal collision")
    stage_entries = sorted(parent.glob(canonical.name + ".stage-*"))
    if any(path.is_symlink() or not path.is_dir() for path in stage_entries):
        raise RepairIntegrityError("ambiguous or invalid transaction staging")
    if len(stage_entries) != 1 or stage_entries[0].resolve() != staging:
        raise RepairIntegrityError("ambiguous or mismatched transaction staging")
    backup_entries = sorted(parent.glob(canonical.name + ".backup-*"))
    if backup_entries:
        raise RepairIntegrityError("transaction backup collision")

    for name in PRESERVED_DENYLIST:
        canonical_sha = file_sha256(canonical / name)
        if canonical_sha != expected_preserved_sha[name]:
            raise RepairIntegrityError("canonical preserved SHA mismatch: " + name)
        if file_sha256(staging / name) != expected_preserved_sha[name]:
            raise RepairIntegrityError("preserved staged SHA mismatch: " + name)
    for name in REPLACEMENT_ALLOWLIST:
        if file_sha256(staging / name) != expected_replacement_sha[name]:
            raise RepairIntegrityError("replacement staged SHA mismatch: " + name)

    return {
        "ready": True,
        "replacement_set_valid": True,
        "preserved_set_valid": True,
        "staged_sha_valid": True,
        "filesystem_preconditions_valid": True,
        "journal_preconditions_valid": True,
    }

def _journal_write(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=True, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def _matching_dirs(parent: Path, prefix: str) -> list[Path]:
    return sorted(path for path in parent.glob(prefix + "*") if path.is_dir())


def recover_transaction(canonical_dir: Path | str, *, validator: Callable[[Path], None] | None = None) -> None:
    canonical, parent = Path(canonical_dir), Path(canonical_dir).parent
    journal = parent / (canonical.name + ".transaction.json")
    stages, backups = _matching_dirs(parent, canonical.name + ".stage-"), _matching_dirs(parent, canonical.name + ".backup-")
    if len(stages) > 1 or len(backups) > 1:
        raise RepairIntegrityError("multiple transaction staging or backup directories")
    if not journal.exists():
        if stages or backups:
            raise RepairIntegrityError("stale transaction staging or backup without journal")
        return
    payload = json.loads(journal.read_text(encoding="utf-8"))
    state = payload.get("state")
    if state not in TRANSACTION_STATES:
        raise RepairIntegrityError("unknown transaction state")
    stage, backup = Path(payload["stage"]), Path(payload["backup"])
    if stages and stage != stages[0] or backups and backup != backups[0]:
        raise RepairIntegrityError("transaction journal path mismatch")
    if state in {"STAGING_PROMOTED", "POST_SWAP_VALIDATED", "COMMITTED", "CLEANUP_COMPLETE"} and canonical.exists():
        if validator is not None:
            validator(canonical)
        if backup.exists():
            shutil.rmtree(backup)
        if stage.exists():
            shutil.rmtree(stage)
        journal.unlink(missing_ok=True)
        return
    if not canonical.exists() and backup.exists():
        os.replace(backup, canonical)
        if stage.exists():
            shutil.rmtree(stage)
        journal.unlink(missing_ok=True)
        return
    raise RepairIntegrityError("stale or incomplete transaction requires explicit recovery")


def run_staged_transaction(
    canonical_dir: Path | str,
    replacement_files: dict[str, bytes],
    *,
    validator: Callable[[Path], None],
    fault_phase: str | None = None,
    deferred_names: tuple[str, ...] = (),
    deferred_builder: Callable[[Path, str], bytes] | None = None,
) -> dict[str, Any]:
    deferred = tuple(deferred_names)

    expected_deferred = (
        FORMAL_FILES[11],
        FORMAL_FILES[12],
    )

    if deferred:
        if deferred != expected_deferred:
            raise RepairContractError(
                "deferred replacement order must be "
                "11 summary then 12 provenance"
            )

        if not isinstance(replacement_files, dict):
            raise RepairContractError(
                "replacement map must be a mapping"
            )

        early_names = set(replacement_files)
        deferred_set = set(deferred)
        allowed = set(REPLACEMENT_ALLOWLIST)

        if early_names & deferred_set:
            raise RepairContractError(
                "early and deferred replacement targets overlap"
            )

        if (early_names | deferred_set) != allowed:
            raise RepairContractError(
                "combined replacement set must be exact"
            )

        for name, content in replacement_files.items():
            if (
                name in PRESERVED_DENYLIST
                or name not in FORMAL_FILES
                or not isinstance(content, bytes)
            ):
                raise RepairContractError(
                    "invalid replacement target " + name
                )

        if not callable(deferred_builder):
            raise RepairContractError(
                "deferred replacement builder must be callable"
            )

    else:
        validate_replacement_map(replacement_files)

        if deferred_builder is not None:
            raise RepairContractError(
                "deferred builder requires deferred names"
            )

    canonical = Path(canonical_dir)
    parent = canonical.parent

    recover_transaction(
        canonical,
        validator=validator,
    )

    if (
        not canonical.is_dir()
        or set(
            item.name
            for item in canonical.iterdir()
            if item.is_file()
        )
        != set(FORMAL_FILES)
    ):
        raise RepairIntegrityError(
            "canonical formal directory is not the exact frozen set"
        )

    stage = Path(
        tempfile.mkdtemp(
            prefix=canonical.name + ".stage-",
            dir=parent,
        )
    )

    backup = parent / (
        canonical.name
        + ".backup-"
        + next(tempfile._get_candidate_names())
    )

    journal = parent / (
        canonical.name + ".transaction.json"
    )

    if backup.exists() or journal.exists():
        shutil.rmtree(stage)

        raise RepairIntegrityError(
            "transaction backup or journal collision"
        )

    expected_preserved_sha = {
        name: file_sha256(canonical / name)
        for name in PRESERVED_DENYLIST
    }

    state: dict[str, Any] = {
        "state": "INIT",
        "canonical": str(canonical),
        "stage": str(stage),
        "backup": str(backup),
        "preserved_files": list(PRESERVED_DENYLIST),
        "replacement_files": list(REPLACEMENT_ALLOWLIST),
        "authorization": "p11_unified_repair",
    }

    try:
        shutil.copytree(
            canonical,
            stage,
            dirs_exist_ok=True,
        )

        materialized = dict(replacement_files)

        # Stage-independent replacements are visible first.
        for name, content in replacement_files.items():
            (stage / name).write_bytes(content)

        # Ordered staged derivation:
        # 11 sees final staged 02-10;
        # 12 sees final staged 00-11 including new 11.
        for name in deferred:
            content = deferred_builder(stage, name)

            if not isinstance(content, bytes):
                raise RepairContractError(
                    "deferred replacement must be bytes: "
                    + name
                )

            (stage / name).write_bytes(content)
            materialized[name] = content

        # At this point the complete replacement set must exist.
        validate_replacement_map(materialized)

        expected_replacement_sha = {
            name: hashlib.sha256(content)
            .hexdigest()
            .upper()
            for name, content in materialized.items()
        }

        validate_transaction_precommit(
            canonical,
            stage,
            expected_replacement_sha=expected_replacement_sha,
            expected_preserved_sha=expected_preserved_sha,
        )

        validator(stage)

        _journal_write(journal, state)

        state["state"] = "STAGING_READY"
        _journal_write(journal, state)

        state["state"] = "STAGING_VALIDATED"
        _journal_write(journal, state)

        if fault_phase == "before_commit":
            raise RuntimeError(
                "fault injection before_commit"
            )

        os.replace(canonical, backup)

        state["state"] = "CANONICAL_MOVED_TO_BACKUP"
        _journal_write(journal, state)

        if fault_phase == "after_backup":
            raise RuntimeError(
                "fault injection after_backup"
            )

        os.replace(stage, canonical)

        state["state"] = "STAGING_PROMOTED"
        _journal_write(journal, state)

        if fault_phase == "after_swap":
            raise RuntimeError(
                "fault injection after_swap"
            )

        validator(canonical)

        state["state"] = "POST_SWAP_VALIDATED"
        _journal_write(journal, state)

        if fault_phase == "before_cleanup":
            raise RuntimeError(
                "fault injection before_cleanup"
            )

        state["state"] = "COMMITTED"
        _journal_write(journal, state)

        shutil.rmtree(backup)

        state["state"] = "CLEANUP_COMPLETE"
        _journal_write(journal, state)

        journal.unlink(missing_ok=True)

        return {
            "status": "PASS",
            "phase": "CLEANUP_COMPLETE",
            "canonical": str(canonical),
            "writes": len(REPLACEMENT_ALLOWLIST),
        }

    except Exception:
        if canonical.exists() and backup.exists():
            shutil.rmtree(canonical)
            os.replace(backup, canonical)

        elif not canonical.exists() and backup.exists():
            os.replace(backup, canonical)

        if stage.exists():
            shutil.rmtree(stage)

        journal.unlink(missing_ok=True)

        raise


def run_unified_maintenance_transaction(
    *,
    canonical_dir: Path | str,
    replacement_files: dict[str, bytes],
    validator: Callable[[Path], None],
    fault_phase: str | None = None,
    deferred_names: tuple[str, ...] = (),
    deferred_builder: Callable[[Path, str], bytes] | None = None,
) -> dict[str, Any]:
    return run_staged_transaction(
        canonical_dir,
        replacement_files,
        validator=validator,
        fault_phase=fault_phase,
        deferred_names=deferred_names,
        deferred_builder=deferred_builder,
    )


def _historical_verify_existing_artifacts(context: dict[str, Any], reference_service: Any) -> dict[str, Any]:
    if context.get("model_fits") != 0 or context.get("writes") != 0:
        raise RepairIntegrityError("verify-existing execution firewall mismatch")
    rows, scores, models, artifacts = _required(context, "rows", "verify-existing"), _required(context, "scores", "verify-existing"), _required(context, "models", "verify-existing"), _required(context, "artifacts", "verify-existing")
    expected_product = reference_service.product_metrics(rows, scores, models)
    if len(expected_product) != len(artifacts["03_product_row_metrics.csv"]):
        raise RepairIntegrityError("verify-existing product ledger row count mismatch")
    if "07_r2_alternative_image_sensitivity.csv" in artifacts:
        validate_r2_rows(rows, artifacts["07_r2_alternative_image_sensitivity.csv"], models, context["primary_scores"], context["sensitivity_scores"])
    if "08_r3_g_exposure_sensitivity.csv" in artifacts:
        validate_r3_rows(artifacts["08_r3_g_exposure_sensitivity.csv"])
    return {"status": "PASS", "writes": 0, "model_fits": 0, "independent_reference": True, "verified_files": list(FORMAL_FILES)}


def _historical_verify_recompute_artifacts(context: dict[str, Any], reference_service: Any) -> dict[str, Any]:
    if context.get("model_fits") != 0 or context.get("writes") != 0:
        raise RepairIntegrityError("verify-recompute execution firewall mismatch")
    rows, scores, models, artifacts = _required(context, "rows", "verify-recompute"), _required(context, "scores", "verify-recompute"), _required(context, "models", "verify-recompute"), _required(context, "artifacts", "verify-recompute")
    expected = {"03_product_row_metrics.csv": reference_service.product_metrics(rows, scores, models), "04_group_metrics.csv": reference_service.group_metrics(rows, scores, models, context["bootstrap_plan_sha256"])}
    for name, value in expected.items():
        if artifacts.get(name) != value:
            raise RepairIntegrityError("verify-recompute artifact mismatch: " + name)
    return {"status": "PASS", "writes": 0, "model_fits": 0, "independent_recompute": True, "verified_files": list(FORMAL_FILES[1:11])}


def _validate_hex_text(value: Any, field: str, length: int, description: str) -> None:
    if not isinstance(value, str) or len(value) != length or any(char not in "0123456789abcdefABCDEF" for char in value):
        raise RepairIntegrityError(field + " is not a valid " + description)


def _validate_git_blob_text(value: Any, field: str) -> None:
    _validate_hex_text(value, field, 40, "Git blob SHA")


def _validate_sha256_text(value: Any, field: str) -> None:
    _validate_hex_text(value, field, 64, "SHA256")


def _git_blob(path: Path, root: Path) -> str | None:
    relative = path.relative_to(root).as_posix()
    result = subprocess.run(["git", "rev-parse", "HEAD:" + relative], cwd=root, capture_output=True, text=True, check=False)
    return result.stdout.strip() or None


PROVENANCE_REQUIRED_KEYS = frozenset({
    "stage", "baseline_main_commit", "execution_version",
    "maintenance_code_commit", "producer_commit", "producer_path",
    "maintenance_producer_path", "producer_file_sha256",
    "maintenance_producer_sha256", "producer_git_blob",
    "oracle_commit", "oracle_path", "reference_oracle_path",
    "oracle_file_sha256", "reference_oracle_sha256", "oracle_git_blob",
    "contract_path", "p11_contract_path", "contract_git_blob",
    "p11_contract_git_blob", "contract_content_sha256",
    "p11_contract_sha256", "spec_path", "p11_execution_spec_path",
    "spec_git_blob", "p11_execution_spec_git_blob",
    "spec_content_sha256", "execution_spec_sha256",
    "opening_seal_sha256", "old_invalid_formal_sha256",
    "old_artifact_status", "new_formal_sha256", "model_fits",
    "formal_primary_rescoring", "formal_maintenance_primary_rescoring",
    "performance_based_decision", "metric_values_used_for_repair_authority",
    "model_ranking_used_for_repair_authority",
    "threshold_selection_used_for_repair_authority",
    "label_definition_selection_used_for_repair_authority",
    "scientific_protocol_changed",
})


def _git_head(root: Path) -> str:
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=False)
    if result.returncode != 0 or not result.stdout.strip():
        raise RepairIntegrityError("cannot resolve repository HEAD")
    return result.stdout.strip()


def _git_object_sha256(relative_path: str, root: Path) -> str:
    result = subprocess.run(
        ["git", "cat-file", "blob", f"HEAD:{relative_path}"],
        cwd=root,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise RepairIntegrityError("cannot read Git object " + relative_path)
    return hashlib.sha256(result.stdout).hexdigest().upper()


def _pre_maintenance_old_ledger(root: Path) -> dict[str, str]:
    authority = json.loads(
        (root / "config/modeling/p11_post_open_scientific_maintenance_authorization.json").read_text(encoding="utf-8")
    )
    ledger = authority.get("pre_maintenance_formal_sha256")
    if not isinstance(ledger, dict):
        raise RepairIntegrityError("pre-maintenance formal ledger authority is missing")
    expected_names = set(REPLACEMENT_ALLOWLIST[:6])
    if set(ledger) != set(FORMAL_FILES) or not expected_names.issubset(ledger):
        raise RepairIntegrityError("pre-maintenance formal ledger key set is not exact")
    return {name: str(ledger[name]).upper() for name in REPLACEMENT_ALLOWLIST[:6]}


def build_provenance_bindings(
    *,
    base: dict[str, Any],
    maintenance_commit: str,
    old_sha: dict[str, str],
    new_sha: dict[str, str],
    producer_git_blob: str | None = None,
    oracle_git_blob: str | None = None,
    contract_git_blob: str | None = None,
    spec_git_blob: str | None = None,
    opening_seal_sha256: str = P11_OPENING_SEAL_SHA256,
    producer_path: str = "scripts/modeling/p11_locked_test_repair.py",
    oracle_path: str | None = None,
    contract_path: str = "config/modeling/p11_locked_test_evaluation_contract.json",
    spec_path: str = "config/modeling/p11_locked_test_execution_spec.json",
) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[2]
    oracle_path = oracle_path or REFERENCE_ORACLE_PATH
    producer_file, oracle_file = root / producer_path, root / oracle_path
    contract_file, spec_file = root / contract_path, root / spec_path
    head = _git_head(root)
    producer_sha = file_sha256(producer_file)
    oracle_sha = file_sha256(oracle_file)
    producer_blob = producer_git_blob or _git_blob(producer_file, root)
    oracle_blob = oracle_git_blob or _git_blob(oracle_file, root)
    contract_blob = contract_git_blob or _git_blob(contract_file, root)
    spec_blob = spec_git_blob or _git_blob(spec_file, root)
    if not all((producer_blob, oracle_blob, contract_blob, spec_blob)):
        raise RepairIntegrityError("provenance Git identity is incomplete")
    value = {
        "stage": "P11 locked-test evaluation freeze",
        "baseline_main_commit": P11_BASELINE_MAIN_COMMIT,
        "execution_version": "p11_exec_v1.0",
        "maintenance_code_commit": maintenance_commit,
        "producer_commit": maintenance_commit,
        "producer_path": producer_path,
        "maintenance_producer_path": producer_path,
        "producer_file_sha256": producer_sha,
        "maintenance_producer_sha256": producer_sha,
        "producer_git_blob": producer_blob,
        "oracle_commit": head,
        "oracle_path": oracle_path,
        "reference_oracle_path": oracle_path,
        "oracle_file_sha256": oracle_sha,
        "reference_oracle_sha256": oracle_sha,
        "oracle_git_blob": oracle_blob,
        "contract_path": contract_path,
        "p11_contract_path": contract_path,
        "contract_git_blob": contract_blob,
        "p11_contract_git_blob": contract_blob,
        "contract_content_sha256": P11_CONTRACT_CONTENT_SHA256,
        "p11_contract_sha256": P11_CONTRACT_CONTENT_SHA256,
        "spec_path": spec_path,
        "p11_execution_spec_path": spec_path,
        "spec_git_blob": spec_blob,
        "p11_execution_spec_git_blob": spec_blob,
        "spec_content_sha256": _git_object_sha256(spec_path, root),
        "execution_spec_sha256": _git_object_sha256(spec_path, root),
        "opening_seal_sha256": opening_seal_sha256,
        "old_invalid_formal_sha256": dict(old_sha),
        "old_artifact_status": {name: "INVALID_SUPERSEDED" for name in old_sha},
        "new_formal_sha256": dict(new_sha),
        "model_fits": 0,
        "formal_primary_rescoring": 0,
        "formal_maintenance_primary_rescoring": 0,
        "performance_based_decision": False,
        "metric_values_used_for_repair_authority": False,
        "model_ranking_used_for_repair_authority": False,
        "threshold_selection_used_for_repair_authority": False,
        "label_definition_selection_used_for_repair_authority": False,
        "scientific_protocol_changed": False,
    }
    if set(value) != set(PROVENANCE_REQUIRED_KEYS):
        raise RepairIntegrityError("provenance builder schema is not exact")
    return value


def build_expected_provenance_bindings(
    *,
    formal_dir: Path | str | None = None,
    maintenance_commit: str | None = None,
    old_sha: dict[str, str] | None = None,
    new_sha: dict[str, str] | None = None,
) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[2]
    directory = Path(formal_dir) if formal_dir is not None else root / "data/processed/modeling_p11_5180/p11_locked_test_evaluation_freeze"
    old = dict(old_sha) if old_sha is not None else _pre_maintenance_old_ledger(root)
    new = dict(new_sha) if new_sha is not None else {name: file_sha256(directory / name) for name in FORMAL_LEDGER_FILES}
    head = maintenance_commit or _git_head(root)
    return build_provenance_bindings(
        base={},
        maintenance_commit=head,
        old_sha=old,
        new_sha=new,
    )


def build_expected_provenance_bindings_for_test() -> dict[str, Any]:
    return build_expected_provenance_bindings()


def _validate_provenance_shape(value: dict[str, Any]) -> bool:
    if not isinstance(value, dict):
        raise RepairIntegrityError("provenance must be an object")
    if "provenance_sha256" in value or "self_sha256" in value:
        raise RepairIntegrityError("provenance self-hash is forbidden")
    if set(value) != set(PROVENANCE_REQUIRED_KEYS):
        raise RepairIntegrityError("provenance schema must be exact")
    for field in ("producer_path", "oracle_path", "contract_path", "spec_path"):
        _required(value, field, "provenance")
    if value["producer_path"] != "scripts/modeling/p11_locked_test_repair.py" or value["oracle_path"] != REFERENCE_ORACLE_PATH:
        raise RepairIntegrityError("provenance code path mismatch")
    if value["contract_path"] != "config/modeling/p11_locked_test_evaluation_contract.json" or value["spec_path"] != "config/modeling/p11_locked_test_execution_spec.json":
        raise RepairIntegrityError("provenance contract/spec path mismatch")
    if value["baseline_main_commit"] != P11_BASELINE_MAIN_COMMIT or value["execution_version"] != "p11_exec_v1.0":
        raise RepairIntegrityError("provenance frozen identity mismatch")
    for field in ("producer_commit", "maintenance_code_commit", "oracle_commit"):
        _validate_hex_text(value[field], field, 40, "Git commit SHA")
    for field in ("producer_git_blob", "oracle_git_blob", "contract_git_blob", "spec_git_blob"):
        _validate_git_blob_text(value[field], field)
    for field in ("producer_file_sha256", "oracle_file_sha256", "contract_content_sha256", "spec_content_sha256", "opening_seal_sha256"):
        _validate_sha256_text(value[field], field)
    if value["opening_seal_sha256"] != P11_OPENING_SEAL_SHA256:
        raise RepairIntegrityError("provenance opening seal binding mismatch")
    if type(value["model_fits"]) is not int or value["model_fits"] != 0:
        raise RepairIntegrityError("provenance model-fit count mismatch")
    if type(value["formal_primary_rescoring"]) is not int or value["formal_primary_rescoring"] != 0:
        raise RepairIntegrityError("provenance primary-rescoring count mismatch")
    if type(value["formal_maintenance_primary_rescoring"]) is not int or value["formal_maintenance_primary_rescoring"] != 0:
        raise RepairIntegrityError("provenance maintenance-rescoring count mismatch")
    for field in (
        "performance_based_decision", "metric_values_used_for_repair_authority",
        "model_ranking_used_for_repair_authority", "threshold_selection_used_for_repair_authority",
        "label_definition_selection_used_for_repair_authority", "scientific_protocol_changed",
    ):
        if type(value[field]) is not bool or value[field] is not False:
            raise RepairIntegrityError("provenance firewall mismatch")
    old_ledger, new_ledger = value["old_invalid_formal_sha256"], value["new_formal_sha256"]
    if set(old_ledger) != set(REPLACEMENT_ALLOWLIST[:6]):
        raise RepairIntegrityError("provenance old invalid ledger must bind exact 03/05/06/07/08/09 set")
    if set(new_ledger) != set(FORMAL_LEDGER_FILES):
        raise RepairIntegrityError("provenance formal ledger must bind exact 00-11 set")
    if set(value["old_artifact_status"]) != set(REPLACEMENT_ALLOWLIST[:6]):
        raise RepairIntegrityError("provenance old status ledger must be exact")
    for field, ledger in (("old_invalid_formal_sha256", old_ledger), ("new_formal_sha256", new_ledger)):
        for name, digest in ledger.items():
            _validate_sha256_text(digest, field + "." + name)
    return True


def validate_candidate_provenance(
    value: dict[str, Any],
    *,
    expected_bindings: dict[str, Any] | None = None,
) -> bool:
    _validate_provenance_shape(value)
    if expected_bindings is not None:
        if set(expected_bindings) != set(PROVENANCE_REQUIRED_KEYS):
            raise RepairIntegrityError("expected provenance schema is not exact")
        if value != expected_bindings:
            raise RepairIntegrityError("provenance external authority mismatch")
    return True

def validate_active_verification_context(context: dict[str, Any]) -> bool:
    required = {
        "rows", "models", "artifacts", "reference_expected_artifacts",
        "opening_seal", "opening_seal_path", "identity", "writes",
        "model_fits", "primary_verification_rescores", "formal_primary_rescoring",
        "live_identity", "r4_product_authority", "r5_categories",
    }
    missing = sorted(required - set(context))
    if missing:
        raise RepairIntegrityError("active verification context missing: " + repr(missing))
    if context["writes"] != 0 or context["model_fits"] != 0 or context["formal_primary_rescoring"] != 0:
        raise RepairIntegrityError("active verification execution firewall mismatch")
    artifacts = context["artifacts"]
    expected = context["reference_expected_artifacts"]
    if set(artifacts) != set(FORMAL_FILES) or set(expected) != set(FORMAL_FILES):
        raise RepairIntegrityError("active verification artifact authority must be exact 00-12")
    identity = _required(context, "identity", "active verification")
    for field in ("contract", "execution_spec", "upstream_sha256", "reference_oracle", "expected_artifact_authority"):
        _required(identity, field, "active identity")
    if list(identity["expected_artifact_authority"]) != list(FORMAL_FILES):
        raise RepairIntegrityError("active artifact authority mismatch")
    upstream = identity["upstream_sha256"]
    if set(upstream) != {"p7_d", "p8_b", "p9", "p10"} or any(not value for value in upstream.values()):
        raise RepairIntegrityError("active upstream SHA ledger is incomplete")
    for name in ("contract", "execution_spec", "reference_oracle"):
        if not isinstance(identity[name], dict):
            raise RepairIntegrityError("active identity namespace is malformed: " + name)
    if context["live_identity"] is True:
        validate_opening_seal(context["opening_seal"], context["opening_seal_path"])
        expected_provenance = _required(context, "expected_provenance", "active verification")
        validate_candidate_provenance(
            artifacts["12_p11_provenance.json"],
            expected_bindings=expected_provenance,
        )
    return True


def historical_command_disabled(command: str) -> None:
    raise RepairContractError(command + " is historical and cannot write the P11 formal directory")


def prepare_contract(spec: dict[str, Any] | None = None, contract: dict[str, Any] | None = None) -> dict[str, Any]:
    if spec is not None and contract is not None:
        validate_execution_spec(spec, contract)
    return {"status": "PASS", "writes": 0, "model_fits": 0, "contract_only": True}


def producer_bootstrap_metrics(rows: list[dict[str, Any]], scores: dict[str, np.ndarray], models: dict[str, dict[str, Any]], iterations: int = 5000, seed: int = 20260818, product: list[dict[str, Any]] | None = None) -> tuple[list[dict[str, Any]], str]:
    keys = sorted({_text(row, "primary_response_sha256") for row in rows})
    draws, plan_sha = producer_bootstrap_plan(keys, iterations, seed)
    point_rows = product if product is not None else producer_product_metrics(rows, scores, models)
    point = {(row["model_id"], metric): float(row[metric]) for row in point_rows for metric in CI_FIELDS}
    groups = {key: [index for index, row in enumerate(rows) if _text(row, "primary_response_sha256") == key] for key in keys}
    names = [_text(row, "parent_asin") for row in rows]
    result = []
    for model_id, model in models.items():
        labels = _binary(rows, _text(model, "outcome"))
        values = np.asarray(scores[model_id], dtype=np.float64)
        distributions = {metric: [] for metric in CI_FIELDS}
        for replicate_index, draw in enumerate(draws):
            selected, occurrences = [], []
            for occurrence, group_index in enumerate(draw):
                members = groups[keys[int(group_index)]]
                selected.extend(members)
                occurrences.extend([occurrence] * len(members))
            selected_array = np.asarray(selected, dtype=np.int64)
            metrics = producer_metric_bundle(labels[selected_array], values[selected_array], [names[index] for index in selected], occurrences=occurrences, context=f"{model_id}/replicate={replicate_index}")
            for metric in CI_FIELDS:
                distributions[metric].append(metrics[metric])
        for metric in CI_FIELDS:
            distribution = np.asarray(distributions[metric], dtype=np.float64)
            result.append({"model_id": model_id, "outcome": _text(model, "outcome"), "track": _text(model, "track"), "metric": metric, "point_estimate": point[(model_id, metric)], "ci_lower": float(np.quantile(distribution, 0.025, method="linear")), "ci_upper": float(np.quantile(distribution, 0.975, method="linear")), "iterations": iterations, "cluster_count": len(keys), "seed": seed, "bootstrap_plan_sha256": plan_sha, "status": "ok"})
    return result, plan_sha


def producer_score(raw: Iterable[float], mean: Iterable[float], scale: Iterable[float], coef: Iterable[float], intercept: float) -> float:
    x, mu, sigma, beta = (np.asarray(list(value), dtype=np.float64) for value in (raw, mean, scale, coef))
    if x.shape != mu.shape or x.shape != sigma.shape or x.shape != beta.shape or np.any(sigma == 0) or not np.all(np.isfinite(x)):
        raise RepairIntegrityError("frozen score parameter shape/domain mismatch")
    linear = float(intercept + np.dot((x - mu) / sigma, beta))
    return float(1.0 / (1.0 + math.exp(-min(700.0, max(-700.0, linear)))))


def producer_r2_metrics(rows: list[dict[str, Any]], primary_scores: dict[str, np.ndarray], sensitivity_features: dict[str, np.ndarray], models: dict[str, dict[str, Any]], parameters: dict[str, dict[str, Iterable[float] | float]]) -> list[dict[str, Any]]:
    available = r2_available_indices(rows)
    result = []
    for model_id, model in models.items():
        if model_id not in primary_scores or model_id not in sensitivity_features or model_id not in parameters:
            raise RepairIntegrityError("missing R2 model binding " + model_id)
        primary = np.asarray(primary_scores[model_id], dtype=np.float64)
        features = np.asarray(sensitivity_features[model_id], dtype=np.float64)
        parameter = parameters[model_id]
        if primary.ndim != 1 or len(primary) != len(rows) or features.ndim != 2 or len(features) != len(rows):
            raise RepairIntegrityError("R2 score/feature shape mismatch")
        for field in ("mean", "scale", "coef", "intercept"):
            _required(parameter, field, "R2")
        for index in available:
            sensitivity = producer_score(features[index], parameter["mean"], parameter["scale"], parameter["coef"], float(parameter["intercept"]))
            primary_value = float(primary[index])
            result.append({"parent_asin": _text(rows[index], "parent_asin"), "primary_response_sha256": _text(rows[index], "primary_response_sha256"), "primary_feature_row_index": _text(rows[index], "primary_feature_row_index"), "sensitivity_feature_row_index": _text(rows[index], "sensitivity_feature_row_index"), "outcome": _text(model, "outcome"), "track": _text(model, "track"), "model_id": model_id, "primary_score": primary_value, "sensitivity_score": sensitivity, "score_delta": sensitivity - primary_value, "no_retrain": True, "same_split_grouping": True})
    return result

def _normal_value(value: Any) -> str:
    if isinstance(value, bool):
        return "1" if value else "0"
    if value is None:
        return ""
    text = str(value)
    if text.strip().lower() in {"true", "1"}:
        return "1"
    if text.strip().lower() in {"false", "0"}:
        return "0"
    return text


def _exact_rows(expected: list[dict[str, Any]], actual: list[dict[str, Any]], label: str) -> None:
    if len(expected) != len(actual):
        raise RepairIntegrityError(label + " row count mismatch")
    expected_rows = [tuple(sorted((key, _normal_value(value)) for key, value in row.items())) for row in expected]
    actual_rows = [tuple(sorted((key, _normal_value(value)) for key, value in row.items())) for row in actual]
    if expected_rows != actual_rows:
        raise RepairIntegrityError(label + " content mismatch")


def _exact_artifact_set(artifacts: dict[str, Any]) -> None:
    if set(artifacts) != set(FORMAL_FILES):
        raise RepairIntegrityError("verify artifact set must be exact 00-12")
    for name in PRESERVED_DENYLIST:
        if name not in artifacts:
            raise RepairIntegrityError("preserved artifact missing: " + name)


def _exact_artifact_value(expected: Any, actual: Any, label: str) -> None:
    if isinstance(expected, list) and isinstance(actual, list):
        _exact_rows(expected, actual, label)
        return
    if isinstance(expected, dict) and isinstance(actual, dict):
        if expected != actual:
            raise RepairIntegrityError(label + " content mismatch")
        return
    if _normal_value(expected) != _normal_value(actual):
        raise RepairIntegrityError(label + " content mismatch")


def verify_existing_artifacts(context: dict[str, Any], reference_service: Any = None) -> dict[str, Any]:
    validate_active_verification_context(context)
    artifacts = context["artifacts"]
    expected = context["reference_expected_artifacts"]
    _exact_artifact_set(artifacts)
    for name in FORMAL_FILES:
        _exact_artifact_value(expected[name], artifacts[name], "verify-existing " + name)
    validate_r3_rows(artifacts["08_r3_g_exposure_sensitivity.csv"])
    validate_r4_core_binding(artifacts["09_r4_label_definition_robustness.csv"], context["r4_product_authority"])
    validate_r5_source_binding(artifacts["10_r5_image_exception_diagnostics.csv"], context["r5_categories"])
    validate_tracked_report(artifacts["11_p11_summary.json"])
    return {
        "status": "PASS",
        "writes": 0,
        "model_fits": 0,
        "primary_verification_rescores": context["primary_verification_rescores"],
        "formal_primary_rescoring": 0,
        "independent_reference": True,
        "verified_files": list(FORMAL_FILES),
    }


def verify_recompute_artifacts(context: dict[str, Any], reference_service: Any = None) -> dict[str, Any]:
    validate_active_verification_context(context)
    artifacts = context["artifacts"]
    reconstructed = _required(context, "reconstructed_artifacts", "verify-recompute")
    expected_names = set(FORMAL_FILES[1:11])
    if set(reconstructed) != expected_names:
        raise RepairIntegrityError("reconstructed artifact coverage must be exact 01-10")
    for name in FORMAL_FILES[1:11]:
        _exact_artifact_value(reconstructed[name], artifacts[name], "verify-recompute " + name)
    validate_r3_rows(artifacts["08_r3_g_exposure_sensitivity.csv"])
    validate_r4_core_binding(artifacts["09_r4_label_definition_robustness.csv"], context["r4_product_authority"])
    validate_r5_source_binding(artifacts["10_r5_image_exception_diagnostics.csv"], context["r5_categories"])
    return {
        "status": "PASS",
        "writes": 0,
        "model_fits": 0,
        "primary_verification_rescores": context["primary_verification_rescores"],
        "formal_primary_rescoring": 0,
        "independent_recompute": True,
        "fits": 0,
        "verified_files": list(FORMAL_FILES[1:11]),
    }


def validate_tracked_report(report: dict[str, Any]) -> bool:
    if _text(report, "status") == "CLOSED":
        raise RepairIntegrityError("tracked P11 report cannot claim CLOSED")
    aggregates = _required(report, "scientific_aggregates", "tracked report")
    expected_counts = {"product": 6, "group": 6, "bootstrap": 24, "r1": 6, "r2": 6, "r3": 96, "r4": 16, "r5": 24}
    if set(aggregates) != set(expected_counts):
        raise RepairIntegrityError("tracked report aggregate namespace mismatch")
    for name, count in expected_counts.items():
        if not isinstance(aggregates[name], list) or len(aggregates[name]) != count:
            raise RepairIntegrityError("tracked report aggregate count mismatch: " + name)
    def reject_row_identity(value: Any) -> None:
        if isinstance(value, dict):
            if "parent_asin" in value:
                raise RepairIntegrityError("tracked report contains row-level parent_asin")
            for child in value.values():
                reject_row_identity(child)
        elif isinstance(value, list):
            for child in value:
                reject_row_identity(child)
    reject_row_identity(aggregates)
    return True
