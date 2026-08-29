#!/usr/bin/env python3
"""P8-A aggregate-only analysis contract freeze and verifier.

This module reads frozen P7 image manifests, the canonical product label table,
and frozen G diagnostics.  It never extracts image features, fits a model,
assigns a split, calls a model API, or makes a network request.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping


ROOT = Path(__file__).resolve().parents[2]
CONTRACT_PATH = ROOT / "config" / "modeling" / "p8_a_analysis_contract.json"
FORMAL_DIR = ROOT / "data" / "processed" / "modeling_readiness_p8_5180" / "p8_a_analysis_contract"
FORMAL_FILES = (
    "01_upstream_freeze_audit.json",
    "02_modeling_population_and_outcome_audit.json",
    "03_g_role_and_sensitivity_audit.json",
    "04_analysis_contract_snapshot.json",
    "05_p8_a_provenance.json",
)
PROVENANCE_FILENAME = "05_p8_a_provenance.json"
PRIMARY_STRATIFICATION_OUTCOME = "has_any_outer_imagery_observed"
CORE_MIN_PRODUCTS = 40

LABEL_PATH = ROOT / "data" / "processed" / "affective_imagery_labels_v21_5180" / "39_product_imagery_labels_v21.csv"
P7D_MANIFEST_PATH = ROOT / "data" / "processed" / "retail_outer_package_images_p7_5180" / "p7_d_final_image_freeze" / "04_final_image_manifest.csv"
G_METHOD_PATH = ROOT / "data" / "processed" / "review_exposure_pu_diagnostics_v21_5180" / "06_method_summary.json"

P7_VERIFY_COMMANDS = {
    "p7_a": [
        sys.executable,
        str(ROOT / "scripts/images/audit_p7_source_inventory.py"),
        "--input",
        "data/processed/review_matching_5180/01_valid_products.csv",
        "--raw-metadata",
        "data/meta_Grocery_and_Gourmet_Food.jsonl/meta_Grocery_and_Gourmet_Food.jsonl",
        "--output-dir",
        "data/processed/retail_outer_package_images_p7_5180/p7_a_source_inventory",
        "--verify-existing",
    ],
    "p7_b": [sys.executable, str(ROOT / "scripts/images/acquire_p7_primary_assets.py"), "--verify-existing"],
    "p7_c": [sys.executable, str(ROOT / "scripts/images/p7_c_primary_manifest_qa.py"), "--verify-existing"],
    "p7_d": [sys.executable, str(ROOT / "scripts/images/p7_d_final_image_freeze.py"), "--verify-existing"],
}

CORE_DIMENSIONS = (
    "general_visual_appeal",
    "cute_friendly",
    "premium_refined",
    "gift_presentation",
    "simple_modern",
    "natural_botanical",
    "calming_soft",
    "cheerful_colorful",
    "traditional_vintage",
    "negative_appearance",
)
MAIN_STATUSES = frozenset({"frozen_primary", "frozen_primary_with_qa_exception"})
ELIGIBILITY_MARKERS = ("eligible_", "_dimension_keep_", "eligibility")
FORBIDDEN_FORMAL_TERMS = ("feature_matrix", "embedding", "split_assignment", "model_performance")


class ContractError(RuntimeError):
    """Raised when an upstream or P8-A contract invariant is not satisfied."""


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest().upper()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def git_output(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def git_blob_sha256(commit: str, repo_path: str | Path) -> str:
    path = Path(repo_path).as_posix()
    return sha256_bytes(subprocess.check_output(["git", "cat-file", "blob", f"{commit}:{path}"], cwd=ROOT))


def repo_path(value: str | Path) -> Path:
    normalized = str(value).replace("\\", "/")
    path = Path(normalized)
    return path if path.is_absolute() else ROOT / path


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ContractError(f"expected JSON object: {path}")
    return value


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def canonical_outcome_hierarchy(secondary_confirmatory: Iterable[str] = ()) -> dict[str, Any]:
    core = [f"{dimension}_observed_positive_core" for dimension in CORE_DIMENSIONS]
    pilot = [f"{dimension}_observed_positive_pilot" for dimension in CORE_DIMENSIONS]
    robust = [f"{dimension}_observed_positive_robust" for dimension in CORE_DIMENSIONS]
    return {
        "primary": PRIMARY_STRATIFICATION_OUTCOME,
        "secondary_confirmatory": list(secondary_confirmatory),
        "robustness": ["has_any_all_level_imagery_evidence", "*_observed_positive_pilot", "*_observed_positive_robust"],
        "canonical_pu_outcomes": [PRIMARY_STRATIFICATION_OUTCOME, "has_any_all_level_imagery_evidence", *core, *pilot, *robust],
    }


def canonical_pu_outcomes(contract: Mapping[str, Any] | None = None) -> tuple[str, ...]:
    values = (contract or {}).get("canonical_pu_outcomes")
    return tuple(values) if values is not None else tuple(canonical_outcome_hierarchy()["canonical_pu_outcomes"])


def main_population_status(status: str) -> bool:
    """Return whether a P7-D row belongs to the frozen main image population."""
    return status in MAIN_STATUSES


def duplicate_group_key(row: Mapping[str, Any]) -> str:
    key = str(row.get("primary_response_sha256") or "").strip().upper()
    if not key:
        raise ContractError("main population row is missing primary_response_sha256")
    return key


def predictor_is_allowed(name: str) -> bool:
    """The P8-A predictor firewall accepts image-derived names only by explicit prefix."""
    normalized = name.strip().lower()
    forbidden = (
        "label", "outcome", "review", "metadata", "ocr", "asin", "title", "brand",
        "store", "categor", "tea_type", "product_form", "web",
    )
    return normalized.startswith("image_") and not any(marker in normalized for marker in forbidden)


def outcome_is_eligible(name: str, outcomes: Iterable[str]) -> bool:
    return name in set(outcomes)


def _as_nonnegative_int(value: Any, column: str, row_number: int) -> int:
    try:
        parsed = int(float(value or 0))
    except (TypeError, ValueError) as exc:
        raise ContractError(f"non-numeric value in {column} row {row_number}") from exc
    if parsed < 0:
        raise ContractError(f"negative value in {column} row {row_number}: {parsed}")
    return parsed


def derive_core_eligibility(
    labels: list[Mapping[str, str]],
    dimensions: Iterable[str] = CORE_DIMENSIONS,
) -> dict[str, Any]:
    """Derive the confirmatory family from frozen V2.1 keep flags."""
    if not labels:
        raise ContractError("cannot derive core eligibility from an empty label table")
    dimensions = tuple(dimensions)
    columns = set(labels[0])
    details: dict[str, dict[str, Any]] = {}
    eligible: list[str] = []
    structural_zero: list[str] = []
    for dimension in dimensions:
        keep_column = f"{dimension}_dimension_keep_core"
        sentence_column = f"{dimension}_sentence_count"
        outcome_column = f"{dimension}_observed_positive_core"
        missing = [column for column in (keep_column, sentence_column, outcome_column) if column not in columns]
        if missing:
            raise ContractError(f"missing frozen core semantic columns for {dimension}: {missing}")
        keep_values = {
            _as_binary(row.get(keep_column, ""), keep_column, index)
            for index, row in enumerate(labels, start=2)
        }
        if keep_values != {0} and keep_values != {1}:
            raise ContractError(f"{keep_column} is not globally consistent: {sorted(keep_values)}")
        keep_core = next(iter(keep_values))
        positive_count = 0
        for index, row in enumerate(labels, start=2):
            sentence_count = _as_nonnegative_int(row.get(sentence_column, ""), sentence_column, index)
            observed = _as_binary(row.get(outcome_column, ""), outcome_column, index)
            expected = int(sentence_count >= 1 and keep_core == 1)
            if observed != expected:
                raise ContractError(
                    f"{outcome_column} contradicts frozen V2.1 semantics at row {index}: "
                    f"expected {expected}, got {observed}"
                )
            positive_count += observed
        if keep_core == 1:
            eligible.append(dimension)
        else:
            structural_zero.append(dimension)
        details[dimension] = {
            "dimension_keep_core": keep_core,
            "eligible_for_core_confirmatory": keep_core == 1,
            "sentence_count_field": sentence_column,
            "observed_positive_core_count": positive_count,
        }
    return {
        "core_min_products": CORE_MIN_PRODUCTS,
        "keep_field_suffix": "_dimension_keep_core",
        "sentence_count_field_suffix": "_sentence_count",
        "observed_positive_field_suffix": "_observed_positive_core",
        "observed_positive_core_semantics": "(sentence_count >= 1) AND (dimension_keep_core == 1)",
        "global_keep_flag_consistency": True,
        "dimensions": details,
        "eligible_dimensions": eligible,
        "structural_zero_dimensions": structural_zero,
    }


def _split_algorithm_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, Mapping):
        return " ".join(_split_algorithm_text(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return " ".join(_split_algorithm_text(item) for item in value)
    return str(value)


def validate_split_policy(split_policy: Mapping[str, Any]) -> None:
    """Reject a non-joint, size-only, or performance-seeking split policy."""
    if split_policy.get("assigned_in_p8_a") is not False:
        raise ContractError("P8-A cannot assign a split")
    if split_policy.get("group_key") != "primary_response_sha256":
        raise ContractError("future split grouping must use primary_response_sha256")
    if split_policy.get("development_proportion") != 0.8 or split_policy.get("locked_test_proportion") != 0.2:
        raise ContractError("future split proportions must be 80/20")
    if split_policy.get("development_cv") != "5-fold group-aware":
        raise ContractError("development split must declare five group-aware folds")
    if split_policy.get("primary_stratification_outcome") != PRIMARY_STRATIFICATION_OUTCOME:
        raise ContractError("future split must stratify on the frozen primary observed-positive outcome")
    tie_break = split_policy.get("hash_tie_break")
    if not isinstance(tie_break, Mapping):
        raise ContractError("split hash ordering/tie-break must be structured")
    if tie_break.get("expression") != "SHA256(p8b_split_v1 + group key)":
        raise ContractError("split hash expression is not the frozen p8b_split_v1 + group key expression")
    if tie_break.get("purpose") != "deterministic ordering and tie-break only":
        raise ContractError("split hash must be limited to deterministic ordering and tie-breaking")
    if tie_break.get("seed_search") is not False or tie_break.get("favorable_split_retries") != 0:
        raise ContractError("split policy cannot search seeds or retry for a favorable split")
    objective = split_policy.get("stratification_objective")
    if not isinstance(objective, Mapping):
        raise ContractError("split policy must declare a structured stratification objective")
    if objective.get("outcome") != PRIMARY_STRATIFICATION_OUTCOME:
        raise ContractError("split objective does not track the frozen primary outcome")
    if objective.get("tracked_totals") != ["product_count", "observed_positive_count"]:
        raise ContractError("split objective must track product and observed-positive totals")
    if objective.get("loss_kind") != "joint_additive_normalized_absolute_deviation":
        raise ContractError("split objective must be a fixed joint balancing loss")
    if objective.get("weights") != {"product_count": 1.0, "observed_positive_count": 1.0}:
        raise ContractError("split objective must weight row and observed-positive deviations equally")
    if objective.get("lexicographic") is not False:
        raise ContractError("split objective cannot be lexicographic")
    if "abs(row deviation) / total rows + abs(observed-positive deviation) / total observed positives" not in objective.get("formula", ""):
        raise ContractError("split objective formula must be joint row plus observed-positive loss")
    locked = split_policy.get("locked_test_algorithm")
    development = split_policy.get("development_five_fold_algorithm")
    for label, algorithm, target in (
        ("locked test", locked, "locked test target"),
        ("development five-fold", development, "development totals after locked test"),
    ):
        if not isinstance(algorithm, Mapping):
            raise ContractError(f"{label} algorithm must be structured")
        if algorithm.get("group_key") != "primary_response_sha256":
            raise ContractError(f"{label} algorithm does not preserve primary response groups")
        if algorithm.get("stratification_outcome") != PRIMARY_STRATIFICATION_OUTCOME:
            raise ContractError(f"{label} algorithm ignores the primary stratification outcome")
        if algorithm.get("tracked_totals") != ["product_count", "observed_positive_count"]:
            raise ContractError(f"{label} algorithm does not track both requested totals")
        if algorithm.get("target") != target:
            raise ContractError(f"{label} algorithm target declaration is incomplete")
        if algorithm.get("loss_kind") != objective["loss_kind"]:
            raise ContractError(f"{label} algorithm does not use the frozen joint loss")
        expected_candidate_evaluation = (
            "evaluate every candidate fold by the aggregate joint loss across all five folds after assignment"
            if label == "development five-fold"
            else "evaluate every candidate whole group assignment using the joint loss"
        )
        if algorithm.get("candidate_evaluation") != expected_candidate_evaluation:
            raise ContractError(f"{label} algorithm must evaluate the declared global joint loss")
        text_value = _split_algorithm_text(algorithm)
        whole_assignment_declared = (
            "whole group" in text_value.lower()
            or "whole-fold" in text_value.lower()
            or "whole fold" in text_value.lower()
        )
        if not whole_assignment_declared or "joint" not in text_value.lower():
            raise ContractError(f"{label} algorithm must state whole-group joint assignment")
    if development.get("input_universe") != "groups assigned to development after locked-test partition only":
        raise ContractError("development folds must consume only post-locked-test development groups")
    if development.get("fold_target_proportion") != 0.2:
        raise ContractError("development fold targets must be 20 percent of development totals")
    forbidden = (
        "size-only",
        "performance-based",
        "seed search",
        "retry until favorable",
        "favorable split",
        "lexicographic",
        "product-count-first",
        "row-first",
        "positive-first",
    )
    text_value = _split_algorithm_text(split_policy).lower()
    if any(term in text_value for term in forbidden):
        raise ContractError("split policy contains a forbidden non-joint or favorable-split adaptation")


def split_hash_order_key(salt: str, group_key: str) -> str:
    """Return the only permitted deterministic ordering/tie-break key."""
    return hashlib.sha256(f"{salt}{group_key}".encode("utf-8")).hexdigest()


def joint_balance_loss(
    row_count: int | float,
    observed_positive_count: int | float,
    target_rows: int | float,
    target_observed_positives: int | float,
    total_rows: int | float,
    total_observed_positives: int | float,
) -> float:
    """Equal-weight scalar loss for joint row and observed-positive balance."""
    return (
        abs(row_count - target_rows) / max(float(total_rows), 1.0)
        + abs(observed_positive_count - target_observed_positives)
        / max(float(total_observed_positives), 1.0)
    )


def _normalise_group_summaries(groups: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    summaries: dict[str, dict[str, Any]] = {}
    for row in groups:
        key = str(row.get("group_key") or row.get("primary_response_sha256") or "").strip().upper()
        if not key:
            raise ContractError("synthetic split group is missing primary_response_sha256")
        products = _as_nonnegative_int(row.get("product_count"), "product_count", 0)
        positives = _as_nonnegative_int(row.get("observed_positive_count"), "observed_positive_count", 0)
        if key in summaries:
            raise ContractError(f"duplicate aggregated split group: {key}")
        summaries[key] = {"group_key": key, "product_count": products, "observed_positive_count": positives}
    if not summaries:
        raise ContractError("split requires at least one group")
    return list(summaries.values())


def assign_groups_to_locked_test(
    groups: Iterable[Mapping[str, Any]],
    salt: str = "p8b_split_v1",
    test_proportion: float = 0.2,
) -> dict[str, str]:
    """Synthetic/reference assignment helper; P8-A never calls it on real data."""
    summaries = _normalise_group_summaries(groups)
    total_rows = sum(row["product_count"] for row in summaries)
    total_positives = sum(row["observed_positive_count"] for row in summaries)
    target_rows = total_rows * test_proportion
    target_positives = total_positives * test_proportion
    remaining = {row["group_key"]: row for row in summaries}
    selected: set[str] = set()
    selected_rows = 0
    selected_positives = 0
    current_loss = joint_balance_loss(
        selected_rows,
        selected_positives,
        target_rows,
        target_positives,
        total_rows,
        total_positives,
    )
    while remaining:
        candidates = []
        for key, row in remaining.items():
            candidate_loss = joint_balance_loss(
                selected_rows + row["product_count"],
                selected_positives + row["observed_positive_count"],
                target_rows,
                target_positives,
                total_rows,
                total_positives,
            )
            candidates.append((candidate_loss, split_hash_order_key(salt, key), key, row))
        candidate_loss, _, key, row = min(candidates, key=lambda item: (item[0], item[1], item[2]))
        if candidate_loss >= current_loss:
            break
        selected.add(key)
        selected_rows += row["product_count"]
        selected_positives += row["observed_positive_count"]
        current_loss = candidate_loss
        del remaining[key]
    return {
        row["group_key"]: ("locked_test" if row["group_key"] in selected else "development")
        for row in summaries
    }


def assign_groups_to_development_folds(
    groups: Iterable[Mapping[str, Any]],
    locked_test_assignments: Mapping[str, str],
    salt: str = "p8b_split_v1",
    fold_count: int = 5,
) -> dict[str, int]:
    """Synthetic/reference five-fold helper; P8-A never calls it on real data."""
    if fold_count < 2:
        raise ContractError("development fold count must be at least two")
    summaries = _normalise_group_summaries(groups)
    assignments = {str(key).strip().upper(): value for key, value in locked_test_assignments.items()}
    keys = {row["group_key"] for row in summaries}
    if set(assignments) != keys:
        raise ContractError("locked-test assignment must cover every group exactly once")
    if set(assignments.values()) != {"locked_test", "development"}:
        raise ContractError("locked-test assignment must contain locked_test and development groups")
    development_groups = [row for row in summaries if assignments[row["group_key"]] == "development"]
    if not development_groups:
        raise ContractError("development folds require at least one post-locked-test development group")
    total_rows = sum(row["product_count"] for row in development_groups)
    total_positives = sum(row["observed_positive_count"] for row in development_groups)
    target_rows = total_rows / fold_count
    target_positives = total_positives / fold_count
    fold_counts = {fold: {"product_count": 0, "observed_positive_count": 0} for fold in range(fold_count)}
    result: dict[str, int] = {}
    ordered = sorted(
        development_groups,
        key=lambda row: (split_hash_order_key(salt, row["group_key"]), row["group_key"]),
    )
    for row in ordered:
        scored = []
        for fold in range(fold_count):
            tentative_counts = {
                candidate_fold: counts.copy()
                for candidate_fold, counts in fold_counts.items()
            }
            tentative_counts[fold]["product_count"] += row["product_count"]
            tentative_counts[fold]["observed_positive_count"] += row["observed_positive_count"]
            global_loss = sum(
                joint_balance_loss(
                    counts["product_count"],
                    counts["observed_positive_count"],
                    target_rows,
                    target_positives,
                    total_rows,
                    total_positives,
                )
                for counts in tentative_counts.values()
            )
            scored.append((global_loss, fold))
        _, chosen = min(scored, key=lambda item: (item[0], item[1]))
        result[row["group_key"]] = chosen
        fold_counts[chosen]["product_count"] += row["product_count"]
        fold_counts[chosen]["observed_positive_count"] += row["observed_positive_count"]
    return result


def load_contract(path: Path = CONTRACT_PATH) -> dict[str, Any]:
    contract = read_json(path)
    if contract.get("contract_version") != "p8a_v1.0":
        raise ContractError("contract_version must be p8a_v1.0")
    hierarchy = canonical_outcome_hierarchy()
    if tuple(contract.get("canonical_pu_outcomes", [])) != tuple(hierarchy["canonical_pu_outcomes"]):
        raise ContractError("canonical PU outcome whitelist is not the frozen 32-column hierarchy")
    outcome_hierarchy = contract.get("outcome_hierarchy", {})
    if outcome_hierarchy.get("primary") != hierarchy["primary"]:
        raise ContractError("primary outcome is not frozen exactly")
    secondary = tuple(outcome_hierarchy.get("secondary_confirmatory", []))
    expected_secondary_shape = tuple(f"{dimension}_observed_positive_core" for dimension in CORE_DIMENSIONS)
    if any(name not in expected_secondary_shape for name in secondary) or len(set(secondary)) != len(secondary):
        raise ContractError("secondary confirmatory outcomes must be unique frozen core columns")
    fdr = outcome_hierarchy.get("dimension_fdr", {})
    if fdr.get("method") != "Benjamini-Hochberg" or fdr.get("q") != 0.05:
        raise ContractError("dimension FDR must remain Benjamini-Hochberg at q=0.05")
    if fdr.get("family_size") != len(secondary) or fdr.get("primary_in_family") is not False:
        raise ContractError("dimension FDR family must match the derived secondary core family")
    if contract.get("g_policy", {}).get("g_main_cohort_filter") is not False:
        raise ContractError("G main cohort filter must be false")
    if contract.get("g_policy", {}).get("g_final_threshold_selected") is not False:
        raise ContractError("G final threshold selected must remain false")
    validate_split_policy(contract.get("split_policy", {}))
    if contract.get("verification_policy", {}).get("verify_existing_writes") != 0:
        raise ContractError("verify-existing write policy must be zero")
    return contract


def run_upstream_verifiers() -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    for name, command in P7_VERIFY_COMMANDS.items():
        completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
        if completed.returncode != 0:
            raise ContractError(f"{name} verify-existing failed:\n{completed.stdout}\n{completed.stderr}")
        try:
            payload = json.loads(completed.stdout)
        except json.JSONDecodeError as exc:
            raise ContractError(f"{name} verifier did not return JSON") from exc
        if payload.get("verification") != "PASS":
            raise ContractError(f"{name} verifier did not PASS")
        results[name] = payload
    return results


def _check_path_sha(path: Path, expected: str, label: str) -> None:
    if not path.exists():
        raise ContractError(f"missing {label}: {path}")
    actual = sha256_file(path)
    if actual != expected.upper():
        raise ContractError(f"{label} SHA mismatch: expected {expected}, got {actual}")


def _as_binary(value: str, column: str, row_number: int) -> int:
    try:
        parsed = int(float(value or 0))
    except (TypeError, ValueError) as exc:
        raise ContractError(f"non-binary value in {column} row {row_number}") from exc
    if parsed not in (0, 1):
        raise ContractError(f"non-binary value in {column} row {row_number}: {parsed}")
    return parsed


def _count_outcome(rows: Iterable[Mapping[str, str]], column: str) -> dict[str, int]:
    values = [_as_binary(row.get(column, ""), column, index) for index, row in enumerate(rows, start=2)]
    return {
        "observed_positive_count": sum(values),
        "pu_unlabeled_count": len(values) - sum(values),
        "row_count": len(values),
    }


def _validate_label_schema(labels: list[dict[str, str]], contract: Mapping[str, Any]) -> dict[str, Any]:
    if len(labels) != 5180 or len({row.get("parent_asin") for row in labels}) != 5180:
        raise ContractError("canonical labels must contain 5180 unique parent_asin rows")
    outcomes = canonical_pu_outcomes(contract)
    columns = set(labels[0]) if labels else set()
    missing = [column for column in outcomes if column not in columns]
    if missing:
        raise ContractError(f"missing canonical PU outcome columns: {missing}")
    eligibility_fields = sorted(column for column in columns if any(marker in column for marker in ELIGIBILITY_MARKERS))
    if set(outcomes) & set(eligibility_fields):
        raise ContractError("eligibility fields leaked into the PU outcome whitelist")
    for column in outcomes:
        _count_outcome(labels, column)
    core_eligibility = derive_core_eligibility(labels)
    declared_core = contract.get("core_eligibility", {})
    if declared_core.get("core_min_products") != core_eligibility["core_min_products"]:
        raise ContractError("core_min_products does not match frozen V2.1 semantics")
    if declared_core.get("eligible_dimensions") != core_eligibility["eligible_dimensions"]:
        raise ContractError("declared core-eligible dimensions do not match frozen keep flags")
    if declared_core.get("structural_zero_dimensions") != core_eligibility["structural_zero_dimensions"]:
        raise ContractError("declared structural-zero dimensions do not match frozen keep flags")
    expected_secondary = [
        f"{dimension}_observed_positive_core"
        for dimension in core_eligibility["eligible_dimensions"]
    ]
    if contract["outcome_hierarchy"].get("secondary_confirmatory") != expected_secondary:
        raise ContractError("secondary confirmatory outcomes do not follow frozen keep_core eligibility")
    if contract["outcome_hierarchy"].get("dimension_fdr", {}).get("family_size") != len(expected_secondary):
        raise ContractError("BH family size does not match frozen keep_core eligibility")
    return {
        "rows": len(labels),
        "unique_parent_asin": len({row["parent_asin"] for row in labels}),
        "outcome_columns": list(outcomes),
        "eligibility_fields_not_outcomes": eligibility_fields,
        "core_eligibility": core_eligibility,
        "label_semantics": contract["label_semantics"],
    }


def _main_manifest_audit(manifest: list[dict[str, str]], contract: Mapping[str, Any]) -> dict[str, Any]:
    if len(manifest) != 5180 or len({row.get("parent_asin") for row in manifest}) != 5180:
        raise ContractError("P7-D final manifest must contain 5180 unique parent_asin rows")
    statuses = {row.get("primary_freeze_status") for row in manifest}
    if not statuses <= {"frozen_primary", "frozen_primary_with_qa_exception", "excluded_non_primary"}:
        raise ContractError(f"unexpected primary freeze status: {statuses}")
    main_rows = [row for row in manifest if main_population_status(row.get("primary_freeze_status", ""))]
    excluded_rows = [row for row in manifest if row.get("primary_freeze_status") == "excluded_non_primary"]
    if len(main_rows) != 5179 or len(excluded_rows) != 1:
        raise ContractError("P7-D main/excluded population accounting mismatch")
    if any(row.get("primary_asset_substituted", "").lower() == "true" for row in manifest):
        raise ContractError("P7-D primary substitution invariant failed")
    if any(row.get("sensitivity_status") == "available" and not row.get("primary_response_sha256") for row in main_rows):
        raise ContractError("sensitivity row is missing its frozen primary exposure")
    groups: dict[str, int] = {}
    for row in main_rows:
        key = duplicate_group_key(row)
        groups[key] = groups.get(key, 0) + 1
    return {
        "manifest_rows": len(manifest),
        "main_modeling_population": len(main_rows),
        "excluded_non_primary": len(excluded_rows),
        "status_counts": {status: sum(row.get("primary_freeze_status") == status for row in manifest) for status in sorted(statuses)},
        "known_qa_exception_count": sum(row.get("primary_freeze_status") == "frozen_primary_with_qa_exception" for row in manifest),
        "sensitivity_available": sum(row.get("sensitivity_status") == "available" for row in manifest),
        "sensitivity_unresolved": sum(row.get("sensitivity_status") == "unresolved" for row in manifest),
        "sensitivity_not_required": sum(row.get("sensitivity_status") == "not_required" for row in manifest),
        "primary_substitutions": sum(row.get("primary_asset_substituted", "").lower() == "true" for row in manifest),
        "duplicate_group_key": "primary_response_sha256",
        "duplicate_group_count": len(groups),
        "duplicate_group_count_gt_one": sum(size > 1 for size in groups.values()),
        "duplicate_rows_in_groups_gt_one": sum(size for size in groups.values() if size > 1),
        "maximum_duplicate_group_size": max(groups.values()),
    }


def _g_audit(contract: Mapping[str, Any]) -> dict[str, Any]:
    source_results: dict[str, str] = {}
    for name, details in contract["g_source_artifacts"].items():
        path = repo_path(details["path"])
        _check_path_sha(path, details["sha256"], f"G artifact {name}")
        source_results[name] = sha256_file(path)
    method = read_json(G_METHOD_PATH)
    if method.get("final_threshold_selected") is not False:
        raise ContractError("frozen G method summary selected a final threshold")
    if method.get("expected_cohort_size") != 5180 or method.get("joined_cohort_size") != 5180:
        raise ContractError("G method summary cohort size mismatch")
    if method.get("run_parameters", {}).get("thresholds") != contract["g_policy"]["existing_threshold_grid"]:
        raise ContractError("G threshold grid does not match the pre-existing frozen grid")
    if method.get("run_parameters", {}).get("strata") != contract["g_policy"]["existing_strata_grid"]:
        raise ContractError("G strata grid does not match the pre-existing frozen grid")
    return {
        "source_artifacts": source_results,
        "method_summary_final_threshold_selected": method["final_threshold_selected"],
        "existing_threshold_grid": method["run_parameters"]["thresholds"],
        "existing_strata_grid": method["run_parameters"]["strata"],
        "g_role": contract["g_policy"],
        "model_performance_inputs_used": False,
    }


def collect_audits(contract: Mapping[str, Any], upstream: Mapping[str, Any] | None = None) -> dict[str, dict[str, Any]]:
    for name, expected in contract["p7_d_formal_sha256"].items():
        _check_path_sha(repo_path(contract["p7_d_formal_paths"][name]), expected, f"P7-D {name}")
    _check_path_sha(LABEL_PATH, contract["canonical_label_source"]["sha256"], "canonical label source")
    manifest = read_csv(P7D_MANIFEST_PATH)
    labels = read_csv(LABEL_PATH)
    manifest_audit = _main_manifest_audit(manifest, contract)
    label_schema = _validate_label_schema(labels, contract)
    label_by_asin = {row["parent_asin"]: row for row in labels}
    main_asins = {row["parent_asin"] for row in manifest if main_population_status(row.get("primary_freeze_status", ""))}
    if not main_asins <= set(label_by_asin):
        raise ContractError("main image population cannot be joined to canonical labels")
    main_labels = [label_by_asin[asin] for asin in sorted(main_asins)]
    primary_count = _count_outcome(labels, "has_any_outer_imagery_observed")
    main_primary_count = _count_outcome(main_labels, "has_any_outer_imagery_observed")
    if primary_count["observed_positive_count"] != contract["canonical_label_source"]["outer_main_observed_positive_count"]:
        raise ContractError("full-universe outer-main observed-positive count is not 232")
    core_dimensions = label_schema["core_eligibility"]["eligible_dimensions"]
    core_counts = {
        f"{dimension}_observed_positive_core": _count_outcome(
            main_labels, f"{dimension}_observed_positive_core"
        )
        for dimension in core_dimensions
    }
    pilot_columns = [f"{dimension}_observed_positive_pilot" for dimension in CORE_DIMENSIONS]
    robust_columns = [f"{dimension}_observed_positive_robust" for dimension in CORE_DIMENSIONS]
    robustness_counts = {column: _count_outcome(main_labels, column) for column in ["has_any_all_level_imagery_evidence", *pilot_columns, *robust_columns]}
    g_audit = _g_audit(contract)
    upstream_audit = {
        "verification": "PASS",
        "upstream_main_commit": contract["upstream_main_commit"],
        "p7_verify": dict(upstream or {name: {"verification": "PASS"} for name in P7_VERIFY_COMMANDS}),
        "p7_d_formal_sha256": dict(contract["p7_d_formal_sha256"]),
        "p7_d_final_manifest_sha256": sha256_file(P7D_MANIFEST_PATH),
        "product_universe": manifest_audit,
        "label_source_path": contract["canonical_label_source"]["path"],
        "label_source_sha256": sha256_file(LABEL_PATH),
    }
    population_audit = {
        "label_universe": label_schema,
        "manifest_population": manifest_audit,
        "main_primary_outcome": main_primary_count,
        "full_universe_primary_outcome": primary_count,
        "core_eligibility": label_schema["core_eligibility"],
        "secondary_confirmatory_outcomes": core_counts,
        "structural_zero_core_outcomes": [
            f"{dimension}_observed_positive_core"
            for dimension in label_schema["core_eligibility"]["structural_zero_dimensions"]
        ],
        "robustness_outcomes": robustness_counts,
        "outcome_hierarchy": contract["outcome_hierarchy"],
        "canonical_pu_outcome_count": len(canonical_pu_outcomes(contract)),
        "pu_zero_semantics": contract["label_semantics"]["zero"],
        "main_analysis_includes_qa_exceptions": True,
        "excluded_non_primary_can_reenter_via_fallback": False,
    }
    g_sensitivity_audit = {
        "g_audit": g_audit,
        "sensitivity_analyses": contract["sensitivity_analyses"],
        "image_exposure_policy": contract["image_exposure_policy"],
        "predictor_boundaries": contract["predictor_boundaries"],
        "duplicate_group_feasibility": {
            "group_key": manifest_audit["duplicate_group_key"],
            "duplicate_groups_gt_one": manifest_audit["duplicate_group_count_gt_one"],
            "duplicate_rows_in_groups_gt_one": manifest_audit["duplicate_rows_in_groups_gt_one"],
            "maximum_group_size": manifest_audit["maximum_duplicate_group_size"],
        },
        "split_policy": contract["split_policy"],
        "evaluation_policy": contract["evaluation_policy"],
        "feature_extraction_started": False,
        "model_fitting_started": False,
        "split_assigned": False,
    }
    snapshot = json.loads(json.dumps(contract, sort_keys=True))
    return {
        "01_upstream_freeze_audit.json": upstream_audit,
        "02_modeling_population_and_outcome_audit.json": population_audit,
        "03_g_role_and_sensitivity_audit.json": g_sensitivity_audit,
        "04_analysis_contract_snapshot.json": snapshot,
    }


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="")


def build_provenance(contract: Mapping[str, Any], formal_run_git_commit: str) -> dict[str, Any]:
    return {
        "formal_run_git_commit": formal_run_git_commit,
        "producer": {
            "path": "scripts/modeling/p8_a_analysis_contract.py",
            "git_blob_sha256": git_blob_sha256(formal_run_git_commit, "scripts/modeling/p8_a_analysis_contract.py"),
        },
        "contract": {
            "path": "config/modeling/p8_a_analysis_contract.json",
            "git_blob_sha256": git_blob_sha256(formal_run_git_commit, "config/modeling/p8_a_analysis_contract.json"),
        },
        "authoritative_upstream": {
            "upstream_main_commit": contract["upstream_main_commit"],
            "p7_d_formal_sha256": contract["p7_d_formal_sha256"],
            "p7_d_final_manifest_sha256": contract["p7_d_formal_sha256"]["04_final_image_manifest.csv"],
            "label_source": contract["canonical_label_source"],
            "g_source_artifacts": contract["g_source_artifacts"],
        },
        "formal_output_sha256": {
            name: sha256_file(FORMAL_DIR / name)
            for name in FORMAL_FILES[:4]
        },
        "provenance_self_sha": None,
    }


def prepare(contract: Mapping[str, Any]) -> dict[str, Any]:
    audits = collect_audits(contract, upstream=run_upstream_verifiers())
    for name, payload in audits.items():
        _write_json(FORMAL_DIR / name, payload)
    provenance = build_provenance(contract, git_output("rev-parse", "HEAD"))
    _write_json(FORMAL_DIR / "05_p8_a_provenance.json", provenance)
    return {"verification": "PASS", "formal_dir": str(FORMAL_DIR.relative_to(ROOT)).replace("\\", "/"), "formal_files": list(FORMAL_FILES)}


def _verify_provenance(contract: Mapping[str, Any], provenance: Mapping[str, Any]) -> None:
    if provenance.get("provenance_self_sha") is not None:
        raise ContractError("P8-A provenance must not contain its own SHA")
    for field in ("formal_output_sha256", "output_sha256"):
        if PROVENANCE_FILENAME in provenance.get(field, {}):
            raise ContractError("P8-A provenance records its own output SHA")
    if set(provenance.get("formal_output_sha256", {})) != set(FORMAL_FILES[:4]):
        raise ContractError("P8-A provenance must bind exactly formal outputs 01-04")
    formal_commit = str(provenance.get("formal_run_git_commit") or "")
    if not formal_commit:
        raise ContractError("formal_run_git_commit missing")
    producer = provenance.get("producer", {})
    contract_trace = provenance.get("contract", {})
    if git_blob_sha256(formal_commit, producer["path"]) != producer["git_blob_sha256"]:
        raise ContractError("producer Git blob provenance mismatch")
    if git_blob_sha256(formal_commit, contract_trace["path"]) != contract_trace["git_blob_sha256"]:
        raise ContractError("contract Git blob provenance mismatch")
    if git_blob_sha256(git_output("rev-parse", "HEAD"), producer["path"]) != producer["git_blob_sha256"]:
        raise ContractError("current producer Git blob differs from formal provenance")
    if git_blob_sha256(git_output("rev-parse", "HEAD"), contract_trace["path"]) != contract_trace["git_blob_sha256"]:
        raise ContractError("current contract Git blob differs from formal provenance")
    for name in FORMAL_FILES[:4]:
        if provenance.get("formal_output_sha256", {}).get(name) != sha256_file(FORMAL_DIR / name):
            raise ContractError(f"formal output SHA mismatch: {name}")


def verify_existing(contract: Mapping[str, Any]) -> dict[str, Any]:
    """Verify frozen upstream and formal P8-A outputs without writing any bytes."""
    if any(not (FORMAL_DIR / name).exists() for name in FORMAL_FILES):
        raise ContractError("P8-A formal outputs are incomplete")
    audits = collect_audits(contract, upstream=run_upstream_verifiers())
    for name, expected in audits.items():
        actual = read_json(FORMAL_DIR / name)
        if actual != expected:
            raise ContractError(f"formal audit reconstruction mismatch: {name}")
    provenance = read_json(FORMAL_DIR / "05_p8_a_provenance.json")
    _verify_provenance(contract, provenance)
    for path in FORMAL_DIR.rglob("*"):
        if path.is_file() and (path.name.endswith(".tmp") or path.name.startswith(".")):
            raise ContractError(f"temporary formal artifact found: {path}")
        if path.is_file() and any(term in path.name.lower() for term in FORBIDDEN_FORMAL_TERMS):
            raise ContractError(f"forbidden formal artifact found: {path}")
    tracked = git_output("ls-files", "--", "data/processed/modeling_readiness_p8_5180")
    if tracked:
        raise ContractError("formal data must remain gitignored")
    return {
        "verification": "PASS",
        "p7_verify": {name: "PASS" for name in P7_VERIFY_COMMANDS},
        "formal_outputs_reconstruct": "PASS",
        "provenance_git_blob_binding": "PASS",
        "zero_network_calls": True,
        "zero_model_calls": True,
        "zero_writes": True,
        "feature_extraction_started": False,
        "model_fitting_started": False,
        "split_assigned": False,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare",), nargs="?")
    parser.add_argument("--verify-existing", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    contract = load_contract()
    if args.verify_existing:
        print(json.dumps(verify_existing(contract), ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    if args.command == "prepare":
        print(json.dumps(prepare(contract), ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    raise ContractError("choose prepare or --verify-existing")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ContractError as exc:
        print(f"P8-A ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
