#!/usr/bin/env python3
"""Materialize and verify the P8-B modeling-ready manifest and frozen split.

This module consumes only the frozen P7-D manifest, the frozen V2.1 label
table, and the frozen G diagnostics. It does not extract features, call a
model, score a model, or make network requests.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.modeling import p8_a_analysis_contract as p8a


ROOT = Path(__file__).resolve().parents[2]
CONTRACT_PATH = ROOT / "config" / "modeling" / "p8_b_modeling_ready_contract.json"
SCHEMA_PATH = ROOT / "config" / "modeling" / "p8_b_modeling_manifest_schema.json"
FORMAL_DIR = ROOT / "data" / "processed" / "modeling_readiness_p8_5180" / "p8_b_modeling_ready_split"
FORMAL_FILES = (
    "01_modeling_ready_manifest.csv",
    "02_split_group_inventory.csv",
    "03_split_assignment.csv",
    "04_split_quality_audit.json",
    "05_modeling_ready_summary.json",
    "06_p8_b_provenance.json",
)
PROVENANCE_FILENAME = "06_p8_b_provenance.json"
SPLIT_ALGORITHM_VERSION = "p8b_split_v1"
GROUP_KEY = "primary_response_sha256"
PRIMARY_OUTCOME = "has_any_outer_imagery_observed"
MAIN_STATUSES = frozenset({"frozen_primary", "frozen_primary_with_qa_exception"})
EXCLUDED_STATUS = "excluded_non_primary"
HASH_PATTERN = re.compile(r"^[0-9A-F]{64}$")
ASSET_ROOT_REPO_RELATIVE = "data/images/retail_outer_package_p7_5180"
BASELINE_MAIN_COMMIT = "84f7518eebe8e1b77bb210b5cea384805a33c35b"
G_COUNT_COLUMNS = (
    "clean_review_count",
    "clean_unique_reviewer_count",
    "clean_sentence_count",
    "packaging_candidate_review_count",
    "packaging_candidate_sentence_count",
    "packaging_candidate_reviewer_count",
    "visual_strict_review_count",
    "visual_strict_sentence_count",
    "visual_strict_reviewer_count",
)
G_THRESHOLDS = (0, 1, 3, 5, 10, 20, 50, 100)
G_STRATA = ("0", "1-2", "3-4", "5-9", "10-19", "20-49", "50-99", "100+")
MANIFEST_BASE_COLUMNS = (
    "parent_asin",
    "input_order",
    "primary_freeze_status",
    "main_analysis_included",
    "main_analysis_exclusion_reason",
    "split_eligible",
    "split_partition",
    "development_fold",
    "split_group_key",
    "split_algorithm_version",
    "split_salt",
    "split_hash_order_key",
    "final_image_stage_status",
    "primary_response_sha256",
    "primary_asset_path",
    "primary_local_path",
    "primary_source_url",
    "primary_decoded_format",
    "primary_width",
    "primary_height",
    "primary_n_frames",
    "primary_response_byte_count",
    "primary_qa_sampled",
    "primary_qa_identity_status",
    "primary_qa_outer_package_status",
    "primary_qa_exception_flag",
    "known_p7c_qa_exception",
    "primary_asset_substituted",
    "sensitivity_queue",
    "sensitivity_status",
    "sensitivity_source_tier",
    "sensitivity_temporal_alignment_status",
    "sensitivity_response_sha256",
    "sensitivity_asset_path",
    "sensitivity_local_path",
    "sensitivity_available_for_sensitivity_only",
    "g_clean_review_count",
    "g_clean_unique_reviewer_count",
    "g_clean_sentence_count",
    "g_packaging_candidate_review_count",
    "g_packaging_candidate_sentence_count",
    "g_packaging_candidate_reviewer_count",
    "g_visual_strict_review_count",
    "g_visual_strict_sentence_count",
    "g_visual_strict_reviewer_count",
    "g_existing_stratum",
    "g_clean_review_count_threshold_0",
    "g_clean_review_count_threshold_1",
    "g_clean_review_count_threshold_3",
    "g_clean_review_count_threshold_5",
    "g_clean_review_count_threshold_10",
    "g_clean_review_count_threshold_20",
    "g_clean_review_count_threshold_50",
    "g_clean_review_count_threshold_100",
    "pu_zero_semantics",
)
GROUP_COLUMNS = (
    "split_group_key",
    "product_count",
    "observed_positive_count",
    "pu_unlabeled_count",
    "qa_exception_count",
    "sensitivity_available_count",
    "member_parent_asin_count",
    "split_hash_order_key",
    "split_partition",
    "development_fold",
)
ASSIGNMENT_COLUMNS = (
    "parent_asin",
    "input_order",
    "split_eligible",
    "split_group_key",
    "primary_outcome",
    "split_algorithm_version",
    "split_salt",
    "product_count",
    "observed_positive_count",
    "split_partition",
    "development_fold",
    "split_hash_order_key",
)


class ContractError(RuntimeError):
    """Raised when a frozen-input or P8-B invariant fails."""


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest().upper()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ContractError(f"expected JSON object: {path}")
    return value


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _expect_equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise ContractError(f"{label} does not match the frozen P8-B contract")


def validate_contract(contract: Mapping[str, Any]) -> None:
    _expect_equal(contract.get("contract_version"), "p8b_v1.0", "contract_version")
    _expect_equal(contract.get("baseline_main_commit"), BASELINE_MAIN_COMMIT, "baseline_main_commit")
    _expect_equal(contract.get("upstream_contract_path"), "config/modeling/p8_a_analysis_contract.json", "upstream_contract_path")
    _expect_equal(contract.get("schema_path"), "config/modeling/p8_b_modeling_manifest_schema.json", "schema_path")
    _expect_equal(contract.get("asset_root_repo_relative"), ASSET_ROOT_REPO_RELATIVE, "asset_root_repo_relative")
    _expect_equal(contract.get("formal_output_files"), list(FORMAL_FILES), "formal_output_files")
    sources = contract.get("source_artifacts", {})
    expected_sources = {
        "p7_d_final_manifest": "data/processed/retail_outer_package_images_p7_5180/p7_d_final_image_freeze/04_final_image_manifest.csv",
        "canonical_label_source": "data/processed/affective_imagery_labels_v21_5180/39_product_imagery_labels_v21.csv",
        "g_product_exposure_table": "data/processed/review_exposure_pu_diagnostics_v21_5180/01_product_exposure_label_table.csv",
        "p7_b_primary_asset_manifest": "data/processed/retail_outer_package_images_p7_5180/p7_b_primary_assets/01_primary_asset_manifest.csv",
        "p7_d_provenance": "data/processed/retail_outer_package_images_p7_5180/p7_d_final_image_freeze/07_p7_d_provenance.json",
        "p7_b_provenance": "data/processed/retail_outer_package_images_p7_5180/p7_b_primary_assets/04_p7_b_provenance.json",
    }
    for key, expected in expected_sources.items():
        _expect_equal(sources.get(key), expected, f"source_artifacts.{key}")
    _expect_equal(contract.get("source_artifact_sha256"), {
        "p7_b_primary_asset_manifest": "B855BBFED7D02945129C2BB90656F738FC69BB914E1C0F9CD064BD388BFA1601",
        "p7_b_provenance": "76165F0C98990EA3FAF628E3309C73A3F5CDA9B1DC4E92217BEF773F193B434A",
        "p7_d_provenance": "7A29A4F1B649D23BA2837126E8DD48FA39B5851A79BE301CA8F1B5B51853BC76",
    }, "source_artifact_sha256")
    _expect_equal(contract.get("p8_a_formal_output_sha256"), {
        "01_upstream_freeze_audit.json": "E24C6C9A8FF60B4679E20C714FC8EDFF98ABD4CB0C480D37D82241DE67A884D9",
        "02_modeling_population_and_outcome_audit.json": "CE96912D0A0CBF385A0F9EACBCCDE719048B0FEBEDD7DCD7B76C844A73DFD9A6",
        "03_g_role_and_sensitivity_audit.json": "7EAFE463FBCFD2CDDD73B8574EF18D495DB4B11EA85C8FCFE436806E1F83EC90",
        "04_analysis_contract_snapshot.json": "E5640861A8844198F1396F5A1DCDF0E04C63D08BA0E0A158EBDFD3C81F5097E3",
        "05_p8_a_provenance.json": "D45BA91F4D705232E7ED9326A978E60D64F4D8F236B9C6E1D05EF2129A77ED51",
    }, "p8_a_formal_output_sha256")
    split = contract.get("split", {})
    for key, expected in {
        "algorithm_version": SPLIT_ALGORITHM_VERSION,
        "group_key": GROUP_KEY,
        "primary_stratification_outcome": PRIMARY_OUTCOME,
        "locked_test_proportion": 0.2,
        "development_proportion": 0.8,
        "development_fold_count": 5,
        "development_fold_target_proportion": 0.2,
        "loss": "joint_additive_normalized_absolute_deviation",
        "weights": {"product_count": 1.0, "observed_positive_count": 1.0},
        "hash_expression": "SHA256(p8b_split_v1 + group key)",
        "hash_purpose": "deterministic ordering and tie-break only",
    }.items():
        _expect_equal(split.get(key), expected, f"split.{key}")
    for key in ("no_seed_search", "no_manual_assignment", "no_performance_tuning"):
        _expect_equal(split.get(key), True, f"split.{key}")
    for key, expected in {
        "full_universe_rows": 5180, "expected_main_rows": 5179,
        "excluded_rows": 1, "known_qa_exception_count": 83,
    }.items():
        _expect_equal(contract.get("population", {}).get(key), expected, f"population.{key}")
    for key, expected in {
        "verify_existing_writes": 0, "git_tracked_formal_data": False,
        "feature_extraction_started": False, "model_fitting_started": False,
        "predictions_or_performance_metrics_started": False, "p9_started": False,
    }.items():
        _expect_equal(contract.get("output_policy", {}).get(key), expected, f"output_policy.{key}")
    _expect_equal(contract.get("provenance_policy", {}).get("provenance_filename"), PROVENANCE_FILENAME, "provenance filename")
    _expect_equal(contract.get("provenance_policy", {}).get("provenance_self_sha_forbidden"), True, "provenance self SHA policy")
    _expect_equal(contract.get("provenance_policy", {}).get("binds_formal_outputs"), list(FORMAL_FILES[:5]), "provenance output binding")


def load_contract() -> dict[str, Any]:
    contract = read_json(CONTRACT_PATH)
    validate_contract(contract)
    return contract


def normalize_repo_path(path_value: Any) -> str:
    text = str(path_value or "").strip().replace("\\", "/")
    path = Path(text)
    if not text or path.is_absolute() or ".." in path.parts:
        raise ContractError(f"path is not a portable repo-relative path: {path_value}")
    return path.as_posix()


def _repo_relative_asset_path(asset_root_repo_relative: str, asset_path: Any) -> str:
    if not asset_path:
        return ""
    return f"{normalize_repo_path(asset_root_repo_relative)}/{normalize_repo_path(asset_path)}"


def _as_bool(value: Any) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "y"}


def _as_binary(value: Any, column: str) -> int:
    try:
        parsed = int(float(value or 0))
    except (TypeError, ValueError) as exc:
        raise ContractError(f"non-binary value in {column}") from exc
    if parsed not in (0, 1):
        raise ContractError(f"non-binary value in {column}: {value}")
    return parsed


def _as_nonnegative_int(value: Any, column: str) -> int:
    try:
        parsed = int(float(value or 0))
    except (TypeError, ValueError) as exc:
        raise ContractError(f"non-numeric value in {column}") from exc
    if parsed < 0:
        raise ContractError(f"negative value in {column}")
    return parsed


def _bool_text(value: bool) -> str:
    return "true" if value else "false"


def _unique_by_parent(rows: Iterable[Mapping[str, Any]], source_name: str) -> dict[str, Mapping[str, Any]]:
    materialized = list(rows)
    keys = [str(row.get("parent_asin") or "").strip() for row in materialized]
    if any(not key for key in keys):
        raise ContractError(f"{source_name} contains a missing parent_asin")
    if len(set(keys)) != len(keys):
        raise ContractError(f"{source_name} must contain unique parent_asin rows")
    return dict(zip(keys, materialized))


def _check_join_sets(sources: Mapping[str, Mapping[str, Mapping[str, Any]]]) -> None:
    names = list(sources)
    reference = set(sources[names[0]])
    for name in names[1:]:
        current = set(sources[name])
        if current != reference:
            raise ContractError(f"{name} parent_asin join is not bijective with {names[0]}")


def _stratum(value: int) -> str:
    if value == 0:
        return G_STRATA[0]
    if value <= 2:
        return G_STRATA[1]
    if value <= 4:
        return G_STRATA[2]
    if value <= 9:
        return G_STRATA[3]
    if value <= 19:
        return G_STRATA[4]
    if value <= 49:
        return G_STRATA[5]
    if value <= 99:
        return G_STRATA[6]
    return G_STRATA[7]


def _validate_primary_key(value: str, asin: str) -> str:
    key = str(value or "").strip().upper()
    if not HASH_PATTERN.fullmatch(key):
        raise ContractError(f"main population row {asin} has invalid primary_response_sha256")
    return key


def build_manifest_rows(
    p7_rows: Iterable[Mapping[str, Any]],
    label_rows: Iterable[Mapping[str, Any]],
    g_rows: Iterable[Mapping[str, Any]],
    p8a_contract: Mapping[str, Any] | None = None,
    p8b_contract: Mapping[str, Any] | None = None,
    *,
    p7b_rows: Iterable[Mapping[str, Any]] | None = None,
) -> list[dict[str, str]]:
    """Join frozen sources and create split-unassigned modeling rows."""
    p8a_contract = p8a_contract or p8a.load_contract()
    p8b_contract = p8b_contract or load_contract()
    sources = {
        "P7-D manifest": _unique_by_parent(p7_rows, "P7-D manifest"),
        "canonical labels": _unique_by_parent(label_rows, "canonical labels"),
        "G product table": _unique_by_parent(g_rows, "G product table"),
    }
    p7b_by_parent = _unique_by_parent(p7b_rows, "P7-B primary asset manifest") if p7b_rows is not None else {}
    _check_join_sets(sources)
    labels = sources["canonical labels"]
    outcomes = p8a.canonical_pu_outcomes(p8a_contract)
    missing_outcomes = [column for column in outcomes if column not in next(iter(labels.values()))]
    if missing_outcomes:
        raise ContractError(f"canonical labels missing PU outcome columns: {missing_outcomes}")
    for asin, label in labels.items():
        label_primary = _as_binary(label.get(PRIMARY_OUTCOME), PRIMARY_OUTCOME)
        g_primary = _as_binary(sources["G product table"][asin].get(PRIMARY_OUTCOME), PRIMARY_OUTCOME)
        if label_primary != g_primary:
            raise ContractError(f"primary outcome mismatch for {asin}")

    rows: list[dict[str, str]] = []
    for asin in sorted(sources["P7-D manifest"], key=lambda key: _as_nonnegative_int(sources["P7-D manifest"][key].get("input_order"), "input_order")):
        p7 = sources["P7-D manifest"][asin]
        label = sources["canonical labels"][asin]
        g = sources["G product table"][asin]
        status = str(p7.get("primary_freeze_status") or "")
        included = status in MAIN_STATUSES
        excluded = status == EXCLUDED_STATUS
        if not included and not excluded:
            raise ContractError(f"unexpected primary_freeze_status for {asin}: {status}")
        p7b = p7b_by_parent.get(asin)
        if included and p7b_rows is not None:
            if p7b is None:
                raise ContractError(f"P7-B primary asset manifest is missing included row {asin}")
            if str(p7b.get("response_sha256") or "").upper() != str(p7.get("primary_response_sha256") or "").upper():
                raise ContractError(f"P7-B/P7-D response SHA mismatch for {asin}")
            if str(p7b.get("asset_path") or "") != str(p7.get("primary_asset_path") or ""):
                raise ContractError(f"P7-B/P7-D asset path mismatch for {asin}")
        group_key = _validate_primary_key(p7.get(GROUP_KEY), asin) if included else ""
        review_count = _as_nonnegative_int(g.get("clean_review_count"), "clean_review_count")
        asset_root_repo_relative = str(p8b_contract.get("asset_root_repo_relative") or ASSET_ROOT_REPO_RELATIVE)
        requested_url = str((p7b or {}).get("requested_url") or "") if p7b_rows is not None else str(p7.get("requested_url") or "")
        row = {
            "parent_asin": asin,
            "input_order": str(p7.get("input_order") or ""),
            "primary_freeze_status": status,
            "main_analysis_included": _bool_text(included),
            "main_analysis_exclusion_reason": "" if included else EXCLUDED_STATUS,
            "split_eligible": _bool_text(included),
            "split_partition": "" if included else "excluded",
            "development_fold": "",
            "split_group_key": group_key,
            "split_algorithm_version": str(p8b_contract["split"]["algorithm_version"]),
            "split_salt": SPLIT_ALGORITHM_VERSION,
            "split_hash_order_key": p8a.split_hash_order_key(SPLIT_ALGORITHM_VERSION, group_key) if included else "",
            "final_image_stage_status": str(p7.get("final_image_stage_status") or ""),
            "primary_response_sha256": group_key,
            "primary_asset_path": str(p7.get("primary_asset_path") or ""),
            "primary_local_path": _repo_relative_asset_path(asset_root_repo_relative, p7.get("primary_asset_path")),
            "primary_source_url": requested_url,
            "primary_decoded_format": str(p7.get("primary_decoded_format") or ""),
            "primary_width": str(p7.get("primary_width") or ""),
            "primary_height": str(p7.get("primary_height") or ""),
            "primary_n_frames": str(p7.get("primary_n_frames") or ""),
            "primary_response_byte_count": str(p7.get("primary_response_byte_count") or ""),
            "primary_qa_sampled": str(p7.get("primary_qa_sampled") or ""),
            "primary_qa_identity_status": str(p7.get("primary_qa_identity_status") or ""),
            "primary_qa_outer_package_status": str(p7.get("primary_qa_outer_package_status") or ""),
            "primary_qa_exception_flag": str(p7.get("primary_qa_exception_flag") or ""),
            "known_p7c_qa_exception": _bool_text(_as_bool(p7.get("c_reviewed_qa_exception"))),
            "primary_asset_substituted": str(p7.get("primary_asset_substituted") or ""),
            "sensitivity_queue": str(p7.get("sensitivity_queue") or ""),
            "sensitivity_status": str(p7.get("sensitivity_status") or ""),
            "sensitivity_source_tier": str(p7.get("sensitivity_source_tier") or ""),
            "sensitivity_temporal_alignment_status": str(p7.get("sensitivity_temporal_alignment_status") or ""),
            "sensitivity_response_sha256": str(p7.get("sensitivity_response_sha256") or ""),
            "sensitivity_asset_path": str(p7.get("sensitivity_asset_path") or ""),
            "sensitivity_local_path": _repo_relative_asset_path(asset_root_repo_relative, p7.get("sensitivity_asset_path")),
            "sensitivity_available_for_sensitivity_only": _bool_text(p7.get("sensitivity_status") == "available"),
            "g_existing_stratum": _stratum(review_count),
            "pu_zero_semantics": "unlabeled",
        }
        for column in G_COUNT_COLUMNS:
            row[f"g_{column}"] = str(g.get(column) or "")
        for threshold in G_THRESHOLDS:
            row[f"g_clean_review_count_threshold_{threshold}"] = _bool_text(review_count >= threshold)
        for outcome in outcomes:
            row[outcome] = str(label.get(outcome) or "")
        rows.append(row)
    return rows


def manifest_columns(p8a_contract: Mapping[str, Any] | None = None) -> tuple[str, ...]:
    contract = p8a_contract or p8a.load_contract()
    return MANIFEST_BASE_COLUMNS + tuple(p8a.canonical_pu_outcomes(contract))


def validate_manifest_schema(schema: Mapping[str, Any], p8a_contract: Mapping[str, Any] | None = None) -> None:
    if schema.get("schema_version") != "p8b_modeling_ready_manifest_v1":
        raise ContractError("unexpected P8-B manifest schema version")
    columns = schema.get("columns")
    if not isinstance(columns, list) or any(not isinstance(item, Mapping) for item in columns):
        raise ContractError("P8-B manifest schema columns must be objects")
    names = [str(item.get("name") or "") for item in columns]
    expected = list(manifest_columns(p8a_contract))
    if names != expected or len(names) != len(set(names)):
        raise ContractError("P8-B manifest schema must exactly match the generated manifest header")
    expected_roles = {
        "final_image_stage_status": "frozen_exposure_stage_status",
        "primary_asset_path": "frozen_exposure_asset_root_relative",
        "primary_local_path": "frozen_exposure_repo_relative",
        "primary_source_url": "frozen_exposure_requested_url",
        "sensitivity_asset_path": "sensitivity_asset_root_relative",
        "sensitivity_local_path": "sensitivity_repo_relative",
    }
    roles = {str(item["name"]): item.get("role") for item in columns}
    for name, role in expected_roles.items():
        if roles.get(name) != role:
            raise ContractError(f"manifest schema role mismatch for {name}")


def load_manifest_schema(p8a_contract: Mapping[str, Any] | None = None) -> dict[str, Any]:
    schema = read_json(SCHEMA_PATH)
    validate_manifest_schema(schema, p8a_contract)
    return schema


def build_group_inventory(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    for row in rows:
        if row.get("split_eligible") != "true":
            continue
        key = str(row.get("split_group_key") or "").strip().upper()
        if not HASH_PATTERN.fullmatch(key):
            raise ContractError("split-eligible row has invalid group inventory key")
        if key not in grouped:
            grouped[key] = {
                "split_group_key": key,
                "product_count": 0,
                "observed_positive_count": 0,
                "pu_unlabeled_count": 0,
                "qa_exception_count": 0,
                "sensitivity_available_count": 0,
                "member_parent_asin_count": 0,
                "split_hash_order_key": p8a.split_hash_order_key(SPLIT_ALGORITHM_VERSION, key),
            }
        entry = grouped[key]
        entry["product_count"] += 1
        positive = _as_binary(row.get(PRIMARY_OUTCOME), PRIMARY_OUTCOME)
        entry["observed_positive_count"] += positive
        entry["pu_unlabeled_count"] += 1 - positive
        entry["qa_exception_count"] += int(row.get("known_p7c_qa_exception") == "true")
        entry["sensitivity_available_count"] += int(row.get("sensitivity_available_for_sensitivity_only") == "true")
        entry["member_parent_asin_count"] += 1
    if not grouped:
        raise ContractError("no eligible groups available for split")
    return [grouped[key] for key in sorted(grouped)]


def build_split_plan(rows: list[dict[str, str]]) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    groups = build_group_inventory(rows)
    summaries = [
        {
            "group_key": group["split_group_key"],
            "product_count": group["product_count"],
            "observed_positive_count": group["observed_positive_count"],
        }
        for group in groups
    ]
    locked = p8a.assign_groups_to_locked_test(summaries, salt=SPLIT_ALGORITHM_VERSION, test_proportion=0.2)
    folds = p8a.assign_groups_to_development_folds(summaries, locked, salt=SPLIT_ALGORITHM_VERSION, fold_count=5)
    assignments: dict[str, dict[str, Any]] = {}
    group_by_key = {group["split_group_key"]: group for group in groups}
    for key, partition in locked.items():
        if partition == "locked_test":
            assignments[key] = {"split_partition": "locked_test", "development_fold": None}
        elif partition == "development":
            if key not in folds:
                raise ContractError(f"development group missing fold assignment: {key}")
            assignments[key] = {"split_partition": "development", "development_fold": int(folds[key])}
        else:
            raise ContractError(f"unexpected split assignment: {partition}")
        group_by_key[key].update(assignments[key])
    if set(assignments) != set(group_by_key):
        raise ContractError("split assignment does not cover every eligible group")
    for row in rows:
        key = row.get("split_group_key", "")
        if row.get("split_eligible") == "true":
            assignment = assignments[key]
            row["split_partition"] = assignment["split_partition"]
            row["development_fold"] = "" if assignment["development_fold"] is None else str(assignment["development_fold"])
        else:
            row["split_partition"] = "excluded"
            row["development_fold"] = ""
    return groups, assignments


def validate_split_plan(
    rows: Iterable[Mapping[str, Any]],
    groups: Iterable[Mapping[str, Any]],
    assignments: Mapping[str, Mapping[str, Any]],
) -> None:
    group_rows: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("split_eligible") == "true":
            key = str(row.get("split_group_key") or "").upper()
            group_rows[key].append(row)
        elif row.get("split_partition") != "excluded" or row.get("development_fold") not in {"", None}:
            raise ContractError("excluded row has an invalid split assignment")
    group_keys = {str(group.get("split_group_key") or "").upper() for group in groups}
    if set(group_rows) != group_keys or set(assignments) != group_keys:
        raise ContractError("split group inventory does not match row group keys")
    for key, members in group_rows.items():
        expected = assignments[key]
        for row in members:
            if row.get("split_partition") != expected["split_partition"]:
                raise ContractError(f"group {key} has split leakage")
            expected_fold = "" if expected["development_fold"] is None else str(expected["development_fold"])
            if row.get("development_fold") != expected_fold:
                raise ContractError(f"group {key} has inconsistent development fold")
    if any(value["split_partition"] == "locked_test" and value["development_fold"] is not None for value in assignments.values()):
        raise ContractError("locked-test group was assigned a development fold")
    development_folds = {
        value["development_fold"]
        for value in assignments.values()
        if value["split_partition"] == "development"
    }
    if not development_folds <= set(range(5)):
        raise ContractError("development assignment contains an invalid fold")


def duplicate_group_partition_distribution(
    rows: Iterable[Mapping[str, Any]],
    groups: Iterable[Mapping[str, Any]] | None = None,
) -> dict[str, dict[str, int]]:
    eligible = [row for row in rows if row.get("split_eligible") == "true"]
    def metrics(members: list[Mapping[str, Any]]) -> dict[str, int]:
        counts = Counter(str(row.get("split_group_key") or "").upper() for row in members)
        duplicate_sizes = [size for size in counts.values() if size > 1]
        return {
            "groups_gt_one": len(duplicate_sizes),
            "rows": sum(duplicate_sizes),
            "row_excess": sum(size - 1 for size in duplicate_sizes),
        }
    return {
        "full_universe": metrics(eligible),
        "locked_test": metrics([row for row in eligible if row.get("split_partition") == "locked_test"]),
        "development": metrics([row for row in eligible if row.get("split_partition") == "development"]),
    }


def validate_primary_asset_bytes(rows: Iterable[Mapping[str, Any]], asset_root: Path, asset_root_repo_relative: str = ASSET_ROOT_REPO_RELATIVE) -> None:
    root = asset_root.resolve()
    for row in rows:
        if row.get("split_eligible") != "true":
            continue
        relative = Path(normalize_repo_path(row.get("primary_asset_path")))
        expected_local_path = _repo_relative_asset_path(asset_root_repo_relative, row.get("primary_asset_path"))
        if row.get("primary_local_path") != expected_local_path:
            raise ContractError(f"primary local path is not repo-relative to the frozen asset root: {row.get('parent_asin')}")
        path = (root / relative).resolve()
        if root not in path.parents:
            raise ContractError(f"primary asset path escapes frozen asset root: {row.get('parent_asin')}")
        if not path.exists():
            raise ContractError(f"missing primary asset bytes: {path}")
        actual = sha256_file(path)
        expected = str(row.get("primary_response_sha256") or "").upper()
        if actual != expected:
            raise ContractError(f"primary asset SHA mismatch for {row.get('parent_asin')}")


def _json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _csv_bytes(rows: Iterable[Mapping[str, Any]], columns: Iterable[str]) -> bytes:
    handle = io.StringIO(newline="")
    writer = csv.DictWriter(handle, fieldnames=list(columns), extrasaction="raise", lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({column: row.get(column, "") for column in writer.fieldnames})
    return handle.getvalue().encode("utf-8")


def _source_path(path_value: str) -> Path:
    return Path(path_value.replace("\\", "/")) if Path(path_value).is_absolute() else ROOT / path_value.replace("\\", "/")


def read_csv_header(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(next(csv.reader(handle)))


def validate_manifest_header(path: Path, p8a_contract: Mapping[str, Any] | None = None) -> None:
    if read_csv_header(path) != list(manifest_columns(p8a_contract)):
        raise ContractError("01_modeling_ready_manifest.csv header does not exactly match the manifest schema")


def _check_sha(path: Path, expected: str, label: str) -> None:
    if not path.exists():
        raise ContractError(f"missing {label}: {path}")
    actual = sha256_file(path)
    if actual != expected.upper():
        raise ContractError(f"{label} SHA mismatch: expected {expected}, got {actual}")


def validate_consumed_source_bindings(
    p8a_contract: Mapping[str, Any],
    p8b_contract: Mapping[str, Any],
) -> dict[str, Path]:
    sources = p8b_contract["source_artifacts"]
    expected = {
        "p7_d_final_manifest": (
            p8a_contract["p7_d_formal_paths"]["04_final_image_manifest.csv"],
            p8a_contract["p7_d_formal_sha256"]["04_final_image_manifest.csv"],
        ),
        "canonical_label_source": (
            p8a_contract["canonical_label_source"]["path"],
            p8a_contract["canonical_label_source"]["sha256"],
        ),
        "g_product_exposure_table": (
            p8a_contract["g_source_artifacts"]["01_product_exposure_label_table.csv"]["path"],
            p8a_contract["g_source_artifacts"]["01_product_exposure_label_table.csv"]["sha256"],
        ),
    }
    bindings: dict[str, Path] = {}
    for key, (authoritative_path, expected_sha) in expected.items():
        actual_path = normalize_repo_path(sources.get(key))
        if actual_path != normalize_repo_path(authoritative_path):
            raise ContractError(f"consumed {key} path is not the P8-A authoritative path")
        path = _source_path(actual_path)
        _check_sha(path, expected_sha, f"consumed {key}")
        bindings[key] = path
    p7b_path = normalize_repo_path(sources.get("p7_b_primary_asset_manifest"))
    expected_p7b_path = "data/processed/retail_outer_package_images_p7_5180/p7_b_primary_assets/01_primary_asset_manifest.csv"
    if p7b_path != expected_p7b_path:
        raise ContractError("consumed p7_b_primary_asset_manifest path is not frozen")
    bindings["p7_b_primary_asset_manifest"] = _source_path(p7b_path)
    _check_sha(bindings["p7_b_primary_asset_manifest"], p8b_contract["source_artifact_sha256"]["p7_b_primary_asset_manifest"], "consumed P7-B primary asset manifest")
    for key, label in (("p7_d_provenance", "P7-D provenance"), ("p7_b_provenance", "P7-B provenance")):
        bindings[key] = _source_path(sources[key])
        _check_sha(bindings[key], p8b_contract["source_artifact_sha256"][key], label)
    p7b_provenance = read_json(bindings["p7_b_provenance"])
    if normalize_repo_path(p7b_provenance.get("asset_root")) != p8b_contract["asset_root_repo_relative"]:
        raise ContractError("P7-B asset root does not match the P8-B asset-root contract")
    asset_root = _source_path(p8b_contract["asset_root_repo_relative"])
    if not asset_root.exists():
        raise ContractError(f"missing frozen P7-B asset root: {asset_root}")
    return bindings


def _quality_audit(rows: list[dict[str, str]], groups: list[dict[str, Any]], p8a_contract: Mapping[str, Any]) -> dict[str, Any]:
    eligible = [row for row in rows if row["split_eligible"] == "true"]
    total_rows = len(eligible)
    total_positive = sum(_as_binary(row[PRIMARY_OUTCOME], PRIMARY_OUTCOME) for row in eligible)
    by_partition: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in eligible:
        by_partition[row["split_partition"]].append(row)
    locked = by_partition["locked_test"]
    development = by_partition["development"]
    locked_target_rows = total_rows * 0.2
    locked_target_positive = total_positive * 0.2
    locked_loss = p8a.joint_balance_loss(len(locked), sum(_as_binary(row[PRIMARY_OUTCOME], PRIMARY_OUTCOME) for row in locked), locked_target_rows, locked_target_positive, total_rows, total_positive)
    def partition_metrics(members: list[dict[str, str]]) -> dict[str, Any]:
        positives = sum(_as_binary(row[PRIMARY_OUTCOME], PRIMARY_OUTCOME) for row in members)
        return {
            "row_count": len(members),
            "observed_positive_count": positives,
            "pu_unlabeled_count": len(members) - positives,
            "group_count": len({row["split_group_key"] for row in members}),
            "known_qa_exception_count": sum(row["known_p7c_qa_exception"] == "true" for row in members),
            "sensitivity_available_count": sum(row["sensitivity_available_for_sensitivity_only"] == "true" for row in members),
        }

    fold_audits = []
    development_rows = len(development)
    development_positive = sum(_as_binary(row[PRIMARY_OUTCOME], PRIMARY_OUTCOME) for row in development)
    fold_target_rows = development_rows * 0.2
    fold_target_positive = development_positive * 0.2
    global_loss = 0.0
    for fold in range(5):
        members = [row for row in development if row["development_fold"] == str(fold)]
        row_count = len(members)
        positive_count = sum(_as_binary(row[PRIMARY_OUTCOME], PRIMARY_OUTCOME) for row in members)
        loss = p8a.joint_balance_loss(row_count, positive_count, fold_target_rows, fold_target_positive, development_rows, development_positive)
        global_loss += loss
        fold_audits.append({
            "fold": fold,
            **partition_metrics(members),
            "target_rows": fold_target_rows,
            "target_observed_positive_count": fold_target_positive,
            "row_deviation": row_count - fold_target_rows,
            "observed_positive_deviation": positive_count - fold_target_positive,
            "joint_loss": loss,
        })
    group_partition = {group["split_group_key"]: group["split_partition"] for group in groups}
    return {
        "verification": "PASS",
        "population": {
            "full_universe_rows": len(rows),
            "main_modeling_population_rows": len(eligible),
            "excluded_non_primary_rows": sum(row["split_partition"] == "excluded" for row in rows),
            "known_p7c_qa_exception_rows": sum(row["known_p7c_qa_exception"] == "true" for row in eligible),
            "full_universe_sensitivity_available_rows": sum(row["sensitivity_available_for_sensitivity_only"] == "true" for row in rows),
            "main_modeling_population_sensitivity_available_rows": sum(row["sensitivity_available_for_sensitivity_only"] == "true" for row in eligible),
            "sensitivity_status_counts_full_universe": dict(sorted(Counter(row["sensitivity_status"] for row in rows).items())),
            "sensitivity_status_counts_main_modeling_population": dict(sorted(Counter(row["sensitivity_status"] for row in eligible).items())),
        },
        "primary_outcome": {
            "name": PRIMARY_OUTCOME,
            "eligible_rows": total_rows,
            "observed_positive_count": total_positive,
            "pu_unlabeled_count": total_rows - total_positive,
        },
        "locked_test": {
            **partition_metrics(locked),
            "target_rows": locked_target_rows,
            "target_observed_positive_count": locked_target_positive,
            "row_deviation": len(locked) - locked_target_rows,
            "observed_positive_deviation": sum(_as_binary(row[PRIMARY_OUTCOME], PRIMARY_OUTCOME) for row in locked) - locked_target_positive,
            "joint_loss": locked_loss,
        },
        "development": {
            **partition_metrics(development),
            "target_fold_rows": fold_target_rows,
            "target_fold_observed_positive_count": fold_target_positive,
            "global_joint_loss": global_loss,
            "folds": fold_audits,
        },
        "group_inventory": {
            "group_key": GROUP_KEY,
            "eligible_group_count": len(groups),
            "duplicate_groups_gt_one": sum(group["member_parent_asin_count"] > 1 for group in groups),
            "rows_in_duplicate_groups_gt_one": sum(group["member_parent_asin_count"] for group in groups if group["member_parent_asin_count"] > 1),
            "maximum_group_size": max(group["member_parent_asin_count"] for group in groups),
            "locked_test_group_count": sum(value == "locked_test" for value in group_partition.values()),
            "development_group_count": sum(value == "development" for value in group_partition.values()),
        },
        "integrity_checks": {
            "primary_response_sha256_group_integrity": True,
            "excluded_non_primary_not_in_split": all(row["split_partition"] == "excluded" for row in rows if row["split_eligible"] == "false"),
            "development_consumes_post_locked_groups_only": True,
            "development_fold_count": 5,
            "locked_test_group_overlap_with_development": 0,
            "cross_fold_group_overlap": 0,
            "excluded_in_split": 0,
            "manual_split_modification": False,
            "seed_search": False,
            "favorable_split_retry": False,
            "secondary_outcome_split_optimization": False,
            "no_secondary_outcome_in_split_objective": True,
            "no_g_threshold_selected": p8a_contract["g_policy"]["g_final_threshold_selected"] is False,
        },
        "not_started": {
            "feature_extraction": False,
            "ocr": False,
            "clip_dino_cnn": False,
            "pca": False,
            "model_fitting": False,
            "hyperparameter_tuning": False,
            "predictions": False,
            "performance_metrics": False,
            "p9": False,
        },
    }


def _summary(
    rows: list[dict[str, str]],
    quality: Mapping[str, Any],
    p8a_contract: Mapping[str, Any],
    output_hashes: Mapping[str, str],
    label_rows: Iterable[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    outcomes = p8a.canonical_pu_outcomes(p8a_contract)
    outcome_counts = {
        outcome: sum(_as_binary(row[outcome], outcome) for row in rows)
        for outcome in outcomes
    }
    core = p8a.derive_core_eligibility(list(label_rows) if label_rows is not None else read_csv(_source_path(p8a_contract["canonical_label_source"]["path"])))
    return {
        "verification": "PASS",
        "stage": "P8-B modeling-ready manifest and deterministic split freeze",
        "contract_version": "p8b_v1.0",
        "population_and_split": quality,
        "duplicate_group_partition_distribution": duplicate_group_partition_distribution(rows),
        "outcome_hierarchy": {
            "primary": p8a_contract["outcome_hierarchy"]["primary"],
            "secondary_confirmatory": p8a_contract["outcome_hierarchy"]["secondary_confirmatory"],
            "robustness": p8a_contract["outcome_hierarchy"]["robustness"],
            "core_eligibility": core,
            "canonical_pu_outcome_count": len(outcomes),
            "observed_positive_counts_full_manifest": outcome_counts,
            "zero_semantics": p8a_contract["label_semantics"]["zero"],
        },
        "g_policy": {
            "role": p8a_contract["g_policy"]["g_role"],
            "thresholds_reported": p8a_contract["g_policy"]["existing_threshold_grid"],
            "strata_reported": p8a_contract["g_policy"]["existing_strata_grid"],
            "final_threshold_selected": False,
        },
        "formal_output_sha256": dict(output_hashes),
        "formal_data_git_tracked": False,
        "features_or_models_started": False,
    }


def _find_producer_commit(payload: Any, required_tokens: tuple[str, ...]) -> str | None:
    if isinstance(payload, Mapping):
        for key, value in payload.items():
            key_text = str(key).lower()
            if all(token in key_text for token in required_tokens) and isinstance(value, str) and re.fullmatch(r"[0-9a-fA-F]{40}", value):
                return value
            found = _find_producer_commit(value, required_tokens)
            if found:
                return found
    elif isinstance(payload, list):
        for value in payload:
            found = _find_producer_commit(value, required_tokens)
            if found:
                return found
    return None


def _provenance(
    p8a_contract: Mapping[str, Any],
    p8b_contract: Mapping[str, Any],
    output_hashes: Mapping[str, str],
    formal_run_git_commit: str,
) -> dict[str, Any]:
    p7_provenance_path = _source_path(p8b_contract["source_artifacts"]["p7_d_provenance"])
    p7_prov = read_json(p7_provenance_path)
    p7_b_prov = read_json(_source_path(p8b_contract["source_artifacts"]["p7_b_provenance"]))
    phase1 = _find_producer_commit(p7_prov, ("phase1", "producer", "commit"))
    phase2 = _find_producer_commit(p7_prov, ("phase2", "diagnostics", "producer", "commit"))
    if not phase1 or not phase2:
        raise ContractError("P7-D provenance does not distinguish Phase-1 and Phase-2 producers")
    tracked = {
        "producer": "scripts/modeling/p8_b_modeling_ready_split.py",
        "contract": "config/modeling/p8_b_modeling_ready_contract.json",
        "manifest_schema": "config/modeling/p8_b_modeling_manifest_schema.json",
        "upstream_p8_a_contract": "config/modeling/p8_a_analysis_contract.json",
    }
    consumed_sources = {
        "p7_d_final_manifest": {
            "path": p8b_contract["source_artifacts"]["p7_d_final_manifest"],
            "sha256": sha256_file(_source_path(p8b_contract["source_artifacts"]["p7_d_final_manifest"])),
        },
        "canonical_label_source": {
            "path": p8b_contract["source_artifacts"]["canonical_label_source"],
            "sha256": sha256_file(_source_path(p8b_contract["source_artifacts"]["canonical_label_source"])),
        },
        "g_product_exposure_table": {
            "path": p8b_contract["source_artifacts"]["g_product_exposure_table"],
            "sha256": sha256_file(_source_path(p8b_contract["source_artifacts"]["g_product_exposure_table"])),
        },
        "p7_b_primary_asset_manifest": {
            "path": p8b_contract["source_artifacts"]["p7_b_primary_asset_manifest"],
            "sha256": sha256_file(_source_path(p8b_contract["source_artifacts"]["p7_b_primary_asset_manifest"])),
        },
    }
    return {
        "baseline_main_commit": p8b_contract["baseline_main_commit"],
        "p8_a_formal_output_sha256": dict(p8b_contract["p8_a_formal_output_sha256"]),
        "asset_root_repo_relative": p8b_contract["asset_root_repo_relative"],
        "consumed_sources": consumed_sources,
        "formal_run_git_commit": formal_run_git_commit,
        "producer": {"path": tracked["producer"], "git_blob_sha256": p8a.git_blob_sha256(formal_run_git_commit, tracked["producer"])},
        "contract": {"path": tracked["contract"], "git_blob_sha256": p8a.git_blob_sha256(formal_run_git_commit, tracked["contract"])},
        "manifest_schema": {"path": tracked["manifest_schema"], "git_blob_sha256": p8a.git_blob_sha256(formal_run_git_commit, tracked["manifest_schema"])},
        "upstream_p8_a_contract": {"path": tracked["upstream_p8_a_contract"], "git_blob_sha256": p8a.git_blob_sha256(formal_run_git_commit, tracked["upstream_p8_a_contract"])},
        "producer_roles": {
            "phase1_acquisition_review_selection": {"commit": phase1, "source": "P7-D provenance"},
            "phase2_diagnostics": {"commit": phase2, "source": "P7-D provenance"},
            "phase2_modeling_ready_manifest_split": {"commit": formal_run_git_commit, "source": "P8-B formal code producer"},
        },
        "authoritative_upstream": {
            "upstream_p8_a_formal_run_git_commit": read_json(p8a.FORMAL_DIR / "05_p8_a_provenance.json").get("formal_run_git_commit"),
            "p7_d_formal_sha256": dict(p8a_contract["p7_d_formal_sha256"]),
            "p7_d_final_manifest_sha256": sha256_file(_source_path(p8a_contract["p7_d_formal_paths"]["04_final_image_manifest.csv"])),
            "canonical_label_source": dict(p8a_contract["canonical_label_source"]),
            "g_source_artifacts": dict(p8a_contract["g_source_artifacts"]),
            "p7_b_provenance": {
                "path": p8b_contract["source_artifacts"]["p7_b_provenance"],
                "sha256": sha256_file(_source_path(p8b_contract["source_artifacts"]["p7_b_provenance"])),
                "asset_root": p7_b_prov.get("asset_root"),
            },
        },
        "split": {
            "algorithm_version": SPLIT_ALGORITHM_VERSION,
            "group_key": GROUP_KEY,
            "primary_stratification_outcome": PRIMARY_OUTCOME,
            "hash_expression": p8b_contract["split"]["hash_expression"],
            "hash_purpose": p8b_contract["split"]["hash_purpose"],
        },
        "formal_output_sha256": {name: output_hashes[name] for name in FORMAL_FILES[:5]},
        "provenance_self_sha": None,
    }


def _build_artifacts(
    p8a_contract: Mapping[str, Any],
    p8b_contract: Mapping[str, Any],
    formal_run_git_commit: str,
) -> dict[str, bytes]:
    validate_contract(p8b_contract)
    load_manifest_schema(p8a_contract)
    bindings = validate_consumed_source_bindings(p8a_contract, p8b_contract)
    p7_rows = read_csv(bindings["p7_d_final_manifest"])
    label_rows = read_csv(bindings["canonical_label_source"])
    g_rows = read_csv(bindings["g_product_exposure_table"])
    p7b_rows = read_csv(bindings["p7_b_primary_asset_manifest"])
    rows = build_manifest_rows(p7_rows, label_rows, g_rows, p8a_contract, p8b_contract, p7b_rows=p7b_rows)
    expected_population = p8b_contract["population"]
    if len(rows) != expected_population["full_universe_rows"]:
        raise ContractError("P8-B full universe row count mismatch")
    if sum(row["split_eligible"] == "true" for row in rows) != expected_population["expected_main_rows"]:
        raise ContractError("P8-B main population row count mismatch")
    if sum(row["known_p7c_qa_exception"] == "true" for row in rows) != expected_population["known_qa_exception_count"]:
        raise ContractError("P8-B QA exception count mismatch")
    groups, assignments = build_split_plan(rows)
    validate_split_plan(rows, groups, assignments)
    if {
        int(row["development_fold"])
        for row in rows
        if row["split_partition"] == "development"
    } != set(range(5)):
        raise ContractError("real development split does not cover all five folds")
    p7b_provenance = read_json(bindings["p7_b_provenance"])
    asset_root = _source_path(str(p7b_provenance["asset_root"]))
    validate_primary_asset_bytes(rows, asset_root, p8b_contract["asset_root_repo_relative"])
    quality = _quality_audit(rows, groups, p8a_contract)
    manifest_columns_expected = tuple(manifest_columns(p8a_contract))
    schema_columns = tuple(item["name"] for item in load_manifest_schema(p8a_contract)["columns"])
    if schema_columns != manifest_columns_expected:
        raise ContractError("manifest schema columns diverge from generated manifest columns")
    manifest_bytes = _csv_bytes(rows, manifest_columns_expected)
    groups_bytes = _csv_bytes(groups, GROUP_COLUMNS)
    group_metrics = {group["split_group_key"]: group for group in groups}
    assignment_rows = []
    for row in rows:
        key = row["split_group_key"]
        group = group_metrics.get(key, {})
        assignment_rows.append({
            "parent_asin": row["parent_asin"],
            "input_order": row["input_order"],
            "split_eligible": row["split_eligible"],
            "split_group_key": key,
            "primary_outcome": row[PRIMARY_OUTCOME],
            "split_algorithm_version": row["split_algorithm_version"],
            "split_salt": row["split_salt"],
            "product_count": group.get("product_count", ""),
            "observed_positive_count": group.get("observed_positive_count", ""),
            "split_partition": row["split_partition"],
            "development_fold": row["development_fold"],
            "split_hash_order_key": row["split_hash_order_key"],
        })
    assignment_bytes = _csv_bytes(assignment_rows, ASSIGNMENT_COLUMNS)
    quality_bytes = _json_bytes(quality)
    preliminary = {
        "01_modeling_ready_manifest.csv": sha256_bytes(manifest_bytes),
        "02_split_group_inventory.csv": sha256_bytes(groups_bytes),
        "03_split_assignment.csv": sha256_bytes(assignment_bytes),
        "04_split_quality_audit.json": sha256_bytes(quality_bytes),
    }
    summary = _summary(rows, quality, p8a_contract, preliminary, label_rows)
    summary_bytes = _json_bytes(summary)
    output_hashes = dict(preliminary)
    output_hashes["05_modeling_ready_summary.json"] = sha256_bytes(summary_bytes)
    provenance = _provenance(p8a_contract, p8b_contract, output_hashes, formal_run_git_commit)
    return {
        "01_modeling_ready_manifest.csv": manifest_bytes,
        "02_split_group_inventory.csv": groups_bytes,
        "03_split_assignment.csv": assignment_bytes,
        "04_split_quality_audit.json": quality_bytes,
        "05_modeling_ready_summary.json": summary_bytes,
        "06_p8_b_provenance.json": _json_bytes(provenance),
    }


def validate_provenance_self_guard(provenance: Mapping[str, Any]) -> None:
    if provenance.get("provenance_self_sha") is not None:
        raise ContractError("P8-B provenance must not contain its own SHA")
    for field in ("formal_output_sha256", "output_sha256"):
        if PROVENANCE_FILENAME in provenance.get(field, {}):
            raise ContractError("P8-B provenance records its own output SHA")


def _validate_provenance(provenance: Mapping[str, Any]) -> None:
    validate_provenance_self_guard(provenance)
    contract = load_contract()
    if provenance.get("baseline_main_commit") != contract["baseline_main_commit"]:
        raise ContractError("P8-B provenance baseline main commit mismatch")
    if provenance.get("p8_a_formal_output_sha256") != contract["p8_a_formal_output_sha256"]:
        raise ContractError("P8-B provenance P8-A formal SHA ledger mismatch")
    if provenance.get("asset_root_repo_relative") != contract["asset_root_repo_relative"]:
        raise ContractError("P8-B provenance asset-root semantics mismatch")
    bindings = validate_consumed_source_bindings(p8a.load_contract(), contract)
    consumed = provenance.get("consumed_sources", {})
    for key in ("p7_d_final_manifest", "canonical_label_source", "g_product_exposure_table", "p7_b_primary_asset_manifest"):
        trace = consumed.get(key, {})
        if normalize_repo_path(trace.get("path")) != normalize_repo_path(contract["source_artifacts"][key]):
            raise ContractError(f"P8-B provenance consumed source path mismatch: {key}")
        if trace.get("sha256") != sha256_file(bindings[key]):
            raise ContractError(f"P8-B provenance consumed source SHA mismatch: {key}")
    output_hashes = provenance.get("formal_output_sha256")
    if set(output_hashes or {}) != set(FORMAL_FILES[:5]):
        raise ContractError("P8-B provenance must bind exactly formal outputs 01-05")
    commit = str(provenance.get("formal_run_git_commit") or "")
    if not re.fullmatch(r"[0-9a-fA-F]{40}", commit):
        raise ContractError("P8-B formal_run_git_commit is missing or invalid")
    for field in ("producer", "contract", "manifest_schema", "upstream_p8_a_contract"):
        trace = provenance.get(field, {})
        path = trace.get("path")
        if not path or p8a.git_blob_sha256(commit, path) != trace.get("git_blob_sha256"):
            raise ContractError(f"P8-B {field} Git blob provenance mismatch")
        if p8a.git_blob_sha256(subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(), path) != trace.get("git_blob_sha256"):
            raise ContractError(f"current {field} Git blob differs from formal provenance")
    for name in FORMAL_FILES[:5]:
        if output_hashes[name] != sha256_file(FORMAL_DIR / name):
            raise ContractError(f"P8-B formal output SHA mismatch: {name}")


def prepare(p8b_contract: Mapping[str, Any] | None = None) -> dict[str, Any]:
    p8b_contract = p8b_contract or load_contract()
    if FORMAL_DIR.exists() and any(FORMAL_DIR.iterdir()):
        raise ContractError("P8-B formal output directory is non-empty; refusing overwrite")
    p8a_contract = p8a.load_contract()
    upstream_result = p8a.verify_existing(p8a_contract)
    artifacts = _build_artifacts(p8a_contract, p8b_contract, subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip())
    FORMAL_DIR.mkdir(parents=True, exist_ok=True)
    for name in FORMAL_FILES:
        (FORMAL_DIR / name).write_bytes(artifacts[name])
    return {
        "verification": "PASS",
        "upstream_p8_a_verification": upstream_result,
        "formal_dir": str(FORMAL_DIR.relative_to(ROOT)).replace("\\", "/"),
        "formal_files": list(FORMAL_FILES),
        "feature_extraction_started": False,
        "model_fitting_started": False,
        "split_assigned": True,
    }


def verify_existing(p8b_contract: Mapping[str, Any] | None = None) -> dict[str, Any]:
    p8b_contract = p8b_contract or load_contract()
    if not FORMAL_DIR.exists() or {path.name for path in FORMAL_DIR.iterdir()} != set(FORMAL_FILES):
        raise ContractError("P8-B formal outputs are incomplete or contain unexpected files")
    p8a_contract = p8a.load_contract()
    upstream_result = p8a.verify_existing(p8a_contract)
    actual_provenance = read_json(FORMAL_DIR / PROVENANCE_FILENAME)
    validate_manifest_header(FORMAL_DIR / FORMAL_FILES[0], p8a_contract)
    _validate_provenance(actual_provenance)
    artifacts = _build_artifacts(p8a_contract, p8b_contract, str(actual_provenance["formal_run_git_commit"]))
    for name in FORMAL_FILES:
        if (FORMAL_DIR / name).read_bytes() != artifacts[name]:
            raise ContractError(f"P8-B formal output reconstruction mismatch: {name}")
    tracked = subprocess.check_output(["git", "ls-files", "--", "data/processed/modeling_readiness_p8_5180"], cwd=ROOT, text=True).strip()
    if tracked:
        raise ContractError("P8-B formal data must remain gitignored")
    for path in FORMAL_DIR.iterdir():
        if path.name.startswith(".") or path.name.endswith(".tmp"):
            raise ContractError(f"temporary formal artifact found: {path}")
        if any(term in path.name.lower() for term in ("feature", "embedding", "prediction", "model_metric", "auc")):
            raise ContractError(f"forbidden formal artifact found: {path}")
    return {
        "verification": "PASS",
        "upstream_p8_a_verification": upstream_result,
        "formal_outputs_reconstruct": "PASS",
        "provenance_git_blob_binding": "PASS",
        "zero_writes": True,
        "zero_network_calls": True,
        "zero_model_calls": True,
        "feature_extraction_started": False,
        "model_fitting_started": False,
        "predictions_or_performance_metrics_started": False,
        "split_assigned": True,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare",), nargs="?")
    parser.add_argument("--verify-existing", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.verify_existing:
        print(json.dumps(verify_existing(), ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    if args.command == "prepare":
        print(json.dumps(prepare(), ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    raise ContractError("choose prepare or --verify-existing")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ContractError as exc:
        print(f"P8-B ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
