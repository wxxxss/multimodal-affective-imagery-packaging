#!/usr/bin/env python3
"""P7-C: label-blind QA of frozen P7-B primary assets and manifest freeze."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[2]
CONTRACT_PATH = ROOT / "config" / "image_assets" / "p7_c_qa_contract.json"
PROMPT_PATH = ROOT / "config" / "image_assets" / "p7_c_review_prompt.json"
PROMPT_REPO_PATH = "config/image_assets/p7_c_review_prompt.json"
PRODUCT_INPUT_PATH = ROOT / "data" / "processed" / "review_matching_5180" / "01_valid_products.csv"
P7A_SCRIPT = ROOT / "scripts" / "images" / "audit_p7_source_inventory.py"
P7B_SCRIPT = ROOT / "scripts" / "images" / "acquire_p7_primary_assets.py"

SAMPLE_FIELDS = (
    "parent_asin", "input_order", "response_sha256", "asset_path", "decoded_format",
    "width", "height", "response_byte_count", "exact_duplicate_asset_group_size",
    "baseline_random", "exact_byte_duplicate_product", "gif_asset",
    "low_min_dimension_asset", "low_byte_size_asset", "qa_strata",
    "title", "store", "categories", "automatic_tea_type", "product_form",
)
REVIEW_FIELDS = (
    "parent_asin", "response_sha256", "identity_status", "outer_package_status",
    "confidence", "reason_code", "reviewer_model", "reviewer_run_id",
    "review_prompt_sha256", "reviewed_at_utc",
)
ADJ_FIELDS = (
    "parent_asin", "response_sha256", "pass_a_identity_status", "pass_a_outer_package_status",
    "pass_a_confidence", "pass_b_identity_status", "pass_b_outer_package_status",
    "pass_b_confidence", "requires_adjudication", "identity_status", "outer_package_status",
    "confidence", "reason_code", "adjudication_basis", "adjudicator_model",
    "adjudicator_run_id", "adjudication_prompt_sha256", "qa_exception_flag",
)
MANIFEST_FIELDS = (
    "parent_asin", "input_order", "source_record_sha256", "image_role",
    "temporal_alignment_status", "eligibility_status", "response_sha256", "asset_path",
    "decoded_format", "width", "height", "n_frames", "response_byte_count",
    "exact_duplicate_asset_id", "exact_duplicate_asset_group_size", "qa_sampled",
    "qa_strata", "qa_identity_status", "qa_outer_package_status", "qa_confidence",
    "qa_adjudication_basis", "qa_exception_flag", "primary_freeze_status",
    "primary_freeze_reason",
)

INT_FIELDS = {"input_order", "width", "height", "n_frames", "response_byte_count", "exact_duplicate_asset_group_size"}
BOOL_FIELDS = {"baseline_random", "exact_byte_duplicate_product", "gif_asset", "low_min_dimension_asset", "low_byte_size_asset", "qa_sampled", "qa_exception_flag", "requires_adjudication"}
CONFIDENCE_RANK = {"low": 0, "medium": 1, "high": 2}


class FormalVerificationError(RuntimeError):
    pass


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest().upper()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def git_blob_sha256(commit: str, repo_path: str | Path) -> str:
    relative = Path(repo_path).as_posix()
    data = subprocess.check_output(["git", "cat-file", "blob", f"{commit}:{relative}"], cwd=ROOT)
    return sha256_bytes(data)


def validate_prompt_contract_identity(contract: Mapping[str, Any]) -> str:
    expected = str(contract["review_prompt_sha256"]).upper()
    current = git_blob_sha256("HEAD", PROMPT_REPO_PATH)
    if current.upper() != expected:
        raise FormalVerificationError("frozen review prompt Git blob SHA does not match contract")
    return current


def validate_prompt_provenance(provenance: Mapping[str, Any], contract: Mapping[str, Any]) -> dict[str, bool]:
    path = str(provenance.get("review_prompt_path") or "")
    if path != PROMPT_REPO_PATH:
        raise FormalVerificationError(f"unexpected frozen review prompt path: {path}")
    expected = str(contract["review_prompt_sha256"]).upper()
    recorded = str(provenance.get("review_prompt_sha256") or "").upper()
    historical = git_blob_sha256(str(provenance["formal_run_git_commit"]), path).upper()
    current = git_blob_sha256("HEAD", path).upper()
    checks = {
        "historical_prompt_git_blob_sha256_match": historical == recorded,
        "current_head_prompt_git_blob_sha256_match": current == expected,
        "review_prompt_sha_matches_contract": recorded == expected,
    }
    if not checks["historical_prompt_git_blob_sha256_match"]:
        raise FormalVerificationError("historical prompt Git blob does not match provenance")
    if not checks["current_head_prompt_git_blob_sha256_match"]:
        raise FormalVerificationError("current prompt Git blob does not match contract")
    if not checks["review_prompt_sha_matches_contract"]:
        raise FormalVerificationError("recorded prompt SHA does not match contract")
    return checks

def load_contract(path: Path = CONTRACT_PATH) -> dict[str, Any]:
    contract = json.loads(path.read_text(encoding="utf-8"))
    required = ("contract_version", "upstream_main_commit", "upstream_p7_a", "upstream_p7_b", "formal_paths", "sample_algorithm", "review_model_identifier", "adjudicator_model_identifier", "review_prompt_sha256", "decoded_placeholder_signatures", "qa_sanity")
    missing = [key for key in required if key not in contract]
    if missing or contract.get("contract_version") != "p7_c_v1.0":
        raise FormalVerificationError(f"invalid P7-C contract: missing={missing}")
    validate_prompt_contract_identity(contract)
    return contract


def load_prompt(path: Path = PROMPT_PATH) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not payload.get("prompt") or payload.get("prompt_version") != "p7c_review_v1.0":
        raise FormalVerificationError("invalid frozen P7-C review prompt")
    return payload


def _parse_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = [dict(row) for row in csv.DictReader(handle)]
    for row in rows:
        for field in INT_FIELDS:
            if field in row:
                row[field] = int(row.get(field) or 0)
        for field in BOOL_FIELDS:
            if field in row:
                row[field] = str(row.get(field) or "").lower() in {"1", "true", "yes"}
    return rows


def _write_csv(path: Path, fields: tuple[str, ...], rows: list[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _paths(contract: Mapping[str, Any]) -> dict[str, Path]:
    return {key: ROOT / value for key, value in contract["formal_paths"].items()}


def _run_zero_write_gate(script: Path, args: list[str]) -> None:
    result = subprocess.run([sys.executable, str(script), *args], cwd=ROOT, capture_output=True, text=True)
    if result.returncode != 0:
        raise FormalVerificationError(f"upstream verification failed: {script.name}\n{result.stdout}\n{result.stderr}")
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise FormalVerificationError(f"upstream verifier returned non-JSON: {script.name}") from exc
    if payload.get("verification") != "PASS":
        raise FormalVerificationError(f"upstream verifier did not PASS: {script.name}")


def p7_a_verify_args() -> list[str]:
    return [
        "--input",
        "data/processed/review_matching_5180/01_valid_products.csv",
        "--raw-metadata",
        "data/meta_Grocery_and_Gourmet_Food.jsonl/meta_Grocery_and_Gourmet_Food.jsonl",
        "--output-dir",
        "data/processed/retail_outer_package_images_p7_5180/p7_a_source_inventory",
        "--verify-existing",
    ]


def validate_upstream(contract: Mapping[str, Any], *, invoke_verifiers: bool = True) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, dict[str, Any]]]:
    a = contract["upstream_p7_a"]
    b = contract["upstream_p7_b"]
    expected_files = [
        (a["inventory_path"], a["inventory_sha256"]), (a["summary_path"], a["summary_sha256"]),
        (a["provenance_path"], a["provenance_sha256"]), (b["manifest_path"], b["manifest_sha256"]),
        (b["unique_inventory_path"], b["unique_inventory_sha256"]), (b["summary_path"], b["summary_sha256"]),
        (b["provenance_path"], b["provenance_sha256"]),
    ]
    for raw_path, expected in expected_files:
        path = ROOT / raw_path
        if not path.exists() or sha256_file(path) != expected:
            raise FormalVerificationError(f"frozen upstream SHA mismatch: {raw_path}")
    if invoke_verifiers:
        _run_zero_write_gate(P7A_SCRIPT, p7_a_verify_args())
        _run_zero_write_gate(P7B_SCRIPT, ["--verify-existing"])
    manifest = _parse_csv(ROOT / b["manifest_path"])
    unique = _parse_csv(ROOT / b["unique_inventory_path"])
    metadata_rows = _parse_csv(PRODUCT_INPUT_PATH)
    if len(manifest) != contract["product_universe_count"] or len({r["parent_asin"] for r in manifest}) != contract["product_universe_count"]:
        raise FormalVerificationError("P7-B manifest universe mismatch")
    if sum(r["eligibility_status"] == "primary_declared_main" for r in manifest) != contract["primary_available_count"]:
        raise FormalVerificationError("P7-B available count mismatch")
    if sum(r["eligibility_status"] == "excluded_non_declared_main" for r in manifest) != contract["excluded_non_primary_count"]:
        raise FormalVerificationError("P7-B excluded count mismatch")
    if len(unique) != contract["unique_asset_count"]:
        raise FormalVerificationError("P7-B unique asset count mismatch")
    metadata = {row["parent_asin"]: {key: row.get(key, "") for key in contract["identity_reference_columns"]} for row in metadata_rows}
    asset_root = ROOT / "data" / "images" / "retail_outer_package_p7_5180"
    for row in manifest:
        if row["eligibility_status"] != "primary_declared_main":
            continue
        asset = asset_root / row["asset_path"]
        if not asset.exists() or sha256_file(asset) != row["response_sha256"]:
            raise FormalVerificationError(f"missing or substituted frozen asset: {row['parent_asin']}")
    return manifest, unique, metadata


def _asset_key(row: Mapping[str, Any]) -> str:
    return str(row.get("response_sha256") or row.get("asset_sha256") or "").upper()


def build_deterministic_sample(products: list[Mapping[str, Any]], unique_assets: list[Mapping[str, Any]], *, baseline_count: int = 400, low_unique_count: int = 50, prefix: str = "p7c_v1|") -> list[dict[str, Any]]:
    eligible = [dict(row) for row in products if row.get("eligibility_status") == "primary_declared_main"]
    if baseline_count > len(eligible):
        raise FormalVerificationError("baseline sample larger than eligible universe")
    baseline = {
        row["parent_asin"] for row in sorted(
            eligible,
            key=lambda row: hashlib.sha256(f"{prefix}{row['parent_asin']}|{row['response_sha256']}".encode()).hexdigest(),
        )[:baseline_count]
    }
    duplicate = {row["parent_asin"] for row in eligible if int(row.get("exact_duplicate_asset_group_size") or 0) > 1}
    gif = {row["parent_asin"] for row in eligible if str(row.get("decoded_format") or "").upper() == "GIF"}
    low_dimension = sorted(unique_assets, key=lambda row: (min(int(row.get("width") or 0), int(row.get("height") or 0)), _asset_key(row)))[:low_unique_count]
    low_bytes = sorted(unique_assets, key=lambda row: (int(row.get("response_byte_count") or 0), _asset_key(row)))[:low_unique_count]
    low_dimension_keys = {_asset_key(row) for row in low_dimension}
    low_bytes_keys = {_asset_key(row) for row in low_bytes}
    strata_by_asin: dict[str, set[str]] = {}
    for row in eligible:
        asin = row["parent_asin"]
        strata_by_asin[asin] = set()
        if asin in baseline:
            strata_by_asin[asin].add("baseline_random")
        if asin in duplicate:
            strata_by_asin[asin].add("exact_byte_duplicate_product")
        if asin in gif:
            strata_by_asin[asin].add("gif_asset")
        if _asset_key(row) in low_dimension_keys:
            strata_by_asin[asin].add("low_min_dimension_asset")
        if _asset_key(row) in low_bytes_keys:
            strata_by_asin[asin].add("low_byte_size_asset")
    sample = []
    for row in sorted(eligible, key=lambda item: int(item["input_order"])):
        strata = strata_by_asin[row["parent_asin"]]
        if not strata:
            continue
        sample.append({
            **{field: row.get(field, "") for field in ("parent_asin", "input_order", "response_sha256", "asset_path", "decoded_format", "width", "height", "response_byte_count", "exact_duplicate_asset_group_size")},
            "baseline_random": "baseline_random" in strata,
            "exact_byte_duplicate_product": "exact_byte_duplicate_product" in strata,
            "gif_asset": "gif_asset" in strata,
            "low_min_dimension_asset": "low_min_dimension_asset" in strata,
            "low_byte_size_asset": "low_byte_size_asset" in strata,
            "qa_strata": "|".join(sorted(strata)),
        })
    return sample


def attach_identity_metadata(sample: list[dict[str, Any]], metadata: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for row in sample:
        if row["parent_asin"] not in metadata:
            raise FormalVerificationError(f"metadata missing sample parent_asin: {row['parent_asin']}")
        output.append({**row, **metadata[row["parent_asin"]]})
    return output


def _validate_sample_binding(row: Mapping[str, Any], sample_by_asin: Mapping[str, Mapping[str, Any]], contract: Mapping[str, Any], role: str) -> None:
    asin = str(row.get("parent_asin") or "")
    if asin not in sample_by_asin:
        raise FormalVerificationError(f"{role} row not in frozen sample: {asin}")
    expected = sample_by_asin[asin]
    if str(row.get("response_sha256") or "").upper() != str(expected.get("response_sha256") or "").upper():
        raise FormalVerificationError(f"{role} response SHA mismatch with frozen sample: {asin}")
    expected_prompt = contract.get("review_prompt_sha256")
    if expected_prompt and str(row.get("review_prompt_sha256") or "").upper() != str(expected_prompt).upper():
        raise FormalVerificationError(f"{role} review prompt SHA mismatch: {asin}")
    expected_model = contract.get("review_model_identifier")
    if expected_model and row.get("reviewer_model") != expected_model:
        raise FormalVerificationError(f"{role} reviewer model mismatch: {asin}")


def validate_review_pass(rows: list[Mapping[str, Any]], sample: list[Mapping[str, Any]], contract: Mapping[str, Any], role: str) -> None:
    sample_by_asin = {row["parent_asin"]: row for row in sample}
    if {row.get("parent_asin") for row in rows} != set(sample_by_asin) or len(rows) != len(sample_by_asin):
        raise FormalVerificationError(f"sample universe mismatch in pass {role}")
    if len({row.get("parent_asin") for row in rows}) != len(rows):
        raise FormalVerificationError(f"duplicate parent_asin in pass {role}")
    for row in rows:
        _validate_enum(row, contract)
        _validate_sample_binding(row, sample_by_asin, contract, f"pass {role}")

def _validate_enum(row: Mapping[str, Any], contract: Mapping[str, Any]) -> None:
    if row.get("identity_status") not in contract["identity_status_enum"]:
        raise FormalVerificationError(f"invalid identity_status: {row.get('identity_status')}")
    if row.get("outer_package_status") not in contract["outer_package_status_enum"]:
        raise FormalVerificationError(f"invalid outer_package_status: {row.get('outer_package_status')}")
    if row.get("confidence") not in contract["confidence_enum"]:
        raise FormalVerificationError(f"invalid confidence: {row.get('confidence')}")
    for field in ("parent_asin", "response_sha256", "reviewer_model", "reviewer_run_id", "review_prompt_sha256", "reviewed_at_utc"):
        if not str(row.get(field) or ""):
            raise FormalVerificationError(f"review field missing: {field}")


def validate_review_rows(pass_a: list[Mapping[str, Any]], pass_b: list[Mapping[str, Any]], sample_asins: list[str], contract: Mapping[str, Any] | None = None, sample_rows: list[Mapping[str, Any]] | None = None) -> None:
    contract = contract or {"identity_status_enum": ["consistent", "contradiction", "indeterminate"], "outer_package_status_enum": ["outer_retail_package", "inner_packaging_only", "product_content_only", "lifestyle_or_serving", "composite_or_collage", "ambiguous", "unavailable_or_corrupt"], "confidence_enum": ["high", "medium", "low"]}
    if sample_rows is not None:
        validate_review_pass(pass_a, sample_rows, contract, "A")
        validate_review_pass(pass_b, sample_rows, contract, "B")
    else:
        expected = set(sample_asins)
        for rows, name in ((pass_a, "A"), (pass_b, "B")):
            if {row.get("parent_asin") for row in rows} != expected or len(rows) != len(expected):
                raise FormalVerificationError(f"sample universe mismatch in pass {name}")
            if len({row.get("parent_asin") for row in rows}) != len(rows):
                raise FormalVerificationError(f"duplicate parent_asin in pass {name}")
            for row in rows:
                _validate_enum(row, contract)
    runs_a = {str(row["reviewer_run_id"]) for row in pass_a}
    runs_b = {str(row["reviewer_run_id"]) for row in pass_b}
    if runs_a & runs_b:
        raise FormalVerificationError("Pass A and Pass B reviewer run IDs must be independent")
    if {row["parent_asin"]: row["response_sha256"] for row in pass_a} != {row["parent_asin"]: row["response_sha256"] for row in pass_b}:
        raise FormalVerificationError("Pass A/B response SHA mismatch")


def requires_adjudication(a: Mapping[str, Any], b: Mapping[str, Any]) -> bool:
    return bool(
        a["identity_status"] != b["identity_status"]
        or a["outer_package_status"] != b["outer_package_status"]
        or a["confidence"] == "low"
        or b["confidence"] == "low"
        or a["identity_status"] == "indeterminate"
        or b["identity_status"] == "indeterminate"
        or a["outer_package_status"] == "ambiguous"
        or b["outer_package_status"] == "ambiguous"
    )


def _as_bool(value: Any) -> bool:
    return value is True or str(value).lower() in {"1", "true", "yes"}


def validate_adjudication_rows(adjudication: list[Mapping[str, Any]], pass_a: list[Mapping[str, Any]], pass_b: list[Mapping[str, Any]], sample: list[Mapping[str, Any]], contract: Mapping[str, Any]) -> None:
    sample_by_asin = {row["parent_asin"]: row for row in sample}
    by_a = {row["parent_asin"]: row for row in pass_a}
    by_b = {row["parent_asin"]: row for row in pass_b}
    expected = {asin for asin in sample_by_asin if requires_adjudication(by_a[asin], by_b[asin])}
    by_adj = {row.get("parent_asin"): row for row in adjudication}
    if set(by_adj) != expected or len(adjudication) != len(expected):
        raise FormalVerificationError("adjudication universe does not match required triggers")
    expected_prompt = str(contract.get("review_prompt_sha256") or "")
    expected_model = contract.get("adjudicator_model_identifier")
    for asin in expected:
        row = by_adj[asin]
        _validate_sample_binding(
            {
                "parent_asin": row.get("parent_asin"),
                "response_sha256": row.get("response_sha256"),
                "review_prompt_sha256": row.get("adjudication_prompt_sha256"),
                "reviewer_model": row.get("adjudicator_model"),
            },
            sample_by_asin,
            {**contract, "review_model_identifier": expected_model, "review_prompt_sha256": expected_prompt},
            "adjudication",
        )
        if row.get("pass_a_identity_status") != by_a[asin]["identity_status"] or row.get("pass_a_outer_package_status") != by_a[asin]["outer_package_status"] or row.get("pass_a_confidence") != by_a[asin]["confidence"]:
            raise FormalVerificationError(f"adjudication Pass A trace mismatch: {asin}")
        if row.get("pass_b_identity_status") != by_b[asin]["identity_status"] or row.get("pass_b_outer_package_status") != by_b[asin]["outer_package_status"] or row.get("pass_b_confidence") != by_b[asin]["confidence"]:
            raise FormalVerificationError(f"adjudication Pass B trace mismatch: {asin}")
        if not _as_bool(row.get("requires_adjudication")) or row.get("adjudication_basis") != "model_assisted_adjudication":
            raise FormalVerificationError(f"invalid adjudication trigger/basis: {asin}")
        if not str(row.get("adjudicator_run_id") or ""):
            raise FormalVerificationError(f"missing adjudicator run id: {asin}")
        if expected_prompt and str(row.get("adjudication_prompt_sha256") or "").upper() != expected_prompt.upper():
            raise FormalVerificationError(f"adjudication prompt SHA mismatch: {asin}")
        final = {
            "identity_status": row.get("identity_status"),
            "outer_package_status": row.get("outer_package_status"),
            "confidence": row.get("confidence"),
        }
        if final["identity_status"] not in contract["identity_status_enum"] or final["outer_package_status"] not in contract["outer_package_status_enum"] or final["confidence"] not in contract["confidence_enum"]:
            raise FormalVerificationError(f"invalid adjudication result: {asin}")
        expected_exception = final["identity_status"] != "consistent" or final["outer_package_status"] != "outer_retail_package"
        if _as_bool(row.get("qa_exception_flag")) != expected_exception:
            raise FormalVerificationError(f"adjudication exception flag mismatch: {asin}")

def derive_primary_freeze_status(excluded: bool, qa_exception: bool) -> tuple[str, str]:
    if excluded:
        return "excluded_non_primary", "historical_first_image_fallback"
    if qa_exception:
        return "frozen_primary_with_qa_exception", "qa_exception"
    return "frozen_primary", "qa_pass"


def _min_confidence(a: str, b: str) -> str:
    return min((a, b), key=lambda value: CONFIDENCE_RANK[value])


def build_adjudicated_rows(pass_a: list[Mapping[str, Any]], pass_b: list[Mapping[str, Any]], adjudication: list[Mapping[str, Any]], contract: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
    contract = contract or {"identity_status_enum": ["consistent", "contradiction", "indeterminate"], "outer_package_status_enum": ["outer_retail_package", "inner_packaging_only", "product_content_only", "lifestyle_or_serving", "composite_or_collage", "ambiguous", "unavailable_or_corrupt"], "confidence_enum": ["high", "medium", "low"]}
    by_a = {row["parent_asin"]: row for row in pass_a}
    by_b = {row["parent_asin"]: row for row in pass_b}
    by_adj = {row["parent_asin"]: row for row in adjudication}
    result = []
    for asin in by_a:
        a, b = by_a[asin], by_b[asin]
        trigger = requires_adjudication(a, b)
        if trigger:
            if asin not in by_adj or by_adj[asin].get("adjudication_basis") in (None, "", "agreement"):
                raise FormalVerificationError(f"required adjudication missing: {asin}")
            final = by_adj[asin]
            basis = "model_assisted_adjudication"
            adjudicator_model = final.get("adjudicator_model", "")
            adjudicator_run_id = final.get("adjudicator_run_id", "")
            adjudication_prompt_sha256 = final.get("adjudication_prompt_sha256", "")
            confidence = final.get("confidence")
            identity = final.get("identity_status")
            outer = final.get("outer_package_status")
            reason = final.get("reason_code", "")
        else:
            identity, outer = a["identity_status"], a["outer_package_status"]
            confidence = _min_confidence(a["confidence"], b["confidence"])
            reason = a.get("reason_code", "")
            basis = "agreement"
            adjudicator_model = adjudicator_run_id = adjudication_prompt_sha256 = ""
        synthetic = {"identity_status": identity, "outer_package_status": outer, "confidence": confidence, "parent_asin": asin, "response_sha256": a["response_sha256"], "reviewer_model": "x", "reviewer_run_id": "x", "review_prompt_sha256": "x", "reviewed_at_utc": "x"}
        _validate_enum(synthetic, contract)
        exception = identity != "consistent" or outer != "outer_retail_package"
        result.append({
            "parent_asin": asin, "response_sha256": a["response_sha256"],
            "pass_a_identity_status": a["identity_status"], "pass_a_outer_package_status": a["outer_package_status"], "pass_a_confidence": a["confidence"],
            "pass_b_identity_status": b["identity_status"], "pass_b_outer_package_status": b["outer_package_status"], "pass_b_confidence": b["confidence"],
            "requires_adjudication": trigger, "identity_status": identity, "outer_package_status": outer, "confidence": confidence, "reason_code": reason,
            "adjudication_basis": basis, "adjudicator_model": adjudicator_model, "adjudicator_run_id": adjudicator_run_id, "adjudication_prompt_sha256": adjudication_prompt_sha256, "qa_exception_flag": exception,
        })
    return result


def wilson_upper_bound(successes: int, total: int, confidence_z: float = 1.6448536269514722) -> float:
    if total <= 0:
        return 1.0
    p = successes / total
    z2 = confidence_z * confidence_z
    denominator = 1 + z2 / total
    center = (p + z2 / (2 * total)) / denominator
    margin = confidence_z * math.sqrt((p * (1 - p) + z2 / (4 * total)) / total) / denominator
    return min(1.0, center + margin)


def _placeholder_signature(row: Mapping[str, Any], signature: Mapping[str, Any]) -> bool:
    return (
        str(row.get("response_sha256") or "").upper() == str(signature["response_sha256"]).upper()
        and str(row.get("decoded_format") or "").upper() == str(signature["decoded_format"]).upper()
        and int(row.get("width") or 0) == int(signature["width"])
        and int(row.get("height") or 0) == int(signature["height"])
        and int(row.get("response_byte_count") or 0) == int(signature["response_byte_count"])
    )


def validate_placeholder_outcomes(sample: list[Mapping[str, Any]], adjudicated: list[Mapping[str, Any]], contract: Mapping[str, Any]) -> None:
    sample_by_asin = {row["parent_asin"]: row for row in sample}
    adj_by_asin = {row["parent_asin"]: row for row in adjudicated}
    signatures = contract.get("decoded_placeholder_signatures", [])
    signature_asins: set[str] = set()
    for signature in signatures:
        matching = {row["parent_asin"] for row in sample if _placeholder_signature(row, signature)}
        expected_count = int(signature["expected_sampled_product_count"])
        if len(matching) != expected_count:
            raise FormalVerificationError(f"placeholder signature sample count mismatch: {signature['name']}")
        signature_asins.update(matching)
        for asin in matching:
            row = adj_by_asin.get(asin)
            if not row or row.get("identity_status") != "indeterminate" or row.get("outer_package_status") != "unavailable_or_corrupt" or row.get("confidence") != "low":
                raise FormalVerificationError(f"placeholder outcome mismatch: {asin}")
    unavailable = {row["parent_asin"] for row in adjudicated if row.get("outer_package_status") == "unavailable_or_corrupt"}
    if unavailable != signature_asins:
        raise FormalVerificationError("unavailable_or_corrupt rows are not exactly the frozen placeholder signatures")


def qa_sanity_status(contradiction: int, baseline_total: int, outer_exception: int, indeterminate: int, unavailable: int, duplicate_risk_contradiction: int, contract: Mapping[str, Any]) -> str:
    sanity = contract["qa_sanity"]
    checks = (
        wilson_upper_bound(contradiction, baseline_total) <= sanity["baseline_identity_contradiction_upper95_max"]
        and wilson_upper_bound(outer_exception, baseline_total) <= sanity["baseline_outer_exception_upper95_max"]
        and (indeterminate / baseline_total if baseline_total else 0.0) <= sanity["baseline_identity_indeterminate_rate_max"]
        and unavailable <= sanity["sampled_unavailable_or_corrupt_count_max"]
        and duplicate_risk_contradiction == 0
    )
    return "PASS" if checks else sanity["status_when_threshold_exceeded"]

def _formal_summary(manifest: list[Mapping[str, Any]], sample: list[Mapping[str, Any]], adjudicated: list[Mapping[str, Any]], contract: Mapping[str, Any], formal_output_sha256: Mapping[str, str] | None = None) -> dict[str, Any]:
    baseline = [row for row in adjudicated if next(sample_row for sample_row in sample if sample_row["parent_asin"] == row["parent_asin"])["baseline_random"]]
    contradiction = sum(row["identity_status"] == "contradiction" for row in baseline)
    indeterminate = sum(row["identity_status"] == "indeterminate" for row in baseline)
    outer_exception = sum(row["outer_package_status"] != "outer_retail_package" for row in baseline)
    duplicate_risk_contradiction = sum(row["identity_status"] == "contradiction" and next(sample_row for sample_row in sample if sample_row["parent_asin"] == row["parent_asin"])["exact_byte_duplicate_product"] for row in adjudicated)
    unavailable_count = sum(row["outer_package_status"] == "unavailable_or_corrupt" for row in adjudicated)
    qa_sanity = qa_sanity_status(contradiction, len(baseline), outer_exception, indeterminate, unavailable_count, duplicate_risk_contradiction, contract)
    return {
        "p7_c_pipeline_integrity_status": "PASS", "p7_c_qa_sanity_status": qa_sanity, "primary_manifest_status": "FROZEN", "contract_version": contract["contract_version"], "formal_output_sha256": dict(formal_output_sha256 or {}),
        "product_universe_count": len(manifest), "primary_available_count": sum(row["eligibility_status"] == "primary_declared_main" for row in manifest), "excluded_non_primary_count": sum(row["primary_freeze_status"] == "excluded_non_primary" for row in manifest),
        "sample_total": len(sample), "baseline_random_count": sum(row["baseline_random"] for row in sample), "duplicate_product_count": sum(row["exact_byte_duplicate_product"] for row in sample), "gif_product_count": sum(row["gif_asset"] for row in sample),
        "low_dimension_unique_asset_count": 50, "low_byte_unique_asset_count": 50,
        "review_a_complete_count": len(sample), "review_b_complete_count": len(sample), "joint_agreement_count": sum(not row["requires_adjudication"] for row in adjudicated), "requires_adjudication_count": sum(row["requires_adjudication"] for row in adjudicated), "adjudicated_count": sum(row["requires_adjudication"] for row in adjudicated),
        "final_identity_distribution": dict(sorted(Counter(row["identity_status"] for row in adjudicated).items())), "final_outer_package_distribution": dict(sorted(Counter(row["outer_package_status"] for row in adjudicated).items())),
        "qa_exception_count": sum(row["qa_exception_flag"] for row in adjudicated), "qa_exception_rate_in_sample": sum(row["qa_exception_flag"] for row in adjudicated) / len(adjudicated) if adjudicated else 0.0,
        "baseline_identity_contradiction_count": contradiction, "baseline_identity_contradiction_rate": contradiction / len(baseline) if baseline else 0.0, "baseline_identity_contradiction_upper95": wilson_upper_bound(contradiction, len(baseline)),
        "baseline_identity_indeterminate_count": indeterminate, "baseline_identity_indeterminate_rate": indeterminate / len(baseline) if baseline else 0.0,
        "baseline_outer_exception_count": outer_exception, "baseline_outer_exception_rate": outer_exception / len(baseline) if baseline else 0.0, "baseline_outer_exception_upper95": wilson_upper_bound(outer_exception, len(baseline)),
        "duplicate_risk_identity_contradiction_count": duplicate_risk_contradiction, "sampled_unavailable_or_corrupt_count": unavailable_count, "frozen_primary_count": sum(row["primary_freeze_status"] == "frozen_primary" for row in manifest), "frozen_primary_with_qa_exception_count": sum(row["primary_freeze_status"] == "frozen_primary_with_qa_exception" for row in manifest), "unavailable_or_corrupt_count": sum(row["outer_package_status"] == "unavailable_or_corrupt" for row in adjudicated),
        **contract["safety"],
    }


def _manifest_from_results(products: list[Mapping[str, Any]], sample: list[Mapping[str, Any]], adjudicated: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    sample_by = {row["parent_asin"]: row for row in sample}
    adj_by = {row["parent_asin"]: row for row in adjudicated}
    output = []
    for product in sorted(products, key=lambda row: int(row["input_order"])):
        asin = product["parent_asin"]
        excluded = product["eligibility_status"] != "primary_declared_main"
        sampled = asin in sample_by
        if excluded:
            freeze, reason = derive_primary_freeze_status(True, False)
            qa = {"qa_identity_status": "not_reviewed", "qa_outer_package_status": "not_reviewed", "qa_confidence": "not_reviewed", "qa_adjudication_basis": "not_reviewed", "qa_exception_flag": False, "qa_strata": ""}
        elif not sampled:
            freeze, reason = derive_primary_freeze_status(False, False)
            qa = {"qa_identity_status": "not_reviewed", "qa_outer_package_status": "not_reviewed", "qa_confidence": "not_reviewed", "qa_adjudication_basis": "not_reviewed", "qa_exception_flag": False, "qa_strata": ""}
        else:
            final = adj_by[asin]
            freeze, reason = derive_primary_freeze_status(False, final["qa_exception_flag"])
            qa = {"qa_identity_status": final["identity_status"], "qa_outer_package_status": final["outer_package_status"], "qa_confidence": final["confidence"], "qa_adjudication_basis": final["adjudication_basis"], "qa_exception_flag": final["qa_exception_flag"], "qa_strata": sample_by[asin]["qa_strata"]}
        output.append({
            "parent_asin": asin, "input_order": product["input_order"], "source_record_sha256": product["source_record_sha256"], "image_role": product["image_role"], "temporal_alignment_status": product["temporal_alignment_status"], "eligibility_status": product["eligibility_status"], "response_sha256": product.get("response_sha256", ""), "asset_path": product.get("asset_path", ""), "decoded_format": product.get("decoded_format", ""), "width": product.get("width", 0), "height": product.get("height", 0), "n_frames": product.get("n_frames", 0), "response_byte_count": product.get("response_byte_count", 0), "exact_duplicate_asset_id": product.get("exact_duplicate_asset_id", ""), "exact_duplicate_asset_group_size": product.get("exact_duplicate_asset_group_size", 0), "qa_sampled": sampled, **qa, "primary_freeze_status": freeze, "primary_freeze_reason": reason,
        })
    return output


def _output_sha_map(paths: Mapping[str, Path]) -> dict[str, str]:
    return {key: sha256_file(paths[key]) for key in ("qa_sample", "review_pass_a", "review_pass_b", "adjudicated", "primary_manifest", "summary")}


def prepare(contract: Mapping[str, Any]) -> dict[str, Any]:
    manifest, unique, metadata = validate_upstream(contract)
    sample = attach_identity_metadata(build_deterministic_sample(manifest, unique), metadata)
    paths = _paths(contract)
    if any(paths[key].exists() for key in ("qa_sample", "review_pass_a", "review_pass_b", "adjudicated", "primary_manifest", "summary", "provenance")):
        raise FormalVerificationError("P7-C formal output already exists")
    _write_csv(paths["qa_sample"], SAMPLE_FIELDS, sample)
    return {"sample_total": len(sample), "baseline_random_count": sum(row["baseline_random"] for row in sample), "duplicate_product_count": sum(row["exact_byte_duplicate_product"] for row in sample), "gif_product_count": sum(row["gif_asset"] for row in sample), "low_dimension_product_count": sum(row["low_min_dimension_asset"] for row in sample), "low_byte_product_count": sum(row["low_byte_size_asset"] for row in sample), "sample_sha256": sha256_file(paths["qa_sample"])}


def _normalize_review_input(path: Path) -> list[dict[str, Any]]:
    return _parse_csv(path)


def import_review(contract: Mapping[str, Any], pass_name: str, input_path: Path) -> dict[str, Any]:
    paths = _paths(contract)
    sample = _parse_csv(paths["qa_sample"])
    rows = _normalize_review_input(input_path)
    target = paths["review_pass_a"] if pass_name == "a" else paths["review_pass_b"]
    if pass_name not in {"a", "b"}:
        raise FormalVerificationError("pass must be a or b")
    validate_review_pass(rows, sample, contract, pass_name.upper())
    _write_csv(target, REVIEW_FIELDS, rows)
    return {"pass": pass_name, "rows": len(rows), "sha256": sha256_file(target)}


def finalize(contract: Mapping[str, Any], adjudication_path: Path | None = None) -> dict[str, Any]:
    paths = _paths(contract)
    manifest, unique, metadata = validate_upstream(contract)
    sample = _parse_csv(paths["qa_sample"])
    pass_a = _parse_csv(paths["review_pass_a"])
    pass_b = _parse_csv(paths["review_pass_b"])
    validate_review_rows(pass_a, pass_b, [row["parent_asin"] for row in sample], contract, sample_rows=sample)
    adjudication = _parse_csv(adjudication_path) if adjudication_path else []
    validate_adjudication_rows(adjudication, pass_a, pass_b, sample, contract)
    adjudicated = build_adjudicated_rows(pass_a, pass_b, adjudication, contract)
    validate_placeholder_outcomes(sample, adjudicated, contract)
    primary_manifest = _manifest_from_results(manifest, sample, adjudicated)
    _write_csv(paths["review_pass_a"], REVIEW_FIELDS, pass_a)
    _write_csv(paths["review_pass_b"], REVIEW_FIELDS, pass_b)
    _write_csv(paths["adjudicated"], ADJ_FIELDS, adjudicated)
    _write_csv(paths["primary_manifest"], MANIFEST_FIELDS, primary_manifest)
    summary_output_sha256 = {name: sha256_file(paths[key]) for name, key in (("01_qa_sample.csv", "qa_sample"), ("02_review_pass_a.csv", "review_pass_a"), ("03_review_pass_b.csv", "review_pass_b"), ("04_adjudicated_qa.csv", "adjudicated"), ("05_primary_manifest.csv", "primary_manifest"))}
    summary = _formal_summary(primary_manifest, sample, adjudicated, contract, summary_output_sha256)
    _write_json(paths["summary"], summary)
    prompt = load_prompt()
    formal_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    review_a_trace = {
        "count": len(pass_a),
        "run_ids": sorted({str(row["reviewer_run_id"]) for row in pass_a}),
        "model_identifiers": sorted({str(row["reviewer_model"]) for row in pass_a}),
        "prompt_sha256s": sorted({str(row["review_prompt_sha256"]).upper() for row in pass_a}),
    }
    review_b_trace = {
        "count": len(pass_b),
        "run_ids": sorted({str(row["reviewer_run_id"]) for row in pass_b}),
        "model_identifiers": sorted({str(row["reviewer_model"]) for row in pass_b}),
        "prompt_sha256s": sorted({str(row["review_prompt_sha256"]).upper() for row in pass_b}),
    }
    adjudication_trace = {
        "count": summary["adjudicated_count"],
        "run_ids": sorted({str(row["adjudicator_run_id"]) for row in adjudication}),
        "model_identifiers": sorted({str(row["adjudicator_model"]) for row in adjudication}),
        "prompt_sha256s": sorted({str(row["adjudication_prompt_sha256"]).upper() for row in adjudication}),
    }
    provenance = {
        "contract_version": contract["contract_version"], "formal_run_git_commit": formal_commit, "producer_script_path": "scripts/images/p7_c_primary_manifest_qa.py", "producer_script_git_blob_sha256": git_blob_sha256(formal_commit, "scripts/images/p7_c_primary_manifest_qa.py"), "contract_path": "config/image_assets/p7_c_qa_contract.json", "contract_git_blob_sha256": git_blob_sha256(formal_commit, "config/image_assets/p7_c_qa_contract.json"), "review_prompt_path": "config/image_assets/p7_c_review_prompt.json", "review_prompt_sha256": git_blob_sha256(formal_commit, PROMPT_REPO_PATH), "upstream_p7_a": contract["upstream_p7_a"], "upstream_p7_b": contract["upstream_p7_b"], "sample_algorithm_version": contract["sample_algorithm"]["version"], "sample_sha256": sha256_file(paths["qa_sample"]), "review_pass_a": {"provider": prompt["provider"], "model_identifier": prompt["model_identifier"], "run_id": pass_a[0]["reviewer_run_id"] if pass_a else "", "prompt_sha256": pass_a[0]["review_prompt_sha256"] if pass_a else ""}, "review_pass_b": {"provider": prompt["provider"], "model_identifier": prompt["model_identifier"], "run_id": pass_b[0]["reviewer_run_id"] if pass_b else "", "prompt_sha256": pass_b[0]["review_prompt_sha256"] if pass_b else ""}, "adjudication": {"provider": prompt["provider"], "model_identifier": prompt["model_identifier"], "count": summary["adjudicated_count"]}, "review_pass_a_trace": review_a_trace, "review_pass_b_trace": review_b_trace, "adjudication_trace": adjudication_trace, "execution_started_utc": utc_now(), "formal_output_sha256": {}, "safety": contract["safety"],
    }
    _write_json(paths["provenance"], provenance)
    provenance["formal_output_sha256"] = {name: sha256_file(paths[key]) for name, key in (("01_qa_sample.csv", "qa_sample"), ("02_review_pass_a.csv", "review_pass_a"), ("03_review_pass_b.csv", "review_pass_b"), ("04_adjudicated_qa.csv", "adjudicated"), ("05_primary_manifest.csv", "primary_manifest"), ("06_p7_c_summary.json", "summary"))}
    _write_json(paths["provenance"], provenance)
    return {"summary": summary, "formal_output_sha256": provenance["formal_output_sha256"], "provenance_sha256": sha256_file(paths["provenance"])}


def verify_existing(contract: Mapping[str, Any]) -> dict[str, Any]:
    paths = _paths(contract)
    required = ("qa_sample", "review_pass_a", "review_pass_b", "adjudicated", "primary_manifest", "summary", "provenance")
    if any(not paths[key].exists() for key in required):
        raise FormalVerificationError("missing P7-C formal output")
    manifest, unique, metadata = validate_upstream(contract)
    sample = _parse_csv(paths["qa_sample"])
    expected_sample = attach_identity_metadata(build_deterministic_sample(manifest, unique), metadata)
    if sample != expected_sample:
        raise FormalVerificationError("deterministic sample mismatch")
    pass_a, pass_b = _parse_csv(paths["review_pass_a"]), _parse_csv(paths["review_pass_b"])
    validate_review_rows(pass_a, pass_b, [row["parent_asin"] for row in sample], contract, sample_rows=sample)
    adjudicated = _parse_csv(paths["adjudicated"])
    validate_adjudication_rows([row for row in adjudicated if _as_bool(row.get("requires_adjudication"))], pass_a, pass_b, sample, contract)
    validate_placeholder_outcomes(sample, adjudicated, contract)
    expected_adjudicated = build_adjudicated_rows(pass_a, pass_b, adjudicated, contract)
    if adjudicated != expected_adjudicated:
        raise FormalVerificationError("adjudication reconstruction mismatch")
    primary_manifest = _parse_csv(paths["primary_manifest"])
    expected_manifest = _manifest_from_results(manifest, sample, adjudicated)
    if primary_manifest != expected_manifest or len(primary_manifest) != contract["product_universe_count"] or len({row["parent_asin"] for row in primary_manifest}) != contract["product_universe_count"]:
        raise FormalVerificationError("Primary Manifest reconstruction mismatch")
    summary = json.loads(paths["summary"].read_text(encoding="utf-8"))
    expected_summary_sha256 = {name: sha256_file(paths[key]) for name, key in (("01_qa_sample.csv", "qa_sample"), ("02_review_pass_a.csv", "review_pass_a"), ("03_review_pass_b.csv", "review_pass_b"), ("04_adjudicated_qa.csv", "adjudicated"), ("05_primary_manifest.csv", "primary_manifest"))}
    expected_summary = _formal_summary(primary_manifest, sample, adjudicated, contract, expected_summary_sha256)
    if summary != expected_summary:
        raise FormalVerificationError("P7-C summary reconstruction mismatch")
    provenance = json.loads(paths["provenance"].read_text(encoding="utf-8"))
    prompt_provenance_checks = validate_prompt_provenance(provenance, contract)
    expected_trace = {
        "review_pass_a_trace": {
            "count": len(pass_a),
            "run_ids": sorted({str(row["reviewer_run_id"]) for row in pass_a}),
            "model_identifiers": sorted({str(row["reviewer_model"]) for row in pass_a}),
            "prompt_sha256s": sorted({str(row["review_prompt_sha256"]).upper() for row in pass_a}),
        },
        "review_pass_b_trace": {
            "count": len(pass_b),
            "run_ids": sorted({str(row["reviewer_run_id"]) for row in pass_b}),
            "model_identifiers": sorted({str(row["reviewer_model"]) for row in pass_b}),
            "prompt_sha256s": sorted({str(row["review_prompt_sha256"]).upper() for row in pass_b}),
        },
        "adjudication_trace": {
            "count": sum(row["requires_adjudication"] for row in adjudicated),
            "run_ids": sorted({str(row["adjudicator_run_id"]) for row in adjudicated if row["requires_adjudication"]}),
            "model_identifiers": sorted({str(row["adjudicator_model"]) for row in adjudicated if row["requires_adjudication"]}),
            "prompt_sha256s": sorted({str(row["adjudication_prompt_sha256"]).upper() for row in adjudicated if row["requires_adjudication"]}),
        },
    }
    checks = {
        "contract_version": provenance.get("contract_version", contract["contract_version"]) == contract["contract_version"],
        "p7_a_inventory_sha": sha256_file(ROOT / contract["upstream_p7_a"]["inventory_path"]) == contract["upstream_p7_a"]["inventory_sha256"],
        "p7_b_manifest_sha": sha256_file(ROOT / contract["upstream_p7_b"]["manifest_path"]) == contract["upstream_p7_b"]["manifest_sha256"],
        "sample_sha": provenance.get("sample_sha256") == sha256_file(paths["qa_sample"]),
        "producer_script_git_blob_sha256_match": git_blob_sha256(provenance["formal_run_git_commit"], provenance["producer_script_path"]) == provenance.get("producer_script_git_blob_sha256"),
        "contract_git_blob_sha256_match": git_blob_sha256(provenance["formal_run_git_commit"], provenance["contract_path"]) == provenance.get("contract_git_blob_sha256"),
        "current_head_contract_git_blob_sha256_match": git_blob_sha256("HEAD", provenance["contract_path"]) == provenance.get("contract_git_blob_sha256"),
        "provenance_does_not_record_own_sha": "07_p7_c_provenance.json" not in provenance.get("formal_output_sha256", {}),
        **prompt_provenance_checks,
        "review_pass_a_trace_match": provenance.get("review_pass_a_trace") == expected_trace["review_pass_a_trace"],
        "review_pass_b_trace_match": provenance.get("review_pass_b_trace") == expected_trace["review_pass_b_trace"],
        "adjudication_trace_match": provenance.get("adjudication_trace") == expected_trace["adjudication_trace"],
    }
    for name, key in (("01_qa_sample.csv", "qa_sample"), ("02_review_pass_a.csv", "review_pass_a"), ("03_review_pass_b.csv", "review_pass_b"), ("04_adjudicated_qa.csv", "adjudicated"), ("05_primary_manifest.csv", "primary_manifest"), ("06_p7_c_summary.json", "summary")):
        checks[f"{key}_sha"] = provenance.get("formal_output_sha256", {}).get(name) == sha256_file(paths[key])
    if not all(checks.values()):
        raise FormalVerificationError(f"P7-C provenance checks failed: {checks}")
    temporary = [path for path in paths["output_dir"].rglob("*") if path.is_file() and (".tmp" in path.name or path.name.endswith(".partial"))]
    if temporary:
        raise FormalVerificationError("temporary P7-C formal files remain")
    return {"verification": "PASS", "sample_total": len(sample), "primary_manifest_rows": len(primary_manifest), "provenance_checks": checks}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "import-pass", "finalize"), nargs="?")
    parser.add_argument("--pass-name", choices=("a", "b"))
    parser.add_argument("--input", type=Path)
    parser.add_argument("--adjudication", type=Path)
    parser.add_argument("--verify-existing", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    contract = load_contract()
    if args.verify_existing:
        print(json.dumps(verify_existing(contract), ensure_ascii=False, indent=2))
        return 0
    if args.command == "prepare":
        print(json.dumps(prepare(contract), ensure_ascii=False, indent=2))
        return 0
    if args.command == "import-pass":
        if not args.pass_name or not args.input:
            raise FormalVerificationError("import-pass requires --pass-name and --input")
        print(json.dumps(import_review(contract, args.pass_name, args.input), ensure_ascii=False, indent=2))
        return 0
    if args.command == "finalize":
        print(json.dumps(finalize(contract, args.adjudication), ensure_ascii=False, indent=2))
        return 0
    raise FormalVerificationError("choose prepare, import-pass, finalize, or --verify-existing")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except FormalVerificationError as exc:
        print(f"P7-C ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
