#!/usr/bin/env python3
"""P7-D label-blind sensitivity layer and final image-stage freeze."""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
from html.parser import HTMLParser
import os
import re
import subprocess
import sys
import time
import uuid
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import requests

# Keep the formal entry point runnable both as a module and as a file.
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.images import acquire_p7_primary_assets as p7b
from scripts.images import audit_p7_source_inventory as p7a


ROOT = Path(__file__).resolve().parents[2]
CONTRACT_PATH = ROOT / "config" / "image_assets" / "p7_d_final_freeze_contract.json"
P7C_SCRIPT = ROOT / "scripts" / "images" / "p7_c_primary_manifest_qa.py"
P7A_SCRIPT = ROOT / "scripts" / "images" / "audit_p7_source_inventory.py"
P7B_SCRIPT = ROOT / "scripts" / "images" / "acquire_p7_primary_assets.py"
P7C_PROMPT_REPO_PATH = "config/image_assets/p7_c_review_prompt.json"
P7C_CONTRACT_PATH = "config/image_assets/p7_c_qa_contract.json"
PRODUCT_INPUT_PATH = ROOT / "data" / "processed" / "review_matching_5180" / "01_valid_products.csv"
RAW_METADATA_PATH = ROOT / "data" / "meta_Grocery_and_Gourmet_Food.jsonl" / "meta_Grocery_and_Gourmet_Food.jsonl"
PRIMARY_ASSET_ROOT = ROOT / "data" / "images" / "retail_outer_package_p7_5180"


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
    path = Path(repo_path).as_posix()
    data = subprocess.check_output(["git", "cat-file", "blob", f"{commit}:{path}"], cwd=ROOT)
    return sha256_bytes(data)


def _as_bool(value: Any) -> bool:
    return value is True or str(value).lower() in {"1", "true", "yes"}


def _as_int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _csv_text(value: Any) -> str:
    return "" if value is None else str(value)


def _resolve_repo_path(value: str | Path) -> Path:
    normalized = str(value).strip().replace("\\", "/")
    path = Path(normalized)
    return path if path.is_absolute() else ROOT / path


def _repo_relative_path(value: str | Path) -> str:
    resolved = _resolve_repo_path(value).resolve()
    try:
        return resolved.relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


def load_contract(path: Path = CONTRACT_PATH) -> dict[str, Any]:
    contract = json.loads(path.read_text(encoding="utf-8"))
    required = (
        "contract_version", "upstream_main_commit", "p7_c_formal_output_sha256",
        "product_universe_count", "primary_available_count", "excluded_non_primary_count",
        "primary_manifest_mutable", "sensitivity_is_primary_replacement",
        "label_blind_before_sensitivity_freeze", "g_final_threshold_selected",
        "formal_paths", "source_tiers", "candidate_review_prompt_sha256",
        "placeholder_signatures", "label_firewall", "safety",
    )
    missing = [key for key in required if key not in contract]
    if missing or contract.get("contract_version") != "p7_d_v1.0":
        raise FormalVerificationError(f"invalid P7-D contract: missing={missing}")
    if contract["primary_manifest_mutable"] or contract["sensitivity_is_primary_replacement"]:
        raise FormalVerificationError("P7-D primary replacement invariant is not frozen")
    if not contract["label_blind_before_sensitivity_freeze"]:
        raise FormalVerificationError("P7-D label-blind firewall is disabled")
    if contract["g_final_threshold_selected"]:
        raise FormalVerificationError("G final threshold must remain unselected")
    if git_blob_sha256("HEAD", P7C_PROMPT_REPO_PATH) != contract["candidate_review_prompt_sha256"]:
        raise FormalVerificationError("P7-D frozen review prompt Git blob mismatch")
    return contract


def _paths(contract: Mapping[str, Any]) -> dict[str, Path]:
    return {key: ROOT / value for key, value in contract["formal_paths"].items()}


def _write_csv(path: Path, fields: list[str] | tuple[str, ...], rows: list[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temp.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temp, path)
    finally:
        if temp.exists():
            temp.unlink()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="")
        os.replace(temp, path)
    finally:
        if temp.exists():
            temp.unlink()


def _read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = [dict(row) for row in csv.DictReader(handle)]
    return rows


def _run_json_gate(script: Path, args: list[str]) -> dict[str, Any]:
    result = subprocess.run([sys.executable, str(script), *args], cwd=ROOT, capture_output=True, text=True)
    if result.returncode != 0:
        raise FormalVerificationError(f"upstream verification failed: {script.name}\n{result.stdout}\n{result.stderr}")
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise FormalVerificationError(f"upstream verifier returned non-JSON: {script.name}") from exc
    if payload.get("verification") != "PASS":
        raise FormalVerificationError(f"upstream verifier did not PASS: {script.name}")
    return payload


def verify_upstream(contract: Mapping[str, Any]) -> dict[str, Any]:
    p7a_result = _run_json_gate(P7A_SCRIPT, [
        "--input", "data/processed/review_matching_5180/01_valid_products.csv",
        "--raw-metadata", "data/meta_Grocery_and_Gourmet_Food.jsonl/meta_Grocery_and_Gourmet_Food.jsonl",
        "--output-dir", "data/processed/retail_outer_package_images_p7_5180/p7_a_source_inventory",
        "--verify-existing",
    ])
    p7b_result = _run_json_gate(P7B_SCRIPT, ["--verify-existing"])
    p7c_result = _run_json_gate(P7C_SCRIPT, ["--verify-existing"])
    for name, expected in contract["p7_c_formal_output_sha256"].items():
        path_key = {
            "01_qa_sample.csv": "qa_sample",
            "02_review_pass_a.csv": "review_pass_a",
            "03_review_pass_b.csv": "review_pass_b",
            "04_adjudicated_qa.csv": "adjudicated",
            "05_primary_manifest.csv": "primary_manifest",
            "06_p7_c_summary.json": "summary",
            "07_p7_c_provenance.json": "provenance",
        }[name]
        p7c_path = ROOT / {
            "qa_sample": "data/processed/retail_outer_package_images_p7_5180/p7_c_primary_manifest/01_qa_sample.csv",
            "review_pass_a": "data/processed/retail_outer_package_images_p7_5180/p7_c_primary_manifest/02_review_pass_a.csv",
            "review_pass_b": "data/processed/retail_outer_package_images_p7_5180/p7_c_primary_manifest/03_review_pass_b.csv",
            "adjudicated": "data/processed/retail_outer_package_images_p7_5180/p7_c_primary_manifest/04_adjudicated_qa.csv",
            "primary_manifest": "data/processed/retail_outer_package_images_p7_5180/p7_c_primary_manifest/05_primary_manifest.csv",
            "summary": "data/processed/retail_outer_package_images_p7_5180/p7_c_primary_manifest/06_p7_c_summary.json",
            "provenance": "data/processed/retail_outer_package_images_p7_5180/p7_c_primary_manifest/07_p7_c_provenance.json",
        }[path_key]
        if sha256_file(p7c_path) != expected:
            raise FormalVerificationError(f"frozen P7-C SHA mismatch: {name}")
    return {"p7_a": p7a_result, "p7_b": p7b_result, "p7_c": p7c_result}


def build_sensitivity_queue(rows: list[Mapping[str, Any]]) -> set[str]:
    return {
        str(row["parent_asin"])
        for row in rows
        if _as_bool(row.get("c_reviewed_qa_exception"))
        or _as_bool(row.get("excluded_non_primary"))
        or bool(str(row.get("global_placeholder_signature") or ""))
    }


def build_exception_inventory(
    primary_rows: list[Mapping[str, Any]],
    global_placeholder_asins: set[str],
    placeholder_by_sha: Mapping[str, str],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for source in sorted(primary_rows, key=lambda row: _as_int(row.get("input_order"))):
        asin = str(source["parent_asin"])
        response = str(source.get("response_sha256") or "").upper()
        signature = placeholder_by_sha.get(response, "") if asin in global_placeholder_asins or response in placeholder_by_sha else ""
        sampled = _as_bool(source.get("qa_sampled"))
        exception = str(source.get("primary_freeze_status") or "") == "frozen_primary_with_qa_exception"
        excluded = str(source.get("primary_freeze_status") or "") == "excluded_non_primary"
        row = {
            "parent_asin": asin,
            "input_order": _as_int(source.get("input_order")),
            "primary_freeze_status": source.get("primary_freeze_status", ""),
            "primary_response_sha256": response,
            "primary_asset_path": source.get("asset_path", ""),
            "primary_decoded_format": source.get("decoded_format", ""),
            "primary_width": _as_int(source.get("width")),
            "primary_height": _as_int(source.get("height")),
            "primary_n_frames": _as_int(source.get("n_frames")),
            "primary_response_byte_count": _as_int(source.get("response_byte_count")),
            "primary_qa_sampled": sampled,
            "primary_qa_status": source.get("qa_identity_status", "not_reviewed") if sampled else "not_reviewed",
            "primary_qa_identity_status": source.get("qa_identity_status", "not_reviewed"),
            "primary_qa_outer_package_status": source.get("qa_outer_package_status", "not_reviewed"),
            "primary_qa_exception_flag": _as_bool(source.get("qa_exception_flag")),
            "c_reviewed_qa_exception": exception,
            "global_placeholder_signature": signature,
            "excluded_non_primary": excluded,
            "sensitivity_queue": False,
        }
        output.append(row)
    queue = build_sensitivity_queue(output)
    for row in output:
        row["sensitivity_queue"] = row["parent_asin"] in queue
    return output


def _candidate_key(row: Mapping[str, Any]) -> str:
    return str(row.get("candidate_id") or f"{row.get('parent_asin','')}|{row.get('response_sha256','')}")


def _is_s3_ui_placeholder_candidate(candidate: Mapping[str, Any]) -> bool:
    if str(candidate.get("source_tier") or "") != "current_exact_parent_asin":
        return False
    if str(candidate.get("source_url_field") or "") != "exact_parent_page_main_image":
        return False
    requested_url = str(candidate.get("requested_url") or "")
    if "01RmK+J4pJL" in requested_url:
        return True
    return _as_int(candidate.get("width")) <= 80 and _as_int(candidate.get("height")) <= 80 and _as_int(candidate.get("response_byte_count")) <= 2048


def candidate_is_structurally_eligible(
    candidate: Mapping[str, Any],
    primary_by_asin: Mapping[str, Mapping[str, Any]],
    placeholder_shas: set[str],
) -> bool:
    if candidate.get("final_asset_status") != "available" or candidate.get("decode_status") != "success":
        return False
    digest = str(candidate.get("response_sha256") or "").upper()
    if not digest or digest in placeholder_shas or _is_s3_ui_placeholder_candidate(candidate):
        return False
    primary = primary_by_asin.get(str(candidate.get("parent_asin") or ""), {})
    if digest == str(primary.get("primary_response_sha256") or primary.get("response_sha256") or "").upper():
        return False
    return candidate.get("candidate_structural_status", "eligible") == "eligible"


def candidate_qualified(candidate: Mapping[str, Any]) -> bool:
    return bool(
        candidate.get("candidate_structural_status") == "eligible"
        and candidate.get("identity_status") == "consistent"
        and candidate.get("outer_package_status") == "outer_retail_package"
    )


def products_requiring_s3_after_s1_s2_adjudication(
    queue: set[str],
    adjudicated: list[Mapping[str, Any]],
) -> set[str]:
    qualified_s1_s2 = {
        str(row.get("parent_asin") or "")
        for row in adjudicated
        if str(row.get("source_tier") or "") != "current_exact_parent_asin"
        and candidate_qualified(row)
    }
    return {str(asin) for asin in queue} - qualified_s1_s2


def validate_s3_escalation_set(
    queue: set[str],
    adjudicated: list[Mapping[str, Any]],
    candidates: list[Mapping[str, Any]],
) -> None:
    required = products_requiring_s3_after_s1_s2_adjudication(queue, adjudicated)
    represented = {
        str(row.get("parent_asin") or "")
        for row in candidates
        if str(row.get("source_tier") or "") == "current_exact_parent_asin"
    }
    if required != represented:
        missing = sorted(required - represented)
        extra = sorted(represented - required)
        raise FormalVerificationError(
            f"P7-D S3 escalation set mismatch: missing={missing} extra={extra}"
        )


def select_sensitivity_candidate(candidates: list[Mapping[str, Any]], contract: Mapping[str, Any]) -> dict[str, Any] | None:
    qualified = [dict(row) for row in candidates if candidate_qualified(row)]
    if not qualified:
        return None
    def key(row: Mapping[str, Any]) -> tuple[int, int, str, str]:
        tier = contract["source_tiers"].get(str(row.get("source_tier")), {})
        return (_as_int(tier.get("rank")), _as_int(row.get("source_image_index")), str(row.get("response_sha256") or "").upper(), _candidate_key(row))
    return min(qualified, key=key)


def build_sensitivity_status(queue: bool, selected: Mapping[str, Any] | None) -> str:
    if not queue:
        return "not_required"
    return "available" if selected else "unresolved"


def preserve_primary_fields(primary: Mapping[str, Any], sensitivity: Mapping[str, Any]) -> dict[str, Any]:
    if _as_bool(sensitivity.get("primary_asset_substituted")):
        raise FormalVerificationError("primary asset substitution is forbidden")
    result = {
        "parent_asin": primary.get("parent_asin", ""),
        "input_order": _as_int(primary.get("input_order")),
        "primary_freeze_status": primary.get("primary_freeze_status", ""),
        "primary_response_sha256": primary.get("primary_response_sha256", primary.get("response_sha256", "")),
        "primary_asset_path": primary.get("primary_asset_path", primary.get("asset_path", "")),
        "primary_decoded_format": primary.get("primary_decoded_format", primary.get("decoded_format", "")),
        "primary_width": _as_int(primary.get("primary_width", primary.get("width"))),
        "primary_height": _as_int(primary.get("primary_height", primary.get("height"))),
        "primary_n_frames": _as_int(primary.get("primary_n_frames", primary.get("n_frames"))),
        "primary_response_byte_count": _as_int(primary.get("primary_response_byte_count", primary.get("response_byte_count"))),
        "primary_qa_sampled": _as_bool(primary.get("primary_qa_sampled", primary.get("qa_sampled"))),
        "primary_qa_identity_status": primary.get("primary_qa_identity_status", primary.get("qa_identity_status", "not_reviewed")),
        "primary_qa_outer_package_status": primary.get("primary_qa_outer_package_status", primary.get("qa_outer_package_status", "not_reviewed")),
        "primary_qa_exception_flag": _as_bool(primary.get("primary_qa_exception_flag", primary.get("qa_exception_flag"))),
        "c_reviewed_qa_exception": _as_bool(primary.get("c_reviewed_qa_exception")),
        "global_placeholder_signature": primary.get("global_placeholder_signature", ""),
        "excluded_non_primary": _as_bool(primary.get("excluded_non_primary")),
        "sensitivity_queue": _as_bool(primary.get("sensitivity_queue")),
        "sensitivity_status": sensitivity.get("sensitivity_status", "unresolved"),
        "sensitivity_source_tier": sensitivity.get("source_tier", ""),
        "sensitivity_temporal_alignment_status": sensitivity.get("temporal_alignment_status", ""),
        "sensitivity_response_sha256": sensitivity.get("response_sha256", ""),
        "sensitivity_asset_path": sensitivity.get("asset_path", ""),
        "sensitivity_decoded_format": sensitivity.get("decoded_format", ""),
        "sensitivity_width": _as_int(sensitivity.get("width")),
        "sensitivity_height": _as_int(sensitivity.get("height")),
        "sensitivity_n_frames": _as_int(sensitivity.get("n_frames")),
        "sensitivity_response_byte_count": _as_int(sensitivity.get("response_byte_count")),
        "sensitivity_identity_status": sensitivity.get("identity_status", ""),
        "sensitivity_outer_package_status": sensitivity.get("outer_package_status", ""),
        "sensitivity_confidence": sensitivity.get("confidence", ""),
        "sensitivity_candidate_count": _as_int(sensitivity.get("candidate_count")),
        "sensitivity_qualified_candidate_count": _as_int(sensitivity.get("qualified_candidate_count")),
        "sensitivity_selection_rank": _as_int(sensitivity.get("selection_rank")),
        "primary_asset_substituted": False,
        "final_image_stage_status": sensitivity.get("final_image_stage_status", "frozen"),
    }
    return result


def _compare_value(kind: str, left: Any, right: Any) -> bool:
    if kind == "int":
        return _as_int(left) == _as_int(right)
    if kind == "bool":
        return _as_bool(left) == _as_bool(right)
    if kind == "sha":
        return str(left or "").upper() == str(right or "").upper()
    return str(left or "") == str(right or "")


def validate_primary_fields_match_p7c(
    manifest: list[Mapping[str, Any]],
    p7c_manifest: list[Mapping[str, Any]],
) -> None:
    p7c_by_asin = {str(row.get("parent_asin") or ""): row for row in p7c_manifest}
    if len(p7c_by_asin) != len(p7c_manifest) or len(manifest) != len(p7c_manifest):
        raise FormalVerificationError("P7-C primary manifest universe mismatch")
    field_pairs = (
        ("input_order", "input_order", "int"),
        ("primary_freeze_status", "primary_freeze_status", "str"),
        ("primary_response_sha256", "response_sha256", "sha"),
        ("primary_asset_path", "asset_path", "str"),
        ("primary_decoded_format", "decoded_format", "str"),
        ("primary_width", "width", "int"),
        ("primary_height", "height", "int"),
        ("primary_n_frames", "n_frames", "int"),
        ("primary_response_byte_count", "response_byte_count", "int"),
        ("primary_qa_sampled", "qa_sampled", "bool"),
        ("primary_qa_identity_status", "qa_identity_status", "str"),
        ("primary_qa_outer_package_status", "qa_outer_package_status", "str"),
        ("primary_qa_exception_flag", "qa_exception_flag", "bool"),
    )
    for row in manifest:
        asin = str(row.get("parent_asin") or "")
        source = p7c_by_asin.get(asin)
        if source is None:
            raise FormalVerificationError(f"P7-C primary field comparison missing ASIN: {asin}")
        for p7d_field, p7c_field, kind in field_pairs:
            if not _compare_value(kind, row.get(p7d_field), source.get(p7c_field)):
                raise FormalVerificationError(f"P7-C primary field mismatch: {asin} {p7d_field}")


def _filter_review_rows_by_keys(rows: list[Mapping[str, Any]], keys: set[str]) -> list[Mapping[str, Any]]:
    return [row for row in rows if _candidate_key(row) in keys]


def validate_candidate_review_pass(
    rows: list[Mapping[str, Any]],
    candidates: Mapping[str, Mapping[str, Any]],
    contract: Mapping[str, Any],
    role: str,
) -> None:
    by_key = dict(candidates)
    expected_keys = set(by_key)
    resolved_keys = []
    for row in rows:
        key = _candidate_key(row)
        if key not in by_key:
            response_sha = str(row.get("response_sha256") or "").upper()
            sha_matches = [candidate_key for candidate_key, candidate in by_key.items() if str(candidate.get("response_sha256") or "").upper() == response_sha]
            if len(sha_matches) == 1:
                key = sha_matches[0]
            else:
                parent_matches = [candidate_key for candidate_key, candidate in by_key.items() if str(candidate.get("parent_asin") or "") == str(row.get("parent_asin") or "")]
                if len(parent_matches) == 1:
                    key = parent_matches[0]
        resolved_keys.append(key)
    actual_keys = set(resolved_keys)
    if actual_keys != expected_keys or len(rows) != len(expected_keys):
        raise FormalVerificationError(f"candidate review universe mismatch in pass {role}")
    for row, key in zip(rows, resolved_keys):
        expected = by_key[key]
        if str(row.get("parent_asin") or "") != str(expected.get("parent_asin") or ""):
            raise FormalVerificationError(f"candidate parent binding mismatch: {key}")
        expected_sha = str(expected.get("response_sha256") or key).upper()
        if str(row.get("response_sha256") or "").upper() != expected_sha:
            raise FormalVerificationError(f"candidate image SHA mismatch: {key}")
        if row.get("reviewer_model") != contract["candidate_review_model_identifier"]:
            raise FormalVerificationError(f"candidate reviewer model mismatch: {key}")
        if str(row.get("review_prompt_sha256") or "").upper() != str(contract["candidate_review_prompt_sha256"]).upper():
            raise FormalVerificationError(f"candidate review prompt mismatch: {key}")
        for field in ("reviewer_run_id", "reviewed_at_utc", "identity_status", "outer_package_status", "confidence"):
            if not str(row.get(field) or ""):
                raise FormalVerificationError(f"candidate review field missing: {field}")
        if row["identity_status"] not in contract["candidate_identity_status_enum"] or row["outer_package_status"] not in contract["candidate_outer_package_status_enum"] or row["confidence"] not in contract["candidate_confidence_enum"]:
            raise FormalVerificationError(f"candidate review enum invalid: {key}")


def validate_dual_review_independence(pass_a: list[Mapping[str, Any]], pass_b: list[Mapping[str, Any]]) -> None:
    by_a = {_candidate_key(row): row for row in pass_a}
    by_b = {_candidate_key(row): row for row in pass_b}
    if set(by_a) != set(by_b):
        raise FormalVerificationError("candidate dual review universe mismatch")
    for key in sorted(by_a):
        run_a = str(by_a[key].get("reviewer_run_id") or "")
        run_b = str(by_b[key].get("reviewer_run_id") or "")
        if not run_a or not run_b or run_a == run_b:
            raise FormalVerificationError(f"candidate reviews must be independent A/B runs: {key}")


def validate_candidate_adjudication(
    adjudication: Mapping[str, Any],
    pass_a: Mapping[str, Any],
    pass_b: Mapping[str, Any],
    contract: Mapping[str, Any],
) -> None:
    key = _candidate_key(adjudication)
    if str(adjudication.get("response_sha256") or "").upper() != str(pass_a.get("response_sha256") or "").upper() or str(adjudication.get("response_sha256") or "").upper() != str(pass_b.get("response_sha256") or "").upper():
        raise FormalVerificationError(f"candidate adjudication image SHA mismatch: {key}")
    pass_a_run_id = adjudication.get("pass_a_reviewer_run_id")
    pass_b_run_id = adjudication.get("pass_b_reviewer_run_id")
    if not pass_a_run_id and not pass_b_run_id:
        pass_a_run_id = pass_a.get("reviewer_run_id")
        pass_b_run_id = pass_b.get("reviewer_run_id")
    if pass_a_run_id != pass_a.get("reviewer_run_id") or pass_b_run_id != pass_b.get("reviewer_run_id"):
        raise FormalVerificationError(f"candidate adjudication review trace mismatch: {key}")
    if adjudication.get("adjudicator_model") != contract["candidate_review_model_identifier"]:
        raise FormalVerificationError(f"candidate adjudicator model mismatch: {key}")
    if str(adjudication.get("adjudication_prompt_sha256") or "").upper() != str(contract["candidate_review_prompt_sha256"]).upper():
        raise FormalVerificationError(f"candidate adjudication prompt mismatch: {key}")
    if not str(adjudication.get("adjudicator_run_id") or ""):
        raise FormalVerificationError(f"candidate adjudicator run missing: {key}")
    expected_qualified = adjudication.get("identity_status") == "consistent" and adjudication.get("outer_package_status") == "outer_retail_package"
    if _as_bool(adjudication.get("candidate_qualified")) != expected_qualified:
        raise FormalVerificationError(f"candidate qualified flag mismatch: {key}")


def build_label_diagnostics(
    manifest: list[Mapping[str, Any]],
    labels: list[Mapping[str, Any]],
    label_columns: list[str],
) -> list[dict[str, Any]]:
    by_asin: dict[str, Mapping[str, Any]] = {}
    for label in labels:
        asin = str(label.get("parent_asin") or "")
        if asin in by_asin:
            raise FormalVerificationError("label join parent_asin must be unique")
        by_asin[asin] = label
    expected = {str(row.get("parent_asin") or "") for row in manifest}
    if set(by_asin) != expected:
        raise FormalVerificationError("label join coverage must equal manifest coverage")
    pu_columns = set(_pu_outcome_label_columns(label_columns))
    output: list[dict[str, Any]] = []
    for source in manifest:
        label = by_asin[str(source["parent_asin"])]
        row = dict(source)
        for column in label_columns:
            value = _csv_text(label.get(column))
            row[column] = value
            if column not in pu_columns:
                continue
            if value.lower() in {"1", "true", "yes"}:
                interpretation = "observed_positive"
            elif value.lower() in {"0", "false", "no"}:
                interpretation = "pu_unlabeled"
            else:
                interpretation = "frozen_label_value"
            row[f"{column}_interpretation"] = interpretation
        output.append(row)
    return output

def _source_record_hash(raw_line: bytes) -> str:
    if raw_line.endswith(b"\n"):
        raw_line = raw_line[:-1]
    if raw_line.endswith(b"\r"):
        raw_line = raw_line[:-1]
    return sha256_bytes(raw_line)


def scan_raw_metadata(path: Path, target_asins: set[str]) -> dict[str, list[dict[str, Any]]]:
    records: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with path.open("rb") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            if not raw_line.strip():
                continue
            record = json.loads(raw_line.decode("utf-8"))
            asin = str(record.get("parent_asin") or "").strip()
            if asin in target_asins:
                records[asin].append({"record": record, "line_number": line_number, "source_record_sha256": _source_record_hash(raw_line)})
    return records


def _candidate_row(asin: str, source: Mapping[str, Any], obj: Mapping[str, Any], index: int, url_field: str, url: str, contract: Mapping[str, Any]) -> dict[str, Any]:
    tier = str(source["source_tier"])
    source_hash = str(source.get("source_record_sha256") or "")
    candidate_id = f"{asin}|{tier}|{index}|{sha256_bytes(url.encode('utf-8'))[:16]}"
    return {
        "candidate_id": candidate_id,
        "parent_asin": asin,
        "source_tier": tier,
        "source_tier_rank": _as_int(contract["source_tiers"][tier]["rank"]),
        "source_record_sha256": source_hash,
        "source_image_index": index,
        "source_variant": str(obj.get("variant") or ""),
        "source_size_role": url_field,
        "source_url_field": url_field,
        "requested_url": url,
        "source_provenance": source.get("source_provenance", ""),
        "temporal_alignment_status": contract["source_tiers"][tier]["temporal_alignment_status"],
        "download_status": "not_attempted",
        "decode_status": "not_attempted",
        "final_asset_status": "not_attempted",
        "response_sha256": "",
        "asset_path": "",
        "decoded_format": "",
        "width": 0,
        "height": 0,
        "n_frames": 0,
        "response_byte_count": 0,
        "candidate_structural_status": "not_checked",
        "reuses_existing_primary_asset": False,
        "error_class": "",
        "error_detail": "",
        "retrieved_at_utc": "",
    }


def reconstruct_source_candidates(exception_rows: list[Mapping[str, Any]], p7b_rows: list[Mapping[str, Any]], p7a_rows: list[Mapping[str, Any]], raw_records: Mapping[str, list[Mapping[str, Any]]], contract: Mapping[str, Any]) -> list[dict[str, Any]]:
    p7b_by = {str(row["parent_asin"]): row for row in p7b_rows}
    p7a_by = {str(row["parent_asin"]): row for row in p7a_rows}
    candidates: list[dict[str, Any]] = []
    for exception in exception_rows:
        if not _as_bool(exception.get("sensitivity_queue")):
            continue
        asin = str(exception["parent_asin"])
        primary = p7b_by[asin]
        audit = p7a_by[asin]
        linked = [r for r in raw_records.get(asin, []) if str(r.get("source_record_sha256")) == str(primary.get("source_record_sha256"))]
        source = linked[0] if linked else None
        if source:
            for index, obj in enumerate(source["record"].get("images") or []):
                if not isinstance(obj, Mapping):
                    continue
                url_field, url = next(((field, str(obj.get(field)).strip()) for field in ("hi_res", "large", "thumb") if str(obj.get(field) or "").strip()), ("", ""))
                if not url:
                    continue
                is_primary_object = (
                    index == _as_int(audit.get("source_image_position"))
                    and str(primary.get("image_role")) == "declared_main"
                ) or (
                    url == str(primary.get("requested_url") or "")
                    and str(obj.get("variant") or "").upper() == "MAIN"
                )
                if is_primary_object:
                    continue
                candidates.append(_candidate_row(asin, {"source_tier": "frozen_2023_metadata_secondary", "source_record_sha256": source["source_record_sha256"], "source_provenance": f"raw_metadata_line:{source['line_number']}"}, obj, index, url_field, url, contract))
        if _as_bool(exception.get("excluded_non_primary")):
            url = str(audit.get("reconstructed_selected_url") or primary.get("requested_url") or "").strip()
            if url:
                obj = {"variant": "", "historical_fallback": True}
                candidates.append(_candidate_row(asin, {"source_tier": "historical_existing_first_image_fallback", "source_record_sha256": primary.get("source_record_sha256", ""), "source_provenance": "p7_b_historical_existing_first_image_fallback"}, obj, _as_int(audit.get("source_image_position")), "requested_url", url, contract))
    return sorted(candidates, key=lambda row: (row["parent_asin"], _as_int(row["source_tier_rank"]), _as_int(row["source_image_index"]), row["requested_url"], row["candidate_id"]))


_AMAZON_IMAGE_PATTERN = re.compile(
    r"https?://(?:m\.media-amazon\.com|images-na\.ssl-images-amazon\.com|images\.amazon\.com)/images/I/[A-Za-z0-9%_+./~:-]+",
    re.IGNORECASE,
)
_ASIN_IN_URL_PATTERN = re.compile(r"/(?:dp|gp/product)/([A-Z0-9]{10})(?:[/?#]|$)", re.IGNORECASE)


class _ExactParentMainImageParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.page_asins: set[str] = set()
        self.landing_candidates: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attr = {str(name).lower(): value or "" for name, value in attrs}
        if tag.lower() == "link" and attr.get("rel", "").lower() == "canonical":
            self._add_asin_from_url(attr.get("href", ""))
        if tag.lower() == "meta" and attr.get("property", "").lower() in {"og:url", "al:android:url", "al:ios:url"}:
            self._add_asin_from_url(attr.get("content", ""))
        if tag.lower() == "input" and attr.get("id", "").lower() == "asin":
            self._add_asin_value(attr.get("value", ""))
        if attr.get("name", "").lower() in {"asin", "parentasin"}:
            self._add_asin_value(attr.get("value", ""))
        for key in ("data-asin", "data-parent-asin"):
            self._add_asin_value(attr.get(key, ""))
        if tag.lower() == "img" and attr.get("id", "").lower() == "landingimage":
            for key in ("data-old-hires", "src"):
                self._add_landing_candidate(attr.get(key, ""))
            for url in _extract_dynamic_image_urls(attr.get("data-a-dynamic-image", "")):
                self._add_landing_candidate(url)
        if tag.lower() == "meta" and attr.get("property", "").lower() == "og:image":
            self._add_landing_candidate(attr.get("content", ""))

    def _add_asin_from_url(self, value: str) -> None:
        match = _ASIN_IN_URL_PATTERN.search(html.unescape(value or ""))
        if match:
            self.page_asins.add(match.group(1).upper())

    def _add_asin_value(self, value: str) -> None:
        normalized = str(value or "").strip().upper()
        if re.fullmatch(r"[A-Z0-9]{10}", normalized):
            self.page_asins.add(normalized)

    def _add_landing_candidate(self, value: str) -> None:
        url = _normalize_amazon_image_url(value)
        if url and url not in self.landing_candidates:
            self.landing_candidates.append(url)


def _normalize_amazon_image_url(value: str) -> str:
    text = html.unescape(str(value or "")).replace("\\/", "/").replace("\\u002F", "/")
    match = _AMAZON_IMAGE_PATTERN.search(text)
    if not match:
        return ""
    return match.group(0).rstrip("\\\\\"'<>),;]")


def _extract_dynamic_image_urls(value: str) -> list[str]:
    text = html.unescape(str(value or "")).replace("\\/", "/").replace("\\u002F", "/")
    if not text.strip():
        return []
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return [_normalize_amazon_image_url(match.group(0)) for match in _AMAZON_IMAGE_PATTERN.finditer(text)]
    urls: list[tuple[int, str]] = []
    if isinstance(payload, Mapping):
        for url, dimensions in payload.items():
            normalized = _normalize_amazon_image_url(str(url))
            if not normalized:
                continue
            area = 0
            if isinstance(dimensions, list) and len(dimensions) >= 2:
                area = _as_int(dimensions[0]) * _as_int(dimensions[1])
            urls.append((area, normalized))
    urls.sort(key=lambda item: (-item[0], item[1]))
    return [url for _, url in urls]


def extract_current_parent_main_image_url(body: bytes, parent_asin: str) -> str | None:
    parser = _ExactParentMainImageParser()
    parser.feed(body.decode("utf-8", errors="replace"))
    requested = str(parent_asin or "").strip().upper()
    if not requested or requested not in parser.page_asins:
        return None
    if parser.page_asins - {requested}:
        return None
    return parser.landing_candidates[0] if parser.landing_candidates else None


def _current_parent_page_row(asin: str, page_url: str, source_hash: str, provenance: str, contract: Mapping[str, Any], error_class: str, error_detail: str) -> dict[str, Any]:
    row = _candidate_row(asin, {"source_tier": "current_exact_parent_asin", "source_record_sha256": source_hash, "source_provenance": provenance}, {"variant": "CURRENT_EXACT_PARENT_PAGE"}, 0, "exact_parent_page", page_url, contract)
    row.update({"download_status": "page_lookup_failed", "final_asset_status": "acquisition_failed", "candidate_structural_status": "ineligible", "error_class": error_class, "error_detail": error_detail, "retrieved_at_utc": utc_now()})
    return row


def fetch_current_exact_parent_candidates(asins: set[str], contract: Mapping[str, Any]) -> list[dict[str, Any]]:
    policy = contract["network_policy"]
    limits = contract["response_limits"]
    if not policy.get("exact_parent_asin_page_lookup", False) or policy.get("search_engine", False) or policy.get("cookies", False) or policy.get("authenticated_session", False) or policy.get("browser_automation", False):
        raise FormalVerificationError("P7-D exact-parent fallback policy is not safely configured")
    output: list[dict[str, Any]] = []
    for asin in sorted(asins):
        page_url = f"https://www.amazon.com/dp/{asin}"
        page_row = None
        for attempt in range(1, int(policy["max_attempts"]) + 1):
            response = None
            try:
                response = requests.get(page_url, headers={"User-Agent": policy["user_agent"], "Cookie": ""}, timeout=(float(policy["connect_timeout_seconds"]), float(policy["read_timeout_seconds"])), allow_redirects=True, stream=True)
                redirects = len(getattr(response, "history", []) or [])
                if redirects > int(policy["max_redirects"]):
                    page_row = _current_parent_page_row(asin, page_url, "", f"exact_parent_asin_page:{page_url}", contract, "maximum_redirects", f"redirect_count={redirects}")
                    break
                if int(response.status_code) != 200:
                    if int(response.status_code) in {int(value) for value in policy["retry_status_codes"]} and attempt < int(policy["max_attempts"]):
                        sleep_seconds = float(response.headers.get("Retry-After") or policy["backoff_seconds"][min(attempt - 1, len(policy["backoff_seconds"]) - 1)])
                        time.sleep(sleep_seconds)
                        continue
                    page_row = _current_parent_page_row(asin, page_url, "", f"exact_parent_asin_page:{page_url}", contract, f"http_{response.status_code}", f"HTTP status {response.status_code}")
                    break
                body, oversized, count = p7b._read_response_body(response, int(limits["maximum_response_bytes"]))
                source_hash = sha256_bytes(body)
                provenance = f"exact_parent_asin_page:{page_url}"
                if oversized:
                    page_row = _current_parent_page_row(asin, page_url, source_hash, provenance, contract, "maximum_response_bytes", f"read_bytes={count}")
                    break
                main_url = extract_current_parent_main_image_url(body, asin)
                if not main_url:
                    page_row = _current_parent_page_row(asin, page_url, source_hash, provenance, contract, "no_bound_main_image", "exact-parent page did not expose a deterministically bound main/landing image for the requested ASIN")
                    break
                output.append(_candidate_row(asin, {"source_tier": "current_exact_parent_asin", "source_record_sha256": source_hash, "source_provenance": provenance}, {"variant": "CURRENT_EXACT_PARENT_MAIN_LANDING_IMAGE"}, 0, "exact_parent_page_main_image", main_url, contract))
                page_row = None
                break
            except (requests.exceptions.Timeout, TimeoutError) as exc:
                if attempt < int(policy["max_attempts"]):
                    time.sleep(float(policy["backoff_seconds"][min(attempt - 1, len(policy["backoff_seconds"]) - 1)]))
                    continue
                page_row = _current_parent_page_row(asin, page_url, "", f"exact_parent_asin_page:{page_url}", contract, "timeout", type(exc).__name__)
            except (requests.exceptions.RequestException, ConnectionError) as exc:
                if attempt < int(policy["max_attempts"]):
                    time.sleep(float(policy["backoff_seconds"][min(attempt - 1, len(policy["backoff_seconds"]) - 1)]))
                    continue
                page_row = _current_parent_page_row(asin, page_url, "", f"exact_parent_asin_page:{page_url}", contract, "network_error", type(exc).__name__)
            finally:
                if response is not None:
                    response.close()
        if page_row is not None:
            output.append(page_row)
    return output

def _sensitivity_relative_path(digest: str, decoded_format: str) -> str:
    extension = p7b.FORMAT_EXTENSIONS.get(str(decoded_format).upper(), "bin")
    return f"sensitivity/{digest[:2].upper()}/{digest.upper()}.{extension}"


def _store_sensitivity_asset(body: bytes, digest: str, decoded_format: str) -> str:
    relative = _sensitivity_relative_path(digest, decoded_format)
    target = ROOT / "data" / "images" / "retail_outer_package_p7_5180" / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if sha256_file(target) != digest:
            raise FormalVerificationError(f"sensitivity asset SHA collision: {relative}")
        return relative
    temp = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temp.open("xb") as handle:
            handle.write(body)
            handle.flush()
            os.fsync(handle.fileno())
        if sha256_file(temp) != digest:
            raise FormalVerificationError("sensitivity temporary SHA mismatch")
        os.replace(temp, target)
    finally:
        if temp.exists():
            temp.unlink()
    return relative


def acquire_candidate(candidate: Mapping[str, Any], contract: Mapping[str, Any], primary_by_asin: Mapping[str, Mapping[str, Any]], session_factory: Any = None, sleep_fn: Any = time.sleep) -> dict[str, Any]:
    row = dict(candidate)
    url = str(row.get("requested_url") or "")
    if not url:
        row.update({"download_status": "network_error", "final_asset_status": "acquisition_failed", "error_class": "empty_requested_url", "error_detail": "missing exact source URL", "retrieved_at_utc": utc_now()})
        return row
    policy = contract["network_policy"]
    limits = contract["response_limits"]
    session_factory = session_factory or requests.Session
    retryable = {int(value) for value in policy["retry_status_codes"]}
    last_retry = None
    for attempt in range(1, int(policy["max_attempts"]) + 1):
        response = None
        try:
            session = session_factory()
            response = session.get(url, headers={"User-Agent": policy["user_agent"]}, timeout=(float(policy["connect_timeout_seconds"]), float(policy["read_timeout_seconds"])), allow_redirects=True, stream=True)
            row.update({"attempt_count": attempt, "http_status": int(response.status_code), "final_url": str(getattr(response, "url", "") or ""), "redirect_count": len(getattr(response, "history", []) or []), "response_content_type": str(response.headers.get("Content-Type", "") or ""), "response_content_length_header": str(response.headers.get("Content-Length", "") or "")})
            if int(row.get("redirect_count") or 0) > int(policy["max_redirects"]):
                row.update({"download_status": "redirect_error", "final_asset_status": "acquisition_failed", "error_class": "maximum_redirects", "error_detail": f"redirect_count={row.get('redirect_count')}"})
                return row
            status = int(response.status_code)
            if status != 200:
                if status in retryable and attempt < int(policy["max_attempts"]):
                    last_retry = float(response.headers.get("Retry-After") or policy["backoff_seconds"][min(attempt - 1, len(policy["backoff_seconds"]) - 1)])
                    sleep_fn(last_retry)
                    continue
                row.update({"download_status": "http_error", "final_asset_status": "acquisition_failed", "error_class": f"http_{status}", "error_detail": f"HTTP status {status}"})
                return row
            body, oversized, count = p7b._read_response_body(response, int(limits["maximum_response_bytes"]))
            row["response_byte_count"] = count
            if oversized:
                row.update({"download_status": "response_too_large", "final_asset_status": "acquisition_failed", "error_class": "maximum_response_bytes", "error_detail": f"read_bytes={count}"})
                return row
            digest = sha256_bytes(body)
            decoded = p7b.decode_image_bytes(body, row.get("response_content_type", ""), {"response_limits": limits})
            row.update({"download_status": "success", "response_sha256": digest, "decode_status": decoded["decode_status"], "decoded_format": decoded["decoded_format"], "width": decoded["width"], "height": decoded["height"], "n_frames": decoded["n_frames"], "error_class": decoded.get("error_class", ""), "error_detail": decoded.get("error_detail", "")})
            if decoded["decode_status"] != "success":
                row["final_asset_status"] = "acquisition_failed"
                return row
            primary = primary_by_asin.get(str(row["parent_asin"]), {})
            known_placeholders = {str(s["response_sha256"]).upper() for s in contract["placeholder_signatures"]}
            if digest == str(primary.get("primary_response_sha256") or "").upper() or digest in known_placeholders or _is_s3_ui_placeholder_candidate(row):
                row["candidate_structural_status"] = "ineligible"
                row["reuses_existing_primary_asset"] = digest == str(primary.get("primary_response_sha256") or "").upper()
                row["asset_path"] = primary.get("primary_asset_path", "") if row["reuses_existing_primary_asset"] else ""
                row["final_asset_status"] = "available"
                if _is_s3_ui_placeholder_candidate(row):
                    row["error_class"] = "s3_ui_placeholder"
                    row["error_detail"] = "exact-parent main image resolved to Amazon UI/no-image placeholder"
                return row
            row["asset_path"] = _store_sensitivity_asset(body, digest, decoded["decoded_format"])
            row["final_asset_status"] = "available"
            row["candidate_structural_status"] = "eligible"
            return row
        except (requests.exceptions.Timeout, TimeoutError) as exc:
            if attempt < int(policy["max_attempts"]):
                sleep_fn(float(policy["backoff_seconds"][min(attempt - 1, len(policy["backoff_seconds"]) - 1)]))
                continue
            row.update({"download_status": "timeout", "final_asset_status": "acquisition_failed", "error_class": type(exc).__name__, "error_detail": str(exc)[:500]})
            return row
        except (requests.exceptions.RequestException, ConnectionError) as exc:
            if attempt < int(policy["max_attempts"]):
                sleep_fn(float(policy["backoff_seconds"][min(attempt - 1, len(policy["backoff_seconds"]) - 1)]))
                continue
            row.update({"download_status": "network_error", "final_asset_status": "acquisition_failed", "error_class": type(exc).__name__, "error_detail": str(exc)[:500]})
            return row
        finally:
            if response is not None:
                response.close()
    row.update({"download_status": "network_error", "final_asset_status": "acquisition_failed", "error_class": "retry_loop_exhausted"})
    return row


def _candidate_fields() -> list[str]:
    return ["candidate_id", "parent_asin", "source_tier", "source_tier_rank", "source_record_sha256", "source_image_index", "source_variant", "source_size_role", "source_url_field", "requested_url", "source_provenance", "temporal_alignment_status", "download_status", "attempt_count", "http_status", "final_url", "redirect_count", "response_content_type", "response_content_length_header", "response_byte_count", "response_sha256", "decoded_format", "width", "height", "n_frames", "decode_status", "final_asset_status", "asset_path", "candidate_structural_status", "reuses_existing_primary_asset", "error_class", "error_detail", "retrieved_at_utc"]


def _review_fields() -> list[str]:
    return ["candidate_id", "parent_asin", "response_sha256", "identity_status", "outer_package_status", "confidence", "reason_code", "reviewer_model", "reviewer_run_id", "review_prompt_sha256", "reviewed_at_utc"]


def _adjudication_fields() -> list[str]:
    return ["candidate_id", "parent_asin", "response_sha256", "pass_a_identity_status", "pass_a_outer_package_status", "pass_a_confidence", "pass_a_reviewer_run_id", "pass_b_identity_status", "pass_b_outer_package_status", "pass_b_confidence", "pass_b_reviewer_run_id", "requires_adjudication", "identity_status", "outer_package_status", "confidence", "reason_code", "adjudication_basis", "adjudicator_model", "adjudicator_run_id", "adjudication_prompt_sha256", "candidate_qualified"]


def _manifest_fields() -> list[str]:
    return ["parent_asin", "input_order", "primary_freeze_status", "primary_response_sha256", "primary_asset_path", "primary_decoded_format", "primary_width", "primary_height", "primary_n_frames", "primary_response_byte_count", "primary_qa_sampled", "primary_qa_identity_status", "primary_qa_outer_package_status", "primary_qa_exception_flag", "c_reviewed_qa_exception", "global_placeholder_signature", "excluded_non_primary", "sensitivity_queue", "sensitivity_status", "sensitivity_source_tier", "sensitivity_temporal_alignment_status", "sensitivity_response_sha256", "sensitivity_asset_path", "sensitivity_decoded_format", "sensitivity_width", "sensitivity_height", "sensitivity_n_frames", "sensitivity_response_byte_count", "sensitivity_identity_status", "sensitivity_outer_package_status", "sensitivity_confidence", "sensitivity_candidate_count", "sensitivity_qualified_candidate_count", "sensitivity_selection_rank", "primary_asset_substituted", "final_image_stage_status"]


def _exception_fields() -> list[str]:
    return ["parent_asin", "input_order", "primary_freeze_status", "primary_response_sha256", "primary_asset_path", "primary_decoded_format", "primary_width", "primary_height", "primary_n_frames", "primary_response_byte_count", "primary_qa_sampled", "primary_qa_status", "primary_qa_identity_status", "primary_qa_outer_package_status", "primary_qa_exception_flag", "c_reviewed_qa_exception", "global_placeholder_signature", "excluded_non_primary", "sensitivity_queue"]


def _diagnostic_fields(
    manifest: list[Mapping[str, Any]],
    label_columns: list[str],
) -> list[str]:
    fields = list(manifest[0].keys()) if manifest else []
    pu_columns = set(_pu_outcome_label_columns(label_columns))
    for column in label_columns:
        if column not in fields:
            fields.append(column)
        if column in pu_columns and f"{column}_interpretation" not in fields:
            fields.append(f"{column}_interpretation")
    return fields

def _build_final_manifest(exception_rows: list[Mapping[str, Any]], adjudicated: list[Mapping[str, Any]], contract: Mapping[str, Any]) -> list[dict[str, Any]]:
    by_asin: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in adjudicated:
        by_asin[str(row["parent_asin"])].append(dict(row))
    output: list[dict[str, Any]] = []
    for primary in exception_rows:
        asin = str(primary["parent_asin"])
        rows = by_asin.get(asin, [])
        selected = select_sensitivity_candidate(rows, contract)
        queue = _as_bool(primary["sensitivity_queue"])
        status = build_sensitivity_status(queue, selected)
        qualified = [row for row in rows if candidate_qualified(row)]
        sensitivity = dict(selected or {})
        sensitivity.update({"sensitivity_status": status, "candidate_count": len(rows), "qualified_candidate_count": len(qualified), "selection_rank": 1 if selected else 0, "final_image_stage_status": "frozen" if status != "unresolved" else "frozen_with_unresolved_sensitivity", "primary_asset_substituted": False})
        output.append(preserve_primary_fields(primary, sensitivity))
    return sorted(output, key=lambda row: _as_int(row["input_order"]))


def _load_p7_rows() -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    p7c_manifest = _read_csv(ROOT / "data/processed/retail_outer_package_images_p7_5180/p7_c_primary_manifest/05_primary_manifest.csv")
    p7b_manifest = _read_csv(ROOT / "data/processed/retail_outer_package_images_p7_5180/p7_b_primary_assets/01_primary_asset_manifest.csv")
    p7a_inventory = _read_csv(ROOT / "data/processed/retail_outer_package_images_p7_5180/p7_a_source_inventory/01_source_identity_inventory.csv")
    if len(p7c_manifest) != 5180 or len(p7b_manifest) != 5180:
        raise FormalVerificationError("P7-D upstream universe is not 5180 rows")
    return p7c_manifest, p7b_manifest, p7a_inventory


def _placeholder_map(contract: Mapping[str, Any]) -> dict[str, str]:
    return {str(row["response_sha256"]).upper(): str(row["name"]) for row in contract["placeholder_signatures"]}


def prepare_exceptions(contract: Mapping[str, Any]) -> dict[str, Any]:
    verify_upstream(contract)
    paths = _paths(contract)
    if paths["exception_inventory"].exists():
        raise FormalVerificationError("P7-D exception inventory already exists")
    p7c_manifest, p7b_manifest, _ = _load_p7_rows()
    primary_rows = [row for row in p7b_manifest if row.get("eligibility_status") == "primary_declared_main"]
    placeholder_by_sha = _placeholder_map(contract)
    global_asins = {str(row["parent_asin"]) for row in primary_rows if str(row.get("response_sha256") or "").upper() in placeholder_by_sha}
    exception_rows = build_exception_inventory(p7c_manifest, global_asins, placeholder_by_sha)
    _write_csv(paths["exception_inventory"], _exception_fields(), exception_rows)
    return {"rows": len(exception_rows), "queue": len(build_sensitivity_queue(exception_rows)), "global_placeholder_product_count": len(global_asins), "sha256": sha256_file(paths["exception_inventory"])}


def acquire_candidates(contract: Mapping[str, Any]) -> dict[str, Any]:
    paths = _paths(contract)
    if not paths["exception_inventory"].exists():
        raise FormalVerificationError("run prepare-exceptions first")
    if paths["candidate_inventory"].exists():
        raise FormalVerificationError("P7-D candidate inventory already exists")
    exception_rows = _read_csv(paths["exception_inventory"])
    _, p7b_manifest, p7a_inventory = _load_p7_rows()
    queue = build_sensitivity_queue(exception_rows)
    raw_records = scan_raw_metadata(RAW_METADATA_PATH, queue)
    initial = reconstruct_source_candidates(exception_rows, p7b_manifest, p7a_inventory, raw_records, contract)
    primary_by = {str(row["parent_asin"]): row for row in exception_rows}
    with ThreadPoolExecutor(max_workers=int(contract["network_policy"]["max_workers"])) as executor:
        acquired = list(executor.map(lambda row: acquire_candidate(row, contract, primary_by), initial))
    acquired.sort(key=lambda row: (row["parent_asin"], _as_int(row["source_tier_rank"]), _as_int(row["source_image_index"]), str(row.get("response_sha256") or ""), row["candidate_id"]))
    _write_csv(paths["candidate_inventory"], _candidate_fields(), acquired)
    return {"candidate_source_object_count": len(initial), "current_exact_parent_fallback_product_count": 0, "s3_deferred_until_after_s1_s2_adjudication": True, "candidate_download_attempt_count": sum(_as_int(row.get("attempt_count")) for row in acquired), "candidate_available_count": sum(row.get("final_asset_status") == "available" for row in acquired), "sha256": sha256_file(paths["candidate_inventory"])}

def _review_candidates_from_inventory(rows: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [dict(row) for row in rows if row.get("candidate_structural_status") == "eligible"]


def write_review_template(contract: Mapping[str, Any]) -> dict[str, Any]:
    paths = _paths(contract)
    rows = _review_candidates_from_inventory(_read_csv(paths["candidate_inventory"]))
    workspace = paths["output_dir"].parent / "p7_d_review_workspace"
    template = []
    for row in rows:
        template.append({"candidate_id": row["candidate_id"], "parent_asin": row["parent_asin"], "response_sha256": row["response_sha256"], "identity_status": "", "outer_package_status": "", "confidence": "", "reason_code": "", "reviewer_model": contract["candidate_review_model_identifier"], "reviewer_run_id": "", "review_prompt_sha256": contract["candidate_review_prompt_sha256"], "reviewed_at_utc": ""})
    target = workspace / "review_template.csv"
    _write_csv(target, _review_fields(), template)
    return {"reviewable_candidates": len(rows), "template": str(target)}


def _candidate_by_id(rows: list[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    return {_candidate_key(row): dict(row) for row in rows}


def _build_review_adjudication_rows(
    reviewable: Mapping[str, Mapping[str, Any]],
    pass_a: list[Mapping[str, Any]],
    pass_b: list[Mapping[str, Any]],
    adjudication_input: list[Mapping[str, Any]],
    contract: Mapping[str, Any],
) -> list[dict[str, Any]]:
    by_a = {_candidate_key(row): row for row in pass_a}
    by_b = {_candidate_key(row): row for row in pass_b}
    adjudication_input = _filter_review_rows_by_keys(list(adjudication_input), set(reviewable))
    adj_by = {_candidate_key(row): row for row in adjudication_input}
    final_rows: list[dict[str, Any]] = []
    for key in sorted(reviewable):
        a, b = by_a[key], by_b[key]
        trigger = a["identity_status"] != b["identity_status"] or a["outer_package_status"] != b["outer_package_status"] or a["confidence"] == "low" or b["confidence"] == "low" or a["identity_status"] == "indeterminate" or b["identity_status"] == "indeterminate" or a["outer_package_status"] == "ambiguous" or b["outer_package_status"] == "ambiguous"
        if trigger:
            if key not in adj_by:
                raise FormalVerificationError(f"candidate adjudication missing: {key}")
            row = dict(adj_by[key])
            row.update({"candidate_id": key, "parent_asin": a["parent_asin"], "response_sha256": a["response_sha256"], "pass_a_identity_status": a["identity_status"], "pass_a_outer_package_status": a["outer_package_status"], "pass_a_confidence": a["confidence"], "pass_a_reviewer_run_id": a["reviewer_run_id"], "pass_b_identity_status": b["identity_status"], "pass_b_outer_package_status": b["outer_package_status"], "pass_b_confidence": b["confidence"], "pass_b_reviewer_run_id": b["reviewer_run_id"], "requires_adjudication": True, "adjudication_basis": "model_assisted_adjudication"})
            validate_candidate_adjudication(row, a, b, contract)
        else:
            row = {"candidate_id": key, "parent_asin": a["parent_asin"], "response_sha256": a["response_sha256"], "pass_a_identity_status": a["identity_status"], "pass_a_outer_package_status": a["outer_package_status"], "pass_a_confidence": a["confidence"], "pass_a_reviewer_run_id": a["reviewer_run_id"], "pass_b_identity_status": b["identity_status"], "pass_b_outer_package_status": b["outer_package_status"], "pass_b_confidence": b["confidence"], "pass_b_reviewer_run_id": b["reviewer_run_id"], "requires_adjudication": False, "identity_status": a["identity_status"], "outer_package_status": a["outer_package_status"], "confidence": "low" if "low" in {a["confidence"], b["confidence"]} else a["confidence"], "reason_code": a.get("reason_code", "agreement"), "adjudication_basis": "agreement", "adjudicator_model": "", "adjudicator_run_id": "", "adjudication_prompt_sha256": "", "candidate_qualified": a["identity_status"] == "consistent" and a["outer_package_status"] == "outer_retail_package"}
        candidate = reviewable[key]
        for field in ("source_tier", "source_tier_rank", "source_record_sha256", "source_image_index", "source_variant", "source_size_role", "source_url_field", "requested_url", "source_provenance", "temporal_alignment_status", "download_status", "decode_status", "final_asset_status", "asset_path", "decoded_format", "width", "height", "n_frames", "response_byte_count", "candidate_structural_status", "reuses_existing_primary_asset"):
            row[field] = candidate.get(field, "")
        row["candidate_qualified"] = row.get("identity_status") == "consistent" and row.get("outer_package_status") == "outer_retail_package"
        final_rows.append(row)
    return final_rows


def acquire_s3_candidates(contract: Mapping[str, Any], pass_a_path: Path, pass_b_path: Path, adjudication_path: Path) -> dict[str, Any]:
    paths = _paths(contract)
    if not paths["candidate_inventory"].exists() or not paths["exception_inventory"].exists():
        raise FormalVerificationError("run prepare-exceptions and acquire-candidates first")
    candidates = _read_csv(paths["candidate_inventory"])
    if any(str(row.get("source_tier") or "") == "current_exact_parent_asin" for row in candidates):
        raise FormalVerificationError("P7-D S3 candidates already exist in candidate inventory")
    s1_s2_reviewable = _candidate_by_id([row for row in _review_candidates_from_inventory(candidates) if str(row.get("source_tier") or "") != "current_exact_parent_asin"])
    s1_s2_keys = set(s1_s2_reviewable)
    pass_a = _filter_review_rows_by_keys(_read_csv(pass_a_path), s1_s2_keys)
    pass_b = _filter_review_rows_by_keys(_read_csv(pass_b_path), s1_s2_keys)
    adjudication_input = _filter_review_rows_by_keys(_read_csv(adjudication_path), s1_s2_keys)
    validate_candidate_review_pass(pass_a, s1_s2_reviewable, contract, "A")
    validate_candidate_review_pass(pass_b, s1_s2_reviewable, contract, "B")
    validate_dual_review_independence(pass_a, pass_b)
    s1_s2_adjudicated = _build_review_adjudication_rows(s1_s2_reviewable, pass_a, pass_b, adjudication_input, contract)
    queue = build_sensitivity_queue(_read_csv(paths["exception_inventory"]))
    missing = products_requiring_s3_after_s1_s2_adjudication(queue, s1_s2_adjudicated)
    primary_by = {str(row["parent_asin"]): row for row in _read_csv(paths["exception_inventory"])}
    fallback_source_rows = fetch_current_exact_parent_candidates(missing, contract) if missing else []
    fallback_image_rows = [row for row in fallback_source_rows if row.get("source_url_field") == "exact_parent_page_main_image"]
    fallback_audit_rows = [row for row in fallback_source_rows if row.get("source_url_field") != "exact_parent_page_main_image"]
    acquired = list(candidates)
    if fallback_image_rows:
        with ThreadPoolExecutor(max_workers=int(contract["network_policy"]["max_workers"])) as executor:
            acquired.extend(list(executor.map(lambda row: acquire_candidate(row, contract, primary_by), fallback_image_rows)))
    acquired.extend(fallback_audit_rows)
    acquired.sort(key=lambda row: (row["parent_asin"], _as_int(row["source_tier_rank"]), _as_int(row["source_image_index"]), str(row.get("response_sha256") or ""), row["candidate_id"]))
    _write_csv(paths["candidate_inventory"], _candidate_fields(), acquired)
    return {"s3_required_product_count": len(missing), "s3_source_object_count": len(fallback_source_rows), "s3_image_candidate_count": len(fallback_image_rows), "s3_unresolved_page_count": len(fallback_audit_rows), "candidate_download_attempt_count": sum(_as_int(row.get("attempt_count")) for row in acquired), "candidate_available_count": sum(row.get("final_asset_status") == "available" for row in acquired), "sha256": sha256_file(paths["candidate_inventory"])}


def freeze_sensitivity(contract: Mapping[str, Any], pass_a_path: Path, pass_b_path: Path, adjudication_path: Path) -> dict[str, Any]:
    pass_a_path = _resolve_repo_path(pass_a_path)
    pass_b_path = _resolve_repo_path(pass_b_path)
    adjudication_path = _resolve_repo_path(adjudication_path)
    paths = _paths(contract)
    if not paths["candidate_inventory"].exists() or not paths["exception_inventory"].exists():
        raise FormalVerificationError("run prepare-exceptions and acquire-candidates first")
    if paths["final_manifest"].exists():
        raise FormalVerificationError("P7-D final manifest already exists")
    candidates = _read_csv(paths["candidate_inventory"])
    reviewable = _candidate_by_id(_review_candidates_from_inventory(candidates))
    reviewable_keys = set(reviewable)
    pass_a = _filter_review_rows_by_keys(_read_csv(pass_a_path), reviewable_keys)
    pass_b = _filter_review_rows_by_keys(_read_csv(pass_b_path), reviewable_keys)
    validate_candidate_review_pass(pass_a, reviewable, contract, "A")
    validate_candidate_review_pass(pass_b, reviewable, contract, "B")
    validate_dual_review_independence(pass_a, pass_b)
    by_a = {_candidate_key(row): row for row in pass_a}
    by_b = {_candidate_key(row): row for row in pass_b}
    adjudication_input = _read_csv(adjudication_path)
    adjudication_input = _filter_review_rows_by_keys(list(adjudication_input), set(reviewable))
    adj_by = {_candidate_key(row): row for row in adjudication_input}
    final_rows: list[dict[str, Any]] = []
    for key in sorted(reviewable):
        a, b = by_a[key], by_b[key]
        trigger = a["identity_status"] != b["identity_status"] or a["outer_package_status"] != b["outer_package_status"] or a["confidence"] == "low" or b["confidence"] == "low" or a["identity_status"] == "indeterminate" or b["identity_status"] == "indeterminate" or a["outer_package_status"] == "ambiguous" or b["outer_package_status"] == "ambiguous"
        if trigger:
            if key not in adj_by:
                raise FormalVerificationError(f"candidate adjudication missing: {key}")
            row = dict(adj_by[key])
            row.update({"candidate_id": key, "parent_asin": a["parent_asin"], "response_sha256": a["response_sha256"], "pass_a_identity_status": a["identity_status"], "pass_a_outer_package_status": a["outer_package_status"], "pass_a_confidence": a["confidence"], "pass_a_reviewer_run_id": a["reviewer_run_id"], "pass_b_identity_status": b["identity_status"], "pass_b_outer_package_status": b["outer_package_status"], "pass_b_confidence": b["confidence"], "pass_b_reviewer_run_id": b["reviewer_run_id"], "requires_adjudication": True, "adjudication_basis": "model_assisted_adjudication"})
            validate_candidate_adjudication(row, a, b, contract)
        else:
            row = {"candidate_id": key, "parent_asin": a["parent_asin"], "response_sha256": a["response_sha256"], "pass_a_identity_status": a["identity_status"], "pass_a_outer_package_status": a["outer_package_status"], "pass_a_confidence": a["confidence"], "pass_a_reviewer_run_id": a["reviewer_run_id"], "pass_b_identity_status": b["identity_status"], "pass_b_outer_package_status": b["outer_package_status"], "pass_b_confidence": b["confidence"], "pass_b_reviewer_run_id": b["reviewer_run_id"], "requires_adjudication": False, "identity_status": a["identity_status"], "outer_package_status": a["outer_package_status"], "confidence": "low" if "low" in {a["confidence"], b["confidence"]} else a["confidence"], "reason_code": a.get("reason_code", "agreement"), "adjudication_basis": "agreement", "adjudicator_model": "", "adjudicator_run_id": "", "adjudication_prompt_sha256": "", "candidate_qualified": a["identity_status"] == "consistent" and a["outer_package_status"] == "outer_retail_package"}
        candidate = reviewable[key]
        for field in ("source_tier", "source_tier_rank", "source_record_sha256", "source_image_index", "source_variant", "source_size_role", "source_url_field", "requested_url", "source_provenance", "temporal_alignment_status", "download_status", "decode_status", "final_asset_status", "asset_path", "decoded_format", "width", "height", "n_frames", "response_byte_count", "candidate_structural_status", "reuses_existing_primary_asset"):
            row[field] = candidate.get(field, "")
        row["candidate_qualified"] = row.get("identity_status") == "consistent" and row.get("outer_package_status") == "outer_retail_package"
        final_rows.append(row)
    exception_rows = _read_csv(paths["exception_inventory"])
    queue = build_sensitivity_queue(exception_rows)
    validate_s3_escalation_set(queue, final_rows, candidates)
    _write_csv(paths["candidate_adjudication"], _adjudication_fields(), final_rows)
    by_candidate = {row["candidate_id"]: row for row in final_rows}
    final_manifest = _build_final_manifest(exception_rows, final_rows, contract)
    _write_csv(paths["final_manifest"], _manifest_fields(), final_manifest)
    pre_label_sha = sha256_file(paths["final_manifest"])
    formal_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    review_sidecars = {
        "pass_a": _review_sidecar_trace(pass_a_path, _read_csv(pass_a_path), "pass_a"),
        "pass_b": _review_sidecar_trace(pass_b_path, _read_csv(pass_b_path), "pass_b"),
        "adjudication": _review_sidecar_trace(adjudication_path, _read_csv(adjudication_path), "adjudication"),
    }
    _write_json(paths["provenance"], {
        "contract_version": contract["contract_version"],
        "formal_run_git_commit": formal_commit,
        "phase1_producer_commit": formal_commit,
        "phase1_producer_script_git_blob_sha256": git_blob_sha256(formal_commit, "scripts/images/p7_d_final_image_freeze.py"),
        "phase2_diagnostics_producer_commit": "",
        "phase2_diagnostics_script_git_blob_sha256": "",
        "producer_script_path": "scripts/images/p7_d_final_image_freeze.py",
        "producer_script_git_blob_sha256": git_blob_sha256(formal_commit, "scripts/images/p7_d_final_image_freeze.py"),
        "contract_path": "config/image_assets/p7_d_final_freeze_contract.json",
        "contract_git_blob_sha256": git_blob_sha256(formal_commit, "config/image_assets/p7_d_final_freeze_contract.json"),
        "review_prompt_path": P7C_PROMPT_REPO_PATH,
        "review_prompt_git_blob_sha256": git_blob_sha256(formal_commit, P7C_PROMPT_REPO_PATH),
        "p7_c_formal_output_sha256": contract["p7_c_formal_output_sha256"],
        "sensitivity_freeze_utc": utc_now(),
        "pre_label_final_image_manifest_sha256": pre_label_sha,
        "post_label_final_image_manifest_sha256": "",
        "label_sources_read_before_sensitivity_freeze": False,
        "label_diagnostics_started_utc": "",
        "label_source_path": "",
        "label_source_sha256": "",
        "review_sidecars": review_sidecars,
        "formal_output_sha256": {},
        "safety": contract["safety"],
    })
    return {"final_manifest_rows": len(final_manifest), "unique_parent_asin": len({row["parent_asin"] for row in final_manifest}), "pre_label_final_image_manifest_sha256": pre_label_sha, "sensitivity_available_count": sum(row["sensitivity_status"] == "available" for row in final_manifest), "sensitivity_unresolved_count": sum(row["sensitivity_status"] == "unresolved" for row in final_manifest), "sensitivity_not_required_count": sum(row["sensitivity_status"] == "not_required" for row in final_manifest)}


def _label_columns(labels: list[Mapping[str, Any]]) -> list[str]:
    if not labels:
        return []
    return [key for key in labels[0] if key != "parent_asin"]


_PU_OUTCOME_LABEL_COLUMNS = frozenset(
    {
        "has_any_outer_imagery_observed",
        "has_any_all_level_imagery_evidence",
        *{
            f"{dimension}_observed_positive_{variant}"
            for dimension in (
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
            for variant in ("pilot", "core", "robust")
        },
    }
)


def _pu_outcome_label_columns(label_columns: list[str]) -> list[str]:
    return [column for column in label_columns if column in _PU_OUTCOME_LABEL_COLUMNS]


def diagnose_labels(contract: Mapping[str, Any], label_path: Path) -> dict[str, Any]:
    label_path = _resolve_repo_path(label_path)
    paths = _paths(contract)
    if not paths["final_manifest"].exists() or not paths["provenance"].exists():
        raise FormalVerificationError("sensitivity manifest must be frozen before label diagnostics")
    provenance = json.loads(paths["provenance"].read_text(encoding="utf-8"))
    if provenance.get("review_sidecars"):
        _validate_review_sidecar_provenance(provenance)
    else:
        review_workspace = paths["output_dir"].parent / "p7_d_review_workspace"
        review_sidecar_paths = {
            "pass_a": review_workspace / "review_pass_a.csv",
            "pass_b": review_workspace / "review_pass_b.csv",
            "adjudication": review_workspace / "adjudication.csv",
        }
        if any(not sidecar.exists() for sidecar in review_sidecar_paths.values()):
            raise FormalVerificationError("P7-D review sidecars are required for provenance binding")
        provenance["review_sidecars"] = {
            role: _review_sidecar_trace(sidecar, _read_csv(sidecar), role)
            for role, sidecar in review_sidecar_paths.items()
        }
    pre_sha = str(provenance.get("pre_label_final_image_manifest_sha256") or "")
    if not pre_sha or sha256_file(paths["final_manifest"]) != pre_sha:
        raise FormalVerificationError("pre-label final manifest SHA checkpoint failed")
    labels = _read_csv(label_path)
    manifest = _read_csv(paths["final_manifest"])
    columns = _label_columns(labels)
    diagnostics = build_label_diagnostics(manifest, labels, columns)
    _write_csv(paths["label_diagnostics"], _diagnostic_fields(diagnostics, columns), diagnostics)
    label_sha = sha256_file(label_path)
    post_sha = sha256_file(paths["final_manifest"])
    if post_sha != pre_sha:
        raise FormalVerificationError("label diagnostics modified frozen final manifest")
    label_path_text = _repo_relative_path(label_path)
    summary = build_summary(contract, manifest, _read_csv(paths["candidate_inventory"]), _read_csv(paths["candidate_adjudication"]), diagnostics, columns, provenance, label_path_text, label_sha, pre_sha, post_sha)
    _write_json(paths["summary"], summary)
    phase1_commit = str(provenance.get("phase1_producer_commit") or provenance["formal_run_git_commit"])
    diagnostics_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    provenance.update({
        "phase1_producer_commit": phase1_commit,
        "phase1_producer_script_git_blob_sha256": git_blob_sha256(phase1_commit, provenance["producer_script_path"]),
        "phase2_diagnostics_producer_commit": diagnostics_commit,
        "phase2_diagnostics_script_git_blob_sha256": git_blob_sha256(diagnostics_commit, provenance["producer_script_path"]),
        "post_label_final_image_manifest_sha256": post_sha,
        "label_sources_read_after_sensitivity_freeze": True,
        "label_diagnostics_started_utc": utc_now(),
        "label_source_path": label_path_text,
        "label_source_sha256": label_sha,
        "formal_output_sha256": {name: sha256_file(paths[key]) for name, key in (("01_exception_inventory.csv", "exception_inventory"), ("02_sensitivity_candidate_inventory.csv", "candidate_inventory"), ("03_sensitivity_candidate_adjudication.csv", "candidate_adjudication"), ("04_final_image_manifest.csv", "final_manifest"), ("05_postfreeze_label_diagnostics.csv", "label_diagnostics"), ("06_p7_d_summary.json", "summary"))},
    })
    _write_json(paths["provenance"], provenance)
    return summary


def _group_stats(rows: list[Mapping[str, Any]], label_column: str, group_field: str) -> dict[str, Any]:
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get(group_field, ""))].append(row)
    output = {}
    for key, group in sorted(groups.items()):
        observed = sum(str(row.get(label_column, "")).lower() in {"1", "true", "yes"} for row in group)
        output[key] = {"rows": len(group), "observed_positive_count": observed, "observed_positive_rate": observed / len(group) if group else 0.0, "pu_unlabeled_count": len(group) - observed}
    return output


def build_summary(contract: Mapping[str, Any], manifest: list[Mapping[str, Any]], candidates: list[Mapping[str, Any]], adjudicated: list[Mapping[str, Any]], diagnostics: list[Mapping[str, Any]], label_columns: list[str], provenance: Mapping[str, Any], label_path: str, label_sha: str, pre_sha: str, post_sha: str) -> dict[str, Any]:
    source_tiers = Counter(str(row.get("sensitivity_source_tier") or "") for row in manifest if str(row.get("sensitivity_status")) == "available")
    temporal = Counter(str(row.get("sensitivity_temporal_alignment_status") or "") for row in manifest if str(row.get("sensitivity_status")) == "available")
    pu_outcome_columns = _pu_outcome_label_columns(label_columns)
    summary: dict[str, Any] = {"p7_d_pipeline_integrity_status": "PASS", "sensitivity_coverage_status": "COMPLETE" if all(str(row.get("sensitivity_status")) != "unresolved" for row in manifest) else "PARTIAL", "final_image_stage_status": "FROZEN", "contract_version": contract["contract_version"], "upstream_main_commit": contract["upstream_main_commit"], "p7_c_formal_output_sha256": contract["p7_c_formal_output_sha256"], "product_universe_count": len(manifest), "primary_available_count": sum(str(row.get("primary_freeze_status")) != "excluded_non_primary" for row in manifest), "excluded_non_primary_count": sum(str(row.get("primary_freeze_status")) == "excluded_non_primary" for row in manifest), "c_reviewed_qa_exception_count": sum(_as_bool(row.get("c_reviewed_qa_exception")) for row in manifest), "global_placeholder_product_count": sum(bool(row.get("global_placeholder_signature")) for row in manifest), "sensitivity_queue_count": sum(_as_bool(row.get("sensitivity_queue")) for row in manifest), "candidate_source_object_count": len(candidates), "candidate_download_attempt_count": sum(_as_int(row.get("attempt_count")) for row in candidates), "candidate_available_count": sum(row.get("final_asset_status") == "available" for row in candidates), "candidate_reviewed_count": len(adjudicated), "candidate_qualified_count": sum(_as_bool(row.get("candidate_qualified")) for row in adjudicated), "sensitivity_available_count": sum(row.get("sensitivity_status") == "available" for row in manifest), "sensitivity_unresolved_count": sum(row.get("sensitivity_status") == "unresolved" for row in manifest), "sensitivity_not_required_count": sum(row.get("sensitivity_status") == "not_required" for row in manifest), "sensitivity_source_tier_distribution": dict(sorted(source_tiers.items())), "sensitivity_temporal_distribution": dict(sorted(temporal.items())), "sensitivity_actual_format_distribution": dict(sorted(Counter(str(row.get("sensitivity_decoded_format") or "") for row in manifest if row.get("sensitivity_status") == "available").items())), "primary_asset_substitution_count": sum(_as_bool(row.get("primary_asset_substituted")) for row in manifest), "pre_label_final_image_manifest_sha256": pre_sha, "post_label_final_image_manifest_sha256": post_sha, "label_source_path": label_path, "label_source_sha256": label_sha, "canonical_label_columns": label_columns, "pu_outcome_label_columns": pu_outcome_columns, "outer_main_observed_positive_count": sum(str(row.get("has_any_outer_imagery_observed", "")).lower() in {"1", "true", "yes"} for row in diagnostics) if "has_any_outer_imagery_observed" in label_columns else None, "label_sources_read_before_sensitivity_freeze": False, "label_sources_read_after_sensitivity_freeze": True, "review_text_read": False, "g_final_threshold_selected": False, "modeling_started": False, "image_feature_extraction_started": False, "formal_data_committed": False, "images_committed": False, "observed_positive_diagnostics": {}}
    for column in pu_outcome_columns:
        observed = sum(str(row.get(column, "")).lower() in {"1", "true", "yes"} for row in diagnostics)
        summary["observed_positive_diagnostics"][column] = {"overall_observed_positive_count": observed, "overall_observed_positive_rate": observed / len(diagnostics) if diagnostics else 0.0, "overall_pu_unlabeled_count": len(diagnostics) - observed, "by_primary_freeze_status": _group_stats(diagnostics, column, "primary_freeze_status"), "by_qa_sampled_status": _group_stats(diagnostics, column, "primary_qa_sampled"), "by_c_reviewed_qa_exception": _group_stats(diagnostics, column, "c_reviewed_qa_exception"), "by_global_placeholder_signature": _group_stats(diagnostics, column, "global_placeholder_signature"), "by_sensitivity_queue": _group_stats(diagnostics, column, "sensitivity_queue"), "by_sensitivity_status": _group_stats(diagnostics, column, "sensitivity_status"), "by_sensitivity_source_tier": _group_stats(diagnostics, column, "sensitivity_source_tier"), "by_temporal_alignment_status": _group_stats(diagnostics, column, "sensitivity_temporal_alignment_status")}
    return summary


def ensure_phase1_label_blind(paths: list[Path], contract: Mapping[str, Any]) -> None:
    forbidden = ["v2.1", "39b", "p5", "p6", "review", "label", "g_"]
    for path in paths:
        normalized = str(path).lower()
        if any(token in normalized for token in forbidden):
            raise FormalVerificationError(f"phase-1 label firewall rejected input path: {path}")
    if not contract["label_firewall"]["phase1_label_sources_read"]:
        return
    raise FormalVerificationError("phase-1 label firewall is marked as read")


def _assert_rows_equal(label: str, actual: list[Mapping[str, Any]], expected: list[Mapping[str, Any]]) -> None:
    if len(actual) != len(expected):
        raise FormalVerificationError(f"P7-D {label} row count mismatch")
    for index, (left, right) in enumerate(zip(actual, expected), start=1):
        left_normalized = {key: _csv_text(value) for key, value in dict(left).items()}
        right_normalized = {key: _csv_text(value) for key, value in dict(right).items()}
        if left_normalized != right_normalized:
            keys = sorted(set(left_normalized) | set(right_normalized))
            mismatch = next((key for key in keys if left_normalized.get(key, "") != right_normalized.get(key, "")), "")
            suffix = f" column {mismatch}" if mismatch else ""
            raise FormalVerificationError(f"P7-D {label} row mismatch at row {index}{suffix}")


def _project_csv_fields(rows: list[Mapping[str, Any]], fields: list[str] | tuple[str, ...]) -> list[dict[str, Any]]:
    return [{field: row.get(field, "") for field in fields} for row in rows]


def _assert_json_equal(label: str, actual: Mapping[str, Any], expected: Mapping[str, Any]) -> None:
    if dict(actual) != dict(expected):
        raise FormalVerificationError(f"P7-D {label} JSON mismatch")


def _validate_no_temp_artifacts(output_dir: Path) -> None:
    for path in output_dir.iterdir():
        name = path.name.lower()
        if name.startswith(".") and ".tmp" in name:
            raise FormalVerificationError(f"P7-D temporary artifact present: {path.name}")


def _rebuild_exception_inventory_for_verify(
    contract: Mapping[str, Any],
    p7c_manifest: list[Mapping[str, Any]],
    p7b_manifest: list[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    primary_rows = [row for row in p7b_manifest if row.get("eligibility_status") == "primary_declared_main"]
    placeholder_by_sha = _placeholder_map(contract)
    global_asins = {str(row["parent_asin"]) for row in primary_rows if str(row.get("response_sha256") or "").upper() in placeholder_by_sha}
    return build_exception_inventory(p7c_manifest, global_asins, placeholder_by_sha)


def _validate_candidate_source_linkage(
    exception_rows: list[Mapping[str, Any]],
    candidates: list[Mapping[str, Any]],
    p7b_manifest: list[Mapping[str, Any]],
    p7a_inventory: list[Mapping[str, Any]],
    contract: Mapping[str, Any],
) -> None:
    queue = build_sensitivity_queue(exception_rows)
    raw_records = scan_raw_metadata(RAW_METADATA_PATH, queue)
    expected = {
        row["candidate_id"]: row
        for row in reconstruct_source_candidates(exception_rows, p7b_manifest, p7a_inventory, raw_records, contract)
    }
    actual_s1_s2 = {row["candidate_id"]: row for row in candidates if str(row.get("source_tier") or "") != "current_exact_parent_asin"}
    if set(actual_s1_s2) != set(expected):
        raise FormalVerificationError("P7-D S1/S2 candidate source universe mismatch")
    source_fields = (
        "candidate_id", "parent_asin", "source_tier", "source_tier_rank", "source_record_sha256",
        "source_image_index", "source_variant", "source_size_role", "source_url_field",
        "requested_url", "source_provenance", "temporal_alignment_status",
    )
    for key, row in actual_s1_s2.items():
        source = expected[key]
        for field in source_fields:
            if _csv_text(row.get(field)) != _csv_text(source.get(field)):
                raise FormalVerificationError(f"P7-D candidate source linkage mismatch: {key} {field}")


def _validate_candidate_asset_integrity(
    exception_rows: list[Mapping[str, Any]],
    candidates: list[Mapping[str, Any]],
    contract: Mapping[str, Any],
) -> None:
    primary_by = {str(row["parent_asin"]): row for row in exception_rows}
    placeholder_shas = {str(row["response_sha256"]).upper() for row in contract["placeholder_signatures"]}
    seen: set[str] = set()
    for row in candidates:
        key = str(row.get("candidate_id") or "")
        if not key or key in seen:
            raise FormalVerificationError(f"P7-D candidate id is missing or duplicated: {key}")
        seen.add(key)
        tier = str(row.get("source_tier") or "")
        if tier not in contract["source_tiers"]:
            raise FormalVerificationError(f"P7-D candidate source tier invalid: {key}")
        if _as_int(row.get("source_tier_rank")) != _as_int(contract["source_tiers"][tier]["rank"]):
            raise FormalVerificationError(f"P7-D candidate source tier rank mismatch: {key}")
        if tier == "current_exact_parent_asin":
            provenance = str(row.get("source_provenance") or "")
            expected_provenance = f"exact_parent_asin_page:https://www.amazon.com/dp/{row.get('parent_asin')}"
            if provenance != expected_provenance:
                raise FormalVerificationError(f"P7-D S3 exact-parent provenance mismatch: {key}")
            if row.get("source_url_field") == "exact_parent_page_main_image":
                if str(row.get("source_variant") or "") != "CURRENT_EXACT_PARENT_MAIN_LANDING_IMAGE":
                    raise FormalVerificationError(f"P7-D S3 main image variant mismatch: {key}")
                if not _normalize_amazon_image_url(str(row.get("requested_url") or "")):
                    raise FormalVerificationError(f"P7-D S3 requested URL is not an Amazon image: {key}")
            elif row.get("source_url_field") != "exact_parent_page":
                raise FormalVerificationError(f"P7-D S3 source field invalid: {key}")
        eligible = candidate_is_structurally_eligible(row, primary_by, placeholder_shas)
        if (str(row.get("candidate_structural_status") or "") == "eligible") != eligible:
            raise FormalVerificationError(f"P7-D candidate structural status mismatch: {key}")
        if str(row.get("final_asset_status") or "") == "available" and str(row.get("asset_path") or ""):
            asset_path = PRIMARY_ASSET_ROOT / str(row["asset_path"])
            if not asset_path.exists():
                raise FormalVerificationError(f"P7-D candidate asset missing: {key}")
            body = asset_path.read_bytes()
            if sha256_bytes(body) != str(row.get("response_sha256") or "").upper():
                raise FormalVerificationError(f"P7-D candidate local SHA mismatch: {key}")
            decoded = p7b.decode_image_bytes(body, str(row.get("response_content_type") or ""), {"response_limits": contract["response_limits"]})
            if decoded["decode_status"] != str(row.get("decode_status") or ""):
                raise FormalVerificationError(f"P7-D candidate local decode mismatch: {key}")
            for field in ("decoded_format", "width", "height", "n_frames"):
                if str(decoded[field]) != str(row.get(field) or ""):
                    raise FormalVerificationError(f"P7-D candidate local decode field mismatch: {key} {field}")


def _review_sidecar_trace(path: Path, rows: list[Mapping[str, Any]], role: str) -> dict[str, Any]:
    fields = {
        "pass_a": ("reviewer_run_id", "reviewer_model", "review_prompt_sha256"),
        "pass_b": ("reviewer_run_id", "reviewer_model", "review_prompt_sha256"),
        "adjudication": ("adjudicator_run_id", "adjudicator_model", "adjudication_prompt_sha256"),
    }
    if role not in fields:
        raise FormalVerificationError(f"unknown P7-D review sidecar role: {role}")
    run_field, model_field, prompt_field = fields[role]
    return {
        "role": role,
        "path": _repo_relative_path(path),
        "sha256": sha256_file(path),
        "row_count": len(rows),
        "run_ids": sorted({str(row.get(run_field) or "") for row in rows if str(row.get(run_field) or "")}),
        "model_identifiers": sorted({str(row.get(model_field) or "") for row in rows if str(row.get(model_field) or "")}),
        "prompt_sha256s": sorted({str(row.get(prompt_field) or "").upper() for row in rows if str(row.get(prompt_field) or "")}),
    }


def _validate_review_sidecar_provenance(
    provenance: Mapping[str, Any],
) -> dict[str, tuple[Path, list[dict[str, Any]]]]:
    sidecars = provenance.get("review_sidecars")
    if not isinstance(sidecars, Mapping):
        raise FormalVerificationError("P7-D review sidecar provenance is missing")
    expected_roles = ("pass_a", "pass_b", "adjudication")
    if set(sidecars) != set(expected_roles):
        raise FormalVerificationError("P7-D review sidecar provenance roles mismatch")
    output: dict[str, tuple[Path, list[dict[str, Any]]]] = {}
    for role in expected_roles:
        trace = sidecars[role]
        if not isinstance(trace, Mapping):
            raise FormalVerificationError(f"P7-D review sidecar trace is invalid: {role}")
        sidecar_path = _resolve_repo_path(str(trace.get("path") or ""))
        if not sidecar_path.exists():
            raise FormalVerificationError(f"P7-D review sidecar is missing: {role}")
        rows = _read_csv(sidecar_path)
        actual = _review_sidecar_trace(sidecar_path, rows, role)
        if dict(trace) != actual:
            raise FormalVerificationError(f"P7-D review sidecar trace mismatch: {role}")
        output[role] = (sidecar_path, rows)
    return output


def _validate_candidate_adjudication_for_verify(
    paths: Mapping[str, Path],
    candidates: list[Mapping[str, Any]],
    adjudicated: list[Mapping[str, Any]],
    contract: Mapping[str, Any],
    review_sidecars: Mapping[str, tuple[Path, list[dict[str, Any]]]] | None = None,
) -> list[dict[str, Any]]:
    reviewable = _candidate_by_id(_review_candidates_from_inventory(candidates))
    if set(reviewable) != {_candidate_key(row) for row in adjudicated} or len(reviewable) != len(adjudicated):
        raise FormalVerificationError("P7-D candidate adjudication universe mismatch")
    if review_sidecars is None:
        review_workspace = paths["output_dir"].parent / "p7_d_review_workspace"
        sidecar_paths = {
            "pass_a": review_workspace / "review_pass_a.csv",
            "pass_b": review_workspace / "review_pass_b.csv",
            "adjudication": review_workspace / "adjudication.csv",
        }
        if any(not path.exists() for path in sidecar_paths.values()):
            raise FormalVerificationError("P7-D review artifacts required for verify-existing are missing")
        sidecar_rows = {role: _read_csv(path) for role, path in sidecar_paths.items()}
    else:
        sidecar_rows = {role: rows for role, (_, rows) in review_sidecars.items()}
    reviewable_keys = set(reviewable)
    pass_a = _filter_review_rows_by_keys(sidecar_rows["pass_a"], reviewable_keys)
    pass_b = _filter_review_rows_by_keys(sidecar_rows["pass_b"], reviewable_keys)
    adjudication_input = _filter_review_rows_by_keys(sidecar_rows["adjudication"], reviewable_keys)
    validate_candidate_review_pass(pass_a, reviewable, contract, "A")
    validate_candidate_review_pass(pass_b, reviewable, contract, "B")
    validate_dual_review_independence(pass_a, pass_b)
    rebuilt = _build_review_adjudication_rows(reviewable, pass_a, pass_b, adjudication_input, contract)
    _assert_rows_equal("03 adjudication rebuild", adjudicated, _project_csv_fields(rebuilt, _adjudication_fields()))
    return rebuilt


def _validate_label_phase_rebuild(
    contract: Mapping[str, Any],
    paths: Mapping[str, Path],
    manifest: list[Mapping[str, Any]],
    candidates: list[Mapping[str, Any]],
    adjudicated: list[Mapping[str, Any]],
    diagnostics: list[Mapping[str, Any]],
    summary: Mapping[str, Any],
    provenance: Mapping[str, Any],
) -> None:
    label_source = str(provenance.get("label_source_path") or "")
    if not label_source:
        raise FormalVerificationError("P7-D label source path missing")
    label_path = _resolve_repo_path(label_source)
    if not label_path.exists():
        raise FormalVerificationError("P7-D label source path does not exist")
    label_sha = sha256_file(label_path)
    if label_sha != provenance.get("label_source_sha256") or label_sha != summary.get("label_source_sha256"):
        raise FormalVerificationError("P7-D label source SHA mismatch")
    labels = _read_csv(label_path)
    if len(labels) != contract["product_universe_count"] or len({str(row.get("parent_asin") or "") for row in labels}) != contract["product_universe_count"]:
        raise FormalVerificationError("P7-D label source universe mismatch")
    columns = _label_columns(labels)
    rebuilt_diagnostics = build_label_diagnostics(manifest, labels, columns)
    _assert_rows_equal("05 label diagnostics rebuild", diagnostics, rebuilt_diagnostics)
    pre_sha = str(provenance.get("pre_label_final_image_manifest_sha256") or "")
    post_sha = str(provenance.get("post_label_final_image_manifest_sha256") or "")
    rebuilt_summary = build_summary(contract, manifest, candidates, adjudicated, diagnostics, columns, provenance, summary.get("label_source_path", label_source), label_sha, pre_sha, post_sha)
    _assert_json_equal("06 summary rebuild", summary, rebuilt_summary)
    if "has_any_outer_imagery_observed" in columns:
        observed = sum(str(row.get("has_any_outer_imagery_observed") or "").lower() in {"1", "true", "yes"} for row in diagnostics)
        if observed != 232:
            raise FormalVerificationError("P7-D outer main observed-positive sanity check failed")

def verify_existing(contract: Mapping[str, Any]) -> dict[str, Any]:
    paths = _paths(contract)
    required = tuple(paths)
    if any(not paths[key].exists() for key in required):
        raise FormalVerificationError("missing P7-D formal output")
    verify_upstream(contract)
    exception_rows = _read_csv(paths["exception_inventory"])
    candidates = _read_csv(paths["candidate_inventory"])
    adjudicated = _read_csv(paths["candidate_adjudication"])
    manifest = _read_csv(paths["final_manifest"])
    diagnostics = _read_csv(paths["label_diagnostics"])
    summary = json.loads(paths["summary"].read_text(encoding="utf-8"))
    provenance = json.loads(paths["provenance"].read_text(encoding="utf-8"))
    _validate_no_temp_artifacts(paths["output_dir"])
    if len(exception_rows) != contract["product_universe_count"] or len(manifest) != contract["product_universe_count"] or len({row["parent_asin"] for row in manifest}) != contract["product_universe_count"]:
        raise FormalVerificationError("P7-D manifest universe mismatch")
    if sum(_as_bool(row.get("primary_asset_substituted")) for row in manifest) != 0:
        raise FormalVerificationError("P7-D primary substitution invariant failed")
    p7c_manifest, p7b_manifest, p7a_inventory = _load_p7_rows()
    expected_exception_rows = _rebuild_exception_inventory_for_verify(contract, p7c_manifest, p7b_manifest)
    _assert_rows_equal("01 exception inventory rebuild", exception_rows, expected_exception_rows)
    validate_primary_fields_match_p7c(manifest, p7c_manifest)
    _validate_candidate_source_linkage(exception_rows, candidates, p7b_manifest, p7a_inventory, contract)
    _validate_candidate_asset_integrity(exception_rows, candidates, contract)
    review_sidecars = _validate_review_sidecar_provenance(provenance)
    rebuilt_adjudicated = _validate_candidate_adjudication_for_verify(paths, candidates, adjudicated, contract, review_sidecars)
    validate_s3_escalation_set(build_sensitivity_queue(exception_rows), rebuilt_adjudicated, candidates)
    rebuilt_manifest = _build_final_manifest(exception_rows, rebuilt_adjudicated, contract)
    _assert_rows_equal("04 final manifest rebuild", manifest, rebuilt_manifest)
    if len(diagnostics) != len(manifest) or len({row["parent_asin"] for row in diagnostics}) != len(manifest):
        raise FormalVerificationError("P7-D label diagnostics join mismatch")
    if provenance.get("pre_label_final_image_manifest_sha256") != provenance.get("post_label_final_image_manifest_sha256") or provenance.get("post_label_final_image_manifest_sha256") != sha256_file(paths["final_manifest"]):
        raise FormalVerificationError("P7-D pre/post label manifest SHA mismatch")
    if provenance.get("review_prompt_git_blob_sha256") != git_blob_sha256(provenance["formal_run_git_commit"], provenance["review_prompt_path"]):
        raise FormalVerificationError("P7-D historical prompt blob mismatch")
    if provenance.get("review_prompt_git_blob_sha256") != git_blob_sha256("HEAD", provenance["review_prompt_path"]):
        raise FormalVerificationError("P7-D current prompt blob mismatch")
    if provenance.get("producer_script_git_blob_sha256") != git_blob_sha256(provenance["formal_run_git_commit"], provenance["producer_script_path"]):
        raise FormalVerificationError("P7-D producer script blob mismatch")
    phase1_commit = str(provenance.get("phase1_producer_commit") or "")
    phase2_commit = str(provenance.get("phase2_diagnostics_producer_commit") or "")
    if not phase1_commit or phase1_commit != str(provenance.get("formal_run_git_commit") or ""):
        raise FormalVerificationError("P7-D Phase-1 producer commit binding mismatch")
    if provenance.get("phase1_producer_script_git_blob_sha256") != git_blob_sha256(phase1_commit, provenance["producer_script_path"]):
        raise FormalVerificationError("P7-D Phase-1 producer script blob mismatch")
    if not phase2_commit:
        raise FormalVerificationError("P7-D Phase-2 diagnostics producer commit is missing")
    if provenance.get("phase2_diagnostics_script_git_blob_sha256") != git_blob_sha256(phase2_commit, provenance["producer_script_path"]):
        raise FormalVerificationError("P7-D Phase-2 diagnostics producer script blob mismatch")
    if provenance.get("contract_git_blob_sha256") != git_blob_sha256(provenance["formal_run_git_commit"], provenance["contract_path"]):
        raise FormalVerificationError("P7-D contract blob mismatch")
    formal_sha = {name: sha256_file(paths[key]) for name, key in (("01_exception_inventory.csv", "exception_inventory"), ("02_sensitivity_candidate_inventory.csv", "candidate_inventory"), ("03_sensitivity_candidate_adjudication.csv", "candidate_adjudication"), ("04_final_image_manifest.csv", "final_manifest"), ("05_postfreeze_label_diagnostics.csv", "label_diagnostics"), ("06_p7_d_summary.json", "summary"))}
    if provenance.get("formal_output_sha256") != formal_sha:
        raise FormalVerificationError("P7-D formal output SHA mismatch")
    if "07_p7_d_provenance.json" in provenance.get("formal_output_sha256", {}):
        raise FormalVerificationError("P7-D provenance records its own SHA")
    if provenance.get("contract_git_blob_sha256") != git_blob_sha256("HEAD", provenance["contract_path"]):
        raise FormalVerificationError("P7-D current contract blob mismatch")
    _validate_label_phase_rebuild(contract, paths, manifest, candidates, adjudicated, diagnostics, summary, provenance)
    return {"verification": "PASS", "final_manifest_rows": len(manifest), "unique_parent_asin": len({row["parent_asin"] for row in manifest}), "primary_asset_substitution_count": 0, "pre_post_manifest_sha_equal": True}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare-exceptions", "acquire-candidates", "acquire-s3-candidates", "write-review-template", "freeze-sensitivity", "diagnose-labels"), nargs="?")
    parser.add_argument("--pass-a", type=Path)
    parser.add_argument("--pass-b", type=Path)
    parser.add_argument("--adjudication", type=Path)
    parser.add_argument("--label-source", type=Path)
    parser.add_argument("--verify-existing", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    contract = load_contract()
    if args.verify_existing:
        print(json.dumps(verify_existing(contract), ensure_ascii=False, indent=2))
        return 0
    if args.command == "prepare-exceptions":
        print(json.dumps(prepare_exceptions(contract), ensure_ascii=False, indent=2))
    elif args.command == "acquire-candidates":
        print(json.dumps(acquire_candidates(contract), ensure_ascii=False, indent=2))
    elif args.command == "acquire-s3-candidates":
        if not args.pass_a or not args.pass_b or not args.adjudication:
            raise FormalVerificationError("acquire-s3-candidates requires --pass-a --pass-b --adjudication")
        print(json.dumps(acquire_s3_candidates(contract, args.pass_a, args.pass_b, args.adjudication), ensure_ascii=False, indent=2))
    elif args.command == "write-review-template":
        print(json.dumps(write_review_template(contract), ensure_ascii=False, indent=2))
    elif args.command == "freeze-sensitivity":
        if not args.pass_a or not args.pass_b or not args.adjudication:
            raise FormalVerificationError("freeze-sensitivity requires --pass-a --pass-b --adjudication")
        print(json.dumps(freeze_sensitivity(contract, args.pass_a, args.pass_b, args.adjudication), ensure_ascii=False, indent=2))
    elif args.command == "diagnose-labels":
        if not args.label_source:
            raise FormalVerificationError("diagnose-labels requires --label-source")
        print(json.dumps(diagnose_labels(contract, args.label_source), ensure_ascii=False, indent=2))
    else:
        raise FormalVerificationError("choose a P7-D command or --verify-existing")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FormalVerificationError, json.JSONDecodeError) as exc:
        print(f"P7-D ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
