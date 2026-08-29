#!/usr/bin/env python3
"""P7-A: read-only source/identity inventory audit for the frozen 5,180-product image universe.

Reads the frozen primary product input and the frozen raw Amazon Reviews'23 Grocery
metadata source, independently reconstructs the historical ``extract_main_image()``
selection semantics (``screen_full_metadata_v2.py``), resolves source-record identity,
and produces a label-blind static URL/format/duplicate inventory.

The script never reads labels, never touches the network, never downloads images,
and never modifies any frozen upstream file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

CONTRACT_VERSION = "p7_v1.2"

EXPECTED_INPUT_SHA256 = "CDF80D7BA5E917982DDCBEC5153C6115CBD1BC5A5B5C158B382D614DFF94A08B"

DEFAULT_INPUT = Path("data") / "processed" / "review_matching_5180" / "01_valid_products.csv"
DEFAULT_RAW_METADATA = (
    Path("data") / "meta_Grocery_and_Gourmet_Food.jsonl" / "meta_Grocery_and_Gourmet_Food.jsonl"
)
DEFAULT_OUTPUT_DIR = (
    Path("data") / "processed" / "retail_outer_package_images_p7_5180" / "p7_a_source_inventory"
)
CONTRACT_PATH = Path("config") / "image_assets" / "p7_source_contract.json"

OUTPUT_FILES = (
    "01_source_identity_inventory.csv",
    "02_p7_a_summary.json",
    "03_p7_a_provenance.json",
)

INVENTORY_COLUMNS = (
    "parent_asin",
    "source_record_count",
    "source_record_index",
    "source_image_variant",
    "source_image_position",
    "source_size_role",
    "main_role_verified",
    "selection_fallback_used",
    "reconstructed_selected_url",
    "existing_main_image_url",
    "selected_url_matches_existing",
    "source_record_sha256",
    "identity_resolution_status",
    "image_role",
    "temporal_alignment_status",
    "main_url_duplicate_group_id",
    "main_url_duplicate_group_size",
    "main_url_format_hint",
    "main_url_is_gif",
    "main_url_has_amazon_size_transform",
    "all_image_url_count",
)

IDENTITY_STATUSES = (
    "exact_single_parent_record",
    "exact_unique_image_match",
    "missing_source_record",
    "ambiguous_multiple_image_matches",
    "source_url_mismatch",
)

SIZE_ROLES = ("hi_res", "large", "thumb")

AMAZON_SIZE_TRANSFORM_RE = re.compile(r"\._[A-Za-z0-9_]+_\.")


class ContractError(RuntimeError):
    """Raised when the frozen P7 source contract is not satisfied."""


def load_contract(path: Path | str) -> dict[str, Any]:
    """Load the frozen machine-readable contract and hard-gate its invariants."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("contract_version") != CONTRACT_VERSION:
        raise ContractError(
            f"contract version mismatch: expected {CONTRACT_VERSION}, got {payload.get('contract_version')}"
        )
    if payload.get("primary_input_sha256") != EXPECTED_INPUT_SHA256:
        raise ContractError(
            "contract primary_input_sha256 does not match the frozen expected input SHA"
        )
    required_enums = (
        "image_role_enum",
        "identity_resolution_status_enum",
        "source_size_role_enum",
        "temporal_definition",
    )
    for key in required_enums:
        if key not in payload:
            raise ContractError(f"contract missing required section: {key}")
    if not isinstance(payload["image_role_enum"], list) or not payload["image_role_enum"]:
        raise ContractError("contract image_role_enum must be a non-empty list")
    if not isinstance(payload["identity_resolution_status_enum"], list) or not payload[
        "identity_resolution_status_enum"
    ]:
        raise ContractError("contract identity_resolution_status_enum must be a non-empty list")
    if not isinstance(payload["source_size_role_enum"], list) or not payload["source_size_role_enum"]:
        raise ContractError("contract source_size_role_enum must be a non-empty list")
    if not isinstance(payload["temporal_definition"].get("temporal_alignment_status_enum"), list):
        raise ContractError("contract temporal_definition.temporal_alignment_status_enum must be a list")
    output_schema = payload.get("p7_a_output_schema")
    if not isinstance(output_schema, dict):
        raise ContractError("contract p7_a_output_schema must be an object")
    contract_columns = output_schema.get("inventory_columns")
    if not isinstance(contract_columns, list):
        raise ContractError("contract p7_a_output_schema.inventory_columns must be a list")
    if tuple(contract_columns) != INVENTORY_COLUMNS:
        raise ContractError(
            "contract p7_a_output_schema.inventory_columns does not match the script INVENTORY_COLUMNS: "
            f"contract={contract_columns} script={list(INVENTORY_COLUMNS)}"
        )
    sanity = payload.get("prior_audit_sanity_expected")
    if not isinstance(sanity, dict) or not sanity:
        raise ContractError("contract prior_audit_sanity_expected must be a non-empty object")
    return payload


def validate_inventory_enums(
    inventory: list[dict[str, Any]], contract: dict[str, Any]
) -> None:
    """Mechanically verify every generated row uses contract enum values."""
    identity_enum = set(contract.get("identity_resolution_status_enum", []))
    size_role_enum = set(contract.get("source_size_role_enum", []))
    image_role_enum = set(contract.get("image_role_enum", []))
    temporal_enum = set(
        contract.get("temporal_definition", {}).get("temporal_alignment_status_enum", [])
    )
    violations: list[str] = []
    for row in inventory:
        status = row["identity_resolution_status"]
        if status not in identity_enum:
            violations.append(
                f"{row['parent_asin']}: invalid identity_resolution_status {status!r}"
            )
        size_role = row["source_size_role"]
        if size_role not in size_role_enum:
            violations.append(
                f"{row['parent_asin']}: invalid source_size_role {size_role!r}"
            )
        image_role = row["image_role"]
        if image_role not in image_role_enum:
            violations.append(f"{row['parent_asin']}: invalid image_role {image_role!r}")
        temporal = row["temporal_alignment_status"]
        if temporal not in temporal_enum:
            violations.append(
                f"{row['parent_asin']}: invalid temporal_alignment_status {temporal!r}"
            )
    if violations:
        preview = "\n".join(violations[:20])
        raise ContractError(
            f"inventory rows violate contract enums ({len(violations)} total):\n{preview}"
        )


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest().upper()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def strip_line_ending(line: bytes) -> bytes:
    """Remove only terminal CR/LF (in any combination) from the exact line bytes."""
    if line.endswith(b"\n"):
        line = line[:-1]
    if line.endswith(b"\r"):
        line = line[:-1]
    return line


def reconstruct_selected_image(images: Any) -> dict[str, Any]:
    """Independently reconstruct the historical ``extract_main_image`` selection.

    Semantics frozen from ``scripts/metadata/screen_full_metadata_v2.py``:

    - per image item: usable URLs in order hi_res -> large -> thumb;
    - usable = string, non-blank after strip;
    - if any item declares variant == MAIN: selected = first usable MAIN URL;
    - else: selected = first usable URL among all image items;
    - all_urls keeps first occurrence only (order preserved).
    """
    if not isinstance(images, list):
        return {
            "selected_url": "",
            "all_urls": [],
            "variant": "",
            "position": -1,
            "size_role": "none",
            "main_role_verified": False,
            "fallback_used": False,
        }

    all_urls: list[str] = []
    seen: set[str] = set()
    main_candidates: list[tuple[str, str, str, int]] = []  # (url, variant, size_role, position)
    fallback_candidates: list[tuple[str, str, str, int]] = []

    for position, item in enumerate(images):
        if not isinstance(item, Mapping):
            continue
        variant = str(item.get("variant") or "").upper()
        usable: list[tuple[str, str]] = []
        for size_role in SIZE_ROLES:
            url = item.get(size_role)
            if isinstance(url, str) and url.strip():
                usable.append((url.strip(), size_role))
        for url, size_role in usable:
            if url not in seen:
                seen.add(url)
                all_urls.append(url)
            if variant == "MAIN":
                main_candidates.append((url, variant, size_role, position))
            fallback_candidates.append((url, variant, size_role, position))

    if main_candidates:
        selected_url, variant, size_role, position = main_candidates[0]
        fallback_used = False
    elif fallback_candidates:
        selected_url, variant, size_role, position = fallback_candidates[0]
        fallback_used = True
    else:
        return {
            "selected_url": "",
            "all_urls": [],
            "variant": "",
            "position": -1,
            "size_role": "none",
            "main_role_verified": False,
            "fallback_used": False,
        }

    return {
        "selected_url": selected_url,
        "all_urls": all_urls,
        "variant": variant,
        "position": position,
        "size_role": size_role,
        "main_role_verified": variant == "MAIN",
        "fallback_used": fallback_used,
    }


def url_format_hint(url: str) -> str:
    if not url:
        return "blank"
    path = url.split("?", 1)[0].lower()
    for suffix in (".jpg", ".jpeg", ".png", ".gif", ".webp", ".avif", ".bmp"):
        if path.endswith(suffix):
            return suffix.lstrip(".")
    return "other"


def main_url_is_gif(url: str) -> bool:
    return url_format_hint(url) == "gif"


def main_url_has_amazon_size_transform(url: str) -> bool:
    return bool(AMAZON_SIZE_TRANSFORM_RE.search(url))


def scan_raw_metadata(raw_path: Path, target_asins: set[str]) -> dict[str, Any]:
    """Stream the raw JSONL source and keep only records for target parent_asins."""
    collected: dict[str, list[dict[str, Any]]] = {}
    counters: Counter[str] = Counter()
    for line_number, raw_line in enumerate(raw_path.open("rb"), start=1):
        counters["lines_scanned"] += 1
        if not raw_line.strip():
            counters["blank_lines"] += 1
            continue
        counters["nonblank_lines"] += 1
        try:
            record = json.loads(raw_line.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            counters["json_error_lines"] += 1
            continue
        if not isinstance(record, Mapping):
            counters["non_object_lines"] += 1
            continue
        counters["valid_json_records"] += 1
        parent_asin = str(record.get("parent_asin") or "").strip()
        if parent_asin and parent_asin in target_asins:
            line_hash = sha256_bytes(strip_line_ending(raw_line))
            collected.setdefault(parent_asin, []).append(
                {"record": record, "line_bytes": raw_line, "line_sha256": line_hash}
            )
    return {"collected": collected, "counters": dict(counters)}


def resolve_identity(
    parent_asin: str,
    records: list[dict[str, Any]],
    existing_main_url: str,
) -> dict[str, Any]:
    """Resolve source-record identity following the frozen status enum rules."""
    if not records:
        return {"status": "missing_source_record", "index": -1, "match_count": 0}

    match_indices: list[int] = []
    for index, entry in enumerate(records):
        recon = reconstruct_selected_image(entry["record"].get("images"))
        if recon["selected_url"] == existing_main_url:
            match_indices.append(index)

    count = len(records)
    if count == 1 and match_indices:
        return {"status": "exact_single_parent_record", "index": 0, "match_count": 1}
    if len(match_indices) == 1:
        return {"status": "exact_unique_image_match", "index": match_indices[0], "match_count": 1}
    if len(match_indices) > 1:
        return {
            "status": "ambiguous_multiple_image_matches",
            "index": match_indices[0],
            "match_count": len(match_indices),
        }
    return {"status": "source_url_mismatch", "index": 0, "match_count": 0}


def load_primary_input(path: Path) -> list[dict[str, str]]:
    """Load the frozen product CSV; only metadata columns are used (label-blind)."""
    import csv

    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(
                {
                    "parent_asin": (row.get("parent_asin") or "").strip(),
                    "main_image_url": (row.get("main_image_url") or "").strip(),
                    "all_image_urls_raw": row.get("all_image_urls") or "",
                }
            )
    return rows


def verify_primary_input(rows: list[dict[str, str]]) -> dict[str, Any]:
    checks = {
        "rows": len(rows),
        "unique_parent_asin": len({r["parent_asin"] for r in rows}),
        "nonblank_main_image_url": sum(1 for r in rows if r["main_image_url"]),
        "nonblank_all_image_urls": sum(1 for r in rows if r["all_image_urls_raw"]),
    }
    return checks


def build_inventory(
    rows: list[dict[str, str]],
    collected: dict[str, list[dict[str, Any]]],
    contract: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    inventory: list[dict[str, Any]] = []
    url_to_group: dict[str, str] = {}
    row_group_ids: list[str] = []
    for row in rows:
        url = row["main_image_url"]
        normalized = url.strip()
        if normalized:
            group_id = sha256_bytes(normalized.encode("utf-8"))
            url_to_group.setdefault(normalized, group_id)
        else:
            group_id = ""
        row_group_ids.append(group_id)

    group_sizes: Counter[str] = Counter(gid for gid in row_group_ids if gid)
    # A duplicate group exists only when more than one product shares the URL.
    duplicate_group_ids = {gid for gid, size in group_sizes.items() if size > 1}
    group_ids = {
        url: (gid if gid in duplicate_group_ids else "")
        for url, gid in url_to_group.items()
    }

    for row in rows:
        parent_asin = row["parent_asin"]
        existing_main_url = row["main_image_url"]
        records = collected.get(parent_asin, [])
        resolution = resolve_identity(parent_asin, records, existing_main_url)

        if resolution["status"] == "missing_source_record":
            recon = {
                "selected_url": "",
                "all_urls": [],
                "variant": "",
                "position": -1,
                "size_role": "none",
                "main_role_verified": False,
                "fallback_used": False,
            }
            record_sha = ""
            source_record_count = 0
            source_record_index = -1
        else:
            entry = records[resolution["index"]]
            recon = reconstruct_selected_image(entry["record"].get("images"))
            record_sha = entry["line_sha256"]
            source_record_count = len(records)
            source_record_index = resolution["index"]

        selected_matches = recon["selected_url"] == existing_main_url
        all_image_urls = [u for u in row["all_image_urls_raw"].split("|") if u]

        # image_role resolves deterministically from the reconstruction.
        if selected_matches and recon["main_role_verified"]:
            image_role = "declared_main"
        elif selected_matches and recon["fallback_used"]:
            image_role = "existing_first_image_fallback"
        else:
            image_role = "none"

        # temporal status records the frozen metadata snapshot when the selected
        # URL is confirmed; unresolved rows stay unresolved.
        if selected_matches:
            temporal_alignment_status = "frozen_metadata_snapshot_2023"
        else:
            temporal_alignment_status = "unresolved"

        group_id = group_ids.get(existing_main_url.strip(), "")
        inventory.append(
            {
                "parent_asin": parent_asin,
                "source_record_count": source_record_count,
                "source_record_index": source_record_index,
                "source_image_variant": recon["variant"],
                "source_image_position": recon["position"],
                "source_size_role": recon["size_role"],
                "main_role_verified": int(recon["main_role_verified"]),
                "selection_fallback_used": int(recon["fallback_used"]),
                "reconstructed_selected_url": recon["selected_url"],
                "existing_main_image_url": existing_main_url,
                "selected_url_matches_existing": int(selected_matches),
                "source_record_sha256": record_sha,
                "identity_resolution_status": resolution["status"],
                "image_role": image_role,
                "temporal_alignment_status": temporal_alignment_status,
                "main_url_duplicate_group_id": group_id,
                "main_url_duplicate_group_size": group_sizes.get(group_id, 0),
                "main_url_format_hint": url_format_hint(existing_main_url),
                "main_url_is_gif": int(main_url_is_gif(existing_main_url)),
                "main_url_has_amazon_size_transform": int(
                    main_url_has_amazon_size_transform(existing_main_url)
                ),
                "all_image_url_count": len(all_image_urls),
            }
        )
    return inventory


def compute_summary(
    rows: list[dict[str, str]],
    inventory: list[dict[str, Any]],
    input_sha: str,
    raw_sha: str,
    inventory_sha: str,
    contract: dict[str, Any],
) -> dict[str, Any]:
    status_dist = Counter(r["identity_resolution_status"] for r in inventory)
    declared_main = sum(1 for r in inventory if r["main_role_verified"] and not r["selection_fallback_used"])
    fallback = sum(1 for r in inventory if r["selection_fallback_used"])
    unresolved_role = sum(
        1 for r in inventory if not r["selected_url_matches_existing"]
    )
    match_count = sum(1 for r in inventory if r["selected_url_matches_existing"])
    mismatch_count = len(inventory) - match_count

    all_url_counts = [r["all_image_url_count"] for r in inventory]
    all_url_counts_sorted = sorted(all_url_counts)
    total_urls = sum(all_url_counts)
    n = len(all_url_counts_sorted)
    median = (
        all_url_counts_sorted[n // 2]
        if n % 2
        else (all_url_counts_sorted[n // 2 - 1] + all_url_counts_sorted[n // 2]) / 2
    )

    group_size_counter: Counter[str] = Counter()
    for r in inventory:
        group_size_counter[r["main_url_duplicate_group_id"]] += 1
    duplicate_groups = {
        gid: size for gid, size in group_size_counter.items() if size > 1 and gid
    }
    gif_count = sum(1 for r in inventory if r["main_url_is_gif"])
    transform_count = sum(1 for r in inventory if r["main_url_has_amazon_size_transform"])
    format_dist = Counter(r["main_url_format_hint"] for r in inventory)
    blank_malformed = sum(
        1
        for r in inventory
        if not r["existing_main_image_url"] or not r["existing_main_image_url"].startswith("http")
    )

    # Deterministic data-sanity gate against the frozen prior-audit expectations.
    sanity_expected = contract.get("prior_audit_sanity_expected", {})
    actual_sanity = {
        "all_image_url_total_count": total_urls,
        "all_image_url_minimum": min(all_url_counts),
        "all_image_url_median": median,
        "all_image_url_maximum": max(all_url_counts),
        "duplicate_main_url_group_count": len(duplicate_groups),
        "largest_duplicate_group_size": max(duplicate_groups.values(), default=0),
        "gif_main_url_count": gif_count,
    }
    sanity_mismatch = {
        key: {"expected": sanity_expected.get(key), "actual": actual_sanity.get(key)}
        for key in sanity_expected
        if sanity_expected.get(key) != actual_sanity.get(key)
    }
    data_sanity_status = "PASS" if not sanity_mismatch else "FAIL"

    p7_a_status = "PASS" if mismatch_count == 0 and data_sanity_status == "PASS" else "FAIL"

    return {
        "p7_a_status": p7_a_status,
        "data_sanity_status": data_sanity_status,
        "data_sanity_mismatch": sanity_mismatch,
        "contract_version": CONTRACT_VERSION,
        "input_rows": len(rows),
        "unique_parent_asin": len({r["parent_asin"] for r in rows}),
        "input_file_sha256": input_sha,
        "raw_metadata_sha256": raw_sha,
        "identity_resolution_distribution": dict(status_dist),
        "declared_main_count": declared_main,
        "first_image_fallback_count": fallback,
        "unresolved_role_count": unresolved_role,
        "selected_url_match_count": match_count,
        "selected_url_mismatch_count": mismatch_count,
        "image_role_distribution": dict(Counter(r["image_role"] for r in inventory)),
        "temporal_alignment_status_distribution": dict(
            Counter(r["temporal_alignment_status"] for r in inventory)
        ),
        "all_image_url_statistics": {
            "total_url_count": total_urls,
            "minimum": min(all_url_counts),
            "median": median,
            "maximum": max(all_url_counts),
        },
        "duplicate_main_url_statistics": {
            "duplicate_group_count": len(duplicate_groups),
            "largest_duplicate_group_size": max(duplicate_groups.values(), default=0),
            "products_in_duplicate_groups": sum(duplicate_groups.values()),
        },
        "gif_main_url_count": gif_count,
        "amazon_size_transform_count": transform_count,
        "url_format_hint_distribution": dict(format_dist),
        "blank_or_malformed_main_url_count": blank_malformed,
        "output_file_sha256": {
            OUTPUT_FILES[0]: inventory_sha,
        },
        "label_sources_read": False,
        "network_access_used": False,
        "images_downloaded": 0,
    }


def git_blob_bytes(commit: str, repo_path: Path) -> bytes:
    """Return exact file bytes at repo_path as stored in a Git commit."""
    import subprocess

    output = subprocess.check_output(
        ["git", "show", f"{commit}:{repo_path.as_posix()}"],
        stderr=subprocess.DEVNULL,
    )
    return output


def git_blob_sha256(commit: str, repo_path: Path) -> str:
    """Hash exact Git blob bytes without checkout-specific text conversion."""
    return sha256_bytes(git_blob_bytes(commit, repo_path))


def verify_provenance_git_identity(
    provenance: Mapping[str, Any],
    *,
    current_head: str | None = None,
) -> dict[str, bool]:
    """Verify producer identity from Git blobs and the current frozen contract blob."""
    checks = {
        "producer_git_commit_known": False,
        "producer_script_git_blob_sha256_match": False,
        "producer_contract_git_blob_sha256_match": False,
        "current_head_contract_git_blob_sha256_match": False,
    }
    git_commit = str(provenance.get("git_commit") or "")
    if git_commit in ("", "unknown"):
        return checks

    checks["producer_git_commit_known"] = True
    try:
        script_path = Path("scripts") / "images" / "audit_p7_source_inventory.py"
        contract_path = Path("config") / "image_assets" / "p7_source_contract.json"
        checks["producer_script_git_blob_sha256_match"] = (
            git_blob_sha256(git_commit, script_path) == provenance.get("script_sha256")
        )
        checks["producer_contract_git_blob_sha256_match"] = (
            git_blob_sha256(git_commit, contract_path) == provenance.get("contract_sha256")
        )
        if current_head is None:
            import subprocess

            current_head = subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        checks["current_head_contract_git_blob_sha256_match"] = (
            git_blob_sha256(current_head, contract_path) == provenance.get("contract_sha256")
        )
    except Exception:
        return checks
    return checks


def build_provenance(
    input_path: Path,
    raw_path: Path,
    output_dir: Path,
    input_sha_before: str,
    input_sha_after: str,
    raw_sha_before: str,
    raw_sha_after: str,
    command_line: str,
    contract: dict[str, Any],
) -> dict[str, Any]:
    import subprocess

    try:
        git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        git_commit = "unknown"

    script_path = Path(__file__).resolve()
    contract_path = Path(CONTRACT_PATH).resolve()
    script_sha = sha256_file(script_path)
    contract_sha = sha256_file(contract_path)

    # Only 01 (inventory) and 02 (summary) are recorded; the provenance never
    # records its own 03 SHA. The 03 SHA is captured externally by hash/report.
    output_shas = {}
    for name in OUTPUT_FILES[:2]:
        target = output_dir / name
        if target.exists():
            output_shas[str(target)] = sha256_file(target)

    return {
        "git_commit": git_commit,
        "script_path": str(script_path),
        "script_sha256": script_sha,
        "contract_path": str(contract_path),
        "contract_sha256": contract_sha,
        "contract_version": CONTRACT_VERSION,
        "input_paths": {"primary_input": str(input_path), "raw_metadata": str(raw_path)},
        "input_sha256_before": input_sha_before,
        "input_sha256_after": input_sha_after,
        "raw_metadata_sha256_before": raw_sha_before,
        "raw_metadata_sha256_after": raw_sha_after,
        "execution_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": sys.version.split()[0],
        "command_line": command_line,
        "output_paths": [str(output_dir / name) for name in OUTPUT_FILES],
        "output_sha256": output_shas,
    }


def write_inventory_csv(path: Path, inventory: list[dict[str, Any]]) -> None:
    import csv

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=INVENTORY_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in inventory:
            writer.writerow(row)


def run_audit(
    input_path: Path,
    raw_path: Path,
    output_dir: Path,
    *,
    force: bool,
    command_line: str,
) -> dict[str, Any]:
    input_path = Path(input_path)
    raw_path = Path(raw_path)
    output_dir = Path(output_dir)

    contract = load_contract(CONTRACT_PATH)
    contract_raw = contract.get("raw_metadata", {})

    for name in OUTPUT_FILES:
        if (output_dir / name).exists() and not force:
            raise SystemExit(
                f"Refusing to overwrite existing formal output: {output_dir / name}\n"
                "Use --verify-existing to validate, or --force only for a documented re-run."
            )

    input_sha_before = sha256_file(input_path)
    raw_sha_before = sha256_file(raw_path)
    if input_sha_before != EXPECTED_INPUT_SHA256:
        raise SystemExit(
            f"P7-A FAIL: primary input SHA mismatch.\n"
            f"expected {EXPECTED_INPUT_SHA256}\ngot      {input_sha_before}"
        )
    if raw_sha_before != contract_raw.get("sha256"):
        raise SystemExit(
            f"P7-A FAIL: raw metadata SHA does not match the frozen contract.\n"
            f"contract {contract_raw.get('sha256')}\ngot      {raw_sha_before}"
        )

    rows = load_primary_input(input_path)
    checks = verify_primary_input(rows)
    expected_checks = {
        "rows": 5180,
        "unique_parent_asin": 5180,
        "nonblank_main_image_url": 5180,
        "nonblank_all_image_urls": 5180,
    }
    if checks != expected_checks:
        raise SystemExit(f"P7-A FAIL: primary input structural check mismatch.\n{checks}")

    target_asins = {r["parent_asin"] for r in rows}
    scanned = scan_raw_metadata(raw_path, target_asins)
    collected = scanned["collected"]

    inventory = build_inventory(rows, collected, contract=contract)
    validate_inventory_enums(inventory, contract)

    output_dir.mkdir(parents=True, exist_ok=True)
    inventory_path = output_dir / OUTPUT_FILES[0]
    summary_path = output_dir / OUTPUT_FILES[1]
    provenance_path = output_dir / OUTPUT_FILES[2]

    write_inventory_csv(inventory_path, inventory)

    input_sha_after = sha256_file(input_path)
    raw_sha_after = sha256_file(raw_path)
    if input_sha_before != input_sha_after or raw_sha_before != raw_sha_after:
        raise SystemExit("P7-A FAIL: input or raw metadata changed during run.")

    inventory_sha = sha256_file(inventory_path)
    summary = compute_summary(rows, inventory, input_sha_after, raw_sha_after, inventory_sha, contract)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    provenance = build_provenance(
        input_path,
        raw_path,
        output_dir,
        input_sha_before,
        input_sha_after,
        raw_sha_before,
        raw_sha_after,
        command_line,
        contract,
    )
    provenance_path.write_text(json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "checks": checks,
        "summary": summary,
        "provenance": provenance,
        "scan_counters": scanned["counters"],
    }


def verify_existing(input_path: Path, raw_path: Path, output_dir: Path) -> dict[str, Any]:
    """Recompute source facts and compare against existing formal outputs (zero writes)."""
    input_path = Path(input_path)
    raw_path = Path(raw_path)
    output_dir = Path(output_dir)

    contract = load_contract(CONTRACT_PATH)

    missing = [name for name in OUTPUT_FILES if not (output_dir / name).exists()]
    if missing:
        raise SystemExit(f"P7-A verify-existing FAIL: missing formal outputs {missing}")

    rows = load_primary_input(input_path)
    target_asins = {r["parent_asin"] for r in rows}
    collected = scan_raw_metadata(raw_path, target_asins)["collected"]
    inventory = build_inventory(rows, collected, contract=contract)
    validate_inventory_enums(inventory, contract)

    inventory_path = output_dir / OUTPUT_FILES[0]
    summary_path = output_dir / OUTPUT_FILES[1]
    provenance_path = output_dir / OUTPUT_FILES[2]

    import csv

    existing_rows: list[dict[str, Any]] = []
    with inventory_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            normalized = {
                k: (int(v) if k in (
                    "source_record_count",
                    "source_record_index",
                    "source_image_position",
                    "main_role_verified",
                    "selection_fallback_used",
                    "selected_url_matches_existing",
                    "main_url_duplicate_group_size",
                    "main_url_is_gif",
                    "main_url_has_amazon_size_transform",
                    "all_image_url_count",
                ) else v)
                for k, v in row.items()
            }
            existing_rows.append(normalized)

    recomputed = [
        {k: r[k] for k in INVENTORY_COLUMNS}
        for r in inventory
    ]
    inventory_match = existing_rows == recomputed
    if not inventory_match:
        for i, (a, b) in enumerate(zip(existing_rows, recomputed)):
            if a != b:
                diffs = {k for k in a if a[k] != b[k]}
                print(f"  first diff at row {i}: fields={sorted(diffs)}")
                break

    existing_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    recomputed_summary = compute_summary(
        rows,
        inventory,
        sha256_file(input_path),
        sha256_file(raw_path),
        sha256_file(inventory_path),
        contract,
    )
    summary_match = existing_summary == recomputed_summary

    # Provenance identity is defined by exact Git blob bytes, not checkout text bytes.
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    provenance_checks = verify_provenance_git_identity(provenance)
    provenance_checks.update({
        "contract_version": provenance.get("contract_version") == CONTRACT_VERSION,
        "input_sha_before": provenance.get("input_sha256_before") == EXPECTED_INPUT_SHA256,
        "input_sha_after": provenance.get("input_sha256_after") == sha256_file(input_path),
        "raw_sha_before": provenance.get("raw_metadata_sha256_before")
        == contract.get("raw_metadata", {}).get("sha256"),
        "raw_sha_after": provenance.get("raw_metadata_sha256_after") == sha256_file(raw_path),
        "inventory_output_sha": provenance.get("output_sha256", {}).get(
            str(inventory_path)
        ) == sha256_file(inventory_path),
        "summary_output_sha": provenance.get("output_sha256", {}).get(
            str(summary_path)
        ) == sha256_file(summary_path),
        "provenance_does_not_record_own_sha": str(provenance_path)
        not in provenance.get("output_sha256", {}),
    })
    provenance_match = all(provenance_checks.values())

    return {
        "inventory_match": inventory_match,
        "summary_match": summary_match,
        "provenance_match": provenance_match,
        "provenance_checks": provenance_checks,
    }


def export_git_summary(summary_path: Path, git_summary_path: Path) -> None:
    """Programmatically export a sanitized Git-visible aggregate snapshot (no URLs/rows)."""
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    allowed = (
        "p7_a_status",
        "data_sanity_status",
        "data_sanity_mismatch",
        "contract_version",
        "input_rows",
        "unique_parent_asin",
        "input_file_sha256",
        "raw_metadata_sha256",
        "identity_resolution_distribution",
        "declared_main_count",
        "first_image_fallback_count",
        "unresolved_role_count",
        "selected_url_match_count",
        "selected_url_mismatch_count",
        "image_role_distribution",
        "temporal_alignment_status_distribution",
        "all_image_url_statistics",
        "duplicate_main_url_statistics",
        "gif_main_url_count",
        "amazon_size_transform_count",
        "url_format_hint_distribution",
        "blank_or_malformed_main_url_count",
        "output_file_sha256",
        "label_sources_read",
        "network_access_used",
        "images_downloaded",
    )
    sanitized = {k: summary[k] for k in allowed if k in summary}
    git_summary_path.write_text(
        json.dumps(sanitized, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="P7-A: label-blind source/identity inventory audit (read-only, no network)."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--raw-metadata", type=Path, default=DEFAULT_RAW_METADATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--force", action="store_true", help="allow overwriting existing formal outputs")
    parser.add_argument(
        "--verify-existing",
        action="store_true",
        help="recompute source facts and compare with existing formal outputs; writes zero bytes",
    )
    parser.add_argument(
        "--export-git-summary",
        type=Path,
        default=None,
        metavar="PATH",
        help="programmatically export the sanitized Git-visible aggregate summary",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    command_line = " ".join(sys.argv)
    if args.verify_existing:
        result = verify_existing(args.input, args.raw_metadata, args.output_dir)
        ok = (
            result["inventory_match"]
            and result["summary_match"]
            and result["provenance_match"]
        )
        print(json.dumps(
            {"verification": "PASS" if ok else "FAIL", **result},
            ensure_ascii=False,
            indent=2,
        ))
        return 0 if ok else 1
    if args.export_git_summary is not None:
        summary_path = Path(args.output_dir) / OUTPUT_FILES[1]
        if not summary_path.exists():
            raise SystemExit(f"formal summary not found: {summary_path}")
        export_git_summary(summary_path, Path(args.export_git_summary))
        print(f"Git-visible sanitized summary written to {args.export_git_summary}")
        return 0
    result = run_audit(
        args.input,
        args.raw_metadata,
        args.output_dir,
        force=args.force,
        command_line=command_line,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
