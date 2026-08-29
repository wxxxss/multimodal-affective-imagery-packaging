#!/usr/bin/env python3
"""P7-B frozen declared-MAIN image acquisition and canonical asset validation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import re
import statistics
import subprocess
import sys
import time
import uuid
import warnings
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from io import BytesIO
from pathlib import Path
from typing import Any, Callable, Mapping

import requests
from PIL import Image, ImageFile, UnidentifiedImageError

ImageFile.LOAD_TRUNCATED_IMAGES = False

CONTRACT_PATH = Path("config") / "image_assets" / "p7_b_asset_contract.json"
DEFAULT_OUTPUT_DIR = Path("data") / "processed" / "retail_outer_package_images_p7_5180" / "p7_b_primary_assets"
DEFAULT_ASSET_ROOT = Path("data") / "images" / "retail_outer_package_p7_5180"
P7_A_AUDIT_SCRIPT = Path("scripts") / "images" / "audit_p7_source_inventory.py"

PRODUCT_MANIFEST_COLUMNS = (
    "parent_asin", "input_order", "source_record_sha256", "image_role",
    "temporal_alignment_status", "eligibility_status", "requested_url",
    "final_url", "redirect_count", "attempt_count", "http_status",
    "response_content_type", "response_content_length_header", "response_etag",
    "response_last_modified", "response_byte_count", "response_sha256",
    "decoded_format", "width", "height", "n_frames", "download_status",
    "decode_status", "final_asset_status", "asset_path",
    "exact_duplicate_asset_id", "exact_duplicate_asset_group_size",
    "retry_after_seconds", "error_class", "error_detail", "retrieved_at_utc",
)
UNIQUE_ASSET_COLUMNS = (
    "asset_sha256", "asset_path", "decoded_format", "width", "height",
    "n_frames", "response_byte_count", "product_count", "source_url_count",
)
INTEGER_FIELDS = {
    "input_order", "redirect_count", "attempt_count", "http_status",
    "response_byte_count", "width", "height", "n_frames",
    "exact_duplicate_asset_group_size",
}
FLOAT_FIELDS = {"retry_after_seconds"}
FORMAT_EXTENSIONS = {
    "JPEG": "jpg", "JPG": "jpg", "PNG": "png", "GIF": "gif",
    "WEBP": "webp", "BMP": "bmp", "TIFF": "tif", "TIF": "tif",
    "ICO": "ico", "PPM": "ppm", "PGM": "pgm", "PBM": "pbm",
    "PCX": "pcx", "XBM": "xbm",
}
MIME_FORMATS = {
    "image/jpeg": {"JPEG", "JPG"}, "image/jpg": {"JPEG", "JPG"},
    "image/png": {"PNG"}, "image/gif": {"GIF"}, "image/webp": {"WEBP"},
    "image/bmp": {"BMP"}, "image/tiff": {"TIFF", "TIF"},
    "image/x-icon": {"ICO"},
}

class ContractError(RuntimeError):
    pass

class IntegrityError(RuntimeError):
    pass

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

def git_blob_bytes(commit: str, repo_path: Path) -> bytes:
    return subprocess.check_output(
        ["git", "show", f"{commit}:{repo_path.as_posix()}"],
        stderr=subprocess.DEVNULL,
    )

def git_blob_sha256(commit: str, repo_path: Path) -> str:
    return sha256_bytes(git_blob_bytes(commit, repo_path))

def git_head() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
    ).strip()

def load_contract(path: Path | str = CONTRACT_PATH) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    required = (
        "contract_version", "upstream_p7_a", "product_universe_count",
        "primary_eligible_count", "excluded_non_declared_main_count",
        "eligible_image_role", "excluded_image_role",
        "primary_temporal_alignment_status", "network_policy",
        "response_limits", "storage_policy", "checkpoint_policy",
        "eligibility_status_enum", "download_status_enum",
        "decode_status_enum", "final_asset_status_enum",
        "product_manifest_columns", "unique_asset_inventory_columns",
        "formal_output_paths", "provenance_policy",
    )
    missing = [key for key in required if key not in payload]
    if missing:
        raise ContractError(f"contract missing required sections: {missing}")
    if tuple(payload["product_manifest_columns"]) != PRODUCT_MANIFEST_COLUMNS:
        raise ContractError("product_manifest_columns do not match script schema")
    if tuple(payload["unique_asset_inventory_columns"]) != UNIQUE_ASSET_COLUMNS:
        raise ContractError("unique_asset_inventory_columns do not match script schema")
    if payload["product_universe_count"] != payload["primary_eligible_count"] + payload["excluded_non_declared_main_count"]:
        raise ContractError("product universe does not equal eligibility accounting")
    for key in ("eligibility_status_enum", "download_status_enum", "decode_status_enum", "final_asset_status_enum"):
        values = payload[key]
        if not isinstance(values, list) or not values or len(values) != len(set(values)):
            raise ContractError(f"{key} must be a unique non-empty list")
    policy = payload["network_policy"]
    if policy.get("method") != "GET" or policy.get("max_workers") != 8:
        raise ContractError("network method/concurrency policy is not frozen")
    if policy.get("max_attempts") != 4 or policy.get("max_redirects") != 5:
        raise ContractError("retry/redirect policy is not frozen")
    if tuple(policy.get("backoff_seconds", [])) != (1, 2, 4):
        raise ContractError("backoff policy is not frozen")
    limits = payload["response_limits"]
    if limits.get("maximum_response_bytes") != 26214400 or limits.get("maximum_decoded_pixels") != 100000000:
        raise ContractError("response limits are not frozen")
    if not payload["provenance_policy"].get("provenance_does_not_record_own_sha"):
        raise ContractError("provenance self-SHA rule is not frozen")
    return payload

def validate_upstream_files(contract: Mapping[str, Any]) -> dict[str, Any]:
    upstream = contract["upstream_p7_a"]
    for path_key, sha_key in (
        ("inventory_path", "inventory_sha256"),
        ("summary_path", "summary_sha256"),
        ("provenance_path", "provenance_sha256"),
        ("input_product_path", "input_product_sha256"),
    ):
        path = Path(upstream[path_key])
        if not path.exists():
            raise FormalVerificationError(f"missing frozen upstream file: {path}")
        actual = sha256_file(path)
        if actual != upstream[sha_key]:
            raise FormalVerificationError(f"upstream SHA mismatch for {path}: {actual}")
    summary = json.loads(Path(upstream["summary_path"]).read_text(encoding="utf-8"))
    if summary.get("p7_a_status") != "PASS" or summary.get("data_sanity_status") != "PASS":
        raise FormalVerificationError("P7-A frozen summary is not PASS")
    if (summary.get("input_rows"), summary.get("unique_parent_asin")) != (5180, 5180):
        raise FormalVerificationError("P7-A universe count mismatch")
    if (summary.get("declared_main_count"), summary.get("first_image_fallback_count")) != (5179, 1):
        raise FormalVerificationError("P7-A eligibility count mismatch")
    return summary

def load_p7_a_inventory(contract: Mapping[str, Any]) -> list[dict[str, str]]:
    path = Path(contract["upstream_p7_a"]["inventory_path"])
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))

def _default_value(field: str) -> Any:
    if field in INTEGER_FIELDS:
        return 0
    if field in FLOAT_FIELDS:
        return ""
    return ""

def build_product_manifest_skeleton(
    inventory_rows: list[Mapping[str, Any]], contract: Mapping[str, Any]
) -> list[dict[str, Any]]:
    products = []
    for input_order, source in enumerate(inventory_rows):
        row = {field: _default_value(field) for field in PRODUCT_MANIFEST_COLUMNS}
        row.update({
            "parent_asin": str(source.get("parent_asin") or ""),
            "input_order": input_order,
            "source_record_sha256": str(source.get("source_record_sha256") or ""),
            "image_role": str(source.get("image_role") or ""),
            "temporal_alignment_status": str(source.get("temporal_alignment_status") or ""),
        })
        existing = str(source.get("existing_main_image_url") or "")
        reconstructed = str(source.get("reconstructed_selected_url") or "")
        selected_match_value = source.get("selected_url_matches_existing")
        matched = (
            existing == reconstructed
            if selected_match_value in (None, "")
            else str(selected_match_value) in {"1", "True", "true"}
        )
        if (
            row["image_role"] == contract["eligible_image_role"]
            and row["temporal_alignment_status"] == contract["primary_temporal_alignment_status"]
            and matched and existing == reconstructed and existing
        ):
            row.update({
                "eligibility_status": "primary_declared_main",
                "requested_url": existing,
                "download_status": "not_attempted",
                "decode_status": "not_attempted",
                "final_asset_status": "acquisition_failed",
            })
        elif row["image_role"] == contract["excluded_image_role"]:
            row.update({
                "eligibility_status": "excluded_non_declared_main",
                "download_status": "not_attempted",
                "decode_status": "not_attempted",
                "final_asset_status": "excluded_non_primary",
            })
        else:
            row.update({
                "eligibility_status": "contract_invalid",
                "download_status": "not_attempted",
                "decode_status": "not_attempted",
                "final_asset_status": "integrity_failed",
                "error_class": "contract_invalid",
                "error_detail": "P7-A row violates frozen P7-B eligibility rule",
            })
        products.append(row)
    return products

def _base_result(row: Mapping[str, Any]) -> dict[str, Any]:
    result = {field: row.get(field, _default_value(field)) for field in PRODUCT_MANIFEST_COLUMNS}
    result.update({
        "retrieved_at_utc": utc_now(),
        "attempt_count": 0,
        "redirect_count": 0,
        "response_byte_count": 0,
        "exact_duplicate_asset_group_size": 0,
        "download_status": "not_attempted",
        "decode_status": "not_attempted",
    })
    return result

def _retry_after_seconds(value: str | None) -> float | None:
    if not value:
        return None
    try:
        return float(value.strip())
    except ValueError:
        try:
            parsed = parsedate_to_datetime(value)
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            return max(0.0, (parsed - datetime.now(timezone.utc)).total_seconds())
        except (TypeError, ValueError, OverflowError):
            return None

def _error_detail(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}".strip()[:500]

def _normalized_content_type(value: str | None) -> str:
    return (value or "").split(";", 1)[0].strip().lower()

def _is_content_type_mismatch(content_type: str, decoded_format: str) -> bool:
    normalized = _normalized_content_type(content_type)
    if not normalized or not decoded_format:
        return False
    expected = MIME_FORMATS.get(normalized)
    return True if expected is None else decoded_format.upper() not in expected

def decode_image_bytes(body: bytes, content_type: str, contract: Mapping[str, Any]) -> dict[str, Any]:
    result = {
        "decode_status": "not_attempted", "decoded_format": "", "width": 0,
        "height": 0, "n_frames": 0, "content_type_mismatch": False,
        "error_class": "", "error_detail": "",
    }
    if not body:
        result["decode_status"] = "empty_body"
        return result
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with Image.open(BytesIO(body)) as image:
                decoded_format = str(image.format or "").upper()
                width, height = image.size
                n_frames = int(getattr(image, "n_frames", 1) or 1)
                if width <= 0 or height <= 0:
                    result.update({"decode_status": "decode_error", "error_class": "non_positive_dimensions"})
                    return result
                if width * height > int(contract["response_limits"]["maximum_decoded_pixels"]):
                    result.update({"decode_status": "pixel_limit_exceeded", "error_class": "maximum_decoded_pixels"})
                    return result
                image.verify()
            with Image.open(BytesIO(body)) as loaded:
                for frame_index in range(n_frames):
                    if frame_index:
                        loaded.seek(frame_index)
                    loaded.load()
        if not decoded_format:
            result.update({"decode_status": "unsupported_format", "error_class": "missing_decoder_format"})
            return result
        result.update({
            "decode_status": "success", "decoded_format": decoded_format,
            "width": int(width), "height": int(height), "n_frames": n_frames,
            "content_type_mismatch": _is_content_type_mismatch(content_type, decoded_format),
        })
        return result
    except UnidentifiedImageError as exc:
        result.update({
            "decode_status": "non_image_response" if not _normalized_content_type(content_type).startswith("image/") else "corrupt_image",
            "error_class": type(exc).__name__, "error_detail": _error_detail(exc),
        })
    except getattr(Image, "DecompressionBombError", OSError) as exc:
        result.update({"decode_status": "pixel_limit_exceeded", "error_class": type(exc).__name__, "error_detail": _error_detail(exc)})
    except OSError as exc:
        result.update({"decode_status": "corrupt_image", "error_class": type(exc).__name__, "error_detail": _error_detail(exc)})
    except Exception as exc:
        result.update({"decode_status": "decode_error", "error_class": type(exc).__name__, "error_detail": _error_detail(exc)})
    return result

def content_addressed_relative_path(response_sha256: str, decoded_format: str) -> str:
    digest = response_sha256.upper()
    extension = FORMAT_EXTENSIONS.get(
        decoded_format.upper(),
        re.sub(r"[^a-z0-9]+", "", decoded_format.lower()) or "bin",
    )
    return f"primary/{digest[:2]}/{digest}.{extension}"

def store_asset_atomically(body: bytes, response_sha256: str, decoded_format: str, asset_root: Path) -> str:
    expected = response_sha256.upper()
    if sha256_bytes(body) != expected:
        raise IntegrityError("response SHA mismatch before storage")
    relative = content_addressed_relative_path(expected, decoded_format)
    target = Path(asset_root) / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if sha256_file(target) == expected:
            return relative
        raise IntegrityError(f"existing canonical asset has wrong SHA: {target}")
    temp_path = target.parent / f".{target.name}.{uuid.uuid4().hex}.tmp"
    try:
        with temp_path.open("xb") as handle:
            handle.write(body)
            handle.flush()
            os.fsync(handle.fileno())
        if sha256_file(temp_path) != expected:
            raise IntegrityError("temporary asset SHA mismatch")
        os.replace(temp_path, target)
    finally:
        if temp_path.exists():
            temp_path.unlink()
    if sha256_file(target) != expected:
        raise IntegrityError("canonical asset SHA mismatch after rename")
    return relative

def _read_response_body(response: Any, maximum_bytes: int) -> tuple[bytes, bool, int]:
    try:
        response.raw.decode_content = True
    except AttributeError:
        pass
    chunks, count = [], 0
    for chunk in response.iter_content(chunk_size=65536):
        if not chunk:
            continue
        chunks.append(chunk)
        count += len(chunk)
        if count > maximum_bytes:
            return b"".join(chunks), True, count
    return b"".join(chunks), False, count

def _set_session(session: Any, maximum_redirects: int) -> None:
    try:
        session.max_redirects = maximum_redirects
    except AttributeError:
        pass
    try:
        session.cookies.clear()
    except AttributeError:
        pass

def _failure_result(
    row: Mapping[str, Any], download_status: str, error_class: str,
    error_detail: str, attempt_count: int, retry_after_seconds: float | None,
) -> dict[str, Any]:
    result = _base_result(row)
    result.update({
        "download_status": download_status, "decode_status": "not_attempted",
        "final_asset_status": "acquisition_failed", "attempt_count": attempt_count,
        "retry_after_seconds": retry_after_seconds if retry_after_seconds is not None else "",
        "error_class": error_class, "error_detail": error_detail,
    })
    return result

def acquire_one(
    row: Mapping[str, Any], contract: Mapping[str, Any], asset_root: Path,
    session_factory: Callable[[], Any] | None = None,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    if row.get("eligibility_status") != "primary_declared_main":
        result = _base_result(row)
        result["final_asset_status"] = "excluded_non_primary"
        return result
    url = str(row.get("requested_url") or "")
    if not url:
        return _failure_result(row, "network_error", "empty_requested_url", "missing exact frozen URL", 0, None)
    policy, limits = contract["network_policy"], contract["response_limits"]
    result, retry_after_last = _base_result(row), None
    session_factory = session_factory or requests.Session
    retryable = {int(value) for value in policy["retry_status_codes"]}
    backoff = [float(value) for value in policy["backoff_seconds"]]
    for attempt in range(1, int(policy["max_attempts"]) + 1):
        result["attempt_count"] = attempt
        response = None
        try:
            session = session_factory()
            _set_session(session, int(policy["max_redirects"]))
            response = session.get(
                url,
                headers={"User-Agent": policy["user_agent"]},
                timeout=(float(policy["connect_timeout_seconds"]), float(policy["read_timeout_seconds"])),
                allow_redirects=True,
                stream=True,
            )
            result.update({
                "http_status": int(response.status_code),
                "final_url": str(getattr(response, "url", "") or ""),
                "redirect_count": len(getattr(response, "history", []) or []),
                "response_content_type": str(response.headers.get("Content-Type", "") or ""),
                "response_content_length_header": str(response.headers.get("Content-Length", "") or ""),
                "response_etag": str(response.headers.get("ETag", "") or ""),
                "response_last_modified": str(response.headers.get("Last-Modified", "") or ""),
            })
            retry_after = _retry_after_seconds(response.headers.get("Retry-After"))
            if retry_after is not None:
                retry_after_last = retry_after
                result["retry_after_seconds"] = retry_after
            if result["redirect_count"] > int(policy["max_redirects"]):
                result.update({
                    "download_status": "redirect_error", "decode_status": "not_attempted",
                    "final_asset_status": "acquisition_failed", "error_class": "maximum_redirects",
                    "error_detail": f"redirect_count={result['redirect_count']}",
                })
                return result
            status = int(response.status_code)
            if status != 200:
                if status in retryable and attempt < int(policy["max_attempts"]):
                    delay = retry_after if retry_after is not None else backoff[min(attempt - 1, len(backoff) - 1)]
                    sleep_fn(float(delay))
                    continue
                result.update({
                    "download_status": "http_error", "decode_status": "not_attempted",
                    "final_asset_status": "acquisition_failed", "error_class": f"http_{status}",
                    "error_detail": f"HTTP status {status}",
                    "retry_after_seconds": retry_after_last if retry_after_last is not None else "",
                })
                return result
            body, oversized, count = _read_response_body(response, int(limits["maximum_response_bytes"]))
            result["response_byte_count"] = count
            if oversized:
                result.update({
                    "download_status": "response_too_large", "decode_status": "not_attempted",
                    "final_asset_status": "acquisition_failed", "error_class": "maximum_response_bytes",
                    "error_detail": f"read_bytes={count}",
                })
                return result
            result["download_status"] = "success"
            result["response_sha256"] = sha256_bytes(body)
            decoded = decode_image_bytes(body, result["response_content_type"], contract)
            for field in ("decode_status", "decoded_format", "width", "height", "n_frames", "error_class", "error_detail"):
                result[field] = decoded[field]
            if decoded["decode_status"] != "success":
                result["final_asset_status"] = "acquisition_failed"
                return result
            result["asset_path"] = store_asset_atomically(body, result["response_sha256"], result["decoded_format"], Path(asset_root))
            result["final_asset_status"] = "available"
            result["error_class"], result["error_detail"] = "", ""
            return result
        except (requests.exceptions.Timeout, TimeoutError) as exc:
            if attempt < int(policy["max_attempts"]):
                sleep_fn(backoff[min(attempt - 1, len(backoff) - 1)])
                continue
            return _failure_result(row, "timeout", type(exc).__name__, _error_detail(exc), attempt, retry_after_last)
        except (requests.exceptions.ConnectionError, ConnectionError) as exc:
            if attempt < int(policy["max_attempts"]):
                sleep_fn(backoff[min(attempt - 1, len(backoff) - 1)])
                continue
            return _failure_result(row, "connection_error", type(exc).__name__, _error_detail(exc), attempt, retry_after_last)
        except requests.exceptions.TooManyRedirects as exc:
            return _failure_result(row, "redirect_error", type(exc).__name__, _error_detail(exc), attempt, retry_after_last)
        except IntegrityError:
            raise
        except requests.exceptions.RequestException as exc:
            if attempt < int(policy["max_attempts"]):
                sleep_fn(backoff[min(attempt - 1, len(backoff) - 1)])
                continue
            return _failure_result(row, "network_error", type(exc).__name__, _error_detail(exc), attempt, retry_after_last)
        finally:
            if response is not None:
                try:
                    response.close()
                except Exception:
                    pass
    return _failure_result(row, "network_error", "retry_loop_exhausted", "retry loop exhausted", int(policy["max_attempts"]), retry_after_last)

def append_checkpoint(path: Path, result: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="") as handle:
        handle.write(json.dumps(dict(result), ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())

def load_checkpoint(path: Path, asset_root: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    loaded = {}
    with path.open("r", encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                result = json.loads(line)
            except json.JSONDecodeError as exc:
                raise FormalVerificationError(f"invalid checkpoint line {number}") from exc
            asin = str(result.get("parent_asin") or "")
            if not asin:
                raise FormalVerificationError(f"checkpoint line {number} has no parent_asin")
            if result.get("final_asset_status") == "available":
                path_on_disk = Path(asset_root) / str(result.get("asset_path") or "")
                if not path_on_disk.exists():
                    continue
                if sha256_file(path_on_disk) != str(result.get("response_sha256") or "").upper():
                    raise IntegrityError(f"checkpoint asset SHA mismatch: {path_on_disk}")
            loaded[asin] = result
    return loaded

def resume_or_acquire(
    row: Mapping[str, Any], checkpoint: Mapping[str, Mapping[str, Any]],
    contract: Mapping[str, Any], asset_root: Path,
    session_factory: Callable[[], Any] | None = None,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    saved = checkpoint.get(str(row["parent_asin"]))
    if saved is not None:
        if saved.get("final_asset_status") != "available":
            return dict(saved)
        path = Path(asset_root) / str(saved.get("asset_path") or "")
        if path.exists() and sha256_file(path) == str(saved.get("response_sha256") or "").upper():
            return dict(saved)
    return acquire_one(row, contract, asset_root, session_factory, sleep_fn)

def validate_manifest_row_enums(row: Mapping[str, Any], contract: Mapping[str, Any]) -> None:
    problems = []
    for field in ("eligibility_status", "download_status", "decode_status", "final_asset_status"):
        if row.get(field) not in set(contract[f"{field}_enum"]):
            problems.append(f"{field}={row.get(field)!r}")
    if row.get("image_role") not in {contract["eligible_image_role"], contract["excluded_image_role"]}:
        problems.append(f"image_role={row.get('image_role')!r}")
    if row.get("temporal_alignment_status") != contract["primary_temporal_alignment_status"]:
        problems.append(f"temporal_alignment_status={row.get('temporal_alignment_status')!r}")
    if problems:
        raise ContractError("manifest enum violation: " + ", ".join(problems))

def annotate_exact_duplicates(products: list[dict[str, Any]]) -> None:
    groups = {}
    for row in products:
        digest = str(row.get("response_sha256") or "").upper()
        if row.get("final_asset_status") == "available" and digest:
            groups.setdefault(digest, []).append(row)
    for digest, rows in groups.items():
        for row in rows:
            row["exact_duplicate_asset_id"] = digest
            row["exact_duplicate_asset_group_size"] = len(rows)

def build_unique_asset_inventory(products: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped, urls = {}, {}
    for row in products:
        if row.get("final_asset_status") != "available":
            continue
        digest = str(row.get("response_sha256") or "").upper()
        if not digest:
            raise FormalVerificationError("available product has no response SHA")
        current = {
            "asset_sha256": digest, "asset_path": row.get("asset_path", ""),
            "decoded_format": row.get("decoded_format", ""),
            "width": int(row.get("width") or 0), "height": int(row.get("height") or 0),
            "n_frames": int(row.get("n_frames") or 0),
            "response_byte_count": int(row.get("response_byte_count") or 0),
            "product_count": 0, "source_url_count": 0,
        }
        if digest not in grouped:
            grouped[digest], urls[digest] = current, set()
        existing = grouped[digest]
        for field in ("asset_path", "decoded_format", "width", "height", "n_frames", "response_byte_count"):
            if existing[field] != current[field]:
                raise FormalVerificationError(f"inconsistent duplicate metadata: {digest}/{field}")
        existing["product_count"] += 1
        if row.get("requested_url"):
            urls[digest].add(str(row["requested_url"]))
    for digest, row in grouped.items():
        row["source_url_count"] = len(urls[digest])
    return list(grouped.values())

def _median(values: list[int]) -> int | float | None:
    if not values:
        return None
    value = statistics.median(values)
    return int(value) if float(value).is_integer() else value

def compute_summary(
    products: list[Mapping[str, Any]], unique_assets: list[Mapping[str, Any]],
    p7_a_summary: Mapping[str, Any], contract: Mapping[str, Any],
    *, orphan_asset_count: int = 0, temp_file_count: int = 0,
) -> dict[str, Any]:
    eligible = [r for r in products if r.get("eligibility_status") == "primary_declared_main"]
    available = [r for r in products if r.get("final_asset_status") == "available"]
    failed = [r for r in eligible if r.get("final_asset_status") != "available"]
    attempted = [r for r in eligible if int(r.get("attempt_count") or 0) > 0]
    terminal = [
        r for r in products
        if r.get("final_asset_status") in set(contract["final_asset_status_enum"])
        and not (r.get("eligibility_status") == "primary_declared_main" and int(r.get("attempt_count") or 0) == 0)
    ]
    pending = len(products) - len(terminal)
    http_dist = Counter(str(r["http_status"]) for r in products if str(r.get("http_status") or ""))
    content_dist = Counter(str(r["response_content_type"]) for r in products if str(r.get("response_content_type") or ""))
    decode_dist = Counter(str(r.get("decode_status") or "") for r in products)
    download_dist = Counter(str(r.get("download_status") or "") for r in products)
    final_dist = Counter(str(r.get("final_asset_status") or "") for r in products)
    format_dist = Counter(str(r.get("decoded_format") or "") for r in available)
    widths = [int(r["width"]) for r in available]
    heights = [int(r["height"]) for r in available]
    sizes = [int(r["response_byte_count"]) for r in available]
    mismatch = sum(
        _is_content_type_mismatch(str(r.get("response_content_type") or ""), str(r.get("decoded_format") or ""))
        for r in available
    )
    rate_403_429 = sum(c for s, c in http_dist.items() if s in {"403", "429"}) / max(1, len(attempted))
    failure_rate = len(failed) / max(1, len(eligible))
    http_200 = [r for r in eligible if str(r.get("http_status") or "") == "200"]
    decode_failure_rate = sum(r.get("final_asset_status") != "available" for r in http_200) / max(1, len(http_200))
    thresholds = contract["acquisition_sanity_thresholds"]
    sanity = "PASS"
    if (
        pending != thresholds["pending_count_must_equal"]
        or rate_403_429 > thresholds["http_403_plus_429_rate_review_above"]
        or failure_rate > thresholds["terminal_failure_rate_review_above"]
        or decode_failure_rate > thresholds["decode_or_integrity_failure_rate_among_http_200_review_above"]
    ):
        sanity = thresholds["review_status"]
    pipeline = (
        len(products) == contract["product_universe_count"]
        and len({r.get("parent_asin") for r in products}) == contract["product_universe_count"]
        and len(eligible) == contract["primary_eligible_count"]
        and len(products) - len(eligible) == contract["excluded_non_declared_main_count"]
        and len(attempted) == contract["primary_eligible_count"]
        and pending == 0
        and all(r.get("final_asset_status") != "available" or r.get("decode_status") == "success" for r in products)
    )
    return {
        "p7_b_status": "PASS" if pipeline else "FAIL",
        "acquisition_sanity_status": sanity,
        "contract_version": contract["contract_version"],
        "upstream_p7_a": {
            "inventory_sha256": contract["upstream_p7_a"]["inventory_sha256"],
            "summary_sha256": contract["upstream_p7_a"]["summary_sha256"],
            "provenance_sha256": contract["upstream_p7_a"]["provenance_sha256"],
            "input_product_sha256": contract["upstream_p7_a"]["input_product_sha256"],
        },
        "product_universe_count": len(products),
        "primary_eligible_count": len(eligible),
        "excluded_non_declared_main_count": len(products) - len(eligible),
        "attempted_count": len(attempted), "terminal_count": len(terminal), "pending_count": pending,
        "download_status_distribution": dict(sorted(download_dist.items())),
        "decode_status_distribution": dict(sorted(decode_dist.items())),
        "final_asset_status_distribution": dict(sorted(final_dist.items())),
        "success_asset_product_count": len(available), "failed_product_count": len(failed),
        "success_rate": len(available) / max(1, len(eligible)),
        "http_status_distribution": dict(sorted(http_dist.items())),
        "content_type_distribution": dict(sorted(content_dist.items())),
        "actual_decoded_format_distribution": dict(sorted(format_dist.items())),
        "jpeg_count": format_dist.get("JPEG", 0) + format_dist.get("JPG", 0),
        "gif_count": format_dist.get("GIF", 0), "png_count": format_dist.get("PNG", 0),
        "webp_count": format_dist.get("WEBP", 0),
        "other_actual_format_count": sum(c for f, c in format_dist.items() if f not in {"JPEG", "JPG", "GIF", "PNG", "WEBP"}),
        "multi_frame_count": sum(int(r.get("n_frames") or 0) > 1 for r in available),
        "width_min": min(widths) if widths else None, "width_median": _median(widths),
        "width_max": max(widths) if widths else None, "height_min": min(heights) if heights else None,
        "height_median": _median(heights), "height_max": max(heights) if heights else None,
        "byte_size_min": min(sizes) if sizes else None, "byte_size_median": _median(sizes),
        "byte_size_max": max(sizes) if sizes else None,
        "content_type_vs_decode_format_mismatch_count": mismatch,
        "source_url_duplicate_group_count": p7_a_summary.get("duplicate_main_url_statistics", {}).get("duplicate_group_count", 0),
        "source_url_products_in_duplicate_groups": p7_a_summary.get("duplicate_main_url_statistics", {}).get("products_in_duplicate_groups", 0),
        "source_url_largest_duplicate_group": p7_a_summary.get("duplicate_main_url_statistics", {}).get("largest_duplicate_group_size", 0),
        "unique_asset_count": len(unique_assets),
        "exact_byte_duplicate_group_count": sum(int(r.get("product_count") or 0) > 1 for r in unique_assets),
        "products_in_exact_byte_duplicate_groups": sum(int(r.get("product_count") or 0) for r in unique_assets if int(r.get("product_count") or 0) > 1),
        "largest_exact_byte_duplicate_group": max((int(r.get("product_count") or 0) for r in unique_assets), default=0),
        "orphan_asset_count": orphan_asset_count, "temp_file_count": temp_file_count,
        "failure_reason_distribution": dict(sorted(Counter(str(r.get("error_class") or "unknown") for r in failed).items())),
        "label_sources_read": False, "current_product_pages_accessed": False,
        "fallback_search_used": False, "secondary_or_child_images_used": False,
        "images_converted_or_reencoded": False,
    }

def _write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temp.write_text(text, encoding="utf-8", newline="")
        os.replace(temp, path)
    finally:
        if temp.exists():
            temp.unlink()

def write_csv(path: Path, fields: tuple[str, ...], rows: list[Mapping[str, Any]]) -> None:
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

def _output_paths(contract: Mapping[str, Any]) -> dict[str, Path]:
    return {key: Path(value) for key, value in contract["formal_output_paths"].items()}

def _asset_files(asset_root: Path) -> list[Path]:
    primary = Path(asset_root) / "primary"
    if not primary.exists():
        return []
    return [p for p in primary.rglob("*") if p.is_file() and ".tmp" not in p.name]

def _asset_relative_path(path: Path, asset_root: Path) -> str:
    return path.relative_to(asset_root).as_posix()

def write_formal_outputs(
    products: list[dict[str, Any]], unique_assets: list[dict[str, Any]],
    summary: dict[str, Any], provenance: dict[str, Any], contract: Mapping[str, Any],
) -> dict[str, str]:
    paths = _output_paths(contract)
    write_csv(paths["product_manifest"], PRODUCT_MANIFEST_COLUMNS, products)
    write_csv(paths["unique_asset_inventory"], UNIQUE_ASSET_COLUMNS, unique_assets)
    _write_text_atomic(paths["summary"], json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    output_shas = {
        name: sha256_file(paths[name])
        for name in ("product_manifest", "unique_asset_inventory", "summary")
    }
    provenance["output_sha256"] = {
        "01_primary_asset_manifest.csv": output_shas["product_manifest"],
        "02_unique_asset_inventory.csv": output_shas["unique_asset_inventory"],
        "03_p7_b_summary.json": output_shas["summary"],
    }
    _write_text_atomic(paths["provenance"], json.dumps(provenance, ensure_ascii=False, indent=2) + "\n")
    output_shas["04_p7_b_provenance.json"] = sha256_file(paths["provenance"])
    return output_shas

def _parse_manifest_row(row: Mapping[str, str]) -> dict[str, Any]:
    parsed = dict(row)
    for field in INTEGER_FIELDS:
        parsed[field] = int(parsed.get(field) or 0)
    for field in FLOAT_FIELDS:
        parsed[field] = "" if parsed.get(field, "") == "" else float(parsed[field])
    return parsed

def read_manifest(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [_parse_manifest_row(r) for r in csv.DictReader(handle)]

def read_unique_inventory(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = []
        for row in csv.DictReader(handle):
            parsed = dict(row)
            for field in ("width", "height", "n_frames", "response_byte_count", "product_count", "source_url_count"):
                parsed[field] = int(parsed[field] or 0)
            rows.append(parsed)
        return rows

def verify_asset_integrity(
    products: list[Mapping[str, Any]], unique_assets: list[Mapping[str, Any]],
    asset_root: Path, contract: Mapping[str, Any],
) -> None:
    for row in products:
        try:
            validate_manifest_row_enums(row, contract)
        except ContractError as exc:
            raise FormalVerificationError(str(exc)) from exc
        eligibility = str(row.get("eligibility_status") or "")
        download_status = str(row.get("download_status") or "")
        decode_status = str(row.get("decode_status") or "")
        final_asset_status = str(row.get("final_asset_status") or "")
        attempt_count = int(row.get("attempt_count") or 0)
        if eligibility == "excluded_non_declared_main":
            if (
                str(row.get("requested_url") or "")
                or attempt_count != 0
                or download_status != "not_attempted"
                or decode_status != "not_attempted"
                or final_asset_status != "excluded_non_primary"
            ):
                raise FormalVerificationError(
                    f"excluded row state transition mismatch: {row.get('parent_asin')}"
                )
        elif final_asset_status == "available":
            if (
                eligibility != "primary_declared_main"
                or attempt_count <= 0
                or download_status != "success"
                or decode_status != "success"
            ):
                raise FormalVerificationError(
                    f"available row state transition mismatch: {row.get('parent_asin')}"
                )
        if eligibility == "primary_declared_main" and attempt_count == 0:
            raise FormalVerificationError(f"eligible row remains unattempted: {row.get('parent_asin')}")
        if row.get("final_asset_status") != "available":
            continue
        digest = str(row.get("response_sha256") or "").upper()
        relative = str(row.get("asset_path") or "")
        if not digest or relative != content_addressed_relative_path(digest, str(row.get("decoded_format") or "")):
            raise FormalVerificationError(f"non-deterministic asset path: {row.get('parent_asin')}")
        path = Path(asset_root) / relative
        if not path.exists() or sha256_file(path) != digest:
            raise FormalVerificationError(f"canonical asset missing or mutated: {path}")
        decoded = decode_image_bytes(path.read_bytes(), str(row.get("response_content_type") or ""), contract)
        if decoded["decode_status"] != "success":
            raise FormalVerificationError(f"canonical asset does not decode: {path}")
        for field in ("decoded_format", "width", "height", "n_frames"):
            if str(decoded[field]) != str(row.get(field)):
                raise FormalVerificationError(f"manifest decode mismatch {field}: {path}")
        if int(row.get("response_byte_count") or 0) != path.stat().st_size:
            raise FormalVerificationError(f"manifest byte count mismatch: {path}")
    expected_products = [dict(row) for row in products]
    annotate_exact_duplicates(expected_products)
    for actual, expected in zip(products, expected_products):
        if (
            actual.get("exact_duplicate_asset_id") != expected.get("exact_duplicate_asset_id")
            or int(actual.get("exact_duplicate_asset_group_size") or 0)
            != int(expected.get("exact_duplicate_asset_group_size") or 0)
        ):
            raise FormalVerificationError("exact duplicate annotation mismatch")
    rebuilt = build_unique_asset_inventory(products)
    if rebuilt != [dict(r) for r in unique_assets]:
        raise FormalVerificationError("unique asset inventory mismatch")
    expected = {str(r.get("asset_path") or "") for r in products if r.get("final_asset_status") == "available"}
    actual = {_asset_relative_path(p, asset_root) for p in _asset_files(Path(asset_root))}
    if expected != actual:
        raise FormalVerificationError("orphan or missing canonical asset")
    temporary = [p for p in Path(asset_root).rglob("*") if p.is_file() and (".tmp" in p.name or p.name.endswith(".partial"))] if Path(asset_root).exists() else []
    if temporary:
        raise FormalVerificationError("temporary canonical files remain")

def verify_producer_identity(provenance: Mapping[str, Any], *, current_head: str | None = None) -> dict[str, bool]:
    checks = {
        "producer_git_commit_known": False,
        "producer_script_git_blob_sha256_match": False,
        "producer_contract_git_blob_sha256_match": False,
        "current_head_contract_git_blob_sha256_match": False,
    }
    commit = str(provenance.get("formal_run_git_commit") or "")
    if not commit:
        return checks
    checks["producer_git_commit_known"] = True
    try:
        script_path = Path("scripts") / "images" / "acquire_p7_primary_assets.py"
        contract_path = Path("config") / "image_assets" / "p7_b_asset_contract.json"
        checks["producer_script_git_blob_sha256_match"] = (
            git_blob_sha256(commit, script_path) == provenance.get("producer_script_git_blob_sha256") == provenance.get("script_sha256")
        )
        checks["producer_contract_git_blob_sha256_match"] = (
            git_blob_sha256(commit, contract_path) == provenance.get("producer_contract_git_blob_sha256") == provenance.get("contract_sha256")
        )
        if current_head is None:
            current_head = git_head()
        checks["current_head_contract_git_blob_sha256_match"] = git_blob_sha256(current_head, contract_path) == provenance.get("producer_contract_git_blob_sha256")
    except Exception:
        return checks
    return checks

def verify_existing(contract: Mapping[str, Any]) -> dict[str, Any]:
    paths = _output_paths(contract)
    required = ("product_manifest", "unique_asset_inventory", "summary", "provenance")
    missing = [key for key in required if not paths[key].exists()]
    if missing:
        raise FormalVerificationError(f"missing formal outputs: {missing}")
    p7_a_summary = validate_upstream_files(contract)
    products, unique_assets = read_manifest(paths["product_manifest"]), read_unique_inventory(paths["unique_asset_inventory"])
    if len(products) != contract["product_universe_count"] or len({r["parent_asin"] for r in products}) != contract["product_universe_count"]:
        raise FormalVerificationError("product manifest universe mismatch")
    if sum(r["eligibility_status"] == "primary_declared_main" for r in products) != contract["primary_eligible_count"]:
        raise FormalVerificationError("eligible count mismatch")
    if sum(r["eligibility_status"] == "excluded_non_declared_main" for r in products) != contract["excluded_non_declared_main_count"]:
        raise FormalVerificationError("excluded count mismatch")
    p7_a_inventory = load_p7_a_inventory(contract)
    for input_order, (product, source) in enumerate(zip(products, p7_a_inventory)):
        if product["input_order"] != input_order:
            raise FormalVerificationError("input order mismatch")
        for field in ("parent_asin", "source_record_sha256", "image_role", "temporal_alignment_status"):
            if str(product.get(field)) != str(source.get(field)):
                raise FormalVerificationError(f"upstream row mismatch: {field}")
        expected_url = str(source.get("existing_main_image_url") or "") if product["eligibility_status"] == "primary_declared_main" else ""
        if product.get("requested_url") != expected_url:
            raise FormalVerificationError("requested URL mismatch")
    verify_asset_integrity(products, unique_assets, Path(contract["storage_policy"]["asset_root"]), contract)
    summary = json.loads(paths["summary"].read_text(encoding="utf-8"))
    if summary != compute_summary(products, unique_assets, p7_a_summary, contract):
        raise FormalVerificationError("summary reconstruction mismatch")
    provenance = json.loads(paths["provenance"].read_text(encoding="utf-8"))
    checks = verify_producer_identity(provenance)
    checks.update({
        "contract_version": provenance.get("contract_version") == contract["contract_version"],
        "p7_a_inventory_sha": sha256_file(Path(contract["upstream_p7_a"]["inventory_path"])) == contract["upstream_p7_a"]["inventory_sha256"],
        "p7_a_summary_sha": sha256_file(Path(contract["upstream_p7_a"]["summary_path"])) == contract["upstream_p7_a"]["summary_sha256"],
        "p7_a_provenance_sha": sha256_file(Path(contract["upstream_p7_a"]["provenance_path"])) == contract["upstream_p7_a"]["provenance_sha256"],
        "provenance_does_not_record_own_sha": "04_p7_b_provenance.json" not in provenance.get("output_sha256", {}),
    })
    for name, key in (
        ("01_primary_asset_manifest.csv", "product_manifest"),
        ("02_unique_asset_inventory.csv", "unique_asset_inventory"),
        ("03_p7_b_summary.json", "summary"),
    ):
        checks[f"{key}_sha"] = provenance.get("output_sha256", {}).get(name) == sha256_file(paths[key])
    if not all(checks.values()):
        raise FormalVerificationError(f"provenance checks failed: {checks}")
    return {"verification": "PASS", "product_manifest_rows": len(products), "unique_assets": len(unique_assets), "provenance_checks": checks}

def _ensure_clean_worktree() -> None:
    status = subprocess.check_output(["git", "status", "--porcelain"], text=True, stderr=subprocess.DEVNULL).strip()
    if status:
        raise FormalVerificationError("formal run requires a clean worktree")

def run_formal(contract: Mapping[str, Any], *, resume: bool, command_line: str) -> dict[str, Any]:
    _ensure_clean_worktree()
    paths = _output_paths(contract)
    if any(paths[key].exists() for key in ("product_manifest", "unique_asset_inventory", "summary", "provenance")):
        raise FormalVerificationError("formal output already exists")
    start = utc_now()
    p7_a_summary = validate_upstream_files(contract)
    inventory = load_p7_a_inventory(contract)
    products = build_product_manifest_skeleton(inventory, contract)
    if len(products) != contract["product_universe_count"]:
        raise FormalVerificationError("P7-B product universe mismatch")
    eligible = [r for r in products if r["eligibility_status"] == "primary_declared_main"]
    excluded = [r for r in products if r["eligibility_status"] == "excluded_non_declared_main"]
    if len(eligible) != contract["primary_eligible_count"] or len(excluded) != contract["excluded_non_declared_main_count"]:
        raise FormalVerificationError("P7-B eligibility mismatch")
    asset_root = Path(contract["storage_policy"]["asset_root"])
    checkpoint_path = paths["checkpoint"]
    checkpoint = load_checkpoint(checkpoint_path, asset_root) if resume else {}
    paths["output_dir"].mkdir(parents=True, exist_ok=True)
    asset_root.mkdir(parents=True, exist_ok=True)
    results = {}
    for row in excluded:
        result = _base_result(row)
        result["final_asset_status"] = "excluded_non_primary"
        results[row["parent_asin"]] = result
        if row["parent_asin"] not in checkpoint:
            append_checkpoint(checkpoint_path, result)
    for row in eligible:
        if row["parent_asin"] in checkpoint:
            results[row["parent_asin"]] = resume_or_acquire(row, checkpoint, contract, asset_root)
    pending = [r for r in eligible if r["parent_asin"] not in checkpoint]
    with ThreadPoolExecutor(max_workers=contract["network_policy"]["max_workers"]) as executor:
        futures = {executor.submit(acquire_one, r, contract, asset_root): r for r in pending}
        for future in as_completed(futures):
            row = futures[future]
            try:
                result = future.result()
            except IntegrityError:
                raise
            except Exception as exc:
                result = _failure_result(row, "network_error", "unexpected_worker_error", _error_detail(exc), 0, None)
            results[row["parent_asin"]] = result
            append_checkpoint(checkpoint_path, result)
    ordered = [results[r["parent_asin"]] for r in products]
    if any(r["eligibility_status"] == "primary_declared_main" and not int(r["attempt_count"]) for r in ordered):
        raise FormalVerificationError("eligible row remained unattempted")
    annotate_exact_duplicates(ordered)
    unique_assets = build_unique_asset_inventory(ordered)
    asset_paths = {r["asset_path"] for r in unique_assets}
    orphan_count = sum(_asset_relative_path(p, asset_root) not in asset_paths for p in _asset_files(asset_root))
    temp_count = sum(p.is_file() and ".tmp" in p.name for p in asset_root.rglob("*")) if asset_root.exists() else 0
    summary = compute_summary(ordered, unique_assets, p7_a_summary, contract, orphan_asset_count=orphan_count, temp_file_count=temp_count)
    commit = git_head()
    script_path, contract_path = Path("scripts") / "images" / "acquire_p7_primary_assets.py", Path("config") / "image_assets" / "p7_b_asset_contract.json"
    script_sha, contract_sha = git_blob_sha256(commit, script_path), git_blob_sha256(commit, contract_path)
    provenance = {
        "formal_run_git_commit": commit,
        "script_path": str(script_path), "contract_path": str(contract_path),
        "script_sha256": script_sha, "contract_sha256": contract_sha,
        "producer_script_git_blob_sha256": script_sha, "producer_contract_git_blob_sha256": contract_sha,
        "contract_version": contract["contract_version"], "p7_a_input_paths": contract["upstream_p7_a"],
        "execution_start_utc": start, "execution_end_utc": utc_now(),
        "python_version": sys.version.split()[0], "requests_version": requests.__version__,
        "pillow_version": Image.__version__, "os_platform": platform.platform(),
        "command_line": command_line, "network_policy": contract["network_policy"],
        "response_limits": contract["response_limits"], "asset_root": str(asset_root),
        "output_paths": {key: str(paths[key]) for key in ("product_manifest", "unique_asset_inventory", "summary", "provenance")},
        "output_sha256": {},
    }
    output_shas = write_formal_outputs(ordered, unique_assets, summary, provenance, contract)
    return {"summary": summary, "provenance": provenance, "output_sha256": output_shas}

def export_git_summary(contract: Mapping[str, Any]) -> None:
    paths = _output_paths(contract)
    summary = json.loads(paths["summary"].read_text(encoding="utf-8"))
    provenance = json.loads(paths["provenance"].read_text(encoding="utf-8"))
    allowed = (
        "p7_b_status", "acquisition_sanity_status", "contract_version", "upstream_p7_a",
        "product_universe_count", "primary_eligible_count", "excluded_non_declared_main_count",
        "attempted_count", "terminal_count", "pending_count", "download_status_distribution",
        "decode_status_distribution", "final_asset_status_distribution", "success_asset_product_count",
        "failed_product_count", "success_rate", "http_status_distribution", "content_type_distribution",
        "actual_decoded_format_distribution", "jpeg_count", "gif_count", "png_count", "webp_count",
        "other_actual_format_count", "multi_frame_count", "width_min", "width_median", "width_max",
        "height_min", "height_median", "height_max", "byte_size_min", "byte_size_median", "byte_size_max",
        "content_type_vs_decode_format_mismatch_count", "source_url_duplicate_group_count",
        "source_url_products_in_duplicate_groups", "source_url_largest_duplicate_group", "unique_asset_count",
        "exact_byte_duplicate_group_count", "products_in_exact_byte_duplicate_groups",
        "largest_exact_byte_duplicate_group", "orphan_asset_count", "temp_file_count",
        "failure_reason_distribution", "label_sources_read", "current_product_pages_accessed",
        "fallback_search_used", "secondary_or_child_images_used", "images_converted_or_reencoded",
    )
    sanitized = {key: summary[key] for key in allowed}
    sanitized.update({
        "formal_run_git_commit": provenance["formal_run_git_commit"],
        "producer_script_git_blob_sha256": provenance["producer_script_git_blob_sha256"],
        "producer_contract_git_blob_sha256": provenance["producer_contract_git_blob_sha256"],
        "formal_output_sha256": provenance["output_sha256"],
        "formal_provenance_sha256": sha256_file(paths["provenance"]),
    })
    _write_text_atomic(Path("docs/stages/images/p7_b_primary_asset_summary.json"), json.dumps(sanitized, ensure_ascii=False, indent=2) + "\n")

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="P7-B primary asset acquisition")
    parser.add_argument("--verify-existing", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--export-git-summary", action="store_true")
    return parser

def main() -> int:
    args = build_parser().parse_args()
    contract = load_contract()
    if args.verify_existing:
        print(json.dumps(verify_existing(contract), ensure_ascii=False, indent=2))
        return 0
    if args.export_git_summary:
        export_git_summary(contract)
        print("Git-visible P7-B summary written.")
        return 0
    print(json.dumps(run_formal(contract, resume=args.resume, command_line=" ".join(sys.argv)), ensure_ascii=False, indent=2))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
