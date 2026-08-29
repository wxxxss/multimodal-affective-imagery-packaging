#!/usr/bin/env python3
"""Shared provenance helpers for v2.1 validation artifacts."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
from typing import Any, Mapping, Sequence

try:
    from .affective_imagery_action_error_mapping_v21 import MappingMetadata
except ImportError:  # pragma: no cover - direct script import
    from affective_imagery_action_error_mapping_v21 import MappingMetadata


PROVENANCE_SCHEMA_VERSION = 1


def sha256_file(path: Path) -> str:
    """Return the SHA-256 of exact file bytes."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_git_state(repo_root: Path) -> dict[str, object]:
    """Return the checked-out commit and whether tracked/untracked changes exist."""
    root = Path(repo_root)
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(f"unable to inspect Git state at {root}") from exc
    if len(commit) != 40:
        raise RuntimeError(f"unexpected Git commit SHA: {commit}")
    return {"git_commit_sha": commit, "git_dirty": bool(status.strip())}


def _file_record(logical_name: str, path: Path) -> dict[str, object]:
    name = str(logical_name).strip()
    if not name:
        raise ValueError("file logical_name must be nonblank")
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"artifact file does not exist: {file_path}")
    absolute = file_path.resolve()
    # Prefer a portable repo-relative path when the file lives inside a Git repo.
    relative = _repo_relative_path(absolute)
    identity = relative if relative is not None else str(absolute)
    return {
        "logical_name": name,
        "path": identity,
        "sha256": sha256_file(file_path),
    }


def _repo_relative_path(path: Path) -> str | None:
    """Return a repo-relative POSIX path, or None when outside any Git repo."""
    candidate = path
    while candidate != candidate.parent:
        if (candidate / ".git").exists():
            try:
                return str(path.relative_to(candidate)).replace("\\", "/")
            except ValueError:
                return None
        candidate = candidate.parent
    return None


def _string_list(values: Sequence[str], field: str) -> list[str]:
    result = [str(value).strip() for value in values]
    if any(not value for value in result):
        raise ValueError(f"{field} values must be nonblank")
    if len(result) != len(set(result)):
        raise ValueError(f"{field} values must be unique")
    return result


def build_provenance(
    *,
    mapping_metadata: MappingMetadata,
    tool_name: str,
    tool_version: str,
    git_state: Mapping[str, object],
    input_files: Mapping[str, Path],
    output_logical_name: str,
    output_path: Path,
    row_count: int,
    mode: str,
    unresolved_count: int,
    allowed_changed_fields: Sequence[str],
    annotator_id: str | None = None,
    annotator_ids: Sequence[str] | None = None,
) -> dict[str, object]:
    """Build a non-self-referential provenance payload from actual artifacts."""
    if annotator_id is not None and annotator_ids is not None:
        raise ValueError("annotator_id and annotator_ids are mutually exclusive")
    if isinstance(row_count, bool) or not isinstance(row_count, int) or row_count < 0:
        raise ValueError("row_count must be a non-negative integer")
    if (
        isinstance(unresolved_count, bool)
        or not isinstance(unresolved_count, int)
        or unresolved_count < 0
    ):
        raise ValueError("unresolved_count must be a non-negative integer")
    name = str(tool_name).strip()
    version = str(tool_version).strip()
    mode_value = str(mode).strip()
    if not name or not version or not mode_value:
        raise ValueError("tool_name, tool_version, and mode must be nonblank")
    commit = str(git_state.get("git_commit_sha", "")).strip()
    if len(commit) != 40:
        raise ValueError("git_commit_sha must contain a 40-character SHA")
    dirty = git_state.get("git_dirty")
    if not isinstance(dirty, bool):
        raise ValueError("git_dirty must be boolean")

    input_records = [
        _file_record(logical_name, input_files[logical_name])
        for logical_name in sorted(input_files)
    ]
    output_record = _file_record(output_logical_name, output_path)
    payload: dict[str, object] = {
        "provenance_schema_version": PROVENANCE_SCHEMA_VERSION,
        "mapping_version": mapping_metadata.mapping_version,
        "mapping_sha256": mapping_metadata.mapping_sha256,
        "contract_schema_version": mapping_metadata.contract_schema_version,
        "tool_name": name,
        "tool_version": version,
        "git_commit_sha": commit,
        "git_dirty": dirty,
        "inputs": input_records,
        "output": output_record,
        "row_count": row_count,
        "mode": mode_value,
        "unresolved_count": unresolved_count,
        "allowed_changed_fields": _string_list(
            allowed_changed_fields, "allowed_changed_fields"
        ),
        "model_assisted": True,
        "human_gold": False,
    }
    if annotator_id is not None:
        identity = str(annotator_id).strip()
        if not identity:
            raise ValueError("annotator_id must be nonblank")
        payload["annotator_id"] = identity
    elif annotator_ids is not None:
        payload["annotator_ids"] = _string_list(annotator_ids, "annotator_ids")
    return payload


def _json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def write_json_new_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    """Atomically publish a new JSON file without overwriting an existing path."""
    output = Path(path)
    if output.exists():
        raise FileExistsError(f"sidecar already exists: {output}")
    if not output.parent.is_dir():
        raise FileNotFoundError(f"sidecar parent directory does not exist: {output.parent}")

    descriptor, temp_name = tempfile.mkstemp(
        dir=output.parent,
        prefix=f".{output.name}.",
        suffix=".tmp",
    )
    temp_path = Path(temp_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(_json_bytes(payload))
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temp_path, output)
        except FileExistsError as exc:
            raise FileExistsError(f"sidecar already exists: {output}") from exc
    finally:
        temp_path.unlink(missing_ok=True)


def load_provenance(path: Path) -> dict[str, Any]:
    provenance_path = Path(path)
    try:
        payload = json.loads(provenance_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"provenance sidecar is not valid JSON: {provenance_path}") from exc
    if not isinstance(payload, dict):
        raise ValueError("provenance sidecar must be a JSON object")
    return payload


def _require_equal(actual: Any, expected: Any, field: str) -> None:
    if actual != expected:
        raise ValueError(
            f"provenance {field} mismatch: actual={actual!r} expected={expected!r}"
        )


def verify_provenance_common(
    provenance: Mapping[str, Any],
    *,
    mapping_metadata: MappingMetadata,
    tool_version: str,
    input_files: Mapping[str, Path],
    output_path: Path,
    row_count: int,
    allowed_changed_fields: Sequence[str],
    annotator_id: str | None = None,
    annotator_ids: Sequence[str] | None = None,
) -> None:
    """Verify common sidecar fields against the current mapping and files."""
    if annotator_id is not None and annotator_ids is not None:
        raise ValueError("annotator_id and annotator_ids are mutually exclusive")
    _require_equal(
        provenance.get("mapping_version"),
        mapping_metadata.mapping_version,
        "mapping_version",
    )
    _require_equal(
        provenance.get("mapping_sha256"),
        mapping_metadata.mapping_sha256,
        "mapping_sha256",
    )
    _require_equal(
        provenance.get("contract_schema_version"),
        mapping_metadata.contract_schema_version,
        "contract_schema_version",
    )
    _require_equal(provenance.get("tool_version"), tool_version, "tool_version")
    _require_equal(provenance.get("row_count"), row_count, "row_count")
    _require_equal(
        provenance.get("allowed_changed_fields"),
        list(allowed_changed_fields),
        "allowed_changed_fields",
    )
    _require_equal(provenance.get("model_assisted"), True, "model_assisted")
    _require_equal(provenance.get("human_gold"), False, "human_gold")

    if annotator_id is not None:
        if provenance.get("annotator_id") != annotator_id or "annotator_ids" in provenance:
            raise ValueError("provenance annotator identity mismatch")
    elif annotator_ids is not None:
        if provenance.get("annotator_ids") != list(annotator_ids) or "annotator_id" in provenance:
            raise ValueError("provenance annotator identity mismatch")

    expected_inputs = [
        _file_record(logical_name, input_files[logical_name])
        for logical_name in sorted(input_files)
    ]
    actual_inputs = provenance.get("inputs")
    if not isinstance(actual_inputs, list) or len(actual_inputs) != len(expected_inputs):
        raise ValueError("provenance input SHA-256 records mismatch")
    for actual, expected in zip(actual_inputs, expected_inputs):
        if actual != expected:
            raise ValueError("provenance input SHA-256 records mismatch")

    output_record = provenance.get("output")
    if not isinstance(output_record, dict):
        raise ValueError("provenance output record is missing")
    current_output_sha = sha256_file(output_path)
    if output_record.get("sha256") != current_output_sha:
        raise ValueError("provenance output SHA-256 mismatch")
