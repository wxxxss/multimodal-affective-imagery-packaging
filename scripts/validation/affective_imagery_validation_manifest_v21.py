#!/usr/bin/env python3
"""Read-only derivation and atomic refresh of the F validation manifest."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import tempfile
from collections import OrderedDict
from pathlib import Path
from typing import Any

import pandas as pd

try:
    from .prepare_affective_imagery_validation_v21 import (
        ADJUDICATION_TEMPLATE_FILENAME,
        ANNOTATIONS_A1_FILENAME,
        ANNOTATIONS_A2_FILENAME,
        CONTEXT_FILENAME,
        DEFAULT_SELECTION_SEED,
        MANIFEST_FILENAME,
        PREPARE_OUTPUTS,
        PRODUCT_DIMENSION_ITEMS_FILENAME,
        REVIEWER_CONTEXT_FILENAME,
        SENTENCE_ITEMS_FILENAME,
        TOOL_VERSION,
        UNLABELED_CONTEXT_CAP,
        build_product_dimension_reviewer_context,
        read_csv,
        sha256_file,
    )
    from .validate_affective_imagery_annotations_v21 import (
        validate_context_frame,
        validate_reviewer_context,
    )
except ImportError:  # pragma: no cover - direct script execution fallback
    from prepare_affective_imagery_validation_v21 import (
        ADJUDICATION_TEMPLATE_FILENAME,
        ANNOTATIONS_A1_FILENAME,
        ANNOTATIONS_A2_FILENAME,
        CONTEXT_FILENAME,
        DEFAULT_SELECTION_SEED,
        MANIFEST_FILENAME,
        PREPARE_OUTPUTS,
        PRODUCT_DIMENSION_ITEMS_FILENAME,
        REVIEWER_CONTEXT_FILENAME,
        SENTENCE_ITEMS_FILENAME,
        TOOL_VERSION,
        UNLABELED_CONTEXT_CAP,
        build_product_dimension_reviewer_context,
        read_csv,
        sha256_file,
    )
    from validate_affective_imagery_annotations_v21 import (
        validate_context_frame,
        validate_reviewer_context,
    )


PROTECTED_WORKSPACE_FILENAMES = [
    SENTENCE_ITEMS_FILENAME,
    PRODUCT_DIMENSION_ITEMS_FILENAME,
    ANNOTATIONS_A1_FILENAME,
    ANNOTATIONS_A2_FILENAME,
    ADJUDICATION_TEMPLATE_FILENAME,
    CONTEXT_FILENAME,
    REVIEWER_CONTEXT_FILENAME,
]

TIER_CATEGORY_MEMBERS: "OrderedDict[str, tuple[str, ...]]" = OrderedDict(
    [
        (
            "formal",
            (
                "mandatory_focus_or_direct_target",
                "formal_other_outer",
            ),
        ),
        ("strict", ("upstream_visual_strict",)),
        ("uncertain", ("upstream_uncertain",)),
        ("excluded", ("upstream_excluded",)),
        ("other", ("other_candidate",)),
    ]
)

COUNT_FIELDS = [
    "candidate_count",
    "initial_quota_selected_count",
    "backfill_selected_count",
    "final_selected_count",
]


def _workspace_hashes(output_dir: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for filename in PROTECTED_WORKSPACE_FILENAMES:
        path = output_dir / filename
        if not path.is_file():
            raise FileNotFoundError(f"missing protected F workspace file: {path}")
        hashes[filename] = sha256_file(path)
    return hashes


def _manifest_json_text(manifest: dict[str, Any]) -> str:
    return json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    except Exception:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
        raise


def _frame_as_strings(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.fillna("").astype(str).reset_index(drop=True)


def _assert_reviewer_rows_unchanged(
    existing: pd.DataFrame,
    rebuilt: pd.DataFrame,
) -> None:
    if list(existing.columns) != list(rebuilt.columns):
        raise ValueError(
            "existing 08 columns do not match deterministic reviewer packet columns"
        )
    left = _frame_as_strings(existing)
    right = _frame_as_strings(rebuilt)
    if left.shape != right.shape or not left.equals(right):
        raise ValueError(
            "existing 08 rows/order do not match deterministic in-memory selection; "
            "manifest-only refresh refuses to rewrite or reinterpret 08"
        )


def _tier_category_stats(
    tier_stats: dict[str, dict[str, Any]],
) -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = {}
    for category, members in TIER_CATEGORY_MEMBERS.items():
        result[category] = {
            field: sum(
                int(tier_stats.get(member, {}).get(field, 0) or 0)
                for member in members
            )
            for field in COUNT_FIELDS
        }
    return result


def _annotations_started(output_dir: Path) -> bool:
    for filename in [ANNOTATIONS_A1_FILENAME, ANNOTATIONS_A2_FILENAME]:
        frame = read_csv(output_dir / filename)
        human_columns = [column for column in frame.columns if column.startswith("human_")]
        for column in human_columns:
            if frame[column].astype(str).str.strip().ne("").any():
                return True
    return False


def _unlabeled_candidate_context_counts(
    context: pd.DataFrame,
    product_items: pd.DataFrame,
) -> tuple[int, int]:
    model_labels = pd.to_numeric(
        product_items["model_label_value"], errors="coerce"
    ).fillna(0).astype(int)
    unlabeled_ids = product_items.loc[
        model_labels == 0, "annotation_item_id"
    ].astype(str).tolist()
    with_candidates = 0
    without_candidates = 0
    context_ids = context["annotation_item_id"].astype(str)
    for aid in unlabeled_ids:
        sub = context.loc[context_ids == aid]
        if (
            len(sub) == 1
            and str(sub.iloc[0].get("context_status", "")).strip()
            == "no_candidate_context"
        ):
            without_candidates += 1
        else:
            with_candidates += 1
    return with_candidates, without_candidates


def _build_refreshed_manifest(
    output_dir: Path,
    current_manifest: dict[str, Any],
    protected_hashes: dict[str, str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    sentence_items = read_csv(output_dir / SENTENCE_ITEMS_FILENAME)
    product_items = read_csv(output_dir / PRODUCT_DIMENSION_ITEMS_FILENAME)
    context = read_csv(output_dir / CONTEXT_FILENAME)
    reviewer_existing = read_csv(output_dir / REVIEWER_CONTEXT_FILENAME)

    context_summary = validate_context_frame(context, product_items)
    validate_reviewer_context(reviewer_existing, product_items, context)

    reviewer_packet_old = current_manifest.get("reviewer_packet")
    if not isinstance(reviewer_packet_old, dict):
        reviewer_packet_old = {}
    selection_seed = int(
        reviewer_packet_old.get("selection_seed", DEFAULT_SELECTION_SEED)
        or DEFAULT_SELECTION_SEED
    )
    unlabeled_cap = int(
        reviewer_packet_old.get("unlabeled_context_cap", UNLABELED_CONTEXT_CAP)
        or UNLABELED_CONTEXT_CAP
    )

    reviewer_rebuilt, reviewer_stats = build_product_dimension_reviewer_context(
        context,
        product_items,
        selection_seed=selection_seed,
        unlabeled_cap=unlabeled_cap,
    )
    _assert_reviewer_rows_unchanged(reviewer_existing, reviewer_rebuilt)

    raw_tier_stats = reviewer_stats.get("tier_stats", {})
    final_tier_total = sum(
        int(stats.get("final_selected_count", 0) or 0)
        for stats in raw_tier_stats.values()
    )
    total_backfill = sum(
        int(stats.get("backfill_selected_count", 0) or 0)
        for stats in raw_tier_stats.values()
    )
    if final_tier_total != int(reviewer_stats.get("unlabeled_rows", 0) or 0):
        raise ValueError(
            "tier final_selected_count total does not equal unlabeled reviewer rows"
        )
    if total_backfill != int(reviewer_stats.get("total_backfill_rows", 0) or 0):
        raise ValueError("tier backfill counts do not equal total_backfill_rows")

    tier_category_stats = _tier_category_stats(raw_tier_stats)
    unlabeled_with_candidates, unlabeled_without_candidates = (
        _unlabeled_candidate_context_counts(context, product_items)
    )

    refreshed = copy.deepcopy(current_manifest)
    refreshed["tool_version"] = TOOL_VERSION
    refreshed["output_dir"] = str(output_dir.resolve())
    output_hashes = refreshed.get("output_hashes")
    if not isinstance(output_hashes, dict):
        output_hashes = {}
    output_hashes.update(protected_hashes)
    refreshed["output_hashes"] = output_hashes
    refreshed["generated_files"] = list(PREPARE_OUTPUTS)
    if "audit_group" in sentence_items.columns:
        refreshed["sentence_group_counts"] = {
            str(group): int(count)
            for group, count in sentence_items["audit_group"].value_counts().items()
        }
    refreshed["sentence_item_count"] = int(len(sentence_items))
    refreshed["product_dimension_item_count"] = int(len(product_items))
    refreshed["annotation_item_count"] = int(len(sentence_items) + len(product_items))
    refreshed["context_scope"] = (
        "07_product_dimension_evidence_context.csv is the complete F product-dimension "
        "review context. Positive tasks must have target-dimension evidence; outer target "
        "evidence is preferred, while positive items with only non-outer target evidence "
        "are retained and explicitly reported for adjudication. Unlabeled review remains "
        "Positive-Unlabeled and does not establish confirmed negatives."
    )
    refreshed["product_dimension_context_row_count"] = int(len(context))
    refreshed["context_annotation_item_count"] = int(
        context["annotation_item_id"].astype(str).nunique()
    )
    refreshed["positive_context_item_count"] = int(
        context_summary["positive_item_count"]
    )
    refreshed["unlabeled_context_item_count"] = int(
        context_summary["unlabeled_item_count"]
    )
    refreshed["positive_items_with_outer_evidence"] = int(
        context_summary["positive_items_with_outer_evidence"]
    )
    refreshed["positive_items_without_outer_evidence"] = int(
        context_summary["positive_items_without_outer_evidence"]
    )
    refreshed["positive_no_outer_annotation_item_ids"] = list(
        context_summary["positive_no_outer_annotation_item_ids"]
    )
    refreshed["unlabeled_items_with_candidate_context"] = unlabeled_with_candidates
    refreshed["unlabeled_items_without_candidate_context"] = unlabeled_without_candidates
    refreshed["no_silent_truncation"] = True
    refreshed["annotations_started"] = _annotations_started(output_dir)

    reviewer_packet = copy.deepcopy(reviewer_packet_old)
    reviewer_packet.update(
        {
            "filename": REVIEWER_CONTEXT_FILENAME,
            "selection_seed": selection_seed,
            "unlabeled_context_cap": unlabeled_cap,
            "reviewer_context_row_count": int(reviewer_stats.get("row_count", 0) or 0),
            "reviewer_annotation_item_count": int(
                reviewer_stats.get("annotation_item_count", 0) or 0
            ),
            "positive_reviewer_rows": int(reviewer_stats.get("positive_rows", 0) or 0),
            "unlabeled_reviewer_rows": int(reviewer_stats.get("unlabeled_rows", 0) or 0),
            "duplicates_removed": int(reviewer_stats.get("duplicates_removed", 0) or 0),
            "truncated_unlabeled_count": int(
                reviewer_stats.get("truncated_unlabeled_count", 0) or 0
            ),
            "exhaustive_unlabeled_count": int(
                reviewer_stats.get("exhaustive_unlabeled_count", 0) or 0
            ),
            "all_tasks_distribution": reviewer_stats.get("all_tasks", {}),
            "positive_tasks_distribution": reviewer_stats.get("positive_tasks", {}),
            "unlabeled_tasks_distribution": reviewer_stats.get("unlabeled_tasks", {}),
            "positive_items_with_outer_evidence": int(
                reviewer_stats.get("positive_items_with_outer_evidence", 0) or 0
            ),
            "positive_items_without_outer_evidence": int(
                reviewer_stats.get("positive_items_without_outer_evidence", 0) or 0
            ),
            "positive_no_outer_annotation_item_ids": list(
                reviewer_stats.get("positive_no_outer_annotation_item_ids", [])
            ),
            "total_backfill_rows": int(
                reviewer_stats.get("total_backfill_rows", 0) or 0
            ),
            "tier_definitions": reviewer_stats.get("tier_definitions", {}),
            "tier_stats": raw_tier_stats,
            "tier_category_stats": tier_category_stats,
            "tier_category_members": {
                category: list(members)
                for category, members in TIER_CATEGORY_MEMBERS.items()
            },
        }
    )
    refreshed["reviewer_packet"] = reviewer_packet
    refreshed["manifest_refresh"] = {
        "mode": "manifest_only",
        "protected_files_sha256": protected_hashes,
        "protected_files_unchanged_required": True,
        "reviewer_rows_rebuilt_in_memory_only": True,
        "reviewer_rows_match_existing_08": True,
    }

    derived_summary = {
        "positive_items_with_outer_evidence": int(
            context_summary["positive_items_with_outer_evidence"]
        ),
        "positive_items_without_outer_evidence": int(
            context_summary["positive_items_without_outer_evidence"]
        ),
        "positive_no_outer_annotation_item_ids": list(
            context_summary["positive_no_outer_annotation_item_ids"]
        ),
        "tier_category_stats": tier_category_stats,
        "total_backfill_rows": int(reviewer_stats.get("total_backfill_rows", 0) or 0),
        "reviewer_context_row_count": int(reviewer_stats.get("row_count", 0) or 0),
        "positive_reviewer_rows": int(reviewer_stats.get("positive_rows", 0) or 0),
        "unlabeled_reviewer_rows": int(reviewer_stats.get("unlabeled_rows", 0) or 0),
    }
    return refreshed, derived_summary


def refresh_validation_manifest_only(
    output_dir: Path,
    *,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Refresh only 06_validation_manifest.json, or compute it without writes."""
    output_dir = Path(output_dir)
    manifest_path = output_dir / MANIFEST_FILENAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"missing validation manifest: {manifest_path}")

    protected_before = _workspace_hashes(output_dir)
    old_manifest_bytes = manifest_path.read_bytes()
    old_manifest_sha = _sha256_bytes(old_manifest_bytes)
    try:
        current_manifest = json.loads(old_manifest_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"validation manifest is not valid UTF-8 JSON: {manifest_path}") from exc
    if not isinstance(current_manifest, dict):
        raise ValueError("validation manifest must be a JSON object")

    refreshed, derived = _build_refreshed_manifest(
        output_dir,
        current_manifest,
        protected_before,
    )
    prospective_bytes = _manifest_json_text(refreshed).encode("utf-8")
    prospective_sha = _sha256_bytes(prospective_bytes)

    if dry_run:
        protected_after = _workspace_hashes(output_dir)
        if protected_after != protected_before:
            raise RuntimeError("protected F workspace files changed during manifest dry-run")
        return {
            "dry_run": True,
            "manifest_path": str(manifest_path.resolve()),
            "manifest_sha256_before": old_manifest_sha,
            "manifest_sha256_after": old_manifest_sha,
            "prospective_manifest_sha256": prospective_sha,
            "would_change_manifest": prospective_sha != old_manifest_sha,
            "protected_files_unchanged": True,
            "protected_files_sha256": protected_before,
            **derived,
        }

    _atomic_write_bytes(manifest_path, prospective_bytes)
    protected_after = _workspace_hashes(output_dir)
    if protected_after != protected_before:
        _atomic_write_bytes(manifest_path, old_manifest_bytes)
        changed = [
            filename
            for filename in PROTECTED_WORKSPACE_FILENAMES
            if protected_before.get(filename) != protected_after.get(filename)
        ]
        raise RuntimeError(
            "protected F workspace files changed during manifest-only refresh; "
            f"06 was rolled back; changed={changed}"
        )

    manifest_after_sha = sha256_file(manifest_path)
    if manifest_after_sha != prospective_sha:
        _atomic_write_bytes(manifest_path, old_manifest_bytes)
        raise RuntimeError("atomic manifest write hash mismatch; 06 was rolled back")

    return {
        "dry_run": False,
        "manifest_path": str(manifest_path.resolve()),
        "manifest_sha256_before": old_manifest_sha,
        "manifest_sha256_after": manifest_after_sha,
        "prospective_manifest_sha256": prospective_sha,
        "would_change_manifest": prospective_sha != old_manifest_sha,
        "protected_files_unchanged": True,
        "protected_files_sha256": protected_before,
        **derived,
    }
