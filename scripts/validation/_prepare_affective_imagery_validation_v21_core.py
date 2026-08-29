#!/usr/bin/env python3
"""Prepare executable v2.1 affective imagery human-validation files.

This script does not create or infer human annotation decisions. It checks the
frozen v2.1 validation inputs, creates deterministic item IDs, samples
product-dimension checks, and writes blank A1/A2/adjudication templates.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import OrderedDict
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


TOOL_VERSION = "affective_imagery_validation_v21.3"
DEFAULT_OUTPUT_DIR = Path("data/manual_validation/affective_imagery_v21")
DEFAULT_SELECTION_SEED = 20260810
UNLABELED_CONTEXT_CAP = 30

AUDIT_SAMPLE_FILENAME = "41_relation_constrained_audit_sample_v21.csv"
DIMENSIONS_FILENAME = "38_relation_constrained_imagery_dimensions_v21.csv"
PRODUCT_LABELS_FILENAME = "39_product_imagery_labels_v21.csv"
PRODUCT_DIMENSION_FILENAME = "39b_product_dimension_evidence_v21.csv"
SUMMARY_FILENAME = "40_relation_constrained_summary_v21.json"

SENTENCE_ITEMS_FILENAME = "01_sentence_items.csv"
PRODUCT_DIMENSION_ITEMS_FILENAME = "02_product_dimension_items.csv"
ANNOTATIONS_A1_FILENAME = "03_annotations_A1.csv"
ANNOTATIONS_A2_FILENAME = "04_annotations_A2.csv"
ADJUDICATION_TEMPLATE_FILENAME = "05_adjudication_template.csv"
MANIFEST_FILENAME = "06_validation_manifest.json"
CONTEXT_FILENAME = "07_product_dimension_evidence_context.csv"
REVIEWER_CONTEXT_FILENAME = "08_product_dimension_reviewer_context.csv"

GROUP_TARGETS: "OrderedDict[str, int]" = OrderedDict(
    [
        ("outer_relation_evidence", 120),
        ("recovered_v21", 120),
        ("uncertain_not_recovered", 120),
        ("inner_relation_evidence", 80),
        ("ambiguous_relation_evidence", 80),
        ("strict_without_relation_evidence", 80),
    ]
)

RELATION_EVIDENCE_GROUPS = {
    "outer_relation_evidence",
    "inner_relation_evidence",
    "ambiguous_relation_evidence",
    "recovered_v21",
}

FALSE_NEGATIVE_GROUPS = {
    "uncertain_not_recovered",
    "strict_without_relation_evidence",
}

DIMENSION_CODES = [
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
]

SENTENCE_CONTEXT_COLUMNS = [
    "annotation_item_id",
    "item_type",
    "sample_order",
    "parent_asin",
    "review_id",
    "sentence_id",
    "user_id",
    "audit_group",
    "sentence",
    "normalized_sentence",
    "clause_text",
    "clause_index",
    "relation_type",
    "object_term",
    "package_level",
    "eligible_for_main_image_model",
    "negated",
    "source_kind",
    "source_type",
    "dimension_code",
    "dimension_name_cn",
    "polarity",
    "expression_raw",
    "expression_lemma",
    "rating",
    "verified_purchase",
    "helpful_vote",
    "pipeline_version",
]

PRODUCT_DIMENSION_CONTEXT_COLUMNS = [
    "annotation_item_id",
    "item_type",
    "sampling_stratum",
    "parent_asin",
    "dimension_code",
    "dimension_name_cn",
    "model_label_value",
    "label_interpretation",
    "label_semantics",
    "keep_for_core_model",
    "keep_for_pilot",
    "product_dimension_sentence_count",
    "product_dimension_review_count",
    "product_dimension_reviewer_count",
    "exposure_review_count",
    "traceable_predicted_evidence",
    "sampling_seed",
    "sampling_rank",
]

EVIDENCE_CONTEXT_COLUMNS = [
    "annotation_item_id",
    "parent_asin",
    "target_dimension_code",
    "model_label_value",
    "label_interpretation",
    "sampling_stratum",
    "context_rank",
    "context_source",
    "context_status",
    "review_id",
    "sentence_id",
    "sentence",
    "normalized_sentence",
    "clause_text",
    "relation_type",
    "object_term",
    "expression_raw",
    "expression_lemma",
    "predicted_dimension_code",
    "package_level",
    "eligible_for_main_image_model",
    "source_kind",
    "source_type",
    "upstream_decision",
    "upstream_reason",
    "is_target_dimension_evidence",
    "is_outer_eligible_evidence",
    "focus_review_flag",
    "focus_review_reason",
]

FOCUS_REVIEW_ITEMS = {
    "prod-96d1031a1b179bc5": {
        "parent_asin": "B0BWLWY25M",
        "reason_en": "presentation appears alongside tea bags; verify whether the inner reclassification is correct",
        "sentence_item_id": "sent-0d82bad8e43882b8",
    },
    "prod-4b13f829c881c342": {
        "parent_asin": "B0C5ZMZBKS",
        "reason_en": "artwork may describe outer retail packaging; check possible false negative",
        "sentence_item_id": "sent-228e38f8b6424b47",
    },
}

ANNOTATION_COLUMNS = [
    "annotator_id",
    "annotation_round",
    "human_packaging_visual",
    "human_relation_valid",
    "human_package_level",
    "human_dimension_code",
    "human_additional_dimension_codes",
    "human_polarity",
    "human_action",
    "human_error_type",
    "human_product_label_traceable",
    "human_unlabeled_missed_signal",
    "human_confidence",
    "human_rationale_cn",
]

ADJUDICATION_COLUMNS = [
    "adjudicated_packaging_visual",
    "adjudicated_relation_valid",
    "adjudicated_package_level",
    "adjudicated_dimension_code",
    "adjudicated_additional_dimension_codes",
    "adjudicated_polarity",
    "adjudicated_action",
    "adjudicated_error_type",
    "adjudicated_product_label_traceable",
    "adjudicated_unlabeled_missed_signal",
    "adjudication_note_cn",
]

PREPARE_OUTPUTS = [
    SENTENCE_ITEMS_FILENAME,
    PRODUCT_DIMENSION_ITEMS_FILENAME,
    ANNOTATIONS_A1_FILENAME,
    ANNOTATIONS_A2_FILENAME,
    ADJUDICATION_TEMPLATE_FILENAME,
    MANIFEST_FILENAME,
    CONTEXT_FILENAME,
    REVIEWER_CONTEXT_FILENAME,
]


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except TypeError:
        pass
    return str(value).strip()


def _is_blank(value: Any) -> bool:
    return _clean_text(value) == ""


def _require_columns(frame: pd.DataFrame, columns: Iterable[str], context: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{context} missing required columns: {', '.join(missing)}")


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig", keep_default_na=False)


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_id(prefix: str, parts: Iterable[Any]) -> str:
    payload = "\x1f".join(_clean_text(part) for part in parts)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
    return f"{prefix}-{digest}"


def _stable_rank(seed: int, *parts: Any) -> float:
    payload = f"{seed}\x1f" + "\x1f".join(_clean_text(part) for part in parts)
    integer = int(hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16], 16)
    return integer / float(16**16)


def _load_manifest(manifest: dict[str, Any] | Path | None) -> dict[str, Any] | None:
    if manifest is None:
        return None
    if isinstance(manifest, Path):
        try:
            loaded = json.loads(manifest.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"source manifest is not valid JSON: {manifest}") from exc
    else:
        loaded = manifest
    if not isinstance(loaded, dict):
        raise ValueError("source manifest must be a JSON object")
    return loaded


def _non_negative_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field} must be a non-negative integer")
    return int(value)


def _manifest_group_mapping(manifest: dict[str, Any], field: str) -> "OrderedDict[str, int]":
    value = manifest.get(field)
    expected_keys = set(GROUP_TARGETS.keys())
    if not isinstance(value, dict) or set(value) != expected_keys:
        raise ValueError(f"{field} must contain exact sentence group keys")
    return OrderedDict(
        (group, _non_negative_int(value[group], f"{field}.{group}"))
        for group in GROUP_TARGETS
    )


def _audit_targets_from_total(sample_size: int) -> "OrderedDict[str, int]":
    if sample_size <= 0:
        return OrderedDict((group, 0) for group in GROUP_TARGETS)

    official_total = sum(GROUP_TARGETS.values())
    if sample_size == official_total:
        return OrderedDict(GROUP_TARGETS)

    targets: "OrderedDict[str, int]" = OrderedDict((group, 0) for group in GROUP_TARGETS)
    remaining = sample_size
    if sample_size >= len(GROUP_TARGETS):
        for group in GROUP_TARGETS:
            targets[group] = 1
        remaining -= len(GROUP_TARGETS)

    scaled: list[tuple[float, int, str, int]] = []
    for order, (group, quota) in enumerate(GROUP_TARGETS.items()):
        raw = quota * sample_size / official_total
        base = int(raw)
        scaled.append((raw - base, order, group, base))

    if sample_size < len(GROUP_TARGETS):
        for _, _, group, base in scaled:
            targets[group] = base
        remaining = sample_size - sum(targets.values())
    else:
        for _, _, group, base in scaled:
            targets[group] = max(targets[group], base)
        remaining = sample_size - sum(targets.values())

    for _, _, group, _ in sorted(scaled, key=lambda item: (-item[0], item[1])):
        if remaining <= 0:
            break
        targets[group] += 1
        remaining -= 1

    while remaining > 0:
        for group in GROUP_TARGETS:
            if remaining <= 0:
                break
            targets[group] += 1
            remaining -= 1

    while sum(targets.values()) > sample_size:
        for group in reversed(GROUP_TARGETS):
            if targets[group] > 0 and sum(targets.values()) > sample_size:
                targets[group] -= 1

    return targets


def _redistribute_group_targets(
    capacities: "OrderedDict[str, int]",
    sample_size: int,
) -> "OrderedDict[str, int]":
    total_available = sum(capacities.values())
    if sample_size <= 0 or total_available <= 0:
        return OrderedDict((group, 0) for group in GROUP_TARGETS)
    if total_available <= sample_size:
        return OrderedDict((group, capacities[group]) for group in GROUP_TARGETS)

    requested = _audit_targets_from_total(sample_size)
    selected: "OrderedDict[str, int]" = OrderedDict(
        (group, min(requested[group], capacities[group]))
        for group in GROUP_TARGETS
    )
    remaining = sample_size - sum(selected.values())
    while remaining > 0:
        progressed = False
        for group in GROUP_TARGETS:
            if remaining <= 0:
                break
            if selected[group] < capacities[group]:
                selected[group] += 1
                remaining -= 1
                progressed = True
        if not progressed:
            raise ValueError("unable to redistribute audit sample quota")
    return selected


def _audit_redistribution_records(
    capacities: "OrderedDict[str, int]",
    requested_quotas: "OrderedDict[str, int]",
    final_quotas: "OrderedDict[str, int]",
) -> list[dict[str, Any]]:
    recipient_remaining = {
        group: max(
            0,
            final_quotas[group] - min(requested_quotas[group], capacities[group]),
        )
        for group in GROUP_TARGETS
    }
    records: list[dict[str, Any]] = []
    for shortage_group in GROUP_TARGETS:
        requested = requested_quotas[shortage_group]
        capacity = capacities[shortage_group]
        shortage = max(0, requested - capacity)
        if shortage == 0:
            continue
        remaining = shortage
        recipients: dict[str, int] = {}
        for recipient in GROUP_TARGETS:
            if remaining <= 0:
                break
            increment = min(recipient_remaining[recipient], remaining)
            if increment <= 0:
                continue
            recipients[recipient] = increment
            recipient_remaining[recipient] -= increment
            remaining -= increment
        records.append(
            {
                "shortage_group": shortage_group,
                "requested_quota": requested,
                "capacity": capacity,
                "shortage": shortage,
                "recipient_increments": recipients,
                "unfilled_shortage": remaining,
            }
        )
    return records


def _validate_source_audit_manifest(
    source_manifest: dict[str, Any] | Path,
) -> dict[str, Any]:
    manifest = _load_manifest(source_manifest)
    if manifest is None:
        raise ValueError("source manifest is required")

    capacities = _manifest_group_mapping(manifest, "sentence_group_capacities")
    requested = _manifest_group_mapping(manifest, "sentence_group_requested_quotas")
    final = _manifest_group_mapping(manifest, "sentence_group_final_quotas")
    actual = _manifest_group_mapping(manifest, "sentence_group_actual_counts")
    audit_seed = _non_negative_int(manifest.get("audit_seed"), "audit_seed")
    requested_size = _non_negative_int(
        manifest.get("audit_requested_sample_size"),
        "audit_requested_sample_size",
    )
    actual_size = _non_negative_int(
        manifest.get("audit_actual_sample_size"),
        "audit_actual_sample_size",
    )

    expected_requested = _audit_targets_from_total(requested_size)
    if requested != expected_requested:
        raise ValueError("sentence_group_requested_quotas do not match requested sample size")
    if any(final[group] > capacities[group] for group in GROUP_TARGETS):
        raise ValueError("final quota exceeds group capacity")
    expected_final = _redistribute_group_targets(capacities, requested_size)
    if final != expected_final:
        raise ValueError("sentence_group_final_quotas do not match deterministic final quotas")
    if actual != final:
        raise ValueError("actual counts must equal final quotas")
    if actual_size != sum(actual.values()):
        raise ValueError("audit_actual_sample_size does not match actual counts")

    redistribution = manifest.get("audit_redistribution")
    if not isinstance(redistribution, list):
        raise ValueError("audit_redistribution must be a list")
    expected_redistribution = _audit_redistribution_records(
        capacities,
        requested,
        final,
    )
    if redistribution != expected_redistribution:
        raise ValueError("audit_redistribution does not match deterministic redistribution")

    return {
        "audit_seed": audit_seed,
        "audit_requested_sample_size": requested_size,
        "audit_actual_sample_size": actual_size,
        "sentence_group_capacities": capacities,
        "sentence_group_requested_quotas": requested,
        "sentence_group_final_quotas": final,
        "sentence_group_actual_counts": actual,
        "audit_redistribution": expected_redistribution,
    }


def expected_group_counts(
    source_manifest: dict[str, Any] | Path | None = None,
    *,
    total_rows: int = 600,
) -> "OrderedDict[str, int]":
    if source_manifest is None:
        return _audit_targets_from_total(total_rows)
    contract = _validate_source_audit_manifest(source_manifest)
    if total_rows != contract["audit_actual_sample_size"]:
        raise ValueError("audit_actual_sample_size does not match expected_rows")
    return OrderedDict(contract["sentence_group_final_quotas"])


def _audit_duplicate_key(row: pd.Series) -> str:
    return "\x1f".join(
        _clean_text(row.get(column))
        for column in [
            "parent_asin",
            "review_id",
            "sentence_id",
            "clause_text",
            "object_term",
            "dimension_code",
            "expression_lemma",
        ]
    )


def validate_sentence_items_input(
    audit_sample: pd.DataFrame,
    *,
    expected_rows: int = 600,
    source_manifest: dict[str, Any] | Path | None = None,
) -> None:
    manifest_contract = (
        _validate_source_audit_manifest(source_manifest)
        if source_manifest is not None
        else None
    )
    effective_expected_rows = expected_rows
    if manifest_contract is not None:
        if expected_rows != 600 and expected_rows != manifest_contract["audit_actual_sample_size"]:
            raise ValueError("expected_rows must match audit_actual_sample_size")
        effective_expected_rows = manifest_contract["audit_actual_sample_size"]

    _require_columns(audit_sample, ["sample_order", "audit_group"], "audit sample")
    if len(audit_sample) != effective_expected_rows:
        raise ValueError(
            f"audit sample must contain {effective_expected_rows} rows, got {len(audit_sample)}"
        )

    sample_order = pd.to_numeric(audit_sample["sample_order"], errors="coerce")
    expected_order = list(range(1, effective_expected_rows + 1))
    if sample_order.isna().any() or sample_order.astype(int).tolist() != expected_order:
        raise ValueError("sample_order must start at 1 and increase by 1 without gaps")
    if audit_sample["sample_order"].duplicated().any():
        raise ValueError("sample_order must be unique")

    allowed_groups = set(GROUP_TARGETS.keys())
    unknown_groups = sorted(set(audit_sample["audit_group"].astype(str)) - allowed_groups)
    if unknown_groups:
        raise ValueError(f"unknown audit_group values: {', '.join(unknown_groups)}")

    actual_counts = OrderedDict(
        (group, int((audit_sample["audit_group"] == group).sum()))
        for group in GROUP_TARGETS
    )
    expected_counts = expected_group_counts(
        manifest_contract,
        total_rows=effective_expected_rows,
    )
    if actual_counts != expected_counts:
        raise ValueError(
            "audit_group quotas do not match expected counts: "
            f"actual={dict(actual_counts)} expected={dict(expected_counts)}"
        )

    relation_required = [
        "parent_asin",
        "sentence_id",
        "sentence",
        "clause_text",
        "object_term",
        "package_level",
        "dimension_code",
        "expression_raw",
        "expression_lemma",
    ]
    _require_columns(audit_sample, relation_required, "audit sample")
    relation_mask = audit_sample["audit_group"].isin(RELATION_EVIDENCE_GROUPS)
    for column in relation_required:
        blank_mask = audit_sample.loc[relation_mask, column].map(_is_blank)
        if bool(blank_mask.any()):
            raise ValueError(f"relation evidence rows must not have empty {column}")

    duplicates = (
        audit_sample.assign(_audit_key=audit_sample.apply(_audit_duplicate_key, axis=1))
        .groupby("_audit_key")["audit_group"]
        .nunique()
    )
    if bool((duplicates > 1).any()):
        raise ValueError("audit items must not be duplicated across audit groups")


def build_sentence_items(audit_sample: pd.DataFrame) -> pd.DataFrame:
    frame = audit_sample.copy()
    for column in SENTENCE_CONTEXT_COLUMNS:
        if column not in frame.columns and column not in {"annotation_item_id", "item_type"}:
            frame[column] = ""
    frame["item_type"] = "sentence"
    frame["annotation_item_id"] = [
        _stable_id(
            "sent",
            [
                row.get("sample_order"),
                row.get("parent_asin"),
                row.get("review_id"),
                row.get("sentence_id"),
                row.get("audit_group"),
                row.get("clause_text"),
                row.get("object_term"),
                row.get("dimension_code"),
                row.get("expression_lemma"),
            ],
        )
        for _, row in frame.iterrows()
    ]
    if frame["annotation_item_id"].duplicated().any():
        raise ValueError("sentence annotation_item_id values must be unique")
    return frame[SENTENCE_CONTEXT_COLUMNS].copy()


def _int_value(value: Any, default: int = 0) -> int:
    try:
        if pd.isna(value):
            return default
    except TypeError:
        pass
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _dimension_priority(dimensions: pd.DataFrame, positive_counts: dict[str, int]) -> dict[str, int]:
    records: list[dict[str, Any]] = []
    for _, row in dimensions.iterrows():
        code = _clean_text(row.get("dimension_code"))
        records.append(
            {
                "dimension_code": code,
                "is_core": _int_value(row.get("keep_for_core_model")) == 1,
                "positive_count": positive_counts.get(code, 0),
            }
        )
    ordered = sorted(
        records,
        key=lambda item: (
            0 if item["is_core"] else 1,
            item["positive_count"],
            item["dimension_code"],
        ),
    )
    return {item["dimension_code"]: index for index, item in enumerate(ordered)}


def _label_value_for_product(labels: pd.Series, code: str) -> int:
    for column in [
        f"{code}_observed_positive_pilot",
        f"{code}_observed_positive_core",
        f"{code}_label_pilot",
        f"{code}_label_robust",
    ]:
        if column in labels.index and _int_value(labels[column]) == 1:
            return 1
    return 0


def _exposure_value(labels: pd.Series) -> int:
    for column in [
        "review_count",
        "clean_review_count",
        "matched_review_count",
        "outer_imagery_review_count",
        "outer_imagery_sentence_count",
    ]:
        if column in labels.index:
            return _int_value(labels[column])
    return 0


def _sample_positive_candidates(
    candidates: pd.DataFrame,
    *,
    seed: int,
    target: int,
    dimension_priority: dict[str, int],
) -> pd.DataFrame:
    if candidates.empty:
        return candidates.copy()
    ranked = candidates.copy()
    ranked["_dimension_priority"] = ranked["dimension_code"].map(dimension_priority)
    ranked["_random_rank"] = [
        _stable_rank(seed, row["parent_asin"], row["dimension_code"], "positive")
        for _, row in ranked.iterrows()
    ]
    ranked = ranked.sort_values(
        ["_dimension_priority", "_random_rank", "parent_asin", "dimension_code"]
    )
    return ranked.head(target).drop(columns=["_dimension_priority", "_random_rank"])


def build_product_dimension_items(
    dimensions: pd.DataFrame,
    product_labels: pd.DataFrame,
    product_dimension_evidence: pd.DataFrame,
    *,
    seed: int = 42,
    total_items: int = 240,
    unlabeled_items: int = 60,
    per_dimension_positive_cap: int = 30,
) -> pd.DataFrame:
    _require_columns(dimensions, ["dimension_code"], "dimensions")
    _require_columns(product_labels, ["parent_asin"], "product labels")
    _require_columns(
        product_dimension_evidence,
        ["parent_asin", "dimension_code"],
        "product-dimension evidence",
    )
    positive_target = total_items - unlabeled_items
    if positive_target <= 0:
        raise ValueError("total_items must be greater than unlabeled_items")

    evidence = product_dimension_evidence.copy()
    evidence["parent_asin"] = evidence["parent_asin"].map(_clean_text)
    evidence["dimension_code"] = evidence["dimension_code"].map(_clean_text)
    evidence = evidence.drop_duplicates(["parent_asin", "dimension_code"])
    positive_counts = evidence.groupby("dimension_code")["parent_asin"].nunique().to_dict()
    priority = _dimension_priority(dimensions, positive_counts)

    selected_groups: list[pd.DataFrame] = []
    for _, row in dimensions.iterrows():
        code = _clean_text(row.get("dimension_code"))
        group = evidence.loc[evidence["dimension_code"] == code].copy()
        if group.empty:
            continue
        group["_random_rank"] = [
            _stable_rank(seed, parent_asin, code, "dimension-cap")
            for parent_asin in group["parent_asin"]
        ]
        group = group.sort_values(["_random_rank", "parent_asin"]).head(
            min(per_dimension_positive_cap, len(group))
        )
        selected_groups.append(group.drop(columns=["_random_rank"]))

    if selected_groups:
        selected = pd.concat(selected_groups, ignore_index=True, sort=False)
    else:
        selected = pd.DataFrame(columns=evidence.columns)

    if len(selected) < positive_target:
        already = set(zip(selected["parent_asin"], selected["dimension_code"]))
        remaining = evidence.loc[
            ~evidence.apply(
                lambda row: (row["parent_asin"], row["dimension_code"]) in already,
                axis=1,
            )
        ]
        selected = pd.concat(
            [
                selected,
                _sample_positive_candidates(
                    remaining,
                    seed=seed,
                    target=positive_target - len(selected),
                    dimension_priority=priority,
                ),
            ],
            ignore_index=True,
            sort=False,
        )

    selected = _sample_positive_candidates(
        selected,
        seed=seed,
        target=positive_target,
        dimension_priority=priority,
    )
    if len(selected) < positive_target:
        raise ValueError(
            f"not enough positive product-dimension pairs: {len(selected)} < {positive_target}"
        )

    dimension_meta = dimensions.set_index("dimension_code").to_dict(orient="index")
    positive_rows: list[dict[str, Any]] = []
    for rank, (_, row) in enumerate(selected.iterrows(), start=1):
        code = _clean_text(row.get("dimension_code"))
        meta = dimension_meta.get(code, {})
        parent_asin = _clean_text(row.get("parent_asin"))
        positive_rows.append(
            {
                "annotation_item_id": _stable_id("prod", [parent_asin, code, "positive"]),
                "item_type": "product_dimension",
                "sampling_stratum": (
                    "positive_core"
                    if _int_value(meta.get("keep_for_core_model")) == 1
                    else "positive_pilot_low_frequency"
                ),
                "parent_asin": parent_asin,
                "dimension_code": code,
                "dimension_name_cn": _clean_text(meta.get("dimension_name_cn")) or code,
                "model_label_value": 1,
                "label_interpretation": "observed_positive",
                "label_semantics": "positive_unlabeled",
                "keep_for_core_model": _int_value(meta.get("keep_for_core_model")),
                "keep_for_pilot": _int_value(meta.get("keep_for_pilot"), 1),
                "product_dimension_sentence_count": _int_value(row.get("sentence_count")),
                "product_dimension_review_count": _int_value(row.get("review_count")),
                "product_dimension_reviewer_count": _int_value(row.get("reviewer_count")),
                "exposure_review_count": _int_value(row.get("review_count")),
                "traceable_predicted_evidence": 1,
                "sampling_seed": seed,
                "sampling_rank": rank,
            }
        )

    positive_pairs = set(zip(evidence["parent_asin"], evidence["dimension_code"]))
    unlabeled_candidates: list[dict[str, Any]] = []
    for _, label_row in product_labels.iterrows():
        parent_asin = _clean_text(label_row.get("parent_asin"))
        exposure = _exposure_value(label_row)
        for _, dim_row in dimensions.iterrows():
            code = _clean_text(dim_row.get("dimension_code"))
            if (parent_asin, code) in positive_pairs:
                continue
            if _label_value_for_product(label_row, code) == 1:
                continue
            unlabeled_candidates.append(
                {
                    "annotation_item_id": _stable_id("prod", [parent_asin, code, "unlabeled"]),
                    "item_type": "product_dimension",
                    "sampling_stratum": "high_exposure_unlabeled",
                    "parent_asin": parent_asin,
                    "dimension_code": code,
                    "dimension_name_cn": _clean_text(dim_row.get("dimension_name_cn")) or code,
                    "model_label_value": 0,
                    "label_interpretation": "unlabeled_not_observed",
                    "label_semantics": "positive_unlabeled",
                    "keep_for_core_model": _int_value(dim_row.get("keep_for_core_model")),
                    "keep_for_pilot": _int_value(dim_row.get("keep_for_pilot"), 1),
                    "product_dimension_sentence_count": 0,
                    "product_dimension_review_count": 0,
                    "product_dimension_reviewer_count": 0,
                    "exposure_review_count": exposure,
                    "traceable_predicted_evidence": 0,
                    "sampling_seed": seed,
                    "_dimension_priority": priority.get(code, 999),
                    "_random_rank": _stable_rank(seed, parent_asin, code, "unlabeled"),
                }
            )
    unlabeled = pd.DataFrame(unlabeled_candidates)
    if len(unlabeled) < unlabeled_items:
        raise ValueError(
            f"not enough unlabeled product-dimension pairs: {len(unlabeled)} < {unlabeled_items}"
        )
    unlabeled = (
        unlabeled.sort_values(
            [
                "exposure_review_count",
                "_dimension_priority",
                "_random_rank",
                "parent_asin",
                "dimension_code",
            ],
            ascending=[False, True, True, True, True],
        )
        .head(unlabeled_items)
        .drop(columns=["_dimension_priority", "_random_rank"])
    )
    unlabeled["sampling_rank"] = range(len(positive_rows) + 1, len(positive_rows) + 1 + len(unlabeled))

    output = pd.concat(
        [pd.DataFrame(positive_rows), unlabeled],
        ignore_index=True,
        sort=False,
    )
    if output["annotation_item_id"].duplicated().any():
        raise ValueError("product-dimension annotation_item_id values must be unique")
    return output[PRODUCT_DIMENSION_CONTEXT_COLUMNS].copy()


def build_annotation_template(items: pd.DataFrame, annotator_id: str) -> pd.DataFrame:
    frame = items.copy()
    for column in ANNOTATION_COLUMNS:
        if column not in frame.columns:
            frame[column] = ""
    frame["annotator_id"] = annotator_id
    frame["annotation_round"] = "1"
    return frame[list(frame.columns)].copy()


def build_blank_adjudication_template(items: pd.DataFrame) -> pd.DataFrame:
    frame = items.copy()
    for column in ADJUDICATION_COLUMNS:
        frame[column] = ""
    return frame.copy()


def _read_summary_label_semantics(path: Path) -> str:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return "positive_unlabeled"
    except json.JSONDecodeError as exc:
        raise ValueError(f"summary JSON is invalid: {path}") from exc
    return _clean_text(data.get("label_semantics")) or "positive_unlabeled"


def _write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def build_product_dimension_evidence_context(
    product_dimension_items: pd.DataFrame,
    relation_evidence: pd.DataFrame,
    upstream_classified: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build 07_product_dimension_evidence_context.csv from V2.1 evidence.

    For positive items: match outer_retail_package evidence from 37 parquet.
    For unlabeled items: provide all packaging candidate context from 15 parquet
    plus any other-dimension formal evidence from 37.
    """
    _require_columns(product_dimension_items, ["annotation_item_id", "parent_asin",
                                                 "dimension_code", "model_label_value",
                                                 "sampling_stratum"],
                      "product dimension items")

    positive_items = product_dimension_items[product_dimension_items["model_label_value"] == 1].copy()
    unlabeled_items = product_dimension_items[product_dimension_items["model_label_value"] == 0].copy()

    context_rows: list[dict[str, Any]] = []

    # ---- positive tasks ----
    for _, item in positive_items.iterrows():
        pid = _clean_text(item["parent_asin"])
        dim = _clean_text(item["dimension_code"])
        aid = _clean_text(item["annotation_item_id"])
        stratum = _clean_text(item.get("sampling_stratum", ""))

        # Find matching outer evidence
        mask = (
            (relation_evidence["parent_asin"].astype(str).str.strip() == pid) &
            (relation_evidence["dimension_code"].astype(str).str.strip() == dim)
        )
        all_target_evidence = relation_evidence[mask].copy()

        if len(all_target_evidence) == 0:
            raise ValueError(
                f"positive product-dimension item {aid} ({pid}, {dim}) "
                f"has NO evidence at all in V2.1 37 parquet"
            )

        # Separate outer eligible and non-outer evidence
        outer_eligible = all_target_evidence[
            (all_target_evidence["package_level"].astype(str).str.strip() == "outer_retail_package") &
            (all_target_evidence["eligible_for_main_image_model"].astype(int) == 1)
        ]
        non_outer = all_target_evidence.drop(outer_eligible.index)

        # Sort: outer eligible first, then non-outer by review_id, sentence_id
        sorted_outer = outer_eligible.sort_values(["review_id", "sentence_id", "dimension_code"])
        sorted_nonouter = non_outer.sort_values(["review_id", "sentence_id", "dimension_code"])

        rank = 0
        focus_flag = 1 if aid in FOCUS_REVIEW_ITEMS else 0
        focus_reason = FOCUS_REVIEW_ITEMS[aid]["reason_en"] if focus_flag else ""

        for _, ev in sorted_outer.iterrows():
            rank += 1
            context_rows.append({
                "annotation_item_id": aid, "parent_asin": pid,
                "target_dimension_code": dim, "model_label_value": 1,
                "label_interpretation": _clean_text(item.get("label_interpretation", "observed_positive")),
                "sampling_stratum": stratum, "context_rank": rank,
                "context_source": "formal_v21_outer_relation_evidence",
                "context_status": "target_dimension_evidence",
                "review_id": _clean_text(ev.get("review_id")),
                "sentence_id": _clean_text(ev.get("sentence_id")),
                "sentence": _clean_text(ev.get("sentence")),
                "normalized_sentence": _clean_text(ev.get("normalized_sentence")),
                "clause_text": _clean_text(ev.get("clause_text")),
                "relation_type": _clean_text(ev.get("relation_type")),
                "object_term": _clean_text(ev.get("object_term")),
                "expression_raw": _clean_text(ev.get("expression_raw")),
                "expression_lemma": _clean_text(ev.get("expression_lemma")),
                "predicted_dimension_code": dim,
                "package_level": _clean_text(ev.get("package_level")),
                "eligible_for_main_image_model": 1,
                "source_kind": _clean_text(ev.get("source_kind")),
                "source_type": _clean_text(ev.get("source_type")),
                "upstream_decision": "", "upstream_reason": "",
                "is_target_dimension_evidence": 1, "is_outer_eligible_evidence": 1,
                "focus_review_flag": focus_flag, "focus_review_reason": focus_reason,
            })

        # Non-outer evidence for positive tasks (e.g., ambiguous/inner for focus review)
        for _, ev in sorted_nonouter.iterrows():
            rank += 1
            context_rows.append({
                "annotation_item_id": aid, "parent_asin": pid,
                "target_dimension_code": dim, "model_label_value": 1,
                "label_interpretation": _clean_text(item.get("label_interpretation", "observed_positive")),
                "sampling_stratum": stratum, "context_rank": rank,
                "context_source": "formal_v21_nonouter_evidence",
                "context_status": "target_dimension_nonouter_evidence",
                "review_id": _clean_text(ev.get("review_id")),
                "sentence_id": _clean_text(ev.get("sentence_id")),
                "sentence": _clean_text(ev.get("sentence")),
                "normalized_sentence": _clean_text(ev.get("normalized_sentence")),
                "clause_text": _clean_text(ev.get("clause_text")),
                "relation_type": _clean_text(ev.get("relation_type")),
                "object_term": _clean_text(ev.get("object_term")),
                "expression_raw": _clean_text(ev.get("expression_raw")),
                "expression_lemma": _clean_text(ev.get("expression_lemma")),
                "predicted_dimension_code": dim,
                "package_level": _clean_text(ev.get("package_level")),
                "eligible_for_main_image_model": int(ev.get("eligible_for_main_image_model", 0) or 0),
                "source_kind": _clean_text(ev.get("source_kind")),
                "source_type": _clean_text(ev.get("source_type")),
                "upstream_decision": "", "upstream_reason": "",
                "is_target_dimension_evidence": 1, "is_outer_eligible_evidence": 0,
                "focus_review_flag": focus_flag, "focus_review_reason": focus_reason,
            })

        # Placeholder if no evidence at all was found (shouldn't happen due to check above)
        if rank == 0:
            raise ValueError(
                f"positive item {aid} has evidence rows in 39b but zero rows in 37 parquet"
            )

    # ---- unlabeled tasks ----
    for _, item in unlabeled_items.iterrows():
        pid = _clean_text(item["parent_asin"])
        dim = _clean_text(item["dimension_code"])
        aid = _clean_text(item["annotation_item_id"])
        stratum = _clean_text(item.get("sampling_stratum", ""))
        rank = 0

        focus_flag = 1 if aid in FOCUS_REVIEW_ITEMS else 0
        focus_reason = FOCUS_REVIEW_ITEMS[aid]["reason_en"] if focus_flag else ""

        # Priority 1: target dimension formal outer evidence (should be none for unlabeled)
        mask_target = (
            (relation_evidence["parent_asin"].astype(str).str.strip() == pid) &
            (relation_evidence["dimension_code"].astype(str).str.strip() == dim) &
            (relation_evidence["package_level"].astype(str).str.strip() == "outer_retail_package") &
            (relation_evidence["eligible_for_main_image_model"].astype(int) == 1)
        )
        target_outer = relation_evidence[mask_target]
        if len(target_outer) > 0:
            raise ValueError(
                f"unlabeled item {aid} has target-dimension outer eligible evidence; "
                f"sampling logic conflict"
            )
        for _, ev in target_outer.iterrows():
            rank += 1
            context_rows.append({
                "annotation_item_id": aid, "parent_asin": pid,
                "target_dimension_code": dim, "model_label_value": 0,
                "label_interpretation": _clean_text(item.get("label_interpretation", "unlabeled_not_observed")),
                "sampling_stratum": stratum, "context_rank": rank,
                "context_source": "formal_v21_target_outer_evidence",
                "context_status": "target_outer_evidence",
                "review_id": _clean_text(ev.get("review_id")),
                "sentence_id": _clean_text(ev.get("sentence_id")),
                "sentence": _clean_text(ev.get("sentence")),
                "normalized_sentence": _clean_text(ev.get("normalized_sentence")),
                "clause_text": _clean_text(ev.get("clause_text")),
                "relation_type": _clean_text(ev.get("relation_type")),
                "object_term": _clean_text(ev.get("object_term")),
                "expression_raw": _clean_text(ev.get("expression_raw")),
                "expression_lemma": _clean_text(ev.get("expression_lemma")),
                "predicted_dimension_code": _clean_text(ev.get("dimension_code")),
                "package_level": _clean_text(ev.get("package_level")),
                "eligible_for_main_image_model": int(ev.get("eligible_for_main_image_model", 0) or 0),
                "source_kind": _clean_text(ev.get("source_kind")),
                "source_type": _clean_text(ev.get("source_type")),
                "upstream_decision": "", "upstream_reason": "",
                "is_target_dimension_evidence": 1, "is_outer_eligible_evidence": 1,
                "focus_review_flag": focus_flag, "focus_review_reason": focus_reason,
            })

        # Priority 2: other-dimension formal outer evidence
        mask_other = (
            (relation_evidence["parent_asin"].astype(str).str.strip() == pid) &
            (relation_evidence["package_level"].astype(str).str.strip() == "outer_retail_package") &
            (relation_evidence["eligible_for_main_image_model"].astype(int) == 1) &
            ~(relation_evidence["dimension_code"].astype(str).str.strip() == dim)
        )
        other_outer = relation_evidence[mask_other].sort_values(["review_id", "sentence_id", "dimension_code"])
        for _, ev in other_outer.iterrows():
            rank += 1
            context_rows.append({
                "annotation_item_id": aid, "parent_asin": pid,
                "target_dimension_code": dim, "model_label_value": 0,
                "label_interpretation": _clean_text(item.get("label_interpretation", "unlabeled_not_observed")),
                "sampling_stratum": stratum, "context_rank": rank,
                "context_source": "formal_v21_other_outer_evidence",
                "context_status": "other_dimension_formal_evidence",
                "review_id": _clean_text(ev.get("review_id")),
                "sentence_id": _clean_text(ev.get("sentence_id")),
                "sentence": _clean_text(ev.get("sentence")),
                "normalized_sentence": _clean_text(ev.get("normalized_sentence")),
                "clause_text": _clean_text(ev.get("clause_text")),
                "relation_type": _clean_text(ev.get("relation_type")),
                "object_term": _clean_text(ev.get("object_term")),
                "expression_raw": _clean_text(ev.get("expression_raw")),
                "expression_lemma": _clean_text(ev.get("expression_lemma")),
                "predicted_dimension_code": _clean_text(ev.get("dimension_code")),
                "package_level": _clean_text(ev.get("package_level")),
                "eligible_for_main_image_model": int(ev.get("eligible_for_main_image_model", 0) or 0),
                "source_kind": _clean_text(ev.get("source_kind")),
                "source_type": _clean_text(ev.get("source_type")),
                "upstream_decision": "", "upstream_reason": "",
                "is_target_dimension_evidence": 0, "is_outer_eligible_evidence": 1,
                "focus_review_flag": focus_flag, "focus_review_reason": focus_reason,
            })

        # Priority 3: other formal relation evidence (non-outer)
        mask_other_any = (
            (relation_evidence["parent_asin"].astype(str).str.strip() == pid) &
            ~(relation_evidence["package_level"].astype(str).str.strip() == "outer_retail_package")
        )
        other_any = relation_evidence[mask_other_any].sort_values(["review_id", "sentence_id", "dimension_code"])
        for _, ev in other_any.iterrows():
            rank += 1
            context_rows.append({
                "annotation_item_id": aid, "parent_asin": pid,
                "target_dimension_code": dim, "model_label_value": 0,
                "label_interpretation": _clean_text(item.get("label_interpretation", "unlabeled_not_observed")),
                "sampling_stratum": stratum, "context_rank": rank,
                "context_source": "formal_v21_nonouter_evidence",
                "context_status": "other_relation_evidence",
                "review_id": _clean_text(ev.get("review_id")),
                "sentence_id": _clean_text(ev.get("sentence_id")),
                "sentence": _clean_text(ev.get("sentence")),
                "normalized_sentence": _clean_text(ev.get("normalized_sentence")),
                "clause_text": _clean_text(ev.get("clause_text")),
                "relation_type": _clean_text(ev.get("relation_type")),
                "object_term": _clean_text(ev.get("object_term")),
                "expression_raw": _clean_text(ev.get("expression_raw")),
                "expression_lemma": _clean_text(ev.get("expression_lemma")),
                "predicted_dimension_code": _clean_text(ev.get("dimension_code")),
                "package_level": _clean_text(ev.get("package_level")),
                "eligible_for_main_image_model": int(ev.get("eligible_for_main_image_model", 0) or 0),
                "source_kind": _clean_text(ev.get("source_kind")),
                "source_type": _clean_text(ev.get("source_type")),
                "upstream_decision": "", "upstream_reason": "",
                "is_target_dimension_evidence": 0, "is_outer_eligible_evidence": 0,
                "focus_review_flag": focus_flag, "focus_review_reason": focus_reason,
            })

        # Priority 4-6: upstream classified candidates
        if upstream_classified is not None:
            uc_cols = ["parent_asin", "review_id", "sentence_id", "sentence",
                        "normalized_sentence", "decision", "reason"]
            available_uc_cols = [c for c in uc_cols if c in upstream_classified.columns]
            uc_mask = upstream_classified["parent_asin"].astype(str).str.strip() == pid
            uc_for_pid = upstream_classified[uc_mask].copy()
            if not uc_for_pid.empty and "decision" in uc_for_pid.columns:
                # Sort: visual_strict first, then uncertain, then excluded
                decision_order = {"visual_strict": 0, "uncertain": 1, "excluded": 2}
                uc_for_pid["_dec_order"] = uc_for_pid["decision"].map(decision_order).fillna(9)
                uc_for_pid = uc_for_pid.sort_values(
                    ["_dec_order", "review_id", "sentence_id"]
                )
                for _, uc in uc_for_pid.iterrows():
                    rank += 1
                    context_rows.append({
                        "annotation_item_id": aid, "parent_asin": pid,
                        "target_dimension_code": dim, "model_label_value": 0,
                        "label_interpretation": _clean_text(item.get("label_interpretation", "unlabeled_not_observed")),
                        "sampling_stratum": stratum, "context_rank": rank,
                        "context_source": f"upstream_{_clean_text(uc.get('decision', 'unknown'))}",
                        "context_status": "packaging_candidate",
                        "review_id": _clean_text(uc.get("review_id")),
                        "sentence_id": _clean_text(uc.get("sentence_id")),
                        "sentence": _clean_text(uc.get("sentence")),
                        "normalized_sentence": _clean_text(uc.get("normalized_sentence")),
                        "clause_text": "",
                        "relation_type": "",
                        "object_term": "",
                        "expression_raw": "",
                        "expression_lemma": "",
                        "predicted_dimension_code": "",
                        "package_level": "",
                        "eligible_for_main_image_model": 0,
                        "source_kind": "",
                        "source_type": "",
                        "upstream_decision": _clean_text(uc.get("decision")),
                        "upstream_reason": _clean_text(uc.get("reason")),
                        "is_target_dimension_evidence": 0,
                        "is_outer_eligible_evidence": 0,
                        "focus_review_flag": focus_flag,
                        "focus_review_reason": focus_reason,
                    })

        # If no context at all, add a placeholder row
        if rank == 0:
            context_rows.append({
                "annotation_item_id": aid, "parent_asin": pid,
                "target_dimension_code": dim, "model_label_value": 0,
                "label_interpretation": _clean_text(item.get("label_interpretation", "unlabeled_not_observed")),
                "sampling_stratum": stratum, "context_rank": 1,
                "context_source": "none", "context_status": "no_candidate_context",
                "review_id": "", "sentence_id": "", "sentence": "",
                "normalized_sentence": "", "clause_text": "", "relation_type": "",
                "object_term": "", "expression_raw": "", "expression_lemma": "",
                "predicted_dimension_code": "", "package_level": "",
                "eligible_for_main_image_model": 0, "source_kind": "", "source_type": "",
                "upstream_decision": "", "upstream_reason": "",
                "is_target_dimension_evidence": 0, "is_outer_eligible_evidence": 0,
                "focus_review_flag": focus_flag, "focus_review_reason": focus_reason,
            })

    context = pd.DataFrame(context_rows, columns=EVIDENCE_CONTEXT_COLUMNS)
    return context


def build_product_dimension_reviewer_context(
    full_context: pd.DataFrame,
    product_dimension_items: pd.DataFrame,
    *,
    selection_seed: int = DEFAULT_SELECTION_SEED,
    unlabeled_cap: int = UNLABELED_CONTEXT_CAP,
) -> tuple[pd.DataFrame, dict]:
    """Build 08_product_dimension_reviewer_context.csv from 07 full context.

    Returns (reviewer_context_df, tier_stats_dict).
    """
    _require_columns(full_context, ["annotation_item_id", "model_label_value",
                                     "is_target_dimension_evidence", "is_outer_eligible_evidence",
                                     "context_source", "upstream_decision",
                                     "predicted_dimension_code", "target_dimension_code",
                                     "focus_review_flag", "review_id", "sentence_id"],
                      "full context")

    TIERS = [
        "mandatory_focus_or_direct_target",
        "formal_other_outer",
        "upstream_visual_strict",
        "upstream_uncertain",
        "upstream_excluded",
        "other_candidate",
    ]

    TIER_QUOTAS = {
        "mandatory_focus_or_direct_target": 10,
        "formal_other_outer": 5,
        "upstream_visual_strict": 5,
        "upstream_uncertain": 5,
        "upstream_excluded": 5,
    }

    def _stable_hash_sort_key(row: pd.Series) -> str:
        parts = [
            _clean_text(row.get("annotation_item_id")),
            _clean_text(row.get("review_id")),
            _clean_text(row.get("sentence_id")),
            _clean_text(row.get("normalized_sentence")),
            _clean_text(row.get("clause_text")),
            _clean_text(row.get("predicted_dimension_code")),
            _clean_text(row.get("context_source")),
            str(selection_seed),
        ]
        return hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()

    def _assign_tier(r: pd.Series) -> str:
        aid = _clean_text(r["annotation_item_id"])
        focus = int(r.get("focus_review_flag", 0) or 0)
        is_target = int(r.get("is_target_dimension_evidence", 0) or 0)
        pred_dim = _clean_text(r.get("predicted_dimension_code"))
        target_dim = _clean_text(r.get("target_dimension_code"))
        ctx_src = _clean_text(r.get("context_source"))
        upd_dec = _clean_text(r.get("upstream_decision"))

        if focus == 1 or is_target == 1 or pred_dim == target_dim:
            return "mandatory_focus_or_direct_target"
        if "formal_v21_other_outer" in ctx_src:
            return "formal_other_outer"
        if "upstream_visual_strict" in ctx_src or upd_dec == "visual_strict":
            return "upstream_visual_strict"
        if "upstream_uncertain" in ctx_src or upd_dec == "uncertain":
            return "upstream_uncertain"
        if "upstream_excluded" in ctx_src or upd_dec == "excluded":
            return "upstream_excluded"
        return "other_candidate"

    all_rows = []
    tier_totals: dict[str, dict] = {
        t: {"candidate_count": 0, "initial_quota_selected_count": 0, "backfill_selected_count": 0,
            "final_selected_count": 0} for t in TIERS
    }
    duplicates_removed = 0
    truncated_unlabeled_count = 0
    exhaustive_unlabeled_count = 0
    positive_per_task_counts: list[int] = []
    unlabeled_per_task_counts: list[int] = []
    positive_outer_count = 0
    positive_no_outer_count = 0
    positive_no_outer_ids: list[str] = []
    total_backfill = 0

    for aid in sorted(full_context["annotation_item_id"].unique()):
        ctx = full_context[full_context["annotation_item_id"] == aid].copy()
        if ctx.empty:
            continue

        model_label = int(ctx.iloc[0]["model_label_value"])
        full_count = len(ctx)

        # Deduplicate on stable hash
        ctx["_dedup_hash"] = ctx.apply(_stable_hash_sort_key, axis=1)
        before_dedup = len(ctx)
        ctx = ctx.drop_duplicates(subset=["_dedup_hash"])
        duplicates_removed += (before_dedup - len(ctx))

        # Assign tiers
        ctx["_tier"] = ctx.apply(_assign_tier, axis=1)
        tier_order = {t: i for i, t in enumerate(TIERS)}
        ctx["_tier_order"] = ctx["_tier"].map(tier_order)

        # Stable sort within tiers (hash-based, then review_id, sentence_id)
        ctx["_sort_hash"] = ctx.apply(_stable_hash_sort_key, axis=1)
        ctx = ctx.sort_values(["_tier_order", "_sort_hash", "review_id", "sentence_id"])

        if model_label == 1:
            # Positive: include ALL target evidence (outer first, then non-outer)
            target_outer = ctx[
                (ctx["is_target_dimension_evidence"] == 1) &
                (ctx["is_outer_eligible_evidence"] == 1)
            ]
            target_nonouter = ctx[
                (ctx["is_target_dimension_evidence"] == 1) &
                (ctx["is_outer_eligible_evidence"] == 0)
            ]
            selected = pd.concat([target_outer, target_nonouter], ignore_index=True)
            if len(selected) == 0:
                raise ValueError(f"positive item {aid} has no target dimension evidence at all")
            if len(target_outer) > 0:
                positive_outer_count += 1
            else:
                positive_no_outer_count += 1
                positive_no_outer_ids.append(aid)
            shortlist_is_exhaustive = 1
            requires_full = 0
            positive_per_task_counts.append(len(selected))
        else:
            # Unlabeled: tiered selection
            selected_rows = []
            remaining = unlabeled_cap
            tier_selected: dict[str, int] = {}
            tier_candidates: dict[str, int] = {}
            for tier in TIERS:
                tier_rows = ctx[ctx["_tier"] == tier]
                tier_candidates[tier] = len(tier_rows)
                quota = TIER_QUOTAS.get(tier, remaining)
                take = min(quota, remaining, len(tier_rows))
                if take > 0:
                    selected_rows.append(tier_rows.head(take))
                    remaining -= take
                tier_selected[tier] = take

            # Backfill if quotas not fully used
            backfill_count = 0
            if remaining > 0:
                already = set()
                for sr in selected_rows:
                    for idx in sr.index:
                        already.add(idx)
                remaining_rows = ctx.drop(index=already)
                backfill = remaining_rows.head(remaining)
                if len(backfill) > 0:
                    selected_rows.append(backfill)
                    backfill_count = len(backfill)
                    total_backfill += backfill_count
                    remaining -= backfill_count

            selected = pd.concat(selected_rows, ignore_index=True) if selected_rows else pd.DataFrame()
            shortlist_is_exhaustive = 1 if len(ctx) <= unlabeled_cap else 0
            requires_full = 1 if not shortlist_is_exhaustive else 0

            if not shortlist_is_exhaustive:
                truncated_unlabeled_count += 1
            else:
                exhaustive_unlabeled_count += 1

            for tier in TIERS:
                tier_totals[tier]["candidate_count"] += tier_candidates.get(tier, 0)
                tier_totals[tier]["initial_quota_selected_count"] += tier_selected.get(tier, 0)
            tier_totals["other_candidate"]["backfill_selected_count"] += backfill_count
            unlabeled_per_task_counts.append(len(selected))

        # Build output rows
        for rank, (_, row) in enumerate(selected.iterrows(), start=1):
            all_rows.append({
                "annotation_item_id": aid,
                "parent_asin": _clean_text(row.get("parent_asin")),
                "target_dimension_code": _clean_text(row.get("target_dimension_code")),
                "model_label_value": model_label,
                "label_interpretation": _clean_text(row.get("label_interpretation")),
                "sampling_stratum": _clean_text(row.get("sampling_stratum")),
                "review_context_rank": rank,
                "review_priority_tier": _clean_text(row["_tier"]),
                "selection_reason": _clean_text(row["_tier"]),
                "full_context_row_count": full_count,
                "reviewer_context_row_count": len(selected),
                "shortlist_is_exhaustive": shortlist_is_exhaustive,
                "selection_seed": selection_seed,
                "requires_full_context_on_uncertainty": requires_full,
                "context_rank": int(row.get("context_rank", 0) or 0),
                "context_source": _clean_text(row.get("context_source")),
                "context_status": _clean_text(row.get("context_status")),
                "review_id": _clean_text(row.get("review_id")),
                "sentence_id": _clean_text(row.get("sentence_id")),
                "sentence": _clean_text(row.get("sentence")),
                "normalized_sentence": _clean_text(row.get("normalized_sentence")),
                "clause_text": _clean_text(row.get("clause_text")),
                "relation_type": _clean_text(row.get("relation_type")),
                "object_term": _clean_text(row.get("object_term")),
                "expression_raw": _clean_text(row.get("expression_raw")),
                "expression_lemma": _clean_text(row.get("expression_lemma")),
                "predicted_dimension_code": _clean_text(row.get("predicted_dimension_code")),
                "package_level": _clean_text(row.get("package_level")),
                "eligible_for_main_image_model": int(row.get("eligible_for_main_image_model", 0) or 0),
                "source_kind": _clean_text(row.get("source_kind")),
                "source_type": _clean_text(row.get("source_type")),
                "upstream_decision": _clean_text(row.get("upstream_decision")),
                "upstream_reason": _clean_text(row.get("upstream_reason")),
                "is_target_dimension_evidence": int(row.get("is_target_dimension_evidence", 0) or 0),
                "is_outer_eligible_evidence": int(row.get("is_outer_eligible_evidence", 0) or 0),
                "focus_review_flag": int(row.get("focus_review_flag", 0) or 0),
                "focus_review_reason": _clean_text(row.get("focus_review_reason")),
            })

    rv = pd.DataFrame(all_rows)

    # Sort output stably
    rv = rv.sort_values(["annotation_item_id", "review_context_rank"])

    # Compute final_selected_count per tier
    for t in TIERS:
        tier_totals[t]["final_selected_count"] = (
            tier_totals[t]["initial_quota_selected_count"] +
            tier_totals[t]["backfill_selected_count"]
        )

    if positive_per_task_counts or unlabeled_per_task_counts:
        positive_per_task_counts.sort()
        unlabeled_per_task_counts.sort()
        all_task_counts = positive_per_task_counts + unlabeled_per_task_counts
        all_task_counts.sort()
        stats = {
            "row_count": len(rv),
            "annotation_item_count": int(rv["annotation_item_id"].nunique()),
            "positive_rows": int(len(rv[rv["model_label_value"] == 1])),
            "unlabeled_rows": int(len(rv[rv["model_label_value"] == 0])),
            "duplicates_removed": duplicates_removed,
            "truncated_unlabeled_count": truncated_unlabeled_count,
            "exhaustive_unlabeled_count": exhaustive_unlabeled_count,
            "all_tasks": {
                "count": len(all_task_counts),
                "min": all_task_counts[0] if all_task_counts else 0,
                "median": all_task_counts[len(all_task_counts) // 2] if all_task_counts else 0,
                "p90": all_task_counts[int(len(all_task_counts) * 0.9)] if all_task_counts else 0,
                "max": all_task_counts[-1] if all_task_counts else 0,
            },
            "positive_tasks": {
                "count": len(positive_per_task_counts),
                "min": positive_per_task_counts[0] if positive_per_task_counts else 0,
                "median": positive_per_task_counts[len(positive_per_task_counts) // 2] if positive_per_task_counts else 0,
                "p90": positive_per_task_counts[int(len(positive_per_task_counts) * 0.9)] if positive_per_task_counts else 0,
                "max": positive_per_task_counts[-1] if positive_per_task_counts else 0,
            },
            "unlabeled_tasks": {
                "count": len(unlabeled_per_task_counts),
                "min": unlabeled_per_task_counts[0] if unlabeled_per_task_counts else 0,
                "median": unlabeled_per_task_counts[len(unlabeled_per_task_counts) // 2] if unlabeled_per_task_counts else 0,
                "p90": unlabeled_per_task_counts[int(len(unlabeled_per_task_counts) * 0.9)] if unlabeled_per_task_counts else 0,
                "max": unlabeled_per_task_counts[-1] if unlabeled_per_task_counts else 0,
            },
            "positive_items_with_outer_evidence": positive_outer_count,
            "positive_items_without_outer_evidence": positive_no_outer_count,
            "positive_no_outer_annotation_item_ids": sorted(positive_no_outer_ids),
            "total_backfill_rows": total_backfill,
            "tier_definitions": {t: TIER_QUOTAS.get(t, "unlimited") for t in TIERS},
            "tier_stats": tier_totals,
            "selection_seed": selection_seed,
            "unlabeled_cap": unlabeled_cap,
        }
    else:
        stats = {"row_count": 0}

    return rv, stats


def _ensure_output_safe(output_dir: Path, overwrite: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    existing = [output_dir / filename for filename in PREPARE_OUTPUTS if (output_dir / filename).exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "validation outputs already exist; pass --overwrite to replace: "
            + ", ".join(path.name for path in existing)
        )
    if overwrite:
        for path in existing:
            path.unlink()


def prepare_validation_workspace(
    *,
    input_dir: Path,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    seed: int = 42,
    overwrite: bool = False,
    source_manifest: Path | None = None,
    upstream_classified: Path | None = None,
    v21_evidence: Path | None = None,
) -> dict[str, Any]:
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    paths = {
        AUDIT_SAMPLE_FILENAME: input_dir / AUDIT_SAMPLE_FILENAME,
        DIMENSIONS_FILENAME: input_dir / DIMENSIONS_FILENAME,
        PRODUCT_LABELS_FILENAME: input_dir / PRODUCT_LABELS_FILENAME,
        PRODUCT_DIMENSION_FILENAME: input_dir / PRODUCT_DIMENSION_FILENAME,
        SUMMARY_FILENAME: input_dir / SUMMARY_FILENAME,
    }
    for filename, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"missing required input {filename}: {path}")

    if source_manifest is None:
        source_manifest_path = paths[SUMMARY_FILENAME].resolve()
        source_manifest_mode = "auto-discovered"
    else:
        source_manifest_path = Path(source_manifest).resolve()
        source_manifest_mode = "explicit"
        if not source_manifest_path.is_file():
            raise FileNotFoundError(
                f"missing source manifest: {source_manifest_path}"
            )
    source_manifest_data = _load_manifest(source_manifest_path)
    source_manifest_contract = _validate_source_audit_manifest(source_manifest_data)
    _ensure_output_safe(output_dir, overwrite)

    audit_sample = read_csv(paths[AUDIT_SAMPLE_FILENAME])
    validate_sentence_items_input(audit_sample, source_manifest=source_manifest_contract)
    dimensions = read_csv(paths[DIMENSIONS_FILENAME])
    product_labels = read_csv(paths[PRODUCT_LABELS_FILENAME])
    product_dimension_evidence = read_csv(paths[PRODUCT_DIMENSION_FILENAME])

    sentence_items = build_sentence_items(audit_sample)
    product_dimension_items = build_product_dimension_items(
        dimensions,
        product_labels,
        product_dimension_evidence,
        seed=seed,
    )
    all_items = pd.concat(
        [sentence_items, product_dimension_items],
        ignore_index=True,
        sort=False,
    )
    annotations_a1 = build_annotation_template(all_items, "A1")
    annotations_a2 = build_annotation_template(all_items, "A2")
    adjudication = build_blank_adjudication_template(all_items)

    output_paths = {
        SENTENCE_ITEMS_FILENAME: output_dir / SENTENCE_ITEMS_FILENAME,
        PRODUCT_DIMENSION_ITEMS_FILENAME: output_dir / PRODUCT_DIMENSION_ITEMS_FILENAME,
        ANNOTATIONS_A1_FILENAME: output_dir / ANNOTATIONS_A1_FILENAME,
        ANNOTATIONS_A2_FILENAME: output_dir / ANNOTATIONS_A2_FILENAME,
        ADJUDICATION_TEMPLATE_FILENAME: output_dir / ADJUDICATION_TEMPLATE_FILENAME,
    }
    write_csv(sentence_items, output_paths[SENTENCE_ITEMS_FILENAME])
    write_csv(product_dimension_items, output_paths[PRODUCT_DIMENSION_ITEMS_FILENAME])
    write_csv(annotations_a1, output_paths[ANNOTATIONS_A1_FILENAME])
    write_csv(annotations_a2, output_paths[ANNOTATIONS_A2_FILENAME])
    write_csv(adjudication, output_paths[ADJUDICATION_TEMPLATE_FILENAME])

    # ---- 07 context ----
    evidence_37 = None
    upstream_15 = None
    if v21_evidence is not None:
        v21_evidence = Path(v21_evidence)
        if not v21_evidence.is_file():
            raise FileNotFoundError(f"missing V2.1 evidence: {v21_evidence}")
        evidence_37 = pd.read_parquet(v21_evidence)
    if upstream_classified is not None:
        upstream_classified = Path(upstream_classified)
        if not upstream_classified.is_file():
            raise FileNotFoundError(f"missing upstream classified: {upstream_classified}")
        upstream_15 = pd.read_parquet(upstream_classified)

    context_df = None
    context_path = output_dir / CONTEXT_FILENAME
    if evidence_37 is not None:
        context_df = build_product_dimension_evidence_context(
            product_dimension_items,
            evidence_37,
            upstream_classified=upstream_15,
        )
        write_csv(context_df, context_path)
        output_paths[CONTEXT_FILENAME] = context_path

    # ---- 08 reviewer context ----
    reviewer_context_df = None
    reviewer_context_stats: dict = {}
    reviewer_path = output_dir / REVIEWER_CONTEXT_FILENAME
    if context_df is not None:
        reviewer_context_df, reviewer_context_stats = build_product_dimension_reviewer_context(
            context_df,
            product_dimension_items,
            selection_seed=DEFAULT_SELECTION_SEED,
            unlabeled_cap=UNLABELED_CONTEXT_CAP,
        )
        write_csv(reviewer_context_df, reviewer_path)
        output_paths[REVIEWER_CONTEXT_FILENAME] = reviewer_path

    # ---- manifest ----
    manifest_output_files = list(PREPARE_OUTPUTS)
    manifest_output_hashes = {
        filename: sha256_file(path)
        for filename, path in output_paths.items()
    }

    # 07 stats
    context_row_count = int(len(context_df)) if context_df is not None else 0
    context_annotation_count = (
        int(context_df["annotation_item_id"].nunique()) if context_df is not None else 0
    )
    positive_context_count = (
        int(len(context_df[context_df["model_label_value"] == 1].drop_duplicates("annotation_item_id")))
        if context_df is not None else 0
    )
    unlabeled_context_count = (
        int(len(context_df[context_df["model_label_value"] == 0].drop_duplicates("annotation_item_id")))
        if context_df is not None else 0
    )
    positive_with_outer = (
        int(len(context_df[(context_df["model_label_value"] == 1) & (context_df["is_target_dimension_evidence"] == 1)]))
        if context_df is not None else 0
    )
    unlabeled_with_candidates = 0
    unlabeled_without_candidates = 0
    if context_df is not None:
        for aid in context_df[context_df["model_label_value"] == 0]["annotation_item_id"].unique():
            sub = context_df[context_df["annotation_item_id"] == aid]
            if len(sub) == 1 and sub.iloc[0]["context_status"] == "no_candidate_context":
                unlabeled_without_candidates += 1
            else:
                unlabeled_with_candidates += 1

    manifest = {
        "tool_version": TOOL_VERSION,
        "seed": seed,
        "input_dir": str(input_dir.resolve()),
        "output_dir": str(output_dir.resolve()),
        "input_hashes": {
            filename: sha256_file(path)
            for filename, path in paths.items()
        },
        "output_hashes": manifest_output_hashes,
        "sentence_group_targets": dict(GROUP_TARGETS),
        "source_manifest_path": str(source_manifest_path),
        "source_manifest_sha256": sha256_file(source_manifest_path),
        "source_manifest_mode": source_manifest_mode,
        "audit_seed": int(source_manifest_contract["audit_seed"]),
        "audit_requested_sample_size": int(
            source_manifest_contract["audit_requested_sample_size"]
        ),
        "audit_actual_sample_size": int(
            source_manifest_contract["audit_actual_sample_size"]
        ),
        "sentence_group_capacities": dict(
            source_manifest_contract["sentence_group_capacities"]
        ),
        "sentence_group_requested_quotas": dict(
            source_manifest_contract["sentence_group_requested_quotas"]
        ),
        "sentence_group_final_quotas": dict(
            source_manifest_contract["sentence_group_final_quotas"]
        ),
        "sentence_group_actual_counts": dict(
            source_manifest_contract["sentence_group_actual_counts"]
        ),
        "audit_redistribution": source_manifest_contract["audit_redistribution"],
        "sentence_group_counts": {
            group: int((sentence_items["audit_group"] == group).sum())
            for group in GROUP_TARGETS
        },
        "sentence_item_count": int(len(sentence_items)),
        "product_dimension_item_count": int(len(product_dimension_items)),
        "annotation_item_count": int(len(all_items)),
        "annotation_key": [
            "annotation_item_id",
            "annotator_id",
            "annotation_round",
        ],
        "label_semantics": _read_summary_label_semantics(paths[SUMMARY_FILENAME]),
        "zero_label_interpretation": "unlabeled_not_observed",
        "generated_files": manifest_output_files,
        "context_scope": (
            "07_product_dimension_evidence_context.csv provides evidence context "
            "for product-dimension tasks. Positive tasks include all matching outer "
            "eligible evidence from 37 parquet. Unlabeled tasks include upstream "
            "packaging candidates (visual_strict, uncertain, excluded) plus other-"
            "dimension formal evidence. Context does not cover raw review text."
        ),
        "product_dimension_context_row_count": context_row_count,
        "context_annotation_item_count": context_annotation_count,
        "positive_context_item_count": positive_context_count,
        "unlabeled_context_item_count": unlabeled_context_count,
        "positive_items_with_outer_evidence": positive_context_count,
        "unlabeled_items_with_candidate_context": unlabeled_with_candidates,
        "unlabeled_items_without_candidate_context": unlabeled_without_candidates,
        "no_silent_truncation": True,
        "focus_review_parent_asins": ["B0BWLWY25M", "B0C5ZMZBKS"],
        "focus_review_annotation_item_ids": list(FOCUS_REVIEW_ITEMS.keys()),
        "focus_review_reasons": {k: v["reason_en"] for k, v in FOCUS_REVIEW_ITEMS.items()},
        "human_gold_validation_pending": True,
        "annotations_started": False,
        "reviewer_packet": {
            "filename": REVIEWER_CONTEXT_FILENAME,
            "selection_seed": DEFAULT_SELECTION_SEED,
            "unlabeled_context_cap": UNLABELED_CONTEXT_CAP,
            "reviewer_context_row_count": reviewer_context_stats.get("row_count", 0),
            "reviewer_annotation_item_count": reviewer_context_stats.get("annotation_item_count", 0),
            "positive_reviewer_rows": reviewer_context_stats.get("positive_rows", 0),
            "unlabeled_reviewer_rows": reviewer_context_stats.get("unlabeled_rows", 0),
            "duplicates_removed": reviewer_context_stats.get("duplicates_removed", 0),
            "truncated_unlabeled_count": reviewer_context_stats.get("truncated_unlabeled_count", 0),
            "exhaustive_unlabeled_count": reviewer_context_stats.get("exhaustive_unlabeled_count", 0),
            "all_tasks_distribution": reviewer_context_stats.get("all_tasks", {}),
            "positive_tasks_distribution": reviewer_context_stats.get("positive_tasks", {}),
            "unlabeled_tasks_distribution": reviewer_context_stats.get("unlabeled_tasks", {}),
            "positive_items_with_outer_evidence": reviewer_context_stats.get("positive_items_with_outer_evidence", 0),
            "positive_items_without_outer_evidence": reviewer_context_stats.get("positive_items_without_outer_evidence", 0),
            "positive_no_outer_annotation_item_ids": reviewer_context_stats.get("positive_no_outer_annotation_item_ids", []),
            "total_backfill_rows": reviewer_context_stats.get("total_backfill_rows", 0),
            "tier_definitions": reviewer_context_stats.get("tier_definitions", {}),
            "tier_stats": reviewer_context_stats.get("tier_stats", {}),
            "method_limitations": (
                "08 is a risk-enhanced, deterministically sampled reviewer packet. "
                "Unlabeled tasks are capped at 30 rows with tiered priority selection. "
                "It is NOT exhaustive review context. On uncertainty, disagreement, "
                "low confidence, or focus_review_flag=1, reviewers MUST consult 07 "
                "for complete context."
            ),
            "escalation_rules": [
                "low confidence -> consult 07 full context",
                "disagreement between A1/A2 -> consult 07 full context",
                "potential missed signal found -> consult 07 full context",
                "focus_review_flag=1 -> consult 07 full context",
                "uncertain -> consult 07 full context",
            ],
        },
    }
    manifest_path = output_dir / MANIFEST_FILENAME
    _write_manifest(manifest_path, manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare v2.1 affective imagery human-validation templates."
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        type=Path,
        help="Directory containing v2.1 output files 38, 39, 39b, 40, and 41.",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        type=Path,
        help="Directory for local manual-validation outputs.",
    )
    parser.add_argument("--seed", default=42, type=int, help="Deterministic sampling seed.")
    parser.add_argument(
        "--source-manifest",
        type=Path,
        help="Optional A-line manifest with audit group capacities.",
    )
    parser.add_argument(
        "--upstream-classified",
        type=Path,
        help="Path to 15_packaging_sentences_rule_classified.parquet for unlabeled context.",
    )
    parser.add_argument(
        "--v21-evidence",
        type=Path,
        help="Path to 37_relation_constrained_sentence_evidence_v21.parquet for evidence context.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing generated validation files in output-dir.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = prepare_validation_workspace(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        seed=args.seed,
        overwrite=args.overwrite,
        source_manifest=args.source_manifest,
        upstream_classified=args.upstream_classified,
        v21_evidence=args.v21_evidence,
    )
    print(
        "Prepared validation workspace: "
        f"{manifest['sentence_item_count']} sentence items, "
        f"{manifest['product_dimension_item_count']} product-dimension items"
    )
    if "product_dimension_context_row_count" in manifest:
        print(f"  Context rows written: {manifest['product_dimension_context_row_count']}")
    print(f"Output directory: {Path(args.output_dir).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
