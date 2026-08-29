#!/usr/bin/env python3
"""Validate A1/A2 v2.1 affective imagery annotation files."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

try:
    from .affective_imagery_final_adjudication_v21 import (
        AdjudicatedMappingConsistencyReport,
        validate_adjudicated_additional_dimensions,
        validate_adjudicated_mapping_consistency as _validate_final_mapping,
        validate_adjudicated_pair_closure,
    )
except ImportError:  # pragma: no cover - direct script execution
    from affective_imagery_final_adjudication_v21 import (
        AdjudicatedMappingConsistencyReport,
        validate_adjudicated_additional_dimensions,
        validate_adjudicated_mapping_consistency as _validate_final_mapping,
        validate_adjudicated_pair_closure,
    )

try:
    from .affective_imagery_action_error_mapping_v21 import (
        MappingContract,
        adapt_row_inputs,
        evaluate_mapping,
        load_mapping,
    )
except ImportError:  # pragma: no cover - direct script execution
    from affective_imagery_action_error_mapping_v21 import (
        MappingContract,
        adapt_row_inputs,
        evaluate_mapping,
        load_mapping,
    )

try:
    from .affective_imagery_annotation_policy_v21 import rationale_required
    from .affective_imagery_decision_sidecar_v21 import (
        decision_sidecar_provenance_path,
        load_canonical_mapping,
        verify_derived_sidecar_with_provenance,
    )
except ImportError:  # pragma: no cover - direct script execution
    from affective_imagery_annotation_policy_v21 import rationale_required
    from affective_imagery_decision_sidecar_v21 import (
        decision_sidecar_provenance_path,
        load_canonical_mapping,
        verify_derived_sidecar_with_provenance,
    )

try:
    from .affective_imagery_validation_provenance_v21 import (
        build_provenance,
        load_provenance,
        read_git_state,
        write_json_new_atomic,
    )
except ImportError:  # pragma: no cover - direct script execution
    from affective_imagery_validation_provenance_v21 import (
        build_provenance,
        load_provenance,
        read_git_state,
        write_json_new_atomic,
    )

try:
    from .prepare_affective_imagery_validation_v21 import (
        ADJUDICATION_COLUMNS,
        ADJUDICATION_TEMPLATE_FILENAME,
        ANNOTATION_COLUMNS,
        ANNOTATIONS_A1_FILENAME,
        ANNOTATIONS_A2_FILENAME,
        CONTEXT_FILENAME,
        DEFAULT_OUTPUT_DIR,
        DIMENSION_CODES,
        MANIFEST_FILENAME,
        PRODUCT_DIMENSION_ITEMS_FILENAME,
        REVIEWER_CONTEXT_FILENAME,
        SENTENCE_ITEMS_FILENAME,
        _clean_text,
        _is_blank,
        read_csv,
        sha256_file,
        write_csv,
    )
except ImportError:  # pragma: no cover - direct script execution
    from prepare_affective_imagery_validation_v21 import (
        ADJUDICATION_COLUMNS,
        ADJUDICATION_TEMPLATE_FILENAME,
        ANNOTATION_COLUMNS,
        ANNOTATIONS_A1_FILENAME,
        ANNOTATIONS_A2_FILENAME,
        CONTEXT_FILENAME,
        DEFAULT_OUTPUT_DIR,
        DIMENSION_CODES,
        MANIFEST_FILENAME,
        PRODUCT_DIMENSION_ITEMS_FILENAME,
        REVIEWER_CONTEXT_FILENAME,
        SENTENCE_ITEMS_FILENAME,
        _clean_text,
        _is_blank,
        read_csv,
        sha256_file,
        write_csv,
    )


TOOL_NAME = "validate_affective_imagery_annotations_v21"
TOOL_VERSION = "affective_imagery_annotation_validator_v21.2"
PRODUCTION_MAPPING_PATH = (
    Path(__file__).parents[2]
    / "config"
    / "affective_imagery"
    / "action_error_mapping_v21.json"
)

YES_NO_UNCERTAIN = {"yes", "no", "uncertain"}
PACKAGE_LEVELS = {"outer", "inner", "ambiguous", "non_packaging", "uncertain"}
POLARITIES = {"positive", "negative", "uncertain"}
HUMAN_ACTIONS = {
    "keep",
    "drop",
    "change_dimension",
    "change_package_level",
    "change_polarity",
    "add_missing",
}
# change_polarity closes ONLY with negation_error.  Other combinations are
# rejected by validate_annotation_frame's action/error closure check.
CHANGE_POLARITY_CLOSED_ERROR = {"negation_error"}
ERROR_TYPES = {
    "none",
    "nonvisual_content",
    "shipping_or_seller",
    "inner_packaging",
    "ambiguous_package_level",
    "wrong_dimension",
    "negation_error",
    "relation_missing",
    "missed_relation",
    "duplicate_or_near_duplicate",
    "context_missing",
    "other",
}
CONFIDENCE_LEVELS = {"high", "medium", "low"}
DIMENSION_VALUES = set(DIMENSION_CODES) | {"none", "uncertain"}

SENTENCE_REQUIRED = {
    "human_packaging_visual": YES_NO_UNCERTAIN,
    "human_relation_valid": YES_NO_UNCERTAIN,
    "human_package_level": PACKAGE_LEVELS,
    "human_dimension_code": DIMENSION_VALUES,
    "human_polarity": POLARITIES,
    "human_action": HUMAN_ACTIONS,
    "human_error_type": ERROR_TYPES,
    "human_confidence": CONFIDENCE_LEVELS,
}

PRODUCT_COMMON_REQUIRED = {
    "human_confidence": CONFIDENCE_LEVELS,
}

VALIDATION_WORKSPACE_FILENAMES = [
    SENTENCE_ITEMS_FILENAME,
    PRODUCT_DIMENSION_ITEMS_FILENAME,
    ANNOTATIONS_A1_FILENAME,
    ANNOTATIONS_A2_FILENAME,
    ADJUDICATION_TEMPLATE_FILENAME,
    MANIFEST_FILENAME,
    CONTEXT_FILENAME,
    REVIEWER_CONTEXT_FILENAME,
]


def load_items(items_dir: Path) -> pd.DataFrame:
    sentence_items = read_csv(items_dir / SENTENCE_ITEMS_FILENAME)
    product_items = read_csv(items_dir / PRODUCT_DIMENSION_ITEMS_FILENAME)
    return pd.concat([sentence_items, product_items], ignore_index=True, sort=False)


def _require_columns(frame: pd.DataFrame, columns: list[str], context: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{context} missing required columns: {', '.join(missing)}")


def _value_allowed(value: Any, allowed: set[str]) -> bool:
    return _clean_text(value) in allowed


# rationale_required is imported from the shared decision-sidecar module
# (single canonical implementation used by validator and normalizer).
_rationale_required = rationale_required


def validate_annotation_frame(
    annotations: pd.DataFrame,
    items: pd.DataFrame,
    *,
    expected_annotator_id: str | None = None,
) -> None:
    required_base = [
        "annotation_item_id",
        "annotator_id",
        "annotation_round",
        "human_confidence",
        "human_rationale_cn",
    ]
    _require_columns(annotations, required_base, "annotation file")
    _require_columns(items, ["annotation_item_id", "item_type"], "items")

    key_columns = ["annotation_item_id", "annotator_id", "annotation_round"]
    if annotations[key_columns].duplicated().any():
        raise ValueError("annotation unique key is duplicated")

    if expected_annotator_id is not None:
        actual = set(annotations["annotator_id"].map(_clean_text))
        if actual != {expected_annotator_id}:
            raise ValueError(f"expected annotator_id {expected_annotator_id}, got {sorted(actual)}")

    item_lookup = items.set_index("annotation_item_id", drop=False)
    item_ids = set(item_lookup.index.astype(str))
    annotation_ids = set(annotations["annotation_item_id"].astype(str))
    if annotation_ids != item_ids:
        missing = sorted(item_ids - annotation_ids)
        extra = sorted(annotation_ids - item_ids)
        raise ValueError(f"annotation ids must match items; missing={missing[:5]} extra={extra[:5]}")

    errors: list[str] = []
    for index, annotation in annotations.iterrows():
        item = item_lookup.loc[_clean_text(annotation["annotation_item_id"])]
        merged = annotation.copy()
        for column in item.index:
            if column not in merged.index:
                merged[column] = item[column]
        item_type = _clean_text(item["item_type"])

        if item_type == "sentence":
            for column, allowed in SENTENCE_REQUIRED.items():
                if column not in annotations.columns or _is_blank(annotation.get(column)):
                    errors.append(f"row {index + 1}: {column} is required")
                    continue
                if not _value_allowed(annotation.get(column), allowed):
                    errors.append(f"row {index + 1}: invalid {column}")
            additional = _clean_text(annotation.get("human_additional_dimension_codes"))
            if additional:
                invalid = [
                    value
                    for value in additional.split("|")
                    if value and value not in DIMENSION_VALUES
                ]
                if invalid:
                    errors.append(f"row {index + 1}: invalid human_additional_dimension_codes")
            if _clean_text(annotation.get("human_action")) == "change_polarity":
                error_value = _clean_text(annotation.get("human_error_type"))
                if error_value not in CHANGE_POLARITY_CLOSED_ERROR:
                    errors.append(
                        f"row {index + 1}: change_polarity requires error_type "
                        f"negation_error (got {error_value or 'blank'})"
                    )
        elif item_type == "product_dimension":
            for column, allowed in PRODUCT_COMMON_REQUIRED.items():
                if column not in annotations.columns or _is_blank(annotation.get(column)):
                    errors.append(f"row {index + 1}: {column} is required")
                    continue
                if not _value_allowed(annotation.get(column), allowed):
                    errors.append(f"row {index + 1}: invalid {column}")
            model_label_value = int(float(item.get("model_label_value", 0) or 0))
            if model_label_value == 1:
                column = "human_product_label_traceable"
                if column not in annotations.columns or _is_blank(annotation.get(column)):
                    errors.append(f"row {index + 1}: {column} is required")
                elif not _value_allowed(annotation.get(column), YES_NO_UNCERTAIN):
                    errors.append(f"row {index + 1}: invalid {column}")
            else:
                column = "human_unlabeled_missed_signal"
                if column not in annotations.columns or _is_blank(annotation.get(column)):
                    errors.append(f"row {index + 1}: {column} is required")
                elif not _value_allowed(annotation.get(column), YES_NO_UNCERTAIN):
                    errors.append(f"row {index + 1}: invalid {column}")
        else:
            errors.append(f"row {index + 1}: invalid item_type")

        if rationale_required(merged, item_type) and _is_blank(annotation.get("human_rationale_cn")):
            errors.append(f"row {index + 1}: human_rationale_cn is required")

    if errors:
        raise ValueError("; ".join(errors[:10]))


def validate_adjudicated_closure(adjudication: pd.DataFrame) -> None:
    """Validate approved action/error closure for nonblank final rows."""
    _require_columns(
        adjudication,
        ["adjudicated_action", "adjudicated_error_type"],
        "adjudication file",
    )
    errors: list[str] = []
    for index, row in adjudication.iterrows():
        action = _clean_text(row.get("adjudicated_action"))
        error_value = _clean_text(row.get("adjudicated_error_type"))
        if not action and not error_value:
            continue
        try:
            validate_adjudicated_pair_closure(
                action,
                error_value,
                row.get("adjudication_note_cn"),
            )
        except ValueError as exc:
            errors.append(f"row {index + 1}: adjudicated {exc}")
    if errors:
        raise ValueError("; ".join(errors[:10]))


def validate_adjudication_frame_schema(
    adjudication: pd.DataFrame,
) -> None:
    """Validate canonical final sentence fields without mutating the frame."""
    _require_columns(
        adjudication,
        ["item_type"] + ADJUDICATION_COLUMNS,
        "adjudication file",
    )
    sentence_required = {
        "adjudicated_packaging_visual": YES_NO_UNCERTAIN,
        "adjudicated_relation_valid": YES_NO_UNCERTAIN,
        "adjudicated_package_level": PACKAGE_LEVELS,
        "adjudicated_dimension_code": DIMENSION_VALUES,
        "adjudicated_polarity": POLARITIES,
        "adjudicated_action": HUMAN_ACTIONS,
        "adjudicated_error_type": ERROR_TYPES,
    }
    errors: list[str] = []
    for index, row in adjudication.iterrows():
        item_type = _clean_text(row.get("item_type"))
        if item_type == "product_dimension":
            if (
                _clean_text(row.get("adjudicated_action"))
                or _clean_text(row.get("adjudicated_error_type"))
            ):
                errors.append(
                    f"row {index + 1}: product_dimension action/error "
                    "must be blank"
                )
            continue
        if item_type != "sentence":
            errors.append(
                f"row {index + 1}: invalid item_type "
                f"{item_type or 'blank'}"
            )
            continue
        for column, allowed in sentence_required.items():
            value = _clean_text(row.get(column))
            if not value:
                errors.append(f"row {index + 1}: {column} is required")
            elif value not in allowed:
                errors.append(f"row {index + 1}: invalid {column}")
        try:
            validate_adjudicated_additional_dimensions(
                row.get("adjudicated_additional_dimension_codes"),
                dimension_values=set(DIMENSION_CODES),
            )
        except ValueError as exc:
            errors.append(f"row {index + 1}: {exc}")
    if errors:
        raise ValueError("; ".join(errors[:10]))


def validate_final_adjudication(
    adjudication: pd.DataFrame,
    contract: MappingContract | None = None,
) -> AdjudicatedMappingConsistencyReport:
    """Validate final schema and frozen-mapping action/error consistency."""
    validate_adjudication_frame_schema(adjudication)
    mapping_contract = contract or load_canonical_mapping(
        PRODUCTION_MAPPING_PATH,
        actions=HUMAN_ACTIONS,
        error_types=ERROR_TYPES,
        dimension_values=DIMENSION_VALUES,
        polarities=POLARITIES,
    )
    return _validate_final_mapping(
        adjudication,
        mapping_contract,
        dimension_values=set(DIMENSION_CODES),
    )


def validate_annotation_mapping_consistency(
    annotations: pd.DataFrame,
    items: pd.DataFrame,
    contract: MappingContract,
) -> None:
    """Validate sentence action/error fields against an explicit mapping contract."""
    _require_columns(annotations, ["annotation_item_id"], "annotation file")
    _require_columns(items, ["annotation_item_id", "item_type"], "items")
    item_lookup = items.set_index("annotation_item_id", drop=False)
    errors: list[str] = []

    for index, annotation in annotations.iterrows():
        annotation_item_id = _clean_text(annotation["annotation_item_id"])
        if annotation_item_id not in item_lookup.index:
            errors.append(f"row {index + 1}: annotation_item_id not found in items")
            continue
        item = item_lookup.loc[annotation_item_id]
        if _clean_text(item["item_type"]) != "sentence":
            continue

        merged = annotation.copy()
        for column in item.index:
            if column not in merged.index:
                merged[column] = item[column]
        result = evaluate_mapping(contract, adapt_row_inputs(merged, contract))
        if result.status == "unresolved":
            errors.append(
                f"row {index + 1}: mapping unresolved ({result.reason_code})"
            )
            continue
        if _clean_text(annotation.get("human_action")) != result.action:
            errors.append(f"row {index + 1}: human_action mismatch")
        if _clean_text(annotation.get("human_error_type")) != result.error_type:
            errors.append(f"row {index + 1}: human_error_type mismatch")

    if errors:
        raise ValueError("; ".join(errors[:10]))


def _annotation_subset_for_prefix(frame: pd.DataFrame, prefix: str) -> pd.DataFrame:
    columns = ["annotation_item_id"] + [
        column
        for column in ANNOTATION_COLUMNS
        if column in frame.columns
    ]
    subset = frame[columns].copy()
    rename = {
        column: f"{prefix}_{column}"
        for column in subset.columns
        if column != "annotation_item_id"
    }
    return subset.rename(columns=rename)


def build_adjudication_template(
    items: pd.DataFrame,
    annotations_a1: pd.DataFrame,
    annotations_a2: pd.DataFrame,
    *,
    derived_a1: pd.DataFrame | None = None,
    derived_a2: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build the canonical adjudication template.

    Raw A1/A2 are validated canonically; raw action/error fields are NOT forced
    to match the mapping.  When derived sidecars are supplied, derived audit
    columns are merged by annotation_item_id into the audit area while
    adjudicated_* columns stay blank.
    """
    validate_annotation_frame(annotations_a1, items, expected_annotator_id="A1")
    validate_annotation_frame(annotations_a2, items, expected_annotator_id="A2")

    base = items.copy()
    merged = base.merge(
        _annotation_subset_for_prefix(annotations_a1, "A1"),
        on="annotation_item_id",
        how="left",
        validate="one_to_one",
    ).merge(
        _annotation_subset_for_prefix(annotations_a2, "A2"),
        on="annotation_item_id",
        how="left",
        validate="one_to_one",
    )
    for column in ADJUDICATION_COLUMNS:
        merged[column] = ""
    if derived_a1 is not None or derived_a2 is not None:
        merged = merge_derived_audit_columns(
            merged,
            derived_a1=derived_a1,
            derived_a2=derived_a2,
        )
    return merged


DERIVED_AUDIT_COLUMN_MAP = {
    "mapping_status": "mapping_status",
    "derived_human_action": "derived_human_action",
    "derived_human_error_type": "derived_human_error_type",
    "reason_code": "mapping_reason_code",
    "mapping_requires_adjudication": "mapping_requires_adjudication",
    "rationale_missing_after_derivation": "rationale_missing_after_derivation",
}


def merge_derived_audit_columns(
    adjudication: pd.DataFrame,
    *,
    derived_a1: pd.DataFrame | None,
    derived_a2: pd.DataFrame | None,
) -> pd.DataFrame:
    """Merge derived decision sidecar columns into the 05 audit area."""
    result = adjudication.copy()
    for prefix, derived in (("A1", derived_a1), ("A2", derived_a2)):
        if derived is None:
            continue
        _require_columns(derived, ["annotation_item_id"], "derived sidecar")
        if derived["annotation_item_id"].map(_clean_text).duplicated().any():
            raise ValueError(f"{prefix} derived sidecar has duplicate annotation_item_id")
        subset = derived[["annotation_item_id", *DERIVED_AUDIT_COLUMN_MAP]].copy()
        subset = subset.rename(columns={
            source: f"{prefix}_{target}"
            for source, target in DERIVED_AUDIT_COLUMN_MAP.items()
        })
        result = result.merge(
            subset,
            on="annotation_item_id",
            how="left",
            validate="one_to_one",
        )
    return result


def _verify_derived_inputs_before_05(
    *,
    items: pd.DataFrame,
    a1_path: Path,
    a2_path: Path,
    derived_a1: pd.DataFrame,
    derived_a2: pd.DataFrame,
    derived_a1_path: Path,
    derived_a2_path: Path,
    mapping_contract: MappingContract,
) -> dict[str, Any]:
    """Verify raw + derived + provenance before any 05 output is produced (F1).

    Fail closed on tampered sidecar, stale input SHA, wrong mapping SHA/version,
    wrong annotator, missing/extra/duplicate IDs, missing provenance, or
    sidecar/provenance mismatch.  Returns pair-level reconciliation counts.
    """
    if derived_a1["annotation_item_id"].map(_clean_text).duplicated().any():
        raise ValueError("A1 derived sidecar has duplicate annotation_item_id")
    if derived_a2["annotation_item_id"].map(_clean_text).duplicated().any():
        raise ValueError("A2 derived sidecar has duplicate annotation_item_id")

    for label, raw_path, derived, derived_path in (
        ("A1", a1_path, derived_a1, derived_a1_path),
        ("A2", a2_path, derived_a2, derived_a2_path),
    ):
        provenance_path = decision_sidecar_provenance_path(derived_path)
        verify_derived_sidecar_with_provenance(
            raw_path=raw_path,
            sidecar_path=derived_path,
            provenance_path=provenance_path,
            contract=mapping_contract,
            annotator_id=label,
            tool_name="normalize_affective_imagery_actions_v21",
            tool_version="affective_imagery_action_error_normalizer_v21.2",
        )

    # Mapping identity identical between A1 and A2 sidecars
    a1_version = _clean_text(derived_a1.iloc[0].get("mapping_version"))
    a2_version = _clean_text(derived_a2.iloc[0].get("mapping_version"))
    a1_sha = _clean_text(derived_a1.iloc[0].get("mapping_sha256"))
    a2_sha = _clean_text(derived_a2.iloc[0].get("mapping_sha256"))
    if a1_version != a2_version or a1_sha != a2_sha:
        raise ValueError(
            "A1/A2 derived sidecars must reference the identical mapping "
            f"(version {a1_version!r} vs {a2_version!r}, sha {a1_sha!r} vs {a2_sha!r})"
        )
    if a1_sha != mapping_contract.metadata.mapping_sha256:
        raise ValueError("A1 sidecar mapping SHA differs from current contract")
    if a2_sha != mapping_contract.metadata.mapping_sha256:
        raise ValueError("A2 sidecar mapping SHA differs from current contract")

    # Exact ID-set match with items (no intersection that silently drops rows)
    item_ids = set(items["annotation_item_id"].map(_clean_text))
    a1_ids = set(derived_a1["annotation_item_id"].map(_clean_text))
    a2_ids = set(derived_a2["annotation_item_id"].map(_clean_text))
    if a1_ids != item_ids or a2_ids != item_ids:
        raise ValueError(
            "derived sidecar ID sets must exactly match items; "
            f"a1_missing={sorted(item_ids - a1_ids)[:5]} "
            f"a2_missing={sorted(item_ids - a2_ids)[:5]} "
            f"a1_extra={sorted(a1_ids - item_ids)[:5]} "
            f"a2_extra={sorted(a2_ids - item_ids)[:5]}"
        )

    # Pair reconciliation counts (sentence-only applicability)
    return _reconcile_derived_pair_counts(derived_a1, derived_a2)


def _reconcile_derived_pair_counts(
    derived_a1: pd.DataFrame,
    derived_a2: pd.DataFrame,
) -> dict[str, Any]:
    """Sentence-applicable pair counts; product N/A rows are excluded (F5)."""
    a1_by_id = derived_a1.set_index(
        derived_a1["annotation_item_id"].map(_clean_text)
    )
    a2_by_id = derived_a2.set_index(
        derived_a2["annotation_item_id"].map(_clean_text)
    )
    a1_status = a1_by_id["mapping_status"].map(_clean_text).to_dict()
    a2_status = a2_by_id["mapping_status"].map(_clean_text).to_dict()
    ids = sorted(a1_status)

    sentence_ids = [
        item_id for item_id in ids
        if a1_status[item_id] != "not_applicable" and a2_status[item_id] != "not_applicable"
    ]
    not_applicable_ids = [
        item_id for item_id in ids
        if a1_status[item_id] == "not_applicable" or a2_status[item_id] == "not_applicable"
    ]
    a1_resolved = sum(1 for item_id in sentence_ids if a1_status[item_id] == "resolved")
    a1_unresolved = sum(1 for item_id in sentence_ids if a1_status[item_id] != "resolved")
    a2_resolved = sum(1 for item_id in sentence_ids if a2_status[item_id] == "resolved")
    a2_unresolved = sum(1 for item_id in sentence_ids if a2_status[item_id] != "resolved")
    both_resolved = [
        item_id for item_id in sentence_ids
        if a1_status[item_id] == "resolved" and a2_status[item_id] == "resolved"
    ]
    any_unresolved = [
        item_id for item_id in sentence_ids
        if a1_status[item_id] != "resolved" or a2_status[item_id] != "resolved"
    ]
    reason_counts: dict[str, int] = {}
    for item_id in any_unresolved:
        for status_map in (a1_status, a2_status):
            if status_map[item_id] != "resolved":
                reason = _clean_text(a1_by_id.loc[item_id, "reason_code"] if status_map is a1_status else a2_by_id.loc[item_id, "reason_code"])
                if reason:
                    reason_counts[reason] = reason_counts.get(reason, 0) + 1
    return {
        "sentence_applicable_pair_total": len(sentence_ids),
        "product_dimension_not_applicable_pair_count": len(not_applicable_ids),
        "a1_resolved_count": a1_resolved,
        "a1_unresolved_count": a1_unresolved,
        "a2_resolved_count": a2_resolved,
        "a2_unresolved_count": a2_unresolved,
        "both_resolved_pair_count": len(both_resolved),
        "pair_with_any_unresolved_count": len(any_unresolved),
        "unresolved_reason_distribution": dict(sorted(reason_counts.items())),
    }


def _verify_adjudication_reread(
    expected: pd.DataFrame,
    actual: pd.DataFrame,
) -> None:
    if list(expected.columns) != list(actual.columns):
        raise ValueError("adjudication output columns changed after write")
    if len(expected) != len(actual):
        raise ValueError("adjudication output row count changed after write")
    for column in expected.columns:
        expected_values = expected[column].map(_clean_text).tolist()
        actual_values = actual[column].map(_clean_text).tolist()
        if expected_values != actual_values:
            raise ValueError(
                f"adjudication output changed after write: {column}"
            )


def _default_output_path(items_dir: Path) -> Path:
    return items_dir / ADJUDICATION_TEMPLATE_FILENAME


def _flag_series(frame: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(frame[column], errors="coerce").fillna(0).astype(int)


def validate_context_frame(
    context: pd.DataFrame,
    product_items: pd.DataFrame,
) -> dict[str, Any]:
    """Validate 07 context and report positive/no-outer items without ID allowlists."""
    _require_columns(
        context,
        [
            "annotation_item_id",
            "parent_asin",
            "target_dimension_code",
            "model_label_value",
            "context_rank",
            "context_source",
            "is_target_dimension_evidence",
            "is_outer_eligible_evidence",
        ],
        "context file",
    )
    _require_columns(
        product_items,
        ["annotation_item_id", "model_label_value", "parent_asin", "dimension_code"],
        "product dimension items",
    )

    product_ids = set(product_items["annotation_item_id"].astype(str))
    context_ids = set(context["annotation_item_id"].astype(str))
    extra = context_ids - product_ids
    if extra:
        raise ValueError(
            f"context contains non-product-dimension IDs: {sorted(extra)[:5]}"
        )
    missing = product_ids - context_ids
    if missing:
        raise ValueError(f"missing context for product-dimension items: {sorted(missing)[:5]}")

    ranks_match = (
        context.groupby("annotation_item_id")["context_rank"].count()
        == context.groupby("annotation_item_id")["context_rank"].nunique()
    )
    if not ranks_match.all():
        bad = ranks_match[~ranks_match].index.tolist()[:5]
        raise ValueError(f"duplicate context_rank within annotation_item_id: {bad}")

    item_lookup = product_items.set_index("annotation_item_id")
    for aid in sorted(product_ids):
        item = item_lookup.loc[aid]
        sub = context[context["annotation_item_id"].astype(str) == aid]
        if sub["parent_asin"].astype(str).str.strip().nunique() != 1:
            raise ValueError(f"parent_asin is inconsistent within context for {aid}")
        if sub["target_dimension_code"].astype(str).str.strip().nunique() != 1:
            raise ValueError(f"target_dimension_code is inconsistent within context for {aid}")
        if sub["model_label_value"].astype(str).str.strip().nunique() != 1:
            raise ValueError(f"model_label_value is inconsistent within context for {aid}")
        if sub["parent_asin"].astype(str).str.strip().iloc[0] != str(item["parent_asin"]).strip():
            raise ValueError(f"parent_asin mismatch for {aid}")
        if sub["target_dimension_code"].astype(str).str.strip().iloc[0] != str(item["dimension_code"]).strip():
            raise ValueError(f"dimension_code mismatch for {aid}")
        if int(float(sub["model_label_value"].iloc[0])) != int(float(item["model_label_value"])):
            raise ValueError(f"model_label_value mismatch for {aid}")

    product_model_labels = pd.to_numeric(
        product_items["model_label_value"], errors="coerce"
    ).fillna(0).astype(int)
    positive_ids = set(
        product_items.loc[product_model_labels == 1, "annotation_item_id"].astype(str)
    )
    unlabeled_ids = set(
        product_items.loc[product_model_labels == 0, "annotation_item_id"].astype(str)
    )

    target_flag = _flag_series(context, "is_target_dimension_evidence")
    outer_flag = _flag_series(context, "is_outer_eligible_evidence")
    context_ids_series = context["annotation_item_id"].astype(str)

    pos_with_target = set(
        context.loc[
            context_ids_series.isin(positive_ids) & (target_flag == 1),
            "annotation_item_id",
        ].astype(str)
    )
    missing_target = positive_ids - pos_with_target
    if missing_target:
        raise ValueError(
            "positive items without target dimension evidence: "
            f"{sorted(missing_target)[:5]}"
        )

    pos_with_outer = set(
        context.loc[
            context_ids_series.isin(positive_ids)
            & (target_flag == 1)
            & (outer_flag == 1),
            "annotation_item_id",
        ].astype(str)
    )
    positive_no_outer_ids = sorted(pos_with_target - pos_with_outer)

    unlabeled_with_target_outer = set(
        context.loc[
            context_ids_series.isin(unlabeled_ids)
            & (target_flag == 1)
            & (outer_flag == 1),
            "annotation_item_id",
        ].astype(str)
    )
    if unlabeled_with_target_outer:
        raise ValueError(
            f"unlabeled items have target outer evidence: {sorted(unlabeled_with_target_outer)[:5]}"
        )

    return {
        "product_dimension_item_count": len(product_ids),
        "positive_item_count": len(positive_ids),
        "unlabeled_item_count": len(unlabeled_ids),
        "positive_items_with_target_evidence": len(pos_with_target),
        "positive_items_with_outer_evidence": len(pos_with_outer),
        "positive_items_without_outer_evidence": len(positive_no_outer_ids),
        "positive_no_outer_annotation_item_ids": positive_no_outer_ids,
    }


def validate_reviewer_context(
    reviewer_context: pd.DataFrame,
    product_items: pd.DataFrame,
    full_context: pd.DataFrame | None = None,
) -> None:
    _require_columns(reviewer_context, [
        "annotation_item_id", "review_context_rank", "review_priority_tier",
        "full_context_row_count", "shortlist_is_exhaustive",
        "requires_full_context_on_uncertainty", "parent_asin",
        "target_dimension_code", "model_label_value",
    ], "reviewer context")
    _require_columns(product_items, ["annotation_item_id", "model_label_value",
                                      "parent_asin", "dimension_code"],
                      "product dimension items")

    product_ids = set(product_items["annotation_item_id"].astype(str))
    ctx_ids = set(reviewer_context["annotation_item_id"].astype(str))
    extra = ctx_ids - product_ids
    if extra:
        raise ValueError(f"reviewer context has non-product-dimension IDs: {sorted(extra)[:5]}")
    missing = product_ids - ctx_ids
    if missing:
        raise ValueError(f"missing reviewer context for items: {sorted(missing)[:5]}")

    for aid in sorted(product_ids):
        sub = reviewer_context[reviewer_context["annotation_item_id"].astype(str) == aid]
        ranks = sorted(sub["review_context_rank"].astype(int))
        expected = list(range(1, len(ranks) + 1))
        if ranks != expected:
            raise ValueError(f"discontinuous ranks for {aid}")

    product_model_labels = pd.to_numeric(
        product_items["model_label_value"], errors="coerce"
    ).fillna(0).astype(int)
    positive_ids = set(
        product_items.loc[product_model_labels == 1, "annotation_item_id"].astype(str)
    )
    for aid in positive_ids:
        sub = reviewer_context[reviewer_context["annotation_item_id"].astype(str) == aid]
        if len(sub) == 0:
            raise ValueError(f"positive item {aid} has no reviewer context")

    unlabeled_ids = set(
        product_items.loc[product_model_labels == 0, "annotation_item_id"].astype(str)
    )
    for aid in unlabeled_ids:
        sub = reviewer_context[reviewer_context["annotation_item_id"].astype(str) == aid]
        if len(sub) > 30:
            raise ValueError(f"unlabeled item {aid} has {len(sub)} rows (cap: 30)")

    human_cols_in_ctx = [c for c in reviewer_context.columns if c.startswith("human_")]
    if human_cols_in_ctx:
        raise ValueError(f"reviewer context contains human columns: {human_cols_in_ctx}")

    item_lookup = product_items.set_index("annotation_item_id")
    for aid in sorted(product_ids):
        item = item_lookup.loc[aid]
        ctx_aid = reviewer_context[reviewer_context["annotation_item_id"].astype(str) == aid]
        if ctx_aid["parent_asin"].astype(str).str.strip().iloc[0] != str(item["parent_asin"]).strip():
            raise ValueError(f"parent_asin mismatch for {aid}")
        if str(ctx_aid["target_dimension_code"].iloc[0]).strip() != str(item["dimension_code"]).strip():
            raise ValueError(f"dimension_code mismatch for {aid}")
        if int(float(ctx_aid["model_label_value"].iloc[0])) != int(float(item["model_label_value"])):
            raise ValueError(f"model_label_value mismatch for {aid}")

    if full_context is not None:
        for aid in sorted(product_ids):
            sub = reviewer_context[reviewer_context["annotation_item_id"].astype(str) == aid]
            fc = full_context[full_context["annotation_item_id"].astype(str) == aid]
            declared_full = int(sub.iloc[0]["full_context_row_count"])
            actual_full = len(fc)
            if declared_full != actual_full:
                raise ValueError(f"{aid}: full_context_row_count mismatch {declared_full} vs {actual_full}")


def _workspace_sha256_snapshot(items_dir: Path) -> dict[str, str]:
    snapshot: dict[str, str] = {}
    for filename in VALIDATION_WORKSPACE_FILENAMES:
        path = items_dir / filename
        if not path.is_file():
            raise FileNotFoundError(f"missing validation workspace file: {path}")
        snapshot[filename] = sha256_file(path)
    return snapshot


def _read_manifest_for_validation(items_dir: Path) -> dict[str, Any]:
    path = items_dir / MANIFEST_FILENAME
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"validation manifest is not valid JSON: {path}") from exc
    if not isinstance(manifest, dict):
        raise ValueError("validation manifest must be a JSON object")
    return manifest


def validate_existing_workspace_read_only(
    *,
    items_dir: Path,
    a1_path: Path,
    a2_path: Path,
    context_path: Path,
    reviewer_path: Path,
    mapping_contract: MappingContract | None = None,
) -> dict[str, Any]:
    """Validate an existing F workspace without creating or modifying any file."""
    product_items = read_csv(items_dir / PRODUCT_DIMENSION_ITEMS_FILENAME)
    sentence_items = read_csv(items_dir / SENTENCE_ITEMS_FILENAME)
    context = read_csv(context_path)
    context_summary = validate_context_frame(context, product_items)

    reviewer_context = read_csv(reviewer_path)
    validate_reviewer_context(reviewer_context, product_items, context)

    manifest = _read_manifest_for_validation(items_dir)
    items = pd.concat([sentence_items, product_items], ignore_index=True, sort=False)
    annotations_a1 = read_csv(a1_path)
    annotations_a2 = read_csv(a2_path)

    annotation_results: dict[str, dict[str, str]] = {}
    for annotator_id, frame in [("A1", annotations_a1), ("A2", annotations_a2)]:
        try:
            validate_annotation_frame(frame, items, expected_annotator_id=annotator_id)
            if mapping_contract is not None:
                validate_annotation_mapping_consistency(frame, items, mapping_contract)
        except ValueError as exc:
            annotation_results[annotator_id] = {
                "status": "FAIL",
                "error": str(exc),
            }
        else:
            annotation_results[annotator_id] = {
                "status": "PASS",
                "error": "",
            }

    return {
        "sentence_item_count": int(len(sentence_items)),
        "product_dimension_item_count": int(len(product_items)),
        "annotation_item_count": int(len(items)),
        "context_row_count": int(len(context)),
        "reviewer_context_row_count": int(len(reviewer_context)),
        "reviewer_annotation_item_count": int(
            reviewer_context["annotation_item_id"].astype(str).nunique()
        ),
        "manifest_tool_version": _clean_text(manifest.get("tool_version")),
        "context": context_summary,
        "annotation_validation": annotation_results,
    }


def _print_validation_summary(summary: dict[str, Any]) -> None:
    context = summary["context"]
    print(
        "Validation summary: "
        f"sentence_items={summary['sentence_item_count']} "
        f"product_dimension_items={summary['product_dimension_item_count']} "
        f"annotation_items={summary['annotation_item_count']} "
        f"context_rows={summary['context_row_count']} "
        f"reviewer_rows={summary['reviewer_context_row_count']}"
    )
    print(
        "Positive evidence summary: "
        f"positive_items_with_outer_evidence={context['positive_items_with_outer_evidence']} "
        f"positive_items_without_outer_evidence={context['positive_items_without_outer_evidence']}"
    )
    print(
        "Positive/no-outer annotation_item_ids: "
        + json.dumps(
            context["positive_no_outer_annotation_item_ids"],
            ensure_ascii=False,
        )
    )
    for annotator_id in ["A1", "A2"]:
        result = summary["annotation_validation"][annotator_id]
        if result["status"] == "PASS":
            print(f"{annotator_id} annotation validation: PASS")
        else:
            print(
                f"{annotator_id} annotation validation: FAIL - {result['error']}",
                file=sys.stderr,
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate A1/A2 annotation files. By default a validated adjudication "
            "template is written; --validate-only is strictly read-only."
        )
    )
    parser.add_argument(
        "--items-dir",
        default=DEFAULT_OUTPUT_DIR,
        type=Path,
        help="Directory containing the existing F validation workspace.",
    )
    parser.add_argument(
        "--a1",
        type=Path,
        help="A1 annotation CSV. Defaults to items-dir/03_annotations_A1.csv.",
    )
    parser.add_argument(
        "--a2",
        type=Path,
        help="A2 annotation CSV. Defaults to items-dir/04_annotations_A2.csv.",
    )
    parser.add_argument(
        "--context",
        type=Path,
        help="Context file path. Defaults to items-dir/07_product_dimension_evidence_context.csv.",
    )
    parser.add_argument(
        "--review-context",
        type=Path,
        help="Reviewer context file path. Defaults to items-dir/08_product_dimension_reviewer_context.csv.",
    )
    parser.add_argument(
        "--mapping",
        type=Path,
        help=(
            "Optional action/error mapping contract. When supplied together with "
            "--require-mapping-consistency, validate derived sentence fields after "
            "canonical schema validation. Also required for the provenance-based "
            "new 05 workflow."
        ),
    )
    parser.add_argument(
        "--require-mapping-consistency",
        action="store_true",
        help=(
            "Only used with --mapping. Applies mapping consistency to already "
            "materialized strict normalized annotations. Raw historical A1/A2 "
            "are validated canonically only by default."
        ),
    )
    parser.add_argument(
        "--derived-a1",
        type=Path,
        help="Optional A1 decision sidecar CSV merged as audit columns into new 05.",
    )
    parser.add_argument(
        "--derived-a2",
        type=Path,
        help="Optional A2 decision sidecar CSV merged as audit columns into new 05.",
    )
    parser.add_argument(
        "--provenance-output",
        type=Path,
        help="Optional new companion provenance JSON for a synthetic/new 05 output.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Adjudication template output path. Defaults to items-dir/05_adjudication_template.csv.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing adjudication template output in write mode.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help=(
            "Read and validate the existing 01-08 workspace without writing any file. "
            "Annotation errors (including missing conditional rationale) return nonzero."
        ),
    )
    args = parser.parse_args()
    if args.validate_only and (
        args.output is not None
        or args.overwrite
        or args.provenance_output is not None
        or args.derived_a1 is not None
        or args.derived_a2 is not None
    ):
        parser.error(
            "--validate-only cannot be combined with --output, --overwrite, "
            "--provenance-output, --derived-a1, or --derived-a2"
        )
    if args.provenance_output is not None and args.mapping is None:
        parser.error("--provenance-output requires --mapping")
    if args.require_mapping_consistency and args.mapping is None:
        parser.error("--require-mapping-consistency requires --mapping")
    if (args.derived_a1 is None) != (args.derived_a2 is None):
        parser.error("--derived-a1 and --derived-a2 must be supplied together")
    if args.derived_a1 is not None and args.mapping is None:
        parser.error("--derived-a1/--derived-a2 require --mapping")
    if args.derived_a1 is not None and args.provenance_output is None:
        parser.error(
            "--derived-a1/--derived-a2 require --provenance-output "
            "(mapping/derived-based 05 must not be produced without a companion provenance sidecar)"
        )
    if args.derived_a1 is not None and args.overwrite:
        parser.error(
            "mapping/derived/provenance-based 05 workflow is new-output-only "
            "and forbids --overwrite"
        )
    if args.provenance_output is not None and args.overwrite:
        parser.error(
            "provenance-based 05 workflow is new-output-only and forbids --overwrite"
        )
    return args


def _run_validate_only(
    *,
    items_dir: Path,
    a1_path: Path,
    a2_path: Path,
    context_path: Path,
    reviewer_path: Path,
    mapping_contract: MappingContract | None = None,
    require_mapping_consistency: bool = False,
) -> int:
    before: dict[str, str] | None = None
    summary: dict[str, Any] | None = None
    validation_error: Exception | None = None

    try:
        before = _workspace_sha256_snapshot(items_dir)
        summary = validate_existing_workspace_read_only(
            items_dir=items_dir,
            a1_path=a1_path,
            a2_path=a2_path,
            context_path=context_path,
            reviewer_path=reviewer_path,
            mapping_contract=(
                mapping_contract if require_mapping_consistency else None
            ),
        )
    except (FileNotFoundError, ValueError) as exc:
        validation_error = exc

    try:
        after = _workspace_sha256_snapshot(items_dir)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Zero-write SHA-256 check: FAIL - {exc}", file=sys.stderr)
        return 2

    if before is None or before != after:
        changed = [] if before is None else [
            name for name in VALIDATION_WORKSPACE_FILENAMES
            if before.get(name) != after.get(name)
        ]
        print(
            "Zero-write SHA-256 check: FAIL"
            + (f" changed={changed}" if changed else ""),
            file=sys.stderr,
        )
        return 2

    if summary is not None:
        _print_validation_summary(summary)
    print(
        "Zero-write SHA-256 check: PASS "
        f"({len(VALIDATION_WORKSPACE_FILENAMES)}/{len(VALIDATION_WORKSPACE_FILENAMES)} unchanged)"
    )

    if validation_error is not None:
        print(f"Validation failed: {validation_error}", file=sys.stderr)
        return 1

    assert summary is not None
    annotation_failures = [
        annotator_id
        for annotator_id, result in summary["annotation_validation"].items()
        if result["status"] != "PASS"
    ]
    if annotation_failures:
        print(
            "Validation failed: annotation validation failed for "
            + ", ".join(annotation_failures),
            file=sys.stderr,
        )
        return 1

    print("Read-only validation: PASS")
    return 0


def main() -> int:
    args = parse_args()
    items_dir = Path(args.items_dir)
    a1_path = args.a1 or (items_dir / ANNOTATIONS_A1_FILENAME)
    a2_path = args.a2 or (items_dir / ANNOTATIONS_A2_FILENAME)
    output_path = args.output or _default_output_path(items_dir)
    context_path = args.context or (items_dir / CONTEXT_FILENAME)
    reviewer_path = args.review_context or (items_dir / REVIEWER_CONTEXT_FILENAME)
    provenance_output = args.provenance_output
    output_existed_before = output_path.exists()
    invocation_created_output = False
    output_written_sha: str | None = None

    try:
        mapping_contract = (
            load_canonical_mapping(
                args.mapping,
                actions=HUMAN_ACTIONS,
                error_types=ERROR_TYPES,
                dimension_values=DIMENSION_VALUES,
                polarities=POLARITIES,
            )
            if args.mapping is not None
            else None
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"Validation failed: {exc}", file=sys.stderr)
        return 1

    if args.validate_only:
        return _run_validate_only(
            items_dir=items_dir,
            a1_path=a1_path,
            a2_path=a2_path,
            context_path=context_path,
            reviewer_path=reviewer_path,
            mapping_contract=mapping_contract,
            require_mapping_consistency=args.require_mapping_consistency,
        )

    try:
        if output_path.exists() and not args.overwrite:
            raise FileExistsError(f"output already exists; pass --overwrite: {output_path}")
        if provenance_output is not None and provenance_output.exists():
            raise FileExistsError(
                f"provenance sidecar already exists: {provenance_output}"
            )

        product_items = read_csv(items_dir / PRODUCT_DIMENSION_ITEMS_FILENAME)
        if context_path.exists():
            context = read_csv(context_path)
            context_summary = validate_context_frame(context, product_items)
            print(
                f"Context validated: {len(context)} rows, "
                f"{context['annotation_item_id'].nunique()} items, "
                f"positive/no-outer={context_summary['positive_items_without_outer_evidence']}"
            )

        if reviewer_path.exists():
            reviewer_ctx = read_csv(reviewer_path)
            full_ctx = read_csv(context_path) if context_path.exists() else None
            validate_reviewer_context(reviewer_ctx, product_items, full_ctx)
            print(
                f"Reviewer context validated: {len(reviewer_ctx)} rows, "
                f"{reviewer_ctx['annotation_item_id'].nunique()} items"
            )

        items = load_items(items_dir)
        annotations_a1 = read_csv(a1_path)
        annotations_a2 = read_csv(a2_path)
        derived_a1 = read_csv(args.derived_a1) if args.derived_a1 is not None else None
        derived_a2 = read_csv(args.derived_a2) if args.derived_a2 is not None else None
        if mapping_contract is not None:
            validate_annotation_frame(
                annotations_a1, items, expected_annotator_id="A1"
            )
            validate_annotation_frame(
                annotations_a2, items, expected_annotator_id="A2"
            )
            if args.require_mapping_consistency:
                validate_annotation_mapping_consistency(
                    annotations_a1, items, mapping_contract
                )
                validate_annotation_mapping_consistency(
                    annotations_a2, items, mapping_contract
                )
        # BLOCKER F1: derived sidecars + provenance must verify BEFORE any 05 output
        pair_counts: dict[str, Any] | None = None
        if derived_a1 is not None and derived_a2 is not None:
            assert mapping_contract is not None
            assert args.derived_a1 is not None and args.derived_a2 is not None
            pair_counts = _verify_derived_inputs_before_05(
                items=items,
                a1_path=a1_path,
                a2_path=a2_path,
                derived_a1=derived_a1,
                derived_a2=derived_a2,
                derived_a1_path=Path(args.derived_a1),
                derived_a2_path=Path(args.derived_a2),
                mapping_contract=mapping_contract,
            )
        adjudication = build_adjudication_template(
            items,
            annotations_a1,
            annotations_a2,
            derived_a1=derived_a1,
            derived_a2=derived_a2,
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        write_csv(adjudication, output_path)
        invocation_created_output = True
        reread = read_csv(output_path)
        _verify_adjudication_reread(adjudication, reread)
        output_written_sha = sha256_file(output_path)

        if provenance_output is not None:
            assert mapping_contract is not None
            try:
                input_files = {
                    "annotations_a1": a1_path,
                    "annotations_a2": a2_path,
                    "product_dimension_items": (
                        items_dir / PRODUCT_DIMENSION_ITEMS_FILENAME
                    ),
                    "sentence_items": items_dir / SENTENCE_ITEMS_FILENAME,
                }
                if derived_a1 is not None:
                    input_files["derived_a1"] = Path(args.derived_a1)
                if derived_a2 is not None:
                    input_files["derived_a2"] = Path(args.derived_a2)
                unresolved_reviewer_decision_count = (
                    int((pair_counts or {}).get("a1_unresolved_count", 0))
                    + int((pair_counts or {}).get("a2_unresolved_count", 0))
                )
                provenance = build_provenance(
                    mapping_metadata=mapping_contract.metadata,
                    tool_name=TOOL_NAME,
                    tool_version=TOOL_VERSION,
                    git_state=read_git_state(Path(__file__).parents[2]),
                    input_files=input_files,
                    output_logical_name="adjudication_template",
                    output_path=output_path,
                    row_count=len(reread),
                    annotator_ids=["A1", "A2"],
                    mode="build_adjudication_template",
                    unresolved_count=unresolved_reviewer_decision_count,
                    allowed_changed_fields=[],
                )
                if pair_counts is not None:
                    provenance.update(pair_counts)
                    provenance["unresolved_reviewer_decision_count"] = (
                        unresolved_reviewer_decision_count
                    )
                write_json_new_atomic(provenance_output, provenance)
            except Exception:
                current_output_sha: str | None = None
                if output_path.is_file():
                    try:
                        current_output_sha = sha256_file(output_path)
                    except OSError:
                        pass
                if (
                    not output_existed_before
                    and invocation_created_output
                    and output_written_sha is not None
                    and current_output_sha == output_written_sha
                ):
                    output_path.unlink()
                raise
        print(
            "Validated A1/A2 annotations and wrote adjudication template: "
            f"{output_path.resolve()}"
        )
        return 0
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"Validation failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
