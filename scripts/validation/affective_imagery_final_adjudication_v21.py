#!/usr/bin/env python3
"""Orchestrate final adjudication through the frozen production evaluator.

This module does not load or interpret mapping JSON. It accepts an existing
MappingContract, adapts adjudicated field names, delegates declared transforms
to adapt_row_inputs(), delegates decisions to evaluate_mapping(), and enforces
the approved human closure for mapping-unresolved sentence rows.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

import pandas as pd

try:
    from .affective_imagery_action_error_mapping_v21 import (
        MappingContract,
        adapt_row_inputs,
        evaluate_mapping,
    )
except ImportError:  # pragma: no cover - direct script execution
    from affective_imagery_action_error_mapping_v21 import (
        MappingContract,
        adapt_row_inputs,
        evaluate_mapping,
    )


_ADJUDICATED_TO_EVALUATOR_FIELDS = {
    "adjudicated_packaging_visual": "human_packaging_visual",
    "adjudicated_relation_valid": "human_relation_valid",
    "adjudicated_package_level": "human_package_level",
    "adjudicated_dimension_code": "human_dimension_code",
    "adjudicated_additional_dimension_codes": (
        "human_additional_dimension_codes"
    ),
    "adjudicated_polarity": "human_polarity",
}


@dataclass(frozen=True)
class FinalAdjudicationResult:
    """One final action/error outcome with its underlying mapping status."""

    status: str
    action: str | None
    error_type: str | None
    reason_code: str | None
    rule_id: str | None = None


@dataclass(frozen=True)
class AdjudicatedMappingConsistencyReport:
    """Complete sentence-level mapping-consistency counts for one frame."""

    total_sentence_count: int
    mapping_resolved_count: int
    mapping_unresolved_count: int
    resolved_mismatch_count: int
    unresolved_reason_distribution: Mapping[str, int]


class AdjudicatedMappingConsistencyError(ValueError):
    """Final consistency failure retaining the complete validation report."""

    def __init__(
        self,
        message: str,
        report: AdjudicatedMappingConsistencyReport,
    ) -> None:
        super().__init__(message)
        self.report = report


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value).strip()


def _item_type(row: Mapping[str, Any]) -> str:
    return _clean_text(row.get("item_type"))


def validate_adjudicated_additional_dimensions(
    value: Any,
    *,
    dimension_values: set[str] | frozenset[str],
) -> None:
    """Validate exact pipe-separated final dimension tokens without rewriting."""

    value_text = _clean_text(value)
    if not value_text:
        return
    tokens = str(value).split("|") if isinstance(value, str) else [value_text]
    if any(
        not token
        or token != token.strip()
        or token not in dimension_values
        for token in tokens
    ):
        raise ValueError("invalid adjudicated_additional_dimension_codes")


def adapt_adjudicated_inputs(
    row: Mapping[str, Any],
    contract: MappingContract,
) -> dict[str, Any]:
    """Rename final semantic fields and invoke the contract's own transforms."""

    adapted = dict(row)
    missing = [
        source
        for source in _ADJUDICATED_TO_EVALUATOR_FIELDS
        if source not in row
    ]
    if missing:
        raise ValueError(
            "final sentence row is missing semantic fields: "
            + ", ".join(missing)
        )
    for source, target in _ADJUDICATED_TO_EVALUATOR_FIELDS.items():
        adapted[target] = row[source]
    return adapt_row_inputs(adapted, contract)


def validate_adjudicated_pair_closure(
    action: Any,
    error_type: Any,
    note: Any = "",
) -> None:
    """Enforce the approved action/error pairs independent of mapping status."""

    action_value = _clean_text(action)
    error_value = _clean_text(error_type)
    note_value = _clean_text(note)
    if action_value == "keep" and error_value != "none":
        raise ValueError(
            "keep requires error_type none "
            f"(got {error_value or 'blank'})"
        )
    if (
        action_value == "change_polarity"
        and error_value != "negation_error"
    ):
        raise ValueError(
            "change_polarity requires error_type negation_error "
            f"(got {error_value or 'blank'})"
        )
    if error_value == "other":
        if action_value == "keep":
            raise ValueError("other requires a non-keep action")
        if not note_value:
            raise ValueError("other requires nonblank adjudication_note_cn")


def validate_manual_adjudication_closure(
    action: Any,
    error_type: Any,
    note: Any,
    *,
    action_enum: frozenset[str],
    error_type_enum: frozenset[str],
) -> None:
    """Validate explicit human closure for an unresolved mapping result."""

    action_value = _clean_text(action)
    error_value = _clean_text(error_type)
    note_value = _clean_text(note)
    if not action_value:
        raise ValueError("manual action is required for mapping unresolved")
    if action_value not in action_enum:
        raise ValueError(f"manual action is not canonical: {action_value}")
    if not error_value:
        raise ValueError(
            "manual error_type is required for mapping unresolved"
        )
    if error_value not in error_type_enum:
        raise ValueError(
            f"manual error_type is not canonical: {error_value}"
        )
    if not note_value:
        raise ValueError(
            "adjudication_note_cn is required for mapping unresolved"
        )
    validate_adjudicated_pair_closure(
        action_value,
        error_value,
        note_value,
    )


def finalize_adjudication(
    row: Mapping[str, Any],
    contract: MappingContract,
    *,
    dimension_values: set[str] | frozenset[str],
    manual_action: Any = None,
    manual_error_type: Any = None,
    adjudication_note_cn: Any = None,
) -> FinalAdjudicationResult:
    """Finalize one row without guessing an unresolved action or error type."""

    if _item_type(row) == "product_dimension":
        return FinalAdjudicationResult(
            status="not_applicable",
            action=None,
            error_type=None,
            reason_code=None,
        )
    if _item_type(row) != "sentence":
        raise ValueError(f"invalid item_type: {_item_type(row) or 'blank'}")

    validate_adjudicated_additional_dimensions(
        row.get("adjudicated_additional_dimension_codes"),
        dimension_values=dimension_values,
    )
    mapping_result = evaluate_mapping(
        contract,
        adapt_adjudicated_inputs(row, contract),
    )
    if mapping_result.status == "resolved":
        return FinalAdjudicationResult(
            status="resolved",
            action=mapping_result.action,
            error_type=mapping_result.error_type,
            reason_code=None,
            rule_id=mapping_result.rule_id,
        )

    validate_manual_adjudication_closure(
        manual_action,
        manual_error_type,
        adjudication_note_cn,
        action_enum=contract.action_enum,
        error_type_enum=contract.error_type_enum,
    )
    return FinalAdjudicationResult(
        status="unresolved",
        action=_clean_text(manual_action),
        error_type=_clean_text(manual_error_type),
        reason_code=mapping_result.reason_code,
        rule_id=mapping_result.rule_id,
    )


def validate_adjudicated_mapping_consistency(
    adjudication: pd.DataFrame,
    contract: MappingContract,
    *,
    dimension_values: set[str] | frozenset[str],
) -> AdjudicatedMappingConsistencyReport:
    """Validate every final sentence against one injected mapping contract."""

    total_sentence_count = 0
    mapping_resolved_count = 0
    mapping_unresolved_count = 0
    resolved_mismatch_count = 0
    unresolved_reasons: Counter[str] = Counter()
    errors: list[str] = []

    for index, row in adjudication.iterrows():
        item_type = _item_type(row)
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

        total_sentence_count += 1
        item_id = _clean_text(row.get("annotation_item_id"))
        row_label = (
            f"row {index + 1}"
            + (f" ({item_id})" if item_id else "")
        )
        try:
            validate_adjudicated_additional_dimensions(
                row.get("adjudicated_additional_dimension_codes"),
                dimension_values=dimension_values,
            )
            mapping_result = evaluate_mapping(
                contract,
                adapt_adjudicated_inputs(row, contract),
            )
        except ValueError as exc:
            errors.append(
                f"{row_label}: mapping evaluation failed: {exc}"
            )
            continue
        if mapping_result.status == "resolved":
            mapping_resolved_count += 1
            action = _clean_text(row.get("adjudicated_action"))
            error_type = _clean_text(
                row.get("adjudicated_error_type")
            )
            mismatch_fields: list[str] = []
            if action != mapping_result.action:
                mismatch_fields.append("adjudicated_action mismatch")
            if error_type != mapping_result.error_type:
                mismatch_fields.append(
                    "adjudicated_error_type mismatch"
                )
            if mismatch_fields:
                resolved_mismatch_count += 1
                errors.append(
                    f"{row_label}: {', '.join(mismatch_fields)}; "
                    f"expected action={mapping_result.action} "
                    f"error_type={mapping_result.error_type}; "
                    f"got action={action or 'blank'} "
                    f"error_type={error_type or 'blank'}"
                )
            continue

        mapping_unresolved_count += 1
        reason_code = mapping_result.reason_code or "unresolved"
        unresolved_reasons[reason_code] += 1
        try:
            validate_manual_adjudication_closure(
                row.get("adjudicated_action"),
                row.get("adjudicated_error_type"),
                row.get("adjudication_note_cn"),
                action_enum=contract.action_enum,
                error_type_enum=contract.error_type_enum,
            )
        except ValueError as exc:
            errors.append(
                f"{row_label}: mapping unresolved "
                f"({reason_code}): {exc}"
            )

    report = AdjudicatedMappingConsistencyReport(
        total_sentence_count=total_sentence_count,
        mapping_resolved_count=mapping_resolved_count,
        mapping_unresolved_count=mapping_unresolved_count,
        resolved_mismatch_count=resolved_mismatch_count,
        unresolved_reason_distribution=MappingProxyType(
            dict(sorted(unresolved_reasons.items()))
        ),
    )
    if errors:
        raise AdjudicatedMappingConsistencyError(
            "; ".join(errors),
            report,
        )
    return report
