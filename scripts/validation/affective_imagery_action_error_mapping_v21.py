#!/usr/bin/env python3
"""Load and evaluate the unified Wave 0/1 action/error mapping contract.

This is the ONLY production evaluator. Business rules are declared in the
external JSON mapping (config/affective_imagery/action_error_mapping_v21.json).
This module implements a thin structural envelope: contract loading and
validation, declared transforms, rule matching, explicit priority order, and
resolved/unresolved outcomes. No packaging/dimension/polarity business rules
are hardcoded here.

Contract design (minimal sufficient):

- operators: exact, in, field_eq, field_ne,
             collection_contains_field, collection_not_contains_field
- conditions in one rule are implicitly ANDed; OR scenarios are split into
  multiple rules
- priority_order is declared by the contract: "lower_wins" (1 = highest
  precedence) is the production choice
- transforms: split_union (dimension union), map_value (package-level alias)
- the automatic evaluator NEVER emits error_type "other"; unresolved results
  route to adjudication
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import pandas as pd


SUPPORTED_CONTRACT_SCHEMA = "affective_imagery.action_error_mapping"
SUPPORTED_CONTRACT_SCHEMA_VERSION = 2
SUPPORTED_SCOPE = "sentence"
SUPPORTED_OPERATORS = {
    "exact",
    "in",
    "field_eq",
    "field_ne",
    "collection_contains_field",
    "collection_not_contains_field",
}
SUPPORTED_PRIORITY_ORDERS = {"lower_wins", "higher_wins"}
SUPPORTED_TRANSFORMS = {"split_union", "map_value"}

_ENVELOPE_FIELDS = {
    "contract_schema",
    "contract_schema_version",
    "mapping_version",
    "scope",
    "priority_order",
    "input_fields",
    "transforms",
    "declared_vocabulary",
    "action_enum",
    "error_type_enum",
    "rules",
    "fallback",
    "provenance",
}


class MappingContractError(ValueError):
    """Raised when a mapping contract or evaluator input is structurally invalid."""


@dataclass(frozen=True)
class MappingMetadata:
    mapping_version: str
    contract_schema_version: int
    mapping_sha256: str
    mapping_path: Path


@dataclass(frozen=True)
class MappingResultSpec:
    status: str
    action: str | None
    error_type: str | None
    reason_code: str | None


@dataclass(frozen=True)
class MappingCondition:
    operator: str
    field: str
    value: Any = None
    values: tuple[Any, ...] = ()
    other_field: str | None = None


@dataclass(frozen=True)
class MappingRule:
    rule_id: str
    priority: int
    conditions: tuple[MappingCondition, ...]
    result: MappingResultSpec


@dataclass(frozen=True)
class MappingTransform:
    op: str
    target: str
    source: str | None = None
    sources: tuple[str, ...] = ()
    mapping: Mapping[str, str] | None = None
    delimiter: str | None = None
    ignore_values: tuple[str, ...] = ()
    on_unmapped: str = "error"


@dataclass(frozen=True)
class MappingContract:
    metadata: MappingMetadata
    scope: str
    priority_order: str
    input_fields: tuple[str, ...]
    transforms: tuple[MappingTransform, ...]
    derived_field_names: tuple[str, ...]
    action_enum: frozenset[str]
    error_type_enum: frozenset[str]
    rules: tuple[MappingRule, ...]
    fallback: MappingResultSpec
    provenance: Mapping[str, Any]


@dataclass(frozen=True)
class EvaluationResult:
    status: str
    action: str | None
    error_type: str | None
    reason_code: str | None
    rule_id: str | None
    priority: int | None
    matched_rule_ids: tuple[str, ...] = ()


def _nonblank_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise MappingContractError(f"{field} must be a nonblank string")
    return value.strip()


def _unique_string_sequence(value: Any, field: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise MappingContractError(f"{field} must be a nonempty string array")
    cleaned = tuple(_nonblank_string(item, field) for item in value)
    if len(cleaned) != len(set(cleaned)):
        raise MappingContractError(f"{field} contains duplicate values")
    return cleaned


def _canonicalize_value(value: Any) -> Any:
    if value is None:
        return ""
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        missing = False
    if isinstance(missing, bool) and missing:
        return ""
    if isinstance(value, str):
        return value.strip()
    return value


def _validate_result(
    value: Any,
    *,
    context: str,
    action_enum: frozenset[str],
    error_type_enum: frozenset[str],
) -> MappingResultSpec:
    if not isinstance(value, dict):
        raise MappingContractError(f"{context} must be an object")
    status = value.get("status")
    if status not in {"resolved", "unresolved"}:
        raise MappingContractError(f"{context}.status must be resolved or unresolved")
    if status == "resolved":
        allowed = {"status", "action", "error_type"}
        if set(value) != allowed:
            raise MappingContractError(
                f"{context} resolved result must contain status/action/error_type"
            )
        action = _nonblank_string(value.get("action"), f"{context}.action")
        error_type = _nonblank_string(value.get("error_type"), f"{context}.error_type")
        if action not in action_enum:
            raise MappingContractError(
                f"{context}.action is not declared in action_enum: {action}"
            )
        if error_type not in error_type_enum:
            raise MappingContractError(
                f"{context}.error_type is not declared in error_type_enum: {error_type}"
            )
        return MappingResultSpec("resolved", action, error_type, None)

    if set(value) != {"status", "reason_code"}:
        raise MappingContractError(
            f"{context} unresolved result must contain status/reason_code"
        )
    reason_code = _nonblank_string(value.get("reason_code"), f"{context}.reason_code")
    return MappingResultSpec("unresolved", None, None, reason_code)


def _validate_transform(
    value: Any,
    *,
    context: str,
    declared_inputs: frozenset[str],
) -> MappingTransform:
    if not isinstance(value, dict):
        raise MappingContractError(f"{context} must be an object")
    op = value.get("op")
    if op not in SUPPORTED_TRANSFORMS:
        raise MappingContractError(f"unsupported transform in {context}: {op}")
    target = _nonblank_string(value.get("target"), f"{context}.target")

    if op == "split_union":
        if set(value) != {"op", "target", "sources", "delimiter", "ignore_values"}:
            raise MappingContractError(
                f"{context} split_union requires op/target/sources/delimiter/ignore_values"
            )
        sources = _unique_string_sequence(value["sources"], f"{context}.sources")
        for source in sources:
            if source not in declared_inputs:
                raise MappingContractError(
                    f"{context}.sources contains undeclared input: {source}"
                )
        delimiter = value["delimiter"]
        if not isinstance(delimiter, str) or not delimiter:
            raise MappingContractError(f"{context}.delimiter must be a nonblank string")
        ignore = value["ignore_values"]
        if not isinstance(ignore, list):
            raise MappingContractError(f"{context}.ignore_values must be an array")
        ignore_tuple = tuple(_canonicalize_value(item) for item in ignore)
        return MappingTransform(
            op=op,
            target=target,
            sources=sources,
            delimiter=delimiter,
            ignore_values=ignore_tuple,
        )

    if set(value) != {"op", "target", "source", "mapping", "on_unmapped"}:
        raise MappingContractError(
            f"{context} map_value requires op/target/source/mapping/on_unmapped"
        )
    source = _nonblank_string(value.get("source"), f"{context}.source")
    if source not in declared_inputs:
        raise MappingContractError(f"{context}.source is not a declared input: {source}")
    raw_mapping = value["mapping"]
    if not isinstance(raw_mapping, dict):
        raise MappingContractError(f"{context}.mapping must be an object")
    mapping = {str(key): str(val) for key, val in raw_mapping.items()}
    on_unmapped = _nonblank_string(value.get("on_unmapped"), f"{context}.on_unmapped")
    if on_unmapped not in {"error", "keep"}:
        raise MappingContractError(f"{context}.on_unmapped must be error or keep")
    return MappingTransform(
        op=op,
        target=target,
        source=source,
        mapping=mapping,
        on_unmapped=on_unmapped,
    )


def _validate_condition(
    value: Any,
    *,
    context: str,
    available_fields: frozenset[str],
) -> MappingCondition:
    if not isinstance(value, dict):
        raise MappingContractError(f"{context} must be an object")
    operator = value.get("operator")
    if operator not in SUPPORTED_OPERATORS:
        raise MappingContractError(f"unsupported condition operator in {context}: {operator}")
    field = _nonblank_string(value.get("field"), f"{context}.field")
    if field not in available_fields:
        raise MappingContractError(f"{context}.field is not a declared input: {field}")

    if operator in {"exact", "in"}:
        if operator == "exact":
            if set(value) != {"operator", "field", "value"}:
                raise MappingContractError(
                    f"{context} exact condition requires operator/field/value"
                )
            return MappingCondition(
                operator, field, value=_canonicalize_value(value["value"])
            )
        if set(value) != {"operator", "field", "values"}:
            raise MappingContractError(
                f"{context} in condition requires operator/field/values"
            )
        values = value["values"]
        if not isinstance(values, list) or not values:
            raise MappingContractError(f"{context}.values must be a nonempty array")
        canonical = tuple(_canonicalize_value(item) for item in values)
        if len(canonical) != len(set(canonical)):
            raise MappingContractError(f"{context}.values contains duplicates")
        return MappingCondition(operator, field, values=canonical)

    if set(value) != {"operator", "field", "other_field"}:
        raise MappingContractError(
            f"{context} field comparison requires operator/field/other_field"
        )
    other_field = _nonblank_string(value.get("other_field"), f"{context}.other_field")
    if other_field not in available_fields:
        raise MappingContractError(f"{context}.other_field is not a declared input: {other_field}")
    return MappingCondition(operator, field, other_field=other_field)


def _apply_transforms(
    inputs: dict[str, Any],
    contract: MappingContract,
) -> dict[str, Any]:
    """Apply declared transforms to produce derived fields (no business defaults)."""
    result = dict(inputs)
    for transform in contract.transforms:
        if transform.op == "split_union":
            collection: list[str] = []
            for source in transform.sources:
                raw_value = result.get(source, "")
                cleaned = _canonicalize_value(raw_value)
                for part in str(cleaned).split(transform.delimiter or "|"):
                    part = part.strip()
                    if not part or part in transform.ignore_values:
                        continue
                    collection.append(part)
            seen: set[str] = set()
            unique_collection: list[str] = []
            for item in collection:
                if item not in seen:
                    seen.add(item)
                    unique_collection.append(item)
            result[transform.target] = unique_collection
            continue
        # map_value
        source_value = _canonicalize_value(result.get(transform.source or "", ""))
        mapping = transform.mapping or {}
        if source_value in mapping:
            result[transform.target] = mapping[source_value]
        elif transform.on_unmapped == "keep":
            result[transform.target] = source_value
        else:
            raise MappingContractError(
                f"map_value transform cannot map source value: {source_value!r}"
            )
    return result


def load_mapping(
    path: Path,
    *,
    schema_action_enum: set[str],
    schema_error_type_enum: set[str],
    schema_dimension_values: set[str] | None = None,
    schema_polarity_values: set[str] | None = None,
) -> MappingContract:
    """Load, validate, and hash a mapping contract without business defaults."""
    mapping_path = Path(path)
    if not mapping_path.is_file():
        raise FileNotFoundError(f"mapping file does not exist: {mapping_path}")
    raw = mapping_path.read_bytes()
    mapping_sha256 = hashlib.sha256(raw).hexdigest()
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MappingContractError(
            f"mapping file is not valid JSON: {mapping_path}"
        ) from exc
    if not isinstance(payload, dict):
        raise MappingContractError("mapping JSON must be an object")
    if set(payload) != _ENVELOPE_FIELDS:
        missing = sorted(_ENVELOPE_FIELDS - set(payload))
        extra = sorted(set(payload) - _ENVELOPE_FIELDS)
        raise MappingContractError(
            f"mapping envelope fields mismatch; missing={missing} extra={extra}"
        )
    if payload["contract_schema"] != SUPPORTED_CONTRACT_SCHEMA:
        raise MappingContractError(
            f"unsupported contract_schema: {payload['contract_schema']}"
        )
    if payload["contract_schema_version"] != SUPPORTED_CONTRACT_SCHEMA_VERSION:
        raise MappingContractError(
            "unsupported contract_schema_version: "
            f"{payload['contract_schema_version']}"
        )
    mapping_version = _nonblank_string(payload["mapping_version"], "mapping_version")
    if payload["scope"] != SUPPORTED_SCOPE:
        raise MappingContractError(f"unsupported mapping scope: {payload['scope']}")
    priority_order = _nonblank_string(payload["priority_order"], "priority_order")
    if priority_order not in SUPPORTED_PRIORITY_ORDERS:
        raise MappingContractError(
            f"unsupported priority_order: {priority_order}"
        )

    input_fields = _unique_string_sequence(payload["input_fields"], "input_fields")
    declared_inputs = frozenset(input_fields)
    transforms_value = payload["transforms"]
    if not isinstance(transforms_value, list):
        raise MappingContractError("transforms must be an array")
    transforms: list[MappingTransform] = []
    derived_names: list[str] = []
    for index, transform_value in enumerate(transforms_value):
        transform = _validate_transform(
            transform_value,
            context=f"transforms[{index}]",
            declared_inputs=declared_inputs,
        )
        transforms.append(transform)
        derived_names.append(transform.target)

    # Declared vocabulary must not expand canonical schema
    declared_vocabulary = payload["declared_vocabulary"]
    if not isinstance(declared_vocabulary, dict):
        raise MappingContractError("declared_vocabulary must be an object")
    dimension_values = _unique_string_sequence(
        declared_vocabulary.get("dimensions"), "declared_vocabulary.dimensions"
    )
    polarity_values = _unique_string_sequence(
        declared_vocabulary.get("human_polarity"),
        "declared_vocabulary.human_polarity",
    )
    if schema_dimension_values is not None:
        outside = sorted(set(dimension_values) - schema_dimension_values)
        if outside:
            raise MappingContractError(
                "declared dimension vocabulary is outside canonical schema: "
                f"{outside}"
            )
    if schema_polarity_values is not None:
        outside = sorted(set(polarity_values) - schema_polarity_values)
        if outside:
            raise MappingContractError(
                "declared human polarity vocabulary is outside canonical schema: "
                f"{outside}"
            )

    action_values = _unique_string_sequence(payload["action_enum"], "action_enum")
    error_values = _unique_string_sequence(
        payload["error_type_enum"], "error_type_enum"
    )
    action_enum = frozenset(action_values)
    error_type_enum = frozenset(error_values)
    action_outside_schema = sorted(action_enum - set(schema_action_enum))
    error_outside_schema = sorted(error_type_enum - set(schema_error_type_enum))
    if action_outside_schema or error_outside_schema:
        raise MappingContractError(
            "mapping enum is outside canonical schema; "
            f"actions={action_outside_schema} errors={error_outside_schema}"
        )

    available_fields = declared_inputs | frozenset(derived_names)
    rules_value = payload["rules"]
    if not isinstance(rules_value, list):
        raise MappingContractError("rules must be an array")
    rules: list[MappingRule] = []
    seen_ids: set[str] = set()
    seen_priorities: set[int] = set()
    for index, rule_value in enumerate(rules_value):
        context = f"rules[{index}]"
        if not isinstance(rule_value, dict) or set(rule_value) != {
            "rule_id", "priority", "conditions", "result"
        }:
            raise MappingContractError(
                f"{context} must contain rule_id/priority/conditions/result"
            )
        rule_id = _nonblank_string(rule_value["rule_id"], f"{context}.rule_id")
        if rule_id in seen_ids:
            raise MappingContractError(f"duplicate rule_id: {rule_id}")
        seen_ids.add(rule_id)
        priority = rule_value["priority"]
        if isinstance(priority, bool) or not isinstance(priority, int):
            raise MappingContractError(f"{context}.priority must be an integer")
        if priority in seen_priorities:
            raise MappingContractError(f"duplicate rule priority: {priority}")
        seen_priorities.add(priority)
        conditions_value = rule_value["conditions"]
        if not isinstance(conditions_value, list):
            raise MappingContractError(f"{context}.conditions must be an array")
        conditions = tuple(
            _validate_condition(
                condition,
                context=f"{context}.conditions[{condition_index}]",
                available_fields=available_fields,
            )
            for condition_index, condition in enumerate(conditions_value)
        )
        result = _validate_result(
            rule_value["result"],
            context=f"{context}.result",
            action_enum=action_enum,
            error_type_enum=error_type_enum,
        )
        rules.append(MappingRule(rule_id, priority, conditions, result))

    fallback = _validate_result(
        payload["fallback"],
        context="fallback",
        action_enum=action_enum,
        error_type_enum=error_type_enum,
    )
    if not isinstance(payload["provenance"], dict):
        raise MappingContractError("provenance must be an object")
    metadata = MappingMetadata(
        mapping_version=mapping_version,
        contract_schema_version=SUPPORTED_CONTRACT_SCHEMA_VERSION,
        mapping_sha256=mapping_sha256,
        mapping_path=mapping_path.resolve(),
    )
    return MappingContract(
        metadata=metadata,
        scope=SUPPORTED_SCOPE,
        priority_order=priority_order,
        input_fields=input_fields,
        transforms=tuple(transforms),
        derived_field_names=tuple(derived_names),
        action_enum=action_enum,
        error_type_enum=error_type_enum,
        rules=tuple(rules),
        fallback=fallback,
        provenance=dict(payload["provenance"]),
    )


def adapt_row_inputs(
    row: Mapping[str, Any],
    contract: MappingContract,
) -> dict[str, Any]:
    """Extract only declared raw fields, then apply declared transforms."""
    missing = [field for field in contract.input_fields if field not in row]
    if missing:
        raise MappingContractError(
            f"row is missing declared input fields: {', '.join(missing)}"
        )
    raw_inputs = {
        field: _canonicalize_value(row[field])
        for field in contract.input_fields
    }
    return _apply_transforms(raw_inputs, contract)


def _collection_value(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if value in (None, ""):
        return []
    return [value]


def _condition_matches(condition: MappingCondition, inputs: Mapping[str, Any]) -> bool:
    current = inputs[condition.field]
    if condition.operator == "exact":
        return current == condition.value
    if condition.operator == "in":
        return current in condition.values
    other = inputs[condition.other_field or ""]
    if condition.operator == "field_eq":
        return current == other
    if condition.operator == "field_ne":
        return current != other
    if condition.operator == "collection_contains_field":
        return other in _collection_value(current)
    return other not in _collection_value(current)


def _evaluation_from_spec(
    result: MappingResultSpec,
    *,
    rule_id: str | None,
    priority: int | None,
    matched_rule_ids: tuple[str, ...] = (),
) -> EvaluationResult:
    return EvaluationResult(
        status=result.status,
        action=result.action,
        error_type=result.error_type,
        reason_code=result.reason_code,
        rule_id=rule_id,
        priority=priority,
        matched_rule_ids=matched_rule_ids,
    )


def evaluate_mapping(
    contract: MappingContract,
    inputs: Mapping[str, Any],
) -> EvaluationResult:
    """Evaluate all AND rules and return the explicitly highest-priority result.

    Priority order is declared by the contract (production: "lower_wins",
    where 1 = highest precedence).
    """
    missing = [field for field in contract.input_fields if field not in inputs]
    missing.extend(
        field for field in contract.derived_field_names if field not in inputs
    )
    if missing:
        raise MappingContractError(
            f"evaluator inputs are missing declared fields: {', '.join(missing)}"
        )
    matches = [
        rule
        for rule in contract.rules
        if all(_condition_matches(condition, inputs) for condition in rule.conditions)
    ]
    if not matches:
        return _evaluation_from_spec(contract.fallback, rule_id=None, priority=None)
    if contract.priority_order == "lower_wins":
        selected = min(matches, key=lambda rule: rule.priority)
    else:
        selected = max(matches, key=lambda rule: rule.priority)
    matched_ids = tuple(rule.rule_id for rule in matches)
    return _evaluation_from_spec(
        selected.result,
        rule_id=selected.rule_id,
        priority=selected.priority,
        matched_rule_ids=matched_ids,
    )
