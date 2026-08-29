#!/usr/bin/env python3
"""Shared decision-sidecar derivation and verification for v2.1 validation.

Single home for:

- DECISION_SIDECAR_COLUMNS
- derive_decision_sidecar() (mapping provenance recorded for every row)
- verify_derived_sidecar() / verify_derived_sidecar_with_provenance()
- canonical contract loading with EXACT vocabulary equality
- rationale-gap semantics (resolved rows only; unresolved are adjudication)
- atomic new-output-only CSV publication

This module imports only neutral helpers (mapping contract, provenance,
prepare core) so both validate_affective_imagery_annotations_v21 and
normalize_affective_imagery_actions_v21 can import it without circularity.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping

import pandas as pd

try:
    from .affective_imagery_action_error_mapping_v21 import (
        MappingContract,
        MappingContractError,
        adapt_row_inputs,
        evaluate_mapping,
        load_mapping,
    )
    from .affective_imagery_annotation_policy_v21 import rationale_required
    from .affective_imagery_validation_provenance_v21 import (
        load_provenance,
        sha256_file,
        verify_provenance_common,
    )
    from .prepare_affective_imagery_validation_v21 import (
        _clean_text,
        _is_blank,
        read_csv,
    )
except ImportError:  # pragma: no cover - direct script execution
    from affective_imagery_action_error_mapping_v21 import (
        MappingContract,
        MappingContractError,
        adapt_row_inputs,
        evaluate_mapping,
        load_mapping,
    )
    from affective_imagery_annotation_policy_v21 import rationale_required
    from affective_imagery_validation_provenance_v21 import (
        load_provenance,
        sha256_file,
        verify_provenance_common,
    )
    from prepare_affective_imagery_validation_v21 import (
        _clean_text,
        _is_blank,
        read_csv,
    )

DECISION_SIDECAR_COLUMNS = [
    "annotation_item_id",
    "annotator_id",
    "annotation_round",
    "mapping_status",
    "derived_human_action",
    "derived_human_error_type",
    "mapping_requires_adjudication",
    "reason_code",
    "selected_rule_id",
    "matched_rule_ids",
    "mapping_version",
    "mapping_sha256",
    "rationale_required_after_derivation",
    "rationale_missing_after_derivation",
    "requires_followup",
]

SIDECAR_PROVENANCE_SUFFIX = ".provenance.json"


def decision_sidecar_provenance_path(sidecar_path: Path) -> Path:
    """Companion provenance path for a decision sidecar CSV."""
    return Path(sidecar_path).with_name(
        f"{Path(sidecar_path).name}{SIDECAR_PROVENANCE_SUFFIX}"
    )


def load_canonical_mapping(
    path: Path,
    *,
    actions: set[str],
    error_types: set[str],
    dimension_values: set[str],
    polarities: set[str],
) -> MappingContract:
    """Load the production mapping and REQUIRE exact canonical vocabulary.

    Unlike load_mapping's subset checks, the production loader requires:
      declared dimensions == DIMENSION_VALUES
      declared human polarity == POLARITIES
    """
    contract = load_mapping(
        path,
        schema_action_enum=actions,
        schema_error_type_enum=error_types,
        schema_dimension_values=dimension_values,
        schema_polarity_values=polarities,
    )
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    declared = raw["declared_vocabulary"]
    declared_dimensions = set(_clean_text(item) for item in declared["dimensions"])
    declared_polarities = set(
        _clean_text(item) for item in declared["human_polarity"]
    )
    if declared_dimensions != set(dimension_values):
        raise MappingContractError(
            "declared dimension vocabulary must EXACTLY equal canonical "
            f"DIMENSION_VALUES; extra={sorted(declared_dimensions - set(dimension_values))} "
            f"missing={sorted(set(dimension_values) - declared_dimensions)}"
        )
    if declared_polarities != set(polarities):
        raise MappingContractError(
            "declared human polarity vocabulary must EXACTLY equal canonical "
            f"POLARITIES; extra={sorted(declared_polarities - set(polarities))} "
            f"missing={sorted(set(polarities) - declared_polarities)}"
        )
    return contract


def derive_decision_sidecar(
    raw: pd.DataFrame,
    contract: MappingContract,
    *,
    annotator_id: str,
) -> pd.DataFrame:
    """Derive per-row mapping decisions without modifying raw A1/A2.

    Every row produces exactly one sidecar row.

    - resolved: derived action/error populated; selected_rule_id and
      matched_rule_ids are recorded for provenance.
    - unresolved: derived action/error blank; mapping_requires_adjudication=1;
      rationale flags stay 0 (blank action/error is NOT pseudo-canonical).
    - product_dimension rows: not_applicable.
    """
    rows: list[dict[str, Any]] = []
    for index, before in raw.iterrows():
        item_id = _clean_text(before.get("annotation_item_id"))
        item_round = _clean_text(before.get("annotation_round", "1"))
        if _clean_text(before.get("item_type")) != "sentence":
            rows.append({
                "annotation_item_id": item_id,
                "annotator_id": annotator_id,
                "annotation_round": item_round,
                "mapping_status": "not_applicable",
                "derived_human_action": "",
                "derived_human_error_type": "",
                "mapping_requires_adjudication": 0,
                "reason_code": "product_dimension_action_error_not_applicable",
                "selected_rule_id": "",
                "matched_rule_ids": "",
                "mapping_version": contract.metadata.mapping_version,
                "mapping_sha256": contract.metadata.mapping_sha256,
                "rationale_required_after_derivation": 0,
                "rationale_missing_after_derivation": 0,
                "requires_followup": 0,
            })
            continue
        inputs = adapt_row_inputs(before, contract)
        evaluation = evaluate_mapping(contract, inputs)
        derived_action = ""
        derived_error = ""
        requires_adjudication = 0
        reason_code = ""
        selected_rule_id = evaluation.rule_id or ""
        matched_rule_ids = "|".join(evaluation.matched_rule_ids)
        rationale_required_flag = 0
        rationale_missing_flag = 0
        if evaluation.status == "resolved":
            derived_action = evaluation.action or ""
            derived_error = evaluation.error_type or ""
            before_requires = rationale_required(before, "sentence")
            after = before.copy()
            after["human_action"] = derived_action
            after["human_error_type"] = derived_error
            after_requires = rationale_required(after, "sentence")
            rationale_required_flag = int(after_requires)
            rationale_missing_flag = int(
                after_requires
                and not before_requires
                and _is_blank(after.get("human_rationale_cn"))
            )
        else:
            requires_adjudication = 1
            reason_code = evaluation.reason_code or ""
        rows.append({
            "annotation_item_id": item_id,
            "annotator_id": annotator_id,
            "annotation_round": item_round,
            "mapping_status": evaluation.status,
            "derived_human_action": derived_action,
            "derived_human_error_type": derived_error,
            "mapping_requires_adjudication": requires_adjudication,
            "reason_code": reason_code,
            "selected_rule_id": selected_rule_id,
            "matched_rule_ids": matched_rule_ids,
            "mapping_version": contract.metadata.mapping_version,
            "mapping_sha256": contract.metadata.mapping_sha256,
            "rationale_required_after_derivation": rationale_required_flag,
            "rationale_missing_after_derivation": rationale_missing_flag,
            "requires_followup": int(requires_adjudication or rationale_missing_flag),
        })
    return pd.DataFrame(rows, columns=DECISION_SIDECAR_COLUMNS)


def verify_derived_sidecar(
    *,
    raw: pd.DataFrame,
    sidecar: pd.DataFrame,
    contract: MappingContract,
    annotator_id: str,
) -> None:
    """Recompute the sidecar from raw input and compare row-by-row."""
    recomputed = derive_decision_sidecar(raw, contract, annotator_id=annotator_id)
    if list(recomputed.columns) != list(sidecar.columns):
        raise ValueError("derived sidecar columns changed")
    if len(recomputed) != len(sidecar):
        raise ValueError("derived sidecar row count changed")
    for column in recomputed.columns:
        if (
            recomputed[column].map(_clean_text).tolist()
            != sidecar[column].map(_clean_text).tolist()
        ):
            raise ValueError(f"derived sidecar mismatch in column: {column}")


def verify_derived_sidecar_with_provenance(
    *,
    raw_path: Path,
    sidecar_path: Path,
    provenance_path: Path,
    contract: MappingContract,
    annotator_id: str,
    tool_name: str,
    tool_version: str,
) -> None:
    """Full artifact verification: sidecar recomputation + provenance checks.

    Fail closed on tampered sidecar, stale input SHA, wrong mapping SHA/version,
    wrong annotator, missing/extra/duplicate IDs, missing provenance, or
    sidecar/provenance mismatch.
    """
    raw = read_csv(raw_path)
    sidecar = read_csv(sidecar_path)
    verify_derived_sidecar(
        raw=raw, sidecar=sidecar, contract=contract, annotator_id=annotator_id
    )
    sidecar_ids = sidecar["annotation_item_id"].map(_clean_text)
    if sidecar_ids.duplicated().any():
        raise ValueError("derived sidecar contains duplicate annotation_item_id")
    provenance = load_provenance(provenance_path)
    verify_provenance_common(
        provenance,
        mapping_metadata=contract.metadata,
        tool_version=tool_version,
        input_files={"annotation_input": raw_path},
        output_path=sidecar_path,
        row_count=len(sidecar),
        annotator_id=annotator_id,
        allowed_changed_fields=[],
    )
    if provenance.get("tool_name") != tool_name:
        raise ValueError("provenance tool_name mismatch")
    if provenance.get("mode") != "derive":
        raise ValueError("provenance mode mismatch")
    # Full statistical provenance: resolved_count / unresolved_count /
    # reason_code_distribution must match the recomputed sidecar exactly.
    statuses = sidecar["mapping_status"].map(_clean_text)
    actual_resolved = int((statuses == "resolved").sum())
    actual_unresolved = int((statuses == "unresolved").sum())
    if provenance.get("resolved_count") != actual_resolved:
        raise ValueError("provenance resolved_count mismatch")
    if provenance.get("unresolved_count") != actual_unresolved:
        raise ValueError("provenance unresolved_count mismatch")
    reason_counter: dict[str, int] = {}
    for reason in sidecar.loc[statuses == "unresolved", "reason_code"].map(_clean_text):
        if reason:
            reason_counter[reason] = reason_counter.get(reason, 0) + 1
    actual_reason_distribution = dict(sorted(reason_counter.items()))
    declared_reason_distribution = provenance.get("reason_code_distribution")
    if declared_reason_distribution != actual_reason_distribution:
        raise ValueError("provenance reason_code_distribution mismatch")


def write_csv_new_atomic(output_path: Path, frame: pd.DataFrame) -> None:
    """Atomically publish a new CSV without overwriting an existing path."""
    output = Path(output_path)
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    if not output.parent.is_dir():
        raise FileNotFoundError(
            f"output parent directory does not exist: {output.parent}"
        )
    descriptor, temp_name = tempfile.mkstemp(
        dir=output.parent,
        prefix=f".{output.name}.",
        suffix=".tmp",
    )
    temp_path = Path(temp_name)
    try:
        with os.fdopen(
            descriptor,
            "w",
            encoding="utf-8-sig",
            newline="",
        ) as handle:
            frame.to_csv(handle, index=False)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temp_path, output)
        except FileExistsError as exc:
            raise FileExistsError(f"output already exists: {output}") from exc
    finally:
        temp_path.unlink(missing_ok=True)
