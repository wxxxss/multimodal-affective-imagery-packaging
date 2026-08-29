#!/usr/bin/env python3
"""Safely normalize v2.1 sentence action/error fields from an explicit contract."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any

import pandas as pd

try:
    from .affective_imagery_action_error_mapping_v21 import (
        MappingContract,
        adapt_row_inputs,
        evaluate_mapping,
        load_mapping,
    )
    from .affective_imagery_annotation_policy_v21 import rationale_required
    from .affective_imagery_decision_sidecar_v21 import (
        DECISION_SIDECAR_COLUMNS,
        decision_sidecar_provenance_path,
        derive_decision_sidecar,
        load_canonical_mapping,
        verify_derived_sidecar,
        verify_derived_sidecar_with_provenance,
        write_csv_new_atomic,
    )
    from .affective_imagery_validation_provenance_v21 import (
        build_provenance,
        load_provenance,
        read_git_state,
        sha256_file,
        verify_provenance_common,
        write_json_new_atomic,
    )
    from .prepare_affective_imagery_validation_v21 import (
        _clean_text,
        _is_blank,
        read_csv,
        write_csv,
    )
    from .validate_affective_imagery_annotations_v21 import (
        DIMENSION_VALUES,
        ERROR_TYPES,
        HUMAN_ACTIONS,
        POLARITIES,
        validate_annotation_frame,
        validate_annotation_mapping_consistency,
    )
except ImportError:  # pragma: no cover - direct script execution
    from affective_imagery_action_error_mapping_v21 import (
        MappingContract,
        adapt_row_inputs,
        evaluate_mapping,
        load_mapping,
    )
    from affective_imagery_annotation_policy_v21 import rationale_required
    from affective_imagery_decision_sidecar_v21 import (
        DECISION_SIDECAR_COLUMNS,
        decision_sidecar_provenance_path,
        derive_decision_sidecar,
        load_canonical_mapping,
        verify_derived_sidecar,
        verify_derived_sidecar_with_provenance,
        write_csv_new_atomic,
    )
    from affective_imagery_validation_provenance_v21 import (
        build_provenance,
        load_provenance,
        read_git_state,
        sha256_file,
        verify_provenance_common,
        write_json_new_atomic,
    )
    from prepare_affective_imagery_validation_v21 import (
        _clean_text,
        _is_blank,
        read_csv,
        write_csv,
    )
    from validate_affective_imagery_annotations_v21 import (
        DIMENSION_VALUES,
        ERROR_TYPES,
        HUMAN_ACTIONS,
        POLARITIES,
        validate_annotation_frame,
        validate_annotation_mapping_consistency,
    )


TOOL_NAME = "normalize_affective_imagery_actions_v21"
TOOL_VERSION = "affective_imagery_action_error_normalizer_v21.2"
ALLOWED_CHANGED_FIELDS = ("human_action", "human_error_type")
KEY_FIELDS = ("annotation_item_id", "annotator_id", "annotation_round")


@dataclass(frozen=True)
class NormalizationResult:
    frame: pd.DataFrame
    row_count: int
    target_sentence_row_count: int
    action_change_count: int
    error_change_count: int
    unresolved: list[dict[str, str]]
    newly_required_rationale_missing_ids: list[str]


def provenance_path_for_output(output_path: Path) -> Path:
    output = Path(output_path)
    return output.with_name(f"{output.name}.provenance.json")


def _merged_row(annotation: pd.Series, item: pd.Series) -> pd.Series:
    merged = annotation.copy()
    for column in item.index:
        if column not in merged.index:
            merged[column] = item[column]
    return merged


def normalize_frame(
    raw: pd.DataFrame,
    contract: MappingContract,
) -> NormalizationResult:
    """Propose contract-derived sentence changes without inventing fallbacks."""
    frame = raw.copy(deep=True)
    target_sentence_row_count = 0
    action_change_count = 0
    error_change_count = 0
    unresolved: list[dict[str, str]] = []
    newly_required: list[str] = []

    for index, before in raw.iterrows():
        if _clean_text(before.get("item_type")) != "sentence":
            continue
        target_sentence_row_count += 1
        item_id = _clean_text(before.get("annotation_item_id"))
        inputs = adapt_row_inputs(before, contract)
        evaluation = evaluate_mapping(contract, inputs)
        if evaluation.status == "unresolved":
            unresolved.append({
                "annotation_item_id": item_id,
                "reason_code": evaluation.reason_code or "",
            })
            continue

        before_requires = rationale_required(before, "sentence")
        if _clean_text(before.get("human_action")) != evaluation.action:
            action_change_count += 1
        if _clean_text(before.get("human_error_type")) != evaluation.error_type:
            error_change_count += 1
        frame.at[index, "human_action"] = evaluation.action
        frame.at[index, "human_error_type"] = evaluation.error_type
        after = frame.loc[index]
        after_requires = rationale_required(after, "sentence")
        if (
            after_requires
            and not before_requires
            and _is_blank(after.get("human_rationale_cn"))
        ):
            newly_required.append(item_id)

    return NormalizationResult(
        frame=frame,
        row_count=len(frame),
        target_sentence_row_count=target_sentence_row_count,
        action_change_count=action_change_count,
        error_change_count=error_change_count,
        unresolved=unresolved,
        newly_required_rationale_missing_ids=newly_required,
    )


def _clean_column(frame: pd.DataFrame, column: str) -> list[str]:
    return [_clean_text(value) for value in frame[column].tolist()]


def _verify_structural_invariants(
    raw: pd.DataFrame,
    normalized: pd.DataFrame,
) -> None:
    if list(raw.columns) != list(normalized.columns):
        raise ValueError("normalized columns or column order changed")
    if len(raw) != len(normalized):
        raise ValueError("normalized row count changed")
    for column in KEY_FIELDS:
        if column not in raw.columns:
            raise ValueError(f"raw input missing key field: {column}")
        if _clean_column(raw, column) != _clean_column(normalized, column):
            raise ValueError(f"normalized key/order changed: {column}")

    non_target_fields = [
        column for column in raw.columns if column not in ALLOWED_CHANGED_FIELDS
    ]
    for column in non_target_fields:
        if _clean_column(raw, column) != _clean_column(normalized, column):
            raise ValueError(f"non-target field changed: {column}")

    for index, before in raw.iterrows():
        if _clean_text(before.get("item_type")) == "sentence":
            continue
        for column in ALLOWED_CHANGED_FIELDS:
            if _clean_text(before.get(column)) != _clean_text(normalized.at[index, column]):
                raise ValueError(
                    f"product-dimension target field changed: {column}"
                )


def verify_normalized_frame(
    raw: pd.DataFrame,
    normalized: pd.DataFrame,
    contract: MappingContract,
    annotator_id: str,
) -> None:
    """Run canonical, mapping, and row-level invariant verification."""
    validate_annotation_frame(raw, raw, expected_annotator_id=annotator_id)
    _verify_structural_invariants(raw, normalized)
    validate_annotation_frame(
        normalized,
        raw,
        expected_annotator_id=annotator_id,
    )
    validate_annotation_mapping_consistency(normalized, raw, contract)


def _load_contract(mapping_path: Path) -> MappingContract:
    return load_canonical_mapping(
        mapping_path,
        actions=HUMAN_ACTIONS,
        error_types=ERROR_TYPES,
        dimension_values=DIMENSION_VALUES,
        polarities=POLARITIES,
    )


def _write_frame_new_atomic(output_path: Path, frame: pd.DataFrame) -> None:
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


def _fail_closed_before_write(result: NormalizationResult) -> None:
    if result.unresolved:
        raise ValueError(
            f"unresolved mapping results: {len(result.unresolved)}"
        )
    if result.newly_required_rationale_missing_ids:
        raise ValueError(
            "rationale newly required but missing: "
            + ", ".join(result.newly_required_rationale_missing_ids)
        )


def _cleanup_owned_output(
    output_path: Path,
    *,
    output_did_not_exist_before: bool,
    invocation_created_output: bool,
    invocation_output_sha: str | None,
) -> None:
    if not (
        output_did_not_exist_before
        and invocation_created_output
        and invocation_output_sha is not None
        and output_path.is_file()
    ):
        return
    try:
        current_sha = sha256_file(output_path)
    except OSError:
        return
    if current_sha == invocation_output_sha:
        output_path.unlink()


def apply_normalization(
    *,
    input_path: Path,
    output_path: Path,
    mapping_path: Path,
    annotator_id: str,
    repo_root: Path | None = None,
) -> NormalizationResult:
    """Apply normalization to a new output and publish provenance last."""
    input_file = Path(input_path)
    output_file = Path(output_path)
    mapping_file = Path(mapping_path)
    sidecar_file = provenance_path_for_output(output_file)
    output_did_not_exist_before = not output_file.exists()
    if not output_did_not_exist_before:
        raise FileExistsError(f"output already exists: {output_file}")
    if sidecar_file.exists():
        raise FileExistsError(f"sidecar already exists: {sidecar_file}")

    input_sha_before = sha256_file(input_file)
    raw = read_csv(input_file)
    contract = _load_contract(mapping_file)
    validate_annotation_frame(raw, raw, expected_annotator_id=annotator_id)
    result = normalize_frame(raw, contract)
    _fail_closed_before_write(result)
    verify_normalized_frame(raw, result.frame, contract, annotator_id)

    invocation_created_output = False
    invocation_output_sha: str | None = None
    try:
        _write_frame_new_atomic(output_file, result.frame)
        invocation_created_output = True
        reread = read_csv(output_file)
        verify_normalized_frame(raw, reread, contract, annotator_id)
        if sha256_file(input_file) != input_sha_before:
            raise ValueError("input SHA-256 changed during normalization")
        invocation_output_sha = sha256_file(output_file)

        git_root = Path(repo_root) if repo_root is not None else Path(__file__).parents[2]
        provenance = build_provenance(
            mapping_metadata=contract.metadata,
            tool_name=TOOL_NAME,
            tool_version=TOOL_VERSION,
            git_state=read_git_state(git_root),
            input_files={"annotation_input": input_file},
            output_logical_name="normalized_annotations",
            output_path=output_file,
            row_count=len(reread),
            annotator_id=annotator_id,
            mode="apply",
            unresolved_count=0,
            allowed_changed_fields=ALLOWED_CHANGED_FIELDS,
        )
        write_json_new_atomic(sidecar_file, provenance)
    except Exception:
        _cleanup_owned_output(
            output_file,
            output_did_not_exist_before=output_did_not_exist_before,
            invocation_created_output=invocation_created_output,
            invocation_output_sha=invocation_output_sha,
        )
        raise
    return NormalizationResult(
        frame=reread,
        row_count=result.row_count,
        target_sentence_row_count=result.target_sentence_row_count,
        action_change_count=result.action_change_count,
        error_change_count=result.error_change_count,
        unresolved=result.unresolved,
        newly_required_rationale_missing_ids=(
            result.newly_required_rationale_missing_ids
        ),
    )


def verify_artifacts(
    *,
    input_path: Path,
    output_path: Path,
    mapping_path: Path,
    annotator_id: str,
) -> None:
    input_file = Path(input_path)
    output_file = Path(output_path)
    mapping_file = Path(mapping_path)
    sidecar_file = provenance_path_for_output(output_file)
    raw = read_csv(input_file)
    normalized = read_csv(output_file)
    contract = _load_contract(mapping_file)
    verify_normalized_frame(raw, normalized, contract, annotator_id)
    provenance = load_provenance(sidecar_file)
    verify_provenance_common(
        provenance,
        mapping_metadata=contract.metadata,
        tool_version=TOOL_VERSION,
        input_files={"annotation_input": input_file},
        output_path=output_file,
        row_count=len(normalized),
        annotator_id=annotator_id,
        allowed_changed_fields=ALLOWED_CHANGED_FIELDS,
    )
    if provenance.get("tool_name") != TOOL_NAME:
        raise ValueError("provenance tool_name mismatch")
    if provenance.get("mode") != "apply":
        raise ValueError("provenance mode mismatch")
    if provenance.get("unresolved_count") != 0:
        raise ValueError("provenance unresolved_count mismatch")


def _summary(result: NormalizationResult) -> dict[str, Any]:
    reasons = Counter(item["reason_code"] for item in result.unresolved)
    return {
        "row_count": result.row_count,
        "target_sentence_row_count": result.target_sentence_row_count,
        "action_change_count": result.action_change_count,
        "error_change_count": result.error_change_count,
        "unresolved_count": len(result.unresolved),
        "unresolved_reason_codes": dict(sorted(reasons.items())),
        "unresolved": result.unresolved,
        "newly_required_rationale_missing_count": len(
            result.newly_required_rationale_missing_ids
        ),
        "newly_required_rationale_missing_ids": (
            result.newly_required_rationale_missing_ids
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Normalize sentence action/error fields from an explicit mapping."
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--mapping", required=True, type=Path)
    parser.add_argument("--annotator-id", required=True)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--dry-run", action="store_true")
    modes.add_argument("--apply", action="store_true")
    modes.add_argument("--verify", action="store_true")
    modes.add_argument("--derive", action="store_true")
    modes.add_argument("--verify-derived", action="store_true")
    return parser.parse_args()


def _derive_sidecar_path(output_path: Path) -> Path:
    """In derive mode --output is the decision sidecar CSV path itself."""
    return Path(output_path)


def _cleanup_owned_sidecar(
    sidecar_csv: Path,
    *,
    sidecar_did_not_exist_before: bool,
    invocation_created_sidecar: bool,
    invocation_sidecar_sha: str | None,
) -> None:
    if not (
        sidecar_did_not_exist_before
        and invocation_created_sidecar
        and invocation_sidecar_sha is not None
        and sidecar_csv.is_file()
    ):
        return
    try:
        current_sha = sha256_file(sidecar_csv)
    except OSError:
        return
    if current_sha == invocation_sidecar_sha:
        sidecar_csv.unlink()


def _run_derive(
    *,
    input_path: Path,
    output_path: Path,
    mapping_path: Path,
    annotator_id: str,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    input_file = Path(input_path)
    mapping_file = Path(mapping_path)
    sidecar_csv = _derive_sidecar_path(output_path)
    provenance_json = decision_sidecar_provenance_path(sidecar_csv)
    sidecar_did_not_exist_before = not sidecar_csv.exists()
    if not sidecar_did_not_exist_before:
        raise FileExistsError(f"decision sidecar already exists: {sidecar_csv}")
    if provenance_json.exists():
        raise FileExistsError(f"sidecar provenance already exists: {provenance_json}")

    input_sha_before = sha256_file(input_file)
    raw = read_csv(input_file)
    validate_annotation_frame(raw, raw, expected_annotator_id=annotator_id)
    contract = _load_contract(mapping_file)
    sidecar = derive_decision_sidecar(raw, contract, annotator_id=annotator_id)
    verify_derived_sidecar(
        raw=raw, sidecar=sidecar, contract=contract, annotator_id=annotator_id
    )

    invocation_created_sidecar = False
    invocation_sidecar_sha: str | None = None
    try:
        write_csv_new_atomic(sidecar_csv, sidecar)
        invocation_created_sidecar = True
        reread = read_csv(sidecar_csv)
        verify_derived_sidecar(
            raw=raw, sidecar=reread, contract=contract, annotator_id=annotator_id
        )
        if sha256_file(input_file) != input_sha_before:
            raise ValueError("input SHA-256 changed during derive")
        invocation_sidecar_sha = sha256_file(sidecar_csv)

        resolved_count = int((reread["mapping_status"] == "resolved").sum())
        unresolved_count = int((reread["mapping_status"] == "unresolved").sum())
        reasons = Counter(
            reread.loc[reread["mapping_status"] == "unresolved", "reason_code"].tolist()
        )
        git_root = Path(repo_root) if repo_root is not None else Path(__file__).parents[2]
        provenance = build_provenance(
            mapping_metadata=contract.metadata,
            tool_name=TOOL_NAME,
            tool_version=TOOL_VERSION,
            git_state=read_git_state(git_root),
            input_files={"annotation_input": input_file},
            output_logical_name="decision_sidecar",
            output_path=sidecar_csv,
            row_count=len(reread),
            annotator_id=annotator_id,
            mode="derive",
            unresolved_count=unresolved_count,
            allowed_changed_fields=[],
        )
        provenance.update({
            "resolved_count": resolved_count,
            "unresolved_count": unresolved_count,
            "reason_code_distribution": dict(sorted(reasons.items())),
            "model_assisted": True,
            "human_gold": False,
        })
        write_json_new_atomic(provenance_json, provenance)
    except Exception:
        _cleanup_owned_sidecar(
            sidecar_csv,
            sidecar_did_not_exist_before=sidecar_did_not_exist_before,
            invocation_created_sidecar=invocation_created_sidecar,
            invocation_sidecar_sha=invocation_sidecar_sha,
        )
        raise
    return {
        "row_count": len(reread),
        "resolved_count": resolved_count,
        "unresolved_count": unresolved_count,
        "unresolved_reason_codes": dict(sorted(reasons.items())),
        "decision_sidecar": str(sidecar_csv.resolve()),
        "provenance": str(provenance_json.resolve()),
    }


def _run_verify_derived(
    *,
    input_path: Path,
    output_path: Path,
    mapping_path: Path,
    annotator_id: str,
) -> dict[str, Any]:
    input_file = Path(input_path)
    mapping_file = Path(mapping_path)
    sidecar_csv = _derive_sidecar_path(output_path)
    provenance_json = decision_sidecar_provenance_path(sidecar_csv)
    if not sidecar_csv.is_file():
        raise FileNotFoundError(f"decision sidecar missing: {sidecar_csv}")
    if not provenance_json.is_file():
        raise FileNotFoundError(f"sidecar provenance missing: {provenance_json}")

    contract = _load_contract(mapping_file)
    verify_derived_sidecar_with_provenance(
        raw_path=input_file,
        sidecar_path=sidecar_csv,
        provenance_path=provenance_json,
        contract=contract,
        annotator_id=annotator_id,
        tool_name=TOOL_NAME,
        tool_version=TOOL_VERSION,
    )
    sidecar = read_csv(sidecar_csv)
    unresolved_count = int((sidecar["mapping_status"] == "unresolved").sum())
    return {
        "row_count": len(sidecar),
        "resolved_count": int((sidecar["mapping_status"] == "resolved").sum()),
        "unresolved_count": unresolved_count,
        "verification": "PASS",
    }


def main() -> int:
    args = parse_args()
    try:
        if args.dry_run:
            raw = read_csv(args.input)
            contract = _load_contract(args.mapping)
            validate_annotation_frame(
                raw,
                raw,
                expected_annotator_id=args.annotator_id,
            )
            result = normalize_frame(raw, contract)
            _verify_structural_invariants(raw, result.frame)
            if not result.unresolved and not result.newly_required_rationale_missing_ids:
                verify_normalized_frame(raw, result.frame, contract, args.annotator_id)
            summary = _summary(result)
            summary.update({
                "mapping_version": contract.metadata.mapping_version,
                "contract_schema_version": contract.metadata.contract_schema_version,
                "mapping_sha256": contract.metadata.mapping_sha256,
                "input_sha256": sha256_file(args.input),
            })
            print(json.dumps(summary, ensure_ascii=False, indent=2))
            return 0
        if args.apply:
            result = apply_normalization(
                input_path=args.input,
                output_path=args.output,
                mapping_path=args.mapping,
                annotator_id=args.annotator_id,
            )
            print(json.dumps(_summary(result), ensure_ascii=False, indent=2))
            return 0
        if args.derive:
            summary = _run_derive(
                input_path=args.input,
                output_path=args.output,
                mapping_path=args.mapping,
                annotator_id=args.annotator_id,
            )
            print(json.dumps(summary, ensure_ascii=False, indent=2))
            return 0
        if args.verify_derived:
            summary = _run_verify_derived(
                input_path=args.input,
                output_path=args.output,
                mapping_path=args.mapping,
                annotator_id=args.annotator_id,
            )
            print(json.dumps(summary, ensure_ascii=False, indent=2))
            return 0
        verify_artifacts(
            input_path=args.input,
            output_path=args.output,
            mapping_path=args.mapping,
            annotator_id=args.annotator_id,
        )
        print("Verification: PASS")
        return 0
    except (FileNotFoundError, FileExistsError, OSError, RuntimeError, ValueError) as exc:
        print(f"Normalization failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
