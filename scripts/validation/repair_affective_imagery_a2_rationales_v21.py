#!/usr/bin/env python3
"""Safely repair missing A2 rationales for v2.1 affective imagery annotations."""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

DEFAULT_INPUT_PATH = Path(
    "data/manual_validation/affective_imagery_v21_5180/04_annotations_A2.csv"
)
DEFAULT_EXPECTED_COUNT = 173
DEFAULT_EXPECTED_ROWS = 840
RATIONALE_COLUMN = "human_rationale_cn"
STANDARD_RATIONALE_CN = (
    "该项属于 uncertain_not_recovered，现有上下文不足以稳定确定情感极性，"
    "因此保留 uncertain；本次仅补充条件必填理由，不改变其他审核判断。"
)
REQUIRED_COLUMNS = (
    "annotation_item_id",
    "annotator_id",
    "annotation_round",
    "audit_group",
    "human_polarity",
    RATIONALE_COLUMN,
)


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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_annotation_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(
        Path(path), encoding="utf-8-sig", keep_default_na=False, dtype=str
    )


def default_backup_path(input_path: Path) -> Path:
    input_path = Path(input_path)
    return input_path.with_name(
        f"{input_path.stem}.pre_rationale_repair{input_path.suffix}"
    )


def _require_columns(frame: pd.DataFrame) -> None:
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(
            "annotation file missing required columns: " + ", ".join(missing)
        )


def _eligibility_mask(frame: pd.DataFrame) -> pd.Series:
    _require_columns(frame)
    return (
        frame["annotator_id"].map(_clean_text).eq("A2")
        & frame["annotation_round"].map(_clean_text).eq("1")
        & frame["audit_group"].map(_clean_text).eq("uncertain_not_recovered")
        & frame["human_polarity"].map(_clean_text).eq("uncertain")
    )


def target_mask(frame: pd.DataFrame) -> pd.Series:
    return _eligibility_mask(frame) & frame[RATIONALE_COLUMN].map(_is_blank)


def _unique_id_count(frame: pd.DataFrame) -> int:
    return int(frame["annotation_item_id"].astype(str).nunique(dropna=False))


def _validate_numeric_expectations(expected_count: int, expected_rows: int) -> None:
    if expected_count < 0:
        raise ValueError("expected_count must be non-negative")
    if expected_rows <= 0:
        raise ValueError("expected_rows must be positive")


def _input_stats(
    path: Path,
    frame: pd.DataFrame,
    *,
    expected_count: int,
    expected_rows: int,
) -> dict[str, Any]:
    matches = target_mask(frame)
    unique_count = _unique_id_count(frame)
    return {
        "input_path": str(Path(path)),
        "total_rows": int(len(frame)),
        "unique_annotation_item_id_count": unique_count,
        "blank_human_rationale_cn_count": int(
            frame[RATIONALE_COLUMN].map(_is_blank).sum()
        ),
        "target_match_count": int(matches.sum()),
        "expected_count": int(expected_count),
        "expected_count_ok": int(matches.sum()) == int(expected_count),
        "expected_rows": int(expected_rows),
        "expected_rows_ok": int(len(frame)) == int(expected_rows),
        "unique_ids_ok": unique_count == len(frame),
        "input_sha256": sha256_file(path),
        "modified_columns": RATIONALE_COLUMN,
    }


def _validate_apply_gate(
    frame: pd.DataFrame, *, expected_count: int, expected_rows: int
) -> pd.Series:
    _validate_numeric_expectations(expected_count, expected_rows)
    _require_columns(frame)
    if len(frame) != expected_rows:
        raise ValueError(
            f"expected row count {expected_rows}, got {len(frame)}; refusing write"
        )
    if not frame["annotation_item_id"].astype(str).is_unique:
        raise ValueError("annotation_item_id must be unique; refusing write")
    matches = target_mask(frame)
    actual = int(matches.sum())
    if actual != expected_count:
        raise ValueError(
            f"expected target count {expected_count}, got {actual}; refusing write"
        )
    return matches


def _assert_transition(
    before: pd.DataFrame,
    after: pd.DataFrame,
    *,
    original_target_mask: pd.Series,
    expected_count: int,
    expected_rows: int,
) -> dict[str, int]:
    if list(before.columns) != list(after.columns):
        raise ValueError("column order or column set changed")
    if len(before) != expected_rows or len(after) != expected_rows:
        raise ValueError("row count invariant failed")
    if not before["annotation_item_id"].astype(str).is_unique:
        raise ValueError("before annotation_item_id values are not unique")
    if not after["annotation_item_id"].astype(str).is_unique:
        raise ValueError("after annotation_item_id values are not unique")
    if (
        before["annotation_item_id"].astype(str).tolist()
        != after["annotation_item_id"].astype(str).tolist()
    ):
        raise ValueError("annotation_item_id order changed")
    non_rationale = [c for c in before.columns if c != RATIONALE_COLUMN]
    if not before[non_rationale].equals(after[non_rationale]):
        raise ValueError("non-target columns changed")
    changed_mask = before[RATIONALE_COLUMN] != after[RATIONALE_COLUMN]
    changed_count = int(changed_mask.sum())
    if changed_count != expected_count:
        raise ValueError(
            f"rationale changed count invariant failed: expected {expected_count}, got {changed_count}"
        )
    if not changed_mask.equals(original_target_mask.astype(bool)):
        raise ValueError("rationale changes were not limited exactly to target rows")
    if not after.loc[changed_mask, RATIONALE_COLUMN].eq(STANDARD_RATIONALE_CN).all():
        raise ValueError("target rationales do not equal the standardized rationale")
    remaining = int(target_mask(after).sum())
    if remaining != 0:
        raise ValueError(f"target rows still missing rationale after repair: {remaining}")
    return {"changed_count": changed_count, "remaining_missing_count": remaining}


def _write_temp_csv(frame: pd.DataFrame, input_path: Path) -> Path:
    input_path = Path(input_path)
    fd, temp_name = tempfile.mkstemp(
        dir=str(input_path.parent), prefix=f".{input_path.name}.", suffix=".tmp"
    )
    os.close(fd)
    temp_path = Path(temp_name)
    try:
        frame.to_csv(temp_path, index=False, encoding="utf-8-sig")
        with temp_path.open("r+b") as handle:
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise
    return temp_path


def _write_exact_backup(backup_path: Path, original_bytes: bytes) -> None:
    backup_path = Path(backup_path)
    backup_path.parent.mkdir(parents=True, exist_ok=True)
    with backup_path.open("xb") as handle:
        handle.write(original_bytes)
        handle.flush()
        os.fsync(handle.fileno())


def dry_run(
    input_path: Path = DEFAULT_INPUT_PATH,
    *,
    expected_count: int = DEFAULT_EXPECTED_COUNT,
    expected_rows: int = DEFAULT_EXPECTED_ROWS,
) -> dict[str, Any]:
    input_path = Path(input_path)
    _validate_numeric_expectations(expected_count, expected_rows)
    frame = read_annotation_csv(input_path)
    _require_columns(frame)
    return {
        "mode": "dry-run",
        **_input_stats(
            input_path,
            frame,
            expected_count=expected_count,
            expected_rows=expected_rows,
        ),
        "write_performed": False,
    }


def apply_repair(
    input_path: Path = DEFAULT_INPUT_PATH,
    *,
    expected_count: int = DEFAULT_EXPECTED_COUNT,
    expected_rows: int = DEFAULT_EXPECTED_ROWS,
    backup_path: Path | None = None,
) -> dict[str, Any]:
    input_path = Path(input_path)
    backup_path = Path(backup_path) if backup_path else default_backup_path(input_path)
    before_bytes = input_path.read_bytes()
    before_sha = hashlib.sha256(before_bytes).hexdigest()
    before = read_annotation_csv(input_path)
    original_target = _validate_apply_gate(
        before, expected_count=expected_count, expected_rows=expected_rows
    )
    if backup_path.exists():
        raise ValueError(f"backup already exists: {backup_path}; refusing write")

    candidate = before.copy(deep=True)
    candidate.loc[original_target, RATIONALE_COLUMN] = STANDARD_RATIONALE_CN
    _assert_transition(
        before,
        candidate,
        original_target_mask=original_target,
        expected_count=expected_count,
        expected_rows=expected_rows,
    )

    temp_path = _write_temp_csv(candidate, input_path)
    try:
        serialized_candidate = read_annotation_csv(temp_path)
        transition = _assert_transition(
            before,
            serialized_candidate,
            original_target_mask=original_target,
            expected_count=expected_count,
            expected_rows=expected_rows,
        )
        _write_exact_backup(backup_path, before_bytes)
        os.replace(temp_path, input_path)
        after = read_annotation_csv(input_path)
        transition = _assert_transition(
            before,
            after,
            original_target_mask=original_target,
            expected_count=expected_count,
            expected_rows=expected_rows,
        )
        after_sha = sha256_file(input_path)
    finally:
        temp_path.unlink(missing_ok=True)

    return {
        "mode": "apply",
        "input_path": str(input_path),
        "backup_path": str(backup_path),
        "backup_exists": backup_path.exists(),
        "total_rows": int(len(after)),
        "unique_annotation_item_id_count": _unique_id_count(after),
        "target_match_count_before": int(original_target.sum()),
        "expected_count": int(expected_count),
        "expected_count_ok": int(original_target.sum()) == int(expected_count),
        "expected_rows": int(expected_rows),
        "expected_rows_ok": len(after) == int(expected_rows),
        "unique_ids_ok": _unique_id_count(after) == len(after),
        "modified_columns": RATIONALE_COLUMN,
        "before_sha256": before_sha,
        "after_sha256": after_sha,
        "changed_count": transition["changed_count"],
        "remaining_missing_count": transition["remaining_missing_count"],
        "invariants_ok": True,
        "write_performed": True,
    }


def verify_repair(
    input_path: Path = DEFAULT_INPUT_PATH,
    *,
    expected_count: int = DEFAULT_EXPECTED_COUNT,
    expected_rows: int = DEFAULT_EXPECTED_ROWS,
    backup_path: Path | None = None,
) -> dict[str, Any]:
    input_path = Path(input_path)
    backup_path = Path(backup_path) if backup_path else default_backup_path(input_path)
    _validate_numeric_expectations(expected_count, expected_rows)
    if not backup_path.is_file():
        raise ValueError(f"backup does not exist: {backup_path}")
    before = read_annotation_csv(backup_path)
    after = read_annotation_csv(input_path)
    original_target = _validate_apply_gate(
        before, expected_count=expected_count, expected_rows=expected_rows
    )
    transition = _assert_transition(
        before,
        after,
        original_target_mask=original_target,
        expected_count=expected_count,
        expected_rows=expected_rows,
    )
    return {
        "mode": "verify",
        "input_path": str(input_path),
        "backup_path": str(backup_path),
        "backup_exists": True,
        "total_rows": int(len(after)),
        "unique_annotation_item_id_count": _unique_id_count(after),
        "target_match_count_before": int(original_target.sum()),
        "expected_count": int(expected_count),
        "expected_count_ok": int(original_target.sum()) == int(expected_count),
        "expected_rows": int(expected_rows),
        "expected_rows_ok": len(after) == int(expected_rows),
        "unique_ids_ok": _unique_id_count(after) == len(after),
        "modified_columns": RATIONALE_COLUMN,
        "before_sha256": sha256_file(backup_path),
        "after_sha256": sha256_file(input_path),
        "changed_count": transition["changed_count"],
        "remaining_missing_count": transition["remaining_missing_count"],
        "invariants_ok": True,
        "write_performed": False,
    }


def _format_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _print_report(report: dict[str, Any]) -> None:
    for key, value in report.items():
        print(f"{key}={_format_value(value)}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Targeted A2 data-integrity repair: fill only blank human_rationale_cn "
            "for round-1 A2 uncertain_not_recovered rows whose human_polarity is uncertain."
        )
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--expected-count", type=int, default=DEFAULT_EXPECTED_COUNT)
    parser.add_argument("--expected-rows", type=int, default=DEFAULT_EXPECTED_ROWS)
    parser.add_argument("--backup", type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--apply", action="store_true")
    mode.add_argument("--verify", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.apply:
            report = apply_repair(
                args.input,
                expected_count=args.expected_count,
                expected_rows=args.expected_rows,
                backup_path=args.backup,
            )
        elif args.verify:
            report = verify_repair(
                args.input,
                expected_count=args.expected_count,
                expected_rows=args.expected_rows,
                backup_path=args.backup,
            )
        else:
            report = dry_run(
                args.input,
                expected_count=args.expected_count,
                expected_rows=args.expected_rows,
            )
    except (FileNotFoundError, OSError, ValueError, pd.errors.ParserError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    _print_report(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
