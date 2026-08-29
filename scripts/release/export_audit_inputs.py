"""Export the minimum publication-safe frozen inputs needed for the AUROC audit.

The source files remain in the private/local research workspace. This exporter
copies only identifiers, split/group fields, the three modeled PU outcomes, and
frozen model scores. It deliberately excludes review text, image URLs/paths,
image binaries, embeddings, and other internal administrative fields.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Sequence

MANIFEST_COLUMNS = (
    "parent_asin",
    "main_analysis_included",
    "split_partition",
    "primary_response_sha256",
    "has_any_outer_imagery_observed",
    "general_visual_appeal_observed_positive_core",
    "cute_friendly_observed_positive_core",
)
PREDICTION_COLUMNS = ("parent_asin", "outcome", "track", "model_id", "score")
OUTCOMES = MANIFEST_COLUMNS[-3:]
TRACKS = ("openclip_512_logistic", "interpretable_36_logistic")
EXPECTED_MODEL_IDS = {f"{outcome}__{track}" for outcome in OUTCOMES for track in TRACKS}


class ExportError(RuntimeError):
    pass


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _require_columns(rows: list[dict[str, str]], columns: Sequence[str], label: str) -> None:
    if not rows:
        raise ExportError(f"{label} is empty")
    missing = [column for column in columns if column not in rows[0]]
    if missing:
        raise ExportError(f"{label} missing required columns: {missing}")


def _write_selected(path: Path, rows: list[dict[str, str]], columns: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row[column] for column in columns})


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def export_audit_inputs(
    manifest_path: Path,
    predictions_path: Path,
    output_dir: Path,
    *,
    require_frozen_counts: bool = True,
) -> dict[str, object]:
    manifest_rows = _read_csv(manifest_path)
    prediction_rows = _read_csv(predictions_path)
    _require_columns(manifest_rows, MANIFEST_COLUMNS, "modeling manifest")
    _require_columns(prediction_rows, PREDICTION_COLUMNS, "held-out predictions")

    parent_ids = [row["parent_asin"].strip() for row in manifest_rows]
    if any(not value for value in parent_ids) or len(parent_ids) != len(set(parent_ids)):
        raise ExportError("modeling manifest parent_asin values must be unique and non-empty")

    modeling_rows = [
        row for row in manifest_rows
        if row["main_analysis_included"].strip().lower() == "true"
    ]
    for row in modeling_rows:
        for outcome in OUTCOMES:
            if row[outcome].strip() not in {"0", "1"}:
                raise ExportError(f"invalid PU outcome value in {outcome}")
        if not row["primary_response_sha256"].strip():
            raise ExportError("missing primary_response_sha256")

    model_ids = {row["model_id"].strip() for row in prediction_rows}
    if require_frozen_counts and model_ids != EXPECTED_MODEL_IDS:
        raise ExportError("prediction model universe does not match the six frozen models")
    prediction_keys = [(row["parent_asin"].strip(), row["model_id"].strip()) for row in prediction_rows]
    if len(prediction_keys) != len(set(prediction_keys)):
        raise ExportError("duplicate parent/model prediction key")
    for row in prediction_rows:
        try:
            float(row["score"])
        except ValueError as exc:
            raise ExportError("invalid frozen prediction score") from exc

    if require_frozen_counts:
        locked = [
            row for row in modeling_rows
            if row["split_partition"].strip() == "locked_test"
        ]
        if len(manifest_rows) != 5180:
            raise ExportError(f"expected 5,180 frozen source rows, found {len(manifest_rows)}")
        if len(modeling_rows) != 5179:
            raise ExportError(f"expected 5,179 modeling rows, found {len(modeling_rows)}")
        if len(locked) != 1036:
            raise ExportError(f"expected 1,036 locked-test rows, found {len(locked)}")
        locked_ids = {row["parent_asin"] for row in locked}
        if len(prediction_rows) != 6216:
            raise ExportError(f"expected 6,216 held-out prediction rows, found {len(prediction_rows)}")
        if {row["parent_asin"] for row in prediction_rows} != locked_ids:
            raise ExportError("held-out prediction parent universe does not match locked-test manifest")
        for model_id in EXPECTED_MODEL_IDS:
            if sum(row["model_id"] == model_id for row in prediction_rows) != 1036:
                raise ExportError(f"prediction coverage mismatch for {model_id}")

    manifest_out = output_dir / "01_modeling_ready_manifest.csv"
    predictions_out = output_dir / "02_locked_test_predictions.csv"
    _write_selected(manifest_out, modeling_rows, MANIFEST_COLUMNS)
    _write_selected(predictions_out, prediction_rows, PREDICTION_COLUMNS)

    result = {
        "manifest_rows": len(modeling_rows),
        "prediction_rows": len(prediction_rows),
        "manifest_columns": list(MANIFEST_COLUMNS),
        "prediction_columns": list(PREDICTION_COLUMNS),
        "manifest_sha256": _sha256(manifest_out),
        "predictions_sha256": _sha256(predictions_out),
        "raw_review_text_exported": False,
        "image_url_or_path_exported": False,
        "image_binary_exported": False,
    }
    (output_dir / "audit_export_provenance.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--allow-nonfrozen-counts",
        action="store_true",
        help="For synthetic/test data only; do not use for the publication export.",
    )
    args = parser.parse_args(argv)
    result = export_audit_inputs(
        args.manifest,
        args.predictions,
        args.output_dir,
        require_frozen_counts=not args.allow_nonfrozen_counts,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
