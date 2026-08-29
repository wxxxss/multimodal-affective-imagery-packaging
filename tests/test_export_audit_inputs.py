from __future__ import annotations

import csv
from pathlib import Path

from scripts.release.export_audit_inputs import export_audit_inputs


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _read_header(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return next(csv.reader(handle))


def test_export_keeps_only_metric_audit_columns(tmp_path: Path):
    manifest = tmp_path / "source_manifest.csv"
    predictions = tmp_path / "source_predictions.csv"
    output = tmp_path / "derived"

    _write_csv(
        manifest,
        [{
            "parent_asin": "A1",
            "main_analysis_included": "true",
            "split_partition": "locked_test",
            "primary_response_sha256": "abc",
            "has_any_outer_imagery_observed": "1",
            "general_visual_appeal_observed_positive_core": "0",
            "cute_friendly_observed_positive_core": "0",
            "primary_source_url": "https://example.invalid/should-not-export",
            "primary_local_path": "private/path.jpg",
        }],
    )
    _write_csv(
        predictions,
        [{
            "parent_asin": "A1",
            "outcome": "has_any_outer_imagery_observed",
            "track": "openclip_512_logistic",
            "model_id": "has_any_outer_imagery_observed__openclip_512_logistic",
            "score": "0.7",
            "primary_response_sha256": "abc",
            "primary_feature_row_index": "9",
        }],
    )

    result = export_audit_inputs(manifest, predictions, output, require_frozen_counts=False)

    assert result["manifest_rows"] == 1
    assert result["prediction_rows"] == 1
    assert _read_header(output / "01_modeling_ready_manifest.csv") == [
        "parent_asin",
        "main_analysis_included",
        "split_partition",
        "primary_response_sha256",
        "has_any_outer_imagery_observed",
        "general_visual_appeal_observed_positive_core",
        "cute_friendly_observed_positive_core",
    ]
    assert _read_header(output / "02_locked_test_predictions.csv") == [
        "parent_asin", "outcome", "track", "model_id", "score"
    ]
