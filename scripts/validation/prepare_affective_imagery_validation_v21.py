#!/usr/bin/env python3
"""Safe public entrypoint for v2.1 affective-imagery validation preparation.

The established preparation implementation is retained in the adjacent internal
core module. This entrypoint applies the F closeout fixes without changing the
reviewer-packet row selection/order, and adds an isolated manifest-only refresh
mode that never rebuilds 01-08.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

try:
    from . import _prepare_affective_imagery_validation_v21_core as _core
except ImportError:  # pragma: no cover - direct script execution
    import _prepare_affective_imagery_validation_v21_core as _core


# Preserve the existing module API, including helpers imported by the validator.
for _name in dir(_core):
    if not _name.startswith("__"):
        globals()[_name] = getattr(_core, _name)

TOOL_VERSION = "affective_imagery_validation_v21.4"
_core.TOOL_VERSION = TOOL_VERSION

_ORIGINAL_BUILD_REVIEWER_CONTEXT = _core.build_product_dimension_reviewer_context
_ORIGINAL_PREPARE_VALIDATION_WORKSPACE = _core.prepare_validation_workspace

_TIER_CATEGORY_MEMBERS = {
    "formal": (
        "mandatory_focus_or_direct_target",
        "formal_other_outer",
    ),
    "strict": ("upstream_visual_strict",),
    "uncertain": ("upstream_uncertain",),
    "excluded": ("upstream_excluded",),
    "other": ("other_candidate",),
}


def _tier_category_stats(
    tier_stats: dict[str, dict[str, Any]],
) -> dict[str, dict[str, int]]:
    fields = [
        "candidate_count",
        "initial_quota_selected_count",
        "backfill_selected_count",
        "final_selected_count",
    ]
    return {
        category: {
            field: sum(
                int(tier_stats.get(tier, {}).get(field, 0) or 0)
                for tier in tiers
            )
            for field in fields
        }
        for category, tiers in _TIER_CATEGORY_MEMBERS.items()
    }


def build_product_dimension_reviewer_context(
    full_context: pd.DataFrame,
    product_dimension_items: pd.DataFrame,
    *,
    selection_seed: int = DEFAULT_SELECTION_SEED,
    unlabeled_cap: int = UNLABELED_CONTEXT_CAP,
) -> tuple[pd.DataFrame, dict]:
    """Build 08 exactly as before, but attribute backfill to each row's real tier."""
    reviewer, stats = _ORIGINAL_BUILD_REVIEWER_CONTEXT(
        full_context,
        product_dimension_items,
        selection_seed=selection_seed,
        unlabeled_cap=unlabeled_cap,
    )
    if not stats.get("tier_stats"):
        return reviewer, stats

    model_labels = pd.to_numeric(
        reviewer.get("model_label_value", pd.Series(dtype=int)),
        errors="coerce",
    ).fillna(0).astype(int)
    unlabeled_rows = reviewer.loc[model_labels == 0]
    actual_final_counts = (
        unlabeled_rows["review_priority_tier"].astype(str).value_counts().to_dict()
        if not unlabeled_rows.empty
        else {}
    )

    recomputed_backfill = 0
    for tier, tier_stats in stats["tier_stats"].items():
        initial = int(tier_stats.get("initial_quota_selected_count", 0) or 0)
        final = int(actual_final_counts.get(tier, 0) or 0)
        backfill = final - initial
        if backfill < 0:
            raise ValueError(
                f"reviewer tier {tier} has final_selected_count {final} below "
                f"initial quota count {initial}"
            )
        tier_stats["backfill_selected_count"] = backfill
        tier_stats["final_selected_count"] = final
        recomputed_backfill += backfill

    if sum(actual_final_counts.values()) != len(unlabeled_rows):
        raise ValueError("reviewer tier final counts do not cover all unlabeled reviewer rows")

    original_total_backfill = int(stats.get("total_backfill_rows", 0) or 0)
    if recomputed_backfill != original_total_backfill:
        raise ValueError(
            "recomputed reviewer backfill total does not match deterministic selection: "
            f"{recomputed_backfill} != {original_total_backfill}"
        )

    stats["total_backfill_rows"] = recomputed_backfill
    stats["tier_category_stats"] = _tier_category_stats(stats["tier_stats"])
    return reviewer, stats


# The retained preparation function resolves this global in its own module at runtime.
# Replacing it here changes statistics only; the returned 08 rows and their order come
# directly from the original deterministic implementation.
_core.build_product_dimension_reviewer_context = build_product_dimension_reviewer_context


REVIEW_CONTENT_COLUMNS = tuple(
    column for column in ANNOTATION_COLUMNS if column.startswith("human_")
)
ANNOTATION_FILENAMES = (
    ANNOTATIONS_A1_FILENAME,
    ANNOTATIONS_A2_FILENAME,
)


def _assert_overwrite_preserves_human_review(output_dir: Path) -> None:
    """Refuse overwrite unless every existing annotation file is provably blank."""
    output_dir = Path(output_dir)
    for filename in ANNOTATION_FILENAMES:
        path = output_dir / filename
        if not path.exists():
            continue
        try:
            with path.open(
                "r", encoding="utf-8-sig", newline=""
            ) as handle:
                header = next(csv.reader(handle))
            invalid_review_headers = [
                column
                for column in REVIEW_CONTENT_COLUMNS
                if header.count(column) != 1
            ]
            if invalid_review_headers:
                raise ValueError(
                    "review columns must appear exactly once: "
                    f"{', '.join(invalid_review_headers)}"
                )
            frame = read_csv(path)
        except Exception as exc:
            raise ValueError(
                "refusing --overwrite: cannot verify existing annotation file "
                f"is blank: {path.name}: {exc}"
            ) from exc

        missing = [
            column for column in REVIEW_CONTENT_COLUMNS if column not in frame.columns
        ]
        if missing:
            raise ValueError(
                "refusing --overwrite: cannot verify existing annotation file "
                f"is blank: {path.name} missing review columns: {', '.join(missing)}"
            )
        if any(
            frame[column].map(_clean_text).ne("").any()
            for column in REVIEW_CONTENT_COLUMNS
        ):
            raise ValueError(
                "refusing --overwrite: existing A1/A2 annotation files "
                f"contain human review content: {path.name}"
            )


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
    """Run the established preparation path with corrected reviewer statistics."""
    if overwrite:
        _assert_overwrite_preserves_human_review(output_dir)
    manifest = _ORIGINAL_PREPARE_VALIDATION_WORKSPACE(
        input_dir=input_dir,
        output_dir=output_dir,
        seed=seed,
        overwrite=overwrite,
        source_manifest=source_manifest,
        upstream_classified=upstream_classified,
        v21_evidence=v21_evidence,
    )
    reviewer_packet = manifest.get("reviewer_packet")
    if isinstance(reviewer_packet, dict):
        tier_stats = reviewer_packet.get("tier_stats", {})
        reviewer_packet["tier_category_stats"] = _tier_category_stats(tier_stats)
        manifest["positive_items_with_outer_evidence"] = int(
            reviewer_packet.get("positive_items_with_outer_evidence", 0) or 0
        )
        manifest["positive_items_without_outer_evidence"] = int(
            reviewer_packet.get("positive_items_without_outer_evidence", 0) or 0
        )
        manifest["positive_no_outer_annotation_item_ids"] = list(
            reviewer_packet.get("positive_no_outer_annotation_item_ids", [])
        )
        _core._write_manifest(Path(output_dir) / MANIFEST_FILENAME, manifest)
    return manifest


_core.prepare_validation_workspace = prepare_validation_workspace


def refresh_validation_manifest_only(
    output_dir: Path,
    *,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Delegate the isolated 06-only refresh to the safety-focused manifest module."""
    try:
        from .affective_imagery_validation_manifest_v21 import (
            refresh_validation_manifest_only as _refresh,
        )
    except ImportError:  # pragma: no cover - direct script execution
        from affective_imagery_validation_manifest_v21 import (
            refresh_validation_manifest_only as _refresh,
        )
    return _refresh(Path(output_dir), dry_run=dry_run)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare v2.1 affective imagery human-validation templates, or safely "
            "refresh only 06_validation_manifest.json in an existing F workspace."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        help=(
            "Directory containing v2.1 output files 38, 39, 39b, 40, and 41. "
            "Required for normal preparation; not used by --refresh-manifest-only."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        type=Path,
        help="Directory for the F manual-validation workspace.",
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
        help=(
            "Replace generated 01-08 files in normal preparation mode. NEVER use this "
            "against the formal F workspace after annotations have started."
        ),
    )
    parser.add_argument(
        "--refresh-manifest-only",
        action="store_true",
        help=(
            "Read the existing F workspace and refresh only 06_validation_manifest.json. "
            "01-05, 07, and 08 are SHA-256 protected and never rewritten."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "With --refresh-manifest-only, compute and validate the prospective manifest "
            "without writing any file."
        ),
    )
    args = parser.parse_args()
    if args.refresh_manifest_only and args.overwrite:
        parser.error("--refresh-manifest-only cannot be combined with --overwrite")
    if args.dry_run and not args.refresh_manifest_only:
        parser.error("--dry-run is only valid with --refresh-manifest-only")
    if not args.refresh_manifest_only and args.input_dir is None:
        parser.error("--input-dir is required for normal preparation")
    return args


def _print_manifest_refresh_summary(summary: dict[str, Any]) -> None:
    if summary["dry_run"]:
        print("Manifest-only refresh: DRY RUN (zero writes)")
    else:
        print("Manifest-only refresh: WROTE 06_validation_manifest.json ONLY")
    print(f"Manifest path: {summary['manifest_path']}")
    print(f"Manifest SHA-256 before: {summary['manifest_sha256_before']}")
    print(f"Manifest SHA-256 after: {summary['manifest_sha256_after']}")
    if summary["dry_run"]:
        print(
            "Prospective manifest SHA-256: "
            f"{summary['prospective_manifest_sha256']}"
        )
    print(
        "Positive evidence: "
        f"with_outer={summary['positive_items_with_outer_evidence']} "
        f"without_outer={summary['positive_items_without_outer_evidence']}"
    )
    print(
        "Positive/no-outer annotation_item_ids: "
        + json.dumps(
            summary["positive_no_outer_annotation_item_ids"],
            ensure_ascii=False,
        )
    )
    final_counts = {
        category: int(stats.get("final_selected_count", 0) or 0)
        for category, stats in summary["tier_category_stats"].items()
    }
    print("Tier final counts: " + json.dumps(final_counts, ensure_ascii=False, sort_keys=True))
    print(f"Total backfill rows: {summary['total_backfill_rows']}")
    print("Protected SHA-256 check: PASS")


def main() -> int:
    args = parse_args()
    try:
        if args.refresh_manifest_only:
            summary = refresh_validation_manifest_only(
                args.output_dir,
                dry_run=args.dry_run,
            )
            _print_manifest_refresh_summary(summary)
            return 0

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
    except (FileNotFoundError, FileExistsError, RuntimeError, ValueError) as exc:
        print(f"F validation preparation failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
