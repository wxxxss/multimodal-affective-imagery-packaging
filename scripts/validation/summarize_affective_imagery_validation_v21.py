#!/usr/bin/env python3
"""Summarize adjudicated v2.1 affective imagery validation results."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any

import pandas as pd

try:
    from .prepare_affective_imagery_validation_v21 import (
        DEFAULT_OUTPUT_DIR,
        FALSE_NEGATIVE_GROUPS,
        PRODUCT_DIMENSION_ITEMS_FILENAME,
        RELATION_EVIDENCE_GROUPS,
        SENTENCE_ITEMS_FILENAME,
        _clean_text,
        read_csv,
    )
except ImportError:  # pragma: no cover - direct script execution
    from prepare_affective_imagery_validation_v21 import (
        DEFAULT_OUTPUT_DIR,
        FALSE_NEGATIVE_GROUPS,
        PRODUCT_DIMENSION_ITEMS_FILENAME,
        RELATION_EVIDENCE_GROUPS,
        SENTENCE_ITEMS_FILENAME,
        _clean_text,
        read_csv,
    )

try:
    from .validate_affective_imagery_annotations_v21 import validate_annotation_frame
except ImportError:  # pragma: no cover - direct script execution
    from validate_affective_imagery_annotations_v21 import validate_annotation_frame


SENTENCE_CORE_SEMANTIC_FIELDS = (
    "human_packaging_visual",
    "human_relation_valid",
    "human_package_level",
    "human_dimension_code",
    "human_additional_dimension_codes",
    "human_polarity",
)
ALLOWED_NORMALIZATION_FIELDS = ("human_action", "human_error_type")


def _ratio(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator


def _item_table(sentence_items: pd.DataFrame, product_dimension_items: pd.DataFrame) -> pd.DataFrame:
    return pd.concat([sentence_items, product_dimension_items], ignore_index=True, sort=False)


def _cohens_kappa(values_a: list[str], values_b: list[str]) -> float | None:
    pairs = [
        (a, b)
        for a, b in zip(values_a, values_b)
        if _clean_text(a) and _clean_text(b)
    ]
    if not pairs:
        return None
    categories = sorted(set([a for a, _ in pairs] + [b for _, b in pairs]))
    if len(categories) < 2:
        return None
    total = len(pairs)
    observed = sum(1 for a, b in pairs if a == b) / total
    expected = 0.0
    for category in categories:
        pa = sum(1 for a, _ in pairs if a == category) / total
        pb = sum(1 for _, b in pairs if b == category) / total
        expected += pa * pb
    if expected == 1:
        return None
    return (observed - expected) / (1 - expected)


def _agreement_key(row: pd.Series, prefix: str) -> tuple[str, ...]:
    if _clean_text(row.get("item_type")) == "sentence":
        return (
            _clean_text(row.get(f"{prefix}_human_relation_valid")),
            _clean_text(row.get(f"{prefix}_human_package_level")),
            _clean_text(row.get(f"{prefix}_human_dimension_code")),
            _clean_text(row.get(f"{prefix}_human_action")),
        )
    return (
        _clean_text(row.get(f"{prefix}_human_product_label_traceable")),
        _clean_text(row.get(f"{prefix}_human_unlabeled_missed_signal")),
    )


def _metric(matches: list[bool]) -> dict[str, int | float | None]:
    numerator = sum(matches)
    denominator = len(matches)
    return {
        "numerator": numerator,
        "denominator": denominator,
        "rate": _ratio(numerator, denominator),
    }


def _empty_agreement_metrics() -> dict[str, Any]:
    return {
        "raw_agreement": None,
        "legacy_raw_agreement": None,
        "raw_agreement_metadata": {
            "authoritative_core": False,
            "description": (
                "legacy metric; not the authoritative core semantic agreement"
            ),
        },
        "cohens_kappa_relation_valid": None,
        "sentence_core_semantic_exact_agreement": None,
        "sentence_action_agreement": None,
        "sentence_error_type_agreement": None,
        "product_dimension_core_agreement_overall": None,
        "product_dimension_core_agreement_positive": None,
        "product_dimension_core_agreement_unlabeled": None,
    }


def agreement_metrics(
    items: pd.DataFrame,
    annotations_a1: pd.DataFrame | None,
    annotations_a2: pd.DataFrame | None,
) -> dict[str, Any]:
    """Return legacy and authoritative layered A1/A2 agreement metrics."""
    if annotations_a1 is None or annotations_a2 is None:
        return _empty_agreement_metrics()
    a1 = annotations_a1.add_prefix("A1_").rename(
        columns={"A1_annotation_item_id": "annotation_item_id"}
    )
    a2 = annotations_a2.add_prefix("A2_").rename(
        columns={"A2_annotation_item_id": "annotation_item_id"}
    )
    item_columns = ["annotation_item_id", "item_type"]
    if "model_label_value" in items.columns:
        item_columns.append("model_label_value")
    merged = (
        items[item_columns]
        .merge(a1, on="annotation_item_id", how="left", validate="one_to_one")
        .merge(a2, on="annotation_item_id", how="left", validate="one_to_one")
    )
    if merged.empty:
        return _empty_agreement_metrics()
    legacy_agreement = [
        _agreement_key(row, "A1") == _agreement_key(row, "A2")
        for _, row in merged.iterrows()
    ]
    sentence_rows = merged.loc[merged["item_type"].map(_clean_text) == "sentence"]
    product_rows = merged.loc[
        merged["item_type"].map(_clean_text) == "product_dimension"
    ].copy()
    sentence_core = [
        all(
            _clean_text(row.get(f"A1_{field}"))
            == _clean_text(row.get(f"A2_{field}"))
            for field in SENTENCE_CORE_SEMANTIC_FIELDS
        )
        for _, row in sentence_rows.iterrows()
    ]
    sentence_action = [
        _clean_text(row.get("A1_human_action"))
        == _clean_text(row.get("A2_human_action"))
        for _, row in sentence_rows.iterrows()
    ]
    sentence_error = [
        _clean_text(row.get("A1_human_error_type"))
        == _clean_text(row.get("A2_human_error_type"))
        for _, row in sentence_rows.iterrows()
    ]

    if not product_rows.empty:
        labels = pd.to_numeric(product_rows.get("model_label_value"), errors="coerce")
        if labels.isna().any() or not labels.isin([0, 1]).all():
            raise ValueError("product-dimension model_label_value must be 0 or 1")
        positive_rows = product_rows.loc[labels == 1]
        unlabeled_rows = product_rows.loc[labels == 0]
    else:
        positive_rows = product_rows
        unlabeled_rows = product_rows
    positive_matches = [
        _clean_text(row.get("A1_human_product_label_traceable"))
        == _clean_text(row.get("A2_human_product_label_traceable"))
        for _, row in positive_rows.iterrows()
    ]
    unlabeled_matches = [
        _clean_text(row.get("A1_human_unlabeled_missed_signal"))
        == _clean_text(row.get("A2_human_unlabeled_missed_signal"))
        for _, row in unlabeled_rows.iterrows()
    ]
    positive_metric = _metric(positive_matches)
    unlabeled_metric = _metric(unlabeled_matches)
    product_numerator = int(positive_metric["numerator"]) + int(
        unlabeled_metric["numerator"]
    )
    product_denominator = int(positive_metric["denominator"]) + int(
        unlabeled_metric["denominator"]
    )
    product_overall = {
        "numerator": product_numerator,
        "denominator": product_denominator,
        "rate": _ratio(product_numerator, product_denominator),
    }
    legacy_rate = sum(legacy_agreement) / len(legacy_agreement)
    return {
        "raw_agreement": legacy_rate,
        "legacy_raw_agreement": legacy_rate,
        "raw_agreement_metadata": {
            "authoritative_core": False,
            "description": (
                "legacy metric; not the authoritative core semantic agreement"
            ),
        },
        "cohens_kappa_relation_valid": _cohens_kappa(
            sentence_rows.get("A1_human_relation_valid", pd.Series(dtype=str)).map(_clean_text).tolist(),
            sentence_rows.get("A2_human_relation_valid", pd.Series(dtype=str)).map(_clean_text).tolist(),
        ),
        "sentence_core_semantic_exact_agreement": _metric(sentence_core),
        "sentence_action_agreement": _metric(sentence_action),
        "sentence_error_type_agreement": _metric(sentence_error),
        "product_dimension_core_agreement_overall": product_overall,
        "product_dimension_core_agreement_positive": positive_metric,
        "product_dimension_core_agreement_unlabeled": unlabeled_metric,
    }


_agreement_metrics = agreement_metrics


def derived_sidecar_agreement(
    derived_a1: pd.DataFrame | None,
    derived_a2: pd.DataFrame | None,
) -> dict[str, Any]:
    """Compute derived action/error agreement only over both-resolved pairs.

    The denominator is explicitly limited to sentence-applicable pairs where
    both A1 and A2 are resolved.  Product-dimension `not_applicable` rows are
    excluded from resolved/unresolved counts and reported separately.  A1/A2
    sidecar ID sets must match exactly; intersection is never used to silently
    drop records.
    """
    if derived_a1 is None or derived_a2 is None:
        return {
            "sentence_applicable_pair_total": 0,
            "product_dimension_not_applicable_pair_count": 0,
            "a1_resolved_count": 0,
            "a1_unresolved_count": 0,
            "a2_resolved_count": 0,
            "a2_unresolved_count": 0,
            "both_resolved_pair_count": 0,
            "pair_with_any_unresolved_count": 0,
            "derived_action_agreement": None,
            "derived_action_agreement_metadata": {
                "denominator": "sentence pairs where both A1 and A2 are resolved",
            },
            "derived_error_type_agreement": None,
            "derived_error_type_agreement_metadata": {
                "denominator": "sentence pairs where both A1 and A2 are resolved",
            },
            "unresolved_reason_distribution": {},
        }

    def _keyed(frame: pd.DataFrame) -> dict[str, dict[str, str]]:
        return {
            _clean_text(row.get("annotation_item_id")): {
                "status": _clean_text(row.get("mapping_status")),
                "action": _clean_text(row.get("derived_human_action")),
                "error_type": _clean_text(row.get("derived_human_error_type")),
                "reason_code": _clean_text(row.get("reason_code")),
            }
            for _, row in frame.iterrows()
        }

    a1 = _keyed(derived_a1)
    a2 = _keyed(derived_a2)
    if set(a1) != set(a2):
        raise ValueError(
            "A1/A2 derived sidecar ID sets must match exactly; "
            f"a1_missing={sorted(set(a2) - set(a1))[:5]} "
            f"a2_missing={sorted(set(a1) - set(a2))[:5]}"
        )
    pair_ids = sorted(a1)
    not_applicable = [
        pair_id
        for pair_id in pair_ids
        if a1[pair_id]["status"] == "not_applicable"
        or a2[pair_id]["status"] == "not_applicable"
    ]
    sentence_ids = [
        pair_id for pair_id in pair_ids if pair_id not in not_applicable
    ]
    a1_resolved = sum(
        1 for pair_id in sentence_ids if a1[pair_id]["status"] == "resolved"
    )
    a1_unresolved = sum(
        1 for pair_id in sentence_ids if a1[pair_id]["status"] != "resolved"
    )
    a2_resolved = sum(
        1 for pair_id in sentence_ids if a2[pair_id]["status"] == "resolved"
    )
    a2_unresolved = sum(
        1 for pair_id in sentence_ids if a2[pair_id]["status"] != "resolved"
    )
    both_resolved = [
        pair_id
        for pair_id in sentence_ids
        if a1[pair_id]["status"] == "resolved" and a2[pair_id]["status"] == "resolved"
    ]
    any_unresolved = [
        pair_id
        for pair_id in sentence_ids
        if a1[pair_id]["status"] != "resolved" or a2[pair_id]["status"] != "resolved"
    ]
    action_matches = [
        a1[pair_id]["action"] == a2[pair_id]["action"]
        for pair_id in both_resolved
    ]
    error_matches = [
        a1[pair_id]["error_type"] == a2[pair_id]["error_type"]
        for pair_id in both_resolved
    ]
    unresolved_reasons: Counter[str] = Counter()
    for pair_id in any_unresolved:
        for key in (a1[pair_id], a2[pair_id]):
            if key["status"] != "resolved" and key["reason_code"]:
                unresolved_reasons[key["reason_code"]] += 1

    return {
        "sentence_applicable_pair_total": len(sentence_ids),
        "product_dimension_not_applicable_pair_count": len(not_applicable),
        "a1_resolved_count": a1_resolved,
        "a1_unresolved_count": a1_unresolved,
        "a2_resolved_count": a2_resolved,
        "a2_unresolved_count": a2_unresolved,
        "both_resolved_pair_count": len(both_resolved),
        "pair_with_any_unresolved_count": len(any_unresolved),
        "derived_action_agreement": _ratio(
            sum(action_matches), len(action_matches)
        ),
        "derived_action_agreement_numerator": sum(action_matches),
        "derived_action_agreement_denominator": len(action_matches),
        "derived_action_agreement_metadata": {
            "denominator": "sentence pairs where both A1 and A2 are resolved",
        },
        "derived_error_type_agreement": _ratio(
            sum(error_matches), len(error_matches)
        ),
        "derived_error_type_agreement_numerator": sum(error_matches),
        "derived_error_type_agreement_denominator": len(error_matches),
        "derived_error_type_agreement_metadata": {
            "denominator": "pairs where both A1 and A2 are resolved",
        },
        "unresolved_reason_distribution": dict(sorted(unresolved_reasons.items())),
    }


def _compare_one_annotator(
    items: pd.DataFrame,
    before: pd.DataFrame,
    after: pd.DataFrame,
) -> dict[str, bool]:
    for frame_name, frame in [("before", before), ("after", after)]:
        if "annotation_item_id" not in frame.columns:
            raise ValueError(f"{frame_name} annotations missing annotation_item_id")
        if frame["annotation_item_id"].map(_clean_text).duplicated().any():
            raise ValueError(f"{frame_name} annotation_item_id is duplicated")

    before_ids = before["annotation_item_id"].map(_clean_text).tolist()
    after_ids = after["annotation_item_id"].map(_clean_text).tolist()
    item_ids = items["annotation_item_id"].map(_clean_text).tolist()
    same_ids = set(before_ids) == set(after_ids) == set(item_ids)
    row_order_unchanged = same_ids and before_ids == after_ids
    if not same_ids:
        return {
            "row_order_unchanged": False,
            "sentence_core_fields_unchanged": False,
            "product_dimension_core_fields_unchanged": False,
            "product_dimension_rows_unchanged": False,
            "non_target_fields_unchanged": False,
        }

    before_by_id = before.copy()
    after_by_id = after.copy()
    before_by_id.index = before_by_id["annotation_item_id"].map(_clean_text)
    after_by_id.index = after_by_id["annotation_item_id"].map(_clean_text)
    item_type_by_id = {
        _clean_text(row["annotation_item_id"]): _clean_text(row["item_type"])
        for _, row in items.iterrows()
    }
    sentence_ids = [
        item_id for item_id in item_ids if item_type_by_id[item_id] == "sentence"
    ]
    product_ids = [
        item_id
        for item_id in item_ids
        if item_type_by_id[item_id] == "product_dimension"
    ]

    def fields_unchanged(
        ids: list[str],
        fields: list[str] | tuple[str, ...],
    ) -> bool:
        if any(
            field not in before.columns or field not in after.columns
            for field in fields
        ):
            return False
        return all(
            _clean_text(before_by_id.at[item_id, field])
            == _clean_text(after_by_id.at[item_id, field])
            for item_id in ids
            for field in fields
        )

    sentence_core_unchanged = fields_unchanged(
        sentence_ids, SENTENCE_CORE_SEMANTIC_FIELDS
    )
    product_core_unchanged = fields_unchanged(
        product_ids,
        (
            "human_product_label_traceable",
            "human_unlabeled_missed_signal",
        ),
    )
    same_columns = list(before.columns) == list(after.columns)
    non_target_fields = [
        field for field in before.columns if field not in ALLOWED_NORMALIZATION_FIELDS
    ]
    non_target_unchanged = same_columns and fields_unchanged(
        item_ids, non_target_fields
    )
    product_rows_unchanged = same_columns and fields_unchanged(
        product_ids, list(before.columns)
    )
    return {
        "row_order_unchanged": row_order_unchanged,
        "sentence_core_fields_unchanged": sentence_core_unchanged,
        "product_dimension_core_fields_unchanged": product_core_unchanged,
        "product_dimension_rows_unchanged": product_rows_unchanged,
        "non_target_fields_unchanged": non_target_unchanged,
    }


def _rate_delta(after: dict[str, Any], before: dict[str, Any]) -> float | None:
    if after["rate"] is None or before["rate"] is None:
        return None
    return after["rate"] - before["rate"]


def compare_normalization_invariance(
    *,
    items: pd.DataFrame,
    before_a1: pd.DataFrame,
    before_a2: pd.DataFrame,
    after_a1: pd.DataFrame,
    after_a2: pd.DataFrame,
) -> dict[str, Any]:
    """Compare normalization row-by-row and report agreement before/after."""
    a1_invariants = _compare_one_annotator(items, before_a1, after_a1)
    a2_invariants = _compare_one_annotator(items, before_a2, after_a2)
    before_metrics = agreement_metrics(items, before_a1, before_a2)
    after_metrics = agreement_metrics(items, after_a1, after_a2)

    flags = {
        name: a1_invariants[name] and a2_invariants[name]
        for name in a1_invariants
    }
    action_before = before_metrics["sentence_action_agreement"]
    action_after = after_metrics["sentence_action_agreement"]
    error_before = before_metrics["sentence_error_type_agreement"]
    error_after = after_metrics["sentence_error_type_agreement"]
    return {
        **flags,
        "annotators": {"A1": a1_invariants, "A2": a2_invariants},
        "invariants_pass": all(flags.values()),
        "sentence_core_agreement_before": before_metrics[
            "sentence_core_semantic_exact_agreement"
        ],
        "sentence_core_agreement_after": after_metrics[
            "sentence_core_semantic_exact_agreement"
        ],
        "product_core_agreement_before": before_metrics[
            "product_dimension_core_agreement_overall"
        ],
        "product_core_agreement_after": after_metrics[
            "product_dimension_core_agreement_overall"
        ],
        "action_agreement_before": action_before,
        "action_agreement_after": action_after,
        "action_agreement_delta": _rate_delta(action_after, action_before),
        "error_type_agreement_before": error_before,
        "error_type_agreement_after": error_after,
        "error_type_agreement_delta": _rate_delta(error_after, error_before),
    }


def summarize_validation(
    *,
    sentence_items: pd.DataFrame,
    product_dimension_items: pd.DataFrame,
    adjudication: pd.DataFrame,
    annotations_a1: pd.DataFrame | None = None,
    annotations_a2: pd.DataFrame | None = None,
) -> dict[str, Any]:
    items = _item_table(sentence_items, product_dimension_items)
    merged = items.merge(
        adjudication,
        on="annotation_item_id",
        how="left",
        suffixes=("", "_adjudication"),
        validate="one_to_one",
    )

    sentence = merged.loc[merged["item_type"] == "sentence"].copy()
    product = merged.loc[merged["item_type"] == "product_dimension"].copy()

    predicted_relation = sentence["audit_group"].isin(RELATION_EVIDENCE_GROUPS)
    relation_denominator = int(predicted_relation.sum())
    relation_numerator = int(
        (sentence.loc[predicted_relation, "adjudicated_relation_valid"].map(_clean_text) == "yes").sum()
    )

    predicted_outer = sentence["audit_group"].map(_clean_text) == "outer_relation_evidence"
    outer_denominator = int(predicted_outer.sum())
    outer_numerator = int(
        (
            (sentence.loc[predicted_outer, "adjudicated_relation_valid"].map(_clean_text) == "yes")
            & (sentence.loc[predicted_outer, "adjudicated_package_level"].map(_clean_text) == "outer")
        ).sum()
    )

    valid_outer = (
        (sentence["adjudicated_relation_valid"].map(_clean_text) == "yes")
        & (sentence["adjudicated_package_level"].map(_clean_text) == "outer")
        & sentence["dimension_code"].map(_clean_text).ne("")
    )
    dimension_denominator = int(valid_outer.sum())
    dimension_numerator = int(
        (
            sentence.loc[valid_outer, "dimension_code"].map(_clean_text)
            == sentence.loc[valid_outer, "adjudicated_dimension_code"].map(_clean_text)
        ).sum()
    )

    recovered = sentence["audit_group"].map(_clean_text) == "recovered_v21"
    recovered_denominator = int(recovered.sum())
    recovered_numerator = int(
        (
            (sentence.loc[recovered, "adjudicated_relation_valid"].map(_clean_text) == "yes")
            & (sentence.loc[recovered, "adjudicated_package_level"].map(_clean_text) == "outer")
        ).sum()
    )

    false_negative_rows = sentence["audit_group"].isin(FALSE_NEGATIVE_GROUPS)
    false_negative_denominator = int(false_negative_rows.sum())
    false_negative_numerator = int(
        (
            sentence.loc[false_negative_rows, "adjudicated_action"].map(_clean_text)
            == "add_missing"
        ).sum()
    )

    positive_product = pd.to_numeric(
        product.get("model_label_value", pd.Series(dtype=int)),
        errors="coerce",
    ).fillna(0).astype(int) == 1
    product_trace_denominator = int(positive_product.sum())
    product_trace_numerator = int(
        (
            product.loc[positive_product, "adjudicated_product_label_traceable"].map(_clean_text)
            == "yes"
        ).sum()
    )

    unlabeled = pd.to_numeric(
        product.get("model_label_value", pd.Series(dtype=int)),
        errors="coerce",
    ).fillna(0).astype(int) == 0

    summary = {
        "relation_precision": _ratio(relation_numerator, relation_denominator),
        "relation_precision_numerator": relation_numerator,
        "relation_precision_denominator": relation_denominator,
        "outer_package_precision": _ratio(outer_numerator, outer_denominator),
        "outer_package_precision_numerator": outer_numerator,
        "outer_package_precision_denominator": outer_denominator,
        "dimension_accuracy": _ratio(dimension_numerator, dimension_denominator),
        "dimension_accuracy_numerator": dimension_numerator,
        "dimension_accuracy_denominator": dimension_denominator,
        "recovered_precision": _ratio(recovered_numerator, recovered_denominator),
        "recovered_precision_numerator": recovered_numerator,
        "recovered_precision_denominator": recovered_denominator,
        "false_negative_signal": _ratio(
            false_negative_numerator,
            false_negative_denominator,
        ),
        "false_negative_signal_numerator": false_negative_numerator,
        "false_negative_signal_denominator": false_negative_denominator,
        "product_label_traceability": _ratio(
            product_trace_numerator,
            product_trace_denominator,
        ),
        "product_label_traceability_numerator": product_trace_numerator,
        "product_label_traceability_denominator": product_trace_denominator,
        "unlabeled_not_observed_count": int(unlabeled.sum()),
        "confirmed_negative_count": 0,
        "label_semantics": "positive_unlabeled",
        "zero_label_interpretation": "unlabeled_not_observed",
    }
    summary.update(agreement_metrics(items, annotations_a1, annotations_a2))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize adjudicated v2.1 affective imagery validation metrics."
    )
    parser.add_argument(
        "--items-dir",
        default=DEFAULT_OUTPUT_DIR,
        type=Path,
        help="Directory containing 01_sentence_items.csv and 02_product_dimension_items.csv.",
    )
    parser.add_argument(
        "--adjudication",
        type=Path,
        help="Final adjudication CSV. Defaults to items-dir/05_adjudication_template.csv.",
    )
    parser.add_argument("--a1", type=Path, help="Optional A1 annotation CSV for agreement.")
    parser.add_argument("--a2", type=Path, help="Optional A2 annotation CSV for agreement.")
    parser.add_argument(
        "--derived-a1",
        type=Path,
        help="Optional A1 decision sidecar CSV for derived action/error agreement.",
    )
    parser.add_argument(
        "--derived-a2",
        type=Path,
        help="Optional A2 decision sidecar CSV for derived action/error agreement.",
    )
    parser.add_argument(
        "--before-a1",
        type=Path,
        help="Optional pre-normalization A1 CSV for row-level comparison.",
    )
    parser.add_argument(
        "--before-a2",
        type=Path,
        help="Optional pre-normalization A2 CSV for row-level comparison.",
    )
    parser.add_argument(
        "--agreement-only",
        action="store_true",
        help="Report agreement without requiring or reading adjudication.",
    )
    parser.add_argument("--output-json", type=Path, help="Optional JSON summary output.")
    parser.add_argument("--output-csv", type=Path, help="Optional CSV summary output.")
    args = parser.parse_args()
    if (args.a1 is None) != (args.a2 is None):
        parser.error("--a1 and --a2 must be supplied together")
    if (args.before_a1 is None) != (args.before_a2 is None):
        parser.error("--before-a1 and --before-a2 must be supplied together")
    if (args.derived_a1 is None) != (args.derived_a2 is None):
        parser.error("--derived-a1 and --derived-a2 must be supplied together")
    if args.agreement_only and args.a1 is None:
        parser.error("--agreement-only requires --a1 and --a2")
    if args.before_a1 is not None and args.a1 is None:
        parser.error("before/after comparison requires --a1 and --a2")
    return args


def main() -> int:
    args = parse_args()
    items_dir = Path(args.items_dir)
    sentence_items = read_csv(items_dir / SENTENCE_ITEMS_FILENAME)
    product_items = read_csv(items_dir / PRODUCT_DIMENSION_ITEMS_FILENAME)
    items = _item_table(sentence_items, product_items)
    a1 = read_csv(args.a1) if args.a1 else None
    a2 = read_csv(args.a2) if args.a2 else None
    if a1 is not None and a2 is not None:
        validate_annotation_frame(a1, items, expected_annotator_id="A1")
        validate_annotation_frame(a2, items, expected_annotator_id="A2")

    if args.agreement_only:
        summary = agreement_metrics(items, a1, a2)
    else:
        adjudication_path = args.adjudication or (
            items_dir / "05_adjudication_template.csv"
        )
        adjudication = read_csv(adjudication_path)
        summary = summarize_validation(
            sentence_items=sentence_items,
            product_dimension_items=product_items,
            adjudication=adjudication,
            annotations_a1=a1,
            annotations_a2=a2,
        )

    comparison: dict[str, Any] | None = None
    if args.before_a1 is not None and args.before_a2 is not None:
        assert a1 is not None and a2 is not None
        before_a1 = read_csv(args.before_a1)
        before_a2 = read_csv(args.before_a2)
        validate_annotation_frame(
            before_a1,
            items,
            expected_annotator_id="A1",
        )
        validate_annotation_frame(
            before_a2,
            items,
            expected_annotator_id="A2",
        )
        comparison = compare_normalization_invariance(
            items=items,
            before_a1=before_a1,
            before_a2=before_a2,
            after_a1=a1,
            after_a2=a2,
        )
        summary.update(comparison)
    if args.derived_a1 is not None and args.derived_a2 is not None:
        derived_a1 = read_csv(args.derived_a1)
        derived_a2 = read_csv(args.derived_a2)
        summary.update(
            derived_sidecar_agreement(derived_a1, derived_a2)
        )
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    if args.output_csv:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([summary]).to_csv(args.output_csv, index=False, encoding="utf-8-sig")
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    if comparison is not None and not comparison["invariants_pass"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
