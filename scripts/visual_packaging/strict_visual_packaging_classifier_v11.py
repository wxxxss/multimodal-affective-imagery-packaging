#!/usr/bin/env python3
"""Strict rule-based filtering of visual packaging review sentences.

The program takes high-recall packaging sentence candidates and assigns:

- visual_strict: explicit packaging object + explicit visual/design evidence
- excluded: clear logistics, order, structure, product-content, or non-packaging
- uncertain: packaging is mentioned but visual evidence is insufficient

Only visual_strict rows are written to the strict visual corpus used by the
next affective-imagery discovery stage.

Supported input:
- .parquet (requires pyarrow)
- .csv.gz
- .csv

Supported detail output:
- parquet (requires pyarrow)
- csv.gz
"""

from __future__ import annotations

import argparse
import csv
import gzip
import html
import json
import math
import random
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Protocol


RULE_VERSION = "strict_visual_packaging_v1.1"

BASE_FIELDS = [
    "parent_asin",
    "review_id",
    "review_fingerprint",
    "sentence_id",
    "sentence_index",
    "sentence_source",
    "sentence",
    "user_id",
    "rating",
    "review_date_utc",
    "verified_purchase",
    "helpful_vote",
    "trigger_groups",
    "matched_trigger_terms",
    "rule_based_context_hint",
]

CLASSIFIED_FIELDS = BASE_FIELDS + [
    "decision",
    "reason",
    "confidence",
    "matched_visual_terms",
    "matched_exclusion_terms",
    "strict_rule_version",
]

PRODUCT_STATS_FIELDS = [
    "parent_asin",
    "title",
    "main_image_url",
    "input_candidate_sentence_count",
    "visual_strict_sentence_count",
    "visual_strict_review_count",
    "visual_strict_reviewer_count",
    "excluded_sentence_count",
    "uncertain_sentence_count",
    "visual_strict_ratio",
    "eligible_visual_ge_1",
    "eligible_visual_ge_2",
    "eligible_visual_ge_3",
    "eligible_visual_ge_5",
]

SAMPLE_FIELDS = [
    "sample_order",
    "parent_asin",
    "review_id",
    "sentence_id",
    "sentence",
    "rating",
    "verified_purchase",
    "helpful_vote",
    "decision",
    "reason",
    "matched_visual_terms",
    "matched_exclusion_terms",
]


@dataclass(frozen=True)
class RuleDecision:
    decision: str
    reason: str
    confidence: str
    matched_visual_terms: tuple[str, ...] = ()
    matched_exclusion_terms: tuple[str, ...] = ()


@dataclass
class ClassificationResult:
    product_count: int
    input_sentence_count: int
    visual_strict_count: int
    excluded_count: int
    uncertain_count: int
    products_with_visual_strict: int
    classified_path: Path
    strict_path: Path


@dataclass
class ProductStats:
    parent_asin: str
    title: str = ""
    main_image_url: str = ""
    input_candidate_sentence_count: int = 0
    visual_strict_sentence_count: int = 0
    visual_reviews: set[str] = field(default_factory=set)
    visual_reviewers: set[str] = field(default_factory=set)
    excluded_sentence_count: int = 0
    uncertain_sentence_count: int = 0


class RowWriter(Protocol):
    path: Path

    def write(self, row: dict[str, Any]) -> None:
        ...

    def close(self) -> None:
        ...


class CsvGzipWriter:
    def __init__(self, path: Path, fieldnames: list[str]) -> None:
        self.path = path
        self._fieldnames = fieldnames
        self._handle = gzip.open(path, "wt", encoding="utf-8-sig", newline="")
        self._writer = csv.DictWriter(self._handle, fieldnames=fieldnames)
        self._writer.writeheader()

    def write(self, row: dict[str, Any]) -> None:
        self._writer.writerow(
            {field: csv_value(row.get(field)) for field in self._fieldnames}
        )

    def close(self) -> None:
        self._handle.close()


class ParquetWriter:
    def __init__(
        self,
        path: Path,
        fieldnames: list[str],
        schema_definition: list[tuple[str, str]],
        batch_size: int,
    ) -> None:
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise RuntimeError(
                "Parquet输入或输出需要pyarrow。请运行："
                "python -m pip install pyarrow"
            ) from exc

        type_map = {
            "string": pa.string(),
            "int64": pa.int64(),
            "float64": pa.float64(),
            "bool": pa.bool_(),
        }
        self.path = path
        self._pa = pa
        self._fieldnames = fieldnames
        self._batch_size = batch_size
        self._buffer: list[dict[str, Any]] = []
        self._schema = pa.schema(
            [(name, type_map[type_name]) for name, type_name in schema_definition]
        )
        self._writer = pq.ParquetWriter(
            path,
            self._schema,
            compression="zstd",
        )

    def write(self, row: dict[str, Any]) -> None:
        self._buffer.append({field: row.get(field) for field in self._fieldnames})
        if len(self._buffer) >= self._batch_size:
            self._flush()

    def _flush(self) -> None:
        if not self._buffer:
            return
        table = self._pa.Table.from_pylist(self._buffer, schema=self._schema)
        self._writer.write_table(table)
        self._buffer.clear()

    def close(self) -> None:
        self._flush()
        self._writer.close()


BASE_SCHEMA = [
    ("parent_asin", "string"),
    ("review_id", "string"),
    ("review_fingerprint", "string"),
    ("sentence_id", "string"),
    ("sentence_index", "int64"),
    ("sentence_source", "string"),
    ("sentence", "string"),
    ("user_id", "string"),
    ("rating", "float64"),
    ("review_date_utc", "string"),
    ("verified_purchase", "bool"),
    ("helpful_vote", "int64"),
    ("trigger_groups", "string"),
    ("matched_trigger_terms", "string"),
    ("rule_based_context_hint", "string"),
]

CLASSIFIED_SCHEMA = BASE_SCHEMA + [
    ("decision", "string"),
    ("reason", "string"),
    ("confidence", "string"),
    ("matched_visual_terms", "string"),
    ("matched_exclusion_terms", "string"),
    ("strict_rule_version", "string"),
]


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    text = html.unescape(str(value))
    text = re.sub(r"<[^>]+>", " ", text)
    text = text.replace("\x00", " ")
    text = re.sub(r"[\u200b\u200c\u200d\ufeff]", "", text)
    return re.sub(r"\s+", " ", text).strip()


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def parse_int(value: Any, default: int = 0) -> int:
    if value is None or value == "":
        return default
    try:
        return int(float(str(value).replace(",", "").strip()))
    except (TypeError, ValueError):
        return default


def parse_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        value_float = float(str(value).replace(",", "").strip())
    except (TypeError, ValueError):
        return None
    return value_float if math.isfinite(value_float) else None


def csv_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "1" if value else "0"
    return str(value)


def compile_terms(terms: dict[str, str]) -> list[tuple[str, re.Pattern[str]]]:
    return [(label, re.compile(pattern, re.I)) for label, pattern in terms.items()]


PACKAGING_OBJECTS = compile_terms(
    {
        "package": r"\bpackag(?:e|es|ed|ing)\b",
        "box": r"\bbox(?:es)?\b",
        "carton": r"\bcarton(?:s)?\b",
        "label": r"\blabel(?:s|ed|ing)?\b",
        "wrapper": r"\bwrapp(?:er|ers|ed|ing)\b",
        "pouch": r"\bpouch(?:es)?\b",
        "tin": r"\btin(?:s)?\b",
        "canister": r"\bcanister(?:s)?\b",
        "container": r"\bcontainer(?:s)?\b",
        "jar": r"\bjar(?:s)?\b",
        "front_panel": r"\bfront (?:of the )?(?:box|package|label)\b",
    }
)

DESIGN_ELEMENTS = compile_terms(
    {
        "design": r"\bdesign(?:s|ed|ing)?\b",
        "illustration": r"\billustrat(?:ion|ions|ed|ive)\b",
        "artwork": r"\bart\s*work\b|\bartwork\b",
        "graphic": r"\bgraphic(?:s|al)?\b",
        "logo": r"\blogo(?:s)?\b",
        "font": r"\bfont(?:s)?\b",
        "typography": r"\btypograph(?:y|ic|ical)\b",
        "lettering": r"\blettering\b",
        "pattern": r"\bpattern(?:s|ed)?\b",
        "print": r"\bprint(?:s|ed|ing)?\b",
        "color_scheme": r"\bcolou?r scheme\b|\bcolour palette\b|\bcolor palette\b",
        "background": r"\bbackground\b",
        "border": r"\bborder(?:s)?\b",
        "layout": r"\blayout\b",
        "floral_design": r"\bfloral (?:design|illustration|artwork|pattern|print|graphic)\b",
        "botanical_design": r"\bbotanical (?:design|illustration|artwork|pattern|print|graphic)\b",
    }
)

STRONG_AESTHETIC = compile_terms(
    {
        "beautiful": r"\bbeautiful(?:ly)?\b",
        "pretty": r"\bpretty\b(?!\s+much)",
        "attractive": r"\battractive\b|\bappealing\b",
        "elegant": r"\belegant(?:ly)?\b",
        "premium": r"\bpremium\b",
        "luxurious": r"\bluxur(?:y|ious|iously)\b",
        "classy": r"\bclassy\b",
        "sophisticated": r"\bsophisticated\b",
        "awesome": r"\bawesome\b",
        "gorgeous": r"\bgorgeous\b",
        "lovely": r"\blovely\b",
        "cute": r"\bcute\b",
        "stylish": r"\bstylish\b",
        "modern": r"\bmodern\b",
        "vintage": r"\bvintage\b|\bretro\b",
        "minimalist": r"\bminimal(?:ist|istic)?\b",
        "natural_looking": r"\bnatural[- ]looking\b",
        "calming_visual": r"\bcalming\b|\bsoothing\b",
        "cheerful": r"\bcheerful\b",
        "clean_looking": r"\bclean[- ]looking\b",
        "eye_catching": r"\beye[- ]catching\b",
        "giftable": r"\bgiftable\b|\bgift[- ]ready\b|\bpresentable\b",
    }
)

NEGATIVE_VISUAL = compile_terms(
    {
        "cheap_looking": r"\bcheap[- ]looking\b|\blooks? cheap\b",
        "ugly": r"\bugly\b",
        "tacky": r"\btacky\b",
        "boring": r"\bboring\b",
        "clinical": r"\bclinical (?:look|appearance)\b|\bclinical-looking\b",
        "dated": r"\bdated\b|\bold[- ]fashioned\b",
        "sad_appearance": r"\bsad (?:box|package|packaging|design|look)\b",
        "plain": r"\bplain[- ]looking\b",
        "unappealing": r"\bunappealing\b|\bnot attractive\b",
    }
)

COLOR_STYLE = compile_terms(
    {
        "specific_color": (
            r"\b(?:pink|green|gold|golden|red|blue|purple|yellow|orange|"
            r"black|white|brown|beige|cream|silver|pastel|dark|bright|"
            r"colorful|colourful)\b"
        ),
        "floral_style": r"\bfloral\b",
        "botanical_style": r"\bbotanical\b",
        "earthy_style": r"\bearthy\b",
    }
)

GIFT_TERMS = compile_terms(
    {
        "gift": r"\bgift(?:s|ed|ing)?\b",
        "giftable": r"\bgiftable\b|\bgift[- ]ready\b",
        "presentation": r"\bpresentation\b|\bpresentable\b",
        "stocking_stuffer": r"\bstocking stuffer\b",
    }
)

SHIPPING_DAMAGE = compile_terms(
    {
        "shipping": r"\bshipping\b|\bdelivery\b|\bdelivered\b",
        "arrived": r"\barriv(?:e|ed|es|ing)\b",
        "crushed": r"\bcrush(?:ed|ing)?\b",
        "damaged": r"\bdamag(?:e|ed|ing)\b",
        "dented": r"\bdent(?:ed|s)?\b",
        "torn": r"\btorn\b|\bripped\b",
        "broken": r"\bbrok(?:e|en)\b",
        "leaking": r"\bleak(?:ed|ing|s)?\b",
        "smashed": r"\bsmashed\b",
    }
)

ORDER_FULFILLMENT = compile_terms(
    {
        "ordered": r"\bordered\b|\border\b",
        "received": r"\breceived\b|\bsent\b|\bshipped\b",
        "missing": r"\bmissing\b|\bdid(?: not|n't) receive\b",
        "wrong_item": r"\bwrong (?:item|product|box|package)\b",
        "only_received": r"\bonly (?:received|got)\b",
        "expected_quantity": r"\bexpected\b.{0,25}\b(?:box|boxes|package|packages)\b",
        "refund_return": r"\brefund\b|\breturnable\b|\breturned\b",
    }
)

STRUCTURAL = compile_terms(
    {
        "sealed": r"\bseal(?:ed|ing|s)?\b",
        "resealable": r"\bresealable\b|\bre-sealable\b",
        "zipper": r"\bzip(?:per|lock)?\b",
        "individually_wrapped": r"\bindividually (?:wrapped|packaged)\b",
        "freshness": r"\bkeep(?:s|ing)? (?:it|them|the tea)? ?fresh\b|\bfor freshness\b",
        "open_close": r"\b(?:easy|hard|difficult) to (?:open|close)\b",
        "vacuum": r"\bvacuum(?:ed)?[- ]sealed\b",
        "foil": r"\bfoil\b",
        "airtight": r"\bairtight\b",
    }
)

QUANTITY_FORMAT = compile_terms(
    {
        "count": r"\b\d+\s*(?:count|ct|(?:tea\s*)?bags?|sachets?|packets?)\b",
        "per_box": r"\bper box\b",
        "number_boxes": r"\b(?:one|two|three|four|five|six|seven|eight|nine|ten|\d+)\s+boxes\b",
        "box_of": r"\bbox of \d+\b|\bbox of (?:tea bags|bags|sachets)\b",
        "servings": r"\bservings? per box\b",
        "size": r"\b(?:ounce|ounces|oz|pound|lb|size)\b",
    }
)

PRODUCT_CONTENT = compile_terms(
    {
        "taste": r"\btast(?:e|es|ed|ing|y|eless)\b|\bflavo(?:r|ur)(?:s|ed|ful)?\b",
        "aroma": r"\baroma(?:s|tic)?\b|\bsmell(?:s|ed|ing)?\b",
        "brew": r"\bbrew(?:s|ed|ing)?\b|\bsteep(?:s|ed|ing)?\b",
        "cup_liquid": r"\bcup(?: of tea)?\b|\bdrink\b|\bliquid\b|\bwater\b",
        "tea_color": r"\b(?:tea|brew|cup|liquid|drink)\b.{0,18}\bcolou?r\b|\bcolou?r\b.{0,18}\b(?:tea|brew|cup|liquid|drink)\b",
        "ingredients": r"\bingredient(?:s)?\b|\bleaves\b|\bflowers\b|\bfruit\b",
        "effect": r"\beffect(?:s)?\b|\bhelps?\b|\bsleep\b|\bstomach\b|\bstress\b|\brelax(?:ing|ed)?\b",
        "caffeine": r"\bcaffeine\b|\bdecaf\b|\bcaffeine[- ]free\b",
        "health": r"\bhealth benefits?\b|\bhormonal\b|\bmedic(?:al|inal)\b",
        "tea_bag_appearance": r"\btea ?bags?\b.{0,30}\blook(?:s|ed|ing)?\b",
    }
)

NON_PACKAGING = compile_terms(
    {
        "looking_for": r"\b(?:was|am|are|were|been|be|is) looking for\b|\bwhat i was looking for\b",
        "hypothetical_storage": r"\b(?:wish|wished) i had\b.{0,40}\b(?:jar|tin|container|canister)\b",
        "photo_advertisement": r"\blook at the photo\b|\blooks? nothing like (?:the )?(?:photo|advertised)\b",
        "general_calming": r"\b(?:tea|it|this)\b.{0,20}\b(?:calming|soothing)\b",
    }
)

WEAK_VISUAL = compile_terms(
    {
        "looks_different": r"\blooks? (?:a little |slightly )?different\b",
        "nice": r"\bnice\b",
        "great": r"\bgreat\b",
        "good": r"\bgood\b",
        "original": r"\boriginal\b",
        "new_packaging": r"\bnew packaging\b",
        "about_packaging": r"\babout the packaging\b",
        "well_packaged": r"\bwell[- ]packaged\b|\bwonderfully packaged\b",
        "love_boxes": r"\blove (?:the )?(?:box|boxes|packaging|package)\b",
    }
)


def matched_labels(
    text: str,
    patterns: list[tuple[str, re.Pattern[str]]],
) -> list[str]:
    return [label for label, pattern in patterns if pattern.search(text)]


def has_near_pair(
    text: str,
    first_patterns: list[tuple[str, re.Pattern[str]]],
    second_patterns: list[tuple[str, re.Pattern[str]]],
    max_chars: int = 80,
) -> bool:
    first_matches = [
        match
        for _, pattern in first_patterns
        for match in pattern.finditer(text)
    ]
    second_matches = [
        match
        for _, pattern in second_patterns
        for match in pattern.finditer(text)
    ]
    for first in first_matches:
        for second in second_matches:
            distance = max(
                second.start() - first.end(),
                first.start() - second.end(),
                0,
            )
            if distance <= max_chars:
                return True
    return False


def has_shipping_event(text: str, shipping_terms: list[str]) -> bool:
    severe_damage = {
        "crushed",
        "damaged",
        "dented",
        "torn",
        "broken",
        "leaking",
        "smashed",
    }
    return bool(
        severe_damage.intersection(shipping_terms)
        or (
            {"shipping", "arrived"}.intersection(shipping_terms)
            and matched_labels(text, PACKAGING_OBJECTS)
        )
    )


def classify_sentence(sentence: str) -> RuleDecision:
    text = normalize_text(sentence)
    if not text:
        return RuleDecision(
            "excluded",
            "not_packaging",
            "high",
            (),
            ("empty_sentence",),
        )

    packaging = matched_labels(text, PACKAGING_OBJECTS)
    design = matched_labels(text, DESIGN_ELEMENTS)
    aesthetic = matched_labels(text, STRONG_AESTHETIC)
    negative_visual = matched_labels(text, NEGATIVE_VISUAL)
    color_style = matched_labels(text, COLOR_STYLE)
    gift = matched_labels(text, GIFT_TERMS)

    shipping = matched_labels(text, SHIPPING_DAMAGE)
    order = matched_labels(text, ORDER_FULFILLMENT)
    structural = matched_labels(text, STRUCTURAL)
    quantity = matched_labels(text, QUANTITY_FORMAT)
    product = matched_labels(text, PRODUCT_CONTENT)
    non_packaging = matched_labels(text, NON_PACKAGING)
    weak = matched_labels(text, WEAK_VISUAL)

    visual_terms = sorted(
        set(packaging + design + aesthetic + negative_visual + color_style + gift)
    )
    exclusion_terms = sorted(
        set(shipping + order + structural + quantity + product + non_packaging)
    )

    # Strong exclusions are evaluated first. This is intentionally conservative.
    if has_shipping_event(text, shipping):
        return RuleDecision(
            "excluded",
            "shipping_damage",
            "high",
            tuple(visual_terms),
            tuple(exclusion_terms),
        )

    if order and packaging:
        return RuleDecision(
            "excluded",
            "order_fulfillment",
            "high",
            tuple(visual_terms),
            tuple(exclusion_terms),
        )

    if "hypothetical_storage" in non_packaging:
        return RuleDecision(
            "excluded",
            "not_packaging",
            "high",
            tuple(visual_terms),
            tuple(exclusion_terms),
        )

    if quantity and packaging and not (
        design or aesthetic or negative_visual or color_style
    ):
        return RuleDecision(
            "excluded",
            "quantity_or_format",
            "high",
            tuple(visual_terms),
            tuple(exclusion_terms),
        )

    if product and not packaging and not design:
        return RuleDecision(
            "excluded",
            "product_content",
            "high",
            tuple(visual_terms),
            tuple(exclusion_terms),
        )

    if non_packaging and not packaging and not design:
        return RuleDecision(
            "excluded",
            "not_packaging",
            "high",
            tuple(visual_terms),
            tuple(exclusion_terms),
        )

    # Explicit design elements linked to packaging are the strongest visual evidence.
    design_linked = bool(design) and (
        bool(packaging)
        or has_near_pair(text, DESIGN_ELEMENTS, PACKAGING_OBJECTS, max_chars=100)
        or bool({"label", "logo", "font", "typography", "lettering"}.intersection(design))
    )
    if design_linked and (
        aesthetic
        or negative_visual
        or color_style
    ):
        return RuleDecision(
            "visual_strict",
            "explicit_design_element",
            "high",
            tuple(visual_terms),
            tuple(exclusion_terms),
        )

    # Color/style is retained only when explicitly attached to packaging.
    color_style_linked = bool(color_style) and bool(packaging) and has_near_pair(
        text,
        COLOR_STYLE,
        PACKAGING_OBJECTS,
        max_chars=65,
    )
    if color_style_linked and (
        aesthetic
        or negative_visual
        or design
    ):
        return RuleDecision(
            "visual_strict",
            "explicit_packaging_color_or_style",
            "high",
            tuple(visual_terms),
            tuple(exclusion_terms),
        )

    # Gift presentation is visual only when package appearance is explicitly praised.
    gift_visual = bool(gift) and bool(packaging) and (
        aesthetic or negative_visual or design or color_style
    )
    if gift_visual:
        return RuleDecision(
            "visual_strict",
            "gift_presentation_visual",
            "high",
            tuple(visual_terms),
            tuple(exclusion_terms),
        )

    # Negative/distinctive packaging appearance is useful affective evidence.
    negative_linked = bool(negative_visual) and bool(packaging) and has_near_pair(
        text,
        NEGATIVE_VISUAL,
        PACKAGING_OBJECTS,
        max_chars=80,
    )
    if negative_linked:
        return RuleDecision(
            "visual_strict",
            "negative_or_distinctive_visual_appearance",
            "high",
            tuple(visual_terms),
            tuple(exclusion_terms),
        )

    # Strong aesthetic adjective must be locally attached to a packaging object.
    aesthetic_linked = bool(aesthetic) and bool(packaging) and has_near_pair(
        text,
        STRONG_AESTHETIC,
        PACKAGING_OBJECTS,
        max_chars=65,
    )
    if aesthetic_linked:
        return RuleDecision(
            "visual_strict",
            "explicit_packaging_aesthetic",
            "high",
            tuple(visual_terms),
            tuple(exclusion_terms),
        )

    # A packaging design element without an explicit affective/style
    # evaluation is relevant but too weak for the strict visual corpus.
    if packaging and design:
        return RuleDecision(
            "uncertain",
            "insufficient_visual_evidence",
            "medium",
            tuple(visual_terms),
            tuple(exclusion_terms),
        )

    # Generic packaging praise/mention remains uncertain even when the
    # sentence also names the product contents. It is not strong enough for
    # visual_strict, but it is more informative than a hard exclusion.
    if packaging and weak:
        return RuleDecision(
            "uncertain",
            "insufficient_visual_evidence",
            "medium",
            tuple(visual_terms + weak),
            tuple(exclusion_terms),
        )

    # Tea content and functional attributes are excluded unless explicit local
    # packaging visual evidence was already captured above.
    if structural:
        return RuleDecision(
            "excluded",
            "structural_packaging",
            "high",
            tuple(visual_terms),
            tuple(exclusion_terms),
        )

    if product:
        return RuleDecision(
            "excluded",
            "product_content",
            "high",
            tuple(visual_terms),
            tuple(exclusion_terms),
        )

    if quantity:
        return RuleDecision(
            "excluded",
            "quantity_or_format",
            "high",
            tuple(visual_terms),
            tuple(exclusion_terms),
        )

    if shipping:
        return RuleDecision(
            "excluded",
            "shipping_damage",
            "high",
            tuple(visual_terms),
            tuple(exclusion_terms),
        )

    if order:
        return RuleDecision(
            "excluded",
            "order_fulfillment",
            "high",
            tuple(visual_terms),
            tuple(exclusion_terms),
        )

    if not packaging and not design:
        return RuleDecision(
            "excluded",
            "not_packaging",
            "high",
            tuple(visual_terms),
            tuple(exclusion_terms),
        )

    if packaging or design or weak:
        return RuleDecision(
            "uncertain",
            "insufficient_visual_evidence",
            "medium",
            tuple(visual_terms + weak),
            tuple(exclusion_terms),
        )

    return RuleDecision(
        "excluded",
        "not_packaging",
        "high",
        tuple(visual_terms),
        tuple(exclusion_terms),
    )


def iter_csv_rows(path: Path) -> Iterator[dict[str, Any]]:
    opener = gzip.open if path.name.lower().endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8-sig", newline="") as handle:
        yield from csv.DictReader(handle)


def iter_rows(path: Path, batch_size: int) -> Iterator[dict[str, Any]]:
    lower_name = path.name.lower()
    if lower_name.endswith(".csv") or lower_name.endswith(".csv.gz"):
        yield from iter_csv_rows(path)
        return

    if lower_name.endswith(".parquet"):
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise RuntimeError(
                "读取Parquet需要pyarrow。请运行：python -m pip install pyarrow"
            ) from exc
        parquet_file = pq.ParquetFile(path)
        for batch in parquet_file.iter_batches(batch_size=batch_size):
            for row in batch.to_pylist():
                yield row
        return

    raise ValueError("输入仅支持.parquet、.csv或.csv.gz。")


def normalize_base_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "parent_asin": normalize_text(row.get("parent_asin")),
        "review_id": normalize_text(row.get("review_id")),
        "review_fingerprint": normalize_text(row.get("review_fingerprint")),
        "sentence_id": normalize_text(row.get("sentence_id")),
        "sentence_index": parse_int(row.get("sentence_index")),
        "sentence_source": normalize_text(row.get("sentence_source")),
        "sentence": normalize_text(row.get("sentence")),
        "user_id": normalize_text(row.get("user_id")),
        "rating": parse_float(row.get("rating")),
        "review_date_utc": normalize_text(row.get("review_date_utc")),
        "verified_purchase": parse_bool(row.get("verified_purchase")),
        "helpful_vote": parse_int(row.get("helpful_vote")),
        "trigger_groups": normalize_text(row.get("trigger_groups")),
        "matched_trigger_terms": normalize_text(row.get("matched_trigger_terms")),
        "rule_based_context_hint": normalize_text(
            row.get("rule_based_context_hint")
        ),
    }


def create_writer(
    output_dir: Path,
    stem: str,
    output_format: str,
    batch_size: int,
) -> RowWriter:
    if output_format == "csv.gz":
        return CsvGzipWriter(output_dir / f"{stem}.csv.gz", CLASSIFIED_FIELDS)
    if output_format == "parquet":
        return ParquetWriter(
            output_dir / f"{stem}.parquet",
            CLASSIFIED_FIELDS,
            CLASSIFIED_SCHEMA,
            batch_size,
        )
    raise ValueError(f"不支持的输出格式: {output_format}")


def read_product_stats(
    path: Path,
) -> tuple[list[str], dict[str, ProductStats]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError("商品统计CSV缺少表头。")
        required = {"parent_asin"}
        missing = required - set(reader.fieldnames)
        if missing:
            raise ValueError(
                f"商品统计CSV缺少字段: {', '.join(sorted(missing))}"
            )

        order: list[str] = []
        result: dict[str, ProductStats] = {}
        duplicates: list[str] = []

        for row in reader:
            parent_asin = normalize_text(row.get("parent_asin"))
            if not parent_asin:
                raise ValueError("商品统计CSV存在空parent_asin。")
            if parent_asin in result:
                duplicates.append(parent_asin)
                continue
            order.append(parent_asin)
            result[parent_asin] = ProductStats(
                parent_asin=parent_asin,
                title=normalize_text(row.get("title")),
                main_image_url=normalize_text(row.get("main_image_url")),
            )

    if duplicates:
        raise ValueError(
            "商品统计CSV存在重复parent_asin: "
            + ", ".join(duplicates[:10])
        )
    return order, result


def write_csv(
    path: Path,
    fieldnames: list[str],
    rows: Iterable[dict[str, Any]],
) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {field: csv_value(row.get(field)) for field in fieldnames}
            )


def product_stats_row(stats: ProductStats) -> dict[str, Any]:
    ratio = (
        stats.visual_strict_sentence_count
        / stats.input_candidate_sentence_count
        if stats.input_candidate_sentence_count
        else None
    )
    count = stats.visual_strict_sentence_count
    return {
        "parent_asin": stats.parent_asin,
        "title": stats.title,
        "main_image_url": stats.main_image_url,
        "input_candidate_sentence_count": stats.input_candidate_sentence_count,
        "visual_strict_sentence_count": count,
        "visual_strict_review_count": len(stats.visual_reviews),
        "visual_strict_reviewer_count": len(stats.visual_reviewers),
        "excluded_sentence_count": stats.excluded_sentence_count,
        "uncertain_sentence_count": stats.uncertain_sentence_count,
        "visual_strict_ratio": f"{ratio:.6f}" if ratio is not None else "",
        "eligible_visual_ge_1": int(count >= 1),
        "eligible_visual_ge_2": int(count >= 2),
        "eligible_visual_ge_3": int(count >= 3),
        "eligible_visual_ge_5": int(count >= 5),
    }


def reservoir_add(
    reservoir: list[dict[str, Any]],
    row: dict[str, Any],
    seen_count: int,
    sample_size: int,
    random_state: random.Random,
) -> None:
    if sample_size <= 0:
        return
    if len(reservoir) < sample_size:
        reservoir.append(row)
        return
    replacement_index = random_state.randint(0, seen_count - 1)
    if replacement_index < sample_size:
        reservoir[replacement_index] = row


def ensure_output_paths(
    output_dir: Path,
    output_format: str,
    overwrite: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    extension = "parquet" if output_format == "parquet" else "csv.gz"
    paths = [
        output_dir / f"15_packaging_sentences_rule_classified.{extension}",
        output_dir / f"16_visual_packaging_sentences_strict.{extension}",
        output_dir / "17_product_visual_packaging_stats.csv",
        output_dir / "18_strict_visual_classification_summary.json",
        output_dir / "19_strict_visual_classification_summary.csv",
        output_dir / "20_visual_strict_audit_sample.csv",
        output_dir / "21_uncertain_audit_sample.csv",
        output_dir / "22_reason_counts.csv",
    ]
    existing = [path for path in paths if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "输出文件已存在，请更换输出目录或添加--overwrite: "
            + ", ".join(path.name for path in existing)
        )
    if overwrite:
        for path in existing:
            path.unlink()


def run_classification(
    *,
    candidates_path: Path,
    product_stats_path: Path,
    output_dir: Path,
    output_format: str = "parquet",
    batch_size: int = 10_000,
    sample_size: int = 500,
    sample_seed: int = 42,
    progress_every: int = 10_000,
    overwrite: bool = False,
) -> ClassificationResult:
    candidates_path = candidates_path.resolve()
    product_stats_path = product_stats_path.resolve()
    output_dir = output_dir.resolve()

    if not candidates_path.is_file():
        raise FileNotFoundError(f"包装候选句文件不存在: {candidates_path}")
    if not product_stats_path.is_file():
        raise FileNotFoundError(f"商品统计文件不存在: {product_stats_path}")

    ensure_output_paths(output_dir, output_format, overwrite)
    product_order, stats_by_asin = read_product_stats(product_stats_path)

    classified_writer = create_writer(
        output_dir,
        "15_packaging_sentences_rule_classified",
        output_format,
        batch_size,
    )
    strict_writer = create_writer(
        output_dir,
        "16_visual_packaging_sentences_strict",
        output_format,
        batch_size,
    )

    input_count = 0
    visual_count = 0
    excluded_count = 0
    uncertain_count = 0
    reason_counts: Counter[str] = Counter()
    decision_counts: Counter[str] = Counter()

    random_state_visual = random.Random(sample_seed)
    random_state_uncertain = random.Random(sample_seed + 1)
    visual_sample: list[dict[str, Any]] = []
    uncertain_sample: list[dict[str, Any]] = []

    try:
        for raw_row in iter_rows(candidates_path, batch_size):
            input_count += 1
            row = normalize_base_row(raw_row)

            parent_asin = row["parent_asin"]
            if parent_asin not in stats_by_asin:
                raise ValueError(
                    "候选句出现商品统计表中不存在的parent_asin: "
                    f"{parent_asin}"
                )

            decision = classify_sentence(row["sentence"])
            output_row = {
                **row,
                "decision": decision.decision,
                "reason": decision.reason,
                "confidence": decision.confidence,
                "matched_visual_terms": "|".join(
                    decision.matched_visual_terms
                ),
                "matched_exclusion_terms": "|".join(
                    decision.matched_exclusion_terms
                ),
                "strict_rule_version": RULE_VERSION,
            }

            classified_writer.write(output_row)

            stats = stats_by_asin[parent_asin]
            stats.input_candidate_sentence_count += 1
            reason_counts[decision.reason] += 1
            decision_counts[decision.decision] += 1

            sample_row = {
                "sample_order": "",
                "parent_asin": parent_asin,
                "review_id": row["review_id"],
                "sentence_id": row["sentence_id"],
                "sentence": row["sentence"],
                "rating": row["rating"],
                "verified_purchase": row["verified_purchase"],
                "helpful_vote": row["helpful_vote"],
                "decision": decision.decision,
                "reason": decision.reason,
                "matched_visual_terms": "|".join(
                    decision.matched_visual_terms
                ),
                "matched_exclusion_terms": "|".join(
                    decision.matched_exclusion_terms
                ),
            }

            if decision.decision == "visual_strict":
                strict_writer.write(output_row)
                visual_count += 1
                stats.visual_strict_sentence_count += 1
                if row["review_id"]:
                    stats.visual_reviews.add(row["review_id"])
                if row["user_id"]:
                    stats.visual_reviewers.add(row["user_id"])
                reservoir_add(
                    visual_sample,
                    sample_row,
                    visual_count,
                    sample_size,
                    random_state_visual,
                )
            elif decision.decision == "excluded":
                excluded_count += 1
                stats.excluded_sentence_count += 1
            else:
                uncertain_count += 1
                stats.uncertain_sentence_count += 1
                reservoir_add(
                    uncertain_sample,
                    sample_row,
                    uncertain_count,
                    sample_size,
                    random_state_uncertain,
                )

            if progress_every and input_count % progress_every == 0:
                print(
                    f"已处理 {input_count:,} 条候选句；"
                    f"严格视觉 {visual_count:,}；"
                    f"排除 {excluded_count:,}；"
                    f"不确定 {uncertain_count:,}。",
                    flush=True,
                )
    finally:
        classified_writer.close()
        strict_writer.close()

    stats_rows = [
        product_stats_row(stats_by_asin[parent_asin])
        for parent_asin in product_order
    ]
    write_csv(
        output_dir / "17_product_visual_packaging_stats.csv",
        PRODUCT_STATS_FIELDS,
        stats_rows,
    )

    for index, row in enumerate(visual_sample, start=1):
        row["sample_order"] = index
    write_csv(
        output_dir / "20_visual_strict_audit_sample.csv",
        SAMPLE_FIELDS,
        visual_sample,
    )

    for index, row in enumerate(uncertain_sample, start=1):
        row["sample_order"] = index
    write_csv(
        output_dir / "21_uncertain_audit_sample.csv",
        SAMPLE_FIELDS,
        uncertain_sample,
    )

    reason_rows = [
        {
            "reason": reason,
            "count": count,
            "percentage_of_all_candidates": (
                f"{count / input_count:.6f}" if input_count else ""
            ),
        }
        for reason, count in sorted(
            reason_counts.items(),
            key=lambda item: (-item[1], item[0]),
        )
    ]
    write_csv(
        output_dir / "22_reason_counts.csv",
        ["reason", "count", "percentage_of_all_candidates"],
        reason_rows,
    )

    products_with_visual = sum(
        stats.visual_strict_sentence_count >= 1
        for stats in stats_by_asin.values()
    )
    product_thresholds = {
        str(threshold): sum(
            stats.visual_strict_sentence_count >= threshold
            for stats in stats_by_asin.values()
        )
        for threshold in [1, 2, 3, 5, 10, 20]
    }
    reviewer_thresholds = {
        str(threshold): sum(
            len(stats.visual_reviewers) >= threshold
            for stats in stats_by_asin.values()
        )
        for threshold in [1, 2, 3, 5]
    }

    summary = {
        "rule_version": RULE_VERSION,
        "candidate_input_path": str(candidates_path),
        "product_stats_input_path": str(product_stats_path),
        "output_format": output_format,
        "product_count": len(product_order),
        "input_candidate_sentence_count": input_count,
        "visual_strict_sentence_count": visual_count,
        "excluded_sentence_count": excluded_count,
        "uncertain_sentence_count": uncertain_count,
        "visual_strict_ratio": (
            visual_count / input_count if input_count else None
        ),
        "products_with_visual_strict": products_with_visual,
        "visual_strict_product_sentence_thresholds": product_thresholds,
        "visual_strict_product_reviewer_thresholds": reviewer_thresholds,
        "decision_counts": dict(decision_counts),
        "reason_counts": dict(reason_counts),
        "policy": (
            "high_precision_low_recall; only visual_strict proceeds to "
            "affective imagery discovery; uncertain is excluded from modeling"
        ),
        "classified_output": str(classified_writer.path),
        "visual_strict_output": str(strict_writer.path),
        "sample_seed": sample_seed,
        "visual_sample_written": len(visual_sample),
        "uncertain_sample_written": len(uncertain_sample),
    }

    with (output_dir / "18_strict_visual_classification_summary.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    flat_summary: list[dict[str, Any]] = []
    for key, value in summary.items():
        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                flat_summary.append(
                    {"metric": f"{key}.{sub_key}", "value": sub_value}
                )
        else:
            flat_summary.append({"metric": key, "value": value})
    write_csv(
        output_dir / "19_strict_visual_classification_summary.csv",
        ["metric", "value"],
        flat_summary,
    )

    return ClassificationResult(
        product_count=len(product_order),
        input_sentence_count=input_count,
        visual_strict_count=visual_count,
        excluded_count=excluded_count,
        uncertain_count=uncertain_count,
        products_with_visual_strict=products_with_visual,
        classified_path=classified_writer.path,
        strict_path=strict_writer.path,
    )



# ---------------------------------------------------------------------------
# v1.1 phrase-level classifier
#
# Root-cause corrections:
# 1. "packaged" as a verb is not treated as a packaging object.
# 2. Visual adjectives must participate in explicit phrase templates, rather
#    than merely appearing near a packaging word.
# 3. "pretty" used as an adverb (pretty steep/good) is not aesthetic evidence.
# 4. Personal storage jars/containers and seller fulfilment are excluded.
# 5. Product-content and quantity language cannot become visual evidence merely
#    because a box/package is mentioned in the same sentence.
# ---------------------------------------------------------------------------

_V11_PACK_NOUN = (
    r"(?:packaging|package|packages|box|boxes|carton|cartons|tin|tins|"
    r"canister|canisters|container|containers|jar|jars|pouch|pouches|"
    r"wrapper|wrappers|label|labels)"
)

_V11_STRONG_POSITIVE = (
    r"(?:beautiful|pretty(?!\s+(?:good|bad|steep|strong|much|well))|cute|"
    r"gorgeous|lovely|elegant|classy|luxurious|premium|sophisticated|"
    r"stylish|modern|vintage|retro|minimalist|attractive|appealing|"
    r"eye[- ]catching|colorful|colourful|cheerful|fancy)"
)

_V11_NEGATIVE = (
    r"(?:ugly|tacky|boring|clinical|plain|unappealing|cheap[- ]looking|"
    r"messy|rough|dated|old[- ]fashioned|incorrect|wrong)"
)

_V11_DESIGN = (
    r"(?:design|illustration|illustrations|artwork|graphic|graphics|logo|"
    r"font|typography|lettering|pattern|print|labeling|colouring|color scheme|colour scheme|"
    r"color palette|colour palette|branding|presentation)"
)

_V11_COLOR_STYLE = (
    r"(?:black|white|pink|green|gold|golden|red|blue|purple|yellow|orange|"
    r"brown|beige|cream|silver|pastel|dark|bright|floral|botanical|earthy)"
)

_V11_RE_PACK_NOUN = re.compile(rf"\b{_V11_PACK_NOUN}\b", re.I)
_V11_RE_PACKAGED_VERB = re.compile(r"\bpackaged\b|\bpackaging\b", re.I)

_V11_RE_SHIPPING = re.compile(
    r"\b(?:arriv(?:e|ed|es|ing)|shipping|delivery|delivered|shipper|"
    r"crushed|damaged|dented|torn|ripped|broken|"
    r"leak(?:ed|ing)|smashed|beat up|beaten up|squashed|bent|lost in transit|"
    r"packed neatly|packed carefully)\b",
    re.I,
)
_V11_RE_ORDER = re.compile(
    r"\b(?:ordered|received|only (?:received|got)|missing|wrong item|"
    r"wrong product|refund|returned|sent me|shipped me|shipped|seller|vendor|amazon|didn['’]?t receive|"
    r"not receive|came with the wrong)\b",
    re.I,
)
_V11_RE_PERSONAL_STORAGE = re.compile(
    r"\b(?:i|we|my|our)\b.{0,70}\b(?:put|keep|kept|store|stored|storing|"
    r"reuse|reusing|refill|refilling|transfer|divide|project)\b.{0,55}\b"
    r"(?:jar|container|canister|tin)\b|"
    r"\b(?:wish|wished|looking for|buy a|bought a|need a|use a)\b.{0,55}\b"
    r"(?:jar|container|canister|tin)\b|"
    r"\b(?:cookie jar|glass jar|clamp[- ]top jar|tea canister)\b|"
    r"\bstored\s+in\b.{0,35}\b(?:jar|container|canister|tin)\b|"
    r"\b(?:buy|bought)\b.{0,45}\b(?:jar|container|canister|tin|tins)\b"
    r".{0,55}\b(?:divide|store|transfer|refill|holiday gifts?)\b|"
    r"\b(?:jar|container|canister|tin)\b.{0,45}\b(?:for afterwards|"
    r"for my counter|for your counter|for the countertop|to store)\b",
    re.I,
)
_V11_RE_QUANTITY_PRICE = re.compile(
    r"\b(?:\d+\s*(?:count|ct|tea ?bags?|sachets?|packets?|boxes)|"
    r"pack of \d+|box of \d+|bulk box|larger package|smaller package|"
    r"one box|two boxes|three boxes|four boxes|five boxes|six boxes|"
    r"price|priced|value|expensive|money friendly|cost[- ]efficient|"
    r"pretty steep|too much for|half empty|1/2 empty)\b",
    re.I,
)
_V11_RE_STRUCTURAL = re.compile(
    r"\b(?:resealable|sealed|zip(?:per|lock)?|airtight|foil|"
    r"individually wrapped|individual wrappers?|for freshness|"
    r"keeps? (?:it|them|the tea)? ?fresh|easy to open|hard to open|"
    r"difficult to open|lid does not close|loose lid|leakproof|"
    r"no staples|no strings|extra paper)\b",
    re.I,
)
_V11_RE_PRODUCT_CONTENT = re.compile(
    r"\b(?:taste|tastes|tasted|tasting|flavor|flavour|aroma|smell|smells|"
    r"brew|brewing|steep|steeping|cup|liquid|water|ingredients|leaves|"
    r"flowers|petals|buds|sleep|stomach|stress|caffeine|decaf|"
    r"health benefits|medicinal|soothing|calming sensation|detox|tea is pretty good|tea is good|tea is great|"
    r"quality bags|tea bags themselves|tea bag design)\b",
    re.I,
)
_V11_RE_PRODUCT_TITLE = re.compile(
    r"\b(?:premium|original)\s+(?:tea|licorice|rooibos|herbal|chamomile|"
    r"peppermint|spearmint|fennel|ginger)\b",
    re.I,
)

_V11_RE_CONTENT_VISUAL_FALSE = re.compile(
    r"\b(?:flowers|petals|buds|leaves|herbs|tea|brew|cup|liquid|drink|"
    r"ingredients)\b.{0,65}\b(?:package|box|pouch|container)\b"
    r".{0,35}\b(?:beautiful|pretty|colorful|colourful|lovely)\b|"
    r"\b(?:beautiful|pretty|colorful|colourful|lovely)\s+"
    r"(?:flowers|petals|buds|leaves|herbs|tea|brew|cup|liquid|drink|"
    r"ingredients)\b",
    re.I,
)
_V11_RE_GENERIC_PACKAGED = re.compile(
    r"\b(?:beautifully|nicely|well|wonderfully|thoughtfully|carefully)\s+"
    r"packaged\b|\bpackaged\s+(?:beautifully|nicely|well|wonderfully|"
    r"thoughtfully|carefully)\b",
    re.I,
)
_V11_RE_GENERIC_PRAISE = re.compile(
    rf"\b(?:awesome|great|good|nice)\s+{_V11_PACK_NOUN}\b|"
    rf"\b{_V11_PACK_NOUN}\b\s+(?:is|was|looks?|looked)?\s*"
    r"(?:awesome|great|good|nice)\b",
    re.I,
)
_V11_RE_NEGATED_OBJECT = re.compile(
    rf"\b(?:not|isn['’]?t|wasn['’]?t|without|no)\b.{{0,30}}"
    rf"\b(?:{_V11_STRONG_POSITIVE}|{_V11_DESIGN})\b.{{0,25}}"
    rf"\b{_V11_PACK_NOUN}\b",
    re.I,
)

# Explicit package noun modified by a strong visual adjective.
_V11_RE_ADJ_BEFORE_NOUN = re.compile(
    rf"\b({_V11_STRONG_POSITIVE})\b(?:\s+(?:little|small|large|metal|paper|"
    r"cardboard|glass|clear|tea|gift|pyramid|shaped)){0,3}\s+"
    rf"\b({_V11_PACK_NOUN})\b",
    re.I,
)

# Package noun as grammatical subject followed by a copular/look predicate.
_V11_RE_NOUN_PREDICATE_POS = re.compile(
    rf"\b({_V11_PACK_NOUN})\b(?:\s+(?:it|itself|they|themselves))?"
    r"(?:\s+\w+){0,2}\s+(?:is|are|was|were|looks?|looked|seems?|felt)\s+"
    r"(?:very\s+|so\s+|super\s+|really\s+|absolutely\s+|rather\s+)?"
    rf"\b({_V11_STRONG_POSITIVE})\b",
    re.I,
)
_V11_RE_NOUN_PREDICATE_NEG = re.compile(
    rf"\b({_V11_PACK_NOUN})\b(?:\s+(?:it|itself|they|themselves))?"
    r"(?:\s+\w+){0,2}\s+(?:is|are|was|were|looks?|looked|seems?|felt|"
    r"could be)\s+(?:very\s+|so\s+|super\s+|really\s+|rather\s+)?"
    rf"\b({_V11_NEGATIVE}|cooler looking)\b",
    re.I,
)

# Very short fragments such as "Pretty box" or "Beautiful packaging".
_V11_RE_SHORT_VISUAL = re.compile(
    rf"^\s*(?:the\s+)?(?:little\s+|small\s+)?"
    rf"(?:({_V11_STRONG_POSITIVE})\s+({_V11_PACK_NOUN})|"
    rf"({_V11_PACK_NOUN})\s+(?:is\s+|was\s+)?({_V11_STRONG_POSITIVE}))"
    r"\s*[.!,:;-]*\s*$",
    re.I,
)

# Explicit design element with an evaluation.
_V11_RE_DESIGN_EVALUATED = re.compile(
    rf"(?:\b({_V11_DESIGN})\b(?:\s+\w+){{0,5}}\s+"
    rf"\b({_V11_STRONG_POSITIVE}|{_V11_NEGATIVE}|correct|incorrect|perky)\b|"
    rf"\b({_V11_STRONG_POSITIVE}|{_V11_NEGATIVE}|gold|golden|clear|clean)\b"
    rf"(?:\s+\w+){{0,4}}\s+\b({_V11_DESIGN})\b)",
    re.I,
)

_V11_RE_DESIGN_CORRECTNESS = re.compile(
    r"\b(?:logo|font|label|labeling|branding|typography)\b.{0,45}\b"
    r"(?:incorrect|wrong|not correct|isn['’]?t(?:\s+even)?\s+correct|aren['’]?t(?:\s+even)?\s+correct|"
    r"doesn['’]?t look right|don['’]?t look right)\b",
    re.I,
)

_V11_RE_DESIGN_PREDICATE = re.compile(
    rf"\b(?:label|labeling|logo|font|branding|typography|design|artwork|"
    rf"illustration)\b(?:\s+(?:and|or)\s+(?:label|labeling|logo|font|"
    rf"branding|typography|design|artwork|illustration))?"
    rf"(?:\s+\w+){{0,2}}\s+(?:is|are|was|were|looks?|looked)\s+"
    rf"(?:very\s+|so\s+|super[- ]|really\s+|rather\s+)?"
    rf"\b({_V11_STRONG_POSITIVE}|{_V11_NEGATIVE}|perky)\b",
    re.I,
)

# Packaging design element explicitly attached to the package.
_V11_RE_DESIGN_ON_PACKAGE = re.compile(
    rf"\b({_V11_DESIGN})\b(?:\s+\w+){{0,6}}\s+\b({_V11_PACK_NOUN})\b|"
    rf"\b({_V11_PACK_NOUN})\b(?:\s+\w+){{0,6}}\s+\b({_V11_DESIGN})\b",
    re.I,
)

# Color/style is strict only when coupled with an affective/design statement.
_V11_RE_COLOR_STYLE_VISUAL = re.compile(
    rf"(?:\b({_V11_STRONG_POSITIVE}|{_V11_NEGATIVE})\b(?:\s+\w+){{0,4}}\s+"
    rf"\b({_V11_COLOR_STYLE})\b(?:\s+\w+){{0,2}}\s+\b({_V11_PACK_NOUN})\b|"
    rf"\b({_V11_COLOR_STYLE})\b(?:\s+\w+){{0,2}}\s+\b({_V11_PACK_NOUN})\b"
    rf"(?:\s+\w+){{0,6}}\s+\b({_V11_STRONG_POSITIVE}|{_V11_NEGATIVE})\b)",
    re.I,
)

_V11_RE_GIFT_VISUAL = re.compile(
    rf"(?:\b({_V11_PACK_NOUN})\b(?:\s+\w+){{0,7}}\s+"
    r"\b(?:giftable|gift[- ]worthy|presentable|perfect gift|great gift|"
    r"nice gift)\b|"
    r"\b(?:giftable|gift[- ]worthy|presentable|perfect gift|great gift|"
    rf"nice gift)\b(?:\s+\w+){{0,7}}\s+\b({_V11_PACK_NOUN})\b)",
    re.I,
)

_V11_RE_PLAIN_CARD_BOX = re.compile(
    r"\bplain\s+(?:cardboard\s+|paper\s+)?box\b",
    re.I,
)
_V11_RE_COOLER_LOOKING = re.compile(
    r"\bbox\b.{0,20}\bcooler looking\b",
    re.I,
)


def _v11_terms(text: str, pattern: re.Pattern[str]) -> tuple[str, ...]:
    matches = []
    for match in pattern.finditer(text):
        value = match.group(0).strip()
        if value:
            matches.append(value)
    return tuple(sorted(set(matches)))


def classify_sentence(sentence: str) -> RuleDecision:
    """High-precision v1.1 phrase-level visual packaging classification."""
    text = normalize_text(sentence)
    if not text:
        return RuleDecision(
            "excluded", "not_packaging", "high", (), ("empty_sentence",)
        )

    has_pack_noun = bool(_V11_RE_PACK_NOUN.search(text))
    visual_hits: list[str] = []
    exclusion_hits: list[str] = []

    # Strong exclusions precede all visual rules.
    if _V11_RE_SHIPPING.search(text):
        exclusion_hits.extend(_v11_terms(text, _V11_RE_SHIPPING))
        return RuleDecision(
            "excluded",
            "shipping_damage",
            "high",
            (),
            tuple(exclusion_hits),
        )

    if _V11_RE_ORDER.search(text) and has_pack_noun:
        exclusion_hits.extend(_v11_terms(text, _V11_RE_ORDER))
        return RuleDecision(
            "excluded",
            "order_fulfillment",
            "high",
            (),
            tuple(exclusion_hits),
        )

    if _V11_RE_PERSONAL_STORAGE.search(text):
        exclusion_hits.extend(_v11_terms(text, _V11_RE_PERSONAL_STORAGE))
        return RuleDecision(
            "excluded",
            "personal_storage_or_accessory",
            "high",
            (),
            tuple(exclusion_hits),
        )

    if _V11_RE_NEGATED_OBJECT.search(text):
        exclusion_hits.extend(_v11_terms(text, _V11_RE_NEGATED_OBJECT))
        return RuleDecision(
            "excluded",
            "not_packaging",
            "high",
            (),
            tuple(exclusion_hits),
        )

    if _V11_RE_CONTENT_VISUAL_FALSE.search(text):
        exclusion_hits.extend(_v11_terms(text, _V11_RE_CONTENT_VISUAL_FALSE))
        return RuleDecision(
            "excluded",
            "product_content",
            "high",
            (),
            tuple(exclusion_hits),
        )

    # Clear phrase-level visual evidence.
    if _V11_RE_DESIGN_CORRECTNESS.search(text):
        visual_hits.extend(_v11_terms(text, _V11_RE_DESIGN_CORRECTNESS))
        return RuleDecision(
            "visual_strict",
            "explicit_design_element",
            "high",
            tuple(visual_hits),
            (),
        )

    if _V11_RE_DESIGN_PREDICATE.search(text):
        visual_hits.extend(_v11_terms(text, _V11_RE_DESIGN_PREDICATE))
        return RuleDecision(
            "visual_strict",
            "explicit_design_element",
            "high",
            tuple(visual_hits),
            (),
        )

    if _V11_RE_DESIGN_EVALUATED.search(text):
        visual_hits.extend(_v11_terms(text, _V11_RE_DESIGN_EVALUATED))
        return RuleDecision(
            "visual_strict",
            "explicit_design_element",
            "high",
            tuple(visual_hits),
            (),
        )

    if _V11_RE_NOUN_PREDICATE_NEG.search(text):
        visual_hits.extend(_v11_terms(text, _V11_RE_NOUN_PREDICATE_NEG))
        return RuleDecision(
            "visual_strict",
            "negative_visual_appearance",
            "high",
            tuple(visual_hits),
            (),
        )

    if _V11_RE_PLAIN_CARD_BOX.search(text) or _V11_RE_COOLER_LOOKING.search(text):
        pattern = (
            _V11_RE_PLAIN_CARD_BOX
            if _V11_RE_PLAIN_CARD_BOX.search(text)
            else _V11_RE_COOLER_LOOKING
        )
        visual_hits.extend(_v11_terms(text, pattern))
        return RuleDecision(
            "visual_strict",
            "negative_visual_appearance",
            "high",
            tuple(visual_hits),
            (),
        )

    if _V11_RE_GIFT_VISUAL.search(text) and (
        _V11_RE_ADJ_BEFORE_NOUN.search(text)
        or _V11_RE_NOUN_PREDICATE_POS.search(text)
        or re.search(r"\b(?:fancy|presentable|giftable|gift[- ]worthy)\b", text, re.I)
    ):
        visual_hits.extend(_v11_terms(text, _V11_RE_GIFT_VISUAL))
        return RuleDecision(
            "visual_strict",
            "gift_presentation_visual",
            "high",
            tuple(visual_hits),
            (),
        )

    if _V11_RE_COLOR_STYLE_VISUAL.search(text):
        visual_hits.extend(_v11_terms(text, _V11_RE_COLOR_STYLE_VISUAL))
        return RuleDecision(
            "visual_strict",
            "explicit_packaging_color_or_style",
            "high",
            tuple(visual_hits),
            (),
        )

    # Do not let "premium tea" or product-name language become package imagery.
    if _V11_RE_PRODUCT_TITLE.search(text) and not (
        _V11_RE_NOUN_PREDICATE_POS.search(text)
        or _V11_RE_ADJ_BEFORE_NOUN.search(text)
    ):
        exclusion_hits.extend(_v11_terms(text, _V11_RE_PRODUCT_TITLE))
        return RuleDecision(
            "excluded",
            "product_content",
            "high",
            (),
            tuple(exclusion_hits),
        )

    if _V11_RE_SHORT_VISUAL.search(text):
        visual_hits.extend(_v11_terms(text, _V11_RE_SHORT_VISUAL))
        return RuleDecision(
            "visual_strict",
            "explicit_packaging_aesthetic",
            "high",
            tuple(visual_hits),
            (),
        )

    if _V11_RE_NOUN_PREDICATE_POS.search(text):
        visual_hits.extend(_v11_terms(text, _V11_RE_NOUN_PREDICATE_POS))
        return RuleDecision(
            "visual_strict",
            "explicit_packaging_aesthetic",
            "high",
            tuple(visual_hits),
            (),
        )

    if _V11_RE_ADJ_BEFORE_NOUN.search(text):
        visual_hits.extend(_v11_terms(text, _V11_RE_ADJ_BEFORE_NOUN))
        return RuleDecision(
            "visual_strict",
            "explicit_packaging_aesthetic",
            "high",
            tuple(visual_hits),
            (),
        )

    # Design words without an evaluation are relevant but not strict enough.
    if _V11_RE_DESIGN_ON_PACKAGE.search(text):
        visual_hits.extend(_v11_terms(text, _V11_RE_DESIGN_ON_PACKAGE))
        return RuleDecision(
            "uncertain",
            "insufficient_visual_evidence",
            "medium",
            tuple(visual_hits),
            (),
        )

    # Generic "beautifully packaged" and "great packaging" do not prove that
    # the reviewer is describing visual design rather than handling or packing.
    if _V11_RE_GENERIC_PACKAGED.search(text) or _V11_RE_GENERIC_PRAISE.search(text):
        visual_hits.extend(_v11_terms(text, _V11_RE_GENERIC_PACKAGED))
        visual_hits.extend(_v11_terms(text, _V11_RE_GENERIC_PRAISE))
        return RuleDecision(
            "uncertain",
            "insufficient_visual_evidence",
            "medium",
            tuple(visual_hits),
            (),
        )

    # Hard content/structure/quantity exclusions after explicit visual rules,
    # allowing a genuine visual clause to survive a separate taste clause.
    if _V11_RE_STRUCTURAL.search(text):
        exclusion_hits.extend(_v11_terms(text, _V11_RE_STRUCTURAL))
        return RuleDecision(
            "excluded",
            "structural_packaging",
            "high",
            (),
            tuple(exclusion_hits),
        )

    if _V11_RE_QUANTITY_PRICE.search(text):
        exclusion_hits.extend(_v11_terms(text, _V11_RE_QUANTITY_PRICE))
        return RuleDecision(
            "excluded",
            "quantity_or_price",
            "high",
            (),
            tuple(exclusion_hits),
        )

    if _V11_RE_PRODUCT_CONTENT.search(text):
        exclusion_hits.extend(_v11_terms(text, _V11_RE_PRODUCT_CONTENT))
        return RuleDecision(
            "excluded",
            "product_content",
            "high",
            (),
            tuple(exclusion_hits),
        )

    if has_pack_noun or _V11_RE_PACKAGED_VERB.search(text):
        return RuleDecision(
            "uncertain",
            "insufficient_visual_evidence",
            "medium",
            (),
            (),
        )

    return RuleDecision("excluded", "not_packaging", "high", (), ())

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "使用高精度规则将包装候选句分为visual_strict、"
            "excluded和uncertain。"
        )
    )
    parser.add_argument(
        "--candidates",
        required=True,
        type=Path,
        help="10_packaging_sentence_candidates.parquet或CSV.GZ",
    )
    parser.add_argument(
        "--product-stats",
        required=True,
        type=Path,
        help="11_product_review_stats_clean.csv",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="输出目录",
    )
    parser.add_argument(
        "--output-format",
        choices=["parquet", "csv.gz"],
        default="parquet",
        help="明细输出格式；默认parquet",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10_000,
        help="Parquet分批读取及写出大小",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=500,
        help="严格视觉与不确定句分别抽取的审计样本数",
    )
    parser.add_argument(
        "--sample-seed",
        type=int,
        default=42,
        help="审计样本随机种子",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=10_000,
        help="每处理多少条候选句打印一次进度；0关闭",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="覆盖同名输出",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        result = run_classification(
            candidates_path=args.candidates,
            product_stats_path=args.product_stats,
            output_dir=args.output_dir,
            output_format=args.output_format,
            batch_size=args.batch_size,
            sample_size=args.sample_size,
            sample_seed=args.sample_seed,
            progress_every=args.progress_every,
            overwrite=args.overwrite,
        )
    except Exception as exc:
        print(f"运行失败: {exc}", file=sys.stderr)
        return 2

    print("=" * 72)
    print("严格视觉包装筛选完成")
    print(f"商品数: {result.product_count:,}")
    print(f"输入候选句: {result.input_sentence_count:,}")
    print(f"visual_strict: {result.visual_strict_count:,}")
    print(f"excluded: {result.excluded_count:,}")
    print(f"uncertain: {result.uncertain_count:,}")
    print(
        "至少有1条visual_strict的商品数: "
        f"{result.products_with_visual_strict:,}"
    )
    print(f"全部分类结果: {result.classified_path}")
    print(f"严格视觉语料: {result.strict_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
