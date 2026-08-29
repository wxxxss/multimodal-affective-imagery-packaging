#!/usr/bin/env python3
"""Clean matched Amazon reviews and extract high-recall packaging sentences.

This stage does NOT assign final visual-packaging labels. It:
- removes empty, duplicate, and preliminary non-English reviews;
- preserves all product IDs from the raw product statistics;
- splits review titles and bodies into traceable sentences;
- extracts high-recall packaging-related sentence candidates;
- records trigger groups for later semantic classification.

Supported input:
- .parquet (requires pyarrow)
- .csv.gz
- .csv

Supported output:
- parquet (requires pyarrow)
- csv.gz
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import html
import json
import math
import random
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Protocol


CLEAN_REVIEW_FIELDS = [
    "parent_asin",
    "asin",
    "review_id",
    "review_fingerprint",
    "user_id",
    "rating",
    "review_title",
    "review_text",
    "clean_text",
    "review_date_utc",
    "verified_purchase",
    "helpful_vote",
    "source_line",
]

SENTENCE_FIELDS = [
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
]

CANDIDATE_FIELDS = SENTENCE_FIELDS + [
    "trigger_groups",
    "matched_trigger_terms",
    "rule_based_context_hint",
]

PRODUCT_STATS_FIELDS = [
    "parent_asin",
    "title",
    "main_image_url",
    "raw_review_count",
    "input_review_rows_seen",
    "excluded_empty_review_count",
    "excluded_non_english_review_count",
    "excluded_duplicate_review_count",
    "clean_review_count",
    "clean_unique_reviewer_count",
    "clean_verified_review_count",
    "clean_verified_review_ratio",
    "clean_helpful_vote_sum",
    "clean_rating_count",
    "clean_rating_mean",
    "clean_rating_std",
    "clean_sentence_count",
    "packaging_candidate_review_count",
    "packaging_candidate_sentence_count",
    "packaging_candidate_reviewer_count",
    "candidate_sentence_from_positive_rating_review_count",
    "candidate_sentence_from_neutral_rating_review_count",
    "candidate_sentence_from_negative_rating_review_count",
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
    "trigger_groups",
    "matched_trigger_terms",
    "rule_based_context_hint",
]


TRIGGER_PATTERNS: dict[str, list[tuple[str, re.Pattern[str]]]] = {
    "direct_packaging": [
        ("packaging", re.compile(r"\bpackag(?:e|ed|es|ing)\b", re.I)),
        ("box", re.compile(r"\bbox(?:es)?\b", re.I)),
        ("carton", re.compile(r"\bcarton(?:s)?\b", re.I)),
        ("container", re.compile(r"\bcontainer(?:s)?\b", re.I)),
        ("pouch", re.compile(r"\bpouch(?:es)?\b", re.I)),
        ("packet", re.compile(r"\bpacket(?:s)?\b", re.I)),
        ("wrapper", re.compile(r"\bwrapp(?:er|ers|ed|ing)\b", re.I)),
        ("tin", re.compile(r"\btin(?:s)?\b", re.I)),
        ("canister", re.compile(r"\bcanister(?:s)?\b", re.I)),
        ("jar", re.compile(r"\bjar(?:s)?\b", re.I)),
        ("label", re.compile(r"\blabel(?:s|ed|ing)?\b", re.I)),
    ],
    "visual_design": [
        ("design", re.compile(r"\bdesign(?:s|ed|ing)?\b", re.I)),
        ("artwork", re.compile(r"\bart\s*work\b|\bartwork\b", re.I)),
        ("illustration", re.compile(r"\billustrat(?:ion|ions|ed|ive)\b", re.I)),
        ("graphic", re.compile(r"\bgraphic(?:s|al)?\b", re.I)),
        ("logo", re.compile(r"\blogo(?:s)?\b", re.I)),
        ("font", re.compile(r"\bfont(?:s)?\b", re.I)),
        ("typography", re.compile(r"\btypograph(?:y|ic|ical)\b", re.I)),
        ("color", re.compile(r"\bcolou?r(?:s|ed|ful|ing)?\b", re.I)),
        ("pattern", re.compile(r"\bpattern(?:s|ed)?\b", re.I)),
        ("print", re.compile(r"\bprint(?:s|ed|ing)?\b", re.I)),
        ("floral", re.compile(r"\bfloral\b", re.I)),
        ("botanical", re.compile(r"\bbotanical\b", re.I)),
        ("image", re.compile(r"\bimage(?:s)?\b", re.I)),
        ("picture", re.compile(r"\bpicture(?:s)?\b", re.I)),
    ],
    "appearance_imagery": [
        ("appearance", re.compile(r"\bappearance\b", re.I)),
        ("aesthetic", re.compile(r"\baesthetic(?:s|ally)?\b", re.I)),
        ("looks", re.compile(r"\blook(?:s|ed|ing)?\b", re.I)),
        ("beautiful", re.compile(r"\bbeautiful(?:ly)?\b", re.I)),
        ("pretty", re.compile(r"\bpretty\b", re.I)),
        ("cute", re.compile(r"\bcute\b", re.I)),
        ("elegant", re.compile(r"\belegant(?:ly)?\b", re.I)),
        ("premium", re.compile(r"\bpremium\b", re.I)),
        ("luxurious", re.compile(r"\bluxur(?:y|ious|iously)\b", re.I)),
        ("classy", re.compile(r"\bclassy\b", re.I)),
        ("sophisticated", re.compile(r"\bsophisticated\b", re.I)),
        ("modern", re.compile(r"\bmodern\b", re.I)),
        ("vintage", re.compile(r"\bvintage\b", re.I)),
        ("natural-looking", re.compile(r"\bnatural[- ]looking\b", re.I)),
        ("calming", re.compile(r"\bcalm(?:ing|ingly)?\b|\bsoothing\b", re.I)),
        ("clean", re.compile(r"\bclean[- ]looking\b|\bclean design\b", re.I)),
        ("simple", re.compile(r"\bsimple\b|\bminimal(?:ist|istic)?\b", re.I)),
        ("attractive", re.compile(r"\battractive\b|\bappealing\b", re.I)),
        ("cheap-looking", re.compile(r"\bcheap[- ]looking\b|\blooks? cheap\b", re.I)),
        ("cheerful", re.compile(r"\bcheerful\b|\bbright and cheerful\b", re.I)),
    ],
    "gift_presentation": [
        ("gift", re.compile(r"\bgift(?:s|ed|ing|able)?\b", re.I)),
        ("presentation", re.compile(r"\bpresentation\b|\bpresentable\b", re.I)),
        ("shelf", re.compile(r"\bon (?:my|the) (?:kitchen )?shelf\b|\bshelf[- ]appeal\b", re.I)),
    ],
    "structural_packaging": [
        ("sealed", re.compile(r"\bseal(?:ed|ing|s)?\b", re.I)),
        ("resealable", re.compile(r"\bresealable\b|\bre-sealable\b", re.I)),
        ("zipper", re.compile(r"\bzip(?:per|lock)?\b", re.I)),
        ("foil", re.compile(r"\bfoil\b", re.I)),
        ("individually wrapped", re.compile(r"\bindividually (?:wrapped|packaged)\b", re.I)),
        ("lid", re.compile(r"\blid(?:s)?\b", re.I)),
        ("open-close", re.compile(r"\beasy to (?:open|close)\b|\bhard to (?:open|close)\b", re.I)),
    ],
    "shipping_damage": [
        ("shipping", re.compile(r"\bshipping\b|\bdelivery\b|\bdelivered\b", re.I)),
        ("arrived", re.compile(r"\barriv(?:e|ed|es|ing)\b", re.I)),
        ("crushed", re.compile(r"\bcrush(?:ed|ing)?\b", re.I)),
        ("damaged", re.compile(r"\bdamag(?:e|ed|ing)\b", re.I)),
        ("dented", re.compile(r"\bdent(?:ed|s)?\b", re.I)),
        ("broken", re.compile(r"\bbrok(?:e|en)\b", re.I)),
        ("leaking", re.compile(r"\bleak(?:ed|ing|s)?\b", re.I)),
        ("torn", re.compile(r"\btorn\b|\btear(?:s|ing)?\b", re.I)),
    ],
}


@dataclass
class CleaningResult:
    product_count: int
    input_review_count: int
    clean_review_count: int
    excluded_empty_count: int
    excluded_non_english_count: int
    excluded_duplicate_count: int
    sentence_count: int
    packaging_candidate_sentence_count: int
    products_with_packaging_candidates: int
    clean_reviews_path: Path
    sentences_path: Path
    candidates_path: Path


@dataclass
class RunningNumericStats:
    count: int = 0
    mean_value: float = 0.0
    m2: float = 0.0

    def add(self, value: float | None) -> None:
        if value is None or not math.isfinite(value):
            return
        self.count += 1
        delta = value - self.mean_value
        self.mean_value += delta / self.count
        delta2 = value - self.mean_value
        self.m2 += delta * delta2

    @property
    def std(self) -> float | None:
        if self.count < 2:
            return None
        return math.sqrt(self.m2 / (self.count - 1))


@dataclass
class ProductCleanStats:
    parent_asin: str
    title: str
    main_image_url: str
    raw_review_count: int
    input_review_rows_seen: int = 0
    excluded_empty_review_count: int = 0
    excluded_non_english_review_count: int = 0
    excluded_duplicate_review_count: int = 0
    clean_review_count: int = 0
    clean_reviewers: set[str] = field(default_factory=set)
    clean_verified_review_count: int = 0
    clean_helpful_vote_sum: int = 0
    ratings: RunningNumericStats = field(default_factory=RunningNumericStats)
    clean_sentence_count: int = 0
    packaging_candidate_reviews: set[str] = field(default_factory=set)
    packaging_candidate_sentence_count: int = 0
    packaging_candidate_reviewers: set[str] = field(default_factory=set)
    candidate_positive_count: int = 0
    candidate_neutral_count: int = 0
    candidate_negative_count: int = 0


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


CLEAN_SCHEMA = [
    ("parent_asin", "string"),
    ("asin", "string"),
    ("review_id", "string"),
    ("review_fingerprint", "string"),
    ("user_id", "string"),
    ("rating", "float64"),
    ("review_title", "string"),
    ("review_text", "string"),
    ("clean_text", "string"),
    ("review_date_utc", "string"),
    ("verified_purchase", "bool"),
    ("helpful_vote", "int64"),
    ("source_line", "int64"),
]

SENTENCE_SCHEMA = [
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
]

CANDIDATE_SCHEMA = SENTENCE_SCHEMA + [
    ("trigger_groups", "string"),
    ("matched_trigger_terms", "string"),
    ("rule_based_context_hint", "string"),
]


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value)
    text = html.unescape(text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = text.replace("\x00", " ")
    text = re.sub(r"[\u200b\u200c\u200d\ufeff]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


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
        parsed = float(str(value).replace(",", "").strip())
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def csv_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "1" if value else "0"
    return str(value)


def is_probably_english(text: str) -> bool:
    text = normalize_text(text)
    if not text:
        return False
    alphabetic = [char for char in text if char.isalpha()]
    if len(alphabetic) < 3:
        return False
    latin = sum(
        ("A" <= char <= "Z") or ("a" <= char <= "z")
        for char in alphabetic
    )
    return latin / len(alphabetic) >= 0.80


def split_sentences(text: str) -> list[str]:
    """Deterministic lightweight English sentence splitting."""
    text = normalize_text(text)
    if not text:
        return []

    protected = text
    replacements = {
        "e.g.": "e§g§",
        "i.e.": "i§e§",
        "Mr.": "Mr§",
        "Mrs.": "Mrs§",
        "Ms.": "Ms§",
        "Dr.": "Dr§",
        "U.S.": "U§S§",
        "oz.": "oz§",
        "lb.": "lb§",
    }
    for original, placeholder in replacements.items():
        protected = protected.replace(original, placeholder)

    parts = re.split(
        r"(?<=[.!?])\s+(?=[\"'“‘(\[]?[A-Z0-9])|[\r\n]+",
        protected,
    )

    sentences = []
    for part in parts:
        restored = part
        for original, placeholder in replacements.items():
            restored = restored.replace(placeholder, original)
        restored = normalize_text(restored)
        if restored:
            sentences.append(restored)
    return sentences


def packaging_trigger_details(sentence: str) -> tuple[list[str], list[str]]:
    groups: list[str] = []
    terms: list[str] = []

    for group, patterns in TRIGGER_PATTERNS.items():
        group_terms = []
        for label, pattern in patterns:
            if pattern.search(sentence):
                group_terms.append(label)
        if group_terms:
            groups.append(group)
            terms.extend(group_terms)

    return groups, sorted(set(terms))


def packaging_trigger_groups(sentence: str) -> list[str]:
    return packaging_trigger_details(sentence)[0]


def context_hint(groups: list[str]) -> str:
    group_set = set(groups)
    visual = bool(
        group_set
        & {"visual_design", "appearance_imagery", "gift_presentation"}
    )
    shipping = "shipping_damage" in group_set
    structural = "structural_packaging" in group_set

    active = sum([visual, shipping, structural])
    if active >= 2:
        return "mixed"
    if visual:
        return "visual_likely"
    if shipping:
        return "shipping_or_damage_likely"
    if structural:
        return "structural_likely"
    return "packaging_general"


def make_sentence_id(review_id: str, sentence_index: int, sentence: str) -> str:
    material = f"{review_id}\x1f{sentence_index}\x1f{sentence}"
    return hashlib.sha1(material.encode("utf-8")).hexdigest()


def make_fallback_fingerprint(row: dict[str, Any]) -> str:
    material = "\x1f".join(
        [
            normalize_text(row.get("parent_asin")),
            normalize_text(row.get("user_id")),
            normalize_text(row.get("review_date_utc")),
            normalize_text(row.get("review_title")).lower(),
            normalize_text(row.get("review_text")).lower(),
        ]
    )
    return hashlib.sha1(material.encode("utf-8")).hexdigest()


def iter_csv_rows(path: Path) -> Iterator[dict[str, Any]]:
    opener = gzip.open if path.suffix.lower() == ".gz" else open
    with opener(path, "rt", encoding="utf-8-sig", newline="") as handle:
        yield from csv.DictReader(handle)


def iter_review_rows(path: Path, batch_size: int) -> Iterator[dict[str, Any]]:
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

    raise ValueError("评论输入仅支持.parquet、.csv或.csv.gz。")


def create_writer(
    output_dir: Path,
    stem: str,
    fieldnames: list[str],
    schema: list[tuple[str, str]],
    output_format: str,
    batch_size: int,
) -> RowWriter:
    if output_format == "csv.gz":
        return CsvGzipWriter(output_dir / f"{stem}.csv.gz", fieldnames)
    if output_format == "parquet":
        return ParquetWriter(
            output_dir / f"{stem}.parquet",
            fieldnames,
            schema,
            batch_size,
        )
    raise ValueError(f"不支持的输出格式: {output_format}")


def read_product_stats(path: Path) -> tuple[list[str], dict[str, ProductCleanStats]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"parent_asin", "raw_review_count"}
        if not reader.fieldnames:
            raise ValueError("商品统计CSV缺少表头。")
        missing = required - set(reader.fieldnames)
        if missing:
            raise ValueError(f"商品统计CSV缺少字段: {', '.join(sorted(missing))}")

        order: list[str] = []
        stats: dict[str, ProductCleanStats] = {}
        duplicates: list[str] = []

        for row in reader:
            parent_asin = normalize_text(row.get("parent_asin"))
            if not parent_asin:
                raise ValueError("商品统计CSV存在空parent_asin。")
            if parent_asin in stats:
                duplicates.append(parent_asin)
                continue
            order.append(parent_asin)
            stats[parent_asin] = ProductCleanStats(
                parent_asin=parent_asin,
                title=normalize_text(row.get("title")),
                main_image_url=normalize_text(row.get("main_image_url")),
                raw_review_count=parse_int(row.get("raw_review_count")),
            )

    if duplicates:
        preview = ", ".join(duplicates[:10])
        raise ValueError(f"商品统计CSV存在重复parent_asin: {preview}")
    return order, stats


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


def product_stats_row(stats: ProductCleanStats) -> dict[str, Any]:
    verified_ratio = (
        stats.clean_verified_review_count / stats.clean_review_count
        if stats.clean_review_count
        else None
    )
    return {
        "parent_asin": stats.parent_asin,
        "title": stats.title,
        "main_image_url": stats.main_image_url,
        "raw_review_count": stats.raw_review_count,
        "input_review_rows_seen": stats.input_review_rows_seen,
        "excluded_empty_review_count": stats.excluded_empty_review_count,
        "excluded_non_english_review_count": stats.excluded_non_english_review_count,
        "excluded_duplicate_review_count": stats.excluded_duplicate_review_count,
        "clean_review_count": stats.clean_review_count,
        "clean_unique_reviewer_count": len(stats.clean_reviewers),
        "clean_verified_review_count": stats.clean_verified_review_count,
        "clean_verified_review_ratio": (
            f"{verified_ratio:.6f}" if verified_ratio is not None else ""
        ),
        "clean_helpful_vote_sum": stats.clean_helpful_vote_sum,
        "clean_rating_count": stats.ratings.count,
        "clean_rating_mean": (
            f"{stats.ratings.mean_value:.4f}" if stats.ratings.count else ""
        ),
        "clean_rating_std": (
            f"{stats.ratings.std:.4f}" if stats.ratings.std is not None else ""
        ),
        "clean_sentence_count": stats.clean_sentence_count,
        "packaging_candidate_review_count": len(
            stats.packaging_candidate_reviews
        ),
        "packaging_candidate_sentence_count": (
            stats.packaging_candidate_sentence_count
        ),
        "packaging_candidate_reviewer_count": len(
            stats.packaging_candidate_reviewers
        ),
        "candidate_sentence_from_positive_rating_review_count": (
            stats.candidate_positive_count
        ),
        "candidate_sentence_from_neutral_rating_review_count": (
            stats.candidate_neutral_count
        ),
        "candidate_sentence_from_negative_rating_review_count": (
            stats.candidate_negative_count
        ),
    }


def ensure_output_paths(
    output_dir: Path,
    output_format: str,
    overwrite: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    extension = "parquet" if output_format == "parquet" else "csv.gz"
    paths = [
        output_dir / f"08_clean_reviews.{extension}",
        output_dir / f"09_review_sentences.{extension}",
        output_dir / f"10_packaging_sentence_candidates.{extension}",
        output_dir / "11_product_review_stats_clean.csv",
        output_dir / "12_review_cleaning_summary.json",
        output_dir / "13_review_cleaning_summary.csv",
        output_dir / "14_packaging_candidate_sample.csv",
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


def run_cleaning(
    *,
    reviews_path: Path,
    raw_stats_path: Path,
    output_dir: Path,
    batch_size: int = 10_000,
    sample_size: int = 500,
    sample_seed: int = 42,
    output_format: str = "parquet",
    overwrite: bool = False,
    progress_every: int = 25_000,
) -> CleaningResult:
    reviews_path = reviews_path.resolve()
    raw_stats_path = raw_stats_path.resolve()
    output_dir = output_dir.resolve()

    if not reviews_path.is_file():
        raise FileNotFoundError(f"评论文件不存在: {reviews_path}")
    if not raw_stats_path.is_file():
        raise FileNotFoundError(f"商品统计文件不存在: {raw_stats_path}")

    ensure_output_paths(output_dir, output_format, overwrite)
    product_order, stats_by_asin = read_product_stats(raw_stats_path)

    clean_writer = create_writer(
        output_dir,
        "08_clean_reviews",
        CLEAN_REVIEW_FIELDS,
        CLEAN_SCHEMA,
        output_format,
        batch_size,
    )
    sentence_writer = create_writer(
        output_dir,
        "09_review_sentences",
        SENTENCE_FIELDS,
        SENTENCE_SCHEMA,
        output_format,
        batch_size,
    )
    candidate_writer = create_writer(
        output_dir,
        "10_packaging_sentence_candidates",
        CANDIDATE_FIELDS,
        CANDIDATE_SCHEMA,
        output_format,
        batch_size,
    )

    seen_fingerprints: set[str] = set()
    input_review_count = 0
    clean_review_count = 0
    excluded_empty_count = 0
    excluded_non_english_count = 0
    excluded_duplicate_count = 0
    sentence_count = 0
    candidate_sentence_count = 0

    random_state = random.Random(sample_seed)
    candidate_sample: list[dict[str, Any]] = []

    try:
        for row in iter_review_rows(reviews_path, batch_size):
            input_review_count += 1

            parent_asin = normalize_text(row.get("parent_asin"))
            if parent_asin not in stats_by_asin:
                raise ValueError(
                    f"评论出现商品统计表中不存在的parent_asin: {parent_asin}"
                )
            product_stats = stats_by_asin[parent_asin]
            product_stats.input_review_rows_seen += 1

            review_title = normalize_text(row.get("review_title"))
            review_text = normalize_text(row.get("review_text"))
            combined_text = normalize_text(
                row.get("combined_text")
                or f"{review_title} {review_text}"
            )
            nonempty = parse_bool(row.get("is_nonempty")) if (
                row.get("is_nonempty") is not None
            ) else bool(combined_text)
            if not nonempty or not combined_text:
                product_stats.excluded_empty_review_count += 1
                excluded_empty_count += 1
                continue

            english = parse_bool(row.get("is_probably_english")) if (
                row.get("is_probably_english") is not None
            ) else is_probably_english(combined_text)
            if not english:
                product_stats.excluded_non_english_review_count += 1
                excluded_non_english_count += 1
                continue

            fingerprint = normalize_text(row.get("review_fingerprint"))
            if not fingerprint:
                fingerprint = make_fallback_fingerprint(row)

            duplicate = (
                parse_bool(row.get("is_duplicate_fingerprint"))
                or fingerprint in seen_fingerprints
            )
            if duplicate:
                product_stats.excluded_duplicate_review_count += 1
                excluded_duplicate_count += 1
                continue
            seen_fingerprints.add(fingerprint)

            review_id = normalize_text(row.get("review_id")) or fingerprint
            user_id = normalize_text(row.get("user_id"))
            rating = parse_float(row.get("rating"))
            verified = parse_bool(row.get("verified_purchase"))
            helpful_vote = parse_int(row.get("helpful_vote"))
            review_date = normalize_text(row.get("review_date_utc"))
            source_line = parse_int(row.get("source_line"))

            clean_text = normalize_text(
                "\n".join(part for part in [review_title, review_text] if part)
            )
            clean_row = {
                "parent_asin": parent_asin,
                "asin": normalize_text(row.get("asin")),
                "review_id": review_id,
                "review_fingerprint": fingerprint,
                "user_id": user_id,
                "rating": rating,
                "review_title": review_title,
                "review_text": review_text,
                "clean_text": clean_text,
                "review_date_utc": review_date,
                "verified_purchase": verified,
                "helpful_vote": helpful_vote,
                "source_line": source_line,
            }
            clean_writer.write(clean_row)

            product_stats.clean_review_count += 1
            if user_id:
                product_stats.clean_reviewers.add(user_id)
            if verified:
                product_stats.clean_verified_review_count += 1
            product_stats.clean_helpful_vote_sum += helpful_vote
            product_stats.ratings.add(rating)
            clean_review_count += 1

            review_has_candidate = False

            sentence_sources: list[tuple[str, str]] = []
            if review_title:
                sentence_sources.append(("title", review_title))
            if review_text:
                for sentence in split_sentences(review_text):
                    sentence_sources.append(("text", sentence))

            # Avoid exact duplicate title/body sentences within one review.
            seen_sentences_in_review: set[str] = set()
            normalized_sentence_index = 0

            for sentence_source, sentence in sentence_sources:
                sentence = normalize_text(sentence)
                sentence_key = sentence.lower()
                if not sentence or sentence_key in seen_sentences_in_review:
                    continue
                seen_sentences_in_review.add(sentence_key)

                sentence_id = make_sentence_id(
                    review_id,
                    normalized_sentence_index,
                    sentence,
                )
                sentence_row = {
                    "parent_asin": parent_asin,
                    "review_id": review_id,
                    "review_fingerprint": fingerprint,
                    "sentence_id": sentence_id,
                    "sentence_index": normalized_sentence_index,
                    "sentence_source": sentence_source,
                    "sentence": sentence,
                    "user_id": user_id,
                    "rating": rating,
                    "review_date_utc": review_date,
                    "verified_purchase": verified,
                    "helpful_vote": helpful_vote,
                }
                sentence_writer.write(sentence_row)
                sentence_count += 1
                product_stats.clean_sentence_count += 1

                groups, terms = packaging_trigger_details(sentence)
                if groups:
                    hint = context_hint(groups)
                    candidate_row = {
                        **sentence_row,
                        "trigger_groups": "|".join(groups),
                        "matched_trigger_terms": "|".join(terms),
                        "rule_based_context_hint": hint,
                    }
                    candidate_writer.write(candidate_row)
                    candidate_sentence_count += 1
                    product_stats.packaging_candidate_sentence_count += 1
                    product_stats.packaging_candidate_reviews.add(review_id)
                    if user_id:
                        product_stats.packaging_candidate_reviewers.add(user_id)
                    review_has_candidate = True

                    if rating is not None:
                        if rating >= 4:
                            product_stats.candidate_positive_count += 1
                        elif rating <= 2:
                            product_stats.candidate_negative_count += 1
                        else:
                            product_stats.candidate_neutral_count += 1

                    sample_row = {
                        "sample_order": "",
                        "parent_asin": parent_asin,
                        "review_id": review_id,
                        "sentence_id": sentence_id,
                        "sentence": sentence,
                        "rating": rating,
                        "verified_purchase": verified,
                        "helpful_vote": helpful_vote,
                        "trigger_groups": "|".join(groups),
                        "matched_trigger_terms": "|".join(terms),
                        "rule_based_context_hint": hint,
                    }

                    if sample_size > 0:
                        if len(candidate_sample) < sample_size:
                            candidate_sample.append(sample_row)
                        else:
                            replace_index = random_state.randint(
                                0,
                                candidate_sentence_count - 1,
                            )
                            if replace_index < sample_size:
                                candidate_sample[replace_index] = sample_row

                normalized_sentence_index += 1

            if progress_every and input_review_count % progress_every == 0:
                print(
                    f"已处理 {input_review_count:,} 条评论；"
                    f"保留 {clean_review_count:,} 条；"
                    f"包装候选句 {candidate_sentence_count:,} 条。",
                    flush=True,
                )
    finally:
        clean_writer.close()
        sentence_writer.close()
        candidate_writer.close()

    stats_rows = [
        product_stats_row(stats_by_asin[parent_asin])
        for parent_asin in product_order
    ]
    write_csv(
        output_dir / "11_product_review_stats_clean.csv",
        PRODUCT_STATS_FIELDS,
        stats_rows,
    )

    for index, row in enumerate(candidate_sample, start=1):
        row["sample_order"] = index
    write_csv(
        output_dir / "14_packaging_candidate_sample.csv",
        SAMPLE_FIELDS,
        candidate_sample,
    )

    products_with_clean_reviews = sum(
        stats.clean_review_count > 0 for stats in stats_by_asin.values()
    )
    products_with_candidates = sum(
        stats.packaging_candidate_sentence_count > 0
        for stats in stats_by_asin.values()
    )

    candidate_count_thresholds = {
        str(threshold): sum(
            stats.packaging_candidate_sentence_count >= threshold
            for stats in stats_by_asin.values()
        )
        for threshold in [1, 2, 3, 5, 10, 20]
    }
    clean_review_thresholds = {
        str(threshold): sum(
            stats.clean_review_count >= threshold
            for stats in stats_by_asin.values()
        )
        for threshold in [1, 3, 5, 10, 20]
    }

    summary = {
        "reviews_input_path": str(reviews_path),
        "raw_product_stats_path": str(raw_stats_path),
        "output_format": output_format,
        "product_count": len(product_order),
        "input_review_count": input_review_count,
        "clean_review_count": clean_review_count,
        "excluded_empty_review_count": excluded_empty_count,
        "excluded_non_english_review_count": excluded_non_english_count,
        "excluded_duplicate_review_count": excluded_duplicate_count,
        "products_with_clean_reviews": products_with_clean_reviews,
        "sentence_count": sentence_count,
        "packaging_candidate_sentence_count": candidate_sentence_count,
        "products_with_packaging_candidates": products_with_candidates,
        "clean_review_product_thresholds": clean_review_thresholds,
        "packaging_candidate_product_thresholds": candidate_count_thresholds,
        "candidate_status": (
            "high_recall_rule_candidates_only_not_final_visual_packaging_labels"
        ),
        "language_status": (
            "uses_preliminary_is_probably_english_flag_from_review_matching"
        ),
        "clean_reviews_output": str(clean_writer.path),
        "sentences_output": str(sentence_writer.path),
        "packaging_candidates_output": str(candidate_writer.path),
        "sample_seed": sample_seed,
        "sample_size_requested": sample_size,
        "sample_size_written": len(candidate_sample),
    }

    with (output_dir / "12_review_cleaning_summary.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    flattened_summary: list[dict[str, Any]] = []
    for key, value in summary.items():
        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                flattened_summary.append(
                    {"metric": f"{key}.{sub_key}", "value": sub_value}
                )
        else:
            flattened_summary.append({"metric": key, "value": value})
    write_csv(
        output_dir / "13_review_cleaning_summary.csv",
        ["metric", "value"],
        flattened_summary,
    )

    return CleaningResult(
        product_count=len(product_order),
        input_review_count=input_review_count,
        clean_review_count=clean_review_count,
        excluded_empty_count=excluded_empty_count,
        excluded_non_english_count=excluded_non_english_count,
        excluded_duplicate_count=excluded_duplicate_count,
        sentence_count=sentence_count,
        packaging_candidate_sentence_count=candidate_sentence_count,
        products_with_packaging_candidates=products_with_candidates,
        clean_reviews_path=clean_writer.path,
        sentences_path=sentence_writer.path,
        candidates_path=candidate_writer.path,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "清洗已匹配评论、切分句子并提取高召回包装相关候选句。"
        )
    )
    parser.add_argument(
        "--reviews",
        required=True,
        type=Path,
        help="02_matched_reviews_raw.parquet或CSV.GZ",
    )
    parser.add_argument(
        "--raw-stats",
        required=True,
        type=Path,
        help="03_product_review_stats_raw.csv",
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
        help="Parquet读取及写出批次大小",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=500,
        help="候选句人工核验样本数量",
    )
    parser.add_argument(
        "--sample-seed",
        type=int,
        default=42,
        help="候选句抽样随机种子",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=25_000,
        help="每处理多少条评论输出一次进度；0表示关闭",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="覆盖已有同名输出",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        result = run_cleaning(
            reviews_path=args.reviews,
            raw_stats_path=args.raw_stats,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
            sample_size=args.sample_size,
            sample_seed=args.sample_seed,
            output_format=args.output_format,
            overwrite=args.overwrite,
            progress_every=args.progress_every,
        )
    except Exception as exc:
        print(f"运行失败: {exc}", file=sys.stderr)
        return 2

    print("=" * 72)
    print("评论清洗与包装候选句提取完成")
    print(f"商品数: {result.product_count:,}")
    print(f"输入评论数: {result.input_review_count:,}")
    print(f"清洗后评论数: {result.clean_review_count:,}")
    print(f"删除空评论: {result.excluded_empty_count:,}")
    print(f"删除初步非英文评论: {result.excluded_non_english_count:,}")
    print(f"删除重复评论: {result.excluded_duplicate_count:,}")
    print(f"句子总数: {result.sentence_count:,}")
    print(
        "包装候选句数: "
        f"{result.packaging_candidate_sentence_count:,}"
    )
    print(
        "有包装候选句商品数: "
        f"{result.products_with_packaging_candidates:,}"
    )
    print(f"清洗评论: {result.clean_reviews_path}")
    print(f"句子明细: {result.sentences_path}")
    print(f"包装候选句: {result.candidates_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
