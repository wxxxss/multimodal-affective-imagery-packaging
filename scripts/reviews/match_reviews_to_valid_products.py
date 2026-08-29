#!/usr/bin/env python3
"""Match reviewed valid products to Amazon Reviews'23 comments by parent_asin.

Outputs:
01_valid_products.csv
02_matched_reviews_raw.parquet OR 02_matched_reviews_raw.csv.gz
03_product_review_stats_raw.csv
04_unmatched_valid_products.csv
05_review_matching_summary.json
06_review_matching_summary.csv
07_jsonl_parse_errors.csv

The program streams the review JSONL and never loads the full review corpus
into memory. All valid products remain in the product statistics table,
including products with zero matched reviews.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import html
import json
import math
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Protocol


PACKAGING_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in [
        r"\bpackag(?:e|ed|es|ing)\b",
        r"\bbox(?:es)?\b",
        r"\bpouch(?:es)?\b",
        r"\bcanister(?:s)?\b",
        r"\btin(?:s)?\b",
        r"\blabel(?:s|ing)?\b",
        r"\blogo(?:s)?\b",
        r"\bfont(?:s)?\b",
        r"\bcolou?r(?:s|ed|ful|ing)?\b",
        r"\bdesign(?:s|ed|ing)?\b",
        r"\billustrat(?:ion|ions|ed|ive)\b",
        r"\bartwork\b",
        r"\bgraphic(?:s)?\b",
        r"\bappearance\b",
        r"\bpresentation\b",
        r"\bgiftable\b",
        r"\bgift[- ]?ready\b",
        r"\blooks?\s+(?:beautiful|nice|great|premium|elegant|cheap|cute|natural|calming|lovely)\b",
        r"\bon\s+(?:my|the)\s+(?:kitchen\s+)?shelf\b",
        r"\bfloral\s+(?:look|design|illustration|artwork|box|label)\b",
        r"\bbotanical\s+(?:look|design|illustration|artwork|box|label)\b",
    ]
]

COMMON_ENGLISH_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "box", "but", "by",
    "for", "from", "good", "great", "has", "have", "i", "in", "is",
    "it", "looks", "love", "my", "nice", "not", "of", "on", "package",
    "packaging", "tea", "that", "the", "this", "to", "very", "was",
    "with", "you",
}

REVIEW_OUTPUT_FIELDS = [
    "parent_asin",
    "asin",
    "review_id",
    "review_fingerprint",
    "user_id",
    "rating",
    "review_title",
    "review_text",
    "combined_text",
    "timestamp_raw",
    "review_date_utc",
    "verified_purchase",
    "helpful_vote",
    "images_json",
    "source_line",
    "is_nonempty",
    "is_probably_english",
    "is_packaging_keyword_candidate",
    "is_duplicate_fingerprint",
]

STATS_FIELDS = [
    "parent_asin",
    "title",
    "main_image_url",
    "raw_review_count",
    "nonempty_review_count",
    "english_review_count",
    "packaging_keyword_candidate_count",
    "verified_review_count",
    "verified_review_ratio",
    "helpful_vote_sum",
    "unique_reviewer_count",
    "unique_review_count",
    "duplicate_review_count",
    "rating_count",
    "rating_mean",
    "rating_std",
    "first_review_date_utc",
    "last_review_date_utc",
]


@dataclass
class MatchRunResult:
    valid_product_count: int
    total_reviews_scanned: int
    matched_review_count: int
    products_with_reviews: int
    products_without_reviews: int
    json_error_count: int
    missing_parent_asin_count: int
    output_reviews_path: Path


@dataclass
class ProductStats:
    parent_asin: str
    title: str = ""
    main_image_url: str = ""
    raw_review_count: int = 0
    nonempty_review_count: int = 0
    english_review_count: int = 0
    packaging_keyword_candidate_count: int = 0
    verified_review_count: int = 0
    helpful_vote_sum: int = 0
    unique_reviewers: set[str] = field(default_factory=set)
    unique_fingerprints: set[str] = field(default_factory=set)
    duplicate_review_count: int = 0
    rating_count: int = 0
    rating_mean_value: float = 0.0
    rating_m2: float = 0.0
    first_timestamp: int | None = None
    last_timestamp: int | None = None

    def add_rating(self, rating: float | None) -> None:
        if rating is None or not math.isfinite(rating):
            return
        self.rating_count += 1
        delta = rating - self.rating_mean_value
        self.rating_mean_value += delta / self.rating_count
        delta2 = rating - self.rating_mean_value
        self.rating_m2 += delta * delta2

    @property
    def rating_std(self) -> float | None:
        if self.rating_count < 2:
            return None
        return math.sqrt(self.rating_m2 / (self.rating_count - 1))

    def update_timestamp(self, timestamp_ms: int | None) -> None:
        if timestamp_ms is None:
            return
        if self.first_timestamp is None or timestamp_ms < self.first_timestamp:
            self.first_timestamp = timestamp_ms
        if self.last_timestamp is None or timestamp_ms > self.last_timestamp:
            self.last_timestamp = timestamp_ms


class ReviewWriter(Protocol):
    path: Path

    def write(self, row: dict[str, Any]) -> None:
        ...

    def close(self) -> None:
        ...


class CsvGzipReviewWriter:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._handle = gzip.open(path, "wt", encoding="utf-8-sig", newline="")
        self._writer = csv.DictWriter(self._handle, fieldnames=REVIEW_OUTPUT_FIELDS)
        self._writer.writeheader()

    def write(self, row: dict[str, Any]) -> None:
        output = {
            field: _csv_value(row.get(field))
            for field in REVIEW_OUTPUT_FIELDS
        }
        self._writer.writerow(output)

    def close(self) -> None:
        self._handle.close()


class ParquetReviewWriter:
    def __init__(self, path: Path, batch_size: int = 10_000) -> None:
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise RuntimeError(
                "输出Parquet需要pyarrow。请运行："
                "python -m pip install pyarrow；"
                "或者使用 --output-format csv.gz。"
            ) from exc

        self.path = path
        self._pa = pa
        self._pq = pq
        self._batch_size = batch_size
        self._buffer: list[dict[str, Any]] = []
        self._schema = pa.schema(
            [
                ("parent_asin", pa.string()),
                ("asin", pa.string()),
                ("review_id", pa.string()),
                ("review_fingerprint", pa.string()),
                ("user_id", pa.string()),
                ("rating", pa.float64()),
                ("review_title", pa.string()),
                ("review_text", pa.string()),
                ("combined_text", pa.string()),
                ("timestamp_raw", pa.string()),
                ("review_date_utc", pa.string()),
                ("verified_purchase", pa.bool_()),
                ("helpful_vote", pa.int64()),
                ("images_json", pa.string()),
                ("source_line", pa.int64()),
                ("is_nonempty", pa.bool_()),
                ("is_probably_english", pa.bool_()),
                ("is_packaging_keyword_candidate", pa.bool_()),
                ("is_duplicate_fingerprint", pa.bool_()),
            ]
        )
        self._writer = pq.ParquetWriter(path, self._schema, compression="zstd")

    def write(self, row: dict[str, Any]) -> None:
        self._buffer.append({field: row.get(field) for field in REVIEW_OUTPUT_FIELDS})
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


def normalize_text(value: Any) -> str:
    """Convert a review field to normalized plain text."""
    if value is None:
        return ""
    if isinstance(value, str):
        text = value
    elif isinstance(value, (list, tuple)):
        text = " ".join(normalize_text(item) for item in value)
    elif isinstance(value, dict):
        text = " ".join(
            f"{normalize_text(key)} {normalize_text(item)}"
            for key, item in value.items()
        )
    else:
        text = str(value)

    text = html.unescape(text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = text.replace("\x00", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def is_probably_english(text: str) -> bool:
    """Deterministic preliminary English flag without external models.

    This is suitable for coverage auditing, not the final language-label gold
    standard. It checks Latin-letter dominance and common English tokens.
    """
    text = normalize_text(text)
    if not text:
        return False

    alphabetic = [char for char in text if char.isalpha()]
    if len(alphabetic) < 3:
        return False

    latin_letters = sum(
        ("A" <= char <= "Z") or ("a" <= char <= "z")
        for char in alphabetic
    )
    if latin_letters / len(alphabetic) < 0.80:
        return False

    tokens = re.findall(r"[A-Za-z']+", text.lower())
    if not tokens:
        return False

    if any(token in COMMON_ENGLISH_WORDS for token in tokens):
        return True

    # Longer Latin-script reviews are provisionally treated as English.
    return len(tokens) >= 5


def has_packaging_candidate(text: str) -> bool:
    text = normalize_text(text)
    return bool(text) and any(pattern.search(text) for pattern in PACKAGING_PATTERNS)


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    text = str(value).strip().lower()
    return text in {"1", "true", "yes", "y", "verified"}


def parse_int(value: Any, default: int = 0) -> int:
    if value is None or value == "":
        return default
    if isinstance(value, bool):
        return int(value)
    try:
        return int(float(str(value).replace(",", "").strip()))
    except (ValueError, TypeError):
        return default


def parse_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        parsed = float(str(value).replace(",", "").strip())
    except (ValueError, TypeError):
        return None
    return parsed if math.isfinite(parsed) else None


def parse_timestamp_ms(value: Any) -> int | None:
    if value is None or value == "":
        return None

    if isinstance(value, (int, float)):
        numeric = float(value)
    else:
        text = str(value).strip()
        try:
            numeric = float(text)
        except ValueError:
            try:
                dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
            except ValueError:
                return None
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return int(dt.timestamp() * 1000)

    # Amazon Reviews'23 timestamps are commonly milliseconds.
    if abs(numeric) < 100_000_000_000:
        numeric *= 1000
    return int(numeric)


def timestamp_ms_to_iso(timestamp_ms: int | None) -> str:
    if timestamp_ms is None:
        return ""
    try:
        dt = datetime.fromtimestamp(timestamp_ms / 1000, tz=timezone.utc)
    except (OverflowError, OSError, ValueError):
        return ""
    return dt.isoformat().replace("+00:00", "Z")


def make_review_fingerprint(
    parent_asin: str,
    user_id: str,
    timestamp_raw: Any,
    review_title: str,
    review_text: str,
) -> str:
    material = "\x1f".join(
        [
            parent_asin.strip(),
            user_id.strip(),
            str(timestamp_raw).strip(),
            review_title.strip().lower(),
            review_text.strip().lower(),
        ]
    )
    return hashlib.sha1(material.encode("utf-8")).hexdigest()


def make_review_id(record: dict[str, Any], fingerprint: str) -> str:
    for key in ("review_id", "reviewId", "id"):
        value = normalize_text(record.get(key))
        if value:
            return value
    return fingerprint


def open_text_auto(path: Path):
    if path.suffix.lower() == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("r", encoding="utf-8")


def iter_jsonl(
    path: Path,
    error_rows: list[dict[str, str]],
    max_error_records: int,
) -> Iterator[tuple[int, dict[str, Any]]]:
    with open_text_auto(path) as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                if len(error_rows) < max_error_records:
                    error_rows.append(
                        {
                            "source_line": str(line_number),
                            "error": str(exc),
                            "line_preview": line[:500].replace("\n", " "),
                        }
                    )
                yield line_number, {"__json_error__": True}
                continue
            if isinstance(record, dict):
                yield line_number, record
            else:
                if len(error_rows) < max_error_records:
                    error_rows.append(
                        {
                            "source_line": str(line_number),
                            "error": "JSON value is not an object",
                            "line_preview": line[:500].replace("\n", " "),
                        }
                    )
                yield line_number, {"__json_error__": True}


def read_valid_products(
    products_path: Path,
    decision_value: str,
) -> tuple[list[dict[str, str]], list[str]]:
    with products_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError("商品CSV缺少表头。")
        required = {"parent_asin", "decision"}
        missing = required - set(reader.fieldnames)
        if missing:
            raise ValueError(f"商品CSV缺少字段: {', '.join(sorted(missing))}")

        products = []
        seen: set[str] = set()
        duplicates: set[str] = set()

        for row in reader:
            if normalize_text(row.get("decision")).lower() != decision_value.lower():
                continue
            parent_asin = normalize_text(row.get("parent_asin"))
            if not parent_asin:
                raise ValueError("valid商品存在空parent_asin。")
            if parent_asin in seen:
                duplicates.add(parent_asin)
            seen.add(parent_asin)
            products.append(dict(row))

    if duplicates:
        preview = ", ".join(sorted(duplicates)[:10])
        raise ValueError(
            f"valid商品parent_asin存在重复: {preview} "
            f"(total {len(duplicates)})"
        )
    if not products:
        raise ValueError(f"没有找到decision={decision_value}的商品。")
    return products, list(reader.fieldnames)


def create_review_writer(output_dir: Path, output_format: str) -> ReviewWriter:
    if output_format == "csv.gz":
        return CsvGzipReviewWriter(output_dir / "02_matched_reviews_raw.csv.gz")
    if output_format == "parquet":
        return ParquetReviewWriter(output_dir / "02_matched_reviews_raw.parquet")
    raise ValueError(f"不支持的输出格式: {output_format}")


def _csv_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "1" if value else "0"
    return str(value)


def _write_csv(
    path: Path,
    fieldnames: list[str],
    rows: Iterable[dict[str, Any]],
) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _csv_value(row.get(field)) for field in fieldnames})


def _stats_to_row(stats: ProductStats) -> dict[str, Any]:
    verified_ratio = (
        stats.verified_review_count / stats.raw_review_count
        if stats.raw_review_count
        else None
    )
    return {
        "parent_asin": stats.parent_asin,
        "title": stats.title,
        "main_image_url": stats.main_image_url,
        "raw_review_count": stats.raw_review_count,
        "nonempty_review_count": stats.nonempty_review_count,
        "english_review_count": stats.english_review_count,
        "packaging_keyword_candidate_count": stats.packaging_keyword_candidate_count,
        "verified_review_count": stats.verified_review_count,
        "verified_review_ratio": (
            f"{verified_ratio:.6f}" if verified_ratio is not None else ""
        ),
        "helpful_vote_sum": stats.helpful_vote_sum,
        "unique_reviewer_count": len(stats.unique_reviewers),
        "unique_review_count": len(stats.unique_fingerprints),
        "duplicate_review_count": stats.duplicate_review_count,
        "rating_count": stats.rating_count,
        "rating_mean": (
            f"{stats.rating_mean_value:.4f}" if stats.rating_count else ""
        ),
        "rating_std": (
            f"{stats.rating_std:.4f}" if stats.rating_std is not None else ""
        ),
        "first_review_date_utc": timestamp_ms_to_iso(stats.first_timestamp),
        "last_review_date_utc": timestamp_ms_to_iso(stats.last_timestamp),
    }


def _ensure_clean_output(
    output_dir: Path,
    output_format: str,
    overwrite: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    expected = [
        output_dir / "01_valid_products.csv",
        output_dir / (
            "02_matched_reviews_raw.parquet"
            if output_format == "parquet"
            else "02_matched_reviews_raw.csv.gz"
        ),
        output_dir / "03_product_review_stats_raw.csv",
        output_dir / "04_unmatched_valid_products.csv",
        output_dir / "05_review_matching_summary.json",
        output_dir / "06_review_matching_summary.csv",
        output_dir / "07_jsonl_parse_errors.csv",
    ]
    existing = [path for path in expected if path.exists()]
    if existing and not overwrite:
        names = ", ".join(path.name for path in existing)
        raise FileExistsError(
            f"输出文件已存在: {names}。请更换输出目录或添加--overwrite。"
        )
    if overwrite:
        for path in existing:
            path.unlink()


def run_matching(
    *,
    products_path: Path,
    reviews_path: Path,
    output_dir: Path,
    output_format: str = "parquet",
    decision_value: str = "valid",
    progress_every: int = 1_000_000,
    overwrite: bool = False,
    max_error_records: int = 1000,
) -> MatchRunResult:
    products_path = products_path.resolve()
    reviews_path = reviews_path.resolve()
    output_dir = output_dir.resolve()

    if not products_path.is_file():
        raise FileNotFoundError(f"商品CSV不存在: {products_path}")
    if not reviews_path.is_file():
        raise FileNotFoundError(f"评论JSONL不存在: {reviews_path}")

    _ensure_clean_output(output_dir, output_format, overwrite)
    products, product_fields = read_valid_products(products_path, decision_value)

    valid_products_path = output_dir / "01_valid_products.csv"
    _write_csv(valid_products_path, product_fields, products)

    product_by_asin = {
        normalize_text(row["parent_asin"]): row
        for row in products
    }
    stats_by_asin = {
        parent_asin: ProductStats(
            parent_asin=parent_asin,
            title=normalize_text(row.get("title")),
            main_image_url=normalize_text(row.get("main_image_url")),
        )
        for parent_asin, row in product_by_asin.items()
    }

    writer = create_review_writer(output_dir, output_format)
    error_rows: list[dict[str, str]] = []

    total_reviews_scanned = 0
    matched_review_count = 0
    json_error_count = 0
    missing_parent_asin_count = 0

    try:
        for source_line, record in iter_jsonl(
            reviews_path,
            error_rows=error_rows,
            max_error_records=max_error_records,
        ):
            total_reviews_scanned += 1

            if record.get("__json_error__"):
                json_error_count += 1
                continue

            parent_asin = normalize_text(
                record.get("parent_asin")
                or record.get("parentAsin")
            )
            if not parent_asin:
                missing_parent_asin_count += 1
                continue
            if parent_asin not in stats_by_asin:
                if progress_every and total_reviews_scanned % progress_every == 0:
                    print(
                        f"已扫描 {total_reviews_scanned:,} 条评论，"
                        f"匹配 {matched_review_count:,} 条。",
                        flush=True,
                    )
                continue

            review_title = normalize_text(record.get("title"))
            review_text = normalize_text(record.get("text"))
            combined_text = normalize_text(f"{review_title} {review_text}")
            nonempty = bool(combined_text)
            english = is_probably_english(combined_text) if nonempty else False
            packaging_candidate = (
                has_packaging_candidate(combined_text) if nonempty else False
            )

            user_id = normalize_text(
                record.get("user_id")
                or record.get("userId")
                or record.get("reviewerID")
            )
            timestamp_raw = record.get("timestamp")
            timestamp_ms = parse_timestamp_ms(timestamp_raw)
            verified = parse_bool(record.get("verified_purchase"))
            helpful_vote = parse_int(record.get("helpful_vote"), default=0)
            rating = parse_float(record.get("rating"))

            fingerprint = make_review_fingerprint(
                parent_asin=parent_asin,
                user_id=user_id,
                timestamp_raw=timestamp_raw,
                review_title=review_title,
                review_text=review_text,
            )
            review_id = make_review_id(record, fingerprint)

            product_stats = stats_by_asin[parent_asin]
            is_duplicate = fingerprint in product_stats.unique_fingerprints

            product_stats.raw_review_count += 1
            if nonempty:
                product_stats.nonempty_review_count += 1
            if english:
                product_stats.english_review_count += 1
            if packaging_candidate:
                product_stats.packaging_keyword_candidate_count += 1
            if verified:
                product_stats.verified_review_count += 1
            product_stats.helpful_vote_sum += helpful_vote
            if user_id:
                product_stats.unique_reviewers.add(user_id)
            if is_duplicate:
                product_stats.duplicate_review_count += 1
            else:
                product_stats.unique_fingerprints.add(fingerprint)
            product_stats.add_rating(rating)
            product_stats.update_timestamp(timestamp_ms)

            images = record.get("images")
            if images is None:
                images_json = ""
            else:
                images_json = json.dumps(images, ensure_ascii=False, separators=(",", ":"))

            output_row = {
                "parent_asin": parent_asin,
                "asin": normalize_text(record.get("asin")),
                "review_id": review_id,
                "review_fingerprint": fingerprint,
                "user_id": user_id,
                "rating": rating,
                "review_title": review_title,
                "review_text": review_text,
                "combined_text": combined_text,
                "timestamp_raw": (
                    "" if timestamp_raw is None else str(timestamp_raw)
                ),
                "review_date_utc": timestamp_ms_to_iso(timestamp_ms),
                "verified_purchase": verified,
                "helpful_vote": helpful_vote,
                "images_json": images_json,
                "source_line": source_line,
                "is_nonempty": nonempty,
                "is_probably_english": english,
                "is_packaging_keyword_candidate": packaging_candidate,
                "is_duplicate_fingerprint": is_duplicate,
            }
            writer.write(output_row)
            matched_review_count += 1

            if progress_every and total_reviews_scanned % progress_every == 0:
                print(
                    f"已扫描 {total_reviews_scanned:,} 条评论，"
                    f"匹配 {matched_review_count:,} 条。",
                    flush=True,
                )
    finally:
        writer.close()

    stats_rows = [
        _stats_to_row(stats_by_asin[normalize_text(product["parent_asin"])])
        for product in products
    ]
    _write_csv(
        output_dir / "03_product_review_stats_raw.csv",
        STATS_FIELDS,
        stats_rows,
    )

    unmatched_products = [
        product
        for product in products
        if stats_by_asin[normalize_text(product["parent_asin"])].raw_review_count == 0
    ]
    _write_csv(
        output_dir / "04_unmatched_valid_products.csv",
        product_fields,
        unmatched_products,
    )

    _write_csv(
        output_dir / "07_jsonl_parse_errors.csv",
        ["source_line", "error", "line_preview"],
        error_rows,
    )

    products_with_reviews = sum(
        stats.raw_review_count > 0
        for stats in stats_by_asin.values()
    )
    products_without_reviews = len(products) - products_with_reviews
    products_with_english = sum(
        stats.english_review_count > 0
        for stats in stats_by_asin.values()
    )
    products_with_packaging_candidates = sum(
        stats.packaging_keyword_candidate_count > 0
        for stats in stats_by_asin.values()
    )
    english_review_count = sum(
        stats.english_review_count
        for stats in stats_by_asin.values()
    )
    packaging_candidate_count = sum(
        stats.packaging_keyword_candidate_count
        for stats in stats_by_asin.values()
    )

    summary = {
        "products_input_path": str(products_path),
        "reviews_input_path": str(reviews_path),
        "decision_filter": decision_value,
        "valid_product_count": len(products),
        "total_reviews_scanned": total_reviews_scanned,
        "matched_review_count": matched_review_count,
        "products_with_reviews": products_with_reviews,
        "products_without_reviews": products_without_reviews,
        "products_with_english_reviews": products_with_english,
        "english_review_count_preliminary": english_review_count,
        "products_with_packaging_keyword_candidates": products_with_packaging_candidates,
        "packaging_keyword_candidate_count_preliminary": packaging_candidate_count,
        "json_error_count": json_error_count,
        "json_error_rows_saved": len(error_rows),
        "reviews_missing_parent_asin_count": missing_parent_asin_count,
        "language_flag_method": "deterministic_latin_english_heuristic_v1",
        "packaging_count_status": (
            "high_recall_keyword_candidate_only_not_final_semantic_classification"
        ),
        "matched_reviews_output": str(writer.path),
    }

    with (output_dir / "05_review_matching_summary.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    _write_csv(
        output_dir / "06_review_matching_summary.csv",
        ["metric", "value"],
        [{"metric": key, "value": value} for key, value in summary.items()],
    )

    return MatchRunResult(
        valid_product_count=len(products),
        total_reviews_scanned=total_reviews_scanned,
        matched_review_count=matched_review_count,
        products_with_reviews=products_with_reviews,
        products_without_reviews=products_without_reviews,
        json_error_count=json_error_count,
        missing_parent_asin_count=missing_parent_asin_count,
        output_reviews_path=writer.path,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "从审核后的商品CSV筛选valid商品，并按parent_asin流式关联"
            "Amazon Reviews'23评论。"
        )
    )
    parser.add_argument(
        "--products",
        required=True,
        type=Path,
        help="5621件审核后合并商品CSV",
    )
    parser.add_argument(
        "--reviews",
        required=True,
        type=Path,
        help="Amazon Reviews'23 Grocery评论JSONL或JSONL.GZ",
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
        help="评论明细格式；默认parquet",
    )
    parser.add_argument(
        "--decision",
        default="valid",
        help="商品decision筛选值；默认valid",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=1_000_000,
        help="每扫描多少条评论打印一次进度；0表示关闭",
    )
    parser.add_argument(
        "--max-error-records",
        type=int,
        default=1000,
        help="最多保存多少条JSON解析错误详情",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="覆盖输出目录中同名结果文件",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        result = run_matching(
            products_path=args.products,
            reviews_path=args.reviews,
            output_dir=args.output_dir,
            output_format=args.output_format,
            decision_value=args.decision,
            progress_every=args.progress_every,
            overwrite=args.overwrite,
            max_error_records=args.max_error_records,
        )
    except Exception as exc:
        print(f"运行失败: {exc}", file=sys.stderr)
        return 2

    print("=" * 72)
    print("评论关联完成")
    print(f"valid商品数: {result.valid_product_count:,}")
    print(f"扫描评论数: {result.total_reviews_scanned:,}")
    print(f"匹配评论数: {result.matched_review_count:,}")
    print(f"有评论商品数: {result.products_with_reviews:,}")
    print(f"无评论商品数: {result.products_without_reviews:,}")
    print(f"JSON解析错误数: {result.json_error_count:,}")
    print(f"评论明细: {result.output_reviews_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
