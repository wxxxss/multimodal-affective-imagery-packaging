#!/usr/bin/env python3
"""Build product-level affective imagery labels from packaging-review sentences.

Pipeline:
1. Read v1.1 rule-classified packaging sentence candidates.
2. Clean residual false positives from visual_strict.
3. Deduplicate exact and near-duplicate sentences within each product.
4. Extract affective expressions using an auditable local lexicon.
5. Confirm preliminary dimensions from strict evidence.
6. Recover only high-confidence uncertain sentences that have:
   explicit packaging object + confirmed imagery expression + no exclusion signal.
7. Aggregate strict and recovered evidence to one row per product.

No online model or API is used.
"""

from __future__ import annotations

import argparse
import gzip
import html
import json
import math
import random
import re
import sys
import unicodedata
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import pandas as pd


PIPELINE_VERSION = "affective_imagery_labels_relation_v2.0"
RECOVERY_RULE_VERSION = "targeted_recovery_relation_v2.0"


@dataclass(frozen=True)
class DimensionSpec:
    code: str
    name_cn: str
    polarity: str
    expressions: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class ExpressionHit:
    dimension_code: str
    dimension_name_cn: str
    polarity: str
    expression_raw: str
    expression_lemma: str
    start: int
    end: int


@dataclass(frozen=True)
class RecoveryDecision:
    accepted: bool
    dimension_codes: tuple[str, ...] = ()
    matched_expressions: tuple[str, ...] = ()
    rejection_reason: str = ""


@dataclass
class PipelineResult:
    product_count: int
    input_visual_strict_count: int
    cleaned_visual_strict_count: int
    exact_duplicate_count: int
    near_duplicate_count: int
    residual_false_positive_count: int
    input_uncertain_count: int
    recovered_count: int
    combined_imagery_sentence_count: int
    products_with_any_imagery: int


DIMENSIONS: tuple[DimensionSpec, ...] = (
    DimensionSpec(
        "general_visual_appeal",
        "一般视觉吸引力",
        "positive",
        (
            ("beautiful", r"\bbeautiful(?:ly)?\b"),
            ("pretty", r"\bpretty\b(?!\s+(?:good|bad|steep|much|well|strong))"),
            ("lovely", r"\blovely\b"),
            ("gorgeous", r"\bgorgeous\b"),
            ("attractive", r"\battractive\b"),
            ("appealing", r"\bappealing\b"),
            ("stunning", r"\bstunning\b"),
            ("eye-catching", r"\beye[- ]catching\b"),
            ("aesthetically pleasing", r"\baesthetically pleasing\b"),
            ("good-looking", r"\bgood[- ]looking\b"),
        ),
    ),
    DimensionSpec(
        "cute_friendly",
        "可爱亲和感",
        "positive",
        (
            ("cute", r"\bcute\b"),
            ("adorable", r"\badorable\b"),
            ("charming", r"\bcharming\b"),
            ("playful", r"\bplayful\b"),
            ("whimsical", r"\bwhimsical\b"),
            ("perky", r"\bperky\b"),
            ("friendly-looking", r"\bfriendly[- ]looking\b"),
            ("fun-looking", r"\bfun[- ]looking\b"),
        ),
    ),
    DimensionSpec(
        "premium_refined",
        "高级精致感",
        "positive",
        (
            ("elegant", r"\belegant(?:ly)?\b"),
            ("classy", r"\bclassy\b"),
            ("fancy", r"\bfancy\b"),
            ("sophisticated", r"\bsophisticated\b"),
            ("luxurious", r"\bluxur(?:ious|iously|y)\b"),
            ("premium-looking", r"\bpremium[- ]looking\b"),
            ("upscale", r"\bupscale\b"),
            ("high-end", r"\bhigh[- ]end\b"),
            ("refined", r"\brefined\b"),
            ("expensive-looking", r"\bexpensive[- ]looking\b"),
            ("professional-looking", r"\bprofessional[- ]looking\b"),
        ),
    ),
    DimensionSpec(
        "gift_presentation",
        "礼赠呈现感",
        "positive",
        (
            ("giftable", r"\bgiftable\b"),
            ("presentable", r"\bpresentable\b"),
            ("gift-worthy", r"\bgift[- ]worthy\b"),
            ("gift-ready", r"\bgift[- ]ready\b"),
            ("perfect gift", r"\bperfect (?:as a )?gift\b"),
            ("great gift", r"\bgreat (?:as a )?gift\b"),
            ("nice gift", r"\bnice (?:as a )?gift\b"),
            ("ideal gift", r"\bideal (?:as a )?gift\b"),
            ("stocking stuffer", r"\bstocking stuffer\b"),
        ),
    ),
    DimensionSpec(
        "simple_modern",
        "简约现代感",
        "positive",
        (
            ("simple", r"\bsimple\b"),
            ("minimal", r"\bminimal\b"),
            ("minimalist", r"\bminimalist(?:ic)?\b"),
            ("clean-looking", r"\bclean[- ]looking\b"),
            ("clean design", r"\bclean design\b"),
            ("modern", r"\bmodern\b"),
            ("sleek", r"\bsleek\b"),
            ("contemporary", r"\bcontemporary\b"),
            ("streamlined", r"\bstreamlined\b"),
        ),
    ),
    DimensionSpec(
        "natural_botanical",
        "自然植物感",
        "positive",
        (
            ("natural-looking", r"\bnatural[- ]looking\b"),
            ("botanical", r"\bbotanical\b"),
            ("floral", r"\bfloral\b"),
            ("earthy", r"\bearthy\b"),
            ("organic-looking", r"\borganic[- ]looking\b"),
            ("plant-inspired", r"\bplant[- ]inspired\b"),
            ("nature-inspired", r"\bnature[- ]inspired\b"),
            ("herbal-looking", r"\bherbal[- ]looking\b"),
        ),
    ),
    DimensionSpec(
        "calming_soft",
        "舒缓柔和感",
        "positive",
        (
            ("calming", r"\bcalming\b"),
            ("soothing", r"\bsoothing\b"),
            ("peaceful", r"\bpeaceful\b"),
            ("serene", r"\bserene\b"),
            ("soft-looking", r"\bsoft[- ]looking\b"),
            ("gentle-looking", r"\bgentle[- ]looking\b"),
            ("relaxing-looking", r"\brelaxing[- ]looking\b"),
        ),
    ),
    DimensionSpec(
        "cheerful_colorful",
        "活力愉悦感",
        "positive",
        (
            ("colorful", r"\bcolorful\b|\bcolourful\b"),
            ("bright", r"\bbright\b"),
            ("cheerful", r"\bcheerful\b"),
            ("vibrant", r"\bvibrant\b"),
            ("lively", r"\blively\b"),
            ("joyful", r"\bjoyful\b"),
            ("festive", r"\bfestive\b"),
        ),
    ),
    DimensionSpec(
        "traditional_vintage",
        "传统复古感",
        "neutral",
        (
            ("traditional", r"\btraditional\b"),
            ("vintage", r"\bvintage\b"),
            ("retro", r"\bretro\b"),
            ("classic", r"\bclassic\b"),
            ("old-fashioned", r"\bold[- ]fashioned\b"),
            ("heritage", r"\bheritage\b"),
            ("nostalgic", r"\bnostalgic\b"),
        ),
    ),
    DimensionSpec(
        "negative_appearance",
        "负面外观感",
        "negative",
        (
            ("ugly", r"\bugly\b"),
            ("tacky", r"\btacky\b"),
            ("boring", r"\bboring\b"),
            ("clinical", r"\bclinical\b"),
            ("plain", r"\bplain\b"),
            ("unappealing", r"\bunappealing\b"),
            ("not attractive", r"\bnot attractive\b"),
            ("cheap-looking", r"\bcheap[- ]looking\b|\blooks? cheap\b"),
            ("messy", r"\bmessy\b"),
            ("rough-looking", r"\brough[- ]looking\b"),
            ("dated", r"\bdated\b"),
            ("incorrect design", r"\b(?:logo|font|label|branding|design)\b.{0,30}\b(?:wrong|incorrect|not correct|isn['’]?t correct|aren['’]?t correct)\b"),
            ("could look better", r"\bcould (?:be|look) (?:better|cooler|nicer)\b"),
            ("not a fan", r"\bnot a fan of (?:the )?(?:packaging|package|box|design|label)\b"),
        ),
    ),
)

DIMENSION_BY_CODE = {dimension.code: dimension for dimension in DIMENSIONS}
COMPILED_DIMENSION_PATTERNS: dict[str, tuple[tuple[str, re.Pattern[str]], ...]] = {
    dimension.code: tuple(
        (lemma, re.compile(pattern, re.IGNORECASE))
        for lemma, pattern in dimension.expressions
    )
    for dimension in DIMENSIONS
}


PACKAGING_NOUN_PATTERN = re.compile(
    r"\b(?:packaging|package|packages|box|boxes|carton|cartons|tin|tins|"
    r"canister|canisters|container|containers|pouch|pouches|wrapper|wrappers|"
    r"label|labels|logo|font|typography|lettering|artwork|illustration|"
    r"illustrations|graphic|graphics|design|branding|presentation)\b",
    re.IGNORECASE,
)

PHYSICAL_PACKAGE_NOUN_PATTERN = re.compile(
    r"\b(?:packaging|package|packages|box|boxes|carton|cartons|tin|tins|"
    r"canister|canisters|container|containers|pouch|pouches|wrapper|wrappers|"
    r"label|labels)\b",
    re.IGNORECASE,
)

SHIPPING_DAMAGE_PATTERN = re.compile(
    r"\b(?:arriv(?:e|ed|es|ing)|shipping|delivery|delivered|crushed|damaged|"
    r"dented|torn|ripped|broken|leak(?:ed|ing)|smashed|beat up|beaten up|"
    r"squashed|bent)\b",
    re.IGNORECASE,
)
ORDER_PATTERN = re.compile(
    r"\b(?:ordered|received|only (?:received|got)|missing|wrong item|"
    r"wrong product|refund|returned|sent me|shipped me|didn['’]?t receive|"
    r"not receive)\b",
    re.IGNORECASE,
)
QUANTITY_PRICE_PATTERN = re.compile(
    r"\b(?:\d+\s*(?:count|ct|tea ?bags?|sachets?|packets?|boxes)|pack of \d+|"
    r"box of \d+|one box|two boxes|three boxes|four boxes|five boxes|six boxes|"
    r"larger package|smaller package|bulk box|price|priced|value|expensive|"
    r"money friendly|cost[- ]efficient|pretty steep|too much for|half empty|"
    r"1/2 empty)\b",
    re.IGNORECASE,
)
STRUCTURAL_PATTERN = re.compile(
    r"\b(?:resealable|sealed|zip(?:per|lock)?|airtight|foil|individually wrapped|"
    r"individual wrappers?|for freshness|keeps? (?:it|them|the tea)? ?fresh|"
    r"easy to open|hard to open|difficult to open|loose lid|leakproof|hold(?:s|ing| the tea| tea)? fresh|would hold the tea fresh|"
    r"no staples|no strings|extra paper)\b",
    re.IGNORECASE,
)
PRODUCT_CONTENT_PATTERN = re.compile(
    r"\b(?:taste|tastes|tasted|tasting|flavor|flavour|aroma|smell|smells|"
    r"brew|brewing|steep|steeping|cup|liquid|water|ingredients|leaves|"
    r"flowers|petals|buds|sleep|stomach|stress|caffeine|decaf|health benefits|"
    r"medicinal|detox)\b",
    re.IGNORECASE,
)

TRACKING_LABEL_PATTERN = re.compile(
    r"\b(?:tracking|shipping|mailing|address|barcode)\s+label\b|"
    r"\blabel\b.{0,20}\b(?:tracking number|shipping address|barcode)\b",
    re.IGNORECASE,
)
EXPIRATION_PATTERN = re.compile(
    r"\b(?:expired|expiration|expiry|best by|best-before|use by|dated)\b"
    r".{0,35}\b(?:box|package|label|date)\b|"
    r"\b(?:box|package|label)\b.{0,35}\b(?:expired|expiration|expiry|dated)\b",
    re.IGNORECASE,
)
PERSONAL_STORAGE_PATTERN = re.compile(
    r"\b(?:i|we|my|our)\b.{0,80}\b(?:put|keep|kept|store|stored|storing|"
    r"reuse|reusing|refill|refilling|transfer|divide|divided)\b.{0,60}\b"
    r"(?:jar|container|canister|tin|tins)\b|"
    r"\b(?:wish|wished|buy|bought|need|use)\b.{0,60}\b"
    r"(?:jar|container|canister|tin|tins)\b.{0,50}\b"
    r"(?:store|divide|transfer|refill|counter|countertop|holiday gifts?)\b|"
    r"\b(?:cookie jar|glass jar|clamp[- ]top jar|storage tin|tea canister)\b",
    re.IGNORECASE,
)
SELLER_PACKAGING_PATTERN = re.compile(
    r"\b(?:seller|vendor|amazon|shipper|warehouse)\b.{0,50}\b"
    r"(?:packaged|packed|packing)\b|"
    r"\b(?:packaged|packed|packing)\b.{0,50}\b"
    r"(?:seller|vendor|amazon|shipper|warehouse|shipping|delivery)\b",
    re.IGNORECASE,
)
PRODUCT_PRESENTATION_PATTERN = re.compile(
    r"\bpresentation\b.{0,80}\b(?:ruby|red|lemon|honey|fruit|fruity|floral|"
    r"flavor|flavour|taste|cup|drink|brew|aroma|garnish)\b|"
    r"\b(?:ruby|red|lemon|honey|fruit|fruity|floral|flavor|flavour|taste|"
    r"cup|drink|brew|aroma|garnish)\b.{0,80}\bpresentation\b",
    re.IGNORECASE,
)
CONTENT_IN_PACKAGE_PATTERN = re.compile(
    r"\b(?:flowers|petals|buds|leaves|ingredients|tea bags?|colors?|colours?)\b"
    r".{0,70}\b(?:in|inside|come in|comes in)\b.{0,30}\b"
    r"(?:package|box|pouch|container)\b.{0,35}\b"
    r"(?:beautiful|pretty|lovely|colorful|colourful|attractive)\b",
    re.IGNORECASE,
)
GENERIC_PRODUCT_PRESENTATION_PATTERN = re.compile(
    r"\b(?:beautiful|pretty|lovely|attractive)\s+presentation\b",
    re.IGNORECASE,
)

RELATION_COPULA_PATTERN_TEMPLATE = (
    r"\b{object}\b(?:\s+\w+){{0,3}}\s+"
    r"(?:is|are|was|were|looks?|looked|seems?|seemed|feels?|felt|"
    r"appears?|appeared|could be|could look)\s+"
    r"(?:very\s+|so\s+|super\s+|really\s+|rather\s+|quite\s+)?"
    r"{expression}"
)

RELATION_ADJ_BEFORE_PATTERN_TEMPLATE = (
    r"{expression}(?:\s+[\w'-]+){{0,3}}\s+\b{object}\b"
)


def normalize_sentence(value: Any) -> str:
    """Normalize text for deterministic within-product deduplication."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    text = html.unescape(str(value))
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("’", "'").replace("‘", "'").replace("`", "'")
    text = re.sub(r"<[^>]+>", " ", text)
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s'-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _safe_text(value: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return str(value)


def _safe_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _safe_int(value: Any, default: int = 0) -> int:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return default
    try:
        return int(float(str(value).replace(",", "").strip()))
    except (TypeError, ValueError):
        return default


def _safe_float(value: Any) -> float | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    try:
        parsed = float(str(value).replace(",", "").strip())
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def extract_dimension_hits(sentence: str) -> tuple[ExpressionHit, ...]:
    """Extract unique dimension-expression hits from a sentence."""
    text = _safe_text(sentence)
    hits: list[ExpressionHit] = []

    for dimension in DIMENSIONS:
        for lemma, pattern in COMPILED_DIMENSION_PATTERNS[dimension.code]:
            for match in pattern.finditer(text):
                hits.append(
                    ExpressionHit(
                        dimension_code=dimension.code,
                        dimension_name_cn=dimension.name_cn,
                        polarity=dimension.polarity,
                        expression_raw=match.group(0),
                        expression_lemma=lemma,
                        start=match.start(),
                        end=match.end(),
                    )
                )

    return tuple(
        sorted(
            hits,
            key=lambda hit: (
                hit.start,
                hit.dimension_code,
                hit.expression_lemma,
            ),
        )
    )


PRENOMINAL_MODIFIERS = {
    "little",
    "small",
    "large",
    "tiny",
    "pink",
    "green",
    "black",
    "white",
    "gold",
    "golden",
    "red",
    "blue",
    "purple",
    "yellow",
    "orange",
    "brown",
    "beige",
    "cream",
    "silver",
    "pastel",
    "dark",
    "bright",
    "floral",
    "botanical",
    "square",
    "round",
    "metal",
    "metallic",
    "paper",
    "cardboard",
    "glass",
    "clear",
    "strong",
    "sturdy",
    "gift",
    "tea",
    "product",
    "outer",
    "front",
    "new",
    "original",
}


def _hit_has_explicit_relation(
    sentence: str,
    hit: ExpressionHit,
    package_matches: Sequence[re.Match[str]],
) -> bool:
    text = _safe_text(sentence)

    for package_match in package_matches:
        # adjective/expression immediately before package/design object;
        # intervening tokens may only be plausible noun modifiers.
        if hit.end <= package_match.start():
            between = text[hit.end : package_match.start()]
            tokens = [
                token.lower()
                for token in re.findall(r"\b[a-zA-Z'-]+\b", between)
            ]
            if len(tokens) <= 3 and all(
                token in PRENOMINAL_MODIFIERS for token in tokens
            ):
                return True

        # package/design object as grammatical subject followed by a
        # copular/appearance verb and the expression.
        if package_match.end() <= hit.start:
            between = text[package_match.end() : hit.start]
            normalized_between = re.sub(
                r"\s+",
                " ",
                between.lower(),
            ).strip(" ,;:()[]{}-'\"")
            if re.fullmatch(
                r"(?:itself |it |they |themselves )?"
                r"(?:is|are|was|were|looks?|looked|seems?|seemed|"
                r"feels?|felt|appears?|appeared|could be|could look)"
                r"(?: (?:very|so|super|really|rather|quite|extremely))?"
                r"(?: (?:a|an|somewhat))?",
                normalized_between,
            ):
                return True

            # Design objects can use verbs such as "uses" or "features":
            if package_match.group(0).lower() in {
                "label",
                "labels",
                "logo",
                "font",
                "typography",
                "lettering",
                "artwork",
                "illustration",
                "illustrations",
                "graphic",
                "graphics",
                "design",
                "branding",
            } and re.fullmatch(
                r"(?:uses?|features?|has|have)"
                r"(?: (?:a|an|the|very|really))?",
                normalized_between,
            ):
                return True

    return False


def _has_explicit_packaging_visual_relation(sentence: str) -> bool:
    """Return True only for phrase-level package-object visual evaluation."""
    text = _safe_text(sentence)
    hits = extract_dimension_hits(text)
    if not hits:
        return False

    package_matches = list(PACKAGING_NOUN_PATTERN.finditer(text))
    if not package_matches:
        return False

    return any(
        _hit_has_explicit_relation(text, hit, package_matches)
        for hit in hits
    )

def classify_residual_false_positive(sentence: str) -> str | None:
    """Identify residual non-visual rows that survived strict v1.1."""
    text = _safe_text(sentence)

    if TRACKING_LABEL_PATTERN.search(text):
        return "tracking_or_shipping_label"
    if EXPIRATION_PATTERN.search(text):
        return "expiration_or_date_label"
    if PERSONAL_STORAGE_PATTERN.search(text):
        return "personal_storage_or_accessory"
    if SELLER_PACKAGING_PATTERN.search(text):
        return "seller_or_shipping_packaging"
    if SHIPPING_DAMAGE_PATTERN.search(text) and not _has_explicit_packaging_visual_relation(text):
        return "shipping_damage"
    if ORDER_PATTERN.search(text) and not _has_explicit_packaging_visual_relation(text):
        return "order_fulfillment"
    if CONTENT_IN_PACKAGE_PATTERN.search(text):
        return "product_content"

    if PRODUCT_PRESENTATION_PATTERN.search(text):
        if not PHYSICAL_PACKAGE_NOUN_PATTERN.search(text):
            return "product_presentation"
        # "gift box has a beautiful presentation" is packaging presentation.
        if not re.search(
            r"\b(?:box|package|packaging|tin|container)\b.{0,45}"
            r"\bpresentation\b",
            text,
            re.IGNORECASE,
        ):
            return "product_presentation"

    if GENERIC_PRODUCT_PRESENTATION_PATTERN.search(text):
        if not re.search(
            r"\b(?:box|package|packaging|tin|container|gift)\b",
            text,
            re.IGNORECASE,
        ):
            return "product_presentation"

    # Product-content adjectives alone are not packaging imagery.
    if PRODUCT_CONTENT_PATTERN.search(text):
        if not _has_explicit_packaging_visual_relation(text):
            return "product_content"

    return None


def _strong_recovery_exclusion(sentence: str) -> str | None:
    residual = classify_residual_false_positive(sentence)
    if residual is not None:
        return residual
    if SHIPPING_DAMAGE_PATTERN.search(sentence):
        return "shipping_damage"
    if ORDER_PATTERN.search(sentence):
        return "order_fulfillment"
    if QUANTITY_PRICE_PATTERN.search(sentence):
        return "quantity_or_price"
    if STRUCTURAL_PATTERN.search(sentence):
        return "structural_packaging"
    return None


def _expression_related_to_package(
    sentence: str,
    hit: ExpressionHit,
) -> bool:
    text = _safe_text(sentence)
    package_matches = list(PACKAGING_NOUN_PATTERN.finditer(text))
    if not package_matches:
        return False
    return _hit_has_explicit_relation(text, hit, package_matches)

def targeted_recovery_decision(
    sentence: str,
    confirmed_dimensions: set[str],
) -> RecoveryDecision:
    """High-precision recovery from uncertain sentences."""
    text = _safe_text(sentence)
    exclusion = _strong_recovery_exclusion(text)
    if exclusion is not None:
        return RecoveryDecision(False, rejection_reason=exclusion)

    if not PACKAGING_NOUN_PATTERN.search(text):
        return RecoveryDecision(False, rejection_reason="no_packaging_object")

    all_hits = extract_dimension_hits(text)
    allowed_hits = [
        hit
        for hit in all_hits
        if hit.dimension_code in confirmed_dimensions
        and _expression_related_to_package(text, hit)
    ]
    if not allowed_hits:
        return RecoveryDecision(
            False,
            rejection_reason="no_confirmed_imagery_relation",
        )

    dimension_codes = tuple(
        sorted({hit.dimension_code for hit in allowed_hits})
    )
    matched_expressions = tuple(
        sorted({hit.expression_lemma for hit in allowed_hits})
    )
    return RecoveryDecision(
        True,
        dimension_codes=dimension_codes,
        matched_expressions=matched_expressions,
    )


def _read_table(path: Path) -> pd.DataFrame:
    lower_name = path.name.lower()
    if lower_name.endswith(".parquet"):
        return pd.read_parquet(path)
    if lower_name.endswith(".csv.gz"):
        return pd.read_csv(path, compression="gzip", encoding="utf-8-sig")
    if lower_name.endswith(".csv"):
        return pd.read_csv(path, encoding="utf-8-sig")
    raise ValueError("输入文件仅支持.parquet、.csv或.csv.gz。")


def _write_detail(
    dataframe: pd.DataFrame,
    output_dir: Path,
    stem: str,
    output_format: str,
) -> Path:
    if output_format == "parquet":
        path = output_dir / f"{stem}.parquet"
        dataframe.to_parquet(path, index=False)
        return path
    if output_format == "csv.gz":
        path = output_dir / f"{stem}.csv.gz"
        dataframe.to_csv(
            path,
            index=False,
            compression="gzip",
            encoding="utf-8-sig",
        )
        return path
    raise ValueError(f"不支持的输出格式: {output_format}")


def _ensure_columns(
    dataframe: pd.DataFrame,
    columns: Sequence[str],
) -> pd.DataFrame:
    result = dataframe.copy()
    for column in columns:
        if column not in result.columns:
            result[column] = ""
    return result


def _prepare_classified(dataframe: pd.DataFrame) -> pd.DataFrame:
    required = {"parent_asin", "sentence", "decision"}
    missing = required - set(dataframe.columns)
    if missing:
        raise ValueError(
            "分类结果缺少字段: " + ", ".join(sorted(missing))
        )

    columns = [
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
        "decision",
        "reason",
        "confidence",
        "matched_visual_terms",
        "matched_exclusion_terms",
        "strict_rule_version",
    ]
    result = _ensure_columns(dataframe, columns)
    result["parent_asin"] = result["parent_asin"].fillna("").astype(str).str.strip()
    result["sentence"] = result["sentence"].fillna("").astype(str)
    result["review_id"] = result["review_id"].fillna("").astype(str)
    result["sentence_id"] = result["sentence_id"].fillna("").astype(str)
    result["user_id"] = result["user_id"].fillna("").astype(str)
    result["normalized_sentence"] = result["sentence"].map(normalize_sentence)
    result["verified_purchase"] = result["verified_purchase"].map(_safe_bool)
    result["helpful_vote"] = result["helpful_vote"].map(_safe_int)
    result["rating"] = result["rating"].map(_safe_float)
    return result


def _rank_for_dedup(dataframe: pd.DataFrame) -> pd.DataFrame:
    ranked = dataframe.copy()
    ranked["_verified_rank"] = ranked["verified_purchase"].astype(int)
    ranked["_helpful_rank"] = ranked["helpful_vote"].map(_safe_int)
    ranked["_text_length_rank"] = ranked["normalized_sentence"].str.len()
    ranked = ranked.sort_values(
        [
            "parent_asin",
            "_verified_rank",
            "_helpful_rank",
            "_text_length_rank",
            "review_id",
            "sentence_id",
        ],
        ascending=[True, False, False, False, True, True],
        kind="mergesort",
    )
    return ranked


def _deduplicate_strict(
    strict: pd.DataFrame,
    near_duplicate_threshold: float,
) -> tuple[pd.DataFrame, pd.DataFrame, int, int]:
    ranked = _rank_for_dedup(strict)
    kept_rows: list[pd.Series] = []
    rejected_rows: list[dict[str, Any]] = []
    exact_count = 0
    near_count = 0

    for parent_asin, group in ranked.groupby("parent_asin", sort=False):
        exact_seen: dict[str, str] = {}
        kept_norms: list[tuple[str, str]] = []

        for _, row in group.iterrows():
            normalized = row["normalized_sentence"]
            sentence_id = row["sentence_id"]

            if not normalized:
                rejected = row.to_dict()
                rejected["cleaning_decision"] = "rejected"
                rejected["cleaning_reason"] = "empty_after_normalization"
                rejected["duplicate_of_sentence_id"] = ""
                rejected_rows.append(rejected)
                continue

            if normalized in exact_seen:
                exact_count += 1
                rejected = row.to_dict()
                rejected["cleaning_decision"] = "rejected"
                rejected["cleaning_reason"] = "exact_duplicate_within_product"
                rejected["duplicate_of_sentence_id"] = exact_seen[normalized]
                rejected_rows.append(rejected)
                continue

            duplicate_of = ""
            if near_duplicate_threshold < 1.0 and len(normalized.split()) >= 7:
                for kept_normalized, kept_sentence_id in kept_norms:
                    if abs(len(normalized) - len(kept_normalized)) > max(
                        20,
                        int(0.20 * max(len(normalized), len(kept_normalized))),
                    ):
                        continue
                    ratio = SequenceMatcher(
                        None,
                        normalized,
                        kept_normalized,
                        autojunk=False,
                    ).ratio()
                    if ratio >= near_duplicate_threshold:
                        duplicate_of = kept_sentence_id
                        break

            if duplicate_of:
                near_count += 1
                rejected = row.to_dict()
                rejected["cleaning_decision"] = "rejected"
                rejected["cleaning_reason"] = "near_duplicate_within_product"
                rejected["duplicate_of_sentence_id"] = duplicate_of
                rejected_rows.append(rejected)
                continue

            exact_seen[normalized] = sentence_id
            kept_norms.append((normalized, sentence_id))
            kept_rows.append(row)

    kept = pd.DataFrame(kept_rows).reset_index(drop=True)
    rejected = pd.DataFrame(rejected_rows)
    return kept, rejected, exact_count, near_count


def _clean_visual_strict(
    strict: pd.DataFrame,
    near_duplicate_threshold: float,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    residual_rejected: list[dict[str, Any]] = []
    residual_keep_mask: list[bool] = []
    residual_counts: dict[str, int] = {}

    for _, row in strict.iterrows():
        reason = classify_residual_false_positive(row["sentence"])
        if reason is None:
            residual_keep_mask.append(True)
        else:
            residual_keep_mask.append(False)
            residual_counts[reason] = residual_counts.get(reason, 0) + 1
            rejected = row.to_dict()
            rejected["cleaning_decision"] = "rejected"
            rejected["cleaning_reason"] = f"residual_false_positive:{reason}"
            rejected["duplicate_of_sentence_id"] = ""
            residual_rejected.append(rejected)

    residual_clean = strict.loc[residual_keep_mask].copy()
    deduped, duplicate_rejected, exact_count, near_count = _deduplicate_strict(
        residual_clean,
        near_duplicate_threshold=near_duplicate_threshold,
    )

    deduped["source_type"] = "visual_strict_cleaned"
    deduped["cleaning_decision"] = "kept"
    deduped["cleaning_reason"] = ""
    deduped["dedup_status"] = "unique_within_product"
    deduped["pipeline_version"] = PIPELINE_VERSION

    rejected_frames = []
    if residual_rejected:
        rejected_frames.append(pd.DataFrame(residual_rejected))
    if not duplicate_rejected.empty:
        rejected_frames.append(duplicate_rejected)

    if rejected_frames:
        rejected = pd.concat(rejected_frames, ignore_index=True, sort=False)
    else:
        rejected = pd.DataFrame(
            columns=list(strict.columns)
            + [
                "cleaning_decision",
                "cleaning_reason",
                "duplicate_of_sentence_id",
            ]
        )
    rejected["pipeline_version"] = PIPELINE_VERSION

    metrics = {
        "residual_false_positive_count": sum(residual_counts.values()),
        "exact_duplicate_count": exact_count,
        "near_duplicate_count": near_count,
        **{
            f"residual_reason.{reason}": count
            for reason, count in sorted(residual_counts.items())
        },
    }
    return deduped, rejected, metrics


def _evidence_from_rows(
    rows: pd.DataFrame,
    source_type: str,
    allowed_dimensions: set[str] | None = None,
) -> pd.DataFrame:
    evidence_rows: list[dict[str, Any]] = []

    for _, row in rows.iterrows():
        hits = extract_dimension_hits(row["sentence"])
        for hit in hits:
            if (
                allowed_dimensions is not None
                and hit.dimension_code not in allowed_dimensions
            ):
                continue
            evidence_rows.append(
                {
                    "parent_asin": row["parent_asin"],
                    "review_id": row.get("review_id", ""),
                    "sentence_id": row.get("sentence_id", ""),
                    "user_id": row.get("user_id", ""),
                    "sentence": row["sentence"],
                    "normalized_sentence": row.get(
                        "normalized_sentence",
                        normalize_sentence(row["sentence"]),
                    ),
                    "rating": row.get("rating"),
                    "verified_purchase": _safe_bool(
                        row.get("verified_purchase")
                    ),
                    "helpful_vote": _safe_int(row.get("helpful_vote")),
                    "source_type": source_type,
                    "dimension_code": hit.dimension_code,
                    "dimension_name_cn": hit.dimension_name_cn,
                    "polarity": hit.polarity,
                    "expression_raw": hit.expression_raw,
                    "expression_lemma": hit.expression_lemma,
                    "pipeline_version": PIPELINE_VERSION,
                }
            )

    if not evidence_rows:
        return pd.DataFrame(
            columns=[
                "parent_asin",
                "review_id",
                "sentence_id",
                "user_id",
                "sentence",
                "normalized_sentence",
                "rating",
                "verified_purchase",
                "helpful_vote",
                "source_type",
                "dimension_code",
                "dimension_name_cn",
                "polarity",
                "expression_raw",
                "expression_lemma",
                "pipeline_version",
            ]
        )
    evidence = pd.DataFrame(evidence_rows)
    return evidence.drop_duplicates(
        [
            "sentence_id",
            "source_type",
            "dimension_code",
            "expression_lemma",
        ],
        keep="first",
    ).reset_index(drop=True)


def _frequency_table(evidence: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "dimension_code",
        "dimension_name_cn",
        "polarity",
        "expression_lemma",
        "sentence_count",
        "product_count",
        "review_count",
        "reviewer_count",
        "strict_sentence_count",
        "recovered_sentence_count",
        "example_sentence",
    ]
    if evidence.empty:
        return pd.DataFrame(columns=columns)

    grouped_rows: list[dict[str, Any]] = []
    for keys, group in evidence.groupby(
        [
            "dimension_code",
            "dimension_name_cn",
            "polarity",
            "expression_lemma",
        ],
        sort=True,
    ):
        (
            dimension_code,
            dimension_name_cn,
            polarity,
            expression_lemma,
        ) = keys
        examples = sorted(
            set(group["sentence"].astype(str)),
            key=lambda value: (len(value), value.lower()),
        )
        grouped_rows.append(
            {
                "dimension_code": dimension_code,
                "dimension_name_cn": dimension_name_cn,
                "polarity": polarity,
                "expression_lemma": expression_lemma,
                "sentence_count": int(group["sentence_id"].nunique()),
                "product_count": int(group["parent_asin"].nunique()),
                "review_count": int(
                    group.loc[group["review_id"].astype(str) != "", "review_id"].nunique()
                ),
                "reviewer_count": int(
                    group.loc[group["user_id"].astype(str) != "", "user_id"].nunique()
                ),
                "strict_sentence_count": int(
                    group.loc[
                        group["source_type"] == "visual_strict_cleaned",
                        "sentence_id",
                    ].nunique()
                ),
                "recovered_sentence_count": int(
                    group.loc[
                        group["source_type"] == "recovered_visual",
                        "sentence_id",
                    ].nunique()
                ),
                "example_sentence": examples[0] if examples else "",
            }
        )

    return pd.DataFrame(grouped_rows, columns=columns).sort_values(
        ["product_count", "sentence_count", "dimension_code", "expression_lemma"],
        ascending=[False, False, True, True],
    )


def _dimension_table(
    strict_evidence: pd.DataFrame,
    combined_evidence: pd.DataFrame,
    pilot_min_products: int,
    core_min_products: int,
    recovery_confirmation_min_products: int,
) -> pd.DataFrame:
    rows = []
    for dimension in DIMENSIONS:
        strict_group = strict_evidence.loc[
            strict_evidence["dimension_code"] == dimension.code
        ]
        combined_group = combined_evidence.loc[
            combined_evidence["dimension_code"] == dimension.code
        ]
        strict_product_count = int(
            strict_group["parent_asin"].nunique()
        )
        total_product_count = int(
            combined_group["parent_asin"].nunique()
        )
        reviewer_count = int(
            combined_group.loc[
                combined_group["user_id"].astype(str) != "",
                "user_id",
            ].nunique()
        )
        rows.append(
            {
                "dimension_code": dimension.code,
                "dimension_name_cn": dimension.name_cn,
                "polarity": dimension.polarity,
                "strict_sentence_count": int(
                    strict_group["sentence_id"].nunique()
                ),
                "strict_product_count": strict_product_count,
                "recovered_sentence_count": int(
                    combined_group.loc[
                        combined_group["source_type"] == "recovered_visual",
                        "sentence_id",
                    ].nunique()
                ),
                "total_sentence_count": int(
                    combined_group["sentence_id"].nunique()
                ),
                "product_count": total_product_count,
                "reviewer_count": reviewer_count,
                "confirmed_for_recovery": int(
                    strict_product_count
                    >= recovery_confirmation_min_products
                ),
                "keep_for_pilot": int(
                    total_product_count >= pilot_min_products
                ),
                "keep_for_core_model": int(
                    total_product_count >= core_min_products
                ),
                "pilot_min_products": pilot_min_products,
                "core_min_products": core_min_products,
                "recovery_confirmation_min_products": (
                    recovery_confirmation_min_products
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["product_count", "total_sentence_count", "dimension_code"],
        ascending=[False, False, True],
    )


def _recover_uncertain(
    uncertain: pd.DataFrame,
    confirmed_dimensions: set[str],
    strict_normalized_keys: set[tuple[str, str]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    recovered_rows: list[dict[str, Any]] = []
    rejected_rows: list[dict[str, Any]] = []
    recovered_keys = set(strict_normalized_keys)

    ranked = _rank_for_dedup(uncertain)
    for _, row in ranked.iterrows():
        normalized = row["normalized_sentence"]
        key = (row["parent_asin"], normalized)

        if not normalized:
            rejected = row.to_dict()
            rejected["recovery_rejection_reason"] = "empty_after_normalization"
            rejected_rows.append(rejected)
            continue
        if key in recovered_keys:
            rejected = row.to_dict()
            rejected["recovery_rejection_reason"] = (
                "duplicate_of_strict_or_recovered"
            )
            rejected_rows.append(rejected)
            continue

        decision = targeted_recovery_decision(
            row["sentence"],
            confirmed_dimensions=confirmed_dimensions,
        )
        if not decision.accepted:
            rejected = row.to_dict()
            rejected["recovery_rejection_reason"] = (
                decision.rejection_reason
            )
            rejected_rows.append(rejected)
            continue

        recovered = row.to_dict()
        recovered["source_type"] = "recovered_visual"
        recovered["recovery_dimension_codes"] = "|".join(
            decision.dimension_codes
        )
        recovered["recovery_matched_expressions"] = "|".join(
            decision.matched_expressions
        )
        recovered["recovery_rule_version"] = RECOVERY_RULE_VERSION
        recovered["pipeline_version"] = PIPELINE_VERSION
        recovered_rows.append(recovered)
        recovered_keys.add(key)

    recovered_columns = list(uncertain.columns) + [
        "source_type",
        "recovery_dimension_codes",
        "recovery_matched_expressions",
        "recovery_rule_version",
        "pipeline_version",
    ]
    rejected_columns = list(uncertain.columns) + [
        "recovery_rejection_reason",
    ]
    recovered_df = pd.DataFrame(
        recovered_rows,
        columns=recovered_columns,
    )
    rejected_df = pd.DataFrame(
        rejected_rows,
        columns=rejected_columns,
    )
    return recovered_df, rejected_df


def _build_product_dimension_evidence(
    combined_evidence: pd.DataFrame,
) -> pd.DataFrame:
    columns = [
        "parent_asin",
        "dimension_code",
        "dimension_name_cn",
        "polarity",
        "strict_sentence_count",
        "recovered_sentence_count",
        "sentence_count",
        "review_count",
        "reviewer_count",
        "verified_sentence_count",
        "verified_sentence_ratio",
        "helpful_vote_sum",
    ]
    if combined_evidence.empty:
        return pd.DataFrame(columns=columns)

    rows: list[dict[str, Any]] = []
    for keys, group in combined_evidence.groupby(
        [
            "parent_asin",
            "dimension_code",
            "dimension_name_cn",
            "polarity",
        ],
        sort=True,
    ):
        parent_asin, dimension_code, dimension_name_cn, polarity = keys
        sentence_count = int(group["sentence_id"].nunique())
        verified_sentence_count = int(
            group.loc[
                group["verified_purchase"].map(_safe_bool),
                "sentence_id",
            ].nunique()
        )
        rows.append(
            {
                "parent_asin": parent_asin,
                "dimension_code": dimension_code,
                "dimension_name_cn": dimension_name_cn,
                "polarity": polarity,
                "strict_sentence_count": int(
                    group.loc[
                        group["source_type"] == "visual_strict_cleaned",
                        "sentence_id",
                    ].nunique()
                ),
                "recovered_sentence_count": int(
                    group.loc[
                        group["source_type"] == "recovered_visual",
                        "sentence_id",
                    ].nunique()
                ),
                "sentence_count": sentence_count,
                "review_count": int(
                    group.loc[
                        group["review_id"].astype(str) != "",
                        "review_id",
                    ].nunique()
                ),
                "reviewer_count": int(
                    group.loc[
                        group["user_id"].astype(str) != "",
                        "user_id",
                    ].nunique()
                ),
                "verified_sentence_count": verified_sentence_count,
                "verified_sentence_ratio": (
                    verified_sentence_count / sentence_count
                    if sentence_count
                    else 0.0
                ),
                "helpful_vote_sum": int(
                    group.drop_duplicates("sentence_id")[
                        "helpful_vote"
                    ].map(_safe_int).sum()
                ),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def _build_product_labels(
    product_stats: pd.DataFrame,
    combined_evidence: pd.DataFrame,
    product_dimension_evidence: pd.DataFrame,
    dimension_table: pd.DataFrame,
) -> pd.DataFrame:
    products = product_stats.copy()
    if "parent_asin" not in products.columns:
        raise ValueError("商品统计文件缺少parent_asin字段。")
    products["parent_asin"] = (
        products["parent_asin"].fillna("").astype(str).str.strip()
    )
    if products["parent_asin"].duplicated().any():
        duplicates = products.loc[
            products["parent_asin"].duplicated(),
            "parent_asin",
        ].head(10).tolist()
        raise ValueError(
            "商品统计文件parent_asin重复: " + ", ".join(duplicates)
        )

    product_index = pd.Index(products["parent_asin"], name="parent_asin")
    summary_columns: dict[str, pd.Series] = {}

    if combined_evidence.empty:
        for column in [
            "strict_imagery_sentence_count",
            "recovered_imagery_sentence_count",
            "total_imagery_sentence_count",
            "imagery_review_count",
            "imagery_reviewer_count",
        ]:
            summary_columns[column] = pd.Series(
                0,
                index=product_index,
                dtype="int64",
            )
    else:
        sentence_rows = combined_evidence.drop_duplicates(
            ["parent_asin", "sentence_id", "source_type"]
        )
        strict_counts = (
            sentence_rows.loc[
                sentence_rows["source_type"]
                == "visual_strict_cleaned"
            ]
            .groupby("parent_asin")["sentence_id"]
            .nunique()
        )
        recovered_counts = (
            sentence_rows.loc[
                sentence_rows["source_type"] == "recovered_visual"
            ]
            .groupby("parent_asin")["sentence_id"]
            .nunique()
        )
        total_counts = (
            sentence_rows.groupby("parent_asin")["sentence_id"].nunique()
        )
        review_counts = (
            combined_evidence.loc[
                combined_evidence["review_id"].astype(str) != ""
            ]
            .groupby("parent_asin")["review_id"]
            .nunique()
        )
        reviewer_counts = (
            combined_evidence.loc[
                combined_evidence["user_id"].astype(str) != ""
            ]
            .groupby("parent_asin")["user_id"]
            .nunique()
        )
        summary_columns = {
            "strict_imagery_sentence_count": (
                strict_counts.reindex(product_index, fill_value=0)
                .astype("int64")
            ),
            "recovered_imagery_sentence_count": (
                recovered_counts.reindex(product_index, fill_value=0)
                .astype("int64")
            ),
            "total_imagery_sentence_count": (
                total_counts.reindex(product_index, fill_value=0)
                .astype("int64")
            ),
            "imagery_review_count": (
                review_counts.reindex(product_index, fill_value=0)
                .astype("int64")
            ),
            "imagery_reviewer_count": (
                reviewer_counts.reindex(product_index, fill_value=0)
                .astype("int64")
            ),
        }

    label_columns: dict[str, pd.Series | int | str] = {
        **summary_columns,
    }
    label_columns["has_any_imagery_label"] = (
        summary_columns["total_imagery_sentence_count"] >= 1
    ).astype("int64")
    label_columns["eligible_model_pilot"] = label_columns[
        "has_any_imagery_label"
    ]
    label_columns["eligible_model_robust"] = (
        (summary_columns["imagery_review_count"] >= 2)
        | (summary_columns["imagery_reviewer_count"] >= 2)
    ).astype("int64")

    dimension_status = dimension_table.set_index("dimension_code")
    dimension_metrics = [
        "strict_sentence_count",
        "recovered_sentence_count",
        "sentence_count",
        "review_count",
        "reviewer_count",
        "verified_sentence_count",
        "helpful_vote_sum",
    ]

    for dimension in DIMENSIONS:
        code = dimension.code
        evidence = product_dimension_evidence.loc[
            product_dimension_evidence["dimension_code"] == code
        ]
        if evidence.empty:
            evidence_indexed = None
        else:
            evidence_indexed = evidence.set_index("parent_asin")

        metric_series: dict[str, pd.Series] = {}
        for metric in dimension_metrics:
            if evidence_indexed is None:
                series = pd.Series(
                    0,
                    index=product_index,
                    dtype="int64",
                )
            else:
                series = (
                    pd.to_numeric(
                        evidence_indexed[metric],
                        errors="coerce",
                    )
                    .reindex(product_index)
                    .fillna(0)
                    .astype("int64")
                )
            metric_series[metric] = series
            label_columns[f"{code}_{metric}"] = series

        verified_ratio = pd.Series(
            0.0,
            index=product_index,
            dtype="float64",
        )
        sentence_mask = metric_series["sentence_count"] > 0
        verified_ratio.loc[sentence_mask] = (
            metric_series["verified_sentence_count"].loc[sentence_mask]
            / metric_series["sentence_count"].loc[sentence_mask]
        )
        label_columns[f"{code}_verified_sentence_ratio"] = verified_ratio

        mention_share = pd.Series(
            0.0,
            index=product_index,
            dtype="float64",
        )
        total_mask = (
            summary_columns["total_imagery_sentence_count"] > 0
        )
        mention_share.loc[total_mask] = (
            metric_series["sentence_count"].loc[total_mask]
            / summary_columns["total_imagery_sentence_count"].loc[
                total_mask
            ]
        )
        label_columns[f"{code}_mention_share"] = mention_share

        keep_pilot = int(
            dimension_status.loc[code, "keep_for_pilot"]
        )
        keep_core = int(
            dimension_status.loc[code, "keep_for_core_model"]
        )
        label_columns[f"{code}_dimension_keep_pilot"] = pd.Series(
            keep_pilot,
            index=product_index,
            dtype="int64",
        )
        label_columns[f"{code}_dimension_keep_core"] = pd.Series(
            keep_core,
            index=product_index,
            dtype="int64",
        )
        label_columns[f"{code}_label_pilot"] = (
            (metric_series["sentence_count"] >= 1)
            & bool(keep_pilot)
        ).astype("int64")
        label_columns[f"{code}_label_robust"] = (
            (
                (metric_series["review_count"] >= 2)
                | (metric_series["reviewer_count"] >= 2)
            )
            & bool(keep_pilot)
        ).astype("int64")

    label_frame = pd.DataFrame(
        label_columns,
        index=product_index,
    ).reset_index(drop=True)

    positive_columns = [
        f"{dimension.code}_label_pilot"
        for dimension in DIMENSIONS
        if dimension.polarity == "positive"
    ]
    negative_columns = [
        f"{dimension.code}_label_pilot"
        for dimension in DIMENSIONS
        if dimension.polarity == "negative"
    ]
    robust_columns = [
        f"{dimension.code}_label_robust"
        for dimension in DIMENSIONS
    ]
    label_frame["pilot_positive_label_count"] = (
        label_frame[positive_columns].sum(axis=1).astype("int64")
    )
    label_frame["pilot_negative_label_count"] = (
        label_frame[negative_columns].sum(axis=1).astype("int64")
    )
    label_frame["robust_label_count"] = (
        label_frame[robust_columns].sum(axis=1).astype("int64")
    )
    label_frame["pipeline_version"] = PIPELINE_VERSION

    return pd.concat(
        [products.reset_index(drop=True), label_frame],
        axis=1,
    )

def _balanced_audit_sample(
    cleaned_strict: pd.DataFrame,
    recovered: pd.DataFrame,
    strict_rejected: pd.DataFrame,
    recovery_rejected: pd.DataFrame,
    sample_size: int,
    seed: int,
) -> pd.DataFrame:
    if sample_size <= 0:
        return pd.DataFrame()

    random_state = random.Random(seed)
    groups: list[tuple[str, pd.DataFrame]] = [
        ("clean_strict", cleaned_strict),
        ("recovered", recovered),
        ("strict_rejected", strict_rejected),
        ("uncertain_not_recovered", recovery_rejected),
    ]
    nonempty = [(name, frame) for name, frame in groups if not frame.empty]
    if not nonempty:
        return pd.DataFrame()

    per_group = max(1, sample_size // len(nonempty))
    sampled_frames: list[pd.DataFrame] = []
    for index, (name, frame) in enumerate(nonempty):
        n = min(per_group, len(frame))
        sampled = frame.sample(
            n=n,
            random_state=seed + index,
            replace=False,
        ).copy()
        sampled["audit_group"] = name
        sampled_frames.append(sampled)

    combined = pd.concat(sampled_frames, ignore_index=True, sort=False)
    if len(combined) < sample_size:
        all_rows = []
        for name, frame in nonempty:
            candidate = frame.copy()
            candidate["audit_group"] = name
            all_rows.append(candidate)
        pool = pd.concat(all_rows, ignore_index=True, sort=False)
        existing_ids = set(
            zip(
                combined.get("audit_group", ""),
                combined.get("sentence_id", ""),
            )
        )
        pool["_audit_key"] = list(
            zip(pool["audit_group"], pool.get("sentence_id", ""))
        )
        pool = pool.loc[~pool["_audit_key"].isin(existing_ids)]
        additional_n = min(sample_size - len(combined), len(pool))
        if additional_n > 0:
            additional = pool.sample(
                n=additional_n,
                random_state=seed + 99,
                replace=False,
            ).drop(columns=["_audit_key"])
            combined = pd.concat(
                [combined, additional],
                ignore_index=True,
                sort=False,
            )

    combined = combined.head(sample_size).copy()
    combined = combined.drop(columns=["sample_order"], errors="ignore")
    combined.insert(0, "sample_order", range(1, len(combined) + 1))
    desired = [
        "sample_order",
        "audit_group",
        "parent_asin",
        "review_id",
        "sentence_id",
        "sentence",
        "decision",
        "reason",
        "cleaning_reason",
        "recovery_rejection_reason",
        "recovery_dimension_codes",
        "recovery_matched_expressions",
        "rating",
        "verified_purchase",
        "helpful_vote",
    ]
    return _ensure_columns(combined, desired)[desired]


def _flatten_summary(summary: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for key, value in summary.items():
        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                rows.append(
                    {
                        "metric": f"{key}.{sub_key}",
                        "value": sub_value,
                    }
                )
        elif isinstance(value, list):
            rows.append(
                {
                    "metric": key,
                    "value": "|".join(map(str, value)),
                }
            )
        else:
            rows.append({"metric": key, "value": value})
    return pd.DataFrame(rows)


def _ensure_output_paths(
    output_dir: Path,
    output_format: str,
    overwrite: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    extension = "parquet" if output_format == "parquet" else "csv.gz"
    paths = [
        output_dir / f"23_visual_strict_cleaned.{extension}",
        output_dir / f"23b_visual_strict_rejected_or_duplicate.{extension}",
        output_dir / "24_affective_expression_frequency.csv",
        output_dir / f"24b_affective_expression_evidence.{extension}",
        output_dir / "25_preliminary_imagery_dimensions.csv",
        output_dir / f"26_uncertain_targeted_recovered.{extension}",
        output_dir / "26b_uncertain_recovery_rejections_sample.csv",
        output_dir / "27_product_imagery_labels.csv",
        output_dir / "27b_product_dimension_evidence.csv",
        output_dir / "28_imagery_extraction_summary.json",
        output_dir / "28b_imagery_extraction_summary.csv",
        output_dir / "29_imagery_audit_sample.csv",
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


def run_pipeline(
    *,
    classified_path: Path,
    product_stats_path: Path,
    output_dir: Path,
    output_format: str = "parquet",
    pilot_min_products: int = 15,
    core_min_products: int = 40,
    recovery_confirmation_min_products: int = 3,
    near_duplicate_threshold: float = 0.96,
    recovery_rejection_sample_size: int = 500,
    audit_sample_size: int = 600,
    audit_seed: int = 42,
    overwrite: bool = False,
) -> PipelineResult:
    classified_path = Path(classified_path).resolve()
    product_stats_path = Path(product_stats_path).resolve()
    output_dir = Path(output_dir).resolve()

    if not classified_path.is_file():
        raise FileNotFoundError(f"分类结果不存在: {classified_path}")
    if not product_stats_path.is_file():
        raise FileNotFoundError(f"商品统计文件不存在: {product_stats_path}")
    if not 0.80 <= near_duplicate_threshold <= 1.0:
        raise ValueError("near_duplicate_threshold必须位于0.80到1.0之间。")

    _ensure_output_paths(output_dir, output_format, overwrite)

    classified = _prepare_classified(_read_table(classified_path))
    product_stats = _read_table(product_stats_path)
    product_count = int(product_stats["parent_asin"].nunique())

    visual_strict = classified.loc[
        classified["decision"] == "visual_strict"
    ].copy()
    uncertain = classified.loc[
        classified["decision"] == "uncertain"
    ].copy()

    (
        cleaned_strict,
        strict_rejected,
        cleaning_metrics,
    ) = _clean_visual_strict(
        visual_strict,
        near_duplicate_threshold=near_duplicate_threshold,
    )

    strict_evidence = _evidence_from_rows(
        cleaned_strict,
        source_type="visual_strict_cleaned",
    )
    strict_dimension_product_counts = (
        strict_evidence.groupby("dimension_code")["parent_asin"]
        .nunique()
        .to_dict()
        if not strict_evidence.empty
        else {}
    )
    confirmed_dimensions = {
        dimension.code
        for dimension in DIMENSIONS
        if strict_dimension_product_counts.get(dimension.code, 0)
        >= recovery_confirmation_min_products
    }

    strict_keys = set(
        zip(
            cleaned_strict["parent_asin"],
            cleaned_strict["normalized_sentence"],
        )
    )
    recovered, recovery_rejected = _recover_uncertain(
        uncertain,
        confirmed_dimensions=confirmed_dimensions,
        strict_normalized_keys=strict_keys,
    )
    if recovered.empty:
        recovered_evidence = _evidence_from_rows(
            recovered,
            source_type="recovered_visual",
            allowed_dimensions=confirmed_dimensions,
        )
    else:
        recovered["normalized_sentence"] = recovered["sentence"].map(
            normalize_sentence
        )
        recovered_evidence = _evidence_from_rows(
            recovered,
            source_type="recovered_visual",
            allowed_dimensions=confirmed_dimensions,
        )

    combined_evidence = pd.concat(
        [strict_evidence, recovered_evidence],
        ignore_index=True,
        sort=False,
    )
    frequency = _frequency_table(combined_evidence)
    dimension_table = _dimension_table(
        strict_evidence,
        combined_evidence,
        pilot_min_products=pilot_min_products,
        core_min_products=core_min_products,
        recovery_confirmation_min_products=(
            recovery_confirmation_min_products
        ),
    )
    product_dimension_evidence = _build_product_dimension_evidence(
        combined_evidence
    )
    product_labels = _build_product_labels(
        product_stats,
        combined_evidence,
        product_dimension_evidence,
        dimension_table,
    )

    cleaned_path = _write_detail(
        cleaned_strict,
        output_dir,
        "23_visual_strict_cleaned",
        output_format,
    )
    strict_rejected_path = _write_detail(
        strict_rejected,
        output_dir,
        "23b_visual_strict_rejected_or_duplicate",
        output_format,
    )
    frequency.to_csv(
        output_dir / "24_affective_expression_frequency.csv",
        index=False,
        encoding="utf-8-sig",
    )
    evidence_path = _write_detail(
        combined_evidence,
        output_dir,
        "24b_affective_expression_evidence",
        output_format,
    )
    dimension_table.to_csv(
        output_dir / "25_preliminary_imagery_dimensions.csv",
        index=False,
        encoding="utf-8-sig",
    )
    recovered_path = _write_detail(
        recovered,
        output_dir,
        "26_uncertain_targeted_recovered",
        output_format,
    )

    if recovery_rejected.empty:
        recovery_rejection_sample = recovery_rejected
    else:
        recovery_rejection_sample = recovery_rejected.sample(
            n=min(
                recovery_rejection_sample_size,
                len(recovery_rejected),
            ),
            random_state=audit_seed + 17,
            replace=False,
        )
    recovery_rejection_sample.to_csv(
        output_dir / "26b_uncertain_recovery_rejections_sample.csv",
        index=False,
        encoding="utf-8-sig",
    )
    product_labels.to_csv(
        output_dir / "27_product_imagery_labels.csv",
        index=False,
        encoding="utf-8-sig",
    )
    product_dimension_evidence.to_csv(
        output_dir / "27b_product_dimension_evidence.csv",
        index=False,
        encoding="utf-8-sig",
    )

    audit_sample = _balanced_audit_sample(
        cleaned_strict,
        recovered,
        strict_rejected,
        recovery_rejected,
        sample_size=audit_sample_size,
        seed=audit_seed,
    )
    audit_sample.to_csv(
        output_dir / "29_imagery_audit_sample.csv",
        index=False,
        encoding="utf-8-sig",
    )

    dimension_coverage = {
        row["dimension_code"]: {
            "strict_product_count": int(row["strict_product_count"]),
            "product_count": int(row["product_count"]),
            "total_sentence_count": int(row["total_sentence_count"]),
            "keep_for_pilot": int(row["keep_for_pilot"]),
            "keep_for_core_model": int(row["keep_for_core_model"]),
        }
        for _, row in dimension_table.iterrows()
    }
    rejection_reason_counts = (
        recovery_rejected["recovery_rejection_reason"]
        .value_counts()
        .to_dict()
        if not recovery_rejected.empty
        else {}
    )
    products_with_any = int(
        (product_labels["has_any_imagery_label"] == 1).sum()
    )
    products_robust = int(
        (product_labels["eligible_model_robust"] == 1).sum()
    )

    summary = {
        "pipeline_version": PIPELINE_VERSION,
        "classified_input_path": str(classified_path),
        "product_stats_input_path": str(product_stats_path),
        "output_format": output_format,
        "product_count": product_count,
        "input_visual_strict_count": int(len(visual_strict)),
        "cleaned_visual_strict_count": int(len(cleaned_strict)),
        "residual_false_positive_count": int(
            cleaning_metrics["residual_false_positive_count"]
        ),
        "exact_duplicate_count": int(
            cleaning_metrics["exact_duplicate_count"]
        ),
        "near_duplicate_count": int(
            cleaning_metrics["near_duplicate_count"]
        ),
        "strict_sentences_without_any_imagery_expression": int(
            len(cleaned_strict)
            - strict_evidence["sentence_id"].nunique()
            if not strict_evidence.empty
            else len(cleaned_strict)
        ),
        "input_uncertain_count": int(len(uncertain)),
        "confirmed_dimensions_for_recovery": sorted(
            confirmed_dimensions
        ),
        "recovery_confirmation_min_products": (
            recovery_confirmation_min_products
        ),
        "recovered_visual_sentence_count": int(len(recovered)),
        "uncertain_not_recovered_count": int(len(recovery_rejected)),
        "combined_imagery_sentence_count": int(
            combined_evidence["sentence_id"].nunique()
            if not combined_evidence.empty
            else 0
        ),
        "combined_affective_evidence_row_count": int(
            len(combined_evidence)
        ),
        "products_with_any_imagery": products_with_any,
        "products_eligible_robust": products_robust,
        "pilot_min_products": pilot_min_products,
        "core_min_products": core_min_products,
        "dimensions_kept_for_pilot": dimension_table.loc[
            dimension_table["keep_for_pilot"] == 1,
            "dimension_code",
        ].tolist(),
        "dimensions_kept_for_core_model": dimension_table.loc[
            dimension_table["keep_for_core_model"] == 1,
            "dimension_code",
        ].tolist(),
        "dimension_coverage": dimension_coverage,
        "strict_cleaning_metrics": cleaning_metrics,
        "uncertain_rejection_reason_counts": {
            str(key): int(value)
            for key, value in rejection_reason_counts.items()
        },
        "policy": (
            "strict evidence is cleaned and deduplicated within product; "
            "uncertain recovery requires explicit package-expression relation "
            "and a dimension confirmed by strict product coverage"
        ),
        "cleaned_strict_output": str(cleaned_path),
        "strict_rejected_output": str(strict_rejected_path),
        "affective_evidence_output": str(evidence_path),
        "recovered_output": str(recovered_path),
    }
    with (output_dir / "28_imagery_extraction_summary.json").open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    _flatten_summary(summary).to_csv(
        output_dir / "28b_imagery_extraction_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    return PipelineResult(
        product_count=product_count,
        input_visual_strict_count=int(len(visual_strict)),
        cleaned_visual_strict_count=int(len(cleaned_strict)),
        exact_duplicate_count=int(
            cleaning_metrics["exact_duplicate_count"]
        ),
        near_duplicate_count=int(
            cleaning_metrics["near_duplicate_count"]
        ),
        residual_false_positive_count=int(
            cleaning_metrics["residual_false_positive_count"]
        ),
        input_uncertain_count=int(len(uncertain)),
        recovered_count=int(len(recovered)),
        combined_imagery_sentence_count=int(
            combined_evidence["sentence_id"].nunique()
            if not combined_evidence.empty
            else 0
        ),
        products_with_any_imagery=products_with_any,
    )



# =============================================================================
# Relation-constrained v2
# =============================================================================


@dataclass(frozen=True)
class Clause:
    text: str
    index: int
    start: int
    end: int


@dataclass(frozen=True)
class RelationEvidence:
    dimension_code: str
    dimension_name_cn: str
    polarity: str
    expression_raw: str
    expression_lemma: str
    source_dimension_code: str
    clause_text: str
    clause_index: int
    relation_type: str
    object_term: str
    negated: bool


@dataclass
class PipelineResultV2:
    product_count: int
    input_visual_strict_count: int
    cleaned_visual_strict_count: int
    relation_evidence_sentence_count: int
    input_uncertain_count: int
    recovered_sentence_count: int
    products_with_any_imagery: int
    environmental_packaging_sentence_count: int


V2_PACKAGING_OBJECT_PATTERN = re.compile(
    r"\b(?:packaging|package|packages|box|boxes|carton|cartons|tin|tins|"
    r"canister|canisters|container|containers|jar|jars|pouch|pouches|"
    r"wrapper|wrappers|envelope|envelopes|bag|bags|label|labels|logo|font|"
    r"typography|lettering|art|artwork|illustration|illustrations|graphic|"
    r"graphics|design|branding|presentation|cylinder|tube)\b",
    re.IGNORECASE,
)

V2_DESIGN_OBJECTS = {
    "label", "labels", "logo", "font", "typography", "lettering", "art",
    "artwork", "illustration", "illustrations", "graphic", "graphics",
    "design", "branding", "presentation",
}

V2_NONPACKAGE_SUBJECT_PATTERN = re.compile(
    r"\b(?:tea|flavo(?:r|ur)|taste|aroma|smell|brew|drink|liquid|cup|"
    r"product|leaves|flowers|petals|buds|ingredients|effect|quality)\b"
    r"(?:\s+\w+){0,2}\s+(?:is|are|was|were|looks?|looked|seems?|felt|"
    r"tastes?|smells?)\b",
    re.IGNORECASE,
)

V2_CONTRAST_SPLIT_PATTERN = re.compile(
    r"\s*(?:;|\s--\s|\s—\s|,?\s+\b(?:but|although|though|however|yet|"
    r"whereas|while|more importantly|on the other hand)\b\s*,?\s*)",
    re.IGNORECASE,
)

V2_ENVIRONMENTAL_PATTERN = re.compile(
    r"\b(?:minimal waste|less waste|reduce(?:d|s|ing)? waste|waste reduction|"
    r"environmentally friendly|environment-friendly|eco[- ]friendly|"
    r"sustainable|sustainability|recyclable|recycled|compostable|biodegradable|"
    r"unnecessary packaging|excess packaging|packaging reduction|less packaging|"
    r"minimal packaging|minimum packaging|uses? minimal packaging|"
    r"committed to (?:simple|minimal) packaging|no extra packaging|"
    r"no unnecessary packaging)\b",
    re.IGNORECASE,
)

V2_DATE_CONTEXT_PATTERN = re.compile(
    r"\b(?:dated|date|expires?|expired|expiration|expiry|best by|best-before|"
    r"use by)\b.{0,35}\b(?:through|until|on|20\d{2}|\d{1,2}[/-]\d{1,2})\b|"
    r"\b(?:dated through|dated until|best by|use by)\b",
    re.IGNORECASE,
)

V2_PRODUCT_STYLE_PATTERN = re.compile(
    r"\b(?:floral|botanical|earthy|natural|calming|soothing|bright|colorful|"
    r"colourful|premium|classic)\b.{0,20}\b(?:tea|flavo(?:r|ur)|aroma|brew|"
    r"drink|leaves|flowers|ingredients)\b|"
    r"\b(?:tea|flavo(?:r|ur)|aroma|brew|drink|leaves|flowers|ingredients)\b"
    r".{0,20}\b(?:floral|botanical|earthy|natural|calming|soothing|bright|"
    r"colorful|colourful|premium|classic)\b",
    re.IGNORECASE,
)

V2_NEGATION_PATTERN = re.compile(
    r"\b(?:not|never|hardly|barely|less|without)\b|n['’]t\b",
    re.IGNORECASE,
)

V2_VISUAL_CUE_PATTERN = re.compile(
    r"\b(?:design|style|look|looks|looking|appearance|aesthetic|visual|layout|"
    r"artwork|illustration|graphic|logo|font|typography|lettering|branding|"
    r"clean|sleek|modern|minimalist|minimalistic)\b",
    re.IGNORECASE,
)

V2_EXTRA_PATTERNS: tuple[tuple[str, str, str, str, re.Pattern[str]], ...] = (
    ("general_visual_appeal", "一般视觉吸引力", "positive", "amazing", re.compile(r"\bamazing\b", re.I)),
    ("general_visual_appeal", "一般视觉吸引力", "positive", "nice-looking", re.compile(r"\bnice[- ]looking\b", re.I)),
    ("premium_refined", "高级精致感", "positive", "premium", re.compile(r"\bpremium\b", re.I)),
    ("simple_modern", "简约现代感", "positive", "clean", re.compile(r"\bclean\b", re.I)),
    ("simple_modern", "简约现代感", "positive", "trendy", re.compile(r"\btrendy\b", re.I)),
    ("simple_modern", "简约现代感", "positive", "neat", re.compile(r"\bneat\b", re.I)),
    ("natural_botanical", "自然植物感", "positive", "natural", re.compile(r"\bnatural\b", re.I)),
)

V2_PRENOMINAL_MODIFIERS = {
    "little", "small", "large", "tiny", "round", "square", "metal", "metallic",
    "paper", "cardboard", "glass", "clear", "gift", "tea", "outer", "front",
    "new", "original", "yellow", "pink", "green", "black", "white", "gold",
    "golden", "red", "blue", "purple", "orange", "brown", "beige", "cream",
    "silver", "pastel", "dark", "bright", "floral", "botanical", "visual",
    "product", "card", "stock", "type", "cylinder", "shaped", "matching",
}


def split_clauses(text: str) -> tuple[Clause, ...]:
    """Split a review sentence at contrastive boundaries without splitting adjective lists."""
    raw = re.sub(r"\s+", " ", _safe_text(text)).strip()
    if not raw:
        return ()
    parts: list[Clause] = []
    cursor = 0
    index = 0
    for match in V2_CONTRAST_SPLIT_PATTERN.finditer(raw):
        segment = raw[cursor:match.start()].strip(" ,;:-")
        if segment:
            start = raw.find(segment, cursor, match.start() + 1)
            parts.append(Clause(segment, index, start, start + len(segment)))
            index += 1
        cursor = match.end()
    segment = raw[cursor:].strip(" ,;:-")
    if segment:
        start = raw.find(segment, cursor)
        parts.append(Clause(segment, index, start, start + len(segment)))
    return tuple(parts)


def _iter_v2_expression_hits(text: str) -> tuple[ExpressionHit, ...]:
    hits = list(extract_dimension_hits(text))
    for code, name_cn, polarity, lemma, pattern in V2_EXTRA_PATTERNS:
        for match in pattern.finditer(text):
            hits.append(
                ExpressionHit(
                    dimension_code=code,
                    dimension_name_cn=name_cn,
                    polarity=polarity,
                    expression_raw=match.group(0),
                    expression_lemma=lemma,
                    start=match.start(),
                    end=match.end(),
                )
            )
    unique: dict[tuple[str, str, int, int], ExpressionHit] = {}
    for hit in hits:
        unique[(hit.dimension_code, hit.expression_lemma, hit.start, hit.end)] = hit
    return tuple(sorted(unique.values(), key=lambda h: (h.start, h.end, h.dimension_code)))


def _nonvisual_contexts(text: str) -> set[str]:
    contexts: set[str] = set()
    if SHIPPING_DAMAGE_PATTERN.search(text):
        contexts.add("shipping_damage")
    if ORDER_PATTERN.search(text):
        contexts.add("order_fulfillment")
    if QUANTITY_PRICE_PATTERN.search(text):
        contexts.add("quantity_or_price")
    if STRUCTURAL_PATTERN.search(text) or re.search(
        r"\b(?:keep(?:s|ing)?|hold(?:s|ing)?|preserv(?:e|es|ing))\b.{0,35}\bfresh\b",
        text,
        re.IGNORECASE,
    ):
        contexts.add("structural_packaging")
    if V2_ENVIRONMENTAL_PATTERN.search(text):
        contexts.add("environmental_packaging")
    if TRACKING_LABEL_PATTERN.search(text):
        contexts.add("tracking_or_shipping_label")
    if EXPIRATION_PATTERN.search(text) or V2_DATE_CONTEXT_PATTERN.search(text):
        contexts.add("expiration_or_date_label")
    if PERSONAL_STORAGE_PATTERN.search(text):
        contexts.add("personal_storage_or_accessory")
    if SELLER_PACKAGING_PATTERN.search(text):
        contexts.add("seller_or_shipping_packaging")
    if PRODUCT_PRESENTATION_PATTERN.search(text):
        contexts.add("product_presentation")
    return contexts


def _negated_between(text: str, relation_start: int, hit_start: int) -> bool:
    context = text[max(relation_start, hit_start - 45):hit_start]
    return bool(V2_NEGATION_PATTERN.search(context))


def _simple_modern_is_visual(clause: str, hit: ExpressionHit) -> bool:
    lemma = hit.expression_lemma.lower()
    if lemma in {"minimalist", "clean-looking", "clean design", "modern", "sleek", "contemporary", "streamlined", "clean", "trendy", "neat"}:
        return True
    if lemma == "minimal" and re.search(r"\bminimal(?:ist|istic)\b", hit.expression_raw, re.I):
        return True
    if lemma in {"simple", "minimal"}:
        if V2_ENVIRONMENTAL_PATTERN.search(clause):
            return False
        return bool(V2_VISUAL_CUE_PATTERN.search(clause))
    return True


def _natural_term_is_visual(clause: str, hit: ExpressionHit, object_term: str) -> bool:
    if hit.expression_lemma in {"floral", "botanical", "earthy", "natural"}:
        if V2_PRODUCT_STYLE_PATTERN.search(clause):
            return False
        return object_term.lower() in V2_DESIGN_OBJECTS or bool(
            re.search(r"\b(?:design|style|look|appearance|artwork|illustration|graphic|pattern|print)\b", clause, re.I)
        )
    return True


def _make_relation_evidence(
    hit: ExpressionHit,
    clause: Clause,
    relation_type: str,
    object_term: str,
    relation_start: int,
) -> RelationEvidence | None:
    if hit.expression_lemma == "pretty" and re.match(
        r"\s+(?:large|small|big|huge|tiny|low|high|damaged|rough|good|bad|steep|strong)\b",
        clause.text[hit.end:],
        re.IGNORECASE,
    ):
        return None
    if hit.dimension_code == "simple_modern" and not _simple_modern_is_visual(clause.text, hit):
        return None
    if hit.dimension_code == "natural_botanical" and not _natural_term_is_visual(clause.text, hit, object_term):
        return None
    if hit.expression_lemma == "clinical" and re.search(r"\bclinical\s+(?:study|studies|trial|evidence|research)\b", clause.text, re.I):
        return None
    if hit.expression_lemma == "dated" and V2_DATE_CONTEXT_PATTERN.search(clause.text):
        return None

    negated = _negated_between(clause.text, relation_start, hit.start)
    if negated and hit.dimension_code != "negative_appearance":
        negative = DIMENSION_BY_CODE["negative_appearance"]
        return RelationEvidence(
            dimension_code="negative_appearance",
            dimension_name_cn=negative.name_cn,
            polarity="negative",
            expression_raw=hit.expression_raw,
            expression_lemma=f"negated:{hit.expression_lemma}",
            source_dimension_code=hit.dimension_code,
            clause_text=clause.text,
            clause_index=clause.index,
            relation_type=f"negated_{relation_type}",
            object_term=object_term,
            negated=True,
        )
    return RelationEvidence(
        dimension_code=hit.dimension_code,
        dimension_name_cn=hit.dimension_name_cn,
        polarity=hit.polarity,
        expression_raw=hit.expression_raw,
        expression_lemma=hit.expression_lemma,
        source_dimension_code=hit.dimension_code,
        clause_text=clause.text,
        clause_index=clause.index,
        relation_type=relation_type,
        object_term=object_term,
        negated=False,
    )


def _clause_relation_evidence(clause: Clause) -> tuple[RelationEvidence, ...]:
    text = clause.text
    object_matches = list(V2_PACKAGING_OBJECT_PATTERN.finditer(text))
    if not object_matches:
        return ()
    hits = _iter_v2_expression_hits(text)
    if not hits:
        return ()

    found: list[RelationEvidence] = []
    for object_match in object_matches:
        object_term = object_match.group(0)
        object_lower = object_term.lower()

        # 1. Affective adjective or style term before a packaging/design object.
        for hit in hits:
            if hit.end > object_match.start():
                continue
            between = text[hit.end:object_match.start()]
            tokens = [token.lower() for token in re.findall(r"\b[a-zA-Z'-]+\b", between)]
            if len(tokens) <= 3 and all(token in V2_PRENOMINAL_MODIFIERS for token in tokens):
                evidence = _make_relation_evidence(
                    hit, clause, "premodifier", object_term, hit.start
                )
                if evidence is not None:
                    found.append(evidence)

        # 2. Packaging/design object followed by copular or appearance predicate.
        tail = text[object_match.end():]
        copula = re.search(
            r"(?:\s+(?:itself|it|they|themselves))?(?:\s+\w+){0,2}\s+"
            r"(?:isn['’]?t|aren['’]?t|wasn['’]?t|weren['’]?t|is|are|was|were|"
            r"looks?|looked|seems?|seemed|feels?|felt|appears?|appeared|"
            r"could be|could look)\b",
            tail,
            re.I,
        )
        if copula and copula.start() <= 35:
            predicate_start = object_match.end() + copula.end()
            predicate_end = len(text)
            other_subject = V2_NONPACKAGE_SUBJECT_PATTERN.search(text, predicate_start)
            if other_subject:
                predicate_end = other_subject.start()
            for hit in hits:
                if predicate_start <= hit.start < predicate_end:
                    evidence = _make_relation_evidence(
                        hit, clause, "copular_predicate", object_term, object_match.start()
                    )
                    if evidence is not None:
                        found.append(evidence)

        # 3. Reviewer explicitly finds/considers the package adjective.
        prefix = text[max(0, object_match.start() - 40):object_match.start()]
        if re.search(r"\b(?:find|finds|found|consider|considers|considered)\s+(?:the\s+|this\s+)?$", prefix, re.I):
            for hit in hits:
                if object_match.end() <= hit.start <= object_match.end() + 45:
                    evidence = _make_relation_evidence(
                        hit, clause, "object_complement", object_term, object_match.start()
                    )
                    if evidence is not None:
                        found.append(evidence)

    # 5. Lexical negative patterns that already contain a packaging object.
    for hit in hits:
        if hit.dimension_code == "negative_appearance" and V2_PACKAGING_OBJECT_PATTERN.search(text):
            evidence = _make_relation_evidence(
                hit, clause, "lexical_negative", V2_PACKAGING_OBJECT_PATTERN.search(text).group(0), hit.start
            )
            if evidence is not None:
                found.append(evidence)

    unique: dict[tuple[str, str, int, str, bool], RelationEvidence] = {}
    for item in found:
        key = (
            item.dimension_code,
            item.expression_lemma,
            item.clause_index,
            item.object_term.lower(),
            item.negated,
        )
        unique[key] = item
    return tuple(sorted(unique.values(), key=lambda item: (item.clause_index, item.dimension_code, item.expression_lemma)))


def extract_relation_evidence(text: str) -> tuple[RelationEvidence, ...]:
    evidence: list[RelationEvidence] = []
    for clause in split_clauses(text):
        evidence.extend(_clause_relation_evidence(clause))
    # Prevent a negated source expression from coexisting with its positive label.
    negated_sources = {
        (item.clause_index, item.source_dimension_code, item.expression_raw.lower())
        for item in evidence
        if item.negated
    }
    filtered = [
        item
        for item in evidence
        if item.negated
        or (item.clause_index, item.dimension_code, item.expression_raw.lower()) not in negated_sources
    ]
    unique: dict[tuple[str, str, int, str], RelationEvidence] = {}
    for item in filtered:
        unique[(item.dimension_code, item.expression_lemma, item.clause_index, item.object_term.lower())] = item
    return tuple(unique.values())


def targeted_recovery_decision_v2(
    sentence: str,
    confirmed_dimensions: set[str],
) -> RecoveryDecision:
    contexts = _nonvisual_contexts(sentence)
    for priority in (
        "shipping_damage", "order_fulfillment", "quantity_or_price",
        "structural_packaging", "tracking_or_shipping_label",
        "expiration_or_date_label", "personal_storage_or_accessory",
        "seller_or_shipping_packaging", "product_presentation",
    ):
        if priority in contexts:
            return RecoveryDecision(False, rejection_reason=priority)

    evidence = [
        item for item in extract_relation_evidence(sentence)
        if item.dimension_code in confirmed_dimensions
    ]
    if "environmental_packaging" in contexts:
        non_simple = [item for item in evidence if item.dimension_code != "simple_modern"]
        if not non_simple:
            return RecoveryDecision(False, rejection_reason="environmental_packaging")
        evidence = non_simple
    if not evidence:
        reason = "no_packaging_object" if not V2_PACKAGING_OBJECT_PATTERN.search(sentence) else "no_confirmed_relation_evidence"
        return RecoveryDecision(False, rejection_reason=reason)
    return RecoveryDecision(
        True,
        dimension_codes=tuple(sorted({item.dimension_code for item in evidence})),
        matched_expressions=tuple(sorted({item.expression_lemma for item in evidence})),
    )


def _relation_evidence_from_rows(
    rows: pd.DataFrame,
    source_type: str,
    allowed_dimensions: set[str] | None = None,
) -> pd.DataFrame:
    output: list[dict[str, Any]] = []
    for _, row in rows.iterrows():
        for item in extract_relation_evidence(row["sentence"]):
            if allowed_dimensions is not None and item.dimension_code not in allowed_dimensions:
                continue
            output.append(
                {
                    "parent_asin": row["parent_asin"],
                    "review_id": row.get("review_id", ""),
                    "sentence_id": row.get("sentence_id", ""),
                    "user_id": row.get("user_id", ""),
                    "sentence": row["sentence"],
                    "normalized_sentence": row.get("normalized_sentence", normalize_sentence(row["sentence"])),
                    "clause_text": item.clause_text,
                    "clause_index": item.clause_index,
                    "relation_type": item.relation_type,
                    "object_term": item.object_term,
                    "negated": item.negated,
                    "source_dimension_code": item.source_dimension_code,
                    "rating": row.get("rating"),
                    "verified_purchase": _safe_bool(row.get("verified_purchase")),
                    "helpful_vote": _safe_int(row.get("helpful_vote")),
                    "source_type": source_type,
                    "dimension_code": item.dimension_code,
                    "dimension_name_cn": item.dimension_name_cn,
                    "polarity": item.polarity,
                    "expression_raw": item.expression_raw,
                    "expression_lemma": item.expression_lemma,
                    "pipeline_version": PIPELINE_VERSION,
                }
            )
    columns = [
        "parent_asin", "review_id", "sentence_id", "user_id", "sentence",
        "normalized_sentence", "clause_text", "clause_index", "relation_type",
        "object_term", "negated", "source_dimension_code", "rating",
        "verified_purchase", "helpful_vote", "source_type", "dimension_code",
        "dimension_name_cn", "polarity", "expression_raw", "expression_lemma",
        "pipeline_version",
    ]
    if not output:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(output, columns=columns).drop_duplicates(
        ["sentence_id", "source_type", "dimension_code", "expression_lemma", "clause_index", "object_term"],
        keep="first",
    ).reset_index(drop=True)


def _recover_uncertain_v2(
    uncertain: pd.DataFrame,
    confirmed_dimensions: set[str],
    strict_keys: set[tuple[str, str]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    recovered: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    seen = set(strict_keys)
    for _, row in _rank_for_dedup(uncertain).iterrows():
        key = (row["parent_asin"], row["normalized_sentence"])
        if key in seen:
            record = row.to_dict()
            record["recovery_rejection_reason"] = "duplicate_of_strict_or_recovered"
            rejected.append(record)
            continue
        decision = targeted_recovery_decision_v2(row["sentence"], confirmed_dimensions)
        if not decision.accepted:
            record = row.to_dict()
            record["recovery_rejection_reason"] = decision.rejection_reason
            rejected.append(record)
            continue
        record = row.to_dict()
        record["source_type"] = "recovered_visual_v2"
        record["recovery_dimension_codes"] = "|".join(decision.dimension_codes)
        record["recovery_matched_expressions"] = "|".join(decision.matched_expressions)
        record["recovery_rule_version"] = RECOVERY_RULE_VERSION
        record["pipeline_version"] = PIPELINE_VERSION
        recovered.append(record)
        seen.add(key)
    recovered_columns = list(dict.fromkeys(list(uncertain.columns) + [
        "source_type", "recovery_dimension_codes", "recovery_matched_expressions",
        "recovery_rule_version", "pipeline_version",
    ]))
    rejected_columns = list(dict.fromkeys(list(uncertain.columns) + ["recovery_rejection_reason"]))
    return pd.DataFrame(recovered, columns=recovered_columns), pd.DataFrame(rejected, columns=rejected_columns)


def _ensure_output_paths_v2(output_dir: Path, output_format: str, overwrite: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    extension = "parquet" if output_format == "parquet" else "csv.gz"
    paths = [
        output_dir / f"30_relation_constrained_sentence_evidence.{extension}",
        output_dir / "31_relation_constrained_imagery_dimensions.csv",
        output_dir / "32_product_imagery_labels_v2.csv",
        output_dir / "32b_product_dimension_evidence_v2.csv",
        output_dir / "33_relation_constrained_summary.json",
        output_dir / "33b_relation_constrained_summary.csv",
        output_dir / "34_relation_constrained_audit_sample.csv",
        output_dir / "35_nonvisual_context_counts.csv",
        output_dir / f"36_uncertain_targeted_recovered_v2.{extension}",
    ]
    existing = [path for path in paths if path.exists()]
    if existing and not overwrite:
        raise FileExistsError("输出文件已存在，请更换输出目录或添加--overwrite: " + ", ".join(path.name for path in existing))
    if overwrite:
        for path in existing:
            path.unlink()


def _build_v2_audit_sample(
    evidence: pd.DataFrame,
    recovered: pd.DataFrame,
    recovery_rejected: pd.DataFrame,
    cleaned_strict: pd.DataFrame,
    sample_size: int,
    seed: int,
) -> pd.DataFrame:
    groups: list[pd.DataFrame] = []
    if not evidence.empty:
        sample = evidence.sample(n=min(max(1, sample_size // 3), len(evidence)), random_state=seed).copy()
        sample["audit_group"] = "relation_evidence"
        groups.append(sample)
    strict_no_evidence = cleaned_strict.loc[~cleaned_strict["sentence_id"].isin(set(evidence["sentence_id"]) if not evidence.empty else set())].copy()
    if not strict_no_evidence.empty:
        sample = strict_no_evidence.sample(n=min(max(1, sample_size // 3), len(strict_no_evidence)), random_state=seed + 1).copy()
        sample["audit_group"] = "strict_without_relation_evidence"
        groups.append(sample)
    if not recovery_rejected.empty:
        sample = recovery_rejected.sample(n=min(max(1, sample_size // 3), len(recovery_rejected)), random_state=seed + 2).copy()
        sample["audit_group"] = "uncertain_not_recovered"
        groups.append(sample)
    if not recovered.empty:
        sample = recovered.sample(n=min(max(1, sample_size // 4), len(recovered)), random_state=seed + 3).copy()
        sample["audit_group"] = "recovered_v2"
        groups.append(sample)
    if not groups:
        return pd.DataFrame()
    groups = [frame.loc[:, ~frame.columns.duplicated()].copy() for frame in groups]
    combined = pd.concat(groups, ignore_index=True, sort=False).head(sample_size).copy()
    combined = combined.drop(columns=["sample_order"], errors="ignore")
    combined.insert(0, "sample_order", range(1, len(combined) + 1))
    return combined


def run_pipeline_v2(
    *,
    classified_path: Path,
    product_stats_path: Path,
    output_dir: Path,
    output_format: str = "parquet",
    pilot_min_products: int = 15,
    core_min_products: int = 40,
    recovery_confirmation_min_products: int = 3,
    near_duplicate_threshold: float = 0.96,
    audit_sample_size: int = 600,
    audit_seed: int = 42,
    overwrite: bool = False,
) -> PipelineResultV2:
    classified_path = Path(classified_path).resolve()
    product_stats_path = Path(product_stats_path).resolve()
    output_dir = Path(output_dir).resolve()
    if not classified_path.is_file():
        raise FileNotFoundError(f"分类结果不存在: {classified_path}")
    if not product_stats_path.is_file():
        raise FileNotFoundError(f"商品统计文件不存在: {product_stats_path}")
    _ensure_output_paths_v2(output_dir, output_format, overwrite)

    classified = _prepare_classified(_read_table(classified_path))
    product_stats = _read_table(product_stats_path)
    visual_strict = classified.loc[classified["decision"] == "visual_strict"].copy()
    uncertain = classified.loc[classified["decision"] == "uncertain"].copy()
    cleaned_strict, _, cleaning_metrics = _clean_visual_strict(
        visual_strict, near_duplicate_threshold=near_duplicate_threshold
    )

    strict_evidence = _relation_evidence_from_rows(cleaned_strict, "visual_strict_relation_v2")
    strict_counts = (
        strict_evidence.groupby("dimension_code")["parent_asin"].nunique().to_dict()
        if not strict_evidence.empty else {}
    )
    confirmed_dimensions = {
        dimension.code for dimension in DIMENSIONS
        if strict_counts.get(dimension.code, 0) >= recovery_confirmation_min_products
    }
    strict_keys = set(zip(cleaned_strict["parent_asin"], cleaned_strict["normalized_sentence"]))
    recovered, recovery_rejected = _recover_uncertain_v2(
        uncertain, confirmed_dimensions, strict_keys
    )
    recovered_evidence = _relation_evidence_from_rows(
        recovered, "recovered_visual_v2", allowed_dimensions=confirmed_dimensions
    )
    combined_evidence = pd.concat([strict_evidence, recovered_evidence], ignore_index=True, sort=False)

    dimension_table = _dimension_table(
        strict_evidence,
        combined_evidence,
        pilot_min_products=pilot_min_products,
        core_min_products=core_min_products,
        recovery_confirmation_min_products=recovery_confirmation_min_products,
    )
    product_dimension_evidence = _build_product_dimension_evidence(combined_evidence)
    product_labels = _build_product_labels(
        product_stats, combined_evidence, product_dimension_evidence, dimension_table
    )
    product_labels["pipeline_version"] = PIPELINE_VERSION

    evidence_path = _write_detail(
        combined_evidence, output_dir, "30_relation_constrained_sentence_evidence", output_format
    )
    dimension_table.to_csv(output_dir / "31_relation_constrained_imagery_dimensions.csv", index=False, encoding="utf-8-sig")
    product_labels.to_csv(output_dir / "32_product_imagery_labels_v2.csv", index=False, encoding="utf-8-sig")
    product_dimension_evidence.to_csv(output_dir / "32b_product_dimension_evidence_v2.csv", index=False, encoding="utf-8-sig")
    recovered_path = _write_detail(
        recovered, output_dir, "36_uncertain_targeted_recovered_v2", output_format
    )

    context_counts: dict[str, int] = {}
    for sentence in classified["sentence"]:
        for context in _nonvisual_contexts(sentence):
            context_counts[context] = context_counts.get(context, 0) + 1
    pd.DataFrame(
        [{"context": key, "sentence_count": value} for key, value in sorted(context_counts.items(), key=lambda item: (-item[1], item[0]))]
    ).to_csv(output_dir / "35_nonvisual_context_counts.csv", index=False, encoding="utf-8-sig")

    audit = _build_v2_audit_sample(
        combined_evidence, recovered, recovery_rejected, cleaned_strict,
        audit_sample_size, audit_seed,
    )
    audit.to_csv(output_dir / "34_relation_constrained_audit_sample.csv", index=False, encoding="utf-8-sig")

    products_with_any = int((product_labels["has_any_imagery_label"] == 1).sum())
    products_robust = int((product_labels["eligible_model_robust"] == 1).sum())
    environmental_count = int(context_counts.get("environmental_packaging", 0))
    dimension_coverage = {
        row["dimension_code"]: {
            "strict_product_count": int(row["strict_product_count"]),
            "product_count": int(row["product_count"]),
            "total_sentence_count": int(row["total_sentence_count"]),
            "keep_for_pilot": int(row["keep_for_pilot"]),
            "keep_for_core_model": int(row["keep_for_core_model"]),
        }
        for _, row in dimension_table.iterrows()
    }
    rejection_counts = (
        recovery_rejected["recovery_rejection_reason"].value_counts().to_dict()
        if not recovery_rejected.empty else {}
    )
    summary = {
        "pipeline_version": PIPELINE_VERSION,
        "classified_input_path": str(classified_path),
        "product_stats_input_path": str(product_stats_path),
        "product_count": int(product_stats["parent_asin"].nunique()),
        "input_visual_strict_count": int(len(visual_strict)),
        "cleaned_visual_strict_count": int(len(cleaned_strict)),
        "strict_relation_evidence_sentence_count": int(strict_evidence["sentence_id"].nunique()) if not strict_evidence.empty else 0,
        "strict_sentences_without_relation_evidence": int(len(cleaned_strict) - (strict_evidence["sentence_id"].nunique() if not strict_evidence.empty else 0)),
        "input_uncertain_count": int(len(uncertain)),
        "confirmed_dimensions_for_recovery": sorted(confirmed_dimensions),
        "recovered_visual_sentence_count": int(len(recovered)),
        "combined_relation_evidence_sentence_count": int(combined_evidence["sentence_id"].nunique()) if not combined_evidence.empty else 0,
        "combined_relation_evidence_row_count": int(len(combined_evidence)),
        "products_with_any_imagery": products_with_any,
        "products_eligible_robust": products_robust,
        "environmental_packaging_sentence_count": environmental_count,
        "dimension_coverage": dimension_coverage,
        "uncertain_rejection_reason_counts": {str(k): int(v) for k, v in rejection_counts.items()},
        "strict_cleaning_metrics": cleaning_metrics,
        "evidence_output": str(evidence_path),
        "recovered_output": str(recovered_path),
        "policy": "clause-level package-object-expression relations only; negated positives become negative evidence; environmental packaging does not create simple_modern labels",
    }
    with (output_dir / "33_relation_constrained_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    _flatten_summary(summary).to_csv(output_dir / "33b_relation_constrained_summary.csv", index=False, encoding="utf-8-sig")

    return PipelineResultV2(
        product_count=summary["product_count"],
        input_visual_strict_count=summary["input_visual_strict_count"],
        cleaned_visual_strict_count=summary["cleaned_visual_strict_count"],
        relation_evidence_sentence_count=summary["combined_relation_evidence_sentence_count"],
        input_uncertain_count=summary["input_uncertain_count"],
        recovered_sentence_count=summary["recovered_visual_sentence_count"],
        products_with_any_imagery=products_with_any,
        environmental_packaging_sentence_count=environmental_count,
    )



# =============================================================================
# Relation-constrained v2.1
# =============================================================================

PIPELINE_VERSION_V21 = "affective_imagery_labels_relation_v2.1.1"
RECOVERY_RULE_VERSION_V21 = "targeted_recovery_relation_v2.1.1"

SOURCE_KIND_STRICT = "strict"
SOURCE_KIND_RECOVERED = "recovered"
SOURCE_TYPE_STRICT_V21 = "visual_strict_relation_v21"
SOURCE_TYPE_RECOVERED_V21 = "recovered_visual_v21"

PACKAGE_LEVEL_OUTER = "outer_retail_package"
PACKAGE_LEVEL_INNER = "inner_sachet_wrapper"
PACKAGE_LEVEL_AMBIGUOUS = "ambiguous_package_level"

LABEL_SEMANTICS = (
    "1=observed consumer imagery mention; "
    "0=unlabeled/not observed, not a confirmed negative"
)


@dataclass(frozen=True)
class RelationEvidenceV21:
    dimension_code: str
    dimension_name_cn: str
    polarity: str
    expression_raw: str
    expression_lemma: str
    source_dimension_code: str
    clause_text: str
    clause_index: int
    relation_type: str
    object_term: str
    package_level: str
    negated: bool


@dataclass
class PipelineResultV21:
    product_count: int
    input_visual_strict_count: int
    cleaned_visual_strict_count: int
    relation_evidence_sentence_count: int
    input_uncertain_count: int
    recovered_sentence_count: int
    products_with_any_outer_imagery: int
    products_with_any_all_level_evidence: int
    environmental_packaging_sentence_count: int


V21_PACKAGING_OBJECT_PATTERN = re.compile(
    r"\b(?:packaging|package|packages|box|boxes|carton|cartons|tin|tins|"
    r"canister|canisters|container|containers|jar|jars|pouch|pouches|"
    r"wrapper|wrappers|envelope|envelopes|bag|bags|sachet|sachets|"
    r"packet|packets|tag|tags|label|labels|logo|font|typography|"
    r"lettering|art|artwork|illustration|illustrations|graphic|graphics|"
    r"design|branding|presentation|cylinder|tube)\b",
    re.IGNORECASE,
)

V21_OUTER_PHYSICAL_TERMS = {
    "packaging",
    "package",
    "packages",
    "box",
    "boxes",
    "carton",
    "cartons",
    "tin",
    "tins",
    "canister",
    "canisters",
    "container",
    "containers",
    "jar",
    "jars",
    "cylinder",
    "tube",
}

V21_INNER_TERMS = {
    "wrapper",
    "wrappers",
    "envelope",
    "envelopes",
    "sachet",
    "sachets",
    "packet",
    "packets",
    "tag",
    "tags",
}

V21_DESIGN_TERMS = {
    "label",
    "labels",
    "logo",
    "font",
    "typography",
    "lettering",
    "art",
    "artwork",
    "illustration",
    "illustrations",
    "graphic",
    "graphics",
    "design",
    "branding",
    "presentation",
}

V21_INNER_CONTEXT_PATTERN = re.compile(
    r"\b(?:tea ?bags?|teabags?|sachets?|individual(?:ly)?\s+"
    r"(?:wrapped|packaged|sealed)|each\s+(?:bag|sachet|packet|wrapper)|"
    r"foil\s+pouches?|pyramid\s+(?:bags?|sachets?)|tea\s+tags?|"
    r"strings?\s+and\s+tags?|inner\s+(?:bag|pouch|wrapper|packet))\b",
    re.IGNORECASE,
)

V21_OUTER_CONTEXT_PATTERN = re.compile(
    r"\b(?:outer|outside|retail|front|front-facing|main|product)\s+"
    r"(?:box|package|packaging|carton|tin|canister|pouch|bag|label|design)"
    r"\b|\bfront\s+of\s+(?:the\s+)?(?:box|package|packaging|carton|tin)\b",
    re.IGNORECASE,
)

V21_OUTER_PHYSICAL_PATTERN = re.compile(
    r"\b(?:packaging|package|packages|box|boxes|carton|cartons|tin|tins|"
    r"canister|canisters|container|containers|jar|jars|cylinder|tube)\b",
    re.IGNORECASE,
)

V21_BUNDLE_VALUE_PATTERN = re.compile(
    r"\b(?:sampler|sample|variety|assortment|bulk|value|family|multi[- ]?"
    r"pack|multipack|combo|combination)\s+(?:package|pack|box|boxes)\b|"
    r"\b(?:package|pack|box)\s+(?:deal|bundle)\b|"
    r"\b(?:amazing|great|good|better|excellent)\s+deal\b|"
    r"\b(?:better|great|good)\s+value\b|\bvalue\s+for\s+(?:the\s+)?price\b",
    re.IGNORECASE,
)

V21_SHIPPING_DAMAGE_PATTERN = re.compile(
    r"\b(?:arriv(?:e|ed|es|ing)|shipping|delivery|delivered|crushed|"
    r"damaged|dented|torn|ripped|broken|leak(?:ed|ing)|smashed|beat up|"
    r"beaten up|squashed|smooshed|smushed|crumpled|flattened|bent|"
    r"used\s+(?:it|the\s+(?:box|package|packaging))\s+like\s+a\s+football|"
    r"kicked\s+(?:the\s+)?(?:box|package|packaging))\b",
    re.IGNORECASE,
)
V21_NOT_ONLY_PATTERN = re.compile(
    r"\bnot\s+only\s+$",
    re.IGNORECASE,
)
V21_STRUCTURAL_EXTRA_PATTERN = re.compile(
    r"\b(?:easy to seal|seal(?:ed|ing)? back up|close(?:d|ing)? back up|"
    r"easy to reseal|keeps? the contents? fresh)\b",
    re.IGNORECASE,
)

V21_QUANTITY_EXTRA_PATTERN = re.compile(
    r"\b(?:not (?:a )?whole lot of product|isn['’]?t (?:a )?whole lot "
    r"of product|whole lot of product|not much product|very little product|"
    r"hardly any product|small amount of product)\b",
    re.IGNORECASE,
)

V21_COORDINATE_NEW_SUBJECT_PATTERN = re.compile(
    r",\s*(?:and|so|then)\s+"
    r"(?=(?:it|they|this|that|the|my|our|its|his|her|their|"
    r"tea|flavo(?:r|ur)|taste|aroma|scent|smell|brew|product|"
    r"leaves|flowers|petals|buds|ingredients|service)\b)",
    re.IGNORECASE,
)

V21_SENTENCE_BOUNDARY_PATTERN = re.compile(r"(?<=[.!?])\s+")

V21_PRODUCT_SUBJECT_PATTERN = re.compile(
    r"\b(?:tea|flavo(?:r|ur)|taste|aroma|scent|smell|brew|drink|liquid|"
    r"cup|product|leaves|flowers|petals|buds|ingredients|service|quality)"
    r"\b",
    re.IGNORECASE,
)

V21_CONTENT_COPULA_PATTERN = re.compile(
    r"\b(?:tea|flavo(?:r|ur)|taste|aroma|scent|smell|brew|drink|liquid|"
    r"product|leaves|flowers|petals|buds|ingredients)\b"
    r"(?:\s+\w+){0,3}\s+"
    r"(?:is|are|was|were|looks?|looked|seems?|seemed|tastes?|smells?)\b",
    re.IGNORECASE,
)

V21_VISUAL_EXTRA_PATTERNS: tuple[
    tuple[str, str, str, str, re.Pattern[str]], ...
] = (
    (
        "general_visual_appeal",
        "一般视觉吸引力",
        "positive",
        "nice-looking",
        re.compile(r"\bnice[- ]looking\b", re.IGNORECASE),
    ),
    (
        "premium_refined",
        "高级精致感",
        "positive",
        "premium",
        re.compile(r"\bpremium\b", re.IGNORECASE),
    ),
    (
        "simple_modern",
        "简约现代感",
        "positive",
        "trendy",
        re.compile(r"\btrendy\b", re.IGNORECASE),
    ),
    (
        "natural_botanical",
        "自然植物感",
        "positive",
        "natural",
        re.compile(r"\bnatural\b", re.IGNORECASE),
    ),
)

V21_AFFECTIVE_SINGLE_WORDS = {
    lemma.lower()
    for dimension in DIMENSIONS
    for lemma, _ in dimension.expressions
    if re.fullmatch(r"[a-zA-Z-]+", lemma)
}
V21_AFFECTIVE_SINGLE_WORDS.update(
    lemma
    for _, _, _, lemma, _ in V21_VISUAL_EXTRA_PATTERNS
)

V21_PRENOMINAL_MODIFIERS = {
    "little",
    "small",
    "large",
    "tiny",
    "round",
    "square",
    "metal",
    "metallic",
    "paper",
    "cardboard",
    "glass",
    "clear",
    "gift",
    "tea",
    "outer",
    "outside",
    "retail",
    "front",
    "stand-up",
    "new",
    "original",
    "yellow",
    "pink",
    "green",
    "black",
    "white",
    "gold",
    "golden",
    "red",
    "blue",
    "purple",
    "orange",
    "brown",
    "beige",
    "cream",
    "silver",
    "pastel",
    "dark",
    "bright",
    "floral",
    "botanical",
    "visual",
    "product",
    "card",
    "stock",
    "type",
    "cylinder",
    "shaped",
    "matching",
    "and",
    "or",
}
V21_PRENOMINAL_MODIFIERS.update(V21_AFFECTIVE_SINGLE_WORDS)

V21_PREDICATE_VERB_PATTERN = re.compile(
    r"^\s*(?:(?:itself|it|they|themselves)\s+)?"
    r"(?:isn['’]?t|aren['’]?t|wasn['’]?t|weren['’]?t|is|are|was|were|"
    r"looks?|looked|seems?|seemed|feels?|felt|appears?|appeared|"
    r"could\s+be|could\s+look)\b",
    re.IGNORECASE,
)

V21_PRETTY_ALLOWED_FOLLOWERS = {
    "nice",
    "cute",
    "beautiful",
    "attractive",
    "elegant",
    "lovely",
    "cool",
    "colorful",
    "colourful",
    "little",
    "small",
    "pink",
    "green",
    "black",
    "white",
    "gold",
    "golden",
    "box",
    "package",
    "packaging",
    "tin",
    "canister",
    "pouch",
    "label",
    "design",
    "wrapper",
}


def split_clauses_v21(text: str) -> tuple[Clause, ...]:
    """Split at contrast, sentence, and new-subject coordinate boundaries."""
    raw = re.sub(r"\s+", " ", _safe_text(text)).strip()
    if not raw:
        return ()

    coarse_parts: list[str] = []
    for sentence_part in V21_SENTENCE_BOUNDARY_PATTERN.split(raw):
        sentence_part = sentence_part.strip()
        if not sentence_part:
            continue
        cursor = 0
        for match in V2_CONTRAST_SPLIT_PATTERN.finditer(sentence_part):
            segment = sentence_part[cursor:match.start()].strip(" ,;:-")
            if segment:
                coarse_parts.append(segment)
            cursor = match.end()
        segment = sentence_part[cursor:].strip(" ,;:-")
        if segment:
            coarse_parts.append(segment)

    final_parts: list[str] = []
    for part in coarse_parts:
        cursor = 0
        for match in V21_COORDINATE_NEW_SUBJECT_PATTERN.finditer(part):
            segment = part[cursor:match.start()].strip(" ,;:-")
            if segment:
                final_parts.append(segment)
            cursor = match.end()
        segment = part[cursor:].strip(" ,;:-")
        if segment:
            final_parts.append(segment)

    clauses: list[Clause] = []
    search_start = 0
    for index, part in enumerate(final_parts):
        start = raw.lower().find(part.lower(), search_start)
        if start < 0:
            start = search_start
        clauses.append(
            Clause(
                text=part,
                index=index,
                start=start,
                end=start + len(part),
            )
        )
        search_start = start + len(part)
    return tuple(clauses)


def classify_package_level(
    object_term: str,
    clause_text: str,
) -> str:
    """Classify the reviewed packaging object relative to a main product image."""
    term = _safe_text(object_term).strip().lower()
    clause = _safe_text(clause_text)

    if term in V21_INNER_TERMS:
        return PACKAGE_LEVEL_INNER

    if V21_INNER_CONTEXT_PATTERN.search(clause):
        if term in V21_DESIGN_TERMS or term in {
            "bag",
            "bags",
            "pouch",
            "pouches",
            "label",
            "labels",
        }:
            return PACKAGE_LEVEL_INNER

    if term in {"pouch", "pouches"}:
        if re.search(
            r"\b(?:foil|individual|inner|tea)\s+(?:tea\s+)?pouch(?:es)?\b|"
            r"\bpouch(?:es)?\b.{0,25}\b(?:inside|within|in the box)\b",
            clause,
            re.IGNORECASE,
        ):
            return PACKAGE_LEVEL_INNER
        if re.search(
            r"\b(?:outer|outside|retail|front|front-facing|main|product|"
            r"resealable|stand[- ]up)(?:\s+(?:outer|outside|retail|front|"
            r"front-facing|main|product|resealable|stand[- ]up))*\s+pouch(?:es)?"
            r"\b|\bpouch(?:es)?\b.{0,25}"
            r"\b(?:outer|outside|retail|front-facing|stand[- ]up)\b",
            clause,
            re.IGNORECASE,
        ):
            return PACKAGE_LEVEL_OUTER
        return PACKAGE_LEVEL_AMBIGUOUS

    if term in {"bag", "bags"}:
        if re.search(
            r"\b(?:tea|pyramid|individual|inner)\s+bags?\b",
            clause,
            re.IGNORECASE,
        ):
            return PACKAGE_LEVEL_INNER
        if re.search(
            r"\b(?:outer|outside|retail|resealable|stand[- ]up|product)"
            r"\s+bags?\b",
            clause,
            re.IGNORECASE,
        ):
            return PACKAGE_LEVEL_OUTER
        return PACKAGE_LEVEL_AMBIGUOUS

    if term in V21_DESIGN_TERMS:
        if V21_INNER_CONTEXT_PATTERN.search(clause):
            return PACKAGE_LEVEL_INNER
        if (
            V21_OUTER_CONTEXT_PATTERN.search(clause)
            or V21_OUTER_PHYSICAL_PATTERN.search(clause)
        ):
            return PACKAGE_LEVEL_OUTER
        return PACKAGE_LEVEL_AMBIGUOUS

    if term in V21_OUTER_PHYSICAL_TERMS:
        return PACKAGE_LEVEL_OUTER

    return PACKAGE_LEVEL_AMBIGUOUS


def _iter_v21_expression_hits(text: str) -> tuple[ExpressionHit, ...]:
    hits = list(extract_dimension_hits(text))
    for code, name_cn, polarity, lemma, pattern in (
        V21_VISUAL_EXTRA_PATTERNS
    ):
        for match in pattern.finditer(text):
            hits.append(
                ExpressionHit(
                    dimension_code=code,
                    dimension_name_cn=name_cn,
                    polarity=polarity,
                    expression_raw=match.group(0),
                    expression_lemma=lemma,
                    start=match.start(),
                    end=match.end(),
                )
            )

    unique: dict[tuple[str, str, int, int], ExpressionHit] = {}
    for hit in hits:
        # "amazing", plain "clean", and plain "neat" from v2 are
        # intentionally not present in the v2.1 extra lexicon.
        unique[
            (
                hit.dimension_code,
                hit.expression_lemma,
                hit.start,
                hit.end,
            )
        ] = hit
    return tuple(
        sorted(
            unique.values(),
            key=lambda hit: (
                hit.start,
                hit.end,
                hit.dimension_code,
            ),
        )
    )


def _nonvisual_contexts_v21(text: str) -> set[str]:
    contexts = set(_nonvisual_contexts(text))
    if V21_SHIPPING_DAMAGE_PATTERN.search(text):
        contexts.add("shipping_damage")
    if V21_BUNDLE_VALUE_PATTERN.search(text):
        contexts.add("product_bundle_or_value")
    if V21_STRUCTURAL_EXTRA_PATTERN.search(text):
        contexts.add("structural_packaging")
    if V21_QUANTITY_EXTRA_PATTERN.search(text):
        contexts.add("quantity_or_price")
    return contexts


def _has_v21_relation_blocking_context(text: str) -> bool:
    contexts = _nonvisual_contexts_v21(text)
    return bool(
        contexts.intersection(
            {
                "shipping_damage",
                "product_bundle_or_value",
                "tracking_or_shipping_label",
                "expiration_or_date_label",
                "personal_storage_or_accessory",
                "seller_or_shipping_packaging",
                "product_presentation",
            }
        )
    )


def _negated_between_v21(
    text: str,
    relation_start: int,
    hit_start: int,
) -> bool:
    context = text[max(relation_start, hit_start - 45) : hit_start]
    context = V21_NOT_ONLY_PATTERN.sub(" ", context)
    return bool(V2_NEGATION_PATTERN.search(context))

def _pretty_is_visual(
    hit: ExpressionHit,
    clause: str,
    relation_type: str,
) -> bool:
    if hit.expression_lemma != "pretty":
        return True
    tail = clause[hit.end:]
    next_word_match = re.match(
        r"\s+([a-zA-Z'-]+)",
        tail,
    )
    if not next_word_match:
        return True
    next_word = next_word_match.group(1).lower()
    if next_word in V21_PRETTY_ALLOWED_FOLLOWERS:
        return True
    if relation_type == "premodifier" and (
        next_word in V21_PRENOMINAL_MODIFIERS
    ):
        return True
    return False


def _simple_modern_is_visual_v21(
    clause: str,
    hit: ExpressionHit,
) -> bool:
    if V2_ENVIRONMENTAL_PATTERN.search(clause):
        return False
    lemma = hit.expression_lemma.lower()
    if lemma in {
        "minimalist",
        "clean-looking",
        "clean design",
        "modern",
        "sleek",
        "contemporary",
        "streamlined",
        "trendy",
    }:
        return True
    if lemma in {"simple", "minimal"}:
        return bool(
            re.search(
                r"\b(?:design|style|look|appearance|visual|layout|"
                r"artwork|illustration|graphic|logo|font|typography|"
                r"lettering|branding|minimalist|minimalistic)\b",
                clause,
                re.IGNORECASE,
            )
        )
    return True


def _natural_term_is_visual_v21(
    clause: str,
    hit: ExpressionHit,
    object_term: str,
) -> bool:
    if hit.expression_lemma not in {
        "floral",
        "botanical",
        "earthy",
        "natural",
    }:
        return True
    if V2_PRODUCT_STYLE_PATTERN.search(clause):
        return False
    return (
        object_term.lower() in V21_DESIGN_TERMS
        or bool(
            re.search(
                r"\b(?:design|style|look|appearance|artwork|"
                r"illustration|graphic|pattern|print)\b",
                clause,
                re.IGNORECASE,
            )
        )
    )


def _make_relation_evidence_v21(
    hit: ExpressionHit,
    clause: Clause,
    relation_type: str,
    object_term: str,
    relation_start: int,
) -> RelationEvidenceV21 | None:
    if not _pretty_is_visual(
        hit,
        clause.text,
        relation_type,
    ):
        return None

    if (
        hit.dimension_code == "simple_modern"
        and not _simple_modern_is_visual_v21(
            clause.text,
            hit,
        )
    ):
        return None

    if (
        hit.dimension_code == "natural_botanical"
        and not _natural_term_is_visual_v21(
            clause.text,
            hit,
            object_term,
        )
    ):
        return None

    if (
        hit.expression_lemma == "clinical"
        and re.search(
            r"\bclinical\s+(?:study|studies|trial|evidence|research)\b",
            clause.text,
            re.IGNORECASE,
        )
    ):
        return None

    if (
        hit.expression_lemma == "dated"
        and V2_DATE_CONTEXT_PATTERN.search(clause.text)
    ):
        return None

    if hit.expression_lemma == "plain":
        if not re.search(
            r"\b(?:too|rather|very|quite|extremely|overly)\s+plain\b|"
            r"\bplain\b.{0,28}\b(?:unattractive|unappealing|boring|ugly|"
            r"cheap[- ]looking|tacky)\b|"
            r"\b(?:unattractive|unappealing|boring|ugly|tacky)\b"
            r".{0,28}\bplain\b",
            clause.text,
            re.IGNORECASE,
        ):
            return None

    if hit.expression_lemma == "not a fan":
        if not re.search(
            r"\b(?:look|looks|appearance|visual|design|style|color|colour|"
            r"artwork|illustration|graphic|font|logo|layout)\b",
            clause.text,
            re.IGNORECASE,
        ):
            return None

    if (
        hit.expression_lemma == "unappealing"
        and re.search(
            r"\b(?:condition|arrived|delivery|shipping|damaged|crushed|"
            r"dented|smooshed|torn|ripped)\b",
            clause.text,
            re.IGNORECASE,
        )
    ):
        return None

    package_level = classify_package_level(
        object_term,
        clause.text,
    )
    negated = _negated_between_v21(
        clause.text,
        relation_start,
        hit.start,
    )

    if (
        negated
        and hit.dimension_code != "negative_appearance"
    ):
        negative = DIMENSION_BY_CODE["negative_appearance"]
        return RelationEvidenceV21(
            dimension_code="negative_appearance",
            dimension_name_cn=negative.name_cn,
            polarity="negative",
            expression_raw=hit.expression_raw,
            expression_lemma=(
                f"negated:{hit.expression_lemma}"
            ),
            source_dimension_code=hit.dimension_code,
            clause_text=clause.text,
            clause_index=clause.index,
            relation_type=f"negated_{relation_type}",
            object_term=object_term,
            package_level=package_level,
            negated=True,
        )

    return RelationEvidenceV21(
        dimension_code=hit.dimension_code,
        dimension_name_cn=hit.dimension_name_cn,
        polarity=hit.polarity,
        expression_raw=hit.expression_raw,
        expression_lemma=hit.expression_lemma,
        source_dimension_code=hit.dimension_code,
        clause_text=clause.text,
        clause_index=clause.index,
        relation_type=relation_type,
        object_term=object_term,
        package_level=package_level,
        negated=False,
    )


def _predicate_span(
    text: str,
    object_end: int,
) -> tuple[int, int] | None:
    tail = text[object_end:]
    verb = V21_PREDICATE_VERB_PATTERN.search(tail)
    if not verb:
        return None

    predicate_start = object_end + verb.end()
    remaining = text[predicate_start:]
    boundary_positions = [len(remaining)]

    punctuation = re.search(r"[,;:]", remaining)
    if punctuation:
        boundary_positions.append(punctuation.start())

    coordinate = V21_COORDINATE_NEW_SUBJECT_PATTERN.search(
        remaining
    )
    if coordinate:
        boundary_positions.append(coordinate.start())

    content_subject = V21_CONTENT_COPULA_PATTERN.search(
        remaining
    )
    if content_subject:
        boundary_positions.append(content_subject.start())

    predicate_end = (
        predicate_start + min(boundary_positions)
    )
    return predicate_start, predicate_end


def _clause_relation_evidence_v21(
    clause: Clause,
) -> tuple[RelationEvidenceV21, ...]:
    text = clause.text
    object_matches = list(
        V21_PACKAGING_OBJECT_PATTERN.finditer(text)
    )
    if not object_matches:
        return ()

    hits = _iter_v21_expression_hits(text)
    if not hits:
        return ()

    found: list[RelationEvidenceV21] = []

    for object_match in object_matches:
        object_term = object_match.group(0)

        # Affective expression before the package/design object.
        for hit in hits:
            if hit.end > object_match.start():
                continue
            between = text[
                hit.end : object_match.start()
            ]
            tokens = [
                token.lower()
                for token in re.findall(
                    r"\b[a-zA-Z'-]+\b",
                    between,
                )
            ]
            if (
                len(tokens) <= 4
                and all(
                    token in V21_PRENOMINAL_MODIFIERS
                    for token in tokens
                )
            ):
                evidence = _make_relation_evidence_v21(
                    hit,
                    clause,
                    "premodifier",
                    object_term,
                    hit.start,
                )
                if evidence is not None:
                    found.append(evidence)

        # Package/design object followed immediately by an
        # appearance/c copular predicate. The span ends at a comma,
        # a new-subject coordinate, or a product-content subject.
        predicate_span = _predicate_span(
            text,
            object_match.end(),
        )
        if predicate_span is not None:
            predicate_start, predicate_end = predicate_span
            for hit in hits:
                if (
                    predicate_start
                    <= hit.start
                    < predicate_end
                ):
                    evidence = _make_relation_evidence_v21(
                        hit,
                        clause,
                        "copular_predicate",
                        object_term,
                        object_match.start(),
                    )
                    if evidence is not None:
                        found.append(evidence)

        # Explicit object complement:
        prefix = text[
            max(0, object_match.start() - 45)
            : object_match.start()
        ]
        if re.search(
            r"\b(?:find|finds|found|consider|considers|"
            r"considered)\s+(?:the\s+|this\s+)?$",
            prefix,
            re.IGNORECASE,
        ):
            for hit in hits:
                if (
                    object_match.end()
                    <= hit.start
                    <= object_match.end() + 45
                ):
                    evidence = _make_relation_evidence_v21(
                        hit,
                        clause,
                        "object_complement",
                        object_term,
                        object_match.start(),
                    )
                    if evidence is not None:
                        found.append(evidence)

    # Lexical negative phrases contain their own relation,
    # for example "not a fan of the packaging".
    first_object = (
        object_matches[0]
        if object_matches
        else None
    )
    if first_object is not None:
        for hit in hits:
            if (
                hit.dimension_code == "negative_appearance"
                and (
                    first_object.start() >= hit.start
                    and first_object.end() <= hit.end
                )
            ):
                evidence = _make_relation_evidence_v21(
                    hit,
                    clause,
                    "lexical_negative",
                    first_object.group(0),
                    hit.start,
                )
                if evidence is not None:
                    found.append(evidence)

    unique: dict[
        tuple[str, str, int, str, str, bool],
        RelationEvidenceV21,
    ] = {}
    for item in found:
        key = (
            item.dimension_code,
            item.expression_lemma,
            item.clause_index,
            item.object_term.lower(),
            item.package_level,
            item.negated,
        )
        unique[key] = item

    return tuple(
        sorted(
            unique.values(),
            key=lambda item: (
                item.clause_index,
                item.dimension_code,
                item.expression_lemma,
                item.object_term,
            ),
        )
    )


def extract_relation_evidence_v21(
    text: str,
) -> tuple[RelationEvidenceV21, ...]:
    """Extract package-object imagery relations with v2.1 exclusions."""
    evidence: list[RelationEvidenceV21] = []
    for clause in split_clauses_v21(text):
        if _has_v21_relation_blocking_context(clause.text):
            continue
        evidence.extend(
            _clause_relation_evidence_v21(clause)
        )

    negated_sources = {
        (
            item.clause_index,
            item.source_dimension_code,
            item.expression_raw.lower(),
            item.object_term.lower(),
        )
        for item in evidence
        if item.negated
    }

    filtered = [
        item
        for item in evidence
        if item.negated
        or (
            item.clause_index,
            item.dimension_code,
            item.expression_raw.lower(),
            item.object_term.lower(),
        )
        not in negated_sources
    ]

    unique: dict[
        tuple[str, str, int, str, str],
        RelationEvidenceV21,
    ] = {}
    for item in filtered:
        key = (
            item.dimension_code,
            item.expression_lemma,
            item.clause_index,
            item.object_term.lower(),
            item.package_level,
        )
        unique[key] = item
    return tuple(unique.values())


def targeted_recovery_decision_v21(
    sentence: str,
    confirmed_dimensions: set[str],
) -> RecoveryDecision:
    contexts = _nonvisual_contexts_v21(sentence)
    evidence = [
        item
        for item in extract_relation_evidence_v21(
            sentence
        )
        if item.dimension_code in confirmed_dimensions
    ]

    if "environmental_packaging" in contexts:
        evidence = [
            item
            for item in evidence
            if item.dimension_code != "simple_modern"
        ]
        if not evidence:
            return RecoveryDecision(
                False,
                rejection_reason=(
                    "environmental_packaging"
                ),
            )

    if evidence:
        return RecoveryDecision(
            True,
            dimension_codes=tuple(
                sorted(
                    {
                        item.dimension_code
                        for item in evidence
                    }
                )
            ),
            matched_expressions=tuple(
                sorted(
                    {
                        item.expression_lemma
                        for item in evidence
                    }
                )
            ),
        )

    for priority in (
        "shipping_damage",
        "order_fulfillment",
        "product_bundle_or_value",
        "quantity_or_price",
        "structural_packaging",
        "tracking_or_shipping_label",
        "expiration_or_date_label",
        "personal_storage_or_accessory",
        "seller_or_shipping_packaging",
        "product_presentation",
    ):
        if priority in contexts:
            return RecoveryDecision(
                False,
                rejection_reason=priority,
            )

    reason = (
        "no_packaging_object"
        if not V21_PACKAGING_OBJECT_PATTERN.search(
            sentence
        )
        else "no_confirmed_relation_evidence"
    )
    return RecoveryDecision(
        False,
        rejection_reason=reason,
    )

def _relation_evidence_from_rows_v21(
    rows: pd.DataFrame,
    *,
    source_kind: str,
    source_type: str,
    allowed_dimensions: set[str] | None = None,
) -> pd.DataFrame:
    columns = [
        "parent_asin",
        "review_id",
        "sentence_id",
        "user_id",
        "sentence",
        "normalized_sentence",
        "clause_text",
        "clause_index",
        "relation_type",
        "object_term",
        "package_level",
        "eligible_for_main_image_model",
        "negated",
        "source_dimension_code",
        "rating",
        "verified_purchase",
        "helpful_vote",
        "source_kind",
        "source_type",
        "dimension_code",
        "dimension_name_cn",
        "polarity",
        "expression_raw",
        "expression_lemma",
        "pipeline_version",
    ]
    records: list[dict[str, Any]] = []

    for _, row in rows.iterrows():
        for evidence in extract_relation_evidence_v21(
            row["sentence"]
        ):
            if (
                allowed_dimensions is not None
                and evidence.dimension_code
                not in allowed_dimensions
            ):
                continue
            records.append(
                {
                    "parent_asin": row["parent_asin"],
                    "review_id": row.get(
                        "review_id",
                        "",
                    ),
                    "sentence_id": row.get(
                        "sentence_id",
                        "",
                    ),
                    "user_id": row.get("user_id", ""),
                    "sentence": row["sentence"],
                    "normalized_sentence": row.get(
                        "normalized_sentence",
                        normalize_sentence(
                            row["sentence"]
                        ),
                    ),
                    "clause_text": (
                        evidence.clause_text
                    ),
                    "clause_index": (
                        evidence.clause_index
                    ),
                    "relation_type": (
                        evidence.relation_type
                    ),
                    "object_term": evidence.object_term,
                    "package_level": (
                        evidence.package_level
                    ),
                    "eligible_for_main_image_model": int(
                        evidence.package_level
                        == PACKAGE_LEVEL_OUTER
                    ),
                    "negated": evidence.negated,
                    "source_dimension_code": (
                        evidence.source_dimension_code
                    ),
                    "rating": row.get("rating"),
                    "verified_purchase": _safe_bool(
                        row.get(
                            "verified_purchase"
                        )
                    ),
                    "helpful_vote": _safe_int(
                        row.get("helpful_vote")
                    ),
                    "source_kind": source_kind,
                    "source_type": source_type,
                    "dimension_code": (
                        evidence.dimension_code
                    ),
                    "dimension_name_cn": (
                        evidence.dimension_name_cn
                    ),
                    "polarity": evidence.polarity,
                    "expression_raw": (
                        evidence.expression_raw
                    ),
                    "expression_lemma": (
                        evidence.expression_lemma
                    ),
                    "pipeline_version": (
                        PIPELINE_VERSION_V21
                    ),
                }
            )

    if not records:
        return pd.DataFrame(columns=columns)

    return (
        pd.DataFrame(records, columns=columns)
        .drop_duplicates(
            [
                "sentence_id",
                "source_kind",
                "dimension_code",
                "expression_lemma",
                "clause_index",
                "object_term",
                "package_level",
            ],
            keep="first",
        )
        .reset_index(drop=True)
    )


def _recover_uncertain_v21(
    uncertain: pd.DataFrame,
    confirmed_dimensions: set[str],
    strict_keys: set[tuple[str, str]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    recovered_rows: list[dict[str, Any]] = []
    rejected_rows: list[dict[str, Any]] = []
    used_keys = set(strict_keys)

    ranked = _rank_for_dedup(uncertain)
    for _, row in ranked.iterrows():
        normalized = row["normalized_sentence"]
        key = (
            row["parent_asin"],
            normalized,
        )
        if not normalized:
            rejected = row.to_dict()
            rejected[
                "recovery_rejection_reason"
            ] = "empty_after_normalization"
            rejected_rows.append(rejected)
            continue

        if key in used_keys:
            rejected = row.to_dict()
            rejected[
                "recovery_rejection_reason"
            ] = "duplicate_of_strict_or_recovered"
            rejected_rows.append(rejected)
            continue

        decision = (
            targeted_recovery_decision_v21(
                row["sentence"],
                confirmed_dimensions,
            )
        )
        if not decision.accepted:
            rejected = row.to_dict()
            rejected[
                "recovery_rejection_reason"
            ] = decision.rejection_reason
            rejected_rows.append(rejected)
            continue

        evidence = [
            item
            for item in extract_relation_evidence_v21(
                row["sentence"]
            )
            if item.dimension_code
            in confirmed_dimensions
        ]
        recovered = row.to_dict()
        recovered["source_kind"] = (
            SOURCE_KIND_RECOVERED
        )
        recovered["source_type"] = (
            SOURCE_TYPE_RECOVERED_V21
        )
        recovered[
            "recovery_dimension_codes"
        ] = "|".join(decision.dimension_codes)
        recovered[
            "recovery_matched_expressions"
        ] = "|".join(
            decision.matched_expressions
        )
        recovered[
            "recovery_package_levels"
        ] = "|".join(
            sorted(
                {
                    item.package_level
                    for item in evidence
                }
            )
        )
        recovered[
            "recovery_rule_version"
        ] = RECOVERY_RULE_VERSION_V21
        recovered[
            "pipeline_version"
        ] = PIPELINE_VERSION_V21
        recovered_rows.append(recovered)
        used_keys.add(key)

    recovered_columns = list(uncertain.columns) + [
        "source_kind",
        "source_type",
        "recovery_dimension_codes",
        "recovery_matched_expressions",
        "recovery_package_levels",
        "recovery_rule_version",
        "pipeline_version",
    ]
    rejected_columns = list(uncertain.columns) + [
        "recovery_rejection_reason",
    ]

    return (
        pd.DataFrame(
            recovered_rows,
            columns=recovered_columns,
        ),
        pd.DataFrame(
            rejected_rows,
            columns=rejected_columns,
        ),
    )


def _dimension_table_v21(
    strict_evidence: pd.DataFrame,
    combined_evidence: pd.DataFrame,
    *,
    pilot_min_products: int,
    core_min_products: int,
    recovery_confirmation_min_products: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    outer = combined_evidence.loc[
        combined_evidence["package_level"]
        == PACKAGE_LEVEL_OUTER
    ]
    strict_outer = strict_evidence.loc[
        strict_evidence["package_level"]
        == PACKAGE_LEVEL_OUTER
    ]

    for dimension in DIMENSIONS:
        code = dimension.code
        strict_group = strict_outer.loc[
            strict_outer["dimension_code"] == code
        ]
        total_group = outer.loc[
            outer["dimension_code"] == code
        ]
        recovered_group = total_group.loc[
            total_group["source_kind"]
            == SOURCE_KIND_RECOVERED
        ]
        all_level_group = combined_evidence.loc[
            combined_evidence["dimension_code"]
            == code
        ]

        product_count = int(
            total_group["parent_asin"].nunique()
        )
        rows.append(
            {
                "dimension_code": code,
                "dimension_name_cn": (
                    dimension.name_cn
                ),
                "polarity": dimension.polarity,
                "strict_sentence_count": int(
                    strict_group[
                        "sentence_id"
                    ].nunique()
                ),
                "recovered_sentence_count": int(
                    recovered_group[
                        "sentence_id"
                    ].nunique()
                ),
                "total_sentence_count": int(
                    total_group[
                        "sentence_id"
                    ].nunique()
                ),
                "strict_product_count": int(
                    strict_group[
                        "parent_asin"
                    ].nunique()
                ),
                "recovered_product_count": int(
                    recovered_group[
                        "parent_asin"
                    ].nunique()
                ),
                "product_count": product_count,
                "review_count": int(
                    total_group.loc[
                        total_group[
                            "review_id"
                        ].astype(str)
                        != "",
                        "review_id",
                    ].nunique()
                ),
                "reviewer_count": int(
                    total_group.loc[
                        total_group[
                            "user_id"
                        ].astype(str)
                        != "",
                        "user_id",
                    ].nunique()
                ),
                "inner_sentence_count": int(
                    all_level_group.loc[
                        all_level_group[
                            "package_level"
                        ]
                        == PACKAGE_LEVEL_INNER,
                        "sentence_id",
                    ].nunique()
                ),
                "ambiguous_sentence_count": int(
                    all_level_group.loc[
                        all_level_group[
                            "package_level"
                        ]
                        == PACKAGE_LEVEL_AMBIGUOUS,
                        "sentence_id",
                    ].nunique()
                ),
                "all_package_level_sentence_count": int(
                    all_level_group[
                        "sentence_id"
                    ].nunique()
                ),
                "confirmed_for_recovery": int(
                    strict_group[
                        "parent_asin"
                    ].nunique()
                    >= recovery_confirmation_min_products
                ),
                "keep_for_pilot": int(
                    product_count
                    >= pilot_min_products
                ),
                "keep_for_core_model": int(
                    product_count
                    >= core_min_products
                ),
                "pilot_min_products": (
                    pilot_min_products
                ),
                "core_min_products": (
                    core_min_products
                ),
                "recovery_confirmation_min_products": (
                    recovery_confirmation_min_products
                ),
                "main_image_package_level": (
                    PACKAGE_LEVEL_OUTER
                ),
                "label_semantics": LABEL_SEMANTICS,
            }
        )

    return pd.DataFrame(rows).sort_values(
        [
            "product_count",
            "total_sentence_count",
            "dimension_code",
        ],
        ascending=[False, False, True],
    )


def _product_dimension_evidence_v21(
    evidence: pd.DataFrame,
) -> pd.DataFrame:
    columns = [
        "parent_asin",
        "dimension_code",
        "dimension_name_cn",
        "polarity",
        "strict_sentence_count",
        "recovered_sentence_count",
        "sentence_count",
        "review_count",
        "reviewer_count",
        "verified_sentence_count",
        "verified_sentence_ratio",
        "helpful_vote_sum",
        "inner_sentence_count",
        "ambiguous_sentence_count",
        "all_package_level_sentence_count",
    ]
    if evidence.empty:
        return pd.DataFrame(columns=columns)

    rows: list[dict[str, Any]] = []
    for keys, group in evidence.groupby(
        [
            "parent_asin",
            "dimension_code",
            "dimension_name_cn",
            "polarity",
        ],
        sort=True,
    ):
        (
            parent_asin,
            dimension_code,
            dimension_name_cn,
            polarity,
        ) = keys
        outer = group.loc[
            group["package_level"]
            == PACKAGE_LEVEL_OUTER
        ]
        strict = outer.loc[
            outer["source_kind"]
            == SOURCE_KIND_STRICT
        ]
        recovered = outer.loc[
            outer["source_kind"]
            == SOURCE_KIND_RECOVERED
        ]
        outer_sentences = outer.drop_duplicates(
            "sentence_id"
        )
        sentence_count = int(
            outer["sentence_id"].nunique()
        )
        verified_count = int(
            outer.loc[
                outer["verified_purchase"].map(
                    _safe_bool
                ),
                "sentence_id",
            ].nunique()
        )
        rows.append(
            {
                "parent_asin": parent_asin,
                "dimension_code": (
                    dimension_code
                ),
                "dimension_name_cn": (
                    dimension_name_cn
                ),
                "polarity": polarity,
                "strict_sentence_count": int(
                    strict[
                        "sentence_id"
                    ].nunique()
                ),
                "recovered_sentence_count": int(
                    recovered[
                        "sentence_id"
                    ].nunique()
                ),
                "sentence_count": sentence_count,
                "review_count": int(
                    outer.loc[
                        outer["review_id"].astype(
                            str
                        )
                        != "",
                        "review_id",
                    ].nunique()
                ),
                "reviewer_count": int(
                    outer.loc[
                        outer["user_id"].astype(
                            str
                        )
                        != "",
                        "user_id",
                    ].nunique()
                ),
                "verified_sentence_count": (
                    verified_count
                ),
                "verified_sentence_ratio": (
                    verified_count
                    / sentence_count
                    if sentence_count
                    else 0.0
                ),
                "helpful_vote_sum": int(
                    outer_sentences[
                        "helpful_vote"
                    ]
                    .map(_safe_int)
                    .sum()
                ),
                "inner_sentence_count": int(
                    group.loc[
                        group["package_level"]
                        == PACKAGE_LEVEL_INNER,
                        "sentence_id",
                    ].nunique()
                ),
                "ambiguous_sentence_count": int(
                    group.loc[
                        group["package_level"]
                        == PACKAGE_LEVEL_AMBIGUOUS,
                        "sentence_id",
                    ].nunique()
                ),
                "all_package_level_sentence_count": int(
                    group["sentence_id"].nunique()
                ),
            }
        )

    return pd.DataFrame(rows, columns=columns)


def _product_labels_v21(
    product_stats: pd.DataFrame,
    evidence: pd.DataFrame,
    product_dimension_evidence: pd.DataFrame,
    dimension_table: pd.DataFrame,
) -> pd.DataFrame:
    products = product_stats.copy()
    if "parent_asin" not in products.columns:
        raise ValueError(
            "商品统计文件缺少parent_asin字段。"
        )
    products["parent_asin"] = (
        products["parent_asin"]
        .fillna("")
        .astype(str)
        .str.strip()
    )
    if products["parent_asin"].duplicated().any():
        raise ValueError(
            "商品统计文件存在重复parent_asin。"
        )

    product_index = pd.Index(
        products["parent_asin"],
        name="parent_asin",
    )

    def counts_for(
        package_level: str | None = None,
        source_kind: str | None = None,
        field: str = "sentence_id",
    ) -> pd.Series:
        frame = evidence
        if package_level is not None:
            frame = frame.loc[
                frame["package_level"]
                == package_level
            ]
        if source_kind is not None:
            frame = frame.loc[
                frame["source_kind"]
                == source_kind
            ]
        if frame.empty:
            return pd.Series(
                0,
                index=product_index,
                dtype="int64",
            )
        frame = frame.loc[
            frame[field].astype(str) != ""
        ]
        return (
            frame.groupby("parent_asin")[
                field
            ]
            .nunique()
            .reindex(
                product_index,
                fill_value=0,
            )
            .astype("int64")
        )

    outer_strict = counts_for(
        PACKAGE_LEVEL_OUTER,
        SOURCE_KIND_STRICT,
    )
    outer_recovered = counts_for(
        PACKAGE_LEVEL_OUTER,
        SOURCE_KIND_RECOVERED,
    )
    outer_total = counts_for(
        PACKAGE_LEVEL_OUTER
    )
    inner_total = counts_for(
        PACKAGE_LEVEL_INNER
    )
    ambiguous_total = counts_for(
        PACKAGE_LEVEL_AMBIGUOUS
    )
    all_total = counts_for()
    outer_reviews = counts_for(
        PACKAGE_LEVEL_OUTER,
        field="review_id",
    )
    outer_reviewers = counts_for(
        PACKAGE_LEVEL_OUTER,
        field="user_id",
    )

    label_data: dict[str, pd.Series | str] = {
        "strict_imagery_sentence_count": (
            outer_strict
        ),
        "recovered_imagery_sentence_count": (
            outer_recovered
        ),
        "outer_imagery_sentence_count": (
            outer_total
        ),
        "inner_imagery_sentence_count": (
            inner_total
        ),
        "ambiguous_imagery_sentence_count": (
            ambiguous_total
        ),
        "all_package_level_imagery_sentence_count": (
            all_total
        ),
        "outer_imagery_review_count": (
            outer_reviews
        ),
        "outer_imagery_reviewer_count": (
            outer_reviewers
        ),
        "has_any_outer_imagery_observed": (
            outer_total >= 1
        ).astype("int64"),
        "has_any_all_level_imagery_evidence": (
            all_total >= 1
        ).astype("int64"),
        "eligible_main_image_model_pilot": (
            outer_total >= 1
        ).astype("int64"),
        "eligible_main_image_model_robust": (
            (outer_reviews >= 2)
            | (outer_reviewers >= 2)
        ).astype("int64"),
    }

    dimension_status = (
        dimension_table.set_index(
            "dimension_code"
        )
    )

    for dimension in DIMENSIONS:
        code = dimension.code
        frame = product_dimension_evidence.loc[
            product_dimension_evidence[
                "dimension_code"
            ]
            == code
        ]
        if frame.empty:
            indexed = None
        else:
            indexed = frame.set_index(
                "parent_asin"
            )

        metric_names = [
            "strict_sentence_count",
            "recovered_sentence_count",
            "sentence_count",
            "review_count",
            "reviewer_count",
            "verified_sentence_count",
            "helpful_vote_sum",
            "inner_sentence_count",
            "ambiguous_sentence_count",
            "all_package_level_sentence_count",
        ]
        metrics: dict[str, pd.Series] = {}
        for metric in metric_names:
            if indexed is None:
                series = pd.Series(
                    0,
                    index=product_index,
                    dtype="int64",
                )
            else:
                series = (
                    pd.to_numeric(
                        indexed[metric],
                        errors="coerce",
                    )
                    .reindex(product_index)
                    .fillna(0)
                    .astype("int64")
                )
            metrics[metric] = series
            label_data[f"{code}_{metric}"] = (
                series
            )

        verified_ratio = pd.Series(
            0.0,
            index=product_index,
            dtype="float64",
        )
        mask = metrics["sentence_count"] > 0
        verified_ratio.loc[mask] = (
            metrics[
                "verified_sentence_count"
            ].loc[mask]
            / metrics[
                "sentence_count"
            ].loc[mask]
        )
        label_data[
            f"{code}_verified_sentence_ratio"
        ] = verified_ratio

        mention_share = pd.Series(
            0.0,
            index=product_index,
            dtype="float64",
        )
        total_mask = outer_total > 0
        mention_share.loc[total_mask] = (
            metrics["sentence_count"].loc[
                total_mask
            ]
            / outer_total.loc[total_mask]
        )
        label_data[
            f"{code}_outer_mention_share"
        ] = mention_share

        keep_pilot = int(
            dimension_status.loc[
                code,
                "keep_for_pilot",
            ]
        )
        keep_core = int(
            dimension_status.loc[
                code,
                "keep_for_core_model",
            ]
        )
        label_data[
            f"{code}_dimension_keep_pilot"
        ] = pd.Series(
            keep_pilot,
            index=product_index,
            dtype="int64",
        )
        label_data[
            f"{code}_dimension_keep_core"
        ] = pd.Series(
            keep_core,
            index=product_index,
            dtype="int64",
        )
        observed_pilot = (
            (metrics["sentence_count"] >= 1)
            & bool(keep_pilot)
        ).astype("int64")
        observed_core = (
            (metrics["sentence_count"] >= 1)
            & bool(keep_core)
        ).astype("int64")
        observed_robust = (
            (
                (metrics["review_count"] >= 2)
                | (
                    metrics["reviewer_count"]
                    >= 2
                )
            )
            & bool(keep_pilot)
        ).astype("int64")

        label_data[
            f"{code}_observed_positive_pilot"
        ] = observed_pilot
        label_data[
            f"{code}_observed_positive_core"
        ] = observed_core
        label_data[
            f"{code}_observed_positive_robust"
        ] = observed_robust
        # Compatibility aliases; semantics remain observed-positive
        # versus unlabeled, not positive versus confirmed-negative.
        label_data[
            f"{code}_label_pilot"
        ] = observed_pilot
        label_data[
            f"{code}_label_robust"
        ] = observed_robust

    label_frame = pd.DataFrame(
        label_data,
        index=product_index,
    ).reset_index(drop=True)
    label_frame[
        "main_image_package_level"
    ] = PACKAGE_LEVEL_OUTER
    label_frame["label_semantics"] = (
        LABEL_SEMANTICS
    )
    label_frame["pipeline_version"] = (
        PIPELINE_VERSION_V21
    )

    return pd.concat(
        [
            products.reset_index(drop=True),
            label_frame,
        ],
        axis=1,
    )


def _ensure_output_paths_v21(
    output_dir: Path,
    output_format: str,
    overwrite: bool,
) -> None:
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )
    extension = (
        "parquet"
        if output_format == "parquet"
        else "csv.gz"
    )
    paths = [
        output_dir
        / (
            "37_relation_constrained_"
            f"sentence_evidence_v21.{extension}"
        ),
        output_dir
        / (
            "38_relation_constrained_"
            "imagery_dimensions_v21.csv"
        ),
        output_dir
        / "39_product_imagery_labels_v21.csv",
        output_dir
        / (
            "39b_product_dimension_"
            "evidence_v21.csv"
        ),
        output_dir
        / (
            "40_relation_constrained_"
            "summary_v21.json"
        ),
        output_dir
        / (
            "40b_relation_constrained_"
            "summary_v21.csv"
        ),
        output_dir
        / (
            "41_relation_constrained_"
            "audit_sample_v21.csv"
        ),
        output_dir
        / "42_nonvisual_context_counts_v21.csv",
        output_dir
        / (
            "43_uncertain_targeted_"
            f"recovered_v21.{extension}"
        ),
    ]
    existing = [
        path
        for path in paths
        if path.exists()
    ]
    if existing and not overwrite:
        raise FileExistsError(
            "输出文件已存在，请更换输出目录或"
            "添加--overwrite: "
            + ", ".join(
                path.name
                for path in existing
            )
        )
    if overwrite:
        for path in existing:
            path.unlink()


AUDIT_GROUP_QUOTAS_V21 = (
    ("outer_relation_evidence", 120),
    ("recovered_v21", 120),
    ("uncertain_not_recovered", 120),
    ("inner_relation_evidence", 80),
    ("ambiguous_relation_evidence", 80),
    ("strict_without_relation_evidence", 80),
)
AUDIT_GROUP_ORDER_V21 = [
    name
    for name, _ in AUDIT_GROUP_QUOTAS_V21
]
AUDIT_RELATION_GROUPS_V21 = {
    "outer_relation_evidence",
    "recovered_v21",
    "inner_relation_evidence",
    "ambiguous_relation_evidence",
}
AUDIT_GROUP_SEED_OFFSETS_V21 = {
    name: index * 1009
    for index, name in enumerate(AUDIT_GROUP_ORDER_V21)
}
AUDIT_RELATION_KEY_COLUMNS_V21 = [
    "sentence_id",
    "clause_text",
    "object_term",
    "expression_lemma",
    "dimension_code",
    "package_level",
    "source_kind",
]
AUDIT_REQUIRED_RELATION_COLUMNS_V21 = [
    "clause_text",
    "object_term",
    "dimension_code",
    "package_level",
]


def _empty_like_v21(frame: pd.DataFrame | None = None) -> pd.DataFrame:
    if frame is None:
        return pd.DataFrame()
    return frame.iloc[0:0].copy()


def _filter_evidence_v21(
    evidence: pd.DataFrame,
    *,
    source_kind: str | None = None,
    package_level: str | None = None,
) -> pd.DataFrame:
    if evidence.empty:
        return _empty_like_v21(evidence)
    frame = evidence.copy()
    if source_kind is not None:
        if "source_kind" not in frame.columns:
            return _empty_like_v21(frame)
        frame = frame.loc[
            frame["source_kind"].astype(str)
            == source_kind
        ]
    if package_level is not None:
        if "package_level" not in frame.columns:
            return _empty_like_v21(frame)
        frame = frame.loc[
            frame["package_level"].astype(str)
            == package_level
        ]
    return frame.copy()


def _relation_audit_frame_v21(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    if frame.empty:
        return _empty_like_v21(frame)
    missing = [
        column
        for column in AUDIT_REQUIRED_RELATION_COLUMNS_V21
        if column not in frame.columns
    ]
    if missing:
        return _empty_like_v21(frame)
    valid = frame.copy()
    for column in AUDIT_REQUIRED_RELATION_COLUMNS_V21:
        valid = valid.loc[
            valid[column].notna()
            & valid[column].astype(str).str.strip().ne("")
        ]
    if valid.empty:
        return _empty_like_v21(frame)
    key_columns = [
        column
        for column in AUDIT_RELATION_KEY_COLUMNS_V21
        if column in valid.columns
    ]
    if key_columns:
        valid = valid.drop_duplicates(
            subset=key_columns,
            keep="first",
        )
    return valid.copy()


def _strict_without_relation_evidence_v21(
    cleaned_strict: pd.DataFrame,
    strict_evidence: pd.DataFrame,
) -> pd.DataFrame:
    if (
        cleaned_strict.empty
        or "sentence_id" not in cleaned_strict.columns
    ):
        return pd.DataFrame()
    strict_ids: set[str] = set()
    if (
        not strict_evidence.empty
        and "sentence_id" in strict_evidence.columns
    ):
        strict_ids = set(
            strict_evidence["sentence_id"]
            .astype(str)
        )
    return cleaned_strict.loc[
        ~cleaned_strict["sentence_id"]
        .astype(str)
        .isin(strict_ids)
    ].copy()


def _audit_targets_v21(sample_size: int) -> dict[str, int]:
    if sample_size <= 0:
        return {
            name: 0
            for name in AUDIT_GROUP_ORDER_V21
        }

    official_total = sum(
        quota
        for _, quota in AUDIT_GROUP_QUOTAS_V21
    )
    if sample_size == official_total:
        return dict(AUDIT_GROUP_QUOTAS_V21)

    targets: dict[str, int] = {
        name: 0
        for name in AUDIT_GROUP_ORDER_V21
    }
    remaining = sample_size
    if sample_size >= len(AUDIT_GROUP_ORDER_V21):
        for name in AUDIT_GROUP_ORDER_V21:
            targets[name] = 1
        remaining -= len(AUDIT_GROUP_ORDER_V21)

    scaled: list[tuple[float, int, str, int]] = []
    for order, (name, quota) in enumerate(
        AUDIT_GROUP_QUOTAS_V21
    ):
        raw = quota * sample_size / official_total
        base = int(raw)
        scaled.append((raw - base, order, name, base))

    if sample_size < len(AUDIT_GROUP_ORDER_V21):
        for _, _, name, base in scaled:
            targets[name] = base
        remaining = sample_size - sum(targets.values())
    else:
        for _, _, name, base in scaled:
            targets[name] = max(targets[name], base)
        remaining = sample_size - sum(targets.values())

    for _, _, name, _ in sorted(
        scaled,
        key=lambda item: (-item[0], item[1]),
    ):
        if remaining <= 0:
            break
        targets[name] += 1
        remaining -= 1

    while remaining > 0:
        for name in AUDIT_GROUP_ORDER_V21:
            if remaining <= 0:
                break
            targets[name] += 1
            remaining -= 1

    while sum(targets.values()) > sample_size:
        for name in reversed(AUDIT_GROUP_ORDER_V21):
            if (
                targets[name] > 0
                and sum(targets.values()) > sample_size
            ):
                targets[name] -= 1

    return targets


def _redistribute_audit_targets_v21(
    group_frames: dict[str, pd.DataFrame],
    sample_size: int,
) -> dict[str, int]:
    availability = {
        name: len(group_frames.get(name, pd.DataFrame()))
        for name in AUDIT_GROUP_ORDER_V21
    }
    total_available = sum(availability.values())
    if sample_size <= 0 or total_available <= 0:
        return {
            name: 0
            for name in AUDIT_GROUP_ORDER_V21
        }
    if total_available <= sample_size:
        return availability

    targets = _audit_targets_v21(sample_size)
    selected = {
        name: min(targets.get(name, 0), availability[name])
        for name in AUDIT_GROUP_ORDER_V21
    }
    remaining = sample_size - sum(selected.values())
    while remaining > 0:
        progressed = False
        for name in AUDIT_GROUP_ORDER_V21:
            if remaining <= 0:
                break
            if selected[name] < availability[name]:
                selected[name] += 1
                remaining -= 1
                progressed = True
        if not progressed:
            break
    return selected


def _dimension_seed_v21(
    seed: int,
    group_name: str,
    dimension_code: str,
) -> int:
    stable_dimension = sum(
        (index + 1) * ord(char)
        for index, char in enumerate(dimension_code)
    )
    return (
        seed
        + AUDIT_GROUP_SEED_OFFSETS_V21[group_name]
        + stable_dimension
    )


def _sample_relation_audit_group_v21(
    frame: pd.DataFrame,
    sample_n: int,
    seed: int,
    group_name: str,
) -> pd.DataFrame:
    if sample_n <= 0 or frame.empty:
        return _empty_like_v21(frame)
    if len(frame) <= sample_n:
        return frame.copy()
    if "dimension_code" not in frame.columns:
        return frame.sample(
            n=sample_n,
            random_state=(
                seed
                + AUDIT_GROUP_SEED_OFFSETS_V21[group_name]
            ),
            replace=False,
        ).copy()

    selected: list[pd.DataFrame] = []
    remaining = sample_n
    grouped = [
        (str(dimension), group.copy())
        for dimension, group in frame.groupby(
            "dimension_code",
            sort=False,
            dropna=False,
        )
    ]
    grouped.sort(
        key=lambda item: (len(item[1]), item[0])
    )
    for dimension, group in grouped:
        if remaining <= 0:
            break
        if len(group) <= remaining:
            selected.append(group.copy())
            remaining -= len(group)
            continue
        selected.append(
            group.sample(
                n=remaining,
                random_state=_dimension_seed_v21(
                    seed,
                    group_name,
                    dimension,
                ),
                replace=False,
            ).copy()
        )
        remaining = 0

    if not selected:
        return _empty_like_v21(frame)
    return pd.concat(
        selected,
        ignore_index=True,
        sort=False,
    )


def _sample_plain_audit_group_v21(
    frame: pd.DataFrame,
    sample_n: int,
    seed: int,
    group_name: str,
) -> pd.DataFrame:
    if sample_n <= 0 or frame.empty:
        return _empty_like_v21(frame)
    if len(frame) <= sample_n:
        return frame.copy()
    return frame.sample(
        n=sample_n,
        random_state=(
            seed
            + AUDIT_GROUP_SEED_OFFSETS_V21[group_name]
        ),
        replace=False,
    ).copy()


def _audit_sample_v21(
    evidence: pd.DataFrame,
    recovered: pd.DataFrame,
    rejected: pd.DataFrame,
    cleaned_strict: pd.DataFrame,
    sample_size: int,
    seed: int,
) -> pd.DataFrame:
    if sample_size <= 0:
        return pd.DataFrame()

    strict_evidence = _filter_evidence_v21(
        evidence,
        source_kind=SOURCE_KIND_STRICT,
    )
    recovered_relation_evidence = _filter_evidence_v21(
        recovered,
        source_kind=SOURCE_KIND_RECOVERED,
    )
    group_frames = {
        "outer_relation_evidence": _relation_audit_frame_v21(
            _filter_evidence_v21(
                strict_evidence,
                package_level=PACKAGE_LEVEL_OUTER,
            )
        ),
        "recovered_v21": _relation_audit_frame_v21(
            recovered_relation_evidence
        ),
        "uncertain_not_recovered": rejected.copy(),
        "inner_relation_evidence": _relation_audit_frame_v21(
            _filter_evidence_v21(
                strict_evidence,
                package_level=PACKAGE_LEVEL_INNER,
            )
        ),
        "ambiguous_relation_evidence": _relation_audit_frame_v21(
            _filter_evidence_v21(
                strict_evidence,
                package_level=PACKAGE_LEVEL_AMBIGUOUS,
            )
        ),
        "strict_without_relation_evidence": (
            _strict_without_relation_evidence_v21(
                cleaned_strict,
                strict_evidence,
            )
        ),
    }
    targets = _redistribute_audit_targets_v21(
        group_frames,
        sample_size,
    )

    groups: list[pd.DataFrame] = []
    for name in AUDIT_GROUP_ORDER_V21:
        frame = group_frames[name]
        requested = targets.get(name, 0)
        if name in AUDIT_RELATION_GROUPS_V21:
            sampled = _sample_relation_audit_group_v21(
                frame,
                requested,
                seed,
                name,
            )
        else:
            sampled = _sample_plain_audit_group_v21(
                frame,
                requested,
                seed,
                name,
            )
        if sampled.empty:
            continue
        sampled = sampled.copy()
        sampled["audit_group"] = name
        groups.append(sampled)

    if not groups:
        return pd.DataFrame()

    combined = pd.concat(
        [
            frame.loc[
                :,
                ~frame.columns.duplicated(),
            ].copy()
            for frame in groups
        ],
        ignore_index=True,
        sort=False,
    )
    combined = combined.drop(
        columns=["sample_order"],
        errors="ignore",
    )
    combined.insert(
        0,
        "sample_order",
        range(1, len(combined) + 1),
    )
    return combined




def _audit_redistribution_records_v21(
    capacities: dict[str, int],
    requested_quotas: dict[str, int],
    final_quotas: dict[str, int],
) -> list[dict[str, Any]]:
    recipient_remaining = {
        name: max(
            0,
            int(final_quotas[name])
            - min(int(requested_quotas[name]), int(capacities[name])),
        )
        for name in AUDIT_GROUP_ORDER_V21
    }
    records: list[dict[str, Any]] = []
    for shortage_group in AUDIT_GROUP_ORDER_V21:
        requested = int(requested_quotas[shortage_group])
        capacity = int(capacities[shortage_group])
        shortage = max(0, requested - capacity)
        if shortage == 0:
            continue
        remaining = shortage
        recipients: dict[str, int] = {}
        for recipient in AUDIT_GROUP_ORDER_V21:
            if remaining <= 0:
                break
            increment = min(recipient_remaining[recipient], remaining)
            if increment <= 0:
                continue
            recipients[recipient] = increment
            recipient_remaining[recipient] -= increment
            remaining -= increment
        records.append(
            {
                "shortage_group": shortage_group,
                "requested_quota": requested,
                "capacity": capacity,
                "shortage": shortage,
                "recipient_increments": recipients,
                "unfilled_shortage": remaining,
            }
        )
    return records


def _redistribute_audit_capacities_v21(
    capacities: dict[str, int],
    sample_size: int,
) -> dict[str, int]:
    total_available = sum(capacities.values())
    if sample_size <= 0 or total_available <= 0:
        return {name: 0 for name in AUDIT_GROUP_ORDER_V21}
    if total_available <= sample_size:
        return dict(capacities)

    targets = _audit_targets_v21(sample_size)
    selected = {
        name: min(targets[name], capacities[name])
        for name in AUDIT_GROUP_ORDER_V21
    }
    remaining = sample_size - sum(selected.values())
    while remaining > 0:
        progressed = False
        for name in AUDIT_GROUP_ORDER_V21:
            if remaining <= 0:
                break
            if selected[name] < capacities[name]:
                selected[name] += 1
                remaining -= 1
                progressed = True
        if not progressed:
            raise ValueError("unable to redistribute audit sample quota")
    return selected


def _validate_audit_manifest_v21(manifest: dict[str, Any]) -> None:
    expected_groups = set(AUDIT_GROUP_ORDER_V21)
    mapping_fields = [
        "sentence_group_capacities",
        "sentence_group_requested_quotas",
        "sentence_group_final_quotas",
        "sentence_group_actual_counts",
    ]
    normalized: dict[str, dict[str, int]] = {}
    for field in mapping_fields:
        value = manifest.get(field)
        if not isinstance(value, dict) or set(value) != expected_groups:
            raise ValueError(f"{field} must contain exactly the six audit group keys")
        if any(
            isinstance(count, bool)
            or not isinstance(count, int)
            or count < 0
            for count in value.values()
        ):
            raise ValueError(f"{field} must contain non-negative integers")
        normalized[field] = {
            name: int(value[name])
            for name in AUDIT_GROUP_ORDER_V21
        }

    for field in [
        "audit_seed",
        "audit_requested_sample_size",
        "audit_actual_sample_size",
    ]:
        value = manifest.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{field} must be a non-negative integer")

    requested_total = int(manifest["audit_requested_sample_size"])
    actual_total = int(manifest["audit_actual_sample_size"])
    capacities = normalized["sentence_group_capacities"]
    requested = normalized["sentence_group_requested_quotas"]
    final = normalized["sentence_group_final_quotas"]
    actual = normalized["sentence_group_actual_counts"]

    expected_requested = _audit_targets_v21(requested_total)
    if requested != expected_requested:
        raise ValueError("requested quotas do not match the deterministic targets")
    expected_final = _redistribute_audit_capacities_v21(
        capacities,
        requested_total,
    )
    if final != expected_final:
        raise ValueError("final quotas do not match deterministic capacity redistribution")
    if any(final[name] > capacities[name] for name in AUDIT_GROUP_ORDER_V21):
        raise ValueError("final quota exceeds group capacity")
    if actual != final:
        raise ValueError("actual counts must equal final quotas")
    if actual_total != sum(actual.values()):
        raise ValueError("actual sample size does not match actual group counts")

    redistribution = manifest.get("audit_redistribution")
    if not isinstance(redistribution, list):
        raise ValueError("audit_redistribution must be a list")
    expected_redistribution = _audit_redistribution_records_v21(
        capacities,
        requested,
        final,
    )
    if redistribution != expected_redistribution:
        raise ValueError("audit redistribution record is inconsistent")
def _audit_sample_and_manifest_v21(
    evidence: pd.DataFrame,
    recovered: pd.DataFrame,
    rejected: pd.DataFrame,
    cleaned_strict: pd.DataFrame,
    sample_size: int,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    audit_sample = _audit_sample_v21(
        evidence,
        recovered,
        rejected,
        cleaned_strict,
        sample_size,
        seed,
    )
    strict_evidence = _filter_evidence_v21(
        evidence,
        source_kind=SOURCE_KIND_STRICT,
    )
    recovered_relation_evidence = _filter_evidence_v21(
        recovered,
        source_kind=SOURCE_KIND_RECOVERED,
    )
    group_frames = {
        "outer_relation_evidence": _relation_audit_frame_v21(
            _filter_evidence_v21(
                strict_evidence,
                package_level=PACKAGE_LEVEL_OUTER,
            )
        ),
        "recovered_v21": _relation_audit_frame_v21(
            recovered_relation_evidence
        ),
        "uncertain_not_recovered": rejected.copy(),
        "inner_relation_evidence": _relation_audit_frame_v21(
            _filter_evidence_v21(
                strict_evidence,
                package_level=PACKAGE_LEVEL_INNER,
            )
        ),
        "ambiguous_relation_evidence": _relation_audit_frame_v21(
            _filter_evidence_v21(
                strict_evidence,
                package_level=PACKAGE_LEVEL_AMBIGUOUS,
            )
        ),
        "strict_without_relation_evidence": (
            _strict_without_relation_evidence_v21(
                cleaned_strict,
                strict_evidence,
            )
        ),
    }
    capacities = {
        name: int(len(group_frames[name]))
        for name in AUDIT_GROUP_ORDER_V21
    }
    requested_quotas = _audit_targets_v21(sample_size)
    final_quotas = _redistribute_audit_targets_v21(
        group_frames,
        sample_size,
    )
    actual_counts = {
        name: int((audit_sample["audit_group"] == name).sum())
        if "audit_group" in audit_sample.columns
        else 0
        for name in AUDIT_GROUP_ORDER_V21
    }
    manifest = {
        "audit_seed": int(seed),
        "audit_requested_sample_size": int(sample_size),
        "sentence_group_capacities": capacities,
        "sentence_group_requested_quotas": requested_quotas,
        "sentence_group_final_quotas": final_quotas,
        "sentence_group_actual_counts": actual_counts,
        "audit_redistribution": (
            _audit_redistribution_records_v21(
                capacities,
                requested_quotas,
                final_quotas,
            )
        ),
        "audit_actual_sample_size": int(len(audit_sample)),
    }
    _validate_audit_manifest_v21(manifest)
    return audit_sample, manifest



def run_pipeline_v21(
    *,
    classified_path: Path,
    product_stats_path: Path,
    output_dir: Path,
    output_format: str = "parquet",
    pilot_min_products: int = 15,
    core_min_products: int = 40,
    recovery_confirmation_min_products: int = 3,
    near_duplicate_threshold: float = 0.96,
    audit_sample_size: int = 600,
    audit_seed: int = 42,
    overwrite: bool = False,
) -> PipelineResultV21:
    classified_path = Path(
        classified_path
    ).resolve()
    product_stats_path = Path(
        product_stats_path
    ).resolve()
    output_dir = Path(output_dir).resolve()

    if not classified_path.is_file():
        raise FileNotFoundError(
            f"分类结果不存在: {classified_path}"
        )
    if not product_stats_path.is_file():
        raise FileNotFoundError(
            f"商品统计文件不存在: "
            f"{product_stats_path}"
        )
    if not 0.80 <= near_duplicate_threshold <= 1.0:
        raise ValueError(
            "near_duplicate_threshold必须位于"
            "0.80到1.0之间。"
        )

    _ensure_output_paths_v21(
        output_dir,
        output_format,
        overwrite,
    )

    classified = _prepare_classified(
        _read_table(classified_path)
    )
    product_stats = _read_table(
        product_stats_path
    )
    visual_strict = classified.loc[
        classified["decision"]
        == "visual_strict"
    ].copy()
    uncertain = classified.loc[
        classified["decision"]
        == "uncertain"
    ].copy()

    (
        cleaned_strict,
        _,
        cleaning_metrics,
    ) = _clean_visual_strict(
        visual_strict,
        near_duplicate_threshold=(
            near_duplicate_threshold
        ),
    )

    strict_evidence = (
        _relation_evidence_from_rows_v21(
            cleaned_strict,
            source_kind=SOURCE_KIND_STRICT,
            source_type=SOURCE_TYPE_STRICT_V21,
        )
    )
    strict_outer = strict_evidence.loc[
        strict_evidence["package_level"]
        == PACKAGE_LEVEL_OUTER
    ]
    strict_counts = (
        strict_outer.groupby(
            "dimension_code"
        )["parent_asin"]
        .nunique()
        .to_dict()
        if not strict_outer.empty
        else {}
    )
    confirmed_dimensions = {
        dimension.code
        for dimension in DIMENSIONS
        if strict_counts.get(
            dimension.code,
            0,
        )
        >= recovery_confirmation_min_products
    }

    strict_keys = set(
        zip(
            cleaned_strict["parent_asin"],
            cleaned_strict[
                "normalized_sentence"
            ],
        )
    )
    recovered, recovery_rejected = (
        _recover_uncertain_v21(
            uncertain,
            confirmed_dimensions,
            strict_keys,
        )
    )
    recovered_evidence = (
        _relation_evidence_from_rows_v21(
            recovered,
            source_kind=(
                SOURCE_KIND_RECOVERED
            ),
            source_type=(
                SOURCE_TYPE_RECOVERED_V21
            ),
            allowed_dimensions=(
                confirmed_dimensions
            ),
        )
    )
    combined_evidence = pd.concat(
        [
            strict_evidence,
            recovered_evidence,
        ],
        ignore_index=True,
        sort=False,
    )

    dimension_table = _dimension_table_v21(
        strict_evidence,
        combined_evidence,
        pilot_min_products=(
            pilot_min_products
        ),
        core_min_products=(
            core_min_products
        ),
        recovery_confirmation_min_products=(
            recovery_confirmation_min_products
        ),
    )
    product_dimension_evidence = (
        _product_dimension_evidence_v21(
            combined_evidence
        )
    )
    product_labels = _product_labels_v21(
        product_stats,
        combined_evidence,
        product_dimension_evidence,
        dimension_table,
    )

    evidence_path = _write_detail(
        combined_evidence,
        output_dir,
        (
            "37_relation_constrained_"
            "sentence_evidence_v21"
        ),
        output_format,
    )
    dimension_table.to_csv(
        output_dir
        / (
            "38_relation_constrained_"
            "imagery_dimensions_v21.csv"
        ),
        index=False,
        encoding="utf-8-sig",
    )
    product_labels.to_csv(
        output_dir
        / "39_product_imagery_labels_v21.csv",
        index=False,
        encoding="utf-8-sig",
    )
    product_dimension_evidence.to_csv(
        output_dir
        / (
            "39b_product_dimension_"
            "evidence_v21.csv"
        ),
        index=False,
        encoding="utf-8-sig",
    )
    recovered_path = _write_detail(
        recovered,
        output_dir,
        (
            "43_uncertain_targeted_"
            "recovered_v21"
        ),
        output_format,
    )

    context_counts: dict[str, int] = {}
    for sentence in classified["sentence"]:
        for context in _nonvisual_contexts_v21(
            sentence
        ):
            context_counts[context] = (
                context_counts.get(
                    context,
                    0,
                )
                + 1
            )
    pd.DataFrame(
        [
            {
                "context": key,
                "sentence_count": value,
            }
            for key, value in sorted(
                context_counts.items(),
                key=lambda item: (
                    -item[1],
                    item[0],
                ),
            )
        ]
    ).to_csv(
        output_dir
        / "42_nonvisual_context_counts_v21.csv",
        index=False,
        encoding="utf-8-sig",
    )

    audit, audit_manifest = _audit_sample_and_manifest_v21(
        combined_evidence,
        recovered_evidence,
        recovery_rejected,
        cleaned_strict,
        audit_sample_size,
        audit_seed,
    )
    audit.to_csv(
        output_dir
        / (
            "41_relation_constrained_"
            "audit_sample_v21.csv"
        ),
        index=False,
        encoding="utf-8-sig",
    )

    outer_evidence = combined_evidence.loc[
        combined_evidence["package_level"]
        == PACKAGE_LEVEL_OUTER
    ]
    all_products = int(
        combined_evidence[
            "parent_asin"
        ].nunique()
        if not combined_evidence.empty
        else 0
    )
    outer_products = int(
        outer_evidence[
            "parent_asin"
        ].nunique()
        if not outer_evidence.empty
        else 0
    )
    robust_products = int(
        (
            product_labels[
                "eligible_main_image_model_robust"
            ]
            == 1
        ).sum()
    )

    package_level_counts = {
        level: {
            "sentence_count": int(
                combined_evidence.loc[
                    combined_evidence[
                        "package_level"
                    ]
                    == level,
                    "sentence_id",
                ].nunique()
            ),
            "product_count": int(
                combined_evidence.loc[
                    combined_evidence[
                        "package_level"
                    ]
                    == level,
                    "parent_asin",
                ].nunique()
            ),
        }
        for level in (
            PACKAGE_LEVEL_OUTER,
            PACKAGE_LEVEL_INNER,
            PACKAGE_LEVEL_AMBIGUOUS,
        )
    }
    source_counts = {
        source_kind: {
            "sentence_count": int(
                combined_evidence.loc[
                    combined_evidence[
                        "source_kind"
                    ]
                    == source_kind,
                    "sentence_id",
                ].nunique()
            ),
            "product_count": int(
                combined_evidence.loc[
                    combined_evidence[
                        "source_kind"
                    ]
                    == source_kind,
                    "parent_asin",
                ].nunique()
            ),
        }
        for source_kind in (
            SOURCE_KIND_STRICT,
            SOURCE_KIND_RECOVERED,
        )
    }

    dimension_coverage = {
        row["dimension_code"]: {
            "strict_sentence_count": int(
                row["strict_sentence_count"]
            ),
            "recovered_sentence_count": int(
                row[
                    "recovered_sentence_count"
                ]
            ),
            "strict_product_count": int(
                row["strict_product_count"]
            ),
            "recovered_product_count": int(
                row[
                    "recovered_product_count"
                ]
            ),
            "product_count": int(
                row["product_count"]
            ),
            "inner_sentence_count": int(
                row["inner_sentence_count"]
            ),
            "ambiguous_sentence_count": int(
                row[
                    "ambiguous_sentence_count"
                ]
            ),
            "keep_for_pilot": int(
                row["keep_for_pilot"]
            ),
            "keep_for_core_model": int(
                row["keep_for_core_model"]
            ),
        }
        for _, row in dimension_table.iterrows()
    }
    rejection_counts = (
        recovery_rejected[
            "recovery_rejection_reason"
        ]
        .value_counts()
        .to_dict()
        if not recovery_rejected.empty
        else {}
    )

    summary = {
        "pipeline_version": (
            PIPELINE_VERSION_V21
        ),
        "classified_input_path": str(
            classified_path
        ),
        "product_stats_input_path": str(
            product_stats_path
        ),
        "product_count": int(
            product_stats[
                "parent_asin"
            ].nunique()
        ),
        "input_visual_strict_count": int(
            len(visual_strict)
        ),
        "cleaned_visual_strict_count": int(
            len(cleaned_strict)
        ),
        "strict_relation_evidence_sentence_count": int(
            strict_evidence[
                "sentence_id"
            ].nunique()
            if not strict_evidence.empty
            else 0
        ),
        "strict_outer_relation_evidence_sentence_count": int(
            strict_outer[
                "sentence_id"
            ].nunique()
            if not strict_outer.empty
            else 0
        ),
        "strict_sentences_without_relation_evidence": int(
            len(cleaned_strict)
            - (
                strict_evidence[
                    "sentence_id"
                ].nunique()
                if not strict_evidence.empty
                else 0
            )
        ),
        "input_uncertain_count": int(
            len(uncertain)
        ),
        "confirmed_dimensions_for_recovery": (
            sorted(confirmed_dimensions)
        ),
        "recovered_visual_sentence_count": int(
            len(recovered)
        ),
        "combined_relation_evidence_sentence_count": int(
            combined_evidence[
                "sentence_id"
            ].nunique()
            if not combined_evidence.empty
            else 0
        ),
        "combined_relation_evidence_row_count": int(
            len(combined_evidence)
        ),
        "products_with_any_imagery": (
            outer_products
        ),
        "products_with_any_outer_imagery": (
            outer_products
        ),
        "products_with_any_all_level_evidence": (
            all_products
        ),
        "products_eligible_main_image_robust": (
            robust_products
        ),
        "environmental_packaging_sentence_count": int(
            context_counts.get(
                "environmental_packaging",
                0,
            )
        ),
        "product_bundle_or_value_sentence_count": int(
            context_counts.get(
                "product_bundle_or_value",
                0,
            )
        ),
        "main_image_package_level": (
            PACKAGE_LEVEL_OUTER
        ),
        "label_semantics": LABEL_SEMANTICS,
        **audit_manifest,
        "source_kind_counts": source_counts,
        "package_level_counts": (
            package_level_counts
        ),
        "dimension_coverage": (
            dimension_coverage
        ),
        "uncertain_rejection_reason_counts": {
            str(key): int(value)
            for key, value in (
                rejection_counts.items()
            )
        },
        "strict_cleaning_metrics": (
            cleaning_metrics
        ),
        "evidence_output": str(
            evidence_path
        ),
        "recovered_output": str(
            recovered_path
        ),
        "policy": (
            "clause-level package-object-expression relations; "
            "canonical strict/recovered source_kind; "
            "outer retail package only for main-image labels; "
            "inner and ambiguous evidence retained for audit; "
            "observed-positive versus unlabeled semantics"
        ),
    }
    with (
        output_dir
        / (
            "40_relation_constrained_"
            "summary_v21.json"
        )
    ).open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            summary,
            handle,
            ensure_ascii=False,
            indent=2,
        )
    _flatten_summary(summary).to_csv(
        output_dir
        / (
            "40b_relation_constrained_"
            "summary_v21.csv"
        ),
        index=False,
        encoding="utf-8-sig",
    )

    return PipelineResultV21(
        product_count=summary["product_count"],
        input_visual_strict_count=summary[
            "input_visual_strict_count"
        ],
        cleaned_visual_strict_count=summary[
            "cleaned_visual_strict_count"
        ],
        relation_evidence_sentence_count=summary[
            "combined_relation_evidence_sentence_count"
        ],
        input_uncertain_count=summary[
            "input_uncertain_count"
        ],
        recovered_sentence_count=summary[
            "recovered_visual_sentence_count"
        ],
        products_with_any_outer_imagery=(
            outer_products
        ),
        products_with_any_all_level_evidence=(
            all_products
        ),
        environmental_packaging_sentence_count=summary[
            "environmental_packaging_sentence_count"
        ],
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "情感意象关系约束v2.1：修复来源计数、"
            "排除组合装/程度副词误报，并分离外包装与"
            "内袋包装；主图标签仅使用外部零售包装。"
        )
    )
    parser.add_argument(
        "--classified",
        required=True,
        type=Path,
        help=(
            "v1.1的15_packaging_sentences_"
            "rule_classified.parquet"
        ),
    )
    parser.add_argument(
        "--product-stats",
        required=True,
        type=Path,
        help=(
            "v1.1的17_product_visual_"
            "packaging_stats.csv"
        ),
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--output-format",
        choices=["parquet", "csv.gz"],
        default="parquet",
    )
    parser.add_argument(
        "--pilot-min-products",
        type=int,
        default=15,
    )
    parser.add_argument(
        "--core-min-products",
        type=int,
        default=40,
    )
    parser.add_argument(
        "--recovery-confirmation-min-products",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--near-duplicate-threshold",
        type=float,
        default=0.96,
    )
    parser.add_argument(
        "--audit-sample-size",
        type=int,
        default=600,
    )
    parser.add_argument(
        "--audit-seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        result = run_pipeline_v21(
            classified_path=args.classified,
            product_stats_path=args.product_stats,
            output_dir=args.output_dir,
            output_format=args.output_format,
            pilot_min_products=args.pilot_min_products,
            core_min_products=args.core_min_products,
            recovery_confirmation_min_products=(
                args.recovery_confirmation_min_products
            ),
            near_duplicate_threshold=(
                args.near_duplicate_threshold
            ),
            audit_sample_size=(
                args.audit_sample_size
            ),
            audit_seed=args.audit_seed,
            overwrite=args.overwrite,
        )
    except Exception as exc:
        print(
            f"运行失败: {exc}",
            file=sys.stderr,
        )
        return 2

    print("=" * 72)
    print("情感意象关系约束v2.1完成")
    print(
        f"商品数: {result.product_count:,}"
    )
    print(
        "输入visual_strict: "
        f"{result.input_visual_strict_count:,}"
    )
    print(
        "清理后visual_strict: "
        f"{result.cleaned_visual_strict_count:,}"
    )
    print(
        "关系证据句: "
        f"{result.relation_evidence_sentence_count:,}"
    )
    print(
        "输入uncertain: "
        f"{result.input_uncertain_count:,}"
    )
    print(
        "定向恢复: "
        f"{result.recovered_sentence_count:,}"
    )
    print(
        "有外包装意象证据的商品: "
        f"{result.products_with_any_outer_imagery:,}"
    )
    print(
        "有任意包装层级证据的商品: "
        f"{result.products_with_any_all_level_evidence:,}"
    )
    print(
        "标签语义: 1=观察到意象提及；"
        "0=未观察到/未标注"
    )
    print(
        f"输出目录: "
        f"{Path(args.output_dir).resolve()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
