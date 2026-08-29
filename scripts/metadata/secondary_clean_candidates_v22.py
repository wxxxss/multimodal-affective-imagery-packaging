#!/usr/bin/env python3
"""V2.2 refinement of V2.1 herbal/fruit-tea candidate decisions.

This script does not rescan the 1.29 GB Amazon metadata JSONL. It reads the
V2.1 all-decisions CSV, normally:

    data/processed/full_v21/candidate_products_v21_all_decisions.csv

V2.2 keeps the three-way decision structure:

- core_candidate: high-confidence dry herbal/fruit infusion product;
- review_required: plausible product requiring metadata/image/LLM review;
- excluded: clear out-of-scope product-specific evidence.

The revision is based on the independently reviewed V2.1 sample. It repairs
food-flavour and bergamot-peel false exclusions, and tightens core-candidate
rules for instant mixes, medical/supplement products, brewed cacao/coffee-fruit
products, and suspected true-tea products.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import random
import re
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

RULES_VERSION = "metadata_secondary_cleaning_v2.2"
DEFAULT_INPUT = (
    Path("data")
    / "processed"
    / "full_v21"
    / "candidate_products_v21_all_decisions.csv"
)
DEFAULT_OUTPUT_DIR = Path("data") / "processed" / "full_v22"

APPENDED_FIELDS = (
    "v22_group",
    "v22_reason",
    "v22_evidence_terms",
    "v22_risk_flags",
    "v22_previous_group",
    "v22_previous_reason",
    "v22_rules_version",
)

REVIEW_SAMPLE_FIELDS = (
    "sample_source",
    "human_product_valid",
    "human_tea_type",
    "human_exclusion_reason",
    "human_notes",
)

TARGET_TERMS = (
    "herbal tea",
    "herb tea",
    "herbal infusion",
    "herbal tisane",
    "tisane",
    "fruit infusion",
    "fruit herbal tea",
    "flower tea",
    "floral tea",
    "botanical infusion",
    "herbal",
    "rooibos",
    "honeybush",
    "caffeine free",
    "caffeine-free",
)

PRIMARY_TARGET_PHRASES = (
    "herbal tea",
    "herb tea",
    "herbal infusion",
    "herbal tisane",
    "fruit infusion",
    "flower tea",
    "floral tea",
    "rooibos tea",
    "honeybush tea",
)

BOTANICAL_TERMS = (
    "chamomile",
    "camomile",
    "hibiscus",
    "peppermint",
    "spearmint",
    "mint",
    "lavender",
    "rosehip",
    "rose hips",
    "rose petal",
    "rose petals",
    "chrysanthemum",
    "lemongrass",
    "lemon grass",
    "lemon balm",
    "elderberry",
    "elderflower",
    "ginger",
    "turmeric",
    "dandelion",
    "nettle",
    "fennel",
    "echinacea",
    "valerian",
    "passionflower",
    "passion flower",
    "moringa",
    "butterfly pea",
    "hawthorn",
    "licorice",
    "liquorice",
    "cinnamon",
    "tulsi",
    "holy basil",
    "lemon verbena",
    "thyme",
    "anise",
    "aniseed",
    "pine needle",
    "bearberry",
    "uva ursi",
    "cassia seed",
    "red clover",
    "gynostemma",
    "vervain",
    "marigold",
    "calendula",
    "linden flower",
    "osmanthus",
    "noni",
    "sappan",
    "tamarind",
    "blue vervain",
    "ginseng",
    "marshmallow leaf",
    "mugwort",
    "motherwort",
    "mistletoe",
    "banaba",
    "bergamot peel",
    "citrus bergamia",
)

TEA_CONTEXT_TERMS = ("tea", "tisane", "infusion", "brew", "steep")

DRY_FORM_TERMS = (
    "tea bag",
    "tea bags",
    "teabag",
    "teabags",
    "sachet",
    "sachets",
    "pouch",
    "pouches",
    "loose leaf",
    "loose-leaf",
    "loose tea",
    "whole leaf",
    "pyramid bag",
    "pyramid bags",
    "individually wrapped",
)

TRUE_TEA_TERMS = (
    "black tea",
    "green tea",
    "white tea",
    "oolong",
    "pu erh",
    "puerh",
    "pu-erh",
    "matcha",
    "sencha",
    "darjeeling",
    "assam",
    "ceylon",
    "earl grey",
    "english breakfast",
    "irish breakfast",
    "scottish breakfast",
    "orange pekoe",
    "gunpowder tea",
    "lord bergamot",
    "jasmine green tea",
    "jasmine black tea",
    "longjing",
    "dragon well",
    "yorkshire tea",
    "pg tips",
    "typhoo",
    "tetley iced tea blend",
    "camellia sinensis",
)

POD_MACHINE_TERMS = (
    "k cup",
    "k cups",
    "k-cup",
    "k-cups",
    "kcup",
    "kcups",
    "keurig",
    "nespresso",
    "tea capsule",
    "tea capsules",
    "coffee capsule",
    "coffee capsules",
    "tea pod",
    "tea pods",
    "coffee pod",
    "t disc",
    "t-disc",
    "tassimo",
    "flavia",
    "verismo",
)

INSTANT_CONCENTRATE_TERMS = (
    "instant tea",
    "instant herbal tea",
    "instant beverage",
    "instant drink",
    "tea crystals",
    "drink crystals",
    "tea powder",
    "powder mix",
    "drink mix",
    "latte mix",
    "chai mix",
    "chai latte mix",
    "syrup",
    "liquid enhancer",
    "drink enhancer",
    "tea concentrate",
    "herbal tea concentrate",
    "iced tea concentrate",
    "liquid concentrate",
    "tea extract",
    "ginseng tea extract",
)

INSTANT_DETAIL_TERMS = (
    "item form granules",
    "item form powder",
    "fast dissolving powder",
    "powder drink mix",
    "instant beverage",
    "drink mix packets",
)

READY_TO_DRINK_TERMS = (
    "ready to drink",
    "ready-to-drink",
    "bottled tea",
    "sparkling tea",
    "tea soda",
    "kombucha",
    "energy drink",
    "fl oz",
    "fluid ounce",
    "cans",
    "bottles",
)

SAMPLER_SET_TERMS = (
    "tea sampler",
    "sampler",
    "variety pack",
    "assortment",
    "assorted",
    "assorted tea",
    "assorted teas",
    "gift set",
    "bundle",
    "collection of teas",
)

STRONG_SUPPLEMENT_TITLE_TERMS = (
    "weight loss",
    "slimming",
    "dieter",
    "dieters drink",
    "laxative",
    "colon cleanse",
    "colonclean",
    "colon clean",
    "senna",
    "skinny tea",
    "herbalife",
    "cleanse tea",
    "dietary supplement",
    "herbal supplement",
    "supplement",
    "capsule",
    "capsules",
    "tablet",
    "tablets",
    "gummy",
    "gummies",
    "tincture",
    "extract drops",
    "labor inducing",
    "labor prep",
    "ripen cervix",
    "ripen the cervix",
    "emmenagogue",
    "water weight gain",
    "temporary water weight",
    "extra bowel movements",
    "bajar de peso",
    "quema de grasa",
    "chupa grasa",
    "fat burner",
)

DIRECT_MEDICAL_PRIMARY_TERMS = (
    "labor inducing",
    "ripen cervix",
    "ripen the cervix",
    "emmenagogue",
    "relieves temporary water weight gain",
    "temporary water weight gain",
    "bajar de peso",
    "quema de grasa",
    "chupa grasa",
    "dieters drink",
    "senna leaf",
    "extra bowel movements",
)

SUPPLEMENT_CONFLICT_REVIEW_TERMS = (
    "supplement",
    "dietary supplement",
    "diatery supplement",
    "herbal supplement",
    "supplements",
    "capsules",
    "weight loss",
    "diuretic",
    "appetite suppressant",
)

FUNCTIONAL_REVIEW_TERMS = (
    "detox",
    "spring cleaning",
    "cholesterol",
    "colesterol",
    "liver care",
    "healthy eyes",
    "blood pressure",
    "immune support",
    "immunity",
    "sleep aid",
    "wellness",
    "tummy",
    "extra strength",
)

NONSTANDARD_BREWED_BEVERAGE_TERMS = (
    "brewed cacao",
    "cacao drink",
    "cocoa drink",
    "cacao tea",
    "coffee fruit tea",
    "coffee cherry tea",
    "coffee leaf tea",
    "cascara tea",
    "substitute to herbal tea",
    "coffee substitute",
    "coffee alternative",
)

FUNGAL_TERMS = ("mushroom", "reishi", "chaga", "ganoderma", "poria", "lichen")

CAFFEINATED_NON_CAMELLIA_TERMS = (
    "yerba mate",
    "mate tea",
    "yaupon",
    "guayusa",
    "coffee leaf",
    "coffee leaves",
    "matevana",
)

GRAIN_INFUSION_TERMS = (
    "barley tea",
    "roasted barley",
    "buckwheat tea",
    "sobacha",
    "corn tea",
    "rice tea",
    "grain tea",
    "job s tears tea",
    "jobs tears tea",
    "adlay tea",
)

TEA_ACCESSORY_TERMS = (
    "empty tea bag",
    "empty tea bags",
    "bubble tea straws",
    "tea straws",
    "tea set",
    "teapot",
    "tea infuser",
    "tea kettle",
    "tea cup",
    "tea strainer",
    "tea filter",
    "tea ball",
)

OBVIOUS_NON_TARGET_FOOD_TERMS = (
    "bubble tea pearls",
    "popping boba",
    "boba pearls",
    "biscuits",
    "cookie",
    "cookies",
    "candy",
    "gumball",
    "chewing gum",
    "noodles",
    "peanut butter",
    "jam",
    "marmalade",
)

FLAVOUR_LIKE_FOOD_TERMS = ("cookie", "cookies", "candy", "marmalade")

RAW_BOTANICAL_OR_FOOD_TERMS = (
    "dates",
    "jujube",
    "peel",
    "root",
    "powder",
    "whole flowers",
    "whole flower",
    "dried flowers",
    "dried flower",
    "petals",
    "buds",
    "dried leaves",
    "whole leaves",
    "seed",
    "seeds",
)

MULTIUSE_TERMS = (
    "for baking",
    "baking",
    "marinades",
    "rubs",
    "cocktails",
    "body care",
    "soap",
    "sachet making",
    "food decoration",
    "culinary",
    "cooking",
    "spice",
    "seasoning",
    "food coloring",
)

SUSPECTED_TRUE_TEA_REVIEW_TERMS = (
    "golden green",
    "tibetan tea",
    "chinese tea",
)

REASON_DESCRIPTIONS = {
    "missing_parent_asin": "缺少parent_asin",
    "missing_title": "缺少商品标题",
    "missing_image": "缺少商品图片",
    "tea_accessory": "茶具、空茶包或饮用附件",
    "kcup_pod_or_machine_pack": "K-Cup、Nespresso等机器胶囊或饮品包",
    "non_botanical_or_fungal": "蘑菇、地衣等非植物原料",
    "caffeinated_non_camellia": "马黛茶、Yaupon、咖啡叶等独立含咖啡因叶饮品",
    "grain_infusion": "大麦、荞麦等谷物浸泡饮品",
    "slimming_laxative_or_supplement": "减肥、泻剂、清肠、补充剂或高度医疗导向",
    "mixed_sampler_or_gift_set": "多口味样品、组合或礼盒",
    "true_tea_base": "红茶、绿茶、白茶、乌龙茶等真茶基底",
    "instant_powder_syrup_or_mix": "速溶粉、浓缩液、糖浆、果酱式茶或饮料混合物",
    "ready_to_drink_or_bottled": "瓶装、罐装或即饮产品",
    "nonstandard_brewed_beverage": "可可冲泡饮品、咖啡果茶或咖啡替代饮品",
    "non_target_food": "糖果、饼干、珍珠等普通食品",
    "functional_claim": "功能性、健康或疾病导向信息需要人工核对",
    "multiuse_botanical": "植物材料同时面向烹饪、护肤、手工等多种用途",
    "raw_botanical_or_food": "原始干花、根、果实、粉末等是否以茶饮为主要用途不清",
    "suspected_true_tea": "存在疑似真茶线索，但不足以自动排除",
    "generic_tea_uncertain": "普通tea商品缺少明确草本或花果证据",
    "explicit_rooibos_honeybush": "标题明确为Rooibos或Honeybush",
    "dry_form_botanical_tea": "标题同时具有干燥茶形态和草本/花果证据",
    "explicit_target_with_botanical": "标题同时具有目标类别和具体植物证据",
    "explicit_target_tea": "标题明确为草本或花果茶",
    "target_without_botanical_or_form": "仅有目标类别词，缺少具体植物或干燥茶形态",
    "named_botanical_tea": "标题明确为具体植物茶",
    "botanical_tea_without_form": "具有植物与tea证据，但产品形态仍需核对",
    "previous_exclusion_unresolved": "V2.1曾排除，但V2.2未找到足够证据确认恢复或继续排除",
    "insufficient_v22_evidence": "V2.2证据不足，保留到人工复核队列",
    "recovered_flavor_named_herbal_tea": "食品风味词只是茶的风味名，商品本身明确为草本茶包",
    "recovered_bergamot_peel_infusion": "纯干燥佛手柑果皮冲泡产品，不是真茶基底",
}


def normalize_text(value: Any) -> str:
    """Normalize JSON-like CSV values into lower-case English token text."""
    if value is None:
        return ""
    if isinstance(value, Mapping):
        value = " ".join(f"{key} {normalize_text(item)}" for key, item in value.items())
    elif isinstance(value, (list, tuple, set)):
        value = " ".join(normalize_text(item) for item in value)
    else:
        value = str(value)
    value = html.unescape(value).lower()
    value = re.sub(r"<[^>]+>", " ", value)
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def find_terms(text: str, terms: Sequence[str]) -> list[str]:
    """Return complete normalized token-sequence matches."""
    padded = f" {normalize_text(text)} "
    found: list[str] = []
    for term in terms:
        normalized = normalize_text(term)
        if normalized and f" {normalized} " in padded:
            found.append(term)
    return found


def _dedupe(items: Sequence[str]) -> list[str]:
    return list(dict.fromkeys(item for item in items if item))


def _decision(
    group: str,
    reason: str,
    evidence: Sequence[str],
    risk_flags: Sequence[str],
    previous_group: str,
    previous_reason: str,
) -> dict[str, Any]:
    return {
        "v22_group": group,
        "v22_reason": reason,
        "v22_evidence_terms": "|".join(_dedupe(evidence)),
        "v22_risk_flags": "|".join(_dedupe(risk_flags)),
        "v22_previous_group": previous_group,
        "v22_previous_reason": previous_reason,
        "v22_rules_version": RULES_VERSION,
    }


def _is_honey_citron_mix(title: str, details: str) -> bool:
    if "honey citron" not in title and "citron ginger" not in title:
        return False
    packaging = find_terms(details, ("tub", "jar", "can", "item form liquid"))
    size_pattern = bool(re.search(r"\b\d+(?:\.\d+)?\s*(?:lb|lbs|pound|pounds|liter|liters)\b", title))
    return bool(packaging or size_pattern)


def _is_bergamot_peel_infusion(title: str, primary_text: str) -> bool:
    if "bergamot" not in title:
        return False
    peel_evidence = find_terms(title, ("bergamot peel", "dried bergamot peel", "citrus bergamia"))
    tea_evidence = find_terms(title, ("herbal tea", "herb tea", "loose tea", "tea"))
    dried_evidence = find_terms(primary_text, ("dried", "peel", "loose herbal tea", "for brewing"))
    strong_true_tea = find_terms(title, TRUE_TEA_TERMS)
    return bool(peel_evidence and tea_evidence and dried_evidence and not strong_true_tea)


def _is_flavour_named_herbal_tea(
    title: str,
    primary_text: str,
    food_matches: Sequence[str],
) -> bool:
    if not set(food_matches).intersection(FLAVOUR_LIKE_FOOD_TERMS):
        return False
    title_dry = find_terms(title, DRY_FORM_TERMS)
    primary_target = find_terms(primary_text, PRIMARY_TARGET_PHRASES)
    return bool(title_dry and primary_target)


def classify_candidate(row: Mapping[str, Any]) -> dict[str, Any]:
    """Refine one V2.1 decision into a V2.2 three-way decision."""
    parent_asin = str(row.get("parent_asin") or "").strip()
    raw_title = str(row.get("title") or "").strip()
    title = normalize_text(raw_title)
    description = normalize_text(row.get("description"))
    features = normalize_text(row.get("features"))
    details = normalize_text(row.get("details"))
    categories = normalize_text(row.get("categories"))
    primary_text = normalize_text([title, description, features, details])
    all_text = normalize_text([primary_text, categories, row.get("store")])
    previous_group = str(row.get("v21_group") or row.get("screening_group") or "").strip()
    previous_reason = str(row.get("v21_reason") or row.get("screening_reason") or "").strip()
    image_url = str(row.get("main_image_url") or "").strip()

    def decide(
        group: str,
        reason: str,
        evidence: Sequence[str],
        risk: Sequence[str] = (),
    ) -> dict[str, Any]:
        return _decision(group, reason, evidence, risk, previous_group, previous_reason)

    if not parent_asin:
        return decide("excluded", "missing_parent_asin", [])
    if not raw_title:
        return decide("excluded", "missing_title", [])
    if not image_url:
        return decide("excluded", "missing_image", [])

    # Clear product-form and category exclusions remain first.
    ordered_title_exclusions = (
        ("tea_accessory", TEA_ACCESSORY_TERMS),
        ("kcup_pod_or_machine_pack", POD_MACHINE_TERMS),
        ("non_botanical_or_fungal", FUNGAL_TERMS),
        ("caffeinated_non_camellia", CAFFEINATED_NON_CAMELLIA_TERMS),
        ("grain_infusion", GRAIN_INFUSION_TERMS),
        ("mixed_sampler_or_gift_set", SAMPLER_SET_TERMS),
        ("true_tea_base", TRUE_TEA_TERMS),
    )
    for reason, terms in ordered_title_exclusions:
        matches = find_terms(title, terms)
        if matches:
            return decide("excluded", reason, matches)

    if (
        ("&" in raw_title or "+" in raw_title)
        and re.search(r"\bset\b", raw_title, flags=re.IGNORECASE)
        and title.count("tea") >= 2
    ):
        return decide("excluded", "mixed_sampler_or_gift_set", ["multi_product_set_pattern"])

    ready_matches = find_terms(title, READY_TO_DRINK_TERMS)
    dry_matches = find_terms(title, DRY_FORM_TERMS)
    if ready_matches and not dry_matches:
        return decide("excluded", "ready_to_drink_or_bottled", ready_matches)

    instant_matches = _dedupe(
        [
            *find_terms(title, INSTANT_CONCENTRATE_TERMS),
            *find_terms(primary_text, INSTANT_DETAIL_TERMS),
        ]
    )
    if _is_honey_citron_mix(title, details):
        instant_matches = _dedupe([*instant_matches, "honey_citron_tub_mix"])
    if instant_matches:
        return decide("excluded", "instant_powder_syrup_or_mix", instant_matches)

    medical_title_matches = find_terms(title, STRONG_SUPPLEMENT_TITLE_TERMS)
    description_head = normalize_text(str(row.get("description") or "")[:700])
    medical_direct_matches = find_terms(
        normalize_text([title, description_head]), DIRECT_MEDICAL_PRIMARY_TERMS
    )
    description_starts_as_supplement = description_head.startswith("dietary supplement")
    if medical_title_matches or medical_direct_matches or description_starts_as_supplement:
        evidence = _dedupe(
            [
                *medical_title_matches,
                *medical_direct_matches,
                *(["description_starts_dietary_supplement"] if description_starts_as_supplement else []),
            ]
        )
        return decide("excluded", "slimming_laxative_or_supplement", evidence)

    supplement_conflict_matches = find_terms(
        primary_text, SUPPLEMENT_CONFLICT_REVIEW_TERMS
    )

    nonstandard_matches = find_terms(primary_text, NONSTANDARD_BREWED_BEVERAGE_TERMS)
    if nonstandard_matches:
        return decide("excluded", "nonstandard_brewed_beverage", nonstandard_matches)

    # Explicit recovery 1: pure dried bergamot peel is not automatically true tea.
    if _is_bergamot_peel_infusion(title, primary_text):
        return decide(
            "core_candidate",
            "recovered_bergamot_peel_infusion",
            ["bergamot peel", "herbal tea", "dried/loose brewing form"],
        )

    food_matches = find_terms(title, OBVIOUS_NON_TARGET_FOOD_TERMS)
    # Explicit recovery 2: a food word can be a flavour name when the product
    # itself is clearly described as herbal tea bags.
    if food_matches and _is_flavour_named_herbal_tea(title, primary_text, food_matches):
        return decide(
            "core_candidate",
            "recovered_flavor_named_herbal_tea",
            [*food_matches, *find_terms(primary_text, PRIMARY_TARGET_PHRASES), *dry_matches],
            ["food_word_used_as_flavour_name"],
        )
    if food_matches:
        return decide("excluded", "non_target_food", food_matches)

    tea_matches = find_terms(title, TEA_CONTEXT_TERMS)
    target_matches = find_terms(title, TARGET_TERMS)
    botanical_matches = find_terms(title, BOTANICAL_TERMS)
    raw_matches = find_terms(title, RAW_BOTANICAL_OR_FOOD_TERMS)
    multiuse_matches = find_terms(all_text, MULTIUSE_TERMS)
    functional_matches = _dedupe(
        [
            *find_terms(title, FUNCTIONAL_REVIEW_TERMS),
            *supplement_conflict_matches,
        ]
    )
    suspected_true_tea_matches = find_terms(title, SUSPECTED_TRUE_TEA_REVIEW_TERMS)

    if functional_matches:
        return decide("review_required", "functional_claim", functional_matches, functional_matches)
    if multiuse_matches:
        return decide("review_required", "multiuse_botanical", multiuse_matches, multiuse_matches)
    if raw_matches and not dry_matches:
        return decide("review_required", "raw_botanical_or_food", raw_matches, raw_matches)

    if (
        " green " in f" {title} "
        and not find_terms(title, ("green rooibos", "honeybush"))
        and tea_matches
    ):
        suspected_true_tea_matches = _dedupe(
            [*suspected_true_tea_matches, "green_without_rooibos_evidence"]
        )
    if suspected_true_tea_matches:
        return decide(
            "review_required",
            "suspected_true_tea",
            suspected_true_tea_matches,
            suspected_true_tea_matches,
        )

    if tea_matches and not (target_matches or botanical_matches):
        return decide(
            "review_required",
            "generic_tea_uncertain",
            tea_matches,
            ["no_target_or_botanical_title_evidence"],
        )

    rooibos_matches = find_terms(title, ("rooibos", "honeybush"))
    if rooibos_matches:
        return decide("core_candidate", "explicit_rooibos_honeybush", rooibos_matches)

    if dry_matches and tea_matches and (target_matches or botanical_matches):
        return decide(
            "core_candidate",
            "dry_form_botanical_tea",
            [*dry_matches, *target_matches, *botanical_matches],
        )

    if tea_matches and target_matches and botanical_matches:
        return decide(
            "core_candidate",
            "explicit_target_with_botanical",
            [*target_matches, *botanical_matches],
        )

    if tea_matches and target_matches:
        if previous_group == "core_candidate":
            return decide("core_candidate", "explicit_target_tea", target_matches)
        return decide(
            "review_required",
            "target_without_botanical_or_form",
            target_matches,
            ["target_without_dry_form_or_botanical"],
        )

    if tea_matches and botanical_matches:
        if len(title.split()) <= 12:
            return decide("core_candidate", "named_botanical_tea", botanical_matches)
        return decide(
            "review_required",
            "botanical_tea_without_form",
            botanical_matches,
            ["no_dry_form_title_evidence"],
        )

    if previous_group == "excluded":
        return decide(
            "review_required",
            "previous_exclusion_unresolved",
            [],
            ["v21_exclusion_not_reconfirmed"],
        )

    return decide(
        "review_required",
        "insufficient_v22_evidence",
        [],
        ["unresolved_v22_candidate"],
    )


@dataclass
class ReservoirSampler:
    capacity: int
    rng: random.Random
    items: list[dict[str, Any]] = field(default_factory=list)
    seen: int = 0

    def add(self, item: Mapping[str, Any]) -> None:
        self.seen += 1
        if self.capacity <= 0:
            return
        copied = dict(item)
        if len(self.items) < self.capacity:
            self.items.append(copied)
            return
        position = self.rng.randrange(self.seen)
        if position < self.capacity:
            self.items[position] = copied


def _open_writer(path: Path, fieldnames: Sequence[str]) -> tuple[Any, csv.DictWriter]:
    handle = path.open("w", encoding="utf-8-sig", newline="")
    writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    return handle, writer


def _write_ids(path: Path, values: set[str]) -> None:
    path.write_text("".join(f"{value}\n" for value in sorted(values)), encoding="utf-8")


def run_secondary_cleaning(
    input_path: Path,
    output_dir: Path,
    *,
    sample_core: int = 100,
    sample_review: int = 150,
    sample_excluded: int = 50,
    seed: int = 42,
    progress_every: int = 5_000,
    max_records: int | None = None,
) -> dict[str, Any]:
    """Run V2.2 on a V2.1 all-decisions CSV and write all output files."""
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV does not exist: {input_path}")
    if max_records is not None and max_records <= 0:
        raise ValueError("max_records must be positive when provided")

    output_dir.mkdir(parents=True, exist_ok=True)
    started_at = datetime.now(timezone.utc).isoformat()
    started_time = time.time()
    rng = random.Random(seed)

    counters: Counter[str] = Counter()
    reason_counts: Counter[str] = Counter()
    transition_counts: Counter[str] = Counter()
    risk_counts: Counter[str] = Counter()
    seen_ids: set[str] = set()
    core_ids: set[str] = set()
    review_ids: set[str] = set()
    excluded_ids: set[str] = set()

    samplers = {
        "core_candidate": ReservoirSampler(sample_core, rng),
        "review_required": ReservoirSampler(sample_review, rng),
        "excluded": ReservoirSampler(sample_excluded, rng),
    }

    with input_path.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        if not reader.fieldnames:
            raise ValueError("Input CSV has no header row")
        required = {"parent_asin", "title", "v21_group", "v21_reason"}
        missing = sorted(required - set(reader.fieldnames))
        if missing:
            raise ValueError(f"Input CSV is missing required columns: {', '.join(missing)}")

        output_fields = list(reader.fieldnames)
        for field_name in APPENDED_FIELDS:
            if field_name not in output_fields:
                output_fields.append(field_name)

        all_handle, all_writer = _open_writer(
            output_dir / "candidate_products_v22_all_decisions.csv", output_fields
        )
        core_handle, core_writer = _open_writer(
            output_dir / "candidate_products_v22_core.csv", output_fields
        )
        review_handle, review_writer = _open_writer(
            output_dir / "candidate_products_v22_review_required.csv", output_fields
        )
        excluded_handle, excluded_writer = _open_writer(
            output_dir / "candidate_products_v22_excluded.csv", output_fields
        )

        try:
            for row in reader:
                counters["records_scanned"] += 1
                if max_records is not None and counters["records_scanned"] > max_records:
                    counters["records_scanned"] -= 1
                    break

                if progress_every and counters["records_scanned"] % progress_every == 0:
                    elapsed = max(time.time() - started_time, 0.001)
                    print(
                        f"[进度] 已处理 {counters['records_scanned']:,} 件；"
                        f"核心 {counters['core_candidate_count']:,}；"
                        f"复核 {counters['review_required_count']:,}；"
                        f"排除 {counters['excluded_count']:,}；"
                        f"速度 {counters['records_scanned']/elapsed:,.0f} 件/秒",
                        flush=True,
                    )

                parent_asin = str(row.get("parent_asin") or "").strip()
                if parent_asin:
                    if parent_asin in seen_ids:
                        counters["duplicate_parent_asins"] += 1
                        continue
                    seen_ids.add(parent_asin)

                decision = classify_candidate(row)
                output_row = dict(row)
                output_row.update(decision)
                group = decision["v22_group"]
                previous_group = decision["v22_previous_group"] or "unknown"

                counters[f"{group}_count"] += 1
                reason_counts[decision["v22_reason"]] += 1
                transition_counts[f"{previous_group}->{group}"] += 1
                if previous_group == "excluded" and group != "excluded":
                    counters["recovered_from_v21_excluded_count"] += 1
                if previous_group == "core_candidate" and group != "core_candidate":
                    counters["demoted_from_v21_core_count"] += 1
                for flag in filter(None, decision["v22_risk_flags"].split("|")):
                    risk_counts[flag] += 1

                all_writer.writerow(output_row)
                samplers[group].add(output_row)
                if group == "core_candidate":
                    core_writer.writerow(output_row)
                    if parent_asin:
                        core_ids.add(parent_asin)
                elif group == "review_required":
                    review_writer.writerow(output_row)
                    if parent_asin:
                        review_ids.add(parent_asin)
                else:
                    excluded_writer.writerow(output_row)
                    if parent_asin:
                        excluded_ids.add(parent_asin)
        finally:
            all_handle.close()
            core_handle.close()
            review_handle.close()
            excluded_handle.close()

    _write_ids(output_dir / "candidate_parent_asins_v22_core.txt", core_ids)
    _write_ids(output_dir / "candidate_parent_asins_v22_review_required.txt", review_ids)
    _write_ids(output_dir / "candidate_parent_asins_v22_retained.txt", core_ids | review_ids)

    sample_fields = output_fields + [
        field_name for field_name in REVIEW_SAMPLE_FIELDS if field_name not in output_fields
    ]
    with (output_dir / "candidate_products_v22_review_sample.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=sample_fields, extrasaction="ignore")
        writer.writeheader()
        for source_name in ("core_candidate", "review_required", "excluded"):
            for item in samplers[source_name].items:
                sample_row = dict(item)
                sample_row.update(
                    {
                        "sample_source": source_name,
                        "human_product_valid": "",
                        "human_tea_type": "",
                        "human_exclusion_reason": "",
                        "human_notes": "",
                    }
                )
                writer.writerow(sample_row)

    elapsed = time.time() - started_time
    report = {
        "rules_version": RULES_VERSION,
        "input_file": str(input_path),
        "input_size_bytes": input_path.stat().st_size,
        "output_dir": str(output_dir),
        "started_at_utc": started_at,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": round(elapsed, 3),
        "max_records": max_records,
        "random_seed": seed,
        "records_scanned": counters["records_scanned"],
        "unique_parent_asins_seen": len(seen_ids),
        "duplicate_parent_asins": counters["duplicate_parent_asins"],
        "core_candidate_count": counters["core_candidate_count"],
        "review_required_count": counters["review_required_count"],
        "excluded_count": counters["excluded_count"],
        "retained_count": counters["core_candidate_count"] + counters["review_required_count"],
        "recovered_from_v21_excluded_count": counters[
            "recovered_from_v21_excluded_count"
        ],
        "demoted_from_v21_core_count": counters["demoted_from_v21_core_count"],
        "core_rate": (
            counters["core_candidate_count"] / counters["records_scanned"]
            if counters["records_scanned"]
            else 0.0
        ),
        "review_required_rate": (
            counters["review_required_count"] / counters["records_scanned"]
            if counters["records_scanned"]
            else 0.0
        ),
        "excluded_rate": (
            counters["excluded_count"] / counters["records_scanned"]
            if counters["records_scanned"]
            else 0.0
        ),
        "decision_reason_counts": dict(reason_counts.most_common()),
        "transition_counts": dict(transition_counts.most_common()),
        "risk_flag_counts": dict(risk_counts.most_common()),
        "reason_descriptions_cn": REASON_DESCRIPTIONS,
        "review_sample_counts": {
            "core_candidate": len(samplers["core_candidate"].items),
            "review_required": len(samplers["review_required"].items),
            "excluded": len(samplers["excluded"].items),
            "total": sum(len(sampler.items) for sampler in samplers.values()),
        },
        "output_files": {
            "all_decisions": "candidate_products_v22_all_decisions.csv",
            "core_candidates": "candidate_products_v22_core.csv",
            "review_required": "candidate_products_v22_review_required.csv",
            "excluded": "candidate_products_v22_excluded.csv",
            "core_ids": "candidate_parent_asins_v22_core.txt",
            "review_ids": "candidate_parent_asins_v22_review_required.txt",
            "retained_ids": "candidate_parent_asins_v22_retained.txt",
            "review_sample": "candidate_products_v22_review_sample.csv",
            "report": "metadata_secondary_cleaning_v22_report.json",
        },
        "interpretation": {
            "core_candidate": "V2.2高置信核心候选；仍需用新抽样样本验证后才能作为最终商品池。",
            "review_required": "不得自动删除；后续结合配料、图片、LLM或人工确认。",
            "excluded": "具有明确产品级排除证据；仍需抽样检查误删除。",
        },
    }

    with (output_dir / "metadata_secondary_cleaning_v22_report.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)

    print("\n=== V2.2候选商品二次清洗完成 ===")
    print(f"处理商品：{report['records_scanned']:,}")
    print(f"核心候选：{report['core_candidate_count']:,}")
    print(f"需要复核：{report['review_required_count']:,}")
    print(f"明确排除：{report['excluded_count']:,}")
    print(f"从V2.1排除组恢复：{report['recovered_from_v21_excluded_count']:,}")
    print(f"从V2.1核心组降级：{report['demoted_from_v21_core_count']:,}")
    print(f"输出目录：{output_dir}")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="V2.2：修订V2.1候选决策，不重新扫描原始JSONL"
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="V2.1全决策CSV路径")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="输出目录")
    parser.add_argument("--sample-core", type=int, default=100, help="核心候选抽样数量")
    parser.add_argument("--sample-review", type=int, default=150, help="待复核组抽样数量")
    parser.add_argument("--sample-excluded", type=int, default=50, help="明确排除组抽样数量")
    parser.add_argument("--seed", type=int, default=42, help="随机抽样种子")
    parser.add_argument(
        "--progress-every",
        type=int,
        default=5_000,
        help="每处理多少件显示进度；0表示关闭",
    )
    parser.add_argument("--max-records", type=int, default=None, help="试运行时限制处理记录数")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        run_secondary_cleaning(
            args.input,
            args.output_dir,
            sample_core=args.sample_core,
            sample_review=args.sample_review,
            sample_excluded=args.sample_excluded,
            seed=args.seed,
            progress_every=args.progress_every,
            max_records=args.max_records,
        )
    except (FileNotFoundError, PermissionError, OSError, ValueError) as exc:
        print(f"错误：{exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
