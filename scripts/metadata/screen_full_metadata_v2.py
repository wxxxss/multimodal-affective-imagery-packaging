#!/usr/bin/env python3
"""V2 full metadata screening for herbal and fruit tea packaging research.

The script streams Amazon Reviews'23 Grocery metadata and identifies dry,
retail herbal/fruit infusion products. It does not read review text and it
does not create final research inclusion labels.

V2 changes are calibrated from the manually reviewed 300-product V1 sample:
- category text is auxiliary evidence only;
- true Camellia sinensis tea bases have exclusion priority;
- ready-to-drink, instant, pod, sampler, supplement, fungal, grain, mate/yaupon
  and generic cooking ingredients are explicitly excluded;
- rooibos and honeybush remain in scope;
- raw dried botanical materials explicitly marketed for tea remain borderline.
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

RULES_VERSION = "metadata_screening_v2.0"

DEFAULT_INPUT = (
    Path("data")
    / "meta_Grocery_and_Gourmet_Food.jsonl"
    / "meta_Grocery_and_Gourmet_Food.jsonl"
)
DEFAULT_OUTPUT_DIR = Path("data") / "processed" / "full_v2"

# Strong evidence must occur in title/description/features, not only categories.
STRONG_TARGET_TERMS = (
    "herbal tea", "herb tea", "herbal infusion", "herbal tisane", "tisane",
    "fruit infusion", "fruit herbal tea", "floral tea", "flower tea",
    "caffeine free herbal tea", "caffeine-free herbal tea",
    "botanical infusion", "rooibos tea", "red rooibos", "green rooibos",
    "honeybush tea",
)

HERBAL_INGREDIENT_TERMS = (
    "chamomile", "camomile", "hibiscus", "peppermint", "spearmint",
    "lavender", "rosehip", "rose hips", "rose tea", "rose petals",
    "chrysanthemum", "lemongrass", "lemon balm", "elderberry",
    "elderflower", "ginger", "turmeric", "dandelion", "nettle",
    "fennel", "echinacea", "valerian", "passionflower", "moringa",
    "butterfly pea", "hawthorn", "licorice root", "liquorice root",
    "cinnamon", "rooibos", "honeybush", "holy basil", "tulsi",
    "lemon verbena", "sage", "thyme", "anise", "aniseed",
)

TEA_CONTEXT_TERMS = (
    "tea", "infusion", "tisane", "brew", "tea bag", "tea bags",
    "teabag", "teabags", "sachet", "sachets", "loose leaf",
    "loose-leaf", "steep", "caffeine free", "caffeine-free",
)

DRY_RETAIL_FORM_TERMS = (
    "tea bag", "tea bags", "teabag", "teabags", "sachet", "sachets",
    "loose leaf", "loose-leaf", "loose tea", "pyramid bag",
    "pyramid sachet", "individually wrapped", "whole leaf",
)

RAW_BOTANICAL_TERMS = (
    "dried flower", "dried flowers", "whole flower", "whole flowers",
    "flower petals", "petals", "cut and sifted", "cut & sifted",
    "dried leaves", "dried leaf", "dried root", "root pieces",
    "whole herb", "whole herbs", "herbal material",
)

TRUE_TEA_TERMS = (
    "black tea", "green tea", "oolong tea", "oolong", "white tea",
    "pu erh", "puerh", "pu-erh", "matcha", "sencha", "gyokuro",
    "darjeeling", "assam tea", "assam black", "ceylon tea", "ceylon",
    "english breakfast", "scottish breakfast", "irish breakfast",
    "earl grey", "jasmine green tea", "jasmine black tea",
    "gunpowder tea", "gunpowder green", "camellia sinensis",
    "orange pekoe", "breakfast tea", "oolong chai",
)

READY_TO_DRINK_TERMS = (
    "ready to drink", "ready-to-drink", "bottled tea", "iced tea bottle",
    "sparkling tea", "tea soda", "kombucha", "tonic", "12 fl oz",
    "16 fl oz", "fluid ounce", "fl oz cans", "cans pack", "bottles pack",
    "cold pressed juice", "ginger ale", "root beer", "soft drink", "soda",
)

INSTANT_MIX_TERMS = (
    "instant tea", "instant herbal tea", "tea crystals", "honey crystals",
    "drink crystals", "powder mix", "tea powder", "latte powder",
    "chai latte mix", "instant chai", "syrup", "liquid concentrate",
    "liquid enhancer", "drink enhancer", "supergreens", "greens powder",
)

POD_MACHINE_TERMS = (
    "k cup", "k cups", "k-cup", "k-cups", "kcup", "kcups", "keurig", "t disc", "t-disc",
    "tassimo", "flavia fresh pack", "flavia", "verismo", "coffee pod",
    "tea pod", "capsule",
)

SAMPLER_SET_TERMS = (
    "tea sampler", "tea assortment", "tasting assortment", "variety pack",
    "gift set", "gift box", "tea chest", "chakra tea set", "bundle",
    "collection of teas", "mixed tea", "assorted tea", "assorted teas",
)

SLIMMING_SUPPLEMENT_TERMS = (
    "slimming tea", "weight loss tea", "lose weight", "laxative tea",
    "diet tea", "slim tea", "skinny tea", "detox slimming",
    "colon cleanse", "senna tea", "dieter tea", "body slim", "healthy weight",
    "supplement", "capsule", "tablet",
    "gummy", "tincture", "extract drops",
)

FUNGAL_NON_BOTANICAL_TERMS = (
    "mushroom tea", "reishi tea", "chaga tea", "ganoderma", "poria cocos",
    "shelf fungus", "lichen tea", "snow tea",
)

GRAIN_INFUSION_TERMS = (
    "barley tea", "roasted barley", "buckwheat tea", "sobacha",
    "corn tea", "roasted corn", "brown rice tea", "rice tea",
    "job's tears tea", "jobs tears tea", "adlay tea", "grain tea",
    "coix seed", "tartary buckwheat", "gorgon fruit",
)

CAFFEINATED_NON_CAMELLIA_TERMS = (
    "yerba mate", "mate tea", "yaupon tea", "guayusa tea", "guayusa",
)

TEAWARE_TERMS = (
    "tea infuser", "tea kettle", "tea cup", "teacup", "teapot",
    "tea strainer", "tea filter", "tea ball", "tea maker",
)

GENERAL_NON_TARGET_TERMS = (
    "coffee", "espresso", "cocoa", "hot chocolate", "energy drink",
    "juice", "candy", "cookie", "cookies", "chocolate", "jam",
    "seasoning", "spice powder", "ground spice", "baking", "snack",
    "soup", "sauce", "essential oil", "protein powder",
)

COOKING_USE_TERMS = (
    "for cooking", "cooking", "for baking", "baking", "seasoning",
    "spice", "spices", "culinary", "soup", "curries", "curry",
    "sauce", "food coloring", "crafts", "potpourri",
)

AMBIGUOUS_TARGET_TERMS = (
    "fruit tea", "wellness tea", "detox tea", "sleep tea", "bedtime tea",
    "herbal blend", "chai", "flowers for tea", "dried flowers for tea",
)

FRUIT_TERMS = (
    "fruit tea", "fruit infusion", "berry tea", "berries", "apple tea",
    "peach tea", "mango tea", "pineapple tea", "citrus tea",
    "orange tea", "lemon tea", "black currant tea",
)

CSV_FIELDS = (
    "parent_asin", "title", "store", "categories", "description", "features",
    "details", "average_rating", "rating_number", "price", "main_image_url",
    "all_image_urls", "matched_primary_target_terms",
    "matched_category_target_terms", "matched_ingredient_terms",
    "matched_context_terms", "matched_true_tea_terms",
    "matched_form_exclusion_terms", "matched_scope_exclusion_terms",
    "matched_ambiguous_terms", "automatic_tea_type", "product_form",
    "screening_group", "screening_reason", "near_excluded", "rules_version",
)

SAMPLE_EXTRA_FIELDS = (
    "sample_source", "human_product_valid", "human_tea_type",
    "human_exclusion_reason", "human_notes",
)


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, Mapping):
        value = " ".join(f"{k} {normalize_text(v)}" for k, v in value.items())
    elif isinstance(value, (list, tuple, set)):
        value = " ".join(normalize_text(item) for item in value)
    else:
        value = str(value)
    value = html.unescape(value).lower()
    value = re.sub(r"<[^>]+>", " ", value)
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def compact_json(value: Any) -> str:
    if value is None:
        return ""
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def find_terms(text: str, terms: Sequence[str]) -> list[str]:
    found: list[str] = []
    padded = f" {text} "
    for term in terms:
        normalized = normalize_text(term)
        if normalized and f" {normalized} " in padded:
            found.append(term)
    return found


def extract_main_image(images: Any) -> tuple[str, list[str]]:
    if not isinstance(images, list):
        return "", []
    all_urls: list[str] = []
    seen: set[str] = set()
    main_candidates: list[str] = []
    fallback_candidates: list[str] = []
    for item in images:
        if not isinstance(item, Mapping):
            continue
        urls = [item.get("hi_res"), item.get("large"), item.get("thumb")]
        usable = [str(url).strip() for url in urls if isinstance(url, str) and url.strip()]
        for url in usable:
            if url not in seen:
                seen.add(url)
                all_urls.append(url)
        if str(item.get("variant", "")).upper() == "MAIN":
            main_candidates.extend(usable)
        fallback_candidates.extend(usable)
    main_url = main_candidates[0] if main_candidates else (fallback_candidates[0] if fallback_candidates else "")
    return main_url, all_urls


def _primary_text(record: Mapping[str, Any]) -> str:
    return normalize_text([record.get("title"), record.get("description"), record.get("features")])


def _category_text(record: Mapping[str, Any]) -> str:
    return normalize_text(record.get("categories"))


def _all_text(record: Mapping[str, Any]) -> str:
    return normalize_text([
        record.get("title"), record.get("description"), record.get("features"),
        record.get("details"), record.get("categories"), record.get("store"),
    ])


def infer_tea_type(text: str) -> str:
    type_rules = (
        ("chamomile", ("chamomile", "camomile")),
        ("hibiscus", ("hibiscus",)),
        ("mint", ("peppermint", "spearmint", "mint tea", "mint infusion")),
        ("rooibos", ("rooibos",)),
        ("honeybush", ("honeybush",)),
        ("rose", ("rose tea", "rose petals", "rosehip", "rose hips")),
        ("lavender", ("lavender",)),
        ("ginger_turmeric", ("ginger", "turmeric")),
        ("fruit_infusion", FRUIT_TERMS),
        ("mixed_herbal", ("herbal blend", "mixed herbal", "herbal assortment")),
    )
    for tea_type, terms in type_rules:
        if find_terms(text, terms):
            return tea_type
    if find_terms(text, HERBAL_INGREDIENT_TERMS):
        return "other_herbal"
    if find_terms(text, ("herbal tea", "herbal infusion", "tisane", "botanical infusion")):
        return "other_herbal"
    return "uncertain"


def infer_product_form(text: str) -> str:
    if find_terms(text, POD_MACHINE_TERMS):
        return "pod_or_machine_pack"
    if find_terms(text, READY_TO_DRINK_TERMS):
        return "ready_to_drink"
    if find_terms(text, INSTANT_MIX_TERMS):
        return "instant_or_mix"
    if find_terms(text, DRY_RETAIL_FORM_TERMS):
        return "tea_bag_or_loose_leaf"
    if find_terms(text, RAW_BOTANICAL_TERMS):
        return "raw_dried_botanical"
    return "unspecified_dry_product"


def _matches_any(text: str, terms: Sequence[str]) -> bool:
    return bool(find_terms(text, terms))


def _description_features_text(record: Mapping[str, Any]) -> str:
    return normalize_text([record.get("description"), record.get("features")])


def _details_text(record: Mapping[str, Any]) -> str:
    return normalize_text(record.get("details"))


def _product_specific_true_tea_terms(text: str) -> list[str]:
    """Find true-tea evidence that clearly describes the product formula."""
    found: list[str] = []
    patterns = (
        r"\b(?:blend|blended|base|made|crafted|prepared|formulated|ingredients?|contains?)\s+(?:of\s+)?(?:(?:premium|top)\s+)?(?:whole leaf\s+)?(black tea|green tea|oolong tea|white tea|matcha|ceylon tea)\b",
        r"\b(black tea|green tea|oolong tea|white tea|matcha|ceylon tea)\s+(?:leaves|blend|base|with spices|with flowers)\b",
        r"\b100\s*%\s+(black tea|green tea|oolong tea|white tea)\b",
        r"\bpremium\s+(?:whole leaf\s+)?(black tea|green tea|oolong tea|white tea)\b",
    )
    for pattern in patterns:
        for match in re.finditer(pattern, text):
            term = normalize_text(match.group(1))
            if term and term not in found:
                found.append(term)
    return found


def _title_true_tea_terms(title_text: str) -> list[str]:
    found = find_terms(title_text, TRUE_TEA_TERMS)
    flexible_patterns = (
        (r"\bblack(?:\s+[a-z0-9]+){0,2}\s+tea\b", "black tea", {"currant", "berry", "berries", "blackberry", "fruit"}),
        (r"\bgreen(?:\s+[a-z0-9]+){0,2}\s+tea\b", "green tea", {"rooibos", "honeybush"}),
        (r"\bwhite(?:\s+[a-z0-9]+){0,2}\s+tea\b", "white tea", {"pine", "needle", "mulberry"}),
    )
    for pattern, label, exclusions in flexible_patterns:
        for match in re.finditer(pattern, title_text):
            segment = set(match.group(0).split())
            if segment & exclusions:
                continue
            if label not in found:
                found.append(label)
    return found


def classify_product(record: Mapping[str, Any], require_image: bool = True) -> dict[str, Any]:
    parent_asin = str(record.get("parent_asin") or "").strip()
    title = str(record.get("title") or "").strip()
    main_image_url, all_image_urls = extract_main_image(record.get("images"))

    title_text = normalize_text(title)
    desc_features = _description_features_text(record)
    primary = normalize_text([title, record.get("description"), record.get("features")])
    category = _category_text(record)
    details = _details_text(record)
    title_category = normalize_text([title, record.get("categories")])
    all_text = normalize_text([primary, category, details, record.get("store")])

    title_target = find_terms(title_text, STRONG_TARGET_TERMS)
    primary_target = find_terms(primary, STRONG_TARGET_TERMS)
    category_target = find_terms(category, STRONG_TARGET_TERMS)
    title_ingredients = find_terms(title_text, HERBAL_INGREDIENT_TERMS)
    primary_ingredients = find_terms(primary, HERBAL_INGREDIENT_TERMS)
    title_context = find_terms(title_text, TEA_CONTEXT_TERMS)
    primary_context = find_terms(primary, TEA_CONTEXT_TERMS)
    title_true_tea = _title_true_tea_terms(title_text)
    secondary_true_tea = find_terms(desc_features, TRUE_TEA_TERMS)
    formulation_true_tea = _product_specific_true_tea_terms(desc_features[:500])
    category_true_tea = find_terms(category, TRUE_TEA_TERMS)
    category_values = record.get("categories") if isinstance(record.get("categories"), list) else []
    category_leaf = normalize_text(category_values[-1]) if category_values else ""
    if category_leaf in {"black", "green", "white", "oolong"}:
        category_true_tea = list(dict.fromkeys(category_true_tea + [f"category_leaf_{category_leaf}"]))
    ambiguous = find_terms(primary, AMBIGUOUS_TARGET_TERMS)

    form_matches: list[str] = []
    scope_matches: list[str] = []
    group = "excluded"
    reason = ""
    near_excluded = False

    explicit_title_target = bool(title_target)
    explicit_title_herbal = bool(find_terms(title_text, (
        "herbal tea", "herb tea", "herbal infusion", "tisane",
        "rooibos", "honeybush", "fruit infusion", "flower tea",
        "floral tea", "caffeine free tea", "caffeine-free tea",
    )))
    title_has_tea = bool(find_terms(title_text, (
        "tea", "teabag", "teabags", "tea bag", "tea bags", "tisane", "infusion", "infuse",
    )))
    title_has_dry_form = bool(find_terms(title_text, DRY_RETAIL_FORM_TERMS))
    details_has_tea_form = bool(find_terms(details, (
        "tea bag", "tea bags", "teabag", "teabags", "loose leaves",
        "loose leaf", "sachet", "sachets",
    )))
    category_supports_tea = bool(
        category_target or find_terms(category, ("tea", "rooibos", "fruit herbal tea", "herbal"))
    )
    raw_botanical = bool(find_terms(primary, RAW_BOTANICAL_TERMS)) or (
        "dried" in primary.split()
        and any(token in primary.split() for token in (
            "flower", "flowers", "petal", "petals", "leaf", "leaves", "root", "roots"
        ))
    )
    explicit_for_tea = bool(find_terms(primary, (
        "for tea", "for teas", "for brewing", "tea infusion", "herbal infusion",
        "brew as tea", "steep as tea", "make tea",
    )))
    ingredient_with_title_context = bool(title_ingredients and title_context)
    ingredient_with_brewing_context = bool(
        primary_ingredients and find_terms(desc_features, (
            "infusion", "tisane", "brew", "steep", "tea bags", "teabags", "loose leaf"
        ))
    )
    generic_auxiliary_tea_support = bool(
        category_supports_tea and (details_has_tea_form or title_has_tea or raw_botanical)
    )
    title_fruit_tea = bool(
        title_has_tea and find_terms(title_text, (
            "fruit tea", "passion fruit", "pomegranate", "black currant",
            "blackberry", "berry", "berries", "tropical fruit", "peach",
            "mango", "pineapple", "orange", "lemon", "apple",
        ))
    )
    product_form = infer_product_form(title_category)

    # Exclusions use title/category first. Descriptive use examples such as
    # "serve with tea", "use a tea infuser", or "instead of cough syrup" do
    # not turn an otherwise valid product into a non-target product.
    sampler_matches = find_terms(title_category, SAMPLER_SET_TERMS)
    if (
        ("variety" in title_text.split() and ("flavor" in title_text or "flavors" in title_text or "one box of each" in title_text))
        or ("assortment" in title_text and title_has_tea)
        or ("measuring spoon included" in title_text)
        or ("2 count" in title_text and "+" in title and title_text.count("tea") >= 2)
    ):
        sampler_matches = list(dict.fromkeys(sampler_matches + ["multi_product_set_pattern"]))
    pod_matches = find_terms(title_category, POD_MACHINE_TERMS)
    ready_matches = find_terms(title_category, READY_TO_DRINK_TERMS)
    if "ginger brew" in title_text and ("ounce" in title_text or "case" in title_text):
        ready_matches = list(dict.fromkeys(ready_matches + ["ginger_brew_bottled_pattern"]))
    instant_matches = find_terms(title_category, INSTANT_MIX_TERMS)
    if "instant" in title_text.split() and title_has_tea:
        instant_matches = list(dict.fromkeys(instant_matches + ["instant_tea_pattern"]))
    slimming_matches = find_terms(title_text, SLIMMING_SUPPLEMENT_TERMS)
    fungal_matches = find_terms(title_text, FUNGAL_NON_BOTANICAL_TERMS)
    grain_matches = find_terms(title_text, GRAIN_INFUSION_TERMS)
    caffeinated_leaf_matches = find_terms(title_text, CAFFEINATED_NON_CAMELLIA_TERMS)
    teaware_matches = find_terms(title_category, TEAWARE_TERMS)
    coffee_alternative_matches = find_terms(title_text, (
        "coffee alternative", "alternative to coffee", "fake coffee",
        "herbal coffee", "chicory coffee", "teeccino",
    ))

    # A mixed sampler is excluded before tea-base classification because the
    # research requires one product/packaging system per parent_asin.
    if not parent_asin:
        reason = "missing_parent_asin"
    elif not title:
        reason = "missing_title"
    elif require_image and not main_image_url:
        reason = "missing_image"
    elif sampler_matches:
        reason = "mixed_sampler_or_gift_set"
        form_matches.extend(sampler_matches)
        near_excluded = bool(primary_target or category_target)
    elif title_true_tea:
        reason = "true_tea_base"
        scope_matches.extend(title_true_tea)
        near_excluded = bool(primary_target or ambiguous)
    elif formulation_true_tea:
        reason = "true_tea_base"
        scope_matches.extend(formulation_true_tea)
        near_excluded = bool(primary_target or ambiguous)
    elif (
        category_leaf in {"black", "green", "white", "oolong"}
        and not explicit_title_herbal
        and not primary_target
        and not title_fruit_tea
        and not find_terms(title_text, FRUIT_TERMS)
        and not find_terms(title_text, ("blackberry", "black currant", "pomegranate", "tropical fruit", "passion fruit"))
        and not find_terms(title_text, ("caffeine free", "caffeine-free"))
    ):
        reason = "true_tea_base"
        scope_matches.append(f"category_leaf_{category_leaf}")
        near_excluded = bool(primary_target or ambiguous)
    elif (
        secondary_true_tea
        and not explicit_title_herbal
        and not find_terms(title_text, ("rooibos", "honeybush"))
        and not (
            find_terms(title_text, ("caffeine free", "caffeine-free"))
            and (title_ingredients or title_fruit_tea or find_terms(title_text, FRUIT_TERMS))
        )
    ):
        reason = "true_tea_base"
        scope_matches.extend(secondary_true_tea)
        near_excluded = bool(primary_target or ambiguous)
    elif (
        find_terms(title_text, ("jasmine tea", "jasmine queen tea"))
        and not find_terms(title_text, ("jasmine flower", "jasmine flowers", "flower infusion"))
    ):
        reason = "true_tea_base"
        scope_matches.append("jasmine tea")
        near_excluded = True
    elif find_terms(title_text, ("masala chai", "chai tea", "chai green", "spiced apple chai")) and not (
        find_terms(title_text, ("rooibos chai", "honeybush chai"))
        or (explicit_title_herbal and find_terms(title_text, ("caffeine free", "caffeine-free")))
    ):
        reason = "true_tea_base"
        scope_matches.append("chai_without_herbal_caffeine_free_evidence")
        near_excluded = True
    elif (
        "breakfast" in title_text.split()
        and "caffeinated" in title_text.split()
        and explicit_title_herbal
    ):
        reason = "suspected_true_tea_base"
        scope_matches.append("caffeinated_breakfast_herbal")
        near_excluded = True
    elif pod_matches:
        reason = "kcup_pod_or_machine_pack"
        form_matches.extend(pod_matches)
        near_excluded = bool(primary_target or category_target)
    elif ready_matches:
        reason = "ready_to_drink_or_bottled"
        form_matches.extend(ready_matches)
        near_excluded = bool(primary_target or category_target)
    elif instant_matches:
        reason = "instant_powder_syrup_or_mix"
        form_matches.extend(instant_matches)
        near_excluded = bool(primary_target or category_target)
    elif slimming_matches:
        reason = "slimming_laxative_or_supplement"
        scope_matches.extend(slimming_matches)
        near_excluded = bool(primary_target or category_target)
    elif fungal_matches:
        reason = "non_botanical_or_fungal"
        scope_matches.extend(fungal_matches)
        near_excluded = bool(primary_target or category_target)
    elif grain_matches:
        reason = "grain_infusion"
        scope_matches.extend(grain_matches)
        near_excluded = bool(primary_target or category_target)
    elif caffeinated_leaf_matches:
        reason = "caffeinated_non_camellia"
        scope_matches.extend(caffeinated_leaf_matches)
        near_excluded = bool(primary_target or category_target)
    elif coffee_alternative_matches:
        reason = "coffee_alternative_or_nonstandard"
        scope_matches.extend(coffee_alternative_matches)
        near_excluded = bool(primary_target or category_target)
    elif teaware_matches:
        reason = "non_target_product"
        scope_matches.extend(teaware_matches)
    else:
        title_non_target = find_terms(title_text, GENERAL_NON_TARGET_TERMS)
        title_cooking = find_terms(title_text, COOKING_USE_TERMS)
        primary_cooking = find_terms(primary, COOKING_USE_TERMS)
        generic_raw_title = raw_botanical or bool(find_terms(title_text, (
            "powder", "ground", "root", "bark", "seed", "peel", "petals",
            "dried leaves", "whole leaves", "cut sifted", "cut and sifted",
        )))
        title_is_obvious_food = bool(find_terms(title_text, (
            "chips", "pita chips", "cookie", "cookies", "candy", "chocolate bar",
            "juice", "soda", "syrup", "seasoning", "spice blend", "protein powder",
        )))

        if title_is_obvious_food and not title_has_tea and not explicit_title_target:
            reason = "non_target_product"
            scope_matches.extend(title_non_target or ["obvious_food_title"])
        elif title_non_target and not title_has_tea and not explicit_title_target:
            reason = "non_target_product"
            scope_matches.extend(title_non_target)
        elif (
            (generic_raw_title or "sticks" in title_text.split() or "spice" in category)
            and (title_cooking or primary_cooking)
            and not explicit_title_target
            and not find_terms(title_text, ("herbal tea", "herb tea", "flower tea", "fruit infusion"))
            and not (title_has_tea and "spice" not in category and "seasoning" not in category)
        ):
            reason = "generic_herb_or_cooking_ingredient"
            scope_matches.extend(title_cooking or primary_cooking)
            near_excluded = bool(category_supports_tea or primary_ingredients)
        elif raw_botanical and (title_has_tea or explicit_for_tea):
            group = "borderline"
            reason = "dried_plant_material_for_tea"
        elif title_fruit_tea and not explicit_title_herbal:
            group = "borderline"
            reason = "ambiguous_target_candidate"
        elif explicit_title_target:
            generic_fruit = bool(find_terms(title_text, ("fruit tea",))) and not bool(
                find_terms(title_text, ("fruit infusion", "fruit herbal tea"))
            )
            wellness_ambiguous = bool(find_terms(title_text, (
                "wellness tea", "detox tea", "sleep tea", "bedtime tea"
            )))
            if generic_fruit or wellness_ambiguous:
                group = "borderline"
                reason = "ambiguous_target_candidate"
            else:
                group = "high_confidence"
                reason = "strong_title_target_evidence"
        elif ingredient_with_title_context:
            if title_has_dry_form:
                group = "high_confidence"
                reason = "ingredient_plus_dry_tea_form"
            else:
                group = "borderline"
                reason = "ingredient_plus_title_tea_context"
        elif title_has_tea and primary_ingredients:
            group = "borderline"
            reason = "botanical_tea_title"
        elif ingredient_with_brewing_context and category_supports_tea:
            group = "borderline"
            reason = "ingredient_plus_brewing_context"
        elif find_terms(title_text, ("fruit tea", "cold infuse")) and category_supports_tea:
            group = "borderline"
            reason = "ambiguous_target_candidate"
        elif generic_auxiliary_tea_support or ("infuse" in title_text.split() and details_has_tea_form):
            group = "borderline"
            reason = "generic_tea_product_with_auxiliary_support"
        else:
            reason = "no_primary_target_evidence"
            near_excluded = bool(category_target or primary_ingredients or ambiguous or category_true_tea)

    true_tea_all = list(dict.fromkeys(title_true_tea + formulation_true_tea + secondary_true_tea + category_true_tea))
    return {
        "parent_asin": parent_asin,
        "title": title,
        "store": str(record.get("store") or ""),
        "categories": compact_json(record.get("categories")),
        "description": compact_json(record.get("description")),
        "features": compact_json(record.get("features")),
        "details": compact_json(record.get("details")),
        "average_rating": record.get("average_rating", ""),
        "rating_number": record.get("rating_number", ""),
        "price": record.get("price", ""),
        "main_image_url": main_image_url,
        "all_image_urls": "|".join(all_image_urls),
        "matched_primary_target_terms": "|".join(primary_target),
        "matched_category_target_terms": "|".join(category_target),
        "matched_ingredient_terms": "|".join(primary_ingredients),
        "matched_context_terms": "|".join(primary_context),
        "matched_true_tea_terms": "|".join(true_tea_all),
        "matched_form_exclusion_terms": "|".join(dict.fromkeys(form_matches)),
        "matched_scope_exclusion_terms": "|".join(dict.fromkeys(scope_matches)),
        "matched_ambiguous_terms": "|".join(ambiguous),
        "automatic_tea_type": infer_tea_type(primary),
        "product_form": product_form,
        "screening_group": group,
        "screening_reason": reason,
        "near_excluded": int(near_excluded),
        "rules_version": RULES_VERSION,
    }


@dataclass
class ReservoirSampler:
    capacity: int
    rng: random.Random
    items: list[dict[str, Any]] = field(default_factory=list)
    seen: int = 0

    def add(self, item: dict[str, Any]) -> None:
        self.seen += 1
        if self.capacity <= 0:
            return
        if len(self.items) < self.capacity:
            self.items.append(dict(item))
            return
        index = self.rng.randrange(self.seen)
        if index < self.capacity:
            self.items[index] = dict(item)


def _open_csv_writer(path: Path, fields: Sequence[str]) -> tuple[Any, csv.DictWriter]:
    handle = path.open("w", encoding="utf-8-sig", newline="")
    writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    return handle, writer


def run_screening(
    input_path: Path,
    output_dir: Path,
    *,
    sample_high: int = 150,
    sample_borderline: int = 100,
    sample_near_excluded: int = 50,
    seed: int = 42,
    progress_every: int = 50_000,
    max_records: int | None = None,
    require_image: bool = True,
) -> dict[str, Any]:
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    if not input_path.exists():
        raise FileNotFoundError(f"Metadata JSONL not found: {input_path}")
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed)
    high_sampler = ReservoirSampler(sample_high, rng)
    border_sampler = ReservoirSampler(sample_borderline, rng)
    near_sampler = ReservoirSampler(sample_near_excluded, rng)

    counters: Counter[str] = Counter()
    tea_type_counts: Counter[str] = Counter()
    reason_counts: Counter[str] = Counter()
    primary_target_counts: Counter[str] = Counter()
    category_target_counts: Counter[str] = Counter()
    seen_parent_asins: set[str] = set()
    candidate_ids: set[str] = set()

    started = time.time()
    started_at = datetime.now(timezone.utc).isoformat()

    all_handle, all_writer = _open_csv_writer(output_dir / "candidate_products_v2_raw.csv", CSV_FIELDS)
    high_handle, high_writer = _open_csv_writer(output_dir / "candidate_products_v2_high_confidence.csv", CSV_FIELDS)
    border_handle, border_writer = _open_csv_writer(output_dir / "candidate_products_v2_borderline.csv", CSV_FIELDS)

    try:
        with input_path.open("r", encoding="utf-8") as source:
            for line in source:
                counters["records_scanned"] += 1
                if max_records is not None and counters["records_scanned"] > max_records:
                    counters["records_scanned"] -= 1
                    break
                if progress_every and counters["records_scanned"] % progress_every == 0:
                    elapsed = max(time.time() - started, 0.001)
                    print(
                        f"[进度] 已扫描 {counters['records_scanned']:,} 行；"
                        f"候选 {counters['candidate_count']:,}；"
                        f"速度 {counters['records_scanned']/elapsed:,.0f} 行/秒",
                        flush=True,
                    )
                if not line.strip():
                    counters["blank_lines"] += 1
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    counters["json_error_lines"] += 1
                    continue
                if not isinstance(record, Mapping):
                    counters["non_object_lines"] += 1
                    continue
                counters["valid_json_records"] += 1

                parent_asin = str(record.get("parent_asin") or "").strip()
                if parent_asin:
                    if parent_asin in seen_parent_asins:
                        counters["duplicate_parent_asins"] += 1
                        continue
                    seen_parent_asins.add(parent_asin)

                result = classify_product(record, require_image=require_image)
                if result["main_image_url"]:
                    counters["records_with_image"] += 1
                group = result["screening_group"]
                counters[f"{group}_count"] += 1
                reason_counts[result["screening_reason"]] += 1
                for term in filter(None, result["matched_primary_target_terms"].split("|")):
                    primary_target_counts[term] += 1
                for term in filter(None, result["matched_category_target_terms"].split("|")):
                    category_target_counts[term] += 1

                if group in {"high_confidence", "borderline"}:
                    counters["candidate_count"] += 1
                    candidate_ids.add(result["parent_asin"])
                    tea_type_counts[result["automatic_tea_type"]] += 1
                    all_writer.writerow(result)
                    if group == "high_confidence":
                        high_writer.writerow(result)
                        high_sampler.add(result)
                    else:
                        border_writer.writerow(result)
                        border_sampler.add(result)
                elif result["near_excluded"]:
                    counters["near_excluded_count"] += 1
                    near_sampler.add(result)
    finally:
        all_handle.close()
        high_handle.close()
        border_handle.close()

    with (output_dir / "candidate_parent_asins_v2.txt").open("w", encoding="utf-8") as handle:
        for parent_asin in sorted(candidate_ids):
            handle.write(parent_asin + "\n")

    sample_fields = tuple(CSV_FIELDS) + SAMPLE_EXTRA_FIELDS
    with (output_dir / "candidate_products_v2_review_sample.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=sample_fields, extrasaction="ignore")
        writer.writeheader()
        for source_name, items in (
            ("high_confidence", high_sampler.items),
            ("borderline", border_sampler.items),
            ("near_excluded", near_sampler.items),
        ):
            for item in items:
                row = dict(item)
                row.update({
                    "sample_source": source_name,
                    "human_product_valid": "",
                    "human_tea_type": "",
                    "human_exclusion_reason": "",
                    "human_notes": "",
                })
                writer.writerow(row)

    elapsed = time.time() - started
    report = {
        "rules_version": RULES_VERSION,
        "input_file": str(input_path),
        "input_size_bytes": input_path.stat().st_size,
        "output_dir": str(output_dir),
        "started_at_utc": started_at,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": round(elapsed, 3),
        "require_image": require_image,
        "max_records": max_records,
        "random_seed": seed,
        "records_scanned": counters["records_scanned"],
        "valid_json_records": counters["valid_json_records"],
        "json_error_lines": counters["json_error_lines"],
        "blank_lines": counters["blank_lines"],
        "non_object_lines": counters["non_object_lines"],
        "duplicate_parent_asins": counters["duplicate_parent_asins"],
        "unique_parent_asins_seen": len(seen_parent_asins),
        "records_with_image": counters["records_with_image"],
        "high_confidence_count": counters["high_confidence_count"],
        "borderline_count": counters["borderline_count"],
        "candidate_count": counters["candidate_count"],
        "excluded_count": counters["excluded_count"],
        "near_excluded_count": counters["near_excluded_count"],
        "candidate_image_rate": (
            1.0 if counters["candidate_count"] and require_image else None
        ),
        "tea_type_counts": dict(tea_type_counts.most_common()),
        "screening_reason_counts": dict(reason_counts.most_common()),
        "matched_primary_target_term_counts": dict(primary_target_counts.most_common()),
        "matched_category_target_term_counts": dict(category_target_counts.most_common()),
        "review_sample_counts": {
            "high_confidence": len(high_sampler.items),
            "borderline": len(border_sampler.items),
            "near_excluded": len(near_sampler.items),
            "total": len(high_sampler.items) + len(border_sampler.items) + len(near_sampler.items),
        },
        "output_files": {
            "all_candidates": "candidate_products_v2_raw.csv",
            "high_confidence": "candidate_products_v2_high_confidence.csv",
            "borderline": "candidate_products_v2_borderline.csv",
            "review_sample": "candidate_products_v2_review_sample.csv",
            "candidate_ids": "candidate_parent_asins_v2.txt",
            "report": "metadata_screening_v2_report.json",
        },
    }
    with (output_dir / "metadata_screening_v2_report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)

    print("\n=== V2元数据初筛完成 ===")
    print(f"扫描记录：{report['records_scanned']:,}")
    print(f"高置信候选：{report['high_confidence_count']:,}")
    print(f"边界候选：{report['borderline_count']:,}")
    print(f"候选合计：{report['candidate_count']:,}")
    print(f"输出目录：{output_dir}")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="V2：全量草本与花果茶商品元数据初筛")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="元数据JSONL路径")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="输出目录")
    parser.add_argument("--sample-high", type=int, default=150)
    parser.add_argument("--sample-borderline", type=int, default=100)
    parser.add_argument("--sample-near-excluded", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--progress-every", type=int, default=50_000)
    parser.add_argument("--max-records", type=int, default=None, help="试运行时限制扫描行数")
    parser.add_argument("--allow-missing-image", action="store_true", help="不把缺少图片作为排除条件")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        run_screening(
            args.input,
            args.output_dir,
            sample_high=args.sample_high,
            sample_borderline=args.sample_borderline,
            sample_near_excluded=args.sample_near_excluded,
            seed=args.seed,
            progress_every=args.progress_every,
            max_records=args.max_records,
            require_image=not args.allow_missing_image,
        )
    except (FileNotFoundError, PermissionError, OSError) as exc:
        print(f"错误：{exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
