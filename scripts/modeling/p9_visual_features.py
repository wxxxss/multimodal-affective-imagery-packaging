"""P9 label-blind visual measurement contract and feature freeze."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import io
import json
import math
import os
import subprocess
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from PIL import Image, ImageOps


ROOT = Path(__file__).resolve().parents[2]
CONTRACT_PATH = ROOT / "config/modeling/p9_visual_feature_contract.json"
SCHEMA_PATH = ROOT / "config/modeling/p9_interpretable_feature_schema.json"
PROMPT_PATH = ROOT / "config/modeling/p9_semantic_prompt_bank.json"
P7D_MANIFEST = ROOT / "data/processed/retail_outer_package_images_p7_5180/p7_d_final_image_freeze/04_final_image_manifest.csv"
FORMAL_DIR = ROOT / "data/processed/modeling_features_p9_5180/p9_visual_feature_freeze"
CHECKPOINT_PATH = ROOT / "data/models/p9_openclip_vit_b32_laion2b_s34b_b79k/open_clip_vit_b32_laion2b_s34b_b79k.pt"
PRIMARY_ROOT = ROOT / "data/images/retail_outer_package_p7_5180"
SENSITIVITY_ROOT = ROOT / "data/images/retail_outer_package_p7_5180/sensitivity"

FORMAL_FILES = (
    "01_feature_source_manifest.csv",
    "02_unique_image_inventory.csv",
    "03_openclip_image_embeddings.npy",
    "04_interpretable_image_features.csv",
    "05_product_feature_map.csv",
    "06_feature_quality_audit.json",
    "07_feature_summary.json",
    "08_p9_provenance.json",
)

CLASSICAL_FEATURES = (
    "luminance_mean",
    "luminance_std",
    "luminance_p10",
    "luminance_p90",
    "saturation_mean",
    "saturation_std",
    "colorfulness_hs",
    "high_saturation_fraction",
    "near_white_fraction",
    "near_black_fraction",
    "warm_hue_fraction",
    "cool_hue_fraction",
    "hue_entropy",
    "grayscale_entropy",
    "edge_density",
    "edge_strength_mean",
    "lr_symmetry",
    "tb_symmetry",
    "center_edge_energy_fraction",
    "quadrant_edge_imbalance",
)

FORBIDDEN_SOURCE_TERMS = (
    "outcome",
    "label",
    "review",
    "split",
    "development_fold",
    "observed_positive",
    "secondary",
    "performance",
)
FORBIDDEN_PROMPT_TERMS = {
    "general_visual_appeal",
    "cute_friendly",
    "premium_refined",
    "gift_presentation",
    "simple_modern",
    "natural_botanical",
    "calming_soft",
    "cheerful_colorful",
    "traditional_vintage",
    "negative_appearance",
}
FORBIDDEN_PROMPT_ALIASES = (
    "cheerful colorful",
    "calming soft",
    "cute friendly",
    "gift presentation",
    "general visual appeal",
    "negative appearance",
    "natural botanical",
    "premium refined",
    "simple modern",
    "traditional vintage",
)
FROZEN_PROMPT_TEMPLATES = (
    "a front-facing retail herbal tea package with {phrase}",
    "product packaging featuring {phrase}",
    "a tea package design showing {phrase}",
)
FROZEN_SEMANTIC_NAMES = (
    "leaf_herb_illustration_score",
    "floral_illustration_score",
    "fruit_motif_score",
    "ingredient_photography_score",
    "character_mascot_score",
    "sparse_layout_score",
    "dense_ornament_score",
    "heritage_ornament_score",
    "geometric_layout_score",
    "kraft_craft_score",
    "ribbon_bow_box_cues_score",
    "pale_pastel_palette_score",
    "vivid_multicolor_palette_score",
    "typography_dominant_score",
    "image_dominant_score",
    "transparent_window_score",
)
FROZEN_CLASSICAL_DEFINITIONS = (
    "mean(0.2126R + 0.7152G + 0.0722B)",
    "std(0.2126R + 0.7152G + 0.0722B)",
    "p10(0.2126R + 0.7152G + 0.0722B)",
    "p90(0.2126R + 0.7152G + 0.0722B)",
    "mean(HSV saturation)",
    "std(HSV saturation)",
    "sqrt(std(R-G)^2 + std(0.5(R+G)-B)^2) + 0.3*sqrt(mean(R-G)^2 + mean(0.5(R+G)-B)^2)",
    "fraction(HSV saturation >= 0.65)",
    "fraction(R>=0.90 and G>=0.90 and B>=0.90)",
    "fraction(R<=0.10 and G<=0.10 and B<=0.10)",
    "chromatic fraction in [330,360) union [0,60) degrees",
    "chromatic fraction in [150,270] degrees",
    "12-bin chromatic hue entropy divided by log(12)",
    "64-bin luminance entropy divided by log(64)",
    "fraction(sqrt(gx^2+gy^2) >= 0.10)",
    "mean(sqrt(gx^2+gy^2))",
    "clip(1 - mean(abs(gray - left_right_flip(gray))), 0, 1)",
    "clip(1 - mean(abs(gray - top_bottom_flip(gray))), 0, 1)",
    "center 50% gradient magnitude sum / total gradient magnitude sum",
    "std of four quadrant gradient-energy proportions",
)
FROZEN_CLASSICAL_RASTER = {
    "exif_transpose": True,
    "first_decoded_frame": True,
    "convert": "RGB",
    "preserve_aspect_ratio": True,
    "longest_side": 512,
    "resample": "LANCZOS",
    "crop": False,
    "segmentation": False,
}


class FeatureContractError(RuntimeError):
    pass


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest().upper()


def sha256_file(path: Path) -> str:
    if not path.is_file():
        raise FeatureContractError(f"missing file for SHA: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _assert_exact(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise FeatureContractError(f"{label} mismatch: {actual!r} != {expected!r}")


def _normalize_prompt_text(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value)).lower()
    normalized = []
    for character in text:
        category = unicodedata.category(character)
        if character in {"_", "-"} or category.startswith("P") or (not character.isalnum() and not character.isspace()):
            normalized.append(" ")
        else:
            normalized.append(character)
    return " ".join("".join(normalized).split())


def validate_contract(contract: Mapping[str, Any] | None = None) -> dict[str, Any]:
    contract = dict(contract or read_json(CONTRACT_PATH))
    expected_keys = {
        "contract_version", "stage", "baseline_main_commit", "feature_firewall", "upstream",
        "model", "classical_analysis_raster", "source_manifest", "feature_outputs",
        "feature_schema", "prompt_bank_path", "interpretable_schema_path", "producer_path",
        "verification_policy", "no_modeling_state",
    }
    _assert_exact(set(contract), expected_keys, "contract top-level fields")
    _assert_exact(contract["contract_version"], "p9_v1.0", "contract version")
    _assert_exact(contract["stage"], "P9 final visual feature freeze", "contract stage")
    _assert_exact(contract["baseline_main_commit"], "ccc3ddc0f778988d5251f536afe1674fb05969b5", "baseline main commit")
    _assert_exact(contract["feature_firewall"], {
        "row_level_source_only": [
            "data/processed/retail_outer_package_images_p7_5180/p7_d_final_image_freeze/04_final_image_manifest.csv",
            "data/images/retail_outer_package_p7_5180 actual frozen image bytes",
        ],
        "forbidden_row_level_sources": [
            "labels", "review text", "G", "P8-B modeling manifest", "P8-B split assignment",
            "outcomes", "split_partition", "development_fold", "primary observed-positive count",
            "secondary outcome", "model performance",
        ],
        "forbidden_methods": [
            "model fitting", "PCA", "supervised dimensionality reduction", "feature selection",
            "OCR", "object detection", "segmentation", "VLM", "encoder comparison", "performance metrics",
        ],
        "labels_read": False,
        "review_text_read": False,
        "G_read": False,
        "split_read": False,
        "model_fitted": False,
        "pca": False,
        "feature_selection": False,
    }, "feature firewall")
    _assert_exact(contract["upstream"], {
        "p7_d_final_manifest_path": "data/processed/retail_outer_package_images_p7_5180/p7_d_final_image_freeze/04_final_image_manifest.csv",
        "p7_d_final_manifest_sha256": "D3B29B0E075DE1137B7A004D12ED5A1764BD548684E067E55D62E63B75416F31",
        "p8_b_formal_sha256": {
            "01_modeling_ready_manifest.csv": "4CC5D4389BE9FEC94CAAC0601A24A49841D3EAB0D0787BAE8A8DD4ACF52B86E9",
            "02_split_group_inventory.csv": "C834E5D71125DD32D69E6A4E8A791F2BB0EFEEAB2702576B300C3C1CB8C5FB89",
            "03_split_assignment.csv": "A0570644DA3BF6078B749562AEF80B120E100596932C0DE348911B48AB6EE6FE",
            "04_split_quality_audit.json": "E7DD1EE95CC340D9CEB13ECB15A005D434CC43760C4FE8834A858351231262B3",
            "05_modeling_ready_summary.json": "EA15823425FF5C589982D88513C9A1BCF0F1A88916EBC9C8655AD567F655DDC5",
            "06_p8_b_provenance.json": "994082F8EE943A2005E2C4C9CA45FC265FC755612D43E568B768F41B3FC2220B",
        },
    }, "upstream ledger")
    expected_model = {
        "library": "open_clip_torch",
        "version": "3.3.0",
        "name": "ViT-B-32",
        "pretrained_tag": "laion2b_s34b_b79k",
        "local_checkpoint_path": "data/models/p9_openclip_vit_b32_laion2b_s34b_b79k/open_clip_vit_b32_laion2b_s34b_b79k.pt",
        "checkpoint_sha256": "ABC9E5336889F261DA3D79936A97B6815C97F546C406337C623AEF4262BAFC19",
        "checkpoint_size_bytes": 605230035,
        "embedding_dimension": 512,
        "inference": {
            "eval": True, "inference_mode": True, "gradients": False, "dtype": "float32",
            "l2_normalized": True, "device": "cpu", "batch_size": 16, "autocast": False,
            "stochastic_augmentation": False, "random_crop": False,
        },
        "preprocessing": {
            "input_resolution": 224,
            "resize": "shortest-side to 224 before center crop",
            "crop": "center crop 224x224",
            "interpolation": "bicubic",
            "resize_mode": "shortest",
            "mean": [0.48145466, 0.4578275, 0.40821073],
            "std": [0.26862954, 0.26130258, 0.27577711],
        },
    }
    for model_field, expected_value in expected_model.items():
        _assert_exact(contract["model"].get(model_field), expected_value, f"model {model_field}")
    _assert_exact(contract["classical_analysis_raster"], FROZEN_CLASSICAL_RASTER, "classical raster")
    _assert_exact(contract["source_manifest"], {
        "path": "data/processed/modeling_features_p9_5180/p9_visual_feature_freeze/01_feature_source_manifest.csv",
        "rows": 5180, "main_rows": 5179, "excluded_rows": 1, "primary_available": 5179,
        "sensitivity_available_full": 14, "sensitivity_available_main": 13,
        "allowed_fields": [
            "parent_asin", "input_order", "primary_freeze_status", "main_analysis_included",
            "primary_response_sha256", "primary_local_path", "primary_decoded_format",
            "primary_width", "primary_height", "sensitivity_status", "sensitivity_response_sha256",
            "sensitivity_local_path", "sensitivity_decoded_format", "sensitivity_width",
            "sensitivity_height", "excluded_non_primary",
        ],
    }, "source manifest contract")
    _assert_exact(contract["feature_outputs"], {
        "directory": "data/processed/modeling_features_p9_5180/p9_visual_feature_freeze",
        "files": list(FORMAL_FILES),
        "git_tracked": False,
    }, "feature outputs")
    _assert_exact(contract["feature_schema"], {
        "classical_count": 20, "semantic_count": 16, "total_interpretable_count": 36,
        "embedding_shape": ["N_unique_images", 512],
        "duplicate_sha_rule": "one image_sha256 maps to one feature_row_index",
    }, "feature schema contract")
    _assert_exact(contract["prompt_bank_path"], "config/modeling/p9_semantic_prompt_bank.json", "prompt bank path")
    _assert_exact(contract["interpretable_schema_path"], "config/modeling/p9_interpretable_feature_schema.json", "schema path")
    _assert_exact(contract["producer_path"], "scripts/modeling/p9_visual_features.py", "producer path")
    _assert_exact(contract["verification_policy"], {
        "formal_prepare_network_calls": 0,
        "verify_existing_network_calls": 0,
        "verify_existing_writes": 0,
        "verify_recompute_network_calls": 0,
        "verify_recompute_writes": 0,
        "verify_recompute_model_training": 0,
        "recompute_embedding_max_abs_diff": 1e-6,
        "recompute_semantic_max_abs_diff": 1e-6,
    }, "verification tolerance policy")
    _assert_exact(contract["no_modeling_state"], {
        "feature_extraction_only": True, "training_started": False, "p10_started": False,
    }, "no-modeling state")
    checkpoint_path = resolve_repo_relative_path(contract["model"]["local_checkpoint_path"])
    if checkpoint_path != CHECKPOINT_PATH.resolve():
        raise FeatureContractError("checkpoint path mismatch")
    if checkpoint_path.stat().st_size != contract["model"]["checkpoint_size_bytes"]:
        raise FeatureContractError("checkpoint size mismatch")
    if sha256_file(checkpoint_path) != contract["model"]["checkpoint_sha256"]:
        raise FeatureContractError("checkpoint SHA mismatch")
    return contract


def validate_schema(schema: Mapping[str, Any] | None = None) -> dict[str, Any]:
    schema = dict(schema or read_json(SCHEMA_PATH))
    _assert_exact(set(schema), {
        "contract_version", "raster", "classical_feature_count", "classical_features",
        "semantic_feature_count", "semantic_features", "semantic_scores_are",
    }, "schema top-level fields")
    _assert_exact(schema["contract_version"], "p9_v1.0", "schema contract version")
    _assert_exact(schema["raster"], {
        "exif_transpose": True, "first_decoded_frame": True, "mode": "RGB",
        "preserve_aspect_ratio": True, "longest_side_pixels": 512, "resample": "LANCZOS",
        "crop": False, "segmentation": False, "background_removal": False,
    }, "schema raster")
    _assert_exact(schema["classical_feature_count"], 20, "classical feature count")
    _assert_exact(schema["classical_features"], [
        {"name": name, "definition": definition}
        for name, definition in zip(CLASSICAL_FEATURES, FROZEN_CLASSICAL_DEFINITIONS)
    ], "classical feature names/definitions")
    _assert_exact(schema["semantic_feature_count"], 16, "semantic feature count")
    _assert_exact(schema["semantic_features"], [
        {"name": name, "prompt_template_count": 3} for name in FROZEN_SEMANTIC_NAMES
    ], "semantic feature names")
    _assert_exact(schema["semantic_scores_are"], "semantic design similarity scores, not presence probabilities or human gold labels", "semantic score semantics")
    return schema


def validate_prompt_bank(bank: Mapping[str, Any] | None = None) -> dict[str, Any]:
    bank = dict(bank or read_json(PROMPT_PATH))
    _assert_exact(set(bank), {"contract_version", "templates", "attributes", "encoding", "anti_circularity"}, "prompt bank top-level fields")
    _assert_exact(bank["contract_version"], "p9_v1.0", "prompt bank contract version")
    _assert_exact(bank["templates"], list(FROZEN_PROMPT_TEMPLATES), "prompt templates")
    phrases = (
        "illustrated leaves, herbs, or plant sprigs",
        "prominent illustrated flowers or blossoms",
        "prominent fruit illustrations or fruit motifs",
        "photographic images of herbs, flowers, fruit, or tea ingredients",
        "a drawn character, mascot, animal, or person",
        "a sparse layout with few visual elements and generous empty space",
        "dense decorative borders, flourishes, patterns, and many visual elements",
        "heritage-style seals, badges, decorative borders, and serif lettering",
        "clean geometric shapes, blocks, lines, and grid-like organization",
        "brown kraft-paper texture, handmade stamp cues, or craft-paper styling",
        "ribbon, bow, decorative box, or boxed display cues",
        "predominantly pale pastel colors",
        "many vivid highly saturated colors",
        "large prominent typography occupying most of the front panel",
        "a large illustration or photograph dominating the front panel",
        "a transparent window showing the tea or ingredients inside",
    )
    prompt_text = _normalize_prompt_text(" ".join(
        template.format(phrase=item["phrase"])
        for item in bank["attributes"] for template in bank["templates"]
    ))
    padded = f" {prompt_text} "
    for alias in FORBIDDEN_PROMPT_ALIASES:
        if f" {alias} " in padded:
            raise FeatureContractError(f"prompt contains forbidden outcome alias: {alias}")
    _assert_exact(bank["attributes"], [
        {"name": name, "phrase": phrase}
        for name, phrase in zip(FROZEN_SEMANTIC_NAMES, phrases)
    ], "prompt attributes")
    _assert_exact(bank["encoding"], {
        "normalize_each_prompt": True,
        "mean_prompt_vectors": True,
        "renormalize_centroid": True,
        "image_score": "normalized_image_embedding dot normalized_attribute_text_centroid",
        "softmax": False,
        "threshold": False,
        "outcome_label_prompt_guard": True,
    }, "prompt encoding")
    _assert_exact(bank["anti_circularity"], {
        "normalization": "NFKC lowercase; underscores/hyphens/punctuation to spaces; collapse whitespace",
        "forbidden_outcome_aliases": list(FORBIDDEN_PROMPT_ALIASES),
    }, "anti-circularity policy")
    return bank


def build_prompt_centroids(prompt_vectors: np.ndarray, template_count: int | None = None) -> np.ndarray:
    values = np.asarray(prompt_vectors, dtype=np.float32)
    if values.ndim != 3 or values.shape[0] == 0 or values.shape[1] == 0 or values.shape[2] == 0:
        raise FeatureContractError(f"prompt vectors must have shape (attributes, templates, dimensions), got {values.shape}")
    if template_count is not None and values.shape[1] != template_count:
        raise FeatureContractError("prompt template count mismatch")
    norms = np.linalg.norm(values, axis=-1, keepdims=True)
    if not np.isfinite(values).all() or np.any(norms <= 0):
        raise FeatureContractError("prompt vectors must be finite and nonzero")
    normalized = values / norms
    centroids = normalized.mean(axis=1)
    centroid_norms = np.linalg.norm(centroids, axis=-1, keepdims=True)
    if np.any(centroid_norms <= 0):
        raise FeatureContractError("prompt centroid is zero")
    return (centroids / centroid_norms).astype(np.float32)


def semantic_similarity_scores(image_embeddings: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    images = np.asarray(image_embeddings, dtype=np.float32)
    attributes = np.asarray(centroids, dtype=np.float32)
    if images.ndim != 2 or attributes.ndim != 2 or images.shape[1] != attributes.shape[1]:
        raise FeatureContractError("semantic score matrix shape mismatch")
    if not np.isfinite(images).all() or not np.isfinite(attributes).all():
        raise FeatureContractError("semantic score inputs must be finite")
    scores = np.matmul(images, attributes.T).astype(np.float32)
    if not np.isfinite(scores).all() or np.any(scores < -1.00001) or np.any(scores > 1.00001):
        raise FeatureContractError("semantic scores must be finite cosine similarities in [-1, 1]")
    return scores


def git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def git_blob_sha(commit: str, path: str | Path) -> str:
    normalized = str(path).replace("\\", "/")
    try:
        data = subprocess.check_output(["git", "show", f"{commit}:{normalized}"], cwd=ROOT)
    except subprocess.CalledProcessError as exc:
        raise FeatureContractError(f"cannot resolve Git blob {commit}:{normalized}") from exc
    return sha256_bytes(data)


def resolve_repo_relative_path(value: str | Path, root: Path = ROOT) -> Path:
    raw = Path(str(value))
    candidate = (raw if raw.is_absolute() else root / raw).resolve()
    root_resolved = root.resolve()
    try:
        candidate.relative_to(root_resolved)
    except ValueError as exc:
        raise FeatureContractError(f"path escape: {value}") from exc
    return candidate


def _resolve_image_path(value: str | Path) -> Path:
    raw = Path(str(value))
    if raw.is_absolute():
        return raw.resolve()
    return (ROOT / raw).resolve()


def _display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(path.resolve()).replace("\\", "/")


def verify_image_bytes(path: str | Path, expected_sha256: str) -> Path:
    image_path = _resolve_image_path(path)
    if not image_path.is_file():
        raise FeatureContractError(f"missing image bytes: {image_path}")
    actual = sha256_file(image_path)
    if actual != str(expected_sha256).upper():
        raise FeatureContractError(f"image SHA mismatch for {image_path}: {actual} != {expected_sha256}")
    return image_path


def _bool_value(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def _asset_path(raw: str, role: str) -> str:
    value = str(raw or "").replace("\\", "/")
    if not value:
        return ""
    if Path(value).is_absolute() or value.startswith("data/"):
        return value
    if role == "primary" and value.startswith("primary/"):
        return (Path("data/images/retail_outer_package_p7_5180") / value).as_posix()
    if role == "sensitivity" and value.startswith("sensitivity/"):
        return (Path("data/images/retail_outer_package_p7_5180") / value).as_posix()
    return value


def build_source_rows(manifest_path: Path = P7D_MANIFEST) -> list[dict[str, str]]:
    """Project only admin/image fields from the P7-D final image manifest."""
    rows: list[dict[str, str]] = []
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        for raw in csv.DictReader(handle):
            excluded = _bool_value(raw.get("excluded_non_primary", False))
            main_included = not excluded
            primary_path = _asset_path(raw.get("primary_local_path") or raw.get("primary_asset_path", ""), "primary")
            sensitivity_path = _asset_path(raw.get("sensitivity_local_path") or raw.get("sensitivity_asset_path", ""), "sensitivity")
            row = {
                "parent_asin": str(raw.get("parent_asin", "")),
                "input_order": str(raw.get("input_order", "")),
                "primary_freeze_status": str(raw.get("primary_freeze_status", "")),
                "main_analysis_included": "1" if main_included else "0",
                "primary_response_sha256": str(raw.get("primary_response_sha256", "")).upper(),
                "primary_local_path": primary_path,
                "primary_decoded_format": str(raw.get("primary_decoded_format", "")),
                "primary_width": str(raw.get("primary_width", "0")),
                "primary_height": str(raw.get("primary_height", "0")),
                "sensitivity_status": str(raw.get("sensitivity_status", "")),
                "sensitivity_response_sha256": str(raw.get("sensitivity_response_sha256", "")).upper(),
                "sensitivity_local_path": sensitivity_path,
                "sensitivity_decoded_format": str(raw.get("sensitivity_decoded_format", "")),
                "sensitivity_width": str(raw.get("sensitivity_width", "0")),
                "sensitivity_height": str(raw.get("sensitivity_height", "0")),
                "excluded_non_primary": "1" if excluded else "0",
            }
            if not row["parent_asin"] or not row["input_order"]:
                raise FeatureContractError("P7-D source row is missing parent_asin or input_order")
            rows.append(row)
    if len({row["parent_asin"] for row in rows}) != len(rows):
        raise FeatureContractError("P9 source parent_asin values must be unique")
    if manifest_path.resolve() == P7D_MANIFEST.resolve():
        if len(rows) != 5180:
            raise FeatureContractError("P9 source population must be 5180 products")
        if sum(row["main_analysis_included"] == "1" for row in rows) != 5179:
            raise FeatureContractError("P9 source main population must be 5179")
        if sum(row["main_analysis_included"] == "0" for row in rows) != 1:
            raise FeatureContractError("P9 source excluded population must be 1")
    for row in rows:
        if any(term in key.lower() for key in row for term in FORBIDDEN_SOURCE_TERMS):
            raise FeatureContractError("P9 source projection contains a forbidden field")
    return rows


def _decode_info(path: Path) -> tuple[str, int, int]:
    try:
        with Image.open(path) as image:
            image.seek(0)
            return str(image.format or ""), int(image.width), int(image.height)
    except Exception as exc:
        raise FeatureContractError(f"cannot decode image: {path}") from exc


def build_unique_image_inventory(source_rows: Sequence[Mapping[str, str]]) -> list[dict[str, str]]:
    grouped: dict[str, dict[str, Any]] = {}

    def add_role(row: Mapping[str, str], role: str) -> None:
        sha_key = f"{role}_response_sha256"
        path_key = f"{role}_local_path"
        expected = str(row.get(sha_key, "")).upper()
        local_path = str(row.get(path_key, ""))
        if not expected or not local_path:
            return
        path = verify_image_bytes(local_path, expected)
        item = grouped.setdefault(expected, {"paths": set(), "roles": []})
        item["paths"].add(path)
        item["roles"].append((str(row["parent_asin"]), role, path))

    for row in source_rows:
        if row.get("main_analysis_included") == "1":
            add_role(row, "primary")
        if row.get("sensitivity_status") == "available":
            add_role(row, "sensitivity")
    inventory: list[dict[str, str]] = []
    for index, image_sha in enumerate(sorted(grouped)):
        item = grouped[image_sha]
        paths = sorted(item["paths"], key=lambda path: _display_path(path))
        canonical = paths[0]
        decoded_format, width, height = _decode_info(canonical)
        primary_products = {asin for asin, role, _ in item["roles"] if role == "primary"}
        sensitivity_products = {asin for asin, role, _ in item["roles"] if role == "sensitivity"}
        inventory.append({
            "feature_row_index": str(index),
            "image_sha256": image_sha,
            "canonical_local_path": _display_path(canonical),
            "primary_product_count": str(len(primary_products)),
            "sensitivity_product_count": str(len(sensitivity_products)),
            "used_as_primary": "1" if primary_products else "0",
            "used_as_sensitivity": "1" if sensitivity_products else "0",
            "decoded_format": decoded_format,
            "width": str(width),
            "height": str(height),
            "source_role_count": str(len(item["roles"])),
        })
    if len(source_rows) == 5180 and sum(int(row["used_as_primary"]) for row in inventory) != 4981:
        raise FeatureContractError("P9 primary unique image count must be 4981")
    return inventory


def build_product_feature_map(source_rows: Sequence[Mapping[str, str]], inventory: Sequence[Mapping[str, str]]) -> list[dict[str, str]]:
    index_by_sha = {row["image_sha256"]: row["feature_row_index"] for row in inventory}
    output: list[dict[str, str]] = []
    for row in source_rows:
        primary_index = ""
        if row.get("main_analysis_included") == "1" and row.get("primary_response_sha256"):
            primary_index = index_by_sha.get(str(row["primary_response_sha256"]).upper(), "")
            if not primary_index:
                raise FeatureContractError(f"missing primary feature mapping: {row['parent_asin']}")
        sensitivity_index = ""
        if row.get("sensitivity_status") == "available" and row.get("sensitivity_response_sha256"):
            sensitivity_index = index_by_sha.get(str(row["sensitivity_response_sha256"]).upper(), "")
            if not sensitivity_index:
                raise FeatureContractError(f"missing sensitivity feature mapping: {row['parent_asin']}")
        output.append({
            "parent_asin": str(row["parent_asin"]),
            "input_order": str(row["input_order"]),
            "main_analysis_included": str(row["main_analysis_included"]),
            "primary_response_sha256": str(row.get("primary_response_sha256", "")),
            "primary_feature_row_index": primary_index,
            "sensitivity_status": str(row.get("sensitivity_status", "")),
            "sensitivity_response_sha256": str(row.get("sensitivity_response_sha256", "")),
            "sensitivity_feature_row_index": sensitivity_index,
            "primary_feature_available": "1" if primary_index else "0",
            "sensitivity_feature_available": "1" if sensitivity_index else "0",
        })
    return output


def _analysis_image(path: Path) -> Image.Image:
    with Image.open(path) as opened:
        image = ImageOps.exif_transpose(opened)
        try:
            image.seek(0)
        except (AttributeError, EOFError):
            pass
        image = image.convert("RGB")
        longest = max(image.width, image.height)
        if longest <= 0:
            raise FeatureContractError(f"empty image dimensions: {path}")
        scale = 512.0 / longest
        size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
        return image.resize(size, Image.Resampling.LANCZOS)


def _entropy(values: np.ndarray, bins: int, value_range: tuple[float, float]) -> float:
    counts, _ = np.histogram(values, bins=bins, range=value_range)
    probabilities = counts.astype(np.float64) / max(1, values.size)
    probabilities = probabilities[probabilities > 0]
    if not probabilities.size:
        return 0.0
    return float(-np.sum(probabilities * np.log(probabilities)) / math.log(bins))


def _hsv(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    maximum = rgb.max(axis=2)
    minimum = rgb.min(axis=2)
    delta = maximum - minimum
    saturation = np.divide(delta, maximum, out=np.zeros_like(maximum), where=maximum > 0)
    hue = np.zeros_like(maximum)
    nonzero = delta > 0
    red, green, blue = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    mask = nonzero & (maximum == red)
    hue[mask] = ((green[mask] - blue[mask]) / delta[mask]) % 6
    mask = nonzero & (maximum == green)
    hue[mask] = ((blue[mask] - red[mask]) / delta[mask]) + 2
    mask = nonzero & (maximum == blue)
    hue[mask] = ((red[mask] - green[mask]) / delta[mask]) + 4
    return (hue / 6.0) % 1.0, saturation, maximum


def compute_classical_features(path: str | Path) -> dict[str, float]:
    image = _analysis_image(_resolve_image_path(path))
    rgb = np.asarray(image, dtype=np.float64) / 255.0
    gray = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
    hue, saturation, _ = _hsv(rgb)
    rg = rgb[..., 0] - rgb[..., 1]
    yb = 0.5 * (rgb[..., 0] + rgb[..., 1]) - rgb[..., 2]
    colorfulness = math.sqrt(float(rg.std() ** 2 + yb.std() ** 2)) + 0.3 * math.sqrt(float(rg.mean() ** 2 + yb.mean() ** 2))
    chromatic = saturation >= 0.20
    warm = chromatic & ((hue >= 330 / 360) | (hue < 60 / 360))
    cool = chromatic & (hue >= 150 / 360) & (hue <= 270 / 360)
    chromatic_count = max(1, int(chromatic.sum()))
    gx = np.zeros_like(gray)
    gy = np.zeros_like(gray)
    gx[:, 1:] = gray[:, 1:] - gray[:, :-1]
    gy[1:, :] = gray[1:, :] - gray[:-1, :]
    gradient = np.sqrt(gx * gx + gy * gy)
    total_energy = float(gradient.sum())
    h, w = gray.shape
    y0, y1 = h // 4, h - h // 4
    x0, x1 = w // 4, w - w // 4
    quadrant_sums = [
        float(gradient[: h // 2, : w // 2].sum()),
        float(gradient[: h // 2, w // 2 :].sum()),
        float(gradient[h // 2 :, : w // 2].sum()),
        float(gradient[h // 2 :, w // 2 :].sum()),
    ]
    quadrant_total = sum(quadrant_sums)
    quadrant_props = np.asarray(quadrant_sums, dtype=np.float64) / (quadrant_total if quadrant_total else 1.0)
    features = {
        "luminance_mean": float(gray.mean()),
        "luminance_std": float(gray.std()),
        "luminance_p10": float(np.percentile(gray, 10)),
        "luminance_p90": float(np.percentile(gray, 90)),
        "saturation_mean": float(saturation.mean()),
        "saturation_std": float(saturation.std()),
        "colorfulness_hs": float(colorfulness),
        "high_saturation_fraction": float(np.mean(saturation >= 0.65)),
        "near_white_fraction": float(np.mean(np.all(rgb >= 0.90, axis=2))),
        "near_black_fraction": float(np.mean(np.all(rgb <= 0.10, axis=2))),
        "warm_hue_fraction": float(warm.sum() / chromatic_count) if chromatic.any() else 0.0,
        "cool_hue_fraction": float(cool.sum() / chromatic_count) if chromatic.any() else 0.0,
        "hue_entropy": _entropy(hue[chromatic], 12, (0.0, 1.0)) if chromatic.any() else 0.0,
        "grayscale_entropy": _entropy(gray, 64, (0.0, 1.0)),
        "edge_density": float(np.mean(gradient >= 0.10)),
        "edge_strength_mean": float(gradient.mean()),
        "lr_symmetry": float(np.clip(1.0 - np.mean(np.abs(gray - np.fliplr(gray))), 0.0, 1.0)),
        "tb_symmetry": float(np.clip(1.0 - np.mean(np.abs(gray - np.flipud(gray))), 0.0, 1.0)),
        "center_edge_energy_fraction": float(gradient[y0:y1, x0:x1].sum() / total_energy) if total_energy else 0.0,
        "quadrant_edge_imbalance": float(np.std(quadrant_props)),
    }
    if tuple(features) != CLASSICAL_FEATURES or not all(np.isfinite(value) for value in features.values()):
        raise FeatureContractError("classical feature output is not the frozen finite 20-field schema")
    return features


def validate_embeddings(embeddings: np.ndarray, contract: Mapping[str, Any] | None = None) -> None:
    contract = contract or read_json(CONTRACT_PATH)
    model = contract["model"]
    expected_dimension = int(model["embedding_dimension"])
    expected_dtype = str(model["inference"]["dtype"])
    if embeddings.ndim != 2 or embeddings.shape[1] != expected_dimension:
        raise FeatureContractError(f"embedding dimension must be {expected_dimension}, got {embeddings.shape}")
    if embeddings.dtype != np.dtype(expected_dtype):
        raise FeatureContractError(f"embedding dtype must be {expected_dtype}, got {embeddings.dtype}")
    if not np.isfinite(embeddings).all():
        raise FeatureContractError("embedding values must be finite")
    norms = np.linalg.norm(embeddings, axis=1)
    if not np.allclose(norms, 1.0, rtol=0, atol=1e-5):
        raise FeatureContractError("embedding rows must be L2 normalized")


def _load_model() -> tuple[Any, Any, Any, list[str]]:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    contract = validate_contract()
    validate_schema()
    bank = validate_prompt_bank()
    try:
        import open_clip
        import torch
        import torch.nn.functional as torch_functional
    except ImportError as exc:
        raise FeatureContractError("P9 requires open_clip_torch 3.3.0 and torch") from exc
    model_config = contract["model"]
    package_version = importlib.metadata.version("open_clip_torch")
    if str(package_version) != model_config["version"] or str(open_clip.__version__) != model_config["version"]:
        raise FeatureContractError(f"open_clip_torch version mismatch: {package_version}/{open_clip.__version__}")
    checkpoint_path = resolve_repo_relative_path(model_config["local_checkpoint_path"])
    if sha256_file(checkpoint_path) != model_config["checkpoint_sha256"]:
        raise FeatureContractError("local checkpoint SHA mismatch")
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_config["name"],
        pretrained=None,
        device=model_config["inference"]["device"],
        precision="fp32",
    )
    checkpoint = torch.load(checkpoint_path, map_location=model_config["inference"]["device"], weights_only=True)
    state_dict = checkpoint.get("state_dict", checkpoint)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    if model.training:
        raise FeatureContractError("model must be in eval mode")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise FeatureContractError("model gradients must be disabled")
    templates = bank["templates"]
    attributes = bank["attributes"]
    prompts = [template.format(phrase=item["phrase"]) for item in attributes for template in templates]
    with torch.inference_mode():
        tokens = open_clip.tokenize(prompts)
        text_features = model.encode_text(tokens)
        text_features = torch_functional.normalize(text_features.float(), dim=-1)
        text_features = text_features.reshape(len(attributes), len(templates), -1).mean(dim=1)
        centroids = torch_functional.normalize(text_features, dim=-1).cpu().numpy().astype(np.float32)
    return model, preprocess, centroids, [item["name"] for item in attributes]


def _extract_model_features(inventory: Sequence[Mapping[str, str]], batch_size: int = 16) -> tuple[np.ndarray, np.ndarray]:
    model, preprocess, centroids, semantic_names = _load_model()
    del semantic_names
    import torch
    import torch.nn.functional as torch_functional

    embeddings: list[np.ndarray] = []
    paths = [_resolve_image_path(row["canonical_local_path"]) for row in inventory]
    for start in range(0, len(paths), batch_size):
        images = []
        for path in paths[start : start + batch_size]:
            with Image.open(path) as opened:
                image = ImageOps.exif_transpose(opened)
                try:
                    image.seek(0)
                except (AttributeError, EOFError):
                    pass
                images.append(image.convert("RGB"))
        batch = torch.stack([preprocess(image) for image in images]).to(dtype=torch.float32)
        with torch.inference_mode():
            encoded = torch_functional.normalize(model.encode_image(batch).float(), dim=-1)
        embeddings.append(encoded.cpu().numpy().astype(np.float32))
    image_embeddings = np.concatenate(embeddings, axis=0).astype(np.float32, copy=False)
    validate_embeddings(image_embeddings)
    semantic_scores = semantic_similarity_scores(image_embeddings, centroids)
    return image_embeddings, semantic_scores


def _feature_rows(inventory: Sequence[Mapping[str, str]], semantic_scores: np.ndarray) -> list[dict[str, str]]:
    bank = read_json(PROMPT_PATH)
    semantic_names = [item["name"] for item in bank["attributes"]]
    if semantic_scores.shape != (len(inventory), len(semantic_names)):
        raise FeatureContractError("semantic score shape mismatch")
    rows: list[dict[str, str]] = []
    for index, item in enumerate(inventory):
        row = {"feature_row_index": item["feature_row_index"], "image_sha256": item["image_sha256"]}
        row.update({name: repr(float(value)) for name, value in compute_classical_features(item["canonical_local_path"]).items()})
        row.update({name: repr(float(value)) for name, value in zip(semantic_names, semantic_scores[index])})
        rows.append(row)
    return rows


def _csv_bytes(rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(fieldnames), lineterminator="\n")
    writer.writeheader()
    writer.writerows({field: row.get(field, "") for field in fieldnames} for row in rows)
    return buffer.getvalue().encode("utf-8")


def _write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    _write_bytes(path, (json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8"))


def _float_values(feature_rows: Sequence[Mapping[str, str]], names: Sequence[str]) -> np.ndarray:
    return np.asarray([[float(row[name]) for name in names] for row in feature_rows], dtype=np.float64)


def _diagnostics(values: np.ndarray, names: Sequence[str]) -> dict[str, dict[str, float]]:
    output: dict[str, dict[str, float]] = {}
    for index, name in enumerate(names):
        column = values[:, index]
        quantiles = np.percentile(column, [1, 10, 50, 90, 99])
        output[name] = {
            "min": float(column.min()),
            "p01": float(quantiles[0]),
            "p10": float(quantiles[1]),
            "median": float(quantiles[2]),
            "p90": float(quantiles[3]),
            "p99": float(quantiles[4]),
            "max": float(column.max()),
            "mean": float(column.mean()),
            "std": float(column.std()),
        }
    return output


def build_quality_audit(source_rows: Sequence[Mapping[str, str]], inventory: Sequence[Mapping[str, str]], embeddings: np.ndarray, feature_rows: Sequence[Mapping[str, str]], product_rows: Sequence[Mapping[str, str]]) -> dict[str, Any]:
    validate_embeddings(embeddings)
    semantic_names = [item["name"] for item in read_json(PROMPT_PATH)["attributes"]]
    classical = _float_values(feature_rows, CLASSICAL_FEATURES)
    semantic = _float_values(feature_rows, semantic_names)
    all_values = np.concatenate([classical, semantic], axis=1)
    duplicate_sha_violations = len({row["image_sha256"] for row in inventory}) != len(inventory)
    return {
        "universe_rows": len(source_rows),
        "main_rows": sum(row["main_analysis_included"] == "1" for row in source_rows),
        "excluded_rows": sum(row["main_analysis_included"] == "0" for row in source_rows),
        "primary_products": sum(int(row["primary_feature_available"]) for row in product_rows),
        "primary_unique_image_count": sum(int(row["used_as_primary"]) for row in inventory),
        "sensitivity_available_products": sum(row["sensitivity_status"] == "available" for row in source_rows),
        "main_sensitivity_available_products": sum(row["sensitivity_status"] == "available" and row["main_analysis_included"] == "1" for row in source_rows),
        "combined_unique_image_count": len(inventory),
        "embedding_shape": list(embeddings.shape),
        "embedding_dtype": str(embeddings.dtype),
        "embedding_dimension": int(embeddings.shape[1]),
        "embedding_nan": int(np.isnan(embeddings).sum()),
        "embedding_inf": int(np.isinf(embeddings).sum()),
        "embedding_norm_min": float(np.linalg.norm(embeddings, axis=1).min()),
        "embedding_norm_median": float(np.median(np.linalg.norm(embeddings, axis=1))),
        "embedding_norm_max": float(np.linalg.norm(embeddings, axis=1).max()),
        "classical_feature_count": len(CLASSICAL_FEATURES),
        "semantic_feature_count": len(semantic_names),
        "interpretable_feature_total": len(CLASSICAL_FEATURES) + len(semantic_names),
        "classical_nan": int(np.isnan(classical).sum()),
        "classical_inf": int(np.isinf(classical).sum()),
        "semantic_nan": int(np.isnan(semantic).sum()),
        "semantic_inf": int(np.isinf(semantic).sum()),
        "semantic_min": float(semantic.min()),
        "semantic_max": float(semantic.max()),
        "duplicate_sha_mapping_violations": int(duplicate_sha_violations),
        "missing_primary_mapping": sum(row["main_analysis_included"] == "1" and row["primary_feature_available"] != "1" for row in product_rows),
        "missing_available_sensitivity_mapping": sum(row["sensitivity_status"] == "available" and row["sensitivity_feature_available"] != "1" for row in product_rows),
        "primary_sensitivity_substitution_violations": sum(
            row["main_analysis_included"] == "1" and row["primary_response_sha256"] != "" and row["primary_feature_row_index"] == row["sensitivity_feature_row_index"]
            and row["sensitivity_status"] == "available" and row["primary_response_sha256"] != row["sensitivity_response_sha256"]
            for row in product_rows
        ),
        "feature_diagnostics": _diagnostics(all_values, (*CLASSICAL_FEATURES, *semantic_names)),
    }


def build_summary(source_rows: Sequence[Mapping[str, str]], inventory: Sequence[Mapping[str, str]], embeddings: np.ndarray, product_rows: Sequence[Mapping[str, str]], output_sha256: Mapping[str, str]) -> dict[str, Any]:
    return {
        "contract_version": "p9_v1.0",
        "p9_a": "PASS",
        "p9_b": "PASS",
        "p9_c": "PASS",
        "p9_d": "PASS",
        "visual_feature_integrity": "PASS",
        "universe": len(source_rows),
        "main": sum(row["main_analysis_included"] == "1" for row in source_rows),
        "excluded": sum(row["main_analysis_included"] == "0" for row in source_rows),
        "unique_primary_images": sum(int(row["used_as_primary"]) for row in inventory),
        "combined_unique_feature_images": len(inventory),
        "sensitivity_available_full": sum(row["sensitivity_status"] == "available" for row in source_rows),
        "sensitivity_available_main": sum(row["sensitivity_status"] == "available" and row["main_analysis_included"] == "1" for row in source_rows),
        "embedding_shape": list(embeddings.shape),
        "embedding_dtype": str(embeddings.dtype),
        "classical_features": 20,
        "semantic_features": 16,
        "total_interpretable_features": 36,
        "primary_products_mapped": sum(row["primary_feature_available"] == "1" for row in product_rows),
        "missing_primary_mapping": sum(row["main_analysis_included"] == "1" and row["primary_feature_available"] != "1" for row in product_rows),
        "duplicate_sha_violations": 0,
        "nan": int(np.isnan(embeddings).sum()),
        "inf": int(np.isinf(embeddings).sum()),
        "labels_read": False,
        "review_text_read": False,
        "G_read": False,
        "split_read": False,
        "pca": False,
        "feature_selection": False,
        "model_fitted": False,
        "performance_inspected": False,
        "p10_started": False,
        "formal_output_sha256": dict(output_sha256),
    }


def _upstream_verification() -> dict[str, Any]:
    # P8-B verify reconstructs P8-A, which reconstructs all four P7 gates.
    command = ['scripts/modeling/p8_b_modeling_ready_split.py', '--verify-existing']
    completed = subprocess.run([sys.executable, *command], cwd=ROOT, capture_output=True, text=True, check=False)
    if completed.returncode:
        raise FeatureContractError('p8_b upstream verification failed')
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise FeatureContractError('p8_b upstream verifier did not emit JSON') from exc
    p8a = payload.get('upstream_p8_a_verification', {})
    p7 = p8a.get('p7_verify', {})
    if payload.get('verification') != 'PASS' or p8a.get('verification') != 'PASS':
        raise FeatureContractError('P8-A/B upstream verification did not PASS')
    if any(p7.get(name) != 'PASS' for name in ('p7_a', 'p7_b', 'p7_c', 'p7_d')):
        raise FeatureContractError('P7 upstream verification did not PASS: ' + repr(p7))
    return {'p7_a': 'PASS', 'p7_b': 'PASS', 'p7_c': 'PASS', 'p7_d': 'PASS', 'p8_a': 'PASS', 'p8_b': 'PASS'}

def _check_upstream_sha(contract: Mapping[str, Any]) -> None:
    upstream = contract["upstream"]
    if sha256_file(P7D_MANIFEST) != upstream["p7_d_final_manifest_sha256"]:
        raise FeatureContractError("P7-D final manifest SHA mismatch")
    p8_dir = ROOT / "data/processed/modeling_readiness_p8_5180/p8_b_modeling_ready_split"
    for name, expected in upstream["p8_b_formal_sha256"].items():
        if name == "06_p8_b_provenance.json":
            path = p8_dir / name
        else:
            path = p8_dir / name
        if sha256_file(path) != expected:
            raise FeatureContractError(f"P8-B formal SHA mismatch: {name}")


def _runtime_environment(contract: Mapping[str, Any]) -> dict[str, Any]:
    try:
        import open_clip
        import torch
        import PIL
    except ImportError as exc:
        raise FeatureContractError("P9 runtime environment is incomplete") from exc
    return {
        "python_version": sys.version.split()[0],
        "open_clip_torch_version": importlib.metadata.version("open_clip_torch"),
        "open_clip_version": str(open_clip.__version__),
        "torch_version": str(torch.__version__),
        "numpy_version": str(np.__version__),
        "pillow_version": str(PIL.__version__),
        "device": contract["model"]["inference"]["device"],
        "cuda_version": torch.version.cuda,
        "batch_size": contract["model"]["inference"]["batch_size"],
        "preprocess_repr": "Resize(224, bicubic, shortest) + CenterCrop(224) + Normalize(OpenCLIP mean/std)",
    }


def validate_provenance(
    provenance: Mapping[str, Any],
    contract: Mapping[str, Any] | None = None,
    *,
    verify_git_traces: bool = False,
) -> None:
    contract = contract or validate_contract()
    outputs = provenance.get("formal_output_sha256", {})
    if "08_p9_provenance.json" in outputs:
        raise FeatureContractError("provenance records its own SHA")
    if provenance.get("provenance_self_sha") is not None:
        raise FeatureContractError("provenance contains its own SHA")
    if provenance.get("model") != contract["model"]:
        raise FeatureContractError("provenance checkpoint/model binding mismatch")
    if provenance.get("baseline_main_commit") != contract["baseline_main_commit"]:
        raise FeatureContractError("provenance baseline mismatch")
    if provenance.get("p7_d_final_manifest", {}).get("path") != contract["upstream"]["p7_d_final_manifest_path"]:
        raise FeatureContractError("provenance P7-D path mismatch")
    if provenance.get("p7_d_final_manifest", {}).get("sha256") != contract["upstream"]["p7_d_final_manifest_sha256"]:
        raise FeatureContractError("provenance P7-D SHA mismatch")
    if provenance.get("p8_b_formal_sha256") != contract["upstream"]["p8_b_formal_sha256"]:
        raise FeatureContractError("provenance P8-B ledger mismatch")
    expected_trace_paths = {
        "producer": contract["producer_path"],
        "contract": "config/modeling/p9_visual_feature_contract.json",
        "interpretable_schema": "config/modeling/p9_interpretable_feature_schema.json",
        "semantic_prompt_bank": "config/modeling/p9_semantic_prompt_bank.json",
    }
    for field, expected_path in expected_trace_paths.items():
        trace = provenance.get(field)
        if not isinstance(trace, Mapping) or trace.get("path") != expected_path:
            raise FeatureContractError(f"P9 {field} provenance path mismatch")
        if not isinstance(trace.get("git_blob_sha256"), str) or len(trace["git_blob_sha256"]) != 64:
            raise FeatureContractError(f"P9 {field} provenance SHA missing")
    if verify_git_traces:
        formal_commit = str(provenance.get("formal_run_git_commit") or "")
        if not formal_commit:
            raise FeatureContractError("P9 formal commit missing")
        for field in expected_trace_paths:
            trace = provenance[field]
            if git_blob_sha(formal_commit, trace["path"]) != trace["git_blob_sha256"]:
                raise FeatureContractError(f"P9 {field} formal Git blob mismatch")
            if git_blob_sha(git_head(), trace["path"]) != trace["git_blob_sha256"]:
                raise FeatureContractError(f"P9 {field} current Git blob mismatch")
    environment = provenance.get("environment")
    if not isinstance(environment, Mapping):
        raise FeatureContractError("P9 environment provenance missing")
    _assert_exact(dict(environment), _runtime_environment(contract), "environment provenance")


def _build_provenance(output_sha256: Mapping[str, str], upstream_verification: Mapping[str, Any]) -> dict[str, Any]:
    contract = read_json(CONTRACT_PATH)
    commit = git_head()
    return {
        "contract_version": "p9_v1.0",
        "baseline_main_commit": contract["baseline_main_commit"],
        "formal_run_git_commit": commit,
        "upstream_verification": dict(upstream_verification),
        "p7_d_final_manifest": {"path": contract["upstream"]["p7_d_final_manifest_path"], "sha256": contract["upstream"]["p7_d_final_manifest_sha256"]},
        "p8_b_formal_sha256": dict(contract["upstream"]["p8_b_formal_sha256"]),
        "model": contract["model"],
        "environment": _runtime_environment(contract),
        "producer": {"path": contract["producer_path"], "git_blob_sha256": git_blob_sha(commit, contract["producer_path"])},
        "contract": {"path": "config/modeling/p9_visual_feature_contract.json", "git_blob_sha256": git_blob_sha(commit, CONTRACT_PATH.relative_to(ROOT))},
        "interpretable_schema": {"path": "config/modeling/p9_interpretable_feature_schema.json", "git_blob_sha256": git_blob_sha(commit, SCHEMA_PATH.relative_to(ROOT))},
        "semantic_prompt_bank": {"path": "config/modeling/p9_semantic_prompt_bank.json", "git_blob_sha256": git_blob_sha(commit, PROMPT_PATH.relative_to(ROOT))},
        "formal_output_sha256": dict(output_sha256),
        "labels_read": False,
        "review_text_read": False,
        "G_read": False,
        "split_read": False,
        "model_fitted": False,
        "pca": False,
        "feature_selection": False,
        "performance_inspected": False,
    }


def prepare() -> dict[str, Any]:
    if FORMAL_DIR.exists() and any(FORMAL_DIR.iterdir()):
        raise FeatureContractError("P9 formal output directory is non-empty; refusing overwrite")
    contract = validate_contract()
    validate_schema()
    validate_prompt_bank()
    if git_head() == "":
        raise FeatureContractError("cannot resolve current Git commit")
    _check_upstream_sha(contract)
    upstream = _upstream_verification()
    source_rows = build_source_rows(P7D_MANIFEST)
    inventory = build_unique_image_inventory(source_rows)
    image_embeddings, semantic_scores = _extract_model_features(inventory, int(contract["model"]["inference"]["batch_size"]))
    feature_rows = _feature_rows(inventory, semantic_scores)
    product_rows = build_product_feature_map(source_rows, inventory)
    validate_embeddings(image_embeddings)
    quality = build_quality_audit(source_rows, inventory, image_embeddings, feature_rows, product_rows)
    field_source = tuple(source_rows[0])
    field_inventory = tuple(inventory[0])
    field_features = tuple(feature_rows[0])
    field_products = tuple(product_rows[0])
    _write_bytes(FORMAL_DIR / FORMAL_FILES[0], _csv_bytes(source_rows, field_source))
    _write_bytes(FORMAL_DIR / FORMAL_FILES[1], _csv_bytes(inventory, field_inventory))
    np.save(FORMAL_DIR / FORMAL_FILES[2], image_embeddings, allow_pickle=False)
    _write_bytes(FORMAL_DIR / FORMAL_FILES[3], _csv_bytes(feature_rows, field_features))
    _write_bytes(FORMAL_DIR / FORMAL_FILES[4], _csv_bytes(product_rows, field_products))
    _write_json(FORMAL_DIR / FORMAL_FILES[5], quality)
    output_sha = {name: sha256_file(FORMAL_DIR / name) for name in FORMAL_FILES[:6]}
    summary = build_summary(source_rows, inventory, image_embeddings, product_rows, output_sha)
    _write_json(FORMAL_DIR / FORMAL_FILES[6], summary)
    output_sha[FORMAL_FILES[6]] = sha256_file(FORMAL_DIR / FORMAL_FILES[6])
    provenance = _build_provenance(output_sha, upstream)
    _write_json(FORMAL_DIR / FORMAL_FILES[7], provenance)
    return {"verification": "PASS", "formal_files": list(FORMAL_FILES), "unique_images": len(inventory), "embedding_shape": list(image_embeddings.shape), "zero_network_calls_after_checkpoint_freeze": True}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _expected_output_bytes(source_rows: Sequence[Mapping[str, str]], inventory: Sequence[Mapping[str, str]], product_rows: Sequence[Mapping[str, str]]) -> dict[str, bytes]:
    return {
        FORMAL_FILES[0]: _csv_bytes(source_rows, tuple(source_rows[0])),
        FORMAL_FILES[1]: _csv_bytes(inventory, tuple(inventory[0])),
        FORMAL_FILES[4]: _csv_bytes(product_rows, tuple(product_rows[0])),
    }


def _validate_formal_files() -> None:
    if not FORMAL_DIR.exists() or {path.name for path in FORMAL_DIR.iterdir()} != set(FORMAL_FILES):
        raise FeatureContractError("P9 formal outputs are incomplete or contain unexpected files")


def _validate_formal_content(contract: Mapping[str, Any]) -> dict[str, Any]:
    _validate_formal_files()
    _check_upstream_sha(contract)
    source_rows = build_source_rows(P7D_MANIFEST)
    inventory = build_unique_image_inventory(source_rows)
    product_rows = build_product_feature_map(source_rows, inventory)
    expected = _expected_output_bytes(source_rows, inventory, product_rows)
    for name, data in expected.items():
        if (FORMAL_DIR / name).read_bytes() != data:
            raise FeatureContractError(f"P9 formal reconstruction mismatch: {name}")
    embeddings = np.load(FORMAL_DIR / FORMAL_FILES[2], allow_pickle=False)
    validate_embeddings(embeddings)
    feature_rows = _read_csv(FORMAL_DIR / FORMAL_FILES[3])
    semantic_names = [item["name"] for item in read_json(PROMPT_PATH)["attributes"]]
    expected_feature_fields = ("feature_row_index", "image_sha256", *CLASSICAL_FEATURES, *semantic_names)
    if len(feature_rows) != len(inventory) or tuple(feature_rows[0]) != expected_feature_fields:
        raise FeatureContractError("P9 interpretable feature schema mismatch")
    if any(row["feature_row_index"] != inv["feature_row_index"] or row["image_sha256"] != inv["image_sha256"] for row, inv in zip(feature_rows, inventory)):
        raise FeatureContractError("P9 feature row order mismatch")
    values = _float_values(feature_rows, (*CLASSICAL_FEATURES, *semantic_names))
    if not np.isfinite(values).all() or np.any(values[:, len(CLASSICAL_FEATURES) :] < -1.00001) or np.any(values[:, len(CLASSICAL_FEATURES) :] > 1.00001):
        raise FeatureContractError("P9 interpretable features are nonfinite or outside semantic cosine range")
    quality = read_json(FORMAL_DIR / FORMAL_FILES[5])
    expected_quality = build_quality_audit(source_rows, inventory, embeddings, feature_rows, product_rows)
    if quality != expected_quality:
        raise FeatureContractError("P9 quality audit reconstruction mismatch")
    output_sha = {name: sha256_file(FORMAL_DIR / name) for name in FORMAL_FILES[:6]}
    summary = read_json(FORMAL_DIR / FORMAL_FILES[6])
    expected_summary = build_summary(source_rows, inventory, embeddings, product_rows, output_sha)
    if summary != expected_summary:
        raise FeatureContractError("P9 summary reconstruction mismatch")
    provenance = read_json(FORMAL_DIR / FORMAL_FILES[7])
    validate_provenance(provenance, contract, verify_git_traces=True)
    if provenance.get("baseline_main_commit") != contract["baseline_main_commit"]:
        raise FeatureContractError("P9 baseline provenance mismatch")
    if provenance.get("p7_d_final_manifest", {}).get("sha256") != contract["upstream"]["p7_d_final_manifest_sha256"]:
        raise FeatureContractError("P9 P7-D provenance mismatch")
    if provenance.get("p8_b_formal_sha256") != contract["upstream"]["p8_b_formal_sha256"]:
        raise FeatureContractError("P9 P8-B ledger mismatch")
    if sha256_file(CHECKPOINT_PATH) != contract["model"]["checkpoint_sha256"]:
        raise FeatureContractError("P9 checkpoint SHA mismatch")
    formal_commit = str(provenance.get("formal_run_git_commit") or "")
    if not formal_commit:
        raise FeatureContractError("P9 formal commit missing")
    for field in ("producer", "contract", "interpretable_schema", "semantic_prompt_bank"):
        trace = provenance.get(field, {})
        if git_blob_sha(formal_commit, trace.get("path", "")) != trace.get("git_blob_sha256"):
            raise FeatureContractError(f"P9 {field} formal Git blob mismatch")
        if git_blob_sha(git_head(), trace.get("path", "")) != trace.get("git_blob_sha256"):
            raise FeatureContractError(f"P9 {field} current Git blob mismatch")
    if provenance.get("formal_output_sha256") != {**output_sha, FORMAL_FILES[6]: sha256_file(FORMAL_DIR / FORMAL_FILES[6])}:
        raise FeatureContractError("P9 formal output SHA ledger mismatch")
    tracked = subprocess.check_output(["git", "ls-files", "--", "data/processed/modeling_features_p9_5180"], cwd=ROOT, text=True).strip()
    if tracked:
        raise FeatureContractError("P9 formal data must remain gitignored")
    return {"verification": "PASS", "unique_images": len(inventory), "embedding_shape": list(embeddings.shape), "zero_writes": True}


def verify_existing() -> dict[str, Any]:
    contract = validate_contract()
    validate_schema()
    validate_prompt_bank()
    upstream = _upstream_verification()
    result = _validate_formal_content(contract)
    result["upstream_verification"] = upstream
    result["zero_network_calls"] = True
    result["zero_model_calls"] = True
    return result


def verify_recompute() -> dict[str, Any]:
    validate_contract()
    validate_schema()
    validate_prompt_bank()
    existing = verify_existing()
    contract = validate_contract()
    source_rows = build_source_rows(P7D_MANIFEST)
    inventory = build_unique_image_inventory(source_rows)
    embeddings, semantic_scores = _extract_model_features(inventory, int(contract["model"]["inference"]["batch_size"]))
    formal_embeddings = np.load(FORMAL_DIR / FORMAL_FILES[2], allow_pickle=False)
    embedding_max_abs_diff = float(np.max(np.abs(embeddings - formal_embeddings)))
    feature_rows = _feature_rows(inventory, semantic_scores)
    formal_feature_rows = _read_csv(FORMAL_DIR / FORMAL_FILES[3])
    names = (*CLASSICAL_FEATURES, *[item["name"] for item in read_json(PROMPT_PATH)["attributes"]])
    recomputed = _float_values(feature_rows, names)
    formal_values = _float_values(formal_feature_rows, names)
    classical_diff = float(np.max(np.abs(recomputed[:, :20] - formal_values[:, :20])))
    semantic_diff = float(np.max(np.abs(recomputed[:, 20:] - formal_values[:, 20:])))
    if embedding_max_abs_diff > 1e-6 or semantic_diff > 1e-6 or classical_diff != 0.0:
        raise FeatureContractError(f"P9 recompute instability: embedding={embedding_max_abs_diff}, classical={classical_diff}, semantic={semantic_diff}")
    return {**existing, "verification": "PASS", "recompute": "PASS", "embedding_max_abs_diff": embedding_max_abs_diff, "classical_max_abs_diff": classical_diff, "semantic_max_abs_diff": semantic_diff, "zero_writes": True, "zero_network_calls": True}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare",), nargs="?")
    parser.add_argument("--verify-existing", action="store_true")
    parser.add_argument("--verify-recompute", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        if args.verify_existing:
            print(json.dumps(verify_existing(), ensure_ascii=False, indent=2, sort_keys=True))
            return 0
        if args.verify_recompute:
            print(json.dumps(verify_recompute(), ensure_ascii=False, indent=2, sort_keys=True))
            return 0
        if args.command == "prepare":
            print(json.dumps(prepare(), ensure_ascii=False, indent=2, sort_keys=True))
            return 0
        raise FeatureContractError("choose prepare, --verify-existing, or --verify-recompute")
    except FeatureContractError as exc:
        print(f"P9 ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
