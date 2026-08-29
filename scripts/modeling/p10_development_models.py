"""P10 development-only predictive and interpretable model freeze."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import io
import json
import math
import os
import platform
import subprocess
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

# These must be set before NumPy/SciPy/scikit-learn import for deterministic CPU runs.
for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[2]
CONTRACT_PATH = ROOT / "config/modeling/p10_development_model_contract.json"
P11_CONTRACT_PATH = ROOT / "config/modeling/p11_locked_test_evaluation_contract.json"
FORMAL_DIR = ROOT / "data/processed/modeling_p10_5180/p10_development_model_freeze"
P8_DIR = ROOT / "data/processed/modeling_readiness_p8_5180/p8_b_modeling_ready_split"
P9_DIR = ROOT / "data/processed/modeling_features_p9_5180/p9_visual_feature_freeze"
P8_MANIFEST = P8_DIR / "01_modeling_ready_manifest.csv"
P9_MAP = P9_DIR / "05_product_feature_map.csv"
P9_INVENTORY = P9_DIR / "02_unique_image_inventory.csv"
P9_EMBEDDINGS = P9_DIR / "03_openclip_image_embeddings.npy"
P9_FEATURES = P9_DIR / "04_interpretable_image_features.csv"

FORMAL_FILES = (
    "01_development_modeling_manifest.csv",
    "02_cv_model_selection.csv",
    "03_development_oof_scores.csv",
    "04_final_model_specification.json",
    "05_final_model_parameters.npz",
    "06_interpretable_coefficient_stability.csv",
    "07_development_model_summary.json",
    "08_p10_provenance.json",
)

CLASSICAL_FEATURES = (
    "luminance_mean", "luminance_std", "luminance_p10", "luminance_p90",
    "saturation_mean", "saturation_std", "colorfulness_hs", "high_saturation_fraction",
    "near_white_fraction", "near_black_fraction", "warm_hue_fraction", "cool_hue_fraction",
    "hue_entropy", "grayscale_entropy", "edge_density", "edge_strength_mean",
    "lr_symmetry", "tb_symmetry", "center_edge_energy_fraction", "quadrant_edge_imbalance",
)
SEMANTIC_FEATURES = (
    "leaf_herb_illustration_score", "floral_illustration_score", "fruit_motif_score",
    "ingredient_photography_score", "character_mascot_score", "sparse_layout_score",
    "dense_ornament_score", "heritage_ornament_score", "geometric_layout_score",
    "kraft_craft_score", "ribbon_bow_box_cues_score", "pale_pastel_palette_score",
    "vivid_multicolor_palette_score", "typography_dominant_score", "image_dominant_score",
    "transparent_window_score",
)
INTERPRETABLE_FEATURES = (*CLASSICAL_FEATURES, *SEMANTIC_FEATURES)
OUTCOMES = (
    ("has_any_outer_imagery_observed", "primary"),
    ("general_visual_appeal_observed_positive_core", "secondary_confirmatory"),
    ("cute_friendly_observed_positive_core", "secondary_confirmatory"),
)
TRACKS = ("openclip_512_logistic", "interpretable_36_logistic")
C_GRID = (0.01, 0.1, 1.0, 10.0, 100.0)
FOLDS = (0, 1, 2, 3, 4)


class P10ContractError(RuntimeError):
    """Raised when a frozen P10 contract or artifact is violated."""


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest().upper()


def sha256_file(path: Path) -> str:
    if not path.is_file():
        raise P10ContractError(f"missing file for SHA: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _assert_exact(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise P10ContractError(f"{label} mismatch: {actual!r} != {expected!r}")


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise P10ContractError(f"missing CSV: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _csv_bytes(rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(fields), extrasaction="raise", lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({field: row[field] for field in fields})
    return buffer.getvalue().encode("utf-8")


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def _git_head() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise P10ContractError("cannot resolve Git HEAD") from exc


def git_blob_sha(commit: str, path: str | Path) -> str:
    normalized = str(path).replace("\\", "/")
    try:
        return subprocess.check_output(["git", "rev-parse", f"{commit}:{normalized}"], cwd=ROOT, text=True).strip().upper()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise P10ContractError(f"cannot resolve Git blob {commit}:{normalized}") from exc


def runtime_environment(contract: Mapping[str, Any] | None = None) -> dict[str, Any]:
    versions = {
        "python_version": sys.version.split()[0],
        "numpy_version": np.__version__,
        "scipy_version": importlib.metadata.version("scipy"),
        "pandas_version": importlib.metadata.version("pandas"),
        "scikit_learn_version": importlib.metadata.version("scikit-learn"),
        "joblib_version": importlib.metadata.version("joblib"),
        "threadpoolctl_version": importlib.metadata.version("threadpoolctl"),
        "platform": platform.platform(),
        "device": "cpu",
        "network_calls": 0,
        "thread_env": {name: os.environ.get(name) for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")},
    }
    if contract is not None:
        _assert_exact(versions, contract["runtime"], "runtime environment")
    return versions


def validate_p11_contract(contract: Mapping[str, Any] | None = None) -> dict[str, Any]:
    value = dict(contract or read_json(P11_CONTRACT_PATH))
    expected_keys = {"contract_version", "stage", "execution_status", "baseline_p10_contract", "locked_test_source", "locked_test_partition", "group_key", "group_rule", "preserve_product_row_metrics", "metrics", "uncertainty", "r2_sensitivity", "firewall", "p8_a_binding", "robustness_analyses"}
    _assert_exact(set(value), expected_keys, "P11 contract top-level fields")
    _assert_exact(value.get("contract_version"), "p11_v1.0", "P11 contract version")
    _assert_exact(value.get("execution_status"), "contract_only_not_executed", "P11 execution status")
    _assert_exact(value.get("baseline_p10_contract"), "config/modeling/p10_development_model_contract.json", "P11 baseline P10 contract")
    _assert_exact(value.get("locked_test_source"), "data/processed/modeling_readiness_p8_5180/p8_b_modeling_ready_split/01_modeling_ready_manifest.csv", "P11 locked source")
    _assert_exact(value.get("locked_test_partition"), "locked_test", "P11 locked partition")
    _assert_exact(value.get("group_key"), "primary_response_sha256", "P11 group key")
    _assert_exact(value.get("group_rule"), "one primary_response_sha256 group is one evaluation unit; group observed label is max product observed label", "P11 group rule")
    _assert_exact(value.get("preserve_product_row_metrics"), True, "P11 product-row preservation")
    _assert_exact(value.get("metrics"), {"primary": "average_precision", "point_estimates": ["average_precision", "roc_auc", "recall_at_top5", "recall_at_top10", "recall_at_top20", "lift_at_top5", "lift_at_top10", "lift_at_top20"], "no_threshold_accuracy": True, "no_threshold_f1": True}, "P11 metrics")
    _assert_exact(value.get("uncertainty"), {"unit": "primary_response_sha256 group", "method": "cluster bootstrap", "iterations": 5000, "random_seed": 20260818, "confidence_level": 0.95, "percentile_metrics": ["average_precision", "roc_auc", "recall_at_top10", "lift_at_top10"], "other_metrics": "point estimates only"}, "P11 uncertainty")
    _assert_exact(value.get("r2_sensitivity"), {"comparison": "frozen final primary model versus sensitivity feature source", "same_frozen_split_grouping": True, "available_products_only": True, "retrain_sensitivity": False, "no_retrain": True}, "P11 R2 sensitivity")
    _assert_exact(value.get("p8_a_binding"), {"contract_path": "config/modeling/p8_a_analysis_contract.json", "contract_sha256": "A22948361EC2C3FAD9F0046D9B6200623222E3F9EC2617951AF7A4E62B7D75C8", "contract_git_blob_sha256": "FE1DA301962FB08A4D6962B744592DDF1FAD4480"}, "P8-A contract binding")
    _assert_exact(value.get("robustness_analyses"), {
        "R1": {"name": "known-QA-exception exclusion", "definition": "exclude only the 83 known P7-C QA exceptions", "promotion_or_relabeling": False},
        "R2": {"name": "primary-versus-frozen-sensitivity-exposure", "definition": "compare primary exposure with frozen sensitivity exposure for available products without changing split grouping", "same_frozen_split_grouping": True, "retrain_sensitivity": False, "promotion_or_relabeling": False},
        "R3": {"name": "G exposure-quality sensitivity", "definition": "use the complete frozen G grid/rules or continuous diagnostic; never select a performance-best exposure threshold", "performance_best_threshold_selection": False, "promotion_or_relabeling": False},
        "R4": {"name": "core-versus-pilot-robust label-definition robustness", "definition": "compare frozen core with pilot/robust definitions without promotion", "promotion_or_relabeling": False},
        "R5": {"name": "known-placeholder-image-exception subgroup", "definition": "descriptive score behavior for known placeholder/image-exception subgroup", "inferential_claim": False},
    }, "P11 R1-R5 robustness definitions")
    _assert_exact(value.get("firewall"), {
        "execution_in_p10": False,
        "locked_rows_read_for_modeling": False,
        "locked_rows_read_for_model_choice": False,
        "locked_predictions_in_p10": False,
        "locked_metrics_in_p10": False,
        "locked_source_read_in_p10": False,
        "locked_partition_in_development_table": False,
        "locked_rows_to_scaler_fit": False,
        "locked_rows_to_model_fit": False,
        "locked_rows_to_metrics": False,
        "locked_rows_to_oof": False,
        "locked_rows_to_scoring_in_p10": False,
        "p11_execution_in_p10": False,
    }, "P11 firewall")
    return value


def validate_contract(contract: Mapping[str, Any] | None = None) -> dict[str, Any]:
    value = dict(contract or read_json(CONTRACT_PATH))
    expected_keys = {"contract_version", "stage", "baseline_main_commit", "producer_path", "formal_output_directory", "formal_output_files", "upstream", "development_population", "pu_semantics", "outcomes", "confirmatory_family", "tracks", "runtime", "logistic", "selection", "cv_audit", "metrics", "oof", "final_refit", "parameter_serialization", "locked_test_firewall", "p11_contract_path", "provenance_policy", "output_policy"}
    _assert_exact(set(value), expected_keys, "P10 contract top-level fields")
    _assert_exact(value.get("contract_version"), "p10_v1.0", "P10 contract version")
    _assert_exact(value.get("baseline_main_commit"), "d0cf90ca6a2bb3836bfb9c368d37efb124ae230b", "P10 baseline main commit")
    _assert_exact(tuple(value.get("formal_output_files", ())), FORMAL_FILES, "P10 formal output files")
    _assert_exact(value.get("development_population", {}).get("development_rows"), 4143, "development row count")
    _assert_exact(value.get("development_population", {}).get("locked_test_rows"), 1036, "locked-test aggregate row count")
    expected_outcomes = [
        {"name": "has_any_outer_imagery_observed", "role": "primary", "p8_source_field": "has_any_outer_imagery_observed", "global_positive_count": 232, "development_positive_count": 186, "locked_test_positive_count": 46},
        {"name": "general_visual_appeal_observed_positive_core", "role": "secondary_confirmatory", "p8_source_field": "general_visual_appeal_observed_positive_core"},
        {"name": "cute_friendly_observed_positive_core", "role": "secondary_confirmatory", "p8_source_field": "cute_friendly_observed_positive_core"},
    ]
    _assert_exact(value.get("outcomes"), expected_outcomes, "outcome definitions")
    _assert_exact(value.get("confirmatory_family"), {"size": 2, "bh_q": 0.05, "in_p10": False, "reserved_for": "P11/P12"}, "confirmatory family")
    _assert_exact(value.get("tracks"), [
        {"name": "openclip_512_logistic", "feature_source": "P9 512-dimensional frozen OpenCLIP embedding", "feature_count": 512, "feature_order": "P9 03 row order resolved by primary_feature_row_index", "scaler": "StandardScaler", "model": "LogisticRegression"},
        {"name": "interpretable_36_logistic", "feature_source": "P9 20 classical + 16 semantic features", "feature_count": 36, "feature_order": "P9 04 exact header order", "scaler": "StandardScaler", "model": "LogisticRegression", "all_features_retained": True},
    ], "track definitions")
    _assert_exact(value.get("pu_semantics"), {
        "label_one": "observed spontaneous affective-imagery mention / observed-positive",
        "label_zero": "PU-unlabeled / unobserved; never confirmed negative",
        "allowed_terms": ["observed-positive score", "observed-positive propensity", "observed-positive discrimination"],
        "predict_proba_one_warning": "model-estimated probability under observational labeling, not latent true perception probability",
        "forbidden_unqualified_terms": ["negative", "true negative", "absence", "not perceived"],
    }, "PU semantics")
    _assert_exact(tuple(value.get("selection", {}).get("c_grid", ())), C_GRID, "C grid")
    _assert_exact(tuple(value.get("selection", {}).get("cv_folds", ())), FOLDS, "CV folds")
    _assert_exact(value.get("selection"), {
        "c_grid": list(C_GRID), "cv_folds": list(FOLDS),
        "selection_metric": "unweighted arithmetic mean of five validation average precision values",
        "tie_break": "exact equal mean average precision chooses smaller C",
        "fits": {"cv": 150, "final": 6, "total": 156},
        "no_model_family_search": True, "no_threshold_search": True, "no_sampling": True, "no_pca": True, "no_pu_native_method": True,
        "ledger_rows": 150,
        "ledger_fields": ["outcome", "outcome_role", "track", "C", "fold", "fold_average_precision", "fold_auroc", "train_rows", "validation_rows", "train_observed_positives", "validation_observed_positives", "n_iter", "convergence_status", "selected"],
        "convergence_warning_is_fail": True,
        "n_iter_strictly_less_than_max_iter": True,
    }, "selection policy")
    _assert_exact(value.get("cv_audit"), {
        "ledger_rows": 150, "cv_fit_count": 150, "final_fit_count": 6, "total_fit_count": 156,
        "convergence_warning_is_fail": True, "max_iter": 5000,
        "n_iter_rule": "strictly less than max_iter for every CV and final fit",
        "required_status": "PASS",
        "required_fields": ["outcome", "outcome_role", "track", "C", "fold", "fold_average_precision", "fold_auroc", "train_rows", "validation_rows", "train_observed_positives", "validation_observed_positives", "n_iter", "convergence_status", "selected"],
    }, "CV audit policy")
    _assert_exact(value.get("logistic"), {"class": "sklearn.linear_model.LogisticRegression", "solver": "lbfgs", "l1_ratio": 0.0, "fit_intercept": True, "class_weight": None, "max_iter": 5000, "tol": 1e-08, "warm_start": False, "random_state": 20260818, "penalty_explicitly_set": False, "n_jobs_explicitly_set": False, "sample_weight": False}, "LogisticRegression configuration")
    _assert_exact(value.get("metrics", {}).get("primary_selection"), "average_precision", "primary selection metric")
    _assert_exact(value.get("metrics", {}).get("top_k_rule"), "ceil(n * fraction), descending score, exact score ties parent_asin ascending", "top-k rule")
    _assert_exact(value.get("oof", {}).get("rows"), 24858, "OOF row count")
    _assert_exact(value.get("final_refit"), {"fit_rows": 4143, "fit_scope": "all development rows only", "final_fit_count": 6, "sensitivity_features": "not fitted and not used for model choice", "convergence_warning_is_fail": True, "n_iter_strictly_less_than_max_iter": True, "required_convergence_status": "PASS"}, "final refit policy")
    _assert_exact(value.get("locked_test_firewall"), {
        "locked_rows_allowed_in_modeling_table": False, "locked_performance_allowed": False, "locked_predictions_allowed": False, "locked_scores_allowed": False, "locked_distribution_allowed": False, "locked_model_choice_allowed": False, "locked_threshold_allowed": False, "locked_features_allowed": False, "aggregate_counts_only": True,
        "locked_rows_to_development_manifest": False, "locked_rows_to_scaler_fit": False, "locked_rows_to_model_fit": False, "locked_rows_to_metrics": False, "locked_rows_to_oof": False, "locked_rows_to_parameter_outputs": False, "locked_rows_to_scoring": False,
    }, "P10 locked-test firewall")
    _assert_exact(value.get("provenance_policy"), {
        "provenance_self_sha_forbidden": True,
        "binds_formal_outputs": ["01_development_modeling_manifest.csv", "02_cv_model_selection.csv", "03_development_oof_scores.csv", "04_final_model_specification.json", "05_final_model_parameters.npz", "06_interpretable_coefficient_stability.csv", "07_development_model_summary.json"],
        "binds_contract_blobs": ["config/modeling/p10_development_model_contract.json", "config/modeling/p11_locked_test_evaluation_contract.json"],
        "binds_producer_path": True, "binds_baseline_main_commit": True, "external_provenance_sha_record": "tracked report/summary/PR body",
        "formal_output_sha_ledger_excludes_self": True, "provenance_self_sha256_key": "provenance_self_sha256", "convergence_audit_bound": True,
    }, "P10 provenance policy")
    _assert_exact(value.get("output_policy"), {
        "prepare_refuses_nonempty_formal_directory": True, "verify_existing_writes": 0, "verify_recompute_writes": 0, "verify_recompute_network_calls": 0, "formal_data_git_tracked": False, "p11_executed": False, "p11_predictions_or_performance_started": False,
        "prepare_network_calls": 0, "verify_existing_model_fits": 0, "verify_recompute_model_fits": 156, "formal_output_ledger_exact": True,
    }, "P10 output policy")
    _assert_exact(value.get("parameter_serialization", {}).get("sklearn_probability_max_abs_tolerance"), 1e-12, "parameter reconstruction tolerance")
    _assert_exact(value.get("parameter_serialization", {}).get("recompute_score_max_abs_tolerance"), 1e-10, "recompute score tolerance")
    _assert_exact(value.get("parameter_serialization", {}).get("recompute_parameter_max_abs_tolerance"), 1e-10, "recompute parameter tolerance")
    runtime_environment(value)
    return value


def validate_pu_text(text: str) -> None:
    lowered = text.lower()
    for forbidden in ("true negative", "absence", "not perceived"):
        if forbidden in lowered:
            raise P10ContractError(f"forbidden unqualified PU wording: {forbidden}")


@dataclass(frozen=True)
class DevelopmentTable:
    parent_asin: tuple[str, ...]
    input_order: np.ndarray
    group_key: tuple[str, ...]
    primary_response_sha256: tuple[str, ...]
    development_fold: np.ndarray
    outcomes: dict[str, np.ndarray]
    split_partition: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        n = len(self.parent_asin)
        if n != 4143 or len(set(self.parent_asin)) != n:
            raise P10ContractError("development table must contain 4143 unique parent_asin rows")
        if len(set(self.group_key)) != n or len(self.primary_response_sha256) != n:
            raise P10ContractError("development table has duplicate or missing primary groups")
        partitions = self.split_partition or tuple("development" for _ in range(n))
        if len(partitions) != n:
            raise P10ContractError("development split partition length mismatch")
        _assert_development_partition(partitions)
        if tuple(sorted(set(int(x) for x in self.development_fold))) != FOLDS:
            raise P10ContractError("development folds are not exactly 0..4")
        if any(len(values) != n for values in self.outcomes.values()):
            raise P10ContractError("development outcome length mismatch")
        expected_counts = {0: 837, 1: 828, 2: 828, 3: 828, 4: 822}
        actual_counts = {fold: int(np.sum(self.development_fold == fold)) for fold in FOLDS}
        _assert_exact(actual_counts, expected_counts, "development fold counts")
        if int(np.sum(self.outcomes[OUTCOMES[0][0]])) != 186:
            raise P10ContractError("development primary observed-positive count mismatch")


@dataclass(frozen=True)
class JoinedDevelopment:
    table: DevelopmentTable
    embedding_features: np.ndarray
    interpretable_features: np.ndarray
    feature_row_index: np.ndarray

    def __post_init__(self) -> None:
        n = len(self.table.parent_asin)
        if self.embedding_features.shape != (n, 512):
            raise P10ContractError(f"development embedding shape mismatch: {self.embedding_features.shape}")
        if self.interpretable_features.shape != (n, 36):
            raise P10ContractError(f"development interpretable shape mismatch: {self.interpretable_features.shape}")
        _assert_development_partition(self.table.split_partition or tuple("development" for _ in range(n)))
        if self.feature_row_index.shape != (n,) or len(set(int(x) for x in self.feature_row_index)) != n:
            raise P10ContractError("development feature row index mismatch")
        if not np.isfinite(self.embedding_features).all() or not np.isfinite(self.interpretable_features).all():
            raise P10ContractError("development feature matrix contains non-finite values")


def build_development_table(contract: Mapping[str, Any] | None = None) -> DevelopmentTable:
    contract = contract or validate_contract()
    rows = _read_csv(P8_MANIFEST)
    development: list[dict[str, Any]] = []
    main_count = 0
    locked_count = 0
    for raw in rows:
        if raw.get("main_analysis_included") != "true":
            continue
        main_count += 1
        partition = raw.get("split_partition")
        if partition == "locked_test":
            locked_count += 1
            continue
        if partition != "development":
            raise P10ContractError(f"unexpected main split partition: {partition!r}")
        development.append({
            "parent_asin": raw["parent_asin"],
            "input_order": int(raw["input_order"]),
            "split_group_key": raw["split_group_key"],
            "primary_response_sha256": raw["primary_response_sha256"],
            "development_fold": int(raw["development_fold"]),
            "has_any_outer_imagery_observed": int(raw["has_any_outer_imagery_observed"]),
            "general_visual_appeal_observed_positive_core": int(raw["general_visual_appeal_observed_positive_core"]),
            "cute_friendly_observed_positive_core": int(raw["cute_friendly_observed_positive_core"]),
        })
    _assert_exact(main_count, int(contract["development_population"]["main_population_rows"]), "P8-B main population count")
    _assert_exact(locked_count, int(contract["development_population"]["locked_test_rows"]), "P8-B locked aggregate count")
    development.sort(key=lambda item: item["input_order"])
    table = DevelopmentTable(
        parent_asin=tuple(item["parent_asin"] for item in development),
        input_order=np.asarray([item["input_order"] for item in development], dtype=np.int64),
        group_key=tuple(item["split_group_key"] for item in development),
        primary_response_sha256=tuple(item["primary_response_sha256"] for item in development),
        development_fold=np.asarray([item["development_fold"] for item in development], dtype=np.int64),
        outcomes={name: np.asarray([item[name] for item in development], dtype=np.int64) for name, _ in OUTCOMES},
        split_partition=tuple("development" for _ in development),
    )
    return table


def join_development_features(table: DevelopmentTable, contract: Mapping[str, Any] | None = None) -> JoinedDevelopment:
    contract = contract or validate_contract()
    product_rows = _read_csv(P9_MAP)
    if len(product_rows) != 5180 or sum(row.get("main_analysis_included") == "1" for row in product_rows) != 5179:
        raise P10ContractError("P9 product feature map row count or main-population marker mismatch")
    product_by_asin = {row["parent_asin"]: row for row in product_rows}
    if len(product_by_asin) != len(product_rows):
        raise P10ContractError("P9 product feature map has duplicate parent_asin")
    inventory_rows = _read_csv(P9_INVENTORY)
    inventory_by_index = {int(row["feature_row_index"]): row for row in inventory_rows}
    embeddings = np.load(P9_EMBEDDINGS, allow_pickle=False)
    if embeddings.ndim != 2 or embeddings.shape[1] != 512 or embeddings.shape[0] != len(inventory_rows):
        raise P10ContractError("P9 embedding inventory shape mismatch")
    feature_rows = _read_csv(P9_FEATURES)
    expected_header = ("feature_row_index", "image_sha256", *INTERPRETABLE_FEATURES)
    if not feature_rows or tuple(feature_rows[0].keys()) != expected_header or len(feature_rows) != len(inventory_rows):
        raise P10ContractError("P9 interpretable feature header or row count mismatch")
    feature_by_index = {int(row["feature_row_index"]): row for row in feature_rows}
    if len(feature_by_index) != len(feature_rows):
        raise P10ContractError("P9 interpretable features have duplicate feature_row_index")
    embedding_values: list[np.ndarray] = []
    interpretable_values: list[np.ndarray] = []
    indices: list[int] = []
    for asin, response_sha in zip(table.parent_asin, table.primary_response_sha256):
        product = product_by_asin.get(asin)
        if product is None or product.get("primary_feature_available") != "1":
            raise P10ContractError(f"missing P9 primary feature map row: {asin}")
        if product["primary_response_sha256"] != response_sha:
            raise P10ContractError(f"P8/P9 primary response SHA mismatch: {asin}")
        index = int(product["primary_feature_row_index"])
        inventory = inventory_by_index.get(index)
        feature = feature_by_index.get(index)
        if inventory is None or feature is None or inventory["image_sha256"] != response_sha or feature["image_sha256"] != response_sha:
            raise P10ContractError(f"P9 feature row binding mismatch: {asin}")
        indices.append(index)
        embedding_values.append(np.asarray(embeddings[index], dtype=np.float64))
        interpretable_values.append(np.asarray([float(feature[name]) for name in INTERPRETABLE_FEATURES], dtype=np.float64))
    return JoinedDevelopment(table, np.vstack(embedding_values), np.vstack(interpretable_values), np.asarray(indices, dtype=np.int64))


def _feature_order(track: str) -> tuple[str, ...]:
    if track == "openclip_512_logistic":
        return tuple(f"embedding_{index:03d}" for index in range(512))
    if track == "interpretable_36_logistic":
        return INTERPRETABLE_FEATURES
    raise P10ContractError(f"unknown track: {track}")


@dataclass(frozen=True)
class FitAudit:
    n_iter: int
    convergence_status: str
    convergence_warning_count: int


def _assert_development_partition(split_partition: str | Sequence[str] | None) -> None:
    if split_partition is None:
        return
    values = (split_partition,) if isinstance(split_partition, str) else tuple(split_partition)
    if any(str(value) != "development" for value in values):
        raise P10ContractError("locked_test or non-development rows cannot enter P10 modeling")


def validate_convergence_audit(n_iter: int, convergence_warning: bool, label: str = "fit") -> FitAudit:
    n_iter = int(n_iter)
    if convergence_warning:
        raise P10ContractError(f"{label} emitted sklearn ConvergenceWarning")
    if n_iter < 1 or n_iter >= 5000:
        raise P10ContractError(f"{label} n_iter must be strictly less than 5000: {n_iter}")
    return FitAudit(n_iter=n_iter, convergence_status="PASS", convergence_warning_count=0)


def _fit_model(X: np.ndarray, y: np.ndarray, C: float, *, split_partition: str | Sequence[str] | None = "development") -> tuple[StandardScaler, LogisticRegression, FitAudit]:
    _assert_development_partition(split_partition)
    if X.dtype != np.float64 or y.dtype != np.int64:
        raise P10ContractError("modeling arrays must be float64 features and int64 labels")
    if len(np.unique(y)) != 2:
        raise P10ContractError("each training fold must contain both PU label states")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X).astype(np.float64, copy=False)
    model = LogisticRegression(
        C=float(C), solver="lbfgs", l1_ratio=0.0, fit_intercept=True,
        class_weight=None, max_iter=5000, tol=1e-8, warm_start=False,
        random_state=20260818,
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(X_scaled, y)
    warning_count = sum(1 for item in caught if issubclass(item.category, ConvergenceWarning))
    audit = validate_convergence_audit(int(np.max(np.asarray(model.n_iter_))), warning_count > 0)
    return scaler, model, audit


def standardize_expected(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return StandardScaler-equivalent transformed values, mean, and scale."""
    values = np.asarray(X, dtype=np.float64)
    mean = values.mean(axis=0)
    variance = ((values - mean) ** 2).mean(axis=0)
    scale = np.sqrt(variance)
    scale[scale == 0.0] = 1.0
    return (values - mean) / scale, mean, scale


def score_from_frozen_parameters(raw_features: np.ndarray, scaler_mean: np.ndarray, scaler_scale: np.ndarray, coef: np.ndarray, intercept: float, *, split_partition: str | Sequence[str] | None = None) -> np.ndarray:
    _assert_development_partition(split_partition)
    X = np.asarray(raw_features, dtype=np.float64)
    mean = np.asarray(scaler_mean, dtype=np.float64)
    scale = np.asarray(scaler_scale, dtype=np.float64)
    weights = np.asarray(coef, dtype=np.float64).reshape(-1)
    if X.ndim != 2 or X.shape[1] != len(mean) or len(scale) != len(mean) or len(weights) != len(mean):
        raise P10ContractError("frozen parameter feature dimension mismatch")
    if np.any(scale <= 0.0) or not np.isfinite(X).all():
        raise P10ContractError("invalid frozen scaler or feature values")
    decision = float(intercept) + ((X - mean) / scale) @ weights
    return 1.0 / (1.0 + np.exp(-np.clip(decision, -745.0, 709.0)))


def select_c(candidates: Sequence[Mapping[str, Any]]) -> float:
    if not candidates:
        raise P10ContractError("C selection has no candidates")
    best = candidates[0]
    for candidate in candidates[1:]:
        candidate_mean = float(candidate["mean_average_precision"])
        best_mean = float(best["mean_average_precision"])
        candidate_c = float(candidate["C"])
        best_c = float(best["C"])
        if candidate_mean > best_mean or (candidate_mean == best_mean and candidate_c < best_c):
            best = candidate
    return float(best["C"])


def top_k_diagnostics(y: np.ndarray, score: np.ndarray, parent_asin: Sequence[str], *, split_partition: str | Sequence[str] | None = None) -> dict[str, float]:
    _assert_development_partition(split_partition)
    y = np.asarray(y, dtype=np.int64)
    score = np.asarray(score, dtype=np.float64)
    if len(y) != len(score) or len(y) != len(parent_asin):
        raise P10ContractError("top-k diagnostic length mismatch")
    order = sorted(range(len(y)), key=lambda index: (-float(score[index]), str(parent_asin[index])))
    total_positive = int(y.sum())
    result: dict[str, float] = {}
    for label, fraction in (("5", 0.05), ("10", 0.10), ("20", 0.20)):
        k = int(math.ceil(len(y) * fraction))
        positive_in_top = int(y[np.asarray(order[:k], dtype=np.int64)].sum())
        recall = positive_in_top / total_positive if total_positive else 0.0
        lift = (positive_in_top / k) / (total_positive / len(y)) if total_positive and k else 0.0
        result[f"recall_at_top{label}"] = float(recall)
        result[f"lift_at_top{label}"] = float(lift)
    return result


def fold_diagnostics(y: np.ndarray, score: np.ndarray, parent_asin: Sequence[str], *, split_partition: str | Sequence[str] | None = None) -> dict[str, float]:
    _assert_development_partition(split_partition)
    y = np.asarray(y, dtype=np.int64)
    score = np.asarray(score, dtype=np.float64)
    if len(y) != len(score):
        raise P10ContractError("fold diagnostic length mismatch")
    return {
        "average_precision": float(average_precision_score(y, score)),
        "roc_auc": float(roc_auc_score(y, score)),
        **top_k_diagnostics(y, score, parent_asin, split_partition=split_partition),
    }


def validate_oof_coverage(rows: Sequence[Mapping[str, Any]], table: DevelopmentTable, *, split_partition: str | Sequence[str] | None = None) -> None:
    _assert_development_partition(split_partition)
    expected = {(asin, outcome, track) for outcome, _ in OUTCOMES for track in TRACKS for asin in table.parent_asin}
    actual: set[tuple[str, str, str]] = set()
    parent_set = set(table.parent_asin)
    fold_by_parent = dict(zip(table.parent_asin, table.development_fold.tolist()))
    for row in rows:
        if str(row.get("split_partition", "development")) != "development":
            raise P10ContractError("locked_test row cannot enter OOF coverage")
        key = (str(row["parent_asin"]), str(row["outcome"]), str(row["track"]))
        if key in actual or key not in expected:
            raise P10ContractError("OOF coverage has duplicate, unknown, or locked row key")
        if key[0] not in parent_set or int(row["development_fold"]) != fold_by_parent[key[0]]:
            raise P10ContractError("OOF fold or parent binding mismatch")
        actual.add(key)
    _assert_exact(actual, expected, "OOF coverage")


def sign_agreement_count(final_value: float, fold_values: Sequence[float]) -> int:
    final_sign = 1 if final_value > 0 else -1 if final_value < 0 else 0
    if final_sign == 0:
        return 0
    return sum(1 for value in fold_values if (value > 0 and final_sign > 0) or (value < 0 and final_sign < 0))


def _model_key(outcome: str, track: str) -> str:
    return f"{outcome}__{track}"


def _c_key(C: float) -> str:
    return format(float(C), ".15g")


def run_development_modeling(joined: JoinedDevelopment) -> dict[str, Any]:
    _assert_development_partition(joined.table.split_partition or tuple("development" for _ in joined.table.parent_asin))
    table = joined.table
    X_by_track = {TRACKS[0]: joined.embedding_features, TRACKS[1]: joined.interpretable_features}
    selection_rows: list[dict[str, Any]] = []
    oof_rows: list[dict[str, Any]] = []
    stability_rows: list[dict[str, Any]] = []
    parameters: dict[str, dict[str, np.ndarray | float]] = {}
    model_specs: list[dict[str, Any]] = []
    final_fit_audits: list[dict[str, Any]] = []
    cv_store: dict[tuple[str, str, str, int], tuple[StandardScaler, LogisticRegression, np.ndarray]] = {}
    fit_count = 0
    for outcome, role in OUTCOMES:
        y = np.asarray(table.outcomes[outcome], dtype=np.int64)
        for track in TRACKS:
            X = np.asarray(X_by_track[track], dtype=np.float64)
            candidates: list[dict[str, Any]] = []
            for C in C_GRID:
                fold_aps: list[float] = []
                fit_rows_for_c: list[dict[str, Any]] = []
                for fold in FOLDS:
                    train_mask = table.development_fold != fold
                    val_mask = table.development_fold == fold
                    scaler, model, audit = _fit_model(X[train_mask], y[train_mask], C, split_partition="development")
                    fit_count += 1
                    score = model.predict_proba(scaler.transform(X[val_mask]).astype(np.float64, copy=False))[:, 1].astype(np.float64)
                    diagnostics = fold_diagnostics(y[val_mask], score, [table.parent_asin[index] for index in np.flatnonzero(val_mask)], split_partition="development")
                    fold_aps.append(float(diagnostics["average_precision"]))
                    fit_rows_for_c.append({
                        "outcome": outcome,
                        "outcome_role": role,
                        "track": track,
                        "C": float(C),
                        "fold": int(fold),
                        "fold_average_precision": float(diagnostics["average_precision"]),
                        "fold_auroc": float(diagnostics["roc_auc"]),
                        "train_rows": int(train_mask.sum()),
                        "validation_rows": int(val_mask.sum()),
                        "train_observed_positives": int(y[train_mask].sum()),
                        "validation_observed_positives": int(y[val_mask].sum()),
                        "n_iter": int(audit.n_iter),
                        "convergence_status": audit.convergence_status,
                        "selected": False,
                    })
                    cv_store[(outcome, track, _c_key(C), fold)] = (scaler, model, score)
                candidates.append({
                    "outcome": outcome,
                    "outcome_role": role,
                    "track": track,
                    "C": float(C),
                    "mean_average_precision": float(np.mean(fold_aps, dtype=np.float64)),
                    "_fit_rows": fit_rows_for_c,
                })
            selected_C = select_c(candidates)
            for candidate in candidates:
                for row in candidate["_fit_rows"]:
                    row["selected"] = bool(float(candidate["C"]) == selected_C)
                    selection_rows.append(row)
            selected_scores: list[np.ndarray] = []
            selected_fold_models: list[tuple[StandardScaler, LogisticRegression]] = []
            for fold in FOLDS:
                scaler, model, score = cv_store[(outcome, track, _c_key(selected_C), fold)]
                selected_scores.append(score)
                selected_fold_models.append((scaler, model))
                val_indices = np.flatnonzero(table.development_fold == fold)
                for index, value in zip(val_indices.tolist(), score.tolist()):
                    oof_rows.append({"parent_asin": table.parent_asin[index], "outcome": outcome, "outcome_role": role, "track": track, "development_fold": int(fold), "observed_positive": int(y[index]), "score": float(value)})
            oof_score = np.empty(len(y), dtype=np.float64)
            for fold, score in zip(FOLDS, selected_scores):
                oof_score[table.development_fold == fold] = score
            selected_record = next(row for row in candidates if float(row["C"]) == selected_C)
            model_id = _model_key(outcome, track)
            final_scaler, final_model, final_audit = _fit_model(X, y, selected_C, split_partition="development")
            fit_count += 1
            final_fit_audits.append({"model_id": model_id, "n_iter": int(final_audit.n_iter), "convergence_status": final_audit.convergence_status})
            parameters[model_id] = {
                "scaler_mean": np.asarray(final_scaler.mean_, dtype=np.float64),
                "scaler_scale": np.asarray(final_scaler.scale_, dtype=np.float64),
                "coef": np.asarray(final_model.coef_[0], dtype=np.float64),
                "intercept": float(final_model.intercept_[0]),
            }
            final_score = final_model.predict_proba(final_scaler.transform(X).astype(np.float64, copy=False))[:, 1].astype(np.float64)
            reconstructed = score_from_frozen_parameters(X, parameters[model_id]["scaler_mean"], parameters[model_id]["scaler_scale"], parameters[model_id]["coef"], float(parameters[model_id]["intercept"]), split_partition="development")
            if float(np.max(np.abs(final_score - reconstructed))) > 1e-12:
                raise P10ContractError(f"frozen parameter reconstruction exceeds tolerance: {model_id}")
            diagnostics = fold_diagnostics(y, oof_score, table.parent_asin, split_partition="development")
            model_specs.append({
                "model_id": model_id, "outcome": outcome, "outcome_role": role, "track": track,
                "C": selected_C, "mean_average_precision": float(selected_record["mean_average_precision"]),
                "pooled_oof_diagnostics": diagnostics, "feature_count": int(X.shape[1]),
                "feature_order": list(_feature_order(track)), "training_rows": len(y),
                "observed_positive_rows": int(y.sum()), "pu_unlabeled_rows": int(len(y) - y.sum()),
                "logistic": {"solver": "lbfgs", "l1_ratio": 0.0, "fit_intercept": True, "class_weight": None, "max_iter": 5000, "tol": 1e-8, "warm_start": False, "random_state": 20260818},
                "scaler": {"class": "StandardScaler", "with_mean": True, "with_std": True, "constant_scale": 1.0},
                "final_convergence": {"n_iter": [int(value) for value in np.asarray(final_model.n_iter_).reshape(-1)]},
                "locked_test_metrics": None,
            })
            if track == "interpretable_36_logistic":
                final_coef = parameters[model_id]["coef"]
                for feature_index, feature_name in enumerate(INTERPRETABLE_FEATURES):
                    fold_values = [float(model.coef_[0, feature_index]) for _, model in selected_fold_models]
                    final_value = float(final_coef[feature_index])
                    sign_count = sign_agreement_count(final_value, fold_values)
                    stability_rows.append({"outcome": outcome, "feature_name": feature_name, "final_standardized_coef": final_value, **{f"fold{fold}": fold_values[fold] for fold in FOLDS}, "sign_agreement_count": int(sign_count), "abs_final_coef": abs(final_value)})
    if fit_count != 156 or len(selection_rows) != 150 or len(final_fit_audits) != 6:
        raise P10ContractError(f"unexpected formal fit counts: {fit_count}, {len(selection_rows)}, {len(final_fit_audits)}")
    oof_rows.sort(key=lambda row: (OUTCOMES.index((row["outcome"], row["outcome_role"])), TRACKS.index(row["track"]), row["parent_asin"]))
    validate_oof_coverage(oof_rows, table, split_partition="development")
    convergence_audit = {
        "cv_fit_count": len(selection_rows),
        "final_fit_count": len(final_fit_audits),
        "total_fit_count": fit_count,
        "all_status_pass": all(row["convergence_status"] == "PASS" for row in selection_rows + final_fit_audits),
        "max_cv_n_iter": max(int(row["n_iter"]) for row in selection_rows),
        "max_final_n_iter": max(int(row["n_iter"]) for row in final_fit_audits),
    }
    if not convergence_audit["all_status_pass"] or convergence_audit["max_cv_n_iter"] >= 5000 or convergence_audit["max_final_n_iter"] >= 5000:
        raise P10ContractError("convergence audit hard gate failed")
    return {"selection_rows": selection_rows, "oof_rows": oof_rows, "stability_rows": stability_rows, "parameters": parameters, "model_specs": model_specs, "fit_count": fit_count, "final_fit_audits": final_fit_audits, "convergence_audit": convergence_audit}


def _manifest_rows(joined: JoinedDevelopment) -> tuple[list[dict[str, Any]], tuple[str, ...]]:
    fields = ("parent_asin", "input_order", "split_group_key", "development_fold", "primary_response_sha256", "primary_feature_row_index", "has_any_outer_imagery_observed", "general_visual_appeal_observed_positive_core", "cute_friendly_observed_positive_core")
    rows = []
    for index, asin in enumerate(joined.table.parent_asin):
        rows.append({
            "parent_asin": asin,
            "input_order": int(joined.table.input_order[index]),
            "split_group_key": joined.table.group_key[index],
            "development_fold": int(joined.table.development_fold[index]),
            "primary_response_sha256": joined.table.primary_response_sha256[index],
            "primary_feature_row_index": int(joined.feature_row_index[index]),
            "has_any_outer_imagery_observed": int(joined.table.outcomes[OUTCOMES[0][0]][index]),
            "general_visual_appeal_observed_positive_core": int(joined.table.outcomes[OUTCOMES[1][0]][index]),
            "cute_friendly_observed_positive_core": int(joined.table.outcomes[OUTCOMES[2][0]][index]),
        })
    return rows, fields


def _selection_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    fields = ("outcome", "outcome_role", "track", "C", "fold", "fold_average_precision", "fold_auroc", "train_rows", "validation_rows", "train_observed_positives", "validation_observed_positives", "n_iter", "convergence_status", "selected")
    return _csv_bytes(rows, fields)


def _oof_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    fields = ("parent_asin", "outcome", "outcome_role", "track", "development_fold", "observed_positive", "score")
    return _csv_bytes(rows, fields)


def _stability_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    fields = ("outcome", "feature_name", "final_standardized_coef", "fold0", "fold1", "fold2", "fold3", "fold4", "sign_agreement_count", "abs_final_coef")
    return _csv_bytes(rows, fields)


def _parameters_bytes(parameters: Mapping[str, Mapping[str, Any]], *, split_partition: str | Sequence[str] | None = None) -> bytes:
    _assert_development_partition(split_partition)
    buffer = io.BytesIO()
    arrays: dict[str, np.ndarray] = {}
    for model_id in (_model_key(outcome, track) for outcome, _ in OUTCOMES for track in TRACKS):
        value = parameters[model_id]
        prefix = model_id
        arrays[f"{prefix}__scaler_mean"] = np.asarray(value["scaler_mean"], dtype=np.float64)
        arrays[f"{prefix}__scaler_scale"] = np.asarray(value["scaler_scale"], dtype=np.float64)
        arrays[f"{prefix}__coef"] = np.asarray(value["coef"], dtype=np.float64)
        arrays[f"{prefix}__intercept"] = np.asarray([float(value["intercept"])], dtype=np.float64)
    np.savez(buffer, **arrays)
    return buffer.getvalue()


def _summary(modeling: Mapping[str, Any], sha: Mapping[str, str]) -> dict[str, Any]:
    return {
        "contract_version": "p10_v1.0",
        "stage": "development-only model freeze",
        "development_rows": 4143,
        "locked_test_rows_aggregate_only": 1036,
        "development_positive_rows": 186,
        "development_fold_counts": {"0": 837, "1": 828, "2": 828, "3": 828, "4": 822},
        "outcomes": [name for name, _ in OUTCOMES],
        "tracks": list(TRACKS),
        "cv_fit_count": 150,
        "cv_ledger_rows": 150,
        "final_fit_count": 6,
        "total_fit_count": int(modeling["fit_count"]),
        "convergence_audit": dict(modeling["convergence_audit"]),
        "oof_rows": 24858,
        "confirmatory_family_size": 2,
        "bh_q_reserved_for_p11_p12": 0.05,
        "model_selection_scope": "development-only unweighted five-fold AP; no locked-test diagnostics",
        "formal_output_sha256_01_to_06": dict(sha),
        "p11_executed": False,
    }


def build_artifacts(joined: JoinedDevelopment) -> tuple[dict[str, bytes], dict[str, Any]]:
    modeling = run_development_modeling(joined)
    manifest_rows, manifest_fields = _manifest_rows(joined)
    manifest_bytes = _csv_bytes(manifest_rows, manifest_fields)
    selection_bytes = _selection_bytes(modeling["selection_rows"])
    oof_bytes = _oof_bytes(modeling["oof_rows"])
    spec = {"contract_version": "p10_v1.0", "models": modeling["model_specs"], "model_count": 6, "locked_test_metrics": None, "p11_executed": False}
    spec_bytes = _json_bytes(spec)
    parameters_bytes = _parameters_bytes(modeling["parameters"], split_partition="development")
    stability_bytes = _stability_bytes(modeling["stability_rows"])
    first_six = {
        "01_development_modeling_manifest.csv": sha256_bytes(manifest_bytes),
        "02_cv_model_selection.csv": sha256_bytes(selection_bytes),
        "03_development_oof_scores.csv": sha256_bytes(oof_bytes),
        "04_final_model_specification.json": sha256_bytes(spec_bytes),
        "05_final_model_parameters.npz": sha256_bytes(parameters_bytes),
        "06_interpretable_coefficient_stability.csv": sha256_bytes(stability_bytes),
    }
    summary = _summary(modeling, first_six)
    summary_bytes = _json_bytes(summary)
    artifacts = {
        "01_development_modeling_manifest.csv": manifest_bytes,
        "02_cv_model_selection.csv": selection_bytes,
        "03_development_oof_scores.csv": oof_bytes,
        "04_final_model_specification.json": spec_bytes,
        "05_final_model_parameters.npz": parameters_bytes,
        "06_interpretable_coefficient_stability.csv": stability_bytes,
        "07_development_model_summary.json": summary_bytes,
    }
    return artifacts, {"modeling": modeling, "manifest_rows": manifest_rows, "manifest_fields": manifest_fields, "spec": spec}


def _verify_upstream(contract: Mapping[str, Any]) -> dict[str, Any]:
    p8_shas = {}
    for name, expected in contract["upstream"]["p8_b_formal_sha256"].items():
        actual = sha256_file(P8_DIR / name)
        _assert_exact(actual, expected, f"P8-B {name} SHA")
        p8_shas[name] = actual
    p9_shas = {}
    for name, expected in contract["upstream"]["p9_formal_sha256"].items():
        actual = sha256_file(P9_DIR / name)
        _assert_exact(actual, expected, f"P9 {name} SHA")
        p9_shas[name] = actual
    if not P8_MANIFEST.is_file() or not P9_MAP.is_file() or not P9_INVENTORY.is_file() or not P9_EMBEDDINGS.is_file() or not P9_FEATURES.is_file():
        raise P10ContractError("required P8/P9 source paths are incomplete")
    return {"p8_b_formal_sha256": p8_shas, "p9_formal_sha256": p9_shas, "zero_network_calls": True}


def _build_provenance(output_sha: Mapping[str, str], upstream: Mapping[str, Any], contract: Mapping[str, Any], modeling: Mapping[str, Any]) -> dict[str, Any]:
    commit = _git_head()
    return {
        "contract_version": "p10_v1.0",
        "baseline_main_commit": contract["baseline_main_commit"],
        "formal_run_git_commit": commit,
        "p8_b_baseline_main_commit": contract["upstream"]["p8_b_baseline_main_commit"],
        "p9_baseline_main_commit": contract["upstream"]["p9_baseline_main_commit"],
        "p8_b_formal_sha256": dict(contract["upstream"]["p8_b_formal_sha256"]),
        "p9_formal_sha256": dict(contract["upstream"]["p9_formal_sha256"]),
        "upstream_verification": dict(upstream),
        "producer": {"path": contract["producer_path"], "git_blob_sha256": git_blob_sha(commit, contract["producer_path"])},
        "p10_contract": {"path": "config/modeling/p10_development_model_contract.json", "git_blob_sha256": git_blob_sha(commit, "config/modeling/p10_development_model_contract.json")},
        "p11_contract": {"path": "config/modeling/p11_locked_test_evaluation_contract.json", "git_blob_sha256": git_blob_sha(commit, "config/modeling/p11_locked_test_evaluation_contract.json")},
        "environment": runtime_environment(contract),
        "modeling_scope": {"rows": 4143, "locked_test_metrics": False, "locked_test_predictions": False, "locked_test_model_choice": False, "p11_executed": False},
        "convergence_audit": dict(modeling["convergence_audit"]),
        "formal_output_sha256": dict(output_sha),
        "provenance_self_sha256": None,
    }


def _validate_provenance(provenance: Mapping[str, Any], contract: Mapping[str, Any], verify_git: bool = True) -> None:
    if "08_p10_provenance.json" in provenance.get("formal_output_sha256", {}):
        raise P10ContractError("P10 provenance records its own SHA")
    if provenance.get("provenance_self_sha256") is not None:
        raise P10ContractError("P10 provenance contains its own SHA")
    _assert_exact(provenance.get("baseline_main_commit"), contract["baseline_main_commit"], "P10 provenance baseline")
    _assert_exact(provenance.get("p8_b_formal_sha256"), contract["upstream"]["p8_b_formal_sha256"], "P8-B provenance ledger")
    _assert_exact(provenance.get("p9_formal_sha256"), contract["upstream"]["p9_formal_sha256"], "P9 provenance ledger")
    _assert_exact(provenance.get("p8_b_baseline_main_commit"), contract["upstream"]["p8_b_baseline_main_commit"], "P8-B provenance baseline")
    _assert_exact(provenance.get("p9_baseline_main_commit"), contract["upstream"]["p9_baseline_main_commit"], "P9 provenance baseline")
    _assert_exact(provenance.get("environment"), runtime_environment(contract), "P10 environment provenance")
    convergence = provenance.get("convergence_audit")
    if not isinstance(convergence, Mapping) or convergence.get("cv_fit_count") != 150 or convergence.get("final_fit_count") != 6 or convergence.get("total_fit_count") != 156 or convergence.get("all_status_pass") is not True or int(convergence.get("max_cv_n_iter", 5000)) >= 5000 or int(convergence.get("max_final_n_iter", 5000)) >= 5000:
        raise P10ContractError("P10 provenance convergence audit mismatch")
    for key, path in (("producer", contract["producer_path"]), ("p10_contract", "config/modeling/p10_development_model_contract.json"), ("p11_contract", "config/modeling/p11_locked_test_evaluation_contract.json")):
        trace = provenance.get(key)
        if not isinstance(trace, Mapping) or trace.get("path") != path or not isinstance(trace.get("git_blob_sha256"), str) or len(trace["git_blob_sha256"]) != 40:
            raise P10ContractError(f"P10 provenance {key} trace missing")
        if verify_git:
            commit = str(provenance.get("formal_run_git_commit") or "")
            if not commit or git_blob_sha(commit, path) != trace["git_blob_sha256"] or git_blob_sha(_git_head(), path) != trace["git_blob_sha256"]:
                raise P10ContractError(f"P10 provenance {key} Git binding mismatch")


def _expected_artifact_bytes(joined: JoinedDevelopment) -> dict[str, bytes]:
    artifacts, _ = build_artifacts(joined)
    return artifacts


def prepare() -> dict[str, Any]:
    if FORMAL_DIR.exists() and any(FORMAL_DIR.iterdir()):
        raise P10ContractError("P10 formal output directory is non-empty; refusing overwrite")
    contract = validate_contract()
    validate_p11_contract()
    upstream = _verify_upstream(contract)
    joined = join_development_features(build_development_table(contract), contract)
    artifacts, details = build_artifacts(joined)
    for name in FORMAL_FILES[:7]:
        _write_bytes(FORMAL_DIR / name, artifacts[name])
    output_sha = {name: sha256_file(FORMAL_DIR / name) for name in FORMAL_FILES[:7]}
    _write_bytes(FORMAL_DIR / FORMAL_FILES[7], _json_bytes(_build_provenance(output_sha, upstream, contract, details["modeling"])))
    return {"verification": "PASS", "development_rows": 4143, "oof_rows": 24858, "cv_fit_count": details["modeling"]["fit_count"] - 6, "final_fit_count": 6, "formal_files": list(FORMAL_FILES), "p11_executed": False}


def _validate_parameters(spec: Mapping[str, Any]) -> None:
    if spec.get("model_count") != 6 or spec.get("locked_test_metrics") is not None or spec.get("p11_executed") is not False:
        raise P10ContractError("final model specification firewall mismatch")
    expected_ids = {_model_key(outcome, track) for outcome, _ in OUTCOMES for track in TRACKS}
    if {item.get("model_id") for item in spec.get("models", [])} != expected_ids:
        raise P10ContractError("final model specification model IDs mismatch")


def _reconstruct_selection_from_ledger(selection_rows: Sequence[Mapping[str, Any]], table: DevelopmentTable) -> dict[tuple[str, str], dict[str, float]]:
    expected_fields = ("outcome", "outcome_role", "track", "C", "fold", "fold_average_precision", "fold_auroc", "train_rows", "validation_rows", "train_observed_positives", "validation_observed_positives", "n_iter", "convergence_status", "selected")
    if len(selection_rows) != 150:
        raise P10ContractError(f"P10 CV ledger must contain exactly 150 rows: {len(selection_rows)}")
    if tuple(selection_rows[0]) != expected_fields:
        raise P10ContractError("P10 CV ledger schema mismatch")
    expected_by_key: dict[tuple[str, str, float, int], dict[str, int]] = {}
    for outcome, _role in OUTCOMES:
        y = table.outcomes[outcome]
        for track in TRACKS:
            for C in C_GRID:
                for fold in FOLDS:
                    train = table.development_fold != fold
                    validation = table.development_fold == fold
                    expected_by_key[(outcome, track, float(C), fold)] = {
                        "train_rows": int(train.sum()),
                        "validation_rows": int(validation.sum()),
                        "train_observed_positives": int(y[train].sum()),
                        "validation_observed_positives": int(y[validation].sum()),
                    }
    grouped: dict[tuple[str, str, float], list[Mapping[str, Any]]] = {}
    for row in selection_rows:
        outcome = str(row["outcome"])
        role = str(row["outcome_role"])
        track = str(row["track"])
        C = float(row["C"])
        fold = int(row["fold"])
        if (outcome, role) not in OUTCOMES or track not in TRACKS or C not in C_GRID or fold not in FOLDS:
            raise P10ContractError("P10 CV ledger contains an unknown outcome, track, C, or fold")
        expected = expected_by_key.get((outcome, track, C, fold))
        if expected is None or any(int(row[field]) != value for field, value in expected.items()):
            raise P10ContractError("P10 CV ledger train/validation counts mismatch")
        if not np.isfinite(float(row["fold_average_precision"])) or not np.isfinite(float(row["fold_auroc"])):
            raise P10ContractError("P10 CV ledger contains non-finite fold metrics")
        if str(row["convergence_status"]) != "PASS":
            raise P10ContractError("P10 CV ledger contains a non-PASS convergence status")
        validate_convergence_audit(int(row["n_iter"]), False, "P10 CV ledger fit")
        grouped.setdefault((outcome, track, C), []).append(row)
    selected_by_key: dict[tuple[str, str], dict[str, float]] = {}
    for outcome, _role in OUTCOMES:
        for track in TRACKS:
            candidates: list[dict[str, Any]] = []
            for C in C_GRID:
                rows = grouped.get((outcome, track, float(C)), [])
                if len(rows) != len(FOLDS) or {int(row["fold"]) for row in rows} != set(FOLDS):
                    raise P10ContractError(f"P10 CV ledger fold coverage mismatch: {outcome}/{track}/{C}")
                selected_values = {str(row["selected"]) for row in rows}
                if selected_values - {"True", "False"} or len(selected_values) != 1:
                    raise P10ContractError("P10 CV ledger selected flag is not group-consistent")
                candidates.append({"C": float(C), "mean_average_precision": float(np.mean([float(row["fold_average_precision"]) for row in rows], dtype=np.float64))})
            selected_C = select_c(candidates)
            for candidate in candidates:
                rows = grouped[(outcome, track, float(candidate["C"]))]
                expected_selected = float(candidate["C"]) == selected_C
                if any(str(row["selected"]) != str(expected_selected) for row in rows):
                    raise P10ContractError(f"P10 CV ledger selected C mismatch: {outcome}/{track}")
            selected = next(candidate for candidate in candidates if float(candidate["C"]) == selected_C)
            selected_by_key[(outcome, track)] = {"C": selected_C, "mean_average_precision": float(selected["mean_average_precision"])}
    if len(grouped) != 30:
        raise P10ContractError("P10 CV ledger does not contain exactly 30 candidate groups")
    return selected_by_key


def _validate_formal_content(contract: Mapping[str, Any]) -> dict[str, Any]:
    if not FORMAL_DIR.exists() or {path.name for path in FORMAL_DIR.iterdir()} != set(FORMAL_FILES):
        raise P10ContractError("P10 formal outputs are incomplete or contain unexpected files")
    joined = join_development_features(build_development_table(contract), contract)
    manifest_rows, manifest_fields = _manifest_rows(joined)
    if (FORMAL_DIR / FORMAL_FILES[0]).read_bytes() != _csv_bytes(manifest_rows, manifest_fields):
        raise P10ContractError("P10 development manifest reconstruction mismatch")
    selection_rows = _read_csv(FORMAL_DIR / FORMAL_FILES[1])
    selected_by_key = _reconstruct_selection_from_ledger(selection_rows, joined.table)
    parameters = np.load(FORMAL_DIR / FORMAL_FILES[4], allow_pickle=False)
    expected_keys = {f"{model_id}__{field}" for model_id in (_model_key(outcome, track) for outcome, _ in OUTCOMES for track in TRACKS) for field in ("scaler_mean", "scaler_scale", "coef", "intercept")}
    if set(parameters.files) != expected_keys:
        raise P10ContractError("P10 parameter NPZ keys mismatch")
    for key in parameters.files:
        if parameters[key].dtype != np.float64 or not np.isfinite(parameters[key]).all():
            raise P10ContractError(f"invalid P10 parameter array: {key}")
    spec = read_json(FORMAL_DIR / FORMAL_FILES[3])
    _validate_parameters(spec)
    for model in spec["models"]:
        selected = selected_by_key[(model["outcome"], model["track"])]
        if float(model["C"]) != selected["C"] or float(model["mean_average_precision"]) != selected["mean_average_precision"] or model["feature_count"] not in (512, 36):
            raise P10ContractError("P10 final model specification does not bind reconstructed CV selection")
    oof_rows = _read_csv(FORMAL_DIR / FORMAL_FILES[2])
    validate_oof_coverage(oof_rows, joined.table, split_partition="development")
    if any(row["parent_asin"] not in set(joined.table.parent_asin) for row in oof_rows):
        raise P10ContractError("OOF contains non-development parent_asin")
    stability_rows = _read_csv(FORMAL_DIR / FORMAL_FILES[5])
    expected_stability_fields = ("outcome", "feature_name", "final_standardized_coef", "fold0", "fold1", "fold2", "fold3", "fold4", "sign_agreement_count", "abs_final_coef")
    if len(stability_rows) != 108 or tuple(stability_rows[0]) != expected_stability_fields or len({(row["outcome"], row["feature_name"]) for row in stability_rows}) != 108:
        raise P10ContractError("P10 coefficient stability schema mismatch")
    summary = read_json(FORMAL_DIR / FORMAL_FILES[6])
    _assert_exact(summary.get("development_rows"), 4143, "P10 summary development rows")
    _assert_exact(summary.get("oof_rows"), 24858, "P10 summary OOF rows")
    _assert_exact(summary.get("cv_ledger_rows"), 150, "P10 summary CV ledger rows")
    _assert_exact(summary.get("cv_fit_count"), 150, "P10 summary CV fit count")
    _assert_exact(summary.get("final_fit_count"), 6, "P10 summary final fit count")
    _assert_exact(summary.get("total_fit_count"), 156, "P10 summary total fit count")
    convergence = summary.get("convergence_audit")
    if not isinstance(convergence, Mapping) or convergence.get("cv_fit_count") != 150 or convergence.get("final_fit_count") != 6 or convergence.get("total_fit_count") != 156 or convergence.get("all_status_pass") is not True or int(convergence.get("max_cv_n_iter", 5000)) >= 5000 or int(convergence.get("max_final_n_iter", 5000)) >= 5000:
        raise P10ContractError("P10 summary convergence audit mismatch")
    _assert_exact(summary.get("p11_executed"), False, "P10 summary P11 status")
    provenance = read_json(FORMAL_DIR / FORMAL_FILES[7])
    _validate_provenance(provenance, contract)
    expected_output_sha = {name: sha256_file(FORMAL_DIR / name) for name in FORMAL_FILES[:7]}
    _assert_exact(provenance.get("formal_output_sha256"), expected_output_sha, "P10 formal output SHA ledger")
    tracked = subprocess.check_output(["git", "ls-files", "--", contract["formal_output_directory"]], cwd=ROOT, text=True).strip()
    if tracked:
        raise P10ContractError("P10 formal data must remain gitignored")
    return {"verification": "PASS", "development_rows": len(joined.table.parent_asin), "oof_rows": len(oof_rows), "zero_writes": True, "zero_network_calls": True, "locked_test_metrics": False, "p11_executed": False}


def verify_existing() -> dict[str, Any]:
    contract = validate_contract()
    validate_p11_contract()
    upstream = _verify_upstream(contract)
    result = _validate_formal_content(contract)
    result["upstream_verification"] = upstream
    result["zero_model_fits"] = True
    return result


def verify_recompute() -> dict[str, Any]:
    contract = validate_contract()
    validate_p11_contract()
    upstream = _verify_upstream(contract)
    existing = _validate_formal_content(contract)
    joined = join_development_features(build_development_table(contract), contract)
    expected = _expected_artifact_bytes(joined)
    for name in (FORMAL_FILES[0], FORMAL_FILES[1], FORMAL_FILES[2], FORMAL_FILES[3], FORMAL_FILES[5], FORMAL_FILES[6]):
        if expected[name] != (FORMAL_DIR / name).read_bytes():
            raise P10ContractError(f"P10 recompute byte mismatch: {name}")
    formal_parameters = np.load(FORMAL_DIR / FORMAL_FILES[4], allow_pickle=False)
    recomputed_parameters = np.load(io.BytesIO(expected[FORMAL_FILES[4]]), allow_pickle=False)
    parameter_max_diff = 0.0
    for key in formal_parameters.files:
        parameter_max_diff = max(parameter_max_diff, float(np.max(np.abs(formal_parameters[key] - recomputed_parameters[key]))))
    if parameter_max_diff > 1e-10:
        raise P10ContractError(f"P10 parameter recompute tolerance exceeded: {parameter_max_diff}")
    oof_existing = np.asarray([float(row["score"]) for row in _read_csv(FORMAL_DIR / FORMAL_FILES[2])], dtype=np.float64)
    oof_recomputed_rows = list(csv.DictReader(io.StringIO(expected[FORMAL_FILES[2]].decode("utf-8"))))
    oof_recomputed = np.asarray([float(row["score"]) for row in oof_recomputed_rows], dtype=np.float64)
    score_max_diff = float(np.max(np.abs(oof_existing - oof_recomputed))) if len(oof_existing) else 0.0
    if score_max_diff > 1e-10:
        raise P10ContractError(f"P10 OOF recompute tolerance exceeded: {score_max_diff}")
    return {**existing, "upstream_verification": upstream, "recompute": "PASS", "parameter_max_abs_diff": parameter_max_diff, "score_max_abs_diff": score_max_diff, "zero_writes": True, "zero_network_calls": True, "zero_locked_test_metrics": True}


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
        raise P10ContractError("choose prepare, --verify-existing, or --verify-recompute")
    except P10ContractError as exc:
        print(f"P10 ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
