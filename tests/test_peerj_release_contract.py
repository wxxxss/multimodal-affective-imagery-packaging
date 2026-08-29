from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

PUBLIC_RUNNABLE_PATHS = [
    "scripts/modeling/p8_a_analysis_contract.py",
    "scripts/modeling/p8_b_modeling_ready_split.py",
    "scripts/modeling/p9_visual_features.py",
    "scripts/modeling/p10_development_models.py",
    "scripts/modeling/public_heldout_evaluation.py",
    "scripts/release/export_audit_inputs.py",
    "scripts/figures/figure01_data_construction_v5.py",
    "scripts/figures/figure03_recognition_performance_v7_auroc_corrected.py",
    "scripts/figures/figure04_design_strategies_v6.py",
]

UPSTREAM_PUBLIC_PATHS = [
    "scripts/metadata/screen_full_metadata_v2.py",
    "scripts/metadata/secondary_clean_candidates_v21.py",
    "scripts/metadata/secondary_clean_candidates_v22.py",
    "scripts/reviews/match_reviews_to_valid_products.py",
    "scripts/reviews/clean_and_extract_packaging_sentences.py",
    "scripts/visual_packaging/strict_visual_packaging_classifier_v11.py",
    "scripts/labels/build_affective_imagery_labels_v21.py",
    "scripts/images/audit_p7_source_inventory.py",
    "scripts/images/acquire_p7_primary_assets.py",
    "scripts/images/p7_c_primary_manifest_qa.py",
    "scripts/images/p7_d_final_image_freeze.py",
    "scripts/validation/_prepare_affective_imagery_validation_v21_core.py",
    "scripts/validation/affective_imagery_action_error_mapping_v21.py",
    "scripts/validation/affective_imagery_annotation_policy_v21.py",
    "scripts/validation/affective_imagery_decision_sidecar_v21.py",
    "scripts/validation/affective_imagery_final_adjudication_v21.py",
    "scripts/validation/affective_imagery_validation_manifest_v21.py",
    "scripts/validation/affective_imagery_validation_provenance_v21.py",
    "scripts/validation/normalize_affective_imagery_actions_v21.py",
    "scripts/validation/prepare_affective_imagery_validation_v21.py",
    "scripts/validation/repair_affective_imagery_a2_rationales_v21.py",
    "scripts/validation/summarize_affective_imagery_validation_v21.py",
    "scripts/validation/validate_affective_imagery_annotations_v21.py",
]

UPSTREAM_RUNTIME_CONFIG_PATHS = [
    "config/affective_imagery/action_error_mapping_v21.json",
]


def test_public_runnable_paths_exist():
    missing = [path for path in PUBLIC_RUNNABLE_PATHS if not (ROOT / path).is_file()]
    assert not missing, "Missing public runnable files: " + ", ".join(missing)


def test_peerj_upstream_source_release_is_complete():
    required = UPSTREAM_PUBLIC_PATHS + UPSTREAM_RUNTIME_CONFIG_PATHS
    missing = [path for path in required if not (ROOT / path).is_file()]
    assert not missing, "Missing PeerJ upstream release files: " + ", ".join(missing)


def test_upstream_source_release_is_documented_as_public_code():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    reproducibility = (ROOT / "docs" / "REPRODUCIBILITY.md").read_text(encoding="utf-8")
    provenance = (ROOT / "docs" / "PREPROCESSING_SOURCE_BINDINGS.md").read_text(encoding="utf-8")
    for path in UPSTREAM_PUBLIC_PATHS + UPSTREAM_RUNTIME_CONFIG_PATHS:
        assert f"`{path}`" in provenance, f"Missing provenance mapping for {path}"
    assert "scripts/metadata/screen_full_metadata_v2.py" in readme
    assert "scripts/reviews/clean_and_extract_packaging_sentences.py" in reproducibility
    assert "scripts/labels/build_affective_imagery_labels_v21.py" in reproducibility
    assert "config/affective_imagery/action_error_mapping_v21.json" in reproducibility


def test_peerj_required_source_and_environment_disclosures_are_present():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    environment = (ROOT / "docs" / "ENVIRONMENT.md").read_text(encoding="utf-8")
    assert "McAuley-Lab/Amazon-Reviews-2023" in readme
    assert "10.48550/arXiv.2403.03952" in readme
    assert "Windows 11" in readme
    assert "Python 3.12.7" in readme
    assert "CPU" in environment
