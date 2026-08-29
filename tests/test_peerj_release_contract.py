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

UPSTREAM_PATHS_NOT_PRESENT_IN_PUBLIC_CHECKOUT = [
    "scripts/metadata/screen_full_metadata_v2.py",
    "scripts/metadata/secondary_clean_candidates_v21.py",
    "scripts/metadata/secondary_clean_candidates_v22.py",
    "scripts/reviews/match_reviews_to_valid_products.py",
    "scripts/reviews/clean_and_extract_packaging_sentences.py",
    "scripts/visual_packaging/strict_visual_packaging_classifier_v11.py",
    "build_affective_imagery_labels_v21.py",
]


def test_public_runnable_paths_exist():
    missing = [path for path in PUBLIC_RUNNABLE_PATHS if not (ROOT / path).is_file()]
    assert not missing, "Missing public runnable files: " + ", ".join(missing)


def test_readme_does_not_present_absent_upstream_sources_as_public_runnable_code():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    for path in UPSTREAM_PATHS_NOT_PRESENT_IN_PUBLIC_CHECKOUT:
        assert f"`{path}`" not in readme, (
            f"README presents absent historical source as a public path: {path}. "
            "Point readers to docs/PREPROCESSING_SOURCE_BINDINGS.md instead."
        )


def test_reproducibility_does_not_instruct_running_absent_upstream_sources():
    text = (ROOT / "docs" / "REPRODUCIBILITY.md").read_text(encoding="utf-8")
    assert "once the publication-facing upstream scripts are present" not in text
    for path in UPSTREAM_PATHS_NOT_PRESENT_IN_PUBLIC_CHECKOUT:
        assert f"python {path}" not in text


def test_peerj_required_source_and_environment_disclosures_are_present():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    environment = (ROOT / "docs" / "ENVIRONMENT.md").read_text(encoding="utf-8")
    assert "McAuley-Lab/Amazon-Reviews-2023" in readme
    assert "10.48550/arXiv.2403.03952" in readme
    assert "Windows 11" in readme
    assert "Python 3.12.7" in readme
    assert "CPU" in environment
