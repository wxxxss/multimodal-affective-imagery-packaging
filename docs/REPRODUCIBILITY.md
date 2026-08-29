# Reproducibility workflow

This document maps the frozen study workflow to the publication-facing repository. Upstream P1–P7 source code is now included in this checkout; exact end-to-end regeneration of historical upstream artifacts still requires the original third-party Amazon Reviews'23 source records and image assets, which are not redistributed.

## 1. Obtain third-party source data

The study used Amazon Reviews'23 Grocery metadata and review resources released by the UCSD McAuley Lab. Source URLs and the reference DOI are given in `docs/DATA_ACCESS.md` and the repository README.

Raw Amazon review text and third-party product-image binaries are not redistributed here.

## 2. Metadata eligibility screening

Public implementation:

```bash
python scripts/metadata/screen_full_metadata_v2.py --help
python scripts/metadata/secondary_clean_candidates_v21.py --help
python scripts/metadata/secondary_clean_candidates_v22.py --help
```

The frozen workflow streamed Grocery metadata, applied ordered product-scope rules for dry herbal/fruit infusion products, and retained candidates for secondary cleaning/review. The final study universe contained 5,180 eligible products.

Exact frozen source identities are recorded in `docs/PREPROCESSING_SOURCE_BINDINGS.md`. `screen_full_metadata_v2.py` contains publication-safe path sanitation only; scientific screening logic is unchanged.

## 3. Review matching and text preprocessing

Public implementation:

```bash
python scripts/reviews/match_reviews_to_valid_products.py --help
python scripts/reviews/clean_and_extract_packaging_sentences.py --help
```

Reviews were joined to eligible products by `parent_asin`, normalized/cleaned, segmented into sentences, and screened with high-recall packaging-language rules before strict visual-package classification. The manuscript reports 14,318,520 Grocery reviews screened, 151,175 matched reviews, 146,160 clean reviews, 539,132 sentences, and 56,197 high-recall packaging-language candidates.

Exact regeneration requires the original Amazon Reviews'23 records; those raw records are not redistributed.

## 4. Visual-package screening and PU outcome construction

Public implementation:

```bash
python scripts/visual_packaging/strict_visual_packaging_classifier_v11.py --help
python scripts/labels/build_affective_imagery_labels_v21.py --help
```

Only clause-level affective expressions referring to the **outer retail package** contributed to modeled outcomes. A value of `1` denotes an observed qualifying mention; `0` is unlabeled/unobserved rather than a confirmed negative.

The exact frozen source identities are recorded in `docs/PREPROCESSING_SOURCE_BINDINGS.md`.

## 5. Validation/adjudication

The publication-facing validation implementation is under `scripts/validation/`; the protocol/schema is under `docs/validation/`. The production action/error mapping contract is `config/affective_imagery/action_error_mapping_v21.json`.

Representative entry points include:

```bash
python scripts/validation/prepare_affective_imagery_validation_v21.py --help
python scripts/validation/validate_affective_imagery_annotations_v21.py --help
python scripts/validation/summarize_affective_imagery_validation_v21.py --help
```

The study used two independent, rule-guided ChatGPT (GPT-5.5; OpenAI)-assisted review passes on the same 840 validation tasks, followed by model-assisted adjudication. This is not described as independent human-gold annotation. Validation findings were used for quality reporting and did not create a sample-derived production-label mask.

## 6. Image acquisition, QA, and primary-image freeze

Public P7 implementation:

```bash
python scripts/images/audit_p7_source_inventory.py --help
python scripts/images/acquire_p7_primary_assets.py --help
python scripts/images/p7_c_primary_manifest_qa.py --help
python scripts/images/p7_d_final_image_freeze.py --help
```

The P7 workflow audited image sources, acquired candidate retail-package images, performed QA, and froze one primary image per product/group under the prespecified contracts in `config/image_assets/`. Raw image binaries and source URLs are not redistributed.

## 7. Create the leakage-controlled modeling split

```bash
python scripts/modeling/p8_a_analysis_contract.py --help
python scripts/modeling/p8_b_modeling_ready_split.py --help
```

The frozen modeling population contained 5,179 products: 4,143 development products and 1,036 held-out products. Products sharing the same primary image-response hash remain in the same partition.

## 8. Extract image representations

```bash
python scripts/modeling/p9_visual_features.py --help
```

The frozen image-representation stage generated 512-dimensional L2-normalized OpenCLIP ViT-B/32 embeddings, 20 classical raster descriptors, and 16 semantic image-text design-similarity scores (36 interpretable features in total). The semantic prompt bank and interpretable-feature schema are under `config/modeling/`.

## 9. Development-only model selection and refit

```bash
python scripts/modeling/p10_development_models.py --help
```

For each of three outcomes and two representation tracks, predictors were standardized and logistic regression was selected over the C grid `{0.01, 0.1, 1, 10, 100}` using five grouped development folds and mean validation average precision. The held-out set was not used for preprocessing, model selection, or fitting.

## 10. Export publication-safe held-out inputs

The release exporter copies only the fields required to audit the held-out results and deliberately excludes raw review text, image URLs/paths, images, and embeddings:

```bash
python scripts/release/export_audit_inputs.py \
  --manifest PATH/TO/FROZEN_MODELING_MANIFEST.csv \
  --predictions PATH/TO/FROZEN_LOCKED_TEST_PREDICTIONS.csv \
  --output-dir data/derived
```

For the frozen study this export must contain 5,179 modeling-manifest rows and 6,216 held-out prediction rows (1,036 products × 6 models). The resulting publication-safe files are intended for PeerJ Supplemental Data 1 or a DOI-bearing data/code archive.

## 11. Held-out evaluation and completed AUROC audit

```bash
python scripts/modeling/public_heldout_evaluation.py \
  --modeling-manifest data/derived/01_modeling_ready_manifest.csv \
  --predictions data/derived/02_locked_test_predictions.csv \
  --output results/heldout_metric_audit.json \
  --bootstrap-iterations 5000 \
  --seed 20260818
```

The evaluator consumes already-frozen held-out predictions and labels. It performs no model fitting or selection and never flips scores. Average precision and top-k metrics use deterministic score-descending ranking; AUROC uses conventional `sklearn.metrics.roc_auc_score`, matching P10. Cluster-bootstrap uncertainty uses `primary_response_sha256` as the sampling unit.

The audit reproduced the original 5,000-draw, seed-`20260818` plan. Conventional held-out AUROC is 0.6877-0.7220 across the six models, with all six corrected 95% interval lower bounds above 0.5. Exact values and provenance are under `data/published_results/p11_auroc_correction/` and `docs/AUROC_AUDIT.md`.

## 12. Post-lock interpretation

Historical P12 is governed by `config/modeling/p12_post_lock_interpretation_contract.json`. It combined development-fold coefficient-direction stability with outcome-level held-out evidence and did not retrain, refit, reselect, rescore, flip scores, change labels, change features, change the split, or select a threshold.

Frozen historical P12 outputs remain under `data/published_results/p12/`. Publication-facing AUROC interpretation is overlaid under `data/published_results/p12_auroc_correction/`. The E2/E1 enrichment grades and resulting Grade A/B design rules are unchanged because they are based on top-10% lift evidence, coefficient stability, and the QA-exclusion lift check rather than AUROC.

## 13. Figures

Figures 1 and 4 retain their frozen manuscript values. Figure 3 is regenerated with the conventional-AUROC correction:

- `scripts/figures/figure01_data_construction_v5.py`
- `scripts/figures/figure03_recognition_performance_v7_auroc_corrected.py`
- `scripts/figures/figure04_design_strategies_v6.py`

The historical Figure 3 v6 script is retained for provenance but is not the publication-facing source after the AUROC audit. Figure 2 is a conceptual workflow diagram rather than a statistical output.

## Verification principle

The public source release uses two independent checks: Git blob identity for frozen-source provenance, and functional repository tests/CI for release integrity. Where exact row-level frozen artifacts are supplied in the publication-safe derived-data release, use their recorded SHA-256 values. Do not regenerate or reselect a frozen upstream stage merely to obtain a different downstream result. Any correction to a verified metric-implementation defect must remain bounded, documented, and based on the already-frozen predictions/labels.
