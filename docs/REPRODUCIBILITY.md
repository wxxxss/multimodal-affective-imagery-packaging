# Reproducibility workflow

This document maps the frozen study workflow to the public code export. It is intentionally organized around the manuscript rather than the internal project-management history.

## 1. Obtain source data

Download the Amazon Reviews'23 Grocery metadata and review resources from the McAuley Lab source repository described in `docs/DATA_ACCESS.md`.

## 2. Screen eligible infusion products

Run the metadata-screening sequence once the publication-facing upstream scripts are present in the release:

```bash
python scripts/metadata/screen_full_metadata_v2.py --help
python scripts/metadata/secondary_clean_candidates_v21.py --help
python scripts/metadata/secondary_clean_candidates_v22.py --help
```

The study universe ultimately contained 5,180 eligible products.

## 3. Match and preprocess reviews

```bash
python scripts/reviews/match_reviews_to_valid_products.py --help
python scripts/reviews/clean_and_extract_packaging_sentences.py --help
```

The manuscript reports 14,318,520 Grocery reviews screened, 151,175 matched reviews, 146,160 clean reviews, 539,132 sentences, and 56,197 high-recall packaging-language candidates.

## 4. Identify qualifying visual-package language and construct PU outcomes

```bash
python scripts/visual_packaging/strict_visual_packaging_classifier_v11.py --help
python build_affective_imagery_labels_v21.py --help
```

Only clause-level affective expressions related to the outer retail package contribute to the modeled outcomes. A zero label is unlabeled/unobserved, not a confirmed negative.

## 5. Validate the label-construction process

The validation protocol and schema are published under `docs/validation/`. The study used two independent, rule-guided ChatGPT (GPT-5.5; OpenAI)-assisted review passes on the same 840 validation tasks, followed by model-assisted adjudication. This is not human-gold annotation. Validation findings were used for reporting and did not change the frozen production labels via a sample-derived mask.

## 6. Acquire and freeze retail-package images

The P7 workflow implements source-inventory audit, image acquisition, QA, and final frozen primary-image selection. Raw third-party image binaries are not redistributed.

## 7. Create the leakage-controlled modeling split

```bash
python scripts/modeling/p8_b_modeling_ready_split.py --help
```

The frozen modeling population contained 5,179 products: 4,143 development products and 1,036 held-out products. Products sharing the same primary image-response hash remain in the same partition.

## 8. Extract image representations

```bash
python scripts/modeling/p9_visual_features.py --help
```

The frozen image representation stage generated 512-dimensional L2-normalized OpenCLIP ViT-B/32 embeddings, 20 classical raster descriptors, and 16 semantic image-text design-similarity scores (36 interpretable features in total). The semantic prompt bank and interpretable-feature schema are under `config/modeling/`.

## 9. Development-only model selection and refit

```bash
python scripts/modeling/p10_development_models.py --help
```

For each of three outcomes and two representation tracks, predictors were standardized and logistic regression was selected over the C grid `{0.01, 0.1, 1, 10, 100}` using five grouped development folds and mean validation average precision. The held-out set was not used for preprocessing, model selection, or fitting.

## 10. Held-out evaluation and completed AUROC audit

The publication-facing evaluator is:

```bash
python scripts/modeling/public_heldout_evaluation.py \
  --modeling-manifest data/derived/01_modeling_ready_manifest.csv \
  --predictions data/derived/02_locked_test_predictions.csv \
  --output results/heldout_metric_audit.json \
  --bootstrap-iterations 5000 \
  --seed 20260818
```

It consumes already-frozen held-out predictions and labels. It performs no model fitting or model selection and never flips scores. Average precision and top-k metrics use the frozen deterministic score-descending ranking convention; AUROC uses conventional `sklearn.metrics.roc_auc_score`, matching P10. Cluster-bootstrap uncertainty uses `primary_response_sha256` as the sampling unit.

The audit is complete. The frozen prediction SHA-256 matches the historical P11 ledger, and the original 5,000-draw/seed-`20260818` bootstrap plan was reproduced exactly. Conventional held-out AUROC is 0.6877-0.7220 across the six models, with all six corrected 95% interval lower bounds above 0.5. Exact values and provenance are under `data/published_results/p11_auroc_correction/` and `docs/AUROC_AUDIT.md`.

The historical P11 reference oracle and historical result files remain unchanged for audit provenance.

## 11. Post-lock interpretation

Historical P12 is governed by `config/modeling/p12_post_lock_interpretation_contract.json`. It combined development-fold coefficient-direction stability with outcome-level held-out evidence and did not retrain, refit, reselect, rescore, flip scores, change labels, change features, change the split, or select a threshold.

Frozen historical P12 outputs remain under `data/published_results/p12/`. Publication-facing AUROC interpretation is overlaid under `data/published_results/p12_auroc_correction/`. The E2/E1 enrichment grades and resulting Grade A/B design rules are unchanged because they are based on top-10% lift evidence, coefficient stability, and the QA-exclusion lift check rather than AUROC.

## 12. Figures

Figures 1 and 4 retain their frozen manuscript values. Figure 3 is regenerated with the bounded conventional-AUROC correction:

- `scripts/figures/figure01_data_construction_v5.py`
- `scripts/figures/figure03_recognition_performance_v7_auroc_corrected.py`
- `scripts/figures/figure04_design_strategies_v6.py`

The historical Figure 3 v6 script is retained for provenance but is not the publication-facing source after the AUROC audit. Figure 2 is a conceptual workflow diagram and is not a statistical output.

## Verification principle

Where exact row-level frozen artifacts are available in the publication-safe derived-data release, use their recorded SHA-256 values. Do not regenerate or reselect a frozen upstream stage merely to obtain a different downstream result. A correction of a verified metric-implementation defect must be bounded, documented, and based on the already-frozen predictions/labels.
