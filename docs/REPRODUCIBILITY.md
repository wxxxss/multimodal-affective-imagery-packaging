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

## 10. Held-out evaluation and AUROC audit

The publication-facing evaluator is:

```bash
python scripts/modeling/public_heldout_evaluation.py \
  --modeling-manifest data/derived/01_modeling_ready_manifest.csv \
  --predictions data/derived/02_locked_test_predictions.csv \
  --output results/heldout_metric_audit.json \
  --bootstrap-iterations 5000 \
  --seed 20260818
```

It consumes already-frozen held-out predictions and labels. It performs no model fitting or model selection and never flips scores. Average precision and top-k metrics use the frozen deterministic score-descending ranking convention; AUROC uses conventional `sklearn.metrics.roc_auc_score`, matching the P10 definition. Cluster-bootstrap uncertainty uses `primary_response_sha256` as the sampling unit.

A metric-consistency audit is required because the historical P11 reference oracle used a different AUROC rank orientation. See `docs/AUROC_AUDIT.md`. The historical P11/P12 files are retained unchanged for provenance until the audit is reconciled with the manuscript.

## 11. Post-lock interpretation

Historical P12 is governed by `config/modeling/p12_post_lock_interpretation_contract.json`. It combined development-fold coefficient-direction stability with outcome-level held-out evidence and did not retrain, refit, reselect, rescore, flip scores, change labels, change features, change the split, or select a threshold.

Frozen historical non-sensitive P12 outputs are published under `data/published_results/p12/`. Any AUROC-dependent P12 language or grade flag must be reconciled after the bounded AUROC audit; feature coefficients themselves are not being re-estimated by that audit.

## 12. Figures

Figures 1, 3, and 4 are generated from fixed manuscript/frozen-result values using the scripts in `scripts/figures/`. Figure 2 is a conceptual workflow diagram and is not a statistical output. Figure 3 must be regenerated if the AUROC audit changes held-out AUROC values or intervals.

## Verification principle

Where exact row-level frozen artifacts are available in the publication-safe derived-data release, use their recorded SHA-256 values. Do not regenerate or reselect a frozen upstream stage merely to obtain a different downstream result. A correction of a verified metric-implementation defect must be bounded, documented, and based on the already-frozen predictions/labels.
