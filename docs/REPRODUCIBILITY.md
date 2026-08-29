# Reproducibility workflow

This document maps the frozen study workflow to the public code export. It is intentionally organized around the manuscript rather than the internal project-management history.

## 1. Obtain source data

Download the Amazon Reviews'23 Grocery metadata and review resources from the McAuley Lab source repository described in `docs/DATA_ACCESS.md`.

## 2. Screen eligible infusion products

Run the metadata-screening sequence:

```bash
python scripts/metadata/screen_full_metadata_v2.py --help
python scripts/metadata/secondary_clean_candidates_v21.py --help
python scripts/metadata/secondary_clean_candidates_v22.py --help
```

The study universe ultimately contained 5,180 eligible products. Metadata-review utilities and prompts are included where they affected the frozen eligibility workflow.

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

The validation protocol and schema are published under `docs/validation/`, with the associated deterministic validation/adjudication utilities under `scripts/validation/`.

The study used two independent, rule-guided ChatGPT (GPT-5.5; OpenAI)-assisted review passes on the same 840 validation tasks, followed by model-assisted adjudication. This is not human-gold annotation. Validation findings were used for reporting and did not change the frozen production labels via a sample-derived mask.

## 6. Acquire and freeze retail-package images

The P7 scripts under `scripts/images/` implement source-inventory audit, image acquisition, QA, and final frozen primary-image selection. Raw image binaries are not redistributed.

## 7. Create the leakage-controlled modeling split

```bash
python scripts/modeling/p8_b_modeling_ready_split.py --help
```

The frozen modeling population contained 5,179 products: 4,143 development products and 1,036 held-out products. Products sharing the same primary image-response hash remain in the same partition.

## 8. Extract image representations

```bash
python scripts/modeling/p9_visual_features.py --help
```

The frozen image representation stage generated:

- 512-dimensional L2-normalized OpenCLIP ViT-B/32 embeddings;
- 20 classical raster descriptors;
- 16 semantic image-text design-similarity scores;
- 36 interpretable features in total.

The semantic prompt bank and interpretable-feature schema are under `config/modeling/`.

## 9. Development-only model selection and refit

```bash
python scripts/modeling/p10_development_models.py --help
```

For each of three outcomes and two representation tracks, predictors were standardized and logistic regression was selected over the C grid `{0.01, 0.1, 1, 10, 100}` using five grouped development folds and mean validation average precision. The held-out set was not used for preprocessing, model selection, or fitting.

## 10. One-time held-out evaluation

```bash
python scripts/modeling/p11_locked_test_evaluation.py --help
```

The held-out evaluation reports average precision, AUROC, top-score recall/lift, cluster-bootstrap uncertainty, and prespecified sensitivity analyses. Score orientation is not flipped after seeing held-out AUROC.

## 11. Post-lock interpretation

P12 is governed by `config/modeling/p12_post_lock_interpretation_contract.json`. It combines development-fold coefficient-direction stability with outcome-level held-out evidence. The P12 materialization was a deterministic post-lock derivation and did not retrain, refit, reselect, rescore, flip scores, change labels, change features, change the split, or select a threshold.

Frozen non-sensitive P12 outputs are published under `data/published_results/p12/`.

## 12. Figures

Figures 1, 3, and 4 are generated from fixed manuscript/frozen-result values using the scripts in `scripts/figures/`. Figure 2 is a conceptual workflow diagram and is not a statistical output.

## Verification principle

Where the exact row-level frozen artifacts are available in the publication-safe derived-data release, use their recorded SHA-256 values and the stage-specific `--verify-existing`/verification modes in the original scripts. Do not regenerate or reselect a frozen upstream stage merely to obtain a different downstream result.
