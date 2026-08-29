# Multimodal machine learning for screening consumer-expressed affective imagery from retail packaging images and e-commerce reviews of herbal and fruit teas

Reproducibility repository accompanying the PeerJ Computer Science submission by **Kaiting Wu** (Jiangxi Institute of Fashion Technology).

## Purpose

This repository is the publication-facing reproducibility package for the manuscript. It contains the machine-learning analysis code, upstream preprocessing/validation/image-workflow source code, configuration files, validation protocol, provenance records, publication-facing metric correction, and non-sensitive derived artifacts that can be shared without redistributing raw third-party reviews or product images.

The study links publicly available Amazon Reviews'23 review/metadata resources to retail-package images and evaluates two visual representations under a positive-unlabeled (PU) observation framework:

- a 512-dimensional OpenCLIP ViT-B/32 image embedding;
- 36 interpretable visual features (20 classical raster descriptors + 16 semantic image-text design-similarity scores).

The two representations are evaluated with the same logistic-regression model family for three observed-positive outcomes: any outer-package affective imagery, general visual appeal, and cute/friendly imagery.

## Public source data

The third-party source dataset is **Amazon Reviews'23**, released by the UCSD McAuley Lab:

- Dataset repository: https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023
- Project documentation: https://amazon-reviews-2023.github.io/
- Reference: Hou Y, Li J, He Z, Yan A, Chen X, McAuley J. 2024. *Bridging Language and Items for Retrieval and Recommendation*. arXiv:2403.03952. https://doi.org/10.48550/arXiv.2403.03952

This repository does **not** redistribute raw Amazon review text or third-party image binaries. Users should obtain those materials from the original source subject to its terms.

## Repository status and AUROC correction

This public repository is a publication-facing export from the frozen research codebase. Source snapshot used for provenance: `1d72e3ba798f93d328c37bfe75b37032b9f21246`.

During publication preparation, a metric-consistency audit found that the historical P11 reference oracle used a reversed rank orientation for AUROC relative to the conventional P10/scikit-learn definition. The audit was completed from the already-frozen held-out scores and labels. No model was retrained or reselected, scores were not inverted, and no label, feature, threshold, or split was changed.

Conventional held-out AUROC is **0.6877-0.7220** across the six frozen models; all six corrected 95% cluster-bootstrap intervals have lower bounds above 0.5. The historical P11/P12 files are preserved unchanged as provenance, while publication-facing correction overlays are provided under:

- `data/published_results/p11_auroc_correction/`
- `data/published_results/p12_auroc_correction/`
- `docs/AUROC_AUDIT.md`

Average precision and top-k recall/lift retain the frozen deterministic ranking definitions and values. P12 E2/E1 enrichment grades and Grade A/B design rules are unchanged because they are driven by top-10% lift, coefficient stability, and the prespecified QA-exclusion check rather than AUROC.

## Computing environment

The frozen manuscript analyses were run under:

- Windows 11 (`Windows-11-10.0.26200-SP0`)
- Python 3.12.7
- CPU-only execution
- NumPy 1.26.4
- SciPy 1.13.1
- pandas 2.2.2
- scikit-learn 1.9.0
- joblib 1.4.2
- threadpoolctl 3.5.0
- Pillow 10.4.0
- `open_clip_torch` 3.3.0
- PyTorch 2.13.0+cpu
- OpenCLIP model: `ViT-B-32`
- OpenCLIP pretrained tag: `laion2b_s34b_b79k`
- OpenCLIP inference device: CPU; batch size 16

Install the Python dependencies with:

```bash
python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

See `docs/ENVIRONMENT.md` for the recorded runtime and `docs/REPRODUCIBILITY.md` for workflow details.

## Public runnable upstream data-construction workflow

The publication checkout now includes the P1–P7 source implementation used to construct the study inputs. Exact public-path ↔ frozen-source bindings are recorded in `docs/PREPROCESSING_SOURCE_BINDINGS.md`.

1. **Metadata eligibility screening**
   - `scripts/metadata/screen_full_metadata_v2.py`
   - `scripts/metadata/secondary_clean_candidates_v21.py`
   - `scripts/metadata/secondary_clean_candidates_v22.py`
2. **Review matching and sentence extraction**
   - `scripts/reviews/match_reviews_to_valid_products.py`
   - `scripts/reviews/clean_and_extract_packaging_sentences.py`
3. **Strict visual-package screening and affective-imagery labels**
   - `scripts/visual_packaging/strict_visual_packaging_classifier_v11.py`
   - `scripts/labels/build_affective_imagery_labels_v21.py`
4. **Rule-guided validation/adjudication**
   - implementation under `scripts/validation/`
   - runtime mapping contract: `config/affective_imagery/action_error_mapping_v21.json`
5. **P7 image source audit, acquisition, QA, and freeze**
   - `scripts/images/audit_p7_source_inventory.py`
   - `scripts/images/acquire_p7_primary_assets.py`
   - `scripts/images/p7_c_primary_manifest_qa.py`
   - `scripts/images/p7_d_final_image_freeze.py`
   - frozen contracts under `config/image_assets/`

These source files can be inspected and invoked from this checkout. Exact end-to-end regeneration of the historical upstream artifacts additionally requires the original Amazon Reviews'23 source files and third-party image assets, which are intentionally not redistributed here.

## Public runnable machine-learning workflow

The publication-facing checkout directly contains the code used for the leakage-controlled modeling and evaluation stages:

1. **Analysis contract and modeling-ready split**
   - `scripts/modeling/p8_a_analysis_contract.py`
   - `scripts/modeling/p8_b_modeling_ready_split.py`
2. **OpenCLIP and interpretable visual feature extraction**
   - `scripts/modeling/p9_visual_features.py`
3. **Development-only model selection/refit**
   - `scripts/modeling/p10_development_models.py`
4. **Held-out evaluation**
   - publication-facing evaluator: `scripts/modeling/public_heldout_evaluation.py`
   - completed metric audit: `docs/AUROC_AUDIT.md`
   - historical audit/reference modules: `scripts/modeling/p11_locked_test_reference_oracle.py`, `scripts/modeling/p11_locked_test_reference_oracle_legacy.py`, `scripts/modeling/p11_locked_test_repair.py`
5. **Post-lock interpretation**
   - historical frozen rules: `config/modeling/p12_post_lock_interpretation_contract.json`
   - historical frozen outputs: `data/published_results/p12/`
   - publication-facing AUROC overlay: `data/published_results/p12_auroc_correction/`
6. **Publication-safe data export**
   - `scripts/release/export_audit_inputs.py`
7. **Manuscript figures**
   - `scripts/figures/figure01_data_construction_v5.py`
   - corrected Figure 3: `scripts/figures/figure03_recognition_performance_v7_auroc_corrected.py`
   - historical Figure 3 source retained for provenance: `scripts/figures/figure03_recognition_performance_v6.py`
   - `scripts/figures/figure04_design_strategies_v6.py`

The workflow deliberately preserves the development/held-out firewall. Held-out results must not be used for model selection, threshold selection, score inversion, or feature selection.

## Positive-unlabeled semantics

A label of `1` means that a qualifying consumer-expressed affective-imagery mention was observed. A label of `0` means **unobserved/unlabeled**, not confirmed absence of the underlying impression. Model scores are therefore interpreted as rankings or propensities for observed qualifying mentions, not calibrated probabilities of a latent psychological state.

## Model-assisted validation disclosure

The validation sample comprised 600 sentence-level tasks and 240 product-dimension tasks. The same 840 tasks were reviewed in two separate, independent **ChatGPT (GPT-5.5; OpenAI)-assisted** contexts under a predefined annotation schema, followed by rule-guided model-assisted adjudication. This model/version identification is author-confirmed for the manuscript disclosure. The process is **not** described as independent human-gold annotation. Validation findings were used for quality reporting and did not create a sample-derived production-label mask.

## Reproducibility boundaries

The repository publishes the scientific source code needed to inspect the upstream P1–P7 data-construction logic and rerun the publication-facing modeling/evaluation logic when required source inputs are available, together with configuration and provenance materials. It excludes raw Amazon review text, raw third-party product images, image-source URLs, the large third-party/OpenCLIP checkpoint binary, credentials/API responses/runtime logs, internal collaboration-history documents unrelated to the manuscript, and personally identifying reviewer/account information.

See `docs/DATA_ACCESS.md` for the exact data boundary, `docs/PREPROCESSING_SOURCE_BINDINGS.md` for frozen upstream source-code bindings, and `docs/SOURCE_PROVENANCE.md` for source-snapshot provenance.

## Citation and archival DOI

If this repository is used before the article receives a final bibliographic record, cite the manuscript title above, author Kaiting Wu, and this GitHub repository. For PeerJ resubmission, archive the final release in a DOI-bearing repository (for example, Zenodo) or upload the code as a PeerJ supplemental file, then enter the resulting DOI/file information in the submission system.
