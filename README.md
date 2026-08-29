# Multimodal machine learning for screening consumer-expressed affective imagery from retail packaging images and e-commerce reviews of herbal and fruit teas

Reproducibility repository accompanying the PeerJ Computer Science submission by **Kaiting Wu** (Jiangxi Institute of Fashion Technology).

## Purpose

This repository contains the code, configuration files, validation protocol, and non-sensitive derived artifacts used in the study. The work links publicly available Amazon Reviews'23 review/metadata resources to retail-package images and evaluates two visual representations under a positive-unlabeled (PU) observation framework:

- a 512-dimensional OpenCLIP ViT-B/32 image embedding;
- 36 interpretable visual features (20 classical raster descriptors + 16 semantic image-text design-similarity scores).

The two representations are evaluated with the same logistic-regression model family for three observed-positive outcomes: any outer-package affective imagery, general visual appeal, and cute/friendly imagery.

## Public source data

The third-party source dataset is **Amazon Reviews'23**, released by the UCSD McAuley Lab:

- Dataset repository: https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023
- Project documentation: https://amazon-reviews-2023.github.io/
- Reference: Hou Y, Li J, He Z, Yan A, Chen X, McAuley J. 2024. *Bridging Language and Items for Retrieval and Recommendation*. arXiv:2403.03952. https://doi.org/10.48550/arXiv.2403.03952

This repository does **not** redistribute raw Amazon review text or third-party image binaries. Users should obtain those materials from the original source subject to its terms.

## Repository status

This public repository is a clean reproducibility export from the frozen research codebase used for the manuscript. The internal research repository remains private and is not required to understand the publication-facing workflow.

Source snapshot used for this export: `1d72e3ba798f93d328c37bfe75b37032b9f21246`.

Frozen scientific source code, contracts, validation documentation, aggregate modeling summaries, and non-sensitive post-lock interpretation outputs are published here. Row-level intermediate artifacts that were intentionally gitignored in the research repository are not reconstructed or fabricated in this Git-only export. A publication-safe row-level derived-data deposit must therefore be exported separately from the frozen local research workspace before PeerJ resubmission; raw review text and copyrighted image binaries will not be redistributed.

**Metric-consistency audit before resubmission.** During preparation of this public release, an inconsistency was identified between the conventional P10 development-stage AUROC (`sklearn.metrics.roc_auc_score`) and the historical P11 reference-oracle rank formula. The historical frozen P11/P12 materials are retained as audit evidence, but held-out AUROC and its bootstrap interval must be revalidated from the already-frozen predictions and labels before the manuscript is resubmitted. No model refitting, model reselection, score inversion, threshold selection, label change, feature change, or split change is authorized by this audit. The publication-facing evaluator is `scripts/modeling/public_heldout_evaluation.py`.

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

## Manuscript workflow mapped to code

1. **Metadata screening**
   - `scripts/metadata/screen_full_metadata_v2.py`
   - `scripts/metadata/secondary_clean_candidates_v21.py`
   - `scripts/metadata/secondary_clean_candidates_v22.py`
2. **Review matching and preprocessing**
   - `scripts/reviews/match_reviews_to_valid_products.py`
   - `scripts/reviews/clean_and_extract_packaging_sentences.py`
3. **Visual-package language screening**
   - `scripts/visual_packaging/strict_visual_packaging_classifier_v11.py`
4. **Affective-imagery label construction**
   - `build_affective_imagery_labels_v21.py`
5. **Rule-guided validation/adjudication utilities**
   - `scripts/validation/`
   - `docs/validation/`
6. **Image acquisition, QA, and frozen primary-image selection**
   - `scripts/images/`
7. **Analysis contract and modeling-ready split**
   - `scripts/modeling/p8_a_analysis_contract.py`
   - `scripts/modeling/p8_b_modeling_ready_split.py`
8. **OpenCLIP and interpretable visual feature extraction**
   - `scripts/modeling/p9_visual_features.py`
9. **Development-only model selection/refit**
   - `scripts/modeling/p10_development_models.py`
10. **Locked held-out evaluation**
   - publication-facing metric implementation: `scripts/modeling/public_heldout_evaluation.py`
   - historical audit/reference modules: `scripts/modeling/p11_locked_test_reference_oracle.py`, `scripts/modeling/p11_locked_test_reference_oracle_legacy.py`, `scripts/modeling/p11_locked_test_repair.py`
11. **Post-lock interpretation**
   - frozen historical rules: `config/modeling/p12_post_lock_interpretation_contract.json`
   - frozen historical non-sensitive outputs: `data/published_results/p12/`
   - these are retained for provenance and will be reconciled with the AUROC audit before resubmission
12. **Manuscript figures**
   - `scripts/figures/figure01_data_construction_v5.py`
   - `scripts/figures/figure03_recognition_performance_v6.py`
   - `scripts/figures/figure04_design_strategies_v6.py`

The internal P11 implementation contains integrity bindings to the original research-repository Git history and locally frozen upstream artifacts. Those bindings are retained as audit evidence; they are not a claim that the historical P11 transaction can be rerun from this public Git checkout alone without the corresponding frozen local artifacts.

The workflow deliberately preserves the development/held-out firewall. Held-out results must not be used for model selection, threshold selection, score inversion, or feature selection.

## Positive-unlabeled semantics

A label of `1` means that a qualifying consumer-expressed affective-imagery mention was observed. A label of `0` means **unobserved/unlabeled**, not confirmed absence of the underlying impression. Model scores are therefore interpreted as rankings or propensities for observed qualifying mentions, not calibrated probabilities of a latent psychological state.

## Model-assisted validation disclosure

The validation sample comprised 600 sentence-level tasks and 240 product-dimension tasks. The same 840 tasks were reviewed in two separate, independent **ChatGPT (GPT-5.5; OpenAI)-assisted** contexts under a predefined annotation schema, followed by rule-guided model-assisted adjudication. This model/version identification is author-confirmed for the manuscript disclosure. The process is **not** described as independent human-gold annotation. Validation findings were used for quality reporting and did not create a sample-derived production-label mask.

## Reproducibility boundaries

The repository publishes scientific code and non-sensitive configuration/provenance materials. It excludes:

- raw Amazon review text;
- raw third-party product images;
- the large third-party/OpenCLIP checkpoint binary;
- credentials, API responses, and runtime logs;
- internal collaboration/history documents unrelated to reproducing the manuscript;
- personally identifying reviewer/account information.

See `docs/DATA_ACCESS.md` for the exact data-access boundary and `docs/SOURCE_PROVENANCE.md` for source-snapshot bindings.

## Citation

If this repository is used before the article receives a final bibliographic record, cite the manuscript title above, author Kaiting Wu, and this GitHub repository. A versioned archival DOI can be added after the reproducibility release is finalized.
