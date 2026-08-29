# PeerJ AI Application resubmission checklist

Before resubmission, confirm **every unchecked item below is resolved**.

## Repository and reproducibility

- [x] Public publication-facing code repository created.
- [x] Public machine-learning split, feature-extraction, model-selection/refit, held-out evaluation, interpretation, figure, and safe-export code is present.
- [x] Public publication-safe copies of the upstream P1–P7 preprocessing, validation, and image-workflow source modules are present.
- [x] README and reproducibility documentation point to the public upstream source paths.
- [x] Frozen upstream preprocessing/validation/image-workflow source identities are recorded by original Git blob SHA-1 in `PREPROCESSING_SOURCE_BINDINGS.md`.
- [x] The validation runtime mapping contract `config/affective_imagery/action_error_mapping_v21.json` is public.
- [x] Amazon Reviews'23 source URL and reference DOI are documented.
- [x] Computing environment is documented without inventing unrecorded hardware details.
- [x] Positive-unlabeled semantics are documented.
- [x] GPT-5.5 model-assisted validation disclosure is documented.
- [x] **PeerJ code-completeness gate:** publication-safe copies of the historical upstream preprocessing/validation/image workflow source modules are included in the public repository, with frozen-source provenance bindings. Raw third-party reviews/images remain excluded.
- [ ] Export the publication-safe frozen analysis data with `scripts/release/export_audit_inputs.py` and include the resulting files in `Supplemental_Data_1_PeerJ.zip` or a DOI-bearing archive. The frozen target is 5,179 modeling-manifest rows and 6,216 held-out prediction rows (1,036 products × 6 models).
- [ ] Archive the final code/data release in a DOI-bearing repository (for example Zenodo) **or** upload the code/data as PeerJ supplemental files, then enter the DOI/file information in the submission system.

## Manuscript

- [ ] Change/article-type language is consistent with **AI Application** throughout the resubmission where applicable.
- [ ] Materials & Methods states the computing infrastructure/environment.
- [ ] Materials & Methods gives the Amazon Reviews'23 source URL/DOI and identifies it as third-party data.
- [ ] Materials & Methods explicitly describes metadata filtering, review matching/cleaning, sentence extraction, visual-package screening, label construction, image acquisition/QA/freeze, split construction, feature extraction, and modeling preprocessing.
- [ ] Abstract spells out **area under the receiver operating characteristic curve (AUROC)** at first use.
- [ ] All manuscript AUROC values, confidence intervals, Figure 3 values, and interpretation agree with `docs/AUROC_AUDIT.md` and the publication-facing correction overlays.
- [ ] Data/code availability statement matches the final public archive/supplemental files exactly.

## PeerJ submission system

- [ ] Confidential Information for PeerJ Staff confirms agreement to switch the article type to **AI Application**.
- [ ] Confidential Information confirms whether the single-author submission is intentional.
- [ ] Authors, author order, corresponding/equal-first status, affiliations, contributions, funding, grants, and competing interests are complete and accurate **in the submission system**.
- [ ] Third-Party Data / Associated Data entry identifies Amazon Reviews'23 and provides the original source URL/DOI.
- [ ] Code/data repository DOI or supplemental-file details are entered in the appropriate Associated Data / code fields.
- [ ] Revised manuscript and submission-system metadata are cross-checked for consistency.

**Do not resubmit while any bold reproducibility/data gate above remains unresolved.** PeerJ explicitly requested both code/reproducibility materials and third-party data provenance before the manuscript can move to peer review.
