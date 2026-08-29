# Data access and redistribution boundary

## Third-party source data

The study used Amazon Reviews'23 Grocery metadata and review records released by the UCSD McAuley Lab.

- Hugging Face dataset: https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023
- Project documentation: https://amazon-reviews-2023.github.io/
- Reference DOI: https://doi.org/10.48550/arXiv.2403.03952

No accession number is assigned by the source repository.

## What is not redistributed

Raw Amazon review text, third-party product-image binaries, image-source URLs, credentials, API responses, and internal account/reviewer information are not redistributed. Those materials remain subject to the original source terms and/or are unnecessary to check the reported machine-learning results.

## PeerJ Supplemental Data 1

The resubmission includes a separate publication-safe dataset, `Supplemental_Data_1_PeerJ.zip`. It contains:

- the 1,036-row held-out product manifest restricted to identifiers, the frozen image-response group hash, and the three modeled positive-unlabeled outcomes;
- the 6,216 frozen held-out model scores (1,036 products × 6 frozen models);
- a convenience joined labels-and-scores table;
- the conventional AUROC correction table and audit provenance;
- SHA-256 checksums.

The package deliberately excludes raw review text, copyrighted image binaries, image URLs/paths, embeddings, and credentials. The public GitHub repository contains the code needed to recompute the publication-facing held-out metrics from this supplemental package.

## Reproducibility scope

The public repository is centered on the manuscript's frozen machine-learning analysis: leakage-controlled split contracts, visual feature extraction specifications, development-only model selection/refit, held-out evaluation, post-lock interpretation, and figures. Raw-source preprocessing rules are described in the manuscript and validation documentation, with frozen source-code bindings recorded in `docs/PREPROCESSING_SOURCE_BINDINGS.md`.

Exact regeneration of all upstream row-level intermediates from raw Amazon records would additionally require the third-party source files and product images under their original terms. Exact checking of the reported held-out ranking results does not require redistribution of those raw materials because the frozen publication-safe labels, group hashes, and scores are supplied as Supplemental Data 1.

## Positive-unlabeled observation semantics

`1` denotes an observed qualifying consumer mention. `0` denotes unobserved/unlabeled status and must not be interpreted as a confirmed negative.
