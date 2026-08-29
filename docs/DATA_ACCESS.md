# Data access and redistribution boundary

## Third-party source data

The study used Amazon Reviews'23 Grocery metadata and review records released by the UCSD McAuley Lab.

- Hugging Face dataset: https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023
- Project documentation: https://amazon-reviews-2023.github.io/
- Reference DOI: https://doi.org/10.48550/arXiv.2403.03952

No accession number is assigned by the source repository.

## What is not redistributed here

This repository does not redistribute raw review text, raw product-image binaries, API responses, credentials, or internal account/reviewer information. Those materials remain subject to the original source terms and/or are unnecessary for publication.

## Derived materials needed for exact frozen-result reproduction

Exact reproduction of the manuscript model outputs requires a de-identified analysis-ready package containing, at minimum:

1. the final eligible `parent_asin` inventory and observed-positive outcome fields;
2. the frozen development/held-out split and grouped-fold assignments;
3. the frozen primary-image identifiers/hashes and analysis mappings;
4. the 512-dimensional OpenCLIP embeddings or sufficient image references to recompute them;
5. the 36 interpretable visual features;
6. development model-selection outputs/final parameter arrays;
7. held-out metric and bootstrap/sensitivity outputs used in the manuscript.

The internal project kept many of these formal row-level artifacts outside Git. A publication-safe export of those derived materials must therefore be added to this repository (or archived separately with a DOI) before PeerJ resubmission. Raw consumer review text and copyrighted image binaries need not be included in that export.

## Positive-unlabeled observation semantics

`1` denotes an observed qualifying consumer mention. `0` denotes unobserved/unlabeled status and must not be interpreted as a confirmed negative.
