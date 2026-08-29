# Held-out AUROC consistency audit

## Status

**Completed.** The bounded audit used the already-frozen P8-B modeling manifest and P11 locked-test predictions. No model was retrained or reselected, no score was inverted, and no label, feature, threshold, or split was changed.

## Root cause

The P10 development-stage implementation computes AUROC with `sklearn.metrics.roc_auc_score(y_true, y_score)`. The historical P11 reference oracle first ordered observations by decreasing score and then applied a positive-rank Mann-Whitney expression whose conventional orientation assumes ascending ranks. This reversed the AUROC orientation. Exact score ties were additionally assigned deterministic identity-key ranks instead of the conventional tie-aware ranking used by `roc_auc_score`.

The frozen historical P11/P12 files remain unchanged as provenance. Publication-facing AUROC values are supplied through explicit correction overlays.

## Frozen-input verification

- P8-B manifest SHA-256: `4CC5D4389BE9FEC94CAAC0601A24A49841D3EAB0D0787BAE8A8DD4ACF52B86E9`
- P11 locked-test predictions SHA-256: `F927730318275EF33EF7C74251E6FAAED67512A20C4678C493AF14C65E85336F`
- The prediction SHA exactly matches the historical P11 formal ledger.
- Modeling population: 5,179 products; 4,143 development and 1,036 held-out.
- Held-out primary-image-response groups: 838.
- Held-out observed-positive counts: 46 / 30 / 17 for any imagery / general visual appeal / cute-friendly.
- Prediction rows: 6,216 = 1,036 products x 6 frozen models.

The original bootstrap plan was reproduced exactly: 5,000 PCG64 draws, seed `20260818`, 838 groups, plan SHA-256 `26DC4E42EC8B23E1CAFD1A0D432EE327A31197DFE6BC5F0B1B7EC8122C9DB32E`.

## Corrected conventional held-out AUROC

| Outcome | Track | Historical P11 AUROC | Corrected AUROC | Corrected 95% CI |
| --- | --- | ---: | ---: | ---: |
| Any imagery | OpenCLIP | 0.302679 | 0.697420 | 0.617048-0.773673 |
| Any imagery | Interpretable-36 | 0.312385 | 0.687714 | 0.613409-0.758554 |
| General visual appeal | OpenCLIP | 0.278661 | 0.721322 | 0.632209-0.805646 |
| General visual appeal | Interpretable-36 | 0.283433 | 0.716551 | 0.624829-0.805560 |
| Cute/friendly | OpenCLIP | 0.292097 | 0.708134 | 0.547545-0.844798 |
| Cute/friendly | Interpretable-36 | 0.278185 | 0.722046 | 0.583065-0.840347 |

All six corrected point estimates and all six lower confidence bounds exceed 0.5. This is not a post-hoc score flip; the conventional AUROC definition was applied to the unchanged frozen scores.

## Other metrics and sensitivity checks

Average precision and top-k recall/lift were independently reconstructed under the documented deterministic score-descending/product-key ranking convention and remain unchanged. The historical AP implementation intentionally resolves exact score ties deterministically and therefore need not equal `sklearn.average_precision_score` for tied duplicate-image scores.

Across 838 unique primary-image-response groups, conventional group-level AUROC ranged from approximately 0.6823 to 0.7181. In the prespecified R1 exclusion of 51 known held-out image-QA rows, primary OpenCLIP conventional AUROC was 0.6985 versus 0.6974 in the main held-out analysis; AP and lift remained on their frozen definitions.

## Publication reconciliation

The correction overlays are:

- `data/published_results/p11_auroc_correction/`
- `data/published_results/p12_auroc_correction/`
- corrected Figure 3 source: `scripts/figures/figure03_recognition_performance_v7_auroc_corrected.py`

The manuscript must use the corrected AUROC values in the Abstract, Results, Discussion, Conclusion, Figure 3, Table 3, and Supplemental Information. The previous claim of a development-to-held-out AUROC reversal is superseded because it was caused by the metric implementation defect, not by the frozen scores.

P12 E2/E1 enrichment grades and Grade A/B design interpretation remain unchanged because those grades are driven by top-10% lift evidence, coefficient-direction stability, and the QA-exclusion lift check rather than AUROC.

Do not approximate the correction as `1 - historical_AUROC`; the exact values above were recomputed from the frozen row-level scores and labels using conventional tie-aware AUROC.
