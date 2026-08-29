# P12 AUROC interpretation correction overlay

The files under `data/published_results/p12/` are retained unchanged as the historical post-lock interpretation freeze. During publication preparation, a bounded audit established that the historical P11 AUROC field had reversed rank orientation. This directory supplies the publication-facing AUROC interpretation overlay without silently rewriting the historical P12 artifacts.

## Corrected outcome-level interpretation

Conventional held-out AUROC, computed from the unchanged frozen scores with `sklearn.metrics.roc_auc_score`, is above 0.5 for all three interpretable models, and each corrected 95% cluster-bootstrap interval has a lower bound above 0.5. Accordingly, the historical P12 `below_half` global-ordering flag is replaced for publication purposes by `above_half_interval_supported`.

## What does not change

P12 design grades remain unchanged. The prespecified E2/E1 outcome enrichment grades are based on top-10% lift and its interval, and design grades combine those enrichment grades with development-fold coefficient-direction stability and the QA-exclusion lift check. The bounded AUROC correction does not alter coefficients, stability counts, AP, lift, feature-level validation status, or causal interpretation guards.

Therefore:

- Any outer-package affective imagery: E2; Grade A rules unchanged.
- General visual appeal: E2; Grade A rules unchanged.
- Cute/friendly: E1; Grade B rules unchanged.

Any historical claim text stating that an AUROC interval remained wholly below 0.5 should be read as superseded by the corrected values in `01_outcome_evidence_auroc_corrected.csv` and the P11 correction provenance under `data/published_results/p11_auroc_correction/`.
