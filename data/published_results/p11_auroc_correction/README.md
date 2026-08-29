# P11 AUROC correction overlay

This directory records a bounded correction to the held-out AUROC implementation discovered during preparation of the PeerJ reproducibility release.

The original frozen P11 artifacts are preserved for audit traceability. They are not silently rewritten here.

## What changed

Only AUROC point estimates and AUROC cluster-bootstrap intervals are corrected. The same frozen held-out scores, labels, split, duplicate-image grouping, bootstrap seed/draw plan, and model parameters are used. No model was retrained or reselected, and scores were not inverted after seeing held-out performance.

The legacy P11 oracle sorted observations by descending score and then applied a Mann-Whitney rank-sum expression whose conventional orientation assumes ascending ranks. This produced the opposite AUROC orientation. The correction uses `sklearn.metrics.roc_auc_score`, matching the P10 development-stage AUROC definition.

## What did not change

- positive-unlabeled outcome semantics;
- frozen model scores and parameters;
- development/held-out partition;
- average precision values as reported under the documented deterministic held-out ranking order;
- recall/lift values and their bootstrap intervals;
- P12 enrichment grades E2/E1, because those grades are based on top-10% lift evidence rather than AUROC;
- feature coefficients, coefficient stability, or causal/feature-level interpretation guards.

## Consequence for interpretation

The previous claim that all held-out AUROC values were below 0.5 is invalid. Conventional held-out AUROC is approximately 0.688–0.722 across the six frozen models, with the six corrected 95% intervals above 0.5. The previously described development-to-held-out AUROC reversal was therefore an evaluation-implementation artifact rather than evidence of a domain-generalization reversal.

`01_auroc_correction.csv` contains the corrected six-model values and `02_correction_provenance.json` records the frozen-input and bootstrap bindings.
