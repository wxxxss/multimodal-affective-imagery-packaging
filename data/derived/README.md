# Publication-safe derived data

This directory is reserved for derived artifacts exported from the frozen local research workspace. It must not contain raw Amazon review text or copyrighted product-image binaries.

## Immediate AUROC-audit inputs

The following two frozen files are required first:

1. `01_modeling_ready_manifest.csv`
   - source: `data/processed/modeling_readiness_p8_5180/p8_b_modeling_ready_split/01_modeling_ready_manifest.csv`
   - contains the frozen modeling population, split/group identities, and observed-positive outcome fields.
2. `02_locked_test_predictions.csv`
   - source: `data/processed/modeling_p11_5180/p11_locked_test_evaluation_freeze/02_locked_test_predictions.csv`
   - contains the already-frozen held-out scores for the six models.

These files permit a bounded metric audit without model refitting or redistribution of raw review text/images.

## Additional PeerJ reproducibility artifacts

A fuller publication deposit should subsequently include publication-safe P8/P9/P10/P11 derived artifacts needed to reproduce the frozen split, image feature mappings, model parameters, and evaluation outputs. Exact source paths and SHA-256 bindings are recorded in the stage contracts under `config/modeling/`.
