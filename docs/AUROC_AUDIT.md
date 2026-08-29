# Held-out AUROC consistency audit

## Why this audit exists

The P10 development-stage implementation computes AUROC with `sklearn.metrics.roc_auc_score(y_true, y_score)`. During construction of the publication-facing repository, the historical P11 reference-oracle implementation was found to use a different rank convention: it first orders observations by decreasing score and then applies a positive-rank Mann–Whitney expression whose orientation is opposite to conventional AUROC. Tied scores are also resolved by a deterministic identity key rather than average ranks.

The frozen historical P11/P12 files are retained unchanged as provenance. They must not be treated as the final publication AUROC authority until this bounded audit is completed.

## Bounded correction protocol

The correction is restricted to metric evaluation of already-frozen held-out scores. It must not retrain or refit a model, select a new model or regularization parameter, invert scores, choose a threshold, alter labels, alter visual features, alter the development/held-out split, or use corrected held-out results for model selection.

Required frozen inputs from the local research workspace:

- `data/processed/modeling_readiness_p8_5180/p8_b_modeling_ready_split/01_modeling_ready_manifest.csv`
- `data/processed/modeling_p11_5180/p11_locked_test_evaluation_freeze/02_locked_test_predictions.csv`

The public audit command is:

```bash
python scripts/modeling/public_heldout_evaluation.py \
  --modeling-manifest data/derived/01_modeling_ready_manifest.csv \
  --predictions data/derived/02_locked_test_predictions.csv \
  --output results/heldout_metric_audit.json \
  --bootstrap-iterations 5000 \
  --seed 20260818
```

The evaluator uses conventional `sklearn.metrics.roc_auc_score` for AUROC, preserves the frozen score orientation, uses deterministic score-descending ranking for AP/top-k metrics, and uses `primary_response_sha256` as the cluster-bootstrap unit.

## Publication reconciliation required after the audit

If corrected AUROC values or intervals differ from the frozen historical P11 values, reconcile every dependent publication artifact before PeerJ resubmission, including the Abstract, Results, Discussion, Conclusion, Figure 3, Table 3, Supplemental Information, P12 outcome-evidence wording, and any `below_half` global-ordering flag. Report the correction transparently in repository provenance.

Do not approximate corrected AUROC as `1 - historical_AUROC`: duplicate-image groups can generate tied scores, and conventional AUROC handles ties differently. Exact correction requires the frozen row-level predictions and labels.
