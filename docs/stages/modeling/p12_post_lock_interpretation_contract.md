# P12 Post-Lock Interpretation Contract

Status: CONTRACT-ONLY / SCIENTIFIC INTERPRETATION NOT EXECUTED

Baseline: `main@061e4f01832bf5a46eafed3a7c41cf468269581b`

Working branch: `p12-post-lock-interpretation`

## Purpose

P12 synthesizes the already frozen P9 interpretable packaging-design features, P10 development-stage coefficient-stability evidence, and P11 locked-test, bootstrap, and R1-R5 evidence into an evidence-graded manuscript interpretation layer.

P12 does not retrain or reselect models and does not change frozen outcomes, labels, image exposure, features, split assignments, or thresholds.

## Frozen Input Bindings

P9 contributes 20 classical and 16 semantic design-similarity features, for 36 interpretable visual features in total.

P10 contributes the frozen coefficient-stability artifact:

`data/processed/modeling_p10_5180/p10_development_model_freeze/06_interpretable_coefficient_stability.csv`

Expected SHA-256:

`2D9CE00D4EE9FF37A7D23AC28D80429E5B8AFE1CAC81DA9CC49B7F018EFC870D`

Expected rows:

`108 = 3 outcomes x 36 features`

The actual 108 coefficient rows were not read during P12-B contract freeze.

P11 contributes frozen product-level metrics, 5000-iteration cluster-bootstrap uncertainty, and R1-R5 robustness evidence.

P11 external provenance SHA-256:

`CFB46FB09B5FEC30E8F858A2E4D8B56D06B76EA958F0D0F5A00A5AC2DE253A49`

## Coefficient-Stability Rules

For each outcome-feature row, compare the sign of the final development standardized coefficient with the five selected-C development-fold coefficient signs.

- 5/5 = directionally stable.
- 4/5 = mostly stable.
- 0-3/5 = unstable.

`final_standardized_coef` is a multivariable regularized development-stage standardized logistic coefficient. It is association evidence, not a causal effect.

`abs_final_coef` is descriptive salience/ranking only. It is not a p-value, significance statistic, marginal effect, or causal effect size.

All 36 features per outcome remain in the complete supplementary ledger.

The manuscript main coefficient figure may show only 5/5 directionally stable features. If more than eight such features occur for an outcome, display the eight largest by `abs_final_coef`. This cap is frozen before reading the actual coefficient values.

## Locked-Test Outcome Evidence

Outcome-level evidence uses frozen P11 `lift_at_top10` and its 95% `primary_response_sha256` cluster-bootstrap percentile interval.

- E2: point estimate > 1 and CI lower bound > 1.
- E1: point estimate > 1 and the 95% CI includes 1.
- E0: point estimate <= 1.

AUROC is recorded separately as a global-ordering flag:

- below-half: CI upper bound < 0.5;
- indeterminate: CI includes 0.5;
- above-half: CI lower bound > 0.5.

Top-tail enrichment and AUROC/global-ordering evidence must be reported together. Score flipping is forbidden.

## R1-R5 Boundaries

### R1

Known-QA-exception exclusion may contribute eligibility context. Enrichment is preserved when frozen R1 `lift_at_top10 > 1`. No post hoc percentage-change tolerance may be invented.

### R2

Alternative frozen image exposure is descriptive caveat evidence only. It does not determine design grade and cannot trigger retraining.

### R3

Review-exposure/G sensitivity is heterogeneity/context evidence only.

Frozen thresholds:

`0, 1, 3, 5, 10, 20, 50, 100`

Frozen strata:

`0, 1-2, 3-4, 5-9, 10-19, 20-49, 50-99, 100+`

Performance-best threshold selection and favorable subgroup promotion are forbidden. `insufficient_class_support` must remain explicit.

### R4

Core remains the primary frozen label definition. Pilot and robust variants remain robustness context only. Promotion and relabeling are forbidden.

### R5

Image-exception diagnostics remain descriptive only and carry no inferential claim. R5 cannot determine design grade.

## Design-Hypothesis Grades

### Grade A — Qualified Design Hypothesis

Requires:

1. 5/5 feature sign stability;
2. E2 outcome evidence;
3. R1 `lift_at_top10 > 1`.

An appropriate AUROC/global-ordering caveat remains mandatory.

### Grade B — Exploratory Design Hypothesis

Either:

- 5/5 stability + E1; or
- 4/5 stability + E2.

Tentative language is mandatory.

### Grade C — Development-Stage Association Only

Feature has at least 4/5 sign agreement but does not meet Grade A or B.

Grade C cannot form a core Practical Applications recommendation.

### Grade D — Do Not Highlight

Feature has 0-3/5 sign agreement.

Grade D remains in the complete supplementary ledger but is not highlighted in the manuscript narrative.

## Feature-Level Locked-Validation Firewall

P11 validates complete frozen model behavior at the outcome level. It does not validate an individual feature effect on the locked test.

Forbidden wording includes claims that a feature:

- was validated on the locked test;
- predicts the locked-test outcome by itself;
- increases or decreases consumer perception;
- causes affective imagery;
- makes consumers feel a particular state.

Causal packaging recommendations are forbidden.

Allowed wording includes:

- associated with;
- directionally stable development-stage visual correlate;
- observed-positive propensity;
- outcome-level locked-test top-tail enrichment;
- qualified design hypothesis;
- exploratory design hypothesis.

## No-P-Value / No-BH-Posthoc Boundary

P10 reserved a two-outcome confirmatory family with `BH q = 0.05`, but frozen P11 did not generate p-values or execute a BH procedure.

P12 therefore may not create a new p-value or BH significance pipeline post hoc.

Any future p-value or BH analysis requires a new frozen contract and separate explicit authorization.

Current P12 evidence grading is restricted to frozen ranking metrics, frozen cluster-bootstrap uncertainty, and predeclared R1-R5 robustness evidence.

## Planned Future Scientific Outputs

When separately authorized, later P12 scientific execution may create:

1. `01_p12_input_binding.json`
2. `02_feature_interpretation_ledger.csv`
3. `03_outcome_evidence_grade.csv`
4. `04_design_hypothesis_matrix.csv`
5. `05_manuscript_claim_ledger.csv`
6. `06_p12_summary.json`
7. `07_p12_provenance.json`

These outputs are not created by P12-B.

## Manuscript Destinations

Primary destinations:

- Results 3.4 — Interpretable Visual Features and Evidence-Based Design Implications.
- Discussion 4.3 — Implications for Packaging Design.

Later P12 synthesis may also support final revisions to Abstract, Practical Applications, Discussion 4.1, and Conclusion.

## Current Authorization Boundary

P12-B does not authorize:

- reading the actual 108 coefficient row values;
- scientific interpretation;
- model fitting or refitting;
- model reselection;
- locked-test rescoring;
- score flipping;
- outcome, label, feature, split, or threshold changes;
- modification of P7-P11 artifacts;
- commit;
- push;
- merge.

A separate explicit authorization is required before P12 scientific interpretation begins.
