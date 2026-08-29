# P12 Post-Lock Interpretation Report

Status: MATERIALIZED / NOT COMMITTED

Contract commit: `f60cdd7e305bbcba962653cccd68111a0f41b0f0`

Baseline main commit: `061e4f01832bf5a46eafed3a7c41cf468269581b`

## Interpretation Boundary

P11 validates complete frozen model behavior at the outcome level. It does not validate any individual feature effect on the locked test.

All feature coefficients below are multivariable standardized development-stage associations. They are not causal effects, marginal effects, p-values, or feature-level locked-test validations.

Semantic feature scores are frozen image-text similarity measures, not probabilities of motif presence and not human gold labels.

No score flipping, model retraining, model refitting, model reselection, threshold selection, p-value construction, or post-hoc BH procedure was performed.

## Outcome-Level Locked-Test Evidence

| Outcome | AP | AUROC (95% CI) | Lift@10 (95% CI) | Evidence | AUROC flag | R1 preserved |
| --- | ---: | --- | --- | --- | --- | --- |
| Any outer-package affective imagery | 0.0892092248237394 | 0.31238471673254281 (0.24163957874646039-0.38667551837869063) | 2.5986622073578598 (1.3671521035598706-3.7742393162393157) | E2 | below_half | True |
| General visual appeal | 0.067527584398593354 | 0.28343273691186216 (0.19438876753549297-0.37505103858572053) | 3.3205128205128207 (1.2029420893571838-4.99514679798357) | E2 | below_half | True |
| Cute/friendly | 0.047420215832912303 | 0.27818507186976854 (0.16022991636165565-0.41770753998476756) | 2.9298642533936654 (0.7654966392830469-4.9951456310679614) | E1 | below_half | True |

Top-tail observed-positive enrichment coexists with poor/below-0.5 global ranking discrimination.

For any outer-package affective imagery and general visual appeal, the lift@10 lower confidence bound exceeded 1 (E2). For cute/friendly, the lift@10 point estimate exceeded 1 but its interval included 1 (E1). All three AUROC intervals remained wholly below 0.5.

## Feature Stability and Design-Hypothesis Grades

| Outcome | 5/5 | 4/5 | 0-3/5 | A | B | C | D |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Any outer-package affective imagery | 24 | 9 | 3 | 24 | 9 | 0 | 3 |
| General visual appeal | 25 | 6 | 5 | 25 | 6 | 0 | 5 |
| Cute/friendly | 21 | 9 | 6 | 0 | 21 | 9 | 6 |

Overall: Grade A = 49, Grade B = 36, Grade C = 9, Grade D = 14.

## Frozen Main-Figure Top-8 Features

### Any outer-package affective imagery

| Rank | Feature | Standardized coefficient | Direction | Grade |
| ---: | --- | ---: | --- | --- |
| 1 | `sparse_layout_score` | -1.1910837348852517 | negative | A |
| 2 | `edge_density` | -0.8339447779748175 | negative | A |
| 3 | `dense_ornament_score` | 0.7060870491456278 | positive | A |
| 4 | `edge_strength_mean` | 0.643477059038845 | positive | A |
| 5 | `luminance_p10` | -0.5854838564210413 | negative | A |
| 6 | `geometric_layout_score` | 0.4952816412536643 | positive | A |
| 7 | `luminance_p90` | 0.39715141903714485 | positive | A |
| 8 | `vivid_multicolor_palette_score` | 0.3732603201418176 | positive | A |

### General visual appeal

| Rank | Feature | Standardized coefficient | Direction | Grade |
| ---: | --- | ---: | --- | --- |
| 1 | `sparse_layout_score` | -1.2062724753539622 | negative | A |
| 2 | `geometric_layout_score` | 0.73884374260686 | positive | A |
| 3 | `edge_density` | -0.706720593183189 | negative | A |
| 4 | `luminance_p10` | -0.6529424166010287 | negative | A |
| 5 | `floral_illustration_score` | 0.6281588004206307 | positive | A |
| 6 | `edge_strength_mean` | 0.55419183580825 | positive | A |
| 7 | `luminance_p90` | 0.54410317442028 | positive | A |
| 8 | `ingredient_photography_score` | -0.49027871676480783 | negative | A |

### Cute/friendly

| Rank | Feature | Standardized coefficient | Direction | Grade |
| ---: | --- | ---: | --- | --- |
| 1 | `dense_ornament_score` | 1.3628411230908242 | positive | B |
| 2 | `sparse_layout_score` | -1.0226443213471852 | negative | B |
| 3 | `leaf_herb_illustration_score` | -0.8705877653862636 | negative | B |
| 4 | `edge_density` | -0.8412784945356365 | negative | B |
| 5 | `edge_strength_mean` | 0.7706996589956482 | positive | B |
| 6 | `luminance_p10` | -0.6426964794413174 | negative | B |
| 7 | `transparent_window_score` | 0.5043114025006747 | positive | B |
| 8 | `heritage_ornament_score` | -0.5015403730815253 | negative | B |

## Robustness Boundaries

- R1: all three interpretable outcomes retained lift@10 > 1 after known-QA-exception exclusion.
- R2: alternative-image sensitivity is descriptive only; it is not a design-grade eligibility gate and no retraining occurred.
- R3: G-exposure analyses describe heterogeneity and coverage only; no performance-best threshold or favorable subgroup was selected.
- R4: alternative label definitions remain robustness analyses; the core outcomes remain primary and no promotion or relabeling occurred.
- R5: image-exception diagnostics are descriptive only; inferential_claim is false and they do not enter feature/design grades.

## Manuscript Use

Grade A features are qualified design hypotheses, not prescriptions. Grade B features are exploratory. Grade C features remain development-stage associations only. Grade D features remain in the complete supplementary ledger and should not be highlighted.

No feature should be described as validated on the locked test, as independently predicting the locked-test outcome, as increasing or decreasing consumer perception, or as causing affective imagery.

Primary manuscript destinations are Results 3.4 and Discussion 4.3, followed only by constrained consistency updates elsewhere in the manuscript.
