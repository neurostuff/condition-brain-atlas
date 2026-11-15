# Plan for the full model (2026-11-14)

James: "you can prepare to build the whole model, just be sure to have a good justification for every complexity being
added and test the addition if you think a statistician would question you."

Rule: start from the additive regression (`code/checks.py`; the baseline) and add one thing at a time. An addition stays
only if it passes its test; every test is scored on data the fit did not see, against the model without the addition, over
several seeds or folds, and the size of the gain is reported with its spread. The three held-out tests are: a fifth of the
papers; one whole measure for a condition; papers from a cut-off year on.

## Baseline (kept as is)

P(peak at v | contrast) = softmax_v(log window[measure, v] + Phi[v] . sum of label coefficients), ridge penalty 20, labels =
condition chain + block. Gains over the window: +0.065 by paper, +0.051 held-out measure, +0.037 future (+0.026 for a measure
new for a known condition). Open questions about the baseline itself, to settle first:

| Question a statistician would ask | Test |
|---|---|
| Is 300 Adam passes converged? | refit at 600 and 1,200 passes and with L-BFGS; held-out score must not move by more than 0.002 |
| Is the bump basis (8 mm, 12 mm lattice) the right resolution? | 6 / 8 / 12 mm widths on the three tests |
| Is the gain robust to the fold assignment? | 5 different paper splits; report the spread |
| Does the window (built from 10 mm spheres, then smoothed) flatter the baseline or the model? | window built from peaks counted once; compare both sides |
| Are peaks within a contrast independent? | score per contrast as well as per peak; bootstrap over papers for all intervals |

## Additions, in the order to try them

| # | Addition | Why (evidence so far) | Test that must pass | Drop if |
|---|---|---|---|---|
| 1 | Penalty per level of the hierarchy (class, condition, direction, family), learned | One penalty for all levels; the class carries two thirds of the gain, so levels differ in how much they should shrink | Held-out papers and future test against one tuned penalty; penalties chosen on inner folds only | gain < 0.002 or within fold spread |
| 2 | Direction by meaning within a measure family (separate "more") instead of a condition-level label shared by both directions | Alzheimer disease "more grey matter" still shows hippocampus (1.3x) because the condition label applies to both directions; split-paper coupling of opposite directions is about 0.02 | Held-out "more" contrasts only; and the Alzheimer case by eye | no gain on "more" contrasts |
| 3 | Learned coupling between measures: a condition's deviation in measure a predicts its deviation in measure b through a learned 28 x 28 (low-rank) matrix | Coupling exists for one cluster (metabolism x perfusion 0.23, grey matter x metabolism 0.13, FA less x diffusivity more 0.09) and not elsewhere; the family label added only +0.006 | Held-out measure for a condition, by pair of measures; compare with family labels | no gain on the coupled cluster's pairs |
| 4 | Explicit "no effect": point mass or heavy-tailed prior on a label's map | Many conditions are diffuse; ridge shrinks but never to zero; needed for honest "little beyond its class" | Simulation with planted null conditions (false positive rate), then held-out score must not drop | held-out score drops by > 0.002 |
| 5 | Posterior intervals (Laplace or HMC in Stan / NumPyro on the label coefficients, basis fixed), with a paper random effect | No uncertainty now; R8 | Coverage on held-out papers: the 80% interval for a region's share of peaks should cover the observed share about 80% of the time, by cell size | coverage far from nominal |
| 6 | Region-restricted analyses scored within a restricted domain | The restricted-scope label only shifts the map | Held-out restricted contrasts | no gain |
| 7 | Similarity outside the tree (text embedding of the condition, or learned low-rank condition factors) | Tree neighbours are imposed; the 1,300 thin conditions have only the class | Conditions with < 15 contrasts, held-out papers | no gain on thin conditions |

Not planned unless a test asks for it: shared spatial networks (the topic model's idea; the regression beats it without
them), per-condition coupling, a neural decoder.

## External validation

ENIGMA Toolbox regional effect maps (cortical thickness, subcortical volume; about a dozen disorders): compare the
regression's cortical thickness and grey matter "less" maps by region, as a test that uses no data from this corpus.

## Stopping rule

If additions 1 to 3 together add less than 0.005 per held-out peak, stop at the regression with intervals (5) and report
that the data do not support more structure.

## Status (2026-11-14, evening)

Baseline questions answered; additions 1 to 3 tested (numbers in NOTES.md, same date).

| # | Outcome |
|---|---|
| Baseline | converged; stable across paper splits (+0.0643 to +0.0656); 8 mm basis kept; window from peaks counted once adopted (better by paper, not on future papers) |
| 1 | KEPT in reduced form: one extra penalty, family level = 5 (others 20). +0.003 by paper, +0.006 held-out measure, +0.002 future |
| 2 | Splitting labels by direction DROPPED. Direction by meaning (diffusivity flipped) KEPT: +0.004 on held-out measures, no parameters |
| 3 | Learned coupling DROPPED: worse than fixed family labels on every test at every rank and penalty tried; loadings collapse onto the direction indicators |

Kept model: `--window-from foci --lam 20 --pen fam=5 --variant full+dirsplit+meaning`. The split by direction was adopted afterwards for map interpretation ("more grey matter" no longer inherits a condition's "less" pattern); it scores the same as the unsplit model on all three tests. Next by the stopping rule: addition 5 (intervals).

Addition 5, intervals: KEPT (2026-11-14, night): papers resampled with replacement (50 refits) plus a within-paper correlation of 0.08 to 0.12 set on training papers. 80% predictive intervals for region peak counts cover 0.82 on held-out papers and 0.82 on held-out measures (95%: 0.95 to 0.97); binomial counts alone cover 0.62. The version with a draw from the ridge prior covers the same and was dropped.

The atlas page shows them (2026-11-14): 50 refits on all papers resampled, within-paper correlation 0.08; per cell and region the 80% interval of the share of predicted peaks and of that share relative to the comparison, and the predictive interval for one new paper's 10 peaks.

Next: additions 4, 6, 7 and the ENIGMA comparison are untouched.

Condition specificity (2026-11-15, NOTES.md): on psychiatric grey matter loss with one audited diagnosis per contrast, the diagnosis adds +0.001 per held-out peak beyond its class (p 0.11); only PTSD passes Holm and a model-free check does not confirm it. Review found three corpus problems to fix before the next atlas build: comorbidity lists counted as diagnoses, 81 meta-analysis papers, the `psychiatric_other` default class.

Corpus rebuilt (2026-11-15, later; NOTES.md): contrasts count toward the primary diagnosis; other listed conditions are kept as
"also present" labels outside the model (as labels of their own they add +0.0006 per held-out peak by paper, +0.0022 on held-out
measures: below the rule); 194 meta-analysis papers dropped. Kept model on it: `fits/v2/prim`, intervals `fits/unc_v2`.
