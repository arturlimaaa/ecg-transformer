# Course spec: PTB-XL superclass classification under structured missing leads

Owners: Artur Lima, Pedro Nogueira. Written 27 Sep 2026 from the submitted midterm, the internal plan's course track, and the completed data and baseline work.

The issue tracker was not updated.

## Problem Statement

The group has already submitted a midterm that promises a fold-10 comparison of 12-lead ECG superclass classification under three fixed missing-lead conditions. The corpus is packed, the official split is checked, and a train-frequency baseline sits at chance. No classifier has been trained, so the comparison table and the single-ECG demo do not exist yet.

## Solution

Train the 1D CNN the midterm names, optionally a small supervised Transformer on the same split, and score every trained model once on fold 10 under the three named conditions. Model selection uses fold 9 only. The course writeup leads with that table, the training curves, the bars against the train-frequency baseline, and a demo in which one test ECG's five superclass scores change when leads are muted.

## User Stories

1. As a student, I want the implementation to start from the branch that already contains the packed corpus dataset, the train-frequency baseline, and the submitted midterm, so that the CNN does not invent a new split.
2. As a student, I want a 1D CNN that takes a 12-lead, 1000-sample ECG and returns five superclass logits, so that the model matches the multi-label task in the midterm.
3. As a student, I want those five logits to stay independent, with no softmax, so that one ECG can be positive for more than one superclass.
4. As a student, I want training to use unweighted binary cross-entropy with logits, so that the loss is the one named in the midterm.
5. As a student, I want the CNN trained only on folds 1–8, so that fold 9 and fold 10 stay out of the weight updates.
6. As a student, I want the checkpoint selected by fold-9 macro-AUROC, so that fold 10 is not used for model selection.
7. As a student, I want fold 10 scored once with that selected checkpoint and all 12 leads present, so that the clean result is a single held-out number.
8. As a student, I want fold-10 macro-AUROC above 0.5, so that the CNN beats the train-frequency baseline's chance floor.
9. As a student, I want per-class AUROC written down for NORM, MI, STTC, CD, and HYP, so that HYP is visible on its own.
10. As a student, I want per-class AUPRC and macro-AUPRC on that same clean fold-10 run, so that both primary metrics from the midterm are recorded.
11. As a student, I want the clean fold-10 evaluation to cover all 2198 test records, including empty-superclass rows, so that the count matches the official test fold.
12. As a student, I want every reported metric to be finite, so that a failed run cannot be filed as a result.
13. As a student, I want the seed, optimizer, learning rate, batch size, epoch budget, patience, and selected epoch recorded with the clean result, so that the hyperparameters the midterm left open can be recovered.
14. As a student, I want a training curve of training loss and fold-9 macro-AUROC by epoch, so that the writeup can show how the checkpoint was chosen.
15. As a student, I want the selected checkpoint saved and identifiable from the metrics record, so that later scoring uses those weights and not a later epoch.
16. As a student, I want the model inputs limited to the 12 by 1000 voltages, so that age, sex, and patient id stay out of the classifier.
17. As a student, I want likelihood-0 diagnostic statements to remain positive superclass labels, so that the label definition stays the PhysioNet aggregation already in use.
18. As a student, I want the 411 empty-superclass rows to remain all-zero label vectors, so that those records stay in the official splits.
19. As a student, I want patient sets to remain disjoint across train, validation, and test, so that the official split stays patient-aware.
20. As a student, I want scoring to use the training-fold lead mean and standard deviation already applied in the packed corpus, so that fold 9 and fold 10 do not refit the scale.
21. As a student, I want a small supervised Transformer trained with the same split, the same unweighted loss, the same selection rule, and the same clean fold-10 record, so that the midterm's optional second architecture can enter the table.
22. As a student, I want that Transformer, when trained, to use the 100 Hz packed corpus only, so that the 500 Hz waveforms stay unused.
23. As a student, I want the Transformer's patch length recorded with its clean result, so that the 100 Hz choice is explicit.
24. As a student, I want the Transformer to have no reconstruction head and no pretraining, so that it stays a supervised model.
25. As a student, I want permission to skip the Transformer, with that skip stated in the writeup, so that the course can finish on the CNN.
26. As a student, I want every trained model scored again on fold 10 with V1, V2, and V3 set to zero, so that the precordial-hole row exists.
27. As a student, I want every trained model scored on fold 10 with only leads I and II kept, so that the I+II-only row exists.
28. As a student, I want a clean row for every trained model in the same table, so that each model has all three conditions side by side.
29. As a student, I want those conditions applied in memory to the already z-scored signal at score time, so that the packed waveforms are not rewritten.
30. As a student, I want the stored lead mask to remain all ones, so that the packed corpus still describes fully observed 12-lead records.
31. As a student, I want zeroing a named lead to leave every other lead's samples unchanged, so that a missing-lead condition is exactly the leads it names.
32. As a student, I want each table cell to hold absolute per-class AUROC, macro-AUROC, per-class AUPRC, and macro-AUPRC, so that the metrics match the midterm.
33. As a student, I want HYP present in every row, so that the rarest superclass is never dropped from a condition.
34. As a student, I want the three rows kept as separate results, so that the writeup has no averaged robustness score.
35. As a student, I want the clean row for a model to match that model's clean fold-10 record, so that the table and the training artifact agree.
36. As a student, I want scoring to load the selected checkpoint and not train further, so that the missing-lead rows are an evaluation of the model already chosen.
37. As a student, I want a demo that loads one selected checkpoint and one fold-10 ECG, so that a single record can be shown under the same three conditions.
38. As a student, I want the demo to show the five superclass probabilities under each condition, so that the scores are on a common 0–1 scale.
39. As a student, I want the three demo probability vectors to differ, so that muting leads changes the displayed scores.
40. As a student, I want the demo's ecg id recorded, so that the example can be repeated.
41. As an instructor, I want the writeup's main figure to be the three-row table with per-class AUROC including HYP, so that the deliverable is the comparison the midterm named.
42. As an instructor, I want the existing 100-record lead-II grid in the writeup, so that the audit figure already produced is part of the report.
43. As an instructor, I want the CNN training curves and bars against the train-frequency baseline in the writeup, so that the visualizations promised in the midterm are present.
44. As an instructor, I want the I+II row discussed with the fact that III and the augmented leads are linearly determined by I and II, so that a small change on that row is read differently from the precordial hole.
45. As an instructor, I want the writeup to state that no self-supervised pretraining was used, so that the method matches the submitted proposal.
46. As an instructor, I want the writeup to describe the models as a course analysis and not as a diagnostic system, so that the claim stays inside the midterm's scope.
47. As a student, I want confusion tables, if any are shown, to be five binary tables with per-class thresholds chosen on fold 9 and frozen for fold 10, so that a multi-label task is not collapsed into one five-by-five matrix.
48. As a student, I want the existing smoke check, superclass aggregation checks, split checks, and train-frequency baseline check to keep passing, so that the new training work does not loosen the data contract.

## Implementation Decisions

- Start from the branch whose tip adds the midterm and the internal plan. The coursework-cut commit that only contains the adversarial review does not contain the dataset or the baseline.
- One new scoring operation is the seam. It takes a model, an official split, and one named condition, and it returns one metric record. CNN training, Transformer training when present, the three-row table, and the single-ECG demo all call that operation. Metric math stays in the existing ranking function used by the train-frequency baseline.
- Superclass order stays NORM, MI, STTC, CD, HYP. Lead order stays I, II, III, AVR, AVL, AVF, V1, V2, V3, V4, V5, V6.
- The CNN is a 1D convolutional network ending in five logits. Depth, width, kernel size, optimizer, learning rate, batch size, epoch budget, and patience are unspecified by the midterm. The run record stores the values actually used. The same is true of Transformer width, depth, and patch length.
- Selection criterion is fold-9 macro-AUROC. The weights that produce the fold-10 numbers are the selected checkpoint, scored once. Fold 10 does not enter early stopping.
- Course missing-lead conditions, and no others:

  | Condition | What is kept | Channels set to 0 on the z-scored tensor |
  | --- | --- | --- |
  | Clean | all 12 leads | none |
  | Precordial hole | V1, V2, V3 off | indices 6, 7, 8 |
  | I+II only | I and II | indices 2 through 11 |

- Zeroing happens at score time on a copy of the z-scored signal. The packed signal array and the stored all-ones lead mask are not edited. Learned mask tokens are not part of this spec.
- A metric record contains `n_test`, the condition name, per-class AUROC for the five superclasses, macro-AUROC, per-class AUPRC for the five superclasses, and macro-AUPRC. A training record adds the seed, the selected epoch, and the hyperparameters listed above, plus an identifier for the checkpoint. The table is one metric record per trained model per condition. It has no row that averages conditions and no field that reports a fraction of the clean score.
- The pass bar for the CNN is clean fold-10 macro-AUROC greater than 0.5, with all five per-class AUROCs present and `n_test` equal to 2198. AUPRC is reported. It is not an additional pass bar beyond what the midterm states.
- Demo scores are sigmoid probabilities of the five logits, not raw logits and not AUROC.
- Device may be the laptop or a hosted GPU. The spec does not require a particular accelerator.
- The final course document's page limit and section template are not in the repo. The writeup content required here is the table, the lead-II grid, the training curves, the baseline bars, the Einthoven remark on the I+II row, and an explicit statement that pretraining was not used.

## Testing Decisions

A good test checks behavior a caller can observe: tensor shape, which leads changed, metric-record fields, the chance-floor inequality, agreement between the clean training record and the table, and a demo whose three probability vectors differ. It does not assert layer counts, kernel sizes, or private training-loop steps.

The single seam under test is the scoring operation. Prior art is the train-frequency baseline: small direct cases for the metric record, and a saved fold-10 record that the suite reads without retraining. Full training of the 17418-record training split stays outside the suite. The saved CNN record is what the suite checks for the real fold-10 number.

Cover:

- The CNN maps a batch of shape `(batch, 12, 1000)` to logits of shape `(batch, 5)`.
- A tiny training step on synthetic labels produces a finite loss that falls, in the same spirit as the existing one-batch smoke check.
- The saved clean CNN record has the five superclass keys, `n_test` of 2198, finite metrics, and macro-AUROC greater than 0.5.
- Applying the precordial hole zeros only V1–V3. Applying I+II only zeros every lead except I and II. Neither application writes the packed corpus.
- The table has exactly the three condition names for each trained model, no averaged condition, and a clean row equal to that model's saved clean record.
- When a Transformer record exists, it satisfies the same clean-record checks. When it does not, the suite does not require it.
- Existing checks for superclass aggregation, disjoint patients, train-only normalization, all-ones lead masks, and the train-frequency baseline still pass.

## Out of Scope

- Publishing this spec to the issue tracker.
- The paper track: a published-scale 1D ResNet, a dropout-matched Transformer without a reconstruction head, masked-precordial reconstruction, extra unlabeled ECG, bootstrap intervals, the artifact-flag slice, and an external corpus. That track stays closed until this spec's CNN table exists. Its rules are in Further Notes so they are not lost. They are not implementation tasks for the stories above.
- Limb-6 as a course-table row.
- Learned mask tokens, lead dropout during training, and random draws of one to three leads.
- Self-supervised pretraining, and any writeup sentence that says pretraining improved robustness.
- 500 Hz waveforms.
- Class-weighted loss. The specified loss is unweighted. A weighted run would be a different analysis.
- Occlusion maps, UMAP, attention maps, and any other interpretability figure.
- Describing the result as a diagnostic device, or describing a small Transformer as a foundation model.
- Refitting lead normalization, dropping empty-superclass rows, or dropping likelihood-0 statements.

## Further Notes

Fixed counts already measured on the packed corpus:

| Split | Records | Empty-superclass rows | NORM | MI | STTC | CD | HYP |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Folds 1–8 | 17418 | 334 | 7596 | 4379 | 4186 | 3907 | 2119 |
| Fold 9 | 2183 | 37 | 955 | 540 | 528 | 495 | 268 |
| Fold 10 | 2198 | 40 | 963 | 550 | 521 | 496 | 262 |

Full corpus: 21799 records, 411 empty-superclass rows. Train prevalence used by the baseline: NORM 0.436, MI 0.251, STTC 0.240, CD 0.224, HYP 0.122. Fold-10 train-frequency baseline: per-class AUROC 0.5, macro-AUROC 0.5, macro-AUPRC 0.254. Packed training leads have mean about 0 and standard deviation about 1. Stored lead masks are entirely ones.

The local `main` checkout at the coursework-cut commit does not contain this implementation. The branch that does is `al/m5-cnn`.

Paper-track rules, gated on the CNN table from this spec, and not stories to implement now:

- Columns fill left to right, and only for models that were actually trained: the train-frequency baseline, the course CNN, the course Transformer if it exists, a published-scale 1D ResNet (`resnet1d_wang` or `xresnet1d101`), a Transformer with the same lead dropout as the reconstruction model and no reconstruction head, and a Transformer with masked-precordial reconstruction fine-tuned with masks still on (or a frozen encoder with a linear probe, reported as well).
- The ResNet's clean fold-10 macro-AUROC has to land in the published band for that model. If it does not, there is no paper. If the dropout-matched Transformer is missing, the comparison is undefined.
- Rows, still not averaged: clean, precordial hole, limb-6 (I, II, III, AVR, AVL, AVF only), and I+II only.
- Cells add bootstrap confidence intervals on fold 10. Overlapping intervals are not a win.
- A missing lead in this track is a learned mask token, and the encoder sees the mask. Zero after z-score is the course rule only.
- Pretraining on the PTB-XL training folds is a pretext ablation. It is called self-supervised only when pretraining uses extra unlabeled ECG and fine-tuning stays on PTB-XL.
- Reconstruction of III or the augmented leads from I and II is identified as Einthoven algebra.
- An optional slice, not a table row, reports records already tagged `electrodes_problems`, `static_noise`, `burst_noise`, or `baseline_drift`.
- An external check, if the paper track gets that far, repeats the four masks on CPSC2018 or Chapman-Shaoxing.
