# Internal plan

Not the midterm. Not for the email to Afzal unless he asks what you are actually doing. If he reads it, it should look like a methods project with a floor and a stretch, not a pitch.

**Owners:** Artur Lima, Pedro Nogueira  
**Date:** 18 Sep 2026  
**Class floor:** BME midterm PDF (`docs/midterm-report.pdf`)  
**This document:** the project we would still respect if the course ended tomorrow.

---

## What this is

A **missing-lead diagnostic benchmark** on PTB-XL superdiagnosis, plus **one result** the existing papers did not pin down.

The benchmark is the thing other people would run. The result is the only scientific sentence we are allowed to care about.

## The sentence

On PTB-XL superdiagnosis, does **masked-precordial reconstruction** beat the **same Transformer trained with identical lead dropout and no reconstruction head**, when the test masks are **clinically structured** rather than random?

A yes means the pretext did something dropout did not. A no means the last three years of “masked leads → robustness” were, at least here, augmentation in a costume. Either one is worth writing down. A new MAE that only beats a small CNN is not.

## What people would use

Not our weights. A script:

```
model checkpoint → fold-10 table
```

Same rows, every time. Same splits. Same metrics. If the next paper cannot dump into this table in an afternoon, we built a notebook, not a benchmark.

---

## The table (this is the output)

**Rows (test conditions). Never average them.**

| Row | Mask | Why it exists |
| --- | --- | --- |
| Clean | all 12 leads | ceiling, comparable to published PTB-XL superdiagnosis |
| Precordial hole | V1, V2, V3 off | chest electrodes, anterior MI, the case algebra cannot fake |
| Limb-6 | I, II, III, aVR, aVL, aVF only | what you get without placing V leads |
| I+II only | everything else off | harsh reduced-lead; III and the augmented leads are Einthoven-reconstructable from these two, V1–V6 are not |

**Columns (models). Fill left to right. Do not invent a column you did not train.**

1. Train-frequency baseline (done: fold-10 macro-AUROC 0.5)
2. Class CNN (M5)
3. Class Transformer, supervised, no pretext (M6, skippable for the course)
4. Published-scale 1D ResNet (`resnet1d_wang` or equivalent). Required before any paper claim.
5. Transformer + **the same lead dropout as the pretext model**, no reconstruction head. This is the control. Without it the sentence is empty.
6. Transformer with masked-precordial reconstruction, then fine-tune **with masks still on**.

**Cells:** absolute macro-AUROC and per-class AUROC (always print HYP). Macro-AUPRC next to them. Bootstrap CIs on fold 10 before you declare a winner. Never lead with “percent of clean performance.”

**Optional extra slice, not a row to average in:** records PTB-XL already tagged `electrodes_problems`, `static_noise`, `burst_noise`, `baseline_drift`. Real artifact, not Gaussian.

**External set, paper track only:** CPSC2018 or Chapman-Shaoxing, same four masks. If the table only exists on PTB-XL, it is a PTB-XL blog post.

---

## Protocol (non-negotiable)

These are the ways the sentence dies. If you break one, stop calling it the result.

1. **Do not pretrain and fine-tune on the same 17k labeled records and say SSL.** Extra unlabeled ECG (MIMIC-IV-ECG or PhysioNet 2021) or drop the word self-supervised and call it a pretext ablation.
2. **Do not use zeros as a missing-lead mask after z-score.** Zero is the training mean. Learned mask tokens. The encoder sees the mask.
3. **Do not sample “1 to 3 random leads” and report one number.** That mixes Einthoven-easy limb drops with hard precordial drops.
4. **Do not treat reconstructing III / aVR / aVL / aVF from I and II as evidence of a representation.** That is algebra. Call it in the text when the I+II row looks “fine.”
5. **Do not use a toy CNN as the deep baseline for a paper.** Match a published PTB-XL 1D ResNet on clean fold 10 before you talk about Transformers.
6. **Keep lead masking on during fine-tune**, or also report a frozen encoder + linear probe. Full clean fine-tune often wipes the pretext.
7. **Official folds 1–8 / 9 / 10.** Patient-aware. Train-only normalization. SCP diagnostic codes via `scp_statements.csv`. Likelihood 0 still counts. 411 empty-superclass rows stay as zeros.

MaeFE, Oh et al. (RLM), ST-MEM, and TolerantECG already did versions of masked leads. We are not claiming the pretext is new. We are claiming the **control + structured test** was not done cleanly enough to believe the headline.

---

## Two tracks

### Track A — what the course gets (floor)

This is what the midterm describes. Ship it even if Track B never opens.

Already done: M1 smoke, M2 100-record audit, M3 packed 100 Hz corpus, M4 chance baseline.

| | Do | Oracle |
| --- | --- | --- |
| M5 | 1D CNN, BCEWithLogits, early stop on fold 9 | fold-10 macro-AUROC > 0.5, per-class AUROCs written down |
| M6 | One small supervised Transformer, same split | same artifacts as M5. **Skip if late.** Course still finishes. |
| M7 | Score every trained model on Clean / V1–V3 / I+II | one table, absolute AUROC, rows not averaged |
| M8 | Demo: one test ECG, mute leads, five scores move. Course writeup uses the table. | both exist |

Track A **does not** include SSL, extra corpora, ResNet-matching, bootstrap, or an external set. Say what you trained. Do not say pretraining improved robustness.

### Track B — what would be worth other people’s time

Open only after M7 exists for the CNN. An unfinished pretext run is not a project.

| | Do | Oracle |
| --- | --- | --- |
| P1 | Eval harness: checkpoint in, four-row table out | someone else can score a model without reading our train loop |
| P2 | `resnet1d_wang` (or xresnet1d101) on this split | clean fold-10 in the published ballpark, not 0.88 vs their 0.93 |
| P3 | Dropout-matched Transformer, no recon head | the control column is filled |
| P4 | Masked-precordial pretext on extra unlabeled ECG if we can get it; otherwise a documented pretext ablation on PTB-XL with the word SSL struck out | P3 and P4 differ by one thing |
| P5 | Bootstrap CIs; artifact-flag slice | no winner inside overlapping CIs |
| P6 | One external set, same masks | the PTB-XL ranking does or does not hold |
| P7 | Short paper + public script | Computing in Cardiology / a methods journal if P2–P6 are real. Not a clinical claim. |

If P2 fails, there is no paper. If P3 is missing, there is no sentence. If P1 is missing, nobody will use it.

---

## What we will not do

- Call this a diagnostic system.
- Call a 1–2M Transformer a foundation model.
- Lead with occlusion maps, UMAP, or attention as evidence.
- Request an A100. These nets are a T4 job.
- Open Track B because M6 felt easy.

## GPU / data (from the locked cut)

- Air: M1–M4, packing, demo. Colab T4: M5, M6, anything in Track B.
- PTB-XL 100 Hz only. Do not unzip `records500` on the 16 GB machine.
- Packed memmap is already ~1 GB at `data/ptb-xl/packed/`.

## How to talk to Afzal

The PDF is a data-analysis proposal: official PTB-XL split, CNN, optional Transformer, a **fixed** missing-lead table. That is the whole class story.

If he asks what is interesting, the table is interesting: same models, three missing-lead conditions that are not random, Einthoven called out, HYP not hidden in a macro. If he says “that comparison is worth doing carefully,” that is the endorsement. Do not walk in with MaeFE, JBHI, or “we will publish.” Do not put Track B in the email.

If he hates missing leads and wants a vanilla classifier, Track A still works with a weaker M7. Track B dies. Accept that in the room, fight it here.

---

## Current state

| Item | Status |
| --- | --- |
| PTB-XL 100 Hz, 21799 records, official folds, train-only z-score | done |
| Patient overlap across train/val/test | 0 |
| M4 baseline | fold-10 AUROC 0.5 all classes; macro-AUPRC 0.254 |
| M5 CNN | next |
| Track B | closed until M7 exists |
