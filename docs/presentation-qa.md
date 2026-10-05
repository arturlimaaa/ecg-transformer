# Presentation Q&A

Likely questions for the 10-minute talk, with short answers. Numbers come from `figures/missing_lead_table.json`, `figures/m9_score_shift.json`, `figures/m8_demo.json`, and `figures/m4_metrics.json`.

## Method

**Why set a missing lead to zero?**
After per-lead z-scoring, zero is the training mean, which is what a flat, disconnected channel becomes. The models never saw such input in training. So the table mixes information that is gone with input the model was not built for. Training with leads dropped, or a learned mask token, would separate the two.

**Is there leakage between train and test?**
No. The official folds are patient-disjoint. Scaling uses training-fold statistics only. The checkpoint is chosen on fold 9, and fold 10 is scored once per model and condition.

**Why AUROC and AUPRC, not accuracy?**
The task is multi-label with five independent outputs. Accuracy needs a threshold per label, and the majority class dominates it. AUROC measures ranking without a threshold. AUPRC is sensitive to rare positives, and its chance level is the label's prevalence (0.119 for HYP).

**Why no confusion matrix?**
One ECG can carry several labels, so a 5×5 matrix does not apply. The correct form is five binary tables with thresholds chosen on fold 9. The ranking metrics need no threshold, so we did not build them.

**Why 100 Hz?**
It is the rate of the published benchmark we compare with. At 1,000 samples per lead, training runs on a laptop.

**Is the baseline a fair comparison?**
It is a floor, not a competitor. It gives every ECG each label's training rate, so it ranks nothing, and its AUROC is 0.5 by construction.

**What happens to ECGs with no label?**
411 ECGs have no diagnostic superclass. They stay in the splits as all-negative, 40 of them in fold 10. Statements with likelihood 0 count, following PhysioNet's aggregation.

## Results

**How does 0.920 compare with the literature?**
Strodthoff et al. (IEEE JBHI 2021) report about 0.93 for a 1D ResNet on the superdiagnostic task, on the same folds. Their test set excludes ECGs with no label, and ours keeps 40. So the two numbers are close but not directly comparable.

**Is the CNN better than the Transformer?**
That is not shown. Each model was trained once, with one seed, and there are no confidence intervals. The clean gap is 0.025. We report it, and we do not claim it.

**Are the differences significant?**
They have not been tested. The large effects are probably real. MI falls from 0.921 to 0.571 with I+II only, and MI and CD lose 0.085 and 0.072 with V1–V3 off, against 0.033 or less for the other labels. Gaps of 0.01–0.02 are not resolved. A bootstrap on the fold-10 predictions would settle this without retraining.

**Why does I+II only drop so much, if I and II determine the limb leads?**
The limb information is still there, because z-scoring keeps the Einthoven relations affine. But the network reads III, aVR, aVL and aVF from channels that are now flat, and it was never trained to derive them. The row also loses all six chest leads, and those cannot be recovered from I and II.

**Why is HYP the weakest label?**
It is the rarest label, at 12% of training ECGs. It has the lowest AUPRC in every row. Hypertrophy voltage criteria use the chest leads, so with I+II only its AUROC falls to 0.665.

**Why did the Transformer degrade more?**
We did not test why. Its best fold-9 epoch came earlier (12 against 23) and scored lower (0.907 against 0.926). This is a single seed.

## The demo

**Why does MI rise to 0.99 for ECG 514 with fewer leads?**
It does not mean the model sees more MI. With I+II only, 92.4% of fold-10 ECGs without MI score above 0.5 for MI, against 4.7% with all leads. The model pushes almost everyone toward MI. One possible mechanism, not tested: flat chest leads look like absent R waves in V1–V3, which is a sign of anterior MI.

**Was ECG 514 cherry-picked?**
It was chosen by a fixed rule: the first fold-10 MI-only ECG with a clean MI probability of at least 0.8. The fold-wide shift above shows it is typical, not a fluke.

**Is 0.5 a decision threshold?**
No. It is only used to describe the shift. A tuned threshold would be chosen on fold 9.

## Scope

**Is this clinically useful?**
No. This is a course analysis, not a diagnostic tool. The practical point is that a model trained on clean ECGs fails without warning when leads are missing. It gives confident, wrong scores, so a deployed model needs to detect missing leads or be trained with them.

**What would you do next?**
Train with leads dropped, add bootstrap intervals on fold 10, and use a published-scale 1D ResNet as the reference model.
