# Evaluation protocol

The purpose of this document is to prevent a common failure: a notebook reports 96% accuracy on a random frame split, and the number collapses when the model meets a new patient.

## 1. Define the question first

Write down, before training:

- **Intended use.** Detection? Classification? Triage? Documentation? Education?
- **Unit of analysis.** Frame, lesion, procedure, or patient.
- **Target condition.** Cancer vs no cancer, NMIBC vs normal, stage, grade, or something else.
- **Reference standard.** Histopathology is the gold standard for cancer. Cystoscopy impression is not a substitute.
- **Population.** Setting, age, sex, prior treatment, device, imaging protocol.
- **Operating point.** What sensitivity and specificity are acceptable, and why?
- **Decision consequence.** What happens after a false negative and after a false positive?

If the intended use is education, say so. Do not borrow clinical metrics to make an educational demo look like a diagnostic device.

## 2. Split by patient

Frames from one cystoscopy are correlated. A frame-level random split leaks patient-specific appearance into test and inflates every metric. Split by patient, and stratify by the outcome.

Minimum credible design:

- Training: fit weights.
- Validation: select the model, threshold, and hyperparameters.
- Test: touch once, after all decisions are frozen.
- External test: a different site, device, or time period.

If the dataset is too small for three splits, use nested cross-validation at the patient level and report the variance.

## 3. Metrics

### Classification

`classification_report` returns:

| Metric | Meaning |
|--------|---------|
| Accuracy | Correct predictions / all predictions. Misleading under class imbalance. |
| Precision (per class) | Of the frames predicted as class C, how many truly are C? |
| Recall / sensitivity (per class) | Of the frames that truly are C, how many were found? |
| F1 (per class) | Harmonic mean of precision and recall. |
| Macro F1 | Unweighted mean of per-class F1. Treats rare classes as important. |
| Weighted F1 | Support-weighted mean. Dominated by common classes. |
| Support | Number of true examples of the class. |
| Confusion matrix | Rows = truth, columns = prediction. Always publish it. |

### Binary screening

`binary_screening_metrics` returns TP, TN, FP, FN, sensitivity, specificity, PPV, NPV, accuracy, F1, and the threshold used.

Report confidence intervals. For a proportion, the Wilson interval is safer than the normal approximation. For paired comparisons, use a paired test or bootstrap over patients.

### Detection

Report:

- precision and recall at a fixed IoU (for example 0.3 and 0.5);
- mean average precision (mAP) if you evaluate multiple thresholds;
- lesion-level detection rate, not just frame-level;
- false positives per video minute, which is what a reviewer actually experiences.

A single-class detector with a high frame-level score can still miss the lesion in the video. Evaluate at the video level.

### Calibration

A confidence score is not a probability of disease. Measure calibration (reliability diagram, expected calibration error, Brier score). If calibration is poor, do not present the score as a risk.

## 4. Threshold selection

Choose the threshold on validation data only, based on the intended clinical or educational trade-off. Freeze it before test. Publish the threshold with every metric. A model without a threshold is not an operating point.

For screening, high sensitivity may be preferred even at the cost of specificity. For documentation or education, the trade-off is different. State the rationale.

## 5. Subgroup and bias analysis

Report performance by:

- sex and age band;
- device, resolution, and frame rate;
- illumination and image quality;
- institution and operator;
- lesion size, location, and stage;
- prior treatment and recurrence status.

If a subgroup is too small, say so instead of pooling silently. A model that works only on one device is a device-specific model.

## 6. Reader study

If the claim is "helps clinicians", run a reader study:

- counterbalanced or crossover design;
- blinded readers;
- clinically meaningful endpoints (detection rate, decision time, confidence);
- pre-registered analysis;
- reporting of automation bias.

A model metric alone does not demonstrate clinical benefit.

## 7. Reproducibility package

Publish:

- code commit;
- data version and access conditions;
- split manifest;
- preprocessing and augmentation;
- random seeds;
- model weights or a reproducible training recipe;
- hardware and library versions;
- evaluation script and raw predictions;
- threshold and confidence intervals.

## 8. Reporting standards

- **STARD 2015** for diagnostic accuracy studies.
- **TRIPOD+AI** for prediction models.
- **CLAIM** for medical imaging AI.
- **CONSORT-AI** and **SPIRIT-AI** for trials and protocols.
- **PROBAST** for risk-of-bias assessment.

Use the checklist that matches the study. Attach it to the manuscript.

## 9. Common mistakes

| Mistake | Why it inflates results |
|---------|-------------------------|
| Random frame split | Same patient in train and test |
| Tuning on test | Test information leaks into the model |
| Reporting accuracy only | Hides false negatives and class imbalance |
| No confidence intervals | Small test sets produce unstable estimates |
| Ignoring device shift | Model learns device artefacts |
| Single threshold for all sites | Operating point is population-specific |
| Claiming clinical utility from AUC | Discrimination is not benefit |
| Hiding the confusion matrix | Prevents independent judgement |

## 10. Minimal experiment template

```text
1. Freeze the question and intended use.
2. Build a patient-level split and manifest.
3. Train only on train; select only on validation.
4. Freeze threshold, preprocessing, and code commit.
5. Evaluate once on test; bootstrap by patient.
6. Report confusion matrix, sensitivity, specificity, PPV, NPV, calibration.
7. Report subgroups and limitations.
8. Publish code, manifest, predictions, and environment.
```

Run the dependency-light checks locally:

```bash
pytest -q
```

The tests cover metric arithmetic, not clinical validity.