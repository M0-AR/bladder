# Research context

## Background

Bladder cancer is a common malignancy and a costly one to surveil. Most new cases are non-muscle-invasive (NMIBC), roughly three quarters of diagnoses. NMIBC is treatable but recurs, so patients undergo repeated cystoscopy. The procedure is operator-dependent: small, flat, or poorly lit lesions can be missed, and documentation quality varies.

Artificial intelligence has two plausible roles in this pathway:

1. **Assistive detection** during or after cystoscopy, to reduce missed lesions.
2. **Structured documentation and training**, to improve consistency and accelerate learning.

Both require more than a model. They require representative data, patient-level evaluation, calibration, human-factors work, and regulatory evidence.

## What the literature suggests

The field has moved from small single-centre experiments to larger retrospective and prospective evaluations. Several themes are consistent:

- **Detection is easier than grading.** Models often discriminate cancer from normal better than they predict grade or stage. A 2026 *iScience* evaluation of an AI-assisted cystoscopy system (AINCDS) reported an accuracy of about 0.919 for cancer identification and about 0.764 for grade prediction. The gap is a useful warning: a single headline number hides the harder task.
- **Video, not just stills.** Cystoscopy is a video procedure. Frame-level metrics can overstate performance; lesion-level and video-level metrics are more meaningful.
- **Data scale is growing but still limited.** Public datasets remain small. CystoDS provides 8,067 images from 160 patients; the Lazo dataset provides 1,754 images from 23 patients. Both are valuable for reproducibility, but neither substitutes for a multi-site external validation.
- **Annotation quality is a bottleneck.** Lesion boundaries are ambiguous, flat lesions are hard to delineate, and inter-annotator agreement is often unreported.
- **Clinical benefit is not established by AUC.** Reader studies and workflow endpoints are needed to show that a model changes decisions or outcomes.
- **Education is promising but under-evidenced.** Game-based learning tends to improve engagement and short-term knowledge, but evidence for durable clinical skill transfer is mixed. A 3D game is an orientation layer, not a substitute for supervised practice.

## Gaps this repository addresses

| Gap | Response |
|-----|----------|
| Scripts with hard-coded paths | Installable package and stable CLI |
| Implicit class order | Canonical `CLASS_NAMES` and a classes sidecar |
| Non-reproducible splits | Seeded splits with a written manifest |
| Frame-level leakage | Documentation and a patient-level split protocol |
| Opaque metrics | Dependency-light metrics with a confusion matrix |
| Unclear model limits | Model facts and a medical disclaimer |
| Poor communication to non-engineers | Interactive 3D site, quiz, glossary, simulator |
| Education without evidence | Explicit framing of the game as orientation, not training |

## What this repository does not claim

- It does not report a validated clinical performance number.
- It does not ship the original training dataset.
- It does not ship a trained classifier.
- It does not establish that the detector generalises.
- It does not claim regulatory approval or clinical utility.

These absences are stated deliberately. A research repository that hides them is less useful, not more.

## A credible experiment plan

### Phase 1: Reproducible baseline

1. Obtain a lawful, de-identified cohort with histopathology.
2. Define the intended use and the unit of analysis (lesion or procedure).
3. Create a patient-level split with a published manifest.
4. Train the classifier and evaluate on validation only.
5. Freeze the threshold and code commit.
6. Evaluate once on test; report the confusion matrix, sensitivity, specificity, PPV, NPV, and calibration with patient-level bootstrap intervals.

### Phase 2: Detection and video

1. Convert annotations to Pascal VOC.
2. Train or fine-tune a single-class detector.
3. Evaluate at the lesion level, not only the frame level.
4. Report false positives per video minute.
5. Test across devices and sites.

### Phase 3: Human factors

1. Run a blinded, counterbalanced reader study.
2. Measure detection, decision time, and confidence.
3. Pre-register the analysis.
4. Report automation bias and failure modes.

### Phase 4: External validation

1. Freeze the model and threshold.
2. Evaluate at a different site or time period.
3. Report subgroup performance and calibration drift.
4. Publish the evaluation package.

## Reporting standards

- **STARD 2015** for diagnostic accuracy.
- **TRIPOD+AI** for prediction models.
- **CLAIM** for medical imaging AI.
- **CONSORT-AI / SPIRIT-AI** for trials and protocols.
- **PROBAST** for risk of bias.

## Selected references and resources

- Siegel RL, et al. Cancer statistics. *CA: A Cancer Journal for Clinicians* (annual updates).
- Babjuk M, et al. EAU Guidelines on Non-Muscle-Invasive Bladder Cancer. European Association of Urology. <https://uroweb.org/guidelines/non-muscle-invasive-bladder-cancer>
- CystoDS: a cystoscopy dataset for AI research (OSF, 2025). 8,067 images from 160 patients.
- Lazo JF, et al. Cystoscopic images dataset. Zenodo record 7741476. <https://zenodo.org/records/7741476>
- 2026 *iScience* evaluation of an AI-assisted cystoscopy diagnosis system (AINCDS), reporting cancer identification accuracy around 0.919 and grade accuracy around 0.764.
- U.S. Food and Drug Administration. Software as a Medical Device (SaMD). <https://www.fda.gov/medical-devices/software-medical-device-samd>
- Regulation (EU) 2017/745 on medical devices. <https://eur-lex.europa.eu/eli/reg/2017/745/oj>
- EQUATOR Network reporting guidelines. <https://www.equator-network.org/>

Citations are provided for orientation. Verify the current version, sample size, and applicability before relying on any number.

## How to cite this repository

See `CITATION.cff`. If you use the pipeline in a paper, cite the software and the datasets you used. If you build on the detector, state its limitations and the fact that it is single-class and unvalidated.

## Open questions

- How much does video-level context improve lesion detection over frame-level inference?
- Which augmentation strategies help under scope and illumination shift?
- Can calibration be preserved across sites?
- What human-factors design reduces automation bias without adding friction?
- Does interactive 3D education improve durable conceptual knowledge compared with text or video? A randomised study would be needed to answer this.

These are good thesis and grant questions. The repository is a starting point, not an answer.