# Business brief

## Executive summary

Bladder cancer is one of the most expensive cancers to manage because it recurs. Cystoscopy is the front door of that pathway: it is operator-dependent, documentation-heavy, and hard to teach at scale. Most AI pilots in this space fail not on model architecture but on the unglamorous work around it — reproducible data splits, honest evaluation, clinician trust, integration, and regulatory evidence.

Bladder Vision is an open, MIT-licensed reference implementation of that unglamorous work. It gives a research or product team a working starting point for cystoscopy frame extraction, annotation conversion, lesion detection, frame classification, evaluation, and education. It is not a product and it is not a medical device. Its commercial value is that it shortens the distance between an idea and a credible pilot.

## The problem

- **Volume and recurrence.** NMIBC is roughly three quarters of new bladder cancer diagnoses, and recurrence is common, so surveillance is continuous.
- **Operator variability.** Lesion visibility depends on technique, experience, scope, and lighting.
- **Documentation burden.** Procedures generate video and free-text notes; structured capture is inconsistent.
- **Training gap.** Trainees need exposure to varied lesions, but live case mix is uneven and feedback is delayed.
- **Pilot failure.** Teams underestimate annotation quality, patient-level splits, device shift, and the evidence a buyer needs.

## Who this is for

| Buyer / user | Job to be done | What this repository provides |
|--------------|----------------|-------------------------------|
| Medtech R&D | Prototype a cystoscopy AI feature | Reference pipeline, model contract, evaluation harness |
| Hospital innovation group | Run a retrospective feasibility study | Reproducible splits, metrics, documentation |
| Academic lab | Publish a reproducible baseline | Clear modules, tests, reporting guidance |
| Surgical training program | Teach concepts and orientation | 3D game, quiz, glossary, visual explanations |
| Investor / diligence team | Understand the technical landscape | Architecture, model facts, regulatory brief |

## Value proposition

1. **Faster first experiment.** A team can go from annotated video to a split dataset and a baseline metric without rebuilding file plumbing.
2. **Lower evidence risk.** The evaluation guide forces patient-level splits, threshold discipline, and subgroup reporting before a pilot starts.
3. **Better communication.** The site explains the pipeline to non-engineers, which reduces the translation cost between clinical, regulatory, and engineering teams.
4. **Honest positioning.** The repository states what it does not have: no clinical validation, no regulatory clearance, no shipped classifier. That protects credibility in diligence.
5. **Permissive licence.** MIT allows commercial use, modification, and distribution, subject to the licence terms.

## What it is not

- Not a diagnostic device.
- Not a validated algorithm.
- Not a substitute for histopathology or clinical judgement.
- Not a turnkey product for a hospital.
- Not a claims engine. Do not use the demo to imply clinical performance.

## Pilot blueprint (12 weeks)

| Phase | Weeks | Output | Go/no-go question |
|-------|-------|--------|-------------------|
| Scoping | 1–2 | Intended use, endpoint, data access, ethics | Is the question clinically and legally viable? |
| Data | 2–4 | De-identified cohort, annotation guideline, patient-level split | Is the reference standard credible? |
| Baseline | 4–6 | Reproducible baseline, confusion matrix, calibration | Is there signal worth pursuing? |
| Reader study | 6–9 | Blinded reader experiment, workflow measure | Does it help a clinician in a realistic task? |
| Evidence review | 9–11 | STARD/TRIPOD+AI/CLAIM package, risk file | Is the evidence sufficient for the next stage? |
| Decision | 11–12 | Scale, iterate, or stop | Would we bet the next budget on this? |

A negative result at any gate is valuable. Stopping early is cheaper than a failed regulatory submission.

## Integration patterns

- **Offline batch:** extract frames and run inference on stored video; write overlays and a CSV; keep data on controlled storage.
- **Edge:** a compact TFLite model in a workstation or device-adjacent box; no patient data leaves the room.
- **Research viewer:** authenticated read-only review of videos and overlays for a study team; audit every access.
- **Education:** static site with no patient data and no backend.

Do not expose an unauthenticated inference endpoint. Do not send patient video to a third-party API without a data processing agreement and a legal basis.

## Regulatory pathway

The intended use determines the pathway. "Research and education" is not a clinical claim. If a team later adds diagnosis, triage, or treatment support, the software may become a medical device.

A credible pathway typically includes:

1. A precise intended-use statement and target population.
2. A quality management system (for example ISO 13485).
3. Risk management (ISO 14971).
4. Usability engineering (IEC 62366).
5. Clinical evaluation and performance evidence.
6. Software lifecycle and cybersecurity controls.
7. Labelling, instructions for use, and training.
8. Post-market surveillance and vigilance.

In the U.S., a low-to-moderate-risk SaMD pathway may be 510(k) with a predicate, or De Novo if no predicate exists. In the EU, classification under MDR depends on the intended purpose and the risk to the patient; cystoscopy decision support is unlikely to be Class I. Get regulatory advice early; this document is not legal advice.

## Economics

Do not model the business on model accuracy alone. The cost drivers are:

- **Data curation and annotation.** Usually the largest early cost.
- **Clinical reference standard.** Pathology adjudication and reader time.
- **Engineering.** Integration, latency, reliability, security, and maintenance.
- **Evidence generation.** Reader studies, external validation, statistical support.
- **Regulatory and quality.** QMS, audits, documentation, post-market obligations.
- **Deployment.** Hardware, support, training, and change management.

A simple ROI model should compare:

```text
value = (time saved per procedure × procedure volume × loaded labour cost)
      + (documentation quality improvement × reimbursement or risk reduction)
      + (training efficiency × trainee throughput)
      - (annotation, engineering, validation, regulatory, support costs)
```

Use local numbers. This repository does not provide them.

## Risks and mitigations

| Risk | Mitigation |
|------|------------|
| Dataset shift | External validation; monitor device and site subgroups |
| Automation bias | Human-in-the-loop design; uncertainty display; training |
| False reassurance | Never use a negative model output to rule out disease |
| Privacy breach | De-identification, access control, encryption, audit |
| Regulatory overreach | Keep research and clinical claims separate |
| Maintenance drift | Version models, data, and thresholds; revalidate |
| Vendor lock-in | Open formats, MIT licence, portable model runtime |

## Due-diligence questions

1. What is the intended use, and what is explicitly out of scope?
2. Is the split patient-level, and where is the manifest?
3. What is the reference standard, and who adjudicated it?
4. What are sensitivity, specificity, PPV, NPV, and calibration at the stated threshold?
5. Which sites, devices, and subgroups were tested?
6. Was the test set touched once?
7. What is the reader-study design, and what was measured?
8. What is the regulatory status and the evidence gap?
9. How is patient data protected, retained, and deleted?
10. What happens when the model is wrong?

## One-paragraph pitch

Bladder Vision is an open research platform for cystoscopy AI. It turns annotated video into reproducible datasets, trains and evaluates lesion models, and explains the whole pipeline through an interactive 3D learning environment. It is deliberately honest about its limits: no clinical validation, no regulatory clearance, no hidden claims. For a medtech team, it is a faster path to a defensible pilot. For a training program, it is a memorable way to teach orientation, vocabulary, and the logic of evidence. For a diligence team, it is a transparent view of what good cystoscopy AI engineering actually requires.

## Licence and commercial use

MIT. You may use, modify, and distribute the code commercially, provided the copyright notice and permission notice are included. The licence does not grant rights to patient data, third-party datasets, trademarks, or regulatory approval. If you build a product, you are responsible for your own evidence, quality system, and regulatory pathway.