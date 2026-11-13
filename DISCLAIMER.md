# Medical and Regulatory Disclaimer

**Bladder Vision is research and education software. It is not a medical device. It is not approved, cleared, certified, or validated for clinical use.**

## 1. No clinical use

The software, models, documentation, demos, and visualisations in this repository:

- are provided for research, engineering, and education only;
- must not be used to diagnose, screen, monitor, treat, or manage any patient;
- must not be used to make, delay, or replace a clinical decision;
- must not be presented to patients as medical advice;
- must not be deployed in a clinical workflow without independent validation, regulatory review, and institutional approval.

A qualified, licensed clinician remains solely responsible for every clinical decision. Nothing produced by this software is a substitute for professional judgement, direct patient assessment, histopathology, or current clinical guidelines.

## 2. Regulatory status

This software has **not** been submitted to, reviewed by, cleared by, or approved by:

- the U.S. Food and Drug Administration (FDA);
- any EU notified body under Regulation (EU) 2017/745 (MDR);
- the UK Medicines and Healthcare products Regulatory Agency (MHRA);
- Health Canada;
- the Australian Therapeutic Goods Administration (TGA);
- any other regulator.

No CE mark, FDA clearance, or equivalent authorisation is claimed. The maintainers are not the "manufacturer" of a medical device under any regulatory framework.

Software can become a medical device when its intended purpose includes diagnosis, prevention, monitoring, prediction, treatment, or alleviation of disease for individual patients. Any organisation that changes the intended purpose, deploys the software clinically, or places it on the market may become responsible for the applicable regulatory obligations.

## 3. Model limitations

The repository may contain research models, partial models, or no trained model at all. Where a model is included:

- it has not been externally validated;
- its training population, acquisition devices, and annotation protocol may not represent your setting;
- performance can degrade under different lighting, scopes, video compression, patient populations, and operators;
- confidence scores are not calibrated probabilities of clinical truth;
- the classifier labels are research categories, not a complete pathology report;
- `Ta` is a stage within `NMIBC`, so the three-class scheme is not a mutually exclusive clinical taxonomy.

Do not infer clinical performance from the interactive teaching simulator on the project website. The simulator is a deliberately simplified visual demonstration.

## 4. No warranty

The software is provided "as is", without warranty of any kind, express or implied, including merchantability, fitness for a particular purpose, and non-infringement. The authors and contributors accept no liability for any use, misuse, or consequence of the software. See `LICENSE`.

## 5. Data protection

If you process patient data:

- obtain the approvals required by your institution and jurisdiction;
- follow applicable law, including HIPAA, GDPR, and local health-data rules;
- minimise, de-identify, encrypt, audit, and delete data according to policy;
- never commit patient data to this repository;
- ensure your deployment has authentication, access control, and logging.

## 6. If you are a patient

This software is not for you. If you have symptoms or questions about bladder cancer, contact a qualified healthcare professional. Do not use this repository, its demos, or its outputs to decide about your care.

## 7. Research collaboration

If you are planning a clinical study or a regulated product, treat this repository as a starting point for engineering and reproducibility, not as evidence of safety or effectiveness. A credible pathway includes a predefined intended use, a clinical evaluation plan, ethics approval, external validation, a quality management system, risk management (ISO 14971), usability engineering (IEC 62366), and post-market surveillance.

Last updated: 2026-11-13.