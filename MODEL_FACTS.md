# Model Facts

This label follows the spirit of a "model facts" or "model card" document: it states what a model is for, what it was built from, and what it must not be used for. It is deliberately short and conservative.

## Model 1: Reference lesion detector

| Field | Value |
|-------|-------|
| File | `models/detect.tflite` |
| Task | Object detection: localise a suspected region in a cystoscopy frame |
| Architecture | TensorFlow Lite object-detection graph (single class) |
| Label | `Suspected_Region` |
| Input | RGB video frame; spatial size is read from the model at runtime |
| Output | Normalised bounding boxes, class ids, confidence scores |
| Default threshold | 0.30 (tune on validation data; the original research script used 0.10) |
| Training data | Not included in this repository |
| Validation | No benchmark or clinical validation is shipped |
| Intended use | Research, engineering, education |
| Not intended for | Diagnosis, screening, triage, treatment, or any patient-care decision |

### Known limitations

- Single-class detector: it does not distinguish cancer from benign mimics.
- The training population, device, illumination, and annotation protocol are not documented in the repository.
- Performance under domain shift is unknown.
- Confidence scores are not calibrated clinical probabilities.
- The original label file used the spelling `Suspected_Regain`; it has been corrected to `Suspected_Region` for display. Class index 0 is unchanged.

## Model 2: Reference frame classifier

| Field | Value |
|-------|-------|
| File | Not shipped. Train with `bladder-vision train-classifier`. |
| Task | Frame-level classification |
| Classes | `NMIBC`, `Normal`, `Ta` (alphabetical, matching folder order) |
| Input | 1730 x 1080 RGB by default, configurable |
| Output | Softmax over three classes |
| Training data | Not included |
| Validation | No benchmark or clinical validation is shipped |
| Intended use | Research, engineering, education |
| Not intended for | Diagnosis, grading, staging, or any patient-care decision |

### Label caution

`Ta` is a stage of non-muscle-invasive bladder cancer, not a separate disease category. The three-class scheme is a research convention from the original project. It is not a mutually exclusive clinical taxonomy and must not be read as a pathology report. A clinically meaningful system would separate detection, staging, grading, and histology, and would be validated against pathology.

## Data and privacy

No patient data is included. Do not use this repository to process identifiable data without the approvals, controls, and legal basis required in your jurisdiction.

## Regulatory status

Not a medical device. Not FDA cleared, CE marked, or approved by any regulator. See `DISCLAIMER.md`.

## Change control

If you retrain, fine-tune, quantise, or change the threshold of a model, publish a new model facts entry with the data, code commit, evaluation protocol, and subgroup results. Do not silently replace a model file.

Last updated: 2026-11-13.