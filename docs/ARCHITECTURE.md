# Architecture

## Why the repository is shaped this way

The original project was a collection of research scripts. That is normal for a first experiment, but it makes reproduction hard: paths are absolute, filenames differ between stages, and the classifier's class order is implicit. The package keeps the science but makes the contracts explicit.

The design goals are:

1. **Reproducible** — seeded splits, written manifests, deterministic filenames.
2. **Light by default** — annotation conversion, splitting, and metrics work with NumPy alone.
3. **Optional heavy dependencies** — OpenCV and TensorFlow are imported only where needed.
4. **Honest** — no metric is reported without a dataset, split, and threshold.
5. **Inspectable** — every stage reads and writes ordinary files.

## Pipeline

```text
                         ┌──────────────────────┐
                         │  cystoscopy video    │
                         └──────────┬───────────┘
                                    │
                     ┌──────────────▼──────────────┐
                     │ VIA / VGG JSON annotations  │
                     └──────────────┬──────────────┘
                                    │
              ┌─────────────────────┼─────────────────────┐
              │                     │                     │
    ┌─────────▼─────────┐ ┌─────────▼─────────┐ ┌─────────▼─────────┐
    │ extract frames    │ │ convert to VOC    │ │ split datasets    │
    │ frames.py         │ │ annotations.py    │ │ dataset.py        │
    └─────────┬─────────┘ └─────────┬─────────┘ └─────────┬─────────┘
              │                     │                     │
              │                     │                     ├── classification: NMIBC / Normal / Ta
              │                     │                     └── detection: images + XML
              │                     │
    ┌─────────▼─────────┐ ┌─────────▼─────────┐
    │ classify video    │ │ detect video      │
    │ video.py          │ │ detector.py       │
    │ classifier.py     │ └─────────┬─────────┘
    └─────────┬─────────┘           │
              │                     │
              └──────────┬──────────┘
                         │
              ┌──────────▼──────────┐
              │ evaluate + report   │
              │ metrics.py          │
              └─────────────────────┘
```

Enhancement (`enhance.py`) and panorama stitching (`panorama.py`) are optional visual aids. They are not required by the classifier or detector.

## Modules

| Module | Responsibility | Heavy dependency |
|--------|----------------|------------------|
| `constants.py` | Canonical labels, defaults, filenames | None |
| `annotations.py` | VIA JSON parsing, VOC XML export, box maths | None |
| `frames.py` | Timestamp lookup, frame extraction, naming, optional VOC sidecars | OpenCV |
| `dataset.py` | Seeded stratified splits, pairing checks, manifests | None |
| `enhance.py` | Lesion-screening masks, contour overlays | OpenCV, NumPy |
| `classifier.py` | CNN training, evaluation, single-frame inference | TensorFlow |
| `detector.py` | TFLite loading, letterboxing, decoding, video overlay | TensorFlow, OpenCV |
| `video.py` | Classifier inference over sampled video frames | TensorFlow, OpenCV |
| `panorama.py` | Stitch sampled frames into a wide view | OpenCV |
| `metrics.py` | Confusion matrix, precision/recall/F1, screening metrics | None |
| `cli.py` | Stable command surface | Lazy |

## Contracts

### Frame naming

An extracted frame is named from the video stem and the annotation timestamp:

```text
<video-stem>_<timestamp>s_frame_<index>.png
```

The same name is used for the image and its optional Pascal VOC sidecar. This is the join key between frames and annotations.

### Class order

```python
CLASS_NAMES = ("NMIBC", "Normal", "Ta")
```

The order is alphabetical because Keras directory loaders sort folder names. A trained model writes `bladder_types.classes.json` next to the weights. Inference reads that sidecar and refuses a mismatch instead of silently swapping labels.

### Split manifest

Both splitters write `split_manifest.json` containing the seed, ratios, per-class counts, and the exact files in each split. A manifest is the evidence that a reported metric used a reproducible partition.

### Detector label

The detector is single-class. The label file contains `Suspected_Region`. Only class index 0 is used. The original spelling `Suspected_Regain` was a typo and has been corrected; the model weights are unchanged.

## Design decisions

**Lazy imports.** `import bladder_vision` works in a minimal environment. This matters for CI, for documentation builds, and for users who only need annotation conversion.

**Seeded random sampling per class.** Splits are stratified by class and reproducible from the seed. A manifest is always written.

**Copy by default.** The splitter copies files so the source dataset remains untouched. `--move` is available for space-constrained environments.

**No silent threshold changes.** The CLI exposes the detector threshold. The default is 0.30 for a usable demo; the legacy script used 0.10. Neither is a validated operating point.

**No hidden model.** The repository ships a detector file but no trained classifier. The README says so. A missing model produces an actionable error, never a fabricated prediction.

## Deployment topologies

### 1. Local research CLI

The default. Data stays on the workstation. Suitable for retrospective studies and teaching.

### 2. Institution batch job

Run the CLI inside a managed environment, write outputs to controlled storage, and keep an audit log. Split the job into extraction, inference, and evaluation so each stage can be retried.

### 3. Research viewer

A read-only viewer that displays videos and overlays for a study team. It must authenticate users, enforce study roles, and log access. This repository does not ship a clinical viewer.

### 4. Static education site

`preview.html` is a self-contained, static teaching experience. It has no backend, no upload, and no patient data. It is deployed to GitHub Pages. The interactive lab is a simulation, not the shipped model.

## Security boundaries

- The CLI reads local files and writes local files. It does not call the network.
- TensorFlow and TFLite model loading must treat model files as code-adjacent artifacts: only load models you trust.
- The website is static and collects no data. Speech uses the browser's local speech synthesis.
- No patient data is included in the repository or the website.

## Failure modes worth knowing

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| `No module named bladder_vision` | Package not installed | `pip install -e .` or set `PYTHONPATH=src` |
| No frames extracted | Timestamp units are not seconds | Pass the correct offset/units or inspect the JSON |
| Split is empty for a class | Too few files for the requested ratios | Lower the test/validation ratios or collect more data |
| Classifier labels look swapped | Missing or stale classes sidecar | Retrain or regenerate `bladder_types.classes.json` |
| Detector finds nothing | Threshold too high or frame is out of distribution | Inspect raw scores before changing the threshold |
| TFLite import error | TensorFlow not installed | `pip install -e ".[ml]"` |

## Extending the pipeline

Add a stage as a pure function with explicit inputs and outputs, a CLI subcommand, a synthetic test, and a paragraph in this document. Keep the file-based contracts stable. If you add a model, add a model facts entry.