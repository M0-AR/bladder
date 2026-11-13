# Dataset contract

This document defines the file layout and annotation schema the pipeline expects. It does not ship patient data.

## Classification layout

```text
data/
├── train/
│   ├── NMIBC/
│   ├── Normal/
│   └── Ta/
├── validation/
│   ├── NMIBC/
│   ├── Normal/
│   └── Ta/
└── test/
    ├── NMIBC/
    ├── Normal/
    └── Ta/
```

Folder names are the labels. They are case-sensitive and must match `CLASS_NAMES` exactly. Class order is alphabetical, matching Keras directory loaders.

Create this layout from a flat, class-labelled source tree:

```bash
bladder-vision split-classification \
  --source data/raw \
  --output data/splits \
  --train 0.7 --validation 0.15 --test 0.15 \
  --seed 42
```

The command writes `data/splits/split_manifest.json`. Keep it with any reported result.

## Detection layout

```text
data/
├── images/
│   ├── train/
│   ├── validation/
│   └── test/
└── annotations/
    ├── train/
    ├── validation/
    └── test/
```

Each image has a same-stem Pascal VOC XML file. Images without annotations are rejected by the pairing check unless you explicitly allow negatives (not supported yet). Pair and split with:

```bash
bladder-vision split-detection \
  --images data/raw/images \
  --annotations data/raw/annotations \
  --output data/detection \
  --seed 42
```

## VIA / VGG JSON schema

The converter accepts the JSON exported by VGG Image Annotator (VIA). The relevant shape is:

```json
{
  "video_1.mp4": {
    "fileref": "",
    "size": 12345678,
    "filename": "video_1.mp4",
    "base64_img_data": "",
    "file_attributes": {},
    "regions": [
      {
        "shape_attributes": {
          "name": "rect",
          "x": 412,
          "y": 208,
          "width": 180,
          "height": 140
        },
        "region_attributes": {
          "label": "tumor",
          "timestamp": 12.4
        }
      }
    ]
  }
}
```

Assumptions:

- `timestamp` is in **seconds** from the start of the video. Verify this before extraction. A dataset that stores milliseconds will produce empty frames or wrong frames.
- Box coordinates are pixels in the source frame.
- `region_attributes.label` is free text. The converter writes one class name, defaulting to `Suspected_Region`; it does not infer pathology.
- Only rectangle regions are supported. Polygons and points are ignored.

Run:

```bash
bladder-vision convert-annotations \
  --json annotations/via_export.json \
  --images data/raw/images \
  --output data/raw/annotations \
  --class-name Suspected_Region
```

## Frame extraction

```bash
bladder-vision extract-frames \
  --video data/raw/video_1.mp4 \
  --json annotations/via_export.json \
  --output data/frames \
  --annotations-dir data/frames_annotations
```

Options:

| Option | Meaning |
|--------|---------|
| `--crop` | Crop to the annotated box instead of saving the full frame |
| `--every` | Keep every Nth annotation |
| `--xy-offset` | Shift boxes when the annotation frame was cropped |
| `--flat` | Do not group output by label |
| `--annotations-dir` | Also write Pascal VOC sidecars |

Frames are named `<video-stem>_<timestamp>s_frame_<index>.png`. Annotation conversion uses the same stem, so the two outputs join.

## Splitting rules

- Split by **patient**, not by frame, whenever patient identity is available. Frames from one procedure are highly correlated; a frame-level split leaks the patient into test and inflates metrics.
- Keep class proportions approximately equal across splits.
- Use a fixed seed and publish the manifest.
- Never tune on the test split. Use training for fitting and validation for model selection. Touch test once.

The provided splitter is file-level because the public interfaces do not carry patient IDs. For a credible study, split at the patient level with your own manifest and pass the resulting folders to the training command.

## Public datasets

These are useful for reproduction and benchmarking. Check each licence before use; some require registration or restrict redistribution.

| Dataset | Scale | Notes |
|---------|-------|-------|
| CystoDS | 8,067 images from 160 patients | Cystoscopy dataset released via OSF; designed for detection and classification research. |
| Lazo cystoscopy dataset | 1,754 images from 23 patients | Public on Zenodo (record 7741476); small, useful for smoke tests and transfer learning. |

Do not re-host restricted data. Link to the source and record the version you used.

## Annotation quality

A model cannot be better than its labels. Minimum checks:

1. Two annotators label a random 10% sample; report agreement (Cohen's kappa or Dice for boxes).
2. A clinician adjudicates disagreements.
3. Record the annotation guideline version.
4. Record the device, illumination, and frame rate.
5. Remove burned-in names, dates, and procedure IDs.
6. Keep a list of excluded frames and the reason.

## Privacy

- De-identify before extraction.
- Keep raw video on controlled storage.
- Do not commit frames to Git.
- Check that cropped images do not contain identifiers from the overlay.
- Document the legal basis and retention period.

## Known limitations

- The class scheme `NMIBC / Normal / Ta` is not mutually exclusive: `Ta` is a stage within `NMIBC`.
- The detector has one class and does not distinguish benign mimics.
- The repository does not contain the original training set, so the shipped detector cannot be retrained from this repository alone.