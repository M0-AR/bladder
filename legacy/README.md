# Legacy research scripts

This directory preserves the original scripts from the first phase of the project. They are kept for provenance and for readers who want to see how the pipeline evolved. They are **not** the supported interface, may contain hard-coded Windows paths, and are excluded from linting.

Use the installable package in `src/bladder_vision/` and the `bladder-vision` command instead.

| Legacy script | Supported replacement |
|---------------|-----------------------|
| `from_json_to_xml.py` | `bladder-vision convert-annotations` |
| `extract_frame.py` | `bladder-vision extract-frames` |
| `save_images_and_annotations_in_desired_structure.py` | `bladder-vision extract-frames --annotations-dir` |
| `randomly_distribute_data.py` | `bladder-vision split-classification` |
| `split_dataset_into_training_validation_testing.py` | `bladder-vision split-detection` |
| `bladder_type.py` | `bladder-vision train-classifier` |
| `object_detection_video.py` | `bladder-vision classify-video` |
| `object_detection_tensorflowlite.py` | `bladder-vision detect-video` |
| `thresholding.py`, `thresholding_with_model.py`, `contour.py` | `bladder_vision.enhance` |
| `panorama.py` | `bladder-vision panorama` |

The original detector paths were updated to point at `models/`. Do not use these scripts with identifiable patient data.