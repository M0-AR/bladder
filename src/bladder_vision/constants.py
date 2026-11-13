"""Shared labels and defaults.

The class order is alphabetical because Keras directory-based loaders sort class
folder names. Training data must therefore use the folder names ``NMIBC``, ``Normal``
and ``Ta``, and inference must decode softmax outputs in the same order.
"""

from __future__ import annotations

CLASS_NAMES: tuple[str, ...] = ("NMIBC", "Normal", "Ta")
CLASS_DESCRIPTIONS: dict[str, str] = {
    "NMIBC": "Non-muscle-invasive bladder cancer (stages Tis, Ta and T1).",
    "Normal": "No suspicious lesion visible in the reviewed frame.",
    "Ta": "Non-invasive papillary urothelial carcinoma (a stage within NMIBC).",
}
DETECTION_LABEL = "Suspected_Region"
DEFAULT_INPUT_HEIGHT = 1080
DEFAULT_INPUT_WIDTH = 1730
DEFAULT_BATCH_SIZE = 20
DEFAULT_EPOCHS = 30
DEFAULT_LEARNING_RATE = 5e-4
DEFAULT_CONFIDENCE = 0.30
DEFAULT_SEED = 42
MODEL_DIR = "models"
DETECTOR_FILE = "detect.tflite"
LABELMAP_FILE = "labelmap.txt"
CLASSIFIER_FILE = "bladder_types.keras"
