"""TensorFlow Lite object detection for suspected bladder regions."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .constants import DEFAULT_CONFIDENCE, DETECTION_LABEL


def load_labels(label_path: str | Path) -> list[str]:
    """Read a TFLite label map, one label per line."""
    lines = Path(label_path).read_text(encoding="utf-8").splitlines()
    return [line.strip() for line in lines if line.strip()]


def _interpreter_class() -> Any:
    try:
        from ai_edge_litert.interpreter import Interpreter

        return Interpreter
    except ImportError:
        pass
    try:
        from tflite_runtime.interpreter import Interpreter

        return Interpreter
    except ImportError:
        pass
    try:
        from tensorflow.lite.python.interpreter import Interpreter

        return Interpreter
    except ImportError as error:
        raise ImportError(
            "No TensorFlow Lite runtime found. Install tensorflow, tflite-runtime "
            "or ai-edge-litert."
        ) from error


@dataclass(frozen=True)
class Detection:
    """One detected region in pixel coordinates."""

    xmin: int
    ymin: int
    xmax: int
    ymax: int
    class_id: int
    label: str
    score: float


class TFLiteDetector:
    """Small wrapper around a single-class TFLite detection model."""

    def __init__(
        self,
        *,
        model_path: str | Path,
        labels_path: str | Path | None = None,
        threshold: float = DEFAULT_CONFIDENCE,
    ) -> None:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be between 0 and 1")
        self.model_path = Path(model_path)
        self.threshold = threshold
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        self.labels = load_labels(labels_path) if labels_path is not None else [DETECTION_LABEL]

        interpreter_class = _interpreter_class()
        self.interpreter = interpreter_class(model_path=str(self.model_path))
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        shape = self.input_details[0]["shape"]
        self.input_height = int(shape[1])
        self.input_width = int(shape[2])
        self.floating_model = self.input_details[0]["dtype"] == np.float32
        self.boxes_index, self.classes_index, self.scores_index = self._output_indices()

    def _output_indices(self) -> tuple[int, int, int]:
        first_name = str(self.output_details[0]["name"])
        if "StatefulPartitionedCall" in first_name:
            return 1, 3, 0
        return 0, 1, 2

    def detect(self, frame: Any) -> list[Detection]:
        """Run inference on one BGR frame and return thresholded detections."""
        import cv2

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (self.input_width, self.input_height))
        input_data = np.expand_dims(resized, axis=0)
        if self.floating_model:
            input_data = (np.float32(input_data) - 127.5) / 127.5
        else:
            input_data = input_data.astype(self.input_details[0]["dtype"])

        self.interpreter.set_tensor(self.input_details[0]["index"], input_data)
        self.interpreter.invoke()

        boxes = self.interpreter.get_tensor(self.output_details[self.boxes_index]["index"])[0]
        classes = self.interpreter.get_tensor(self.output_details[self.classes_index]["index"])[0]
        scores = self.interpreter.get_tensor(self.output_details[self.scores_index]["index"])[0]

        image_height, image_width = frame.shape[:2]
        detections: list[Detection] = []
        for box, class_id, score in zip(boxes, classes, scores, strict=False):
            confidence = float(score)
            if confidence < self.threshold:
                continue
            ymin, xmin, ymax, xmax = (float(value) for value in box[:4])
            x1 = int(max(0, min(image_width, xmin * image_width)))
            y1 = int(max(0, min(image_height, ymin * image_height)))
            x2 = int(max(0, min(image_width, xmax * image_width)))
            y2 = int(max(0, min(image_height, ymax * image_height)))
            if x2 <= x1 or y2 <= y1:
                continue
            index = int(class_id)
            label = self.labels[index] if 0 <= index < len(self.labels) else DETECTION_LABEL
            detections.append(
                Detection(
                    xmin=x1,
                    ymin=y1,
                    xmax=x2,
                    ymax=y2,
                    class_id=index,
                    label=label,
                    score=confidence,
                )
            )
        return detections

    def annotate(
        self,
        frame: Any,
        detections: Sequence[Detection] | None = None,
        *,
        color: tuple[int, int, int] = (0, 255, 120),
        thickness: int = 3,
    ) -> Any:
        """Draw detections on a copy of the frame."""
        import cv2

        output = frame.copy()
        for detection in detections if detections is not None else self.detect(frame):
            cv2.rectangle(
                output,
                (detection.xmin, detection.ymin),
                (detection.xmax, detection.ymax),
                color,
                thickness,
            )
            text = f"{detection.label}: {detection.score * 100:.0f}%"
            (text_width, text_height), baseline = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )
            label_y = max(detection.ymin, text_height + 10)
            cv2.rectangle(
                output,
                (detection.xmin, label_y - text_height - 8),
                (detection.xmin + text_width + 6, label_y + baseline - 8),
                (255, 255, 255),
                cv2.FILLED,
            )
            cv2.putText(
                output,
                text,
                (detection.xmin + 3, label_y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (20, 20, 20),
                2,
            )
        return output


def detect_video(
    *,
    video_path: str | Path,
    detector: TFLiteDetector,
    output_path: str | Path | None = None,
    display: bool = False,
    every_n: int = 1,
    max_frames: int | None = None,
) -> int:
    """Run detection over a video and optionally write an annotated MP4."""
    import cv2

    if every_n < 1:
        raise ValueError("every_n must be at least 1")

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = None
    if output_path is not None:
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(
            str(destination), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
        )

    processed = 0
    frame_index = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame_index % every_n == 0:
                annotated = detector.annotate(frame)
                if writer is not None:
                    writer.write(annotated)
                if display:
                    cv2.imshow("Bladder Vision - detection", annotated)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                processed += 1
                if max_frames is not None and processed >= max_frames:
                    break
            frame_index += 1
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        if display:
            cv2.destroyAllWindows()
    return processed
