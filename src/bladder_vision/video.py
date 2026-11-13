"""Frame-level video classification with confidence overlays."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .classifier import load_class_names, load_classifier, predict_frame
from .constants import CLASS_NAMES, DEFAULT_INPUT_HEIGHT, DEFAULT_INPUT_WIDTH


@dataclass(frozen=True)
class FramePrediction:
    """Prediction for one decoded video frame."""

    frame_index: int
    timestamp: float
    label: str
    score: float
    probabilities: dict[str, float]


def classify_video(
    *,
    video_path: str | Path,
    model_path: str | Path,
    output_path: str | Path | None = None,
    start_seconds: float = 0.0,
    every_n: int = 1,
    display: bool = False,
    max_frames: int | None = None,
    input_size: tuple[int, int] = (DEFAULT_INPUT_WIDTH, DEFAULT_INPUT_HEIGHT),
    class_names: Sequence[str] | None = None,
) -> list[FramePrediction]:
    """Classify a video and return per-frame predictions.

    Inference runs every ``every_n`` frames; intermediate frames reuse the most
    recent prediction, which keeps playback smooth on CPU-only machines.
    """
    import cv2

    if every_n < 1:
        raise ValueError("every_n must be at least 1")
    if start_seconds < 0:
        raise ValueError("start_seconds must not be negative")

    model = load_classifier(model_path)
    names = list(class_names) if class_names else load_class_names(model_path, CLASS_NAMES)

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    if start_seconds:
        capture.set(cv2.CAP_PROP_POS_MSEC, start_seconds * 1000)

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

    predictions: list[FramePrediction] = []
    latest: dict[str, Any] | None = None
    frame_index = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame_index % every_n == 0 or latest is None:
                latest = predict_frame(model, frame, input_size=input_size, class_names=names)
                timestamp = frame_index / fps
                predictions.append(
                    FramePrediction(
                        frame_index=frame_index,
                        timestamp=timestamp,
                        label=str(latest["label"]),
                        score=float(latest["score"]),
                        probabilities=dict(latest["probabilities"]),
                    )
                )

            annotated = frame.copy()
            label = str(latest["label"])
            score = float(latest["score"])
            text = f"{label}  {score * 100:.1f}%"
            cv2.putText(annotated, text, (40, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 255, 120), 3)
            if writer is not None:
                writer.write(annotated)
            if display:
                cv2.imshow("Bladder Vision - classification", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            frame_index += 1
            if max_frames is not None and len(predictions) >= max_frames:
                break
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        if display:
            cv2.destroyAllWindows()
    return predictions
