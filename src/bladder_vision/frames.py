"""Frame extraction from annotated cystoscopy videos."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .annotations import clip_box, frame_filename, iter_annotations, parse_box, write_voc_xml


@dataclass
class ExtractedFrame:
    """A decoded frame with its annotation metadata."""

    frame: Any
    label: str
    timestamp: float
    filename: str
    box: tuple[int, int, int, int] | None = None


def extract_annotated_frames(
    *,
    video_path: str | Path,
    metadata: Mapping[str, Any],
    crop: bool = False,
    every_nth_annotation: int = 1,
    xy_offset: int = 1,
) -> list[ExtractedFrame]:
    """Decode the frame referenced by every point annotation.

    Annotations with more or fewer than one timestamp are skipped, matching the
    original research workflow. When ``crop`` is true the returned ``frame`` is the
    lesion crop, while ``box`` keeps the full-frame coordinates.
    """
    import cv2

    if every_nth_annotation < 1:
        raise ValueError("every_nth_annotation must be at least 1")

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    frames: list[ExtractedFrame] = []
    try:
        for index, annotation in enumerate(iter_annotations(metadata), start=1):
            if (index - 1) % every_nth_annotation != 0:
                continue
            if len(annotation.timestamps) != 1 or not annotation.xy:
                continue

            timestamp = annotation.timestamps[0]
            capture.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
            ok, frame = capture.read()
            if not ok:
                continue

            try:
                raw_box = parse_box(annotation.xy, offset=xy_offset)
                x1, y1, x2, y2 = clip_box(raw_box, frame.shape[1], frame.shape[0])
            except ValueError:
                continue

            selected = frame[y1:y2, x1:x2] if crop else frame
            if selected.size == 0:
                continue
            frames.append(
                ExtractedFrame(
                    frame=selected,
                    label=annotation.label,
                    timestamp=timestamp,
                    filename=frame_filename(timestamp, index),
                    box=(x1, y1, x2, y2),
                )
            )
    finally:
        capture.release()
    return frames


def save_frames(
    frames: list[ExtractedFrame],
    *,
    output_dir: str | Path,
    group_by_label: bool = True,
    annotations_dir: str | Path | None = None,
) -> int:
    """Write extracted frames as PNG files and optionally export VOC XML labels."""
    import cv2

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    annotation_path = Path(annotations_dir) if annotations_dir else None
    if annotation_path:
        annotation_path.mkdir(parents=True, exist_ok=True)

    written = 0
    for item in frames:
        destination = output_path / item.label if group_by_label else output_path
        destination.mkdir(parents=True, exist_ok=True)
        image_path = destination / item.filename
        if not cv2.imwrite(str(image_path), item.frame):
            raise OSError(f"Could not write {image_path}")
        if annotation_path and item.box:
            write_voc_xml(
                filename=item.filename,
                width=item.frame.shape[1],
                height=item.frame.shape[0],
                depth=item.frame.shape[2] if item.frame.ndim == 3 else 1,
                xmin=item.box[0],
                ymin=item.box[1],
                xmax=item.box[2],
                ymax=item.box[3],
                output_path=annotation_path / f"{Path(item.filename).stem}.xml",
            )
        written += 1
    return written


def sample_video(
    *,
    video_path: str | Path,
    output_dir: str | Path,
    every_n: int = 30,
    max_frames: int | None = None,
) -> int:
    """Save every Nth frame from a video, useful for unannotated review footage."""
    import cv2

    if every_n < 1:
        raise ValueError("every_n must be at least 1")
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    written = 0
    index = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if index % every_n == 0:
                image_path = destination / f"frame_{index:06d}.png"
                if not cv2.imwrite(str(image_path), frame):
                    raise OSError(f"Could not write {image_path}")
                written += 1
                if max_frames is not None and written >= max_frames:
                    break
            index += 1
    finally:
        capture.release()
    return written
