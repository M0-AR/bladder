"""Panorama stitching for wide-field cystoscopy review."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def stitch_frames(frames: list[Any]) -> Any:
    """Stitch a list of BGR frames into a panorama using OpenCV's stitcher."""
    import cv2

    if len(frames) < 2:
        raise ValueError("At least two frames are required for stitching")
    create = getattr(cv2, "Stitcher_create", None) or cv2.createStitcher
    stitcher = create(cv2.Stitcher_PANORAMA) if hasattr(cv2, "Stitcher_PANORAMA") else create()
    status, panorama = stitcher.stitch(frames)
    if status != cv2.Stitcher_OK:
        messages = {
            cv2.Stitcher_ERR_NEED_MORE_IMGS: "not enough overlapping detail",
            cv2.Stitcher_ERR_HOMOGRAPHY_EST_FAIL: "homography estimation failed",
            cv2.Stitcher_ERR_CAMERA_PARAMS_ADJUST_FAIL: "camera parameter adjustment failed",
        }
        reason = messages.get(status, f"status {status}")
        raise RuntimeError(f"Panorama stitching failed: {reason}")
    return panorama


def stitch_video(
    *,
    video_path: str | Path,
    every_n: int = 30,
    max_frames: int = 60,
) -> Any:
    """Sample frames from a video and stitch them into a panorama."""
    import cv2

    if every_n < 1:
        raise ValueError("every_n must be at least 1")
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    frames: list[Any] = []
    index = 0
    try:
        while len(frames) < max_frames:
            ok, frame = capture.read()
            if not ok:
                break
            if index % every_n == 0:
                frames.append(frame)
            index += 1
    finally:
        capture.release()
    return stitch_frames(frames)


def save_panorama(panorama: Any, output_path: str | Path) -> Path:
    """Write a panorama to disk."""
    import cv2

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(destination), panorama):
        raise OSError(f"Could not write {destination}")
    return destination


def blend_pair(left: Any, right: Any, homography: Any) -> Any:
    """Warp ``right`` into ``left`` and blend overlapping pixels.

    This is a lightweight alternative for two frames when the full stitcher is
    unavailable. ``homography`` is a 3x3 matrix mapping right to left.
    """
    import cv2

    height, width = left.shape[:2]
    warped = cv2.warpPerspective(right, homography, (width, height))
    mask = (warped > 0).astype(np.uint8) * 255
    blended = left.copy()
    blended[mask > 0] = warped[mask > 0]
    return blended
