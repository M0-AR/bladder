from __future__ import annotations

import numpy as np
import pytest


def test_enhance_frame_returns_annotated_copy():
    cv2 = pytest.importorskip("cv2")
    frame = np.zeros((200, 200, 3), dtype=np.uint8)
    cv2.rectangle(frame, (50, 50), (150, 150), (255, 255, 255), -1)
    original = frame.copy()

    from bladder_vision.enhance import auto_canny, enhance_frame, to_grayscale

    annotated, contours = enhance_frame(frame)
    assert annotated.shape == frame.shape
    assert len(contours) > 0
    assert np.array_equal(frame, original)
    gray = to_grayscale(frame)
    edges = auto_canny(gray)
    assert edges.shape == gray.shape
