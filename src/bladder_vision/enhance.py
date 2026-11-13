"""Classical image-enhancement and contour helpers used by the pipeline."""

from __future__ import annotations

from typing import Any

import numpy as np


def to_grayscale(frame: Any) -> Any:
    """Convert a BGR frame to grayscale."""
    import cv2

    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def threshold_image(gray: Any, threshold: int = 125, invert: bool = True) -> Any:
    """Apply binary thresholding. The research pipeline inverts by default."""
    import cv2

    mode = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
    _, mask = cv2.threshold(gray, threshold, 255, mode)
    return mask


def clean_mask(
    mask: Any,
    *,
    kernel_size: tuple[int, int] = (10, 5),
    close: bool = True,
    erode_iterations: int = 14,
    dilate_iterations: int = 13,
) -> Any:
    """Close small gaps, then erode and dilate to suppress speckle noise."""
    import cv2

    result = mask
    if close:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
        result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel)
    if erode_iterations:
        result = cv2.erode(result, None, iterations=erode_iterations)
    if dilate_iterations:
        result = cv2.dilate(result, None, iterations=dilate_iterations)
    return result


def auto_canny(image: Any, sigma: float = 0.33) -> Any:
    """Canny edge detection with thresholds derived from the image median."""
    import cv2

    median = float(np.median(image))
    lower = int(max(0, (1.0 - sigma) * median))
    upper = int(min(255, (1.0 + sigma) * median))
    return cv2.Canny(image, lower, upper)


def find_contours(mask: Any) -> list[Any]:
    """Return external contours from a binary mask."""
    import cv2

    contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return list(contours)


def draw_contours(
    frame: Any,
    contours: list[Any],
    *,
    color: tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
) -> Any:
    """Draw contours on a copy of the frame and return it."""
    import cv2

    output = frame.copy()
    cv2.drawContours(output, contours, -1, color, thickness)
    return output


def contour_mask(
    frame: Any,
    *,
    threshold: int = 125,
    invert: bool = True,
    kernel_size: tuple[int, int] = (10, 5),
    erode_iterations: int = 14,
    dilate_iterations: int = 13,
    sigma: float = 0.33,
) -> Any:
    """Run the full threshold -> morphology -> Canny -> contour mask chain."""
    gray = to_grayscale(frame)
    mask = threshold_image(gray, threshold=threshold, invert=invert)
    mask = clean_mask(
        mask,
        kernel_size=kernel_size,
        erode_iterations=erode_iterations,
        dilate_iterations=dilate_iterations,
    )
    return auto_canny(mask, sigma=sigma)


def enhance_frame(
    frame: Any,
    *,
    threshold: int = 125,
    invert: bool = True,
    kernel_size: tuple[int, int] = (10, 5),
    erode_iterations: int = 14,
    dilate_iterations: int = 13,
    sigma: float = 0.33,
    color: tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
) -> tuple[Any, list[Any]]:
    """Return a contour-annotated copy of a frame and the detected contours."""
    edges = contour_mask(
        frame,
        threshold=threshold,
        invert=invert,
        kernel_size=kernel_size,
        erode_iterations=erode_iterations,
        dilate_iterations=dilate_iterations,
        sigma=sigma,
    )
    contours = find_contours(edges)
    return draw_contours(frame, contours, color=color, thickness=thickness), contours
