"""
JPEG Compression Artifact Detector.

Detects 8x8 block boundary discontinuities characteristic of JPEG compression.
"""
from typing import Tuple

import cv2
import numpy as np
from PIL import Image
from loguru import logger


def detect_jpeg_artifacts(image: Image.Image, threshold: float = 0.40) -> Tuple[bool, float, float]:
    """
    Detect JPEG blocking artifacts by analyzing 8x8 block boundary differences.

    Args:
        image: Input PIL Image (RGB).
        threshold: Blocking score threshold.

    Returns:
        Tuple of (has_jpeg_artifacts: bool, severity: float [0.0-1.0], blocking_score: float)
    """
    np_img = np.array(image)
    if np_img.ndim == 3:
        gray = cv2.cvtColor(np_img, cv2.COLOR_RGB2GRAY).astype(np.float32)
    else:
        gray = np_img.astype(np.float32)

    h, w = gray.shape
    if h < 16 or w < 16:
        return False, 0.0, 0.0

    # Vertical block boundaries (column differences across 8x8 blocks)
    col_bounds = np.arange(8, w, 8)
    b_col_diff = np.abs(gray[:, col_bounds] - gray[:, col_bounds - 1])
    nb_col_diff = np.abs(gray[:, col_bounds - 1] - gray[:, col_bounds - 2])

    # Horizontal block boundaries (row differences across 8x8 blocks)
    row_bounds = np.arange(8, h, 8)
    b_row_diff = np.abs(gray[row_bounds, :] - gray[row_bounds - 1, :])
    nb_row_diff = np.abs(gray[row_bounds - 1, :] - gray[row_bounds - 2, :])

    boundary_mean = (float(np.mean(b_col_diff)) + float(np.mean(b_row_diff))) / 2.0
    non_boundary_mean = (float(np.mean(nb_col_diff)) + float(np.mean(nb_row_diff))) / 2.0 + 1e-5

    blocking_score = float(boundary_mean / non_boundary_mean - 1.0)
    blocking_score = max(0.0, blocking_score)

    has_artifacts = blocking_score > threshold

    if has_artifacts:
        severity = float(np.clip((blocking_score - threshold) / 0.6, 0.0, 1.0))
    else:
        severity = 0.0

    logger.debug(f"JPEG detector: blocking_score={blocking_score:.3f}, threshold={threshold}, has_artifacts={has_artifacts}")
    return has_artifacts, severity, blocking_score
