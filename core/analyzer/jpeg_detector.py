"""
JPEG Compression Artifact Detector.

Detects 8x8 block boundary discontinuities characteristic of JPEG compression.
"""
from typing import Tuple

import cv2
import numpy as np
from PIL import Image
from loguru import logger


def detect_jpeg_artifacts(image: Image.Image, threshold: float = 0.40) -> Tuple[bool, float, float | None, dict]:
    """
    Detect JPEG blocking artifacts by analyzing 8x8 block boundary differences.
    Requires both a relative ratio jump AND a minimal absolute edge discontinuity
    to prevent false positive saturation on smooth natural portrait images.
    """
    np_img = np.array(image)
    if np_img.ndim == 3:
        gray = cv2.cvtColor(np_img, cv2.COLOR_RGB2GRAY).astype(np.float32)
    else:
        gray = np_img.astype(np.float32)

    h, w = gray.shape
    if h < 32 or w < 32:
        return False, 0.0, 0.50, {}

    # Vertical block boundaries (column differences across 8x8 blocks)
    col_bounds = np.arange(8, w - 8, 8)
    b_col_diff = np.median(np.abs(gray[:, col_bounds] - gray[:, col_bounds - 1]))
    nb_col_diff = np.median(np.abs(gray[:, col_bounds - 1] - gray[:, col_bounds - 2]))

    # Horizontal block boundaries (row differences across 8x8 blocks)
    row_bounds = np.arange(8, h - 8, 8)
    b_row_diff = np.median(np.abs(gray[row_bounds, :] - gray[row_bounds - 1, :]))
    nb_row_diff = np.median(np.abs(gray[row_bounds - 1, :] - gray[row_bounds - 2, :]))

    boundary_score = (float(b_col_diff) + float(b_row_diff)) / 2.0
    non_boundary_score = (float(nb_col_diff) + float(nb_row_diff)) / 2.0 + 1e-5

    ratio_score = float(boundary_score / non_boundary_score - 1.0)
    ratio_score = max(0.0, ratio_score)

    # Require minimum absolute block boundary step (at least 2.0 intensity levels difference)
    absolute_diff = boundary_score - non_boundary_score
    if absolute_diff < 1.5:
        # Ignore tiny relative ratio spikes on near-zero noise backgrounds
        blocking_score = ratio_score * max(0.0, absolute_diff / 1.5)
    else:
        blocking_score = ratio_score

    has_artifacts = blocking_score > threshold

    if has_artifacts:
        severity = float(np.clip((blocking_score - threshold) / 0.8, 0.0, 1.0))
    else:
        severity = 0.0

    # Calibrated heuristic confidence based on distance from decision boundary
    conf_delta = abs(blocking_score - threshold) / (threshold + 1e-5)
    confidence = float(np.clip(0.55 + 0.35 * conf_delta, 0.50, 0.95))

    details = {
        "jpeg_blocking_score": round(blocking_score, 3),
        "boundary_diff": round(boundary_score, 3),
        "non_boundary_diff": round(non_boundary_score, 3),
        "jpeg_blocking_threshold": threshold
    }

    logger.debug(f"JPEG detector: blocking_score={blocking_score:.3f} (abs_diff={absolute_diff:.2f}), threshold={threshold}, has_artifacts={has_artifacts}")
    return has_artifacts, severity, confidence, details
