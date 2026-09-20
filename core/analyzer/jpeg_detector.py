"""
JPEG Compression Artifact Detector.

Detects 8x8 block boundary discontinuities characteristic of JPEG compression.
"""
'''
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
'''



"""
JPEG Compression Artifact Detector.

Detects JPEG-like blocking artifacts using multiple signals:

1. 8x8 block boundary discontinuity
2. Relative boundary/non-boundary difference
3. Horizontal/vertical consistency
4. Local block-boundary strength

This is a heuristic detector, not a calibrated probability model.
"""

from typing import Tuple, Dict, Any

import cv2
import numpy as np

from PIL import Image
from loguru import logger


def _boundary_statistics(
    gray: np.ndarray
) -> Tuple[float, float, float, float]:
    """
    Calculate median boundary and non-boundary differences.

    Returns:
        vertical_boundary
        vertical_non_boundary
        horizontal_boundary
        horizontal_non_boundary
    """

    h, w = gray.shape

    # Need several 8x8 blocks.
    if h < 32 or w < 32:
        return 0.0, 0.0, 0.0, 0.0

    # --------------------------------------------------------
    # Vertical 8x8 boundaries
    # --------------------------------------------------------

    col_bounds = np.arange(
        8,
        w,
        8
    )

    # Remove invalid final positions.
    col_bounds = col_bounds[
        col_bounds < w
    ]

    if len(col_bounds) > 0:

        boundary_vertical = np.abs(
            gray[:, col_bounds]
            -
            gray[:, col_bounds - 1]
        )

        non_boundary_vertical = np.abs(
            gray[:, col_bounds - 2]
            -
            gray[:, col_bounds - 1]
        )

        vertical_boundary = float(
            np.median(
                boundary_vertical
            )
        )

        vertical_non_boundary = float(
            np.median(
                non_boundary_vertical
            )
        )

    else:

        vertical_boundary = 0.0
        vertical_non_boundary = 0.0

    # --------------------------------------------------------
    # Horizontal 8x8 boundaries
    # --------------------------------------------------------

    row_bounds = np.arange(
        8,
        h,
        8
    )

    row_bounds = row_bounds[
        row_bounds < h
    ]

    if len(row_bounds) > 0:

        boundary_horizontal = np.abs(
            gray[row_bounds, :]
            -
            gray[row_bounds - 1, :]
        )

        non_boundary_horizontal = np.abs(
            gray[row_bounds - 2, :]
            -
            gray[row_bounds - 1, :]
        )

        horizontal_boundary = float(
            np.median(
                boundary_horizontal
            )
        )

        horizontal_non_boundary = float(
            np.median(
                non_boundary_horizontal
            )
        )

    else:

        horizontal_boundary = 0.0
        horizontal_non_boundary = 0.0

    return (
        vertical_boundary,
        vertical_non_boundary,
        horizontal_boundary,
        horizontal_non_boundary,
    )


def detect_jpeg_artifacts(
    image: Image.Image,
    threshold: float = 0.35
) -> Tuple[
    bool,
    float,
    float,
    Dict[str, Any]
]:

    """
    Detect JPEG compression artifacts.

    Args:
        image:
            Input PIL image.

        threshold:
            Blocking score threshold.

    Returns:
        (
            has_artifacts,
            severity,
            confidence,
            details
        )
    """

    # ========================================================
    # IMAGE CONVERSION
    # ========================================================

    np_img = np.array(
        image
    )

    if np_img.ndim == 3:

        gray = cv2.cvtColor(
            np_img,
            cv2.COLOR_RGB2GRAY
        )

    else:

        gray = np_img.copy()

    gray = gray.astype(
        np.float32
    )

    h, w = gray.shape

    # ========================================================
    # SMALL IMAGE CHECK
    # ========================================================

    if h < 32 or w < 32:

        return (
            False,
            0.0,
            0.50,
            {
                "jpeg_blocking_score": 0.0,
                "boundary_diff": 0.0,
                "non_boundary_diff": 0.0,
                "jpeg_blocking_threshold": threshold,
            }
        )

    # ========================================================
    # 1. BOUNDARY STATISTICS
    # ========================================================

    (
        vertical_boundary,
        vertical_non_boundary,
        horizontal_boundary,
        horizontal_non_boundary,
    ) = _boundary_statistics(
        gray
    )

    # ========================================================
    # 2. GLOBAL BOUNDARY SCORE
    # ========================================================

    boundary_diff = (
        vertical_boundary +
        horizontal_boundary
    ) / 2.0

    non_boundary_diff = (
        vertical_non_boundary +
        horizontal_non_boundary
    ) / 2.0

    eps = 1e-5

    ratio_score = (
        boundary_diff /
        (non_boundary_diff + eps)
    ) - 1.0

    ratio_score = max(
        0.0,
        ratio_score
    )

    # ========================================================
    # 3. ABSOLUTE BOUNDARY DIFFERENCE
    # ========================================================

    absolute_diff = (
        boundary_diff -
        non_boundary_diff
    )

    # Very tiny differences should not count as JPEG.
    absolute_strength = float(
        np.clip(
            absolute_diff / 4.0,
            0.0,
            1.0
        )
    )

    # ========================================================
    # 4. HORIZONTAL / VERTICAL CONSISTENCY
    # ========================================================

    vertical_ratio = (
        vertical_boundary /
        (
            vertical_non_boundary +
            eps
        )
    ) - 1.0

    horizontal_ratio = (
        horizontal_boundary /
        (
            horizontal_non_boundary +
            eps
        )
    ) - 1.0

    vertical_ratio = max(
        0.0,
        float(vertical_ratio)
    )

    horizontal_ratio = max(
        0.0,
        float(horizontal_ratio)
    )

    # JPEG blocking often appears in both directions.
    directional_consistency = min(
        vertical_ratio,
        horizontal_ratio
    )

    directional_strength = float(
        np.clip(
            directional_consistency,
            0.0,
            1.0
        )
    )

    # ========================================================
    # 5. EDGE-AWARE JPEG SIGNAL
    # ========================================================

    # Strong natural edges can create boundary differences.
    # We therefore compare the image against a lightly
    # smoothed version.

    blurred = cv2.GaussianBlur(
        gray,
        (3, 3),
        0
    )

    residual = (
        gray -
        blurred
    )

    residual_std = float(
        np.std(
            residual
        )
    )

    # A very strong residual means the image contains lots
    # of high-frequency information. JPEG blocking becomes
    # more plausible when this exists together with
    # repeated 8x8 boundary jumps.

    high_frequency_factor = float(
        np.clip(
            residual_std / 15.0,
            0.0,
            1.0
        )
    )

    # ========================================================
    # 6. COMBINED BLOCKING SCORE
    # ========================================================

    # Main signal is still the boundary ratio.
    #
    # Absolute difference prevents tiny ratios caused by
    # near-zero backgrounds.
    #
    # Directional signal provides additional evidence.

    blocking_score = (
        0.55 * ratio_score
        +
        0.20 * absolute_strength
        +
        0.15 * directional_strength
        +
        0.10 * high_frequency_factor
    )

    blocking_score = float(
        np.clip(
            blocking_score,
            0.0,
            1.0
        )
    )

    # ========================================================
    # 7. DETECTION
    # ========================================================

    has_artifacts = bool(
        blocking_score >= threshold
        and
        absolute_diff >= 1.0
    )

    # ========================================================
    # 8. SEVERITY
    # ========================================================

    if has_artifacts:

        severity = float(
            np.clip(
                (
                    blocking_score -
                    threshold
                )
                /
                max(
                    1.0 - threshold,
                    1e-5
                ),
                0.0,
                1.0
            )
        )

        # Don't make detected JPEG artifacts
        # look artificially tiny.
        severity = max(
            severity,
            0.15
        )

    else:

        severity = 0.0

    # ========================================================
    # 9. CONFIDENCE
    # ========================================================

    evidence_count = sum(
        [
            ratio_score >= threshold,
            absolute_diff >= 1.0,
            directional_strength >= 0.20,
            high_frequency_factor >= 0.30,
        ]
    )

    if has_artifacts:

        confidence = float(
            np.clip(
                0.55 +
                0.08 * evidence_count,
                0.55,
                0.90
            )
        )

    else:

        confidence = float(
            np.clip(
                0.60 +
                0.05 * (
                    4 -
                    evidence_count
                ),
                0.60,
                0.80
            )
        )

    # ========================================================
    # 10. DETAILS
    # ========================================================

    details = {

        "jpeg_blocking_score": round(
            blocking_score,
            3
        ),

        "boundary_diff": round(
            boundary_diff,
            3
        ),

        "non_boundary_diff": round(
            non_boundary_diff,
            3
        ),

        "absolute_boundary_diff": round(
            absolute_diff,
            3
        ),

        "vertical_boundary_diff": round(
            vertical_boundary,
            3
        ),

        "horizontal_boundary_diff": round(
            horizontal_boundary,
            3
        ),

        "vertical_ratio": round(
            vertical_ratio,
            3
        ),

        "horizontal_ratio": round(
            horizontal_ratio,
            3
        ),

        "directional_strength": round(
            directional_strength,
            3
        ),

        "high_frequency_factor": round(
            high_frequency_factor,
            3
        ),

        "jpeg_blocking_threshold": (
            threshold
        ),
    }

    # ========================================================
    # 11. LOGGING
    # ========================================================

    logger.debug(
        f"JPEG detector | "
        f"score={blocking_score:.3f} | "
        f"boundary={boundary_diff:.2f} | "
        f"non_boundary={non_boundary_diff:.2f} | "
        f"absolute={absolute_diff:.2f} | "
        f"vertical={vertical_ratio:.2f} | "
        f"horizontal={horizontal_ratio:.2f} | "
        f"detected={has_artifacts} | "
        f"severity={severity:.2f}"
    )

    return (
        has_artifacts,
        severity,
        confidence,
        details,
    )