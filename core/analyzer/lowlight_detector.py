"""
Exposure Detection Module.

Calculates exposure statistics (mean, median, dark_ratio, bright_ratio)
to detect under-exposed (low-light) and over-exposed images.
"""
'''
from typing import Tuple, List, Dict

import cv2
import numpy as np
from PIL import Image
from loguru import logger

from app.schemas.image import DegradationItem


def detect_low_light(image: Image.Image, config: dict) -> Tuple[List[DegradationItem], Dict]:
    """
    Detect exposure degradations.

    Args:
        image: Input PIL Image (RGB).
        config: Exposure configuration dict.

    Returns:
        Tuple of (list of DegradationItems, raw_metrics dict)
    """
    np_img = np.array(image)
    if np_img.ndim == 3:
        ycbcr = cv2.cvtColor(np_img, cv2.COLOR_RGB2YCrCb)
        y_channel = ycbcr[:, :, 0].astype(np.float32) / 255.0
    else:
        y_channel = np_img.astype(np.float32) / 255.0

    mean_lum = float(np.mean(y_channel))
    median_lum = float(np.median(y_channel))
    dark_ratio = float(np.mean(y_channel < 0.1))
    bright_ratio = float(np.mean(y_channel > 0.9))

    # Thresholds
    low_thresh = float(config.get("low_light_threshold", 0.35))
    over_thresh = float(config.get("overexposure_threshold", 0.70))
    dark_ratio_thresh = float(config.get("dark_ratio_threshold", 0.20))
    bright_ratio_thresh = float(config.get("bright_ratio_threshold", 0.20))

    metrics = {
        "mean_luminance": round(mean_lum, 3),
        "median_luminance": round(median_lum, 3),
        "dark_ratio": round(dark_ratio, 3),
        "bright_ratio": round(bright_ratio, 3),
    }

    degradations = []
    
    # 1. Low light check
    is_low = mean_lum < low_thresh or dark_ratio > dark_ratio_thresh
    if is_low:
        severity = float(np.clip(1.0 - (mean_lum / max(low_thresh, 1e-5)), 0.0, 1.0))
        # If dark_ratio dominates
        severity = max(severity, float(np.clip(dark_ratio / max(dark_ratio_thresh + 0.3, 1e-5), 0.0, 1.0)))
        
        degradations.append(DegradationItem(
            name="low_light",
            score=round(severity, 2),
            severity="HIGH" if severity >= 0.65 else ("MEDIUM" if severity >= 0.35 else "LOW"),
            confidence=0.85,
            details={"mean_luminance": metrics["mean_luminance"], "dark_ratio": metrics["dark_ratio"], "threshold": low_thresh}
        ))
        
    # 2. Overexposure check (only if not low light)
    is_over = not is_low and (mean_lum > over_thresh or bright_ratio > bright_ratio_thresh)
    if is_over:
        severity = float(np.clip((mean_lum - over_thresh) / (1.0 - over_thresh + 1e-5), 0.0, 1.0))
        severity = max(severity, float(np.clip(bright_ratio / max(bright_ratio_thresh + 0.3, 1e-5), 0.0, 1.0)))
        
        degradations.append(DegradationItem(
            name="overexposure",
            score=round(severity, 2),
            severity="HIGH" if severity >= 0.65 else ("MEDIUM" if severity >= 0.35 else "LOW"),
            confidence=0.85,
            details={"mean_luminance": metrics["mean_luminance"], "bright_ratio": metrics["bright_ratio"]}
        ))

    logger.debug(f"Exposure detector: mean={mean_lum:.2f}, dark_ratio={dark_ratio:.2f}, bright_ratio={bright_ratio:.2f}, detected={len(degradations)}")
    return degradations, metrics
'''


"""
Exposure Detection Module.

Detects exposure-related degradations:

1. Low-light / underexposure
2. Overexposure
3. Shadow clipping
4. Highlight clipping
5. Poor exposure distribution

The detector uses global and regional luminance statistics.
"""

from typing import Tuple, List, Dict

import cv2
import numpy as np

from PIL import Image
from loguru import logger

from app.schemas.image import DegradationItem


def detect_low_light(
    image: Image.Image,
    config: dict
) -> Tuple[List[DegradationItem], Dict]:

    """
    Detect exposure degradations.

    Args:
        image:
            Input PIL Image.

        config:
            Exposure configuration dictionary.

    Returns:
        Tuple containing:

        - List of DegradationItem
        - Raw exposure metrics
    """

    # =========================================================
    # IMAGE CONVERSION
    # =========================================================

    np_img = np.array(
        image
    )

    if np_img.ndim == 3:

        ycbcr = cv2.cvtColor(
            np_img,
            cv2.COLOR_RGB2YCrCb
        )

        y_channel = (
            ycbcr[:, :, 0]
            .astype(np.float32)
            / 255.0
        )

    else:

        y_channel = (
            np_img
            .astype(np.float32)
            / 255.0
        )

    # =========================================================
    # GLOBAL STATISTICS
    # =========================================================

    mean_lum = float(
        np.mean(y_channel)
    )

    median_lum = float(
        np.median(y_channel)
    )

    std_lum = float(
        np.std(y_channel)
    )

    # =========================================================
    # DARK / BRIGHT PIXEL RATIOS
    # =========================================================

    dark_ratio = float(
        np.mean(
            y_channel < 0.10
        )
    )

    shadow_ratio = float(
        np.mean(
            y_channel < 0.20
        )
    )

    bright_ratio = float(
        np.mean(
            y_channel > 0.90
        )
    )

    highlight_ratio = float(
        np.mean(
            y_channel > 0.80
        )
    )

    # =========================================================
    # EXTREME CLIPPING
    # =========================================================

    black_clip_ratio = float(
        np.mean(
            y_channel <= 0.02
        )
    )

    white_clip_ratio = float(
        np.mean(
            y_channel >= 0.98
        )
    )

    # =========================================================
    # CONFIGURATION
    # =========================================================

    low_thresh = float(
        config.get(
            "low_light_threshold",
            0.35
        )
    )

    over_thresh = float(
        config.get(
            "overexposure_threshold",
            0.70
        )
    )

    dark_ratio_thresh = float(
        config.get(
            "dark_ratio_threshold",
            0.20
        )
    )

    bright_ratio_thresh = float(
        config.get(
            "bright_ratio_threshold",
            0.20
        )
    )

    # =========================================================
    # HISTOGRAM DISTRIBUTION
    # =========================================================

    histogram, _ = np.histogram(
        y_channel,
        bins=32,
        range=(0.0, 1.0)
    )

    histogram = (
        histogram /
        max(
            np.sum(histogram),
            1
        )
    )

    # Percentage of pixels in lower 25%
    lower_quarter = float(
        np.sum(
            histogram[:8]
        )
    )

    # Percentage of pixels in upper 25%
    upper_quarter = float(
        np.sum(
            histogram[-8:]
        )
    )

    # =========================================================
    # REGIONAL ANALYSIS
    # =========================================================

    h, w = y_channel.shape

    grid = 4

    tile_h = h // grid
    tile_w = w // grid

    tile_means = []

    if tile_h >= 8 and tile_w >= 8:

        for r in range(grid):

            for c in range(grid):

                y1 = r * tile_h

                y2 = (
                    (r + 1) * tile_h
                    if r < grid - 1
                    else h
                )

                x1 = c * tile_w

                x2 = (
                    (c + 1) * tile_w
                    if c < grid - 1
                    else w
                )

                tile = y_channel[
                    y1:y2,
                    x1:x2
                ]

                tile_means.append(
                    float(
                        np.mean(tile)
                    )
                )

    if tile_means:

        darkest_tile = float(
            np.min(tile_means)
        )

        brightest_tile = float(
            np.max(tile_means)
        )

        exposure_range = (
            brightest_tile -
            darkest_tile
        )

    else:

        darkest_tile = mean_lum
        brightest_tile = mean_lum
        exposure_range = 0.0

    # =========================================================
    # LOW-LIGHT SIGNALS
    # =========================================================

    mean_is_dark = (
        mean_lum <
        low_thresh
    )

    dark_regions = (
        dark_ratio >
        dark_ratio_thresh
    )

    shadow_dominant = (
        shadow_ratio >
        0.35
    )

    # Significant clipping in shadows
    shadow_clipping = (
        black_clip_ratio >
        0.05
    )

    # =========================================================
    # OVEREXPOSURE SIGNALS
    # =========================================================

    mean_is_bright = (
        mean_lum >
        over_thresh
    )

    bright_regions = (
        bright_ratio >
        bright_ratio_thresh
    )

    highlight_dominant = (
        highlight_ratio >
        0.35
    )

    highlight_clipping = (
        white_clip_ratio >
        0.05
    )

    # =========================================================
    # REGIONAL EXTREMES
    # =========================================================

    severely_dark_region = (
        darkest_tile <
        0.15
    )

    severely_bright_region = (
        brightest_tile >
        0.85
    )

    # =========================================================
    # FINAL LOW-LIGHT DECISION
    # =========================================================

    low_light_signals = [
        mean_is_dark,
        dark_regions,
        shadow_dominant,
        shadow_clipping,
        severely_dark_region,
    ]

    low_signal_count = sum(
        bool(x)
        for x in low_light_signals
    )

    is_low = bool(
        low_signal_count >= 2
        or
        mean_is_dark
        or
        dark_regions
    )

    # =========================================================
    # FINAL OVEREXPOSURE DECISION
    # =========================================================

    overexposure_signals = [
        mean_is_bright,
        bright_regions,
        highlight_dominant,
        highlight_clipping,
        severely_bright_region,
    ]

    over_signal_count = sum(
        bool(x)
        for x in overexposure_signals
    )

    is_over = bool(
        over_signal_count >= 2
        or
        mean_is_bright
        or
        bright_regions
    )

    # IMPORTANT:
    #
    # We intentionally DO NOT use:
    #
    #     is_over = not is_low and ...
    #
    # because an image can contain both dark shadows
    # and blown highlights.

    # =========================================================
    # LOW-LIGHT SEVERITY
    # =========================================================

    low_severity_mean = float(
        np.clip(
            1.0 -
            (
                mean_lum /
                max(
                    low_thresh,
                    1e-5
                )
            ),
            0.0,
            1.0
        )
    )

    low_severity_dark = float(
        np.clip(
            (
                dark_ratio -
                dark_ratio_thresh
            )
            /
            0.40,
            0.0,
            1.0
        )
    )

    low_severity_clip = float(
        np.clip(
            black_clip_ratio /
            0.20,
            0.0,
            1.0
        )
    )

    low_severity = float(
        np.clip(
            0.45 * low_severity_mean
            +
            0.35 * low_severity_dark
            +
            0.20 * low_severity_clip,
            0.0,
            1.0
        )
    )

    # If low light is clearly detected but the score is
    # extremely small, don't report an artificial zero.

    if is_low:

        low_severity = max(
            low_severity,
            0.15
        )

    # =========================================================
    # OVEREXPOSURE SEVERITY
    # =========================================================

    over_severity_mean = float(
        np.clip(
            (
                mean_lum -
                over_thresh
            )
            /
            max(
                1.0 -
                over_thresh,
                1e-5
            ),
            0.0,
            1.0
        )
    )

    over_severity_bright = float(
        np.clip(
            (
                bright_ratio -
                bright_ratio_thresh
            )
            /
            0.40,
            0.0,
            1.0
        )
    )

    over_severity_clip = float(
        np.clip(
            white_clip_ratio /
            0.20,
            0.0,
            1.0
        )
    )

    over_severity = float(
        np.clip(
            0.45 * over_severity_mean
            +
            0.35 * over_severity_bright
            +
            0.20 * over_severity_clip,
            0.0,
            1.0
        )
    )

    if is_over:

        over_severity = max(
            over_severity,
            0.15
        )

    # =========================================================
    # MIXED EXPOSURE
    # =========================================================

    mixed_exposure = bool(
        is_low
        and
        is_over
    )

    # =========================================================
    # CONFIDENCE
    # =========================================================

    low_confidence = float(
        np.clip(
            0.55 +
            0.08 * low_signal_count,
            0.55,
            0.90
        )
    )

    over_confidence = float(
        np.clip(
            0.55 +
            0.08 * over_signal_count,
            0.55,
            0.90
        )
    )

    # =========================================================
    # METRICS
    # =========================================================

    metrics = {

        "mean_luminance": round(
            mean_lum,
            3
        ),

        "median_luminance": round(
            median_lum,
            3
        ),

        "luminance_std": round(
            std_lum,
            3
        ),

        "dark_ratio": round(
            dark_ratio,
            3
        ),

        "shadow_ratio": round(
            shadow_ratio,
            3
        ),

        "bright_ratio": round(
            bright_ratio,
            3
        ),

        "highlight_ratio": round(
            highlight_ratio,
            3
        ),

        "black_clip_ratio": round(
            black_clip_ratio,
            3
        ),

        "white_clip_ratio": round(
            white_clip_ratio,
            3
        ),

        "lower_quarter_ratio": round(
            lower_quarter,
            3
        ),

        "upper_quarter_ratio": round(
            upper_quarter,
            3
        ),

        "darkest_tile_luminance": round(
            darkest_tile,
            3
        ),

        "brightest_tile_luminance": round(
            brightest_tile,
            3
        ),

        "exposure_range": round(
            exposure_range,
            3
        ),

        "low_light_detected": is_low,

        "overexposure_detected": is_over,

        "mixed_exposure": mixed_exposure,
    }

    # =========================================================
    # DEGRADATIONS
    # =========================================================

    degradations = []

    # ---------------------------------------------------------
    # LOW LIGHT
    # ---------------------------------------------------------

    if is_low:

        degradations.append(
            DegradationItem(
                name="low_light",

                score=round(
                    low_severity,
                    2
                ),

                severity=(
                    "HIGH"
                    if low_severity >= 0.65
                    else
                    "MEDIUM"
                    if low_severity >= 0.35
                    else
                    "LOW"
                ),

                confidence=round(
                    low_confidence,
                    2
                ),

                details={
                    "mean_luminance": round(
                        mean_lum,
                        3
                    ),

                    "median_luminance": round(
                        median_lum,
                        3
                    ),

                    "dark_ratio": round(
                        dark_ratio,
                        3
                    ),

                    "shadow_ratio": round(
                        shadow_ratio,
                        3
                    ),

                    "black_clip_ratio": round(
                        black_clip_ratio,
                        3
                    ),

                    "darkest_tile_luminance": round(
                        darkest_tile,
                        3
                    ),

                    "threshold": low_thresh,
                }
            )
        )

    # ---------------------------------------------------------
    # OVEREXPOSURE
    # ---------------------------------------------------------

    if is_over:

        degradations.append(
            DegradationItem(
                name="overexposure",

                score=round(
                    over_severity,
                    2
                ),

                severity=(
                    "HIGH"
                    if over_severity >= 0.65
                    else
                    "MEDIUM"
                    if over_severity >= 0.35
                    else
                    "LOW"
                ),

                confidence=round(
                    over_confidence,
                    2
                ),

                details={
                    "mean_luminance": round(
                        mean_lum,
                        3
                    ),

                    "median_luminance": round(
                        median_lum,
                        3
                    ),

                    "bright_ratio": round(
                        bright_ratio,
                        3
                    ),

                    "highlight_ratio": round(
                        highlight_ratio,
                        3
                    ),

                    "white_clip_ratio": round(
                        white_clip_ratio,
                        3
                    ),

                    "brightest_tile_luminance": round(
                        brightest_tile,
                        3
                    ),

                    "threshold": over_thresh,
                }
            )
        )

    # =========================================================
    # LOGGING
    # =========================================================

    logger.debug(
        f"Exposure detector | "
        f"mean={mean_lum:.3f} | "
        f"median={median_lum:.3f} | "
        f"dark={dark_ratio:.3f} | "
        f"bright={bright_ratio:.3f} | "
        f"black_clip={black_clip_ratio:.3f} | "
        f"white_clip={white_clip_ratio:.3f} | "
        f"low={is_low} | "
        f"over={is_over} | "
        f"mixed={mixed_exposure}"
    )

    return (
        degradations,
        metrics
    )