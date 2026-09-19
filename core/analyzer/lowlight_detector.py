"""
Exposure Detection Module.

Calculates exposure statistics (mean, median, dark_ratio, bright_ratio)
to detect under-exposed (low-light) and over-exposed images.
"""
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
