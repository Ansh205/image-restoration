"""
Low-Light Detection Module.

Calculates mean luminance across the image to detect under-exposed or dark images.
"""
from typing import Tuple

import cv2
import numpy as np
from PIL import Image
from loguru import logger


def detect_low_light(image: Image.Image, threshold: float = 0.35) -> Tuple[bool, float, float]:
    """
    Detect low-light conditions.

    Args:
        image: Input PIL Image (RGB).
        threshold: Mean luminance threshold in range [0.0, 1.0].

    Returns:
        Tuple of (is_low_light: bool, severity: float [0.0-1.0], mean_luminance: float)
    """
    np_img = np.array(image)
    if np_img.ndim == 3:
        # Convert RGB to YCbCr to extract Y (Luminance) channel
        ycbcr = cv2.cvtColor(np_img, cv2.COLOR_RGB2YCrCb)
        y_channel = ycbcr[:, :, 0].astype(np.float32) / 255.0
    else:
        y_channel = np_img.astype(np.float32) / 255.0

    mean_luminance = float(np.mean(y_channel))
    is_low_light = mean_luminance < threshold

    if is_low_light:
        # Scale severity: mean_luminance 0 -> severity 1.0; mean_luminance threshold -> severity 0.0
        severity = float(np.clip(1.0 - (mean_luminance / max(threshold, 1e-5)), 0.0, 1.0))
    else:
        severity = 0.0

    logger.debug(f"Low-light detector: mean_luminance={mean_luminance:.3f}, threshold={threshold}, is_low_light={is_low_light}")
    return is_low_light, severity, mean_luminance
