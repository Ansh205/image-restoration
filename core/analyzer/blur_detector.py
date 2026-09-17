"""
Blur Detection Module.

Uses OpenCV Laplacian variance to estimate image sharpness and detect blur.
"""
from typing import Any, Tuple

import cv2
import numpy as np
from PIL import Image
from loguru import logger


def detect_blur(image: Image.Image, threshold: float = 100.0) -> Tuple[bool, float, float]:
    """
    Detect blur using Laplacian variance on the grayscale image.

    Args:
        image: PIL Image (RGB).
        threshold: Variance threshold below which an image is considered blurry.

    Returns:
        Tuple of (is_blurry: bool, severity: float [0.0-1.0], laplacian_variance: float)
    """
    # Convert PIL Image to Grayscale OpenCV numpy array
    np_img = np.array(image)
    if np_img.ndim == 3:
        gray = cv2.cvtColor(np_img, cv2.COLOR_RGB2GRAY)
    else:
        gray = np_img

    # Compute Laplacian variance
    laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    is_blurry = laplacian_var < threshold

    # Calculate severity score [0.0, 1.0] where 1.0 is extremely blurry
    if is_blurry:
        # Scale severity linearly: variance 0 -> severity 1.0; variance threshold -> severity 0.0
        severity = float(np.clip(1.0 - (laplacian_var / max(threshold, 1e-5)), 0.0, 1.0))
    else:
        severity = 0.0

    logger.debug(f"Blur detector: laplacian_var={laplacian_var:.2f}, threshold={threshold}, is_blurry={is_blurry}")
    return is_blurry, severity, laplacian_var
