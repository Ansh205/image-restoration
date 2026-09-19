"""
Blur Detection Module.

Uses OpenCV Laplacian variance to estimate image sharpness and detect blur.
"""
from typing import Any, Tuple

import cv2
import numpy as np
from PIL import Image
from loguru import logger


def detect_blur(image: Image.Image, threshold: float = 250.0) -> Tuple[bool, float, float | None, dict]:
    """
    Detect blur using Laplacian variance and Tenengrad on the grayscale image.

    Args:
        image: PIL Image (RGB).
        threshold: Variance threshold below which an image is considered blurry.

    Returns:
        Tuple of (is_blurry: bool, severity: float [0.0-1.0], confidence, details_dict)
    """
    # Convert PIL Image to Grayscale OpenCV numpy array
    np_img = np.array(image)
    if np_img.ndim == 3:
        gray = cv2.cvtColor(np_img, cv2.COLOR_RGB2GRAY)
    else:
        gray = np_img

    # Apply mild Gaussian blur to suppress high-frequency noise prior to measuring structural blur
    smoothed = cv2.GaussianBlur(gray, (3, 3), 0)
    
    # Compute Laplacian variance
    laplacian_var = float(cv2.Laplacian(smoothed, cv2.CV_64F).var())
    
    # Compute Tenengrad variance (gradient magnitude)
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    tenengrad_var = float(np.mean(sobel_x**2 + sobel_y**2))

    is_blurry = laplacian_var < threshold

    # Calculate severity score [0.0, 1.0] where 1.0 is extremely blurry
    if is_blurry:
        # Scale severity linearly: variance 0 -> severity 1.0; variance threshold -> severity 0.0
        severity = float(np.clip(1.0 - (laplacian_var / max(threshold, 1e-5)), 0.0, 1.0))
    else:
        severity = 0.0

    details = {
        "laplacian_variance": round(laplacian_var, 3),
        "tenengrad_variance": round(tenengrad_var, 3),
        "blur_threshold": threshold
    }
    # Calibrated heuristic confidence based on signal distance from threshold
    conf_delta = abs(laplacian_var - threshold) / (threshold + 1e-5)
    confidence = float(np.clip(0.55 + 0.35 * conf_delta, 0.50, 0.95))

    logger.debug(f"Blur detector: laplacian_var={laplacian_var:.2f}, tenengrad={tenengrad_var:.2f}, threshold={threshold}, is_blurry={is_blurry}")
    return is_blurry, severity, confidence, details
