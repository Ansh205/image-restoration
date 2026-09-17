"""
Noise Detection Module.

Estimates Gaussian noise level (sigma) using local patch standard deviation.
"""
from typing import Tuple

import cv2
import numpy as np
from PIL import Image
from loguru import logger


def estimate_noise_sigma(gray: np.ndarray) -> float:
    """
    Estimate noise standard deviation (sigma) using median absolute deviation of Laplacian response.
    Implements Immerkaer / Donoho noise estimation method.
    """
    h, w = gray.shape
    # Kernel for noise estimation
    kernel = np.array([[1, -2, 1],
                       [-2, 4, -2],
                       [1, -2, 1]], dtype=np.float32)
    
    sigma = np.sum(np.abs(cv2.filter2D(gray.astype(np.float32), -1, kernel)))
    sigma = sigma * np.sqrt(0.5 * np.pi) / (6.0 * (w - 2) * (h - 2))
    return float(sigma)


def detect_noise(image: Image.Image, threshold: float = 15.0) -> Tuple[bool, float, float]:
    """
    Detect noise in PIL Image.

    Args:
        image: Input PIL Image (RGB).
        threshold: Noise sigma threshold (scale 0-255).

    Returns:
        Tuple of (is_noisy: bool, severity: float [0.0-1.0], estimated_sigma: float)
    """
    np_img = np.array(image)
    if np_img.ndim == 3:
        gray = cv2.cvtColor(np_img, cv2.COLOR_RGB2GRAY)
    else:
        gray = np_img

    sigma = estimate_noise_sigma(gray)
    is_noisy = sigma > threshold

    if is_noisy:
        # Scale severity: threshold -> 0.0; threshold + 35.0 -> 1.0
        severity = float(np.clip((sigma - threshold) / 35.0, 0.0, 1.0))
    else:
        severity = 0.0

    logger.debug(f"Noise detector: estimated_sigma={sigma:.2f}, threshold={threshold}, is_noisy={is_noisy}")
    return is_noisy, severity, sigma
