"""
Resolution Detection Module.

Checks image dimensions against min_dimension threshold (e.g., 512px).
"""
from typing import Tuple

import numpy as np
from PIL import Image
from loguru import logger


def detect_resolution(image: Image.Image, min_dimension: int = 512) -> Tuple[bool, float, float | None, dict]:
    """
    Detect if an image is low resolution using minimum dimension.

    Args:
        image: Input PIL Image.
        min_dimension: Minimum dimension threshold (width or height).

    Returns:
        Tuple of (is_low_res, severity, confidence, details_dict)
    """
    width, height = image.size
    min_dim = min(width, height)
    is_low_res = min_dim < min_dimension

    if is_low_res:
        # Severity = 1.0 when min_dim is very low; 0.0 when it is close to min_dimension
        severity = float(np.clip(1.0 - (min_dim / float(max(min_dimension, 1))), 0.0, 1.0))
    else:
        severity = 0.0

    details = {
        "dimensions": [width, height],
        "min_dimension_found": min_dim,
        "min_dimension_threshold": min_dimension
    }
    
    # Confidence for resolution is deterministic (1.0) because it's a known measurement
    confidence = 1.0

    logger.debug(f"Resolution detector: dimensions=({width}, {height}), min_dim={min_dim}, threshold={min_dimension}, is_low_res={is_low_res}")
    return is_low_res, severity, confidence, details
