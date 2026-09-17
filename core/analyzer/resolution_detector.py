"""
Resolution Detection Module.

Checks image dimensions against min_dimension threshold (e.g., 512px).
"""
from typing import Tuple

from PIL import Image
from loguru import logger


def detect_resolution(image: Image.Image, min_dimension: int = 512) -> Tuple[bool, float, Tuple[int, int]]:
    """
    Detect if an image is low resolution.

    Args:
        image: Input PIL Image.
        min_dimension: Minimum dimension threshold (width or height).

    Returns:
        Tuple of (is_low_res: bool, severity: float [0.0-1.0], (width, height))
    """
    width, height = image.size
    max_dim = max(width, height)
    is_low_res = max_dim < min_dimension

    if is_low_res:
        # Severity = 1.0 when max_dim is 64px or lower; 0.0 when max_dim is min_dimension
        severity = float(max(0.0, min(1.0, 1.0 - (max_dim / float(min_dimension)))))
    else:
        severity = 0.0

    logger.debug(f"Resolution detector: dimensions=({width}, {height}), max_dim={max_dim}, min_dim={min_dimension}, is_low_res={is_low_res}")
    return is_low_res, severity, (width, height)
