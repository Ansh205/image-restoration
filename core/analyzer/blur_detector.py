"""
Blur Detection Module.

Uses OpenCV Laplacian variance to estimate image sharpness and detect blur.
"""
'''from typing import Any, Tuple

import cv2
import numpy as np
from PIL import Image
from loguru import logger


def detect_blur(image: Image.Image, threshold: float = 150.0) -> Tuple[bool, float, float | None, dict]:
    """
    Detect blur using multi-metric analysis:
    1. Spatial resolution scale normalization (ref reference 512px diagonal)
    2. Tile-based (4x4) sharpness variation (detecting sparse high-edge text vs uniform blur)
    3. High-frequency detail ratio (fine texture vs primary edges)
    4. Combined normalized blur score in [0, 1]

    Args:
        image: PIL Image (RGB).
        threshold: Base Laplacian variance threshold (default 250.0).

    Returns:
        Tuple of (is_blurry: bool, severity: float [0.0-1.0], confidence, details_dict)
    """
    np_img = np.array(image)
    if np_img.ndim == 3:
        gray = cv2.cvtColor(np_img, cv2.COLOR_RGB2GRAY)
    else:
        gray = np_img.copy()

    h, w = gray.shape[:2]
    diag = np.sqrt(w * w + h * h)
    # Scale factor relative to reference 512x512 image (diag ~ 724)
    ref_diag = np.sqrt(512**2 + 512**2)
    resolution_scale = float(diag / ref_diag)

    # 1. Global Metrics (Raw)
    smoothed = cv2.GaussianBlur(gray, (3, 3), 0)
    laplacian_map = cv2.Laplacian(smoothed, cv2.CV_64F)
    laplacian_var = float(laplacian_map.var())

    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    mag_sq = sobel_x**2 + sobel_y**2
    tenengrad_var = float(np.mean(mag_sq))

    # Scale-normalized metrics (smaller images have fewer total samples/shorter gradients)
    norm_laplacian = laplacian_var / max(resolution_scale**2, 0.1)
    norm_tenengrad = tenengrad_var / max(resolution_scale**2, 0.1)

    # 2. Tile-Based Local Sharpness Analysis (4x4 Grid)
    grid_rows, grid_cols = 4, 4
    tile_h, tile_w = h // grid_rows, w // grid_cols
    tile_sharpnesses = []

    if tile_h >= 8 and tile_w >= 8:
        for r in range(grid_rows):
            for c in range(grid_cols):
                tile = smoothed[r*tile_h:(r+1)*tile_h, c*tile_w:(c+1)*tile_w]
                tile_var = float(cv2.Laplacian(tile, cv2.CV_64F).var())
                tile_sharpnesses.append(tile_var)
    else:
        tile_sharpnesses = [laplacian_var]

    tile_p10 = float(np.percentile(tile_sharpnesses, 10))
    tile_p50 = float(np.percentile(tile_sharpnesses, 50))
    tile_p90 = float(np.percentile(tile_sharpnesses, 90))
    tile_std = float(np.std(tile_sharpnesses))

    # 3. High-Frequency / Fine-Detail Energy Ratio
    # Compare fine Laplacian energy (kernel size 1/3) vs coarse gradient energy
    high_freq_map = np.abs(laplacian_map)
    high_freq_energy = float(np.mean(high_freq_map))
    grad_energy = float(np.mean(np.sqrt(mag_sq) + 1e-5))
    fine_detail_ratio = float(high_freq_energy / (grad_energy + 1e-5))

    # 4. Multi-Metric Blur Decision Logic
    # Effective threshold scaled by resolution
    scaled_thresh = threshold * (resolution_scale**2)

    # Blur signals:
    # A. Global raw Laplacian or normalized Laplacian below low-pass threshold factor (0.25)
    global_blur_sub = (laplacian_var < (threshold * 0.25)) or (norm_laplacian < (threshold * 0.25))
    # B. Median tile sharpness is low relative to resolution
    median_tile_blur = tile_p50 < (scaled_thresh * 0.8)
    # C. Severe sub-pixel fine detail loss (fine_detail_ratio < 0.11 AND tile_p50 < threshold * 1.5)
    fine_detail_blur = (fine_detail_ratio < 0.11) and (tile_p50 < threshold * 1.5)

    # Image is considered blurry if global metric triggers OR local tile sharpness is low AND fine detail is lost
    is_blurry = global_blur_sub or fine_detail_blur or (median_tile_blur and fine_detail_ratio < 0.25)

    # Compute continuous severity [0.0, 1.0]
    # Blend normalized global laplacian and median tile sharpness
    ref_thresh = max(threshold, 1.0)
    lap_score = float(np.clip(1.0 - (norm_laplacian / ref_thresh), 0.0, 1.0))
    tile_score = float(np.clip(1.0 - (tile_p50 / (scaled_thresh + 1e-5)), 0.0, 1.0))
    fine_score = float(np.clip((0.35 - fine_detail_ratio) / 0.35, 0.0, 1.0)) if fine_detail_ratio < 0.35 else 0.0

    if is_blurry:
        severity = float(np.clip(0.5 * lap_score + 0.3 * tile_score + 0.2 * fine_score, 0.15, 1.0))
    else:
        severity = 0.0

    details = {
        "laplacian_variance": round(laplacian_var, 3),
        "tenengrad_variance": round(tenengrad_var, 3),
        "norm_laplacian": round(norm_laplacian, 3),
        "norm_tenengrad": round(norm_tenengrad, 3),
        "resolution_scale": round(resolution_scale, 3),
        "tile_p10": round(tile_p10, 3),
        "tile_p50": round(tile_p50, 3),
        "tile_p90": round(tile_p90, 3),
        "fine_detail_ratio": round(fine_detail_ratio, 3),
        "blur_threshold": threshold
    }

    # Calibrated confidence based on multi-metric agreement
    signals_count = sum([global_blur_sub, median_tile_blur, fine_detail_blur])
    confidence = float(np.clip(0.60 + 0.12 * signals_count, 0.50, 0.95))

    logger.debug(
        f"Blur detector: laplacian={laplacian_var:.1f} (norm={norm_laplacian:.1f}), "
        f"tile_p50={tile_p50:.1f}, fine_ratio={fine_detail_ratio:.3f}, scale={resolution_scale:.2f}, "
        f"is_blurry={is_blurry}, severity={severity:.2f}"
    )
    return is_blurry, severity, confidence, details
'''

"""
Robust Blur Detection Module.

Detects loss of fine detail using multiple image-quality signals.

Important:
- Low resolution is NOT automatically classified as blur.
- Blur detection is resolution-aware.
- Scores are heuristic, not calibrated probabilities.
"""

from typing import Tuple, Dict, Any

import cv2
import numpy as np
from PIL import Image
from loguru import logger


def detect_blur(
    image: Image.Image,
    threshold: float = 150.0,
    min_dimension: int = 512,
) -> Tuple[bool, float, float, Dict[str, Any]]:

    np_img = np.array(image)

    if np_img.ndim == 3:
        gray = cv2.cvtColor(
            np_img,
            cv2.COLOR_RGB2GRAY
        )
    else:
        gray = np_img.copy()

    gray = gray.astype(np.uint8)

    h, w = gray.shape[:2]

    # =========================================================
    # 1. IMAGE SIZE / RESOLUTION
    # =========================================================

    min_dim = min(w, h)

    resolution_scale = float(
        min_dim / max(min_dimension, 1)
    )

    # Cap scale so very large images do not create
    # unreasonable values.
    resolution_scale = float(
        np.clip(
            resolution_scale,
            0.25,
            2.0
        )
    )

    # =========================================================
    # 2. GLOBAL LAPLACIAN
    # =========================================================

    smoothed = cv2.GaussianBlur(
        gray,
        (3, 3),
        0
    )

    laplacian = cv2.Laplacian(
        smoothed,
        cv2.CV_64F
    )

    laplacian_variance = float(
        laplacian.var()
    )

    # =========================================================
    # 3. TENENGRAD
    # =========================================================

    sobel_x = cv2.Sobel(
        gray,
        cv2.CV_64F,
        1,
        0,
        ksize=3
    )

    sobel_y = cv2.Sobel(
        gray,
        cv2.CV_64F,
        0,
        1,
        ksize=3
    )

    gradient_magnitude = np.sqrt(
        sobel_x ** 2 +
        sobel_y ** 2
    )

    tenengrad_variance = float(
        np.mean(
            gradient_magnitude ** 2
        )
    )

    # =========================================================
    # 4. FINE DETAIL RATIO
    # =========================================================

    high_frequency_energy = float(
        np.mean(
            np.abs(laplacian)
        )
    )

    gradient_energy = float(
        np.mean(
            gradient_magnitude
        )
    )

    fine_detail_ratio = float(
        high_frequency_energy /
        (gradient_energy + 1e-6)
    )

    # =========================================================
    # 5. LOCAL / TILE SHARPNESS
    # =========================================================

    grid = 4

    tile_h = h // grid
    tile_w = w // grid

    tile_scores = []

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

                tile = gray[
                    y1:y2,
                    x1:x2
                ]

                tile = cv2.GaussianBlur(
                    tile,
                    (3, 3),
                    0
                )

                tile_lap = cv2.Laplacian(
                    tile,
                    cv2.CV_64F
                )

                tile_scores.append(
                    float(tile_lap.var())
                )

    if not tile_scores:
        tile_scores = [
            laplacian_variance
        ]

    tile_p10 = float(
        np.percentile(
            tile_scores,
            10
        )
    )

    tile_p25 = float(
        np.percentile(
            tile_scores,
            25
        )
    )

    tile_p50 = float(
        np.percentile(
            tile_scores,
            50
        )
    )

    tile_p75 = float(
        np.percentile(
            tile_scores,
            75
        )
    )

    tile_p90 = float(
        np.percentile(
            tile_scores,
            90
        )
    )

    # =========================================================
    # 6. EDGE DENSITY
    # =========================================================

    edges = cv2.Canny(
        gray,
        50,
        150
    )

    edge_density = float(
        np.mean(edges > 0)
    )

    # =========================================================
    # 7. NORMALIZED SHARPNESS
    # =========================================================

    # We use a softer resolution normalization than before.
    #
    # Very small images should not automatically become
    # "blurry", but their raw metrics should not be
    # compared directly with huge images.

    scale_factor = max(
        resolution_scale,
        0.5
    )

    normalized_laplacian = (
        laplacian_variance /
        scale_factor
    )

    normalized_tenengrad = (
        tenengrad_variance /
        scale_factor
    )

    # =========================================================
    # 8. RELATIVE LOCAL SHARPNESS
    # =========================================================

    # If P50 is much lower than P90, sharpness is concentrated
    # in only a few regions. This can happen with text/strong
    # edges while other areas remain soft.

    local_uniformity = float(
        tile_p50 /
        (tile_p90 + 1e-6)
    )

    # =========================================================
    # 9. BLUR SIGNALS
    # =========================================================

    # A. Extremely low global sharpness
    global_blur = (
        normalized_laplacian <
        threshold * 0.75
    )

    # B. Weak local detail
    #
    # Use a less aggressive resolution adjustment.
    local_threshold = max(
        threshold * 0.9,
        80.0
    )

    local_blur = (
        tile_p50 <
        local_threshold
    )

    # C. Fine detail loss
    fine_detail_loss = (
        fine_detail_ratio < 0.12
    )

    # D. Low local uniformity
    #
    # Sharpness exists in a few areas but not across
    # the image.
    uneven_sharpness = (
        local_uniformity < 0.35
        and
        tile_p50 < threshold * 2.0
    )

    # E. Low edge density
    low_edge_density = (
        edge_density < 0.015
    )

    # =========================================================
    # 10. COMBINED DECISION
    # =========================================================

    # Strong evidence:
    strong_blur = (
        global_blur
        and
        (
            fine_detail_loss
            or
            local_blur
        )
    )

    # Fine detail loss + weak local detail
    detail_blur = (
        fine_detail_loss
        and
        (
            local_blur
            or
            uneven_sharpness
        )
    )

    # Very low edge information
    weak_image_structure = (
        global_blur
        and
        low_edge_density
    )

    is_blurry = bool(
        strong_blur
        or
        detail_blur
        or
        weak_image_structure
    )

    # =========================================================
    # 11. BLUR SEVERITY
    # =========================================================

    lap_score = float(
        np.clip(
            1.0 -
            (
                normalized_laplacian /
                max(threshold * 2.0, 1.0)
            ),
            0.0,
            1.0
        )
    )

    tile_score = float(
        np.clip(
            1.0 -
            (
                tile_p50 /
                max(threshold * 2.0, 1.0)
            ),
            0.0,
            1.0
        )
    )

    fine_score = float(
        np.clip(
            (
                0.20 -
                fine_detail_ratio
            ) / 0.20,
            0.0,
            1.0
        )
    )

    uniformity_score = float(
        np.clip(
            1.0 -
            local_uniformity,
            0.0,
            1.0
        )
    )

    if is_blurry:

        severity = float(
            np.clip(
                0.35 * lap_score
                + 0.30 * tile_score
                + 0.20 * fine_score
                + 0.15 * uniformity_score,
                0.15,
                1.0
            )
        )

    else:

        severity = 0.0

    # =========================================================
    # 12. HEURISTIC CONFIDENCE
    # =========================================================

    signals = [
        global_blur,
        local_blur,
        fine_detail_loss,
        uneven_sharpness,
    ]

    signal_count = sum(
        bool(x)
        for x in signals
    )

    if is_blurry:

        confidence = float(
            np.clip(
                0.55 +
                0.10 * signal_count,
                0.55,
                0.90
            )
        )

    else:

        confidence = 0.0

    # =========================================================
    # 13. DETAILS
    # =========================================================

    details = {

        "width": w,
        "height": h,

        "min_dimension_found": min_dim,
        "min_dimension_threshold": min_dimension,

        "resolution_scale": round(
            resolution_scale,
            3
        ),

        "laplacian_variance": round(
            laplacian_variance,
            3
        ),

        "tenengrad_variance": round(
            tenengrad_variance,
            3
        ),

        "norm_laplacian": round(
            normalized_laplacian,
            3
        ),

        "norm_tenengrad": round(
            normalized_tenengrad,
            3
        ),

        "tile_p10": round(
            tile_p10,
            3
        ),

        "tile_p25": round(
            tile_p25,
            3
        ),

        "tile_p50": round(
            tile_p50,
            3
        ),

        "tile_p75": round(
            tile_p75,
            3
        ),

        "tile_p90": round(
            tile_p90,
            3
        ),

        "fine_detail_ratio": round(
            fine_detail_ratio,
            3
        ),

        "edge_density": round(
            edge_density,
            4
        ),

        "local_uniformity": round(
            local_uniformity,
            3
        ),

        "blur_signals": {

            "global_blur": bool(
                global_blur
            ),

            "local_blur": bool(
                local_blur
            ),

            "fine_detail_loss": bool(
                fine_detail_loss
            ),

            "uneven_sharpness": bool(
                uneven_sharpness
            ),

            "weak_image_structure": bool(
                weak_image_structure
            ),
        },

        "blur_threshold": threshold,
    }

    logger.debug(
        f"Blur detector | "
        f"size={w}x{h} | "
        f"lap={laplacian_variance:.2f} | "
        f"tile_p50={tile_p50:.2f} | "
        f"fine={fine_detail_ratio:.3f} | "
        f"edge_density={edge_density:.4f} | "
        f"uniformity={local_uniformity:.3f} | "
        f"blur={is_blurry} | "
        f"severity={severity:.2f}"
    )

    return (
        is_blurry,
        severity,
        confidence,
        details,
    )