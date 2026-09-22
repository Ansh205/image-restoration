"""
Noise Detection Module.

Estimates Gaussian noise level (sigma) using local patch standard deviation.
"""
'''
from typing import Tuple

import cv2
import numpy as np
from PIL import Image
from loguru import logger


def estimate_noise_sigma(gray: np.ndarray) -> float:
    """
    Estimate noise standard deviation (sigma) using median absolute deviation of Laplacian response.
    Implements Immerkaer / Donoho noise estimation method with MAD for robustness against edges.
    """
    # Kernel for noise estimation
    kernel = np.array([[1, -2, 1],
                       [-2, 4, -2],
                       [1, -2, 1]], dtype=np.float32)
    
    lap = cv2.filter2D(gray.astype(np.float32), -1, kernel)
    sigma = np.median(np.abs(lap)) * np.sqrt(0.5 * np.pi) / (6.0 * 0.6745)
    return float(sigma)


def estimate_smooth_tile_noise(gray: np.ndarray) -> float:
    """
    Estimate noise standard deviation in smooth (low-gradient) patches of the image.
    Helps isolate true sensor/grain noise from strong structural edges.
    """
    h, w = gray.shape[:2]
    grid = 4
    tile_h, tile_w = h // grid, w // grid
    if tile_h < 8 or tile_w < 8:
        return float(np.std(gray))

    tile_stds = []
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = np.sqrt(sobel_x**2 + sobel_y**2)

    for r in range(grid):
        for c in range(grid):
            tile_grad = grad_mag[r*tile_h:(r+1)*tile_h, c*tile_w:(c+1)*tile_w]
            tile_pix = gray[r*tile_h:(r+1)*tile_h, c*tile_w:(c+1)*tile_w]
            # If tile has relatively low mean gradient (smooth area)
            if np.mean(tile_grad) < 15.0:
                tile_stds.append(float(np.std(tile_pix)))

    if tile_stds:
        return float(np.median(tile_stds))
    return float(np.std(gray))


def detect_noise(image: Image.Image, threshold: float = 15.0) -> Tuple[bool, float, float | None, dict]:
    """
    Detect noise in PIL Image using Donoho MAD estimation and smooth tile variance analysis.

    Args:
        image: Input PIL Image (RGB).
        threshold: Noise sigma threshold (scale 0-255).

    Returns:
        Tuple of (is_noisy, severity, confidence, details_dict)
    """
    np_img = np.array(image)
    if np_img.ndim == 3:
        gray = cv2.cvtColor(np_img, cv2.COLOR_RGB2GRAY)
    else:
        gray = np_img.copy()

    mad_sigma = estimate_noise_sigma(gray)
    smooth_sigma = estimate_smooth_tile_noise(gray)

    # Combined estimated sigma (taking conservative minimum to prevent edge false positives)
    effective_sigma = float(min(mad_sigma, max(smooth_sigma, mad_sigma * 0.7)))
    is_noisy = effective_sigma > threshold

    if is_noisy:
        severity = float(np.clip((effective_sigma - threshold) / 35.0, 0.0, 1.0))
    else:
        severity = 0.0

    details = {
        "estimated_noise_sigma": round(effective_sigma, 3),
        "mad_noise_sigma": round(mad_sigma, 3),
        "smooth_tile_sigma": round(smooth_sigma, 3),
        "noise_sigma_threshold": threshold
    }
    confidence = float(np.clip(0.60 + 0.30 * abs(effective_sigma - threshold) / (threshold + 1e-5), 0.50, 0.95))

    logger.debug(f"Noise detector: effective_sigma={effective_sigma:.2f} (mad={mad_sigma:.2f}, smooth={smooth_sigma:.2f}), threshold={threshold}, is_noisy={is_noisy}")
    return is_noisy, severity, confidence, details
'''



"""
Robust Noise Detection Module.

Detects image noise using multiple signals:

1. Robust MAD/Laplacian noise estimation
2. Gaussian residual noise
3. Noise estimation in smooth regions
4. Local noise consistency
5. High-frequency noise ratio

The detector is designed to avoid confusing strong edges and
natural texture with sensor/compression noise.

NOTE:
This is a heuristic detector. Without a labeled dataset, the
score should be treated as an image-quality indicator rather
than a calibrated probability.
"""

from typing import Tuple, Dict, Any

import cv2
import numpy as np

from PIL import Image
from loguru import logger


# ============================================================
# 1. ROBUST MAD NOISE ESTIMATION
# ============================================================

def estimate_noise_sigma(
    gray: np.ndarray
) -> float:

    """
    Estimate noise sigma using a robust Laplacian/MAD method.

    Strong edges can produce large responses, therefore the
    median is used instead of the mean.
    """

    gray = gray.astype(
        np.float32
    )

    kernel = np.array(
        [
            [1, -2, 1],
            [-2, 4, -2],
            [1, -2, 1],
        ],
        dtype=np.float32,
    )

    lap = cv2.filter2D(
        gray,
        -1,
        kernel
    )

    median_abs = np.median(
        np.abs(lap)
    )

    sigma = (
        median_abs
        * np.sqrt(0.5 * np.pi)
        / (6.0 * 0.6745)
    )

    return float(
        sigma
    )


# ============================================================
# 2. RESIDUAL NOISE ESTIMATION
# ============================================================

def estimate_residual_noise(
    gray: np.ndarray
) -> float:

    """
    Estimate high-frequency residual noise.

    A Gaussian-smoothed image is treated as the underlying
    signal and the residual is treated as high-frequency
    noise/detail.
    """

    gray_f = gray.astype(
        np.float32
    )

    smooth = cv2.GaussianBlur(
        gray_f,
        (5, 5),
        0
    )

    residual = (
        gray_f - smooth
    )

    # Robust standard deviation
    median_residual = np.median(
        residual
    )

    mad = np.median(
        np.abs(
            residual -
            median_residual
        )
    )

    sigma = (
        1.4826 * mad
    )

    return float(
        sigma
    )


# ============================================================
# 3. SMOOTH REGION NOISE
# ============================================================

def estimate_smooth_region_noise(
    gray: np.ndarray
) -> Tuple[float, float, int]:

    """
    Estimate noise primarily inside smooth regions.

    Returns:

        noise_sigma
        percentage_of_image_considered_smooth
        number_of_valid_tiles
    """

    h, w = gray.shape[:2]

    grid = 4

    tile_h = h // grid
    tile_w = w // grid

    if tile_h < 8 or tile_w < 8:

        return (
            0.0,
            0.0,
            0
        )

    gray_f = gray.astype(
        np.float32
    )

    sobel_x = cv2.Sobel(
        gray_f,
        cv2.CV_32F,
        1,
        0,
        ksize=3
    )

    sobel_y = cv2.Sobel(
        gray_f,
        cv2.CV_32F,
        0,
        1,
        ksize=3
    )

    gradient = np.sqrt(
        sobel_x ** 2 +
        sobel_y ** 2
    )

    tile_noise = []

    total_tiles = grid * grid

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

            tile = gray_f[
                y1:y2,
                x1:x2
            ]

            tile_gradient = gradient[
                y1:y2,
                x1:x2
            ]

            mean_gradient = float(
                np.mean(
                    tile_gradient
                )
            )

            # Smooth region.
            #
            # This threshold is deliberately not extremely
            # restrictive because small/compressed images
            # may have weak gradients even when noisy.

            if mean_gradient < 25.0:

                # Remove low-frequency illumination
                # variation before estimating noise.

                tile_smooth = cv2.GaussianBlur(
                    tile,
                    (5, 5),
                    0
                )

                residual = (
                    tile -
                    tile_smooth
                )

                median_residual = np.median(
                    residual
                )

                mad = np.median(
                    np.abs(
                        residual -
                        median_residual
                    )
                )

                sigma = (
                    1.4826 * mad
                )

                tile_noise.append(
                    float(sigma)
                )

    if not tile_noise:

        return (
            0.0,
            0.0,
            0
        )

    noise_sigma = float(
        np.median(
            tile_noise
        )
    )

    smooth_percentage = (
        len(tile_noise) /
        total_tiles
    )

    return (
        noise_sigma,
        float(smooth_percentage),
        len(tile_noise)
    )


# ============================================================
# 4. LOCAL NOISE VARIATION
# ============================================================

def estimate_local_noise_variation(
    gray: np.ndarray
) -> float:

    """
    Estimate how consistently high-frequency residuals
    appear across the image.

    This helps distinguish isolated texture from
    image-wide noise.
    """

    h, w = gray.shape[:2]

    grid = 4

    tile_h = h // grid
    tile_w = w // grid

    if tile_h < 8 or tile_w < 8:

        return 0.0

    gray_f = gray.astype(
        np.float32
    )

    noise_values = []

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

            tile = gray_f[
                y1:y2,
                x1:x2
            ]

            smooth = cv2.GaussianBlur(
                tile,
                (5, 5),
                0
            )

            residual = (
                tile -
                smooth
            )

            median_residual = np.median(
                residual
            )

            mad = np.median(
                np.abs(
                    residual -
                    median_residual
                )
            )

            sigma = (
                1.4826 * mad
            )

            noise_values.append(
                float(sigma)
            )

    if not noise_values:

        return 0.0

    values = np.asarray(
        noise_values,
        dtype=np.float32
    )

    mean_value = float(
        np.mean(values)
    )

    if mean_value < 1e-6:

        return 0.0

    # Coefficient of variation.
    #
    # Lower = noise is distributed more uniformly.
    # Higher = noise is concentrated in certain areas.

    variation = float(
        np.std(values) /
        mean_value
    )

    return variation


# ============================================================
# 5. MAIN NOISE DETECTOR
# ============================================================

def detect_noise(
    image: Image.Image,
    threshold: float = 8.0
) -> Tuple[
    bool,
    float,
    float,
    Dict[str, Any]
]:

    """
    Detect image noise using multiple measurements.

    Args:
        image:
            Input PIL image.

        threshold:
            Base noise sigma threshold.

            This is NOT the only decision criterion.
            Multiple signals are combined.

    Returns:

        (
            is_noisy,
            severity,
            confidence,
            details
        )
    """

'''
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
        np.uint8
    )

    h, w = gray.shape[:2]

    # ========================================================
    # 1. MAD NOISE
    # ========================================================

    mad_sigma = estimate_noise_sigma(
        gray
    )

    # ========================================================
    # 2. RESIDUAL NOISE
    # ========================================================

    residual_sigma = estimate_residual_noise(
        gray
    )

    # ========================================================
    # 3. SMOOTH REGION NOISE
    # ========================================================

    smooth_sigma, smooth_percentage, valid_tiles = (
        estimate_smooth_region_noise(
            gray
        )
    )

    # ========================================================
    # 4. LOCAL VARIATION
    # ========================================================

    local_variation = (
        estimate_local_noise_variation(
            gray
        )
    )

    # ========================================================
    # 5. HIGH FREQUENCY ENERGY
    # ========================================================

    gray_f = gray.astype(
        np.float32
    )

    smooth = cv2.GaussianBlur(
        gray_f,
        (5, 5),
        0
    )

    residual = (
        gray_f -
        smooth
    )

    high_frequency_energy = float(
        np.std(
            residual
        )
    )

    # ========================================================
    # 6. EDGE DENSITY
    # ========================================================

    edges = cv2.Canny(
        gray,
        50,
        150
    )

    edge_density = float(
        np.mean(
            edges > 0
        )
    )

    # ========================================================
    # 7. COMBINE ESTIMATES
    # ========================================================

    estimates = [
        mad_sigma,
        residual_sigma,
    ]

    if smooth_sigma > 0:

        estimates.append(
            smooth_sigma
        )

    # Median is more robust than taking the minimum.
    #
    # Your previous implementation used min(...), which could
    # suppress the final noise estimate.

    effective_sigma = float(
        np.median(
            estimates
        )
    )

    # ========================================================
    # 8. SIGNALS
    # ========================================================

    # Strong noise estimate
    strong_noise = (
        effective_sigma >
        threshold
    )

    # Residual high-frequency noise
    residual_noise = (
        residual_sigma >
        threshold * 0.90
    )

    # Smooth-region noise
    smooth_region_noise = (
        smooth_sigma > 0
        and
        smooth_sigma >
        threshold * 0.80
    )

    # Multiple independent estimators agree.
    estimator_agreement = (
        sum(
            [
                mad_sigma > threshold * 0.80,
                residual_sigma > threshold * 0.80,
                (
                    smooth_sigma > 0
                    and
                    smooth_sigma >
                    threshold * 0.80
                ),
            ]
        )
        >= 2
    )

    # ========================================================
    # 9. FINAL DECISION
    # ========================================================

    is_noisy = bool(
        strong_noise
        or
        (
            residual_noise
            and
            smooth_region_noise
        )
        or
        (
            estimator_agreement
            and
            high_frequency_energy >
            threshold * 0.75
        )
    )

    # ========================================================
    # 10. SEVERITY
    # ========================================================

    # 1.0 around roughly 3x threshold.

    severity = float(
        np.clip(
            (
                effective_sigma -
                threshold
            )
            /
            (
                threshold * 2.0
            ),
            0.0,
            1.0
        )
    )

    # If several independent signals agree,
    # slightly increase severity.

    if estimator_agreement:

        severity = float(
            np.clip(
                severity + 0.10,
                0.0,
                1.0
            )
        )

    # ========================================================
    # 11. CONFIDENCE
    # ========================================================

    agreement_count = sum(
        [
            mad_sigma > threshold * 0.80,
            residual_sigma > threshold * 0.80,
            (
                smooth_sigma > 0
                and
                smooth_sigma >
                threshold * 0.80
            ),
        ]
    )

    if is_noisy:

        confidence = float(
            np.clip(
                0.55 +
                0.10 * agreement_count,
                0.55,
                0.90
            )
        )

    else:

        confidence = float(
            np.clip(
                0.60 +
                0.05 * (
                    3 -
                    agreement_count
                ),
                0.60,
                0.80
            )
        )

    # ========================================================
    # 12. DETAILS
    # ========================================================

    details = {

        "width": w,

        "height": h,

        "estimated_noise_sigma": round(
            effective_sigma,
            3
        ),

        "mad_noise_sigma": round(
            mad_sigma,
            3
        ),

        "residual_noise_sigma": round(
            residual_sigma,
            3
        ),

        "smooth_tile_sigma": round(
            smooth_sigma,
            3
        ),

        "smooth_region_percentage": round(
            smooth_percentage,
            3
        ),

        "valid_smooth_tiles": (
            valid_tiles
        ),

        "high_frequency_energy": round(
            high_frequency_energy,
            3
        ),

        "local_noise_variation": round(
            local_variation,
            3
        ),

        "edge_density": round(
            edge_density,
            4
        ),

        "noise_sigma_threshold": (
            threshold
        ),

        "noise_signals": {

            "strong_noise": bool(
                strong_noise
            ),

            "residual_noise": bool(
                residual_noise
            ),

            "smooth_region_noise": bool(
                smooth_region_noise
            ),

            "estimator_agreement": bool(
                estimator_agreement
            ),
        },
    }

    # ========================================================
    # 13. LOGGING
    # ========================================================

    logger.debug(
        f"Noise detector | "
        f"size={w}x{h} | "
        f"effective={effective_sigma:.2f} | "
        f"mad={mad_sigma:.2f} | "
        f"residual={residual_sigma:.2f} | "
        f"smooth={smooth_sigma:.2f} | "
        f"HF={high_frequency_energy:.2f} | "
        f"edge={edge_density:.4f} | "
        f"noisy={is_noisy} | "
        f"severity={severity:.2f}"
    )

    return (
        is_noisy,
        severity,
        confidence,
        details,
    )

'''




"""
Noise Detector.

Detects random / sensor-like image noise using multiple signals:

1. MAD/Laplacian noise estimate
2. Gaussian residual noise
3. Smooth-region noise
4. Local noise variation
5. High-frequency energy
6. Edge density

The detector is heuristic because the project currently has
no labeled noise dataset.
"""

from typing import Any, Dict, Tuple

import cv2
import numpy as np
from PIL import Image


def _clip(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return float(np.clip(value, low, high))


# ============================================================
# 1. MAD / LAPLACIAN NOISE ESTIMATION
# ============================================================

def estimate_noise_sigma(gray: np.ndarray) -> float:
    """
    Estimate noise using a high-pass Laplacian-like filter.

    The result is approximately in 8-bit pixel intensity units.
    """

    gray_f = gray.astype(np.float32)

    kernel = np.array(
        [
            [1, -2, 1],
            [-2, 4, -2],
            [1, -2, 1],
        ],
        dtype=np.float32,
    )

    lap = cv2.filter2D(
        gray_f,
        cv2.CV_32F,
        kernel,
    )

    median_abs = np.median(
        np.abs(lap)
    )

    sigma = (
        median_abs
        * np.sqrt(0.5 * np.pi)
        / (6.0 * 0.6745)
    )

    return float(sigma)


# ============================================================
# 2. GAUSSIAN RESIDUAL NOISE
# ============================================================

def estimate_residual_noise(gray: np.ndarray) -> float:
    """
    Estimate high-frequency residual after removing
    low-frequency image structure.
    """

    gray_f = gray.astype(np.float32)

    smooth = cv2.GaussianBlur(
        gray_f,
        (5, 5),
        0,
    )

    residual = gray_f - smooth

    # Robust MAD estimate
    median = np.median(residual)

    mad = np.median(
        np.abs(residual - median)
    )

    sigma = 1.4826 * mad

    return float(sigma)


# ============================================================
# 3. SMOOTH-REGION NOISE
# ============================================================

def estimate_smooth_region_noise(
    gray: np.ndarray,
) -> Tuple[float, int]:
    """
    Estimate noise in relatively smooth image regions.

    Strong edges/textures are avoided because they can
    incorrectly look like noise.
    """

    h, w = gray.shape

    grid = 4

    tile_h = h // grid
    tile_w = w // grid

    if tile_h < 8 or tile_w < 8:
        return float(np.std(gray)), 0

    gray_f = gray.astype(np.float32)

    sobel_x = cv2.Sobel(
        gray_f,
        cv2.CV_32F,
        1,
        0,
        ksize=3,
    )

    sobel_y = cv2.Sobel(
        gray_f,
        cv2.CV_32F,
        0,
        1,
        ksize=3,
    )

    grad_mag = cv2.magnitude(
        sobel_x,
        sobel_y,
    )

    tile_sigmas = []

    for row in range(grid):

        for col in range(grid):

            y1 = row * tile_h
            y2 = (
                (row + 1) * tile_h
                if row < grid - 1
                else h
            )

            x1 = col * tile_w
            x2 = (
                (col + 1) * tile_w
                if col < grid - 1
                else w
            )

            tile_gray = gray_f[
                y1:y2,
                x1:x2,
            ]

            tile_grad = grad_mag[
                y1:y2,
                x1:x2,
            ]

            if tile_gray.size == 0:
                continue

            mean_gradient = float(
                np.mean(tile_grad)
            )

            # Smooth region
            if mean_gradient < 20.0:

                sigma = float(
                    np.std(tile_gray)
                )

                tile_sigmas.append(
                    sigma
                )

    if not tile_sigmas:
        return 0.0, 0

    return (
        float(np.median(tile_sigmas)),
        len(tile_sigmas),
    )


# ============================================================
# 4. LOCAL NOISE VARIATION
# ============================================================

def estimate_local_noise_variation(
    gray: np.ndarray,
) -> float:
    """
    Measure variation between local neighborhoods.
    """

    gray_f = gray.astype(np.float32)

    local_mean = cv2.GaussianBlur(
        gray_f,
        (7, 7),
        0,
    )

    local_sq_mean = cv2.GaussianBlur(
        gray_f ** 2,
        (7, 7),
        0,
    )

    local_variance = np.maximum(
        local_sq_mean
        - local_mean ** 2,
        0,
    )

    local_std = np.sqrt(
        local_variance
    )

    # Use a robust percentile instead of maximum.
    value = np.percentile(
        local_std,
        75,
    )

    return float(value)


# ============================================================
# 5. HIGH FREQUENCY ENERGY
# ============================================================

def estimate_high_frequency_energy(
    gray: np.ndarray,
) -> float:
    """
    Estimate high-frequency image energy.
    """

    gray_f = gray.astype(np.float32)

    blur = cv2.GaussianBlur(
        gray_f,
        (5, 5),
        0,
    )

    high_freq = (
        gray_f - blur
    )

    energy = float(
        np.std(high_freq)
    )

    return energy


# ============================================================
# 6. EDGE DENSITY
# ============================================================

def estimate_edge_density(
    gray: np.ndarray,
) -> float:

    edges = cv2.Canny(
        gray,
        50,
        150,
    )

    density = np.mean(
        edges > 0
    )

    return float(density)


# ============================================================
# MAIN DETECTOR
# ============================================================

def detect_noise(
    image: Image.Image,
    threshold: float = 3.0,
) -> Tuple[
    bool,
    float,
    float,
    Dict[str, Any],
]:

    # --------------------------------------------------------
    # Convert image
    # --------------------------------------------------------

    gray = np.array(
        image.convert("L"),
        dtype=np.uint8,
    )

    h, w = gray.shape

    # --------------------------------------------------------
    # Calculate noise signals
    # --------------------------------------------------------

    mad_sigma = estimate_noise_sigma(
        gray
    )

    residual_sigma = estimate_residual_noise(
        gray
    )

    smooth_sigma, valid_smooth_tiles = (
        estimate_smooth_region_noise(
            gray
        )
    )

    local_noise_variation = (
        estimate_local_noise_variation(
            gray
        )
    )

    high_frequency_energy = (
        estimate_high_frequency_energy(
            gray
        )
    )

    edge_density = estimate_edge_density(
        gray
    )

    # --------------------------------------------------------
    # Combine independent estimates
    # --------------------------------------------------------

    estimates = [
        mad_sigma,
        residual_sigma,
    ]

    effective_sigma = float(
        np.median(estimates)
    )

    noise_score = float(
        round(
            max(
                effective_sigma,
                mad_sigma,
                residual_sigma,
            ),
            3,
        )
    )

    # --------------------------------------------------------
    # Final decision: Actionable IF AND ONLY IF noise_score >= threshold
    # --------------------------------------------------------

    is_noisy = bool(
        noise_score >= threshold
    )

    # --------------------------------------------------------
    # Severity
    # --------------------------------------------------------

    if is_noisy:

        excess = (
            noise_score
            - threshold
        )

        severity = _clip(
            excess
            / max(
                threshold * 2.0,
                1e-6,
            )
        )

        severity = _clip(
            severity
        )

    else:

        severity = 0.0

    # --------------------------------------------------------
    # Confidence
    # --------------------------------------------------------
    #
    # This is heuristic confidence.
    # It is NOT statistically calibrated because there is
    # currently no labeled noise dataset.
    # --------------------------------------------------------

    confidence = 0.85 if is_noisy else 0.75

    # --------------------------------------------------------
    # Detailed metrics
    # --------------------------------------------------------

    details = {

        "estimated_noise_sigma": round(
            noise_score,
            3,
        ),

        "mad_noise_sigma": round(
            mad_sigma,
            3,
        ),

        "residual_noise_sigma": round(
            residual_sigma,
            3,
        ),

        "smooth_tile_sigma": round(
            smooth_sigma,
            3,
        ),

        "valid_smooth_tiles": int(
            valid_smooth_tiles
        ),

        "local_noise_variation": round(
            local_noise_variation,
            3,
        ),

        "high_frequency_energy": round(
            high_frequency_energy,
            3,
        ),

        "edge_density": round(
            edge_density,
            4,
        ),

        "noise_sigma_threshold": float(
            threshold
        ),

        "image_width": int(w),

        "image_height": int(h),
    }

    return (
        bool(is_noisy),
        float(severity),
        float(confidence),
        details,
    )