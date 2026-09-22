"""
UMSN Diagnostic Follow-Up Script.
Executes Same-Resolution Evaluation, Upsampling Method Comparisons, and Blending Loss Isolation.
"""

import os
import sys
sys.path.insert(0, os.path.abspath("."))

import cv2
import numpy as np
from PIL import Image
from loguru import logger

from models.deblurring.umsn import UMSNModel


def compute_metrics(img_bgr: np.ndarray) -> dict:
    """Compute Laplacian, Tenengrad, and NoiseSigma metrics for a BGR image/crop."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY) if len(img_bgr.shape) == 3 else img_bgr

    # Laplacian variance
    lap = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    # Tenengrad gradient magnitude
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    tenengrad = float(np.mean(gx ** 2 + gy ** 2))

    # Noise sigma (Immerkaer method)
    h, w = gray.shape[:2]
    kernel = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float64)
    sigma = float(np.sum(np.abs(cv2.filter2D(gray.astype(np.float64), -1, kernel))))
    noise_sigma = sigma * np.sqrt(0.5 * np.pi) / (6.0 * max(1, w - 2) * max(1, h - 2))

    return {
        "laplacian": round(lap, 2),
        "tenengrad": round(tenengrad, 2),
        "noise_sigma": round(noise_sigma, 2),
    }


def generate_test_face_image(width=819, height=1024) -> Image.Image:
    """Generate high-contrast test image containing face structure for YuNet detection."""
    img = np.full((height, width, 3), (220, 220, 220), dtype=np.uint8)

    cx, cy = 256 + 316 // 2, 149 + 416 // 2
    axes = (316 // 2, 416 // 2)

    cv2.ellipse(img, (cx, cy), axes, 0, 0, 360, (180, 200, 230), -1)
    cv2.ellipse(img, (cx, cy), axes, 0, 0, 360, (50, 50, 50), 4)

    # Eyes
    cv2.ellipse(img, (cx - 70, cy - 60), (30, 18), 0, 0, 360, (255, 255, 255), -1)
    cv2.circle(img, (cx - 70, cy - 60), 12, (80, 40, 20), -1)
    cv2.circle(img, (cx - 70, cy - 60), 5, (0, 0, 0), -1)

    cv2.ellipse(img, (cx + 70, cy - 60), (30, 18), 0, 0, 360, (255, 255, 255), -1)
    cv2.circle(img, (cx + 70, cy - 60), 12, (80, 40, 20), -1)
    cv2.circle(img, (cx + 70, cy - 60), 5, (0, 0, 0), -1)

    # Eyebrows
    cv2.ellipse(img, (cx - 70, cy - 90), (35, 8), 0, 180, 360, (20, 20, 20), 6)
    cv2.ellipse(img, (cx + 70, cy - 90), (35, 8), 0, 180, 360, (20, 20, 20), 6)

    # Nose & Mouth
    cv2.line(img, (cx, cy - 40), (cx - 12, cy + 30), (120, 140, 170), 4)
    cv2.line(img, (cx - 12, cy + 30), (cx + 12, cy + 30), (120, 140, 170), 4)
    cv2.ellipse(img, (cx, cy + 90), (55, 25), 0, 0, 180, (40, 40, 160), -1)
    cv2.ellipse(img, (cx, cy + 90), (55, 25), 0, 0, 360, (20, 20, 100), 3)

    blurred = cv2.GaussianBlur(img, (9, 9), 3.0)
    return Image.fromarray(cv2.cvtColor(blurred, cv2.COLOR_BGR2RGB))


def main():
    logger.info("=== UMSN DIAGNOSTIC FOLLOW-UP ===")
    debug_dir = "debug/umsn"
    os.makedirs(debug_dir, exist_ok=True)

    test_img = generate_test_face_image()
    umsn = UMSNModel()
    restored_img = umsn.restore(test_img, debug_dir=debug_dir)

    # Read saved face crops & raw output
    orig_face = cv2.imread(os.path.join(debug_dir, "original_face_1.png"))
    raw_umsn = cv2.imread(os.path.join(debug_dir, "umsn_raw_output_1.png"))
    blended_face = cv2.imread(os.path.join(debug_dir, "umsn_blended_face_1.png"))

    h_orig, w_orig = orig_face.shape[:2]

    # =========================================================================
    # TASK 1: SAME-RESOLUTION EVALUATION (128x128)
    # =========================================================================
    orig_face_128 = cv2.resize(orig_face, (128, 128), interpolation=cv2.INTER_CUBIC)

    m_orig_128 = compute_metrics(orig_face_128)
    m_raw_128 = compute_metrics(raw_umsn)

    logger.info("\n------------------------------------------------------------")
    logger.info("TASK 1: SAME-RESOLUTION EVALUATION (128x128)")
    logger.info("------------------------------------------------------------")
    logger.info(f"Original @128:   Laplacian={m_orig_128['laplacian']}, Tenengrad={m_orig_128['tenengrad']}, NoiseSigma={m_orig_128['noise_sigma']}")
    logger.info(f"UMSN raw @128:   Laplacian={m_raw_128['laplacian']}, Tenengrad={m_raw_128['tenengrad']}, NoiseSigma={m_raw_128['noise_sigma']}")

    # =========================================================================
    # TASK 2: ISOLATE UPSAMPLING LOSS (128x128 -> original face size)
    # =========================================================================
    up_bicubic = cv2.resize(raw_umsn, (w_orig, h_orig), interpolation=cv2.INTER_CUBIC)
    up_lanczos = cv2.resize(raw_umsn, (w_orig, h_orig), interpolation=cv2.INTER_LANCZOS4)
    up_nearest = cv2.resize(raw_umsn, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)

    p_bicubic = os.path.join(debug_dir, "upsample_bicubic.png")
    p_lanczos = os.path.join(debug_dir, "upsample_lanczos.png")
    p_nearest = os.path.join(debug_dir, "upsample_nearest.png")

    cv2.imwrite(p_bicubic, up_bicubic)
    cv2.imwrite(p_lanczos, up_lanczos)
    cv2.imwrite(p_nearest, up_nearest)

    m_bicubic = compute_metrics(up_bicubic)
    m_lanczos = compute_metrics(up_lanczos)
    m_nearest = compute_metrics(up_nearest)

    logger.info("\n------------------------------------------------------------")
    logger.info("TASK 2: ISOLATE UPSAMPLING LOSS (Resized to original dimensions)")
    logger.info("------------------------------------------------------------")
    logger.info(f"Bicubic ({w_orig}x{h_orig}): Laplacian={m_bicubic['laplacian']} | Saved: {p_bicubic}")
    logger.info(f"Lanczos ({w_orig}x{h_orig}): Laplacian={m_lanczos['laplacian']} | Saved: {p_lanczos}")
    logger.info(f"Nearest ({w_orig}x{h_orig}): Laplacian={m_nearest['laplacian']} | Saved: {p_nearest}")

    # =========================================================================
    # TASK 3: ISOLATE BLENDING LOSS
    # =========================================================================
    m_blended = compute_metrics(blended_face)

    logger.info("\n------------------------------------------------------------")
    logger.info("TASK 3: ISOLATE BLENDING LOSS")
    logger.info("------------------------------------------------------------")
    logger.info(f"UMSN raw 128x128 Laplacian:        {m_raw_128['laplacian']}")
    logger.info(f"UMSN bicubic upsampled Laplacian:  {m_bicubic['laplacian']}")
    logger.info(f"UMSN blended Laplacian:            {m_blended['laplacian']}")

    # Determine largest quality loss phase
    loss_upsample_pct = ((m_bicubic['laplacian'] - m_raw_128['laplacian']) / max(m_raw_128['laplacian'], 1e-5)) * 100.0
    loss_blend_pct = ((m_blended['laplacian'] - m_bicubic['laplacian']) / max(m_bicubic['laplacian'], 1e-5)) * 100.0

    logger.info("\n============================================================")
    logger.info("FINAL SUMMARY")
    logger.info("============================================================")
    logger.info(f"Original @128:     {m_orig_128['laplacian']}")
    logger.info(f"UMSN raw @128:     {m_raw_128['laplacian']}")
    logger.info(f"Bicubic:           {m_bicubic['laplacian']}")
    logger.info(f"Lanczos:           {m_lanczos['laplacian']}")
    logger.info(f"Nearest:           {m_nearest['laplacian']}")
    logger.info(f"Bicubic + blend:   {m_blended['laplacian']}")
    logger.info("------------------------------------------------------------")
    if abs(loss_upsample_pct) > abs(loss_blend_pct):
        logger.info(f"LARGEST QUALITY LOSS OCCURS DURING: 2. upsampling (Laplacian dropped by {loss_upsample_pct:.1f}% from 128x128 to native dimensions)")
    else:
        logger.info(f"LARGEST QUALITY LOSS OCCURS DURING: 3. Gaussian blending (Laplacian dropped by {loss_blend_pct:.1f}% during blending)")


if __name__ == "__main__":
    main()
