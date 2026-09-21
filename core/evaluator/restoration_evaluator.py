"""
Per-Operation Restoration Evaluator.

Evaluates image quality metrics before and after each restoration operation (no ground truth required).
Applies operation-specific decision rules to KEEP or DISCARD model outputs.
"""
from dataclasses import dataclass
from typing import Dict, Any, Tuple, Optional
import cv2
import numpy as np
from PIL import Image
from loguru import logger

from app.config import load_config
from core.analyzer.noise_detector import estimate_noise_sigma
from core.analyzer.jpeg_detector import detect_jpeg_artifacts


@dataclass
class EvaluationResult:
    """Structured record of a single operation's before/after evaluation."""
    operation: str
    decision: str  # "KEEP" or "DISCARD"
    reason: str
    before_metrics: Dict[str, Any]
    after_metrics: Dict[str, Any]
    improvement_metrics: Dict[str, Any]


class RestorationEvaluator:
    """
    Quality control evaluator for restoration model outputs.
    Compares image statistics before and after an operation to ensure that:
    1. Primary target metric improves.
    2. Secondary safety metrics stay within acceptable tolerances.
    3. Output image is valid and non-identical.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        if config is None:
            try:
                config = load_config()
            except Exception:
                config = {}
        self.config = config.get("evaluator", {})

    def validate_image_safety(
        self, before_image: Image.Image, after_image: Image.Image
    ) -> Tuple[bool, str]:
        """
        Verify safety constraints:
        - Output is a valid PIL Image
        - Output has valid non-zero dimensions
        - Pixel values are finite (no NaN/Inf)
        - Output is not completely identical to input
        """
        if after_image is None or not isinstance(after_image, Image.Image):
            return False, "Output is not a valid Image object"

        if after_image.width <= 0 or after_image.height <= 0:
            return False, f"Invalid image dimensions: {after_image.width}x{after_image.height}"

        try:
            arr_after = np.array(after_image.convert("RGB"), dtype=np.float32)
        except Exception as e:
            return False, f"Failed to convert image to array: {e}"

        if not np.isfinite(arr_after).all():
            return False, "Output image contains NaN or infinite values"

        # Check if identical (for same dimensions)
        if before_image.size == after_image.size:
            arr_before = np.array(before_image.convert("RGB"), dtype=np.float32)
            diff = np.abs(arr_after - arr_before)
            mad = float(np.mean(diff))
            if mad < 1e-3:
                return False, "Model output is identical to input image (no change)"

        return True, "Valid"

    def compute_image_metrics(self, image: Image.Image) -> Dict[str, float]:
        """Compute heuristic quality and structural metrics for an image."""
        rgb = np.array(image.convert("RGB"))
        gray_u8 = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        gray_f32 = gray_u8.astype(np.float32)

        # 1. Laplacian Sharpness
        lap_var = float(cv2.Laplacian(gray_u8, cv2.CV_64F).var())

        # 2. Tenengrad Sharpness
        sobelx = cv2.Sobel(gray_u8, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray_u8, cv2.CV_64F, 0, 1, ksize=3)
        tenengrad = float(np.mean(sobelx**2 + sobely**2))

        # 3. Noise Sigma (Immerkaer MAD method)
        noise_sig = estimate_noise_sigma(gray_u8)

        # 4. Exposure stats
        y_norm = gray_f32 / 255.0
        mean_lum = float(np.mean(y_norm))
        dark_ratio = float(np.mean(y_norm < 0.1))
        bright_ratio = float(np.mean(y_norm > 0.9))

        # 5. JPEG Blocking Score
        _, blocking_score, _, _ = detect_jpeg_artifacts(image)

        return {
            "laplacian": round(lap_var, 3),
            "tenengrad": round(tenengrad, 3),
            "noise_sigma": round(noise_sig, 3),
            "mean_luminance": round(mean_lum, 4),
            "dark_ratio": round(dark_ratio, 4),
            "bright_ratio": round(bright_ratio, 4),
            "blocking_score": round(blocking_score, 4),
            "width": float(image.width),
            "height": float(image.height),
        }

    def evaluate(
        self,
        operation: str,
        before_image: Image.Image,
        after_image: Image.Image,
    ) -> EvaluationResult:
        """
        Evaluate an operation step output against its before image.
        Returns EvaluationResult with decision 'KEEP' or 'DISCARD'.
        """
        # Step 1: Safety validation
        is_safe, safety_reason = self.validate_image_safety(before_image, after_image)
        if not is_safe:
            b_metrics = self.compute_image_metrics(before_image)
            return EvaluationResult(
                operation=operation,
                decision="DISCARD",
                reason=f"Safety check failed: {safety_reason}",
                before_metrics=b_metrics,
                after_metrics=b_metrics,
                improvement_metrics={},
            )

        before_m = self.compute_image_metrics(before_image)
        after_m = self.compute_image_metrics(after_image)

        def pct_change(after_val, before_val):
            val_b = max(abs(before_val), 1e-5)
            return ((after_val - before_val) / val_b) * 100.0

        changes = {
            "laplacian_pct": round(pct_change(after_m["laplacian"], before_m["laplacian"]), 2),
            "tenengrad_pct": round(pct_change(after_m["tenengrad"], before_m["tenengrad"]), 2),
            "noise_sigma_pct": round(pct_change(after_m["noise_sigma"], before_m["noise_sigma"]), 2),
            "mean_lum_pct": round(pct_change(after_m["mean_luminance"], before_m["mean_luminance"]), 2),
            "dark_ratio_change": round(after_m["dark_ratio"] - before_m["dark_ratio"], 4),
            "bright_ratio_change": round(after_m["bright_ratio"] - before_m["bright_ratio"], 4),
            "blocking_score_pct": round(pct_change(after_m["blocking_score"], before_m["blocking_score"]), 2),
        }

        min_imp = self.config.get("min_improvement_pct", 0.03) * 100.0

        decision = "DISCARD"
        reason = ""

        # Step 2: Operation-specific decision logic
        if operation == "denoise":
            cfg = self.config.get("denoise", {})
            min_noise_red = cfg.get("min_noise_reduction_pct", 0.03) * 100.0
            max_sharp_loss = cfg.get("max_sharpness_loss_pct", 0.15) * 100.0

            noise_reduced = (before_m["noise_sigma"] - after_m["noise_sigma"])
            noise_reduced_pct = -changes["noise_sigma_pct"]
            sharpness_loss_pct = -changes["laplacian_pct"]

            if noise_reduced_pct < min_noise_red and noise_reduced < 0.2:
                decision = "DISCARD"
                reason = f"Noise reduction insufficient (reduced by {noise_reduced_pct:.1f}%, min required: {min_noise_red:.1f}%)"
            elif before_m["noise_sigma"] < 5.0 and sharpness_loss_pct > max_sharp_loss:
                decision = "DISCARD"
                reason = f"Excessive loss of detail/sharpness on low-noise image (sharpness dropped by {sharpness_loss_pct:.1f}%, max allowed: {max_sharp_loss:.1f}%)"
            elif after_m["tenengrad"] < 5.0 and before_m["tenengrad"] > 50.0:
                decision = "DISCARD"
                reason = f"Denoising completely destroyed image structural gradients (Tenengrad dropped to {after_m['tenengrad']:.1f})"
            else:
                decision = "KEEP"
                reason = f"Noise estimate reduced by {noise_reduced_pct:.1f}% with acceptable detail preservation"

        elif operation == "deblur":
            cfg = self.config.get("deblur", {})
            min_sharp_gain = cfg.get("min_sharpness_gain_pct", 0.03) * 100.0
            max_noise_inc = cfg.get("max_noise_increase_pct", 0.35) * 100.0

            sharp_gain_pct = changes["laplacian_pct"]
            noise_inc_pct = changes["noise_sigma_pct"]

            if sharp_gain_pct < min_sharp_gain:
                decision = "DISCARD"
                reason = f"Sharpness did not improve sufficiently (Laplacian change: {sharp_gain_pct:+.1f}%, min required: +{min_sharp_gain:.1f}%)"
            elif noise_inc_pct > max_noise_inc and (after_m["noise_sigma"] - before_m["noise_sigma"]) > 1.0:
                decision = "DISCARD"
                reason = f"Excessive noise amplification (noise increased by {noise_inc_pct:.1f}%, max allowed: {max_noise_inc:.1f}%)"
            else:
                decision = "KEEP"
                reason = f"Sharpness improved by {sharp_gain_pct:+.1f}% without excessive noise increase"

        elif operation == "low_light":
            cfg = self.config.get("low_light", {})
            min_lum_gain = cfg.get("min_luminance_gain_pct", 0.03) * 100.0
            max_bright_clip = cfg.get("max_bright_clip_ratio", 0.25)

            lum_gain_pct = changes["mean_lum_pct"]
            dark_reduced = changes["dark_ratio_change"] < -0.01

            if lum_gain_pct < min_lum_gain and not dark_reduced:
                decision = "DISCARD"
                reason = f"Exposure did not improve (mean luminance gain: {lum_gain_pct:+.1f}%, min required: +{min_lum_gain:.1f}%)"
            elif after_m["bright_ratio"] > max_bright_clip and changes["bright_ratio_change"] > 0.05:
                decision = "DISCARD"
                reason = f"Excessive highlight clipping (bright ratio: {after_m['bright_ratio']:.2f}, max allowed: {max_bright_clip:.2f})"
            else:
                decision = "KEEP"
                reason = f"Luminance improved by {lum_gain_pct:+.1f}% with dark ratio reduction of {abs(changes['dark_ratio_change']):.2f}"

        elif operation == "jpeg_artifacts":
            cfg = self.config.get("jpeg_artifacts", {})
            min_block_red = cfg.get("min_blocking_reduction_pct", 0.03) * 100.0
            max_sharp_loss = cfg.get("max_sharpness_loss_pct", 0.15) * 100.0

            block_red_pct = -changes["blocking_score_pct"]
            sharp_loss_pct = -changes["laplacian_pct"]

            if block_red_pct < min_block_red and (before_m["blocking_score"] - after_m["blocking_score"]) < 0.02:
                decision = "DISCARD"
                reason = f"JPEG blocking score did not improve (reduced by {block_red_pct:.1f}%, min required: {min_block_red:.1f}%)"
            elif sharp_loss_pct > max_sharp_loss:
                decision = "DISCARD"
                reason = f"Excessive loss of detail during JPEG artifact removal (sharpness dropped by {sharp_loss_pct:.1f}%)"
            else:
                decision = "KEEP"
                reason = f"JPEG blocking score reduced by {block_red_pct:.1f}% while preserving image details"

        elif operation == "super_resolution":
            scaled = (after_image.width > before_image.width) or (after_image.height > before_image.height)
            if not scaled:
                decision = "DISCARD"
                reason = f"Super resolution output dimensions ({after_image.width}x{after_image.height}) did not increase from input ({before_image.width}x{before_image.height})"
            else:
                decision = "KEEP"
                reason = f"Resolution increased successfully from {before_image.width}x{before_image.height} to {after_image.width}x{after_image.height}"

        else:
            decision = "KEEP"
            reason = f"Generic operation '{operation}' completed safety checks"

        return EvaluationResult(
            operation=operation,
            decision=decision,
            reason=reason,
            before_metrics=before_m,
            after_metrics=after_m,
            improvement_metrics=changes,
        )
