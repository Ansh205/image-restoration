"""
Per-Operation Restoration Evaluator.

Evaluates image quality metrics before and after each restoration operation (no ground truth required).
Applies Level 1 Global Hard Safety Gates and Level 2 Operation-Specific decision rules to KEEP or DISCARD model outputs.
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
    hard_safety_passed: bool = True
    before_metrics: Dict[str, Any] = None
    after_metrics: Dict[str, Any] = None
    improvement_metrics: Dict[str, Any] = None

    def __post_init__(self):
        if self.before_metrics is None:
            self.before_metrics = {}
        if self.after_metrics is None:
            self.after_metrics = {}
        if self.improvement_metrics is None:
            self.improvement_metrics = {}


# Configurable Safety and Pass-Specific Threshold Constants
MAX_LUMINANCE_DROP = 0.15           # 15% max luminance drop under conditional evaluation
CATASTROPHIC_LUMINANCE_DROP = 0.30  # 30% drop -> IMMEDIATE DISCARD (hard safety failure)

PASS1_MAX_NOISE_INCREASE = 2.0      # +200% max noise increase in Pass 1 (permissive)
PASS2_NOISE_SOFT_LIMIT = 1.5        # +150% soft noise limit in Pass 2 (medium/relaxed)
PASS2_NOISE_HARD_LIMIT = 2.0        # +200% hard noise limit in Pass 2
PASS3_NOISE_MAX_INCREASE = 0.50     # +50% max noise increase in Pass 3 (strict)


class RestorationEvaluator:
    """
    Quality control evaluator for restoration model outputs.
    Compares image statistics before and after an operation to ensure that:
    1. Output passes Level 1 Global Hard Safety Gates (luminance preservation, non-collapse, finite values).
    2. Target degradation metric improves without violating Level 2 operation-specific rules.
    3. Output image is valid and non-identical.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        if config is None:
            try:
                config = load_config()
            except Exception:
                config = {}
        self.config = config.get("evaluator", {})

        # Expose threshold constants on evaluator instance for configurability
        self.max_luminance_drop = float(self.config.get("max_luminance_drop", MAX_LUMINANCE_DROP))
        self.catastrophic_luminance_drop = float(self.config.get("catastrophic_luminance_drop", CATASTROPHIC_LUMINANCE_DROP))

        self.pass1_max_noise_increase = float(self.config.get("pass1_max_noise_increase", PASS1_MAX_NOISE_INCREASE))
        self.pass2_noise_soft_limit = float(self.config.get("pass2_noise_soft_limit", PASS2_NOISE_SOFT_LIMIT))
        self.pass2_noise_hard_limit = float(self.config.get("pass2_noise_hard_limit", PASS2_NOISE_HARD_LIMIT))
        self.pass3_noise_max_increase = float(self.config.get("pass3_noise_max_increase", PASS3_NOISE_MAX_INCREASE))

    def validate_image_safety(
        self, before_image: Image.Image, after_image: Image.Image
    ) -> Tuple[bool, str]:
        """
        Verify basic object safety constraints:
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
                return True, "Model output is identical to input image (no change)"

        return True, "Valid"

    def check_hard_safety_gates(
        self,
        before_m: Dict[str, Any],
        after_m: Dict[str, Any],
        changes: Dict[str, Any],
        operation: str,
    ) -> Tuple[bool, str]:
        """
        Level 1: Global Hard Safety Gates.
        Applies across ALL THREE PASSES to override pass permissiveness.
        Prevents catastrophic image failures (such as -79% luminance drops, near-black collapse,
        massive noise explosions, or gradient annihilation).
        Returns (passed: bool, reason: str).
        """
        before_lum = float(before_m.get("mean_luminance", 0.0))
        after_lum = float(after_m.get("mean_luminance", 0.0))

        # 1. Catastrophic Luminance Drop Gate
        if before_lum > 1e-4:
            lum_drop_ratio = (before_lum - after_lum) / before_lum
            if lum_drop_ratio > self.catastrophic_luminance_drop:
                return False, (
                    f"HARD SAFETY FAILURE: Catastrophic luminance drop from {before_lum:.4f} to {after_lum:.4f} "
                    f"(-{lum_drop_ratio * 100.0:.1f}%, limit: {self.catastrophic_luminance_drop * 100.0:.0f}%)"
                )

        # 2. Near-Black Image Collapse Gate
        if after_lum < 0.08 and before_lum >= 0.15:
            return False, (
                f"HARD SAFETY FAILURE: Output image collapsed to near-black "
                f"(mean luminance: {before_lum:.4f} -> {after_lum:.4f})"
            )

        # 3. Severe Luminance Drop + Dark Ratio Surge Gate
        if before_lum > 1e-4:
            lum_drop_ratio = (before_lum - after_lum) / before_lum
            dark_ratio_change = float(after_m.get("dark_ratio", 0.0) - before_m.get("dark_ratio", 0.0))
            if lum_drop_ratio > self.max_luminance_drop and dark_ratio_change > 0.10:
                return False, (
                    f"HARD SAFETY FAILURE: Luminance dropped by {lum_drop_ratio * 100.0:.1f}% "
                    f"with dark pixel surge ({before_m.get('dark_ratio', 0.0):.2f} -> {after_m.get('dark_ratio', 0.0):.2f})"
                )

        # 4. Catastrophic Noise Explosion Gate
        noise_inc_pct = float(changes.get("noise_sigma_pct", 0.0))
        noise_diff = float(after_m.get("noise_sigma", 0.0) - before_m.get("noise_sigma", 0.0))
        if noise_inc_pct > 300.0 and noise_diff > 5.0:
            return False, (
                f"HARD SAFETY FAILURE: Severe noise explosion "
                f"(noise increased by {noise_inc_pct:.1f}%, +{noise_diff:.2f} sigma)"
            )

        # 5. Structural Contrast / Gradient Annihilation Gate
        before_ten = float(before_m.get("tenengrad", 0.0))
        after_ten = float(after_m.get("tenengrad", 0.0))
        if after_ten < 1.0 and before_ten > 20.0:
            return False, (
                f"HARD SAFETY FAILURE: Structural contrast annihilated "
                f"(Tenengrad dropped from {before_ten:.1f} to {after_ten:.1f})"
            )

        return True, "Hard safety gates passed"

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
        pass_number: int = 1,
        op_run_count: int = 1,
    ) -> EvaluationResult:
        """
        Two-Level Quality Evaluation:
        Level 1: Global Hard Safety Gates (applies in ALL 3 passes, overrides pass permissiveness).
        Level 2: Operation-Specific Rules (evaluates target degradation improvements and pass noise limits).

        Returns EvaluationResult with decision 'KEEP' or 'DISCARD'.
        """
        # Step 1: Object Safety validation
        is_safe, safety_reason = self.validate_image_safety(before_image, after_image)
        if not is_safe:
            b_metrics = self.compute_image_metrics(before_image)
            return EvaluationResult(
                operation=operation,
                decision="DISCARD",
                reason=f"Safety check failed: {safety_reason}",
                hard_safety_passed=False,
                before_metrics=b_metrics,
                after_metrics=b_metrics,
                improvement_metrics={},
            )

        img_before_arr = np.array(before_image.convert("RGB"))
        img_after_arr = np.array(after_image.convert("RGB"))
        h_img, w_img = img_before_arr.shape[:2]

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

        # Check if output image is identical to input image
        if np.array_equal(img_before_arr, img_after_arr):
            return EvaluationResult(
                operation=operation,
                decision="DISCARD",
                reason="Model output is identical/no meaningful restoration change.",
                hard_safety_passed=True,
                before_metrics=before_m,
                after_metrics=after_m,
                improvement_metrics=changes,
            )

        # Step 2: Level 1 — Global Hard Safety Gates (Applies across ALL 3 Passes)
        hard_safe, hard_reason = self.check_hard_safety_gates(before_m, after_m, changes, operation)
        if not hard_safe:
            logger.warning(f"[{operation.upper()}] {hard_reason}")
            return EvaluationResult(
                operation=operation,
                decision="DISCARD",
                reason=hard_reason,
                hard_safety_passed=False,
                before_metrics=before_m,
                after_metrics=after_m,
                improvement_metrics=changes,
            )

        # Select max noise increase ratio based on pass_number
        if pass_number == 1:
            max_noise_ratio = self.pass1_max_noise_increase   # 2.0 = 200%
        elif pass_number == 2:
            max_noise_ratio = self.pass2_noise_hard_limit     # 2.0 = 200%
        else:
            max_noise_ratio = self.pass3_noise_max_increase   # 0.50 = 50%

        max_noise_inc_pct = max_noise_ratio * 100.0

        decision = "KEEP"
        reason = ""

        # Step 3: Level 2 — Operation-specific decision logic per pass_number
        if operation == "low_light":
            lum_gain_pct = changes["mean_lum_pct"]
            noise_inc_pct = changes["noise_sigma_pct"]
            dark_reduced = changes["dark_ratio_change"] < -0.01

            # Rule: Low-light model MUST NOT make the image darker in any pass!
            if lum_gain_pct < -5.0 or (after_m["mean_luminance"] < before_m["mean_luminance"] - 0.02 and not dark_reduced):
                decision = "DISCARD"
                reason = (
                    f"[Low Light] Low light restoration made image darker "
                    f"(mean luminance: {before_m['mean_luminance']:.4f} -> {after_m['mean_luminance']:.4f}, change: {lum_gain_pct:+.1f}%)"
                )
            elif pass_number == 1:
                if noise_inc_pct > max_noise_inc_pct and (after_m["noise_sigma"] - before_m["noise_sigma"]) > 3.0:
                    decision = "DISCARD"
                    reason = f"[Pass 1] Excessive noise amplification during low-light enhancement (noise increased by {noise_inc_pct:.1f}%, Pass 1 limit: {max_noise_inc_pct:.1f}%)"
                else:
                    decision = "KEEP"
                    reason = f"[Pass 1] Low light accepted under permissive Pass 1 thresholds (luminance change: {lum_gain_pct:+.1f}%)"

            elif pass_number == 2:
                soft_limit_pct = self.pass2_noise_soft_limit * 100.0
                if noise_inc_pct > max_noise_inc_pct and (after_m["noise_sigma"] - before_m["noise_sigma"]) > 2.0:
                    decision = "DISCARD"
                    reason = f"[Pass 2] Noise increased by {noise_inc_pct:.1f}%, exceeding Pass 2 hard limit of {max_noise_inc_pct:.1f}%"
                elif noise_inc_pct > soft_limit_pct and lum_gain_pct <= 0 and not dark_reduced:
                    decision = "DISCARD"
                    reason = f"[Pass 2] Noise increased by {noise_inc_pct:.1f}% without luminance improvement"
                else:
                    decision = "KEEP"
                    reason = f"[Pass 2] Low light accepted under Pass 2 medium thresholds (luminance change: {lum_gain_pct:+.1f}%)"

            else:
                if noise_inc_pct > max_noise_inc_pct and (after_m["noise_sigma"] - before_m["noise_sigma"]) > 1.0:
                    decision = "DISCARD"
                    reason = f"[Pass 3] Noise increased by {noise_inc_pct:.1f}%, exceeding Pass 3 strict limit of {max_noise_inc_pct:.1f}%"
                elif lum_gain_pct < 0 and not dark_reduced:
                    decision = "DISCARD"
                    reason = f"[Pass 3] Exposure did not improve in strict Pass 3 (luminance change: {lum_gain_pct:+.1f}%)"
                else:
                    decision = "KEEP"
                    reason = f"[Pass 3] Low light accepted under Pass 3 strict thresholds (luminance change: {lum_gain_pct:+.1f}%)"

        elif operation == "denoise":
            noise_reduced_pct = -changes["noise_sigma_pct"]
            noise_inc_pct = changes["noise_sigma_pct"]
            sharpness_loss_pct = -changes["laplacian_pct"]
            lum_loss_pct = -changes["mean_lum_pct"]

            # Rule: Denoising MUST NOT cause unacceptable luminance drop (>10%)
            if lum_loss_pct > 10.0:
                decision = "DISCARD"
                reason = f"Denoising caused unacceptable luminance decrease of {lum_loss_pct:.1f}% (mean lum: {before_m['mean_luminance']:.4f} -> {after_m['mean_luminance']:.4f})"
            elif pass_number == 1:
                if noise_inc_pct > max_noise_inc_pct and (after_m["noise_sigma"] - before_m["noise_sigma"]) > 2.0:
                    decision = "DISCARD"
                    reason = f"[Pass 1] Excessive noise increase (noise increased by {noise_inc_pct:.1f}%, Pass 1 limit: {max_noise_inc_pct:.1f}%)"
                elif after_m["tenengrad"] < 2.0 and before_m["tenengrad"] > 50.0:
                    decision = "DISCARD"
                    reason = f"[Pass 1] Denoising completely destroyed image structural gradients (Tenengrad dropped to {after_m['tenengrad']:.1f})"
                else:
                    decision = "KEEP"
                    reason = f"[Pass 1] Accepted under permissive Pass 1 thresholds (noise change: {changes['noise_sigma_pct']:+.1f}%)"

            elif pass_number == 2:
                soft_limit_pct = self.pass2_noise_soft_limit * 100.0
                if noise_inc_pct > max_noise_inc_pct and (after_m["noise_sigma"] - before_m["noise_sigma"]) > 1.5:
                    decision = "DISCARD"
                    reason = f"[Pass 2] Noise increased by {noise_inc_pct:.1f}%, exceeding Pass 2 hard limit of {max_noise_inc_pct:.1f}%"
                elif noise_inc_pct > soft_limit_pct and noise_reduced_pct <= 0 and (after_m["noise_sigma"] - before_m["noise_sigma"]) > 1.0:
                    decision = "DISCARD"
                    reason = f"[Pass 2] Noise increased by {noise_inc_pct:.1f}%, exceeding Pass 2 soft limit of {soft_limit_pct:.1f}% without improvement"
                elif sharpness_loss_pct > 35.0 and before_m["noise_sigma"] < 5.0:
                    decision = "DISCARD"
                    reason = f"[Pass 2] Excessive loss of detail (sharpness dropped by {sharpness_loss_pct:.1f}%)"
                else:
                    decision = "KEEP"
                    reason = f"[Pass 2] Accepted under Pass 2 medium thresholds (noise change: {changes['noise_sigma_pct']:+.1f}%)"

            else:
                if noise_inc_pct > max_noise_inc_pct and (after_m["noise_sigma"] - before_m["noise_sigma"]) > 0.5:
                    decision = "DISCARD"
                    reason = f"[Pass 3] Noise increased by {noise_inc_pct:.1f}%, exceeding Pass 3 strict limit of {max_noise_inc_pct:.1f}%"
                elif sharpness_loss_pct > 25.0:
                    decision = "DISCARD"
                    reason = f"[Pass 3] Excessive sharpness loss (dropped by {sharpness_loss_pct:.1f}%, Pass 3 max allowed: 25.0%)"
                else:
                    decision = "KEEP"
                    reason = f"[Pass 3] Accepted under Pass 3 strict thresholds (noise change: {changes['noise_sigma_pct']:+.1f}%)"

        elif operation == "deblur":
            lum_loss_pct = -changes["mean_lum_pct"]

            # Rule: Deblurring MUST NOT cause unacceptable luminance drop (>10%)
            if lum_loss_pct > 10.0:
                decision = "DISCARD"
                reason = f"Deblurring caused unacceptable luminance decrease of {lum_loss_pct:.1f}% (mean lum: {before_m['mean_luminance']:.4f} -> {after_m['mean_luminance']:.4f})"
            else:
                # Detect face crop regions if available
                faces = []
                try:
                    from models.deblurring.umsn import UMSNModel
                    bgr_before = cv2.cvtColor(img_before_arr, cv2.COLOR_RGB2BGR)
                    dummy_umsn = UMSNModel()
                    faces = dummy_umsn._detect_faces(bgr_before)
                except Exception:
                    faces = []

                if len(faces) > 0:
                    fx, fy, fw, fh = faces[0]
                    pad_x, pad_y = int(fw * 0.2), int(fh * 0.2)
                    x1, y1 = max(0, fx - pad_x), max(0, fy - pad_y)
                    x2, y2 = min(w_img, fx + fw + pad_x), min(h_img, fy + fh + pad_y)

                    crop_b = Image.fromarray(img_before_arr[y1:y2, x1:x2])
                    crop_a = Image.fromarray(img_after_arr[y1:y2, x1:x2])

                    fb_m = self.compute_image_metrics(crop_b)
                    fa_m = self.compute_image_metrics(crop_a)

                    face_sharp_gain = pct_change(fa_m["laplacian"], fb_m["laplacian"])
                    face_noise_inc = pct_change(fa_m["noise_sigma"], fb_m["noise_sigma"])

                    changes["laplacian_pct"] = round(face_sharp_gain, 2)
                    changes["noise_sigma_pct"] = round(face_noise_inc, 2)

                    logger.info(f"[DEBLUR FACE - PASS {pass_number}] Sharpness change = {face_sharp_gain:+.2f}%, Noise change = {face_noise_inc:+.2f}%")

                    if face_sharp_gain > 300.0 and (face_noise_inc > 80.0 or (fa_m["noise_sigma"] - fb_m["noise_sigma"]) > 2.0):
                        decision = "DISCARD"
                        reason = (
                            f"[DEBLUR EVALUATION] Sharpness change: {face_sharp_gain:+.1f}%, Face noise change: {face_noise_inc:+.1f}%. "
                            f"Decision: DISCARD. Reason: Potential over-sharpening detected. Sharpness increased substantially together with excessive noise/edge amplification. Candidate: Previous accepted candidate preserved."
                        )
                    elif op_run_count >= 2 and (face_noise_inc > 30.0 or face_sharp_gain < 10.0):
                        decision = "DISCARD"
                        reason = (
                            f"[DEBLUR EVALUATION] [Execution 2/2] Stricter second execution evaluation: face noise change ({face_noise_inc:+.1f}%) "
                            f"or face sharpness gain ({face_sharp_gain:+.1f}%). Decision: DISCARD. Reason: Second execution requires stronger quality improvement without edge/noise amplification. Candidate: Previous accepted candidate preserved."
                        )
                    elif pass_number == 1:
                        if face_noise_inc > max_noise_inc_pct and (fa_m["noise_sigma"] - fb_m["noise_sigma"]) > 3.0:
                            decision = "DISCARD"
                            reason = f"[Pass 1] Excessive face noise amplification (increased by {face_noise_inc:.1f}%, Pass 1 limit: {max_noise_inc_pct:.1f}%)"
                        elif face_sharp_gain < -30.0:
                            decision = "DISCARD"
                            reason = f"[Pass 1] Face region severely degraded (sharpness dropped by {abs(face_sharp_gain):.1f}%)"
                        else:
                            decision = "KEEP"
                            reason = f"[Pass 1] Face crop accepted under permissive Pass 1 thresholds (sharpness change: {face_sharp_gain:+.1f}%)"

                    elif pass_number == 2:
                        soft_limit_pct = self.pass2_noise_soft_limit * 100.0
                        if face_noise_inc > max_noise_inc_pct and (fa_m["noise_sigma"] - fb_m["noise_sigma"]) > 2.0:
                            decision = "DISCARD"
                            reason = f"[Pass 2] Face noise increased by {face_noise_inc:.1f}%, exceeding Pass 2 hard limit of {max_noise_inc_pct:.1f}%"
                        elif face_noise_inc > soft_limit_pct and face_sharp_gain <= 0:
                            decision = "DISCARD"
                            reason = f"[Pass 2] Face noise increased by {face_noise_inc:.1f}% without sharpness improvement"
                        elif face_sharp_gain < -15.0:
                            decision = "DISCARD"
                            reason = f"[Pass 2] Face sharpness degraded significantly (dropped by {abs(face_sharp_gain):.1f}%)"
                        else:
                            decision = "KEEP"
                            reason = f"[Pass 2] Face crop accepted under Pass 2 medium thresholds (sharpness change: {face_sharp_gain:+.1f}%)"

                    else:
                        if face_noise_inc > max_noise_inc_pct and (fa_m["noise_sigma"] - fb_m["noise_sigma"]) > 1.0:
                            decision = "DISCARD"
                            reason = f"[Pass 3] Face noise increased by {face_noise_inc:.1f}%, exceeding Pass 3 strict limit of {max_noise_inc_pct:.1f}%"
                        elif face_sharp_gain < -5.0:
                            decision = "DISCARD"
                            reason = f"[Pass 3] Face sharpness decreased in strict Pass 3 (dropped by {abs(face_sharp_gain):.1f}%)"
                        else:
                            decision = "KEEP"
                            reason = f"[Pass 3] Face crop accepted under Pass 3 strict thresholds (sharpness change: {face_sharp_gain:+.1f}%)"

                else:
                    sharp_gain_pct = changes["laplacian_pct"]
                    noise_inc_pct = changes["noise_sigma_pct"]

                    if sharp_gain_pct > 300.0 and (noise_inc_pct > 80.0 or (after_m["noise_sigma"] - before_m["noise_sigma"]) > 2.0):
                        decision = "DISCARD"
                        reason = (
                            f"[DEBLUR EVALUATION] Sharpness change: {sharp_gain_pct:+.1f}%, Noise change: {noise_inc_pct:+.1f}%. "
                            f"Decision: DISCARD. Reason: Potential over-sharpening detected. Sharpness increased substantially together with excessive noise/edge amplification. Candidate: Previous accepted candidate preserved."
                        )
                    elif op_run_count >= 2 and (noise_inc_pct > 30.0 or sharp_gain_pct < 10.0):
                        decision = "DISCARD"
                        reason = (
                            f"[DEBLUR EVALUATION] [Execution 2/2] Stricter second execution evaluation: noise change ({noise_inc_pct:+.1f}%) "
                            f"or sharpness gain ({sharp_gain_pct:+.1f}%). Decision: DISCARD. Reason: Second execution requires stronger quality improvement without edge/noise amplification. Candidate: Previous accepted candidate preserved."
                        )
                    elif pass_number == 1:
                        if noise_inc_pct > max_noise_inc_pct and (after_m["noise_sigma"] - before_m["noise_sigma"]) > 3.0:
                            decision = "DISCARD"
                            reason = f"[Pass 1] Excessive noise amplification (noise increased by {noise_inc_pct:.1f}%, Pass 1 limit: {max_noise_inc_pct:.1f}%)"
                        elif sharp_gain_pct < -30.0:
                            decision = "DISCARD"
                            reason = f"[Pass 1] Severe sharpness loss (dropped by {abs(sharp_gain_pct):.1f}%)"
                        else:
                            decision = "KEEP"
                            reason = f"[Pass 1] Deblur output accepted under permissive Pass 1 thresholds (sharpness change: {sharp_gain_pct:+.1f}%)"

                    elif pass_number == 2:
                        soft_limit_pct = self.pass2_noise_soft_limit * 100.0
                        if noise_inc_pct > max_noise_inc_pct and (after_m["noise_sigma"] - before_m["noise_sigma"]) > 2.0:
                            decision = "DISCARD"
                            reason = f"[Pass 2] Noise increased by {noise_inc_pct:.1f}%, exceeding Pass 2 hard limit of {max_noise_inc_pct:.1f}%"
                        elif noise_inc_pct > soft_limit_pct and sharp_gain_pct <= 0:
                            decision = "DISCARD"
                            reason = f"[Pass 2] Noise increased by {noise_inc_pct:.1f}% without sharpness improvement"
                        elif sharp_gain_pct < -15.0:
                            decision = "DISCARD"
                            reason = f"[Pass 2] Sharpness dropped by {abs(sharp_gain_pct):.1f}%"
                        else:
                            decision = "KEEP"
                            reason = f"[Pass 2] Deblur output accepted under Pass 2 medium thresholds (sharpness change: {sharp_gain_pct:+.1f}%)"

                    else:
                        if noise_inc_pct > max_noise_inc_pct and (after_m["noise_sigma"] - before_m["noise_sigma"]) > 1.0:
                            decision = "DISCARD"
                            reason = f"[Pass 3] Noise increased by {noise_inc_pct:.1f}%, exceeding Pass 3 strict limit of {max_noise_inc_pct:.1f}%"
                        elif sharp_gain_pct < -5.0:
                            decision = "DISCARD"
                            reason = f"[Pass 3] Sharpness decreased in strict Pass 3 (dropped by {abs(sharp_gain_pct):.1f}%)"
                        else:
                            decision = "KEEP"
                            reason = f"[Pass 3] Deblur output accepted under Pass 3 strict thresholds (sharpness change: {sharp_gain_pct:+.1f}%)"

        elif operation == "jpeg_artifacts":
            sharp_loss_pct = -changes["laplacian_pct"]
            lum_loss_pct = -changes["mean_lum_pct"]

            if lum_loss_pct > 10.0:
                decision = "DISCARD"
                reason = f"JPEG artifact removal caused unacceptable luminance decrease of {lum_loss_pct:.1f}%"
            elif pass_number == 1:
                if sharp_loss_pct > 40.0:
                    decision = "DISCARD"
                    reason = f"[Pass 1] Severe detail destruction during JPEG artifact removal (sharpness dropped by {sharp_loss_pct:.1f}%)"
                else:
                    decision = "KEEP"
                    reason = f"[Pass 1] JPEG artifact removal accepted under permissive Pass 1 thresholds"

            elif pass_number == 2:
                if sharp_loss_pct > 25.0:
                    decision = "DISCARD"
                    reason = f"[Pass 2] Excessive detail loss during JPEG artifact removal (sharpness dropped by {sharp_loss_pct:.1f}%)"
                else:
                    decision = "KEEP"
                    reason = f"[Pass 2] JPEG artifact removal accepted under Pass 2 medium thresholds"

            else:
                if sharp_loss_pct > 15.0:
                    decision = "DISCARD"
                    reason = f"[Pass 3] Excessive detail loss in strict Pass 3 (sharpness dropped by {sharp_loss_pct:.1f}%)"
                else:
                    decision = "KEEP"
                    reason = f"[Pass 3] JPEG artifact removal accepted under Pass 3 strict thresholds"

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
            hard_safety_passed=True,
            before_metrics=before_m,
            after_metrics=after_m,
            improvement_metrics=changes,
        )
