"""
OSDFace Evaluator — Dedicated face-region quality, sharpness, noise, and color shift evaluator
for OSDFace restored candidates in Phase 2.
"""
import cv2
import numpy as np
from typing import List, Tuple, Dict, Any, Union
from PIL import Image
from loguru import logger


class OSDFaceEvaluator:
    """
    Evaluator to determine whether an OSDFace restored candidate should be accepted (KEEP)
    or rejected (DISCARD) before passing it to the default restoration pipeline.
    """

    @staticmethod
    def _compute_sharpness(img_bgr: np.ndarray) -> float:
        """Compute Laplacian variance as a proxy for image sharpness."""
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())

    @staticmethod
    def _compute_noise(img_bgr: np.ndarray) -> float:
        """Estimate high-frequency noise standard deviation using Gaussian blur difference."""
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        diff = gray - blurred
        return float(np.std(diff))

    def evaluate(
        self,
        baseline_bgr: np.ndarray,
        candidate_bgr: np.ndarray,
        bboxes: List[Tuple[int, int, int, int]]
    ) -> Tuple[str, str, Dict[str, Any]]:
        """
        Evaluate OSDFace candidate against baseline over face regions.

        Args:
            baseline_bgr: Original baseline image (BGR numpy array).
            candidate_bgr: OSDFace restored candidate image (BGR numpy array).
            bboxes: List of valid face bounding boxes (fx, fy, fw, fh).

        Returns:
            Tuple of (decision, reason, metrics_dict)
            decision: "KEEP" or "DISCARD"
            reason: Explanatory text string
            metrics_dict: Per-face and overall metric values
        """
        if not bboxes or len(bboxes) == 0:
            return "DISCARD", "No valid face bounding boxes provided.", {}

        h_img, w_img = baseline_bgr.shape[:2]
        face_count = len(bboxes)

        total_mad = 0.0
        total_changed_pct = 0.0
        sharpness_changes = []
        noise_changes = []
        color_shifts = []

        logger.info("============================================================")
        logger.info("[OSDFACE EVALUATION]")
        logger.info(f"Evaluating OSDFace candidate over {face_count} face region(s)...")

        for idx, (fx, fy, fw, fh) in enumerate(bboxes, start=1):
            x1 = max(0, fx)
            y1 = max(0, fy)
            x2 = min(w_img, fx + fw)
            y2 = min(h_img, fy + fh)

            crop_base = baseline_bgr[y1:y2, x1:x2]
            crop_cand = candidate_bgr[y1:y2, x1:x2]

            if crop_base.size == 0 or crop_cand.size == 0:
                continue

            # 1. MAD & Changed Pixels %
            diff = np.abs(crop_cand.astype(np.float32) - crop_base.astype(np.float32))
            mad = float(np.mean(diff))
            changed_pct = float(np.mean(diff > 1.0) * 100.0)
            total_mad += mad
            total_changed_pct += changed_pct

            # 2. Sharpness Change %
            s_base = self._compute_sharpness(crop_base)
            s_cand = self._compute_sharpness(crop_cand)
            s_change_pct = ((s_cand - s_base) / max(1.0, s_base)) * 100.0
            sharpness_changes.append(s_change_pct)

            # 3. Noise Change %
            n_base = self._compute_noise(crop_base)
            n_cand = self._compute_noise(crop_cand)
            n_change_pct = ((n_cand - n_base) / max(0.1, n_base)) * 100.0
            noise_changes.append(n_change_pct)

            # 4. Color Shifts (in RGB channel space)
            rgb_base = cv2.cvtColor(crop_base, cv2.COLOR_BGR2RGB).astype(np.float32)
            rgb_cand = cv2.cvtColor(crop_cand, cv2.COLOR_BGR2RGB).astype(np.float32)
            dR = float(rgb_cand[:, :, 0].mean() - rgb_base[:, :, 0].mean())
            dG = float(rgb_cand[:, :, 1].mean() - rgb_base[:, :, 1].mean())
            dB = float(rgb_cand[:, :, 2].mean() - rgb_base[:, :, 2].mean())
            color_shifts.append((dR, dG, dB))

            logger.info(f"Face {idx} Evaluation:")
            logger.info(f"    MAD: {mad:.4f}, Changed Pixels: {changed_pct:.2f}%")
            logger.info(f"    Sharpness Change: {s_change_pct:+.2f}% (Base: {s_base:.1f} -> Cand: {s_cand:.1f})")
            logger.info(f"    Noise Change: {n_change_pct:+.2f}%")
            logger.info(f"    RGB Shift: dR={dR:+.2f}, dG={dG:+.2f}, dB={dB:+.2f}")

        avg_mad = total_mad / max(1, face_count)
        avg_changed_pct = total_changed_pct / max(1, face_count)
        avg_sharpness_change = float(np.mean(sharpness_changes)) if sharpness_changes else 0.0
        avg_noise_change = float(np.mean(noise_changes)) if noise_changes else 0.0

        avg_dR = float(np.mean([cs[0] for cs in color_shifts])) if color_shifts else 0.0
        avg_dG = float(np.mean([cs[1] for cs in color_shifts])) if color_shifts else 0.0
        avg_dB = float(np.mean([cs[2] for cs in color_shifts])) if color_shifts else 0.0

        metrics = {
            "faces_evaluated": face_count,
            "avg_mad": avg_mad,
            "avg_changed_pct": avg_changed_pct,
            "avg_sharpness_change": avg_sharpness_change,
            "avg_noise_change": avg_noise_change,
            "avg_dR": avg_dR,
            "avg_dG": avg_dG,
            "avg_dB": avg_dB,
        }

        # Decision Logic
        # 1. Check if change is negligible
        if avg_mad < 0.1 or avg_changed_pct < 0.2:
            decision = "DISCARD"
            reason = "Negligible face restoration change detected (MAD < 0.1)."
        # 2. Check for extreme color cast / artifacting
        elif abs(avg_dR - avg_dG) > 20.0 or abs(avg_dB - avg_dG) > 20.0:
            decision = "DISCARD"
            reason = f"Excessive color cast detected (dR={avg_dR:+.1f}, dG={avg_dG:+.1f}, dB={avg_dB:+.1f})."
        # 3. Check for severe noise amplification without sharpness gain
        elif avg_noise_change > 150.0 and avg_sharpness_change < 0.0:
            decision = "DISCARD"
            reason = f"Excessive noise introduced (+{avg_noise_change:.1f}%) with negative sharpness change."
        else:
            decision = "KEEP"
            reason = f"Meaningful face restoration accepted (MAD={avg_mad:.2f}, Sharpness {avg_sharpness_change:+.1f}%, clean color balance)."

        logger.info("------------------------------------------------------------")
        logger.info(f"Face Sharpness Change: {avg_sharpness_change:+.1f}%")
        logger.info(f"Face Noise Change: {avg_noise_change:+.1f}%")
        logger.info(f"RGB Shift: dR={avg_dR:+.2f}, dG={avg_dG:+.2f}, dB={avg_dB:+.2f}")
        logger.info(f"MAD: {avg_mad:.4f}")
        logger.info(f"Decision: {decision}")
        logger.info(f"Reason: {reason}")
        logger.info("============================================================")

        return decision, reason, metrics
