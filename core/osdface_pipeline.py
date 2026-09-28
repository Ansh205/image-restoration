"""
OSDFace Pipeline — Orchestrates Phase 2 OSDFace face-specialist preprocessing stage
with the default 3-pass restoration pipeline.
"""
import time
import cv2
import numpy as np
from typing import Tuple, Dict, Any, Optional, List
from PIL import Image
from loguru import logger

from core.osdface_processor import OSDFaceProcessor
from core.osdface_evaluator import OSDFaceEvaluator
from core.pipeline.engine import RestorationEngine, EngineResult
from core.analyzer.analyzer import DegradationAnalyzer
from core.pipeline.planner import PipelinePlanner


class OSDFaceAssistedPipeline:
    """
    Phase 2 OSDFace-Assisted Restoration Pipeline.
    Runs YuNet -> OSDFace -> OSDFace Evaluator -> (if KEEP: candidate, else: original baseline) -> Degradation Analyzer -> 3-Pass Engine.
    """

    def __init__(self, device: str = "cpu"):
        self.device = device
        self.processor = OSDFaceProcessor(device=device)
        self.evaluator = OSDFaceEvaluator()

    def run(
        self,
        image: Image.Image,
        original_image: Image.Image,
        image_id: str,
        analyzer: DegradationAnalyzer,
        planner: PipelinePlanner,
        engine: RestorationEngine,
        custom_operations: Optional[List[str]] = None,
        upscale_4k: bool = False,
    ) -> Tuple[EngineResult, Dict[str, Any], Optional[Image.Image], str]:
        """
        Execute OSDFace-Assisted Pipeline.

        Args:
            image: Working inference image (PIL Image).
            original_image: Original uploaded baseline image (PIL Image).
            image_id: Unique image identifier.
            analyzer: DegradationAnalyzer instance.
            planner: PipelinePlanner instance.
            engine: RestorationEngine instance.
            custom_operations: Optional custom operation list.
            upscale_4k: 4K AI upscaling toggle.

        Returns:
            Tuple of (engine_result, osdface_metadata, osdface_candidate_pil, osdface_decision)
        """
        logger.info("============================================================")
        logger.info("[PHASE 2 — OSDFACE ASSISTED PIPELINE]")
        logger.info("============================================================")

        baseline_img = original_image or image
        baseline_bgr = cv2.cvtColor(np.array(baseline_img.convert("RGB")), cv2.COLOR_RGB2BGR)

        # 1. Detect faces using YuNet
        detected_bboxes = self.processor.detector.detect(baseline_bgr)
        h_img, w_img = baseline_bgr.shape[:2]
        img_area = h_img * w_img
        valid_bboxes = []

        for fx, fy, fw, fh in detected_bboxes:
            area_ratio = (fw * fh) / max(1, img_area)
            aspect_ratio = fw / max(1, fh)
            if area_ratio > 0.85 and (aspect_ratio < 0.3 or aspect_ratio > 3.0):
                continue
            valid_bboxes.append((fx, fy, fw, fh))

        face_count = len(valid_bboxes)
        logger.info(f"[OSDFACE] Faces detected: {face_count}")

        # ---------------------------------------------------------------------
        # CASE 1: NO VALID FACES DETECTED
        # ---------------------------------------------------------------------
        if face_count == 0:
            logger.info("[OSDFACE]")
            logger.info("No valid faces detected.")
            logger.info("[OSDFACE → DEFAULT PIPELINE]")
            logger.info("Skipping OSDFace and running default restoration pipeline.")

            engine_res = engine.run_three_pass(
                image=image,
                analyzer=analyzer,
                planner=planner,
                custom_operations=custom_operations,
                upscale_4k=upscale_4k,
                original_image=original_image,
                image_id=image_id,
            )

            osdface_meta = {
                "enabled": True,
                "faces_detected": 0,
                "candidate_created": False,
                "evaluation_decision": "SKIPPED",
                "evaluation_reason": "No valid face detected by YuNet detector.",
                "inference_time": 0.0,
                "total_time": engine_res.total_time_seconds,
            }

            return engine_res, osdface_meta, None, "SKIPPED"

        # ---------------------------------------------------------------------
        # CASE 2: 1+ VALID FACES DETECTED — RUN OSDFACE PREPROCESSING
        # ---------------------------------------------------------------------
        logger.info("[OSDFACE]")
        logger.info("Candidate generated.")

        t_osd_start = time.time()
        osd_out, _, face_restored = self.processor.process(baseline_img)
        t_osd_end = time.time()
        osd_time = round(t_osd_end - t_osd_start, 3)

        osd_candidate_pil = osd_out if isinstance(osd_out, Image.Image) else Image.fromarray(cv2.cvtColor(osd_out, cv2.COLOR_BGR2RGB))
        candidate_bgr = cv2.cvtColor(np.array(osd_candidate_pil.convert("RGB")), cv2.COLOR_RGB2BGR)

        # 2. Run OSDFace Evaluation
        decision, reason, eval_metrics = self.evaluator.evaluate(
            baseline_bgr=baseline_bgr,
            candidate_bgr=candidate_bgr,
            bboxes=valid_bboxes,
        )

        logger.info(f"[OSDFACE EVALUATION]")
        logger.info(f"Decision: {decision}")
        logger.info(f"Reason: {reason}")

        osdface_meta = {
            "enabled": True,
            "faces_detected": face_count,
            "candidate_created": True,
            "evaluation_decision": decision,
            "evaluation_reason": reason,
            "inference_time": osd_time,
            "avg_mad": eval_metrics.get("avg_mad", 0.0),
            "avg_changed_pct": eval_metrics.get("avg_changed_pct", 0.0),
            "avg_sharpness_change": eval_metrics.get("avg_sharpness_change", 0.0),
            "avg_noise_change": eval_metrics.get("avg_noise_change", 0.0),
            "avg_dR": eval_metrics.get("avg_dR", 0.0),
            "avg_dG": eval_metrics.get("avg_dG", 0.0),
            "avg_dB": eval_metrics.get("avg_dB", 0.0),
        }

        # 3. Feed Accepted / Rejected Candidate to Standard Restoration Pipeline
        if decision == "KEEP":
            logger.info("[OSDFACE → DEFAULT PIPELINE]")
            logger.info("Accepted OSDFace candidate passed to DegradationAnalyzer.")
            input_for_standard = osd_candidate_pil
        else:
            logger.info("[OSDFACE → DEFAULT PIPELINE]")
            logger.info("OSDFace candidate rejected.")
            logger.info("Using original baseline.")
            input_for_standard = image

        logger.info("[DEFAULT PIPELINE]")
        logger.info("Starting 3-pass restoration...")

        engine_res = engine.run_three_pass(
            image=input_for_standard,
            analyzer=analyzer,
            planner=planner,
            custom_operations=custom_operations,
            upscale_4k=upscale_4k,
            original_image=original_image,
            initial_candidate=(input_for_standard if decision == "KEEP" else None),
            image_id=image_id,
        )

        return engine_res, osdface_meta, osd_candidate_pil, decision
