"""
Restoration Engine — Sequentially executes model wrappers in an ordered restoration pipeline.
"""
import time
from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple, Optional
from PIL import Image
import numpy as np
import cv2
from loguru import logger

from models.factory import get_model
from core.pipeline.image_utils import image_to_bytes
from core.evaluator.restoration_evaluator import RestorationEvaluator, EvaluationResult


@dataclass
class PipelineStepResult:
    """Result metadata for a single pipeline execution step."""
    step_number: int
    operation: str
    model_name: str
    execution_time_seconds: float
    input_size: Tuple[int, int]
    output_size: Tuple[int, int]


@dataclass
class EngineResult:
    """Overall result from running the restoration engine."""
    final_image: Image.Image
    pipeline_steps: List[Dict[str, Any]]
    total_time_seconds: float
    metrics: Dict[str, Any] = field(default_factory=dict)
    analysis_report: Optional[Any] = None


def _compute_basic_metrics(original: Image.Image, restored: Image.Image) -> Dict[str, Any]:
    """Compute comprehensive before/after quality and difference metrics with float precision."""
    # Ensure both images are evaluated at the same comparison size
    if original.size != restored.size:
        orig_eval = original.resize(restored.size, Image.Resampling.LANCZOS).convert("RGB")
    else:
        orig_eval = original.convert("RGB")
    rest_eval = restored.convert("RGB")

    orig_np = np.array(orig_eval, dtype=np.float32)
    rest_np = np.array(rest_eval, dtype=np.float32)

    # 1. Pixel-level differences
    diff = np.abs(rest_np - orig_np)
    mad = float(np.mean(diff))
    max_diff = float(np.max(diff))
    changed_pixels_pct = float(np.mean(diff > 1.0) * 100.0)

    # 2. Laplacian Variance (Sharpness)
    orig_gray = cv2.cvtColor(orig_np.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    rest_gray = cv2.cvtColor(rest_np.astype(np.uint8), cv2.COLOR_RGB2GRAY)

    orig_lap = float(cv2.Laplacian(orig_gray, cv2.CV_64F).var())
    rest_lap = float(cv2.Laplacian(rest_gray, cv2.CV_64F).var())

    lap_change_pct = ((rest_lap - orig_lap) / max(orig_lap, 1e-5)) * 100.0

    # 3. Tenengrad Sharpness (Sobel Gradient Variance)
    sobelx_orig = cv2.Sobel(orig_gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely_orig = cv2.Sobel(orig_gray, cv2.CV_64F, 0, 1, ksize=3)
    orig_tenengrad = float(np.mean(sobelx_orig**2 + sobely_orig**2))

    sobelx_rest = cv2.Sobel(rest_gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely_rest = cv2.Sobel(rest_gray, cv2.CV_64F, 0, 1, ksize=3)
    rest_tenengrad = float(np.mean(sobelx_rest**2 + sobely_rest**2))

    tenengrad_change_pct = ((rest_tenengrad - orig_tenengrad) / max(orig_tenengrad, 1e-5)) * 100.0

    return {
        "original_sharpness": round(orig_lap, 2),
        "restored_sharpness": round(rest_lap, 2),
        "sharpness_change_percent": round(lap_change_pct, 2),
        "tenengrad_change_percent": round(tenengrad_change_pct, 2),
        "mean_absolute_difference": round(mad, 3),
        "max_pixel_difference": round(max_diff, 1),
        "changed_pixels_percent": round(changed_pixels_pct, 2),
        "original_resolution": f"{original.width}x{original.height}",
        "restored_resolution": f"{restored.width}x{restored.height}",
    }


class RestorationEngine:
    """
    Sequentially processes an image through an ordered list of restoration operations.
    Supports single-pass execution and two-pass post-restoration degradation re-evaluation
    with per-operation before/after quality evaluation.
    """

    def __init__(self, device: str = "cpu", evaluator: Optional[RestorationEvaluator] = None):
        self.device = device
        self.evaluator = evaluator or RestorationEvaluator()

    def run(
        self,
        image: Image.Image,
        operations: List[str],
        original_image: Optional[Image.Image] = None,
    ) -> EngineResult:
        """
        Execute pipeline operations sequentially while preserving resolution and tracking detailed step metrics.

        Args:
            image: Working PIL Image (RGB) for model inference.
            operations: Ordered list of operation names (e.g. ['denoise', 'deblur']).
            original_image: Original high-res uploaded PIL Image for reference comparison and size reconstruction.

        Returns:
            EngineResult containing final restored image, step breakdown, total time, and metrics.
        """
        original_ref = original_image or image
        
        if not operations:
            logger.info("Empty pipeline. Returning original image.")
            return EngineResult(
                final_image=original_ref.copy(),
                pipeline_steps=[],
                total_time_seconds=0.0,
                metrics=_compute_basic_metrics(original_ref, original_ref),
            )

        start_total_time = time.time()
        working_img = image.copy()
        step_records = []

        logger.info(f"Starting restoration engine with pipeline: {operations}")

        for idx, op in enumerate(operations, start=1):
            t0 = time.time()
            input_size = (working_img.width, working_img.height)
            image_before = working_img.copy()

            try:
                model = get_model(op)
                logger.info(f"Step {idx}/{len(operations)}: Running '{op}' using {model.name} on size {input_size[0]}x{input_size[1]}")
                restored_step = model.predict(image_before)
            except Exception as e:
                logger.error(f"Error executing step '{op}': {e}. Skipping step.")
                continue

            dt = round(time.time() - t0, 3)
            output_size = (restored_step.width, restored_step.height)

            # Evaluate before/after
            eval_res = self.evaluator.evaluate(op, image_before, restored_step)

            logger.info(f"[{op.upper()}] Before:")
            logger.info(f"    Laplacian={eval_res.before_metrics.get('laplacian', 0.0):.2f}")
            logger.info(f"    NoiseSigma={eval_res.before_metrics.get('noise_sigma', 0.0):.2f}")
            logger.info(f"[{op.upper()}] After:")
            logger.info(f"    Laplacian={eval_res.after_metrics.get('laplacian', 0.0):.2f}")
            logger.info(f"    NoiseSigma={eval_res.after_metrics.get('noise_sigma', 0.0):.2f}")
            logger.info(f"[{op.upper()}] Decision: {eval_res.decision}")
            logger.info(f"[{op.upper()}] Reason: {eval_res.reason}")

            if image_before.size != restored_step.size:
                step_inp_eval = np.array(image_before.resize(restored_step.size, Image.Resampling.LANCZOS).convert("RGB"), dtype=np.float32)
            else:
                step_inp_eval = np.array(image_before.convert("RGB"), dtype=np.float32)
            step_out_eval = np.array(restored_step.convert("RGB"), dtype=np.float32)
            step_diff = np.abs(step_out_eval - step_inp_eval)
            step_mad = float(np.mean(step_diff))
            step_changed_pct = float(np.mean(step_diff > 1.0) * 100.0)

            step_info = {
                "step_number": idx,
                "pass_number": 1,
                "operation": op,
                "model_name": model.name,
                "decision": eval_res.decision,
                "reason": eval_res.reason,
                "execution_time_seconds": dt,
                "input_size": f"{input_size[0]}x{input_size[1]}",
                "output_size": f"{output_size[0]}x{output_size[1]}",
                "laplacian_before": eval_res.before_metrics.get("laplacian"),
                "laplacian_after": eval_res.after_metrics.get("laplacian"),
                "laplacian_change_percent": eval_res.improvement_metrics.get("laplacian_pct", 0.0),
                "noise_sigma_before": eval_res.before_metrics.get("noise_sigma"),
                "noise_sigma_after": eval_res.after_metrics.get("noise_sigma"),
                "mean_absolute_difference": round(step_mad, 3),
                "changed_pixels_percent": round(step_changed_pct, 2),
                "before_metrics": eval_res.before_metrics,
                "after_metrics": eval_res.after_metrics,
                "improvement_metrics": eval_res.improvement_metrics,
            }
            step_records.append(step_info)

            if eval_res.decision == "KEEP":
                working_img = restored_step
            else:
                logger.warning(f"[{op.upper()}] Output DISCARDED. Retaining image_before for subsequent operations.")

        if "super_resolution" not in operations:
            orig_w, orig_h = original_ref.width, original_ref.height
            if (working_img.width, working_img.height) != (orig_w, orig_h):
                logger.info(f"Reconstructing final output size from model resolution ({working_img.width}x{working_img.height}) back to original resolution ({orig_w}x{orig_h})")
                working_img = working_img.resize((orig_w, orig_h), Image.Resampling.LANCZOS)

        total_dt = round(time.time() - start_total_time, 3)
        metrics = _compute_basic_metrics(original_ref, working_img)

        logger.info(f"Pipeline completed in {total_dt}s across {len(step_records)} steps. Final resolution: {working_img.width}x{working_img.height}")

        return EngineResult(
            final_image=working_img,
            pipeline_steps=step_records,
            total_time_seconds=total_dt,
            metrics=metrics,
        )

    # ===================================================================
    # EFFECT CLASSIFICATION CONSTANTS
    # ===================================================================
    MAX_OPERATION_ATTEMPTS = 2

    @staticmethod
    def _classify_effect(eval_res: EvaluationResult, operation: str) -> str:
        """
        Classify the effect of an operation based on evaluator metrics.
        Returns one of: IMPROVING, NEUTRAL, HARMFUL, SEVERELY_HARMFUL.

        Uses the existing per-operation evaluator thresholds rather than
        inventing new arbitrary metrics.
        """
        changes = eval_res.improvement_metrics
        before = eval_res.before_metrics
        after = eval_res.after_metrics

        lap_pct = changes.get("laplacian_pct", 0.0)
        noise_pct = changes.get("noise_sigma_pct", 0.0)
        noise_before = before.get("noise_sigma", 0.0)
        noise_after = after.get("noise_sigma", 0.0)
        lap_before = before.get("laplacian", 0.0)
        lap_after = after.get("laplacian", 0.0)

        if operation == "deblur":
            # Sharpness jumped hugely AND noise also jumped hugely → high-frequency amplification
            if noise_pct > 80.0 and (noise_after - noise_before) > 3.0:
                return "SEVERELY_HARMFUL"
            if lap_pct < -20.0:
                return "SEVERELY_HARMFUL"
            if noise_pct > 50.0 and (noise_after - noise_before) > 1.5:
                return "HARMFUL"
            if lap_pct < -5.0:
                return "HARMFUL"
            if eval_res.decision == "KEEP":
                return "IMPROVING"
            return "NEUTRAL"

        elif operation == "denoise":
            # If noise actually INCREASED substantially
            if noise_pct > 50.0 and (noise_after - noise_before) > 3.0:
                return "SEVERELY_HARMFUL"
            if noise_pct > 20.0 and (noise_after - noise_before) > 1.0:
                return "HARMFUL"
            # Destroyed structural detail
            if lap_pct < -60.0:
                return "SEVERELY_HARMFUL"
            if lap_pct < -30.0:
                return "HARMFUL"
            if eval_res.decision == "KEEP":
                return "IMPROVING"
            return "NEUTRAL"

        elif operation == "low_light":
            lum_pct = changes.get("mean_lum_pct", 0.0)
            if lum_pct < -10.0:
                return "HARMFUL"
            if noise_pct > 80.0 and (noise_after - noise_before) > 3.0:
                return "SEVERELY_HARMFUL"
            if eval_res.decision == "KEEP":
                return "IMPROVING"
            return "NEUTRAL"

        elif operation == "super_resolution":
            if eval_res.decision == "KEEP":
                return "IMPROVING"
            return "NEUTRAL"

        # Generic fallback
        if eval_res.decision == "KEEP":
            return "IMPROVING"
        return "NEUTRAL"

    def _is_operation_eligible(
        self, op: str, op_tracker: Dict[str, Dict[str, Any]], pass_num: int
    ) -> tuple:
        """
        Determine whether an operation is eligible to run.
        Returns (eligible: bool, reason: str).
        """
        info = op_tracker.get(op, {"attempt_count": 0, "last_effect": "NOT_RUN"})
        attempt_count = info.get("attempt_count", 0)
        last_effect = info.get("last_effect", "NOT_RUN")

        if attempt_count >= self.MAX_OPERATION_ATTEMPTS:
            return False, f"maximum attempts reached ({attempt_count}/{self.MAX_OPERATION_ATTEMPTS})"

        if last_effect == "SEVERELY_HARMFUL":
            return False, f"blocked: previous attempt was SEVERELY_HARMFUL"

        if last_effect == "NOT_RUN":
            return True, "not yet attempted"

        if last_effect == "IMPROVING":
            return False, f"previously completed with effect IMPROVING"

        if last_effect == "NEUTRAL":
            return False, f"previously completed with effect NEUTRAL (no meaningful change)"

        if last_effect == "HARMFUL":
            return False, f"previously HARMFUL, retry blocked"

        if last_effect == "ERROR":
            if attempt_count < self.MAX_OPERATION_ATTEMPTS:
                return True, f"retry after ERROR (attempt {attempt_count + 1})"
            return False, f"maximum attempts reached after ERROR"

        # RUN state — eligible if under attempt limit
        if attempt_count < self.MAX_OPERATION_ATTEMPTS:
            return True, f"eligible (attempt {attempt_count + 1})"
        return False, f"maximum attempts reached"

    def run_three_pass(
        self,
        image: Image.Image,
        analyzer: Any,
        planner: Any,
        custom_operations: Optional[List[str]] = None,
        upscale_4k: bool = False,
        original_image: Optional[Image.Image] = None,
        image_id: str = "",
    ) -> EngineResult:
        """
        3-PASS restoration architecture:
        - The degradation analyzer decides WHAT restoration models should run in each pass.
        - The evaluator decides WHETHER the output of an individual model should be kept or discarded.
        - Previous-pass evaluator results NEVER permanently prevent a model from running.
        - Candidate management: before -> run_model -> evaluate -> KEEP (candidate=result) / DISCARD (candidate=before).
        - Discarded model outputs NEVER contaminate subsequent inputs.
        - Never revert all the way to original baseline merely because a model output fails evaluation.
        """
        baseline_image = (original_image or image).copy()
        start_total_time = time.time()

        # candidate starts as baseline/original
        current_candidate = baseline_image.copy()

        all_step_records = []
        global_step_counter = 1
        operation_history = []
        any_operation_accepted = False
        pass1_candidate_created = False

        per_pass_info = {}

        logger.info("=" * 60)
        logger.info("Starting 3-pass restoration pipeline...")
        logger.info("=" * 60)
        logger.info(f"[STATE] Baseline image size: {baseline_image.width}x{baseline_image.height}")
        logger.info(f"[STATE] Initial Candidate = Baseline image")

        # Track execution attempts and effect states per operation
        pass1_analysis = None
        seen_ops = set()
        op_tracker: Dict[str, Dict[str, Any]] = {}

        for pass_num in (1, 2, 3):
            logger.info("")
            logger.info("=" * 60)
            logger.info(f"[PASS {pass_num}] Starting Pass {pass_num}/3 on current candidate ({current_candidate.width}x{current_candidate.height})...")
            logger.info("=" * 60)

            # 1. Run degradation analyzer on current candidate
            pass_analysis = analyzer.analyze(current_candidate, image_id=f"{image_id}_pass{pass_num}" if pass_num > 1 else image_id)
            if pass_num == 1:
                pass1_analysis = pass_analysis

            pass_degradations = [
                deg.name for deg in pass_analysis.degradations
                if getattr(deg, "detected", True)
            ]
            logger.info(f"[PASS {pass_num}]")
            logger.info(f"[ANALYSIS]")
            logger.info(f"Detected degradations: {pass_degradations}")

            # 2. Plan operations for current pass
            planned_ops = planner.plan(
                analysis=pass_analysis,
                custom_operations=custom_operations if pass_num == 1 else None,
            )
            if pass_num == 1 and upscale_4k and "super_resolution" not in planned_ops:
                logger.info("4K AI Upscaling toggle ON: appending 'super_resolution' to Pass 1 pipeline.")
                planned_ops.append("super_resolution")

            logger.info(f"[PLANNER - PASS {pass_num}] Planned operations: {planned_ops}")

            pass_executed_ops = []

            # 3. Execute planned operations if safe and under maximum execution limit (2/2)
            for op in planned_ops:
                info = op_tracker.setdefault(op, {"attempt_count": 0, "last_effect": "NOT_RUN"})

                # Check if blocked due to previous HARMFUL or SEVERELY_HARMFUL attempt (applies in Pass 2 and Pass 3)
                '''
                if pass_num > 1 and (info["blocked"] or info["last_effect"] in ["HARMFUL", "SEVERELY_HARMFUL"]):
                    logger.info("[SKIP]")
                    logger.info(f"{op}")
                    logger.info("Reason:")
                    logger.info(f"Operation blocked due to previous {info['last_effect']} effect")
                    operation_history.append({
                        "pass": pass_num,
                        "operation": op,
                        "effect": "SKIPPED",
                        "reason": f"Blocked due to previous {info['last_effect']} effect",
                    })
                    continue'''
                if pass_num > 1 and info["last_effect"] in ["HARMFUL", "SEVERELY_HARMFUL"]:
                    logger.info("[SKIP]")
                    logger.info(f"{op}")
                    logger.info("Reason:")
                    logger.info(f"Not retrying after previous {info['last_effect']} effect")

                    operation_history.append({"pass": pass_num,
                        "operation": op,
                        "effect": "SKIPPED",
                        "reason": f"Not retried after previous {info['last_effect']} effect",
                    })

                    continue

                if info["attempt_count"] >= self.MAX_OPERATION_ATTEMPTS:
                    logger.info("[SKIP]")
                    logger.info(f"{op}")
                    logger.info("Reason:")
                    logger.info("Maximum execution limit reached (2/2)")
                    operation_history.append({
                        "pass": pass_num,
                        "operation": op,
                        "effect": "SKIPPED",
                        "reason": "Maximum execution limit reached (2/2)",
                    })
                    continue

                op_run_count = info["attempt_count"] + 1
                seen_ops.add(op)

                logger.info("[OPERATION]")
                logger.info(f"{op} | Execution {op_run_count}/2")

                t0 = time.time()
                input_size = (current_candidate.width, current_candidate.height)
                image_before = current_candidate.copy()

                try:
                    model = get_model(op)
                    logger.info(f"Step {global_step_counter} (Pass {pass_num}): Running '{op}' using {model.name} on size {input_size[0]}x{input_size[1]}")

                    # Check if model has real weights (Restormer checkpoint diagnosis)
                    if hasattr(model, '_model_instance') and hasattr(model._model_instance, 'real_checkpoint_loaded'):
                        if not model._model_instance.real_checkpoint_loaded:
                            logger.warning(f"[{op.upper()}] Real checkpoint loaded: NO — using fallback")
                            logger.warning(f"[{op.upper()}] Restormer fallback active: YES")

                    model_output = model.predict(image_before)
                except Exception as e:
                    logger.error(f"Error executing step '{op}' in Pass {pass_num}: {e}. Skipping step.")
                    operation_history.append({
                        "pass": pass_num,
                        "operation": op,
                        "effect": "DISCARD",
                        "reason": f"Execution error: {e}",
                    })
                    continue

                dt = round(time.time() - t0, 3)
                output_size = (model_output.width, model_output.height)

                # 4. Evaluate before/after for THIS operation in THIS pass
                eval_res = self.evaluator.evaluate(
                    op, image_before, model_output, pass_number=pass_num, op_run_count=op_run_count
                )

                logger.info(f"[METRICS BEFORE]")
                logger.info(f"    Laplacian = {eval_res.before_metrics.get('laplacian', 0.0):.2f}")
                logger.info(f"    NoiseSigma = {eval_res.before_metrics.get('noise_sigma', 0.0):.2f}")
                logger.info(f"    MeanLuminance = {eval_res.before_metrics.get('mean_luminance', 0.0):.4f}")
                logger.info(f"[METRICS AFTER]")
                logger.info(f"    Laplacian = {eval_res.after_metrics.get('laplacian', 0.0):.2f}")
                logger.info(f"    NoiseSigma = {eval_res.after_metrics.get('noise_sigma', 0.0):.2f}")
                logger.info(f"    MeanLuminance = {eval_res.after_metrics.get('mean_luminance', 0.0):.4f}")
                logger.info(f"[EVALUATION]")
                logger.info(f"{eval_res.decision}")
                logger.info(f"[REASON]")
                logger.info(f"{eval_res.reason}")

                # 5. Classify Operation Effect and Update Candidate Image State
                effect = self._classify_effect(eval_res, op)
                if eval_res.decision == "KEEP":
                    effect = "IMPROVING"

                info["attempt_count"] += 1
                info["last_effect"] = effect

                logger.info(f"[OPERATION EFFECT] Effect: {effect}")

                if pass_num == 1:
                    current_candidate = model_output
                    pass_executed_ops.append(op)
                    pass1_candidate_created = True
                    logger.info(f"[CANDIDATE - PASS 1] Candidate UPDATED with '{op}' output regardless of evaluator decision: {eval_res.decision}")
                else:
                    if effect == "IMPROVING":
                        current_candidate = model_output
                        any_operation_accepted = True
                        pass_executed_ops.append(op)
                        logger.info(f"[CANDIDATE] Candidate: UPDATED with accepted '{op}' result ({current_candidate.width}x{current_candidate.height})")
                    else:
                        logger.info(
                            f"[CANDIDATE] Candidate: UNCHANGED "
                            f"(keeping previous accepted candidate). "
                            f"Operation output rejected as {effect}."
                        )

                # Compute pixel diff metrics for step record
                if image_before.size != model_output.size:
                    step_inp_eval = np.array(image_before.resize(model_output.size, Image.Resampling.LANCZOS).convert("RGB"), dtype=np.float32)
                else:
                    step_inp_eval = np.array(image_before.convert("RGB"), dtype=np.float32)
                step_out_eval = np.array(model_output.convert("RGB"), dtype=np.float32)
                step_diff = np.abs(step_out_eval - step_inp_eval)
                step_mad = float(np.mean(step_diff))
                step_changed_pct = float(np.mean(step_diff > 1.0) * 100.0)

                is_step_accepted = (pass_num == 1) or (effect == "IMPROVING")

                step_info = {
                    "step_number": global_step_counter,
                    "pass_number": pass_num,
                    "operation": op,
                    "model_name": model.name,
                    "effect": eval_res.decision,
                    "evaluator_decision": eval_res.decision,
                    "evaluator_reason": eval_res.reason,
                    "execution_time_seconds": dt,
                    "input_size": f"{input_size[0]}x{input_size[1]}",
                    "output_size": f"{output_size[0]}x{output_size[1]}",
                    "laplacian_before": eval_res.before_metrics.get("laplacian"),
                    "laplacian_after": eval_res.after_metrics.get("laplacian"),
                    "laplacian_change_percent": eval_res.improvement_metrics.get("laplacian_pct", 0.0),
                    "noise_sigma_before": eval_res.before_metrics.get("noise_sigma"),
                    "noise_sigma_after": eval_res.after_metrics.get("noise_sigma"),
                    "noise_sigma_change_percent": eval_res.improvement_metrics.get("noise_sigma_pct", 0.0),
                    "mean_absolute_difference": round(step_mad, 3),
                    "changed_pixels_percent": round(step_changed_pct, 2),
                    "before_metrics": eval_res.before_metrics,
                    "after_metrics": eval_res.after_metrics,
                    "improvement_metrics": eval_res.improvement_metrics,
                    "is_accepted": is_step_accepted,
                }
                if is_step_accepted:
                    step_info["output_image"] = model_output.copy()
                all_step_records.append(step_info)
                global_step_counter += 1

                operation_history.append({
                    "pass": pass_num,
                    "operation": op,
                    "effect": eval_res.decision,
                    "noise_change": eval_res.improvement_metrics.get("noise_sigma_pct", 0.0) / 100.0,
                    "reason": eval_res.reason,
                })

            per_pass_info[f"pass_{pass_num}"] = {
                "degradations": pass_degradations,
                "planned": planned_ops,
                "executed": pass_executed_ops,
            }

        # ===================================================================
        # FINAL OUTPUT RESOLUTION & RECONSTRUCTION
        # ===================================================================
        logger.info("")
        logger.info("=" * 60)
        logger.info("[FINAL RESULT SUMMARY]")
        logger.info("=" * 60)

        if any_operation_accepted or pass1_candidate_created:
            final_output_image = current_candidate
            final_decision = "KEEP"
            final_reason = "Accepted candidate from 3-pass restoration sequence."
            logger.info(f"[FINAL DECISION] KEEP accepted candidate ({final_output_image.width}x{final_output_image.height})")
        else:
            final_output_image = baseline_image
            final_decision = "DISCARD"
            final_reason = "No restoration operation was accepted across any pass; returning original baseline."
            logger.warning("[FINAL DECISION] DISCARD. Returning original baseline image because no operation was accepted.")

        # Reconstruct baseline dimensions if super_resolution was not applied
        if "super_resolution" not in [s["operation"] for s in all_step_records if s.get("evaluator_decision") == "KEEP"]:
            orig_w, orig_h = baseline_image.width, baseline_image.height
            if (final_output_image.width, final_output_image.height) != (orig_w, orig_h):
                logger.info(f"Reconstructing output resolution to original baseline dimensions ({orig_w}x{orig_h})")
                final_output_image = final_output_image.resize((orig_w, orig_h), Image.Resampling.LANCZOS)

        total_dt = round(time.time() - start_total_time, 3)
        metrics = _compute_basic_metrics(baseline_image, final_output_image)

        metrics.update({
            "pass_1_degradations": per_pass_info.get("pass_1", {}).get("degradations", []),
            "pass_1_planned": per_pass_info.get("pass_1", {}).get("planned", []),
            "pass_1_executed": per_pass_info.get("pass_1", {}).get("executed", []),
            "pass_2_degradations": per_pass_info.get("pass_2", {}).get("degradations", []),
            "pass_2_planned": per_pass_info.get("pass_2", {}).get("planned", []),
            "pass_2_executed": per_pass_info.get("pass_2", {}).get("executed", []),
            "pass_3_degradations": per_pass_info.get("pass_3", {}).get("degradations", []),
            "pass_3_planned": per_pass_info.get("pass_3", {}).get("planned", []),
            "pass_3_executed": per_pass_info.get("pass_3", {}).get("executed", []),
            "operation_history": operation_history,
            "final_decision": final_decision,
            "final_reason": final_reason,
            "passes_executed": 3,
        })

        return EngineResult(
            final_image=final_output_image,
            pipeline_steps=all_step_records,
            total_time_seconds=total_dt,
            metrics=metrics,
            analysis_report=pass1_analysis,
        )

    def run_two_pass(
        self,
        image: Image.Image,
        analyzer: Any,
        planner: Any,
        custom_operations: Optional[List[str]] = None,
        upscale_4k: bool = False,
        original_image: Optional[Image.Image] = None,
        image_id: str = "",
    ) -> EngineResult:
        """Backward compatibility wrapper delegating to run_three_pass."""
        return self.run_three_pass(
            image=image,
            analyzer=analyzer,
            planner=planner,
            custom_operations=custom_operations,
            upscale_4k=upscale_4k,
            original_image=original_image,
            image_id=image_id,
        )


