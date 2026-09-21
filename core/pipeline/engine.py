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
        """
        Executes a TWO-PASS restoration architecture:
        - PASS 1: Analyzes image, plans pipeline, and executes planned operations with quality evaluation.
        - PASS 2: Re-analyzes Pass-1 output image. Filters out already applied operations.
                  Executes any remaining/newly exposed operations with quality evaluation. Max 2 passes.
        """
        original_ref = original_image or image
        start_total_time = time.time()
        working_img = image.copy()

        applied_operations = set()
        all_step_records = []
        global_step_counter = 1

        logger.info("Starting restoration request...")

        # ===================================================================
        # PASS 1 / 2
        # ===================================================================
        logger.info("PASS 1/2")

        pass1_analysis = analyzer.analyze(working_img, image_id=image_id)
        initial_degradations = [
            deg.name for deg in pass1_analysis.degradations
            if getattr(deg, "detected", True)
        ]
        logger.info(f"Initial degradations:\n{initial_degradations}")

        pass1_planned = planner.plan(
            analysis=pass1_analysis,
            custom_operations=custom_operations,
        )

        if upscale_4k and "super_resolution" not in pass1_planned:
            logger.info("4K AI Upscaling toggle ON: appending 'super_resolution' to operations pipeline.")
            pass1_planned.append("super_resolution")

        logger.info(f"Planned PASS 1 pipeline:\n{pass1_planned}")

        pass1_executed = []
        for op in pass1_planned:
            logger.info(f"Running PASS 1 operation: {op}")
            t0 = time.time()
            input_size = (working_img.width, working_img.height)
            image_before = working_img.copy()

            try:
                model = get_model(op)
                logger.info(f"Step {global_step_counter} (Pass 1): Running '{op}' using {model.name} on size {input_size[0]}x{input_size[1]}")
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
                "step_number": global_step_counter,
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
            all_step_records.append(step_info)
            global_step_counter += 1

            if eval_res.decision == "KEEP":
                applied_operations.add(op)
                pass1_executed.append(op)
                working_img = restored_step
            else:
                logger.warning(f"[{op.upper()}] Output DISCARDED. Retaining image_before for subsequent steps.")

        logger.info("PASS 1 completed.")

        # ===================================================================
        # PASS 2 / 2
        # ===================================================================
        logger.info("Running post-restoration degradation analysis...")
        logger.info("PASS 2/2")

        pass2_analysis = analyzer.analyze(working_img, image_id=f"{image_id}_pass2")
        remaining_degradations = [
            deg.name for deg in pass2_analysis.degradations
            if getattr(deg, "detected", True)
        ]
        logger.info(f"Remaining/new degradations:\n{remaining_degradations}")

        candidate_pass2_pipeline = planner.plan(analysis=pass2_analysis)
        logger.info(f"Candidate PASS 2 pipeline:\n{candidate_pass2_pipeline}")

        pass2_planned = []
        pass2_skipped = []
        for op in candidate_pass2_pipeline:
            if op in applied_operations:
                logger.info(f"Skipping '{op}': already applied in PASS 1")
                pass2_skipped.append(op)
            else:
                pass2_planned.append(op)

        pass2_executed = []
        if not pass2_planned:
            logger.info("No new restoration operations required.\nRestoration completed.")
        else:
            logger.info(f"Planned PASS 2 pipeline:\n{pass2_planned}")
            for op in pass2_planned:
                logger.info(f"Running PASS 2 operation: {op}")
                t0 = time.time()
                input_size = (working_img.width, working_img.height)
                image_before = working_img.copy()

                try:
                    model = get_model(op)
                    logger.info(f"Step {global_step_counter} (Pass 2): Running '{op}' using {model.name} on size {input_size[0]}x{input_size[1]}")
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
                    "step_number": global_step_counter,
                    "pass_number": 2,
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
                all_step_records.append(step_info)
                global_step_counter += 1

                if eval_res.decision == "KEEP":
                    applied_operations.add(op)
                    pass2_executed.append(op)
                    working_img = restored_step
                else:
                    logger.warning(f"[{op.upper()}] Output DISCARDED. Retaining image_before for subsequent steps.")

            logger.info("PASS 2 completed.\nRestoration completed.")

        # Final Resolution Reconstruction (if super_resolution was NOT executed across either pass)
        if "super_resolution" not in applied_operations:
            orig_w, orig_h = original_ref.width, original_ref.height
            if (working_img.width, working_img.height) != (orig_w, orig_h):
                logger.info(f"Reconstructing final output size from model resolution ({working_img.width}x{working_img.height}) back to original resolution ({orig_w}x{orig_h})")
                working_img = working_img.resize((orig_w, orig_h), Image.Resampling.LANCZOS)

        total_dt = round(time.time() - start_total_time, 3)
        metrics = _compute_basic_metrics(original_ref, working_img)

        metrics.update({
            "pass_1_degradations": initial_degradations,
            "pass_1_planned": pass1_planned,
            "pass_1_executed": pass1_executed,
            "pass_2_degradations": remaining_degradations,
            "pass_2_candidate": candidate_pass2_pipeline,
            "pass_2_planned": pass2_planned,
            "pass_2_skipped": pass2_skipped,
            "pass_2_executed": pass2_executed,
            "applied_operations": list(applied_operations),
            "passes_executed": 2 if pass2_planned else 1,
        })

        return EngineResult(
            final_image=working_img,
            pipeline_steps=all_step_records,
            total_time_seconds=total_dt,
            metrics=metrics,
            analysis_report=pass1_analysis,
        )


