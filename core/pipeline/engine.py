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
    """

    def __init__(self, device: str = "cpu"):
        self.device = device

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

            # Step input stats
            inp_gray = cv2.cvtColor(np.array(working_img.convert("RGB")), cv2.COLOR_RGB2GRAY)
            inp_lap = float(cv2.Laplacian(inp_gray, cv2.CV_64F).var())

            try:
                model = get_model(op)
                logger.info(f"Step {idx}/{len(operations)}: Running '{op}' using {model.name} on size {input_size[0]}x{input_size[1]}")
                restored_step = model.predict(working_img)
            except Exception as e:
                logger.error(f"Error executing step '{op}': {e}. Skipping step.")
                continue

            dt = round(time.time() - t0, 3)
            output_size = (restored_step.width, restored_step.height)

            # Step output stats
            out_gray = cv2.cvtColor(np.array(restored_step.convert("RGB")), cv2.COLOR_RGB2GRAY)
            out_lap = float(cv2.Laplacian(out_gray, cv2.CV_64F).var())
            step_lap_change = ((out_lap - inp_lap) / max(inp_lap, 1e-5)) * 100.0

            # Pixel diff stats between step input and output (resizing for comparison if step changed dimensions)
            if working_img.size != restored_step.size:
                step_inp_eval = np.array(working_img.resize(restored_step.size, Image.Resampling.LANCZOS).convert("RGB"), dtype=np.float32)
            else:
                step_inp_eval = np.array(working_img.convert("RGB"), dtype=np.float32)
            step_out_eval = np.array(restored_step.convert("RGB"), dtype=np.float32)
            step_diff = np.abs(step_out_eval - step_inp_eval)
            step_mad = float(np.mean(step_diff))
            step_changed_pct = float(np.mean(step_diff > 1.0) * 100.0)

            step_info = {
                "step_number": idx,
                "operation": op,
                "model_name": model.name,
                "execution_time_seconds": dt,
                "input_size": f"{input_size[0]}x{input_size[1]}",
                "output_size": f"{output_size[0]}x{output_size[1]}",
                "laplacian_before": round(inp_lap, 2),
                "laplacian_after": round(out_lap, 2),
                "laplacian_change_percent": round(step_lap_change, 2),
                "mean_absolute_difference": round(step_mad, 3),
                "changed_pixels_percent": round(step_changed_pct, 2),
            }
            step_records.append(step_info)
            logger.info(
                f"[{op.upper()}] Done in {dt}s | Input: {input_size[0]}x{input_size[1]} -> Output: {output_size[0]}x{output_size[1]} | "
                f"Laplacian: {inp_lap:.2f} -> {out_lap:.2f} ({step_lap_change:+.2f}%) | MAD: {step_mad:.3f} | Changed: {step_changed_pct:.2f}%"
            )

            working_img = restored_step

        # Final Resolution Restoration:
        # If super_resolution was NOT executed and working_img size differs from original_ref, reconstruct back to original resolution!
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
