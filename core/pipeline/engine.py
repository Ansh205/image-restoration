"""
Restoration Engine — Sequentially executes model wrappers in an ordered restoration pipeline.
"""
import time
from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple
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
    """Compute basic before/after quality indicators."""
    orig_np = np.array(original.convert("RGB"))
    rest_np = np.array(restored.convert("RGB"))

    orig_gray = cv2.cvtColor(orig_np, cv2.COLOR_RGB2GRAY)
    rest_gray = cv2.cvtColor(rest_np, cv2.COLOR_RGB2GRAY)

    orig_sharpness = float(cv2.Laplacian(orig_gray, cv2.CV_64F).var())
    rest_sharpness = float(cv2.Laplacian(rest_gray, cv2.CV_64F).var())

    sharpness_change_pct = round(((rest_sharpness - orig_sharpness) / (orig_sharpness + 1e-5)) * 100.0, 2)

    return {
        "original_sharpness": round(orig_sharpness, 2),
        "restored_sharpness": round(rest_sharpness, 2),
        "sharpness_change_percent": sharpness_change_pct,
        "original_resolution": f"{original.width}x{original.height}",
        "restored_resolution": f"{restored.width}x{restored.height}",
    }


class RestorationEngine:
    """
    Sequentially processes an image through an ordered list of restoration operations.
    """

    def __init__(self, device: str = "cpu"):
        self.device = device

    def run(self, image: Image.Image, operations: List[str]) -> EngineResult:
        """
        Execute pipeline operations sequentially.

        Args:
            image: Input PIL Image (RGB).
            operations: Ordered list of operation names (e.g. ['denoise', 'deblur']).

        Returns:
            EngineResult containing final restored image, step breakdown, total time, and metrics.
        """
        if not operations:
            logger.info("Empty pipeline. Returning original image.")
            return EngineResult(
                final_image=image,
                pipeline_steps=[],
                total_time_seconds=0.0,
                metrics=_compute_basic_metrics(image, image),
            )

        start_total_time = time.time()
        current_img = image.copy()
        step_records = []

        logger.info(f"Starting restoration engine with pipeline: {operations}")

        for idx, op in enumerate(operations, start=1):
            t0 = time.time()
            input_size = (current_img.width, current_img.height)

            try:
                model = get_model(op)
                logger.info(f"Step {idx}/{len(operations)}: Running '{op}' using {model.name}")
                current_img = model.predict(current_img)
            except Exception as e:
                logger.error(f"Error executing step '{op}': {e}. Skipping step.")
                continue

            dt = round(time.time() - t0, 3)
            output_size = (current_img.width, current_img.height)

            step_info = {
                "step_number": idx,
                "operation": op,
                "model_name": model.name,
                "execution_time_seconds": dt,
                "input_size": f"{input_size[0]}x{input_size[1]}",
                "output_size": f"{output_size[0]}x{output_size[1]}",
            }
            step_records.append(step_info)
            logger.info(f"Finished step '{op}' in {dt}s. Output size: {output_size}")

        total_dt = round(time.time() - start_total_time, 3)
        metrics = _compute_basic_metrics(image, current_img)

        logger.info(f"Pipeline completed in {total_dt}s across {len(step_records)} steps.")

        return EngineResult(
            final_image=current_img,
            pipeline_steps=step_records,
            total_time_seconds=total_dt,
            metrics=metrics,
        )
