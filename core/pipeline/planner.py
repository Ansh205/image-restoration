"""
Pipeline Planner — Maps detected degradations to an ordered execution pipeline.
"""
from typing import List, Optional
from loguru import logger

from app.schemas.image import AnalysisResponse, DegradationItem


# Canonical restoration sequence to prevent compounding artifacts
CANONICAL_RESTORATION_ORDER = [
    "low_light",        # 1. Zero-DCE++: Brighten image first so downstream models see full detail
    "denoise",          # 2. DRUNet: Clean noise before deblurring to prevent noise amplification
    "deblur",           # 3. Restormer: Deblur clean image
    "jpeg",             # 4. SwinIR: Remove 8x8 DCT grid blocking artifacts
    "super_resolution", # 5. Real-ESRGAN: Upscale image as the final step
]

DEGRADATION_TO_OPERATION = {
    "low_light": "low_light",
    "noise": "denoise",
    "blur": "deblur",
    "jpeg_artifacts": "jpeg",
    "low_resolution": "super_resolution",
}


class PipelinePlanner:
    """
    Analyzes detected degradations and produces an ordered sequence of model operations.
    """

    def __init__(self, canonical_order: Optional[List[str]] = None):
        self.canonical_order = canonical_order or CANONICAL_RESTORATION_ORDER

    def plan(
        self,
        analysis: AnalysisResponse,
        custom_operations: Optional[List[str]] = None,
    ) -> List[str]:
        """
        Generate an ordered list of operation names.

        Args:
            analysis: Structured degradation report from DegradationAnalyzer.
            custom_operations: Optional list of specific operations requested by the user.

        Returns:
            List of operation names ordered by the canonical restoration pipeline rules.
        """
        if custom_operations:
            logger.info(f"Using user-specified custom operations: {custom_operations}")
            # Order custom operations according to canonical rules if possible
            ordered_custom = [op for op in self.canonical_order if op in custom_operations]
            # Append any extra operations not in canonical order
            for op in custom_operations:
                if op not in ordered_custom:
                    ordered_custom.append(op)
            return ordered_custom

        # Extract operation names from detected degradations
        required_ops = set()
        for deg in analysis.degradations:
            op = DEGRADATION_TO_OPERATION.get(deg.name)
            if op:
                required_ops.add(op)

        if not required_ops:
            logger.info("No degradations detected. Pipeline is empty.")
            return []

        # Order detected operations using canonical order
        planned_pipeline = [op for op in self.canonical_order if op in required_ops]

        logger.info(f"Planned restoration pipeline: {planned_pipeline}")
        return planned_pipeline
