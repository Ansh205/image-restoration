"""
Pipeline Planner — Maps detected degradations to an ordered execution pipeline.
"""
'''
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
        min_severity: str = "LOW",
    ) -> List[str]:
        """
        Generate an ordered list of operation names.

        Args:
            analysis: Structured degradation report from DegradationAnalyzer.
            custom_operations: Optional list of specific operations requested by the user.
            min_severity: Minimum degradation severity to schedule model ("LOW", "MEDIUM", or "HIGH").

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

        severity_rank = {"LOW": 1, "MEDIUM": 2, "HIGH": 3}
        min_rank = severity_rank.get(min_severity.upper(), 1)

        # Extract operation names from detected degradations that meet minimum severity & restoration_required threshold
        required_ops = set()
        for deg in analysis.degradations:
            deg_rank = severity_rank.get(str(deg.severity).upper(), 1)
            op = DEGRADATION_TO_OPERATION.get(deg.name)

            if hasattr(deg, "detected") and not deg.detected:
                logger.info(f"Skipping '{deg.name}': Not detected.")
                continue

            # Special policy for JPEG artifacts: require score >= 0.35 or MEDIUM/HIGH severity
            if deg.name == "jpeg_artifacts" and deg.score < 0.35:
                logger.info(f"Skipping 'jpeg_artifacts' (Score: {deg.score:.2f}, Severity: {deg.severity}): Below restoration_required threshold (0.35).")
                continue

            if deg_rank < min_rank:
                logger.info(f"Skipping '{deg.name}' (Severity: {deg.severity}, Score: {deg.score:.2f}): Below minimum threshold '{min_severity}'.")
                continue

            if op:
                logger.info(f"Scheduling '{op}' for degradation '{deg.name}' (Severity: {deg.severity}, Score: {deg.score:.2f}, restoration_required=True).")
                required_ops.add(op)

        if not required_ops:
            logger.info("No degradations above threshold. Pipeline is empty.")
            return []

        # Order detected operations using canonical order
        planned_pipeline = [op for op in self.canonical_order if op in required_ops]

        logger.info(f"Planned restoration pipeline: {planned_pipeline}")
        return planned_pipeline
'''


"""
Pipeline Planner.

Maps detected degradations to restoration operations.
"""
from typing import List, Optional

from loguru import logger

from app.schemas.image import AnalysisResponse


# ============================================================
# CANONICAL RESTORATION ORDER
# ============================================================
#
# The order in which restoration operations should be applied.
#
# low_light
#     ↓
# denoise
#     ↓
# deblur
#     ↓
# jpeg
#     ↓
# super_resolution
#
# Keep this order centralized so the planner does not depend
# on the order in which detectors return their results.
# ============================================================

CANONICAL_RESTORATION_ORDER = [
    "low_light",
    "denoise",
    "deblur",
    "jpeg",
    "super_resolution",
]


# ============================================================
# DEGRADATION → RESTORATION OPERATION
# ============================================================
#
# These are only degradations for which a restoration operation
# currently exists in the project.
#
# IMPORTANT:
# "overexposure" is intentionally NOT included because there
# is currently no overexposure restoration model.
#
# The analyzer can still detect/report overexposure, but the
# planner will simply log that no restoration operation exists.
# ============================================================

DEGRADATION_TO_OPERATION = {
    "low_light": "low_light",
    "noise": "denoise",
    "blur": "deblur",
    "jpeg_artifacts": "jpeg",
    "low_resolution": "super_resolution",
}


# ============================================================
# PIPELINE PLANNER
# ============================================================

class PipelinePlanner:
    """
    Converts degradation analysis results into a restoration
    pipeline.

    The planner does NOT perform restoration.

    It only decides:
        1. Which restoration operations are required.
        2. Which operations are available.
        3. Which operations satisfy the minimum severity.
        4. What order the operations should run in.
    """

    def __init__(
        self,
        canonical_order: Optional[List[str]] = None,
    ):
        """
        Initialize the pipeline planner.

        Args:
            canonical_order:
                Optional custom restoration order.
                If not provided, the default canonical order
                is used.
        """

        self.canonical_order = (
            canonical_order
            or CANONICAL_RESTORATION_ORDER
        )

    def plan(
        self,
        analysis: AnalysisResponse,
        custom_operations: Optional[List[str]] = None,
        min_severity: str = "LOW",
    ) -> List[str]:
        """
        Create a restoration pipeline from degradation analysis.

        Args:
            analysis:
                Output returned by DegradationAnalyzer.

            custom_operations:
                Optional user-specified restoration operations.
                When provided, automatic degradation-based
                planning is skipped.

            min_severity:
                Minimum severity required to schedule a
                restoration operation.

                Options:
                    LOW
                    MEDIUM
                    HIGH

        Returns:
            Ordered list of restoration operations.

        Example:
            ["low_light", "denoise", "deblur", "super_resolution"]
        """

        # =====================================================
        # 1. USER REQUESTED OPERATIONS
        # =====================================================

        if custom_operations:
            logger.info(
                f"Using custom operations: "
                f"{custom_operations}"
            )

            # First keep operations that are part of the
            # canonical order.
            ordered = [
                operation
                for operation in self.canonical_order
                if operation in custom_operations
            ]

            # Then append custom operations that are not
            # present in the canonical order.
            for operation in custom_operations:
                if operation not in ordered:
                    ordered.append(operation)

            logger.info(
                f"Custom restoration pipeline: {ordered}"
            )

            return ordered

        # =====================================================
        # 2. SEVERITY CONFIGURATION
        # =====================================================

        severity_rank = {
            "LOW": 1,
            "MEDIUM": 2,
            "HIGH": 3,
        }

        min_rank = severity_rank.get(
            min_severity.upper(),
            1,
        )

        required_operations = set()

        # =====================================================
        # 3. PROCESS DETECTED DEGRADATIONS
        # =====================================================

        for degradation in analysis.degradations:

            name = degradation.name

            score = float(
                degradation.score
            )

            severity = str(
                degradation.severity
            ).upper()

            current_rank = severity_rank.get(
                severity,
                1,
            )

            # -------------------------------------------------
            # Find restoration operation
            # -------------------------------------------------

            operation = DEGRADATION_TO_OPERATION.get(
                name
            )

            # -------------------------------------------------
            # No restoration model available
            # -------------------------------------------------

            if operation is None:

                logger.warning(
                    f"No restoration operation mapped "
                    f"for degradation '{name}'. "
                    f"Detected with score={score:.2f}, "
                    f"severity={severity}."
                )

                continue

            # -------------------------------------------------
            # Severity filtering
            # -------------------------------------------------

            if current_rank < min_rank:

                logger.info(
                    f"Skipping '{name}': "
                    f"severity={severity}, "
                    f"minimum={min_severity}"
                )

                continue

            # -------------------------------------------------
            # Schedule restoration
            # -------------------------------------------------

            required_operations.add(
                operation
            )

            logger.info(
                f"Scheduling '{operation}' "
                f"for '{name}' "
                f"(score={score:.2f}, "
                f"severity={severity})"
            )

        # =====================================================
        # 4. NOTHING TO RESTORE
        # =====================================================

        if not required_operations:

            logger.info(
                "No available restoration operations "
                "require execution."
            )

            return []

        # =====================================================
        # 5. APPLY CANONICAL ORDER
        # =====================================================

        planned_pipeline = [
            operation
            for operation in self.canonical_order
            if operation in required_operations
        ]

        # =====================================================
        # 6. LOG FINAL PIPELINE
        # =====================================================

        logger.info(
            f"Planned restoration pipeline: "
            f"{planned_pipeline}"
        )

        return planned_pipeline