"""
Degradation Analyzer Aggregator.

Executes all 5 degradation detectors (blur, noise, resolution, low-light, jpeg)
and returns a structured degradation report.
"""
from typing import Any, Dict, List

from PIL import Image
from loguru import logger

from app.config import load_config
from app.schemas.image import DegradationItem, AnalysisResponse
from core.analyzer.blur_detector import detect_blur
from core.analyzer.noise_detector import detect_noise
from core.analyzer.resolution_detector import detect_resolution
from core.analyzer.lowlight_detector import detect_low_light
from core.analyzer.jpeg_detector import detect_jpeg_artifacts


def _get_severity_label(score: float) -> str:
    if score >= 0.65:
        return "HIGH"
    if score >= 0.35:
        return "MEDIUM"
    return "LOW"


class DegradationAnalyzer:
    """
    Main analyzer class that orchestrates individual detection modules.
    """

    def __init__(self, config: Dict[str, Any] | None = None):
        """
        Args:
            config: Optional config dict. If None, loads from configs/default.yaml.
        """
        full_cfg = config or load_config()
        self.cfg = full_cfg.get("analyzer", {})

        self.blur_thresh = float(self.cfg.get("blur_laplacian_threshold", 250.0))
        self.noise_thresh = float(self.cfg.get("noise_sigma_threshold", 2.0))
        self.min_dim = int(self.cfg.get("min_dimension", 512))
        self.low_light_thresh = float(self.cfg.get("low_light_mean_threshold", 0.35))
        self.jpeg_thresh = float(self.cfg.get("jpeg_artifact_threshold", 0.05))

    def analyze(self, image: Image.Image, image_id: str = "") -> AnalysisResponse:
        """
        Analyze a PIL Image and produce a full degradation report.

        Args:
            image: PIL Image (RGB).
            image_id: Unique image identifier.

        Returns:
            AnalysisResponse schema.
        """
        degradations: List[DegradationItem] = []
        raw_metrics: Dict[str, Any] = {}

        # 1. Blur Detection
        is_blurry, blur_sev, lap_var = detect_blur(image, threshold=self.blur_thresh)
        raw_metrics["laplacian_variance"] = round(lap_var, 2)
        if is_blurry:
            degradations.append(
                DegradationItem(
                    name="blur",
                    score=round(blur_sev, 2),
                    severity=_get_severity_label(blur_sev),
                    details={"laplacian_variance": round(lap_var, 1), "threshold": self.blur_thresh},
                )
            )

        # 2. Noise Detection
        is_noisy, noise_sev, sigma = detect_noise(image, threshold=self.noise_thresh)
        raw_metrics["estimated_noise_sigma"] = round(sigma, 2)
        if is_noisy:
            degradations.append(
                DegradationItem(
                    name="noise",
                    score=round(noise_sev, 2),
                    severity=_get_severity_label(noise_sev),
                    details={"noise_sigma": round(sigma, 1), "threshold": self.noise_thresh},
                )
            )

        # 3. Resolution Check
        is_low_res, res_sev, (w, h) = detect_resolution(image, min_dimension=self.min_dim)
        raw_metrics["dimensions"] = [w, h]
        if is_low_res:
            degradations.append(
                DegradationItem(
                    name="low_resolution",
                    score=round(res_sev, 2),
                    severity=_get_severity_label(res_sev),
                    details={"dimensions": [w, h], "min_dimension": self.min_dim},
                )
            )

        # 4. Low-Light Detection
        is_dark, light_sev, mean_lum = detect_low_light(image, threshold=self.low_light_thresh)
        raw_metrics["mean_luminance"] = round(mean_lum, 3)
        if is_dark:
            degradations.append(
                DegradationItem(
                    name="low_light",
                    score=round(light_sev, 2),
                    severity=_get_severity_label(light_sev),
                    details={"mean_luminance": round(mean_lum, 2), "threshold": self.low_light_thresh},
                )
            )

        # 5. JPEG Artifact Detection
        has_jpeg, jpeg_sev, block_score = detect_jpeg_artifacts(image, threshold=self.jpeg_thresh)
        raw_metrics["jpeg_blocking_score"] = round(block_score, 3)
        if has_jpeg:
            degradations.append(
                DegradationItem(
                    name="jpeg_artifacts",
                    score=round(jpeg_sev, 2),
                    severity=_get_severity_label(jpeg_sev),
                    details={"blocking_score": round(block_score, 2), "threshold": self.jpeg_thresh},
                )
            )

        logger.info(f"Analysis complete for image {image_id}: detected {len(degradations)} degradation(s)")
        return AnalysisResponse(
            image_id=image_id,
            degradations=degradations,
            raw_metrics=raw_metrics,
        )
