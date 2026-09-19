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

        self.blur_thresh = float(self.cfg.get("blur", {}).get("laplacian_threshold", 250.0))
        self.noise_thresh = float(self.cfg.get("noise", {}).get("sigma_threshold", 15.0))
        self.min_dim = int(self.cfg.get("resolution", {}).get("min_dimension", 512))
        
        self.exposure_cfg = self.cfg.get("exposure", {})
        self.jpeg_thresh = float(self.cfg.get("jpeg_artifacts", {}).get("blocking_threshold", 0.40))

    def analyze(self, image: Image.Image, image_id: str = "") -> AnalysisResponse:
        """
        Analyze a PIL Image and produce a full degradation report.
        """
        degradations: List[DegradationItem] = []
        raw_metrics: Dict[str, Any] = {}

        # 1. Blur Detection
        is_blurry, blur_sev, blur_conf, blur_details = detect_blur(image, threshold=self.blur_thresh)
        raw_metrics.update(blur_details)
        if is_blurry:
            degradations.append(
                DegradationItem(
                    name="blur",
                    score=round(blur_sev, 2),
                    severity=_get_severity_label(blur_sev),
                    confidence=blur_conf,
                    details=blur_details,
                )
            )

        # 2. Noise Detection
        is_noisy, noise_sev, noise_conf, noise_details = detect_noise(image, threshold=self.noise_thresh)
        raw_metrics.update(noise_details)
        if is_noisy:
            degradations.append(
                DegradationItem(
                    name="noise",
                    score=round(noise_sev, 2),
                    severity=_get_severity_label(noise_sev),
                    confidence=noise_conf,
                    details=noise_details,
                )
            )

        # 3. Resolution Check
        is_low_res, res_sev, res_conf, res_details = detect_resolution(image, min_dimension=self.min_dim)
        raw_metrics.update(res_details)
        if is_low_res:
            degradations.append(
                DegradationItem(
                    name="low_resolution",
                    score=round(res_sev, 2),
                    severity=_get_severity_label(res_sev),
                    confidence=res_conf,
                    details=res_details,
                )
            )

        # 4. Exposure Detection
        exposure_items, exposure_metrics = detect_low_light(image, self.exposure_cfg)
        raw_metrics.update(exposure_metrics)
        for item in exposure_items:
            degradations.append(item)

        # 5. JPEG Artifact Detection
        has_jpeg, jpeg_sev, jpeg_conf, jpeg_details = detect_jpeg_artifacts(image, threshold=self.jpeg_thresh)
        raw_metrics.update(jpeg_details)
        if has_jpeg:
            degradations.append(
                DegradationItem(
                    name="jpeg_artifacts",
                    score=round(jpeg_sev, 2),
                    severity=_get_severity_label(jpeg_sev),
                    confidence=jpeg_conf,
                    details=jpeg_details,
                )
            )

        logger.info(f"Analysis complete for image {image_id}: detected {len(degradations)} degradation(s)")
        return AnalysisResponse(
            image_id=image_id,
            degradations=degradations,
            raw_metrics=raw_metrics,
        )
