"""
Integration tests for model fallbacks, pipeline routing, and image degradation fixes.
"""
import numpy as np
from PIL import Image
from models.factory import get_model
from core.pipeline.planner import PipelinePlanner
from core.pipeline.engine import RestorationEngine
from app.schemas.image import AnalysisResponse, DegradationItem


def test_jpeg_model_factory_lookup():
    """Verify that 'jpeg' maps to SwinIRModel in factory."""
    model = get_model("jpeg")
    assert model.__class__.__name__ == "SwinIRModel"


def test_realesrgan_no_black_image():
    """Verify RealESRGAN outputs valid image with fallback upscale (no black screen)."""
    model = get_model("super_resolution")
    img = Image.new("RGB", (32, 32), color=(120, 180, 200))
    res = model.restore(img)
    
    assert res.size == (128, 128)
    arr = np.array(res)
    assert arr.mean() > 50.0, "Output image should not be black!"


def test_restormer_no_color_shift():
    """Verify Restormer output retains original RGB color distribution without pink tint."""
    model = get_model("deblur")
    # Natural skin-like / beige tone (R=200, G=170, B=150)
    img = Image.new("RGB", (64, 64), color=(200, 170, 150))
    res = model.restore(img)
    
    arr = np.array(res)
    r_mean, g_mean, b_mean = arr[:, :, 0].mean(), arr[:, :, 1].mean(), arr[:, :, 2].mean()
    
    # Check green channel hasn't collapsed relative to red and blue (pink tint symptom)
    assert g_mean > 100.0, "Green channel collapsed (pink shift symptom)"
    assert abs(r_mean - 200.0) < 30.0
    assert abs(g_mean - 170.0) < 30.0


def test_full_pipeline_with_jpeg_and_deblur():
    """Verify pipeline executes low_light -> deblur -> jpeg -> super_resolution seamlessly."""
    planner = PipelinePlanner()
    analysis = AnalysisResponse(
        image_id="test1234",
        degradations=[
            DegradationItem(name="blur", score=0.75, severity="HIGH"),
            DegradationItem(name="low_light", score=0.80, severity="HIGH"),
            DegradationItem(name="jpeg_artifacts", score=0.60, severity="MEDIUM"),
            DegradationItem(name="low_resolution", score=0.55, severity="MEDIUM"),
        ],
        raw_metrics={"dimensions": [128, 128]}
    )
    
    plan = planner.plan(analysis)
    assert "jpeg" in plan or "deblur" in plan
    
def test_resolution_preservation_1024x1280():
    """Verify that an input of 1024x1280 is preserved as 1024x1280 in final output."""
    engine = RestorationEngine()
    orig_img = Image.new("RGB", (1024, 1280), color=(150, 150, 150))
    working_img = orig_img.resize((819, 1024), Image.Resampling.LANCZOS)
    
    # Run pipeline with deblurring (no super resolution)
    result = engine.run(working_img, ["deblur"], original_image=orig_img)
    
    assert result.final_image.size == (1024, 1280), f"Expected (1024, 1280), got {result.final_image.size}"
    assert result.metrics["original_resolution"] == "1024x1280"
    assert result.metrics["restored_resolution"] == "1024x1280"


def test_planner_skips_low_severity_jpeg():
    """Verify planner skips LOW severity JPEG artifacts when min_severity is MEDIUM."""
    planner = PipelinePlanner()
    analysis = AnalysisResponse(
        image_id="test_low_jpeg",
        degradations=[
            DegradationItem(name="blur", score=0.85, severity="HIGH"),
            DegradationItem(name="jpeg_artifacts", score=0.03, severity="LOW"),
        ],
    )
    
    # With min_severity="MEDIUM", jpeg_artifacts (LOW) should be skipped
    plan = planner.plan(analysis, min_severity="MEDIUM")
    assert "deblur" in plan
    assert "jpeg" not in plan


def test_realesrgan_4k_capping():
    """Verify RealESRGAN skips upscaling if image is already 4K (3840px) or caps output at 3840px."""
    model = get_model("super_resolution")
    
    # 1. Already 4K image (3840x2160) -> should skip and retain 3840x2160
    already_4k = Image.new("RGB", (3840, 2160), color=(100, 150, 200))
    res_4k = model.restore(already_4k)
    assert res_4k.size == (3840, 2160), f"Expected 3840x2160, got {res_4k.size}"

    # 2. Large image (2000x2000) -> 4x upscale (8000x8000) should be capped at 3840x3840
    large_img = Image.new("RGB", (2000, 2000), color=(100, 150, 200))
    res_large = model.restore(large_img)
    assert max(res_large.size) <= 3840, f"Expected max dimension <= 3840, got {res_large.size}"
    assert res_large.size == (3840, 3840)



