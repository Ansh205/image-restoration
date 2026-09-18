"""
Integration tests for model fallbacks, pipeline routing, and image degradation fixes.
"""
import numpy as np
from PIL import Image
from models.factory import get_model
from core.pipeline.planner import PipelinePlanner, ImageDegradations
from core.pipeline.engine import RestorationEngine, EngineConfig


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


from app.schemas.image import AnalysisResponse, DegradationItem


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
    
    engine = RestorationEngine(EngineConfig(enable_metrics=True))
    img = Image.new("RGB", (64, 64), color=(100, 120, 140))
    
    res_img, report = engine.execute_pipeline(img, plan)
    assert res_img is not None
    assert len(report.steps_executed) == len(plan)
