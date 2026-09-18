"""
Unit and integration tests for Phase 5: Pipeline Planner, Restoration Engine, and API endpoint.
"""
import io
import pytest
import numpy as np
from PIL import Image

from app.schemas.image import AnalysisResponse, DegradationItem
from core.pipeline.planner import PipelinePlanner, CANONICAL_RESTORATION_ORDER
from core.pipeline.engine import RestorationEngine, _compute_basic_metrics


# ---------------------------------------------------------------------------
# Test Helpers
# ---------------------------------------------------------------------------
def _make_sample_image(width: int = 256, height: int = 256) -> Image.Image:
    """Create a sample synthetic RGB image for testing."""
    arr = np.random.randint(50, 200, (height, width, 3), dtype=np.uint8)
    return Image.fromarray(arr)


# ---------------------------------------------------------------------------
# 1. Pipeline Planner Unit Tests
# ---------------------------------------------------------------------------
class TestPipelinePlanner:
    def test_planner_canonical_ordering(self):
        planner = PipelinePlanner()

        # Input degradations out of canonical order
        analysis = AnalysisResponse(
            degradations=[
                DegradationItem(name="low_resolution", score=0.8, severity="HIGH"),
                DegradationItem(name="blur", score=0.6, severity="MEDIUM"),
                DegradationItem(name="noise", score=0.7, severity="HIGH"),
                DegradationItem(name="low_light", score=0.9, severity="HIGH"),
                DegradationItem(name="jpeg_artifacts", score=0.5, severity="LOW"),
            ]
        )

        plan = planner.plan(analysis)

        # Expected canonical order: low_light -> denoise -> deblur -> jpeg -> super_resolution
        expected = ["low_light", "denoise", "deblur", "jpeg", "super_resolution"]
        assert plan == expected

    def test_planner_empty_degradations(self):
        planner = PipelinePlanner()
        analysis = AnalysisResponse(degradations=[])
        plan = planner.plan(analysis)
        assert plan == []

    def test_planner_custom_operations_override(self):
        planner = PipelinePlanner()
        analysis = AnalysisResponse(
            degradations=[DegradationItem(name="noise", score=0.7, severity="HIGH")]
        )
        custom = ["super_resolution", "denoise"]
        plan = planner.plan(analysis, custom_operations=custom)
        # Custom operations should be sorted by canonical rule: denoise -> super_resolution
        assert plan == ["denoise", "super_resolution"]


# ---------------------------------------------------------------------------
# 2. Restoration Engine Unit Tests
# ---------------------------------------------------------------------------
class TestRestorationEngine:
    def test_engine_empty_pipeline(self):
        engine = RestorationEngine()
        img = _make_sample_image()
        result = engine.run(img, [])

        assert result.final_image == img
        assert result.pipeline_steps == []
        assert result.total_time_seconds == 0.0

    def test_engine_single_step(self):
        engine = RestorationEngine()
        img = _make_sample_image(128, 128)
        # Run Zero-DCE++ low_light step
        result = engine.run(img, ["low_light"])

        assert result.final_image is not None
        assert len(result.pipeline_steps) == 1
        assert result.pipeline_steps[0]["operation"] == "low_light"
        assert result.total_time_seconds >= 0.0

    def test_compute_basic_metrics(self):
        orig = _make_sample_image(100, 100)
        restored = _make_sample_image(100, 100)
        metrics = _compute_basic_metrics(orig, restored)

        assert "original_sharpness" in metrics
        assert "restored_sharpness" in metrics
        assert "sharpness_change_percent" in metrics
        assert metrics["original_resolution"] == "100x100"


# ---------------------------------------------------------------------------
# 3. API Integration Test (POST /api/restore & GET /api/image/{id})
# ---------------------------------------------------------------------------
class TestRestoreAPI:
    @pytest.fixture
    def client(self):
        from fastapi.testclient import TestClient
        from app.main import app
        from app.routes.upload import _image_store
        _image_store.clear()
        c = TestClient(app)
        yield c
        _image_store.clear()

    def test_restore_endpoint_not_found(self, client):
        resp = client.post("/api/restore", json={"image_id": "non_existent_id"})
        assert resp.status_code == 404

    def test_full_upload_and_restore_flow(self, client):
        # 1. Upload sample image
        img = _make_sample_image(200, 200)
        buf = io.BytesIO()
        img.save(buf, format="PNG")

        upload_resp = client.post(
            "/api/upload",
            files={"file": ("test.png", buf.getvalue(), "image/png")},
        )
        assert upload_resp.status_code == 200
        image_id = upload_resp.json()["image_id"]

        # 2. Call /api/restore
        restore_resp = client.post("/api/restore", json={"image_id": image_id})
        assert restore_resp.status_code == 200
        data = restore_resp.json()

        assert data["success"] is True
        assert data["image_id"] == image_id
        assert "original_meta" in data
        assert "restored_meta" in data
        assert "pipeline_steps" in data
        assert "metrics" in data
        assert "inference_time_seconds" in data

        # 3. Serve restored image bytes via /api/image/{image_id}
        restored_id = f"restored_{image_id}"
        img_resp = client.get(f"/api/image/{restored_id}")
        assert img_resp.status_code == 200
        assert img_resp.headers["content-type"] == "image/png"
        assert len(img_resp.content) > 0
