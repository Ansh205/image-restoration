"""
Unit and integration tests for the Degradation Analyzer.
"""
import io
import pytest
import numpy as np
from PIL import Image, ImageFilter

from core.analyzer.blur_detector import detect_blur
from core.analyzer.noise_detector import detect_noise
from core.analyzer.resolution_detector import detect_resolution
from core.analyzer.lowlight_detector import detect_low_light
from core.analyzer.jpeg_detector import detect_jpeg_artifacts
from core.analyzer.analyzer import DegradationAnalyzer


# ---------------------------------------------------------------------------
# Synthetic Test Image Generators
# ---------------------------------------------------------------------------
def _make_clean_image(width: int = 600, height: int = 600) -> Image.Image:
    """Create a clean, sharp synthetic image with 20px checkerboard pattern."""
    arr = np.zeros((height, width, 3), dtype=np.uint8)
    for y in range(0, height, 20):
        for x in range(0, width, 20):
            if ((x // 20) + (y // 20)) % 2 == 0:
                arr[y:y+20, x:x+20] = [180, 180, 180]
            else:
                arr[y:y+20, x:x+20] = [100, 100, 100]
    return Image.fromarray(arr)


def _make_blurry_image() -> Image.Image:
    """Create a heavily blurred image."""
    img = _make_clean_image()
    return img.filter(ImageFilter.GaussianBlur(radius=8))


def _make_noisy_image() -> Image.Image:
    """Create an image with heavy Gaussian noise."""
    img_np = np.array(_make_clean_image(), dtype=np.float32)
    noise = np.random.normal(0, 30, img_np.shape)
    noisy_np = np.clip(img_np + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy_np)


def _make_low_res_image() -> Image.Image:
    """Create a small 128x128 image."""
    return Image.new("RGB", (128, 128), (100, 150, 200))


def _make_dark_image() -> Image.Image:
    """Create a very dark low-light image."""
    arr = np.ones((600, 600, 3), dtype=np.uint8) * 30  # Dark gray
    return Image.fromarray(arr)


def _make_jpeg_artifact_image() -> Image.Image:
    """Create a JPEG-compressed image with heavy blocking artifacts."""
    img = _make_clean_image()
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=5)  # Extreme compression
    buf.seek(0)
    return Image.open(buf)


# ---------------------------------------------------------------------------
# Individual Detector Unit Tests
# ---------------------------------------------------------------------------
class TestIndividualDetectors:
    def test_blur_detection(self):
        clean = _make_clean_image()
        blurry = _make_blurry_image()

        is_blurry_clean, _, _, clean_det = detect_blur(clean, threshold=150.0)
        is_blurry_blur, sev, _, blur_det = detect_blur(blurry, threshold=150.0)

        assert is_blurry_clean is False
        assert is_blurry_blur is True
        assert sev > 0.3
        assert blur_det["laplacian_variance"] < clean_det["laplacian_variance"]

    def test_noise_detection(self):
        clean = _make_clean_image()
        noisy = _make_noisy_image()

        is_noisy_clean, _, _, _ = detect_noise(clean, threshold=15.0)
        is_noisy_dirty, sev, _, dict_det = detect_noise(noisy, threshold=15.0)

        assert is_noisy_dirty is True
        assert sev > 0.0
        assert dict_det["estimated_noise_sigma"] > 15.0

    def test_resolution_detection(self):
        low_res = _make_low_res_image()
        high_res = _make_clean_image(1024, 1024)

        is_low, sev, _, _ = detect_resolution(low_res, min_dimension=512)
        assert is_low is True
        assert sev > 0.5

        is_high, _, _, _ = detect_resolution(high_res, min_dimension=512)
        assert is_high is False

    def test_lowlight_detection(self):
        dark = _make_dark_image()
        normal = _make_clean_image()
        cfg = {"low_light_threshold": 0.35, "overexposure_threshold": 0.70, "dark_ratio_threshold": 0.20, "bright_ratio_threshold": 0.20}

        degradations_dark, metrics_dark = detect_low_light(dark, config=cfg)
        names = [d.name for d in degradations_dark]
        assert "low_light" in names
        assert metrics_dark["mean_luminance"] < 0.35

        degradations_normal, metrics_norm = detect_low_light(normal, config=cfg)
        names_norm = [d.name for d in degradations_normal]
        assert "low_light" not in names_norm

    def test_jpeg_detection(self):
        jpeg_img = _make_jpeg_artifact_image()
        has_jpeg, sev, _, _ = detect_jpeg_artifacts(jpeg_img, threshold=0.10)
        assert has_jpeg is True
        assert sev > 0.0


# ---------------------------------------------------------------------------
# DegradationAnalyzer Aggregator Tests
# ---------------------------------------------------------------------------
class TestDegradationAnalyzer:
    def test_analyzer_blurry_noisy_dark(self):
        analyzer = DegradationAnalyzer()
        dark_img = _make_dark_image()

        report = analyzer.analyze(dark_img, image_id="test1234")
        assert report.image_id == "test1234"
        names = [d.name for d in report.degradations]
        assert "low_light" in names

    def test_analyzer_clean_image(self):
        analyzer = DegradationAnalyzer()
        clean_img = _make_clean_image(800, 800)

        report = analyzer.analyze(clean_img, image_id="clean001")
        # Clean synthetic sine pattern should not trigger dark, low-res, or jpeg
        names = [d.name for d in report.degradations]
        assert "low_light" not in names
        assert "low_resolution" not in names


# ---------------------------------------------------------------------------
# API Integration Test
# ---------------------------------------------------------------------------
class TestAnalyzeAPI:
    @pytest.fixture
    def client(self):
        from fastapi.testclient import TestClient
        from app.main import app
        return TestClient(app)

    def test_analyze_endpoint_not_found(self, client):
        resp = client.post("/api/analyze", json={"image_id": "non_existent_id"})
        assert resp.status_code == 404

    def test_upload_and_analyze_flow(self, client):
        # 1. Upload dark image
        dark = _make_dark_image()
        buf = io.BytesIO()
        dark.save(buf, format="PNG")

        upload_resp = client.post(
            "/api/upload",
            files={"file": ("dark.png", buf.getvalue(), "image/png")},
        )
        assert upload_resp.status_code == 200
        image_id = upload_resp.json()["image_id"]

        # 2. Analyze uploaded image
        analyze_resp = client.post("/api/analyze", json={"image_id": image_id})
        assert analyze_resp.status_code == 200
        data = analyze_resp.json()
        assert data["image_id"] == image_id
        deg_names = [d["name"] for d in data["degradations"]]
        assert "low_light" in deg_names
