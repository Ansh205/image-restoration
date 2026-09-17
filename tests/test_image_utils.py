"""
Tests for image I/O utilities and upload endpoint.
"""
import io
import pytest
from PIL import Image

from core.pipeline.image_utils import (
    load_image_from_bytes,
    validate_extension,
    validate_file_size,
    resize_if_needed,
    convert_to_rgb,
    pil_to_numpy,
    numpy_to_pil,
    image_to_bytes,
    ImageValidationError,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_test_image(width: int = 100, height: int = 80, mode: str = "RGB", fmt: str = "PNG") -> bytes:
    """Create a minimal test image as bytes."""
    img = Image.new(mode, (width, height), color=(128, 64, 32))
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Validation tests
# ---------------------------------------------------------------------------
class TestValidation:
    def test_valid_extensions(self):
        for ext in [".jpg", ".jpeg", ".png", ".webp", ".bmp"]:
            validate_extension(f"photo{ext}")  # Should not raise

    def test_invalid_extension(self):
        with pytest.raises(ImageValidationError, match="Unsupported"):
            validate_extension("document.pdf")

    def test_invalid_extension_gif(self):
        with pytest.raises(ImageValidationError, match="Unsupported"):
            validate_extension("animation.gif")

    def test_file_size_ok(self):
        small_data = b"x" * (1024 * 1024)  # 1 MB
        validate_file_size(small_data, max_mb=10)  # Should not raise

    def test_file_size_too_large(self):
        big_data = b"x" * (11 * 1024 * 1024)  # 11 MB
        with pytest.raises(ImageValidationError, match="too large"):
            validate_file_size(big_data, max_mb=10)


# ---------------------------------------------------------------------------
# Loading tests
# ---------------------------------------------------------------------------
class TestLoadImage:
    def test_load_valid_png(self):
        data = _make_test_image(200, 150, "RGB", "PNG")
        img, meta = load_image_from_bytes(data, "test.png")
        assert img.mode == "RGB"
        assert meta.width == 200
        assert meta.height == 150
        assert meta.channels == 3

    def test_load_valid_jpeg(self):
        data = _make_test_image(300, 200, "RGB", "JPEG")
        img, meta = load_image_from_bytes(data, "test.jpg")
        assert img.mode == "RGB"
        assert meta.width == 300

    def test_load_rgba_converts_to_rgb(self):
        data = _make_test_image(100, 100, "RGBA", "PNG")
        img, meta = load_image_from_bytes(data, "test.png")
        assert img.mode == "RGB"

    def test_load_grayscale_converts_to_rgb(self):
        # Create a grayscale image
        gray_img = Image.new("L", (100, 100), 128)
        buf = io.BytesIO()
        gray_img.save(buf, format="PNG")
        data = buf.getvalue()

        img, meta = load_image_from_bytes(data, "gray.png")
        assert img.mode == "RGB"

    def test_reject_invalid_extension(self):
        data = _make_test_image()
        with pytest.raises(ImageValidationError):
            load_image_from_bytes(data, "test.gif")

    def test_reject_corrupt_image(self):
        with pytest.raises(ImageValidationError, match="Cannot open"):
            load_image_from_bytes(b"not an image at all", "fake.png")

    def test_reject_empty_bytes(self):
        with pytest.raises(ImageValidationError):
            load_image_from_bytes(b"", "empty.png")


# ---------------------------------------------------------------------------
# Preprocessing tests
# ---------------------------------------------------------------------------
class TestPreprocessing:
    def test_convert_rgb_noop(self):
        img = Image.new("RGB", (50, 50))
        result = convert_to_rgb(img)
        assert result.mode == "RGB"

    def test_convert_rgba(self):
        img = Image.new("RGBA", (50, 50), (255, 0, 0, 128))
        result = convert_to_rgb(img)
        assert result.mode == "RGB"

    def test_resize_not_needed(self):
        img = Image.new("RGB", (500, 300))
        result, resized = resize_if_needed(img, max_size=1024)
        assert not resized
        assert result.size == (500, 300)

    def test_resize_needed(self):
        img = Image.new("RGB", (2000, 1000))
        result, resized = resize_if_needed(img, max_size=1024)
        assert resized
        assert max(result.size) == 1024
        # Check aspect ratio preserved
        assert result.size == (1024, 512)

    def test_resize_tall_image(self):
        img = Image.new("RGB", (500, 2000))
        result, resized = resize_if_needed(img, max_size=1024)
        assert resized
        assert result.size[1] == 1024

    def test_numpy_roundtrip(self):
        img = Image.new("RGB", (50, 50), (100, 150, 200))
        arr = pil_to_numpy(img)
        assert arr.shape == (50, 50, 3)
        assert arr.dtype.name == "float32"
        assert 0.0 <= arr.min() and arr.max() <= 1.0

        restored = numpy_to_pil(arr)
        assert restored.mode == "RGB"
        assert restored.size == (50, 50)


# ---------------------------------------------------------------------------
# Encoding tests
# ---------------------------------------------------------------------------
class TestEncoding:
    def test_image_to_png_bytes(self):
        img = Image.new("RGB", (50, 50), (255, 0, 0))
        data = image_to_bytes(img, fmt="PNG")
        assert len(data) > 0
        # Verify it's a valid PNG
        reloaded = Image.open(io.BytesIO(data))
        assert reloaded.size == (50, 50)

    def test_image_to_jpeg_bytes(self):
        img = Image.new("RGB", (50, 50), (0, 255, 0))
        data = image_to_bytes(img, fmt="JPEG", quality=85)
        assert len(data) > 0


# ---------------------------------------------------------------------------
# API tests (using FastAPI TestClient)
# ---------------------------------------------------------------------------
class TestUploadAPI:
    @pytest.fixture
    def client(self):
        from fastapi.testclient import TestClient
        from app.main import app
        return TestClient(app)

    def test_health_check(self, client):
        resp = client.get("/api/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "healthy"

    def test_upload_valid_image(self, client):
        data = _make_test_image(200, 150, "RGB", "PNG")
        resp = client.post(
            "/api/upload",
            files={"file": ("test.png", data, "image/png")},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["success"] is True
        assert "image_id" in body
        assert body["meta"]["width"] == 200
        assert body["meta"]["height"] == 150

    def test_upload_invalid_extension(self, client):
        data = _make_test_image()
        resp = client.post(
            "/api/upload",
            files={"file": ("test.gif", data, "image/gif")},
        )
        assert resp.status_code == 422

    def test_upload_empty_file(self, client):
        resp = client.post(
            "/api/upload",
            files={"file": ("empty.png", b"", "image/png")},
        )
        assert resp.status_code == 400
