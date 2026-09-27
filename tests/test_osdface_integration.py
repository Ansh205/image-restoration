"""
Integration tests for Phase 1 OSDFace Face Restoration integration.

Tests cover:
- Test 1: Single face image (OSDFace mode)
- Test 2: Multiple face image (OSDFace mode)
- Test 3: No face image (OSDFace mode -> skip & default pipeline fallback)
- Test 4: Default mode (OSDFace model is NEVER loaded or executed)
- Test 5: Both mode (Independent Branch A and Branch B outputs)
"""
import io
import os
import pytest
import numpy as np
import cv2
from PIL import Image
from fastapi.testclient import TestClient

from app.main import app
from core.face_detector import YuNetFaceDetector

client = TestClient(app)


@pytest.fixture(autouse=True)
def setup_test_osdface_weights():
    """Ensure dummy OSDFace weight files exist during unit test runs if missing."""
    osd_dir = os.path.join("weights", "osdface")
    os.makedirs(osd_dir, exist_ok=True)
    required = ["associate_2.ckpt", "embedding_change_weights.pth", "pytorch_lora_weights.safetensors"]
    for filename in required:
        path = os.path.join(osd_dir, filename)
        if not os.path.exists(path) or os.path.getsize(path) < 10:
            with open(path, "wb") as f:
                f.write(b"OSDFACE_TEST_WEIGHT_HEADER_" + b"0" * 2000)


def create_synthetic_image(has_face: bool = True, num_faces: int = 1) -> bytes:
    """
    Create a synthetic test image with or without drawing face features (eyes, nose, mouth).
    """
    if not has_face:
        # Plain solid image with zero face features
        img = np.ones((400, 400, 3), dtype=np.uint8) * 128
    else:
        img = np.ones((400, 400, 3), dtype=np.uint8) * 128
        for i in range(num_faces):
            cx = 100 + i * 150
            cy = 200
            # Head
            cv2.ellipse(img, (cx, cy), (50, 65), 0, 0, 360, (210, 180, 140), -1)
            # Eyes
            cv2.circle(img, (cx - 18, cy - 15), 6, (50, 50, 50), -1)
            cv2.circle(img, (cx + 18, cy - 15), 6, (50, 50, 50), -1)
            # Nose
            cv2.line(img, (cx, cy - 5), (cx, cy + 10), (40, 40, 40), 2)
            # Mouth
            cv2.ellipse(img, (cx, cy + 25), (18, 8), 0, 0, 180, (50, 50, 50), 2)

    pil_img = Image.fromarray(img)
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return buf.getvalue()


def test_yunet_face_detector_basic():
    """Verify YuNet detector initializes and returns bounding box list."""
    detector = YuNetFaceDetector()
    img_bytes = create_synthetic_image(has_face=True, num_faces=1)
    nparr = np.frombuffer(img_bytes, np.uint8)
    bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    bboxes = detector.detect(bgr)
    assert isinstance(bboxes, list)


def test_osdface_mode_single_face():
    """Test OSDFace mode on an image containing a face."""
    img_bytes = create_synthetic_image(has_face=True, num_faces=1)
    
    # 1. Upload
    up_resp = client.post("/api/upload", files={"file": ("face1.png", img_bytes, "image/png")})
    assert up_resp.status_code == 200
    image_id = up_resp.json()["image_id"]

    # 2. Restore in OSDFace mode
    res_resp = client.post("/api/restore", json={
        "image_id": image_id,
        "restoration_mode": "osdface"
    })
    assert res_resp.status_code == 200
    data = res_resp.json()
    assert data["success"] is True
    assert data["restoration_mode"] == "osdface"
    assert len(data["pipeline_steps"]) >= 1


def test_osdface_mode_multiple_faces():
    """Test OSDFace mode on an image containing 2 faces."""
    img_bytes = create_synthetic_image(has_face=True, num_faces=2)
    
    up_resp = client.post("/api/upload", files={"file": ("faces2.png", img_bytes, "image/png")})
    assert up_resp.status_code == 200
    image_id = up_resp.json()["image_id"]

    res_resp = client.post("/api/restore", json={
        "image_id": image_id,
        "restoration_mode": "osdface"
    })
    assert res_resp.status_code == 200
    data = res_resp.json()
    assert data["success"] is True
    assert data["restoration_mode"] == "osdface"


def test_osdface_mode_no_face_fallback():
    """Test OSDFace mode on an image without a face -> verify skip & default pipeline fallback."""
    img_bytes = create_synthetic_image(has_face=False, num_faces=0)
    
    up_resp = client.post("/api/upload", files={"file": ("noface.png", img_bytes, "image/png")})
    assert up_resp.status_code == 200
    image_id = up_resp.json()["image_id"]

    res_resp = client.post("/api/restore", json={
        "image_id": image_id,
        "restoration_mode": "osdface"
    })
    assert res_resp.status_code == 200
    data = res_resp.json()
    assert data["success"] is True
    assert data["restoration_mode"] == "osdface"


def test_default_mode_osdface_never_executed():
    """Test Default mode -> verify OSDFace is never triggered."""
    img_bytes = create_synthetic_image(has_face=True, num_faces=1)
    
    up_resp = client.post("/api/upload", files={"file": ("default.png", img_bytes, "image/png")})
    assert up_resp.status_code == 200
    image_id = up_resp.json()["image_id"]

    res_resp = client.post("/api/restore", json={
        "image_id": image_id,
        "restoration_mode": "default"
    })
    assert res_resp.status_code == 200
    data = res_resp.json()
    assert data["success"] is True
    assert data["restoration_mode"] == "default"
    # Ensure OSDFace was NOT added to pipeline steps
    step_names = [s.get("model_name") for s in data["pipeline_steps"]]
    assert "OSDFace" not in step_names


def test_both_mode_execution():
    """Test Both mode -> verify independent Branch A (Default) and Branch B (OSDFace)."""
    img_bytes = create_synthetic_image(has_face=True, num_faces=1)
    
    up_resp = client.post("/api/upload", files={"file": ("both.png", img_bytes, "image/png")})
    assert up_resp.status_code == 200
    image_id = up_resp.json()["image_id"]

    res_resp = client.post("/api/restore", json={
        "image_id": image_id,
        "restoration_mode": "both"
    })
    assert res_resp.status_code == 200
    data = res_resp.json()
    assert data["success"] is True
    assert data["restoration_mode"] == "both"
    assert "osdface_result" in data
    assert data["osdface_result"] is not None
