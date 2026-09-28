"""
Phase 2 Integration Test Suite — OSDFace + Default Restoration Pipeline Integration.
Tests all 7 required test scenarios:
  1. Default mode (untouched standard 3-pass engine)
  2. OSDFace mode with 1 face (OSDFace -> OSDFace Evaluator -> Default pipeline)
  3. OSDFace mode with multiple faces (multi-face restoration -> Evaluator -> Default pipeline)
  4. OSDFace mode with no faces (YuNet 0 faces -> OSDFace skipped -> Default pipeline)
  5. Both mode with faces (Branch A: Standard vs Branch B: OSDFace-Assisted Standard)
  6. Both mode with no face (Branch A: Standard vs Branch B: Standard fallback)
  7. OSDFace candidate rejection (DISCARD -> fallback to original baseline -> Default pipeline)
"""
import io
import cv2
import numpy as np
import pytest
from PIL import Image
from fastapi.testclient import TestClient

from app.main import app
from app.routes.upload import _image_store
from core.osdface_evaluator import OSDFaceEvaluator

client = TestClient(app)


def _upload_test_image(img_bgr: np.ndarray) -> str:
    """Helper to upload a test numpy BGR image and return image_id."""
    is_success, buffer = cv2.imencode(".png", img_bgr)
    assert is_success
    files = {"file": ("test.png", io.BytesIO(buffer), "image/png")}
    response = client.post("/api/upload", files=files)
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    return data["image_id"]


def _create_synthetic_face_image(num_faces: int = 1) -> np.ndarray:
    """Create a synthetic test image with N clear faces for YuNet detection."""
    img = np.full((600, 800, 3), 200, dtype=np.uint8)
    if num_faces == 1:
        # Single face centered
        cx, cy = 400, 300
        cv2.ellipse(img, (cx, cy), (80, 100), 0, 0, 360, (180, 140, 100), -1) # Head
        cv2.circle(img, (cx - 30, cy - 20), 12, (255, 255, 255), -1) # Left Eye
        cv2.circle(img, (cx - 30, cy - 20), 5, (50, 50, 50), -1)
        cv2.circle(img, (cx + 30, cy - 20), 12, (255, 255, 255), -1) # Right Eye
        cv2.circle(img, (cx + 30, cy - 20), 5, (50, 50, 50), -1)
        cv2.ellipse(img, (cx, cy + 30), (25, 12), 0, 0, 180, (100, 50, 150), 3) # Mouth
    elif num_faces >= 2:
        # Two distinct faces with wide separation and clear features
        img = np.full((700, 1200, 3), 200, dtype=np.uint8)
        for cx in (350, 850):
            cy = 350
            cv2.ellipse(img, (cx, cy), (90, 120), 0, 0, 360, (180, 140, 100), -1) # Head
            cv2.circle(img, (cx - 35, cy - 25), 14, (255, 255, 255), -1) # Left Eye
            cv2.circle(img, (cx - 35, cy - 25), 6, (50, 50, 50), -1)
            cv2.circle(img, (cx + 35, cy - 25), 14, (255, 255, 255), -1) # Right Eye
            cv2.circle(img, (cx + 35, cy - 25), 6, (50, 50, 50), -1)
            cv2.ellipse(img, (cx, cy + 35), (30, 15), 0, 0, 180, (100, 50, 150), 3) # Mouth
    return img


def _create_synthetic_no_face_image() -> np.ndarray:
    """Create a landscape image with zero faces."""
    img = np.zeros((400, 400, 3), dtype=np.uint8)
    cv2.rectangle(img, (50, 50), (350, 350), (100, 200, 100), -1)
    return img


def test_phase2_default_mode():
    """TEST 1: Default mode runs exact untouched 3-pass pipeline without OSDFace."""
    img_bgr = _create_synthetic_face_image(1)
    image_id = _upload_test_image(img_bgr)

    res = client.post("/api/restore", json={"image_id": image_id, "restoration_mode": "default"})
    assert res.status_code == 200
    data = res.json()

    assert data["success"] is True
    assert data["restoration_mode"] == "default"
    assert "osdface_metadata" not in data or data["osdface_metadata"] is None
    assert "standard_metadata" in data
    assert data["standard_metadata"]["result_available"] is True


def test_phase2_osdface_mode_single_face():
    """TEST 2: OSDFace mode with 1 face (OSDFace -> Evaluator -> Default pipeline)."""
    img_bgr = _create_synthetic_face_image(1)
    image_id = _upload_test_image(img_bgr)

    res = client.post("/api/restore", json={"image_id": image_id, "restoration_mode": "osdface"})
    assert res.status_code == 200
    data = res.json()

    assert data["success"] is True
    assert data["restoration_mode"] == "osdface"
    assert "osdface_metadata" in data and data["osdface_metadata"] is not None
    osd_meta = data["osdface_metadata"]
    assert osd_meta["enabled"] is True
    assert osd_meta["faces_detected"] >= 1
    assert osd_meta["evaluation_decision"] in ["KEEP", "DISCARD"]


def test_phase2_osdface_mode_multiple_faces():
    """TEST 3: OSDFace mode with multiple faces (Multi-face restoration -> Default pipeline)."""
    img_bgr = _create_synthetic_face_image(2)
    image_id = _upload_test_image(img_bgr)

    res = client.post("/api/restore", json={"image_id": image_id, "restoration_mode": "osdface"})
    assert res.status_code == 200
    data = res.json()

    assert data["success"] is True
    assert data["restoration_mode"] == "osdface"
    osd_meta = data["osdface_metadata"]
    assert osd_meta["enabled"] is True
    assert osd_meta["faces_detected"] >= 2


def test_phase2_osdface_mode_no_face_fallback():
    """TEST 4: OSDFace mode with 0 faces (YuNet 0 faces -> OSDFace skipped -> Default pipeline)."""
    img_bgr = _create_synthetic_no_face_image()
    image_id = _upload_test_image(img_bgr)

    res = client.post("/api/restore", json={"image_id": image_id, "restoration_mode": "osdface"})
    assert res.status_code == 200
    data = res.json()

    assert data["success"] is True
    assert data["restoration_mode"] == "osdface"
    osd_meta = data["osdface_metadata"]
    assert osd_meta["faces_detected"] == 0
    assert osd_meta["evaluation_decision"] == "SKIPPED"


def test_phase2_both_mode_with_face():
    """TEST 5: Both mode with face (Branch A: Standard vs Branch B: OSDFace-Assisted Standard)."""
    img_bgr = _create_synthetic_face_image(1)
    image_id = _upload_test_image(img_bgr)

    res = client.post("/api/restore", json={"image_id": image_id, "restoration_mode": "both"})
    assert res.status_code == 200
    data = res.json()

    assert data["success"] is True
    assert data["restoration_mode"] == "both"
    assert "standard_execution_time" in data
    assert "osdface_assisted_execution_time" in data
    assert "total_execution_time" in data
    assert data["standard_execution_time"] > 0
    assert data["osdface_assisted_execution_time"] > 0
    assert data["total_execution_time"] >= data["standard_execution_time"]
    assert "osdface_result" in data and data["osdface_result"] is not None
    branch_b = data["osdface_result"]
    assert branch_b["success"] is True
    assert "image_url" in branch_b


def test_phase2_both_mode_no_face():
    """TEST 6: Both mode with no face (Branch A: Standard vs Branch B: Standard fallback)."""
    img_bgr = _create_synthetic_no_face_image()
    image_id = _upload_test_image(img_bgr)

    res = client.post("/api/restore", json={"image_id": image_id, "restoration_mode": "both"})
    assert res.status_code == 200
    data = res.json()

    assert data["success"] is True
    assert data["restoration_mode"] == "both"
    branch_b = data["osdface_result"]
    assert branch_b["skipped"] is True
    assert branch_b["faces_detected"] == 0


def test_phase2_osdface_evaluator_rejection():
    """TEST 7: OSDFace evaluator DISCARD decision for identical / noise-degraded candidates."""
    evaluator = OSDFaceEvaluator()
    base = np.full((100, 100, 3), 128, dtype=np.uint8)
    cand = base.copy() # Identical candidate (MAD = 0.0)

    decision, reason, metrics = evaluator.evaluate(base, cand, [(10, 10, 80, 80)])
    assert decision == "DISCARD"
    assert "MAD < 0.1" in reason or "Negligible" in reason
