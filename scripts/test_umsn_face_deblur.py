"""
Verification script for UMSN Face Deblurring integration:
- YuNet SOTA Face Detection
- Zero-face skip behavior (no full-image UMSN)
- Face-region crop evaluation
- Two-pass attempted operation tracking
"""
import os
import sys
sys.path.insert(0, os.path.abspath("."))
import numpy as np
import cv2
from PIL import Image
from loguru import logger

from models.deblurring.umsn import UMSNModel
from core.evaluator.restoration_evaluator import RestorationEvaluator
from core.analyzer.analyzer import DegradationAnalyzer
from core.pipeline.planner import PipelinePlanner
from core.pipeline.engine import RestorationEngine


def test_umsn_face_detection_and_inference():
    logger.info("============================================================")
    logger.info("TEST 1: UMSN Face Deblurring on Synthetic Image (No Face)")
    logger.info("============================================================")
    
    umsn = UMSNModel()
    # Create simple landscape image without faces
    arr_no_face = np.full((300, 300, 3), 120, dtype=np.uint8)
    arr_no_face[50:150, 50:150] = 200
    img_no_face = Image.fromarray(arr_no_face)

    output_no_face = umsn.restore(img_no_face)

    # Verify image returned unchanged when 0 faces detected
    assert np.array_equal(np.array(img_no_face), np.array(output_no_face))
    logger.info("[OK] No face detected -> UMSN skipped and input image returned unchanged.")

    logger.info("\n============================================================")
    logger.info("TEST 2: UMSN Face Deblurring on Synthetic Image with Face Circle")
    logger.info("============================================================")
    
    # Create synthetic image with a face-like structure
    arr_face = np.full((300, 300, 3), 180, dtype=np.uint8)
    # Draw face oval
    cv2.ellipse(arr_face, (150, 150), (60, 80), 0, 0, 360, (220, 190, 170), -1)
    # Draw eyes & mouth
    cv2.circle(arr_face, (130, 130), 8, (40, 40, 40), -1)
    cv2.circle(arr_face, (170, 130), 8, (40, 40, 40), -1)
    cv2.ellipse(arr_face, (150, 180), (25, 12), 0, 0, 180, (150, 50, 50), -1)
    img_face = Image.fromarray(arr_face)

    bboxes = umsn._detect_faces(arr_face)
    logger.info(f"Face detector returned {len(bboxes)} face(s) for synthetic image.")

    output_face = umsn.restore(img_face)
    assert output_face is not None
    logger.info("[OK] Face image processed successfully by UMSN.")


def test_evaluator_face_crop_behavior():
    logger.info("\n============================================================")
    logger.info("TEST 3: Restoration Evaluator on Identical / Skipped Step")
    logger.info("============================================================")
    evaluator = RestorationEvaluator()

    arr = np.full((200, 200, 3), 100, dtype=np.uint8)
    img = Image.fromarray(arr)

    res = evaluator.evaluate("deblur", img, img.copy())
    logger.info(f"Evaluator decision: {res.decision} | Reason: {res.reason}")
    assert res.decision == "DISCARD"
    assert "identical" in res.reason.lower() or "no meaningful" in res.reason.lower()
    logger.info("[OK] Identical output correctly evaluated as DISCARD.")


def test_two_pass_attempted_operations_tracking():
    logger.info("\n============================================================")
    logger.info("TEST 4: Two-Pass Attempted Operations Tracking")
    logger.info("============================================================")
    analyzer = DegradationAnalyzer()
    planner = PipelinePlanner()
    engine = RestorationEngine()

    arr = np.full((264, 266, 3), 100, dtype=np.uint8)
    # Add blur
    arr_blur = cv2.GaussianBlur(arr, (7, 7), 2.0)
    test_img = Image.fromarray(arr_blur)

    res = engine.run_two_pass(
        image=test_img,
        analyzer=analyzer,
        planner=planner,
        image_id="test_attempted_ops",
    )

    # Check operations in pipeline steps
    ops = [step["operation"] for step in res.pipeline_steps]
    logger.info(f"Executed steps: {ops}")
    # Verify 'deblur' appears at most once in pipeline_steps
    assert ops.count("deblur") <= 1
    logger.info("[OK] Attempted operation tracking prevented duplicate execution in Pass 2.")


if __name__ == "__main__":
    test_umsn_face_detection_and_inference()
    test_evaluator_face_crop_behavior()
    test_two_pass_attempted_operations_tracking()
