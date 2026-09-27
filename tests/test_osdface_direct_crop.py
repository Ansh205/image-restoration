"""
Standalone Diagnostic Test for Direct OSDFace Model Inference.

Isolates OSDFaceModel.predict() from YuNet face detection and blending.
Saves input_face.png and osdface_output.png and verifies output difference statistics.
"""
import os
import sys
sys.path.insert(0, os.path.abspath("."))
import cv2
import math
import numpy as np
from PIL import Image
from loguru import logger

from models.osdface_model import OSDFaceModel


def test_direct_osdface_crop_inference():
    logger.info("============================================================")
    logger.info("[DIRECT OSDFACE CROP DIAGNOSTIC TEST]")
    logger.info("============================================================")

    # 1. Create a synthetic test face crop (512x512 with facial features & noise)
    h, w = 512, 512
    face_crop = np.full((h, w, 3), 180, dtype=np.uint8)
    
    # Draw face shape & features (eyes, nose, mouth)
    cv2.ellipse(face_crop, (256, 256), (150, 200), 0, 0, 360, (220, 190, 170), -1) # face oval
    cv2.circle(face_crop, (190, 210), 25, (50, 40, 30), -1)   # left eye
    cv2.circle(face_crop, (322, 210), 25, (50, 40, 30), -1)   # right eye
    cv2.line(face_crop, (256, 240), (256, 300), (140, 100, 90), 4) # nose
    cv2.ellipse(face_crop, (256, 340), (60, 20), 0, 0, 180, (80, 40, 40), 6) # mouth
    
    # Add blur & noise degradation
    face_crop = cv2.GaussianBlur(face_crop, (11, 11), 3.0)
    noise = np.random.normal(0, 15, face_crop.shape).astype(np.int16)
    face_crop = np.clip(face_crop.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # Save input image
    out_dir = "tests/artifacts"
    os.makedirs(out_dir, exist_ok=True)
    input_path = os.path.join(out_dir, "input_face.png")
    output_path = os.path.join(out_dir, "osdface_output.png")

    cv2.imwrite(input_path, face_crop)
    logger.info(f"Saved input face crop to: {input_path}")

    # 2. Instantiate and load real OSDFace model wrapper
    model = OSDFaceModel(device="cpu")
    model.load()

    assert model.real_checkpoint_loaded, "[TEST ERROR] Real OSDFace checkpoint was not loaded!"

    # 3. Run direct model inference
    logger.info("Running OSDFaceModel.predict(face_crop)...")
    restored_crop = model.predict(face_crop)

    # Save output image
    cv2.imwrite(output_path, restored_crop)
    logger.info(f"Saved restored face crop to: {output_path}")

    # 4. Calculate Output Difference Statistics
    in_arr = face_crop.astype(np.float32)
    out_arr = restored_crop.astype(np.float32)

    abs_diff = np.abs(out_arr - in_arr)
    mad = float(np.mean(abs_diff))
    max_diff = float(np.max(abs_diff))
    changed_pixels_pct = float(np.mean(abs_diff > 1.0) * 100.0)

    mse = float(np.mean((out_arr - in_arr) ** 2))
    psnr = 20.0 * math.log10(255.0 / math.sqrt(mse)) if mse > 1e-6 else 99.99

    logger.info("============================================================")
    logger.info("[DIRECT OSDFACE DIAGNOSTIC METRICS]")
    logger.info(f"Input Shape: {face_crop.shape}")
    logger.info(f"Output Shape: {restored_crop.shape}")
    logger.info(f"Mean Abs Difference (MAD): {mad:.4f}")
    logger.info(f"Max Pixel Difference: {max_diff:.1f}")
    logger.info(f"Changed Pixels: {changed_pixels_pct:.2f}%")
    logger.info(f"PSNR: {psnr:.2f} dB")
    logger.info(f"Mathematically Identical (in == out): {np.array_equal(face_crop, restored_crop)}")
    logger.info("============================================================")

    # Assertions
    assert not np.array_equal(face_crop, restored_crop), "[TEST FAIL] Input and Output crops are mathematically identical!"
    assert mad > 0.1, f"[TEST FAIL] MAD is too low ({mad:.4f}), inference produced negligible change!"
    assert changed_pixels_pct > 1.0, f"[TEST FAIL] Changed pixels percentage is too low ({changed_pixels_pct:.2f}%)!"

    logger.info("✅ Direct OSDFace face-crop diagnostic test PASSED successfully!")


if __name__ == "__main__":
    test_direct_osdface_crop_inference()
