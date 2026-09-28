"""
Diagnostic script for OSDFace color space tracing, channel means logging, and debug image generation.
"""
import os
import cv2
import numpy as np
import torch
from pathlib import Path
from PIL import Image
from loguru import logger

from core.osdface_processor import OSDFaceProcessor, get_osdface_model
from models.osdface_model import OSDFaceModel


def test_osdface_color_and_mask():
    logger.info("============================================================")
    logger.info("[OSDFACE COLOR & MASK DIAGNOSTIC TEST]")
    logger.info("============================================================")

    # 1. Load or create a realistic test face image
    test_img_path = Path("tests/artifacts/input_face.png")
    if test_img_path.exists():
        img_bgr = cv2.imread(str(test_img_path))
    else:
        img_bgr = np.full((512, 512, 3), 180, dtype=np.uint8)
        cv2.ellipse(img_bgr, (256, 256), (150, 200), 0, 0, 360, (180, 150, 210), -1)

    h_img, w_img, c_img = img_bgr.shape
    logger.info(f"Loaded test image: shape={img_bgr.shape}, dtype={img_bgr.dtype}")

    # Trace Color Space:
    logger.info("------------------------------------------------------------")
    logger.info("[COLOR SPACE TRACE]")
    logger.info("Original image loaded with OpenCV: BGR")
    logger.info(f"Original BGR channel means -> B: {img_bgr[:, :, 0].mean():.2f}, G: {img_bgr[:, :, 1].mean():.2f}, R: {img_bgr[:, :, 2].mean():.2f}")

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    logger.info(f"Converted to RGB -> R: {img_rgb[:, :, 0].mean():.2f}, G: {img_rgb[:, :, 1].mean():.2f}, B: {img_rgb[:, :, 2].mean():.2f}")

    # Load Model
    model = OSDFaceModel(device="cpu")
    model.load()

    # Model Input (RGB float32)
    input_float = img_rgb.astype(np.float32) / 255.0
    in_tensor = torch.from_numpy(input_float).permute(2, 0, 1).unsqueeze(0)
    in_tensor_norm = (in_tensor - 0.5) * 2.0

    with torch.no_grad():
        latents = model.vre_encoder(in_tensor_norm)
        b, c, h_lat, w_lat = latents.shape
        latents_flat = latents.view(b, c, h_lat * w_lat)
        proj_1d = model.embedding_proj(latents_flat)
        proj_spatial = proj_1d[:, :512, :].view(b, c, h_lat, w_lat)
        
        out_raw = model.vre_decoder(latents + proj_spatial * 0.15)
        out_raw_np = out_raw.squeeze(0).permute(1, 2, 0).numpy()

    logger.info("------------------------------------------------------------")
    logger.info("[OSDFACE COLOR DEBUG — RAW DECODER OUTPUT]")
    logger.info(f"Input Crop Shape: {img_rgb.shape}, Channel Order: RGB")
    logger.info(f"Input Crop Means -> R: {img_rgb[:, :, 0].mean():.2f}, G: {img_rgb[:, :, 1].mean():.2f}, B: {img_rgb[:, :, 2].mean():.2f}")
    
    logger.info(f"Raw Decoder Output Shape: {out_raw_np.shape}, Channel Order: RGB")
    logger.info(f"Raw Decoder Channel Means -> Ch0(R): {out_raw_np[:, :, 0].mean():.4f}, Ch1(G): {out_raw_np[:, :, 1].mean():.4f}, Ch2(B): {out_raw_np[:, :, 2].mean():.4f}")
    logger.info(f"Raw Decoder Channel Std   -> Ch0(R): {out_raw_np[:, :, 0].std():.4f}, Ch1(G): {out_raw_np[:, :, 1].std():.4f}, Ch2(B): {out_raw_np[:, :, 2].std():.4f}")
    logger.info(f"Raw Decoder Min/Max/Mean  -> Min: {out_raw_np.min():.4f}, Max: {out_raw_np.max():.4f}, Mean: {out_raw_np.mean():.4f}")

    # Test Residual Addition
    restored_tensor = torch.clamp(in_tensor + out_raw * 0.25, 0.0, 1.0)
    restored_np_float = restored_tensor.squeeze(0).permute(1, 2, 0).numpy()
    restored_rgb = (restored_np_float * 255.0).clip(0, 255).astype(np.uint8)

    logger.info("------------------------------------------------------------")
    logger.info("[OSDFACE COLOR DEBUG — RESTORED CROP]")
    logger.info(f"Restored Crop Means -> R: {restored_rgb[:, :, 0].mean():.2f}, G: {restored_rgb[:, :, 1].mean():.2f}, B: {restored_rgb[:, :, 2].mean():.2f}")
    
    dR = restored_rgb[:, :, 0].mean() - img_rgb[:, :, 0].mean()
    dG = restored_rgb[:, :, 1].mean() - img_rgb[:, :, 1].mean()
    dB = restored_rgb[:, :, 2].mean() - img_rgb[:, :, 2].mean()
    
    logger.info(f"Per-channel shifts -> ΔR: {dR:+.2f}, ΔG: {dG:+.2f}, ΔB: {dB:+.2f}")
    if dR > dG + 5.0 and dB > dG + 5.0:
        logger.warning("[OSDFACE WARNING] Significant magenta (Red + Blue) shift detected in restored output!")
    elif dR > dG + 5.0:
        logger.warning("[OSDFACE WARNING] Significant red-channel shift detected in restored output!")

    # Save debug artifacts
    debug_dir = Path("tests/artifacts")
    debug_dir.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(str(debug_dir / "debug_osdface_input_face_1.png"), cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(debug_dir / "debug_osdface_output_face_1.png"), cv2.cvtColor(restored_rgb, cv2.COLOR_RGB2BGR))

    logger.info(f"Saved debug crops to {debug_dir}")
    logger.info("============================================================")


if __name__ == "__main__":
    test_osdface_color_and_mask()
