"""
Script to upload local model weights to Hugging Face repository: Ansh205/image-restoration-models.

Usage:
    python scripts/upload_weights_to_hf.py --token YOUR_HF_TOKEN

Or login first with HF CLI:
    huggingface-cli login
    python scripts/upload_weights_to_hf.py
"""
import os
import sys
import argparse
from pathlib import Path
from loguru import logger
from huggingface_hub import HfApi

REPO_ID = "Ansh205/image-restoration-models"

WEIGHT_FILES = [
    ("weights/umsn/Deblur_epoch_Best.pth", "umsn/Deblur_epoch_Best.pth"),
    ("weights/retinexformer/LOL_v2_real.pth", "retinexformer/LOL_v2_real.pth"),
    ("weights/scunet/scunet_color_real_psnr.pth", "scunet/scunet_color_real_psnr.pth"),
    ("weights/drunet/drunet_color.pth", "drunet/drunet_color.pth"),
    ("weights/restormer/restormer_deblurring.pth", "restormer/restormer_deblurring.pth"),
    ("weights/realesrgan/RealESRGAN_x4plus.pth", "realesrgan/RealESRGAN_x4plus.pth"),
    ("weights/zerodce/zero_dce_pp.pth", "zerodce/zero_dce_pp.pth"),
    ("weights/swinir/swinir_jpeg.pth", "swinir/swinir_jpeg.pth"),
]

def upload_weights(token: str = None):
    api = HfApi(token=token)
    logger.info(f"Target Hugging Face repository: {REPO_ID}")

    for local_rel_path, remote_rel_path in WEIGHT_FILES:
        local_path = Path(local_rel_path).resolve()
        if not local_path.exists():
            logger.warning(f"File not found locally: {local_path}. Skipping.")
            continue

        logger.info(f"Uploading {local_rel_path} -> {REPO_ID}/{remote_rel_path} ...")
        try:
            api.upload_file(
                path_or_fileobj=str(local_path),
                path_in_repo=remote_rel_path,
                repo_id=REPO_ID,
                repo_type="model",
            )
            logger.info(f"[OK] Uploaded {remote_rel_path} successfully!")
        except Exception as e:
            logger.error(f"Failed to upload {remote_rel_path}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload model weights to Hugging Face")
    parser.add_argument("--token", type=str, default=None, help="Hugging Face access token (write permissions)")
    args = parser.parse_args()
    upload_weights(args.token)
