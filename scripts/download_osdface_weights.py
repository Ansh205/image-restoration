"""
Download OSDFace pretrained model weights from official Hugging Face mirror with retries.
Required files:
- associate_2.ckpt
- embedding_change_weights.pth
- pytorch_lora_weights.safetensors
"""
import os
import time
from pathlib import Path
from huggingface_hub import hf_hub_download
from loguru import logger

WEIGHTS_DIR = Path("weights/osdface")
PRETRAINED_DIR = Path("pretrained")
REPO_ID = "alecccdd/OSDFace"

REQUIRED_FILES = [
    "associate_2.ckpt",
    "embedding_change_weights.pth",
    "pytorch_lora_weights.safetensors"
]


def ensure_osdface_weights(max_retries: int = 5) -> dict[str, Path]:
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    PRETRAINED_DIR.mkdir(parents=True, exist_ok=True)

    downloaded_paths = {}

    for filename in REQUIRED_FILES:
        target_path = WEIGHTS_DIR / filename
        alt_path = PRETRAINED_DIR / filename

        if target_path.exists() and target_path.stat().st_size > 1000:
            logger.info(f"[OSDFACE] Found local weight: {target_path} ({target_path.stat().st_size / (1024*1024):.2f} MB)")
            downloaded_paths[filename] = target_path
            continue
        elif alt_path.exists() and alt_path.stat().st_size > 1000:
            logger.info(f"[OSDFACE] Found local weight in pretrained/: {alt_path} ({alt_path.stat().st_size / (1024*1024):.2f} MB)")
            downloaded_paths[filename] = alt_path
            continue

        success = False
        for attempt in range(1, max_retries + 1):
            logger.info(f"[OSDFACE] Downloading {filename} (Attempt {attempt}/{max_retries})...")
            try:
                downloaded_file = hf_hub_download(
                    repo_id=REPO_ID,
                    filename=filename,
                    local_dir=str(WEIGHTS_DIR),
                )
                file_path = Path(downloaded_file)
                if file_path.exists() and file_path.stat().st_size > 1000:
                    logger.info(f"[OSDFACE WEIGHT DOWNLOAD] Successfully downloaded {filename} ({file_path.stat().st_size / 1024 / 1024:.2f} MB)")
                    downloaded_paths[filename] = file_path
                    success = True
                    break
            except Exception as e:
                logger.warning(f"[OSDFACE RETRY] Attempt {attempt} failed for {filename}: {e}")
                time.sleep(2 * attempt)

        if not success:
            logger.error(f"[OSDFACE ERROR] Missing required weight: {filename}")
            raise FileNotFoundError(f"[OSDFACE ERROR] Missing required weight: {filename}")

    return downloaded_paths


if __name__ == "__main__":
    paths = ensure_osdface_weights()
    print("Downloaded/verified OSDFace weights:")
    for k, v in paths.items():
        print(f"  - {k}: {v} ({v.stat().st_size / (1024*1024):.2f} MB)")
