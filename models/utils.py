"""
Weight management and model downloading utilities.

Handles downloading pretrained checkpoints from Hugging Face Hub
or official repositories with automatic local caching.
"""
from pathlib import Path
from typing import Any

from loguru import logger


# Base directory for local model weight cache
WEIGHTS_DIR = Path(__file__).resolve().parent.parent / "weights"


def get_weights_path(repo_id: str, filename: str) -> Path:
    """
    Get the local path for a model checkpoint.
    If the checkpoint does not exist locally, download it using huggingface_hub.

    Args:
        repo_id: HuggingFace repository ID (e.g., "Ansh205/image-restoration-models").
        filename: Relative path in repo (e.g., "drunet/drunet_color.pth").

    Returns:
        Path to local weight file.
    """
    local_path = WEIGHTS_DIR / filename
    if local_path.exists():
        logger.debug(f"Using cached weights: {local_path}")
        return local_path

    # Ensure parent directory exists
    local_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Downloading weights for {filename} from HF Hub ({repo_id})...")
    try:
        from huggingface_hub import hf_hub_download

        downloaded = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=str(WEIGHTS_DIR),
            local_dir_use_symlinks=False,
        )
        logger.info(f"Weights downloaded successfully to {downloaded}")
        return Path(downloaded)
    except Exception as e:
        logger.warning(
            f"Could not download weights from {repo_id}/{filename}: {e}. "
            f"Model will initialize with unweighted architecture for testing."
        )
        return local_path
