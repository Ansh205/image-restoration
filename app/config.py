"""
Configuration loader.
Reads configs/default.yaml and provides project-wide settings.
"""
from pathlib import Path
from typing import Any

import yaml


CONFIG_DIR = Path(__file__).resolve().parent.parent / "configs"
DEFAULT_CONFIG = CONFIG_DIR / "default.yaml"


def load_config(config_path: Path | None = None) -> dict[str, Any]:
    """
    Load YAML configuration.

    Args:
        config_path: Optional path to a custom config file.
                     Falls back to configs/default.yaml.

    Returns:
        Dictionary of configuration values.
    """
    path = config_path or DEFAULT_CONFIG

    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return config


def get_device(config: dict[str, Any]) -> str:
    """
    Resolve the device string from config.

    Returns:
        "cuda" if available and configured, else "cpu".
    """
    import torch

    device_setting = config.get("device", "auto")

    if device_setting == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_setting
