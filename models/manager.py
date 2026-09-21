"""Lazy, cached access to configured restoration models."""

from __future__ import annotations

from typing import Any

from app.config import get_device, load_config
from models.base import BaseRestorationModel
from models.factory import get_model

_CONFIG_KEY = {
    "denoise": "denoising",
    "deblur": "deblurring",
    "super_resolution": "super_resolution",
    "low_light": "low_light",
    "jpeg": "jpeg_artifacts",
}


class ModelManager:
    """Creates each model once and keeps it available for later requests."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or load_config()
        self.device = get_device(self.config)
        self._models: dict[str, BaseRestorationModel] = {}

    def get(self, operation: str) -> BaseRestorationModel:
        """Return a cached, lazily loaded model for one pipeline operation."""
        if operation not in _CONFIG_KEY:
            raise ValueError(f"Unsupported restoration operation: {operation}")
        if operation not in self._models:
            model_config = self.config.get("models", {}).get(_CONFIG_KEY[operation], {})
            self._models[operation] = get_model(operation, config=model_config, device=self.device)
        return self._models[operation]

    def unload_all(self) -> None:
        """Release cached model memory."""
        for model in self._models.values():
            model.unload()
        self._models.clear()
