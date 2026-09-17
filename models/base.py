"""
Abstract base class for all restoration models.

Every model wrapper (DRUNet, Restormer, Real-ESRGAN, etc.)
must inherit from BaseRestorationModel and implement load() and restore().
"""
from abc import ABC, abstractmethod
from typing import Any

import torch
from PIL import Image
from loguru import logger


class BaseRestorationModel(ABC):
    """
    Base class that every restoration model wrapper must extend.

    Provides:
        - Automatic device selection (CUDA / CPU)
        - Lazy loading pattern (call load() before first inference)
        - Common interface for the pipeline to call
    """

    def __init__(self, config: dict[str, Any] | None = None, device: str | None = None):
        """
        Args:
            config: Model-specific configuration dict.
            device: Force a specific device. If None, auto-detect.
        """
        self.config = config or {}
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self._loaded = False

    @property
    def name(self) -> str:
        """Human-readable model name."""
        return self.__class__.__name__

    @abstractmethod
    def load(self) -> None:
        """
        Load model weights into memory.
        Must set self.model and self._loaded = True.
        """
        pass

    @abstractmethod
    def restore(self, image: Image.Image) -> Image.Image:
        """
        Run restoration on a PIL Image.

        Args:
            image: Input PIL Image (RGB).

        Returns:
            Restored PIL Image (RGB).
        """
        pass

    def ensure_loaded(self) -> None:
        """Load the model if not already loaded."""
        if not self._loaded:
            logger.info(f"Loading model: {self.name} on {self.device}")
            self.load()
            self._loaded = True
            logger.info(f"Model {self.name} loaded successfully")

    def unload(self) -> None:
        """Release model from memory."""
        if self.model is not None:
            del self.model
            self.model = None
            self._loaded = False
            if self.device == "cuda":
                torch.cuda.empty_cache()
            logger.info(f"Model {self.name} unloaded")
