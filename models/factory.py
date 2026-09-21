"""
Model Factory and Registry.

Provides a unified interface to instantiate and manage any of the 5 restoration models
by operation type or model name.
"""
from typing import Any, Type

from loguru import logger

from models.base import BaseRestorationModel
from models.denoising.drunet import DRUNetModel
from models.denoising.scunet import SCUNetModel
from models.deblurring.restormer import RestormerModel
from models.super_resolution.realesrgan import RealESRGANModel
from models.low_light.zerodce import ZeroDCEModel
from models.jpeg.swinir import SwinIRModel


# Registry mapping operation key -> Model Class
MODEL_REGISTRY: dict[str, Type[BaseRestorationModel]] = {
    "denoise": SCUNetModel,
    "deblur": RestormerModel,
    "super_resolution": RealESRGANModel,
    "low_light": ZeroDCEModel,
    "jpeg": SwinIRModel,
    "jpeg_artifacts": SwinIRModel,
}

# Registry mapping model name -> Model Class
NAME_REGISTRY: dict[str, Type[BaseRestorationModel]] = {
    "scunet": SCUNetModel,
    "scunetmodel": SCUNetModel,
    "drunet": DRUNetModel,
    "restormer": RestormerModel,
    "real-esrgan": RealESRGANModel,
    "realesrgan": RealESRGANModel,
    "zero-dce++": ZeroDCEModel,
    "zerodce": ZeroDCEModel,
    "swinir": SwinIRModel,
}


def get_model(
    key_or_name: str,
    config: dict[str, Any] | None = None,
    device: str | None = None,
) -> BaseRestorationModel:
    """
    Factory function to instantiate a restoration model.

    Args:
        key_or_name: Operation key (e.g., 'denoise') or model name (e.g., 'drunet').
        config: Optional model-specific configuration.
        device: Force specific device ('cuda' or 'cpu').

    Returns:
        An instance of a subclass of BaseRestorationModel.
    """
    key_lower = key_or_name.lower().strip()

    cls = MODEL_REGISTRY.get(key_lower) or NAME_REGISTRY.get(key_lower)

    if cls is None:
        valid_keys = sorted(set(list(MODEL_REGISTRY.keys()) + list(NAME_REGISTRY.keys())))
        raise ValueError(f"Unknown model key or name '{key_or_name}'. Valid options: {valid_keys}")

    logger.debug(f"Factory creating model '{cls.__name__}' for key '{key_or_name}'")
    return cls(config=config, device=device)
