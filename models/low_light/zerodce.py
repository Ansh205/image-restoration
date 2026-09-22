"""
Zero-DCE++: Zero-Reference Deep Curve Estimation for Low-Light Image Enhancement.
Paper: https://arxiv.org/abs/2103.00860

[LEGACY / DISABLED] - Superceded as active low-light model by Retinexformer (models/low_light/retinexformer.py).
Retained for backwards compatibility and easy switching.
"""
from typing import Any

import torch
import torch.nn as nn
from PIL import Image, ImageEnhance
from loguru import logger

from models.base import BaseRestorationModel
from models.utils import get_weights_path
from core.pipeline.image_utils import pil_to_numpy, numpy_to_pil


# ---------------------------------------------------------------------------
# Zero-DCE++ Light Curve Estimation Architecture
# ---------------------------------------------------------------------------
class CNet(nn.Module):
    """Depthwise separable convolutional block for Zero-DCE++."""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.depth_conv = nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1, groups=in_ch, bias=False)
        self.point_conv = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.depth_conv(x)
        out = self.relu(self.point_conv(out))
        return out


class ZeroDCEPlusPlusArch(nn.Module):
    """Zero-DCE++ curve estimation network."""
    def __init__(self, number_f: int = 32, iteration: int = 8):
        super().__init__()
        self.iteration = iteration
        self.e_conv1 = CNet(3, number_f)
        self.e_conv2 = CNet(number_f, number_f)
        self.e_conv3 = CNet(number_f, number_f)
        self.e_conv4 = CNet(number_f, number_f)
        self.e_conv5 = CNet(number_f * 2, number_f)
        self.e_conv6 = CNet(number_f * 2, number_f)
        self.e_conv7 = nn.Conv2d(number_f * 2, 3, kernel_size=3, padding=1, bias=True)
        nn.init.zeros_(self.e_conv7.weight)
        if self.e_conv7.bias is not None:
            nn.init.zeros_(self.e_conv7.bias)

    def enhance(self, x: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        """Apply curve enhancement iteratively: LE(x) = x + A*x*(1-x)."""
        r = x
        for i in range(self.iteration):
            r = r + A * (torch.pow(r, 2) - r)
        return r

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x1 = self.e_conv1(x)
        x2 = self.e_conv2(x1)
        x3 = self.e_conv3(x2)
        x4 = self.e_conv4(x3)

        x5 = self.e_conv5(torch.cat([x3, x4], dim=1))
        x6 = self.e_conv6(torch.cat([x2, x5], dim=1))
        A = torch.tanh(self.e_conv7(torch.cat([x1, x6], dim=1)))

        enhanced = self.enhance(x, A)
        return enhanced, A


# ---------------------------------------------------------------------------
# Zero-DCE++ Wrapper Class
# ---------------------------------------------------------------------------
class ZeroDCEModel(BaseRestorationModel):
    """
    Zero-DCE++ low-light enhancement model wrapper.
    Estimates enhancement curves to brighten low-light PIL Images.
    """

    def __init__(self, config: dict[str, Any] | None = None, device: str | None = None):
        super().__init__(config, device)
        self.has_weights: bool = False

    def load(self) -> None:
        self.model = ZeroDCEPlusPlusArch(number_f=32, iteration=8).to(self.device)
        self.model.eval()

        repo_id = self.config.get("weights_repo", "Ansh205/image-restoration-models")
        weights_path = self.config.get("weights_path", "zerodce/zero_dce_pp.pth")

        local_weights = get_weights_path(repo_id, weights_path)
        if local_weights.exists():
            try:
                state_dict = torch.load(str(local_weights), map_location=self.device)
                self.model.load_state_dict(state_dict, strict=False)
                self.has_weights = True
                logger.info(f"Loaded Zero-DCE++ weights from {local_weights}")
            except Exception as e:
                logger.warning(f"Failed to load Zero-DCE++ checkpoint state dict: {e}")
                self.has_weights = False
        else:
            self.has_weights = False

        self._loaded = True

    def restore(self, image: Image.Image) -> Image.Image:
        self.ensure_loaded()

        if not self.has_weights:
            logger.info("Zero-DCE++ using PIL Brightness enhancement fallback")
            enhancer = ImageEnhance.Brightness(image)
            return enhancer.enhance(1.25)

        np_img = pil_to_numpy(image)
        tensor_img = torch.from_numpy(np_img).permute(2, 0, 1).unsqueeze(0).to(self.device)

        with torch.no_grad():
            enhanced_tensor, _ = self.model(tensor_img)
            enhanced_tensor = torch.clamp(enhanced_tensor, 0.0, 1.0)

        output_np = enhanced_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
        return numpy_to_pil(output_np)
