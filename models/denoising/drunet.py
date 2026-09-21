"""
DRUNet model wrapper for image denoising.

DRUNet (Deep Residual UNet) accepts an RGB image along with a noise level map
and outputs a clean, denoised image.
"""
from typing import Any

import torch
import torch.nn as nn
from PIL import Image
import numpy as np
from loguru import logger

from models.base import BaseRestorationModel
from models.utils import get_weights_path
from core.pipeline.image_utils import pil_to_numpy, numpy_to_pil


# ---------------------------------------------------------------------------
# Lightweight PyTorch DRUNet Architecture
# ---------------------------------------------------------------------------
class ResidualBlock(nn.Module):
    def __init__(self, channels: int = 64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.conv(x)


class DRUNetArch(nn.Module):
    """UNet architecture with residual blocks for image denoising."""
    def __init__(self, in_nc: int = 4, out_nc: int = 3, nc: list[int] | None = None):
        super().__init__()
        nc = nc or [64, 128, 256, 512]
        # Input: 3 RGB channels + 1 noise map channel = 4 channels
        self.head = nn.Conv2d(in_nc, nc[0], 3, padding=1)

        # Encoder
        self.enc1 = nn.Sequential(ResidualBlock(nc[0]), ResidualBlock(nc[0]))
        self.down1 = nn.Conv2d(nc[0], nc[1], 2, stride=2)
        self.enc2 = nn.Sequential(ResidualBlock(nc[1]), ResidualBlock(nc[1]))
        self.down2 = nn.Conv2d(nc[1], nc[2], 2, stride=2)
        self.enc3 = nn.Sequential(ResidualBlock(nc[2]), ResidualBlock(nc[2]))

        # Decoder
        self.up2 = nn.ConvTranspose2d(nc[2], nc[1], 2, stride=2)
        self.dec2 = nn.Sequential(ResidualBlock(nc[1]), ResidualBlock(nc[1]))
        self.up1 = nn.ConvTranspose2d(nc[1], nc[0], 2, stride=2)
        self.dec1 = nn.Sequential(ResidualBlock(nc[0]), ResidualBlock(nc[0]))

        # Output
        self.tail = nn.Conv2d(nc[0], out_nc, 3, padding=1)
        nn.init.zeros_(self.tail.weight)
        if self.tail.bias is not None:
            nn.init.zeros_(self.tail.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.head(x)
        e1 = self.enc1(h)
        d1 = self.down1(e1)
        e2 = self.enc2(d1)
        d2 = self.down2(e2)
        e3 = self.enc3(d2)

        u2 = self.up2(e3)
        d2 = self.dec2(u2 + e2)
        u1 = self.up1(d2)
        d1 = self.dec1(u1 + e1)

        res = self.tail(d1)
        out = x[:, :3, :, :] + res
        return out


from models.denoising.scunet import SCUNetModel

# Expose DRUNetModel as alias of SCUNetModel for backward compatibility
class DRUNetModel(SCUNetModel):
    """
    DRUNetModel wrapper alias.
    Delegates to SCUNetModel for blind real-world denoising with Real-PSNR checkpoint.
    """
    pass

