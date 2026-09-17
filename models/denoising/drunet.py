"""
DRUNet model wrapper for image denoising.

DRUNet (Deep Residual UNet) accepts an RGB image along with a noise level map
and outputs a clean, denoised image.
"""
from typing import Any

import torch
import torch.nn as nn
from PIL import Image
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

        out = self.tail(d1)
        return out


# ---------------------------------------------------------------------------
# DRUNet Wrapper Class
# ---------------------------------------------------------------------------
class DRUNetModel(BaseRestorationModel):
    """
    DRUNet denoising wrapper.
    Converts PIL Image → Tensor + Noise Map → DRUNet → Denoised PIL Image.
    """

    def __init__(self, config: dict[str, Any] | None = None, device: str | None = None):
        super().__init__(config, device)
        self.noise_level: float = self.config.get("noise_level", 15.0)

    def load(self) -> None:
        self.model = DRUNetArch(in_nc=4, out_nc=3).to(self.device)
        self.model.eval()

        repo_id = self.config.get("weights_repo", "Ansh205/image-restoration-models")
        weights_path = self.config.get("weights_path", "drunet/drunet_color.pth")

        local_weights = get_weights_path(repo_id, weights_path)
        if local_weights.exists():
            try:
                state_dict = torch.load(str(local_weights), map_location=self.device)
                self.model.load_state_dict(state_dict, strict=False)
                logger.info(f"Loaded DRUNet weights from {local_weights}")
            except Exception as e:
                logger.warning(f"Failed to load DRUNet checkpoint state dict: {e}")

        self._loaded = True

    def restore(self, image: Image.Image) -> Image.Image:
        self.ensure_loaded()

        np_img = pil_to_numpy(image)  # (H, W, 3) float32 [0, 1]
        h, w, _ = np_img.shape

        # Pad dimensions to multiples of 4 for UNet downsampling
        pad_h = (4 - h % 4) % 4
        pad_w = (4 - w % 4) % 4
        if pad_h > 0 or pad_w > 0:
            np_img = np.pad(np_img, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")

        tensor_img = torch.from_numpy(np_img).permute(2, 0, 1).unsqueeze(0).to(self.device)

        # Create noise level map (normalized)
        sigma = self.noise_level / 255.0
        noise_map = torch.full((1, 1, tensor_img.shape[2], tensor_img.shape[3]), sigma, device=self.device)

        # Concatenate image + noise map -> (1, 4, H, W)
        input_tensor = torch.cat([tensor_img, noise_map], dim=1)

        with torch.no_grad():
            output_tensor = self.model(input_tensor)
            # Residual learning: output = input - residual noise (or direct output)
            output_tensor = torch.clamp(output_tensor, 0.0, 1.0)

        output_np = output_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()

        # Unpad
        if pad_h > 0 or pad_w > 0:
            output_np = output_np[:h, :w, :]

        return numpy_to_pil(output_np)
