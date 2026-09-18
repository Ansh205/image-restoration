"""
Real-ESRGAN model wrapper for 4x super-resolution.

Real-ESRGAN enhances low-resolution images by upscaling them 4x using
Residual-in-Residual Dense Blocks (RRDB).
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
# Real-ESRGAN RRDB Architecture
# ---------------------------------------------------------------------------
class DenseBlock(nn.Module):
    def __init__(self, nf: int = 64, gc: int = 32):
        super().__init__()
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1)
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1)
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1)
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class RRDB(nn.Module):
    """Residual in Residual Dense Block."""
    def __init__(self, nf: int = 64, gc: int = 32):
        super().__init__()
        self.rdb1 = DenseBlock(nf, gc)
        self.rdb2 = DenseBlock(nf, gc)
        self.rdb3 = DenseBlock(nf, gc)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        return out * 0.2 + x


class RealESRGANArch(nn.Module):
    """Real-ESRGAN Net for 4x Super-Resolution."""
    def __init__(self, in_nc: int = 3, out_nc: int = 3, nf: int = 64, nb: int = 6, gc: int = 32, scale: int = 4):
        super().__init__()
        self.scale = scale
        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1)
        self.rrdb_blocks = nn.Sequential(*[RRDB(nf, gc) for _ in range(nb)])
        self.conv_body = nn.Conv2d(nf, nf, 3, 1, 1)

        # Upsampling
        self.conv_up1 = nn.Conv2d(nf, nf, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(nf, nf, 3, 1, 1)
        self.conv_hr = nn.Conv2d(nf, nf, 3, 1, 1)
        self.conv_last = nn.Conv2d(nf, out_nc, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        fea = self.conv_first(x)
        trunk = self.conv_body(self.rrdb_blocks(fea))
        fea = fea + trunk

        # 2x upsampling
        fea = self.lrelu(self.conv_up1(torch.nn.functional.interpolate(fea, scale_factor=2, mode="nearest")))
        # 4x upsampling
        fea = self.lrelu(self.conv_up2(torch.nn.functional.interpolate(fea, scale_factor=2, mode="nearest")))

        out = self.conv_last(self.lrelu(self.conv_hr(fea)))
        return out


# ---------------------------------------------------------------------------
# Real-ESRGAN Wrapper Class
# ---------------------------------------------------------------------------
class RealESRGANModel(BaseRestorationModel):
    """
    Real-ESRGAN 4x super-resolution wrapper.
    Upscales PIL Image 4x.
    """

    def __init__(self, config: dict[str, Any] | None = None, device: str | None = None):
        super().__init__(config, device)
        self.has_weights: bool = False

    def load(self) -> None:
        self.model = RealESRGANArch(scale=4, nb=6).to(self.device)
        self.model.eval()

        repo_id = self.config.get("weights_repo", "Ansh205/image-restoration-models")
        weights_path = self.config.get("weights_path", "realesrgan/RealESRGAN_x4plus.pth")

        local_weights = get_weights_path(repo_id, weights_path)
        if local_weights.exists():
            try:
                state_dict = torch.load(str(local_weights), map_location=self.device)
                if "params_strict" in state_dict:
                    state_dict = state_dict["params_strict"]
                self.model.load_state_dict(state_dict, strict=False)
                self.has_weights = True
                logger.info(f"Loaded Real-ESRGAN weights from {local_weights}")
            except Exception as e:
                logger.warning(f"Failed to load Real-ESRGAN checkpoint state dict: {e}")
                self.has_weights = False
        else:
            self.has_weights = False

        self._loaded = True

    def restore(self, image: Image.Image) -> Image.Image:
        self.ensure_loaded()

        if not self.has_weights:
            logger.info("RealESRGAN using high-quality Lanczos 4x upscaling fallback")
            target_w = image.width * 4
            target_h = image.height * 4
            return image.resize((target_w, target_h), Image.Resampling.LANCZOS)

        np_img = pil_to_numpy(image)
        tensor_img = torch.from_numpy(np_img).permute(2, 0, 1).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output_tensor = self.model(tensor_img)
            output_tensor = torch.clamp(output_tensor, 0.0, 1.0)

        output_np = output_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
        return numpy_to_pil(output_np)
