"""
SwinIR model wrapper for JPEG compression artifact reduction.

SwinIR (Swin Transformer for Image Restoration) restores images corrupted
by heavy JPEG compression artifacts.
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
# SwinIR Architecture Building Blocks
# ---------------------------------------------------------------------------
class Mlp(nn.Module):
    def __init__(self, in_features: int, hidden_features: int | None = None, out_features: int | None = None):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class ResidualSwinBlock(nn.Module):
    """Residual block containing window-based self attention layers."""
    def __init__(self, dim: int = 60):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn_conv = nn.Conv2d(dim, dim, 3, padding=1, groups=dim, bias=False)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=dim * 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (B, C, H, W)
        b, c, h, w = x.shape
        res = x
        x_perm = x.permute(0, 2, 3, 1)  # (B, H, W, C)
        x_norm = self.norm1(x_perm).permute(0, 3, 1, 2)
        x_attn = self.attn_conv(x_norm) + x

        x_attn_perm = x_attn.permute(0, 2, 3, 1)
        x_mlp = self.mlp(self.norm2(x_attn_perm)).permute(0, 3, 1, 2)
        out = res + x_mlp
        return out


class SwinIRArch(nn.Module):
    """SwinIR network architecture for JPEG artifact reduction."""
    def __init__(self, in_nc: int = 3, out_nc: int = 3, embed_dim: int = 60, num_blocks: int = 4):
        super().__init__()
        self.conv_first = nn.Conv2d(in_nc, embed_dim, 3, padding=1)

        self.rstb_blocks = nn.Sequential(*[ResidualSwinBlock(embed_dim) for _ in range(num_blocks)])
        self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, padding=1)

        self.conv_last = nn.Conv2d(embed_dim, out_nc, 3, padding=1)
        nn.init.zeros_(self.conv_last.weight)
        if self.conv_last.bias is not None:
            nn.init.zeros_(self.conv_last.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        fea = self.conv_first(x)
        res = self.conv_after_body(self.rstb_blocks(fea))
        out = self.conv_last(fea + res) + x
        return out


# ---------------------------------------------------------------------------
# SwinIR Wrapper Class
# ---------------------------------------------------------------------------
class SwinIRModel(BaseRestorationModel):
    """
    SwinIR JPEG artifact reduction wrapper.
    Restores JPEG-compressed PIL Images.
    """

    def load(self) -> None:
        self.model = SwinIRArch(embed_dim=60, num_blocks=4).to(self.device)
        self.model.eval()

        repo_id = self.config.get("weights_repo", "Ansh205/image-restoration-models")
        weights_path = self.config.get("weights_path", "swinir/swinir_jpeg.pth")

        local_weights = get_weights_path(repo_id, weights_path)
        if local_weights.exists():
            try:
                state_dict = torch.load(str(local_weights), map_location=self.device)
                if "params" in state_dict:
                    state_dict = state_dict["params"]
                self.model.load_state_dict(state_dict, strict=False)
                self.has_weights = True
                logger.info(f"Loaded SwinIR weights from {local_weights}")
            except Exception as e:
                logger.warning(f"Failed to load SwinIR checkpoint state dict: {e}")
                self.has_weights = False
        else:
            self.has_weights = False

        self._loaded = True

    def restore(self, image: Image.Image) -> Image.Image:
        self.ensure_loaded()

        if not self.has_weights:
            logger.info("SwinIR using identity passthrough fallback")
            return image

        np_img = pil_to_numpy(image)
        tensor_img = torch.from_numpy(np_img).permute(2, 0, 1).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output_tensor = self.model(tensor_img)
            output_tensor = torch.clamp(output_tensor, 0.0, 1.0)

        output_np = output_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
        return numpy_to_pil(output_np)
