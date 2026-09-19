"""
Restormer model wrapper for image deblurring.

Restormer is an efficient Transformer-based model for image restoration.
Supports loading base pretrained weights and optional LoRA PEFT adapters.
"""
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageFilter
import numpy as np
from loguru import logger

from models.base import BaseRestorationModel
from models.utils import get_weights_path
from core.pipeline.image_utils import pil_to_numpy, numpy_to_pil


# ---------------------------------------------------------------------------
# Restormer Transformer Building Blocks
# ---------------------------------------------------------------------------
class MDTA(nn.Module):
    """Multi-Dconv Head Transposed Attention."""
    def __init__(self, dim: int = 48, num_heads: int = 4):
        super().__init__()
        self.num_heads = num_heads
        self.qkv = nn.Conv2d(dim, dim * 3, 1, bias=False)
        self.qkv_dw = nn.Conv2d(dim * 3, dim * 3, 3, padding=1, groups=dim * 3, bias=False)
        self.project_out = nn.Conv2d(dim, dim, 1, bias=False)
        self.scale = torch.nn.Parameter(torch.ones(num_heads, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        qkv = self.qkv_dw(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = q.view(b, self.num_heads, c // self.num_heads, h * w)
        k = k.view(b, self.num_heads, c // self.num_heads, h * w)
        v = v.view(b, self.num_heads, c // self.num_heads, h * w)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        out = (attn @ v).view(b, c, h, w)
        out = self.project_out(out)
        return out


class GDFN(nn.Module):
    """Gated-Dconv Feed-Forward Network."""
    def __init__(self, dim: int = 48, ffn_expansion_factor: float = 2.66):
        super().__init__()
        hidden_dim = int(dim * ffn_expansion_factor)
        self.project_in = nn.Conv2d(dim, hidden_dim * 2, 1, bias=False)
        self.dwconv = nn.Conv2d(hidden_dim * 2, hidden_dim * 2, 3, padding=1, groups=hidden_dim * 2, bias=False)
        self.project_out = nn.Conv2d(hidden_dim, dim, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = self.dwconv(self.project_in(x)).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, dim: int = 48, num_heads: int = 4):
        super().__init__()
        self.norm1 = nn.GroupNorm(1, dim)
        self.attn = MDTA(dim, num_heads)
        self.norm2 = nn.GroupNorm(1, dim)
        self.ffn = GDFN(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class RestormerArch(nn.Module):
    """Restormer Transformer architecture for image deblurring."""
    def __init__(self, in_nc: int = 3, out_nc: int = 3, dim: int = 48, num_blocks: list[int] | None = None):
        super().__init__()
        num_blocks = num_blocks or [2, 3, 3, 4]
        heads = [1, 2, 4, 8]

        self.patch_embed = nn.Conv2d(in_nc, dim, 3, padding=1, bias=False)

        # Encoder Level 1
        self.encoder_level1 = nn.Sequential(*[TransformerBlock(dim, heads[0]) for _ in range(num_blocks[0])])

        # Down 1 -> Level 2
        self.down1_2 = nn.Conv2d(dim, dim * 2, 2, stride=2, bias=False)
        self.encoder_level2 = nn.Sequential(*[TransformerBlock(dim * 2, heads[1]) for _ in range(num_blocks[1])])

        # Down 2 -> Level 3
        self.down2_3 = nn.Conv2d(dim * 2, dim * 4, 2, stride=2, bias=False)
        self.encoder_level3 = nn.Sequential(*[TransformerBlock(dim * 4, heads[2]) for _ in range(num_blocks[2])])

        # Decoder Level 3 -> Up 2
        self.up3_2 = nn.ConvTranspose2d(dim * 4, dim * 2, 2, stride=2, bias=False)
        self.reduce_chan2 = nn.Conv2d(dim * 4, dim * 2, 1, bias=False)
        self.decoder_level2 = nn.Sequential(*[TransformerBlock(dim * 2, heads[1]) for _ in range(num_blocks[1])])

        # Up 1 -> Level 1
        self.up2_1 = nn.ConvTranspose2d(dim * 2, dim, 2, stride=2, bias=False)
        self.reduce_chan1 = nn.Conv2d(dim * 2, dim, 1, bias=False)
        self.decoder_level1 = nn.Sequential(*[TransformerBlock(dim, heads[0]) for _ in range(num_blocks[0])])

        self.output = nn.Conv2d(dim, out_nc, 3, padding=1, bias=False)
        nn.init.zeros_(self.output.weight)

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        fo = self.patch_embed(inp)
        out_enc1 = self.encoder_level1(fo)

        inp_enc2 = self.down1_2(out_enc1)
        out_enc2 = self.encoder_level2(inp_enc2)

        inp_enc3 = self.down2_3(out_enc2)
        out_enc3 = self.encoder_level3(inp_enc3)

        inp_dec2 = self.up3_2(out_enc3)
        inp_dec2 = self.reduce_chan2(torch.cat([inp_dec2, out_enc2], dim=1))
        out_dec2 = self.decoder_level2(inp_dec2)

        inp_dec1 = self.up2_1(out_dec2)
        inp_dec1 = self.reduce_chan1(torch.cat([inp_dec1, out_enc1], dim=1))
        out_dec1 = self.decoder_level1(inp_dec1)

        out = self.output(out_dec1) + inp
        return out


import cv2


# ---------------------------------------------------------------------------
# Restormer Wrapper Class
# ---------------------------------------------------------------------------
class RestormerModel(BaseRestorationModel):
    """
    Restormer deblurring model wrapper.
    Supports base model weights and optional LoRA PEFT adapters.
    """

    def __init__(self, config: dict[str, Any] | None = None, device: str | None = None):
        super().__init__(config, device)
        self.has_weights: bool = False

    def load(self) -> None:
        self.model = RestormerArch(dim=48).to(self.device)
        self.model.eval()

        repo_id = self.config.get("weights_repo", "Ansh205/image-restoration-models")
        weights_path = self.config.get("weights_path", "restormer/restormer_deblurring.pth")

        local_weights = get_weights_path(repo_id, weights_path)
        if local_weights.exists():
            try:
                state_dict = torch.load(str(local_weights), map_location=self.device)
                self.model.load_state_dict(state_dict, strict=False)
                
                # Check if output weights are all zeros (dummy initial state dict)
                out_w = self.model.output.weight
                if torch.all(out_w == 0):
                    logger.warning("Restormer checkpoint has zero-initialized output weights (dummy initialized state dict). Fallback active.")
                    self.has_weights = False
                else:
                    self.has_weights = True
                    logger.info(f"Loaded Restormer pretrained weights from {local_weights}")
            except Exception as e:
                logger.warning(f"Failed to load Restormer checkpoint state dict: {e}")
                self.has_weights = False
        else:
            self.has_weights = False

        # Check if LoRA is requested
        if self.config.get("use_lora", False):
            lora_path = self.config.get("lora_path", "restormer-lora/")
            self._load_lora_adapter(repo_id, lora_path)

        self._loaded = True

    def _load_lora_adapter(self, repo_id: str, lora_path: str) -> None:
        """Load LoRA PEFT adapter weights if available."""
        try:
            from peft import PeftModel
            local_lora = get_weights_path(repo_id, f"{lora_path}adapter_model.bin")
            if local_lora.exists():
                self.model = PeftModel.from_pretrained(self.model, str(local_lora.parent))
                logger.info("Successfully attached LoRA adapter to Restormer")
        except Exception as e:
            logger.warning(f"Could not load LoRA adapter for Restormer: {e}")

    def restore(self, image: Image.Image) -> Image.Image:
        self.ensure_loaded()

        # Compute BEFORE stats
        np_before = pil_to_numpy(image) * 255.0  # [0, 255] float
        h, w, _ = np_before.shape

        gray_before = cv2.cvtColor(np_before.astype(np.uint8), cv2.COLOR_RGB2GRAY)
        lap_before = float(cv2.Laplacian(gray_before, cv2.CV_64F).var())
        sobelx_b = cv2.Sobel(gray_before, cv2.CV_64F, 1, 0, ksize=3)
        sobely_b = cv2.Sobel(gray_before, cv2.CV_64F, 0, 1, ksize=3)
        tenengrad_before = float(np.mean(sobelx_b**2 + sobely_b**2))

        logger.debug(
            f"[RESTORMER BEFORE] Range: [{np_before.min():.1f}, {np_before.max():.1f}] | "
            f"Mean: {np_before.mean():.2f}, Std: {np_before.std():.2f} | "
            f"Laplacian: {lap_before:.3f}, Tenengrad: {tenengrad_before:.3f}"
        )

        if not self.has_weights:
            logger.info("Restormer using high-pass sharpening filter fallback for deblurring")
            # Unsharp mask high-pass sharpening to recover edge contrast
            blurred = image.filter(ImageFilter.GaussianBlur(radius=1.5))
            np_orig = np.array(image, dtype=np.float32)
            np_blur = np.array(blurred, dtype=np.float32)
            np_sharp = np.clip(np_orig + 1.2 * (np_orig - np_blur), 0, 255).astype(np.uint8)
            restored_img = Image.fromarray(np_sharp)
        else:
            np_img = pil_to_numpy(image)
            # Pad for 2-level downsampling (multiple of 4)
            pad_h = (4 - h % 4) % 4
            pad_w = (4 - w % 4) % 4
            if pad_h > 0 or pad_w > 0:
                np_img = np.pad(np_img, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")

            tensor_img = torch.from_numpy(np_img).permute(2, 0, 1).unsqueeze(0).to(self.device)

            with torch.no_grad():
                output_tensor = self.model(tensor_img)
                output_tensor = torch.clamp(output_tensor, 0.0, 1.0)

            output_np = output_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
            if pad_h > 0 or pad_w > 0:
                output_np = output_np[:h, :w, :]

            restored_img = numpy_to_pil(output_np)

        # Compute AFTER stats
        np_after = pil_to_numpy(restored_img) * 255.0
        gray_after = cv2.cvtColor(np_after.astype(np.uint8), cv2.COLOR_RGB2GRAY)
        lap_after = float(cv2.Laplacian(gray_after, cv2.CV_64F).var())
        sobelx_a = cv2.Sobel(gray_after, cv2.CV_64F, 1, 0, ksize=3)
        sobely_a = cv2.Sobel(gray_after, cv2.CV_64F, 0, 1, ksize=3)
        tenengrad_after = float(np.mean(sobelx_a**2 + sobely_a**2))

        diff = np.abs(np_after - np_before)
        mad = float(np.mean(diff))
        max_diff = float(np.max(diff))
        changed_pct = float(np.mean(diff > 1.0) * 100.0)

        lap_change_pct = ((lap_after - lap_before) / max(lap_before, 1e-5)) * 100.0
        tenengrad_change_pct = ((tenengrad_after - tenengrad_before) / max(tenengrad_before, 1e-5)) * 100.0

        logger.info(
            f"[RESTORMER AFTER] Range: [{np_after.min():.1f}, {np_after.max():.1f}] | "
            f"Mean: {np_after.mean():.2f}, Std: {np_after.std():.2f} | "
            f"Laplacian: {lap_before:.2f} -> {lap_after:.2f} ({lap_change_pct:+.2f}%) | "
            f"Tenengrad: {tenengrad_before:.2f} -> {tenengrad_after:.2f} ({tenengrad_change_pct:+.2f}%) | "
            f"MAD: {mad:.3f}, MaxDiff: {max_diff:.1f}, ChangedPixels: {changed_pct:.2f}%"
        )

        return restored_img
