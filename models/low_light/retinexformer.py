"""
Retinexformer: One-stage Retinex-based Transformer for Low-light Image Enhancement.

Official Reference: https://github.com/caiyuanhao1998/Retinexformer
Pretrained Checkpoint: LOL_v2_real.pth
"""

import math
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from pathlib import Path
from typing import Dict, Any, List, Optional
from loguru import logger

from models.base import BaseRestorationModel
from models.utils import get_weights_path


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    def norm_cdf(x):
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn(
            "mean is more than 2 std from [a, b] in nn.init.trunc_normal_.",
            stacklevel=2,
        )
    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.0))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0):
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


class PreNorm(nn.Module):
    def __init__(self, dim: int, fn: nn.Module):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        x = self.norm(x)
        return self.fn(x, *args, **kwargs)


class GELU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(x)


class Illumination_Estimator(nn.Module):
    def __init__(self, n_fea_middle: int, n_fea_in: int = 4, n_fea_out: int = 3):
        super(Illumination_Estimator, self).__init__()
        self.conv1 = nn.Conv2d(n_fea_in, n_fea_middle, kernel_size=1, bias=True)
        self.depth_conv = nn.Conv2d(
            n_fea_middle, n_fea_middle, kernel_size=5, padding=2, bias=True, groups=n_fea_in
        )
        self.conv2 = nn.Conv2d(n_fea_middle, n_fea_out, kernel_size=1, bias=True)

    def forward(self, img: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean_c = img.mean(dim=1).unsqueeze(1)
        input_tensor = torch.cat([img, mean_c], dim=1)
        x_1 = self.conv1(input_tensor)
        illu_fea = self.depth_conv(x_1)
        illu_map = self.conv2(illu_fea)
        return illu_fea, illu_map


class IG_MSA(nn.Module):
    def __init__(self, dim: int, dim_head: int = 64, heads: int = 8):
        super().__init__()
        self.num_heads = heads
        self.dim_head = dim_head
        self.to_q = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_k = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_v = nn.Linear(dim, dim_head * heads, bias=False)
        self.rescale = nn.Parameter(torch.ones(heads, 1, 1))
        self.proj = nn.Linear(dim_head * heads, dim, bias=True)
        self.pos_emb = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
            GELU(),
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
        )
        self.dim = dim

    def forward(self, x_in: torch.Tensor, illu_fea_trans: torch.Tensor) -> torch.Tensor:
        b, h, w, c = x_in.shape
        x = x_in.reshape(b, h * w, c)
        q_inp = self.to_q(x)
        k_inp = self.to_k(x)
        v_inp = self.to_v(x)
        illu_attn = illu_fea_trans
        q, k, v, illu_attn = map(
            lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.num_heads),
            (q_inp, k_inp, v_inp, illu_attn.flatten(1, 2)),
        )
        v = v * illu_attn
        q = q.transpose(-2, -1)
        k = k.transpose(-2, -1)
        v = v.transpose(-2, -1)
        q = F.normalize(q, dim=-1, p=2)
        k = F.normalize(k, dim=-1, p=2)
        attn = k @ q.transpose(-2, -1)
        attn = attn * self.rescale
        attn = attn.softmax(dim=-1)
        x = attn @ v
        x = x.permute(0, 3, 1, 2)
        x = x.reshape(b, h * w, self.num_heads * self.dim_head)
        out_c = self.proj(x).view(b, h, w, c)
        out_p = self.pos_emb(v_inp.reshape(b, h, w, c).permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        return out_c + out_p


class FeedForward(nn.Module):
    def __init__(self, dim: int, mult: int = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, dim * mult, 1, 1, bias=False),
            GELU(),
            nn.Conv2d(dim * mult, dim * mult, 3, 1, 1, bias=False, groups=dim * mult),
            GELU(),
            nn.Conv2d(dim * mult, dim, 1, 1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x.permute(0, 3, 1, 2).contiguous())
        return out.permute(0, 2, 3, 1)


class IGAB(nn.Module):
    def __init__(self, dim: int, dim_head: int = 64, heads: int = 8, num_blocks: int = 2):
        super().__init__()
        self.blocks = nn.ModuleList([])
        for _ in range(num_blocks):
            self.blocks.append(
                nn.ModuleList(
                    [
                        IG_MSA(dim=dim, dim_head=dim_head, heads=heads),
                        PreNorm(dim, FeedForward(dim=dim)),
                    ]
                )
            )

    def forward(self, x: torch.Tensor, illu_fea: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 3, 1)
        for attn, ff in self.blocks:
            x = attn(x, illu_fea_trans=illu_fea.permute(0, 2, 3, 1)) + x
            x = ff(x) + x
        return x.permute(0, 3, 1, 2)


class Denoiser(nn.Module):
    def __init__(self, in_dim: int = 3, out_dim: int = 3, dim: int = 31, level: int = 2, num_blocks: List[int] = [2, 4, 4]):
        super(Denoiser, self).__init__()
        self.dim = dim
        self.level = level

        self.embedding = nn.Conv2d(in_dim, self.dim, 3, 1, 1, bias=False)

        self.encoder_layers = nn.ModuleList([])
        dim_level = dim
        for i in range(level):
            self.encoder_layers.append(
                nn.ModuleList(
                    [
                        IGAB(dim=dim_level, num_blocks=num_blocks[i], dim_head=dim, heads=dim_level // dim),
                        nn.Conv2d(dim_level, dim_level * 2, 4, 2, 1, bias=False),
                        nn.Conv2d(dim_level, dim_level * 2, 4, 2, 1, bias=False),
                    ]
                )
            )
            dim_level *= 2

        self.bottleneck = IGAB(dim=dim_level, dim_head=dim, heads=dim_level // dim, num_blocks=num_blocks[-1])

        self.decoder_layers = nn.ModuleList([])
        for i in range(level):
            self.decoder_layers.append(
                nn.ModuleList(
                    [
                        nn.ConvTranspose2d(dim_level, dim_level // 2, stride=2, kernel_size=2, padding=0, output_padding=0),
                        nn.Conv2d(dim_level, dim_level // 2, 1, 1, bias=False),
                        IGAB(dim=dim_level // 2, num_blocks=num_blocks[level - 1 - i], dim_head=dim, heads=(dim_level // 2) // dim),
                    ]
                )
            )
            dim_level //= 2

        self.mapping = nn.Conv2d(self.dim, out_dim, 3, 1, 1, bias=False)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x: torch.Tensor, illu_fea: torch.Tensor) -> torch.Tensor:
        fea = self.embedding(x)
        fea_encoder = []
        illu_fea_list = []
        for IGAB_layer, FeaDownSample, IlluFeaDownsample in self.encoder_layers:
            fea = IGAB_layer(fea, illu_fea)
            illu_fea_list.append(illu_fea)
            fea_encoder.append(fea)
            fea = FeaDownSample(fea)
            illu_fea = IlluFeaDownsample(illu_fea)

        fea = self.bottleneck(fea, illu_fea)

        for i, (FeaUpSample, Fution, LeWinBlcok) in enumerate(self.decoder_layers):
            fea = FeaUpSample(fea)
            fea = Fution(torch.cat([fea, fea_encoder[self.level - 1 - i]], dim=1))
            illu_fea = illu_fea_list[self.level - 1 - i]
            fea = LeWinBlcok(fea, illu_fea)

        return self.mapping(fea) + x


class RetinexFormer_Single_Stage(nn.Module):
    def __init__(self, in_channels: int = 3, out_channels: int = 3, n_feat: int = 40, level: int = 2, num_blocks: List[int] = [1, 2, 2]):
        super(RetinexFormer_Single_Stage, self).__init__()
        self.estimator = Illumination_Estimator(n_feat)
        self.denoiser = Denoiser(in_dim=in_channels, out_dim=out_channels, dim=n_feat, level=level, num_blocks=num_blocks)

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        illu_fea, illu_map = self.estimator(img)
        input_img = img * illu_map + img
        return self.denoiser(input_img, illu_fea)


class RetinexFormer(nn.Module):
    def __init__(self, in_channels: int = 3, out_channels: int = 3, n_feat: int = 40, stage: int = 1, num_blocks: List[int] = [1, 2, 2]):
        super(RetinexFormer, self).__init__()
        self.stage = stage
        modules_body = [
            RetinexFormer_Single_Stage(in_channels=in_channels, out_channels=out_channels, n_feat=n_feat, level=2, num_blocks=num_blocks)
            for _ in range(stage)
        ]
        self.body = nn.Sequential(*modules_body)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


class RetinexformerModel(BaseRestorationModel):
    """
    Wrapper for official Retinexformer Low-Light Enhancement Model (`LOL_v2_real.pth`).
    """

    def __init__(self, config: dict[str, Any] | None = None, device: str | None = None):
        super().__init__(config=config, device=device)
        self.has_weights: bool = False

    def load(self) -> None:
        device_obj = torch.device(self.device)

        repo_id = self.config.get("weights_repo", "Ansh205/image-restoration-models")
        weights_path = self.config.get("weights_path", "retinexformer/LOL_v2_real.pth")

        local_weights = get_weights_path(repo_id, weights_path)
        if not local_weights.exists():
            alt_path = Path("weights/retinexformer/LOL_v2_real.pth")
            if alt_path.exists():
                local_weights = alt_path

        logger.info(f"Loading official Retinexformer model architecture...")

        # Instantiate RetinexFormer with LOL_v2_real parameters (n_feat=40, stage=1, num_blocks=[1,2,2])
        model = RetinexFormer(in_channels=3, out_channels=3, n_feat=40, stage=1, num_blocks=[1, 2, 2]).to(device_obj)

        if not local_weights.exists():
            logger.warning(
                f"Retinexformer checkpoint not found at: {local_weights.resolve()}\n"
                f"Model initialized with unweighted fallback for testing."
            )
            self.has_weights = False
        else:
            try:
                logger.info(f"Loading Retinexformer weights from {local_weights}")
                checkpoint = torch.load(local_weights, map_location=device_obj)
                if isinstance(checkpoint, dict):
                    if "params" in checkpoint:
                        state_dict = checkpoint["params"]
                    elif "params_ema" in checkpoint:
                        state_dict = checkpoint["params_ema"]
                    elif "state_dict" in checkpoint:
                        state_dict = checkpoint["state_dict"]
                    else:
                        state_dict = checkpoint
                else:
                    state_dict = checkpoint

                new_state_dict = {}
                for k, v in state_dict.items():
                    name = k[7:] if k.startswith("module.") else k
                    new_state_dict[name] = v

                model.load_state_dict(new_state_dict, strict=True)
                self.has_weights = True
            except Exception as e:
                logger.warning(f"Could not load Retinexformer weights: {e}")
                self.has_weights = False

        model.eval()
        self.model = model
        self._loaded = True
        logger.info("Retinexformer low-light model initialized successfully")

    def restore(self, image: Any) -> Any:
        self.ensure_loaded()
        from PIL import Image

        is_pil = isinstance(image, Image.Image)
        if is_pil:
            rgb_in = np.array(image).astype(np.float32) / 255.0
        elif isinstance(image, np.ndarray):
            rgb_in = (image[:, :, ::-1].astype(np.float32)) / 255.0
        else:
            raise ValueError("Invalid input image provided to RetinexformerModel.restore()")

        if not self.has_weights:
            # Mild gamma curve fallback for low-light enhancement when unweighted
            enhanced_rgb = np.power(np.clip(rgb_in, 0.0, 1.0), 0.7)
            if is_pil:
                return Image.fromarray((enhanced_rgb * 255.0).astype(np.uint8))
            else:
                return (enhanced_rgb * 255.0).astype(np.uint8)[:, :, ::-1]

        device_obj = torch.device(self.device)
        h, w = rgb_in.shape[:2]

        multiple = 16
        pad_h = (multiple - h % multiple) % multiple
        pad_w = (multiple - w % multiple) % multiple

        if pad_h > 0 or pad_w > 0:
            rgb_padded = np.pad(rgb_in, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
        else:
            rgb_padded = rgb_in

        t_input = torch.from_numpy(rgb_padded).permute(2, 0, 1).unsqueeze(0).float().to(device_obj)

        with torch.no_grad():
            t_output = self.model(t_input)

        out_rgb = t_output.squeeze(0).permute(1, 2, 0).cpu().numpy()

        if pad_h > 0 or pad_w > 0:
            out_rgb = out_rgb[:h, :w, :]

        out_rgb = np.clip(out_rgb, 0.0, 1.0)

        if is_pil:
            return Image.fromarray((out_rgb * 255.0).astype(np.uint8))
        else:
            return (out_rgb * 255.0).astype(np.uint8)[:, :, ::-1]


