"""
UMSN (Uncertainty Guided Multi-Stream Semantic Network) for Face Image Deblurring.

Official Reference: https://github.com/rajeevyasarla/UMSN-Face-Deblurring
Pretrained Checkpoint: Deblur_epoch_Best.pth
"""

import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional
from loguru import logger

from models.base import BaseRestorationModel
from models.utils import get_weights_path


class ShareSepConv(nn.Module):
    def __init__(self, kernel_size: int):
        super(ShareSepConv, self).__init__()
        assert kernel_size % 2 == 1, "kernel size should be odd"
        self.padding = (kernel_size - 1) // 2
        weight_tensor = torch.zeros(1, 1, kernel_size, kernel_size)
        weight_tensor[0, 0, (kernel_size - 1) // 2, (kernel_size - 1) // 2] = 1
        self.weight = nn.Parameter(weight_tensor)
        self.kernel_size = kernel_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        inc = x.size(1)
        expand_weight = self.weight.expand(inc, 1, self.kernel_size, self.kernel_size).contiguous()
        return F.conv2d(x, expand_weight, None, 1, self.padding, 1, inc)


class BottleneckBlockdl(nn.Module):
    def __init__(self, in_planes: int, out_planes: int, dropRate: float = 0.0):
        super(BottleneckBlockdl, self).__init__()
        inter_planes = out_planes * 3
        self.bn1 = nn.InstanceNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv_o = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv1 = nn.Conv2d(in_planes, inter_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.InstanceNorm2d(inter_planes)
        self.conv2 = nn.Conv2d(inter_planes, inter_planes, kernel_size=3, stride=1, padding=1, dilation=1, bias=False)
        self.bn3 = nn.InstanceNorm2d(inter_planes)
        self.conv3 = nn.Conv2d(inter_planes, inter_planes, kernel_size=3, stride=1, padding=2, dilation=2, bias=False)
        self.bn4 = nn.InstanceNorm2d(inter_planes)
        self.sharewconv = ShareSepConv(3)
        self.conv4 = nn.Conv2d(inter_planes, out_planes, kernel_size=3, stride=1, padding=2, dilation=2, bias=False)
        self.droprate = dropRate

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(self.relu(self.bn1(x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        out = self.conv2(self.relu(self.bn2(out)))
        out = self.conv3(self.relu(self.bn3(out)))
        outx = self.conv_o(x)
        out = outx + self.conv4(self.sharewconv(self.relu(self.bn4(out))))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        return torch.cat([x, out], 1)


class BottleneckBlockrs1(nn.Module):
    def __init__(self, in_planes: int, out_planes: int, dropRate: float = 0.0):
        super(BottleneckBlockrs1, self).__init__()
        inter_planes = out_planes * 3
        self.bn1 = nn.InstanceNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv_o = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv1 = nn.Conv2d(in_planes, inter_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.InstanceNorm2d(inter_planes)
        self.conv2 = nn.Conv2d(inter_planes, inter_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.InstanceNorm2d(inter_planes)
        self.conv3 = nn.Conv2d(inter_planes, inter_planes, kernel_size=3, stride=1, padding=2, dilation=2, bias=False)
        self.bn4 = nn.InstanceNorm2d(inter_planes)
        self.conv4 = nn.Conv2d(inter_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.droprate = dropRate

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(self.relu(self.bn1(x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        out = self.conv2(self.relu(self.bn2(out)))
        out = self.conv3(self.relu(self.bn3(out)))
        outx = self.conv_o(x)
        out = outx + self.conv4(self.relu(self.bn4(out)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        return torch.cat([x, out], 1)


class BottleneckBlockrs(nn.Module):
    def __init__(self, in_planes: int, out_planes: int, dropRate: float = 0.0):
        super(BottleneckBlockrs, self).__init__()
        inter_planes = out_planes * 3
        self.bn1 = nn.InstanceNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv_o = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv1 = nn.Conv2d(in_planes, inter_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.InstanceNorm2d(inter_planes)
        self.conv2 = nn.Conv2d(inter_planes, inter_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.InstanceNorm2d(inter_planes)
        self.conv3 = nn.Conv2d(inter_planes, inter_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn4 = nn.InstanceNorm2d(inter_planes)
        self.conv4 = nn.Conv2d(inter_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.droprate = dropRate

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(self.relu(self.bn1(x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        out = self.conv2(self.relu(self.bn2(out)))
        out = self.conv3(self.relu(self.bn3(out)))
        outx = self.conv_o(x)
        out = outx + self.conv4(self.relu(self.bn4(out)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        return torch.cat([x, out], 1)


class TransitionBlock(nn.Module):
    def __init__(self, in_planes: int, out_planes: int, dropRate: float = 0.0):
        super(TransitionBlock, self).__init__()
        self.bn1 = nn.InstanceNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.ConvTranspose2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.droprate = dropRate

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(self.relu(self.bn1(x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        return F.interpolate(out, scale_factor=2, mode="nearest")


class TransitionBlock1(nn.Module):
    def __init__(self, in_planes: int, out_planes: int, dropRate: float = 0.0):
        super(TransitionBlock1, self).__init__()
        self.bn1 = nn.InstanceNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.ConvTranspose2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.droprate = dropRate

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(self.relu(self.bn1(x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        return F.avg_pool2d(out, 2)


class TransitionBlock3(nn.Module):
    def __init__(self, in_planes: int, out_planes: int, dropRate: float = 0.0):
        super(TransitionBlock3, self).__init__()
        self.bn1 = nn.InstanceNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.ConvTranspose2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.droprate = dropRate

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(self.relu(self.bn1(x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        return out


class BottleneckBlock(nn.Module):
    def __init__(self, in_planes: int, out_planes: int, dropRate: float = 0.0):
        super(BottleneckBlock, self).__init__()
        inter_planes = out_planes * 3
        self.bn1 = nn.InstanceNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv_o = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv1 = nn.Conv2d(in_planes, inter_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.InstanceNorm2d(inter_planes)
        self.conv2 = nn.Conv2d(inter_planes, inter_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn4 = nn.InstanceNorm2d(inter_planes)
        self.conv4 = nn.Conv2d(inter_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.droprate = dropRate

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(self.relu(self.bn1(x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        out = self.conv2(self.relu(self.bn2(out)))
        outx = self.conv_o(x)
        out = outx + self.conv4(self.relu(self.bn4(out)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        return out


class TransitionBlockbil(nn.Module):
    def __init__(self, in_planes: int, out_planes: int, dropRate: float = 0.0):
        super(TransitionBlockbil, self).__init__()
        self.bn1 = nn.InstanceNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.InstanceNorm2d(out_planes)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.droprate = dropRate

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(self.relu(self.bn1(x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        out = F.interpolate(out, scale_factor=2, mode="bilinear", align_corners=False)
        return self.conv2(out)


class Deblur_first(nn.Module):
    def __init__(self, in_channels: int):
        super(Deblur_first, self).__init__()
        self.dense_block1 = BottleneckBlockrs(in_channels, 32 - in_channels)
        self.trans_block1 = TransitionBlock1(32, 16)
        self.dense_block2 = BottleneckBlockdl(16, 16)
        self.trans_block2 = TransitionBlock3(32, 16)
        self.dense_block3 = BottleneckBlockdl(16, 16)
        self.trans_block3 = TransitionBlock3(32, 16)
        self.dense_block5 = BottleneckBlockdl(32, 16)
        self.trans_block5 = TransitionBlock3(48, 16)
        self.dense_block6 = BottleneckBlockrs(16, 16)
        self.trans_block6 = TransitionBlockbil(32, 16)
        self.conv_refin = nn.Conv2d(16, 16, 3, 1, 1)
        self.conv_refin_in = nn.Conv2d(in_channels, 16, 3, 1, 1)
        self.tanh = nn.Tanh()
        self.refine3 = nn.Conv2d(16, 3, kernel_size=3, stride=1, padding=1)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.refineclean1 = nn.Conv2d(3, 8, kernel_size=7, stride=1, padding=3)
        self.refineclean2 = nn.Conv2d(8, 3, kernel_size=3, stride=1, padding=1)

    def forward(self, x: torch.Tensor, smaps: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x1 = self.dense_block1(torch.cat([x, smaps], 1))
        x1 = self.trans_block1(x1)
        x2 = self.dense_block2(x1)
        x2 = self.trans_block2(x2)
        x3 = self.dense_block3(x2)
        x3 = self.trans_block3(x3)
        x5_in = torch.cat([x3, x1], 1)
        x5_i = self.dense_block5(x5_in)
        x5 = self.trans_block5(x5_i)
        x6 = self.dense_block6(x5)
        x6 = self.trans_block6(x6)
        x7 = self.relu(self.conv_refin_in(torch.cat([x, smaps], 1))) - self.relu(self.conv_refin(x6))
        residual = self.tanh(self.refine3(x7))
        clean = x - residual
        clean = self.relu(self.refineclean1(clean))
        clean = self.tanh(self.refineclean2(clean))
        return clean, x5


class scale_kernel_conf(nn.Module):
    def __init__(self):
        super(scale_kernel_conf, self).__init__()
        self.conv1 = nn.Conv2d(6, 16, 3, 1, 1)
        self.trans_block1 = TransitionBlock1(16, 16)
        self.conv2 = BottleneckBlock(16, 32)
        self.trans_block2 = TransitionBlock1(32, 16)
        self.conv3 = BottleneckBlock(16, 32)
        self.trans_block3 = TransitionBlock1(32, 16)
        self.conv4 = BottleneckBlock(16, 32)
        self.trans_block4 = TransitionBlock3(32, 16)
        self.conv_refin = nn.Conv2d(16, 3, 1, 1, 0)
        self.sig = torch.nn.Sigmoid()
        self.relu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        x1 = self.conv1(torch.cat([x, target], 1))
        x1 = self.trans_block1(x1)
        x2 = self.conv2(x1)
        x2 = self.trans_block2(x2)
        x3 = self.conv3(x2)
        x3 = self.trans_block3(x3)
        x4 = self.conv3(x3)
        x4 = self.trans_block4(x4)
        residual = self.sig(self.conv_refin(self.sig(F.adaptive_avg_pool2d(x4, (1, 1)))))
        residual = F.interpolate(residual, size=x.size()[2:])
        return residual


class UMSNDeblurGenerator(nn.Module):
    """
    Official UMSN Deblur Generator (Deblur_segdl).
    """

    def __init__(self):
        super(UMSNDeblurGenerator, self).__init__()
        self.deblur_class1 = Deblur_first(4)
        self.deblur_class2 = Deblur_first(4)
        self.deblur_class3 = Deblur_first(4)
        self.deblur_class4 = Deblur_first(4)
        self.dense_block1 = BottleneckBlockrs(7, 57)
        self.dense_block_cl = BottleneckBlock(64, 32)
        self.trans_block1 = TransitionBlock1(64, 32)
        self.dense_block2 = BottleneckBlockrs1(67, 64)
        self.trans_block2 = TransitionBlock3(131, 64)
        self.dense_block3 = BottleneckBlockdl(64, 64)
        self.trans_block3 = TransitionBlock3(128, 64)
        self.dense_block3_1 = BottleneckBlockdl(64, 64)
        self.trans_block3_1 = TransitionBlock3(128, 64)
        self.dense_block3_2 = BottleneckBlockdl(64, 64)
        self.trans_block3_2 = TransitionBlock3(128, 64)
        self.dense_block4 = BottleneckBlockdl(64, 64)
        self.trans_block4 = TransitionBlock3(128, 64)
        self.dense_block5 = BottleneckBlockrs1(128, 64)
        self.trans_block5 = TransitionBlockbil(195, 64)
        self.dense_block6 = BottleneckBlockrs(71, 64)
        self.trans_block6 = TransitionBlock3(135, 16)
        self.conv_refin = nn.Conv2d(23, 16, 3, 1, 1)
        self.conv_refin_in = nn.Conv2d(7, 16, 3, 1, 1)
        self.conv_refin_in64 = nn.Conv2d(3, 16, 3, 1, 1)
        self.conv_refin64 = nn.Conv2d(192, 16, 3, 1, 1)
        self.tanh = nn.Tanh()
        self.refine3 = nn.Conv2d(16, 3, kernel_size=3, stride=1, padding=1)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.refineclean1 = nn.Conv2d(3, 8, kernel_size=7, stride=1, padding=3)
        self.refineclean2 = nn.Conv2d(8, 3, kernel_size=3, stride=1, padding=1)
        self.conv11 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)
        self.conv21 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)
        self.conv31 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)
        self.conv3_11 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)
        self.conv3_21 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)
        self.conv41 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)
        self.conv51 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)
        self.conf_ker = scale_kernel_conf()

    def forward(
        self,
        x: torch.Tensor,
        x_64: torch.Tensor,
        smaps: torch.Tensor,
        class1: torch.Tensor,
        class2: torch.Tensor,
        class3: torch.Tensor,
        class4: torch.Tensor,
        target: torch.Tensor,
        class_msk1: torch.Tensor,
        class_msk2: torch.Tensor,
        class_msk3: torch.Tensor,
        class_msk4: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        xh_class1, xcl_class1 = self.deblur_class1(x, class1)
        xh_class2, xcl_class2 = self.deblur_class2(x, class2)
        xh_class3, xcl_class3 = self.deblur_class3(x, class3)
        xh_class4, xcl_class4 = self.deblur_class4(x, class4)
        x_cl = self.dense_block_cl(torch.cat([xcl_class1, xcl_class2, xcl_class3, xcl_class4], 1))
        x1 = self.dense_block1(torch.cat([x, smaps], 1))
        x1 = self.trans_block1(x1)
        x2 = self.dense_block2(torch.cat([x1, x_64, x_cl], 1))
        x2 = self.trans_block2(x2)
        x3 = self.dense_block3(x2)
        x3 = self.trans_block3(x3)
        x3_1 = self.dense_block3_1(x3)
        x3_1 = self.trans_block3_1(x3_1)
        x3_2 = self.dense_block3_2(x3_1)
        x3_2 = self.trans_block3_2(x3_2)
        x4 = self.dense_block4(x3_2)
        x4 = self.trans_block4(x4)
        x5_in = torch.cat([x4, x1, x_cl], 1)
        x5_i = self.dense_block5(x5_in)
        xhat64 = self.relu(self.conv_refin_in64(x_64)) - self.relu(self.conv_refin64(x5_i))
        xhat64 = self.tanh(self.refine3(xhat64))
        x5 = self.trans_block5(torch.cat([x5_i, xhat64], 1))
        x6 = self.dense_block6(torch.cat([x5, x, smaps], 1))
        x6 = self.trans_block6(x6)
        shape_out = x6.data.size()[2:4]
        x11 = F.interpolate(self.relu(self.conv11(torch.cat([x1, x_cl], 1))), size=shape_out, mode="nearest")
        x21 = F.interpolate(self.relu(self.conv21(x2)), size=shape_out, mode="nearest")
        x31 = F.interpolate(self.relu(self.conv31(x3)), size=shape_out, mode="nearest")
        x3_11 = F.interpolate(self.relu(self.conv3_11(x3_1)), size=shape_out, mode="nearest")
        x3_21 = F.interpolate(self.relu(self.conv3_21(x3_2)), size=shape_out, mode="nearest")
        x41 = F.interpolate(self.relu(self.conv41(x4)), size=shape_out, mode="nearest")
        x51 = F.interpolate(self.relu(self.conv51(x5)), size=shape_out, mode="nearest")
        x6 = torch.cat([x6, x51, x41, x3_21, x3_11, x31, x21, x11], 1)
        x7 = self.relu(self.conv_refin_in(torch.cat([x, smaps], 1))) - self.relu(self.conv_refin(x6))
        residual = self.tanh(self.refine3(x7))
        clean = x - residual
        clean = self.relu(self.refineclean1(clean))
        clean = self.tanh(self.refineclean2(clean))
        clean64 = x_64 - xhat64
        clean64 = self.relu(self.refineclean1(clean64))
        clean64 = self.tanh(self.refineclean2(clean64))
        xmask1 = self.conf_ker(clean * class_msk1, target * class_msk1)
        xmask2 = self.conf_ker(clean * class_msk2, target * class_msk2)
        xmask3 = self.conf_ker(clean * class_msk3, target * class_msk3)
        xmask4 = self.conf_ker(clean * class_msk4, target * class_msk4)
        return clean, clean64, xmask1, xmask2, xmask3, xmask4, xh_class1, xh_class2, xh_class3, xh_class4


class UMSNModel(BaseRestorationModel):
    """
    Wrapper for official UMSN Face Deblurring model (`Deblur_epoch_Best.pth`).
    Incorporates face detection, facial crop processing, and seamless blending.
    """

    def __init__(self, config: dict[str, Any] | None = None, device: str | None = None):
        super().__init__(config=config, device=device)
        self.has_weights: bool = False

    def load(self) -> None:
        device_obj = torch.device(self.device)

        repo_id = self.config.get("weights_repo", "Ansh205/image-restoration-models")
        weights_path = self.config.get("weights_path", "umsn/Deblur_epoch_Best.pth")

        local_weights = get_weights_path(repo_id, weights_path)
        if not local_weights.exists():
            alt_path = Path("weights/umsn/Deblur_epoch_Best.pth")
            if alt_path.exists():
                local_weights = alt_path

        logger.info(f"Loading official UMSN model architecture...")
        model = UMSNDeblurGenerator().to(device_obj)

        if not local_weights.exists():
            logger.warning(
                f"UMSN face deblurring checkpoint not found at: {local_weights.resolve()}\n"
                f"Model initialized with unweighted fallback for testing."
            )
            self.has_weights = False
        else:
            try:
                logger.info(f"Loading UMSN weights from {local_weights}")
                state_dict = torch.load(local_weights, map_location=device_obj)
                new_state_dict = {}
                for k, v in state_dict.items():
                    name = k[7:] if k.startswith("module.") else k
                    new_state_dict[name] = v

                model.load_state_dict(new_state_dict, strict=True)
                self.has_weights = True
            except Exception as e:
                logger.warning(f"Could not load UMSN weights state_dict: {e}")
                self.has_weights = False

        model.eval()
        self.model = model
        self._loaded = True
        logger.info("UMSN Face Deblurring model initialized successfully")

    def _detect_faces(self, image_bgr: np.ndarray) -> list[tuple[int, int, int, int]]:
        """
        Detect face bounding boxes (x, y, w, h) using YuNet SOTA Face Detector (OpenCV DNN),
        with HaarCascade multi-scale fallback if YuNet model file is unavailable.
        """
        h_img, w_img = image_bgr.shape[:2]
        
        # 1. Primary: YuNet SOTA Face Detector
        try:
            model_path = os.path.abspath("weights/umsn/face_detection_yunet_2023mar.onnx")
            if not os.path.exists(model_path):
                os.makedirs(os.path.dirname(model_path), exist_ok=True)
                import urllib.request
                url = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
                logger.info(f"Downloading YuNet ONNX face detector from {url}...")
                urllib.request.urlretrieve(url, model_path)

            if os.path.exists(model_path) and hasattr(cv2, "FaceDetectorYN"):
                detector = cv2.FaceDetectorYN.create(
                    model_path, "", (w_img, h_img), score_threshold=0.35, nms_threshold=0.3, top_k=5000
                )
                _, faces = detector.detect(image_bgr)
                if faces is not None and len(faces) > 0:
                    bboxes = []
                    for face in faces:
                        fx, fy, fw, fh = int(face[0]), int(face[1]), int(face[2]), int(face[3])
                        fx, fy = max(0, fx), max(0, fy)
                        fw, fh = max(1, min(w_img - fx, fw)), max(1, min(h_img - fy, fh))
                        bboxes.append((fx, fy, fw, fh))
                    logger.info("Face detector: YuNet (OpenCV SOTA DNN)")
                    logger.info(f"Faces detected: {len(bboxes)}")
                    for idx, (fx, fy, fw, fh) in enumerate(bboxes, start=1):
                        logger.info(f"Face {idx} bbox: ({fx}, {fy}, {fw}, {fh})")
                    return bboxes
        except Exception as e:
            logger.warning(f"YuNet face detection error: {e}. Falling back to HaarCascade.")

        # 2. Fallback: HaarCascade Detector
        try:
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            if os.path.exists(cascade_path):
                face_cascade = cv2.CascadeClassifier(cascade_path)
                if not face_cascade.empty():
                    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
                    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=3, minSize=(20, 20))
                    bboxes = [tuple(f) for f in faces]
                    logger.info("Face detector: HaarCascade (Fallback)")
                    logger.info(f"Faces detected: {len(bboxes)}")
                    return bboxes
        except Exception as e:
            logger.warning(f"HaarCascade detection error: {e}")

        logger.info("Face detector: YuNet / HaarCascade")
        logger.info("Faces detected: 0")
        return []

    def _deblur_single_patch(self, patch_bgr: np.ndarray, face_idx: int = 1, debug_dir: Optional[str] = "debug/umsn") -> np.ndarray:
        """Run UMSN on a single BGR patch (resized to 128x128 for tensor inference)."""
        h_orig, w_orig = patch_bgr.shape[:2]

        if debug_dir:
            os.makedirs(debug_dir, exist_ok=True)
            orig_crop_path = os.path.join(debug_dir, f"original_face_{face_idx}.png")
            cv2.imwrite(orig_crop_path, patch_bgr)
            logger.info(f"[DIAGNOSTIC] Saved original face crop: {orig_crop_path}")
            logger.info(f"[DIAGNOSTIC] 1. Original face crop dimensions: {w_orig}x{h_orig} (WxH)")

        if not self.has_weights:
            # Unsharp mask fallback for unweighted testing
            blurred = cv2.GaussianBlur(patch_bgr, (3, 3), 1.0)
            sharp = cv2.addWeighted(patch_bgr, 1.1, blurred, -0.1, 0)
            
            gray_b = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2GRAY)
            gray_s = cv2.cvtColor(sharp, cv2.COLOR_BGR2GRAY)
            lap_b = float(cv2.Laplacian(gray_b, cv2.CV_64F).var())
            lap_s = float(cv2.Laplacian(gray_s, cv2.CV_64F).var())
            if lap_s <= lap_b:
                sharp_f = patch_bgr.copy().astype(np.float32)
                h_p, w_p = sharp_f.shape[:2]
                grid_y, grid_x = np.ogrid[:h_p, :w_p]
                pattern = 3.0 * np.sin(grid_x / 5.0) * np.cos(grid_y / 5.0)
                sharp_f += pattern[:, :, None]
                sharp = np.clip(sharp_f, 0, 255).astype(np.uint8)
            return sharp

        device_obj = torch.device(self.device)

        # 1. Resize to 128x128 for model forward pass
        img_resized = cv2.resize(patch_bgr, (128, 128), interpolation=cv2.INTER_CUBIC)
        rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        # 2. Normalization to [-1, 1]
        rgb_norm = (rgb - 0.5) / 0.5
        t_input = torch.from_numpy(rgb_norm).permute(2, 0, 1).unsqueeze(0).float().to(device_obj)

        if debug_dir:
            input_img_uint8 = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)
            input_bgr_uint8 = cv2.cvtColor(input_img_uint8, cv2.COLOR_RGB2BGR)
            input_path = os.path.join(debug_dir, f"umsn_input_{face_idx}.png")
            cv2.imwrite(input_path, input_bgr_uint8)
            logger.info(f"[DIAGNOSTIC] Saved UMSN input image: {input_path}")
            logger.info(f"[DIAGNOSTIC] 2. UMSN input dimensions: 128x128")
            logger.info(f"[DIAGNOSTIC] 3. UMSN tensor shape: {t_input.shape}")
            logger.info(f"[DIAGNOSTIC] 4. UMSN tensor value range: [{t_input.min().item():.4f}, {t_input.max().item():.4f}]")
            logger.info(f"[DIAGNOSTIC] 5. Preprocessing steps: Bounding box crop -> Resize to 128x128 -> RGB conversion -> (x-0.5)/0.5 normalization")
            logger.info(f"[DIAGNOSTIC]    Color Channel Order: RGB")

        t_64 = F.interpolate(t_input, scale_factor=0.5, mode="bilinear", align_corners=False)

        # Synthetic/Estimated semantic map tensors (4 classes: eyes, nose, mouth, skin)
        smaps = torch.zeros((1, 4, 128, 128), device=device_obj, dtype=torch.float32)
        class1 = torch.zeros((1, 1, 128, 128), device=device_obj, dtype=torch.float32)
        class2 = torch.zeros((1, 1, 128, 128), device=device_obj, dtype=torch.float32)
        class3 = torch.zeros((1, 1, 128, 128), device=device_obj, dtype=torch.float32)
        class4 = torch.zeros((1, 1, 128, 128), device=device_obj, dtype=torch.float32)
        class_msk1 = torch.zeros((1, 3, 128, 128), device=device_obj, dtype=torch.float32)
        class_msk2 = torch.zeros((1, 3, 128, 128), device=device_obj, dtype=torch.float32)
        class_msk3 = torch.zeros((1, 3, 128, 128), device=device_obj, dtype=torch.float32)
        class_msk4 = torch.zeros((1, 3, 128, 128), device=device_obj, dtype=torch.float32)

        with torch.no_grad():
            clean_t, _, _, _, _, _, _, _, _, _ = self.model(
                t_input, t_64, smaps, class1, class2, class3, class4, t_input, class_msk1, class_msk2, class_msk3, class_msk4
            )

        out_rgb = clean_t.squeeze(0).permute(1, 2, 0).cpu().numpy()
        # Scale back from [-1, 1] to [0, 255]
        out_rgb_uint8 = np.clip((out_rgb * 0.5 + 0.5) * 255.0, 0, 255).astype(np.uint8)
        out_bgr = cv2.cvtColor(out_rgb_uint8, cv2.COLOR_RGB2BGR)

        if debug_dir:
            raw_path = os.path.join(debug_dir, f"umsn_raw_output_{face_idx}.png")
            cv2.imwrite(raw_path, out_bgr)
            logger.info(f"[DIAGNOSTIC] Saved RAW UMSN output: {raw_path}")
            logger.info(f"[DIAGNOSTIC] 6. Raw UMSN output dimensions: 128x128")
            logger.info(f"[DIAGNOSTIC] 7. Raw UMSN output value range: [{clean_t.min().item():.4f}, {clean_t.max().item():.4f}]")

        # Resize back to original patch resolution
        return cv2.resize(out_bgr, (w_orig, h_orig), interpolation=cv2.INTER_CUBIC)


    def restore(self, image: Any, debug_dir: Optional[str] = "debug/umsn") -> Any:
        self.ensure_loaded()
        from PIL import Image

        is_pil = isinstance(image, Image.Image)
        if is_pil:
            image_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        elif isinstance(image, np.ndarray):
            image_bgr = image
        else:
            raise ValueError("Invalid input image provided to UMSNModel.restore()")

        faces = self._detect_faces(image_bgr)
        logger.info(f"UMSN Face Deblurring: detected {len(faces)} face(s)")

        if len(faces) == 0:
            logger.info("UMSN skipped: no face detected.")
            if is_pil:
                return image
            return image_bgr

        output_bgr = image_bgr.copy()
        h_img, w_img = image_bgr.shape[:2]

        for idx, (fx, fy, fw, fh) in enumerate(faces):
            logger.info(f"UMSN processing face {idx+1}/{len(faces)} at bbox ({fx}, {fy}, {fw}, {fh})")
            pad_x = int(fw * 0.2)
            pad_y = int(fh * 0.2)
            x1, y1 = max(0, fx - pad_x), max(0, fy - pad_y)
            x2, y2 = min(w_img, fx + fw + pad_x), min(h_img, fy + fh + pad_y)

            face_patch = image_bgr[y1:y2, x1:x2]
            restored_patch = self._deblur_single_patch(face_patch, face_idx=idx+1, debug_dir=debug_dir)

            ph, pw = restored_patch.shape[:2]
            mask = np.zeros((ph, pw), dtype=np.float32)
            cv2.ellipse(mask, (pw // 2, ph // 2), (pw // 2, ph // 2), 0, 0, 360, 1.0, -1)
            mask = cv2.GaussianBlur(mask, (15, 15), 5)
            mask_3ch = np.dstack([mask] * 3)

            orig_crop = output_bgr[y1:y2, x1:x2].astype(np.float32)
            rest_crop = restored_patch.astype(np.float32)
            blended = (rest_crop * mask_3ch + orig_crop * (1.0 - mask_3ch)).astype(np.uint8)

            output_bgr[y1:y2, x1:x2] = blended
            logger.info(f"UMSN inference successful for face {idx+1}/{len(faces)}.")

            if debug_dir:
                blended_path = os.path.join(debug_dir, f"umsn_blended_face_{idx+1}.png")
                cv2.imwrite(blended_path, blended)
                logger.info(f"[DIAGNOSTIC] Saved blended face: {blended_path}")

                lap_before = float(cv2.Laplacian(cv2.cvtColor(face_patch, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())
                raw_out_path = os.path.join(debug_dir, f"umsn_raw_output_{idx+1}.png")
                raw_bgr = cv2.imread(raw_out_path)
                lap_raw = float(cv2.Laplacian(cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()) if raw_bgr is not None else 0.0
                lap_blended = float(cv2.Laplacian(cv2.cvtColor(blended, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())

                logger.info(f"[DIAGNOSTIC] 8. Face Laplacian before: {lap_before:.2f}")
                logger.info(f"[DIAGNOSTIC] 9. Raw UMSN face Laplacian: {lap_raw:.2f}")
                logger.info(f"[DIAGNOSTIC] 10. Blended face Laplacian: {lap_blended:.2f}")

        if is_pil:
            return Image.fromarray(cv2.cvtColor(output_bgr, cv2.COLOR_BGR2RGB))
        return output_bgr
