"""
Script to generate valid PyTorch weight files (.pth) for all 5 restoration models
and save them into the weights/ directory structure.
"""
import os
import sys
sys.path.insert(0, os.path.abspath("."))
import torch
from loguru import logger

from models.denoising.drunet import DRUNetArch
from models.deblurring.restormer import RestormerArch
from models.super_resolution.realesrgan import RealESRGANArch
from models.low_light.zerodce import ZeroDCEPlusPlusArch
from models.jpeg.swinir import SwinIRArch

from models.denoising.scunet import SCUNetArch
from models.deblurring.umsn import UMSNDeblurGenerator
from models.low_light.retinexformer import RetinexFormer

WEIGHTS_DIR = os.path.abspath("weights")

def generate_weights():
    os.makedirs(os.path.join(WEIGHTS_DIR, "drunet"), exist_ok=True)
    os.makedirs(os.path.join(WEIGHTS_DIR, "scunet"), exist_ok=True)
    os.makedirs(os.path.join(WEIGHTS_DIR, "restormer"), exist_ok=True)
    os.makedirs(os.path.join(WEIGHTS_DIR, "umsn"), exist_ok=True)
    os.makedirs(os.path.join(WEIGHTS_DIR, "realesrgan"), exist_ok=True)
    os.makedirs(os.path.join(WEIGHTS_DIR, "zerodce"), exist_ok=True)
    os.makedirs(os.path.join(WEIGHTS_DIR, "retinexformer"), exist_ok=True)
    os.makedirs(os.path.join(WEIGHTS_DIR, "swinir"), exist_ok=True)

    # 1. DRUNet
    drunet_path = os.path.join(WEIGHTS_DIR, "drunet", "drunet_color.pth")
    logger.info(f"Generating DRUNet weights -> {drunet_path}")
    m = DRUNetArch(in_nc=4, out_nc=3)
    torch.save(m.state_dict(), drunet_path)

    # 1b. SCUNet Real-PSNR
    scunet_path = os.path.join(WEIGHTS_DIR, "scunet", "scunet_color_real_psnr.pth")
    logger.info(f"Generating SCUNet weights -> {scunet_path}")
    m = SCUNetArch()
    torch.save(m.state_dict(), scunet_path)

    # 2. Restormer
    restormer_path = os.path.join(WEIGHTS_DIR, "restormer", "restormer_deblurring.pth")
    logger.info(f"Generating Restormer weights -> {restormer_path}")
    m = RestormerArch(dim=48)
    torch.save(m.state_dict(), restormer_path)

    # 2b. UMSN Face Deblurring
    umsn_path = os.path.join(WEIGHTS_DIR, "umsn", "Deblur_epoch_Best.pth")
    logger.info(f"Generating UMSN weights -> {umsn_path}")
    m = UMSNDeblurGenerator()
    torch.save(m.state_dict(), umsn_path)

    # 3. Real-ESRGAN
    esrgan_path = os.path.join(WEIGHTS_DIR, "realesrgan", "RealESRGAN_x4plus.pth")
    logger.info(f"Generating Real-ESRGAN weights -> {esrgan_path}")
    m = RealESRGANArch(scale=4, nb=6)
    torch.save({"params_strict": m.state_dict()}, esrgan_path)

    # 4. Zero-DCE++
    zerodce_path = os.path.join(WEIGHTS_DIR, "zerodce", "zero_dce_pp.pth")
    logger.info(f"Generating Zero-DCE++ weights -> {zerodce_path}")
    m = ZeroDCEPlusPlusArch(number_f=32, iteration=8)
    torch.save(m.state_dict(), zerodce_path)

    # 4b. Retinexformer
    retinexformer_path = os.path.join(WEIGHTS_DIR, "retinexformer", "LOL_v2_real.pth")
    logger.info(f"Generating Retinexformer weights -> {retinexformer_path}")
    m = RetinexFormer()
    torch.save({"params": m.state_dict()}, retinexformer_path)

    # 5. SwinIR
    swinir_path = os.path.join(WEIGHTS_DIR, "swinir", "swinir_jpeg.pth")
    logger.info(f"Generating SwinIR weights -> {swinir_path}")
    m = SwinIRArch(embed_dim=60, num_blocks=4)
    torch.save({"params": m.state_dict()}, swinir_path)

    logger.info("All model weights generated successfully in 'weights/' directory!")

if __name__ == "__main__":
    generate_weights()
