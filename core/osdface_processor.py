"""
OSDFace Processor — Orchestrates face detection, 1:1 square crop alignment,
OSDFace restoration, and seamless feather-blending back into original images.
"""
import cv2
import numpy as np
from typing import Tuple, List, Union
from PIL import Image
from loguru import logger

from core.face_detector import YuNetFaceDetector
from models.osdface_model import OSDFaceModel

# Singleton cache for OSDFace model instance to prevent reloading per request
_osdface_model_instance: Union[OSDFaceModel, None] = None


def get_osdface_model(device: str = "cpu") -> OSDFaceModel:
    """
    Get or instantiate the singleton OSDFace model wrapper.
    Ensures model weights are loaded once and reused across requests.
    """
    global _osdface_model_instance
    if _osdface_model_instance is None:
        logger.info("[OSDFACE] Initializing OSDFace model instance (Singleton)...")
        _osdface_model_instance = OSDFaceModel(device=device)
        _osdface_model_instance.load()
    return _osdface_model_instance


class OSDFaceProcessor:
    """
    Processor to execute OSDFace face restoration pipeline on images with 0, 1, or multiple faces.
    Follows official 512x512 aligned square face cropping and feathered ellipse mask reinsertion.
    """

    def __init__(self, device: str = "cpu"):
        self.device = device
        self.detector = YuNetFaceDetector()

    def process(self, image: Union[Image.Image, np.ndarray]) -> Tuple[Union[Image.Image, np.ndarray], int, bool]:
        """
        Process an input image with OSDFace face restoration.

        Args:
            image: Input image (PIL Image or BGR uint8 numpy array).

        Returns:
            Tuple of (restored_image, face_count, face_restored_bool)
        """
        logger.info("============================================================")
        logger.info("[OSDFACE MODE]")
        logger.info("Mode: OSDFACE")
        logger.info("============================================================")

        is_pil = isinstance(image, Image.Image)
        if is_pil:
            image_bgr = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)
        else:
            image_bgr = image.copy()

        h_img, w_img = image_bgr.shape[:2]
        img_area = h_img * w_img

        # 1. Detect faces using YuNet detector
        detected_bboxes = self.detector.detect(image_bgr)
        valid_bboxes = []

        logger.info("[OSDFACE DETECTION DIAGNOSTICS]")
        for idx, (fx, fy, fw, fh) in enumerate(detected_bboxes, start=1):
            face_area = fw * fh
            area_ratio = face_area / img_area
            width_ratio = fw / w_img
            height_ratio = fh / h_img
            aspect_ratio = fw / max(1, fh)

            logger.info(f"Detected Box {idx}:")
            logger.info(f"    bbox = ({fx}, {fy}, {fw}, {fh})")
            logger.info(f"    area_ratio = {area_ratio:.4f}")
            logger.info(f"    width_ratio = {width_ratio:.4f}")
            logger.info(f"    height_ratio = {height_ratio:.4f}")
            logger.info(f"    aspect_ratio = {aspect_ratio:.2f}")

            # Filter extreme false positives (e.g. boxes covering >75% of non-square image with extreme ratio)
            if area_ratio > 0.85 and (aspect_ratio < 0.3 or aspect_ratio > 3.0):
                logger.warning(f"    -> Skipping false positive box {idx} (extreme aspect ratio / area ratio)")
                continue

            valid_bboxes.append((fx, fy, fw, fh))

        face_count = len(valid_bboxes)
        logger.info(f"[OSDFACE] Valid faces after filtering: {face_count}")

        if face_count == 0:
            logger.info("[OSDFACE] 0 faces detected by YuNet. Processing full image through OSDFace model fallback...")
            valid_bboxes = [(0, 0, w_img, h_img)]
            face_count = 1

        # Load OSDFace model
        model = get_osdface_model(device=self.device)
        output_bgr = image_bgr.copy()

        for idx, (fx, fy, fw, fh) in enumerate(valid_bboxes, start=1):
            logger.info(f"[OSDFACE PROCESS FACE {idx}]")

            # 2. Square Face Alignment & Padding (1:1 aspect ratio preserving)
            cx = fx + fw // 2
            cy = fy + fh // 2
            side = max(fw, fh)
            side_padded = int(side * 1.35)

            x1 = max(0, cx - side_padded // 2)
            y1 = max(0, cy - side_padded // 2)
            x2 = min(w_img, cx + side_padded // 2)
            y2 = min(h_img, cy + side_padded // 2)

            face_patch_bgr = image_bgr[y1:y2, x1:x2]
            orig_crop_h, orig_crop_w = face_patch_bgr.shape[:2]

            if orig_crop_h < 4 or orig_crop_w < 4:
                logger.warning(f"[OSDFACE WARNING] Face patch {idx} is too small ({orig_crop_w}x{orig_crop_h}). Skipping.")
                continue

            logger.info(f"Face {idx} crop region: {x1}:{x2}, {y1}:{y2} (Square patch: {orig_crop_w}x{orig_crop_h})")

            # 3. Model Inference
            if is_pil:
                patch_pil = Image.fromarray(cv2.cvtColor(face_patch_bgr, cv2.COLOR_BGR2RGB))
                restored_pil = model.predict(patch_pil)
                restored_patch_bgr = cv2.cvtColor(np.array(restored_pil), cv2.COLOR_RGB2BGR)
            else:
                restored_patch_bgr = model.predict(face_patch_bgr)

            rest_h, rest_w = restored_patch_bgr.shape[:2]

            # 4. Feathered Ellipse Mask Blending
            mask = np.zeros((orig_crop_h, orig_crop_w), dtype=np.float32)
            cv2.ellipse(
                mask,
                (orig_crop_w // 2, orig_crop_h // 2),
                (orig_crop_w // 2, orig_crop_h // 2),
                0, 0, 360, 1.0, -1
            )
            ksize = max(7, (min(orig_crop_w, orig_crop_h) // 6) | 1)
            mask = cv2.GaussianBlur(mask, (ksize, ksize), 0)
            mask_3ch = np.dstack([mask] * 3)

            orig_crop_float = output_bgr[y1:y2, x1:x2].astype(np.float32)
            rest_crop_float = restored_patch_bgr.astype(np.float32)

            blended_crop = (rest_crop_float * mask_3ch + orig_crop_float * (1.0 - mask_3ch)).astype(np.uint8)
            output_bgr[y1:y2, x1:x2] = blended_crop

            logger.info(f"[OSDFACE BLEND]")
            logger.info(f"Face {idx} restored and blended back successfully")

        logger.info(f"[OSDFACE]")
        logger.info("Final OSDFace output generated")

        if is_pil:
            final_out = Image.fromarray(cv2.cvtColor(output_bgr, cv2.COLOR_BGR2RGB))
        else:
            final_out = output_bgr

        return final_out, face_count, True
