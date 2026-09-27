"""
Shared YuNet SOTA Face Detector (OpenCV DNN).

Reused across UMSN and OSDFace models to avoid duplicating detector logic.
"""
import os
import cv2
import numpy as np
from typing import List, Tuple
from loguru import logger

YUNET_MODEL_PATH = "weights/umsn/face_detection_yunet_2023mar.onnx"
YUNET_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"


class YuNetFaceDetector:
    """
    OpenCV DNN YuNet Face Detector wrapper.
    Detects face bounding boxes in (x, y, w, h) format.
    """

    def __init__(self, score_threshold: float = 0.20, nms_threshold: float = 0.3):
        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold
        self._ensure_model_exists()

    def _ensure_model_exists(self) -> None:
        abs_path = os.path.abspath(YUNET_MODEL_PATH)
        if not os.path.exists(abs_path):
            os.makedirs(os.path.dirname(abs_path), exist_ok=True)
            import urllib.request
            logger.info(f"[YUNET] Downloading YuNet ONNX model from {YUNET_URL}...")
            urllib.request.urlretrieve(YUNET_URL, abs_path)
            logger.info(f"[YUNET] YuNet model saved to {abs_path}")

    def detect(self, image_bgr: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detect face bounding boxes in an image.

        Args:
            image_bgr: Input image in BGR format (numpy ndarray).

        Returns:
            List of bounding boxes [(x, y, w, h), ...].
        """
        h_img, w_img = image_bgr.shape[:2]
        abs_path = os.path.abspath(YUNET_MODEL_PATH)

        if os.path.exists(abs_path) and hasattr(cv2, "FaceDetectorYN"):
            try:
                detector = cv2.FaceDetectorYN.create(
                    abs_path, "", (w_img, h_img),
                    score_threshold=self.score_threshold,
                    nms_threshold=self.nms_threshold,
                    top_k=5000
                )
                _, faces = detector.detect(image_bgr)
                if faces is not None and len(faces) > 0:
                    bboxes = []
                    for face in faces:
                        fx, fy, fw, fh = int(face[0]), int(face[1]), int(face[2]), int(face[3])
                        fx, fy = max(0, fx), max(0, fy)
                        fw, fh = max(1, min(w_img - fx, fw)), max(1, min(h_img - fy, fh))
                        bboxes.append((fx, fy, fw, fh))
                    return bboxes
            except Exception as e:
                logger.warning(f"[YUNET ERROR] YuNet face detection failed: {e}. Falling back to HaarCascade.")

        # Fallback to OpenCV HaarCascade if YuNet DNN fails
        try:
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            if os.path.exists(cascade_path):
                face_cascade = cv2.CascadeClassifier(cascade_path)
                if not face_cascade.empty():
                    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
                    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=3, minSize=(20, 20))
                    bboxes = [tuple(f) for f in faces]
                    return bboxes
        except Exception as e:
            logger.warning(f"[HAARCASCADE ERROR] HaarCascade detection error: {e}")

        return []
