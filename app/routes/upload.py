"""
Upload route — handles image upload, validation, and preprocessing.
"""
import uuid
from pathlib import Path

from fastapi import APIRouter, UploadFile, File, HTTPException
from loguru import logger

from core.pipeline.image_utils import (
    load_image_from_bytes,
    resize_if_needed,
    image_to_bytes,
    ImageValidationError,
)
from app.schemas.image import UploadResponse, ImageMeta as ImageMetaSchema


router = APIRouter(prefix="/api", tags=["upload"])

# In-memory store for uploaded images (replaced with proper storage later)
_image_store: dict[str, dict] = {}


@router.post("/upload", response_model=UploadResponse)
async def upload_image(file: UploadFile = File(...)):
    """
    Upload an image for restoration.

    - Validates file type, size, and dimensions
    - Converts to RGB
    - Assigns a unique image_id
    - Returns metadata about the uploaded image
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    # Read file bytes
    try:
        file_bytes = await file.read()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read file: {e}")

    if len(file_bytes) == 0:
        raise HTTPException(status_code=400, detail="Empty file")

    # Validate and load
    try:
        img, meta = load_image_from_bytes(file_bytes, filename=file.filename)
    except ImageValidationError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error loading image: {e}")
        raise HTTPException(status_code=500, detail="Internal error processing image")

    # Resize for inference if needed (configurable max)
    img_for_inference, was_resized = resize_if_needed(img, max_size=1024)

    # Generate unique ID
    image_id = str(uuid.uuid4())[:8]

    # Store in memory (the PIL image object for later pipeline use)
    _image_store[image_id] = {
        "original": img,
        "for_inference": img_for_inference,
        "was_resized": was_resized,
        "meta": meta,
    }

    logger.info(f"Image uploaded: id={image_id}, file={file.filename}")

    return UploadResponse(
        image_id=image_id,
        meta=ImageMetaSchema(
            width=meta.width,
            height=meta.height,
            channels=meta.channels,
            file_size_bytes=meta.file_size_bytes,
            format=meta.format,
            filename=meta.filename,
        ),
    )


def get_stored_image(image_id: str) -> dict:
    """Retrieve a stored image by its ID. Used by other routes."""
    if image_id not in _image_store:
        raise HTTPException(status_code=404, detail=f"Image not found: {image_id}")
    return _image_store[image_id]
