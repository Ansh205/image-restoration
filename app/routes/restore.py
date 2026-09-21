"""
Restoration Route — Handles end-to-end image restoration and image serving.
"""
import io
import uuid
from typing import List, Optional

from fastapi import APIRouter, HTTPException, status, Response
from pydantic import BaseModel, Field
from loguru import logger
from PIL import Image

from app.schemas.image import (
    RestorationResponse,
    ImageMeta as ImageMetaSchema,
    ErrorResponse,
)
from app.routes.upload import get_stored_image, _image_store
from core.analyzer.analyzer import DegradationAnalyzer
from core.pipeline.planner import PipelinePlanner
from core.pipeline.engine import RestorationEngine
from core.pipeline.image_utils import image_to_bytes


router = APIRouter(prefix="/api", tags=["Restoration"])

analyzer_instance = DegradationAnalyzer()
planner_instance = PipelinePlanner()
engine_instance = RestorationEngine()


class RestoreRequest(BaseModel):
    image_id: str = Field(..., description="Unique image identifier returned from /api/upload")
    custom_operations: Optional[List[str]] = Field(
        default=None,
        description="Optional explicit list of operations to run (overrides automatic planner)"
    )
    upscale_4k: bool = Field(
        default=False,
        description="Enable 4K AI Upscaling (Real-ESRGAN capped at 3840px max dimension)"
    )


@router.post(
    "/restore",
    response_model=RestorationResponse,
    responses={
        404: {"model": ErrorResponse, "description": "Image ID not found"},
        422: {"model": ErrorResponse, "description": "Validation error"},
    },
)
async def restore_image(req: RestoreRequest) -> RestorationResponse:
    """
    Run end-to-end degradation analysis and pipeline restoration on an uploaded image.
    """
    stored = get_stored_image(req.image_id)
    if not stored:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Image with ID '{req.image_id}' not found. Please upload first.",
        )

    pil_img = stored.get("for_inference") or stored["original"]
    orig_meta = stored["meta"]

    logger.info(f"Processing restoration request for image_id='{req.image_id}' ({pil_img.width}x{pil_img.height}), upscale_4k={req.upscale_4k}")

    # Execute two-pass restoration engine
    original_img = stored["original"]
    engine_result = engine_instance.run_two_pass(
        image=pil_img,
        analyzer=analyzer_instance,
        planner=planner_instance,
        custom_operations=req.custom_operations,
        upscale_4k=req.upscale_4k,
        original_image=original_img,
        image_id=req.image_id,
    )
    restored_img = engine_result.final_image
    analysis_report = engine_result.analysis_report

    # 4. Generate restored image metadata & store in memory
    restored_bytes = image_to_bytes(restored_img, fmt=orig_meta.format or "PNG")
    restored_meta_obj = ImageMetaSchema(
        width=restored_img.width,
        height=restored_img.height,
        channels=3,
        file_size_bytes=len(restored_bytes),
        format=orig_meta.format or "PNG",
        filename=f"restored_{orig_meta.filename}",
    )

    restored_id = f"restored_{req.image_id}"
    _image_store[restored_id] = {
        "original": restored_img,
        "for_inference": restored_img,
        "was_resized": False,
        "meta": restored_meta_obj,
    }

    return RestorationResponse(
        success=True,
        image_id=req.image_id,
        original_meta=ImageMetaSchema(
            width=orig_meta.width,
            height=orig_meta.height,
            channels=orig_meta.channels,
            file_size_bytes=orig_meta.file_size_bytes,
            format=orig_meta.format,
            filename=orig_meta.filename,
        ),
        restored_meta=restored_meta_obj,
        degradations=analysis_report.degradations,
        pipeline_steps=engine_result.pipeline_steps,
        metrics=engine_result.metrics,
        inference_time_seconds=engine_result.total_time_seconds,
    )


@router.get("/image/{image_id}")
async def serve_image(image_id: str):
    """
    Serve raw image bytes for preview and download.
    """
    stored = get_stored_image(image_id)
    if not stored:
        raise HTTPException(status_code=404, detail="Image not found")

    pil_img = stored.get("original") or stored.get("for_inference")
    img_bytes = image_to_bytes(pil_img, fmt="PNG")
    return Response(content=img_bytes, media_type="image/png")
