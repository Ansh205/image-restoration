"""
Analysis API Endpoint Route.

POST /api/analyze — Accepts an image_id, runs DegradationAnalyzer,
and returns a structured degradation report.
"""
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field
from loguru import logger

from app.schemas.image import AnalysisResponse, ErrorResponse
from app.routes.upload import get_stored_image
from core.analyzer.analyzer import DegradationAnalyzer


router = APIRouter(prefix="/api", tags=["Analysis"])
analyzer_instance = DegradationAnalyzer()


class AnalyzeRequest(BaseModel):
    image_id: str = Field(..., description="Unique image identifier returned from /api/upload")


@router.post(
    "/analyze",
    response_model=AnalysisResponse,
    responses={
        404: {"model": ErrorResponse, "description": "Image ID not found"},
        422: {"model": ErrorResponse, "description": "Validation error"},
    },
)
async def analyze_image(req: AnalyzeRequest) -> AnalysisResponse:
    """
    Run degradation analysis on a previously uploaded image.

    - **image_id**: UUID returned during upload.
    """
    stored = get_stored_image(req.image_id)
    if not stored:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Image with ID '{req.image_id}' not found. Please upload first.",
        )

    pil_img = stored.get("for_inference") or stored["original"]
    logger.info(f"Analyzing image '{req.image_id}' ({pil_img.width}x{pil_img.height})")

    report = analyzer_instance.analyze(pil_img, image_id=req.image_id)
    return report
