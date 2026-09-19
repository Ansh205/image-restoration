"""
Pydantic schemas for image upload request/response validation.
"""
from pydantic import BaseModel, Field


class ImageMeta(BaseModel):
    """Metadata returned after loading an image."""
    width: int = Field(..., description="Image width in pixels")
    height: int = Field(..., description="Image height in pixels")
    channels: int = Field(default=3, description="Number of color channels")
    file_size_bytes: int = Field(..., description="Original file size in bytes")
    format: str | None = Field(default=None, description="Detected image format")
    filename: str = Field(..., description="Original filename")


class UploadResponse(BaseModel):
    """Response after successfully uploading an image."""
    success: bool = True
    message: str = "Image uploaded and validated successfully"
    image_id: str = Field(..., description="Unique identifier for this upload")
    meta: ImageMeta


class ErrorResponse(BaseModel):
    """Error response for failed operations."""
    success: bool = False
    error: str = Field(..., description="Error message")
    detail: str | None = Field(default=None, description="Additional error details")


class DegradationItem(BaseModel):
    """A single degradation detection result."""
    name: str = Field(..., description="Degradation type (e.g., 'noise', 'blur')")
    score: float = Field(..., ge=0.0, le=1.0, description="Raw severity score 0-1")
    severity: str = Field(..., description="Severity level: LOW, MEDIUM, or HIGH")
    confidence: float | None = Field(default=None, description="Heuristic confidence score 0-1")
    details: dict = Field(default_factory=dict, description="Detector-specific info")


class AnalysisResponse(BaseModel):
    """Response containing the degradation analysis."""
    image_id: str = Field(default="", description="Unique image identifier")
    degradations: list[DegradationItem]
    pipeline: list[str] = Field(
        default_factory=list,
        description="Ordered list of operations to apply"
    )
    raw_metrics: dict = Field(
        default_factory=dict,
        description="Raw numerical values from detectors"
    )


class RestorationResponse(BaseModel):
    """Response after restoration is complete."""
    success: bool = True
    image_id: str
    original_meta: ImageMeta
    restored_meta: ImageMeta
    degradations: list[DegradationItem]
    pipeline_steps: list[dict] = Field(
        default_factory=list,
        description="List of executed pipeline steps with model names"
    )
    metrics: dict = Field(
        default_factory=dict,
        description="Before/after quality indicators"
    )
    inference_time_seconds: float = Field(..., description="Total inference time")
