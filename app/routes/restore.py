"""
Restoration Route — Handles end-to-end image restoration and image serving.
Supports Default, OSDFace, and Both restoration modes.
"""
import io
import time
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
from core.pipeline.engine import RestorationEngine, _compute_basic_metrics
from core.pipeline.image_utils import image_to_bytes
from core.osdface_processor import OSDFaceProcessor


router = APIRouter(prefix="/api", tags=["Restoration"])

analyzer_instance = DegradationAnalyzer()
planner_instance = PipelinePlanner()
engine_instance = RestorationEngine()
osdface_processor_instance = OSDFaceProcessor()


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
    restoration_mode: str = Field(
        default="default",
        description="Restoration mode: 'default', 'osdface', or 'both'"
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
    Run end-to-end image restoration supporting Default, OSDFace, and Both modes.
    """
    stored = get_stored_image(req.image_id)
    if not stored:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Image with ID '{req.image_id}' not found. Please upload first.",
        )

    pil_img = stored.get("for_inference") or stored["original"]
    orig_meta = stored["meta"]
    original_img = stored["original"]

    mode = req.restoration_mode.lower().strip() if req.restoration_mode else "default"
    logger.info(f"Processing restoration request for image_id='{req.image_id}' ({pil_img.width}x{pil_img.height}), mode='{mode}', upscale_4k={req.upscale_4k}")

    # =========================================================================
    # MODE 1: DEFAULT (Standard Degradation Analyzer + 3-Pass Engine)
    # =========================================================================
    if mode == "default":
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

        pipeline_steps = engine_result.pipeline_steps
        for step in pipeline_steps:
            if step.get("is_accepted") and "output_image" in step:
                step_img = step.pop("output_image")
                inter_id = f"intermediate_{req.image_id}_step_{step['step_number']}"
                _image_store[inter_id] = {
                    "original": step_img,
                    "for_inference": step_img,
                    "was_resized": False,
                    "meta": ImageMetaSchema(
                        width=step_img.width,
                        height=step_img.height,
                        channels=3,
                        file_size_bytes=0,
                        format=orig_meta.format or "PNG",
                        filename=f"intermediate_step_{step['step_number']}.png",
                    ),
                }
                step["image_url"] = f"/api/image/{inter_id}"
            elif "output_image" in step:
                step.pop("output_image")

        return RestorationResponse(
            success=True,
            image_id=req.image_id,
            restoration_mode="default",
            original_meta=ImageMetaSchema(
                width=orig_meta.width,
                height=orig_meta.height,
                channels=orig_meta.channels,
                file_size_bytes=orig_meta.file_size_bytes,
                format=orig_meta.format,
                filename=orig_meta.filename,
            ),
            restored_meta=restored_meta_obj,
            degradations=analysis_report.degradations if analysis_report else [],
            pipeline_steps=engine_result.pipeline_steps,
            metrics=engine_result.metrics,
            inference_time_seconds=engine_result.total_time_seconds,
        )

    # =========================================================================
    # MODE 2: OSDFACE (YuNet -> OSDFace -> Blend, or Fallback if 0 faces)
    # =========================================================================
    elif mode == "osdface":
        t0 = time.time()
        osd_out, face_count, face_restored = osdface_processor_instance.process(original_img)

        # IF NO FACE: Fallback to default pipeline
        if not face_restored or face_count == 0:
            logger.info("[OSDFACE] Fallback to DEFAULT pipeline because no face was detected.")
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
            dt = round(time.time() - t0, 3)
            restored_bytes = image_to_bytes(restored_img, fmt=orig_meta.format or "PNG")
            restored_meta_obj = ImageMetaSchema(
                width=restored_img.width,
                height=restored_img.height,
                channels=3,
                file_size_bytes=len(restored_bytes),
                format=orig_meta.format or "PNG",
                filename=f"restored_{orig_meta.filename}",
            )
            _image_store[f"restored_{req.image_id}"] = {
                "original": restored_img,
                "meta": restored_meta_obj,
            }
            pipeline_steps = engine_result.pipeline_steps
            for step in pipeline_steps:
                if step.get("is_accepted") and "output_image" in step:
                    step_img = step.pop("output_image")
                    inter_id = f"intermediate_{req.image_id}_step_{step['step_number']}"
                    _image_store[inter_id] = {
                        "original": step_img,
                        "for_inference": step_img,
                        "was_resized": False,
                        "meta": ImageMetaSchema(
                            width=step_img.width,
                            height=step_img.height,
                            channels=3,
                            file_size_bytes=0,
                            format=orig_meta.format or "PNG",
                            filename=f"intermediate_step_{step['step_number']}.png",
                        ),
                    }
                    step["image_url"] = f"/api/image/{inter_id}"
                elif "output_image" in step:
                    step.pop("output_image")

            return RestorationResponse(
                success=True,
                image_id=req.image_id,
                restoration_mode="osdface",
                original_meta=ImageMetaSchema(
                    width=orig_meta.width,
                    height=orig_meta.height,
                    channels=orig_meta.channels,
                    file_size_bytes=orig_meta.file_size_bytes,
                    format=orig_meta.format,
                    filename=orig_meta.filename,
                ),
                restored_meta=restored_meta_obj,
                degradations=engine_result.analysis_report.degradations if engine_result.analysis_report else [],
                pipeline_steps=pipeline_steps + [{
                    "step_number": 99,
                    "operation": "osdface_check",
                    "model_name": "OSDFace",
                    "faces_detected": 0,
                    "status": "SKIPPED — Fallback to Default Pipeline",
                    "execution_time_seconds": dt
                }],
                metrics=engine_result.metrics,
                inference_time_seconds=dt,
            )

        # IF FACE DETECTED: Return OSDFace restored image
        dt = round(time.time() - t0, 3)
        restored_img = osd_out if isinstance(osd_out, Image.Image) else Image.fromarray(osd_out[:, :, ::-1])
        restored_bytes = image_to_bytes(restored_img, fmt=orig_meta.format or "PNG")
        restored_meta_obj = ImageMetaSchema(
            width=restored_img.width,
            height=restored_img.height,
            channels=3,
            file_size_bytes=len(restored_bytes),
            format=orig_meta.format or "PNG",
            filename=f"osdface_{orig_meta.filename}",
        )

        _image_store[f"restored_{req.image_id}"] = {
            "original": restored_img,
            "meta": restored_meta_obj,
        }

        osd_metrics = _compute_basic_metrics(original_img, restored_img)
        osd_metrics.update({
            "faces_detected": face_count,
            "osdface_inference_mode": "One-step diffusion",
            "model_name": "OSDFace (SD 2.1 Base)",
            "real_weights_loaded": True,
        })

        osd_step = {
            "step_number": 1,
            "pass_number": 1,
            "operation": "face_restoration",
            "model_name": "OSDFace",
            "faces_detected": face_count,
            "execution_time_seconds": dt,
            "input_size": f"{original_img.width}x{original_img.height}",
            "output_size": f"{restored_img.width}x{restored_img.height}",
            "is_accepted": True,
        }

        return RestorationResponse(
            success=True,
            image_id=req.image_id,
            restoration_mode="osdface",
            original_meta=ImageMetaSchema(
                width=orig_meta.width,
                height=orig_meta.height,
                channels=orig_meta.channels,
                file_size_bytes=orig_meta.file_size_bytes,
                format=orig_meta.format,
                filename=orig_meta.filename,
            ),
            restored_meta=restored_meta_obj,
            degradations=[],
            pipeline_steps=[osd_step],
            metrics=osd_metrics,
            inference_time_seconds=dt,
        )

    # =========================================================================
    # MODE 3: BOTH (Independent Branch A [Default] and Branch B [OSDFace])
    # =========================================================================
    elif mode == "both":
        t_start_both = time.time()

        # Branch A: Default Restoration Pipeline
        logger.info("[BOTH MODE] Running Branch A: Standard Restoration Pipeline...")
        engine_result_a = engine_instance.run_two_pass(
            image=pil_img,
            analyzer=analyzer_instance,
            planner=planner_instance,
            custom_operations=req.custom_operations,
            upscale_4k=req.upscale_4k,
            original_image=original_img,
            image_id=req.image_id,
        )
        restored_a = engine_result_a.final_image

        restored_a_bytes = image_to_bytes(restored_a, fmt=orig_meta.format or "PNG")
        meta_a = ImageMetaSchema(
            width=restored_a.width,
            height=restored_a.height,
            channels=3,
            file_size_bytes=len(restored_a_bytes),
            format=orig_meta.format or "PNG",
            filename=f"default_{orig_meta.filename}",
        )
        _image_store[f"restored_{req.image_id}"] = {
            "original": restored_a,
            "meta": meta_a,
        }

        # Branch B: OSDFace
        logger.info("[BOTH MODE] Running Branch B: OSDFace Restoration...")
        tb0 = time.time()
        osd_out_b, face_count_b, face_restored_b = osdface_processor_instance.process(original_img)
        dt_b = round(time.time() - tb0, 3)

        osd_b_id = f"osdface_{req.image_id}"
        if face_restored_b and face_count_b > 0:
            restored_b = osd_out_b if isinstance(osd_out_b, Image.Image) else Image.fromarray(osd_out_b[:, :, ::-1])
            bytes_b = image_to_bytes(restored_b, fmt=orig_meta.format or "PNG")
            meta_b = ImageMetaSchema(
                width=restored_b.width,
                height=restored_b.height,
                channels=3,
                file_size_bytes=len(bytes_b),
                format=orig_meta.format or "PNG",
                filename=f"osdface_{orig_meta.filename}",
            )
            _image_store[osd_b_id] = {
                "original": restored_b,
                "meta": meta_b,
            }
            metrics_b = _compute_basic_metrics(original_img, restored_b)
            osdface_payload = {
                "success": True,
                "skipped": False,
                "image_id": osd_b_id,
                "image_url": f"/api/image/{osd_b_id}",
                "faces_detected": face_count_b,
                "inference_mode": "One-step diffusion",
                "model_name": "OSDFace",
                "execution_time_seconds": dt_b,
                "meta": meta_b.model_dump(),
                "metrics": metrics_b,
            }
        else:
            osdface_payload = {
                "success": True,
                "skipped": True,
                "reason": "No face detected by YuNet detector",
                "faces_detected": 0,
                "execution_time_seconds": dt_b,
            }

        dt_total = round(time.time() - t_start_both, 3)

        pipeline_steps_a = engine_result_a.pipeline_steps
        for step in pipeline_steps_a:
            if step.get("is_accepted") and "output_image" in step:
                step_img = step.pop("output_image")
                inter_id = f"intermediate_{req.image_id}_step_{step['step_number']}"
                _image_store[inter_id] = {
                    "original": step_img,
                    "for_inference": step_img,
                    "was_resized": False,
                    "meta": ImageMetaSchema(
                        width=step_img.width,
                        height=step_img.height,
                        channels=3,
                        file_size_bytes=0,
                        format=orig_meta.format or "PNG",
                        filename=f"intermediate_step_{step['step_number']}.png",
                    ),
                }
                step["image_url"] = f"/api/image/{inter_id}"
            elif "output_image" in step:
                step.pop("output_image")

        return RestorationResponse(
            success=True,
            image_id=req.image_id,
            restoration_mode="both",
            original_meta=ImageMetaSchema(
                width=orig_meta.width,
                height=orig_meta.height,
                channels=orig_meta.channels,
                file_size_bytes=orig_meta.file_size_bytes,
                format=orig_meta.format,
                filename=orig_meta.filename,
            ),
            restored_meta=meta_a,
            degradations=engine_result_a.analysis_report.degradations if engine_result_a.analysis_report else [],
            pipeline_steps=pipeline_steps_a,
            metrics=engine_result_a.metrics,
            inference_time_seconds=dt_total,
            osdface_result=osdface_payload,
        )

    else:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid restoration_mode '{req.restoration_mode}'. Must be 'default', 'osdface', or 'both'."
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
