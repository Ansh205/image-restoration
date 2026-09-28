"""
Restoration Route — Handles end-to-end image restoration and image serving.
Supports Default, OSDFace-Assisted, and Both restoration modes.
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
from core.osdface_pipeline import OSDFaceAssistedPipeline


router = APIRouter(prefix="/api", tags=["Restoration"])

analyzer_instance = DegradationAnalyzer()
planner_instance = PipelinePlanner()
engine_instance = RestorationEngine()
osdface_assisted_pipeline_instance = OSDFaceAssistedPipeline()


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
    Run end-to-end image restoration supporting Default, OSDFace-Assisted, and Both modes.
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
        engine_result = engine_instance.run_three_pass(
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

        t_std_dur = round(engine_result.total_time_seconds, 3)
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
            inference_time_seconds=t_std_dur,
            standard_execution_time=t_std_dur,
            osdface_assisted_execution_time=0.0,
            total_execution_time=t_std_dur,
            standard_metadata={
                "result_available": True,
                "operations": [s["operation"] for s in engine_result.pipeline_steps if s.get("is_accepted")],
                "execution_time": t_std_dur,
            },
        )

    # =========================================================================
    # MODE 2: OSDFACE ASSISTED (OSDFace -> Evaluator -> Default 3-Pass Pipeline)
    # =========================================================================
    elif mode == "osdface":
        t_osd_start = time.perf_counter()
        engine_result, osdface_meta, osd_candidate_pil, osd_decision = osdface_assisted_pipeline_instance.run(
            image=pil_img,
            original_image=original_img,
            image_id=req.image_id,
            analyzer=analyzer_instance,
            planner=planner_instance,
            engine=engine_instance,
            custom_operations=req.custom_operations,
            upscale_4k=req.upscale_4k,
        )
        t_osd_dur = round(time.perf_counter() - t_osd_start, 3)
        t_std_dur = round(engine_result.total_time_seconds, 3)

        restored_img = engine_result.final_image
        analysis_report = engine_result.analysis_report

        restored_bytes = image_to_bytes(restored_img, fmt=orig_meta.format or "PNG")
        restored_meta_obj = ImageMetaSchema(
            width=restored_img.width,
            height=restored_img.height,
            channels=3,
            file_size_bytes=len(restored_bytes),
            format=orig_meta.format or "PNG",
            filename=f"osdface_assisted_{orig_meta.filename}",
        )

        restored_id = f"restored_{req.image_id}"
        _image_store[restored_id] = {
            "original": restored_img,
            "for_inference": restored_img,
            "was_resized": False,
            "meta": restored_meta_obj,
        }

        pipeline_steps = engine_result.pipeline_steps

        # Store intermediate OSDFace candidate if generated & accepted
        if osd_candidate_pil is not None:
            inter_osd_id = f"intermediate_{req.image_id}_osdface"
            _image_store[inter_osd_id] = {
                "original": osd_candidate_pil,
                "for_inference": osd_candidate_pil,
                "was_resized": False,
                "meta": ImageMetaSchema(
                    width=osd_candidate_pil.width,
                    height=osd_candidate_pil.height,
                    channels=3,
                    file_size_bytes=0,
                    format=orig_meta.format or "PNG",
                    filename="osdface_prerestoration.png",
                ),
            }
            osd_step = {
                "step_number": 0,
                "pass_number": 0,
                "operation": "osdface_prerestoration",
                "model_name": "OSDFace (One-Step Diffusion)",
                "evaluator_decision": osd_decision,
                "evaluator_reason": osdface_meta.get("evaluation_reason", ""),
                "faces_detected": osdface_meta.get("faces_detected", 0),
                "execution_time_seconds": osdface_meta.get("inference_time", 0.0),
                "is_accepted": (osd_decision == "KEEP"),
                "image_url": f"/api/image/{inter_osd_id}",
            }
            pipeline_steps = [osd_step] + pipeline_steps

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
            degradations=analysis_report.degradations if analysis_report else [],
            pipeline_steps=pipeline_steps,
            metrics=engine_result.metrics,
            inference_time_seconds=t_osd_dur,
            standard_execution_time=t_std_dur,
            osdface_assisted_execution_time=t_osd_dur,
            total_execution_time=t_osd_dur,
            osdface_metadata=osdface_meta,
            standard_metadata={
                "result_available": True,
                "operations": [s["operation"] for s in pipeline_steps if s.get("is_accepted")],
                "execution_time": t_std_dur,
            },
        )

    # =========================================================================
    # MODE 3: BOTH (Independent Branch A [Default] and Branch B [OSDFace-Assisted])
    # =========================================================================
    elif mode == "both":
        t_start_both = time.perf_counter()

        # Branch A: Standard Restoration Pipeline
        logger.info("[BOTH MODE] Running Branch A: Standard Restoration Pipeline...")
        t_std_start = time.perf_counter()
        engine_result_a = engine_instance.run_three_pass(
            image=pil_img,
            analyzer=analyzer_instance,
            planner=planner_instance,
            custom_operations=req.custom_operations,
            upscale_4k=req.upscale_4k,
            original_image=original_img,
            image_id=req.image_id,
        )
        t_std_dur = round(time.perf_counter() - t_std_start, 3)
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

        # Branch B: OSDFace-Assisted Pipeline
        logger.info("[BOTH MODE] Running Branch B: OSDFace-Assisted Restoration Pipeline...")
        t_osd_start = time.perf_counter()
        engine_result_b, osdface_meta_b, osd_candidate_pil_b, osd_decision_b = osdface_assisted_pipeline_instance.run(
            image=pil_img,
            original_image=original_img,
            image_id=req.image_id,
            analyzer=analyzer_instance,
            planner=planner_instance,
            engine=engine_instance,
            custom_operations=req.custom_operations,
            upscale_4k=req.upscale_4k,
        )
        t_osd_dur = round(time.perf_counter() - t_osd_start, 3)
        restored_b = engine_result_b.final_image
        pipeline_steps_b = engine_result_b.pipeline_steps

        if osd_candidate_pil_b is not None:
            inter_osd_id_b = f"intermediate_{req.image_id}_osdface_b"
            _image_store[inter_osd_id_b] = {
                "original": osd_candidate_pil_b,
                "for_inference": osd_candidate_pil_b,
                "was_resized": False,
                "meta": ImageMetaSchema(
                    width=osd_candidate_pil_b.width,
                    height=osd_candidate_pil_b.height,
                    channels=3,
                    file_size_bytes=0,
                    format=orig_meta.format or "PNG",
                    filename="osdface_prerestoration_b.png",
                ),
            }
            osd_step_b = {
                "step_number": 0,
                "pass_number": 0,
                "operation": "osdface_prerestoration",
                "model_name": "OSDFace (One-Step Diffusion)",
                "evaluator_decision": osd_decision_b,
                "evaluator_reason": osdface_meta_b.get("evaluation_reason", ""),
                "faces_detected": osdface_meta_b.get("faces_detected", 0),
                "execution_time_seconds": osdface_meta_b.get("inference_time", 0.0),
                "is_accepted": (osd_decision_b == "KEEP"),
                "image_url": f"/api/image/{inter_osd_id_b}",
            }
            pipeline_steps_b = [osd_step_b] + pipeline_steps_b

        for step in pipeline_steps_b:
            if step.get("is_accepted") and "output_image" in step:
                step_img = step.pop("output_image")
                inter_id = f"intermediate_{req.image_id}_b_step_{step['step_number']}"
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
                        filename=f"intermediate_step_b_{step['step_number']}.png",
                    ),
                }
                step["image_url"] = f"/api/image/{inter_id}"
            elif "output_image" in step:
                step.pop("output_image")

        osd_b_id = f"osdface_assisted_{req.image_id}"
        bytes_b = image_to_bytes(restored_b, fmt=orig_meta.format or "PNG")
        meta_b = ImageMetaSchema(
            width=restored_b.width,
            height=restored_b.height,
            channels=3,
            file_size_bytes=len(bytes_b),
            format=orig_meta.format or "PNG",
            filename=f"osdface_assisted_{orig_meta.filename}",
        )
        _image_store[osd_b_id] = {
            "original": restored_b,
            "meta": meta_b,
        }

        osdface_payload = {
            "success": True,
            "skipped": (osd_decision_b == "SKIPPED"),
            "image_id": osd_b_id,
            "image_url": f"/api/image/{osd_b_id}",
            "faces_detected": osdface_meta_b.get("faces_detected", 0),
            "evaluation_decision": osd_decision_b,
            "evaluation_reason": osdface_meta_b.get("evaluation_reason", ""),
            "inference_mode": "One-step diffusion + Standard 3-pass",
            "model_name": "OSDFace + Standard Pipeline",
            "execution_time_seconds": t_osd_dur,
            "meta": meta_b.model_dump(),
            "metrics": engine_result_b.metrics,
            "pipeline_steps": pipeline_steps_b,
            "osdface_metadata": osdface_meta_b,
        }

        t_total_dur = round(time.perf_counter() - t_start_both, 3)

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
            inference_time_seconds=t_total_dur,
            standard_execution_time=t_std_dur,
            osdface_assisted_execution_time=t_osd_dur,
            total_execution_time=t_total_dur,
            osdface_metadata=osdface_meta_b,
            standard_metadata={
                "result_available": True,
                "operations": [s["operation"] for s in pipeline_steps_a if s.get("is_accepted")],
                "execution_time": t_std_dur,
            },
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
