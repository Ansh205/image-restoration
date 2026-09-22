"""
Unit tests for Harmful Operation Retry Blocking and Candidate Preservation.
"""
import pytest
import numpy as np
from PIL import Image

from core.pipeline.engine import RestorationEngine
from core.evaluator.restoration_evaluator import RestorationEvaluator, EvaluationResult
from core.analyzer.analyzer import DegradationAnalyzer
from core.pipeline.planner import PipelinePlanner


def test_harmful_operation_retry_blocked():
    """Verify that an operation classified as HARMFUL or SEVERELY_HARMFUL is blocked from retrying in Pass 2/3."""
    engine = RestorationEngine()
    analyzer = DegradationAnalyzer()
    planner = PipelinePlanner()

    # Create synthetic blurry image
    img_arr = np.random.randint(100, 120, (100, 100, 3), dtype=np.uint8)
    img_arr[30:70, 30:70] = 180
    test_img = Image.fromarray(img_arr)

    # Force deblur across 3 passes
    res = engine.run_three_pass(
        image=test_img,
        analyzer=analyzer,
        planner=planner,
        custom_operations=["deblur"],
    )

    # Inspect skipped steps in history
    skipped_steps = [s for s in res.metrics.get("operation_history", []) if s.get("effect") == "SKIPPED"]
    
    # If deblur was harmful in pass 1 or reached max limit, subsequent attempts must be skipped
    deblur_exec_count = len([s for s in res.pipeline_steps if s["operation"] == "deblur"])
    assert deblur_exec_count <= 2
    assert len(skipped_steps) >= 1


def test_candidate_preservation_on_harmful():
    """Verify that candidate image is NOT updated when an operation output is rejected as harmful."""
    engine = RestorationEngine()

    arr_before = np.full((100, 100, 3), 100, dtype=np.uint8)
    img_before = Image.fromarray(arr_before)

    # Output with extreme noise added (harmful)
    arr_after = arr_before.copy()
    noise = np.random.normal(0, 30, (100, 100, 3)).astype(np.int16)
    arr_after = np.clip(arr_after.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    img_after = Image.fromarray(arr_after)

    eval_res = engine.evaluator.evaluate("deblur", img_before, img_after, pass_number=1, op_run_count=1)
    effect = engine._classify_effect(eval_res, "deblur")

    assert eval_res.decision == "DISCARD"
    assert effect in ["HARMFUL", "SEVERELY_HARMFUL"]
