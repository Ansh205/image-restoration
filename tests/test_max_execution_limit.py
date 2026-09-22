"""
Unit tests for Global Max 2 Execution Limit & Deblur Over-Sharpening DISCARD behavior.
"""
import pytest
import numpy as np
from PIL import Image

from core.evaluator.restoration_evaluator import RestorationEvaluator
from core.pipeline.engine import RestorationEngine
from core.analyzer.analyzer import DegradationAnalyzer
from core.pipeline.planner import PipelinePlanner


def test_deblur_over_sharpening_discard():
    evaluator = RestorationEvaluator()

    # Smooth image before
    arr_before = np.full((100, 100, 3), 100, dtype=np.uint8)
    arr_before[40:60, 40:60] = 120
    img_before = Image.fromarray(arr_before)

    # Extremely high contrast edge (sharpened +300%+) + moderate noise amplification (+100%)
    arr_after = np.full((100, 100, 3), 100, dtype=np.uint8)
    arr_after[40:60, 40:60] = 240
    noise = np.random.normal(0, 4, (100, 100, 3)).astype(np.int16)
    arr_after = np.clip(arr_after.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    img_after = Image.fromarray(arr_after)

    res = evaluator.evaluate("deblur", img_before, img_after, pass_number=1, op_run_count=1)

    assert res.decision == "DISCARD"
    assert "over-sharpening" in res.reason.lower()


def test_max_execution_limit_enforced_in_3pass():
    engine = RestorationEngine()
    analyzer = DegradationAnalyzer()
    planner = PipelinePlanner()

    # Image with severe blur
    img_arr = np.random.randint(100, 120, (100, 100, 3), dtype=np.uint8)
    img_arr[30:70, 30:70] = 180
    test_img = Image.fromarray(img_arr)

    # Force deblur in all 3 passes using custom_operations
    res = engine.run_three_pass(
        image=test_img,
        analyzer=analyzer,
        planner=planner,
        custom_operations=["deblur"],
    )

    deblur_steps = [s for s in res.pipeline_steps if s["operation"] == "deblur"]
    skipped_steps = [s for s in res.metrics.get("operation_history", []) if s.get("effect") == "SKIPPED"]

    # Deblur should execute AT MOST 2 times
    assert len(deblur_steps) <= 2
    # The 3rd attempt (or subsequent attempts if skipped) must be skipped
    assert any("Maximum execution limit reached" in s.get("reason", "") or "Blocked" in s.get("reason", "") or "Not retried" in s.get("reason", "") for s in skipped_steps)
