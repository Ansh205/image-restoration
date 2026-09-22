"""
Unit tests for RestorationEvaluator Global Hard Safety Gates & Operation Rules.
"""
import pytest
import numpy as np
from PIL import Image

from core.evaluator.restoration_evaluator import RestorationEvaluator


def test_catastrophic_luminance_drop_gate():
    evaluator = RestorationEvaluator()

    # Image before: medium luminance (~0.28)
    arr_before = np.full((100, 100, 3), 72, dtype=np.uint8)  # 72/255 = 0.282
    img_before = Image.fromarray(arr_before)

    # Image after: very dark/near-black (~0.058) - 79% drop
    arr_after = np.full((100, 100, 3), 15, dtype=np.uint8)   # 15/255 = 0.0588
    img_after = Image.fromarray(arr_after)

    # Evaluate under Pass 3 denoise (the exact scenario where earlier evaluator failed)
    res = evaluator.evaluate("denoise", img_before, img_after, pass_number=3)

    assert res.decision == "DISCARD"
    assert res.hard_safety_passed is False
    assert "HARD SAFETY FAILURE" in res.reason
    assert "luminance" in res.reason.lower()


def test_low_light_darkening_discard():
    evaluator = RestorationEvaluator()

    # Low light input: 0.3246 luminance
    arr_before = np.full((100, 100, 3), 83, dtype=np.uint8)
    img_before = Image.fromarray(arr_before)

    # Output: 0.2825 luminance (darker)
    arr_after = np.full((100, 100, 3), 72, dtype=np.uint8)
    img_after = Image.fromarray(arr_after)

    # Even in Pass 1 (permissive), low-light making image darker MUST be discarded!
    res = evaluator.evaluate("low_light", img_before, img_after, pass_number=1)

    assert res.decision == "DISCARD"
    assert "darker" in res.reason.lower()


def test_denoise_luminance_drop_discard():
    evaluator = RestorationEvaluator()

    arr_before = np.full((100, 100, 3), 120, dtype=np.uint8)
    img_before = Image.fromarray(arr_before)

    # 15% drop (120 -> 100)
    arr_after = np.full((100, 100, 3), 100, dtype=np.uint8)
    img_after = Image.fromarray(arr_after)

    res = evaluator.evaluate("denoise", img_before, img_after, pass_number=1)

    assert res.decision == "DISCARD"
    assert "luminance" in res.reason.lower()
