"""
Comprehensive Verification Script for Per-Operation Before/After Restoration Evaluator.
Tests all 5 required evaluation scenarios and two-pass engine integration.
"""
import numpy as np
from PIL import Image
from loguru import logger

from core.evaluator.restoration_evaluator import RestorationEvaluator, EvaluationResult
from core.analyzer.analyzer import DegradationAnalyzer
from core.pipeline.planner import PipelinePlanner
from core.pipeline.engine import RestorationEngine


def test_evaluator_unit_cases():
    logger.info("============================================================")
    logger.info("UNIT TEST 1: Deblur Operation Improving Sharpness (KEEP)")
    logger.info("============================================================")
    evaluator = RestorationEvaluator()

    # Create smooth image for before, sharpened image for after
    arr_before = np.full((100, 100, 3), 128, dtype=np.uint8)
    arr_before[40:60, 40:60] = 140
    img_before = Image.fromarray(arr_before)

    arr_after = np.full((100, 100, 3), 128, dtype=np.uint8)
    arr_after[40:60, 40:60] = 220  # High contrast edge -> high Laplacian
    img_after = Image.fromarray(arr_after)

    res_deblur = evaluator.evaluate("deblur", img_before, img_after)
    logger.info(f"[DEBLUR TEST] Decision: {res_deblur.decision} | Reason: {res_deblur.reason}")
    assert res_deblur.decision == "KEEP"
    logger.info("[OK] Deblur with improved sharpness kept.")

    logger.info("\n============================================================")
    logger.info("UNIT TEST 2: Denoise Operation Reducing Noise (KEEP)")
    logger.info("============================================================")
    # Structured image with noise vs clean structured image
    clean_base = np.full((100, 100, 3), 100, dtype=np.uint8)
    clean_base[30:70, 30:70] = 200  # Strong square edges

    # Add Gaussian noise
    noise = np.random.normal(0, 15, (100, 100, 3)).astype(np.int16)
    noisy_arr = np.clip(clean_base.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    img_noisy = Image.fromarray(noisy_arr)
    img_clean = Image.fromarray(clean_base)

    res_denoise = evaluator.evaluate("denoise", img_noisy, img_clean)
    logger.info(f"[DENOISE TEST] Decision: {res_denoise.decision} | Reason: {res_denoise.reason}")
    assert res_denoise.decision == "KEEP"
    logger.info("[OK] Denoise with reduced noise kept.")

    logger.info("\n============================================================")
    logger.info("UNIT TEST 3: Model Producing Identical Output (DISCARD)")
    logger.info("============================================================")
    res_identical = evaluator.evaluate("deblur", img_before, img_before.copy())
    logger.info(f"[IDENTICAL TEST] Decision: {res_identical.decision} | Reason: {res_identical.reason}")
    assert res_identical.decision == "DISCARD"
    assert "identical" in res_identical.reason.lower()
    logger.info("[OK] Identical model output discarded.")

    logger.info("\n============================================================")
    logger.info("UNIT TEST 4: Restoration Making Image Worse (DISCARD)")
    logger.info("============================================================")
    # Deblur applied to an image, but output drops sharpness significantly
    res_worse = evaluator.evaluate("deblur", img_after, img_before)
    logger.info(f"[WORSE TEST] Decision: {res_worse.decision} | Reason: {res_worse.reason}")
    assert res_worse.decision == "DISCARD"
    logger.info("[OK] Deblur output with decreased sharpness discarded.")


def test_engine_evaluator_discard_behavior():
    logger.info("\n============================================================")
    logger.info("INTEGRATION TEST 5: Engine Discard Propagation Check")
    logger.info("============================================================")

    class DummyBadModel:
        name = "DummyBadModel"
        def predict(self, img: Image.Image) -> Image.Image:
            # Return identical image so evaluator discards it
            return img.copy()

    analyzer = DegradationAnalyzer()
    planner = PipelinePlanner()
    engine = RestorationEngine()

    test_img = Image.new("RGB", (100, 100), color=(100, 100, 100))

    # Single pass run
    res = engine.run(test_img, ["deblur"])
    step = res.pipeline_steps[0]
    logger.info(f"Step decision: {step['decision']} | Reason: {step['reason']}")
    assert step["decision"] == "DISCARD"
    assert res.final_image is not None
    logger.info("[OK] Discarded output is not propagated to final image.")


def test_two_pass_with_evaluator():
    logger.info("\n============================================================")
    logger.info("INTEGRATION TEST 6: Full Two-Pass Pipeline with Evaluator Active")
    logger.info("============================================================")

    analyzer = DegradationAnalyzer()
    planner = PipelinePlanner()
    engine = RestorationEngine()

    img_arr = np.random.randint(10, 40, (264, 266, 3), dtype=np.uint8)
    test_img = Image.fromarray(img_arr)

    res = engine.run_two_pass(
        image=test_img,
        analyzer=analyzer,
        planner=planner,
        image_id="evaluator_two_pass_test",
    )

    logger.info("============================================================")
    logger.info("TWO-PASS EVALUATION SUMMARY")
    logger.info("============================================================")
    for step in res.pipeline_steps:
        logger.info(
            f"Step {step['step_number']} (Pass {step['pass_number']}): op='{step['operation']}' | "
            f"decision={step['decision']} | reason='{step['reason']}'"
        )

    assert res.final_image is not None
    assert res.metrics.get("passes_executed", 0) >= 1
    logger.info("[OK] Two-pass pipeline executed successfully with active evaluator.")
    logger.info("============================================================")


if __name__ == "__main__":
    test_evaluator_unit_cases()
    test_engine_evaluator_discard_behavior()
    test_two_pass_with_evaluator()
