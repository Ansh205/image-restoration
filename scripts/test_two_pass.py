"""
Standalone verification script for Two-Pass Restoration Architecture.
"""
import sys
import numpy as np
from PIL import Image
from loguru import logger

from core.analyzer.analyzer import DegradationAnalyzer
from core.pipeline.planner import PipelinePlanner
from core.pipeline.engine import RestorationEngine
from app.schemas.image import AnalysisResponse, DegradationItem


def test_two_pass_flow():
    logger.info("============================================================")
    logger.info("Testing Two-Pass Restoration Architecture Execution Flow")
    logger.info("============================================================")

    analyzer = DegradationAnalyzer()
    planner = PipelinePlanner()
    engine = RestorationEngine()

    analyze_call_count = 0
    original_analyze = analyzer.analyze

    def counting_analyze(img, image_id=""):
        nonlocal analyze_call_count
        analyze_call_count += 1
        logger.info(f"-> DegradationAnalyzer call #{analyze_call_count} for image_id='{image_id}'")
        return original_analyze(img, image_id=image_id)

    analyzer.analyze = counting_analyze

    # Create a synthetic image with dark noisy content
    img_arr = np.random.randint(10, 40, (264, 266, 3), dtype=np.uint8)
    test_img = Image.fromarray(img_arr)

    logger.info(f"Created synthetic test image: size={test_img.size}, mode={test_img.mode}")

    res = engine.run_two_pass(
        image=test_img,
        analyzer=analyzer,
        planner=planner,
        image_id="test_two_pass_img",
    )

    logger.info("============================================================")
    logger.info("VERIFICATION RESULTS (Scenario 1)")
    logger.info("============================================================")
    logger.info(f"Total analyzer calls: {analyze_call_count}")
    assert analyze_call_count == 2, f"Expected analyzer to be called exactly 2 times, got {analyze_call_count}"
    logger.info("[OK] Analyzer ran exactly TWICE.")

    logger.info(f"Pipeline steps executed: {len(res.pipeline_steps)}")
    for step in res.pipeline_steps:
        logger.info(f"  Step {step['step_number']} (Pass {step['pass_number']}): operation='{step['operation']}', model='{step['model_name']}'")

    applied_ops = res.metrics.get("applied_operations", [])
    logger.info(f"Applied operations set across all passes: {applied_ops}")
    
    ops_executed_list = [step["operation"] for step in res.pipeline_steps]
    assert len(ops_executed_list) == len(set(ops_executed_list)), f"Duplicate operation detected! Executed: {ops_executed_list}"
    logger.info("[OK] No operation was executed twice in a single request.")

    expected_size = (test_img.width * 4, test_img.height * 4) if "super_resolution" in applied_ops else test_img.size
    assert res.final_image.size == expected_size, f"Expected size {expected_size}, got {res.final_image.size}"
    logger.info(f"[OK] Final image generated cleanly with expected resolution {expected_size}.")


def test_two_pass_newly_exposed_degradation():
    logger.info("\n============================================================")
    logger.info("Testing Scenario 2: Newly Exposed Degradation in Pass 2")
    logger.info("============================================================")

    class MockAnalyzer:
        def __init__(self):
            self.calls = 0

        def analyze(self, img, image_id=""):
            self.calls += 1
            if self.calls == 1:
                # Pass 1: blur and low_light
                return AnalysisResponse(
                    image_id=image_id,
                    degradations=[
                        DegradationItem(name="blur", score=0.8, severity="HIGH"),
                        DegradationItem(name="low_light", score=0.8, severity="HIGH"),
                    ]
                )
            else:
                # Pass 2: noise exposed after deblurring, plus remaining blur
                return AnalysisResponse(
                    image_id=image_id,
                    degradations=[
                        DegradationItem(name="noise", score=0.75, severity="HIGH"),
                        DegradationItem(name="blur", score=0.6, severity="MEDIUM"),
                    ]
                )

    mock_analyzer = MockAnalyzer()
    planner = PipelinePlanner()
    engine = RestorationEngine()

    test_img = Image.new("RGB", (100, 100), color=(50, 50, 50))

    res = engine.run_two_pass(
        image=test_img,
        analyzer=mock_analyzer,
        planner=planner,
        image_id="test_scenario2",
    )

    logger.info("============================================================")
    logger.info("VERIFICATION RESULTS (Scenario 2)")
    logger.info("============================================================")
    logger.info(f"Total analyzer calls: {mock_analyzer.calls}")
    assert mock_analyzer.calls == 2, f"Expected 2 analyzer calls, got {mock_analyzer.calls}"

    logger.info(f"Pass 1 Executed: {res.metrics['pass_1_executed']}")
    logger.info(f"Pass 2 Executed: {res.metrics['pass_2_executed']}")
    logger.info(f"Pass 2 Skipped: {res.metrics['pass_2_skipped']}")

    assert res.metrics["pass_1_executed"] == ["low_light", "deblur"]
    assert res.metrics["pass_2_skipped"] == ["deblur"]
    assert res.metrics["pass_2_executed"] == ["denoise"]
    assert set(res.metrics["applied_operations"]) == {"low_light", "deblur", "denoise"}
    logger.info("[OK] Pass 2 successfully executed newly exposed operation 'denoise' and skipped already applied 'deblur'.")
    logger.info("============================================================")


if __name__ == "__main__":
    test_two_pass_flow()
    test_two_pass_newly_exposed_degradation()
