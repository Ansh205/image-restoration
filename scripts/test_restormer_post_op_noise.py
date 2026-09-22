import os
import sys
sys.path.insert(0, os.path.abspath("."))

import cv2
import numpy as np
from PIL import Image
from loguru import logger

from core.analyzer.analyzer import DegradationAnalyzer
from core.pipeline.planner import PipelinePlanner
from core.pipeline.engine import RestorationEngine


def run_test():
    logger.info("============================================================")
    logger.info("TESTING TWO-PASS FLOW: PASS 1 DEBLUR -> PASS 2 DENOISE")
    logger.info("============================================================")

    img_path = "debug/umsn/original_face_1.png"
    if not os.path.exists(img_path):
        logger.error(f"Image '{img_path}' not found!")
        return

    face_img = Image.open(img_path).convert("RGB")
    arr = np.array(face_img)
    blurred_arr = cv2.GaussianBlur(arr, (7, 7), 2.5)
    blurry_face_img = Image.fromarray(blurred_arr)

    analyzer = DegradationAnalyzer()
    planner = PipelinePlanner()
    engine = RestorationEngine()

    result = engine.run_two_pass(
        image=blurry_face_img,
        analyzer=analyzer,
        planner=planner,
        image_id="test_deblur_pass1_denoise_pass2",
    )

    logger.info("============================================================")
    logger.info("TEST EXECUTION RESULTS")
    logger.info("============================================================")
    logger.info(f"Pass 1 Degradations: {result.metrics.get('pass_1_degradations')}")
    logger.info(f"Pass 1 Planned:      {result.metrics.get('pass_1_planned')}")
    logger.info(f"Pass 1 Executed:     {result.metrics.get('pass_1_executed')}")
    logger.info(f"Pass 2 Degradations: {result.metrics.get('pass_2_degradations')}")
    logger.info(f"Pass 2 Planned:      {result.metrics.get('pass_2_planned')}")
    logger.info(f"Pass 2 Executed:     {result.metrics.get('pass_2_executed')}")
    logger.info(f"Attempted Ops:       {result.metrics.get('attempted_operations')}")
    logger.info(f"Applied Ops:         {result.metrics.get('applied_operations')}")

    for step in result.pipeline_steps:
        decision_val = step.get('effect', step.get('evaluator_decision', step.get('decision', 'N/A')))
        reason_val = step.get('evaluator_reason', step.get('reason', 'N/A'))
        logger.info(f"Step {step['step_number']} (Pass {step['pass_number']}): op='{step['operation']}', model='{step['model_name']}', effect/decision='{decision_val}', reason='{reason_val}'")


def test_pure_blur_image():
    logger.info("\n============================================================")
    logger.info("TESTING PURE BLUR SCENARIO: PASS 1 DEBLUR -> PASS 2 DENOISE")
    logger.info("============================================================")

    img_path = "debug/umsn/original_face_1.png"
    face_img = Image.open(img_path).convert("RGB")
    arr = np.array(face_img)
    # Heavy Gaussian blur
    blurred_arr = cv2.GaussianBlur(arr, (11, 11), 4.0)
    blurry_face_img = Image.fromarray(blurred_arr)

    analyzer = DegradationAnalyzer()
    planner = PipelinePlanner()
    engine = RestorationEngine()

    result = engine.run_two_pass(
        image=blurry_face_img,
        analyzer=analyzer,
        planner=planner,
        custom_operations=["deblur"],
        image_id="test_pure_blur",
    )

    logger.info("============================================================")
    logger.info("PURE BLUR EXECUTION SUMMARY")
    logger.info("============================================================")
    logger.info(f"Pass 1 Planned:      {result.metrics.get('pass_1_planned')}")
    logger.info(f"Pass 1 Executed:     {result.metrics.get('pass_1_executed')}")
    logger.info(f"Pass 2 Degradations: {result.metrics.get('pass_2_degradations')}")
    logger.info(f"Pass 2 Planned:      {result.metrics.get('pass_2_planned')}")
    logger.info(f"Pass 2 Skipped:      {result.metrics.get('pass_2_skipped')}")
    logger.info(f"Pass 2 Executed:     {result.metrics.get('pass_2_executed')}")
    logger.info(f"Attempted Ops:       {result.metrics.get('attempted_operations')}")
    logger.info(f"Applied Ops:         {result.metrics.get('applied_operations')}")

    for step in result.pipeline_steps:
        decision_val = step.get('effect', step.get('evaluator_decision', step.get('decision', 'N/A')))
        reason_val = step.get('evaluator_reason', step.get('reason', 'N/A'))
        logger.info(f"Step {step['step_number']} (Pass {step['pass_number']}): op='{step['operation']}', model='{step['model_name']}', effect/decision='{decision_val}', reason='{reason_val}'")


if __name__ == "__main__":
    run_test()
    test_pure_blur_image()

