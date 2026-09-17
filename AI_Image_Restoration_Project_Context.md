# AI Image Restoration & Enhancement System — Coding Context

## 1. Project goal

Build a Python-first, end-to-end web application that automatically analyzes an uploaded image, identifies which degradations are significant, selects only the necessary restoration models, executes them in an appropriate pipeline, and reports how the image changed.

This is a resume/portfolio project. Image quality is the primary objective; the application should also demonstrate strong skills in computer vision, deep learning, transformers, PEFT/LoRA, evaluation, backend development, and deployment.

The user should NOT have to select the restoration model manually.

Example:
Input image -> degradation analysis -> "blur: high, noise: medium, resolution: low" -> planner -> denoise -> deblur -> super-resolution -> enhanced image -> metrics/results.

## 2. Hard constraints

- The developer has no local GPU.
- Training/fine-tuning must use free cloud GPU resources, primarily Kaggle.
- Do not design the system around paid infrastructure.
- Deployment should use free/low-cost student-friendly resources, initially Hugging Face Spaces CPU deployment.
- 5–10+ seconds inference is acceptable for the portfolio demo; production traffic/low latency is NOT the goal.
- The developer is primarily a Python developer.
- Backend: FastAPI.
- Frontend: ordinary HTML/CSS/JavaScript is preferred; do not require React.
- Do not use an LLM API, image-generation API, or external AI API for the core restoration process.
- Models should run through local/cloud-hosted PyTorch inference.
- Do not fine-tune every model.
- Fine-tune ONE model only, using LoRA/PEFT. Preferred model: Restormer for deblurring.
- Other restoration models should use reliable pretrained weights.
- Never invent benchmark numbers. All reported metrics must come from actual experiments.
- Do not blindly apply every restoration model. Apply only the operations whose degradation severity warrants them.
- Ground-truth metrics such as PSNR/SSIM/LPIPS require a ground-truth reference. For arbitrary user uploads, use appropriate no-reference/image-statistics metrics instead.

## 3. Restoration models

Use the following initial model set unless experiments show a strong reason to change it:

1. Denoising: DRUNet
2. Deblurring: Restormer
3. Super-resolution: Real-ESRGAN
4. Low-light enhancement: Zero-DCE++
5. JPEG artifact reduction: SwinIR

Fine-tuning:
- Fine-tune Restormer only.
- Use LoRA/PEFT adapters.
- Freeze the base Restormer parameters.
- Train only LoRA parameters/adapters.
- Training runs on Kaggle GPU.
- Compare pretrained Restormer vs LoRA-adapted Restormer on a held-out validation/test split.

## 4. Automatic degradation analysis

The analyzer should estimate, at minimum:

- Noise severity
- Blur severity
- Resolution adequacy
- Low-light severity
- JPEG/compression artifact severity

Prefer explainable CV/statistical techniques initially rather than adding a large classifier for every degradation.

Possible starting signals:
- Blur: variance of Laplacian and related sharpness statistics
- Brightness: luminance statistics
- Noise: high-frequency residual/local variance/noise estimation
- Resolution: image dimensions and project-defined thresholds
- JPEG artifacts: compression/blocking/high-frequency statistics

Important:
- Treat detector outputs as scores/probabilities/severity values, not universal truths.
- Thresholds must be validated experimentally.
- Detection thresholds should be configurable.
- The system should expose both score and human-readable severity: LOW/MEDIUM/HIGH.
- Avoid claiming a score is a calibrated probability unless it has actually been calibrated.

## 5. Pipeline planner

The planner receives the degradation report and decides:
1. Which operations are necessary.
2. Which model performs each operation.
3. The order of operations.

Example:
{
  "noise": "high",
  "blur": "medium",
  "resolution": "low",
  "low_light": "low",
  "jpeg_artifacts": "low"
}
might produce:
["denoise", "deblur", "super_resolution"]

Do not assume one order is universally optimal. Test candidate orderings on benchmark data. The guiding principle is:
"Apply the minimum necessary restoration operations that produce measurable visual/quantitative improvement."

The planner should skip unnecessary models.

## 6. Inference UX

During inference, the web application should visibly communicate what the system detected and what it is doing.

Example:

IMAGE ANALYSIS
- Noise: HIGH
- Blur: MEDIUM
- Resolution: LOW
- Low-light: LOW
- JPEG artifacts: LOW

REQUIRED IMPROVEMENTS
- Noise reduction required
- Deblurring required
- Resolution enhancement required
- Low-light enhancement skipped
- JPEG restoration skipped

RESTORATION PIPELINE
1. DRUNet — denoising
2. Restormer — deblurring
3. Real-ESRGAN — 4x super-resolution

The frontend should update progress/status during inference. FastAPI may expose a job/status endpoint or use a simple polling approach; do not over-engineer real-time streaming unless it materially improves the demo.

## 7. Metrics

### Arbitrary uploaded images without ground truth
Do NOT claim PSNR/SSIM/LPIPS improvement because the true clean image is unknown.

Show before/after values for appropriate no-reference or image-statistics measures, for example:
- NIQE
- BRISQUE
- sharpness
- noise estimate
- brightness
- contrast
- resolution/dimensions

Clearly label these as quality indicators/statistics, not restoration accuracy.

### Benchmark/evaluation mode with ground truth
Use:
- PSNR (higher is better)
- SSIM (higher is better)
- LPIPS (lower is better)

Report:
- degraded input vs ground truth
- restored output vs ground truth
- absolute improvement
- inference time
- optionally model memory
- optionally number/percentage of trainable parameters for LoRA

The benchmark mode should be reproducible and use fixed train/validation/test splits where the dataset permits.

## 8. Evaluation and experiments

The project should include:
- Individual model evaluation
- Automatic degradation detector validation
- Pipeline-order experiments
- End-to-end pipeline evaluation
- Pretrained Restormer vs LoRA Restormer comparison
- Inference-time benchmarking on CPU and/or available cloud hardware
- Visual before/after examples
- Failure cases

Do not optimize only for PSNR. Consider perceptual quality and artifacts as well.

## 9. Suggested datasets

Use task-appropriate public datasets. Initial candidates:
- Denoising: BSD68, SIDD, or controlled synthetic noise
- Super-resolution: DIV2K, Set5, Set14, Urban100
- Deblurring: GoPro, HIDE, RealBlur
- Low-light: LOL / LOL-v2

Choose datasets based on model compatibility, licensing/access, compute limits, and reproducibility. Do not download or train on massive datasets unnecessarily.

## 10. Training strategy

Most models:
- Use pretrained weights.
- First make inference work.
- Benchmark before modifying them.

Restormer:
- Load pretrained checkpoint.
- Add LoRA to appropriate attention/linear layers.
- Freeze base parameters.
- Train only LoRA adapters.
- Use Kaggle GPU.
- Save adapter weights and configuration.
- Evaluate against the untouched pretrained baseline.

The goal is not to prove LoRA is universally superior. The goal is to demonstrate parameter-efficient adaptation and measure whether it improves the chosen task/dataset.

## 11. Infrastructure

Development:
- Local laptop: Python, FastAPI, frontend, tests, small CPU inference, Git.

Training:
- Kaggle notebooks with free GPU.

Code:
- GitHub.

Model storage:
- Hugging Face Hub.

Deployment:
- Hugging Face Spaces or another free CPU-friendly host.
- Keep the public demo input-size constrained if necessary.
- Lazy-load models where useful.
- Avoid loading all large models into RAM if memory becomes a deployment problem.

Deployment architecture:
Browser -> HTML/CSS/JS -> FastAPI -> degradation analyzer -> pipeline planner -> PyTorch models -> metrics -> response.

Kaggle is for training/experimentation, NOT the live production inference server.

## 12. Suggested repository structure

image-restoration/
├── app/
│   ├── main.py
│   ├── routes/
│   └── schemas/
├── frontend/
│   ├── index.html
│   ├── styles.css
│   └── app.js
├── core/
│   ├── analyzer/
│   ├── planner/
│   ├── pipeline/
│   └── metrics/
├── models/
│   ├── denoising/
│   ├── deblurring/
│   ├── super_resolution/
│   ├── low_light/
│   └── jpeg/
├── training/
│   ├── restormer_lora/
│   └── configs/
├── evaluation/
├── datasets/
├── notebooks/
├── tests/
├── scripts/
├── requirements.txt
├── Dockerfile
├── README.md
└── LICENSE

Keep model adapters/wrapper code separate from downloaded model weights. Do not commit huge checkpoints to Git.

## 13. Engineering principles for any coding model

When generating code:
- Build incrementally; do not generate the entire application in one giant response unless explicitly requested.
- Preserve the architecture above.
- Use modular Python classes/functions.
- Use type hints and clear docstrings.
- Add error handling for corrupt images, unsupported formats, oversized images, missing model weights, and insufficient memory.
- Keep model loading separate from request handling.
- Cache/load models once rather than loading them for every request.
- Use configuration files/environment variables for paths, thresholds, device selection, and model settings.
- Support CPU fallback automatically.
- Do not assume CUDA exists.
- Add logging for each inference stage.
- Return structured JSON from FastAPI.
- Keep frontend independent from model implementation.
- Add tests for degradation detectors, planner decisions, API validation, and metrics.
- Make experiments reproducible with seeds/configs where applicable.
- Do not silently change model choice, threshold, dataset, or metric definition.
- Explain any approximation or heuristic.
- Prefer correctness and maintainability over unnecessary abstraction.

## 14. Device strategy

Use:
- CUDA when available.
- CPU otherwise.

Example conceptual configuration:
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

Never hard-code CUDA.

Because deployment is CPU-oriented, benchmark memory and latency on CPU for the final demo path.

## 15. Security and robustness

For uploads:
- Validate extension and MIME/type where practical.
- Limit maximum file size.
- Limit maximum width/height.
- Convert safely to RGB.
- Use temporary files or in-memory processing carefully.
- Never execute uploaded files.
- Avoid path traversal.
- Clean temporary artifacts.
- Handle malicious/corrupt image inputs gracefully.

## 16. Resume/portfolio goal

The project should demonstrate:
- Deep learning for computer vision
- Image restoration
- Transformer-based vision
- Transfer learning
- LoRA/PEFT
- Automatic model selection
- Evaluation methodology
- PyTorch
- FastAPI
- Web application development
- Cloud GPU experimentation
- Free/student-friendly deployment
- Performance/resource awareness

The resume should use actual measured results only.

## 17. Recommended build order

Phase 1: Project skeleton + environment
Phase 2: Image I/O + preprocessing
Phase 3: Individual pretrained model inference
Phase 4: Degradation analyzer
Phase 5: Pipeline planner
Phase 6: End-to-end automatic restoration
Phase 7: Metrics and benchmark mode
Phase 8: Restormer LoRA fine-tuning on Kaggle
Phase 9: Web UI + FastAPI integration
Phase 10: Deployment
Phase 11: Optimization, testing, documentation, and resume-ready benchmarking

## 18. Definition of done

A strong final version should allow a user to:
1. Open the web app.
2. Upload an image.
3. See detected degradation/severity.
4. See which improvements are required.
5. See the selected pipeline and live progress.
6. Receive an enhanced image.
7. Compare original and enhanced images.
8. See before/after no-reference quality indicators.
9. Run benchmark mode on known ground-truth examples and see PSNR/SSIM/LPIPS.
10. Download the result.
11. Understand which models were used and why.
12. Reproduce the major experiments from the repository.

The final project should prioritize actual restoration quality while using the system architecture to showcase engineering and ML skills.
