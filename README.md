# AI Image Restoration & Enhancement System

An AI-powered web application that **automatically analyzes** uploaded images, detects degradations (noise, blur, low resolution, poor lighting, JPEG artifacts), selects the appropriate deep-learning restoration models, and enhances the image — all without manual model selection.

## Features

- **Automatic degradation analysis** — detects 5 types of image degradation with severity scoring
- **Smart pipeline planning** — only applies necessary restoration models
- **5 restoration models**: DRUNet, Restormer, Real-ESRGAN, Zero-DCE++, SwinIR
- **LoRA fine-tuned Restormer** — parameter-efficient adaptation for deblurring
- **Before/after quality metrics** — no-reference (BRISQUE, NIQE) + full-reference (PSNR, SSIM, LPIPS)
- **Web UI** with drag-and-drop upload and live pipeline progress
- **Free deployment** on Hugging Face Spaces (CPU)

## Tech Stack

| Component        | Technology                     |
|------------------|--------------------------------|
| Backend          | Python, FastAPI, PyTorch       |
| Frontend         | HTML, CSS, JavaScript          |
| Models           | DRUNet, Restormer, Real-ESRGAN, Zero-DCE++, SwinIR |
| Fine-tuning      | LoRA/PEFT on Restormer         |
| Training         | Kaggle GPU (free tier)         |
| Model Storage    | Hugging Face Hub               |
| Deployment       | Hugging Face Spaces (Docker)   |

## Project Structure

```
image-restoration/
├── app/                  # FastAPI backend
├── frontend/             # HTML/CSS/JS frontend
├── core/
│   ├── analyzer/         # Degradation detection
│   ├── planner/          # Pipeline planning
│   ├── pipeline/         # Orchestration
│   └── metrics/          # Quality metrics
├── models/               # Model wrappers
├── training/             # LoRA fine-tuning code
├── evaluation/           # Benchmark scripts & results
├── notebooks/            # Kaggle notebooks
├── tests/                # Unit & integration tests
├── configs/              # YAML configuration
├── Dockerfile
└── requirements.txt
```

## Quick Start

```bash
# Clone
git clone https://github.com/Ansh205/image-restoration.git
cd image-restoration

# Setup environment
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt

# Run
uvicorn app.main:app --reload
```

Visit `http://localhost:8000` to use the application.

## Build Phases

- [x] Phase 1: Project skeleton + environment
- [ ] Phase 2: Image I/O + preprocessing
- [ ] Phase 3: Individual pretrained model inference
- [ ] Phase 4: Degradation analyzer
- [ ] Phase 5: Pipeline planner
- [ ] Phase 6: End-to-end automatic restoration
- [ ] Phase 7: Metrics and benchmark mode
- [ ] Phase 8: Restormer LoRA fine-tuning (Kaggle)
- [ ] Phase 9: Web UI + FastAPI integration
- [ ] Phase 10: Deployment (Hugging Face Spaces)
- [ ] Phase 11: Optimization, testing, documentation

## License

MIT — see [LICENSE](LICENSE)

## Author

[Ansh205](https://github.com/Ansh205)
