"""
AI Image Restoration & Enhancement System
FastAPI Application Entry Point
"""
import os
import sys
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from loguru import logger

from app.config import load_config


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

logger.remove()  # Remove default stderr handler
logger.add(sys.stderr, level="INFO", format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | {message}")
logger.add(LOG_DIR / "app.log", rotation="10 MB", retention="7 days", level="DEBUG")


# ---------------------------------------------------------------------------
# Application lifespan — load config, pre-warm models (later phases)
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown logic."""
    config = load_config()
    app.state.config = config
    logger.info("Configuration loaded successfully")
    logger.info(f"Device strategy: {config.get('device', 'auto')}")
    yield
    logger.info("Application shutting down")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="AI Image Restoration & Enhancement",
    description="Automatically analyzes image degradations and restores them using deep learning models.",
    version="0.1.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Mount frontend static files
# ---------------------------------------------------------------------------
FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/")
async def serve_frontend():
    """Serve the frontend index.html."""
    index_path = FRONTEND_DIR / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    return {"message": "Frontend not found. Place index.html in frontend/"}


@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "version": "0.1.0",
        "project": "AI Image Restoration & Enhancement",
    }
