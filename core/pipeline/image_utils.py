"""
Image I/O and preprocessing utilities.

Handles loading, validating, resizing, converting, and saving images
with all the safety checks from the project context.
"""
import io
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from loguru import logger


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
MAX_FILE_SIZE_MB = 10
MAX_DIMENSION = 4096


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class ImageMeta:
    """Metadata about a loaded image."""
    width: int
    height: int
    channels: int
    file_size_bytes: int
    format: str | None
    filename: str


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
class ImageValidationError(Exception):
    """Raised when image validation fails."""
    pass


def validate_extension(filename: str) -> None:
    """Check that the file extension is allowed."""
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise ImageValidationError(
            f"Unsupported file type '{ext}'. "
            f"Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
        )


def validate_file_size(file_bytes: bytes, max_mb: float = MAX_FILE_SIZE_MB) -> None:
    """Check that file size is within limits."""
    size_mb = len(file_bytes) / (1024 * 1024)
    if size_mb > max_mb:
        raise ImageValidationError(
            f"File too large: {size_mb:.2f} MB. Maximum: {max_mb} MB."
        )


def validate_dimensions(img: Image.Image, max_dim: int = MAX_DIMENSION) -> None:
    """Check that image dimensions are within limits."""
    w, h = img.size
    if w > max_dim or h > max_dim:
        raise ImageValidationError(
            f"Image too large: {w}x{h}. Maximum dimension: {max_dim}px."
        )


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_image_from_bytes(
    file_bytes: bytes,
    filename: str = "unknown",
    max_file_size_mb: float = MAX_FILE_SIZE_MB,
    max_dimension: int = MAX_DIMENSION,
) -> tuple[Image.Image, ImageMeta]:
    """
    Load an image from raw bytes with full validation.

    Args:
        file_bytes: Raw bytes of the uploaded file.
        filename: Original filename (for extension check + metadata).
        max_file_size_mb: Maximum allowed file size in MB.
        max_dimension: Maximum allowed width or height in pixels.

    Returns:
        Tuple of (PIL Image in RGB, ImageMeta).

    Raises:
        ImageValidationError: If any validation check fails.
    """
    # 1. Validate extension
    validate_extension(filename)

    # 2. Validate file size
    validate_file_size(file_bytes, max_file_size_mb)

    # 3. Try opening the image
    try:
        img = Image.open(io.BytesIO(file_bytes))
        img.verify()  # Check for corrupt images
        # Re-open after verify (verify leaves file in unusable state)
        img = Image.open(io.BytesIO(file_bytes))
    except Exception as e:
        raise ImageValidationError(f"Cannot open image: {e}")

    # 4. Validate dimensions
    validate_dimensions(img, max_dimension)

    # 5. Convert to RGB
    img = convert_to_rgb(img)

    # 6. Build metadata
    meta = ImageMeta(
        width=img.size[0],
        height=img.size[1],
        channels=3,
        file_size_bytes=len(file_bytes),
        format=img.format,
        filename=filename,
    )

    logger.info(f"Loaded image: {filename} ({meta.width}x{meta.height}, {len(file_bytes) / 1024:.1f} KB)")
    return img, meta


def load_image_from_path(path: str | Path) -> tuple[Image.Image, ImageMeta]:
    """Load an image from a file path."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Image file not found: {path}")

    file_bytes = path.read_bytes()
    return load_image_from_bytes(file_bytes, filename=path.name)


# ---------------------------------------------------------------------------
# Conversion & Preprocessing
# ---------------------------------------------------------------------------
def convert_to_rgb(img: Image.Image) -> Image.Image:
    """Convert image to RGB, handling RGBA, grayscale, palette modes."""
    if img.mode == "RGB":
        return img
    if img.mode == "RGBA":
        # Composite on white background
        background = Image.new("RGB", img.size, (255, 255, 255))
        background.paste(img, mask=img.split()[3])
        logger.debug("Converted RGBA → RGB (white background)")
        return background
    logger.debug(f"Converted {img.mode} → RGB")
    return img.convert("RGB")


def resize_if_needed(
    img: Image.Image,
    max_size: int = 1024,
) -> tuple[Image.Image, bool]:
    """
    Resize image if its largest dimension exceeds max_size.
    Maintains aspect ratio.

    Returns:
        Tuple of (image, was_resized).
    """
    w, h = img.size
    if max(w, h) <= max_size:
        return img, False

    # Calculate new dimensions maintaining aspect ratio
    if w >= h:
        new_w = max_size
        new_h = int(h * (max_size / w))
    else:
        new_h = max_size
        new_w = int(w * (max_size / h))

    resized = img.resize((new_w, new_h), Image.LANCZOS)
    logger.info(f"Resized image: {w}x{h} → {new_w}x{new_h}")
    return resized, True


def pil_to_numpy(img: Image.Image) -> np.ndarray:
    """Convert PIL Image to NumPy array (H, W, C), float32, range [0, 1]."""
    arr = np.array(img).astype(np.float32) / 255.0
    return arr


def numpy_to_pil(arr: np.ndarray) -> Image.Image:
    """Convert NumPy array (H, W, C) float32 [0, 1] back to PIL Image."""
    arr = np.clip(arr, 0.0, 1.0)
    arr = (arr * 255.0).astype(np.uint8)
    return Image.fromarray(arr)


# ---------------------------------------------------------------------------
# Saving / Encoding
# ---------------------------------------------------------------------------
def image_to_bytes(img: Image.Image, fmt: str = "PNG", quality: int = 95) -> bytes:
    """Encode a PIL Image to bytes."""
    buffer = io.BytesIO()
    save_kwargs = {}
    if fmt.upper() in ("JPEG", "JPG"):
        save_kwargs["quality"] = quality
    img.save(buffer, format=fmt, **save_kwargs)
    return buffer.getvalue()


def save_image(img: Image.Image, path: str | Path, quality: int = 95) -> Path:
    """Save a PIL Image to disk."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    save_kwargs = {}
    if path.suffix.lower() in (".jpg", ".jpeg"):
        save_kwargs["quality"] = quality

    img.save(str(path), **save_kwargs)
    logger.info(f"Saved image: {path}")
    return path
