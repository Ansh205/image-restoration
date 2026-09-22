"""
Unit tests for all 5 model wrappers and the model factory.
"""
import pytest
from PIL import Image

from models.base import BaseRestorationModel
from models.factory import get_model, MODEL_REGISTRY
from models.denoising.drunet import DRUNetModel
from models.deblurring.umsn import UMSNModel
from models.low_light.retinexformer import RetinexformerModel
from models.deblurring.restormer import RestormerModel
from models.super_resolution.realesrgan import RealESRGANModel
from models.low_light.zerodce import ZeroDCEModel
from models.jpeg.swinir import SwinIRModel


@pytest.fixture
def sample_image() -> Image.Image:
    """Create a small 64x64 test RGB image."""
    return Image.new("RGB", (64, 64), color=(100, 150, 200))


class TestModelFactory:
    def test_factory_instantiation(self):
        for key in MODEL_REGISTRY:
            model = get_model(key, device="cpu")
            assert isinstance(model, BaseRestorationModel)
            assert model.device == "cpu"

    def test_factory_invalid_key(self):
        with pytest.raises(ValueError, match="Unknown model key"):
            get_model("invalid_model_name")


class TestDRUNet:
    def test_drunet_restoration(self, sample_image):
        model = DRUNetModel(device="cpu")
        out = model.restore(sample_image)
        assert isinstance(out, Image.Image)
        assert out.size == sample_image.size
        assert out.mode == "RGB"


class TestUMSN:
    def test_umsn_restoration(self, sample_image):
        model = UMSNModel(device="cpu")
        out = model.restore(sample_image)
        assert isinstance(out, Image.Image)
        assert out.size == sample_image.size
        assert out.mode == "RGB"


class TestRetinexformer:
    def test_retinexformer_restoration(self, sample_image):
        model = RetinexformerModel(device="cpu")
        out = model.restore(sample_image)
        assert isinstance(out, Image.Image)
        assert out.size == sample_image.size
        assert out.mode == "RGB"


class TestRestormer:
    def test_restormer_restoration(self, sample_image):
        model = RestormerModel(device="cpu")
        out = model.restore(sample_image)
        assert isinstance(out, Image.Image)
        assert out.size == sample_image.size
        assert out.mode == "RGB"


class TestRealESRGAN:
    def test_realesrgan_super_resolution(self, sample_image):
        model = RealESRGANModel(device="cpu")
        out = model.restore(sample_image)
        assert isinstance(out, Image.Image)
        # 4x upscaling check: 64x64 -> 256x256
        assert out.size == (256, 256)
        assert out.mode == "RGB"


class TestZeroDCE:
    def test_zerodce_restoration(self, sample_image):
        model = ZeroDCEModel(device="cpu")
        out = model.restore(sample_image)
        assert isinstance(out, Image.Image)
        assert out.size == sample_image.size
        assert out.mode == "RGB"


class TestSwinIR:
    def test_swinir_restoration(self, sample_image):
        model = SwinIRModel(device="cpu")
        out = model.restore(sample_image)
        assert isinstance(out, Image.Image)
        assert out.size == sample_image.size
        assert out.mode == "RGB"

