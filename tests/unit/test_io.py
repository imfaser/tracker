import base64
import os
from io import BytesIO

import torch
from PIL import Image

from sam3.io import decode_image, decode_mask, save_mask


def _make_base64_image(width=200, height=200):
    """创建测试用的 base64 图像字符串"""
    img = Image.new("RGB", (width, height), color=(255, 255, 255))
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def _make_base64_mask(width=100, height=100):
    """创建测试用的 base64 mask 字符串"""
    img = Image.new("L", (width, height), color=128)
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def _file_to_base64(filepath: str) -> str:
    """将文件转换为 base64 字符串"""
    with open(filepath, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def test_decode_image():
    image_b64 = _make_base64_image(200, 200)
    img = decode_image(image_b64)
    assert img.mode == "RGB"
    assert img.size == (200, 200)


def test_decode_mask():
    mask_b64 = _make_base64_mask(100, 100)
    tensor = decode_mask(mask_b64)
    assert tensor.shape == (1, 1, 100, 100)
    assert tensor.dtype == torch.float32
    assert 0.0 <= tensor.min() <= tensor.max() <= 1.0


def test_save_and_load_mask(tmp_path):
    mask = torch.rand(1, 100, 100)
    filepath = save_mask(mask, str(tmp_path))
    assert os.path.exists(filepath)
    assert filepath.endswith("_mask.png")

    loaded = decode_mask(_file_to_base64(filepath))
    assert loaded.shape == (1, 1, 100, 100)
    assert loaded.dtype == torch.float32
    assert 0.0 <= loaded.min() <= loaded.max() <= 1.0


def test_save_mask_creates_output_dir(tmp_path):
    output_dir = str(tmp_path / "sub" / "dir")
    mask = torch.zeros(1, 10, 10)
    filepath = save_mask(mask, output_dir)
    assert os.path.exists(filepath)
    assert os.path.isdir(output_dir)


def test_save_mask_2d(tmp_path):
    mask = torch.rand(50, 50)
    filepath = save_mask(mask, str(tmp_path))
    loaded = decode_mask(_file_to_base64(filepath))
    assert loaded.shape == (1, 1, 50, 50)


def test_load_mask_values(tmp_path):
    mask = torch.ones(1, 10, 10)
    filepath = save_mask(mask, str(tmp_path))
    loaded = decode_mask(_file_to_base64(filepath))
    assert torch.allclose(loaded, torch.ones(1, 1, 10, 10), atol=0.01)
