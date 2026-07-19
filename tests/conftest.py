import pytest
from PIL import Image, ImageDraw


@pytest.fixture
def test_image(tmp_path):
    """生成测试图片：白色背景上的蓝色圆形"""
    path = str(tmp_path / "test.png")
    img = Image.new("RGB", (200, 200), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.ellipse([50, 50, 150, 150], fill=(0, 100, 200))
    img.save(path)
    return path


@pytest.fixture
def test_mask(tmp_path):
    """生成测试灰度 mask"""
    import torch

    from sam3.io import save_mask

    mask = torch.rand(1, 100, 100)
    return save_mask(mask, str(tmp_path))
