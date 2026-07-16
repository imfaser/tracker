import os

import torch

from sam3.io import load_image, load_mask, save_mask


def test_load_image(test_image):
    img = load_image(test_image)
    assert img.mode == "RGB"
    assert img.size == (200, 200)


def test_save_and_load_mask(tmp_path):
    mask = torch.rand(1, 100, 100)
    filepath = save_mask(mask, str(tmp_path))
    assert os.path.exists(filepath)
    assert filepath.endswith("_mask.png")

    loaded = load_mask(filepath)
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
    loaded = load_mask(filepath)
    assert loaded.shape == (1, 1, 50, 50)


def test_load_mask_values(tmp_path):
    mask = torch.ones(1, 10, 10)
    filepath = save_mask(mask, str(tmp_path))
    loaded = load_mask(filepath)
    assert torch.allclose(loaded, torch.ones(1, 1, 10, 10), atol=0.01)
