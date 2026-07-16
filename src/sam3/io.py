import os
import uuid

import numpy as np
import torch
from PIL import Image


def load_image(path: str) -> Image.Image:
    """加载图片并转换为 RGB"""
    return Image.open(path).convert("RGB")


def load_mask(path: str) -> torch.Tensor:
    """加载灰度 PNG mask 并转换为 soft mask tensor

    返回 shape (1, 1, H, W) 的 float tensor，值范围 [0, 1]
    """

    mask = Image.open(path).convert("L")
    arr = np.array(mask, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr)
    return tensor.unsqueeze(0).unsqueeze(0)


def save_mask(mask: torch.Tensor, output_dir: str) -> str:
    """将 mask tensor 保存为灰度 PNG，返回文件路径

    mask: shape (H, W) 或 (1, H, W) 的 float tensor，值范围 [0, 1]
    """
    os.makedirs(output_dir, exist_ok=True)

    if mask.dim() == 3:
        mask = mask.squeeze(0)

    mask_np = (mask.detach().cpu().numpy() * 255).clip(0, 255).astype("uint8")
    img = Image.fromarray(mask_np, mode="L")

    filename = f"{uuid.uuid4().hex}_mask.png"
    filepath = os.path.join(output_dir, filename)
    img.save(filepath)
    return filepath
