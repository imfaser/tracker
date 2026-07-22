import base64
import os
import uuid
from io import BytesIO

import numpy as np
import torch
from PIL import Image


def decode_image(base64_str: str) -> Image.Image:
    """将 base64 字符串解码为 RGB PIL.Image"""
    try:
        image_bytes = base64.b64decode(base64_str)
    except Exception as e:
        raise ValueError(f"无效的 base64 字符串: {e}") from e

    try:
        return Image.open(BytesIO(image_bytes)).convert("RGB")
    except Exception as e:
        raise ValueError(f"无法解码图像: {e}") from e


def decode_mask(base64_str: str) -> torch.Tensor:
    """将 base64 字符串解码为 mask tensor

    返回 shape (1, 1, H, W) 的 float tensor，值范围 [0, 1]
    """
    try:
        mask_bytes = base64.b64decode(base64_str)
    except Exception as e:
        raise ValueError(f"无效的 base64 字符串: {e}") from e

    try:
        mask = Image.open(BytesIO(mask_bytes)).convert("L")
    except Exception as e:
        raise ValueError(f"无法解码 mask 图像: {e}") from e

    arr = np.array(mask, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr)
    return tensor.unsqueeze(0).unsqueeze(0)


def save_mask(mask: torch.Tensor, output_dir: str) -> str:
    """将 mask tensor 保存为灰度 PNG，返回文件路径

    mask: shape (H, W) 或 (1, H, W) 或 (1, 1, H, W) 的 float tensor，值范围 [0, 1]
    """
    os.makedirs(output_dir, exist_ok=True)

    while mask.dim() > 2:
        if mask.shape[0] != 1:
            raise ValueError(f"mask shape {tuple(mask.shape)} 无法压缩到 2D")
        mask = mask.squeeze(0)

    mask_np = (mask.detach().cpu().numpy() * 255).clip(0, 255).astype("uint8")
    img = Image.fromarray(mask_np, mode="L")

    filename = f"{uuid.uuid4().hex}_mask.png"
    filepath = os.path.join(output_dir, filename)
    img.save(filepath)
    return filepath
