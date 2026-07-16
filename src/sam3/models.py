import os
from dataclasses import dataclass

import torch
from transformers import Sam3TrackerModel, Sam3TrackerProcessor


@dataclass
class LoadedModels:
    tracker_model: Sam3TrackerModel
    tracker_processor: Sam3TrackerProcessor
    device: torch.device


def load_models() -> LoadedModels:
    """从 MODEL_NAME 环境变量加载 SAM3 模型"""
    model_path = os.getenv("MODEL_NAME", r"D:\code\tmp\sam3")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Sam3TrackerModel.from_pretrained(model_path, device_map=str(device))
    processor = Sam3TrackerProcessor.from_pretrained(model_path)

    return LoadedModels(
        tracker_model=model,
        tracker_processor=processor,
        device=device,
    )
