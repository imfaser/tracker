from dataclasses import dataclass

import torch
from transformers import Sam3TrackerModel, Sam3TrackerProcessor

from sam3.config import Settings


@dataclass
class LoadedModels:
    tracker_model: Sam3TrackerModel
    tracker_processor: Sam3TrackerProcessor
    device: torch.device


def load_models(settings: Settings) -> LoadedModels:
    """从 Settings 加载 SAM3 模型"""
    if settings.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(settings.device)

    model = Sam3TrackerModel.from_pretrained(settings.model_name, device_map=str(device))
    processor = Sam3TrackerProcessor.from_pretrained(settings.model_name)

    return LoadedModels(
        tracker_model=model,
        tracker_processor=processor,
        device=device,
    )
