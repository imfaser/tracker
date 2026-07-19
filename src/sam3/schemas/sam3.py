from __future__ import annotations

from typing import Optional

import torch
from PIL import Image
from pydantic import BaseModel, ConfigDict


class SAM3Request(BaseModel):
    """SAM3 内部格式"""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    image: Image.Image
    input_points: Optional[torch.Tensor] = None
    input_labels: Optional[torch.Tensor] = None
    input_boxes: Optional[torch.Tensor] = None
    input_masks: Optional[torch.Tensor] = None
    multimask_output: bool = True
