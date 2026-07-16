from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, model_validator


class SegmentRequest(BaseModel):
    """图像分割请求"""

    image_path: str
    p_point: list[list[int]] = []
    n_point: list[list[int]] = []
    boxes: list[list[float]] = []
    prev_mask: Optional[str] = None
    multimask_output: bool = True

    @model_validator(mode="after")
    def validate_at_least_one_prompt(self):
        if not self.p_point and not self.n_point and not self.boxes and not self.prev_mask:
            raise ValueError("至少需要一种提示：p_point, n_point, boxes, 或 prev_mask")
        return self

    @model_validator(mode="after")
    def validate_boxes_format(self):
        for i, box in enumerate(self.boxes):
            if len(box) != 4:
                raise ValueError(f"boxes[{i}] 需要恰好 4 个值 [x1, y1, x2, y2]，实际为 {len(box)} 个")
        return self
