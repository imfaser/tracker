from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, model_validator


class PointPrompt(BaseModel):
    """单个点击提示"""

    coords: tuple[int, int]
    label: int

    @model_validator(mode="after")
    def validate_label(self) -> PointPrompt:
        if self.label not in (0, 1):
            raise ValueError(f"label 必须是 0 或 1，实际为 {self.label}")
        return self


class BoundingBox(BaseModel):
    """边界框提示"""

    coords: tuple[float, float, float, float]


class Object(BaseModel):
    """一个待分割物体 = 一组提示"""

    points: list[PointPrompt] = []
    box: Optional[BoundingBox] = None

    @model_validator(mode="after")
    def validate_has_prompt(self) -> Object:
        if not self.points and self.box is None:
            raise ValueError("每个 object 至少需要一个提示（points 或 box）")
        return self


class MCPRequest(BaseModel):
    """MCP 协议输入格式"""

    image: str
    objects: list[Object]
    prev_mask: Optional[str] = None
    multimask_output: bool = True

    @model_validator(mode="after")
    def validate_objects_not_empty(self) -> MCPRequest:
        if not self.objects:
            raise ValueError("至少需要一个 object")
        return self
