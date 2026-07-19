from typing import Optional

import torch

from sam3.io import decode_image, decode_mask
from sam3.schemas.mcp import MCPRequest
from sam3.schemas.sam3 import SAM3Request


def build_points(
    p_point: list[list[int]], n_point: list[list[int]]
) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """将 MCP 点列表转换为 SAM3 期望的 tensor 格式

    SAM3 期望:
    - points: shape [batch, num_points, 1, 2] (每个点一个 object)
    - labels: shape [batch, 1, num_points] (1=positive, 0=negative)
    """
    if not p_point and not n_point:
        return None, None

    points = []
    labels = []
    for pt in p_point:
        points.append(pt)
        labels.append(1)
    for pt in n_point:
        points.append(pt)
        labels.append(0)

    points_tensor = torch.tensor([[points]], dtype=torch.float)
    labels_tensor = torch.tensor([[labels]], dtype=torch.long)
    return points_tensor, labels_tensor


def build_boxes(boxes: list[list[float]]) -> Optional[torch.Tensor]:
    """将 MCP boxes 转换为 SAM3 期望的 3D tensor 格式

    SAM3 期望: [image, box, coordinates] = [1, M, 4]
    """
    if not boxes:
        return None
    return torch.tensor([boxes], dtype=torch.float)


def mcp_to_sam3(req: MCPRequest) -> SAM3Request:
    """将 MCPRequest 转换为 SAM3Request"""
    image = decode_image(req.image)

    input_points, input_labels = build_points(req.p_point, req.n_point)
    input_boxes = build_boxes(req.boxes)
    input_masks = decode_mask(req.prev_mask) if req.prev_mask else None

    return SAM3Request(
        image=image,
        input_points=input_points,
        input_labels=input_labels,
        input_boxes=input_boxes,
        input_masks=input_masks,
        multimask_output=req.multimask_output,
    )
