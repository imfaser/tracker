from typing import Optional

import torch

from sam3.io import decode_image, decode_mask
from sam3.schemas.mcp import MCPRequest, Object
from sam3.schemas.sam3 import SAM3Request


def build_points(
    objects: list[Object],
) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """将 Object 列表转换为 SAM3 期望的 4D tensor 格式

    SAM3 期望:
    - points: shape [batch, num_objects, max_points_per_obj, 2]
    - labels: shape [batch, num_objects, max_points_per_obj]
    """
    if not objects or all(not obj.points for obj in objects):
        return None, None

    all_obj_points = []
    all_obj_labels = []

    for obj in objects:
        obj_pts = []
        obj_lbls = []
        for pt in obj.points:
            obj_pts.append(list(pt.coords))
            obj_lbls.append(pt.label)
        all_obj_points.append(obj_pts)
        all_obj_labels.append(obj_lbls)

    max_points = max(len(pts) for pts in all_obj_points) if all_obj_points else 0

    padded_points = []
    padded_labels = []
    for pts, lbls in zip(all_obj_points, all_obj_labels, strict=True):
        pad_count = max_points - len(pts)
        padded_points.append(pts + [[0, 0]] * pad_count)
        padded_labels.append(lbls + [-10] * pad_count)

    points_tensor = torch.tensor([padded_points], dtype=torch.float)
    labels_tensor = torch.tensor([padded_labels], dtype=torch.long)
    return points_tensor, labels_tensor


def build_boxes(objects: list[Object]) -> Optional[torch.Tensor]:
    """将 Object 列表转换为 SAM3 期望的 3D tensor 格式

    SAM3 期望: [image, box, coordinates] = [1, M, 4]
    """
    box_list = []
    for obj in objects:
        if obj.box is not None:
            box_list.append(list(obj.box.coords))

    if not box_list:
        return None
    return torch.tensor([box_list], dtype=torch.float)


def mcp_to_sam3(req: MCPRequest) -> SAM3Request:
    """将 MCPRequest 转换为 SAM3Request"""
    image = decode_image(req.image)

    input_points, input_labels = build_points(req.objects)
    input_boxes = build_boxes(req.objects)
    input_masks = decode_mask(req.prev_mask) if req.prev_mask else None

    return SAM3Request(
        image=image,
        input_points=input_points,
        input_labels=input_labels,
        input_boxes=input_boxes,
        input_masks=input_masks,
        multimask_output=req.multimask_output,
    )
