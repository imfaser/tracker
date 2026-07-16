import os
from typing import Optional

import torch

from sam3.io import load_image, load_mask, save_mask
from sam3.models import LoadedModels
from sam3.schemas import SegmentRequest


class Sam3Tracker:
    """SAM3 图像分割器"""

    def __init__(self, models: LoadedModels) -> None:
        self._model = models.tracker_model
        self._processor = models.tracker_processor
        self._device = models.device

    def segment(self, req: SegmentRequest) -> str:
        """执行图像分割，返回 mask 文件路径"""
        image = load_image(req.image_path)

        input_points, input_labels = self._build_points(req)
        input_boxes = self._build_boxes(req)
        input_masks = self._build_masks(req)

        inputs = self._processor(
            images=image,
            input_points=input_points,
            input_labels=input_labels,
            input_boxes=input_boxes,
            return_tensors="pt",
        ).to(self._device)

        with torch.no_grad():
            outputs = self._model(
                **inputs,
                input_masks=input_masks,
                multimask_output=req.multimask_output,
            )

        best_idx = outputs.iou_scores.argmax(dim=-1).cpu()

        masks = self._processor.post_process_masks(
            outputs.pred_masks.cpu(), inputs["original_sizes"]
        )

        batch_size, point_batch_size = best_idx.shape
        result_mask = torch.stack(masks)[
            torch.arange(batch_size), torch.arange(point_batch_size), best_idx
        ].squeeze()

        output_dir = os.getenv("OUTPUT_DIR", "./out")
        return save_mask(result_mask, output_dir)

    def _build_points(self, req: SegmentRequest) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if not req.p_point and not req.n_point:
            return None, None

        points = []
        labels = []
        for pt in req.p_point:
            points.append([pt])
            labels.append(1)
        for pt in req.n_point:
            points.append([pt])
            labels.append(0)

        # SAM3 期望 4D points: [image, object, point_per_object, coordinates]
        # SAM3 期望 3D labels: [image, object, point_per_object]
        points_tensor = torch.tensor([points], dtype=torch.float)
        labels_tensor = torch.tensor([[labels]], dtype=torch.long)
        return points_tensor, labels_tensor

    def _build_boxes(self, req: SegmentRequest) -> Optional[torch.Tensor]:
        if not req.boxes:
            return None
        return torch.tensor([req.boxes], dtype=torch.float)

    def _build_masks(self, req: SegmentRequest) -> Optional[torch.Tensor]:
        if not req.prev_mask:
            return None
        return load_mask(req.prev_mask).to(self._device)
