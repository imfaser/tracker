import torch

from sam3.models import LoadedModels
from sam3.schemas.sam3 import SAM3Request


class Sam3Tracker:
    """SAM3 图像分割器"""

    def __init__(self, models: LoadedModels) -> None:
        self._model = models.tracker_model
        self._processor = models.tracker_processor
        self._device = models.device

    def segment(self, req: SAM3Request) -> torch.Tensor:
        """执行图像分割，多物体时合并为单个 mask 返回"""
        inputs = self._processor(
            images=req.image,
            input_points=req.input_points,
            input_labels=req.input_labels,
            input_boxes=req.input_boxes,
            return_tensors="pt",
        ).to(self._device)

        input_masks = req.input_masks.to(self._device) if req.input_masks is not None else None

        with torch.no_grad():
            outputs = self._model(
                **inputs,
                input_masks=input_masks,
                multimask_output=req.multimask_output,
            )

        best_idx = outputs.iou_scores.argmax(dim=-1).cpu()

        masks = self._processor.post_process_masks(outputs.pred_masks.cpu(), inputs["original_sizes"])

        if best_idx.dim() == 1:
            best_idx = best_idx.unsqueeze(0)

        batch_size, point_batch_size = best_idx.shape
        stacked = torch.stack(masks)  # (batch, objects, num_masks, H, W)
        per_object = stacked[torch.arange(batch_size), torch.arange(point_batch_size), best_idx]

        merged = per_object[0].bool()
        for i in range(1, per_object.shape[0]):
            merged = merged | per_object[i].bool()

        return merged.float()
