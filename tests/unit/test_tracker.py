from unittest.mock import MagicMock

import torch
from PIL import Image

from sam3.schemas.sam3 import SAM3Request
from sam3.tracker import Sam3Tracker


def _make_tracker():
    models = MagicMock()
    models.tracker_model = MagicMock()
    models.tracker_processor = MagicMock()
    models.device = "cpu"
    return Sam3Tracker(models)


def _make_sam3_request(**kwargs):
    """创建测试用的 SAM3Request"""
    defaults = {
        "image": Image.new("RGB", (100, 100)),
        "multimask_output": True,
    }
    defaults.update(kwargs)
    return SAM3Request(**defaults)


def test_tracker_segment_returns_tensor():
    tracker = _make_tracker()

    mock_outputs = MagicMock()
    mock_outputs.iou_scores = torch.tensor([[0.9, 0.8, 0.7]])
    mock_outputs.pred_masks = torch.randn(1, 3, 100, 100)
    tracker._model.return_value = mock_outputs  # type: ignore[attr-defined]

    tracker._processor.post_process_masks.return_value = [torch.randn(3, 100, 100)]  # type: ignore[attr-defined]
    mock_processor_result = MagicMock()
    mock_processor_result.to.return_value = {"original_sizes": torch.tensor([[100, 100]])}
    tracker._processor.return_value = mock_processor_result  # type: ignore[attr-defined]

    req = _make_sam3_request(input_points=torch.tensor([[[[50, 50]]]]))
    result = tracker.segment(req)

    assert isinstance(result, torch.Tensor)
    tracker._model.assert_called_once()  # type: ignore[attr-defined]


def test_tracker_segment_multi_object_merges_to_2d():
    """2 个 object（box）时，point_batch_size=2，结果应合并为 (H, W)"""
    tracker = _make_tracker()

    mock_outputs = MagicMock()
    mock_outputs.iou_scores = torch.tensor([[[0.9, 0.8, 0.7], [0.6, 0.95, 0.5]]])
    mock_outputs.pred_masks = torch.randn(1, 2, 3, 100, 100)
    tracker._model.return_value = mock_outputs  # type: ignore[attr-defined]

    tracker._processor.post_process_masks.return_value = [torch.randn(2, 3, 100, 100)]  # type: ignore[attr-defined]
    mock_processor_result = MagicMock()
    mock_processor_result.to.return_value = {"original_sizes": torch.tensor([[100, 100]])}
    tracker._processor.return_value = mock_processor_result  # type: ignore[attr-defined]

    req = _make_sam3_request(input_boxes=torch.tensor([[[10, 10, 50, 50], [60, 60, 90, 90]]]))
    result = tracker.segment(req)

    assert isinstance(result, torch.Tensor)
    assert result.dim() == 2
    assert result.shape == (100, 100)
