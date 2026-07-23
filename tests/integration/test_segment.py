import base64
import os

import pytest
import torch

from sam3.config import Settings
from sam3.io import save_mask
from sam3.models import load_models
from sam3.schemas.mcp import BoundingBox, MCPRequest, Object, PointPrompt
from sam3.tracker import Sam3Tracker
from transform.segment import mcp_to_sam3

IMAGE_SIZE = (200, 200)


def _image_to_base64(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _assert_valid_mask(mask: torch.Tensor, expected_shape=IMAGE_SIZE) -> None:
    """验证 mask 基本性质：2D、形状正确、非空、有前景和背景"""
    assert mask.dim() == 2, f"mask 应为 2D，实际 {mask.dim()}D，shape={tuple(mask.shape)}"
    assert tuple(mask.shape) == expected_shape
    assert mask.dtype.is_floating_point
    binary = mask > 0.5
    fg_count = binary.sum().item()
    bg_count = (~binary).sum().item()
    assert fg_count > 0, "mask 全为背景，分割失败"
    assert bg_count > 0, "mask 全为前景，分割失败"


@pytest.fixture(scope="module")
def tracker():
    settings = Settings(
        model_name=r"D:\code\tmp\sam3",
        device="auto",
        output_dir="./out",
    )
    models = load_models(settings)
    return Sam3Tracker(models), settings


@pytest.mark.integration
def test_segment_with_point(test_image, tracker, tmp_path):
    tracker_instance, settings = tracker
    settings.output_dir = str(tmp_path)

    image_b64 = _image_to_base64(test_image)
    req = MCPRequest(
        image=image_b64,
        objects=[Object(points=[PointPrompt(coords=(100, 100), label=1)])],
    )
    sam3_req = mcp_to_sam3(req)

    result_mask = tracker_instance.segment(sam3_req)
    result_path = save_mask(result_mask, settings.output_dir)

    assert os.path.exists(result_path)
    assert result_path.endswith("_mask.png")
    _assert_valid_mask(result_mask)


@pytest.mark.integration
def test_segment_with_box(test_image, tracker, tmp_path):
    tracker_instance, settings = tracker
    settings.output_dir = str(tmp_path)

    image_b64 = _image_to_base64(test_image)
    req = MCPRequest(
        image=image_b64,
        objects=[Object(box=BoundingBox(coords=(50, 50, 150, 150)))],
    )
    sam3_req = mcp_to_sam3(req)

    result_mask = tracker_instance.segment(sam3_req)
    result_path = save_mask(result_mask, settings.output_dir)

    assert os.path.exists(result_path)
    _assert_valid_mask(result_mask)


@pytest.mark.integration
def test_segment_with_prev_mask(test_image, tracker, tmp_path):
    tracker_instance, settings = tracker
    settings.output_dir = str(tmp_path)

    image_b64 = _image_to_base64(test_image)

    req1 = MCPRequest(
        image=image_b64,
        objects=[Object(points=[PointPrompt(coords=(100, 100), label=1)])],
    )
    sam3_req1 = mcp_to_sam3(req1)
    first_mask = tracker_instance.segment(sam3_req1)
    first_mask_path = save_mask(first_mask, settings.output_dir)

    with open(first_mask_path, "rb") as f:
        mask_b64 = base64.b64encode(f.read()).decode("utf-8")

    req2 = MCPRequest(
        image=image_b64,
        prev_mask=mask_b64,
        objects=[Object(points=[PointPrompt(coords=(100, 100), label=1)])],
    )
    sam3_req2 = mcp_to_sam3(req2)
    second_mask = tracker_instance.segment(sam3_req2)
    second_mask_path = save_mask(second_mask, settings.output_dir)

    assert os.path.exists(second_mask_path)
    assert first_mask_path != second_mask_path
    _assert_valid_mask(second_mask)


@pytest.mark.integration
def test_segment_mask_only(test_image, tracker, tmp_path):
    """仅 prev_mask（objects=[]）端到端分割"""
    tracker_instance, settings = tracker
    settings.output_dir = str(tmp_path)

    image_b64 = _image_to_base64(test_image)

    req1 = MCPRequest(
        image=image_b64,
        objects=[Object(points=[PointPrompt(coords=(100, 100), label=1)])],
    )
    sam3_req1 = mcp_to_sam3(req1)
    first_mask = tracker_instance.segment(sam3_req1)
    first_mask_path = save_mask(first_mask, settings.output_dir)

    with open(first_mask_path, "rb") as f:
        mask_b64 = base64.b64encode(f.read()).decode("utf-8")

    req2 = MCPRequest(
        image=image_b64,
        prev_mask=mask_b64,
        objects=[],
    )
    sam3_req2 = mcp_to_sam3(req2)

    assert sam3_req2.input_points is None
    assert sam3_req2.input_labels is None
    assert sam3_req2.input_boxes is None
    assert sam3_req2.input_masks is not None

    second_mask = tracker_instance.segment(sam3_req2)
    second_mask_path = save_mask(second_mask, settings.output_dir)

    assert os.path.exists(second_mask_path)
    assert first_mask_path != second_mask_path
    _assert_valid_mask(second_mask)


@pytest.mark.integration
def test_segment_multi_object_merges_mask(test_image, tracker, tmp_path):
    tracker_instance, settings = tracker
    settings.output_dir = str(tmp_path)

    image_b64 = _image_to_base64(test_image)
    req = MCPRequest(
        image=image_b64,
        objects=[
            Object(points=[PointPrompt(coords=(80, 80), label=1)]),
            Object(points=[PointPrompt(coords=(120, 120), label=1)]),
        ],
    )
    sam3_req = mcp_to_sam3(req)

    result_mask = tracker_instance.segment(sam3_req)
    result_path = save_mask(result_mask, settings.output_dir)

    assert os.path.exists(result_path)
    _assert_valid_mask(result_mask)


@pytest.mark.integration
def test_segment_multi_box_merges_mask(test_image, tracker, tmp_path):
    """2 个 box 时 point_batch_size=2，曾触发 save_mask 无限循环"""
    tracker_instance, settings = tracker
    settings.output_dir = str(tmp_path)

    image_b64 = _image_to_base64(test_image)
    req = MCPRequest(
        image=image_b64,
        objects=[
            Object(box=BoundingBox(coords=(40, 40, 100, 100))),
            Object(box=BoundingBox(coords=(100, 100, 160, 160))),
        ],
    )
    sam3_req = mcp_to_sam3(req)

    result_mask = tracker_instance.segment(sam3_req)
    result_path = save_mask(result_mask, settings.output_dir)

    assert os.path.exists(result_path)
    assert result_path.endswith("_mask.png")
    _assert_valid_mask(result_mask)


@pytest.mark.integration
def test_segment_box_with_points(test_image, tracker, tmp_path):
    """单对象同时带 box + point：模型 forward 要求 points/boxes 的 object 维对齐"""
    tracker_instance, settings = tracker
    settings.output_dir = str(tmp_path)

    image_b64 = _image_to_base64(test_image)
    req = MCPRequest(
        image=image_b64,
        objects=[
            Object(
                points=[PointPrompt(coords=(100, 100), label=1)],
                box=BoundingBox(coords=(50, 50, 150, 150)),
            )
        ],
    )
    sam3_req = mcp_to_sam3(req)

    assert sam3_req.input_points is not None
    assert sam3_req.input_boxes is not None
    assert sam3_req.input_points.shape[1] == sam3_req.input_boxes.shape[1]

    result_mask = tracker_instance.segment(sam3_req)
    result_path = save_mask(result_mask, settings.output_dir)

    assert os.path.exists(result_path)
    _assert_valid_mask(result_mask)


@pytest.mark.integration
def test_segment_multi_box_with_points(test_image, tracker, tmp_path):
    """两对象各自带 box + point：points 和 boxes 的 object 维必须都为 2 且对齐"""
    tracker_instance, settings = tracker
    settings.output_dir = str(tmp_path)

    image_b64 = _image_to_base64(test_image)
    req = MCPRequest(
        image=image_b64,
        objects=[
            Object(
                points=[PointPrompt(coords=(70, 70), label=1)],
                box=BoundingBox(coords=(40, 40, 100, 100)),
            ),
            Object(
                points=[PointPrompt(coords=(130, 130), label=1)],
                box=BoundingBox(coords=(100, 100, 160, 160)),
            ),
        ],
    )
    sam3_req = mcp_to_sam3(req)

    assert sam3_req.input_points is not None
    assert sam3_req.input_boxes is not None
    assert sam3_req.input_points.shape[1] == 2
    assert sam3_req.input_boxes.shape[1] == 2
    assert sam3_req.input_points.shape[1] == sam3_req.input_boxes.shape[1]

    result_mask = tracker_instance.segment(sam3_req)
    result_path = save_mask(result_mask, settings.output_dir)

    assert os.path.exists(result_path)
    _assert_valid_mask(result_mask)
