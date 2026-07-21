import base64
import os

import pytest

from sam3.config import Settings
from sam3.io import save_mask
from sam3.models import load_models
from sam3.schemas.mcp import BoundingBox, MCPRequest, Object, PointPrompt
from sam3.tracker import Sam3Tracker
from transform.segment import mcp_to_sam3


def _image_to_base64(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


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
    assert result_mask.dim() == 2
