import base64
from io import BytesIO

import pytest
from PIL import Image
from pydantic import ValidationError

from sam3.schemas.mcp import BoundingBox, MCPRequest, Object, PointPrompt


def _make_base64_image():
    img = Image.new("RGB", (100, 100), color=(255, 255, 255))
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


# --- PointPrompt ---


def test_point_prompt_positive():
    p = PointPrompt(coords=(100, 200), label=1)
    assert p.coords == (100, 200)
    assert p.label == 1


def test_point_prompt_negative():
    p = PointPrompt(coords=(50, 50), label=0)
    assert p.label == 0


def test_point_prompt_invalid_label():
    with pytest.raises(ValidationError, match="label 必须是 0 或 1"):
        PointPrompt(coords=(10, 10), label=2)


# --- BoundingBox ---


def test_bounding_box_valid():
    b = BoundingBox(coords=(75.0, 275.0, 1725.0, 850.0))
    assert b.coords == (75.0, 275.0, 1725.0, 850.0)


# --- Object ---


def test_object_points_only():
    obj = Object(points=[PointPrompt(coords=(100, 200), label=1)])
    assert len(obj.points) == 1
    assert obj.box is None


def test_object_box_only():
    obj = Object(box=BoundingBox(coords=(1, 2, 3, 4)))
    assert obj.points == []
    assert obj.box is not None


def test_object_mixed():
    obj = Object(
        points=[PointPrompt(coords=(10, 10), label=1)],
        box=BoundingBox(coords=(1, 2, 3, 4)),
    )
    assert len(obj.points) == 1
    assert obj.box is not None


def test_object_no_prompt_raises():
    with pytest.raises(ValidationError, match="至少需要一个提示"):
        Object()


# --- MCPRequest ---


def test_valid_single_object():
    image_b64 = _make_base64_image()
    req = MCPRequest(
        image=image_b64,
        objects=[Object(points=[PointPrompt(coords=(100, 200), label=1)])],
    )
    assert len(req.objects) == 1
    assert req.multimask_output is True


def test_valid_multi_object():
    image_b64 = _make_base64_image()
    req = MCPRequest(
        image=image_b64,
        objects=[
            Object(points=[PointPrompt(coords=(100, 200), label=1)]),
            Object(box=BoundingBox(coords=(10, 10, 100, 100))),
        ],
    )
    assert len(req.objects) == 2


def test_valid_prev_mask():
    image_b64 = _make_base64_image()
    mask_b64 = _make_base64_image()
    req = MCPRequest(
        image=image_b64,
        objects=[Object(points=[PointPrompt(coords=(0, 0), label=1)])],
        prev_mask=mask_b64,
    )
    assert req.prev_mask == mask_b64


def test_empty_objects_raises():
    image_b64 = _make_base64_image()
    with pytest.raises(ValidationError, match="objects 和 prev_mask 至少提供一个"):
        MCPRequest(image=image_b64, objects=[])


def test_mask_only_passes():
    image_b64 = _make_base64_image()
    mask_b64 = _make_base64_image()
    req = MCPRequest(image=image_b64, objects=[], prev_mask=mask_b64)
    assert req.objects == []
    assert req.prev_mask == mask_b64


def test_missing_image_raises():
    with pytest.raises(ValidationError):
        MCPRequest(objects=[Object(points=[PointPrompt(coords=(0, 0), label=1)])])  # type: ignore[call-arg]


def test_multimask_false():
    image_b64 = _make_base64_image()
    req = MCPRequest(
        image=image_b64,
        objects=[Object(points=[PointPrompt(coords=(0, 0), label=1)])],
        multimask_output=False,
    )
    assert req.multimask_output is False
