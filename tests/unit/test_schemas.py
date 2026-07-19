import base64
from io import BytesIO

import pytest
from PIL import Image
from pydantic import ValidationError

from sam3.schemas.mcp import MCPRequest


def _make_base64_image():
    """创建测试用的 base64 图像字符串"""
    img = Image.new("RGB", (100, 100), color=(255, 255, 255))
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def test_valid_point_request():
    image_b64 = _make_base64_image()
    req = MCPRequest(image=image_b64, p_point=[[100, 200]])
    assert req.image == image_b64
    assert req.p_point == [[100, 200]]


def test_valid_box_request():
    image_b64 = _make_base64_image()
    req = MCPRequest(image=image_b64, boxes=[[75, 275, 1725, 850]])
    assert req.boxes == [[75, 275, 1725, 850]]


def test_valid_prev_mask_request():
    image_b64 = _make_base64_image()
    mask_b64 = _make_base64_image()
    req = MCPRequest(image=image_b64, prev_mask=mask_b64)
    assert req.prev_mask == mask_b64


def test_valid_mixed_request():
    image_b64 = _make_base64_image()
    req = MCPRequest(
        image=image_b64,
        p_point=[[100, 200]],
        n_point=[[50, 50]],
        boxes=[[10, 10, 100, 100]],
    )
    assert len(req.p_point) == 1
    assert len(req.n_point) == 1
    assert len(req.boxes) == 1


def test_no_prompt_raises():
    image_b64 = _make_base64_image()
    with pytest.raises(ValidationError, match="至少需要一种提示"):
        MCPRequest(image=image_b64)


def test_empty_lists_raises():
    image_b64 = _make_base64_image()
    with pytest.raises(ValidationError, match="至少需要一种提示"):
        MCPRequest(image=image_b64, p_point=[], n_point=[], boxes=[])


def test_box_wrong_count_raises():
    image_b64 = _make_base64_image()
    with pytest.raises(ValidationError, match="boxes\\[0\\] 需要恰好 4 个值"):
        MCPRequest(image=image_b64, boxes=[[75, 275]])


def test_box_too_many_values_raises():
    image_b64 = _make_base64_image()
    with pytest.raises(ValidationError, match="boxes\\[0\\] 需要恰好 4 个值"):
        MCPRequest(image=image_b64, boxes=[[1, 2, 3, 4, 5]])


def test_multimask_default_true():
    image_b64 = _make_base64_image()
    req = MCPRequest(image=image_b64, p_point=[[0, 0]])
    assert req.multimask_output is True


def test_multimask_false():
    image_b64 = _make_base64_image()
    req = MCPRequest(image=image_b64, p_point=[[0, 0]], multimask_output=False)
    assert req.multimask_output is False


def test_missing_image_raises():
    with pytest.raises(ValidationError):
        MCPRequest(p_point=[[0, 0]])  # type: ignore[call-arg]
