import base64
from io import BytesIO

from PIL import Image

from sam3.schemas.mcp import MCPRequest
from transform.segment import build_boxes, build_points, mcp_to_sam3


def _make_base64_image(width=200, height=200):
    """创建测试用的 base64 图像字符串"""
    img = Image.new("RGB", (width, height), color=(255, 255, 255))
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def test_build_points_positive():
    points, labels = build_points([[100, 200], [300, 400]], [])
    assert points is not None
    assert points.shape == (1, 2, 1, 2)
    assert labels is not None
    assert labels.tolist() == [[[1, 1]]]


def test_build_points_negative():
    points, labels = build_points([], [[50, 50]])
    assert points is not None
    assert labels is not None
    assert labels.tolist() == [[[0]]]


def test_build_points_mixed():
    points, labels = build_points([[100, 200]], [[50, 50]])
    assert points is not None
    assert points.shape == (1, 2, 1, 2)
    assert labels is not None
    assert labels.tolist() == [[[1, 0]]]


def test_build_points_none():
    points, labels = build_points([], [])
    assert points is None
    assert labels is None


def test_build_boxes():
    boxes = build_boxes([[75, 275, 1725, 850]])
    assert boxes is not None
    assert boxes.shape == (1, 1, 4)


def test_build_boxes_multiple():
    boxes = build_boxes([[1, 2, 3, 4], [5, 6, 7, 8]])
    assert boxes is not None
    assert boxes.shape == (1, 2, 4)


def test_build_boxes_none():
    boxes = build_boxes([])
    assert boxes is None


def test_mcp_to_sam3_with_points():
    image_b64 = _make_base64_image()
    req = MCPRequest(image=image_b64, p_point=[[100, 200]])
    sam3_req = mcp_to_sam3(req)

    assert sam3_req.image.mode == "RGB"
    assert sam3_req.input_points is not None
    assert sam3_req.input_points.shape == (1, 1, 1, 2)
    assert sam3_req.input_labels is not None
    assert sam3_req.input_boxes is None
    assert sam3_req.input_masks is None
    assert sam3_req.multimask_output is True


def test_mcp_to_sam3_with_boxes():
    image_b64 = _make_base64_image()
    req = MCPRequest(image=image_b64, boxes=[[10, 10, 100, 100]])
    sam3_req = mcp_to_sam3(req)

    assert sam3_req.input_points is None
    assert sam3_req.input_boxes is not None
    assert sam3_req.input_boxes.shape == (1, 1, 4)


def test_mcp_to_sam3_with_multimask_false():
    image_b64 = _make_base64_image()
    req = MCPRequest(image=image_b64, p_point=[[0, 0]], multimask_output=False)
    sam3_req = mcp_to_sam3(req)

    assert sam3_req.multimask_output is False
