import base64
from io import BytesIO

from PIL import Image

from sam3.schemas.mcp import BoundingBox, MCPRequest, Object, PointPrompt
from transform.segment import build_boxes, build_points, mcp_to_sam3


def _make_base64_image(width=200, height=200):
    img = Image.new("RGB", (width, height), color=(255, 255, 255))
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


# --- build_points ---


def test_build_points_single_object_single_point():
    objects = [Object(points=[PointPrompt(coords=(100, 200), label=1)])]
    points, labels = build_points(objects)
    assert points is not None
    assert points.shape == (1, 1, 1, 2)
    assert labels is not None
    assert labels.shape == (1, 1, 1)
    assert labels.tolist() == [[[1]]]  # type: ignore[union-attr]


def test_build_points_single_object_multi_point():
    objects = [
        Object(
            points=[
                PointPrompt(coords=(100, 200), label=1),
                PointPrompt(coords=(300, 400), label=0),
            ]
        )
    ]
    points, labels = build_points(objects)
    assert points is not None
    assert points.shape == (1, 1, 2, 2)
    assert labels is not None
    assert labels.tolist() == [[[1, 0]]]  # type: ignore[union-attr]


def test_build_points_multi_object_equal_length():
    objects = [
        Object(points=[PointPrompt(coords=(100, 200), label=1)]),
        Object(points=[PointPrompt(coords=(300, 400), label=1)]),
    ]
    points, labels = build_points(objects)
    assert points is not None
    assert points.shape == (1, 2, 1, 2)
    assert labels is not None
    assert labels.tolist() == [[[1], [1]]]  # type: ignore[union-attr]


def test_build_points_multi_object_padding():
    objects = [
        Object(points=[PointPrompt(coords=(100, 200), label=1)]),
        Object(
            points=[
                PointPrompt(coords=(300, 400), label=1),
                PointPrompt(coords=(500, 600), label=0),
            ]
        ),
    ]
    points, labels = build_points(objects)
    assert points is not None
    assert points.shape == (1, 2, 2, 2)
    assert labels is not None
    assert labels.tolist() == [[[1, -10], [1, 0]]]  # type: ignore[union-attr]


def test_build_points_empty():
    points, labels = build_points([])
    assert points is None
    assert labels is None


# --- build_boxes ---


def test_build_boxes_single():
    objects = [Object(box=BoundingBox(coords=(75, 275, 1725, 850)))]
    boxes = build_boxes(objects)
    assert boxes is not None
    assert boxes.shape == (1, 1, 4)


def test_build_boxes_multiple():
    objects = [
        Object(box=BoundingBox(coords=(1, 2, 3, 4))),
        Object(box=BoundingBox(coords=(5, 6, 7, 8))),
    ]
    boxes = build_boxes(objects)
    assert boxes is not None
    assert boxes.shape == (1, 2, 4)


def test_build_boxes_partial():
    objects = [
        Object(box=BoundingBox(coords=(1, 2, 3, 4))),
        Object(points=[PointPrompt(coords=(10, 10), label=1)]),
        Object(box=BoundingBox(coords=(5, 6, 7, 8))),
    ]
    boxes = build_boxes(objects)
    assert boxes is not None
    assert boxes.shape == (1, 2, 4)


def test_build_boxes_none():
    objects = [Object(points=[PointPrompt(coords=(10, 10), label=1)])]
    boxes = build_boxes(objects)
    assert boxes is None


# --- mcp_to_sam3 end-to-end ---


def test_mcp_to_sam3_with_points():
    image_b64 = _make_base64_image()
    req = MCPRequest(
        image=image_b64,
        objects=[Object(points=[PointPrompt(coords=(100, 200), label=1)])],
    )
    sam3_req = mcp_to_sam3(req)

    assert sam3_req.image.mode == "RGB"
    assert sam3_req.input_points is not None
    assert sam3_req.input_points.shape == (1, 1, 1, 2)
    assert sam3_req.input_labels is not None
    assert sam3_req.input_labels.shape == (1, 1, 1)
    assert sam3_req.input_boxes is None
    assert sam3_req.input_masks is None
    assert sam3_req.multimask_output is True


def test_mcp_to_sam3_with_boxes():
    image_b64 = _make_base64_image()
    req = MCPRequest(
        image=image_b64,
        objects=[Object(box=BoundingBox(coords=(10, 10, 100, 100)))],
    )
    sam3_req = mcp_to_sam3(req)

    assert sam3_req.input_points is None
    assert sam3_req.input_boxes is not None
    assert sam3_req.input_boxes.shape == (1, 1, 4)


def test_mcp_to_sam3_mixed():
    image_b64 = _make_base64_image()
    req = MCPRequest(
        image=image_b64,
        objects=[
            Object(
                points=[PointPrompt(coords=(100, 200), label=1)],
                box=BoundingBox(coords=(10, 10, 100, 100)),
            )
        ],
    )
    sam3_req = mcp_to_sam3(req)
    assert sam3_req.input_points is not None
    assert sam3_req.input_boxes is not None
    assert sam3_req.input_points.shape[1] == sam3_req.input_boxes.shape[1]


def test_mcp_to_sam3_multi_object_box_and_points():
    """多对象各自带 box+point 时，points 和 boxes 的 object 维必须对齐"""
    image_b64 = _make_base64_image()
    req = MCPRequest(
        image=image_b64,
        objects=[
            Object(
                points=[PointPrompt(coords=(50, 50), label=1)],
                box=BoundingBox(coords=(10, 10, 80, 80)),
            ),
            Object(
                points=[PointPrompt(coords=(150, 150), label=1)],
                box=BoundingBox(coords=(120, 120, 190, 190)),
            ),
        ],
    )
    sam3_req = mcp_to_sam3(req)
    assert sam3_req.input_points is not None
    assert sam3_req.input_boxes is not None
    assert sam3_req.input_points.shape[1] == 2
    assert sam3_req.input_boxes.shape[1] == 2


def test_mcp_to_sam3_multimask_false():
    image_b64 = _make_base64_image()
    req = MCPRequest(
        image=image_b64,
        objects=[Object(points=[PointPrompt(coords=(0, 0), label=1)])],
        multimask_output=False,
    )
    sam3_req = mcp_to_sam3(req)
    assert sam3_req.multimask_output is False
