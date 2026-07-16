from unittest.mock import MagicMock

from sam3.schemas import SegmentRequest
from sam3.tracker import Sam3Tracker


def _make_tracker():
    models = MagicMock()
    models.tracker_model = MagicMock()
    models.tracker_processor = MagicMock()
    models.device = "cpu"
    return Sam3Tracker(models)


def test_build_points_with_positive():
    tracker = _make_tracker()
    req = SegmentRequest(image_path="test.png", p_point=[[100, 200], [300, 400]])
    points, labels = tracker._build_points(req)
    assert points is not None
    assert points.shape == (1, 2, 1, 2)
    assert labels is not None
    assert labels.tolist() == [[[1, 1]]]


def test_build_points_with_negative():
    tracker = _make_tracker()
    req = SegmentRequest(image_path="test.png", n_point=[[50, 50]])
    points, labels = tracker._build_points(req)
    assert points is not None
    assert labels.tolist() == [[[0]]]


def test_build_points_mixed():
    tracker = _make_tracker()
    req = SegmentRequest(image_path="test.png", p_point=[[100, 200]], n_point=[[50, 50]])
    points, labels = tracker._build_points(req)
    assert points.shape == (1, 2, 1, 2)
    assert labels.tolist() == [[[1, 0]]]


def test_build_points_none():
    tracker = _make_tracker()
    req = SegmentRequest(image_path="test.png", boxes=[[1, 2, 3, 4]])
    points, labels = tracker._build_points(req)
    assert points is None
    assert labels is None


def test_build_boxes():
    tracker = _make_tracker()
    req = SegmentRequest(image_path="test.png", boxes=[[75, 275, 1725, 850]])
    boxes = tracker._build_boxes(req)
    assert boxes is not None
    assert boxes.shape == (1, 1, 4)


def test_build_boxes_multiple():
    tracker = _make_tracker()
    req = SegmentRequest(image_path="test.png", boxes=[[1, 2, 3, 4], [5, 6, 7, 8]])
    boxes = tracker._build_boxes(req)
    assert boxes.shape == (1, 2, 4)


def test_build_boxes_none():
    tracker = _make_tracker()
    req = SegmentRequest(image_path="test.png", p_point=[[0, 0]])
    boxes = tracker._build_boxes(req)
    assert boxes is None


def test_build_masks_none():
    tracker = _make_tracker()
    req = SegmentRequest(image_path="test.png", p_point=[[0, 0]])
    masks = tracker._build_masks(req)
    assert masks is None
