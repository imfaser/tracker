import os

import pytest

from sam3.models import load_models
from sam3.schemas import SegmentRequest
from sam3.tracker import Sam3Tracker


@pytest.fixture(scope="module")
def tracker():
    os.environ["MODEL_NAME"] = r"D:\code\tmp\sam3"
    models = load_models()
    return Sam3Tracker(models)


@pytest.mark.integration
def test_segment_with_point(test_image, tracker, tmp_path):
    os.environ["OUTPUT_DIR"] = str(tmp_path)
    req = SegmentRequest(image_path=test_image, p_point=[[100, 100]])
    result = tracker.segment(req)
    assert os.path.exists(result)
    assert result.endswith("_mask.png")


@pytest.mark.integration
def test_segment_with_box(test_image, tracker, tmp_path):
    os.environ["OUTPUT_DIR"] = str(tmp_path)
    req = SegmentRequest(image_path=test_image, boxes=[[50, 50, 150, 150]])
    result = tracker.segment(req)
    assert os.path.exists(result)


@pytest.mark.integration
def test_segment_with_prev_mask(test_image, tracker, tmp_path):
    os.environ["OUTPUT_DIR"] = str(tmp_path)

    req1 = SegmentRequest(image_path=test_image, p_point=[[100, 100]])
    first_mask = tracker.segment(req1)

    req2 = SegmentRequest(image_path=test_image, prev_mask=first_mask, p_point=[[100, 100]])
    second_mask = tracker.segment(req2)
    assert os.path.exists(second_mask)
    assert first_mask != second_mask
