import pytest
from pydantic import ValidationError

from sam3.schemas import SegmentRequest


def test_valid_point_request():
    req = SegmentRequest(image_path="test.png", p_point=[[100, 200]])
    assert req.image_path == "test.png"
    assert req.p_point == [[100, 200]]


def test_valid_box_request():
    req = SegmentRequest(image_path="test.png", boxes=[[75, 275, 1725, 850]])
    assert req.boxes == [[75, 275, 1725, 850]]


def test_valid_prev_mask_request():
    req = SegmentRequest(image_path="test.png", prev_mask="mask.png")
    assert req.prev_mask == "mask.png"


def test_valid_mixed_request():
    req = SegmentRequest(
        image_path="test.png",
        p_point=[[100, 200]],
        n_point=[[50, 50]],
        boxes=[[10, 10, 100, 100]],
    )
    assert len(req.p_point) == 1
    assert len(req.n_point) == 1
    assert len(req.boxes) == 1


def test_no_prompt_raises():
    with pytest.raises(ValidationError, match="至少需要一种提示"):
        SegmentRequest(image_path="test.png")


def test_empty_lists_raises():
    with pytest.raises(ValidationError, match="至少需要一种提示"):
        SegmentRequest(image_path="test.png", p_point=[], n_point=[], boxes=[])


def test_box_wrong_count_raises():
    with pytest.raises(ValidationError, match="boxes\\[0\\] 需要恰好 4 个值"):
        SegmentRequest(image_path="test.png", boxes=[[75, 275]])


def test_box_too_many_values_raises():
    with pytest.raises(ValidationError, match="boxes\\[0\\] 需要恰好 4 个值"):
        SegmentRequest(image_path="test.png", boxes=[[1, 2, 3, 4, 5]])


def test_multimask_default_true():
    req = SegmentRequest(image_path="test.png", p_point=[[0, 0]])
    assert req.multimask_output is True


def test_multimask_false():
    req = SegmentRequest(image_path="test.png", p_point=[[0, 0]], multimask_output=False)
    assert req.multimask_output is False


def test_missing_image_path_raises():
    with pytest.raises(ValidationError):
        SegmentRequest(p_point=[[0, 0]])
