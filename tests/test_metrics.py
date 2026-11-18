import numpy as np
import pytest

from camus_seg.metrics import aggregate, binary_metrics, bootstrap_mean


def test_identical_masks():
    mask = np.zeros((10, 10), bool)
    mask[2:7, 3:8] = True
    result = binary_metrics(mask, mask, (.7, .2))
    assert result["dice"] == result["iou"] == 1
    assert result["hd95_mm"] == result["assd_mm"] == 0


@pytest.mark.parametrize("axis,expected", [(0, 2.0), (1, .5)])
def test_physical_spacing_axis_order(axis, expected):
    target, prediction = np.zeros((5, 5), bool), np.zeros((5, 5), bool)
    target[2, 2] = True
    point = [2, 2]
    point[axis] += 1
    prediction[tuple(point)] = True
    result = binary_metrics(prediction, target, (2.0, .5))
    assert result["hd95_mm"] == result["assd_mm"] == expected
    assert result["dice"] == 0


def test_overlap_formula():
    result = binary_metrics([[1, 1], [0, 0]], [[1, 0], [1, 0]], (1, 1))
    assert result["dice"] == .5
    assert result["iou"] == pytest.approx(1/3)


@pytest.mark.parametrize("prediction,target,dice,distance,failure", [
    (False, False, 1, 0, False), (False, True, 0, None, True),
    (True, False, 0, None, True)])
def test_empty_mask_policy(prediction, target, dice, distance, failure):
    result = binary_metrics(np.full((3, 3), prediction), np.full((3, 3), target), (1, 1))
    assert result["dice"] == dice
    assert result["hd95_mm"] == distance
    assert result["surface_failure"] == failure


def test_patients_have_equal_weight_and_missing_surfaces_are_counted():
    zero = binary_metrics(np.zeros((2, 2)), np.ones((2, 2)), (1, 1))
    one = binary_metrics(np.ones((2, 2)), np.ones((2, 2)), (1, 1))
    rows = [dict(patient="a", **zero) for _ in range(10)] + [dict(patient="b", **one)]
    result = aggregate(rows, 100)
    assert result["dice"]["mean"] == .5
    assert result["dice"]["n_patients"] == 2
    assert result["surface_failures"] == 10
    assert result["hd95_mm"]["excluded_frame_class_pairs"] == 10
    assert result["hd95_mm"]["n_patients"] == 1
    assert result["hd95_mm"]["ci95"] is None


def test_bootstrap_seed_and_single_patient():
    assert bootstrap_mean([.1, .5, .9], 100, 9) == bootstrap_mean([.1, .5, .9], 100, 9)
    assert bootstrap_mean([.5], 100)["ci95"] is None


@pytest.mark.parametrize("spacing", [(0, 1), (float("nan"), 1), (1,)])
def test_invalid_spacing_rejected(spacing):
    with pytest.raises(ValueError):
        binary_metrics(np.ones((2, 2)), np.ones((2, 2)), spacing)
