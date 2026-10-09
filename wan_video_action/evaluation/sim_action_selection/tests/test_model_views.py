import numpy as np
import pytest

from ..core.decision_capture import select_model_views


def test_default_preserves_every_native_view():
    views = np.arange(2 * 3 * 4 * 5 * 3, dtype=np.uint8).reshape(2, 3, 4, 5, 3)
    np.testing.assert_array_equal(select_model_views(views), views)


def test_model_camera_selection_preserves_native_observations():
    views = np.arange(2 * 3 * 4 * 5 * 3, dtype=np.uint8).reshape(2, 3, 4, 5, 3)
    original = views.copy()
    selected = select_model_views(views, [2, 0])
    np.testing.assert_array_equal(selected, original[:, [2, 0]])
    selected[:] = 0
    np.testing.assert_array_equal(views, original)


def test_main_camera_selection_keeps_camera_axis():
    views = np.zeros((1, 2, 224, 224, 3), dtype=np.uint8)
    assert select_model_views(views, [0]).shape == (1, 1, 224, 224, 3)


@pytest.mark.parametrize("indices", [[], [-1], [2], [0, 0], [0.0], [True], [[0]]])
def test_invalid_model_camera_selection_is_rejected(indices):
    with pytest.raises(ValueError):
        select_model_views(np.zeros((1, 2, 4, 4, 3), dtype=np.uint8), indices)
