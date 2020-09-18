import numpy as np
import pytest

from AssistanceTransform import estimator
from AssistanceTransform.exceptions import DimensionError


def test_area():
    """Estimates the area of a polygon."""
    # Basic, simple shape
    poly1 = np.array([[1, 1], [1, 3], [2, 4], [3, 3], [3, 1], [1, 1]])
    assert estimator.area(poly1) == 5

    # Simple shape with negatives
    poly2 = poly1 - 2
    assert estimator.area(poly2) == 5

    # More complex shape with floats
    poly3 = np.array([[1, 1], [1.5, 1.5], [1, 2.5], [0.5, 3],
                      [1, 3.5], [2.5, 1.5], [2, 0.5]])
    assert estimator.area(poly3) == 2.375

    # More complex shape, but smaller
    poly4 = poly3 / 15
    assert round(estimator.area(poly4), 3) == 0.011

    # More complex shape, but far greater
    poly5 = poly4 * 1000
    assert round(estimator.area(poly5)) == 10556

    # Check poly shape
    # 1D
    poly6 = np.array([1, 2])
    with pytest.raises(DimensionError) as excinfo:
        estimator.area(poly6)
    assert f"Expected dimension (n, 2), not {poly6.shape}" in str(excinfo)
    # nD where n > 2
    poly7 = np.array([[[1, 2]]])
    with pytest.raises(DimensionError) as excinfo:
        estimator.area(poly7)
    assert f"Expected dimension (n, 2), not {poly7.shape}" in str(excinfo)

    # 1D coordinate
    poly8 = np.array([[1]])
    with pytest.raises(DimensionError) as excinfo:
        estimator.area(poly8)
    assert f"Expected dimension (n, 2), not {poly8.shape}" in str(excinfo)

    # nD where n > 2 coordinate
    poly9 = np.array([[1, 2, 3]])
    with pytest.raises(DimensionError) as excinfo:
        estimator.area(poly9)
    assert f"Expected dimension (n, 2), not {poly9.shape}" in str(excinfo)

    # Check for correct type
    type_mistakes = ["str", 0, [0, 1]]
    for arr in type_mistakes:
        with pytest.raises(TypeError) as excinfo:
            estimator.area(arr)
        assert f"Expected poly to be `np.ndarray`, not {type(arr)}" in str(
            excinfo)


def test_estimate_release_rate():
    """Estimates release rate of fluid based on size of fluid spill at `n` different time indices."""
    assert False
