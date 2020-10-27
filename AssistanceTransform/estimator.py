"""
Area estimations

This module is used for estimating the area of a polygon, and estimating the rate at which the polygon increases in size (a.k.a., the release rate)
"""

import datetime as dt
from typing import List

import numpy as np

from AssistanceTransform.exceptions import DimensionError


def area(poly: np.ndarray) -> float:
    """Estimates the area of a polygon using the shoelace formula.

    :param poly: Polygon
    :type poly: Numpy array
    :return: Area of polygon
    :rtype: float
    """
    if not isinstance(poly, np.ndarray):
        raise TypeError(f"Expected poly to be `np.ndarray`, not {type(poly)}")
    if len(poly.shape) != 2:
        raise DimensionError(f"Expected dimension (n, 2), not {poly.shape}")
    if poly.shape[1] != 2:
        raise DimensionError(f"Expected dimension (n, 2), not {poly.shape}")
    x, y = poly.T
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def estimate_release_rate(polys: List[np.ndarray], time_indices: List[dt.datetime]) -> float:
    """Estimates release rate of fluid based on size of fluid spill at `n` different time indices.

    :param polys: List of polygons describing fluid spill
    :type polys: List[np.ndarray]
    :param time_indices: List of time indices
    :type time_indices: List[datetime]
    :return: m^2 s^-1
    :rtype: float
    """
    raise NotImplementedError(
        "Function `estimate_release_rate` in module `estimator.py` has not yet been implemented")
