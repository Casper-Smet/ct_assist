from datetime.datetime import datetime
from typing import List

import numpy as np


def area(poly: np.ndarray) -> float:
    """Estimates the area of a polygon.

    :param poly: Polygon
    :type poly: Numpy array
    :return: Area of polygon in m
    :rtype: float
    """
    pass


def estimate_release_rate(polys: List[np.ndarray], time_indices: List[datetime]) -> float:
    """Estimates release rate of fluid based on size of fluid spill at `n` different time indices.

    :param polys: List of polygons describing fluid spill
    :type polys: List[np.ndarray]
    :param time_indices: List of time indices
    :type time_indices: List[datetime]
    :return: m^2 s^-1
    :rtype: float
    """
    pass
