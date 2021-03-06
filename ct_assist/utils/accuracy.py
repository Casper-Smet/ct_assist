from multiprocessing import Pool
from typing import List, Tuple

import numpy as np
from sklearn.metrics import mean_squared_error
from tqdm import tqdm
# from tqdm.contrib.concurrent import process_map

from ct_assist import transform
from ct_assist.exceptions import DimensionError


def _extract_props(params):
    cam = transform.fit(**params)
    return (cam.roll_deg, cam.tilt_deg, cam.heading_deg, cam.elevation_m)


def camera_properties(X_test: List[dict], Y_true: List[Tuple[float, float, float, float]], verbose: bool = True, mp: bool = True) -> float:
    """Accuracy test for camera properties (roll_deg, tilt_deg, heading_deg, elevation_m)

    :param X_test: List of kwargs for `transform.fit`
    :type X_test: List[dict]
    :param Y_true: Spatial orientation roll_deg, tilt_deg, heading_deg, elevation_m
    :type Y_true: List[Tuple[float, float, float, float]]
    :param verbose: TQDM, defaults to True
    :type verbose: bool
    :param mp: Multiprocessing used, defaults to True
    :type mp: bool
    :return: RMSE for roll_deg, tilt_deg, heading_deg, elevation_m
    :rtype: float
    """
    if mp:
        with Pool() as p:
            Y_pred = p.map(_extract_props, X_test)
    else:
        Y_pred = list(map(_extract_props, tqdm(X_test, disable=not verbose)))
    Y_pred = np.array(Y_pred)
    Y_true = np.array(Y_true)

    return (mean_squared_error(Y_true[:, 0], Y_pred[:, 0], squared=False),
            mean_squared_error(Y_true[:, 1], Y_pred[:, 1], squared=False),
            mean_squared_error(Y_true[:, 2], Y_pred[:, 2], squared=False),
            mean_squared_error(Y_true[:, 3], Y_pred[:, 3], squared=False)), Y_pred, Y_true


def calc_area(poly: np.ndarray) -> float:
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
    ar = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
    return ar if not np.isnan(ar).any() else 0


def _area(params):
    return sum(calc_area(
        poly[:, :2]) for poly in transform.fit_transform(**params))


def area(X_test: List[dict], y_true: List[float], verbose: bool = True, mp: bool = True) -> float:
    """Accuracy test for area.

    :param X_test: List of kwargs for transform.fit
    :type X_test: List[dict]
    :param y_true: Areas of Polygons
    :type y_true: List[float]
    :param verbose: TQDM, defaults to True
    :type verbose: bool
    :param mp: Multiprocessing used, defaults to True
    :type mp: bool
    :return: RMSE, Aka loss
    :rtype: float
    """
    if mp:
        with Pool() as p:
            Y_pred = p.map(_area, X_test)
    else:
        Y_pred = list(map(_area, tqdm(X_test, disable=not verbose)))
    return mean_squared_error(y_true, Y_pred, squared=False), Y_pred
