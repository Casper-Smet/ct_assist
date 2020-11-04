from typing import List, Tuple

import numpy as np
from sklearn.metrics import mean_squared_error
from tqdm import tqdm

from ct_assist import transform, estimator


def camera_properties(X_test: List[dict], Y_true: List[Tuple[float, float, float, float]], verbose: bool = True) -> float:
    """Accuracy test for camera properties (roll_deg, tilt_deg, heading_deg, elevation_m)

    :param X_test: List of kwargs for `transform.fit`
    :type X_test: List[dict]
    :param Y_true: Spatial orientation roll_deg, tilt_deg, heading_deg, elevation_m
    :type Y_true: List[Tuple[float, float, float, float]]
    :param verbose: TQDM, defaults to True
    :type verbose: bool
    :return: RMSE for roll_deg, tilt_deg, heading_deg, elevation_m
    :rtype: float
    """
    cam_generator = map(lambda params: transform.fit(
        **params).orientation.parameters, tqdm(X_test, disable=not verbose))
    Y_pred = list(map(lambda cam: (cam.roll_deg, cam.tilt_deg,
                                   cam.heading_deg, cam.elevation_m), cam_generator))
    Y_pred = np.array(Y_pred)
    Y_true = np.array(Y_true)

    return (mean_squared_error(Y_true[:, 0], Y_pred[:, 0], squared=False),
            mean_squared_error(Y_true[:, 1], Y_pred[:, 1], squared=False),
            mean_squared_error(Y_true[:, 2], Y_pred[:, 2], squared=False),
            mean_squared_error(Y_true[:, 3], Y_pred[:, 3], squared=False))


def area(X_test: List[dict], y_true: List[float], verbose: bool = True) -> float:
    """Accuracy test for area.

    :param X_test: List of kwargs for transform.fit
    :type X_test: List[dict]
    :param y_true: Areas of Polygons
    :type y_true: List[float]
    :param verbose: TQDM, defaults to True
    :type verbose: bool
    :return: RMSE, Aka loss
    :rtype: float
    """
    Y_pred = map(lambda params: estimator.area(
        transform.fit_transform(**params)[:, :2]), tqdm(X_test))
    return mean_squared_error(y_true, list(Y_pred), squared=False)
