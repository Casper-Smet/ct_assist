from typing import List, Tuple

import numpy as np
from sklearn.metrics import mean_squared_error

from AssistanceTransform import transform, estimator


def camera_properties(X_test: List[dict], Y_true: List[Tuple[float, float, float, float]]) -> float:
    """Accuracy test for camera properties (roll_deg, tilt_deg, heading_deg, elevation_m)

    :param X_test: List of kwargs for `transform.fit`
    :type X_test: List[dict]
    :param Y_true: Spatial orientation roll_deg, tilt_deg, heading_deg, elevation_m
    :type Y_true: List[Tuple[float, float, float, float]]
    :return: RMSE for roll_deg, tilt_deg, heading_deg, elevation_m
    :rtype: float
    """
    cam_generator = map(lambda params: transform.fit(**params).orientation.parameters, X_test)
    Y_pred = map(lambda cam: (cam.roll_deg, cam.tilt_deg, cam.heading_deg, cam.elevation_m), cam_generator)
    return mean_squared_error(Y_true, list(Y_pred), squared=False)


def area(X_test: List[dict], y_true: List[float]) -> float:
    """Accuracy test for area.

    :param X_test: List of kwargs for transform.fit
    :type X_test: List[dict]
    :param y_true: Areas of Polygons
    :type y_true: List[float]
    :return: RMSE, Aka loss
    :rtype: float
    """
    Y_pred = map(lambda params: estimator.area(transform.fit_transform(**params)[:, :2]), X_test)
    return mean_squared_error(y_true, list(Y_pred), squared=False)
