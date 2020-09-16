from typing import Tuple

import cameratransform as ct
import numpy as np
from PIL import Image
from PIL.ExifTags import TAGS


def transform_image(img: Image.Image, reference: Tuple[np.ndarray, np.ndarray], height: np.ndarray, STD: int, image_coords: np.ndarray,
                    meta_data: dict = None, z: int = 0, *args, **kwargs) -> np.ndarray:
    """Function composition for transforming image-coordinates to real-world coordinates
    using the other functions declared in transform.py.

    :param img: Photograph in PIL image format
    :type img: Image.Image
    :param reference: Tuple with reference object (heads, feet)
    :type reference: Tuple[np.ndarray, np.ndarray]
    :param height: Height(s) of reference
    :type height: np.ndarray
    :param image_coords: Coordinates you wish to transform to real-world
    :type image_coords: np.ndarray
    :param z: Points, defaults to 0
    :type z: int, optional
    :param meta_data: image meta data for intrinsic camera properties, defaults to None
    :return: image_coords transformed to real-world coordinates
    :rtype: np.ndarray
    """
    pass


def getExif(img: Image.Image) -> Tuple[float, Tuple[int, int], Tuple[float, float]]:
    """Extracts or estimates image meta data for Camera intrinsic properties.

    Extracts:
     * Focal length
     * Image size
     * Sensor size

    :param img: Photograph in PIL image format
    :type img: Image.Image
    :return: (focal length, image size, sensor size)
    :rtype: Tuple[float, Tuple[int, int], Tuple[float, float]]
    """
    pass


def sensor_size_resolution(resolution: Tuple, image_size: Tuple[int, int]) -> Tuple[float, float]:
    """Estimates sensor size based on FocalPlaneXResolution and FocalPlaneYResolution and image size.
    Based on CameraTransform's sensor size estimation.

    :param resolution: (FocalPlaneXResolution, FocalPlaneYResolution)
    :type resolution: Tuple[FocalPlaneXResolution, FocalPlaneYResolution]
    :param image_size: Image size in pixels, w * h
    :type image_size: Tuple[int, int]
    :return: Sensor size
    :rtype: Tuple[float, float]
    """
    pass


def sensor_size_crop_factor(effective_f: float, actual_f: float) -> Tuple[float, float]:
    """Estimates sensor size based on effective and actual focal length, comparing to a standard 35 mm film camera.

    :param effective_f: Effective focal length
    :type effective_f: float
    :param actual_f: Actual focal length
    :type actual_f: float
    :return: Sensor size
    :rtype: Tuple[float, float]
    """
    pass
