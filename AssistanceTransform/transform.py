from typing import Tuple

import cameratransform as ct
import numpy as np
from PIL import Image

from AssistanceTransform.exceptions import MissingExifError, DimensionError


def transform_image(img: Image.Image, reference: Tuple[np.ndarray, np.ndarray], height: np.ndarray, STD: int, image_coords: np.ndarray,
                    meta_data: dict = None, z: float = 0.0, *args, **kwargs) -> np.ndarray:
    """Function composition for transforming image-coordinates to real-world coordinates
    using the other functions declared in transform.py.

    :param img: Photograph in PIL image format
    :type img: Image.Image
    :param reference: Tuple with reference object (heads, feet)
    :type reference: Tuple[np.ndarray, np.ndarray]
    :param height: Height(s) of reference
    :type height: np.ndarray or float
    :param image_coords: Coordinates you wish to transform to real-world
    :type image_coords: np.ndarray
    :param z: Points, defaults to 0
    :type z: int, optional
    :param meta_data: image meta data for intrinsic camera properties, defaults to None
    :return: image_coords transformed to real-world coordinates
    :rtype: np.ndarray
    """
    pass


def get_Exif(img: Image.Image) -> Tuple[float, Tuple[int, int], Tuple[float, float]]:
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
    # TODO: Add support for lens look up table
    exif_data = img.getexif()
    f = exif_data.get(37386)
    # If focal length is unknown, CameraTransform cannot be used
    if f is None:
        raise MissingExifError("Actual Focal Length not found in exif")

    img_size = img.size
    # Image dimensions must 1x1 or greater
    if img_size[0] < 1 or img_size[1] < 1:
        raise DimensionError(f"Dimensions must be greater than 0, not {img_size}")

    resolution = exif_data.get(41486), exif_data.get(41487)
    if resolution[0] is not None and resolution[1] is not None:
        sensor_size = sensor_size_resolution(resolution, img_size)
    else:
        effective_f = exif_data.get(41989)
        # If neither FocalPlaneResolution or effective focal length is known, CameraTransform may not be used
        if effective_f is None:
            raise MissingExifError(
                "FocalPlane(X/Y)Resolution and effective focal length not found in exif")
        sensor_size = sensor_size_crop_factor(effective_f, f)
    return f, img_size, sensor_size


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
    # `resolution` and `image_size` must be of type tuple
    if not isinstance(resolution, tuple):
        raise TypeError("Expected `resolution` as tuple(float, float)")
    if not isinstance(image_size, tuple):
        raise TypeError("Expected `image_size` as tuple(float, float)")

    if resolution[0] == 0:
        raise ZeroDivisionError("FocalPlaneXResolution must be greater than 0")
    if resolution[1] == 0:
        raise ZeroDivisionError("FocalPlaneYResolution must be greater than 0")

    sensor_size = (image_size[0] / resolution[0] * 25.4,
                   image_size[1] / resolution[1] * 25.4)
    return sensor_size


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
