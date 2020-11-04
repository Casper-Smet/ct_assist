"""
Transform

This module contains helper functions of fitting a cameratransform.Camera object and using it to transform image point to real-world points,
and functions for extracting exif data, and estimating sensor size.

This module is a work in progress.
"""

from numbers import Rational
from typing import Tuple

import cameratransform as ct
import numpy as np
from PIL import Image

from ct_assist.exceptions import DimensionError, MissingExifError


def fit_transform(img: Image.Image, reference: np.ndarray, height: np.ndarray, STD: int, image_coords: np.ndarray,
                  meta_data: dict = None, z: float = 0.0, iters=1e4, verbose=False, seed: int = None, *args, **kwargs) -> np.ndarray:
    """Function composition for transforming image-coordinates to real-world coordinates
    using the other functions declared in transform.py.


    :param img: Photograph in PIL image format
    :type img: Image.Image
    :param reference: Tuple with reference object (heads, feet), dim=(2, n, 2)
    :type reference: np.ndarray
    :param height: Height(s) of reference
    :type height: np.ndarray or float
    :param STD: STD(s) of reference
    :type STD: np.ndarray or float
    :param meta_data: image meta data for intrinsic camera properties, defaults to None
    :type meta_data: dict
    :param iters: Amount of iterations in Monte Carlo simulation
    :type iters: int
    :param verbose: If progress bar and trace should be printed, defaults to False
    :type verbose: bool
    :param seed: Random seed to be passed to numpy.random.seed (not recommended)
    :type seed: int
    :return: image_coords transformed to real-world coordinates
    :rtype: np.ndarray
    """
    # TODO: Test for types `seed`, `verbose`, `iters`
    if not isinstance(image_coords, np.ndarray):
        raise TypeError(f"Expected `image_coords` to be of type np.ndarray, not {type(image_coords)}")
    if not isinstance(z, (float, np.ndarray)):
        raise TypeError(f"Expected `z` to be of type float|np.ndarray, not {type(z)}")
    
    cam = fit(img=img, reference=reference, height=height, STD=STD,
              meta_data=meta_data, iters=iters, verbose=verbose, seed=seed)

    real_pos = cam.spaceFromImage(points=image_coords, Z=z)

    return real_pos


def fit(img: Image.Image, reference: np.ndarray, height: np.ndarray, STD: int, meta_data: dict = None, iters=1e4, verbose=False,
        seed: int = None, *args, **kwargs) -> ct.Camera:
    """Creates a trained CameraTransform.Camera object. See "https://cameratransform.readthedocs.io/en/latest/camera.html".

    :param img: Photograph in PIL image format
    :type img: Image.Image
    :param reference: Tuple with reference object (heads, feet), dim=(2, n, 2)
    :type reference: np.ndarray
    :param height: Height(s) of reference
    :type height: np.ndarray or float
    :param STD: STD(s) of reference
    :type STD: np.ndarray or float
    :param meta_data: image meta data for intrinsic camera properties, defaults to None
    :type meta_data: dict
    :param iters: Amount of iterations in Monte Carlo simulation
    :type iters: int
    :param verbose: If progress bar and trace should be printed, defaults to False
    :type verbose: bool
    :param seed: Random seed to be passed to numpy.random.seed (not recommended)
    :type seed: int
    :return: image_coords transformed to real-world coordinates
    :rtype: np.ndarray
    """
    # TODO: Test for types `seed`, `verbose`, `iters`
    # Check if img is PIL.Image.Image
    if not isinstance(img, Image.Image):
        raise TypeError(
            f"Expected `img` to be PIL.Image.Image, not {type(img)}")

    # Check if reference is a np.ndarray
    if not isinstance(reference, np.ndarray):
        raise TypeError(
            f"Expected `reference` to be np.ndarray, not {type(reference)}")
    # Check dimensionality of reference
    if reference.shape[0] != 2 or reference.shape[2] != 2:
        raise DimensionError(
            f"Expected `reference` with dimension (2, n, 2), not {reference.shape}")

    # Check if height is int or float or np.ndarray
    if not isinstance(height, (int, float, np.ndarray)):
        raise TypeError(
            f"Expected `height` to be np.ndarray or float, not {type(height)}")

    # Check if STD is int or float or np.ndarray
    if not isinstance(STD, (int, float, np.ndarray)):
        raise TypeError(
            f"Expected `STD` to be np.ndarray or float, not {type(STD)}")

    # If meta_data was passed by user, use that instead
    if meta_data is not None:
        if not isinstance(meta_data, dict):
            raise TypeError(
                f"Expected `meta_data` to be of type dict, got {type(meta_data)} instead")
        f, image_size, sensor_size = meta_data.get("focal_length"), meta_data.get(
            "image_size"), meta_data.get("sensor_size")
        if (not isinstance(f, (int, float))) or (not isinstance(image_size, tuple)) or (not isinstance(sensor_size, tuple)):
            raise ValueError(f"Metadata incorrect, check typing: {meta_data}")
    else:
        # Get Focal length, image size, sensor size from image meta data (exif)
        f, image_size, sensor_size = get_Exif(img)

    # If random seed is given, set numpy.random.seed
    if seed is not None:
        # As CameraTransform uses the np legacy version of RNG, this is the only way one can set the seed
        # This is, however, not the prefered way to do this.
        # See: https://numpy.org/doc/stable/reference/random/generated/numpy.random.seed.html
        np.random.seed(seed)

    # Initialise projection
    proj = ct.RectilinearProjection(focallength_mm=f,
                                    sensor=sensor_size,
                                    image=image_size)
    # Initialise Camera
    cam = ct.Camera(projection=proj)

    # Add objects to Camera
    cam.addObjectHeightInformation(
        points_head=reference[0], points_feet=reference[1], height=height, variation=STD)

    # Fit for all spatial parameters
    trace = cam.metropolis([
        ct.FitParameter("elevation_m", lower=0,
                        upper=200, value=2),
        ct.FitParameter("tilt_deg", lower=0, upper=180, value=80),
        ct.FitParameter("heading_deg", lower=-180, upper=180, value=-77),
        ct.FitParameter("roll_deg", lower=-180, upper=180, value=0)
    ], iterations=iters, print_trace=verbose, disable_bar=not verbose)

    return cam


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

    # Check if img is PILL.Image
    if not isinstance(img, Image.Image):
        raise TypeError(f"Expected PIL.Image.Image, not {type(img)}")

    img_size = img.size
    # Image dimensions must 1x1 or greater
    if img_size[0] < 1 or img_size[1] < 1:
        raise DimensionError(
            f"Dimensions must be greater than 0, not {img_size}")

    exif_data = img.getexif()
    f = exif_data.get(37386)
    # If focal length is unknown, CameraTransform cannot be used
    if f is None:
        raise MissingExifError("Actual Focal Length not found in exif")
    # Actual focal length must be float-like
    if not issubclass(type(f), (Rational, float, int)):
        raise TypeError(f"Actual focal length must be float, not {type(f)}")
    # f is most likely Rational, convert to float
    f = float(f)

    # Get model name
    name = exif_data.get(272)
    # Try getting sensor size from model name
    sensor_size = sensor_size_look_up(name)

    # Get FocalPlaneXResolution and FocalPlaneYResolution
    resolution = exif_data.get(41486), exif_data.get(41487)
    if isinstance(sensor_size, tuple):
        pass
    elif resolution[0] is not None and resolution[1] is not None:
        # FocalPlaneResolutions must be float-like
        if not issubclass(type(resolution[0]), (Rational, float, int)):
            raise TypeError(
                f"FocalPlaneXResolution must be float, not {type(resolution[0])}")
        if not issubclass(type(resolution[1]), (Rational, float, int)):
            raise TypeError(
                f"FocalPlaneYResolution must be float, not {type(resolution[1])}")
        # FocalPlaneResolution most likely Rational, convert to float
        resolution = float(resolution[0]), float(resolution[1])
        sensor_size = sensor_size_resolution(resolution, img_size)
    else:
        effective_f = exif_data.get(41989)
        # If neither FocalPlaneResolution or effective focal length is known, CameraTransform may not be used
        if effective_f is None:
            raise MissingExifError(
                "FocalPlane(X/Y)Resolution and effective focal length not found in exif")
        # Effective focal length must be float-like
        if not issubclass(type(effective_f), (Rational, float, int)):
            raise TypeError(
                f"Effective focal length must be float, not {type(effective_f)}")
        # f is most likely Rational, convert to float
        effective_f = float(effective_f)
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

    # `FocalPlaneXResolution` and `FocalPlaneYresolution` must be greater than 0
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
    if effective_f == 0:
        raise ZeroDivisionError("Effective focal length may not be 0")
    if actual_f == 0:
        raise ZeroDivisionError("Actual focal length may not be 0")
    crop_factor = effective_f / actual_f
    sensor_size = (36 / crop_factor, 24 / crop_factor)
    return sensor_size


def sensor_size_look_up(model_name: str):
    """Looks up the sensor size of the photographic device with name `model_name`.

    :param model_name: Model name
    :type model_name: str
    """
    table = {
        "iPhone SE": (4.8, 3.6),
        "iPhone 11": (5.76, 4.29),  # Approx
        "iPhone 8 Plus": (4.8, 3.5),
        "SamsungSM-A202F": (6.40, 4.80),  # Approx
        "SM-A515F": (6.16, 4.55)  # Approx
    }

    return table.get(model_name)
