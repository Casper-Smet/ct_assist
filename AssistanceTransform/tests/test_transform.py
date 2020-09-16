from typing import Tuple

import cameratransform as ct
import numpy as np
from PIL import Image
from PIL.ExifTags import TAGS

from AssistanceTransform import transform


def test_transform_image():
    """Function composition for transforming image-coordinates to real-world coordinates
    using the other functions declared in transform.py."""
    assert False


def test_getExif():
    """Extracts or estimates image meta data for Camera intrinsic properties."""
    assert False


def test_sensor_size_resolution():
    """Estimates sensor size based on FocalPlaneXResolution and FocalPlaneYResolution and image size.
    Based on CameraTransform's sensor size estimation."""
    assert False


def test_sensor_size_crop_factor():
    """Estimates sensor size based on effective and actual focal length, comparing to a standard 35 mm film camera."""
    assert False
