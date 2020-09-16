from typing import Tuple

import cameratransform as ct
import numpy as np
import pytest
from PIL import Image
from PIL.ExifTags import TAGS

from AssistanceTransform import transform
from AssistanceTransform.exceptions import DimensionError, MissingExifError


def test_transform_image():
    """Function composition for transforming image-coordinates to real-world coordinates
    using the other functions declared in transform.py.
    
    Transformation of images is non-deterministic due to Metropolis Monte Carlo sampling"""
    assert False


def test_get_Exif(monkeypatch):
    """Extracts or estimates image meta data for Camera intrinsic properties."""
    # Fake the Image object and Exif data, FocalPlaneimage_size
    with monkeypatch.context() as m:
        img = Image.new("RGB", (30, 30), color="red")
        m.setattr(img, "getexif", {37386: 0.6, 41486: 15, 41487: 7.5})
        assert transform.get_Exif(img) == (0.6, (30, 30), (50.8, 101.6))

    # Fake the Image object and Exif data, crop factor
    with monkeypatch.context() as m:
        img = Image.new("RGB", (30, 30), color='red')
        m.setattr(img, "getexif", {37386: 4, 41989: 8})
        assert transform.getExif(img) == (0.6, (30, 30), (18, 12))

    # Missing actual focal length
    with monkeypatch.context() as m:
        img = Image.new("RGB", (30, 30), color='red')
        m.setattr(img, "getexif", {41486: 15, 41487: 7.5})
        with pytest.raises(MissingExifError) as excinfo:
            transform.get_Exif(img)
        assert "Actual Focal Length" in str(excinfo)

    # Missing both FocalPlaneResolution and effective focal length
    with monkeypatch.context() as m:
        img = Image.new("RGB", (30, 30), color='red')
        m.setattr(img, "getexif", {37386: 4})
        with pytest.raises(MissingExifError) as excinfo:
            transform.get_Exif(img)
        assert "FocalPlane(X/Y)Resolution and effective focal length" in str(excinfo)

    # Check image size - (2D)
    img = Image.new("RGB", (30, 30), color="red")
    with pytest.raises(DimensionError) as excinfo:
        transform.get_Exif(img)
    assert "Expected two image dimensions" in str(excinfo)

    # Check image size - each image dimension must be size 1 or greater
    img = Image.new("RGB", (0, 0), color="red")
    with pytest.raises(DimensionError) as excinfo:
        transform.get_Exif(img)
    assert "Dimensions must be greater than 0" in str(excinfo)

    # Incorrect type for img - str
    with pytest.raises(TypeError) as excinfo:
        transform.get_Exif("str")
    assert "Expected PIL.Image.Image" in str(excinfo)

    # Incorrect type for img - int
    with pytest.raises(TypeError) as excinfo:
        transform.get_Exif(0)
    assert "Expected PIL.Image.Image" in str(excinfo)

    # Incorrect type for img - dict
    with pytest.raises(TypeError) as excinfo:
        transform.get_Exif({"img": True})
    assert "Expected PIL.Image.Image" in str(excinfo)

    # Incorrect type for actual focal length - str
    with monkeypatch.context() as m:
        img = Image.new("RGB", (30, 30), color='red')
        m.setattr(img, "getexif", {37386: "str", 41486: 15, 41487: 7.5})
        with pytest.raises(TypeError) as excinfo:
            transform.get_Exif(img)
        assert "Actual focal length" in str(excinfo)

    # Incorrect type for actual focal length - list
    with monkeypatch.context() as m:
        img = Image.new("RGB", (30, 30), color='red')
        m.setattr(img, "getexif", {37386: [1, 2], 41486: 15, 41487: 7.5})
        with pytest.raises(TypeError) as excinfo:
            transform.get_Exif(img)
        assert "Actual focal length" in str(excinfo)

    # Incorrect type for FocalPlaneXResolution - str
    with monkeypatch.context() as m:
        img = Image.new("RGB", (30, 30), color='red')
        m.setattr(img, "getexif", {37386: 28.1, 41486: "str", 41487: 7.5})
        with pytest.raises(TypeError) as excinfo:
            transform.get_Exif(img)
        assert "FocalPlaneXResolution" in str(excinfo)

    # Incorrect type for FocalPlaneXResolution - list
    with monkeypatch.context() as m:
        img = Image.new("RGB", (30, 30), color='red')
        m.setattr(img, "getexif", {37386: 28.1, 41486: [1, 2], 41487: 7.5})
        with pytest.raises(TypeError) as excinfo:
            transform.get_Exif(img)
        assert "FocalPlaneXResolution" in str(excinfo)

    # Incorrect type for FocalPlaneYResolution - str
    with monkeypatch.context() as m:
        img = Image.new("RGB", (30, 30), color='red')
        m.setattr(img, "getexif", {37386: 7.5, 41486: 15, 41487: "str"})
        with pytest.raises(TypeError) as excinfo:
            transform.get_Exif(img)
        assert "FocalPlaneYResolution" in str(excinfo)

    # Incorrect type for FocalPlaneYResolution - list
    with monkeypatch.context() as m:
        img = Image.new("RGB", (30, 30), color='red')
        m.setattr(img, "getexif", {37386: 7.5, 41486: 15, 41487: [1, 2]})
        with pytest.raises(TypeError) as excinfo:
            transform.get_Exif(img)
        assert "FocalPlaneYResolution" in str(excinfo)

    # Incorrect type for effective focal length - str
    with monkeypatch.context() as m:
        img = Image.new("RGB", (30, 30), color='red')
        m.setattr(img, "getexif", {37386: 4, 41989: "str"})
        with pytest.raises(TypeError) as excinfo:
            transform.get_Exif(img)
        assert "Effective focal length" in str(excinfo)

    # Incorrect type for effective focal length - list
    with monkeypatch.context() as m:
        img = Image.new("RGB", (30, 30), color='red')
        m.setattr(img, "getexif", {37386: 4, 41989: [1, 2]})
        with pytest.raises(TypeError) as excinfo:
            transform.get_Exif(img)
        assert "Effective focal length" in str(excinfo)


def test_sensor_size_resolution():
    """Estimates sensor size based on FocalPlaneXResolution and FocalPlaneYResolution and image size.
    Based on CameraTransform's sensor size estimation."""
    assert transform.sensor_size_resolution(
        (15, 7.5), (30, 30)) == (50.8, 101.6)

    # FocalPlaneXResolution equals 0
    with pytest.raises(ZeroDivisionError) as excinfo:
        transform.sensor_size_resolution((0, 1), (1, 1))
    assert "FocalPlaneXResolution must be greater than 0" in str(excinfo)

    # FocalPlaneYResolution equals 0
    with pytest.raises(ZeroDivisionError) as excinfo:
        transform.sensor_size_resolution((1, 0), (1, 1))
    assert "FocalPlaneYResolution must be greater than 0"

    # Incorrect type for resolution - str
    with pytest.raises(TypeError) as excinfo:
        transform.sensor_size_resolution("str", (0, 0))
    assert "Expected `resolution` as tuple(float, float)" in str(excinfo)

    # Incorrect type for resolution - list
    with pytest.raises(TypeError) as excinfo:
        transform.sensor_size_resolution([1, 2], (0, 0))
    assert "Expected `resolution` as tuple(float, float)" in str(excinfo)

    # Incorrect type for resolution - int
    with pytest.raises(TypeError) as excinfo:
        transform.sensor_size_resolution(0, (0, 0))
    assert "Expected `resolution` as tuple(float, float)" in str(excinfo)

    # Incorrect type for image_size - str
    with pytest.raises(TypeError) as excinfo:
        transform.sensor_size_resolution((0, 0), "str")
    assert "Expected `image_size` as tuple(float, float)" in str(excinfo)

    # Incorrect type for image_size - list
    with pytest.raises(TypeError) as excinfo:
        transform.sensor_size_resolution((0, 0), [1, 2])
    assert "Expected `image_size` as tuple(float, float)" in str(excinfo)

    # Incorrect type for image_size - int
    with pytest.raises(TypeError) as excinfo:
        transform.sensor_size_resolution((0, 0), 0)
    assert "Expected `image_size` as tuple(float, float)" in str(excinfo)


def test_sensor_size_crop_factor():
    """Estimates sensor size based on effective and actual focal length, comparing to a standard 35 mm film camera."""
    assert transform.sensor_size_crop_factor(8, 4) == (18, 12)

    # Effective focal length may not equal 0
    with pytest.raises(ZeroDivisionError) as excinfo:
        transform.sensor_size_crop_factor(0, 1)
    assert "Effective" in str(excinfo)

    # Actual focal length may not equal 0
    with pytest.raises(ZeroDivisionError) as excinfo:
        transform.sensor_size_crop_factor(1, 0)
    assert "Actual" in str(excinfo)
