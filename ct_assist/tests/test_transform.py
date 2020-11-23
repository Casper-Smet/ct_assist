import json
import os
import re

import numpy as np
import pytest
from ct_assist import transform
from ct_assist.exceptions import DimensionError, MissingExifError
from PIL import Image


def setup_vars():
    """Loads data for test_fit_transform"""
    data_dir = r"./notebooks/data/table"
    json_fp = os.path.join(data_dir, "anno.json")
    arr_fp = os.path.join(data_dir, "anno.npz")
    with open(json_fp, "r") as fp:
        mappings = json.load(fp)

    with np.load(arr_fp) as arrs:
        anno_dict = {img: {"heads": arrs[f"{prefix}heads"],
                           "feet": arrs[f"{prefix}feet"]}
                     for img, prefix in mappings.items()}

    annotations = anno_dict["D:\\University\\2020-2021\\Internship\\ct_assist\\notebooks\\data\\table\\img_03.jpg"]
    # feet and heads have been swapped in annotations
    reference = np.array([annotations["feet"], annotations["heads"]])
    height = 0.095  # m
    STD = 0.01  # m
    img = Image.open(
        r"./notebooks/data/table/img_03.jpg")

    image_coords = np.array(
        [[1216, 1398], [2215, 1754], [3268, 1530], [2067, 1282]])

    return (img, reference, height, STD, image_coords)


def test_fit_transform(monkeypatch):
    """Function composition for transforming image-coordinates to real-world coordinates
    using the other functions declared in transform.py.

    Transformation of images is non-deterministic due to Metropolis Monte Carlo sampling,
    accuracy will be tested seperately."""

    # TODO: Test for types `seed`, `verbose`, `iters`
    # Test using real data
    real_points = np.array([[-3.3518353, 0.4124983, 0.],
                            [-1.6383052, 0.72337879, 0.],
                            [-1.82506649, 1.6064894, 0.],
                            [-3.6821136, 1.45151892, 0.]])

    params = setup_vars()

    #  Test if transformed_points equal real_points
    transformed_points = transform.fit_transform(*params, iters=1e4, seed=0)
    # This test really doesn't say much for the accuracy, it is only useful for consistency testing
    assert np.allclose(real_points, transformed_points)  # Not needed

    # Passing meta data through dict
    meta_data = {"focal_length": 3.99, "image_size": (
        4032, 3024), "sensor_size": (4.8, 3.5)}
    # Test if transformed points equal real_points with meta_data
    transformed_points = transform.fit_transform(
        *params, meta_data=meta_data, iters=1e4, seed=0)
    # This test really doesn't say much for the accuracy, it is only useful for consistency testing
    assert np.allclose(real_points, transformed_points)

    multi_params = [[p] for p in params[1:]]
    transformed_points = transform.fit_transform(
        params[0], *multi_params, meta_data=meta_data, iters=1e4, seed=0, multi=True)

    type_mistakes = ["str", 0, [0, 1]]
    # Wrong type for meta_data
    for _meta_data in type_mistakes:
        with pytest.raises(TypeError, match=f"Expected `meta_data` to be of type dict, got {type(_meta_data)} instead"):
            transform.fit_transform(*params, meta_data=_meta_data)

    # Check for missing/wrong type focal_length
    _meta_data = meta_data.copy()
    del _meta_data["focal_length"]
    with pytest.raises(ValueError, match=re.escape(f"Metadata incorrect, check typing: {_meta_data}")):
        transform.fit_transform(*params, meta_data=_meta_data)

    # Check for missing/wrong type image_size
    _meta_data = meta_data.copy()
    del _meta_data["image_size"]
    with pytest.raises(ValueError, match=re.escape(f"Metadata incorrect, check typing: {_meta_data}")):
        transform.fit_transform(*params, meta_data=_meta_data)

    # Check for missing/wrong type sensor_size
    _meta_data = meta_data.copy()
    del _meta_data["sensor_size"]
    with pytest.raises(ValueError, match=re.escape(f"Metadata incorrect, check typing: {_meta_data}")):
        transform.fit_transform(*params, meta_data=_meta_data)

    img = Image.new("RGB", (30, 30), color="red")
    # Check `reference` dimensions
    with pytest.raises(DimensionError,
                       match=re.escape(f"Expected `reference` with dimension (2, n, 2), not {np.array([np.array([[1]]), ]).shape}")):
        transform.fit_transform(
            img, np.array([np.array([[1]]), ]), 1.0, 1, np.array([1]))

    # Fake image fake exif data
    with monkeypatch.context() as m:
        img = Image.new("RGB", (30, 30), color="red")
        m.setattr(img, "getexif", lambda: {37386: 0.6, 41486: 15, 41487: 7.5})

        # Wrong type for image
        for img_ in type_mistakes:
            with pytest.raises(TypeError, match="Expected `img` to be PIL.Image.Image"):
                transform.fit_transform(
                    img_, (np.array([1]), np.array([1])), 1.0, 1, np.array([1]))

        # Wrong type for reference
        for reference in type_mistakes:
            with pytest.raises(TypeError, match=f"Expected `reference` to be np.ndarray, not {type(reference)}"):
                transform.fit_transform(
                    img=img, reference=reference, height=1.0, STD=1, image_coords=np.array([1]))
            if not isinstance(reference, list):
                with pytest.raises(TypeError, match=f"Expected `reference` to be a list, not {type(reference)}"):
                    transform.fit_transform(
                        img=img, reference=reference, height=1.0, STD=1, image_coords=np.array([1]), multi=True)

        # Wrong type for height
        for height in type_mistakes:
            if isinstance(height, (list, int)):
                continue
            with pytest.raises(TypeError, match="Expected `height` to be np.ndarray or float"):
                transform.fit_transform(
                    img=img, reference=np.array([np.array([[1, 1]]), np.array([[1, 1]])]), height=height, STD=1, image_coords=np.array([1]))

        # Wrong type for STD
        for STD in type_mistakes:
            if isinstance(STD, (int, list)):
                continue
            with pytest.raises(TypeError, match="Expected `STD` to be np.ndarray or float"):
                transform.fit_transform(
                    img=img, reference=np.array([np.array([[1, 1]]), np.array([[1, 1]])]), height=1.0, STD=STD, image_coords=np.array([1]))

        # Wrong type for z
        for z in type_mistakes:
            with pytest.raises(TypeError, match=f"Expected `z` to be of type float|np.ndarray, not {type(z)}"):
                transform.fit_transform(
                    img=img, reference=np.array([np.array([[1, 1]]), np.array([[1, 1]])]), height=1.0, STD=1, image_coords=np.array([1]),
                    z=z)

        # Wrong type for image_coords
        for img_coords in type_mistakes:
            if isinstance(img_coords, list):
                continue
            with pytest.raises(TypeError, match=f"Expected `image_coords` to be of type np.ndarray, not {type(img_coords)}"):
                transform.fit_transform(
                    img=img, reference=np.array([np.array([[1, 1]]), np.array([[1, 1]])]), height=1.0, STD=1, image_coords=img_coords)


def test_fit(monkeypatch):
    # TODO: Test for types `z`, `image_coords`, `seed`, `verbose`, `iters`
    # Load preset parameters, without image_coords
    params = setup_vars()[:-1]

    orientation_parameters = transform.fit(
        *params, seed=1).orientation.parameters
    pred_params = (orientation_parameters.roll_deg,
                   orientation_parameters.tilt_deg, orientation_parameters.heading_deg)
    for param in pred_params:
        assert -180 <= param <= 180, "Params must be within bounds"
    # real_params = (1.0475075736599635,
    #                74.32266805581958,
    #                -77.54609982919105)
    # This test really doesn't say much for the accuracy, it is only useful for consistency testing
    # assert pred_params == real_params  # Doesn't work as intended

    # Passing meta data through dict
    meta_data = {"focal_length": 3.99, "image_size": (
        4032, 3024), "sensor_size": (4.8, 3.5)}

    type_mistakes = ["str", 0, [0, 1]]
    # Wrong type for meta_data
    for _meta_data in type_mistakes:
        with pytest.raises(TypeError, match=f"Expected `meta_data` to be of type dict, got {type(_meta_data)} instead"):
            transform.fit(*params, meta_data=_meta_data)

    # Check for missing/wrong type focal_length
    _meta_data = meta_data.copy()
    del _meta_data["focal_length"]
    with pytest.raises(ValueError, match=re.escape(f"Metadata incorrect, check typing: {_meta_data}")):
        transform.fit(*params, meta_data=_meta_data)

    # Check for missing/wrong type image_size
    _meta_data = meta_data.copy()
    del _meta_data["image_size"]
    with pytest.raises(ValueError, match=re.escape(f"Metadata incorrect, check typing: {_meta_data}")):
        transform.fit(*params, meta_data=_meta_data)

    # Check for missing/wrong type sensor_size
    _meta_data = meta_data.copy()
    del _meta_data["sensor_size"]
    with pytest.raises(ValueError, match=re.escape(f"Metadata incorrect, check typing: {_meta_data}")):
        transform.fit(*params, meta_data=_meta_data)

    img = Image.new("RGB", (30, 30), color="red")

    # Check `reference` dimensions
    with pytest.raises(DimensionError,
                       match=re.escape(f"Expected `reference` with dimension (2, n, 2), not {np.array([np.array([[1]]), ]).shape}")):
        transform.fit(
            img, np.array([np.array([[1]]), ]), 1.0, 1, np.array([1]))

    # Fake image fake exif data
    with monkeypatch.context() as m:
        img = Image.new("RGB", (30, 30), color="red")
        m.setattr(img, "getexif", lambda: {37386: 0.6, 41486: 15, 41487: 7.5})

        # Wrong type for image
        for img_ in type_mistakes:
            with pytest.raises(TypeError, match="Expected `img` to be PIL.Image.Image"):
                transform.fit(
                    img_, (np.array([1]), np.array([1])), 1.0, 1, np.array([1]))

        # Wrong type for reference
        for reference in type_mistakes:
            with pytest.raises(TypeError, match=f"Expected `reference` to be np.ndarray, not {type(reference)}"):
                transform.fit(
                    img, reference, 1.0, 1, np.array([1]))

        # Wrong type for height
        for height in type_mistakes:
            if isinstance(height, (list, int)):
                continue
            with pytest.raises(TypeError, match="Expected `height` to be np.ndarray or float"):
                transform.fit(
                    img=img, reference=np.array([np.array([[1, 1]]), np.array([[1, 1]])]), height=height, STD=1, image_coords=np.array([1]))

        # Wrong type for STD
        for STD in type_mistakes:
            if isinstance(STD, (list, int)):
                continue
            with pytest.raises(TypeError, match="Expected `STD` to be np.ndarray or float"):
                transform.fit(
                    img=img, reference=np.array([np.array([[1, 1]]), np.array([[1, 1]])]), height=1.0, STD=STD, image_coords=np.array([1]))


def test_get_Exif(monkeypatch):
    """Extracts or estimates image meta data for Camera intrinsic properties."""

    # Fake iPhone SE model name
    with monkeypatch.context() as m:
        img = Image.new("RGB", (30, 30), color="red")
        m.setattr(img, "getexif", lambda: {37386: 0.6, 272: "iPhone SE"})
        assert transform.get_Exif(img) == (0.6, (30, 30), (4.8, 3.6))

    # Fake the Image object and Exif data, FocalPlaneimage_size
    with monkeypatch.context() as m:
        img = Image.new("RGB", (30, 30), color="red")
        m.setattr(img, "getexif", lambda: {37386: 0.6, 41486: 15, 41487: 7.5})
        assert transform.get_Exif(img) == (0.6, (30, 30), (50.8, 101.6))

    # Fake the Image object and Exif data, crop factor
    with monkeypatch.context() as m:
        img = Image.new("RGB", (30, 30), color='red')
        m.setattr(img, "getexif", lambda: {37386: 4, 41989: 8})
        assert transform.get_Exif(img) == (4, (30, 30), (18, 12))

    # Missing actual focal length
    with monkeypatch.context() as m:
        img = Image.new("RGB", (30, 30), color='red')
        m.setattr(img, "getexif", lambda: {41486: 15, 41487: 7.5})
        with pytest.raises(MissingExifError, match="Actual Focal Length"):
            transform.get_Exif(img)

    # Missing both FocalPlaneResolution and effective focal length
    with monkeypatch.context() as m:
        img = Image.new("RGB", (30, 30), color='red')
        m.setattr(img, "getexif", lambda: {37386: 4})
        with pytest.raises(MissingExifError, match=re.escape("FocalPlane(X/Y)Resolution and effective focal length")):
            transform.get_Exif(img)

    # Check image size - each image dimension must be size 1 or greater
    img = Image.new("RGB", (0, 0), color="red")
    with pytest.raises(DimensionError, match="Dimensions must be greater than 0"):
        transform.get_Exif(img)

    # Incorrect type for img - str
    with pytest.raises(TypeError, match="Expected PIL.Image.Image"):
        transform.get_Exif("str")

    # Incorrect type for img - int
    with pytest.raises(TypeError, match="Expected PIL.Image.Image"):
        transform.get_Exif(0)

    # Incorrect type for img - dict
    with pytest.raises(TypeError, match="Expected PIL.Image.Image"):
        transform.get_Exif({"img": True})

    # Incorrect type for actual focal length - str
    with monkeypatch.context() as m:
        img = Image.new("RGB", (30, 30), color='red')
        m.setattr(img, "getexif", lambda: {
                  37386: "str", 41486: 15, 41487: 7.5})
        with pytest.raises(TypeError, match="Actual focal length"):
            transform.get_Exif(img)

    # Incorrect type for actual focal length - list
    with monkeypatch.context() as m:
        img = Image.new("RGB", (30, 30), color='red')
        m.setattr(img, "getexif", lambda: {
                  37386: [1, 2], 41486: 15, 41487: 7.5})
        with pytest.raises(TypeError, match="Actual focal length"):
            transform.get_Exif(img)

    # Incorrect type for FocalPlaneXResolution - str
    with monkeypatch.context() as m:
        img = Image.new("RGB", (30, 30), color='red')
        m.setattr(img, "getexif", lambda: {
                  37386: 28.1, 41486: "str", 41487: 7.5})
        with pytest.raises(TypeError, match="FocalPlaneXResolution"):
            transform.get_Exif(img)

    # Incorrect type for FocalPlaneXResolution - list
    with monkeypatch.context() as m:
        img = Image.new("RGB", (30, 30), color='red')
        m.setattr(img, "getexif", lambda: {
                  37386: 28.1, 41486: [1, 2], 41487: 7.5})
        with pytest.raises(TypeError, match="FocalPlaneXResolution"):
            transform.get_Exif(img)

    # Incorrect type for FocalPlaneYResolution - str
    with monkeypatch.context() as m:
        img = Image.new("RGB", (30, 30), color='red')
        m.setattr(img, "getexif", lambda: {
                  37386: 7.5, 41486: 15, 41487: "str"})
        with pytest.raises(TypeError, match="FocalPlaneYResolution"):
            transform.get_Exif(img)

    # Incorrect type for FocalPlaneYResolution - list
    with monkeypatch.context() as m:
        img = Image.new("RGB", (30, 30), color='red')
        m.setattr(img, "getexif", lambda: {
                  37386: 7.5, 41486: 15, 41487: [1, 2]})
        with pytest.raises(TypeError, match="FocalPlaneYResolution"):
            transform.get_Exif(img)

    # Incorrect type for effective focal length - str
    with monkeypatch.context() as m:
        img = Image.new("RGB", (30, 30), color='red')
        m.setattr(img, "getexif", lambda: {37386: 4, 41989: "str"})
        with pytest.raises(TypeError, match="Effective focal length"):
            transform.get_Exif(img)

    # Incorrect type for effective focal length - list
    with monkeypatch.context() as m:
        img = Image.new("RGB", (30, 30), color='red')
        m.setattr(img, "getexif", lambda: {37386: 4, 41989: [1, 2]})
        with pytest.raises(TypeError, match="Effective focal length"):
            transform.get_Exif(img)


def test_sensor_size_resolution():
    """Estimates sensor size based on FocalPlaneXResolution and FocalPlaneYResolution and image size.
    Based on CameraTransform's sensor size estimation."""
    assert transform.sensor_size_resolution(
        (15, 7.5), (30, 30)) == (50.8, 101.6)

    # FocalPlaneXResolution equals 0
    with pytest.raises(ZeroDivisionError, match="FocalPlaneXResolution must be greater than 0"):
        transform.sensor_size_resolution((0, 1), (1, 1))

    # FocalPlaneYResolution equals 0
    with pytest.raises(ZeroDivisionError, match="FocalPlaneYResolution must be greater than 0"):
        transform.sensor_size_resolution((1, 0), (1, 1))
    assert "FocalPlaneYResolution must be greater than 0"

    # Incorrect type for resolution - str
    with pytest.raises(TypeError, match=re.escape("Expected `resolution` as tuple(float, float)")):
        transform.sensor_size_resolution("str", (0, 0))

    # Incorrect type for resolution - list
    with pytest.raises(TypeError, match=re.escape("Expected `resolution` as tuple(float, float)")):
        transform.sensor_size_resolution([1, 2], (0, 0))

    # Incorrect type for resolution - int
    with pytest.raises(TypeError, match=re.escape("Expected `resolution` as tuple(float, float)")):
        transform.sensor_size_resolution(0, (0, 0))

    # Incorrect type for image_size - str
    with pytest.raises(TypeError, match=re.escape("Expected `image_size` as tuple(float, float)")):
        transform.sensor_size_resolution((0, 0), "str")

    # Incorrect type for image_size - list
    with pytest.raises(TypeError, match=re.escape("Expected `image_size` as tuple(float, float)")):
        transform.sensor_size_resolution((0, 0), [1, 2])

    # Incorrect type for image_size - int
    with pytest.raises(TypeError, match=re.escape("Expected `image_size` as tuple(float, float)")):
        transform.sensor_size_resolution((0, 0), 0)


def test_sensor_size_crop_factor():
    """Estimates sensor size based on effective and actual focal length, comparing to a standard 35 mm film camera."""
    assert transform.sensor_size_crop_factor(8, 4) == (18, 12)

    # Effective focal length may not equal 0
    with pytest.raises(ZeroDivisionError, match="Effective"):
        transform.sensor_size_crop_factor(0, 1)

    # Actual focal length may not equal 0
    with pytest.raises(ZeroDivisionError, match="Actual"):
        transform.sensor_size_crop_factor(1, 0)


def test_sensor_size_look_up():
    """Looks up the sensor size of the photographic device with name `model_name`."""
    assert transform.sensor_size_look_up("iPhone SE") == (4.8, 3.6)
    assert transform.sensor_size_look_up("iPhone 11") == (5.76, 4.29)
    assert transform.sensor_size_look_up("SamsungSM-A202F") == (6.40, 4.80)

    assert transform.sensor_size_look_up("test") is None
