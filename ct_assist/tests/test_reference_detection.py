import pytest

import torch
import numpy as np
from ct_assist import reference_detection as rd
from ct_assist.exceptions import SkipFieldWarning

from detectron2.config.config import CfgNode
from detectron2.engine import DefaultPredictor
from detectron2.structures import Instances


def test_load_model():
    """Loads Detectron2 model"""
    assert isinstance(
        rd.load_model(), DefaultPredictor), "rd.load_model() != DefaultPredictor"
    dp, cfg = rd.load_model(return_cfg=True)
    assert isinstance(
        dp, DefaultPredictor), "rd.load_model(return_cfg=True), != (DefaultPredictor, ...)"
    assert isinstance(
        cfg, CfgNode), "rd.load_model(return_cfg=True), != (... CfgNode)"
    assert cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST == 0.7


def test_instances_to_dict():
    """Take Detectron2.engine.DefaultPredictor output, and turns it into an easily parsable dictionary."""
    instances = Instances(image_size=(1920, 1080))
    instances.set("pred_masks", ["test-masks"])

    class test_tensor:
        def item():
            return 1

    instances.set("pred_classes", [test_tensor])
    classes = {1: "test_class"}
    preds = {"instances": instances}
    assert {"test_class": ["test-masks"]
            } == rd.instances_to_dict(preds, classes)


def test_get_heads_feet():
    mask = torch.zeros(10, 10)
    mask[2:6, 1:4] = 1
    mask[4:8, 6:10] = 1
    reference = [[[1, 2],
                  [2, 2],
                  [3, 2],
                  [6, 4],
                  [7, 4],
                  [8, 4],
                  [9, 4]],

                 [[1, 5],
                  [2, 5],
                  [3, 5],
                  [6, 7],
                  [7, 7],
                  [8, 7],
                  [9, 7]]]
    reference = np.array(reference)
    assert (rd.get_heads_feet(mask, step_size=1, offset=0) == reference).all()
    assert (rd.get_heads_feet(mask, step_size=2, offset=0) == reference[:, 0::2]).all()


def test_extract_reference():
    """Extracts reference objects' feet and heads"""
    mask = torch.zeros(10, 10)
    mask[2:6, 1:4] = 1
    mask[4:8, 6:10] = 1
    obj = {"truck": [mask]}
    step_size = 1
    offset = 0  # Include all
    height_dict = {"truck": (3, 1)}
    reference = [[[1, 2],
                  [2, 2],
                  [3, 2],
                  [6, 4],
                  [7, 4],
                  [8, 4],
                  [9, 4]],

                 [[1, 5],
                  [2, 5],
                  [3, 5],
                  [6, 7],
                  [7, 7],
                  [8, 7],
                  [9, 7]]]
    reference = np.array(reference)
    # Test for expected output
    test_ref, *test_height = rd.extract_reference(objects=obj, step_size=step_size, offset=offset, height_dict=height_dict)[0]
    assert (test_ref == reference).all(), "reference != test_ref"
    assert tuple(test_height) == height_dict.get("truck")

    # Test for warning
    obj["test"] = []
    with pytest.warns(SkipFieldWarning, match="Key `test` not found in `height_dict`. Skipped this field."):
        test_ref, *test_height = rd.extract_reference(objects=obj, step_size=step_size, offset=offset, height_dict=height_dict)[0]

    # Test for skip field
    assert (test_ref == reference).all(), "reference != test_ref"
    assert tuple(test_height) == height_dict.get("truck")
