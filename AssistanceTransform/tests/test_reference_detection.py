from AssistanceTransform.reference_detection import get_heads_feet
import torch
import numpy as np
from AssistanceTransform import reference_detection as rf

from detectron2.config.config import CfgNode
from detectron2.engine import DefaultPredictor
from detectron2.structures import Instances


def test_load_model():
    """Loads Detectron2 model"""
    assert isinstance(
        rf.load_model(), DefaultPredictor), "rf.load_model() != DefaultPredictor"
    dp, cfg = rf.load_model(return_cfg=True)
    assert isinstance(
        dp, DefaultPredictor), "rf.load_model(return_cfg=True), != (DefaultPredictor, ...)"
    assert isinstance(
        cfg, CfgNode), "rf.load_model(return_cfg=True), != (... CfgNode)"
    assert cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST == 0.7


def test_extract_reference():
    """Extracts reference objects' feet and heads"""
    raise NotImplementedError(
        "Function `extract_reference` in `reference_detection.py` has not yet been implemented")


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
            } == rf.instances_to_dict(preds, classes)


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
    assert (get_heads_feet(mask, step_size=1, min_size=0) == reference).all()
    assert (get_heads_feet(mask, step_size=2, min_size=0) == reference[:, 0::2]).all()
