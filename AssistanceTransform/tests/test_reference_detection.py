from AssistanceTransform import reference_detection as rf

from detectron2.config.config import CfgNode
from detectron2.engine import DefaultPredictor
from detectron2.structures import Instances


def test_load_model():
    """Loads Detectron2 model"""
    assert isinstance(rf.load_model(), DefaultPredictor), "rf.load_model() != DefaultPredictor"
    dp, cfg = rf.load_model(return_cfg=True)
    assert isinstance(dp, DefaultPredictor), "rf.load_model(return_cfg=True), != (DefaultPredictor, ...)"
    assert isinstance(cfg, CfgNode), "rf.load_model(return_cfg=True), != (... CfgNode)"
    assert cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST == 0.7


def test_extract_reference():
    """Extracts reference objects' feet and heads"""
    raise NotImplementedError("Function `extract_reference` in `reference_detection.py` has not yet been implemented")


def test_instances_to_dict(monkeypatch):
    """Take Detectron2.engine.DefaultPredictor output, and turns it into an easily parsable dictionary."""
    instances = Instances(image_size=(1920, 1080))
    instances.set("pred_masks", ["test-masks"])

    class test_tensor:
        def item():
            return 1

    instances.set("pred_classes", [test_tensor])
    classes = {1: "test_class"}
    preds = {"instances": instances}
    assert {"test_class": ["test-masks"]} == rf.instances_to_dict(preds, classes)