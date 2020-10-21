from AssistanceTransform import reference_detection as rf

from detectron2.engine import DefaultPredictor
from detectron2.config.config import CfgNode


def test_load_model():
    """Loads Detectron2 model"""
    assert isinstance(rf.load_model(), DefaultPredictor), "rf.load_model() != DefaultPredictor"
    dp, cfg = rf.load_model(return_cfg=True)
    assert isinstance(dp, DefaultPredictor), "rf.load_model(return_cfg=True), != (DefaultPredictor, ...)"
    assert isinstance(cfg, CfgNode), "rf.load_model(return_cfg=True), != (... CfgNode)"
    assert cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST == 0.7


def test_extract_reference():
    """Extracts reference objects' feet and heads"""
    raise NotImplementedError("Function `load_model` in `reference_detection.py` has not yet been implemented")
