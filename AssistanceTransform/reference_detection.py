import detectron2
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
import numpy as np


def load_model(model_url: str = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml", threshold: float = 0.7,
               return_cfg: bool = False) -> DefaultPredictor:
    """Loads detectron2 model at model dir with threshold, option to return

    :param model_url: Points to yaml file containing config, defaults to "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
    :type model_url: str, optional
    :param threshold: Confidence threshold, defaults to 0.7
    :type threshold: float, optional
    :param return_cfg: If CfgNode obj should be returned, defaults to False
    :type return_cfg: bool, optional
    :return: Detectron2 default predictor
    :rtype: DefaultPredictor
    """
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(model_url))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model_url)
    predictor = DefaultPredictor(cfg)
    if return_cfg:
        return predictor, cfg
    else:
        return predictor


def extract_reference(objects: np.ndarray) -> np.ndarray:
    """Extracts reference objects' feet and heads

    :param objects: Segmentation masks
    :type objects: np.ndarray
    :return: Reference feet, heads, average height, STD height
    :rtype: np.ndarray
    """
    raise NotImplementedError(
        "Function `load_model` in `reference_detection.py` has not yet been implemented")
