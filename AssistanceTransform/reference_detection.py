from collections import defaultdict

import detectron2
import numpy as np
import torch
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.engine import DefaultPredictor
from detectron2.structures import Instances
from detectron2.utils.visualizer import Visualizer


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


def instances_to_dict(preds: dict, thing_classes: dict) -> defaultdict:
    """Take Detectron2.engine.DefaultPredictor output, and turns it into an easily parsable dictionary.

    Have a cfg ready? Use: detectron2.data.MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).get("thing_classes"))
    for your thing_classes.

    :param preds: dict to Instances object containing segmentation masks, output from defaultpredictor
    :type preds: dict
    :param thing_classes: Dictionary mapping integers to string
    :type thing_classes: dict
    :return: object-name : [masks]
    :rtype: defaultdict
    """
    # Initialise empty defaultdict with list constructor
    class_dict = defaultdict(list)
    ins: Instances = preds["instances"]
    masks = ins.get("pred_masks")
    classes = ins.get("pred_classes")
    for i in range(len(ins)):
        class_int = classes[i].item()
        class_str = thing_classes[class_int]
        mask = masks[i]
        class_dict[class_str].append(mask)
    return class_dict


def get_heads_feet(mask: torch.tensor, step_size=5, min_size=0.9):
    head, feet = [], []
    # Get all points where point == 1 (mask)
    mask_points = mask.nonzero()
    # For each unique value for the x-plane
    for x in torch.unique(mask_points[...,1]):
        # Get the indices at which mask[:, x] == x
        index = (mask_points.T[1] == x).nonzero()
        # Get all values for y where mask[:, x] == x
        ys = mask_points[index, 0]
        # Get max and min y, cast to CPU
        max_y, min_y = ys.max().item(), ys.min().item()
        # Remove max_y == min_y
        if max_y != min_y:
            # Cast x to CPU
            x = x.item()
            head.append([x, max_y])
            feet.append([x, min_y])
    # Turn head, feet into a numpy array and reverse
    reference = np.array([head, feet])[::-1]
    min_dist = min_size * np.max(reference[1] - reference[0])
    # Remove those that are outside the minimum threshold
    reference = reference[:, (reference[1] - reference[0])[:,1] >= min_dist]
    
    # Apply step size
    return reference[:, 0::step_size]

# def extract_reference(objects: np.ndarray) -> np.ndarray:
#     """Extracts reference objects' feet and heads

#     :param objects: Segmentation masks
#     :type objects: np.ndarray
#     :return: Reference feet, heads, average height, STD height
#     :rtype: np.ndarray
#     """
#     raise NotImplementedError(
#         "Function `load_model` in `reference_detection.py` has not yet been implemented")
