"""
This module contains the functions necessary to extract head-feet pairs from images, load Detectron2 models, and parse the detectron2 outpu\
t.

Use `get_heads_feet` to extract a head-feet pair from a single instance/binary mask.

Use `extract_reference` to extract a series of head-feet pairs from a single image.

Usage example
-----------------
.. code-block:: python

    # If you plan on using detectron2
    predictor, cfg = load_model()  # Basic COCO-trained mask-rcnn, threshold = 0.7
    preds = predictor(image)  # Your image here
    from detectron2.data import MetadataCatalog
    objects = instances_to_dict(preds, MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).get("thing_classes"))
    reference = extract_reference(objects)  # These are the head-feet pairs!

Functions
---------------------
"""

import warnings
from collections import defaultdict
from typing import List, Tuple

import numpy as np
import torch
# TODO: Move detectron2 to util files
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.structures import Instances

from ct_assist.exceptions import SkipFieldWarning


HEIGHT_DICT: dict = {"truck": (3, 1),
                     "person": (1.741, 0.05),
                     "car": (1.425, 0.0247)}

# TODO: Move to util file


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
    try:
        predictor = DefaultPredictor(cfg)
    except AssertionError as a:
        if "NVIDIA driver" in str(a):
            cfg.MODEL.DEVICE = "cpu"
            predictor = DefaultPredictor(cfg)
        else:
            raise a
    if return_cfg:
        return predictor, cfg
    else:
        return predictor

# TODO: Move to util file


def instances_to_dict(preds: dict, thing_classes: list) -> defaultdict:
    """Take Detectron2.engine.DefaultPredictor output, and turns it into an easily parsable dictionary.

    Have a cfg ready? Use: detectron2.data.MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).get("thing_classes"))
    for your thing_classes.

    :param preds: dict to Instances object containing segmentation masks, output from defaultpredictor
    :type preds: dict
    :param thing_classes: List mapping integers to string
    :type thing_classes: list
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


def get_heads_feet(mask: torch.tensor, step_size=5, offset=0.1) -> np.ndarray:
    """Gets head and feet from torch tensor segmentation mask.

    :param mask: Segmentation mask
    :type mask: torch.tensor
    :param step_size: Amount of found head feet pairs to skip to avoid overfitting, defaults to 5
    :type step_size: int, optional
    :param offset: Permitted distance from median head-feet distance in percentages, defaults to 0.1
    :type offset: float, optional
    :return: Returns numpy array with [heads, feet]
    :rtype: np.ndarray
    """
    head, feet = [], []
    # Get all points where point == 1 (mask)
    mask_points = torch.nonzero(mask, as_tuple=False)  # .nonzero()
    # For each unique value for the x-plane
    for x in torch.unique(mask_points[..., 1]):
        # Get the indices at which mask[:, x] == x
        index = torch.nonzero(mask_points.T[1] == x, as_tuple=False)  # .nonzero()
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
    # Calculate all distances between heads and feet
    dist = (reference[1] - reference[0])[:, 1]
    median_dist = np.median(dist)
    min_dist = (1 - offset) * median_dist
    max_dist = (1 + offset) * median_dist
    # Threshold is median_dist +- offset
    min_mask = min_dist <= dist
    max_mask = dist <= max_dist
    # Remove those that are outside the threshold
    reference = reference[:, min_mask == max_mask]

    # Apply step size
    return reference[:, 0::step_size]


def extract_reference(objects: dict, step_size: int = 10, offset: float = 0.1,
                      height_dict: dict = HEIGHT_DICT, always_warn: bool = True) -> List[Tuple[np.ndarray, float, float]]:
    """Extracts references from dictionary filled with predictions.

    See instances_to_dict for objects' format. The output is based on the output for get_heads_feet.

    :param objects: key : [mask]
    :type objects: dict
    :param step_size: How many pixels to skip, defaults to 10
    :type step_size: int, optional
    :param offset: Minimum size relative to median distance between heads and feet, defaults to 0.1
    :type offset: float, optional
    :param height_dict: dictionary mapping object-type to height, defaults to HEIGHT_DICT
    :type height_dict: dictionary
    :param always_warn: If SkipFieldWarning should always be thrown
    :return: [(reference, height, STD)]
    :rtype: List[Tuple[np.ndarray, float, float]]
    """
    if always_warn:
        warnings.simplefilter("always", SkipFieldWarning)
    args = []
    for key, masks in objects.items():
        try:
            height, STD = height_dict.get(key)
        except TypeError:
            warnings.warn(
                f"Key `{key}` not found in `height_dict`. Skipped this field.", SkipFieldWarning)
            continue
        refs = [get_heads_feet(mask, step_size=step_size,
                               offset=offset) for mask in masks]
        refs = np.concatenate(refs, axis=1)
        args.append((refs, height, STD))
    return args
