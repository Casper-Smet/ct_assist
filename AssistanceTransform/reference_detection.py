from detectron2.engine import DefaultPredictor
import numpy as np


def load_model(classes: list, model_dir: str = None, model_name: str = None) -> DefaultPredictor:
    """Loads Detectron2 model

    :param classes: Classes to detect for
    :type classes: list
    :param model_dir: Directory containig model, defaults to None
    :type model_dir: str, optional
    :param model_name: Model file name, defaults to None
    :type model_name: str, optional
    :return: Detectron2 predictor for objects
    :rtype: DefaultPredictor
    """
    raise NotImplementedError("Function `load_model` in `reference_detection.py` has not yet been implemented")


def extract_reference(objects: np.ndarray) -> np.ndarray:
    """Extracts reference objects' feet and heads

    :param objects: Segmentation masks
    :type objects: np.ndarray
    :return: Reference feet, heads, average height, STD height
    :rtype: np.ndarray
    """
    raise NotImplementedError("Function `load_model` in `reference_detection.py` has not yet been implemented")
