from typing import List, Tuple
from PIL import Image, ImageDraw

import numpy as np


def visualize(img_path: str, args: List[Tuple[np.ndarray, float, float]], show=True) -> Image.Image:
    """Visualizes extract_reference's output.

    :param img_path: Path to image
    :type img_path: str
    :param args: Ouput from extract_reference
    :type args: List[Tuple[np.ndarray, float, float]]
    :param show: If the image should be shown directly, defaults to True
    :type show: bool, optional
    :return: Image with heads and feet drawn on it
    :rtype: Image.Image
    """
    img = Image.open(img_path)
    draw = ImageDraw.Draw(img)
    for reference, *_ in args:
        for i in range(reference.shape[1]):
            heads, feet = reference[:, i]
            draw.line([(*heads), (*feet)], fill="yellow", width=2)
    if show:
        img.show()
    return img
