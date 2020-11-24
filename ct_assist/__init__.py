"""

ct_assist, or CameraTransform-Assistance, is a Python package for finding reference objects (head-feet pairs) from images, and using them with CameraTransform.

This package was made for TNO and EU-project ASSISTANCE.

transform.py contains functions for fitting CameraTransform's camera, extracting exif data from images, and estimating sensor size based.

reference_detection.py contains functions for finding head-feet pairs on images, used for CameraTransform fitting.
"""

from . import transform
from . import reference_detection

__all__ = ['transform', "reference_detection"]
