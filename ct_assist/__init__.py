"""
AssitanceTransform

Python package for finding reference objects (head-feet pairs) from images, and using them with CameraTransform.

This package was made for TNO and EU-project ASSISTANCE.

Transform contains functions for fitting CameraTransform's camera, extracting exif data from images, and estimating sensor size based.
"""

from . import transform
from . import reference_detection