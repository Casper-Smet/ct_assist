"""
AssitanceTransform

Python package for finding reference objects (head-feet pairs) from images, and using them with CameraTransform.

This package was made for TNO and EU-project ASSISTANCE.

Estimator contains functions calculating the area of a polygon, and the "release rate", the rate at which the size of a polygon increases over time.

Transform contains functions for fitting CameraTransform's camera, extracting exif data from images, and estimating sensor size based.
"""

from . import estimator
from . import transform
from . import reference_detection