"""Image loading helpers with explicit EXIF-orientation normalization."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps


def load_image(path: str | Path) -> np.ndarray | None:
    """Load an image as BGR after applying the EXIF display orientation.

    This avoids treating a portrait photograph whose JPEG pixels are stored
    landscape plus an EXIF rotation tag as a landscape image.
    """
    try:
        with Image.open(path) as source:
            oriented = ImageOps.exif_transpose(source).convert("RGB")
            rgb = np.asarray(oriented)
    except (OSError, ValueError):
        return None
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        return None
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def frame_orientation(image: np.ndarray) -> str:
    """Return the display orientation of a BGR image."""
    if image is None or image.ndim < 2:
        return "unknown"
    h, w = image.shape[:2]
    if w > h:
        return "landscape"
    if h > w:
        return "portrait"
    return "square"
