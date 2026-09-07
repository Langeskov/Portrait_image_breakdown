"""Image loading helpers with explicit EXIF-orientation normalization."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def _jpeg_exif_orientation(path: str | Path) -> int:
    """Read JPEG EXIF orientation (1..8) without a Pillow dependency."""
    try:
        data = Path(path).read_bytes()
    except OSError:
        return 1
    if not data.startswith(b"\xff\xd8"):
        return 1
    pos = 2
    while pos + 4 <= len(data):
        if data[pos] != 0xFF:
            pos += 1
            continue
        marker = data[pos + 1]
        pos += 2
        if marker in (0xD8, 0xD9):
            continue
        if pos + 2 > len(data):
            break
        length = int.from_bytes(data[pos:pos + 2], "big")
        if length < 2 or pos + length > len(data):
            break
        segment = data[pos + 2:pos + length]
        if marker == 0xE1 and segment.startswith(b"Exif\x00\x00"):
            tiff = segment[6:]
            if len(tiff) < 8:
                return 1
            endian = tiff[:2]
            order = "little" if endian == b"II" else "big" if endian == b"MM" else None
            if order is None or int.from_bytes(tiff[2:4], order) != 42:
                return 1
            ifd0 = int.from_bytes(tiff[4:8], order)
            if ifd0 + 2 > len(tiff):
                return 1
            count = int.from_bytes(tiff[ifd0:ifd0 + 2], order)
            entry_pos = ifd0 + 2
            for _ in range(count):
                if entry_pos + 12 > len(tiff):
                    break
                tag = int.from_bytes(tiff[entry_pos:entry_pos + 2], order)
                if tag == 0x0112:
                    typ = int.from_bytes(tiff[entry_pos + 2:entry_pos + 4], order)
                    num = int.from_bytes(tiff[entry_pos + 4:entry_pos + 8], order)
                    if typ == 3 and num >= 1:
                        value = int.from_bytes(tiff[entry_pos + 8:entry_pos + 10], order)
                        return value if 1 <= value <= 8 else 1
                    return 1
                entry_pos += 12
        pos += length
    return 1


def _apply_exif_orientation(image: np.ndarray, orientation: int) -> np.ndarray:
    """Apply the standard EXIF orientation transform to a BGR array."""
    if orientation == 2:
        return cv2.flip(image, 1)
    if orientation == 3:
        return cv2.rotate(image, cv2.ROTATE_180)
    if orientation == 4:
        return cv2.flip(image, 0)
    if orientation == 5:
        return cv2.transpose(cv2.flip(image, 1))
    if orientation == 6:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    if orientation == 7:
        return cv2.transpose(cv2.flip(image, 0))
    if orientation == 8:
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return image


def load_image(path: str | Path) -> np.ndarray | None:
    """Load raw BGR pixels and apply JPEG EXIF orientation exactly once."""
    # OpenCV applies EXIF orientation unless IMREAD_IGNORE_ORIENTATION is set;
    # explicitly disable that behavior because this module applies the transform.
    flags = cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION
    image = cv2.imread(str(path), flags)
    if image is None:
        return None
    return _apply_exif_orientation(image, _jpeg_exif_orientation(path))


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
