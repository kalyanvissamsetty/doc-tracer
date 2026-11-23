"""Pixel differences on aligned RGB or grayscale uint8 raster data."""
import cv2
import numpy as np

from redline.models import Region


def pixel_diff(a: np.ndarray, b: np.ndarray, threshold: int = 20, min_area: int = 4) -> list[Region]:
    """Return connected bounding boxes in pixel coordinates, with exclusive right and bottom edges."""
    if a.shape != b.shape or a.dtype != np.uint8 or b.dtype != np.uint8:
        raise ValueError("Inputs must be same-sized uint8 rasters.")
    if a.size == 0 or (a.ndim != 2 and (a.ndim != 3 or a.shape[2] != 3)):
        raise ValueError("Inputs must be nonempty grayscale or RGB rasters.")
    if not 0 <= threshold <= 255 or min_area < 1:
        raise ValueError("Use a threshold between 0 and 255 and a positive minimum area.")
    delta = cv2.absdiff(a, b)
    if a.ndim == 3:
        delta = delta.max(axis=2)
    mask = (delta > threshold).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), dtype=np.uint8))
    _, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    regions = [
        Region(bbox=(x, y, x + width, y + height), area=area)
        for x, y, width, height, area in stats[1:]
        if area >= min_area
    ]
    return sorted(regions, key=lambda region: (region.bbox[1], region.bbox[0]))
