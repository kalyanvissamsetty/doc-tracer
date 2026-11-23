"""Pixel differences on aligned RGB or grayscale uint8 raster data."""
import cv2
import numpy as np

from redline.models import Region


def difference_mask(a: np.ndarray, b: np.ndarray, threshold: int = 20) -> np.ndarray:
    """Return the raster mask after thresholding and closing small gaps."""
    if a.shape != b.shape or a.dtype != np.uint8 or b.dtype != np.uint8:
        raise ValueError("Inputs must be same-sized uint8 rasters.")
    if a.size == 0 or (a.ndim != 2 and (a.ndim != 3 or a.shape[2] != 3)):
        raise ValueError("Inputs must be nonempty grayscale or RGB rasters.")
    if not 0 <= threshold <= 255:
        raise ValueError("Use a threshold between 0 and 255.")
    delta = cv2.absdiff(a, b)
    if a.ndim == 3:
        delta = delta.max(axis=2)
    mask = (delta > threshold).astype(np.uint8)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), dtype=np.uint8))


def pixel_diff(a: np.ndarray, b: np.ndarray, threshold: int = 20, min_area: int = 4) -> list[Region]:
    """Return connected bounding boxes in pixel coordinates, with exclusive right and bottom edges."""
    if min_area < 1:
        raise ValueError("Use a positive minimum area.")
    mask = difference_mask(a, b, threshold)
    _, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    regions = [
        Region(bbox=(x, y, x + width, y + height), area=area)
        for x, y, width, height, area in stats[1:]
        if area >= min_area
    ]
    return sorted(regions, key=lambda region: (region.bbox[1], region.bbox[0]))
