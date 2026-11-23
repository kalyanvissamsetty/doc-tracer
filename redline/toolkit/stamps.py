"""Locate colored stamp candidates in an upright RGB raster."""
import cv2
import numpy as np

from redline.models import Region


def detect_stamps(image: np.ndarray, *, min_area: int = 35, join_radius: int = 2) -> list[Region]:
    """Join nearby colored ink fragments and discard components inside another box.

    Results are candidates, since colored logos and graphics can have the same appearance.
    Choose a join radius smaller than half the gap between adjacent stamps.
    """
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8 or image.size == 0:
        raise ValueError("Stamp detection requires a nonempty RGB uint8 image.")
    if min_area < 1 or join_radius < 0:
        raise ValueError("Use a positive minimum area and nonnegative joining radius.")
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv, (0, 85, 45), (179, 255, 255))
    size = 2 * join_radius + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    joined = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    _, labels, stats, _ = cv2.connectedComponentsWithStats(joined)
    support = np.bincount(labels[mask != 0], minlength=len(stats))
    candidates = [
        Region(bbox=(x, y, x + w, y + h), area=int(support[index]))
        for index, (x, y, w, h, _) in enumerate(stats[1:], 1)
        if support[index] >= min_area
    ]

    def box_area(region):
        x0, y0, x1, y1 = region.bbox
        return (x1 - x0) * (y1 - y0)

    kept = []
    for region in sorted(candidates, key=box_area, reverse=True):
        x0, y0, x1, y1 = region.bbox
        if any(a <= x0 and b <= y0 and c >= x1 and d >= y1 for a, b, c, d in (item.bbox for item in kept)):
            continue
        kept.append(region)
    return sorted(kept, key=lambda region: (region.bbox[1], region.bbox[0]))
