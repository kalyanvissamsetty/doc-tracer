"""Compare embedded image occurrences using content digests and PDF bounds."""
from collections import defaultdict
from typing import Literal

import numpy as np
import pymupdf
from pydantic import BaseModel
from scipy.optimize import linear_sum_assignment

from redline.models import EmbeddedImage


class ImageChange(BaseModel):
    kind: Literal["moved", "resized", "replaced", "added", "removed"]
    v1: EmbeddedImage | None = None
    v2: EmbeddedImage | None = None
    moved: bool = False
    resized: bool = False


def embedded_images(page: pymupdf.Page) -> list[EmbeddedImage]:
    return [
        EmbeddedImage(
            bbox=tuple(image["bbox"]), digest=image["digest"].hex(),
            width=image["width"], height=image["height"],
        )
        for image in page.get_image_info(hashes=True)
    ]


def _group(images: list[EmbeddedImage]) -> dict[str, list[int]]:
    groups = defaultdict(list)
    for index, image in enumerate(images):
        groups[image.digest].append(index)
    return groups


def _overlap(a: EmbeddedImage, b: EmbeddedImage) -> float:
    first, second = pymupdf.Rect(a.bbox), pymupdf.Rect(b.bbox)
    common = (first & second).get_area()
    union = first.get_area() + second.get_area() - common
    return common / union if union else 0.0


def _assignment(scores: np.ndarray, *, maximize: bool = False):
    return zip(*linear_sum_assignment(scores, maximize=maximize))


def compare_embedded_images(
    page_v1: pymupdf.Page, page_v2: pymupdf.Page, *,
    tolerance: float = 1.0, replacement_iou: float = 0.6,
) -> list[ImageChange]:
    """Match identical content globally before checking which placements were replaced."""
    if not np.isfinite(tolerance) or tolerance < 0 or not 0 < replacement_iou <= 1:
        raise ValueError("Use a nonnegative point tolerance and a replacement IoU in (0, 1].")
    before, after = embedded_images(page_v1), embedded_images(page_v2)
    before_groups, after_groups = _group(before), _group(after)
    old_left, new_left = set(range(len(before))), set(range(len(after)))
    changes = []

    for digest, old_indices in before_groups.items():
        new_indices = after_groups.get(digest)
        if not new_indices:
            continue
        old_boxes = np.array([before[i].bbox for i in old_indices])
        new_boxes = np.array([after[j].bbox for j in new_indices])
        distances = np.abs(old_boxes[:, None, :] - new_boxes[None, :, :])
        costs = distances.sum(axis=2)
        # Exact placements take priority when the same image occurs multiple times.
        costs += (distances.max(axis=2) > tolerance) * (costs.max() * (min(costs.shape) + 1) + 1)
        for row, col in _assignment(costs):
            i, j = old_indices[row], new_indices[col]
            first, second = pymupdf.Rect(before[i].bbox), pymupdf.Rect(after[j].bbox)
            moved = abs(first.x0 - second.x0) > tolerance or abs(first.y0 - second.y0) > tolerance
            resized = abs(first.width - second.width) > tolerance or abs(first.height - second.height) > tolerance
            if moved or resized:
                changes.append(ImageChange(
                    kind="resized" if resized else "moved", v1=before[i], v2=after[j],
                    moved=moved, resized=resized,
                ))
            old_left.remove(i)
            new_left.remove(j)

    old_indices, new_indices = sorted(old_left), sorted(new_left)
    if old_indices and new_indices:
        overlaps = np.array([[_overlap(before[i], after[j]) for j in new_indices] for i in old_indices])
        scores = np.where(overlaps >= replacement_iou, overlaps + min(overlaps.shape), 0)
        for row, col in _assignment(scores, maximize=True):
            if scores[row, col] == 0:
                continue
            i, j = old_indices[row], new_indices[col]
            changes.append(ImageChange(kind="replaced", v1=before[i], v2=after[j]))
            old_left.remove(i)
            new_left.remove(j)

    changes.extend(ImageChange(kind="removed", v1=before[i]) for i in sorted(old_left))
    changes.extend(ImageChange(kind="added", v2=after[j]) for j in sorted(new_left))
    return changes
