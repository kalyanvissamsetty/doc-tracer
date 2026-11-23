"""Extract ruled tables with coordinates and match rows by ordered key sets."""
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
import pymupdf

from redline.models import BBox
from redline.toolkit.values import normalize_value


@dataclass(frozen=True)
class Grid:
    """Grid rules in displayed raster pixels, with the source image dimensions."""

    x_rules: tuple[int, ...]
    y_rules: tuple[int, ...]
    width: int
    height: int


@dataclass(frozen=True)
class Cell:
    value: str
    bbox: BBox


@dataclass(frozen=True)
class RowMatch:
    """Zero-based data row indices (excluding the header)."""

    v1_row: int | None
    v2_row: int | None
    matched_by: tuple[str, ...] = ()


def _rule_centers(projection: np.ndarray, minimum: float) -> tuple[int, ...]:
    indices = np.flatnonzero(projection >= minimum)
    runs = np.split(indices, np.flatnonzero(np.diff(indices) > 1) + 1)
    return tuple(int(round(run.mean())) for run in runs if len(run))


def detect_table_grid(image: np.ndarray) -> Grid | None:
    """Find the largest coherent ruled grid in an upright grayscale or RGB image."""
    if image.dtype != np.uint8 or image.size == 0:
        raise ValueError("Table detection requires a nonempty uint8 image.")
    if image.ndim == 2:
        gray = image
    elif image.ndim == 3 and image.shape[2] == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        raise ValueError("Table detection requires a grayscale or RGB image.")

    height, width = gray.shape
    _, ink = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    horizontal = cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((1, max(12, width // 35)), np.uint8))
    vertical = cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((max(12, height // 35), 1), np.uint8))
    rules = cv2.bitwise_or(horizontal, vertical)
    _, _, stats, _ = cv2.connectedComponentsWithStats(rules)
    candidates = sorted(stats[1:], key=lambda item: int(item[2]) * int(item[3]), reverse=True)
    for x, y, w, h, _ in candidates:
        xs = _rule_centers(np.count_nonzero(vertical[y:y + h, x:x + w], axis=0), h * 0.55)
        ys = _rule_centers(np.count_nonzero(horizontal[y:y + h, x:x + w], axis=1), w * 0.55)
        if len(xs) < 2 or len(ys) < 3:
            continue
        # Require repeated crossings so unrelated graphic rules do not form a table.
        junctions = rules[np.ix_(np.array(ys) + y, np.array(xs) + x)]
        if np.count_nonzero(junctions) >= junctions.size * 0.8:
            return Grid(tuple(int(x + col) for col in xs), tuple(int(y + row) for row in ys), width, height)
    return None


def _ocr_words(layout_page: Mapping[str, Any], grid: Grid) -> list[tuple[float, float, str]]:
    width, height = layout_page["width"], layout_page["height"]
    if width <= 0 or height <= 0:
        raise ValueError("OCR page dimensions must be positive.")
    if not np.isclose(width / height, grid.width / grid.height, rtol=0.03):
        raise ValueError("OCR page and raster grid must have the same orientation and aspect ratio.")
    words = []
    for word in layout_page.get("words", []):
        polygon = word.get("polygon")
        if polygon and len(polygon) % 2 == 0:
            words.append((
                float(np.mean(polygon[::2])) * grid.width / width,
                float(np.mean(polygon[1::2])) * grid.height / height,
                word["content"],
            ))
    return words


def extract_table(
    page: pymupdf.Page, grid: Grid, *, layout_page: Mapping[str, Any] | None = None
) -> list[list[Cell]]:
    """Return the full cell matrix including the header and empty cells.

    Supply a Document Intelligence page in the same displayed orientation for scans.
    Cell boxes use unrotated PDF points.
    """
    if grid.width <= 0 or grid.height <= 0:
        raise ValueError("Grid dimensions must be positive.")
    page_x, page_y = page.rect.width / grid.width, page.rect.height / grid.height
    if not np.isclose(page_x, page_y, rtol=0.03):
        raise ValueError("Grid and PDF page must have the same orientation and aspect ratio.")

    if layout_page is None:
        words = []
        for x0, y0, x1, y1, content, *_ in page.get_text("words", sort=True):
            center = pymupdf.Point((x0 + x1) / 2, (y0 + y1) / 2) * page.rotation_matrix
            words.append((center.x / page_x, center.y / page_y, content))
    else:
        words = _ocr_words(layout_page, grid)
    buckets = defaultdict(list)
    for x, y, content in words:
        col = int(np.searchsorted(grid.x_rules, x, side="right")) - 1
        row = int(np.searchsorted(grid.y_rules, y, side="right")) - 1
        buckets[row, col].append(content)

    return [
        [
            Cell(
                " ".join(buckets[row, col]),
                tuple(pymupdf.Rect(x0 * page_x, y0 * page_y, x1 * page_x, y1 * page_y) * page.derotation_matrix),
            )
            for col, (x0, x1) in enumerate(zip(grid.x_rules, grid.x_rules[1:]))
        ]
        for row, (y0, y1) in enumerate(zip(grid.y_rules, grid.y_rules[1:]))
    ]


def match_rows(
    rows_v1: Sequence[Mapping[str, str]],
    rows_v2: Sequence[Mapping[str, str]],
    keys: Sequence[str | tuple[str, ...]] = ("item number", ("description", "heat number")),
) -> list[RowMatch]:
    """Match unique nonempty keys in priority order and preserve ambiguous rows as unpaired."""

    def index(rows, remaining, fields):
        indexed = defaultdict(list)
        for i in sorted(remaining):
            normalized = {normalize_value(name): value for name, value in rows[i].items()}
            identity = tuple(normalize_value(normalized.get(normalize_value(field), "")) for field in fields)
            if all(identity):
                indexed[identity].append(i)
        return indexed

    remaining_v1, remaining_v2 = set(range(len(rows_v1))), set(range(len(rows_v2)))
    matched = []
    for key in keys:
        fields = (key,) if isinstance(key, str) else key
        if not fields:
            raise ValueError("Row matching keys must include at least one field.")
        old = index(rows_v1, remaining_v1, fields)
        new = index(rows_v2, remaining_v2, fields)
        for identity, before in old.items():
            after = new.get(identity, [])
            if len(before) == len(after) == 1:
                i, j = before[0], after[0]
                matched.append(RowMatch(i, j, fields))
                remaining_v1.remove(i)
                remaining_v2.remove(j)
    matched.extend(RowMatch(None, j) for j in remaining_v2)
    return [
        *sorted(matched, key=lambda match: match.v2_row),
        *(RowMatch(i, None) for i in sorted(remaining_v1)),
    ]
