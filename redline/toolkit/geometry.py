"""Fit a proposed redline box to a ruled cell without discarding its evidence."""
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
import pymupdf

from redline.models import Change, GeometryEdit
from redline.toolkit.tables import Grid, _rule_centers, detect_table_grid


def _faint_grid(gray: np.ndarray) -> Grid | None:
    """Recover long, faint rulings on fairly large scanned tables."""
    height, width = gray.shape
    ink = (gray < 185).astype(np.uint8)
    horizontal = cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((1, max(12, width // 4)), np.uint8))
    vertical = cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((max(12, height // 8), 1), np.uint8))
    xs = _rule_centers(np.count_nonzero(vertical, axis=0), height * .23)
    ys = _rule_centers(np.count_nonzero(horizontal, axis=1), width * .40)
    if len(xs) < 3 or len(ys) < 4:
        return None
    crossings = 0
    for x in xs:
        for y in ys:
            window = np.s_[max(0, y - 4):y + 5, max(0, x - 4):x + 5]
            crossings += bool(np.any(horizontal[window]) and np.any(vertical[window]))
    if crossings < .8 * len(xs) * len(ys):
        return None
    return Grid(xs, ys, width, height)


def snap_to_detected_cell(
    path: Path, page_number: int, change: Change, *, actor: Literal["agent", "reviewer"]
) -> GeometryEdit:
    """Propose the unique detected cell covering the existing evidence box."""
    if change.category != "value" or change.bbox_v2 is None:
        raise ValueError("Only individual value changes can snap to a detected cell.")
    evidence = pymupdf.Rect(change.bbox_v2)
    if evidence.is_empty or not evidence.is_valid or not np.isfinite(tuple(evidence)).all():
        raise ValueError("The change needs a valid evidence box.")
    with pymupdf.open(path) as pdf:
        if not 1 <= page_number <= len(pdf):
            raise ValueError("The revised page was not found.")
        page = pdf[page_number - 1]
        evidence_display = evidence * page.rotation_matrix
        zoom = min(2.5, 2100 / max(page.rect.width, page.rect.height))
        pix = page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), colorspace=pymupdf.csGRAY, annots=False)
        gray = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width)
        grid = detect_table_grid(gray) or _faint_grid(gray)
        if grid is None:
            raise ValueError("No reliable ruled grid was detected on this page.")
        sx = 1 / zoom
        matches = []
        for row, (y0, y1) in enumerate(zip(grid.y_rules, grid.y_rules[1:]), 1):
            for col, (x0, x1) in enumerate(zip(grid.x_rules, grid.x_rules[1:]), 1):
                target = pymupdf.Rect(x0 * sx, y0 * sx, x1 * sx, y1 * sx)
                if (target & evidence_display).get_area() < .9 * evidence_display.get_area():
                    continue
                gutter = (target + (-1, -1, 1, 1))
                if evidence_display in gutter:
                    matches.append((row, col, target))
        if len(matches) != 1:
            raise ValueError("The evidence does not fit inside one detected cell. Keep the original box for review.")
        row, column, target = matches[0]
        # A partial grid detector can miss a faint rule and accidentally combine two cells.
        inset = (target + (2, 2, -2, -2)) * zoom
        if not inset.is_empty:
            x0, y0, x1, y1 = map(int, inset)
            ink = gray[y0:y1, x0:x1] < 185
            if ink.size and (ink.mean(axis=0).max() > .8 or ink.mean(axis=1).max() > .8):
                raise ValueError("The detected cell contains another ruling. Keep the original box for review.")
        after = target * page.derotation_matrix
        if max(abs(a - b) for a, b in zip(evidence, after)) <= .8:
            raise ValueError("The box already fits the detected cell.")
        return GeometryEdit(
            change_id=change.id, v2_page=page_number, before=tuple(evidence), after=tuple(after),
            row=row, column=column, actor=actor,
        )
