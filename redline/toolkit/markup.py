"""Preserve the revised PDF and add commentable redline annotations."""
from pathlib import Path
from tempfile import NamedTemporaryFile
import os

import cv2
import numpy as np
import pymupdf

from redline.models import ChangeLog, Change

COLORS = {"addition": (0.04, 0.53, 0.19), "deletion": (0.81, 0.12, 0.13), "modification": (0.10, 0.35, 0.86)}
LEGEND = "Redline legend: green = addition, red = deletion, blue = modification."


def _comment(change: Change) -> str:
    return (
        f"#{change.id} {change.type.title()} | {change.field}\n"
        f"V1: {change.v1_value or ''}\nV2: {change.v2_value or ''}"
    )


class TagPlanner:
    """Place a numeric badge without covering existing ink or another badge."""

    def __init__(self, page: pymupdf.Page, boxes: list[pymupdf.Rect]):
        self.bounds = page.rect
        self.boxes = [box * page.rotation_matrix for box in boxes]
        self.occupied: list[pymupdf.Rect] = []
        self.words = [pymupdf.Rect(*word[:4]) * page.rotation_matrix for word in page.get_text("words")]
        self.zoom = min(2, 1800 / max(self.bounds.width, self.bounds.height))
        pix = page.get_pixmap(matrix=pymupdf.Matrix(self.zoom, self.zoom), colorspace=pymupdf.csGRAY)
        pixels = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width)
        self.ink = cv2.integral((pixels < 200).astype(np.uint8), sdepth=cv2.CV_32S)

    def _clear(self, footprint: pymupdf.Rect) -> bool:
        if footprint not in self.bounds:
            return False
        x0, y0 = (int(np.floor(value * self.zoom)) for value in (footprint.x0, footprint.y0))
        x1, y1 = (int(np.ceil(value * self.zoom)) for value in (footprint.x1, footprint.y1))
        occupied_pixels = self.ink[y1, x1] - self.ink[y0, x1] - self.ink[y1, x0] + self.ink[y0, x0]
        return occupied_pixels == 0 and all((footprint & word).is_empty for word in self.words)

    def place(self, index: int, radius: float) -> pymupdf.Point | None:
        box = self.boxes[index]
        padding = radius + 1
        interior = padding + 1.5  # Keep the footprint off the box edge and any printed grid rule.
        inset = box + (interior, interior, -interior, -interior)
        left, top = box.x0 + interior, box.y0 + interior
        right, bottom = box.x1 - interior, box.y1 - interior
        middle_x, middle_y = (box.x0 + box.x1) / 2, (box.y0 + box.y1) / 2
        anchors = []
        if inset.is_valid and not inset.is_empty:
            anchors.extend((x, y) for y in (top, bottom, middle_y)
                           for x in (left, right, (left + middle_x) / 2, (right + middle_x) / 2, middle_x))
        anchors.extend((
            (left, box.y0 - padding), (right, box.y0 - padding),
            (left, box.y1 + padding), (right, box.y1 + padding),
            (box.x0 - padding, top), (box.x1 + padding, top),
            (box.x0 - padding, middle_y), (box.x1 + padding, middle_y),
        ))
        for x, y in anchors:
            footprint = pymupdf.Rect(x - padding, y - padding, x + padding, y + padding)
            if not self._clear(footprint):
                continue
            if any((footprint & other).get_area() > 0 for other in self.occupied):
                continue
            if any((footprint & other).get_area() > 0
                   for i, other in enumerate(self.boxes) if i != index):
                continue
            self.occupied.append(footprint)
            return pymupdf.Point(x, y)
        return None


def _draw_tag(
    page: pymupdf.Page, center: pymupdf.Point, radius: float, change: Change, *, fontsize: float,
) -> None:
    text = str(change.id)
    color = COLORS[change.type]
    page.draw_circle(center * page.derotation_matrix, radius, color=color, fill=color, overlay=True)
    offset = pymupdf.get_text_length(text, fontsize=fontsize) / 2
    point = pymupdf.Point(center.x - offset, center.y + fontsize / 3) * page.derotation_matrix
    page.insert_text(point, text, fontsize=fontsize, color=(1, 1, 1), rotate=page.rotation)


def _place_legend(page: pymupdf.Page) -> bool:
    pix = page.get_pixmap(colorspace=pymupdf.csGRAY)
    pixels = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width)
    width = int(pymupdf.get_text_length(LEGEND, fontsize=8)) + 10
    height = 16
    # Use only an empty strip near a page edge, leaving drawing content intact.
    right = pix.width - width - 6
    for y in (*range(6, min(52, pix.height - height), 4), *range(max(6, pix.height - 52), pix.height - height, 4)):
        for x in (6, right):
            if x < 6 or x + width > pix.width:
                continue
            if np.all(pixels[y:y + height, x:x + width] >= 250):
                point = pymupdf.Point(x + 5, y + 11) * page.derotation_matrix
                page.insert_text(point, LEGEND, fontsize=8, rotate=page.rotation)
                return True
    return False


def apply_page_markup(
    page: pymupdf.Page, changes: list[Change], *, annotation_subject: str | None = None,
) -> tuple[int | None, list[int]]:
    """Add a page of redlines and return its first annotation and comment-only IDs."""
    if not changes:
        return None, []
    bounds = page.rect * page.derotation_matrix
    boxes = []
    for change in changes:
        if change.bbox_v2 is None and change.category != "page":
            raise ValueError(f"Change {change.id} requires a V2 bounding box.")
        box = pymupdf.Rect(change.bbox_v2) & bounds if change.bbox_v2 else bounds
        if box.is_empty or box.is_infinite or not box.is_valid:
            raise ValueError(f"Change {change.id} has an invalid V2 bounding box.")
        boxes.append(box)
    planner = TagPlanner(page, boxes)
    first_xref = None
    comment_only = []
    for index, (change, box) in enumerate(zip(changes, boxes)):
        annot = page.add_rect_annot(box)
        color = COLORS[change.type]
        annot.set_colors(stroke=color, fill=color)
        annot.set_border(width=1.25)
        annot.set_info(title=f"Redline #{change.id}", content=_comment(change), subject=annotation_subject)
        annot.update(opacity=0.24)
        if first_xref is None:
            first_xref = annot.xref
        for fontsize in (8, 7, 6):
            radius = max(fontsize * .72, pymupdf.get_text_length(str(change.id), fontsize=fontsize) / 2 + 1.5)
            center = planner.place(index, radius)
            if center is not None:
                _draw_tag(page, center, radius, change, fontsize=fontsize)
                break
        else:
            comment_only.append(change.id)
    return first_xref, comment_only


def place_page_legend(page: pymupdf.Page, first_xref: int) -> None:
    if not _place_legend(page):
        annot = page.load_annot(first_xref)
        annot.set_info(content=f"{annot.info['content']}\n\n{LEGEND}")


def write_markup(
    v2_path: str | Path, changes: ChangeLog, out_path: str | Path, *, show_cosmetic: bool = False,
    annotation_subject: str | None = None,
) -> dict[int, list[int]]:
    """Write V2 redlines and return the changes whose number lives in the PDF comment."""
    source = Path(v2_path)
    target = Path(out_path)
    if source.resolve() == target.resolve():
        raise ValueError("Markup output must be distinct from V2.")
    comment_only = {}
    with pymupdf.open(source, filetype="pdf") as pdf:
        by_page: dict[int, list[Change]] = {}
        for group in changes.pages:
            if group.v2_page is None:
                continue
            if not 1 <= group.v2_page <= len(pdf):
                raise ValueError(f"Invalid V2 page {group.v2_page}.")
            by_page.setdefault(group.v2_page, []).extend(
                change for change in group.changes if show_cosmetic or change.category != "cosmetic"
            )
        for number, page_changes in sorted(by_page.items()):
            page = pdf[number - 1]
            first_xref, hidden_tags = apply_page_markup(page, page_changes, annotation_subject=annotation_subject)
            if hidden_tags:
                comment_only[number] = hidden_tags
            if first_xref is not None:
                place_page_legend(pdf[number - 1], first_xref)

        with NamedTemporaryFile(dir=target.parent, suffix=".pdf", delete=False) as temporary:
            intermediate = Path(temporary.name)
        try:
            pdf.save(intermediate, garbage=3, deflate=True)
            os.replace(intermediate, target)
        finally:
            intermediate.unlink(missing_ok=True)
    return comment_only
