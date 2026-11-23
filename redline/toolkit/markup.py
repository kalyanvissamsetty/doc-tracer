"""Preserve the revised PDF and add commentable redline annotations."""
from pathlib import Path
from tempfile import NamedTemporaryFile
import os

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


def _draw_tag(page: pymupdf.Page, box: pymupdf.Rect, change: Change) -> None:
    text = str(change.id)
    radius = max(7, 3 + len(text) * 2)
    displayed = box * page.rotation_matrix
    x = min(max(displayed.x0 + radius, radius), page.rect.width - radius)
    y = displayed.y0 - radius
    if y < radius:
        y = displayed.y1 + radius
    y = min(max(y, radius), page.rect.height - radius)
    center = pymupdf.Point(x, y)
    color = COLORS[change.type]
    page.draw_circle(center * page.derotation_matrix, radius, color=color, fill=color, overlay=True)
    offset = pymupdf.get_text_length(text, fontsize=8) / 2
    point = pymupdf.Point(x - offset, y + 2.7) * page.derotation_matrix
    page.insert_text(point, text, fontsize=8, color=(1, 1, 1), rotate=page.rotation)


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


def write_markup(
    v2_path: str | Path, changes: ChangeLog, out_path: str | Path, *, show_cosmetic: bool = False,
    annotation_subject: str | None = None,
) -> None:
    """Write V2 redlines. Deleted pages stay in the log because V2 has no surface for them."""
    source = Path(v2_path)
    target = Path(out_path)
    if source.resolve() == target.resolve():
        raise ValueError("Markup output must be distinct from V2.")
    with pymupdf.open(source, filetype="pdf") as pdf:
        first_annotation_by_page = {}
        for group in changes.pages:
            if group.v2_page is None:
                continue
            if not 1 <= group.v2_page <= len(pdf):
                raise ValueError(f"Invalid V2 page {group.v2_page}.")
            page = pdf[group.v2_page - 1]
            bounds = page.rect * page.derotation_matrix
            for change in group.changes:
                if change.category == "cosmetic" and not show_cosmetic:
                    continue
                if change.bbox_v2 is None and change.category != "page":
                    raise ValueError(f"Change {change.id} requires a V2 bounding box.")
                box = pymupdf.Rect(change.bbox_v2) & bounds if change.bbox_v2 else bounds
                if box.is_empty or box.is_infinite or not box.is_valid:
                    raise ValueError(f"Change {change.id} has an invalid V2 bounding box.")
                annot = page.add_rect_annot(box)
                color = COLORS[change.type]
                annot.set_colors(stroke=color, fill=color)
                annot.set_border(width=1.25)
                annot.set_info(title=f"Redline #{change.id}", content=_comment(change), subject=annotation_subject)
                annot.update(opacity=0.24)
                _draw_tag(page, box, change)
                first_annotation_by_page.setdefault(group.v2_page, annot.xref)

        for page_number, xref in sorted(first_annotation_by_page.items()):
            page = pdf[page_number - 1]
            if not _place_legend(page):
                annot = page.load_annot(xref)
                annot.set_info(content=f"{annot.info['content']}\n\n{LEGEND}")

        with NamedTemporaryFile(dir=target.parent, suffix=".pdf", delete=False) as temporary:
            intermediate = Path(temporary.name)
        try:
            pdf.save(intermediate, garbage=3, deflate=True)
            os.replace(intermediate, target)
        finally:
            intermediate.unlink(missing_ok=True)
