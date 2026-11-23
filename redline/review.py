"""Human review, page thumbnails, and accepted-change exports."""
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Literal

import pymupdf
from pydantic import BaseModel, ConfigDict, Field

from redline.models import BBox, Change, ChangeLog, PageChanges
from redline.toolkit import write_markup
from redline.toolkit.markup import apply_page_markup, place_page_legend


class ReviewEdit(BaseModel):
    model_config = ConfigDict(extra="forbid")
    field: str = Field(min_length=1, max_length=300)
    type: Literal["addition", "deletion", "modification"]
    v1_value: str | None = None
    v2_value: str | None = None


def find_change(log: ChangeLog, change_id: int) -> Change:
    for group in log.pages:
        for change in group.changes:
            if change.id == change_id:
                return change
    raise ValueError("This change no longer exists.")


def render_preview(path: Path, number: int, box: BBox | None = None, *, context: float = 16, annots: bool = True) -> bytes:
    with pymupdf.open(path) as pdf:
        page = pdf[number - 1]
        clip = page.rect
        if box is not None:
            region = pymupdf.Rect(box) * page.rotation_matrix
            clip &= pymupdf.Rect(
                region.x0 - context, region.y0 - context, region.x1 + context, region.y1 + context,
            )
        if clip.is_empty:
            raise ValueError("The selected region is outside the page.")
        zoom = min(2, 1450 / max(clip.width, clip.height))
        return page.get_pixmap(clip=clip, matrix=pymupdf.Matrix(zoom, zoom), annots=annots).tobytes("png")



def render_edited_page(path: Path, group: PageChanges, *, show_cosmetic: bool) -> tuple[bytes, list[int]]:
    with pymupdf.open(path) as pdf:
        page = pdf[group.v2_page - 1]
        changes = [change for change in group.changes if show_cosmetic or change.category != "cosmetic"]
        first_xref, comment_only = apply_page_markup(page, changes)
        if first_xref is not None:
            page = pdf[group.v2_page - 1]
            place_page_legend(page, first_xref)
        zoom = min(2, 1450 / max(page.rect.width, page.rect.height))
        return page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom)).tobytes("png"), comment_only

def _summary(doc: pymupdf.Document, accepted: list[Change]) -> None:
    page = doc.new_page(width=612, height=792)
    teal = (0.03, 0.41, 0.39)
    page.draw_rect((0, 0, 612, 14), color=teal, fill=teal)
    page.insert_text((48, 79), "Redline summary", fontsize=27, fontname="hebo")
    page.insert_text((48, 113), f"{len(accepted)} accepted change{'s' if len(accepted) != 1 else ''}", fontsize=13)
    for idx, kind in enumerate(("addition", "deletion", "modification")):
        y = 170 + idx * 43
        page.insert_text((48, y), kind.title() + "s", fontsize=14)
        page.insert_text((510, y), str(sum(change.type == kind for change in accepted)), fontsize=14, fontname="hebo")
        page.draw_line((48, y + 11), (564, y + 11), color=(0.85, 0.89, 0.89))
    page.insert_text((48, 320), "Removed pages are listed in the accompanying change log.", fontsize=11)


def write_final(v2: Path, log: ChangeLog, output: Path, *, summary: bool) -> dict[int, list[int]]:
    selected = log.model_copy(deep=True)
    for group in selected.pages:
        group.changes = [change for change in group.changes if change.human_status == "accepted"]

    if not summary:
        return write_markup(v2, selected, output, show_cosmetic=True)
    with TemporaryDirectory(dir=output.parent) as directory:
        base = Path(directory) / "marked.pdf"
        comment_only = write_markup(v2, selected, base, show_cosmetic=True)
        with pymupdf.open(base) as pdf:
            _summary(pdf, [change for group in selected.pages for change in group.changes])
            pdf.save(output, deflate=True)
    return comment_only
