from pathlib import Path

import numpy as np
import pymupdf
import pytest

from redline.models import Change, ChangeLog, PageChanges
from redline.storage import LocalStorage
from redline.toolkit import pair_pages, pixel_diff, probe_pdf, render_page, write_markup


def make_pdf(path: Path, titles: list[str], rotation: int = 0) -> Path:
    with pymupdf.open() as pdf:
        for title in titles:
            page = pdf.new_page(width=500, height=350)
            page.insert_text((50, 90), title)
            page.set_rotation(rotation)
        pdf.save(path)
    return path


def test_probe_scan_and_review_annotations(tmp_path):
    path = tmp_path / "scan.pdf"
    with pymupdf.open() as pdf:
        page = pdf.new_page(width=200, height=300)
        image = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 200, 300), False)
        image.clear_with(230)
        page.insert_image(page.rect, pixmap=image)
        page.draw_line((30, 60), (80, 60))
        page.add_text_annot((30, 30), "Check value")
        page.set_rotation(90)
        pdf.save(path)
    result = probe_pdf(path)
    page = result.pages[0]
    assert result.page_count == 1
    assert (page.width, page.height, page.rotation) == (200, 300, 90)
    assert page.full_page_scan and not page.has_text
    assert page.images[0].bbox == (0, 0, 200, 300)
    assert page.images[0].digest
    assert page.vector_count >= 1
    assert page.annotations[0].kind == "Text"


def test_render_dpi_rotation_and_annotation_switch(tmp_path):
    path = make_pdf(tmp_path / "annotated.pdf", ["Drawing A"], 270)
    with pymupdf.open(path) as pdf:
        pdf[0].add_text_annot((30, 40), "Review")
        pdf.saveIncr()
    store = LocalStorage(tmp_path / "artifacts")
    plain = render_page(path, 1, 144, False, storage=store)
    reviewed = render_page(path, 1, 144, True, storage=store)
    assert plain != reviewed
    assert plain == render_page(path, 1, 144, False, storage=store)
    assert (pymupdf.Pixmap(store.read(plain)).width, pymupdf.Pixmap(store.read(plain)).height) == (700, 1000)
    with pytest.raises(ValueError, match="Page must be"):
        render_page(path, 0, 72, storage=store)


def test_pair_reordered_revised_added_and_removed_pages(tmp_path):
    first = make_pdf(tmp_path / "old.pdf", ["Pump Sheet A-10", "Motor Sheet B-20", "Deprecated"])
    second = make_pdf(tmp_path / "new.pdf", ["Revised Motor Sheet B-20", "Safety training", "Pump Sheet A-10"])
    assert [(p.v1_page, p.v2_page) for p in pair_pages(first, second)] == [
        (2, 1), (None, 2), (1, 3), (3, None),
    ]


def test_pixel_diff_preserves_separate_color_changes_and_filters_specks():
    first = np.full((55, 70, 3), 255, np.uint8)
    second = first.copy()
    second[5:10, 5:11] = (255, 0, 255)
    second[5:11, 20:26] = (0, 255, 255)
    second[23, 30] = 0
    regions = pixel_diff(first, second, threshold=20, min_area=4)
    assert [(r.bbox, r.area) for r in regions] == [((5, 5, 11, 10), 30), ((20, 5, 26, 11), 36)]
    assert pixel_diff(first, first) == []
    with pytest.raises(ValueError):
        pixel_diff(first, second[..., 0])


@pytest.mark.parametrize("rotation", [0, 90, 180, 270])
def test_markup_round_trip(tmp_path, rotation):
    source = make_pdf(tmp_path / "v2.pdf", ["Drawing"], rotation)
    with pymupdf.open(source) as pdf:
        pdf[0].add_text_annot((25, 25), "Existing note")
        pdf.saveIncr()
    original = source.read_bytes()
    changes = [
        Change(id=i, type=kind, category="value", field="Pressure", v1_value="100", v2_value="120",
               bbox_v2=(50 + i * 70, 130, 100 + i * 70, 160), evidence="Changed pixels", confidence=0.92)
        for i, kind in enumerate(("addition", "deletion", "modification"), 1)
    ]
    changes.append(changes[-1].model_copy(update={"id": 4, "category": "cosmetic"}))
    removed_page = changes[1].model_copy(update={"id": 5, "category": "page", "bbox_v2": None})
    log = ChangeLog(job_id="test", v1="v1.pdf", v2="v2.pdf", pages=[
        PageChanges(v1_page=1, v2_page=1, method="text", changes=changes),
        PageChanges(v1_page=2, v2_page=None, method="structural", changes=[removed_page]),
    ])
    output = tmp_path / "result.pdf"
    write_markup(source, log, output)
    assert source.read_bytes() == original
    with pymupdf.open(output) as pdf:
        assert len(pdf) == 1 and pdf[0].rotation == rotation
        page = pdf[0]
        assert "Redline legend" in page.get_text()
        assert all(str(number) in page.get_text().splitlines() for number in (1, 2, 3))
        annotations = list(page.annots())
        assert annotations[0].info["content"] == "Existing note"
        assert len(annotations) == 4
        for annot, change in zip(annotations[1:], changes):
            assert tuple(annot.rect) == pytest.approx(change.bbox_v2, abs=1.25)
            assert annot.opacity == pytest.approx(0.24)
            assert f"#{change.id} {change.type.title()}" in annot.info["content"]
            assert "V1: 100\nV2: 120" in annot.info["content"]
        assert annotations[1].colors["fill"][1] > annotations[1].colors["fill"][0]
        assert annotations[2].colors["fill"][0] > annotations[2].colors["fill"][1]
        assert annotations[3].colors["fill"][2] > annotations[3].colors["fill"][1]


def test_dense_page_keeps_legend_in_new_comment(tmp_path):
    source = tmp_path / "dense.pdf"
    with pymupdf.open() as pdf:
        page = pdf.new_page(width=500, height=350)
        page.draw_rect(page.rect, fill=(0.95, 0.95, 0.95))
        note = page.add_text_annot((50, 80), "Keep my note")
        note.set_info(title="Redline #previous")
        pdf.save(source)
    change = Change(
        id=1, type="modification", category="value", field="Inspection",
        bbox_v2=(120, 130, 200, 160), evidence="Changed pixels", confidence=0.6,
    )
    assert change.needs_review
    log = ChangeLog(job_id="test", v1="v1.pdf", v2="dense.pdf", pages=[
        PageChanges(v1_page=1, v2_page=1, method="pixel_diff", changes=[change]),
    ])
    output = tmp_path / "dense-marked.pdf"
    write_markup(source, log, output)
    with pymupdf.open(output) as pdf:
        page = pdf[0]
        existing, added = list(page.annots())
        assert existing.info["content"] == "Keep my note"
        assert "Redline legend" in added.info["content"]
        assert "Redline legend" not in page.get_text()
