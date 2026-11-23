from io import BytesIO

import cv2
import numpy as np
from PIL import Image
import pymupdf
import pytest

from redline.storage import LocalStorage
from redline.toolkit import (
    compare_embedded_images, detect_stamps, detect_table_grid, extract_table, match_rows,
    normalize_value, render_page, split_annotation_layer, view_crop, view_page,
)


def _table(page):
    for x in (30, 170, 290):
        page.draw_line((x, 30), (x, 150), width=2)
    for y in (30, 70, 110, 150):
        page.draw_line((30, y), (290, y), width=2)
    for x, y, text in [
        (40, 55, "Item number"), (180, 55, "Description"),
        (40, 95, "7"), (180, 95, "Steel flange"), (40, 135, "9"),
    ]:
        page.insert_text((x, y), text)


def _rgb(pix):
    return np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)


def test_table_grid_and_pdf_extraction():
    with pymupdf.open() as pdf:
        page = pdf.new_page(width=340, height=220)
        _table(page)
        grid = detect_table_grid(_rgb(page.get_pixmap(matrix=pymupdf.Matrix(2, 2))))
        assert grid is not None
        assert grid.x_rules == pytest.approx((60, 340, 580), abs=2)
        assert grid.y_rules == pytest.approx((60, 140, 220, 300), abs=2)
        rows = extract_table(page, grid)
        assert [[cell.value for cell in row] for row in rows] == [
            ["Item number", "Description"], ["7", "Steel flange"], ["9", ""],
        ]
        assert rows[1][1].bbox == pytest.approx((170, 70, 290, 110), abs=1)
        assert detect_table_grid(np.full((220, 340), 255, np.uint8)) is None


def test_scanned_table_uses_ocr_coordinates():
    with pymupdf.open() as source, pymupdf.open() as flattened:
        page = source.new_page(width=340, height=220)
        _table(page)
        pix = page.get_pixmap()
        scan = flattened.new_page(width=340, height=220)
        scan.insert_image(scan.rect, pixmap=pix)
        grid = detect_table_grid(_rgb(pix))
        assert grid is not None and not scan.get_text()
        layout_page = {"width": 340 / 72, "height": 220 / 72, "words": [
            {"content": "Steel", "polygon": [v / 72 for v in (180, 80, 205, 80, 205, 96, 180, 96)]},
            {"content": "flange", "polygon": [v / 72 for v in (209, 80, 240, 80, 240, 96, 209, 96)]},
        ]}
        assert extract_table(scan, grid, layout_page=layout_page)[1][1].value == "Steel flange"
        with pytest.raises(ValueError, match="orientation"):
            extract_table(scan, grid, layout_page={**layout_page, "width": 1})


def test_match_rows_by_unique_item_then_composite_key():
    def row(item, description, heat=""):
        return {"Item Number": item, "Description": description, "Heat Number": heat}

    old = [row("7", "Flange"), row("8", "Pipe", "X7"), row("9", "Washer"),
           row("duplicate", "Ring"), row("duplicate", "Ring")]
    new = [row("80", "  PIPE ", "x7"), row("7", "Forged flange"), row("12", "Gasket"),
           row("duplicate", "Ring"), row("duplicate", "Ring")]
    matches = match_rows(old, new)
    assert [(m.v1_row, m.v2_row) for m in matches] == [
        (1, 0), (0, 1), (None, 2), (None, 3), (None, 4), (2, None), (3, None), (4, None),
    ]
    assert matches[0].matched_by == ("description", "heat number")
    assert matches[1].matched_by == ("item number",)


@pytest.mark.parametrize(("source", "expected"), [
    ("  TEST\n Pressure  100PSI ", "test pressure 100 psi"),
    ('2″', "2 in"), ("2''", "2 in"), ("2′", "2 ft"),
    ("#  52", "#52"), ("10 Pounds", "10 lb"), ("2026/4/5", "2026-04-05"),
    ("23 November 2026", "2026-11-23"), ("11/23/2026", "2026-11-23"),
    ("04/05/2026", "04/05/2026"), ("LOT S10", "lot s10"),
])
def test_normalize_value(source, expected):
    assert normalize_value(source) == expected


def test_numeric_ocr_is_opt_in():
    assert normalize_value("1O0 pSi") == "1o0 psi"
    assert normalize_value("1O0 pSi", numeric_ocr=True) == "100 psi"
    assert normalize_value("l.S / O", numeric_ocr=True) == "1.5 / 0"


def test_stamp_candidates_group_fragments_keep_neighbors_and_drop_nested():
    image = np.full((130, 260, 3), 255, np.uint8)
    blue, red = (25, 60, 220), (210, 25, 60)
    cv2.rectangle(image, (20, 20), (109, 71), blue, 2)
    cv2.putText(image, "OK", (48, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.9, blue, 2)
    image[26:58, 118:153] = red
    image[26:58, 155:193] = red
    image[90:108, 15:200] = 50
    boxes = [item.bbox for item in detect_stamps(image)]
    assert boxes == [(19, 19, 111, 73), (118, 26, 193, 58)]


def _png(color):
    stream = BytesIO()
    Image.new("RGB", (10, 10), color).save(stream, format="PNG")
    return stream.getvalue()


def test_image_changes_identify_swaps_resize_replacement_and_unpaired_images():
    with pymupdf.open() as v1, pymupdf.open() as v2:
        old, new = v1.new_page(), v2.new_page()
        boxes = [(x, 30, x + 30, 60) for x in (10, 60, 110, 160, 210, 260)]
        for index, color in enumerate(("red", "blue", "green", "orange", "yellow", "red")):
            old.insert_image(boxes[index], stream=_png(color))
        for index, color in ((0, "blue"), (1, "red"), (3, "purple"), (5, "red")):
            new.insert_image(boxes[index], stream=_png(color))
        new.insert_image((110, 30, 155, 75), stream=_png("green"))
        new.insert_image((320, 30, 350, 60), stream=_png("black"))
        changes = compare_embedded_images(old, new)
        assert sorted(item.kind for item in changes) == ["added", "moved", "moved", "removed", "replaced", "resized"]
        assert {(item.v1.bbox, item.v2.bbox) for item in changes if item.kind == "moved"} == {
            (boxes[0], boxes[1]), (boxes[1], boxes[0]),
        }
        assert next(item for item in changes if item.kind == "removed").v1.bbox == boxes[4]


def test_split_annotations_and_crops(tmp_path):
    path = tmp_path / "review.pdf"
    with pymupdf.open() as pdf:
        page = pdf.new_page(width=300, height=200)
        page.insert_text((40, 60), "Review this value")
        highlight = page.add_highlight_annot(page.search_for("value"))
        highlight.set_info(content="Verify pressure.")
        page.insert_link({"kind": pymupdf.LINK_URI, "from": pymupdf.Rect(120, 120, 160, 130),
                          "uri": "https://example.test"})
        page.set_rotation(90)
        pdf.save(path)
    storage = LocalStorage(tmp_path / "artifacts")
    plain, annotations = split_annotation_layer(path, 1, dpi=72, storage=storage)
    with_annotations = render_page(path, 1, 72, storage=storage)
    assert plain != with_annotations
    assert annotations[0].kind == "Highlight"
    assert annotations[0].content == "Verify pressure."
    assert annotations[1].kind == "Link"
    assert annotations[1].bbox == (120, 120, 160, 130)
    with Image.open(BytesIO(view_page(plain, storage=storage))) as page:
        assert page.size == (200, 300)
    with Image.open(BytesIO(view_crop(plain, (-2, 15, 25, 32), storage=storage))) as crop:
        assert crop.size == (25, 17)
    with pytest.raises(ValueError, match="outside"):
        view_crop(plain, (300, 400, 320, 420), storage=storage)
