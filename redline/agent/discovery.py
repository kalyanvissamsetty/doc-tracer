"""Inventory changes and retain bounded render references for later visual interpretation."""
from collections import Counter, defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from PIL import Image
import cv2
import numpy as np
import pymupdf

from redline.models import Annotation, BBox, Change, PageChanges
from redline.storage import Storage
from redline.toolkit import (
    compare_embedded_images, detect_table_grid, extract_table, match_rows,
    compare_scanned_drawing, normalize_value, pair_pages, split_annotation_layer, vector_diff, view_page,
)
from redline.toolkit.visual import difference_mask


@dataclass(frozen=True)
class CropRef:
    image_id: str
    bbox: BBox


@dataclass(frozen=True)
class PdfCropRef:
    path: Path
    page: int
    bbox: BBox


@dataclass
class Discovery:
    pages: list[PageChanges]
    crops: dict[int, tuple[CropRef, CropRef] | tuple[PdfCropRef, PdfCropRef]]
    source_boxes: dict[int, BBox | None]
    drawing_diagnostics: list[dict]


@dataclass(frozen=True)
class PageImage:
    image_id: str
    pixels: np.ndarray
    page: pymupdf.Page

    def pdf_box(self, box: BBox) -> BBox:
        height, width = self.pixels.shape[:2]
        x0, y0, x1, y1 = box
        sx, sy = self.page.rect.width / width, self.page.rect.height / height
        return tuple(pymupdf.Rect(x0 * sx, y0 * sy, x1 * sx, y1 * sy) * self.page.derotation_matrix)

    def crop(self, box: BBox) -> CropRef:
        height, width = self.pixels.shape[:2]
        displayed = pymupdf.Rect(box) * self.page.rotation_matrix
        x0, y0, x1, y1 = displayed
        sx, sy = width / self.page.rect.width, height / self.page.rect.height
        return CropRef(self.image_id, (x0 * sx - 8, y0 * sy - 8, x1 * sx + 8, y1 * sy + 8))


def page_dpi(page: pymupdf.Page, maximum_edge: int = 1500) -> float:
    return min(110, maximum_edge * 72 / max(page.rect.width, page.rect.height))


def _pixels(content: bytes) -> np.ndarray:
    from io import BytesIO

    with Image.open(BytesIO(content)) as image:
        return np.array(image.convert("RGB"))


def _bounds(page: pymupdf.Page) -> pymupdf.Rect:
    return page.rect * page.derotation_matrix


def _map_box(box: BBox, source: pymupdf.Page, target: pymupdf.Page) -> BBox:
    displayed = pymupdf.Rect(box) * source.rotation_matrix
    sx, sy = target.rect.width / source.rect.width, target.rect.height / source.rect.height
    x0, y0, x1, y1 = displayed
    return tuple(pymupdf.Rect(x0 * sx, y0 * sy, x1 * sx, y1 * sy) * target.derotation_matrix)


def _covers(box: BBox, evidence: BBox) -> bool:
    region = pymupdf.Rect(box)
    return (region & pymupdf.Rect(evidence)).get_area() >= 0.6 * region.get_area()


def _widget_changes(before: list[Annotation], after: list[Annotation], old, new):
    groups = defaultdict(lambda: ([], []))
    for index, marks in enumerate((before, after)):
        for mark in marks:
            if mark.kind.startswith("Widget: ") and mark.field_name:
                groups[mark.kind, mark.field_name][index].append(mark)

    for (_, name), (first, second) in groups.items():
        if len(first) <= 1 and len(second) <= 1:
            pairs = [(first[0] if first else None, second[0] if second else None)]
        else:
            positions = [
                {tuple(round(v) for v in mark.bbox): mark for mark in group} for group in (first, second)
            ]
            keys = sorted(positions[0].keys() | positions[1].keys())
            pairs = [(positions[0].get(key), positions[1].get(key)) for key in keys]
        for a, b in pairs:
            old_box = _map_box(a.bbox, old, new) if a else None
            box = b.bbox if b else old_box
            if (pymupdf.Rect(box) & _bounds(new)).is_empty:
                continue
            moved = bool(a and b and max(abs(x - y) for x, y in zip(old_box, box)) > 1)
            same_value = normalize_value(a.content or "") == normalize_value(b.content or "") if a and b else False
            if same_value and not moved:
                continue
            before_value = (a.content or None) if a else None
            after_value = (b.content or None) if b else None
            if moved and same_value:
                kind = "modification"
            else:
                kind = "addition" if not a or (not before_value and after_value) else (
                    "deletion" if not b or (before_value and not after_value) else "modification"
                )
            yield dict(
                type=kind, category="value", field=name, v1_value=before_value, v2_value=after_value,
                bbox_v2=box, bbox_v1=a.bbox if a else None, evidence="PDF form widget states or placements differ for this field.",
                confidence=0.96,
            )


def _annotation_changes(before: list[Annotation], after: list[Annotation], same_content: bool):
    def identity(mark: Annotation):
        return mark.kind, tuple(round(value, 1) for value in mark.bbox), mark.content

    # Widgets are content fields, and URI changes require link metadata.
    groups = [[mark for mark in marks if not mark.kind.startswith("Widget: ") and mark.kind != "Link"]
              for marks in (before, after)]
    for index, (marks, opposite) in enumerate((groups, tuple(reversed(groups)))):
        remaining = Counter(map(identity, opposite))
        for mark in marks:
            key = identity(mark)
            if remaining[key]:
                remaining[key] -= 1
                continue
            added = index == 1
            yield mark, {
                "type": "addition" if added else "deletion",
                "category": "stamp" if mark.kind == "Stamp" else "review_markup",
                "field": mark.kind,
                "v1_value": None if added else mark.content,
                "v2_value": mark.content if added else None,
                "review_request_status": (
                    "content_unchanged" if same_content else "unverified"
                ) if not added and mark.kind != "Stamp" else None,
                "evidence": f"PDF {mark.kind} annotation exists only in {'V2' if added else 'V1'}.",
                "confidence": 0.95,
            }


def _structural(old: PageImage, new: PageImage, layout_pages):
    """Return table changes and the V2 table footprint when extraction is reliable."""
    grids = [detect_table_grid(view.pixels) for view in (old, new)]
    if any(grid is None for grid in grids):
        return [], None
    tables = []
    used_ocr = False
    for view, grid, fetch_layout in zip((old, new), grids, layout_pages):
        layout = None
        if not view.page.get_text().strip():
            layout = fetch_layout()
            if layout is None:
                return [], None
            used_ocr = True
        try:
            header, *rows = extract_table(view.page, grid, layout_page=layout)
        except (KeyError, ValueError):
            return [], None
        keys = [normalize_value(cell.value) for cell in header]
        if "item number" not in keys or len(set(keys)) != len(keys) or not all(keys):
            return [], None
        tables.append((grid, keys, rows, [dict(zip(keys, (cell.value for cell in row))) for row in rows]))
    (_, keys_old, old_cells, old_rows), (grid, keys, new_cells, new_rows) = tables
    if set(keys_old) != set(keys):
        return [], None
    for rows in (old_rows, new_rows):
        identities = [normalize_value(row["item number"]) for row in rows]
        if not all(identities) or len(set(identities)) != len(identities):
            return [], None

    changes = []
    confidence = 0.79 if used_ocr else 0.97
    old_columns = {key: col for col, key in enumerate(keys_old)}
    for match in match_rows(old_rows, new_rows):
        i, j = match.v1_row, match.v2_row
        if i is None or j is None:
            added = i is None
            row = new_rows[j] if added else old_rows[i]
            cells = new_cells[j] if added else old_cells[i]
            first, last = cells[0].bbox, cells[-1].bbox
            box = tuple(pymupdf.Rect(first) | pymupdf.Rect(last))
            changes.append(dict(
                type="addition" if added else "deletion", category="row",
                field=f"Item {row['item number']}",
                v1_value=None if added else " | ".join(row.values()),
                v2_value=" | ".join(row.values()) if added else None,
                bbox_v2=box if added else _map_box(box, old.page, new.page), bbox_v1=None if added else box,
                evidence="Extracted table row has no matching key in the other version.", confidence=confidence,
            ))
            continue
        a, b = old_rows[i], new_rows[j]
        for column, key in enumerate(keys):
            if normalize_value(a[key]) == normalize_value(b[key]):
                continue
            changes.append(dict(
                type="addition" if not a[key] else "deletion" if not b[key] else "modification",
                category="value", field=f"Item {b['item number']}, {key}",
                v1_value=a[key] or None, v2_value=b[key] or None, bbox_v2=new_cells[j][column].bbox,
                bbox_v1=old_cells[i][old_columns[key]].bbox,
                evidence="Different extracted values in a key-matched table cell.", confidence=confidence,
            ))
    footprint = new.pdf_box((grid.x_rules[0], grid.y_rules[0], grid.x_rules[-1], grid.y_rules[-1]))
    return changes, footprint


def _raster_candidates(old: PageImage, new: PageImage, covered: list[BBox]):
    height, width = new.pixels.shape[:2]
    scaled = old.pixels
    if scaled.shape != new.pixels.shape:
        scaled = cv2.resize(old.pixels, (width, height), interpolation=cv2.INTER_AREA)
    mask = difference_mask(scaled, new.pixels, threshold=30)
    _, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    for label, (x, y, w, h, area) in enumerate(stats[1:], 1):
        span = (x, y, x + w, y + h)
        if area < 4 or any(_covers(new.pdf_box(span), box) for box in covered):
            patch = np.s_[y:y + h, x:x + w]
            mask[patch][labels[patch] == label] = 0
    merged = cv2.dilate(mask, np.ones((3, 11), np.uint8))
    count, labels, _, _ = cv2.connectedComponentsWithStats(merged)
    ys, xs = np.nonzero(mask)
    memberships = labels[ys, xs]
    min_x, min_y = np.full(count, width), np.full(count, height)
    max_x, max_y = np.zeros(count, dtype=int), np.zeros(count, dtype=int)
    np.minimum.at(min_x, memberships, xs)
    np.minimum.at(min_y, memberships, ys)
    np.maximum.at(max_x, memberships, xs + 1)
    np.maximum.at(max_y, memberships, ys + 1)
    grid_extent = None
    grid_checked = False
    for i in range(1, count):
        x0, y0, x1, y1 = min_x[i], min_y[i], max_x[i], max_y[i]
        box = new.pdf_box((x0, y0, x1, y1))
        old_box = _map_box(box, new.page, old.page)
        # A connected table grid or scan shift can span most of the sheet. Pixel
        # connectivity at that scale is insufficient to mark a value change.
        area_fraction = (x1 - x0) * (y1 - y0) / (width * height)
        if .08 <= area_fraction < .28 and not grid_checked:
            grid_checked = True
            grid = detect_table_grid(new.pixels)
            if grid and len(grid.x_rules) >= 3 and len(grid.y_rules) >= 4:
                grid_extent = pymupdf.Rect(grid.x_rules[0], grid.y_rules[0], grid.x_rules[-1], grid.y_rules[-1])
        spans_table = grid_extent is not None and (
            (pymupdf.Rect(x0, y0, x1, y1) & grid_extent).get_area() >= .9 * grid_extent.get_area()
        )
        if area_fraction >= .28 or spans_table:
            yield dict(
                type="modification", category="value", field="Unlocalized raster difference",
                bbox_v1=old_box, bbox_v2=box, confidence=0.4,
                evidence="Connected pixel differences span a substantial part of the page.",
                markup_withheld_reason="This difference is too broad to locate a changed value. Review the source pages or use cell-level extraction.",
            ), None
            continue
        # Empty white regions provide direct evidence of additions or deletions.
        a = scaled[y0:y1, x0:x1]
        b = new.pixels[y0:y1, x0:x1]
        blank_a = bool(np.all(a >= 245))
        blank_b = bool(np.all(b >= 245))
        kind = "addition" if blank_a else "deletion" if blank_b else "modification"
        yield dict(
            type=kind, category="value", field="Changed raster region", bbox_v1=old_box, bbox_v2=box,
            evidence="Nontrivial pixel differences in annotation-free page renders.",
            confidence=0.75 if (blank_a or blank_b) and old.pixels.shape == new.pixels.shape else 0.55,
        ), (old.crop(old_box), new.crop(box))


def _drawing_route(old: pymupdf.Page, new: pymupdf.Page, document_type: str) -> str | None:
    pages = old, new
    explicit = normalize_value(document_type) in {"as-built", "as built", "drawing", "cad"}
    large = min(max(page.rect.width, page.rect.height) for page in pages) >= 1000
    if not explicit and not (large and normalize_value(document_type) == "auto"):
        return None
    scans = [
        any((pymupdf.Rect(image["bbox"]) & _bounds(page)).get_area() >= 0.8 * page.rect.get_area()
            for image in page.get_image_info())
        for page in pages
    ]
    if (explicit and any(scans)) or all(scans):
        return "scan"
    if not any(scans) and any(len(page.get_cdrawings()) >= (1 if explicit else 8) for page in pages):
        return "vector"
    return None


def discover_changes(
    v1: Path, v2: Path, storage: Storage, *,
    layout: Callable[[Path, int], dict | None] | None = None, document_type: str = "auto",
) -> Discovery:
    pages = []
    crops = {}
    source_boxes = {}
    diagnostics = []
    next_id = 1
    with pymupdf.open(v1) as before, pymupdf.open(v2) as after:
        for pair in pair_pages(v1, v2):
            i, j = pair.v1_page, pair.v2_page
            group = PageChanges(v1_page=i, v2_page=j, method="identical")
            pages.append(group)

            def add(fields, crop_refs=None):
                nonlocal next_id
                source_boxes[next_id] = fields.pop("bbox_v1", None)
                group.changes.append(Change(id=next_id, **fields))
                if crop_refs:
                    crops[next_id] = crop_refs
                next_id += 1

            if i is None or j is None:
                group.method = "structural"
                add(dict(
                    type="addition" if i is None else "deletion", category="page", field=f"Page {i or j}",
                    evidence="Page pairing left this page unmatched.", confidence=0.92,
                ))
                continue

            page_old, page_new = before[i - 1], after[j - 1]
            id_old, annots_old = split_annotation_layer(v1, i, dpi=page_dpi(page_old), storage=storage)
            id_new, annots_new = split_annotation_layer(v2, j, dpi=page_dpi(page_new), storage=storage)
            route = _drawing_route(page_old, page_new, document_type)
            same_content = id_old == id_new
            widgets = list(_widget_changes(annots_old, annots_new, page_old, page_new))
            for fields in widgets:
                group.method = "structural"
                add(fields)
            for mark, fields in _annotation_changes(annots_old, annots_new, same_content and not widgets):
                box = mark.bbox if fields["type"] == "addition" else _map_box(mark.bbox, page_old, page_new)
                if (pymupdf.Rect(box) & _bounds(page_new)).is_empty:
                    continue
                group.method = "structural"
                add({**fields, "bbox_v1": mark.bbox if fields["type"] == "deletion" else None, "bbox_v2": box})
            # Identical thumbnails can hide fine line edits on drawing-sized sheets.
            if same_content and route is None:
                continue

            old = PageImage(id_old, _pixels(view_page(id_old, storage=storage)), page_old)
            new = PageImage(id_new, _pixels(view_page(id_new, storage=storage)), page_new)
            if route == "scan":
                scan = compare_scanned_drawing(page_old, page_new, old.pixels, new.pixels)
                registered = scan.alignment.transform is not None
                diagnostics.append({
                    "page": j, "method": "tiled_scan" if registered else "raster_fallback",
                    "inliers": scan.alignment.inliers, "alignment_quality": scan.alignment.quality,
                    "regions": len(scan.regions),
                })
                if registered:
                    if scan.regions:
                        group.method = "pixel_diff"
                    for region in scan.regions:
                        add(dict(
                            type=region.kind, category="value", field="Changed scan linework",
                            bbox_v1=region.bbox_v1, bbox_v2=region.bbox_v2,
                            evidence=f"{region.pixels} changed ink pixels in a registered drawing tile.",
                            confidence=min(0.69, scan.alignment.quality),
                        ), (PdfCropRef(v1, i, region.bbox_v1), PdfCropRef(v2, j, region.bbox_v2)))
                    continue

            covered = []
            for image in compare_embedded_images(page_old, page_new):
                source = _map_box(image.v1.bbox, page_old, page_new) if image.v1 else None
                target = image.v2.bbox if image.v2 else None
                box = target or source
                if (pymupdf.Rect(box) & _bounds(page_new)).is_empty:
                    continue
                if image.kind in {"replaced", "added", "removed"} and pymupdf.Rect(box).get_area() >= 0.8 * _bounds(page_new).get_area():
                    # A full-page scan is a source of regional changes, rather than one whole-page image edit.
                    continue
                group.method = "image_compare"
                add(dict(
                    type={"added": "addition", "removed": "deletion"}.get(image.kind, "modification"),
                    category="image", field=f"Embedded image {image.kind}", bbox_v2=box,
                    bbox_v1=image.v1.bbox if image.v1 else None,
                    evidence=f"Embedded image digest and placement comparison reported {image.kind}.",
                    confidence=0.93 if image.kind != "replaced" else 0.8,
                ))
                covered.extend(b for b in (source, target) if b)
            if route == "vector":
                edits = vector_diff(page_old, page_new)
                diagnostics.append({"page": j, "method": "vector", "regions": len(edits)})
                if edits:
                    group.method = "vector"
                for edit in edits:
                    bbox = _map_box(edit.bbox_v2, page_old, page_new) if edit.after is None else edit.bbox_v2
                    if (pymupdf.Rect(bbox) & _bounds(page_new)).is_empty:
                        continue
                    label = (edit.after or edit.before).kind == "label"
                    add(dict(
                        type=edit.kind, category="value", field="Drawing label" if label else "Drawing primitive",
                        v1_value=edit.before.label if edit.before else None,
                        v2_value=edit.after.label if edit.after else None,
                        bbox_v1=edit.before.bbox if edit.before else None, bbox_v2=bbox,
                        evidence="Native PDF text or path geometry differs between drawing versions.",
                        confidence=0.94,
                    ))
                if edits or same_content or group.method == "image_compare":
                    continue
            changes, table = ([], None) if route else _structural(
                old, new, (lambda: layout(v1, i) if layout else None, lambda: layout(v2, j) if layout else None),
            )
            if table is not None:
                group.method = "structural"
                covered.append(table)
                for fields in changes:
                    add(fields)
            for fields, refs in _raster_candidates(old, new, covered):
                if group.method == "identical":
                    group.method = "pixel_diff"
                if route == "scan":
                    fields["confidence"] = min(fields["confidence"], 0.48)
                    fields["evidence"] += " Scan registration could not be established."
                add(fields, refs)
    return Discovery(pages, crops, source_boxes, diagnostics)
