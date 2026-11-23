"""Compare PDF path primitives and native text labels in unrotated page points."""
from collections import defaultdict, deque
from dataclasses import dataclass
from math import hypot
from typing import Literal

import pymupdf
from scipy.spatial import cKDTree

from redline.models import BBox


@dataclass(frozen=True)
class DrawingPart:
    kind: str
    geometry: tuple[float, ...]
    style: tuple
    bbox: BBox
    label: str | None = None

    @property
    def center(self) -> tuple[float, float]:
        x0, y0, x1, y1 = self.bbox
        return (x0 + x1) / 2, (y0 + y1) / 2

    @property
    def identity(self) -> tuple:
        return self.kind, self.geometry, self.style, self.label


@dataclass(frozen=True)
class VectorChange:
    kind: Literal["addition", "deletion", "modification"]
    before: DrawingPart | None
    after: DrawingPart | None
    bbox_v2: BBox


def _geometry(item) -> tuple[float, ...]:
    kind, *values = item
    if kind == "l":
        values = sorted(values)
    if kind == "re":
        return tuple(values[0])
    if kind == "qu":
        values = list(values[0])
    return tuple(component for value in values for component in value)


def _bounds(coords: tuple[float, ...], padding: float) -> BBox:
    xs, ys = coords[::2], coords[1::2]
    return min(xs) - padding, min(ys) - padding, max(xs) + padding, max(ys) + padding


def _parts(page: pymupdf.Page) -> list[DrawingPart]:
    if page.first_annot is not None or page.first_widget is not None:
        # PyMuPDF's drawing extractor includes annotation appearance streams.
        with pymupdf.open() as clean:
            clean.insert_pdf(page.parent, from_page=page.number, to_page=page.number, annots=False, widgets=False, links=False)
            return _parts(clean[0])

    parts = []
    for path in page.get_cdrawings():
        style = (
            path["type"], path.get("color"), path.get("fill"), path.get("width"),
            path.get("dashes"), path.get("lineCap"), path.get("lineJoin"), path.get("even_odd"),
            path.get("stroke_opacity"), path.get("fill_opacity"),
        )
        width = path.get("width") or 0
        # Preserve closed filled shapes as a unit so their fill and winding stay associated.
        path_items = [(item[0], _geometry(item)) for item in path["items"] if item[0] in {"l", "c", "re", "qu"}]
        if "f" in path["type"] and len(path_items) > 1:
            types, geometries = zip(*path_items)
            source_box = tuple(path["rect"])
            parts.append(DrawingPart(
                "fill", tuple(component for item in geometries for component in item),
                (*style, types), _bounds(source_box, max(1, width / 2)),
            ))
            continue
        for kind, geometry in path_items:
            parts.append(DrawingPart(kind, geometry, style, _bounds(geometry, max(1, width / 2))))

    text = page.get_text("dict", flags=pymupdf.TEXTFLAGS_DICT & ~pymupdf.TEXT_PRESERVE_IMAGES)
    for block in text["blocks"]:
        for line in block.get("lines", []):
            for span in line["spans"]:
                label = span["text"].strip()
                if label:
                    box = tuple(span["bbox"])
                    style = span["font"], span["size"], span["color"], tuple(line["dir"])
                    parts.append(DrawingPart("label", box, style, box, label))
    return parts


def _distance(a: DrawingPart, b: DrawingPart) -> float:
    return hypot(a.center[0] - b.center[0], a.center[1] - b.center[1])


def _overlap(a: DrawingPart, b: DrawingPart) -> float:
    old, new = pymupdf.Rect(a.bbox), pymupdf.Rect(b.bbox)
    area = max(old.get_area(), new.get_area())
    return (old & new).get_area() / area if area else 0.0


def vector_diff(page_v1: pymupdf.Page, page_v2: pymupdf.Page, *, tolerance: float = 0.25) -> list[VectorChange]:
    """Return coordinate-backed additions, deletions, and nearby primitive or label edits."""
    if not 0 < tolerance <= 2:
        raise ValueError("Vector tolerance must be in (0, 2] points.")
    old, new = _parts(page_v1), _parts(page_v2)
    exact = defaultdict(deque)
    for j, part in enumerate(new):
        exact[part.identity].append(j)
    old_left, new_left = set(range(len(old))), set(range(len(new)))
    for i, part in enumerate(old):
        matches = exact[part.identity]
        if matches:
            old_left.remove(i)
            new_left.remove(matches.popleft())

    js = sorted(new_left)
    candidates = []
    if js:
        tree = cKDTree([new[j].center for j in js])
        for i in sorted(old_left):
            a = old[i]
            for index in tree.query_ball_point(a.center, 28):
                j = js[index]
                b = new[j]
                if a.kind != b.kind:
                    continue
                distance = _distance(a, b)
                nearly_equal = (
                    a.style == b.style and a.label == b.label and len(a.geometry) == len(b.geometry)
                    and max(abs(x - y) for x, y in zip(a.geometry, b.geometry)) <= tolerance
                )
                overlap = _overlap(a, b)
                if nearly_equal or overlap >= 0.15 or (distance <= 6 and a.kind != "label"):
                    candidates.append((0 if nearly_equal else 1, 1 - overlap, distance, i, j))

    changes = []
    for priority, _, _, i, j in sorted(candidates):
        if i not in old_left or j not in new_left:
            continue
        old_left.remove(i)
        new_left.remove(j)
        if priority:
            changes.append(VectorChange("modification", old[i], new[j], new[j].bbox))
    changes.extend(VectorChange("deletion", old[i], None, old[i].bbox) for i in sorted(old_left))
    changes.extend(VectorChange("addition", None, new[j], new[j].bbox) for j in sorted(new_left))
    return sorted(changes, key=lambda item: (item.bbox_v2[1], item.bbox_v2[0], item.kind))
