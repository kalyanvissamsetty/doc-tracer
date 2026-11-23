"""Register drawing scans, then compare bounded, overlapping high-resolution tiles."""
from dataclasses import dataclass
from math import ceil

import cv2
import numpy as np
import pymupdf

from redline.models import BBox


@dataclass(frozen=True)
class Alignment:
    transform: np.ndarray | None
    inliers: int
    quality: float


@dataclass(frozen=True)
class ScanRegion:
    kind: str
    bbox_v1: BBox
    bbox_v2: BBox
    pixels: int


@dataclass(frozen=True)
class ScanComparison:
    alignment: Alignment
    regions: list[ScanRegion]


def _gray(image: np.ndarray) -> np.ndarray:
    if image.dtype != np.uint8 or image.size == 0:
        raise ValueError("Registration requires nonempty uint8 rasters.")
    if image.ndim == 2:
        return image
    if image.ndim == 3 and image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    raise ValueError("Registration requires grayscale or RGB rasters.")


def align_images(before: np.ndarray, after: np.ndarray) -> Alignment:
    """Estimate a V1-to-V2 preview homography with feature matches and RANSAC."""
    old, new = _gray(before), _gray(after)
    if np.array_equal(old, new):
        return Alignment(np.eye(3), 0, 1.0)

    for detector, norm, ratio in (
        (cv2.ORB_create(nfeatures=2800, fastThreshold=12), cv2.NORM_HAMMING, 0.76),
        (cv2.SIFT_create(nfeatures=2200), cv2.NORM_L2, 0.72),
    ):
        first, a = detector.detectAndCompute(old, None)
        second, b = detector.detectAndCompute(new, None)
        if a is None or b is None or min(len(a), len(b)) < 12:
            continue
        matcher = cv2.BFMatcher(norm)
        matches = [m for m, n in matcher.knnMatch(a, b, k=2) if m.distance < ratio * n.distance]
        # Repetitive linework can send several source features to the same destination.
        unique = {}
        for match in sorted(matches, key=lambda candidate: candidate.distance):
            unique.setdefault(match.trainIdx, match)
        if len(unique) < 12:
            continue
        src = np.float32([first[m.queryIdx].pt for m in unique.values()])
        dst = np.float32([second[m.trainIdx].pt for m in unique.values()])
        transform, mask = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
        if transform is None or mask is None:
            continue
        selected = mask.ravel().astype(bool)
        count = int(selected.sum())
        if count < 12 or count / len(selected) < 0.42:
            continue

        h, w = old.shape
        nh, nw = new.shape
        coverage = cv2.contourArea(cv2.convexHull(src[selected])) / (w * h)
        if coverage < 0.025:
            continue
        corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(1, 4, 2)
        mapped = cv2.perspectiveTransform(corners, transform)[0]
        ideal = np.float32([[0, 0], [nw, 0], [nw, nh], [0, nh]])
        if not np.isfinite(mapped).all() or not cv2.isContourConvex(mapped):
            continue
        if np.max(np.abs((mapped - ideal) / [nw, nh])) > 0.24:
            continue
        # Similarity is more stable than a homography when translation, scale, and deskew are sufficient.
        affine, _ = cv2.estimateAffinePartial2D(src[selected], dst[selected], method=cv2.RANSAC)
        if affine is not None:
            projected = cv2.transform(src[selected][None], affine)[0]
            residual = np.linalg.norm(projected - dst[selected], axis=1)
            if float(np.median(residual)) < 1.1 and float(np.quantile(residual, 0.9)) < 2.5:
                transform = np.vstack([affine, [0, 0, 1]])
        return Alignment(transform, count, min(1.0, 0.55 + 0.3 * count / len(selected) + 0.3 * coverage))
    return Alignment(None, 0, 0.0)


def _scale(x: float, y: float) -> np.ndarray:
    return np.diag([x, y, 1.0])


def _translate(x: float, y: float) -> np.ndarray:
    return np.array([[1.0, 0, x], [0, 1.0, y], [0, 0, 1.0]])


def _project(box: pymupdf.Rect, transform: np.ndarray) -> pymupdf.Rect:
    points = np.array([[[box.x0, box.y0], [box.x1, box.y0], [box.x1, box.y1], [box.x0, box.y1]]])
    result = cv2.perspectiveTransform(points, transform)[0]
    return pymupdf.Rect(*result.min(axis=0), *result.max(axis=0))


def _raster(page, zoom, box):
    pix = page.get_pixmap(
        matrix=pymupdf.Matrix(zoom, zoom), clip=box, colorspace=pymupdf.csGRAY, annots=False,
    )
    return np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width), pix.x, pix.y


def _tile_regions(
    before, after, transform, *, zoom, tile_size, overlap, min_area,
):
    width, height = ceil(after.rect.width * zoom), ceil(after.rect.height * zoom)
    reverse = np.linalg.inv(transform)
    margin = np.ones((5, 5), np.uint8)
    for row, y in enumerate(range(0, height, tile_size)):
        for column, x in enumerate(range(0, width, tile_size)):
            x1, y1 = min(x + tile_size, width), min(y + tile_size, height)
            dest_box = pymupdf.Rect(
                max(0, x - overlap) / zoom, max(0, y - overlap) / zoom,
                min(width, x1 + overlap) / zoom, min(height, y1 + overlap) / zoom,
            )
            src_box = _project(dest_box, reverse)
            source_box = (src_box + (-4, -4, 4, 4)) & before.rect
            if source_box.is_empty:
                continue
            prior, sx, sy = _raster(before, zoom, source_box)
            revised, dx, dy = _raster(after, zoom, dest_box)
            local = _translate(-dx, -dy) @ _scale(zoom, zoom) @ transform @ _scale(1 / zoom, 1 / zoom) @ _translate(sx, sy)
            shape = revised.shape[::-1]
            aligned = cv2.warpPerspective(prior, local, shape, borderValue=255)
            valid = cv2.warpPerspective(np.full(prior.shape, 255, np.uint8), local, shape)
            valid = cv2.erode(valid, margin, borderType=cv2.BORDER_CONSTANT, borderValue=0)
            guard = 3
            if dx < guard:
                valid[:, :guard - dx] = 0
            if dy < guard:
                valid[:guard - dy] = 0
            if dx + revised.shape[1] > width - guard:
                valid[:, max(0, width - guard - dx):] = 0
            if dy + revised.shape[0] > height - guard:
                valid[max(0, height - guard - dy):, :] = 0
            # A 2-pixel neighborhood absorbs small residual scan misalignment around existing ink.
            new_ink = (cv2.erode(aligned, margin).astype(np.int16) - revised > 36) & (valid == 255)
            old_ink = (cv2.erode(revised, margin).astype(np.int16) - aligned > 36) & (valid == 255)
            mask = cv2.morphologyEx((new_ink | old_ink).astype(np.uint8), cv2.MORPH_CLOSE, margin)
            _, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
            for label, (lx, ly, w, h, _) in enumerate(stats[1:], 1):
                owned = (max(x, lx + dx), max(y, ly + dy), min(x1, lx + w + dx), min(y1, ly + h + dy))
                ox, oy, ex, ey = owned
                if ex <= ox or ey <= oy:
                    continue
                sl = np.s_[oy - dy:ey - dy, ox - dx:ex - dx]
                belongs = labels[sl] == label
                additions = int(np.count_nonzero(new_ink[sl] & belongs))
                deletions = int(np.count_nonzero(old_ink[sl] & belongs))
                touches_seam = (ox == x and x > 0 or oy == y and y > 0 or
                                ex == x1 and x1 < width or ey == y1 and y1 < height)
                if additions + deletions >= (max(1, min_area // 3) if touches_seam else min_area):
                    yield (row, column), owned, additions, deletions


def _join_seams(tiles):
    """Union fragments across shared tile seams, without merging adjacent features in one tile."""
    by_tile = {}
    parent = list(range(len(tiles)))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, ((row, col), (x0, y0, x1, y1), _, _) in enumerate(tiles):
        for dr, dc in ((0, -1), (-1, 0)):
            for j in by_tile.get((row + dr, col + dc), []):
                X0, Y0, X1, Y1 = tiles[j][1]
                if (dc and abs(x0 - X1) <= 4 and min(y1, Y1) > max(y0, Y0)) or (
                    dr and abs(y0 - Y1) <= 4 and min(x1, X1) > max(x0, X0)
                ):
                    parent[root(i)] = root(j)
        by_tile.setdefault((row, col), []).append(i)
    components = {}
    for i, (_, (x0, y0, x1, y1), added, removed) in enumerate(tiles):
        key = root(i)
        if key in components:
            a, b, c, d, p, q = components[key]
            components[key] = min(a, x0), min(b, y0), max(c, x1), max(d, y1), p + added, q + removed
        else:
            components[key] = x0, y0, x1, y1, added, removed
    return components.values()


def compare_scanned_drawing(
    before: pymupdf.Page, after: pymupdf.Page, preview_v1: np.ndarray, preview_v2: np.ndarray, *,
    dpi: float = 175, tile_size: int = 960, overlap: int = 32, min_area: int = 10,
) -> ScanComparison:
    """Return localized point boxes in both versions, or no registration on weak evidence."""
    if not 40 <= dpi <= 300 or not 64 <= tile_size <= 2048 or not 8 <= overlap < tile_size / 2 or min_area < 1:
        raise ValueError("Invalid drawing tile or DPI configuration.")
    alignment = align_images(preview_v1, preview_v2)
    if alignment.transform is None:
        return ScanComparison(alignment, [])
    bh, bw = preview_v1.shape[:2]
    ah, aw = preview_v2.shape[:2]
    pdf_transform = (
        _scale(after.rect.width / aw, after.rect.height / ah) @ alignment.transform
        @ _scale(bw / before.rect.width, bh / before.rect.height)
    )
    inverse = np.linalg.inv(pdf_transform)
    zoom = min(dpi / 72, 12000 / max(after.rect.width, after.rect.height, before.rect.width, before.rect.height))
    tiles = list(_tile_regions(
        before, after, pdf_transform, zoom=zoom, tile_size=tile_size, overlap=overlap, min_area=min_area,
    ))
    regions = []
    for x0, y0, x1, y1, added, removed in _join_seams(tiles):
        if added + removed < min_area:
            continue
        dest = pymupdf.Rect(x0 / zoom, y0 / zoom, x1 / zoom, y1 / zoom) & after.rect
        source = _project(dest, inverse) & before.rect
        if source.is_empty or dest.is_empty:
            continue
        kind = "modification" if min(added, removed) > 0.18 * max(added, removed) else (
            "addition" if added >= removed else "deletion"
        )
        regions.append(ScanRegion(
            kind, tuple(source * before.derotation_matrix), tuple(dest * after.derotation_matrix), added + removed,
        ))
    return ScanComparison(alignment, regions)
