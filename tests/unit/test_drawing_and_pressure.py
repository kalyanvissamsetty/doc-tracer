from decimal import Decimal
from pathlib import Path

import cv2
import numpy as np
import pymupdf
import pytest
import yaml

from redline.agent.discovery import discover_changes
from redline.comparison import ComparisonOptions, compare
from redline.config import ConfigurationError
from redline.storage import LocalStorage
from redline.toolkit import align_images, compare_scanned_drawing, vector_diff
from redline.tracing import JsonlTracer
from redline.validation.pressure import PressureTestValidator, load_pressure_rules


def _draw_sheet(page, *, revised):
    for j in range(10):
        y = 70 + 35 * j
        page.draw_line((100, y), (530, y), width=2)
    page.draw_line((195, 480), (470, 480), width=2, color=(0, 0, 1) if revised else (0, 0, 0))
    page.insert_text((195, 520), "Valve V-202" if revised else "Valve V-201", fontsize=17)
    if revised:
        page.draw_rect((700, 430, 810, 490), width=2)
    else:
        page.draw_rect((680, 135, 740, 182), width=2)
        page.add_rect_annot((550, 500, 590, 533))
    page.insert_text((100, 58), "Sheet AB-7", fontsize=19)


def test_vector_paths_labels_styles_and_annotation_isolation(tmp_path):
    paths = [tmp_path / "v1.pdf", tmp_path / "v2.pdf"]
    for revised, path in zip((False, True), paths):
        with pymupdf.open() as doc:
            _draw_sheet(doc.new_page(width=1200, height=800), revised=revised)
            doc.save(path)
    with pymupdf.open(paths[0]) as old, pymupdf.open(paths[1]) as new:
        changes = vector_diff(old[0], new[0])
    assert sorted(delta.kind for delta in changes) == ["addition", "deletion", "modification", "modification"]
    labels = [(delta.before.label, delta.after.label) for delta in changes if delta.before and delta.before.label]
    assert labels == [("Valve V-201", "Valve V-202")]

    result = compare(
        *paths, options=ComparisonOptions(60, Decimal(5), document_type="as-built"),
        root=tmp_path / "artifacts", tracer=JsonlTracer(tmp_path / "run.jsonl"),
    )
    with pymupdf.open(result.pdf) as marked:
        assert len(list(marked[0].annots())) == 5
    assert '"method": "vector"' in result.changes.read_text()
    assert '"category": "review_markup"' in result.changes.read_text()


def test_vector_ignores_small_geometry_rounding():
    with pymupdf.open() as old, pymupdf.open() as new:
        a, b = old.new_page(), new.new_page()
        a.draw_line((40, 40), (110, 40))
        b.draw_line((110.08, 40.09), (40.1, 40.09))
        assert vector_diff(a, b, tolerance=0.25) == []


def _scan_fixture(root):
    rng = np.random.default_rng(521)
    base = np.full((850, 1280), 250, np.uint8)
    cv2.rectangle(base, (23, 24), (1256, 826), 30, 2)
    for section, (sx, sy) in enumerate(((50, 55), (450, 55), (850, 55), (50, 475), (450, 475), (850, 475))):
        cv2.putText(base, f"{chr(65 + section)} - 10{section}", (sx + 8, sy + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 30, 2)
        for _ in range(10):
            x, y = rng.integers([sx + 5, sy + 56], [sx + 350, sy + 295])
            cv2.rectangle(base, (x, y), (x + int(rng.integers(12, 36)), y + int(rng.integers(12, 28))), 35, 2)
            cv2.line(base, (x, y), (x + 45, y + 30), 40, 2)
    cv2.rectangle(base, (120, 395), (180, 430), 35, -1)
    revised = base.copy()
    revised[393:433, 118:183] = 250
    cv2.rectangle(revised, (370, 387), (530, 412), 32, -1)
    transform = cv2.getRotationMatrix2D((640, 425), 1.6, 1.0)
    transform[:, 2] += [18, -11]
    warped = cv2.warpAffine(revised, transform, base.shape[::-1], borderValue=250)
    for name, scan in (("v1.pdf", base), ("v2.pdf", warped)):
        with pymupdf.open() as doc:
            page = doc.new_page(width=1280, height=850)
            page.insert_image(page.rect, stream=cv2.imencode(".png", scan)[1].tobytes())
            doc.save(root / name)
    return base, warped, transform


def test_scan_registration_combines_changes_across_tiles(tmp_path):
    base, warped, expected = _scan_fixture(tmp_path)
    alignment = align_images(base, warped)
    assert alignment.inliers > 100
    assert alignment.transform[:2] == pytest.approx(expected, abs=0.6)
    paths = tmp_path / "v1.pdf", tmp_path / "v2.pdf"
    with pymupdf.open(paths[0]) as old, pymupdf.open(paths[1]) as new:
        result = compare_scanned_drawing(old[0], new[0], base, warped, dpi=126, tile_size=256)
    assert sorted(region.kind for region in result.regions) == ["addition", "deletion"]
    added = next(region for region in result.regions if region.kind == "addition")
    assert added.bbox_v2 == pytest.approx((387, 379, 549, 410), abs=3)
    assert added.bbox_v1 == pytest.approx((369, 381, 531, 418), abs=5)

    discovered = discover_changes(*paths, LocalStorage(tmp_path / "store"))
    assert discovered.drawing_diagnostics[0]["method"] == "tiled_scan"
    assert len(discovered.pages[0].changes) == 2
    assert all(c.needs_review for c in discovered.pages[0].changes)
    assert set(discovered.crops) == {change.id for change in discovered.pages[0].changes}


def test_alignment_refuses_insufficient_features():
    old = np.full((160, 200), 255, np.uint8)
    new = old.copy()
    new[35:60, 35:60] = 0
    assert align_images(old, new).transform is None
    assert np.array_equal(align_images(old, old).transform, np.eye(3))


def _pressure_pdf(path: Path, *, bad: bool, attach: bool):
    with pymupdf.open() as doc:
        page = doc.new_page()
        y = 55
        records = (
            "Pressure test report",
            "Test pressure: 180 psi" if bad else "Test pressure: 120 psi",
            "Minimum allowable pressure: 100 psi",
            "Maximum allowable pressure: 1 MPa",
            "Hold time: 120 seconds" if bad else "Hold time: 00:35",
            "Test medium: " if bad else "Test medium: Water",
            "Test Chart Required = Yes",
            "Test start date: 2026-11-20",
            "Test end date: 2026-11-19" if bad else "Test end date: 2026-11-20",
            "Signature date: 2026-11-21",
            "Signature: Unsigned" if bad else "Signature: Alex Test",
            "Employee ID: N/A" if bad else "Employee ID: T-42",
        )
        for text in records:
            page.insert_text((45, y), text)
            y += 22
        if attach:
            doc.embfile_add("pressure_test_chart.csv", b"time,pressure\n0,120\n60,120\n")
        doc.save(path)


def test_pressure_validation_uses_configured_limits_without_regulatory_defaults(tmp_path):
    defaults = load_pressure_rules()
    assert defaults.minimum_duration is None
    config = tmp_path / "rules.yaml"
    config.write_text(yaml.safe_dump({
        **defaults.model_dump(), "minimum_pressure": "100 psi", "maximum_pressure": "1 MPa",
        "minimum_duration": "30 minutes",
    }))
    validator = PressureTestValidator(load_pressure_rules(config))
    e1, e2 = tmp_path / "pass.pdf", tmp_path / "fail.pdf"
    _pressure_pdf(e1, bad=False, attach=True)
    _pressure_pdf(e2, bad=True, attach=False)
    report = validator.validate(e1)
    assert len(report.findings) == 7
    assert all(finding.status == "pass" for finding in report.findings)
    failed = validator.validate(e2)
    assert all(finding.status == "fail" for finding in failed.findings)
    default_findings = {f.rule: f.status for f in PressureTestValidator(defaults).validate(e1).findings}
    assert default_findings["pressure_range"] == default_findings["duration"] == "review"
    with pytest.raises(ConfigurationError):
        config.write_text(config.read_text().replace("30 minutes", "bad-duration"))
        load_pressure_rules(config)


def test_unreadable_pressure_report_needs_review(tmp_path):
    file = tmp_path / "scan.pdf"
    with pymupdf.open() as doc:
        doc.new_page()
        doc.save(file)
    validator = PressureTestValidator(load_pressure_rules())
    assert validator.validate(file) is None
    assert all(item.status == "review" for item in validator.validate(file, explicit=True).findings)
