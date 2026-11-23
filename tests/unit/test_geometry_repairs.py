from decimal import Decimal
import json

import pymupdf
import pytest

from redline.agent.budget import Budget
from redline.agent.tools import JobTools, ToolError
from redline.comparison import ComparisonOptions
from redline.jobs import JobService
from redline.models import Change, ChangeLog, PageChanges
from redline.review import render_edited_page
from redline.toolkit.geometry import snap_to_detected_cell
from redline.toolkit.markup import TagPlanner, write_markup
from redline.tracing import JsonlTracer


def make_table(path, values=("2", "4"), rotation=0):
    with pymupdf.open() as pdf:
        page = pdf.new_page(width=240, height=225)
        page.insert_text((24, 42), "BOM review sample")
        for x in (24, 108, 216):
            page.draw_line((x, 70), (x, 184))
        for y in (70, 108, 146, 184):
            page.draw_line((24, y), (216, y))
        for index, (label, value) in enumerate(zip(("Flange", "Bolt"), values)):
            y = 135 + 38 * index
            page.insert_text((36, y), label)
            page.insert_text((130, y), value)
        page.set_rotation(rotation)
        pdf.save(path)
    return path


@pytest.mark.parametrize("rotation", [0, 90, 180, 270])
def test_snap_uses_unrotated_cell_coordinates_and_rejects_cross_cell_evidence(tmp_path, rotation):
    path = make_table(tmp_path / "v2.pdf", rotation=rotation)
    change = Change(
        id=1, type="modification", category="value", field="Quantity", bbox_v2=(129, 123, 144, 137),
        evidence="Visible digit", confidence=0.85,
    )
    edit = snap_to_detected_cell(path, 1, change, actor="agent")
    assert edit.before == change.bbox_v2
    assert edit.after == pytest.approx((108, 108, 216, 146), abs=.8)
    change.bbox_v2 = (100, 122, 140, 136)
    with pytest.raises(ValueError, match="does not fit"):
        snap_to_detected_cell(path, 1, change, actor="agent")


def test_tag_placement_keeps_clear_of_previous_row_and_adjacent_tags(tmp_path):
    with pymupdf.open() as pdf:
        page = pdf.new_page(width=220, height=180)
        page.insert_text((53, 78), "PREVIOUS ROW", fontsize=12)
        page.insert_text((53, 105), "NEW VALUE", fontsize=12)
        page.insert_text((53, 131), "NEXT VALUE", fontsize=12)
        boxes = [pymupdf.Rect(49, 80, 150, 107), pymupdf.Rect(49, 107, 150, 134)]
        planner = TagPlanner(page, boxes)
        assert planner.place(0, 6) is not None
        assert planner.place(1, 6) is not None
        assert (planner.occupied[0] & planner.occupied[1]).is_empty
        for footprint in planner.occupied:
            for word in page.get_text("words"):
                # Glyphs stay clear, including text in the preceding row.
                text = pymupdf.Rect(*word[:4])
                assert (footprint & text).is_empty
        source = tmp_path / "source.pdf"
        pdf.save(source)

    original = source.read_bytes()
    change = Change(id=1, type="modification", category="value", field="Value", bbox_v2=(49, 80, 150, 107),
                    evidence="Changed value", confidence=.9)
    log = ChangeLog(job_id="test", v1="old.pdf", v2="new.pdf", pages=[
        PageChanges(v1_page=1, v2_page=1, changes=[change], method="text"),
    ])
    result = tmp_path / "marked.pdf"
    assert write_markup(source, log, result) == {}
    assert source.read_bytes() == original
    with pymupdf.open(result) as marked:
        assert "1" in marked[0].get_text().splitlines()


def test_ink_covered_page_keeps_the_number_in_the_annotation_comment(tmp_path):
    source = tmp_path / "dense.pdf"
    with pymupdf.open() as pdf:
        page = pdf.new_page(width=100, height=100)
        page.draw_rect(page.rect, fill=(0, 0, 0))
        pdf.save(source)
    change = Change(id=8, type="modification", category="value", field="Value", bbox_v2=(20, 20, 50, 50),
                    evidence="Changed region", confidence=.9)
    log = ChangeLog(job_id="test", v1="old.pdf", v2="new.pdf", pages=[
        PageChanges(v1_page=1, v2_page=1, changes=[change], method="pixel_diff"),
    ])
    result = tmp_path / "marked.pdf"
    assert write_markup(source, log, result) == {1: [8]}
    with pymupdf.open(result) as marked:
        page = marked[0]
        assert "#8" in next(page.annots()).info["content"]
        assert "8" not in page.get_text()


def _approve(job, result):
    job.call("review_page", json.dumps({
        "page": 1, "view_id": result.summary["view_id"], "passed": True,
    }))


def test_agent_snap_requires_a_new_render_and_approval(tmp_path):
    v1 = make_table(tmp_path / "v1.pdf")
    v2 = make_table(tmp_path / "v2.pdf", ("8", "7"))
    root = tmp_path / "job"
    job = JobTools(v1, v2, root, Budget(40, Decimal(5)), JsonlTracer(root / "trace.jsonl"), None)
    job.call("discover")
    first = job.call("list_candidates", '{"offset":0}').summary["items"][0]
    assert first["v2_page"] == 1
    job.call("write_markup")
    older = job.call("view_page", '{"page":1}')
    _approve(job, older)
    edit = job.call("snap_to_cell", json.dumps({"change_id": first["id"]})).summary
    assert edit["before"] == first["bbox_v2"]
    assert edit["after"] == pytest.approx((108, 108, 216, 146), abs=.8)
    with pytest.raises(ToolError, match="current markup"):
        job.call("deliver")
    job.call("write_markup")
    with pytest.raises(ToolError, match="View the page"):
        _approve(job, older)
    newer = job.call("view_page", '{"page":1}')
    assert newer.image_id != older.image_id
    _approve(job, newer)
    job.call("deliver")
    log = ChangeLog.model_validate_json((root / "changes.json").read_bytes())
    assert log.geometry_edits[0].actor == "agent"
    assert log.pages[0].changes[0].bbox_v2 == tuple(edit["after"])


def test_reviewer_snap_invalidates_page_acceptances_and_export(tmp_path):
    v1 = make_table(tmp_path / "v1.pdf")
    v2 = make_table(tmp_path / "v2.pdf", ("8", "7"))
    service = JobService(tmp_path / "jobs")
    try:
        job_id, future = service.submit(
            (v1.name, v1.read_bytes()), (v2.name, v2.read_bytes()), ComparisonOptions(60, Decimal(5)),
        )
        future.result(timeout=15)
        job = service.get(job_id)
        log = ChangeLog.model_validate_json(job["review_log"])
        change_ids = [c.id for c in log.pages[0].changes]
        assert len(change_ids) >= 2
        for revision, change_id in enumerate(change_ids):
            service.update_review(job_id, revision, action="accept", change_id=change_id)
        revision = len(change_ids)
        service.generate_final(job_id, revision)
        service.update_review(job_id, revision, action="snap_to_cell", change_id=change_ids[0])
        job = service.get(job_id)
        assert job["final_revision"] is None
        log = ChangeLog.model_validate_json(job["review_log"])
        page = log.pages[0]
        assert log.geometry_edits[0].actor == "reviewer"
        assert [c.human_status for c in page.changes] == ["edited"] + ["pending"] * (len(change_ids) - 1)
        png, _ = render_edited_page(v2, page, show_cosmetic=False)
        assert png.startswith(b"\x89PNG")
    finally:
        service.close()


def test_snap_rejects_a_cell_with_an_unreported_internal_rule(tmp_path, monkeypatch):
    from redline.toolkit.tables import Grid

    path = make_table(tmp_path / "v2.pdf")
    monkeypatch.setattr(
        "redline.toolkit.geometry.detect_table_grid",
        lambda _: Grid((60, 540), (175, 270, 365, 460), 600, 563),
    )
    change = Change(
        id=1, type="modification", category="value", field="Value", bbox_v2=(128, 123, 143, 137),
        evidence="Visible digit", confidence=.85,
    )
    with pytest.raises(ValueError, match="another ruling"):
        snap_to_detected_cell(path, 1, change, actor="agent")


@pytest.mark.parametrize("table_width,table_height", [(300, 256), (180, 140)])
def test_raster_grid_shift_stays_in_review_without_highlighting_table(tmp_path, table_width, table_height):
    from redline.models import ChangeLog
    from redline.agent.loop import run_comparison

    paths = [tmp_path / "original.pdf", tmp_path / "revised.pdf"]
    for offset, path in zip((0, 4), paths):
        with pymupdf.open() as pdf:
            page = pdf.new_page(width=380, height=340)
            page.insert_text((35, 30), "Bill of materials")
            left, top = 30 + offset, 49 + offset
            for column in range(5):
                x = left + column * table_width / 4
                page.draw_line((x, top), (x, top + table_height))
            for row in range(7):
                y = top + row * table_height / 6
                page.draw_line((left, y), (left + table_width, y))
            if offset:
                page.draw_rect((352, 20, 361, 29), color=(0, 0, 0), fill=(0, 0, 0))
            pdf.save(path)

    # A grid may be detected via the same connected run or via separate scan lines.
    # Keep the regression deterministic by comparing a scan grid with its raster omitted.
    if table_width < 200:
        with pymupdf.open() as pdf:
            page = pdf.new_page(width=380, height=340)
            page.insert_text((35, 30), "Bill of materials")
            pdf.save(paths[0])

    root = tmp_path / "job"
    result = run_comparison(*paths, root=root, budget=Budget(40, Decimal(5)), tracer=JsonlTracer(root / "trace.jsonl"))
    log = ChangeLog.model_validate_json(result.changes.read_bytes())
    changed = log.pages[0].changes
    broad = [change for change in changed if change.markup_withheld_reason]
    assert broad and all(change.needs_review and not change.should_draw(show_cosmetic=True) for change in broad)
    with pymupdf.open(result.pdf) as pdf:
        page = pdf[0]
        marks = list(page.annots())
        assert marks and all(mark.info["title"] not in {f"Redline #{change.id}" for change in broad} for mark in marks)
        assert any(mark.rect.x0 > 345 and mark.rect.y1 < 34 for mark in marks)
