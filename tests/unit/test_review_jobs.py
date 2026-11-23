from decimal import Decimal
import json
from pathlib import Path

import pymupdf
import pytest

from redline.comparison import ComparisonOptions
from redline.eval import GoldenCase, run_suite, score
from redline.jobs import JobService, ReviewConflict
from redline.models import ChangeLog
from redline.review import ReviewEdit, render_preview

GOLDEN = Path(__file__).resolve().parents[1] / "golden"


def test_background_job_edit_and_revisioned_export(tmp_path):
    service = JobService(tmp_path / "jobs")
    folder = GOLDEN / "pressure-form"
    try:
        job_id, worker = service.submit(
            ("original.pdf", (folder / "v1.pdf").read_bytes()),
            ("revised.pdf", (folder / "v2.pdf").read_bytes()),
            ComparisonOptions(60, Decimal("5"), summary_page=True),
        )
        worker.result(timeout=15)
        job = service.get(job_id)
        assert job["status"] == "completed"
        assert job["cost"] == 0
        work = service.directory(job)
        assert json.loads((work / "review-regions.json").read_text())["1"] == [260, 137, 394, 160]

        service.update_review(job_id, 0, action="edit", change_id=1, edit=ReviewEdit(
            field="Reviewed pressure", type="modification", v1_value="100 psi", v2_value="120 psi",
        ))
        with pytest.raises(ReviewConflict):
            service.update_review(job_id, 0, action="reject", change_id=1)
        service.update_review(job_id, 1, action="reject", change_id=2)
        service.update_review(job_id, 2, action="accept_high")
        service.generate_final(job_id, 3)
        assert service.get(job_id)["final_revision"] == 3
        with pymupdf.open(work / "final-3.pdf") as pdf:
            assert len(pdf) == 2
            assert len(list(pdf[0].annots())) == 1
            assert "1 accepted change" in pdf[-1].get_text()

        service.update_review(job_id, 3, action="accept", change_id=1)
        assert service.get(job_id)["final_revision"] is None
        service.generate_final(job_id, 4)
        with pymupdf.open(work / "final-4.pdf") as pdf:
            assert len(list(pdf[0].annots())) == 2
            assert "2 accepted changes" in pdf[-1].get_text()
        changes = json.loads((work / "final-4.json").read_text())["pages"][0]["changes"]
        assert [(c["field"], c["human_status"]) for c in changes] == [
            ("Reviewed pressure", "accepted"), ("Gauge ID", "rejected"), ("Square", "accepted"),
        ]
    finally:
        service.close()


def test_golden_run_and_reordered_source_crop_boxes(tmp_path):
    report = run_suite(GOLDEN)
    assert set(report["by_document_type"]) == {"form", "table/BOM"}
    for stats in report["by_document_type"].values():
        assert stats["recall"] == stats["precision"] == stats["type_accuracy"] == stats["value_accuracy"] == 1.0
    service = JobService(tmp_path / "jobs")
    case = GOLDEN / "parts-list"
    try:
        job_id, worker = service.submit(
            ("before.pdf", (case / "v1.pdf").read_bytes()), ("after.pdf", (case / "v2.pdf").read_bytes()),
            ComparisonOptions(60, Decimal("5")),
        )
        worker.result(timeout=15)
        job = service.get(job_id)
        log = ChangeLog.model_validate_json(job["review_log"])
        box = json.loads((service.directory(job) / "review-regions.json").read_text())["1"]
        assert box[1] < log.pages[0].changes[0].bbox_v2[1]
        assert render_preview(service.inputs(job)[0], 1, tuple(box)).startswith(b"\x89PNG")
    finally:
        service.close()


def test_evaluation_counts_one_to_one_matches_and_reports_mistyped_localizations():
    example = GoldenCase.model_validate({
        "document_type": "form",
        "changes": [{"v2_page": 1, "type": "addition", "bbox_v2": [0, 0, 10, 10], "v2_value": "new"}],
    })
    change = {
        "type": "deletion", "category": "value", "field": "test", "evidence": "pixel diff", "confidence": 0.55,
        "bbox_v2": [0, 0, 10, 10],
    }
    log = ChangeLog.model_validate({
        "job_id": "sample", "v1": "v1.pdf", "v2": "v2.pdf",
        "pages": [{"v1_page": 1, "v2_page": 1, "method": "text", "changes": [{"id": 1, **change}]}],
    })
    wrong = score(log, example, pages=1, elapsed=1).metrics()
    assert wrong["recall"] == wrong["type_accuracy"] == 0
    assert wrong["localized"] == 1
    log.pages[0].changes += [
        log.pages[0].changes[0].model_copy(update={"id": i, "type": "addition", "v2_value": "new"})
        for i in (2, 3)
    ]
    duplicates = score(log, example, pages=1, elapsed=1).metrics()
    assert duplicates["recall"] == 1 and duplicates["precision"] == pytest.approx(1 / 3)
