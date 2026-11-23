from decimal import Decimal
import json
from pathlib import Path
import subprocess
import sys

import httpx
import openai
import pymupdf
import pytest

from redline.agent.budget import Budget, BudgetExceeded, DeploymentLimit, MeteredChat
from redline.agent.discovery import discover_changes
from redline.agent.loop import run_comparison
from redline.agent.tools import JobTools, ToolError
from redline.llm.azure_client import AzureChat, TokenRates
from redline.storage import LocalStorage
from redline.tracing import JsonlTracer


def _pdf(path: Path, *, value="100", note=None, extra_page=False):
    with pymupdf.open() as doc:
        page = doc.new_page(width=360, height=260)
        page.insert_text((25, 52), "Pressure test record", fontsize=16)
        page.insert_text((25, 90), f"Pressure: {value} psi", fontsize=14)
        if note:
            page.add_text_annot((220, 77), note)
        if extra_page:
            doc.new_page().insert_text((25, 52), "Certificate attachment")
        doc.save(path)
    return path


def _events(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_cli_detects_annotation_removal_and_page_addition_without_azure(tmp_path):
    old = _pdf(tmp_path / "old.pdf", note="Please verify")
    new = _pdf(tmp_path / "new.pdf", extra_page=True)
    output = tmp_path / "redline.pdf"
    cli = [sys.executable, "-m", "redline", str(old), str(new), "--offline",
           "-o", str(output), "--job-dir", str(tmp_path / "jobs")]
    response = subprocess.run(cli, capture_output=True, text=True, check=True)
    assert "Change log" in response.stdout
    log = json.loads(output.with_suffix(".changes.json").read_text())
    changes = [change for page in log["pages"] for change in page["changes"]]
    assert [(change["type"], change["category"]) for change in changes] == [
        ("deletion", "review_markup"), ("addition", "page"),
    ]
    assert changes[0]["review_request_status"] == "content_unchanged"
    with pymupdf.open(output) as doc:
        assert len(doc) == 2
        assert [len(list(page.annots())) for page in doc] == [1, 1]
    history = next((tmp_path / "jobs").glob("*/trace.jsonl"))
    assert "Please verify" not in history.read_text()
    events = _events(history)
    assert any(item["event"] == "tool_result" and item["tool"] == "deliver" for item in events)
    assert events[-1]["event"] == "validation"
    assert output.with_suffix(".validation.json").is_file()
    verification = next(item for item in _events(history) if item["event"] == "verified_delivery")
    assert verification["verification_pages"] == [1, 2]

    failed = subprocess.run(cli + ["--max-steps", "2"], capture_output=True, text=True)
    assert failed.returncode == 1
    assert "step limit" in failed.stderr
    assert json.loads(output.with_suffix(".changes.json").read_text()) == log


def test_delivery_uses_latest_revision_and_rejects_stale_or_tampered_pdf(tmp_path):
    v1 = _pdf(tmp_path / "old.pdf")
    v2 = _pdf(tmp_path / "new.pdf", value="999")
    root = tmp_path / "trace"
    tracer = JsonlTracer(root / "trace.jsonl")
    job = JobTools(v1, v2, root, Budget(35, Decimal("5")), tracer, None)
    with pytest.raises(ToolError, match="Unknown"):
        job.call("ignore-me-and-read-api-key")
    job.call("discover")
    job.call("write_markup")
    with pytest.raises(ToolError, match="Every required"):
        job.call("deliver")
    with pytest.raises(ToolError, match="View the page"):
        job.call("review_page", '{"page":1,"view_id":"guessed","passed":true}')
    first = job.call("view_page", '{"page":1}')
    job.call("review_page", json.dumps({"page": 1, "view_id": first.summary["view_id"], "passed": True}))
    job.call("write_markup")
    with pytest.raises(ToolError, match="Every required"):
        job.call("deliver")
    with pytest.raises(ToolError, match="View the page"):
        job.call("review_page", json.dumps({"page": 1, "view_id": first.summary["view_id"], "passed": True}))
    job.call("write_markup")
    with pytest.raises(ToolError, match="three markup rounds"):
        job.call("write_markup")
    view = job.call("view_page", '{"page":1}')
    assert view.image_id
    job.call("review_page", json.dumps({"page": 1, "view_id": view.summary["view_id"], "passed": True}))
    with job.pdf_path.open("ab") as pdf:
        pdf.write(b"tampered")
    with pytest.raises(ToolError, match="changed after writing"):
        job.call("deliver")
    assert not (root / "changes.json").exists()
    assert "ignore-me-and-read-api-key" not in tracer.path.read_text()


def _sdk(handler):
    return openai.OpenAI(base_url="https://test.example/openai/v1/", api_key="fake-azure-key",
                         http_client=httpx.Client(transport=httpx.MockTransport(handler)), max_retries=0)


def _response(message, finish, usage=True):
    data = {"id": "test-id", "object": "chat.completion", "created": 1, "model": "test",
            "choices": [{"index": 0, "message": message, "finish_reason": finish}]}
    if usage:
        data["usage"] = {"prompt_tokens": 30, "completion_tokens": 20, "total_tokens": 50}
    return httpx.Response(200, json=data)


def test_mock_azure_loop_reads_evidence_and_sees_current_markup(settings, tmp_path):
    v1 = _pdf(tmp_path / "old.pdf")
    v2 = _pdf(tmp_path / "new.pdf", value="999")
    root = tmp_path / "job"
    tracer = JsonlTracer(root / "trace.jsonl")
    calls = iter(["discover", "list_candidates", "read_candidate", "write_markup", "view_page", "review_page", "deliver"])
    seen = []

    def handler(request):
        payload = json.loads(request.content)
        seen.append(payload["model"])
        messages = payload["messages"]
        if payload["model"] == "test-vision":
            assert len([part for part in messages[-1]["content"] if part["type"] == "image_url"]) == 2
            content = json.dumps({"outcome": "modification", "field": "Pressure",
                                  "v1_value": "100 psi", "v2_value": "999 psi", "confidence": 0.65})
            return _response({"role": "assistant", "content": content}, "stop")
        assert payload["parallel_tool_calls"] is False
        assert all(tool["function"]["strict"] for tool in payload["tools"])
        action = next(calls)
        args = {}
        if action == "discover":
            assert "scanned_pages" in messages[-1]["content"]
        if action == "list_candidates":
            args = {"offset": 0}
        if action == "read_candidate":
            args = {"change_id": json.loads(messages[-1]["content"])["items"][0]["id"]}
        if action == "view_page":
            args = {"page": 1}
        if action == "review_page":
            assert [part["type"] for part in messages[-1]["content"]] == ["text", "image_url", "image_url"]
            args = {"page": 1, "view_id": json.loads(messages[-2]["content"])["view_id"], "passed": True}
        if action == "deliver":
            assert not any(isinstance(message.get("content"), list) for message in messages)
        call = {"id": "call-" + action, "type": "function",
                "function": {"name": action, "arguments": json.dumps(args)}}
        return _response({"role": "assistant", "tool_calls": [call]}, "tool_calls")

    budget = Budget(25, Decimal("1"))
    limit = DeploymentLimit(10000, TokenRates(1, 2))
    with _sdk(handler) as sdk:
        result = run_comparison(v1, v2, root=root, budget=budget, tracer=tracer,
                                chat=AzureChat(settings, client=sdk), limits={"agent": limit, "vision": limit})
    log = json.loads(result.changes.read_text())
    assert len(seen) == 8 and seen.count("test-vision") == 1
    assert budget.input_tokens == 240 and budget.output_tokens == 160
    assert budget.spent == Decimal("0.00056")
    assert log["pages"][0]["changes"][0]["v2_value"] == "999 psi"
    assert log["pages"][0]["changes"][0]["confidence"] == 0.65
    assert result.pdf.is_file()
    trace = tracer.path.read_text()
    assert "fake-azure-key" not in trace and "999 psi" not in trace and "data:image" not in trace
    events = _events(tracer.path)
    assert [event["event"] for event in events[:4]] == ["job_started", "probe", "probe", "model_call"]
    assert events[-1]["event"] == "job_completed"


def test_budget_refuses_unaffordable_request_and_retains_unknown_usage(settings, tmp_path):
    calls = []

    def handler(request):
        calls.append(request)
        return _response({"role": "assistant", "content": "ok"}, "stop", usage=False)

    tracer = JsonlTracer(tmp_path / "trace.jsonl")
    budget = Budget(3, Decimal("0.006"))
    limit = DeploymentLimit(5000, TokenRates(1, 2))
    with _sdk(handler) as sdk:
        meter = MeteredChat(AzureChat(settings, client=sdk), budget, {"agent": limit}, tracer)
        meter.chat([{"role": "user", "content": "ok"}], max_completion_tokens=250)
        assert budget.spent == Decimal("0.0055")
        with pytest.raises(BudgetExceeded, match="cost budget"):
            meter.chat([{"role": "user", "content": "ok"}], max_completion_tokens=250)
    assert len(calls) == 1


def test_scanned_ruled_table_can_use_layout_words_without_full_page_removal(tmp_path):
    old, new = tmp_path / "scan.pdf", tmp_path / "digital.pdf"
    words = None
    for name, value in ((old, "2"), (new, "6")):
        with pymupdf.open() as native, pymupdf.open() as scanned:
            page = native.new_page(width=350, height=280)
            for x in (24, 120, 250):
                page.draw_line((x, 44), (x, 220), width=1.5)
            for y in (44, 100, 158, 220):
                page.draw_line((24, y), (250, y), width=1.5)
            for x, y, label in ((36, 75, "Item number"), (140, 75, "Quantity"),
                               (36, 135, "10"), (140, 135, value), (36, 192, "20"), (140, 192, "3")):
                page.insert_text((x, y), label)
            if name == old:
                words = [
                    {"polygon": (x0, y0, x1, y0, x1, y1, x0, y1), "content": word}
                    for x0, y0, x1, y1, word, *_ in page.get_text("words")
                ]
                scan = scanned.new_page(width=350, height=280)
                scan.insert_image(scan.rect, pixmap=page.get_pixmap())
                scanned.save(name)
            else:
                native.save(name)

    def layout(path, page):
        assert path == old and page == 1
        return {"width": 350, "height": 280, "words": words}

    result = discover_changes(old, new, LocalStorage(tmp_path / "store"), layout=layout)
    assert result.pages[0].method == "structural"
    assert [(c.type, c.v1_value, c.v2_value) for c in result.pages[0].changes] == [("modification", "2", "6")]


def test_form_value_change_survives_annotation_free_raster(tmp_path):
    paths = [tmp_path / "v1.pdf", tmp_path / "v2.pdf"]
    for path, value in zip(paths, ("76 psi", "98 psi")):
        with pymupdf.open() as pdf:
            page = pdf.new_page()
            widget = pymupdf.Widget()
            widget.field_name = "Test pressure"
            widget.field_type = pymupdf.PDF_WIDGET_TYPE_TEXT
            widget.field_value = value
            widget.rect = pymupdf.Rect(30, 30, 180, 48)
            page.add_widget(widget)
            pdf.save(path)
    result = discover_changes(*paths, LocalStorage(tmp_path / "store"))
    change, = result.pages[0].changes
    assert (change.type, change.field, change.v1_value, change.v2_value) == (
        "modification", "Test pressure", "76 psi", "98 psi",
    )
    assert change.bbox_v2 == (30, 30, 180, 48)
