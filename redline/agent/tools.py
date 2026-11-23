"""Job-scoped tools with typed arguments, evidence IDs, and revisioned review."""
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from hashlib import sha256
from json import dumps
from math import ceil, floor
from pathlib import Path
from secrets import token_hex

import numpy as np
import pymupdf
from pydantic import BaseModel, ConfigDict, Field

from redline.agent.budget import Budget, MeteredChat
from redline.agent.discovery import Discovery, PdfCropRef, discover_changes, page_dpi
from redline.llm.region_reader import RegionContext, read_region_pair
from redline.models import Change, ChangeLog, Cost, GeometryEdit
from redline.review import render_preview
from redline.sandbox import SandboxError, run_python
from redline.storage import LocalStorage
from redline.toolkit import normalize_value, render_page, view_crop, view_page, write_markup
from redline.toolkit.geometry import snap_to_detected_cell
from redline.tracing import Tracer


class ToolError(ValueError):
    pass


class Args(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class ListArgs(Args):
    offset: int = Field(ge=0)


class CandidateArgs(Args):
    change_id: int = Field(ge=1)


class PythonArgs(Args):
    code: str


class PageArgs(Args):
    page: int = Field(ge=1)


class ReviewArgs(PageArgs):
    view_id: str
    passed: bool


@dataclass(frozen=True)
class Tool:
    args: type[Args]
    description: str

    def schema(self, name: str):
        parameters = self.args.model_json_schema()
        for entry in parameters["properties"].values():
            entry.pop("minimum", None)  # Enforced locally for deployment schema compatibility.
        return {"type": "function", "function": {
            "name": name, "description": self.description, "strict": True,
            "parameters": parameters,
        }}


REGISTRY = {
    "run_python": Tool(PythonArgs, "Optional last resort for pure document calculations that require no files or network. Do not use for comparison, visualization, or as an alternative to a registered PDF tool. Returns bounded stdout and exit code."),
    "discover": Tool(Args, "Use first to pair pages and inventory evidence. Use once per job. Returns counts."),
    "list_candidates": Tool(ListArgs, "Use after discovery for 12 candidate summaries at an offset. Do not request it before discovery. Returns candidate IDs, types, evidence, and reading availability."),
    "read_candidate": Tool(CandidateArgs, "Use to interpret an ambiguous raster candidate with paired crops. Do not use for structural or page-level candidates. Returns an updated evidence-backed candidate."),
    "snap_to_cell": Tool(CandidateArgs, "Use when a value box needs to align with one ruled cell. Do not use for whole rows or ambiguous evidence spanning multiple cells. Returns the detected cell and revised bounds. Write and review again after a successful snap."),
    "write_markup": Tool(Args, "Use after discovering and reviewing candidates. Do not write before discovery or after the three-round limit. Returns revision and required review pages."),
    "view_page": Tool(PageArgs, "Use after writing markup to inspect a required V2 page visually. Do not use for old revisions or unrelated pages. Returns the marked page and a review ID."),
    "review_page": Tool(ReviewArgs, "Use only after seeing the current page image. Do not approve incorrect or illegible marks. Records pass or fail for the current revision and view ID."),
    "deliver": Tool(Args, "Use after approving every required current page. Do not use before visual verification. Returns a verified delivery manifest."),
}


@dataclass(frozen=True)
class ToolResult:
    summary: dict
    image_id: str | None = None


def _file_digest(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class JobTools:
    def __init__(
        self, v1: Path, v2: Path, root: Path, budget: Budget, tracer: Tracer,
        meter: MeteredChat | None, *, show_cosmetic: bool = False,
        layout: Callable[[Path, int], dict | None] | None = None, document_type: str = "auto", allow_sandbox: bool = False,
    ):
        self.v1, self.v2 = v1, v2
        self.root = root
        self.storage = LocalStorage(root)
        self.budget, self.tracer, self.meter = budget, tracer, meter
        self.show_cosmetic = show_cosmetic
        self.layout = layout
        self.document_type = document_type
        self.allow_sandbox = allow_sandbox
        self.annotation_subject = f"PDF Redline Agent {token_hex(12)}"
        self.discovery: Discovery | None = None
        self.revision = 0
        self.pending_pages: set[int] = set()
        self.views: dict[int, str] = {}
        self.approved: set[int] = set()
        self.written_digest: str | None = None
        self.written_candidate_revision = -1
        self.candidate_revision = 0
        self.geometry_edits: list[GeometryEdit] = []
        self.comment_only_tags: dict[int, list[int]] = {}
        self.delivered = False

    @property
    def tools(self) -> dict[str, Tool]:
        return {name: tool for name, tool in REGISTRY.items() if name != "run_python" or self.allow_sandbox}

    def _tool_run_python(self, code: str) -> ToolResult:
        try:
            result = run_python(code, tracer=self.tracer)
        except SandboxError as exc:
            raise ToolError(str(exc)) from None
        return ToolResult({"stdout": result.stdout, "stderr": result.stderr, "exit_code": result.exit_code})

    @property
    def pdf_path(self) -> Path:
        return self.root / "redline.pdf"

    def call(self, name: str, arguments: str = "{}") -> ToolResult:
        self.budget.step()
        if self.delivered:
            raise ToolError("This comparison has already been delivered.")
        entry = self.tools.get(name)
        if entry is None:
            self.tracer.record("tool_rejected", reason="unregistered_tool")
            raise ToolError("Unknown comparison tool.")
        try:
            args = entry.args.model_validate_json(arguments)
        except ValueError:
            self.tracer.record("tool_rejected", tool=name, reason="invalid_arguments")
            raise ToolError("Invalid arguments for this tool.") from None
        identifiers = {field: value for field, value in args.model_dump().items() if field not in {"view_id", "code"}}
        self.tracer.record("tool_call", tool=name, step=self.budget.steps, inputs=identifiers)
        try:
            if name != "discover" and self.discovery is None:
                raise ToolError("Run discover before calling other tools.")
            result = getattr(self, f"_tool_{name}")(**args.model_dump())
        except ToolError:
            self.tracer.record("tool_rejected", tool=name, **identifiers)
            raise
        # Keep raw document text, connection data, and image bytes out of the trace.
        if name == "list_candidates":
            outputs = {"candidate_ids": [item["id"] for item in result.summary["items"]],
                       "next_offset": result.summary["next_offset"]}
        elif name == "read_candidate":
            outputs = {key: result.summary[key] for key in ("id", "type", "category", "confidence")}
        elif name == "run_python":
            outputs = {"exit_code": result.summary["exit_code"]}
        else:
            outputs = {key: value for key, value in result.summary.items() if key != "view_id"}
        self.tracer.record(
            "tool_result", tool=name, revision=self.revision, artifact_id=result.image_id,
            inputs=identifiers, outputs=outputs,
        )
        return result

    def _tool_discover(self) -> ToolResult:
        if self.discovery is not None:
            raise ToolError("This job has already run discovery.")
        self.discovery = discover_changes(
            self.v1, self.v2, self.storage, layout=self.layout, document_type=self.document_type,
        )
        for event in self.discovery.drawing_diagnostics:
            self.tracer.record("drawing_comparison", **event)
        changes = self._changes()
        self.tracer.record("discovery", pages=len(self.discovery.pages), candidates=len(changes))
        return ToolResult({"page_pairs": len(self.discovery.pages), "candidates": len(changes)})

    def _changes(self) -> list[Change]:
        return [change for page in self.discovery.pages for change in page.changes]

    def _summary(self, change: Change) -> dict:
        return {
            "id": change.id, "type": change.type, "category": change.category,
            "field": change.field[:150], "evidence": change.evidence,
            "confidence": change.confidence, "can_read": change.id in self.discovery.crops,
            "markup_withheld_reason": change.markup_withheld_reason,
        }

    def _tool_list_candidates(self, offset: int) -> ToolResult:
        located = [(group.v2_page, change) for group in self.discovery.pages for change in group.changes]
        batch = located[offset:offset + 12]
        next_offset = offset + len(batch)
        return ToolResult({
            "items": [{**self._summary(change), "v2_page": page, "bbox_v2": change.bbox_v2}
                      for page, change in batch],
            "next_offset": next_offset if next_offset < len(located) else None,
        })

    def _tool_snap_to_cell(self, change_id: int) -> ToolResult:
        if self.revision >= 3:
            raise ToolError("No markup rounds remain for reviewing a geometry repair.")
        for group in self.discovery.pages:
            for change in group.changes:
                if change.id != change_id:
                    continue
                if group.v2_page is None:
                    raise ToolError("This change has no revised page.")
                try:
                    edit = snap_to_detected_cell(self.v2, group.v2_page, change, actor="agent")
                except ValueError as exc:
                    raise ToolError(str(exc)) from None
                change.bbox_v2 = edit.after
                self.geometry_edits.append(edit)
                self.candidate_revision += 1
                return ToolResult(edit.model_dump())
        raise ToolError("Change ID was not found.")

    def _tool_read_candidate(self, change_id: int) -> ToolResult:
        refs = self.discovery.crops.get(change_id)
        if refs is None or self.meter is None:
            raise ToolError("This candidate has no available Azure crop reading.")
        change = next(item for item in self._changes() if item.id == change_id)
        crops = []
        ocr_notes = []
        for index, ref in enumerate(refs, 1):
            if isinstance(ref, PdfCropRef):
                crop = render_preview(ref.path, ref.page, ref.bbox, annots=False)
            else:
                crop = view_crop(ref.image_id, ref.bbox, storage=self.storage)
            crops.append(crop)
            artifact_id = f"crops/{change_id}-v{index}.png"
            self.storage.write(artifact_id, crop)
            self.tracer.record("crop", change_id=change_id, artifact_id=artifact_id)
            if isinstance(ref, PdfCropRef) and self.layout is not None:
                reading = self.layout(self.root / artifact_id, 1) or {}
                text = " ".join(word["content"] for word in reading.get("words", [])
                                if word.get("confidence", 0) >= 0.7)
                ocr_notes.append(f"V{index}: {text[:400]}")
        result = read_region_pair(
            *crops, RegionContext(change.evidence, change.field, "; ".join(ocr_notes)),
            chat=self.meter, temperature=None,
        ).interpretation
        if result.field:
            change.field = result.field[:150]
        change.v1_value, change.v2_value = result.v1_value, result.v2_value
        equivalent = bool(result.v1_value and result.v2_value) and (
            normalize_value(result.v1_value) == normalize_value(result.v2_value)
        )
        valid_change = {
            "addition": not result.v1_value and bool(result.v2_value),
            "deletion": bool(result.v1_value) and not result.v2_value,
            "modification": bool(result.v1_value and result.v2_value) and not equivalent,
        }
        if valid_change.get(result.outcome):
            change.type = result.outcome
            change.category = "value"
        elif result.outcome in {"cosmetic", "unchanged"} and equivalent and not result.needs_review:
            change.category = "cosmetic"
        else:
            change.category = "value"
        change.confidence = (
            min(0.69, result.confidence)
            if result.needs_review or not (valid_change.get(result.outcome) or change.category == "cosmetic")
            else result.confidence
        )
        self.candidate_revision += 1
        return ToolResult(self._summary(change))

    def _log(self) -> ChangeLog:
        deployments = self.meter.chat_client.deployments if self.meter else {}
        return ChangeLog(
            job_id=self.root.name, v1=self.v1.name, v2=self.v2.name,
            model_deployments=deployments,
            cost=Cost(
                input_tokens=self.budget.input_tokens, output_tokens=self.budget.output_tokens,
                estimated_usd=float(self.budget.spent),
            ),
            pages=self.discovery.pages, geometry_edits=self.geometry_edits, comment_only_tags=self.comment_only_tags,
        )

    def _expected(self, page: int) -> list[Change]:
        return [
            change for group in self.discovery.pages if group.v2_page == page for change in group.changes
            if change.should_draw(show_cosmetic=self.show_cosmetic)
        ]

    def _tool_write_markup(self) -> ToolResult:
        if self.revision >= 3:
            raise ToolError("The three markup rounds have been used.")
        log = self._log()
        self.comment_only_tags = write_markup(
            self.v2, log, self.pdf_path, show_cosmetic=self.show_cosmetic,
            annotation_subject=self.annotation_subject,
        )
        self.revision += 1
        self.written_candidate_revision = self.candidate_revision
        self.written_digest = _file_digest(self.pdf_path)
        self.views.clear()
        self.approved.clear()
        self.pending_pages = {
            group.v2_page for group in log.pages if group.v2_page is not None and self._expected(group.v2_page)
        } or {1}
        return ToolResult({"revision": self.revision, "review_pages": sorted(self.pending_pages)})

    def _check_revision(self) -> None:
        if self.written_digest is None or self.written_candidate_revision != self.candidate_revision:
            raise ToolError("Write a current markup revision first.")
        if not self.pdf_path.is_file() or _file_digest(self.pdf_path) != self.written_digest:
            raise ToolError("The marked PDF changed after writing. Generate a new revision.")

    def _tool_view_page(self, page: int) -> ToolResult:
        self._check_revision()
        if page not in self.pending_pages:
            raise ToolError("Page is outside the current verification set.")
        with pymupdf.open(self.pdf_path) as marked, pymupdf.open(self.v2) as baseline:
            revised, original = marked[page - 1], baseline[page - 1]
            if len(marked) != len(baseline) or revised.rect != original.rect or revised.rotation != original.rotation:
                raise ToolError("Markup changed page geometry.")
            def inventory(page):
                return Counter((mark.type[1], mark.info["title"], mark.info["subject"])
                               for mark in page.annots() or ())

            existing, actual = inventory(original), inventory(revised)
            expected = {f"Redline #{item.id}": item for item in self._expected(page)}
            if actual != existing + Counter(("Square", title, self.annotation_subject) for title in expected):
                raise ToolError("The rendered page has an unexpected annotation inventory.")
            bounds = revised.rect * revised.derotation_matrix
            boxes = []
            for mark in revised.annots() or ():
                if mark.info["subject"] != self.annotation_subject:
                    continue
                item = expected[mark.info["title"]]
                box = bounds if item.bbox_v2 is None else pymupdf.Rect(item.bbox_v2) & bounds
                if max(abs(a - b) for a, b in zip(mark.rect, box)) > 2:
                    raise ToolError("A redline annotation does not match its evidence coordinates.")
                boxes.append(box * revised.rotation_matrix)
            dpi = page_dpi(revised)
            width, height = revised.rect.width, revised.rect.height
        image_id = render_page(self.pdf_path, page, dpi, storage=self.storage)
        baseline_id = render_page(self.v2, page, dpi, storage=self.storage)
        if boxes:
            self._check_visible_boxes(boxes, width, height, baseline_id, image_id)
        view_id = token_hex(10)
        self.views[page] = view_id
        self.approved.discard(page)
        return ToolResult({
            "revision": self.revision, "page": page, "view_id": view_id, "baseline_id": baseline_id,
            "marked_changes": len(expected), "comment_only_tags": self.comment_only_tags.get(page, []),
        }, image_id)

    def _check_visible_boxes(self, boxes, width, height, baseline_id, image_id) -> None:
        images = [pymupdf.Pixmap(view_page(item, storage=self.storage)) for item in (baseline_id, image_id)]
        dimensions = images[0].width, images[0].height
        raster_width, raster_height = dimensions
        before, after = (
            np.frombuffer(pix.samples, dtype=np.uint8).reshape(raster_height, raster_width, 3) for pix in images
        )
        changed = np.any(before != after, axis=2)
        sx, sy = raster_width / width, raster_height / height
        for box in boxes:
            region = changed[floor(box.y0 * sy):ceil(box.y1 * sy), floor(box.x0 * sx):ceil(box.x1 * sx)]
            if not np.any(region):
                raise ToolError("A redline box did not produce a visible render difference.")

    def _tool_review_page(self, page: int, view_id: str, passed: bool) -> ToolResult:
        self._check_revision()
        if self.views.get(page) != view_id:
            raise ToolError("View the page in the current revision before reviewing it.")
        if passed:
            self.approved.add(page)
        else:
            self.approved.discard(page)
        return ToolResult({"revision": self.revision, "page": page, "approved": passed})

    def _tool_deliver(self) -> ToolResult:
        self._check_revision()
        if self.approved != self.pending_pages:
            raise ToolError("Every required page must pass review for the latest markup revision.")
        self.storage.write("changes.json", self._log().model_dump_json(indent=2).encode())
        self.storage.write("review-regions.json", dumps(self.discovery.source_boxes).encode())
        self.delivered = True
        self.tracer.record(
            "verified_delivery", revision=self.revision, verification_pages=sorted(self.approved),
            verification="agent_visual_review" if self.meter else "local_render_checks",
        )
        return ToolResult({"verified": True, "revision": self.revision})
