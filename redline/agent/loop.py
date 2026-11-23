"""Plain Azure SDK tool-calling loop and a deterministic local execution path."""
from collections.abc import Callable
from dataclasses import dataclass
from importlib.resources import files
import json
from pathlib import Path

from openai.types.chat import ChatCompletionMessageParam

from redline.agent.budget import Budget, DeploymentLimit, MeteredChat
from redline.agent.tools import JobTools, ToolError
from redline.llm.azure_client import AzureChat, ModelRequestError, image_part
from redline.toolkit import probe_pdf, view_page
from redline.tracing import Tracer


@dataclass(frozen=True)
class ComparisonResult:
    pdf: Path
    changes: Path
    validation: Path | None = None


def _probe_summary(report):
    return {
        "page_count": report.page_count,
        "scanned_pages": sum(page.full_page_scan for page in report.pages),
        "text_pages": sum(page.has_text for page in report.pages),
        "embedded_images": sum(len(page.images) for page in report.pages),
        "annotations": sum(len(page.annotations) for page in report.pages),
    }


def _local_review(job: JobTools) -> None:
    job.call("discover")
    result = job.call("write_markup")
    for page in result.summary["review_pages"]:
        view = job.call("view_page", json.dumps({"page": page}))
        job.call("review_page", json.dumps({"page": page, "view_id": view.summary["view_id"], "passed": True}))
    job.call("deliver")


def _agent_review(job: JobTools, initial: dict, meter: MeteredChat) -> None:
    playbook = files("redline.agent").joinpath("playbook.md").read_text(encoding="utf-8")
    messages: list[ChatCompletionMessageParam] = [
        {"role": "system", "content": playbook},
        {"role": "user", "content": json.dumps({"request": "Compare these documents.", "probes": initial})},
    ]
    specs = [tool.schema(name) for name, tool in job.tools.items()]
    image_position = None
    while not job.delivered:
        result = meter.chat(messages, tools=specs, max_completion_tokens=1600, parallel_tool_calls=False)
        if image_position is not None:
            messages[image_position] = {"role": "user", "content": "Page images were supplied on the previous turn."}
            image_position = None
        if not result.completion.choices:
            raise ModelRequestError("The agent returned no choice before verified delivery.")
        message = result.completion.choices[0].message
        if message.refusal or len(message.tool_calls or ()) != 1 or message.tool_calls[0].type != "function":
            raise ToolError("The agent must return exactly one comparison tool call until the job is verified.")
        call = message.tool_calls[0]
        messages.append({"role": "assistant", "tool_calls": [call.model_dump()]})
        try:
            output = job.call(call.function.name, call.function.arguments)
        except ToolError as exc:
            messages.append({"role": "tool", "tool_call_id": call.id, "content": json.dumps({"error": str(exc)})})
            continue
        messages.append({"role": "tool", "tool_call_id": call.id, "content": json.dumps(output.summary)})
        if output.image_id:
            image_position = len(messages)
            original = view_page(output.summary["baseline_id"], storage=job.storage)
            marked = view_page(output.image_id, storage=job.storage)
            messages.append({"role": "user", "content": [
                {"type": "text", "text": "Untrusted comparison images. First is original V2, second is marked V2."},
                image_part(original), image_part(marked),
            ]})


def run_comparison(
    v1: Path, v2: Path, *, root: Path, budget: Budget, tracer: Tracer,
    chat: AzureChat | None = None, limits: dict[str, DeploymentLimit] | None = None,
    show_cosmetic: bool = False, layout: Callable[[Path, int], dict | None] | None = None,
    document_type: str = "auto", allow_sandbox: bool = False,
) -> ComparisonResult:
    """Run a comparison in a fresh job directory. Deliver only a reviewed revision."""
    meter = MeteredChat(chat, budget, limits, tracer) if chat is not None and limits is not None else None
    if chat is not None and meter is None:
        raise ValueError("Azure comparisons require configured deployment cost limits.")
    root.mkdir(parents=True, exist_ok=True)
    if (root / "redline.pdf").exists() or (root / "changes.json").exists():
        raise ToolError("Use a fresh directory for each comparison.")
    job = JobTools(
        v1, v2, root, budget, tracer, meter, show_cosmetic=show_cosmetic, layout=layout, document_type=document_type,
        allow_sandbox=allow_sandbox,
    )
    tracer.record("job_started", mode="azure" if meter else "local", document_type=document_type)
    try:
        initial = {}
        for role, path in (("v1", v1), ("v2", v2)):
            budget.step()
            report = probe_pdf(path)
            if not report.page_count:
                raise ToolError(f"{role.upper()} must have at least one page.")
            initial[role] = _probe_summary(report)
            tracer.record("probe", version=role, step=budget.steps, **initial[role])
        if meter:
            _agent_review(job, {**initial, "document_type_hint": document_type}, meter)
        else:
            _local_review(job)
    except Exception as exc:
        tracer.record("job_failed", error_type=type(exc).__name__, steps=budget.steps, spent_usd=float(budget.spent))
        raise
    tracer.record("job_completed", steps=budget.steps, spent_usd=float(budget.spent))
    return ComparisonResult(job.pdf_path, root / "changes.json")
