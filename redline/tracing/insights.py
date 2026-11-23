"""Forward minimal trace metadata through the official Azure Monitor OTel exporter."""
from azure.monitor.opentelemetry.exporter import AzureMonitorTraceExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

NUMERIC = {
    "step", "steps", "page", "duration_ms", "input_tokens", "output_tokens", "spent_usd",
    "reserved_usd", "regions", "alignment_quality", "candidates", "revision", "passed", "failed", "review",
}
ENUMS = {
    "role": {"agent", "vision"}, "mode": {"azure", "local"}, "version": {"v1", "v2"},
    "verification": {"agent_visual_review", "local_render_checks"},
}
EVENTS = {
    "job_started", "probe", "model_call", "model_error", "model_result", "tool_call", "tool_result",
    "tool_rejected", "discovery", "drawing_comparison", "crop", "ocr_call", "ocr_fallback",
    "verified_delivery", "job_completed", "job_failed", "validation", "archive", "sandbox_execution",
}
TOOLS = {
    "discover", "list_candidates", "read_candidate", "write_markup", "view_page", "review_page", "deliver", "run_python",
}


class InsightsTracer:
    def __init__(self, tracer, job_id: str):
        self.tracer = tracer
        self.job_id = job_id

    def record(self, event: str, **data: object) -> None:
        if event not in EVENTS:
            return
        attributes = {f"redline.{key}": value for key, value in data.items()
                      if key in NUMERIC and type(value) in {int, float}}
        for key, allowed in {**ENUMS, "tool": TOOLS}.items():
            value = data.get(key)
            if isinstance(value, str) and value in allowed:
                attributes[f"redline.{key}"] = value
        attributes["redline.job_id"] = self.job_id
        try:
            with self.tracer.start_as_current_span(f"redline.{event}", attributes=attributes):
                pass
        except Exception:
            # Local JSONL remains the authoritative job trace if telemetry is unavailable.
            pass


def create_provider(connection_string: str) -> TracerProvider:
    provider = TracerProvider(resource=Resource.create({"service.name": "pdf-redline-agent"}))
    exporter = AzureMonitorTraceExporter(connection_string=connection_string, disable_offline_storage=True)
    provider.add_span_processor(BatchSpanProcessor(exporter))
    return provider
