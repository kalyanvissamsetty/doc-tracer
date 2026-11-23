from decimal import Decimal
import json
from pathlib import Path
from runpy import run_path
import subprocess
import sys
from time import monotonic
from unittest.mock import Mock

import pytest

from redline.cloud import CloudServices
from redline.comparison import ComparisonOptions
from redline.config import CloudSettings, ConfigurationError, load_settings
from redline.jobs import JobService
from redline.sandbox import SandboxError, _collect, _command, run_python
from redline.storage.blob import BlobStorage
from redline.tracing import JsonlTracer


def test_blob_backend_round_trip_and_validates_ids():
    from azure.core.exceptions import ResourceNotFoundError

    container = Mock()
    blob = container.get_blob_client.return_value
    blob.download_blob.return_value.readall.return_value = b"pdf"
    backend = BlobStorage(container)
    backend.write("jobs/test/redline.pdf", b"pdf")
    container.get_blob_client.assert_called_with("redline/jobs/test/redline.pdf")
    blob.upload_blob.assert_called_once_with(b"pdf", overwrite=True)
    assert backend.read("jobs/test/redline.pdf") == b"pdf"
    blob.download_blob.side_effect = ResourceNotFoundError("test")
    with pytest.raises(FileNotFoundError):
        backend.read("jobs/test/missing")
    for path in ("../output", "/output", "x//y", "x\\y"):
        with pytest.raises(ValueError):
            backend.write(path, b"x")


def test_web_job_archives_verified_and_reviewed_outputs(tmp_path):
    from redline.storage import LocalStorage

    case = Path(__file__).parents[1] / "golden" / "pressure-form"
    cloud = CloudServices(CloudSettings())
    cloud.store = LocalStorage(tmp_path / "synthetic-blob")
    service = JobService(tmp_path / "jobs", cloud=cloud)
    try:
        job_id, worker = service.submit(
            ("before.pdf", (case / "v1.pdf").read_bytes()), ("after.pdf", (case / "v2.pdf").read_bytes()),
            ComparisonOptions(60, Decimal(5)),
        )
        worker.result(timeout=15)
        prefix = f"jobs/{job_id}"
        assert service.get(job_id)["status"] == "completed"
        manifest = json.loads(cloud.store.read(f"{prefix}/comparison/manifest.json"))
        assert {"v1.pdf", "v2.pdf", "redline.pdf", "changes.json", "validation.json"} <= set(manifest["artifacts"])
        trace = cloud.store.read(f"{prefix}/comparison/trace.jsonl")
        assert b"verified_delivery" in trace
        service.generate_final(job_id, 0)
        final = json.loads(cloud.store.read(f"{prefix}/reviews/0/manifest.json"))
        assert final["artifacts"] == ["changes.json", "redline.pdf"]
    finally:
        service.close()


def test_missing_or_conflicting_blob_configuration():
    with pytest.raises(ConfigurationError, match="exactly one"):
        CloudServices(CloudSettings(azure_storage_container="private"))
    with pytest.raises(ConfigurationError, match="exactly one"):
        CloudServices(CloudSettings(azure_storage_container="private", azure_storage_connection_string="example",
                                   azure_storage_account_url="https://example.test"))


def test_insights_exports_allowlisted_fields_without_document_data():
    pytest.importorskip("azure.monitor.opentelemetry.exporter")
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    from redline.tracing.insights import InsightsTracer

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    try:
        tracer = InsightsTracer(provider.get_tracer("test"), "job-42")
        tracer.record("model_result", role="vision", input_tokens=100, evidence="private text",
                      api_key="fake-key", tool="not-registered")
        tracer.record("unregistered-event", role="vision")
        spans = exporter.get_finished_spans()
        assert len(spans) == 1
        assert spans[0].name == "redline.model_result"
        assert dict(spans[0].attributes) == {
            "redline.job_id": "job-42", "redline.role": "vision", "redline.input_tokens": 100,
        }
    finally:
        provider.shutdown()


def test_entra_mode_loads_without_api_keys(tmp_path, monkeypatch):
    for key in ("AZURE_OPENAI_API_KEY", "AZURE_DOCINTEL_KEY", "AZURE_AUTH_MODE"):
        monkeypatch.delenv(key, raising=False)
    file = tmp_path / ".env"
    file.write_text(
        "AZURE_AUTH_MODE=entra\nAZURE_OPENAI_ENDPOINT=https://example.test\n"
        "AZURE_OPENAI_DEPLOYMENT_AGENT=agent\nAZURE_OPENAI_DEPLOYMENT_VISION=vision\n"
        "AZURE_DOCINTEL_ENDPOINT=https://doc.example.test\n"
    )
    settings = load_settings(file)
    assert settings.azure_auth_mode == "entra"
    assert settings.azure_openai_api_key is settings.azure_docintel_key is None


def test_sandbox_command_and_cgroup_guard(tmp_path, monkeypatch):
    command = _command("test-sandbox")
    for option in ("--network=none", "--read-only", "--cap-drop=ALL", "--cpus=0.5", "--memory=128m",
                   "--memory-swap=128m", "--user=65534:65534", "--pids-limit=16", "--pull=never"):
        assert option in command
    assert "--privileged" not in command and "--volume" not in command
    script = Path(__file__).parents[2] / "docker" / "sandbox_entry.py"
    guard = run_path(str(script))["limited"]
    monkeypatch.setattr("os.getuid", lambda: 65534, raising=False)
    for name, data in {"cpu.max": "50000 100000", "memory.max": str(128 * 1024 * 1024),
                       "memory.swap.max": "0", "pids.max": "16"}.items():
        (tmp_path / name).write_text(data)
    assert guard(tmp_path)
    (tmp_path / "memory.max").write_text("max")
    assert not guard(tmp_path)


def test_sandbox_rejects_large_input(tmp_path):
    tracer = JsonlTracer(tmp_path / "trace.jsonl")
    with pytest.raises(SandboxError, match="input size"):
        run_python("sensitive sample" * 5000, tracer=tracer)
    event = json.loads(tracer.path.read_text())
    assert event["status"] == "rejected"
    assert "sensitive" not in tracer.path.read_text()


@pytest.mark.skipif(sys.platform == "win32", reason="The sandbox pipe reader uses POSIX selectors.")
def test_sandbox_limits_output():
    with subprocess.Popen([sys.executable, "-c", "print('x' * 20000)"],
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE) as process:
        with pytest.raises(SandboxError, match="output limit"):
            _collect(process, monotonic() + 5)
        process.kill()


def test_entra_comparison_sends_a_bearer_token(settings, tmp_path, monkeypatch):
    pytest.importorskip("azure.identity")
    import httpx
    import openai
    from azure.core.credentials import AccessToken

    from redline.agent.loop import ComparisonResult
    from redline.comparison import compare

    identity = Mock(spec=["get_token", "close"])
    identity.get_token.return_value = AccessToken("synthetic-token", 4102444800)
    monkeypatch.setattr("redline.comparison.entra_credential", lambda: identity)
    openai_client = openai.OpenAI

    def respond(request):
        assert request.headers["Authorization"] == "Bearer synthetic-token"
        assert "api-key" not in request.headers
        return httpx.Response(200, json={"id": "test", "object": "chat.completion", "created": 1,
                                        "model": "agent", "choices": []})

    def create(**kwargs):
        transport = httpx.MockTransport(respond)
        return openai_client(http_client=httpx.Client(transport=transport), **kwargs)

    monkeypatch.setattr("redline.llm.azure_client.openai.OpenAI", create)

    def run(v1, v2, **kwargs):
        kwargs["chat"].chat([{"role": "user", "content": "test"}])
        return ComparisonResult(v1, v2)

    monkeypatch.setattr("redline.comparison.run_comparison", run)
    pricing = {f"{role}_{key}": value for role in ("agent", "vision") for key, value in (
        ("context_tokens", 32000), ("input_usd_per_million", 1), ("output_usd_per_million", 2),
    )}
    config = settings.model_copy(update={
        "azure_auth_mode": "entra", "azure_openai_api_key": None, "azure_docintel_key": None, **pricing,
    })
    compare(tmp_path / "v1.pdf", tmp_path / "v2.pdf", root=tmp_path,
            options=ComparisonOptions(60, Decimal(5), document_type="form"),
            tracer=JsonlTracer(tmp_path / "trace.jsonl"), settings=config)
    assert identity.get_token.call_args.args == ("https://cognitiveservices.azure.com/.default",)
    identity.close.assert_called_once()
