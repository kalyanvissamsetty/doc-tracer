"""Launch the shared comparison engine with optional metered Azure services."""
from contextlib import ExitStack
from dataclasses import dataclass, replace
from decimal import Decimal
from pathlib import Path

from redline.agent.budget import Budget, DeploymentLimit
from redline.agent.loop import ComparisonResult, run_comparison
from redline.cloud import entra_credential
from redline.config import ConfigurationError, Settings
from redline.llm.azure_client import AzureChat, TokenRates, build_client
from redline.ocr.docintel import DocumentIntel, OCRRequestError
from redline.storage import LocalStorage, Storage
from redline.storage.blob import archive_artifacts
from redline.tracing import Tracer
from redline.validation import ValidationPlugin
from redline.validation.pressure import PressureTestValidator, load_pressure_rules


@dataclass(frozen=True)
class ComparisonOptions:
    max_steps: int
    max_cost_usd: Decimal
    show_cosmetic: bool = False
    summary_page: bool = False
    document_type: str = "auto"
    docintel_layout: bool = False
    pressure_rules: Path | None = None
    allow_sandbox: bool = False


def deployment_limits(settings: Settings) -> dict[str, DeploymentLimit]:
    limits = {}
    missing = []
    for role in ("agent", "vision"):
        names = [f"{role}_{field}" for field in ("context_tokens", "input_usd_per_million", "output_usd_per_million")]
        tokens, input_rate, output_rate = (getattr(settings, name) for name in names)
        if any(value is None for value in (tokens, input_rate, output_rate)):
            missing.extend(name.upper() for name in names if getattr(settings, name) is None)
        else:
            limits[role] = DeploymentLimit(tokens, TokenRates(input_rate, output_rate))
    if missing:
        raise ConfigurationError("Configure the deployment context bounds and rates: " + ", ".join(missing))
    return limits


def validate_options(options: ComparisonOptions, settings: Settings | None) -> None:
    if options.allow_sandbox and settings is None:
        raise ConfigurationError("The optional Python fallback is for the Azure agent workflow.")
    if options.docintel_layout and (settings is None or settings.docintel_layout_usd_per_page is None):
        raise ConfigurationError("Metered Layout OCR requires Azure settings and DOCINTEL_LAYOUT_USD_PER_PAGE.")
    if settings is not None:
        deployment_limits(settings)
    Budget(options.max_steps, options.max_cost_usd)


def compare(
    v1: Path, v2: Path, *, root: Path, options: ComparisonOptions, tracer: Tracer,
    settings: Settings | None = None,
) -> ComparisonResult:
    validate_options(options, settings)
    kind = options.document_type.strip().casefold()
    validation = None
    if kind in {"auto", "pressure test", "pressure-test"} or options.pressure_rules is not None:
        plugin: ValidationPlugin = PressureTestValidator(load_pressure_rules(options.pressure_rules))
        validation = plugin.validate(v2, explicit=kind != "auto" or options.pressure_rules is not None)
    budget = Budget(options.max_steps, options.max_cost_usd)
    with ExitStack() as resources:
        limits = deployment_limits(settings) if settings else None
        credential = None
        token_provider = None
        if settings and settings.azure_auth_mode == "entra":
            credential = entra_credential()
            from azure.identity import get_bearer_token_provider

            resources.callback(credential.close)
            token_provider = get_bearer_token_provider(credential, "https://cognitiveservices.azure.com/.default")
        sdk = resources.enter_context(build_client(settings, token_provider, max_retries=0)) if settings else None
        chat = AzureChat(settings, client=sdk) if sdk else None
        layout = None
        if options.docintel_layout:
            rate = Decimal(str(settings.docintel_layout_usd_per_page))
            ocr = DocumentIntel(settings, LocalStorage(root), retry_total=0, credential=credential)
            resources.callback(ocr.client.close)

            def layout(path: Path, page: int) -> dict | None:
                def reserve():
                    budget.step()
                    budget.reserve(rate)
                    budget.docintel_pages += 1
                    tracer.record("ocr_call", page=page, step=budget.steps, spent_usd=float(budget.spent))

                try:
                    pages = ocr.extract(path, page, before_request=reserve).get("pages", [])
                except OCRRequestError:
                    tracer.record("ocr_fallback", page=page)
                    return None
                return next((item for item in pages if item.get("pageNumber") == page), None)

        result = run_comparison(
            v1, v2, root=root, tracer=tracer, budget=budget, chat=chat, limits=limits,
            show_cosmetic=options.show_cosmetic, layout=layout, document_type=options.document_type,
            allow_sandbox=options.allow_sandbox,
        )
    if validation is not None:
        filename = "validation.json"
        LocalStorage(root).write(filename, validation.model_dump_json(indent=2).encode())
        tracer.record(
            "validation", plugin=validation.plugin, artifact_id=filename,
            passed=sum(item.status == "pass" for item in validation.findings),
            failed=sum(item.status == "fail" for item in validation.findings),
            review=sum(item.status == "review" for item in validation.findings),
        )
        result = replace(result, validation=root / filename)
    return result


def archive_comparison(
    store: Storage, job_id: str, inputs: tuple[Path, Path], result: ComparisonResult, trace: Path,
) -> None:
    files = {
        "v1.pdf": inputs[0], "v2.pdf": inputs[1], "redline.pdf": result.pdf,
        "changes.json": result.changes, "trace.jsonl": trace,
        "review-regions.json": result.changes.with_name("review-regions.json"),
    }
    if result.validation:
        files["validation.json"] = result.validation
    archive_artifacts(store, f"jobs/{job_id}/comparison", files)
