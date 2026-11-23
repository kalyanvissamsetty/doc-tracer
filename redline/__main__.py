"""Command line entry point for a local or Azure-backed comparison."""
import argparse
from decimal import Decimal, InvalidOperation
import os
from pathlib import Path
from shutil import copyfileobj
import sys
from tempfile import NamedTemporaryFile, mkdtemp

import pymupdf

from redline.agent.budget import BudgetExceeded
from redline.agent.tools import ToolError
from redline.cloud import CloudServices
from redline.config import ConfigurationError, load_cloud_settings, load_job_defaults, load_settings
from redline.llm.azure_client import ModelRequestError
from redline.comparison import ComparisonOptions, archive_comparison, compare
from redline.storage.blob import ArtifactArchiveError


def _decimal_argument(value: str) -> Decimal:
    try:
        return Decimal(value)
    except InvalidOperation:
        raise argparse.ArgumentTypeError("Enter an amount in USD.") from None


def _arguments(argv: list[str] | None):
    parser = argparse.ArgumentParser(description="Compare PDF revisions and write a verified markup.")
    parser.add_argument("v1", type=Path, help="Original PDF")
    parser.add_argument("v2", type=Path, help="Revised PDF")
    parser.add_argument("-o", "--output", type=Path, required=True, help="Output PDF")
    parser.add_argument("--document-type", choices=["auto", "as-built", "pressure-test", "form", "table"],
                        default="auto", help="Select a drawing or validation workflow explicitly")
    parser.add_argument("--pressure-rules", type=Path, help="Override the pressure-test rules YAML for this run")
    parser.add_argument("--allow-sandbox", action="store_true", help="Enable the constrained Python fallback on a Docker host")
    parser.add_argument("--offline", action="store_true", help="Use deterministic comparison and local render checks")
    parser.add_argument("--docintel-layout", action="store_true", help="Use metered Layout OCR for scanned tables and selected drawing crops")
    parser.add_argument("--show-cosmetic", action="store_true", help="Draw changes classified as cosmetic")
    parser.add_argument("--max-steps", type=int, help="Maximum probe, tool, and model steps")
    parser.add_argument("--max-cost-usd", type=_decimal_argument, help="Estimated Azure spending ceiling")
    parser.add_argument("--job-dir", type=Path, default=Path(".redline/jobs"), help="Directory for job traces and artifacts")
    return parser.parse_args(argv)


def _copy_output(source: Path, target: Path) -> None:
    with NamedTemporaryFile(dir=target.parent, delete=False) as output:
        temp = Path(output.name)
        try:
            with source.open("rb") as stream:
                copyfileobj(stream, output)
            output.close()
            os.replace(temp, target)
        finally:
            temp.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    args = _arguments(argv)
    output = args.output.resolve()
    sidecar = output.with_suffix(".changes.json")
    validation_sidecar = output.with_suffix(".validation.json")
    root = None
    try:
        if not args.v1.is_file() or not args.v2.is_file():
            raise ToolError("Both inputs must be readable PDF files.")
        if {output, sidecar, validation_sidecar} & {args.v1.resolve(), args.v2.resolve()}:
            raise ToolError("Output files must be distinct from both inputs.")
        if args.offline and args.docintel_layout:
            raise ToolError("Document Intelligence Layout requires an Azure comparison.")
        settings = None if args.offline else load_settings()
        defaults = settings if settings is not None else load_job_defaults()
        options = ComparisonOptions(
            args.max_steps if args.max_steps is not None else defaults.max_agent_steps,
            args.max_cost_usd if args.max_cost_usd is not None else Decimal(str(defaults.max_job_cost_usd)),
            show_cosmetic=args.show_cosmetic, docintel_layout=args.docintel_layout,
            document_type=args.document_type, pressure_rules=args.pressure_rules, allow_sandbox=args.allow_sandbox,
        )
        args.job_dir.mkdir(parents=True, exist_ok=True)
        root = Path(mkdtemp(prefix="compare-", dir=args.job_dir.resolve()))
        trace_path = root / "trace.jsonl"
        with CloudServices(load_cloud_settings()) as cloud:
            tracer = cloud.tracer(trace_path, root.name)
            result = compare(args.v1, args.v2, root=root, options=options, tracer=tracer, settings=settings)
            if cloud.store:
                archive_comparison(cloud.store, root.name, (args.v1, args.v2), result, trace_path)
                tracer.record("archive")
        output.parent.mkdir(parents=True, exist_ok=True)
        _copy_output(result.changes, sidecar)
        _copy_output(result.pdf, output)
        if result.validation:
            _copy_output(result.validation, validation_sidecar)
    except (ArtifactArchiveError, BudgetExceeded, ConfigurationError, ModelRequestError, ToolError) as exc:
        print(str(exc), file=sys.stderr)
        if root is not None:
            print(f"Job trace: {root / 'trace.jsonl'}", file=sys.stderr)
        return 1
    except (OSError, ValueError, RuntimeError, InvalidOperation, pymupdf.FileDataError) as exc:
        print(f"Comparison failed ({type(exc).__name__}). Check inputs, settings, and destination.", file=sys.stderr)
        if root is not None:
            print(f"Job trace: {root / 'trace.jsonl'}", file=sys.stderr)
        return 1
    print(f"PDF: {output}\nChange log: {sidecar}\nJob trace: {root / 'trace.jsonl'}")
    if result.validation:
        print(f"Validation: {validation_sidecar}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
