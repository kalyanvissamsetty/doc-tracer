"""Check documented pressure test evidence against explicitly configured rules."""
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
import re
import sysconfig

import pymupdf
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator
import yaml

from redline.config import ConfigurationError
from redline.toolkit.values import normalize_value
from redline.validation import Finding, ValidationReport

RULES_FILE = Path("config/pressure_test_rules.yaml")
PRESSURE_UNITS = {
    "pa": Decimal("0.001"), "kpa": Decimal(1), "mpa": Decimal(1000),
    "bar": Decimal(100), "psi": Decimal("6.894757293168"),
}
DURATION_UNITS = {
    "s": Decimal(1), "sec": Decimal(1), "second": Decimal(1), "seconds": Decimal(1),
    "min": Decimal(60), "h": Decimal(3600),
}
REQUIRED_KEYS = {
    "pressure", "pressure_min", "pressure_max", "duration", "medium",
    "chart_required", "start", "end", "signed", "signature", "employee",
}
EMPTY = {"", "-", "—", "n/a", "na", "none", "null", "not provided", "not signed", "unsigned", "pending"}


def _quantity(value: str | None, units: dict[str, Decimal]) -> Decimal | None:
    if not value:
        return None
    text = normalize_value(value)
    if units is DURATION_UNITS:
        clock = re.fullmatch(r"(\d+):([0-5]\d)(?::([0-5]\d))?", text)
        if clock:
            hours, minutes, seconds = clock.groups()
            return Decimal(int(hours) * 3600 + int(minutes) * 60 + int(seconds or 0))
    match = re.fullmatch(r"([+-]?(?:\d+(?:\.\d+)?|\.\d+))\s*([a-z]+)", text)
    if not match:
        return None
    number, unit = match.groups()
    factor = units.get(unit)
    return Decimal(number) * factor if factor is not None else None


class PressureRules(BaseModel):
    model_config = ConfigDict(extra="forbid")
    minimum_pressure: str | None = None
    maximum_pressure: str | None = None
    minimum_duration: str | None = None
    fields: dict[str, list[str]]
    chart_identifiers: list[str] = Field(min_length=1)

    @model_validator(mode="after")
    def check_values(self):
        if set(self.fields) != REQUIRED_KEYS or any(not aliases or any(not _label(s) for s in aliases)
                                                   for aliases in self.fields.values()):
            raise ValueError("Supply nonempty aliases for every pressure-test field.")
        if any(not _label(name) for name in self.chart_identifiers):
            raise ValueError("Supply nonempty chart identifiers.")
        lower = _quantity(self.minimum_pressure, PRESSURE_UNITS)
        upper = _quantity(self.maximum_pressure, PRESSURE_UNITS)
        duration = _quantity(self.minimum_duration, DURATION_UNITS)
        if any(setting is not None and (result is None or result < 0) for setting, result in (
            (self.minimum_pressure, lower), (self.maximum_pressure, upper), (self.minimum_duration, duration),
        )):
            raise ValueError("Use numeric thresholds with supported units.")
        if lower is not None and upper is not None and lower > upper:
            raise ValueError("Minimum pressure must not exceed maximum pressure.")
        return self


def load_pressure_rules(path: Path | None = None) -> PressureRules:
    if path is None:
        candidates = (RULES_FILE, Path(__file__).resolve().parents[2] / RULES_FILE,
                      Path(sysconfig.get_path("data")) / RULES_FILE)
        path = next((candidate for candidate in candidates if candidate.is_file()), RULES_FILE)
    try:
        return PressureRules.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
    except (OSError, UnicodeError, ValidationError, yaml.YAMLError):
        raise ConfigurationError("Check config/pressure_test_rules.yaml for field aliases and valid limits with units.") from None


def _label(text: str) -> str:
    return " ".join(re.sub(r"[\W_]+", " ", text.casefold()).split())


def _date(text: str | None):
    if not text:
        return None
    value = normalize_value(text)
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}(?:[t ]\d{2}:\d{2}(?::\d{2})?(?:z|[+-]\d{2}:\d{2})?)?", value):
        return None
    try:
        return datetime.fromisoformat(value.upper())
    except ValueError:
        return None


@dataclass(frozen=True)
class Evidence:
    value: str | None
    pages: tuple[int, ...]
    ambiguous: bool = False


class PressureTestValidator:
    def __init__(self, rules: PressureRules):
        self.rules = rules
        self.aliases = {_label(alias): name for name, aliases in rules.fields.items() for alias in aliases}
        self.chart_names = [_label(text) for text in rules.chart_identifiers]

    def _facts(self, path: Path):
        facts = defaultdict(list)
        chart_pages = set()
        announced = False
        with pymupdf.open(path) as doc:
            embedded_chart = any(self._chart(name) for name in doc.embfile_names())
            for page in doc:
                number = page.number + 1
                lines = [line.strip() for line in page.get_text(sort=True).splitlines() if line.strip()]
                announced |= any("pressure test" in _label(line) for line in lines[:12])
                widgets = list(page.widgets() or ())
                for widget in widgets:
                    field = self.aliases.get(_label(widget.field_name or ""))
                    if widget.field_type == pymupdf.PDF_WIDGET_TYPE_SIGNATURE:
                        field = field or "signature"
                        kind, value = doc.xref_get_key(widget.xref, "V")
                        signed = (kind == "xref" or kind == "dict") and value != "null"
                        facts[field].append(("Signed field" if signed else "", number))
                    elif field:
                        facts[field].append((str(widget.field_value or ""), number))
                widget_fields = {self.aliases.get(_label(widget.field_name or "")) for widget in widgets}
                for i, line in enumerate(lines):
                    parts = re.split(r"\s*[:=]\s*|\s{2,}", line, maxsplit=1)
                    field = self.aliases.get(_label(parts[0]))
                    if not field or field in widget_fields:
                        continue
                    if len(parts) == 2:
                        value = parts[1]
                    else:
                        value = lines[i + 1] if i + 1 < len(lines) else ""
                        if self.aliases.get(_label(re.split(r"[:=]", value, maxsplit=1)[0])):
                            value = ""
                    facts[field].append((value, number))
                headings = [_label(line) for line in lines[:5] if ":" not in line and "=" not in line]
                if any(heading in self.chart_names for heading in headings):
                    if page.get_image_info() or len(page.get_cdrawings()) >= 3:
                        chart_pages.add(number)
                for mark in page.annots() or ():
                    if mark.type[1] == "FileAttachment" and any(self._chart(str(name)) for name in mark.file_info.values()):
                        chart_pages.add(number)
        evidence = {}
        for name, members in facts.items():
            unique = {normalize_value(value): value for value, _ in members if normalize_value(value) not in EMPTY}
            evidence[name] = Evidence(next(iter(unique.values())) if len(unique) == 1 else None,
                                      tuple(sorted({page for _, page in members})), len(unique) > 1)
        return evidence, chart_pages, embedded_chart, announced

    def _chart(self, text: str) -> bool:
        normalized = _label(text)
        return any(re.search(rf"\b{re.escape(name)}\b", normalized) for name in self.chart_names)

    def validate(self, path: Path, *, explicit: bool = False) -> ValidationReport | None:
        facts, chart_pages, embedded_chart, announced = self._facts(path)
        if not explicit and not announced and not ("pressure" in facts and len(facts) >= 3):
            return None
        findings = []

        def evidence(name):
            return facts.get(name, Evidence(None, ()))

        def get(name):
            return evidence(name).value

        def add(rule, status, message, *fields, extra=()):
            pages = {page for field in fields for page in evidence(field).pages}
            findings.append(Finding(rule=rule, status=status, message=message, pages=sorted(pages | set(extra))))

        evidence_found = bool(facts)
        raw_pressure = get("pressure")
        raw_minimum = self.rules.minimum_pressure
        raw_maximum = self.rules.maximum_pressure
        measured = _quantity(raw_pressure, PRESSURE_UNITS)
        lower = _quantity(raw_minimum, PRESSURE_UNITS)
        upper = _quantity(raw_maximum, PRESSURE_UNITS)
        if None in (measured, lower, upper) or lower > upper:
            add("pressure_range", "review", "A readable test pressure and configured minimum and maximum pressures with units are required.",
                "pressure", "pressure_min", "pressure_max")
        else:
            passed = lower <= measured <= upper
            add("pressure_range", "pass" if passed else "fail",
                f"Test pressure is {raw_pressure.strip()} against an allowable range of "
                f"{raw_minimum.strip()} to {raw_maximum.strip()}.",
                "pressure", "pressure_min", "pressure_max")

        duration = _quantity(get("duration"), DURATION_UNITS)
        minimum = _quantity(self.rules.minimum_duration, DURATION_UNITS)
        if minimum is None or duration is None:
            add("duration", "review", "A configured minimum and a readable duration with units are required.", "duration")
        else:
            passed = duration >= minimum
            add("duration", "pass" if passed else "fail",
                f"Recorded duration is {get('duration').strip()}. Configured minimum is {self.rules.minimum_duration.strip()}.",
                "duration")

        required = normalize_value(get("chart_required") or "")
        if required in {"no", "false", "off", "0", "not required"}:
            add("chart", "pass", "The report states that a chart is not required.", "chart_required")
        elif required in {"yes", "true", "on", "1", "required"}:
            attached = bool(chart_pages or embedded_chart)
            add("chart", "pass" if attached else "fail", "A required chart was found." if attached
                else "A required chart could not be located among the report pages or attachments.",
                "chart_required", extra=chart_pages)
        else:
            add("chart", "review", "The chart requirement could not be read.", "chart_required")

        start, end, signed = (_date(get(key)) for key in ("start", "end", "signed"))
        if start is None or end is None or evidence("signed").ambiguous or (get("signed") and signed is None):
            add("dates", "review", "Start, end, or supplied signature dates were missing or ambiguous.", "start", "end", "signed")
        else:
            dates = [value for value in (start, end, signed) if value is not None]
            if any(value.tzinfo is not None for value in dates) and not all(value.tzinfo is not None for value in dates):
                add("dates", "review", "Dates mix local times and explicit time zones.", "start", "end", "signed")
            else:
                comparable = [value.astimezone(timezone.utc) if value.tzinfo else value for value in dates]
                ordered = comparable == sorted(comparable)
                add("dates", "pass" if ordered else "fail", "Test dates are in order." if ordered
                    else "Test dates are out of order.", "start", "end", "signed")

        for source, rule, name in (("medium", "test_medium", "Test medium"), ("signature", "signature", "Signature"),
                                   ("employee", "employee_id", "Employee ID")):
            present = bool(get(source))
            if evidence(source).ambiguous:
                add(rule, "review", f"Conflicting values were found for {name.lower()}.", source)
            else:
                add(rule, "pass" if present else "fail" if evidence_found else "review",
                    f"{name} is present." if present else f"{name} could not be confirmed.", source)
        return ValidationReport(plugin="pressure_test", findings=findings)
