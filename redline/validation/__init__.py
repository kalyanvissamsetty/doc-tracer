"""Report validation plugin contract."""
from pathlib import Path
from typing import Literal, Protocol

from pydantic import BaseModel


class Finding(BaseModel):
    rule: str
    status: Literal["pass", "fail", "review"]
    message: str
    pages: list[int]


class ValidationReport(BaseModel):
    plugin: str
    findings: list[Finding]


class ValidationPlugin(Protocol):
    def validate(self, path: Path, *, explicit: bool = False) -> ValidationReport | None: ...
