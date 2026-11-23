"""Job tracing contract and local JSONL writer."""
from datetime import datetime, timezone
import json
from pathlib import Path
from threading import Lock
from typing import Protocol


class Tracer(Protocol):
    def record(self, event: str, **data: object) -> None: ...


class JsonlTracer:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()

    def record(self, event: str, **data: object) -> None:
        record = {"time": datetime.now(timezone.utc).isoformat(), "event": event, **data}
        with self._lock, self.path.open("a", encoding="utf-8") as output:
            output.write(json.dumps(record, ensure_ascii=True) + "\n")


class CompositeTracer:
    def __init__(self, *tracers: Tracer):
        self.tracers = tracers

    def record(self, event: str, **data: object) -> None:
        for tracer in self.tracers:
            tracer.record(event, **data)
