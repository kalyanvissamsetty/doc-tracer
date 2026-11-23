"""SQLite-backed comparison jobs, worker, and revisioned human review."""
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
import json
from pathlib import Path
import sqlite3
from time import time
from uuid import uuid4

from redline.agent.budget import BudgetExceeded
from redline.agent.tools import ToolError
from redline.cloud import CloudServices
from redline.comparison import ComparisonOptions, archive_comparison, compare, validate_options
from redline.config import ConfigurationError, Settings
from redline.llm.azure_client import ModelRequestError
from redline.models import ChangeLog
from redline.review import ReviewEdit, find_change, write_final
from redline.storage import LocalStorage
from redline.storage.blob import ArtifactArchiveError, archive_artifacts
from redline.toolkit.geometry import snap_to_detected_cell
from redline.tracing import JsonlTracer


class ReviewConflict(ValueError):
    pass


def read_trace(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    events = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue  # The worker may still be writing the last line.
    return events


def _name(filename: str, default: str) -> str:
    name = filename.replace("\\", "/").split("/")[-1].replace("\x00", "").strip()
    return name[:220] if name and name not in {".", ".."} else default


class JobService:
    """One local process owns the worker pool. Each operation uses its own SQLite connection."""

    def __init__(self, root: Path, cloud: CloudServices | None = None):
        self.cloud = cloud
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.db = self.root / "jobs.sqlite3"
        with self._connect() as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("""CREATE TABLE IF NOT EXISTS jobs (
                id TEXT PRIMARY KEY, v1 TEXT NOT NULL, v2 TEXT NOT NULL,
                status TEXT NOT NULL, mode TEXT NOT NULL, document_type TEXT NOT NULL,
                created REAL NOT NULL, started REAL, finished REAL, error TEXT,
                cost REAL, show_cosmetic INTEGER NOT NULL, summary INTEGER NOT NULL,
                review_log TEXT, revision INTEGER NOT NULL DEFAULT 0, final_revision INTEGER
            )""")
            conn.execute("""UPDATE jobs SET status='failed', finished=?, error=?
                WHERE status IN ('running','queued')""", (time(), "The app restarted before this job completed."))
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="redline")

    @contextmanager
    def _connect(self):
        conn = sqlite3.connect(self.db, timeout=15)
        conn.row_factory = sqlite3.Row
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    def close(self):
        self._pool.shutdown(wait=True)
        if self.cloud is not None:
            self.cloud.close()

    def get(self, job_id: str) -> dict:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
        if row is None:
            raise ValueError("Comparison job was not found.")
        return dict(row)

    def list_jobs(self) -> list[dict]:
        with self._connect() as conn:
            return [dict(row) for row in conn.execute("""
                SELECT id, v1, v2, status, mode, document_type, created, started, finished, cost
                FROM jobs ORDER BY created DESC
            """)]

    def directory(self, job: dict) -> Path:
        return self.root / job["id"]

    def inputs(self, job: dict) -> tuple[Path, Path]:
        directory = self.directory(job)
        return directory / "input-v1" / job["v1"], directory / "input-v2" / job["v2"]

    def submit(
        self, v1: tuple[str, bytes], v2: tuple[str, bytes], options: ComparisonOptions,
        settings: Settings | None = None,
    ) -> tuple[str, Future]:
        validate_options(options, settings)
        job_id = uuid4().hex
        directory = self.root / job_id
        directory.mkdir(mode=0o700)
        storage = LocalStorage(directory)
        names = _name(v1[0], "v1.pdf"), _name(v2[0], "v2.pdf")
        for index, (name, content) in enumerate(zip(names, (v1[1], v2[1])), 1):
            storage.write(f"input-v{index}/{name}", content)
        with self._connect() as conn:
            conn.execute("""INSERT INTO jobs
                (id,v1,v2,status,mode,document_type,created,show_cosmetic,summary)
                VALUES (?,?,?,?,?,?,?,?,?)""",
                (job_id, *names, "queued", "Azure" if settings else "Local", options.document_type,
                 time(), options.show_cosmetic, options.summary_page))
        return job_id, self._pool.submit(self._run, job_id, options, settings)

    def _run(self, job_id: str, options: ComparisonOptions, settings: Settings | None) -> None:
        job = self.get(job_id)
        directory = self.directory(job)
        trace_path = directory / "trace.jsonl"
        tracer = self.cloud.tracer(trace_path, job_id) if self.cloud else JsonlTracer(trace_path)
        with self._connect() as conn:
            conn.execute("UPDATE jobs SET status='running',started=? WHERE id=?", (time(), job_id))
        try:
            result = compare(*self.inputs(job), root=directory, options=options, settings=settings, tracer=tracer)
            if self.cloud and self.cloud.store:
                archive_comparison(self.cloud.store, job_id, self.inputs(job), result, trace_path)
                tracer.record("archive")
            log = ChangeLog.model_validate_json(result.changes.read_bytes())
            with self._connect() as conn:
                conn.execute("""UPDATE jobs SET status='completed',finished=?,cost=?,review_log=? WHERE id=?""",
                             (time(), log.cost.estimated_usd, log.model_dump_json(), job_id))
        except Exception as exc:
            events = read_trace(trace_path)
            cost = next((item["spent_usd"] for item in reversed(events) if "spent_usd" in item), 0)
            if isinstance(exc, (BudgetExceeded, ConfigurationError, ModelRequestError, ToolError, ArtifactArchiveError)):
                message = str(exc)
            else:
                message = "Comparison failed. Check that both uploads are readable, unlocked PDFs."
            with self._connect() as conn:
                conn.execute("UPDATE jobs SET status='failed', finished=?, error=?, cost=? WHERE id=?",
                             (time(), message, cost, job_id))

    def _lock_review(self, conn: sqlite3.Connection, job_id: str, revision: int) -> tuple[dict, ChangeLog]:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute("SELECT * FROM jobs WHERE id=? AND status='completed'", (job_id,)).fetchone()
        if row is None:
            raise ValueError("This job is not ready for review.")
        if row["revision"] != revision:
            raise ReviewConflict("The review changed in another session. Review the current values and try again.")
        return dict(row), ChangeLog.model_validate_json(row["review_log"])

    def update_review(
        self, job_id: str, revision: int, *, action: str, change_id: int | None = None,
        edit: ReviewEdit | None = None,
    ) -> None:
        with self._connect() as conn:
            job, log = self._lock_review(conn, job_id, revision)
            if action == "accept_high":
                for page in log.pages:
                    for change in page.changes:
                        if (
                            change.human_status == "pending" and change.confidence >= 0.7
                            and (job["show_cosmetic"] or change.category != "cosmetic")
                        ):
                            change.human_status = "accepted"
            else:
                change = find_change(log, change_id)
                if action in {"accept", "reject"}:
                    change.human_status = "accepted" if action == "accept" else "rejected"
                elif action == "edit" and edit is not None:
                    for name, value in edit.model_dump().items():
                        setattr(change, name, value)
                    change.human_status = "edited"
                elif action == "snap_to_cell":
                    group = next(group for group in log.pages if any(item.id == change_id for item in group.changes))
                    if group.v2_page is None:
                        raise ValueError("This change has no revised page.")
                    geometry = snap_to_detected_cell(self.inputs(job)[1], group.v2_page, change, actor="reviewer")
                    change.bbox_v2 = geometry.after
                    log.geometry_edits.append(geometry)
                    for page in log.pages:
                        if page.v2_page == group.v2_page:
                            for item in page.changes:
                                if item.human_status == "accepted":
                                    item.human_status = "pending"
                    change.human_status = "edited"
                else:
                    raise ValueError("Unknown review action.")
            conn.execute("UPDATE jobs SET review_log=?,revision=revision+1,final_revision=NULL WHERE id=?",
                         (log.model_dump_json(), job_id))

    def generate_final(self, job_id: str, revision: int) -> None:
        with self._connect() as conn:
            job, log = self._lock_review(conn, job_id, revision)
            directory = self.directory(job)
            output = directory / f"final-{revision}.pdf"
            if job["final_revision"] == revision and output.is_file():
                return
            log.comment_only_tags = write_final(self.inputs(job)[1], log, output, summary=bool(job["summary"]))
            json_name = f"final-{revision}.json"
            LocalStorage(directory).write(json_name, log.model_dump_json(indent=2).encode())
            if self.cloud and self.cloud.store:
                archive_artifacts(self.cloud.store, f"jobs/{job_id}/reviews/{revision}", {
                    "redline.pdf": output, "changes.json": directory / json_name,
                })
            conn.execute("UPDATE jobs SET final_revision=? WHERE id=?", (revision, job_id))
