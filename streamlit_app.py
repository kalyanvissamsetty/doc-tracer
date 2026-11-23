"""Streamlit UI for PDF comparisons and human review."""
import atexit
from decimal import Decimal
import json
from pathlib import Path
from time import time

import streamlit as st
from streamlit.errors import StreamlitSecretNotFoundError

from redline.comparison import ComparisonOptions, deployment_limits
from redline.cloud import CloudServices
from redline.config import ConfigurationError, JobDefaults, load_cloud_settings, load_job_defaults, load_settings
from redline.eval import run_suite
from redline.jobs import JobService, ReviewConflict, read_trace
from redline.models import ChangeLog, PageChanges
from redline.review import ReviewEdit, render_edited_page, render_preview
from redline.storage import LocalStorage
from redline.validation import ValidationReport


ROOT = Path(__file__).resolve().parent
GOLDEN = ROOT / "tests" / "golden"
SECTIONS = ("New comparison", "Review", "Jobs & traces")
TOOLS = {
    "discover": "Find candidate changes",
    "list_candidates": "Inspect change evidence",
    "read_candidate": "Read a pair of crops",
    "write_markup": "Write the proposed redline",
    "snap_to_cell": "Align a box to a detected table cell",
    "view_page": "Render a page for verification",
    "review_page": "Review the rendered page",
    "deliver": "Deliver the verified revision",
}
STATUS = {"queued": "Queued", "running": "Running", "completed": "Ready for review", "failed": "Failed"}


@st.cache_resource
def jobs() -> JobService:
    settings = load_cloud_settings(streamlit_secrets=_secrets())
    cloud = CloudServices(settings)
    try:
        service = JobService(settings.redline_data_dir, cloud=cloud)
    except Exception:
        cloud.close()
        raise
    atexit.register(service.close)
    return service


@st.cache_data(show_spinner=False, max_entries=64)
def preview(path: Path, page: int, box=None) -> bytes:
    return render_preview(path, page, box)



@st.cache_data(show_spinner=False, max_entries=64)
def preview_edited(path: Path, page_json: str, show_cosmetic: bool) -> tuple[bytes, list[int]]:
    return render_edited_page(path, PageChanges.model_validate_json(page_json), show_cosmetic=show_cosmetic)

def _chrome():
    st.markdown("""
<style>
.block-container {max-width: 1180px; padding-top: 2.15rem; padding-bottom: 4rem;}
[data-testid="stHeader"] {background: transparent;}
[data-testid="stAppViewContainer"] {background: #f5f7f6;}
[data-testid="stVerticalBlockBorderWrapper"] {background: #fff; border-color: #dfe7e5; border-radius: 15px;}
[data-testid="stFileUploaderDropzone"] {background: #edf3f1; border: 1px dashed #a0bcb6; border-radius: 11px;}
[data-testid="stCaptionContainer"] {color: #4e6561;}
.brand {display:flex; gap:11px; align-items:center; color:#152a29; font-weight:710; font-size:1.05rem; letter-spacing:-.025em;}
.mark {display:grid; place-items:center; width:34px; height:34px; border-radius:10px; background:#0b6960; color:white; flex-shrink:0;}
.hero {background: linear-gradient(116deg,#102f30 0%,#154341 70%,#1b5450 100%); border-radius:17px;
       padding:clamp(1.5rem,4vw,2.4rem); margin:1rem 0 .7rem; position:relative; overflow:hidden;}
.hero h1 {color:#fff; font-weight:690; font-size:clamp(2rem,4vw,3rem); letter-spacing:-.055em; padding:0; margin:0 0 .38rem;}
.hero p {color:#ddede8; margin:0; font-size:1rem; max-width:36rem; position:relative;}
.hero::after {content:''; position:absolute; inset:0 0 0 auto; width:30%; opacity:.32; pointer-events:none;
             background: repeating-linear-gradient(135deg,transparent,transparent 22px,#73a49d 23px,transparent 24px);
             mask-image:linear-gradient(90deg,transparent,#000);}
</style>
""", unsafe_allow_html=True)
    brand, nav = st.columns([1, 1.8], vertical_alignment="center")
    with brand:
        st.markdown("""<div class="brand"><span class="mark" aria-hidden="true">
        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.65"
        stroke-linecap="round" stroke-linejoin="round"><path d="M7 3h7l4 4v14H7zM14 3v5h4M10 12h5m-5 4h5"/></svg>
        </span>PDF Redline</div>""", unsafe_allow_html=True)
    with nav:
        for section, column in zip(SECTIONS, st.columns(3)):
            column.button(section, type="primary" if st.session_state.section == section else "secondary",
                          on_click=_navigate, args=(section,), width="stretch")


def _navigate(section: str, job_id: str | None = None):
    st.session_state.section = section
    if job_id is not None:
        st.session_state.active_job = job_id


def _hero(title: str, detail: str):
    st.markdown(f'<div class="hero"><h1>{title}</h1><p>{detail}</p></div>', unsafe_allow_html=True)


def _secrets():
    try:
        return st.secrets.to_dict()
    except (StreamlitSecretNotFoundError, FileNotFoundError):
        return None


def _job_name(job: dict) -> str:
    return f"{job['v1']} → {job['v2']} · {job['id'][:6]}"


def _elapsed(job: dict) -> str:
    if job["started"] is None:
        return "—"
    return f"{max(0, (job['finished'] or time()) - job['started']):.1f} s"


def _event_label(event: dict) -> str:
    kind = event["event"]
    if "tool" in event:
        return TOOLS.get(event["tool"], "Comparison tool")
    label = {
        "job_started": "Start comparison", "probe": "Inspect PDF",
        "model_call": "Ask the Azure model", "model_result": "Receive Azure result",
        "discovery": "Inventory candidate changes", "crop": "Save comparison crop",
        "ocr_call": "Read a scanned page with Layout OCR", "ocr_fallback": "Fall back to raster comparison",
        "drawing_comparison": "Compare drawing geometry",
        "validation": "Validate the revised pressure test report",
        "verified_delivery": "Confirm verified delivery", "job_completed": "Comparison complete",
        "job_failed": "Comparison stopped",
    }
    return label.get(kind, kind.replace("_", " ").capitalize())


@st.fragment(run_every=1)
def _progress(service: JobService, job_id: str):
    job = service.get(job_id)
    events = read_trace(service.directory(job) / "trace.jsonl")
    with st.container(border=True):
        st.subheader("Latest comparison")
        st.text(_job_name(job))
        if job["status"] == "failed":
            st.error(job["error"])
        elif job["status"] == "completed":
            st.success("Comparison ready")
            if st.button("Open review", type="primary", key="open-progress"):
                _navigate("Review", job_id)
                st.rerun()
        else:
            st.info(_event_label(events[-1]) if events else STATUS[job["status"]])
        step = max((event.get("step", 0) for event in events), default=0)
        tokens = sum(
            (event.get("input_tokens") or 0) + (event.get("output_tokens") or 0) for event in events
            if event["event"] == "model_result"
        )
        spend = next((event["spent_usd"] for event in reversed(events) if "spent_usd" in event), 0)
        for metric, (label, value) in zip(st.columns(3), (
            ("Step", step), ("Tokens", f"{tokens:,}"), ("Estimated cost", f"${spend:.4f}"),
        )):
            metric.metric(label, value)
        recent = [{"Step": item.get("step", "—"), "Activity": _event_label(item)}
                  for item in events if item["event"] == "tool_call"]
        if recent:
            st.dataframe(recent[-8:], hide_index=True, width="stretch")


def _submit(service: JobService, inputs, options: ComparisonOptions, settings=None):
    try:
        job_id, _ = service.submit(*inputs, options, settings)
    except (ConfigurationError, ValueError) as exc:
        st.error(str(exc))
        return
    except OSError:
        st.error("Could not save the uploads. Check that the job directory is writable.")
        return
    st.session_state.active_job = job_id
    st.rerun()


def new_comparison(service: JobService):
    _hero("Compare PDF revisions", "Upload the original and revised files to find changes and create a reviewable redline.")
    try:
        defaults = load_job_defaults(streamlit_secrets=_secrets())
    except ConfigurationError as exc:
        st.error(str(exc))
        defaults = JobDefaults()

    settings = None
    inputs, options_col = st.columns([1.72, 1], gap="large")
    with inputs, st.container(border=True):
        st.subheader("Upload PDFs")
        first, second = st.columns(2, gap="medium")
        original = first.file_uploader("Original · V1", type="pdf", key="upload-v1")
        revised = second.file_uploader("Revised · V2", type="pdf", key="upload-v2")
        hint = st.selectbox("Document type", ["Auto", "Form", "Table / BOM", "Field report", "Pressure test", "As-built"])
        sample = st.button("Run sample comparison", help="Compare a built-in pressure test form in local mode.")
    with options_col, st.container(border=True):
        st.subheader("Comparison settings")
        mode = st.selectbox("Mode", ["Local", "Azure agent"], help="Local mode can run without Azure credentials.")
        show = st.checkbox("Show cosmetic changes")
        summary = st.checkbox("Include a summary page")
        with st.expander("Step and cost limits"):
            steps = st.number_input("Maximum steps", min_value=1, value=defaults.max_agent_steps)
            cost = st.number_input("Maximum cost (USD)", min_value=0.0, value=float(defaults.max_job_cost_usd), format="%.4f")
            ocr = st.checkbox("Use Layout OCR for scanned tables and drawing crops", disabled=mode == "Local")
        if mode == "Azure agent":
            try:
                settings = load_settings(streamlit_secrets=_secrets())
                deployment_limits(settings)
                if ocr and settings.docintel_layout_usd_per_page is None:
                    raise ConfigurationError("Set DOCINTEL_LAYOUT_USD_PER_PAGE to enable metered OCR.")
            except ConfigurationError as exc:
                settings = None
                st.error(str(exc))
        submit = st.button(
            "Run comparison", type="primary", width="stretch",
            disabled=not (original and revised) or (mode == "Azure agent" and settings is None),
        )

    options = ComparisonOptions(int(steps), Decimal(str(cost)), show, summary, hint.lower(), ocr and mode != "Local")
    if submit:
        _submit(service, ((original.name, original.getvalue()), (revised.name, revised.getvalue())), options, settings)
    if sample:
        case = GOLDEN / "pressure-form"
        sample_options = ComparisonOptions(int(steps), Decimal(str(cost)), show, summary, "pressure test")
        _submit(service, tuple(((case / name).name, (case / name).read_bytes()) for name in ("v1.pdf", "v2.pdf")),
                sample_options)
    if st.session_state.get("active_job"):
        _progress(service, st.session_state.active_job)


def _selection(service: JobService, name: str, *, completed: bool = False) -> dict | None:
    jobs = [job for job in service.list_jobs() if not completed or job["status"] == "completed"]
    if not jobs:
        st.info("Completed comparisons will appear here." if completed else "Run a comparison to see its job record.")
        return None
    job_map = {job["id"]: job for job in jobs}
    current = st.session_state.get("active_job")
    job_id = st.selectbox("Comparison", list(job_map), index=list(job_map).index(current) if current in job_map else 0,
                          format_func=lambda option: _job_name(job_map[option]), key=f"{name}-job-{current}")
    return service.get(job_id)


def _update(service, job, action, change_id=None, edit=None):
    try:
        service.update_review(job["id"], job["revision"], action=action, change_id=change_id, edit=edit)
    except (ReviewConflict, ValueError) as exc:
        st.warning(str(exc))
        return
    st.rerun()


def _crop(column, title, path, page, box, message):
    with column:
        st.caption(title)
        if page is None or message:
            st.caption(message or "No corresponding page.")
            return
        st.image(preview(path, page, box), width="stretch")


def _review_change(service, job, group, change, folder):
    with st.container(border=True):
        st.markdown(f"**#{change.id} · {change.type.title()}**")
        st.text(change.field)
        st.caption(f"{change.confidence:.0%} confidence · {change.human_status.title()}")
        if change.needs_review:
            st.warning("Low confidence. Check both versions before accepting.")
        st.text(f"V1  {change.v1_value or '—'}")
        st.text(f"V2  {change.v2_value or '—'}")
        st.caption(change.evidence)
        if change.review_request_status:
            st.text(f"Review request: {change.review_request_status.replace('_', ' ')}")
        accept, reject = st.columns(2)
        if accept.button("Accept", key=f"accept-{job['id']}-{change.id}", width="stretch"):
            _update(service, job, "accept", change.id)
        if reject.button("Reject", key=f"reject-{job['id']}-{change.id}", width="stretch"):
            _update(service, job, "reject", change.id)

        if group.v2_page is not None and change.category == "value" and change.bbox_v2 is not None:
            if st.button("Snap box to detected cell", key=f"snap-{job['id']}-{change.id}",
                         help="Keeps the original evidence and resets accepted changes on this page for review."):
                _update(service, job, "snap_to_cell", change.id)

        with st.expander("Edit change"):
            with st.form(f"edit-{job['id']}-{change.id}-{job['revision']}"):
                field = st.text_input("Field", value=change.field)
                types = ["addition", "deletion", "modification"]
                kind = st.selectbox("Change type", types, index=types.index(change.type), format_func=str.title)
                before = st.text_input("Original value", value=change.v1_value or "")
                after = st.text_input("Revised value", value=change.v2_value or "")
                if st.form_submit_button("Save edit"):
                    if not field.strip() or len(field.strip()) > 300:
                        st.error("Enter a field name of 1 to 300 characters.")
                    else:
                        _update(service, job, "edit", change.id, ReviewEdit(
                            field=field.strip(), type=kind, v1_value=before or None, v2_value=after or None,
                        ))
            st.caption("Accept the edited change when it is ready to include.")

    paths = service.inputs(job)
    boxes_path = folder / "review-regions.json"
    boxes = json.loads(boxes_path.read_bytes()) if boxes_path.is_file() else {}
    old_box = boxes.get(str(change.id), change.bbox_v2)
    crop1, crop2 = st.columns(2, gap="medium")
    _crop(crop1, "Original crop · V1", paths[0], group.v1_page, tuple(old_box) if old_box else None,
          "No corresponding region." if old_box is None and change.category != "page" else None)
    _crop(crop2, "Revised crop · V2", paths[1], group.v2_page, change.bbox_v2, None)


def review_page(service: JobService):
    _hero("Review changes", "Check each proposed change, then export the approved redline.")
    job = _selection(service, "review", completed=True)
    if not job:
        return
    log = ChangeLog.model_validate_json(job["review_log"])
    folder = service.directory(job)
    changes = [c for page in log.pages for c in page.changes if job["show_cosmetic"] or c.category != "cosmetic"]
    accepted = sum(c.human_status == "accepted" for c in changes)
    st.caption(f"{len(changes)} proposed · {accepted} accepted")
    left, right = st.columns([1.25, 1], gap="large")
    with left, st.container(border=True):
        def page_title(index):
            group = log.pages[index]
            return f"V2 page {group.v2_page}" if group.v2_page is not None else f"Removed V1 page {group.v1_page}"

        page_index = st.selectbox("Page", range(len(log.pages)), format_func=page_title, key=f"page-{job['id']}")
        group = log.pages[page_index]
        if group.v2_page is None:
            st.caption("This removed page is listed in the change log.")
            st.image(preview(service.inputs(job)[0], group.v1_page), width="stretch")
        else:
            if any(edit.v2_page == group.v2_page and edit.actor == "reviewer" for edit in log.geometry_edits):
                marked, comment_only = preview_edited(
                    service.inputs(job)[1], group.model_dump_json(), bool(job["show_cosmetic"]),
                )
                st.caption("Geometry edited. Review the revised boxes before accepting.")
            else:
                marked = preview(folder / "redline.pdf", group.v2_page)
                comment_only = log.comment_only_tags.get(group.v2_page, [])
            st.image(marked, width="stretch")
            if comment_only:
                ids = ", ".join(f"#{number}" for number in comment_only)
                st.caption(f"See PDF annotation comments for {ids}. The page had no clear space for those tags.")
    with right:
        on_page = [c for c in group.changes if job["show_cosmetic"] or c.category != "cosmetic"]
        st.subheader("Proposed changes")
        if st.button("Accept all high-confidence", disabled=not any(
            c.human_status == "pending" and c.confidence >= 0.7 for c in changes
        )):
            _update(service, job, "accept_high")
        if on_page:
            by_id = {c.id: c for c in on_page}
            selected = st.selectbox("Change", list(by_id), format_func=lambda value: f"#{value} · {by_id[value].field}",
                                    key=f"change-{job['id']}-{page_index}")
            _review_change(service, job, group, by_id[selected], folder)
        else:
            st.info("No proposed changes on this page.")

    validation = folder / "validation.json"
    if validation.is_file():
        st.subheader("Validation")
        report = ValidationReport.model_validate_json(validation.read_bytes())
        labels = {
            "pressure_range": "Pressure range", "duration": "Test duration", "test_medium": "Test medium",
            "chart": "Chart attachment", "dates": "Date order", "signature": "Signature", "employee_id": "Employee ID",
        }
        states = {"pass": "Pass", "fail": "Fail", "review": "Needs review"}
        st.dataframe([
            {"Check": labels.get(item.rule, item.rule), "Result": states[item.status], "Finding": item.message,
             "V2 pages": ", ".join(map(str, item.pages)) or "—"}
            for item in report.findings
        ], hide_index=True, width="stretch")
        st.download_button("Download validation report", validation.read_bytes(),
                           file_name="validation.json", mime="application/json", on_click="ignore")

    st.subheader("Export")
    if st.button("Generate final PDF", type="primary"):
        try:
            service.generate_final(job["id"], job["revision"])
        except (ValueError, OSError, RuntimeError):
            st.error("Could not generate this review revision. Reload the job and try again.")
        else:
            st.rerun()
    revision = job["final_revision"]
    if revision is not None:
        first, second = st.columns(2)
        first.download_button("Download final PDF", (folder / f"final-{revision}.pdf").read_bytes(),
                              file_name="redline.pdf", mime="application/pdf", on_click="ignore", width="stretch")
        second.download_button("Download change log", (folder / f"final-{revision}.json").read_bytes(),
                               file_name="changes.json", mime="application/json", on_click="ignore", width="stretch")
    else:
        st.caption("The final PDF includes accepted changes only.")


def _history(service: JobService):
    records = service.list_jobs()
    if records:
        st.dataframe([{
            "Comparison": _job_name(job), "Status": STATUS[job["status"]], "Mode": job["mode"],
            "Duration": _elapsed(job), "Estimated cost": "—" if job["cost"] is None else f"${job['cost']:.4f}",
        } for job in records], hide_index=True, width="stretch")
    if any(job["status"] in {"running", "queued"} for job in records):
        st.button("Refresh jobs")
    selected = _selection(service, "trace")
    if not selected:
        return
    folder = service.directory(selected)
    events = read_trace(folder / "trace.jsonl")
    if selected["error"]:
        st.error(selected["error"])
    with st.expander("Agent trace", expanded=True):
        if not events:
            st.caption("No events recorded yet.")
            return
        index = st.selectbox("Event", range(len(events)), index=len(events) - 1,
                             format_func=lambda i: f"{i + 1}. {_event_label(events[i])} · {events[i]['event']}",
                             key=f"event-{selected['id']}")
        event = events[index]
        st.json(event, expanded=True)
        artifact = event.get("artifact_id")
        if artifact and artifact.endswith(".png"):
            store = LocalStorage(folder)
            baseline = event.get("outputs", {}).get("baseline_id")
            if baseline:
                first, second = st.columns(2)
                first.image(store.read(baseline), caption="Revised source", width=220)
                second.image(store.read(artifact), caption="Proposed markup", width=220)
            else:
                st.image(store.read(artifact), width=220)
        st.download_button("Download JSONL trace", (folder / "trace.jsonl").read_bytes(), mime="application/x-ndjson",
                           file_name="trace.jsonl", on_click="ignore")
    if selected["status"] == "completed" and st.button("Review this comparison"):
        _navigate("Review", selected["id"])
        st.rerun()


def jobs_page(service: JobService):
    _hero("Jobs & traces", "Inspect comparison history, agent activity, and evaluation results.")
    _history(service)
    st.subheader("Golden-set evaluation")
    st.caption("Run the local engine on labeled sample PDFs and score its detected changes.")
    if st.button("Run evaluation", type="primary"):
        try:
            with st.spinner("Comparing golden cases…"):
                st.session_state.evaluation = run_suite(GOLDEN)
        except (ValueError, OSError, RuntimeError):
            st.error("Evaluation failed. Check the golden case PDFs and labels.")

    report = st.session_state.get("evaluation")
    if report:
        def percent(value):
            return f"{value:.0%}" if value is not None else "—"

        st.dataframe([{
            "Document type": kind, "Recall": percent(stats["recall"]), "Precision": percent(stats["precision"]),
            "Type accuracy": percent(stats["type_accuracy"]), "Value accuracy": percent(stats["value_accuracy"]),
            "Sec / page": round(stats["seconds_per_page"], 3), "USD / page": stats["usd_per_page"],
        } for kind, stats in report["by_document_type"].items()], hide_index=True, width="stretch")
        st.download_button("Download evaluation report", json.dumps(report, indent=2),
                           mime="application/json", file_name="evaluation.json", on_click="ignore")


def main():
    st.set_page_config(page_title="PDF Redline", layout="wide")
    if "section" not in st.session_state:
        st.session_state.section = SECTIONS[0]
    _chrome()
    section = st.session_state.section
    render = {SECTIONS[0]: new_comparison, SECTIONS[1]: review_page, SECTIONS[2]: jobs_page}[section]
    try:
        service = jobs()
    except ConfigurationError as exc:
        st.error(str(exc))
        return
    render(service)


if __name__ == "__main__":
    main()
