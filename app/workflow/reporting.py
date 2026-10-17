"""Small application-owned final-report artifact/print journal.

report_print.json is separate from the authoritative tuning session, which is
never modified here. Atomic writes persist print intent BEFORE external I/O;
an intent without an outcome is uncertain after process restart. Callers MUST
serialize operations per run. This is not a persistent LangGraph checkpointer.
PDF validation requires the supplied PNG's exact decoded pixels to be painted
on the experiment-images page, fully inside it at full usable width. Human approval still
checks layout, occlusion, and scientific wording; this is not a semantic audit.
"""

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile
from uuid import uuid4

from PIL import Image

from app.print_tuning.final_figures import comparison_metadata, render_report_comparison as render_final_heatmaps
from app.report_policy import REPORT_PAGES, REPORT_ALLOWED_PAGES, REPORT_VERSION, COMPARISON_VERSION, COMPARISON_PAGE
from app.print_tuning.tuning_tools import _load_selection, _session_inputs


def _need(condition, message):
    if not condition:
        raise ValueError(message)


def _now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def load_record(run_dir):
    path = Path(run_dir) / "report_print.json"
    if not path.exists():
        return {"contract_version": 1, "figure": None, "figure_history": [], "pending": None,
                "attempts": [], "skip": None, "stopped": False}
    record = json.loads(path.read_text(encoding="utf-8"))
    _need(isinstance(record, dict) and record.get("contract_version") == 1
          and isinstance(record.get("attempts"), list), "Invalid report-print journal")
    return record


def save_record(run_dir, record):
    """Same-directory fsync and replace; serialize before touching the old record."""
    path = Path(run_dir) / "report_print.json"
    data = json.dumps(record, indent=2, ensure_ascii=False, allow_nan=False)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=path.name + ".", suffix=".tmp", delete=False) as file:
            temporary = Path(file.name)
            file.write(data)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return record


def prepare_figure(session_path, session, run_dir):
    """Reuse only an intact figure bound to completed settings and original data.

    The graph checks completion with its existing gate before calling. Missing,
    modified or stale figures are deliberately regenerated here, at most twice
    for a given input fingerprint. Old artifact records are retained.
    """
    inputs = _session_inputs(session, None)
    selection, _ = _load_selection(inputs)
    for item in selection["items"]:
        for panel in item["panels"]:
            _need(sha(panel["data_path"]) == panel["sha256"], "Original scientific array changed after selection")
    expected = comparison_metadata(session)
    from app.print_tuning import final_figures
    fingerprint = _digest({"report_version": REPORT_VERSION, "comparison": expected,
        "renderer_sha256": sha(final_figures.__file__),
        "session_id": session["session_id"], "run_id": session["run_id"],
        "selection_sha256": session["selection_sha256"], "config": session["config_snapshot"],
        "selected_experiments": session["selected_experiments"], "final_settings": session["final_settings"]})
    record = load_record(run_dir)
    old = record["figure"]
    if old and old.get("comparison_version") == COMPARISON_VERSION and old["inputs_sha256"] == fingerprint and Path(old["figure_path"]).is_file() \
            and sha(old["figure_path"]) == old["figure_sha256"] and old.get("session_id") == session["session_id"] \
            and all(old.get(key) == value for key, value in expected.items()):
        return deepcopy(old), False
    previous = sum(f["inputs_sha256"] == fingerprint for f in record["figure_history"])
    _need(previous < 3, "Final figure regeneration limit reached; inspect the artifacts before retrying")
    result = render_final_heatmaps(session_path, Path(run_dir) / "print_tuning")
    _need(result.get("status") == "ok" and sha(result["figure_path"]) == result["figure_sha256"],
          "Renderer returned an invalid final figure")
    _need(all(result.get(key) == value for key, value in expected.items()),
          "Rendered figure differs from frozen experiment order/settings/limits")
    figure = {**result, "inputs_sha256": fingerprint, "session_id": session["session_id"]}
    record.update(figure=figure, pending=None, skip=None)
    record["figure_history"].append(deepcopy(figure))
    save_record(run_dir, record)
    return deepcopy(figure), True


def inspect_tuned_report(path, figure):
    """Inspect real painted images, not PDF filenames, metadata, or prompt claims.

    Require exact RGB pixels, unchanged aspect ratio, full page containment,
    and full 180 mm figure width (10 pt labels). Reject recompression/resampling. The report
    script should insert the original PNG directly (or use interpolation='none').
    """
    from app.agents.report_and_print import inspect_report
    import pymupdf

    report = inspect_report(path)
    report.update(figure_embedded=False, figure_sha256=figure["figure_sha256"],
                  inputs_sha256=figure["inputs_sha256"])
    try:
        _need(figure.get("comparison_version") == COMPARISON_VERSION
              and figure.get("groups") == ["Before printer tuning", "After printer tuning"]
              and figure.get("columns") == ["Initial", "Final", "Initial", "Final"]
              and figure.get("minimum_font_pt", 0) >= 10,
              "Stale comparison layout; regenerate the four-panel figure")
        _need(sha(figure["figure_path"]) == figure["figure_sha256"], "Final figure changed")
        _need(report["exists"] and report["pages"] in REPORT_ALLOWED_PAGES, "report.pdf must have 2 or 3 pages")
        with Image.open(figure["figure_path"]) as source:
            expected = source.convert("RGB")
            size, pixels = expected.size, expected.tobytes()
        with pymupdf.open(path) as document:
            _need(all(abs(p.rect.width - 595.276) < 3 and abs(p.rect.height - 841.89) < 3 for p in document),
                  "Report must have A4 portrait pages")
            page = document[COMPARISON_PAGE]
            images = page.get_image_info(xrefs=True)
            _need(len(images) <= 64, "Too many PDF images to validate")
            for image in images:
                if not image.get("xref") or (image["width"], image["height"]) != size:
                    continue
                box = pymupdf.Rect(image["bbox"])
                # Subpixel rounding at 300 dpi changes the fitted width by <0.5 pt.
                if (not page.rect.contains(box) or box.width < figure["width_mm"] * 72 / 25.4 - .5
                        or box.height < page.rect.height * .25
                        or abs(box.width / box.height - size[0] / size[1]) > .01):
                    continue
                pixmap = pymupdf.Pixmap(document, image["xref"])
                if pixmap.alpha:
                    continue
                decoded = Image.frombytes("RGB" if pixmap.n == 3 else "L", size, pixmap.samples).convert("RGB")
                if decoded.tobytes() == pixels:
                    report["figure_embedded"] = True
                    break
        _need(report["figure_embedded"], "Report must visibly embed the supplied final tuned PNG unchanged")
    except (OSError, ValueError, RuntimeError) as error:
        report["error"] = str(error)
    return report


def require_report(report):
    _need(report.get("exists") and report.get("pages") in REPORT_ALLOWED_PAGES and report.get("figure_embedded")
          and not report.get("error"), "Invalid final report: " + str(report.get("error", "missing tuned figure")))


def request_revision(run_dir, instructions):
    """Persist exact human wording and invalidate approval, retaining all attempts."""
    _need(isinstance(instructions, str) and instructions.strip(), "Revision instructions are required")
    record = load_record(run_dir)
    previous = record.get("revision")
    revision = {"id": uuid4().hex, "instructions": instructions,
                "before_sha256": sha(Path(run_dir) / "report.pdf"), "requested_at": _now()}
    if previous:
        revision["prior_instructions"] = [*previous.get("prior_instructions", []), previous["instructions"]]
    record.update(revision=revision, pending=None, skip=None)
    save_record(run_dir, record)
    return deepcopy(revision)


def complete_revision(run_dir, revision_id, report, checks):
    record = load_record(run_dir)
    revision = record.get("revision")
    _need(revision and revision["id"] == revision_id, "Revision changed during generation")
    require_report(report)
    _need(report["sha256"] != revision["before_sha256"],
          "No content change was produced; revision remains pending. Correct the edits and retry.")
    record.setdefault("revisions", []).append({**revision, "after_sha256": report["sha256"],
        "checks": checks, "completed_at": _now(), "review": "Human review required; semantics not proven"})
    record.update(revision=None, pending=None, skip=None)
    save_record(run_dir, record)


def pending_request(run_dir, args, report, figure):
    _need(not load_record(run_dir).get("revision"), "Report revision is pending; rebuild before approval")
    require_report(report)
    args = deepcopy(args)
    _need(set(args) == {"pdf_path", "printer", "copies"} and type(args["copies"]) is int
          and args["copies"] == 1 and isinstance(args["printer"], str) and bool(args["printer"].strip()),
          "Report printing requires an exact printer and one copy")
    _need(Path(args["pdf_path"]).resolve() == (Path(run_dir) / "report.pdf").resolve(), "Submission must use this run's report.pdf")
    binding = {"args": args, "report_sha256": report["sha256"], "figure_sha256": figure["figure_sha256"],
               "inputs_sha256": figure["inputs_sha256"]}
    record = load_record(run_dir)
    previous = record["pending"]
    if previous and all(previous.get(k) == v for k, v in binding.items()):
        return deepcopy(previous)
    pending = {**binding, "attempt_id": uuid4().hex}
    record.update(pending=pending, skip=None)
    save_record(run_dir, record)
    return pending


def start_submission(run_dir, pending, report, figure):
    """Return (attempt, new_intent). False MUST never call the printer again."""
    record = load_record(run_dir)
    _need(not record.get("revision"), "Report revision is pending; physical submission is blocked")
    for attempt in record["attempts"]:
        if attempt["attempt_id"] == pending["attempt_id"]:
            _need(all(attempt[k] == pending.get(k) for k in ("args", "report_sha256", "figure_sha256", "inputs_sha256")),
                  "Conflicting report attempt replay")
            return deepcopy(attempt), False
    require_report(report)
    _need(record["pending"] == pending, "Pending report request or submission arguments changed; request fresh approval")
    _need(report["sha256"] == pending["report_sha256"] and figure["figure_sha256"] == pending["figure_sha256"]
          and figure["inputs_sha256"] == pending["inputs_sha256"], "Report inputs changed; request fresh approval")
    latest = record["attempts"][-1] if record["attempts"] else None
    _need(not latest or latest.get("resolution", {}).get("outcome") == "not_printed"
          or print_state(record) == "printed" and not attempt_matches(latest, report, figure),
          "Previous report submission must be resolved before submitting again")
    attempt = {**deepcopy(pending), "approved_at": _now(), "intent_at": _now(), "result": None,
               "readings": [], "resolution": {}}
    record["attempts"].append(attempt)
    record["pending"] = None
    save_record(run_dir, record)
    return deepcopy(attempt), True


def record_outcome(run_dir, attempt_id, result, *, reading=False):
    record = load_record(run_dir)
    attempt = next(a for a in record["attempts"] if a["attempt_id"] == attempt_id)
    if reading:
        _need(isinstance(attempt["result"], dict) and result.get("job_id") == attempt["result"].get("job_id")
              and bool(result.get("job_id")), "Wrong report job identity")
        if result not in attempt["readings"]:
            attempt["readings"].append(deepcopy(result))
    else:
        _need(attempt["result"] is None or attempt["result"] == result, "Conflicting report submission outcome")
        attempt["result"] = deepcopy(result)
    return save_record(run_dir, record)


def resolve_attempt(run_dir, outcome, *, source="human"):
    _need(outcome in ("printed", "not_printed", "stop"), "Invalid report print resolution")
    record = load_record(run_dir)
    attempt = record["attempts"][-1]
    old = attempt["resolution"]
    _need(not old or old["outcome"] == outcome, "Report print outcome already resolved")
    if not old:
        attempt["resolution"] = {"outcome": outcome, "source": source, "recorded_at": _now()}
    record["pending"] = None
    if outcome == "stop":
        record["stopped"] = True
    return save_record(run_dir, record)


def record_status_error(run_dir, job_id, result):
    """Bind an operational read failure by request, without inventing a result ID."""
    record = load_record(run_dir)
    attempt = record["attempts"][-1]
    _need(attempt["result"].get("job_id") == job_id and bool(job_id)
          and result.get("ok") is False and result.get("job_id") in (None, job_id), "Wrong status-error request identity")
    attempt["status_error"] = {"requested_job_id": job_id, "result": deepcopy(result)}
    return save_record(run_dir, record)


def print_state(record):
    """Journal status, with uncertainty taking precedence over apparent success."""
    if record.get("stopped"):
        return "stopped"
    if not record["attempts"]:
        return "new"
    attempt = record["attempts"][-1]
    resolution = attempt["resolution"].get("outcome")
    if resolution:
        return resolution
    if attempt.get("status_error"):
        return "uncertain"
    result = attempt["result"]
    reading = (attempt["readings"] or [result])[-1] or {}
    if result is None or result.get("tracked") is False or reading.get("state") == "uncertain" \
            or reading.get("confirmed") is False or reading.get("state") == "done" and reading.get("confirmed") is not True:
        return "uncertain"
    # MCP wraps connector exceptions as ok=false. Such a transport/operational
    # failure says nothing about whether a submission reached the printer.
    if result.get("ok") is False or reading.get("ok") is False:
        return "uncertain"
    if reading.get("state") == "failed":
        return "failed" if reading.get("confirmed") is True else "uncertain"
    if reading.get("state") == "done" and reading.get("confirmed") is True:
        return "printed"
    if not isinstance(result.get("job_id"), str) or not result["job_id"]:
        return "uncertain"
    return "pending"


def attempt_matches(attempt, report, figure):
    return (report.get("figure_embedded") and not report.get("error")
            and report.get("sha256") == attempt["report_sha256"]
            and figure["figure_sha256"] == attempt["figure_sha256"]
            and figure["inputs_sha256"] == attempt["inputs_sha256"])
