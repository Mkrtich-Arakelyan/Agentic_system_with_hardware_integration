"""api_client against the in-memory mock API, through FastAPI's TestClient (no network, no models)."""
from pathlib import Path

from fastapi.testclient import TestClient
import pytest

from frontend.api_client import ApiClient, ApiError, find_preview_files, relative_run_path
from frontend.mock_api import create_app

ALL_KINDS = {"budget", "tuning_printer", "trial_print_approval", "tuning_print_confirmation", "tuning_scan",
             "tuning_poll", "tuning_problem", "report_workflow_problem", "print_approval", "printer_problem",
             "report_print_confirmation"}
PNG = b"\x89PNG\r\n\x1a\n" + b"\0" * 64


def make_client(tmp_path, step_seconds=0.0):
    app = create_app(tmp_path / "runs", step_seconds=step_seconds)
    return ApiClient("http://testserver", http=TestClient(app)), app


@pytest.fixture
def api(tmp_path):
    return make_client(tmp_path)[0]


def start(api, **options):
    return api.start_session(request=options.pop("request", "Compare four experiments."), fake_printer=True, **options)


def pending(api, session_id):
    session = api.get_session(session_id)
    assert session["status"] == "waiting_for_human", session
    return session["pending"]


def walk_to(api, session_id, kind, seen=None):
    """Answer valid defaults until the given pause kind is pending; return its payload."""
    while True:
        payload = pending(api, session_id)
        if payload["kind"] == kind:
            return payload
        if seen is not None:
            seen.append(payload["kind"])
        api.answer(session_id, default_answer(api, session_id, payload))


def default_answer(api, session_id, payload):
    kind = payload["kind"]
    if kind == "budget":
        return "yes keep the report short"
    if kind == "tuning_printer":
        return {"action": "printer", "printer": api.printers()[0]}
    if kind == "tuning_scan":
        paths = api.upload_scans(session_id, [("page_1.png", PNG, "image/png"), ("page_2.png", PNG, "image/png")])
        return {"action": "scan", "scan_paths": paths}
    if kind == "print_approval":
        return "yes"
    if kind == "printer_problem":
        return "retry"
    return {"action": {"trial_print_approval": "approve", "tuning_poll": "check", "tuning_problem": "retry",
                       "report_workflow_problem": "retry"}.get(kind, "printed")}


def test_full_run_reaches_every_pause_and_finishes(api):
    session_id = start(api)
    assert any(row["session_id"] == session_id for row in api.list_sessions())
    kinds, scan_rounds, approvals = [], [], []
    change_requested = corrected = False
    while True:
        session = api.get_session(session_id)
        if session["status"] != "waiting_for_human":
            break
        payload = session["pending"]
        kinds.append(payload["kind"])
        if payload["kind"] == "tuning_scan":
            scan_rounds.append(payload["round_id"])
        if payload["kind"] == "trial_print_approval":
            approvals.append(payload["round_id"])
            assert payload["page_count"] == 2 and payload["print_settings"]["paper"] == "A4"
            relative = relative_run_path(payload["pdf_path"], session["run_dir"])
            assert api.download_file(session_id, relative).startswith(b"%PDF")
        if payload["kind"] == "print_approval" and not change_requested:
            change_requested = True          # a change request leads to report_workflow_problem with "correct"
            api.answer(session_id, "Make the verdict one sentence.")
            continue
        if payload["kind"] == "report_workflow_problem" and "correct" in payload["choices"]:
            corrected = True
            api.answer(session_id, {"action": "correct", "instructions": "Shorter verdict, keep the graph."})
            continue
        api.answer(session_id, default_answer(api, session_id, payload))

    assert session["status"] == "finished" and session["stop_reason"] == "finished"
    assert session["pending"] is None
    assert set(kinds) == ALL_KINDS
    assert scan_rounds == [1, 2, 3] and approvals == [1, 2, 3]
    assert change_requested and corrected
    roles = {turn["role"] for turn in session["dialogue"]}
    assert {"orchestrator", "scientist", "executor", "general", "photo_quality_specialist", "human",
            "system"} <= roles
    assert {"run_focusing_experiment", "make_trial_sheet", "read_scan", "submit_print"} <= {
        record["tool_name"] for record in session["execution_results"]}
    assert all(set(record) == {"tool_name", "tool_args", "result"} for record in session["execution_results"])
    assert api.download_file(session_id, "report.pdf").startswith(b"%PDF")
    previews = find_preview_files(session["execution_results"], session["run_dir"])
    assert any(path.endswith(".png") for path in previews)
    assert api.download_file(session_id, previews[0])


def test_answer_is_refused_when_nothing_is_pending(tmp_path):
    api, _ = make_client(tmp_path, step_seconds=60)      # the agents are still "working"
    session_id = start(api)
    assert api.get_session(session_id)["status"] == "running"
    with pytest.raises(ApiError) as error:
        api.answer(session_id, "yes")
    assert error.value.status == 409 and "nothing is waiting" in str(error.value)


def test_answer_is_refused_after_the_run_stopped(api):
    session_id = start(api)
    api.answer(session_id, "no")
    session = api.get_session(session_id)
    assert (session["status"], session["stop_reason"], session["pending"]) == ("stopped", "stopped_by_human", None)
    with pytest.raises(ApiError) as error:
        api.answer(session_id, "yes")
    assert error.value.status == 409


@pytest.mark.parametrize("kind, answer, message", [
    ("budget", {"action": "yes"}, "budget takes a text"),
    ("budget", "maybe", "budget takes a text"),
    ("tuning_printer", {"action": "printer", "printer": "  "}, "exact printer name"),
    ("trial_print_approval", {"action": "print it"}, "action must be one of approve, stop"),
    ("tuning_scan", {"action": "scan", "scan_paths": []}, "non-empty list"),
    ("tuning_scan", {"action": "scan", "scan_paths": ["/etc/hosts"]}, "not an uploaded scan of round 1"),
    ("print_approval", "   ", "print_approval takes"),
    ("printer_problem", {"action": "retry"}, 'printer_problem takes the text "retry" or "skip"'),
])
def test_invalid_answers_get_422_with_a_message(api, kind, answer, message):
    session_id = start(api)
    walk_to(api, session_id, kind)
    with pytest.raises(ApiError) as error:
        api.answer(session_id, answer)
    assert error.value.status == 422 and message in str(error.value)
    assert pending(api, session_id)["kind"] == kind          # still waiting for a valid answer


def test_upload_scans_into_the_active_round(api):
    session_id = start(api)
    with pytest.raises(ApiError) as error:                   # no tuning round yet
        api.upload_scans(session_id, [("scan.png", PNG, "image/png")])
    assert error.value.status == 409
    walk_to(api, session_id, "tuning_scan")
    run_dir = Path(api.get_session(session_id)["run_dir"])
    paths = api.upload_scans(session_id, [("scan.png", PNG, "image/png"), ("scan.png", PNG, "image/png"),
                                          ("../../evil name.pdf", b"%PDF-1.4", "application/pdf")])
    folder = run_dir / "scans" / "round_1"
    assert [Path(p).name for p in paths] == ["scan.png", "scan_2.png", "evil name.pdf"]
    assert all(Path(p).parent == folder and Path(p).read_bytes() for p in paths)
    with pytest.raises(ApiError) as error:                   # refused before sending
        api.upload_scans(session_id, [("notes.txt", b"x", "text/plain")])
    assert "Not a scan file" in str(error.value)
    response = api.http.post(f"/api/sessions/{session_id}/scans", files=[("files", ("notes.txt", b"x"))])
    assert response.status_code == 422                       # and refused by the server too
    api.answer(session_id, {"action": "scan", "scan_paths": paths})
    assert pending(api, session_id)["kind"] == "trial_print_approval"
    assert pending(api, session_id)["round_id"] == 2


def test_download_files_only_inside_the_run_folder(api, tmp_path):
    session_id = start(api)
    payload = walk_to(api, session_id, "trial_print_approval")
    run_dir = api.get_session(session_id)["run_dir"]
    assert api.download_file(session_id, relative_run_path(payload["pdf_path"], run_dir)).startswith(b"%PDF")
    (tmp_path / "runs" / "secret.txt").write_text("not yours")
    for outside in ("../secret.txt", "print_tuning/../../secret.txt", str(tmp_path / "runs" / "secret.txt")):
        with pytest.raises(ApiError) as error:
            api.download_file(session_id, outside)
        assert error.value.status == 403, outside
        assert "outside the run folder" in str(error.value)
    with pytest.raises(ApiError) as error:
        api.download_file(session_id, "print_tuning/missing.pdf")
    assert error.value.status == 404


def test_reprint_and_not_printed_ask_for_a_fresh_approval(api):
    session_id = start(api)
    walk_to(api, session_id, "tuning_print_confirmation")
    api.answer(session_id, {"action": "not_printed"})
    assert pending(api, session_id)["kind"] == "trial_print_approval"
    walk_to(api, session_id, "tuning_scan")
    api.answer(session_id, {"action": "reprint"})
    assert pending(api, session_id)["kind"] == "trial_print_approval"
    assert pending(api, session_id)["round_id"] == 1


def test_stop_and_skip_end_the_run(api):
    stopped = start(api)
    walk_to(api, stopped, "tuning_poll")
    api.answer(stopped, {"action": "stop"})
    assert api.get_session(stopped)["status"] == "stopped"
    skipped = start(api)
    walk_to(api, skipped, "printer_problem")
    api.answer(skipped, "skip")
    session = api.get_session(skipped)
    assert (session["status"], session["stop_reason"]) == ("finished", "finished")


def test_failed_run_and_unknown_session(api):
    session_id = start(api, request="Compare four experiments [fail]")
    session = api.get_session(session_id)
    assert (session["status"], session["pending"]) == ("failed", None)
    statuses = {row["session_id"]: row["status"] for row in api.list_sessions()}
    assert statuses[session_id] == "failed"
    with pytest.raises(ApiError) as error:
        api.get_session("no-such-session")
    assert error.value.status == 404


def test_events_stream_of_a_finished_run(api):
    session_id = start(api)
    api.answer(session_id, "no")
    with api.http.stream("GET", f"/api/sessions/{session_id}/events") as response:
        body = "".join(response.iter_text())
    assert "event: dialogue" in body and "event: tool" in body
    assert body.rstrip().splitlines()[-2] == "event: end"


def test_start_and_connection_errors_are_clear_messages(api):
    with pytest.raises(ApiError) as error:
        api.start_session(request="   ")
    assert error.value.status == 422 and "request must not be blank" in str(error.value)
    with pytest.raises(ApiError) as error:
        api.start_session(request="x", printer="Nope")
    assert "unknown printer" in str(error.value)
    with pytest.raises(ApiError) as error:
        ApiClient("http://127.0.0.1:9", timeout=2).printers()
    assert error.value.status is None and "Cannot reach the backend" in str(error.value)


def test_path_helpers():
    assert relative_run_path("/runs/a/print_tuning/t.pdf", "/runs/a") == "print_tuning/t.pdf"
    assert relative_run_path("report.pdf", "/runs/a") == "report.pdf"
    assert relative_run_path("/runs/b/t.pdf", "/runs/a") is None
    assert relative_run_path("/runs/a/../b/t.pdf", "/runs/a") is None
    assert relative_run_path(r"C:\runs\a\scans\round_1\p.png", r"C:\runs\a") == "scans/round_1/p.png"
    assert relative_run_path(None, "/runs/a") is None
    value = {"x": ["/runs/a/one.pdf", {"y": "/runs/a/two.PNG"}], "z": "/elsewhere/three.png", "w": "notes.txt"}
    assert find_preview_files(value, "/runs/a") == ["one.pdf", "two.PNG"]
    assert ApiClient.file_path("s 1", "../x y.pdf") == "/api/sessions/s%201/files/%2E%2E/x%20y.pdf"
