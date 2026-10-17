"""Smoke tests of the Streamlit page with Streamlit's AppTest, against the mock served on a local port."""
from pathlib import Path
import socket
import threading
import time
import json

import pytest
import requests

pytest.importorskip("streamlit")
import streamlit as st  # noqa: E402
from streamlit.testing.v1 import AppTest  # noqa: E402
import uvicorn  # noqa: E402

from frontend.api_client import ApiClient  # noqa: E402
from frontend.mock_api import create_app  # noqa: E402

APP = str(Path(__file__).resolve().parents[1] / "streamlit_app.py")


@pytest.fixture(scope="module")
def backend(tmp_path_factory):
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(create_app(tmp_path_factory.mktemp("runs"), step_seconds=0),
                                           host="127.0.0.1", port=port, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 10
    while not server.started and time.monotonic() < deadline:
        time.sleep(0.05)
    yield f"http://127.0.0.1:{port}"
    server.should_exit = True
    thread.join(5)


def page(url, monkeypatch, session_id=None):
    monkeypatch.setenv("API_URL", url)
    st.cache_resource.clear()
    st.cache_data.clear()
    app = AppTest.from_file(APP, default_timeout=30)
    if session_id:
        app.session_state["session_id"] = session_id
    return app.run()


def button(app, label):
    return next(b for b in app.button if b.label == label)


def test_page_explains_an_unreachable_backend(monkeypatch):
    app = page("http://127.0.0.1:9", monkeypatch)
    assert not app.exception
    assert any("Cannot reach the backend" in e.value for e in app.error)


def test_polling_reports_backend_disconnect_and_clears_on_recovery(monkeypatch):
    session = {"session_id": "connection-test", "status": "running", "run_dir": "/runs/connection-test",
               "dialogue": [], "execution_results": [], "pending": None}
    session_reads = []
    def respond(http, method, url, **kwargs):
        assert method == "GET", "Polling must not start or resume work"
        response = requests.Response(); response.status_code = 200
        if url.endswith("/api/sessions/connection-test"):
            session_reads.append(url)
            if len(session_reads) == 2:  # initial page succeeds; the polling fragment disconnects
                raise requests.ConnectionError("offline test")
        body = {"printers": ["Office_Printer"]} if url.endswith("/printers") else [session] if url.endswith("/sessions") else session
        response._content = json.dumps(body).encode()
        return response
    monkeypatch.setattr(requests.Session, "request", respond)
    app = page("http://ui-test", monkeypatch, "connection-test")
    assert not app.exception
    assert any("Cannot reach the backend" in e.value for e in app.error)
    app.run()
    assert not app.exception
    assert not app.error
    assert any("Running" in m.value for m in app.markdown)


def test_start_a_run_from_the_sidebar(backend, monkeypatch):
    requests_sent = []
    original = requests.Session.request
    def record(http, method, url, **kwargs):
        if method == "POST" and url.endswith("/api/sessions"):
            requests_sent.append(kwargs["json"])
        return original(http, method, url, **kwargs)
    monkeypatch.setattr(requests.Session, "request", record)
    app = page(backend, monkeypatch)
    assert not app.exception
    assert not app.number_input  # no initial/additional decision-budget widgets
    assert {w.label for w in app.sidebar.text_input} >= {"Model", "Specialist model", "Paper color"}
    assert any(w.label == "Enable real printer selection" for w in app.sidebar.checkbox)
    app.session_state["new-run-model"] = "offline-test-model"
    app.session_state["new-run-paper"] = "light yellow"
    app.run()
    submit = next(b for b in app.sidebar.button if b.label == "Start run")
    submit.click().run()
    assert not app.exception
    session_id = app.session_state["session_id"]
    session = ApiClient(backend).get_session(session_id)
    assert session["status"] == "waiting_for_human"
    assert session["pending"]["decisions_used"] == 10 and session["pending"]["extra"] == 5
    assert "initial_decisions" not in requests_sent[-1] and "extra_decisions" not in requests_sent[-1]
    assert requests_sent[-1]["model"] == "offline-test-model" and requests_sent[-1]["paper_color"] == "light yellow"
    assert next(w for w in app.sidebar.text_input if w.label == "Model").value == "offline-test-model"


@pytest.mark.parametrize("kind,choices,labels", [
    ("tuning_cycle", ["continue", "finish", "stop"], {"Continue", "Finish", "Stop"}),
    ("trial_print_approval", ["approve", "stop"], {"Approve", "Stop"}),
    ("tuning_scan", ["scan", "reprint", "stop"], {"Upload", "Reprint", "Stop"}),
    ("print_approval", [], {"Approve and print", "Request this change"}),
    ("report_workflow_problem", ["retry", "correct", "stop"], {"Retry", "Correct instructions", "Stop"}),
])
def test_essential_controls_and_framework_chrome(kind, choices, labels, monkeypatch):
    session = {"session_id": "ui-controls", "status": "waiting_for_human", "run_dir": "/runs/ui-controls",
        "dialogue": [], "execution_results": [], "fake_printer": True,
        "pending": {"kind": kind, "choices": choices, "interrupt_token": "stable-token", "question": "Review this step",
                    "cycle_id": 2, "step_id": 1, "round_id": 1, "trial_id": "trial",
                    "pdf_path": "/runs/ui-controls/report.pdf", "printer": "Office_Printer", "pdf_sha256": "a" * 64}}
    def respond(http, method, url, **kwargs):
        assert method == "GET", "Rendering/polling must never submit an answer"
        response = requests.Response()
        response.status_code = 200
        if "/files/" in url:
            response._content = b"%PDF-1.4\n%%EOF"
        else:
            body = {"printers": ["Office_Printer"]} if url.endswith("/printers") else [session] if url.endswith("/sessions") else session
            response._content = json.dumps(body).encode()
        return response
    monkeypatch.setattr(requests.Session, "request", respond)
    app = page("http://ui-test", monkeypatch, session["session_id"])
    assert not app.exception
    assert not app.number_input
    assert labels <= {b.label for b in app.button}
    assert st.get_option("client.toolbarMode") == "minimal"
    assert any("Waiting for your input" in m.value for m in app.markdown)
    if kind in ("trial_print_approval", "print_approval"):
        assert app.get("download_button")
    if kind == "tuning_scan":
        assert app.get("file_uploader")
    # AppTest does not send unsent form buffers on .run(); seed committed widget
    # state to check retention. Unsubmitted browser text is a manual browser check.
    app.session_state["new-run-model"] = "keep-this-model"
    app.run()
    app.run()
    assert not app.exception
    assert next(w for w in app.sidebar.text_input if w.label == "Model").value == "keep-this-model"


def test_recovery_and_report_revision_controls_remain(monkeypatch):
    session = {"session_id": "paused", "status": "stopped", "run_dir": "/runs/paused",
               "dialogue": [], "execution_results": [], "pending": None, "recovery_required": True}
    def respond(http, method, url, **kwargs):
        assert method == "GET"
        response = requests.Response(); response.status_code = 200
        body = {"printers": ["Office_Printer"]} if url.endswith("/printers") else [session] if url.endswith("/sessions") else session
        response._content = json.dumps(body).encode()
        return response
    monkeypatch.setattr(requests.Session, "request", respond)
    app = page("http://ui-test", monkeypatch, "paused")
    assert not app.exception
    assert any(b.label == "Resume saved run" for b in app.button)
    assert any("Explicitly reopen human-stopped tuning" in c.label for c in app.checkbox)
    assert any(w.label == "Report revision instructions (optional)" for w in app.text_area)


def test_answer_pauses_from_the_page(backend, monkeypatch):
    api = ApiClient(backend)
    session_id = api.start_session(request="UI test", fake_printer=True)
    app = page(backend, monkeypatch, session_id)
    assert not app.exception
    assert any("Continue the run?" in m.value for m in app.markdown)
    app.text_input[0].input("focus on cost")
    button(app, "Continue").click().run()
    assert not app.exception
    human = [t["text"] for t in api.get_session(session_id)["dialogue"] if t["role"] == "human"]
    assert human == ["yes focus on cost"]

    app.run()                                    # now: which printer prints the trial sheets?
    assert api.get_session(session_id)["pending"]["kind"] == "tuning_printer"
    button(app, "Use this printer").click().run()
    payload = api.get_session(session_id)["pending"]
    assert payload["kind"] == "trial_print_approval"

    app.run()                                    # the trial card shows the PDF and its sha256
    assert not app.exception
    assert any(payload["pdf_sha256"] in m.value for m in app.markdown)
    assert any(b.label == "Download" for b in app.get("download_button"))
    button(app, "Approve").click().run()
    assert api.get_session(session_id)["pending"]["kind"] == "tuning_print_confirmation"

    app.run()
    button(app, "Printed").click().run()
    assert api.get_session(session_id)["pending"]["kind"] == "tuning_scan"
    app.run()
    button(app, "Upload").click().run()          # nothing chosen: a message, nothing sent
    assert any("Choose at least one scan file" in e.value for e in app.error)
    assert api.get_session(session_id)["pending"]["kind"] == "tuning_scan"


def test_an_invalid_answer_shows_the_backend_message(backend, monkeypatch):
    api = ApiClient(backend)
    session_id = api.start_session(request="UI 422 test", fake_printer=True)
    for answer in ("yes", {"action": "printer", "printer": "Office_Printer"}, {"action": "approve"},
                   {"action": "printed"}):
        api.answer(session_id, answer)
    payload = api.get_session(session_id)["pending"]
    assert payload["kind"] == "tuning_scan"
    app = page(backend, monkeypatch, session_id)
    app.session_state["uploads"] = {(session_id, 1, payload["trial_id"]): ["/etc/hosts"]}
    app.run()
    button(app, "Use these 1 scans").click().run()
    assert not app.exception
    assert any("Invalid input: tuning_scan: /etc/hosts is not an uploaded scan" in e.value for e in app.error)
    assert api.get_session(session_id)["pending"]["kind"] == "tuning_scan"


def test_polling_while_waiting_keeps_typed_text(backend, monkeypatch):
    api = ApiClient(backend)
    session_id = api.start_session(request="UI polling test", fake_printer=True)
    app = page(backend, monkeypatch, session_id)
    app.text_input[0].input("half-typed guidance")
    app.run()                                    # a poll with nothing new: the page keeps the text
    app.run()
    assert not app.exception
    assert app.text_input[0].value == "half-typed guidance"
    assert api.get_session(session_id)["pending"]["kind"] == "budget"


def test_answer_for_a_question_that_changed_is_not_sent(backend, monkeypatch):
    api = ApiClient(backend)
    session_id = api.start_session(request="UI stale answer test", fake_printer=True)
    app = page(backend, monkeypatch, session_id)
    api.answer(session_id, "yes")                # answered in another tab: now tuning_printer is pending
    button(app, "Continue").click().run()        # the page still showed the budget card
    assert not app.exception
    assert any("The question changed before your answer was sent" in e.value for e in app.error)
    session = api.get_session(session_id)
    assert session["pending"]["kind"] == "tuning_printer"
    assert [t["text"] for t in session["dialogue"] if t["role"] == "human"] == ["yes"]
    app.run()                                    # the page now shows the new question
    assert any("Which printer prints the trial sheets?" in m.value for m in app.markdown)
