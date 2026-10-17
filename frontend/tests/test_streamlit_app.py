"""Smoke tests of the Streamlit page with Streamlit's AppTest, against the mock served on a local port."""
from pathlib import Path
import socket
import threading
import time

import pytest

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


def test_start_a_run_from_the_sidebar(backend, monkeypatch):
    app = page(backend, monkeypatch)
    assert not app.exception
    submit = next(b for b in app.sidebar.button if b.label == "Start run")
    submit.click().run()
    assert not app.exception
    session_id = app.session_state["session_id"]
    assert ApiClient(backend).get_session(session_id)["status"] == "waiting_for_human"


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
