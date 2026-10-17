"""HTTP lifecycle and concurrency tests: no network providers or hardware."""
from concurrent.futures import ThreadPoolExecutor
import io
import json
from pathlib import Path
import threading
import time

from fastapi.testclient import TestClient
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import START, END, StateGraph
from langgraph.types import interrupt
from PIL import Image
import pytest

from app.api.main import create_app
from app.api.session_manager import validate_answer, SessionError
from app.workflow.state import State


def wait(client, sid, status="waiting_for_human", timeout=60):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        value = client.get(f"/api/sessions/{sid}").json()
        if value["status"] == status:
            session = client.app.state.sessions.get(sid)
            with session.lock:
                if not session.busy:
                    return value
        if value["status"] == "failed":
            raise AssertionError(value)
        time.sleep(.02)
    raise AssertionError(value)


def send(client, sid, payload, answer):
    return client.post(f"/api/sessions/{sid}/answer", json={"interrupt_token": payload["interrupt_token"], "answer": answer})


def png():
    out = io.BytesIO()
    Image.new("L", (20, 20), 200).save(out, format="PNG")
    return out.getvalue()


def preparation(pauses, gate=None):
    def prepare(root, state, options, **kwargs):
        if gate:
            gate.wait(10)
        builder = StateGraph(State)
        for i, payload in enumerate(pauses):
            def node(state, p=payload):
                p = dict(p)
                if p["kind"] == "tuning_scan":
                    p["suggested_scan_directory"] = str(root / "scans" / "cycle_2" / "step_2")
                reply = interrupt(p)
                return {"dialogue": [{"role": "human", "text": json.dumps(reply)}]}
            builder.add_node(str(i), node)
            builder.add_edge(START if i == 0 else str(i-1), str(i))
        builder.add_node("done", lambda state: {"stop_reason": "finished"})
        builder.add_edge(str(len(pauses)-1), "done")
        builder.add_edge("done", END)
        return builder.compile(checkpointer=InMemorySaver()), state
    return prepare


def test_prompt_creation_claims_and_restart(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-offline-sensitive-key")
    gate = threading.Event()
    pauses = [{"kind": "trial_print_approval", "choices": ["approve", "stop"]}] * 2
    app = create_app(tmp_path, prepare_fn=preparation(pauses, gate))
    with TestClient(app) as client:
        start = time.monotonic()
        response = client.post("/api/sessions", json={"request": "test"})
        assert response.status_code == 201
        assert time.monotonic() - start < 2
        sid = response.json()["session_id"]
        assert app.state.sessions.get(sid).record["options"]["fake_printer"] is True
        assert client.post(f"/api/sessions/{sid}/resume", json={}).status_code == 409
        gate.set()
        first = wait(client, sid)["pending"]
        with ThreadPoolExecutor(2) as pool:
            replies = list(pool.map(lambda _: send(client, sid, first, {"action": "approve"}).status_code, range(2)))
        assert sorted(replies) == [200, 409]
        second = wait(client, sid)["pending"]
        assert first["interrupt_token"] != second["interrupt_token"]
        assert send(client, sid, first, {"action": "approve"}).status_code == 409
        root = app.state.sessions.get(sid).root
        current = app.state.sessions.get(sid)
        with current.lock:
            current.state["dialogue"].append({"role": "system", "text": "sk-offline-sensitive-key"})
        assert "sk-offline-sensitive-key" not in client.get(f"/api/sessions/{sid}").text
        assert client.post("/api/sessions", json={"request": "x", "api_key": "sk-offline-sensitive-key"}).status_code == 422
    with TestClient(create_app(tmp_path, prepare_fn=preparation(pauses))) as client:
        recovered = client.get(f"/api/sessions/{sid}").json()
        assert recovered["status"] == "stopped" and recovered["recovery_required"]
        assert recovered["pending"] is None
        assert send(client, sid, second, {"action": "approve"}).status_code == 409
        assert not (root / "fake_printer_jobs.json").exists()
        assert client.post(f"/api/sessions/{sid}/resume", json={}).status_code == 200
        assert wait(client, sid)["pending"]["interrupt_token"] != second["interrupt_token"]


def test_scan_validation_and_artifact_boundaries(tmp_path):
    pauses = [{"kind": "tuning_scan", "cycle_id": 2, "step_id": 2, "choices": ["scan", "stop"]}]
    with TestClient(create_app(tmp_path, prepare_fn=preparation(pauses), upload_limit=2048)) as client:
        sid = client.post("/api/sessions", json={"request": "scan"}).json()["session_id"]
        pending = wait(client, sid)["pending"]
        url = f"/api/sessions/{sid}/scans"
        data = {"interrupt_token": pending["interrupt_token"]}
        upload = lambda name, content: client.post(url, data=data, files=[("files", (name, content, "image/png"))])
        assert upload("bad.png", b"bad").status_code == 422
        assert upload("huge.png", b"x" * 2049).status_code == 413
        paths = [upload("../../scan.png", png()).json()["paths"][0] for _ in range(2)]
        assert len(set(paths)) == 2
        assert all("scans/cycle_2/step_2/" in p.replace("\\", "/") for p in paths)
        assert wait(client, sid)["pending"] == pending  # upload never resumes
        root = Path(client.get(f"/api/sessions/{sid}").json()["run_dir"])
        (root / ".env").write_text("secret")
        (root / "private.json").write_text("secret")
        assert client.get(f"/api/sessions/{sid}/files/.env").status_code == 403
        assert client.get(f"/api/sessions/{sid}/files/private.json").status_code == 403
        assert client.get(f"/api/sessions/{sid}/files/%2E%2E/private.png").status_code == 403
        relative = Path(paths[0]).relative_to(root).as_posix()
        result = client.get(f"/api/sessions/{sid}/files/{relative}")
        assert result.status_code == 200 and result.headers["content-type"] == "image/png"
        assert "no-store" in result.headers["cache-control"]
        assert send(client, sid, pending, {"action": "scan", "scan_paths": [str(root / ".env")]}).status_code == 422
        assert send(client, sid, pending, {"action": "scan", "scan_paths": paths}).status_code == 200
        wait(client, sid, "finished")
        assert upload("late.png", png()).status_code == 409


def test_scan_upload_rechecks_interrupt_after_validation(tmp_path, monkeypatch):
    import app.api.routes as routes
    entered, release = threading.Event(), threading.Event()
    original = routes.validate_scan

    def validate(data, suffix):
        entered.set()
        assert release.wait(10)
        original(data, suffix)

    monkeypatch.setattr(routes, "validate_scan", validate)
    with TestClient(create_app(tmp_path, prepare_fn=preparation([
        {"kind": "tuning_scan", "choices": ["scan", "stop"]}]
    ))) as client:
        sid = client.post("/api/sessions", json={"request": "race"}).json()["session_id"]
        pending = wait(client, sid)["pending"]
        with ThreadPoolExecutor(1) as pool:
            upload = pool.submit(client.post, f"/api/sessions/{sid}/scans",
                data={"interrupt_token": pending["interrupt_token"]}, files=[("files", ("scan.png", png(), "image/png"))])
            assert entered.wait(10)
            assert send(client, sid, pending, {"action": "stop"}).status_code == 200
            wait(client, sid, "finished")
            release.set()
            assert upload.result().status_code == 409
        root = Path(client.get(f"/api/sessions/{sid}").json()["run_dir"])
        assert not list(root.rglob("*.png"))


def test_file_symlink_escape(tmp_path):
    with TestClient(create_app(tmp_path / "runs", prepare_fn=preparation([{"kind": "budget"}]))) as client:
        sid = client.post("/api/sessions", json={"request": "files"}).json()["session_id"]
        root = Path(wait(client, sid)["run_dir"])
        outside = tmp_path / "private.png"
        outside.write_bytes(png())
        try:
            (root / "escape.png").symlink_to(outside)
        except OSError:
            pytest.skip("Creating symlinks is not permitted on this host")
        assert client.get(f"/api/sessions/{sid}/files/escape.png").status_code == 403


@pytest.mark.parametrize("kind,answer", [
    ("budget", "yes additional guidance"), ("printer_problem", "retry"), ("print_approval", "Change the title"),
    ("tuning_printer", {"action": "printer", "printer": "Office_Printer"}),
    ("trial_print_approval", {"action": "approve"}), ("tuning_scan", {"action": "scan"}),
    ("tuning_print_confirmation", {"action": "printed"}), ("tuning_poll", {"action": "check"}),
    ("tuning_problem", {"action": "regenerate"}), ("tuning_cycle", {"action": "continue"}),
    ("report_workflow_problem", {"action": "correct", "instructions": "New title"}),
    ("report_print_confirmation", {"action": "not_printed"}),
])
def test_answer_shapes(kind, answer):
    payload = {"kind": kind, "choices": [answer["action"]] if isinstance(answer, dict) else []}
    assert validate_answer(payload, answer) == answer
    with pytest.raises(SessionError):
        validate_answer(payload, "" if isinstance(answer, str) else {"action": "invalid"})


def test_readonly_printers_and_failures(tmp_path):
    def broken(fake):
        raise RuntimeError("sk-private-value")
    with TestClient(create_app(tmp_path, printers_fn=broken)) as client:
        response = client.get("/api/printers?fake_printer=false")
        assert response.status_code == 503 and "sk-private-value" not in response.text


def test_frontend_omits_budget_settings_and_real_api_uses_existing_defaults(tmp_path, monkeypatch):
    from frontend.api_client import ApiClient
    from app.workflow import runner
    from app.agents.common import DEFAULT_INITIAL_DECISIONS
    from app.agents.orchestrator import DEFAULT_EXTRA_DECISIONS
    configured = []
    monkeypatch.setenv("OPENAI_API_KEY", "offline-budget-test")
    def setup(root, state, **kwargs):
        configured.append(kwargs)
        assert state["decision_budget"] == DEFAULT_INITIAL_DECISIONS
        return preparation([{"kind": "budget", "decisions_used": kwargs["initial_decisions"],
                             "extra": kwargs["extra_decisions"]}])(root, state, {})[0]
    # Exercise real API preparation/default resolution without constructing models or hardware tools.
    monkeypatch.setattr(runner, "prepare_workflow", setup)
    with TestClient(create_app(tmp_path)) as client:
        api = ApiClient("http://testserver", http=client)
        sid = api.start_session(request="Use the server's existing budgets")
        current = wait(client, sid)
        assert configured[0]["initial_decisions"] == DEFAULT_INITIAL_DECISIONS
        assert configured[0]["extra_decisions"] == DEFAULT_EXTRA_DECISIONS
        assert current["pending"]["extra"] == DEFAULT_EXTRA_DECISIONS
        assert send(client, sid, current["pending"], "no").status_code == 200
        wait(client, sid, "finished")


def test_discovers_terminal_failures_without_starting_workers(tmp_path):
    for reason in ("finished", "execution_failure", "running"):
        root = tmp_path / reason
        root.mkdir()
        (root / "state.json").write_text(json.dumps({"dialogue": [], "execution_results": [], "stop_reason": reason}))
        (root / "config.json").write_text(json.dumps({"request": "Saved terminal run", "fake_printer": True}))
    def forbidden(*args, **kwargs):
        raise AssertionError("Discovery must not start a workflow")
    with TestClient(create_app(tmp_path, prepare_fn=forbidden)) as client:
        rows = {r["session_id"]: r["status"] for r in client.get("/api/sessions").json()}
        assert rows == {"finished": "finished", "execution_failure": "failed", "running": "stopped"}
        assert client.get("/api/sessions/running").json()["stop_reason"] == "execution_interrupted"


def test_all_interrupts_roundtrip_and_cooperative_stop(tmp_path):
    cases = [
        ({"kind": "budget"}, "yes keep the controls fixed"),
        ({"kind": "tuning_printer", "choices": ["printer", "stop"]}, {"action": "printer", "printer": "Office_Printer"}),
        ({"kind": "trial_print_approval", "choices": ["approve", "stop"]}, {"action": "approve"}),
        ({"kind": "tuning_print_confirmation", "choices": ["printed", "not_printed", "stop"]}, {"action": "printed"}),
        ({"kind": "tuning_poll", "choices": ["check", "stop"]}, {"action": "check"}),
        ({"kind": "tuning_problem", "choices": ["retry", "regenerate", "stop"]}, {"action": "retry"}),
        ({"kind": "tuning_problem", "choices": ["reprint", "stop"]}, {"action": "reprint"}),
        ({"kind": "tuning_cycle", "choices": ["continue", "finish", "stop"]}, {"action": "finish"}),
        ({"kind": "report_workflow_problem", "choices": ["retry", "correct", "stop"]}, {"action": "correct", "instructions": "Keep the figure"}),
        ({"kind": "print_approval"}, "Please revise the closing note"),
        ({"kind": "printer_problem", "choices": ["retry", "skip"]}, "retry"),
        ({"kind": "report_print_confirmation", "choices": ["printed", "not_printed", "stop"]}, {"action": "not_printed"}),
    ]
    with TestClient(create_app(tmp_path, prepare_fn=preparation([p for p, _ in cases]))) as client:
        sid = client.post("/api/sessions", json={"request": "all interrupts"}).json()["session_id"]
        for payload, answer in cases:
            pending = wait(client, sid)["pending"]
            assert pending["kind"] == payload["kind"]
            assert send(client, sid, pending, answer).status_code == 200
        final = wait(client, sid, "finished")
        assert [json.loads(t["text"]) for t in final["dialogue"] if t["role"] == "human"] == [a for _, a in cases]

    gate = threading.Event()
    with TestClient(create_app(tmp_path / "stop", prepare_fn=preparation([{"kind": "budget"}], gate))) as client:
        sid = client.post("/api/sessions", json={"request": "pause"}).json()["session_id"]
        assert client.post(f"/api/sessions/{sid}/stop").status_code == 200
        gate.set()
        paused = wait(client, sid, "stopped")
        assert paused["recovery_required"] and paused["pending"] is None
