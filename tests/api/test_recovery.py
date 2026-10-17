"""Disk recovery through HTTP must preserve stopped sessions and unresolved intents."""
import json
from unittest.mock import Mock
from types import SimpleNamespace

from fastapi.testclient import TestClient
from langgraph.checkpoint.memory import InMemorySaver

from app.api.main import create_app
from app.workflow import runner, tuning
from app.workflow.graph import build_graph
from app.print_tuning.tuning_tools import create_tuning_tools
from tests.api.test_sessions import wait, preparation
from tests.workflow import test_tuning as fixtures


def fixture():
    f = fixtures.SessionTests()
    f.setUp()
    f.init(max_cycles=2)
    f.propose()
    _, trial = f.trial(1)
    state = {"dialogue": [], "execution_results": [{"tool_name": "run_focusing_experiment", "result": r} for r in f.results],
             "experiment_selection": f.selection, "tuning_session_path": str(f.path), "tuning_printer": "Office_Printer",
             "stop_reason": "running", "decision_count": 0, "decision_budget": 10}
    runner.save_run(state, f.root)
    config = json.loads((f.root / "config.json").read_text())
    config.update(request="Offline recovery", fake_printer=True)
    (f.root / "config.json").write_text(json.dumps(config))
    return f, trial


def test_restart_unresolved_trial_intent_requires_confirmation_without_submission():
    f, trial = fixture()
    try:
        tuning.record_print_approval(f.path, approval_id="approval", trial_id=trial["trial_id"],
            pdf_sha256=trial["pdf_sha256"], printer="Office_Printer", print_settings={"sides": "two-sided-long-edge"},
            approved_by="offline test")
        tuning.begin_print_attempt(f.path, approval_id="approval", attempt_id="intent")
        before = f.path.read_bytes()

        def prepare(root, state, options, **kwargs):
            assert kwargs["resume"]
            state, _ = runner.load_resume(root, 10)
            model = Mock()
            model.bind_tools.side_effect = AssertionError("No executor calls before human confirmation")
            printer_tools = [SimpleNamespace(name=name) for name in ("job_status", "printer_status", "submit_trial_print")]
            graph = build_graph(model, Mock(), Mock(), printer_tools, Mock(), root, 5, InMemorySaver(),
                                tuning_tools=create_tuning_tools(root), max_cycles=2)
            return graph, state

        with TestClient(create_app(f.root.parent, prepare_fn=prepare)) as client:
            sid = f.root.name
            assert f.path.read_bytes() == before  # discovery performs no tuning/hardware action
            assert client.get(f"/api/sessions/{sid}").json()["pending"] is None
            assert client.post(f"/api/sessions/{sid}/resume", json={}).status_code == 200
            current = wait(client, sid)
            assert current["pending"]["kind"] == "tuning_print_confirmation"
            session = tuning.load_tuning_session(f.path)
            assert len(session["rounds"]["1"]["print_attempts"]) == 1
            assert session["rounds"]["1"]["print_attempts"][0]["result"] is None
    finally:
        f.doCleanups()


def test_human_stop_requires_explicit_backed_up_recovery():
    f, _ = fixture()
    try:
        tuning.record_stop(f.path, "Human stopped print tuning")
        before = f.path.read_bytes()

        def prepare(root, state, options, **kwargs):
            state, config = runner.load_resume(root, 10, resume_stopped_tuning=kwargs["recovery"])
            return preparation([{"kind": "budget"}])(root, state, options)

        with TestClient(create_app(f.root.parent, prepare_fn=prepare)) as client:
            sid = f.root.name
            assert client.post(f"/api/sessions/{sid}/resume", json={}).status_code == 409
            assert f.path.read_bytes() == before
            # Recovery uses the actual runner backup/reopen logic; graph then asks for approval.
            response = client.post(f"/api/sessions/{sid}/resume", json={"recover_stopped_tuning": True})
            assert response.status_code == 200
            assert wait(client, sid)["pending"]["kind"] == "budget"
            assert list((f.root / "recovery_backups").glob("*/tuning_session.json"))
            assert tuning.load_tuning_session(f.path)["status"] != "stopped"
    finally:
        f.doCleanups()
