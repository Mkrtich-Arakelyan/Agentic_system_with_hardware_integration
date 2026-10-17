"""The terminal runner and print tuning: tuning pauses as dicts, scan folders, tools and resume."""

import json
import os
from pathlib import Path
from uuid import uuid4

import pytest

from app.print_tuning.tuning_tools import create_tuning_tools
from app.workflow import runner
from app.printing import connector
from app.workflow.tuning import (
    begin_print_attempt, load_tuning_session, record_print_approval,
    record_scan_for_executor, record_stop,
)
from tests.workflow import test_tuning_flow   # module import: its TestCase is not collected here again


def answers(monkeypatch, *replies):
    """input() gives these replies in order; the prompts it was shown are returned."""
    replies, prompts = list(replies), []

    def fake_input(prompt=""):
        prompts.append(prompt)
        if not replies:
            raise AssertionError("asked more often than expected: " + prompt)
        reply = replies.pop(0)
        return reply() if callable(reply) else reply
    monkeypatch.setattr("builtins.input", fake_input)
    return prompts


def test_report_revision_input_keeps_exact_wording_and_correction(monkeypatch):
    text = "  Add a reader question at the end.  "
    answers(monkeypatch, text)
    assert runner.ask_human({"kind": "print_approval", "question": "Print?"}) == text
    answers(monkeypatch, "correct", text)
    assert runner.ask_human({"kind": "report_workflow_problem", "question": "Retry/correct/stop?",
                            "choices": ["retry", "correct", "stop"]}) == {"action": "correct", "instructions": text}


def test_missing_sumatra_stops_startup_before_resume_or_approval(monkeypatch, tmp_path, capsys):
    run = tmp_path / "run"
    run.mkdir()
    (run / "config.json").write_text('{}', encoding="utf-8")
    original = (run / "config.json").read_bytes()
    monkeypatch.setattr(runner.sys, "argv", ["main.py", "--resume", str(run),
        "--resume-stopped-tuning", "--no-fake-printer", "--printer", "Offline"])
    monkeypatch.setenv("OPENAI_API_KEY", "offline-test-key")
    monkeypatch.setenv("SUMATRA_PATH", "missing-sumatra.exe")
    monkeypatch.setattr(connector, "_windows", lambda: True)
    monkeypatch.setattr(connector.shutil, "which", lambda value: None)
    monkeypatch.setattr(runner, "load_resume", lambda *a, **k: pytest.fail("must not mutate resume records"))
    monkeypatch.setattr(runner, "ask_human", lambda *a, **k: pytest.fail("must not request approval"))
    monkeypatch.setattr(connector, "_run", lambda *a: pytest.fail("must not contact printer"))
    with pytest.raises(SystemExit) as exit:
        runner.main()
    assert exit.value.code == 2
    assert "SUMATRA_PATH" in capsys.readouterr().err
    assert (run / "config.json").read_bytes() == original
    assert list(run.iterdir()) == [run / "config.json"]


# ---------- ask_human: the tuning pauses ----------

PRINTER = {"kind": "tuning_printer", "question": "Supply the exact printer name, or stop.", "choices": ["printer", "stop"]}


def test_printer_uses_the_default_or_a_typed_name(monkeypatch, capsys):
    answers(monkeypatch, "printer")
    assert runner.ask_human(PRINTER, default_printer="Office_Printer") == {"action": "printer", "printer": "Office_Printer"}
    assert "Default printer: Office_Printer" in capsys.readouterr().out
    answers(monkeypatch, "maybe", "", "printer HP LaserJet 4")
    assert runner.ask_human(PRINTER, default_printer="Office_Printer") == {"action": "printer", "printer": "HP LaserJet 4"}
    assert "Please answer with one of: printer, stop" in capsys.readouterr().out
    answers(monkeypatch, "printer", "printer Lab_Printer")          # no default: a bare "printer" is not enough
    assert runner.ask_human(PRINTER) == {"action": "printer", "printer": "Lab_Printer"}
    answers(monkeypatch, "STOP")
    assert runner.ask_human(PRINTER, default_printer="Office_Printer") == {"action": "stop"}


def test_trial_print_approval_shows_the_exact_sheet(monkeypatch, capsys):
    payload = {"kind": "trial_print_approval", "pdf_path": "/run/print_tuning/trial_r1_abcd1234.pdf", "page_count": 2,
               "printer": "Office_Printer", "round_id": 1, "trial_id": "trial-1", "pdf_sha256": "f" * 64,
               "question": "Approve printing this exact trial PDF? Choose approve or stop.", "choices": ["approve", "stop"]}
    answers(monkeypatch, "yes", "approve")
    assert runner.ask_human(payload) == {"action": "approve"}
    out = capsys.readouterr().out
    for text in ("trial_r1_abcd1234.pdf", "Pages: 2", "Office_Printer", "round 1", "trial-1", "f" * 64,
                 "A4, grayscale, two-sided (long-edge), Actual size / 100%"):
        assert text in out
    answers(monkeypatch, "stop")
    assert runner.ask_human(payload) == {"action": "stop"}


def scan_payload(folder):
    return {"kind": "tuning_scan", "round_id": 2, "trial_id": "trial-2", "suggested_scan_directory": str(folder),
            "question": "Scan the printed pages ...", "choices": ["scan", "reprint", "stop"]}


def test_scan_requires_explicit_files_even_when_folder_contains_old_scans(monkeypatch, tmp_path, capsys):
    folder = tmp_path / "scans" / "round_2"
    folder.mkdir(parents=True)
    for name in ("b.png", "a.JPG", "old_rescan.png", "notes.txt", ".hidden.png"):
        (folder / name).write_bytes(b"x")
    prompts = answers(monkeypatch, "scan", "", str(folder), "", "b.png", "a.JPG", "", "confirm")
    result = runner.ask_human(scan_payload(folder))
    assert result == {"action": "scan", "scan_paths": [str((folder / "b.png").resolve()), str((folder / "a.JPG").resolve())]}
    out = capsys.readouterr().out
    assert str(folder) in out and "give explicit scan file paths" in out and "directory given" in out
    assert any('Type "confirm"' in prompt for prompt in prompts)


def test_scan_prompt_creates_missing_folder_without_selecting_files(monkeypatch, tmp_path):
    folder = tmp_path / "scans" / "round_1"
    answers(monkeypatch, "scan", "", "stop")
    assert runner.ask_human(scan_payload(folder)) == {"action": "stop"}
    assert folder.is_dir()


def test_scan_accepts_one_pdf_or_multiple_images(monkeypatch, tmp_path, capsys):
    folder, other = tmp_path / "round_2", tmp_path / "phone"
    folder.mkdir(), other.mkdir()
    (folder / "page2.png").write_bytes(b"x")
    pdf = tmp_path / "both pages.pdf"
    pdf.write_bytes(b"x")
    image = other / "phone page 1.jpeg"
    image.write_bytes(b"x")
    # A whole input line is one path, even with spaces and no quotes.
    answers(monkeypatch, "scan", str(pdf), "", "confirm")
    assert runner.ask_human(scan_payload(folder)) == {"action": "scan", "scan_paths": [str(pdf.resolve())]}
    answers(monkeypatch, "scan", f'"{image}"', "'page2.png'", "", "confirm")
    assert runner.ask_human(scan_payload(folder))["scan_paths"] == [str(image.resolve()), str((folder / "page2.png").resolve())]
    out = capsys.readouterr().out
    assert "Selected scan files:" in out and str(pdf.resolve()) in out and str(image.resolve()) in out


def test_scan_rejects_missing_files_and_wrong_types(monkeypatch, tmp_path, capsys):
    folder = tmp_path / "round_1"
    folder.mkdir()
    (folder / "notes.txt").write_text("x")
    answers(monkeypatch, "scan", "missing.png", "notes.txt", "", "", "stop")
    assert runner.ask_human(scan_payload(folder)) == {"action": "stop"}
    out = capsys.readouterr().out
    assert "file not found" in out and "notes.txt is not a scan file" in out and "give explicit scan file paths" in out
    answers(monkeypatch, "stop")
    assert runner.ask_human(scan_payload(folder)) == {"action": "stop"}
    answers(monkeypatch, "reprint")
    assert runner.ask_human(scan_payload(folder)) == {"action": "reprint"}


def test_scan_confirmation_can_correct_selection(monkeypatch, tmp_path, capsys):
    for name in ("wrong.jpg", "page 1.jpg", "page 2.png"):
        (tmp_path / name).write_bytes(b"x")
    prompts = answers(monkeypatch, "scan", "wrong.jpg", "", "correct", "page 1.jpg", "page 2.png", "",
                      "yes", "", "confirm")
    assert runner.ask_human(scan_payload(tmp_path)) == {
        "action": "scan", "scan_paths": [str((tmp_path / name).resolve()) for name in ("page 1.jpg", "page 2.png")]}
    out = capsys.readouterr().out
    assert "wrong.jpg" in out and "page 1.jpg" in out
    assert "Please answer with one of: confirm, correct, stop" in out
    assert sum('Type "confirm"' in prompt for prompt in prompts) == 4


@pytest.mark.parametrize("at_confirmation", [False, True])
def test_scan_can_stop_during_entry_or_confirmation(monkeypatch, tmp_path, at_confirmation):
    (tmp_path / "page.jpg").write_bytes(b"x")
    replies = ("scan", "page.jpg", "", "stop") if at_confirmation else ("scan", "page.jpg", "stop")
    answers(monkeypatch, *replies)
    assert runner.ask_human(scan_payload(tmp_path)) == {"action": "stop"}


def test_scan_only_accepts_files_entered_after_choosing_scan(monkeypatch, tmp_path, capsys):
    (tmp_path / "old.jpg").write_bytes(b"old")
    answers(monkeypatch, "scan old.jpg", "scan", "stop")
    assert runner.ask_human(scan_payload(tmp_path)) == {"action": "stop"}
    assert 'Type "scan" alone' in capsys.readouterr().out


@pytest.mark.parametrize("text, expected", [
    (r"C:\Users\Mkrtich\scans\page1.jpg", r"C:\Users\Mkrtich\scans\page1.jpg"),
    (r"C:\Users\First Last\scans\page 1.jpg", r"C:\Users\First Last\scans\page 1.jpg"),
    (r"C:\Users\O'Neil\scans\page1.jpg", r"C:\Users\O'Neil\scans\page1.jpg"),
    ('"C:\\Users\\First Last\\scans\\page1.jpg"', r"C:\Users\First Last\scans\page1.jpg"),
    (r"\\server\share\page1.jpg", r"\\server\share\page1.jpg"),
    ('"\\\\server\\scans with spaces\\page1.jpg"', r"\\server\scans with spaces\page1.jpg"),
    ("'/tmp/scans with spaces/page1.jpg'", "/tmp/scans with spaces/page1.jpg"),
    (r"C:\scans\page#1.jpg", r"C:\scans\page#1.jpg"),
])
def test_scan_lines_preserve_paths_on_every_platform(monkeypatch, tmp_path, text, expected):
    # Exercise the real line handling, isolating only platform-specific resolution
    # so Windows/UNC regressions are also caught when these tests run on POSIX.
    file = tmp_path / "page1.jpg"
    file.write_bytes(b"x")
    seen = []
    def path(value):
        seen.append(value)
        return file
    monkeypatch.setattr(runner, "Path", path)
    paths, error = runner.resolve_scan_paths(text, tmp_path, formats=("jpg",))
    assert seen == [expected]
    assert paths == [str(file.resolve())] and not error


def test_explicit_unquoted_paths_preserve_order_and_do_not_expand_folders(tmp_path):
    files = [tmp_path / name for name in ("page2.jpg", "page1.jpg")]
    for file in files:
        file.write_bytes(b"x")
    (tmp_path / "old_rescan.jpg").write_bytes(b"old")
    # Spaces and backslashes do not need shell quoting; each line is one file.
    text = "\n".join(str(file) for file in files)
    assert runner.resolve_scan_paths(text, tmp_path, formats=("jpg",)) == ([str(f.resolve()) for f in files], "")
    paths, error = runner.resolve_scan_paths(f'"{files[0]}"\n"{tmp_path}"', tmp_path, formats=("jpg",))
    assert paths == [] and "directory given" in error


@pytest.mark.parametrize("text", ['"unclosed.jpg', "'unclosed.jpg"])
def test_scan_rejects_unmatched_quotes(tmp_path, text):
    paths, error = runner.resolve_scan_paths(text, tmp_path, formats=("jpg",))
    assert paths == [] and "surrounding quotes must match" in error


def test_scan_formats_are_the_ones_read_scan_accepts():
    assert set(runner.scan_formats()) == {"pdf", "png", "jpg", "jpeg"}


def test_print_confirmation_poll_and_problem(monkeypatch, capsys):
    confirmation = {"kind": "tuning_print_confirmation", "attempt_id": "a1", "question": "Did this trial print?",
                    "choices": ["printed", "not_printed", "stop"]}
    answers(monkeypatch, "yes", "not_printed")
    assert runner.ask_human(confirmation) == {"action": "not_printed"}
    assert "could not be confirmed" in capsys.readouterr().out
    answers(monkeypatch, "printed")
    assert runner.ask_human(confirmation) == {"action": "printed"}

    poll = {"kind": "tuning_poll", "question": "Check again or stop?", "choices": ["check", "stop"]}
    answers(monkeypatch, "wait", "check")
    assert runner.ask_human(poll) == {"action": "check"}

    problem = {"kind": "tuning_problem", "message": "Printer is not ready: out_of_paper", "question": "?",
               "choices": ["retry", "regenerate", "stop"]}
    answers(monkeypatch, "regenerate")
    assert runner.ask_human(problem) == {"action": "regenerate"}
    assert "Printer is not ready: out_of_paper" in capsys.readouterr().out
    recovery = {"kind": "tuning_problem", "question": "Reprint or stop?", "choices": ["reprint", "stop"]}
    answers(monkeypatch, "retry", "reprint")                          # choices come from the payload
    assert runner.ask_human(recovery) == {"action": "reprint"}


@pytest.mark.parametrize("choice", ["printed", "not_printed", "stop"])
def test_report_print_confirmation_returns_payload_choice_as_action(monkeypatch, choice):
    # Choices belong to the graph's payload, not a second hardcoded runner list.
    payload = {"kind": "report_print_confirmation", "question": "Confirm the report print outcome.",
               "choices": ["printed", "not_printed", "stop"]}
    answers(monkeypatch, "", "yes", choice)
    assert runner.ask_human(payload) == {"action": choice}


def test_report_workflow_recovery_and_exact_pdf_display(monkeypatch, capsys):
    payload = {"kind": "report_workflow_problem", "message": "Executor did not request submit_print.",
               "question": "Retry workflow or stop?", "choices": ["retry", "stop"]}
    answers(monkeypatch, "yes", "retry")
    assert runner.ask_human(payload) == {"action": "retry"}
    assert "submit_print" in capsys.readouterr().out
    answers(monkeypatch, "retry")
    assert runner.ask_human({"kind": "print_approval", "pdf_path": "report.pdf", "printer": "ir2425",
                             "pages": 2, "copies": 1, "pdf_sha256": "a" * 64}) == "retry"
    out = capsys.readouterr().out
    assert "a" * 64 in out and "ir2425" in out and "report.pdf" in out


def test_resume_discards_approval_and_report_visit_state_without_changing_records(tmp_path):
    state = {"dialogue": [], "execution_results": [], "report": {"sha256": "pdf"},
             "approval": {"approved": True}, "pending_print": {"attempt_id": "pending"},
             "report_request_passes": 2, "report_workflow_failure": "stalled",
             "experiment_selection": {"selections": ["frozen"]}, "active_job_id": "existing-job"}
    (tmp_path / "state.json").write_text(json.dumps(state))
    (tmp_path / "config.json").write_text(json.dumps({"bench": {}, "request": "saved", "fake_printer": False,
                                                    "printer_name": "ir2425"}))
    # An intent with unknown outcome is kept byte-for-byte for graph reconciliation.
    (tmp_path / "report_print.json").write_text('{"attempts": [{"result": null}]}')
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    resumed, config = runner.load_resume(tmp_path, 10)
    assert not any(k in resumed for k in ("approval", "pending_print", "report_request_passes", "report_workflow_failure"))
    assert resumed["report"] == state["report"] and resumed["experiment_selection"] == state["experiment_selection"]
    assert resumed["active_job_id"] == "existing-job"
    assert config["printer_name"] == "ir2425" and config["fake_printer"] is False
    assert {p.name: p.read_bytes() for p in tmp_path.iterdir()} == before


def test_unknown_kinds_and_the_old_kinds(monkeypatch):
    answers(monkeypatch, "maybe", "b")
    assert runner.ask_human({"kind": "something_new", "question": "?", "choices": ["a", "b"]}) == {"action": "b"}
    answers(monkeypatch, "free text")
    assert runner.ask_human({"kind": "something_new", "question": "?"}) == "free text"
    answers(monkeypatch, "yes, continue")                              # old kinds still return their text
    assert runner.ask_human({"kind": "budget", "summary": "s", "decisions_used": 3, "question": "?"}) == "yes, continue"
    answers(monkeypatch, "retry")
    assert runner.ask_human({"kind": "printer_problem", "reading": {}, "question": "?", "choices": ["retry", "skip"]}) == "retry"
    answers(monkeypatch, "yes")
    assert runner.ask_human({"kind": "print_approval", "pdf_path": "r.pdf", "pages": 2, "printer": "P",
                             "copies": 1, "question": "?"}) == "yes"


# ---------- scan folders and tools ----------

def test_scan_folders_are_created(tmp_path):
    folders = runner.scan_folders(tmp_path)
    assert folders == [tmp_path / "scans" / f"round_{n}" for n in (1, 2, 3)]
    assert all(folder.is_dir() for folder in folders)
    assert runner.scan_folders(tmp_path) == folders                     # again, e.g. on resume


def test_tuning_tools_reach_the_graph_with_the_fake_printer(tmp_path, monkeypatch):
    from app.printing.tools import load_printer_tools

    monkeypatch.setenv("FAKE_PRINTER_FILE", str(tmp_path / "fake_printer_jobs.json"))
    printer_tools = load_printer_tools({"PRINTER_MODE": "fake"})
    tuning_tools = create_tuning_tools(tmp_path)
    runner.check_tuning_tools(printer_tools, tuning_tools)
    names = {tool.name for tool in [*printer_tools, *tuning_tools]}
    assert {"submit_trial_print", "job_status", "printer_status", "make_trial_sheet", "read_scan"} <= names
    with pytest.raises(RuntimeError, match="submit_trial_print"):
        runner.check_tuning_tools([t for t in printer_tools if t.name != "submit_trial_print"], tuning_tools)


def test_resume_drops_only_per_visit_tuning_keys(tmp_path):
    for key in ("tuning_action", "tuning_failure", "tuning_failures", "tuning_polls", "pending_trial_print"):
        assert key in runner.RESUME_DROPPED_KEYS
    for key in ("tuning_session_path", "tuning_printer", "paper_color", "experiment_selection"):
        assert key not in runner.RESUME_DROPPED_KEYS
    run = tmp_path / "run"
    run.mkdir()
    (run / "config.json").write_text(json.dumps({"bench": {}, "request": "r"}))
    saved = {"dialogue": [], "execution_results": [], "tuning_session_path": "s.json", "tuning_printer": "P",
             "tuning_action": {"action": "submit_trial_print"}, "tuning_failure": "x", "tuning_failures": 2,
             "tuning_polls": 3, "pending_trial_print": {"approval_id": "a"}}
    (run / "state.json").write_text(json.dumps(saved))
    state, _ = runner.load_resume(run, 10)
    assert state["tuning_session_path"] == "s.json" and state["tuning_printer"] == "P"
    assert not {"tuning_action", "tuning_failure", "tuning_failures", "tuning_polls", "pending_trial_print"} & set(state)


# ---------- resume in each phase: the real graph, mocked models and printer ----------

class _Flow(test_tuning_flow.TuningFlowTests):
    """The real-graph harness of test_tuning_flow, without running its tests again here."""
    __test__ = False


@pytest.fixture(scope="module")
def flow_class():
    _Flow.setUpClass()
    yield _Flow
    _Flow.doClassCleanups()


@pytest.fixture
def flow(flow_class):
    harness = flow_class()
    harness.setUp()
    yield harness
    harness.doCleanups()


def resume_like_the_runner(flow, folder: Path, state: dict | None = None):
    """Save the graph state as the runner does, load it with load_resume and run a NEW graph and checkpointer."""
    values = state if state is not None else dict(flow.graph.get_state(flow.config).values)
    folder.mkdir()
    (folder / "config.json").write_text(json.dumps({"bench": {}, "request": "r"}))
    (folder / "state.json").write_text(json.dumps(values, ensure_ascii=False))
    resumed, _ = runner.load_resume(folder, 10)
    flow.build()
    flow.config = {"configurable": {"thread_id": uuid4().hex}, "recursion_limit": 200}
    return resumed, flow.graph.invoke(resumed, flow.config)


def payload(result):
    return result["__interrupt__"][0].value


def test_resume_ready_for_trial_makes_the_sheet_and_asks_before_printing(flow, tmp_path):
    flow.fixture.propose()                                               # phase ready_for_trial, nothing drawn yet
    assert load_tuning_session(flow.path)["phase"] == "ready_for_trial"
    stale = {"tuning_action": {"action": "submit_trial_print", "printer": "Offline"}, "tuning_polls": 2,
             "tuning_failure": "old", "tuning_failures": 2, "pending_trial_print": {"approval_id": "old"}}
    resumed, result = resume_like_the_runner(flow, tmp_path / "saved", {**flow.state, **stale})
    assert not set(stale) & set(resumed)
    assert flow.kind(result) == "trial_print_approval"
    assert flow.submissions == [] and flow.stages == []                  # no print, no second proposal
    assert payload(result)["trial_id"] == load_tuning_session(flow.path)["active_trial_id"]


def test_resume_at_the_approval_keeps_the_trial_and_never_prints_unasked(flow, tmp_path):
    result = flow.start()
    assert flow.kind(result) == "trial_print_approval"
    trial_id = load_tuning_session(flow.path)["active_trial_id"]
    resumed, result = resume_like_the_runner(flow, tmp_path / "saved")
    assert "pending_trial_print" not in resumed and resumed["tuning_printer"] == "Offline"
    assert flow.kind(result) == "trial_print_approval" and payload(result)["trial_id"] == trial_id
    assert flow.submissions == []
    assert load_tuning_session(flow.path)["active_trial_id"] == trial_id   # no new sheet was drawn
    assert flow.kind(flow.resume("approve")) == "tuning_scan"
    assert len(flow.submissions) == 1


def test_resume_awaiting_scan_after_print_never_reprints(flow, tmp_path):
    flow.printed()
    session = load_tuning_session(flow.path)
    trial_id = session["active_trial_id"]
    assert session["phase"] == "awaiting_scan" and len(flow.submissions) == 1
    _, result = resume_like_the_runner(flow, tmp_path / "saved")
    assert flow.kind(result) == "tuning_scan" and payload(result)["trial_id"] == trial_id
    assert len(flow.submissions) == 1
    # build_graph resolves run_dir (on macOS /var/... is /private/var/...), as the runner's run folder is
    assert payload(result)["suggested_scan_directory"] == str((flow.root / "scans" / "round_1").resolve())
    result = flow.resume("scan", scan_paths=["phone/round-1.pdf"])
    assert flow.kind(result) == "trial_print_approval" and payload(result)["round_id"] == 2
    assert len(flow.submissions) == 1                                    # round 1 was not printed again


def test_resume_awaiting_review_reviews_once_without_reprinting(flow, tmp_path):
    flow.printed()
    record_scan_for_executor(flow.path, flow.read_scan(["phone/round-1.pdf"]), record_id="before-crash")
    session = load_tuning_session(flow.path)
    trial_id = session["active_trial_id"]
    assert session["phase"] == "awaiting_review"
    _, result = resume_like_the_runner(flow, tmp_path / "saved")
    assert flow.stages == ["propose_gamma", "review_gamma"]
    assert flow.kind(result) == "trial_print_approval" and payload(result)["round_id"] == 2
    assert len(flow.submissions) == 1
    session = load_tuning_session(flow.path)
    assert session["rounds"]["1"]["trial_result"]["trial_id"] == trial_id and session["active_round"] == 2


def stopped_run(flow):
    record_stop(flow.path, "Human stopped print tuning")
    state = {**flow.state, "stop_reason": "stopped_by_human"}
    (flow.root / "state.json").write_text(json.dumps(state), encoding="utf-8")
    (flow.root / "config.json").write_text(json.dumps({"bench": {}, "request": "r"}), encoding="utf-8")
    return load_tuning_session(flow.path)


def test_ordinary_resume_respects_tuning_stop_and_describes_it_correctly(flow):
    flow.path.write_bytes(flow.printable)
    stopped_run(flow)
    before = flow.path.read_bytes()
    resumed, _ = runner.load_resume(flow.root, 10)
    result = flow.build().invoke(resumed, flow.config)
    assert result["stop_reason"] == "stopped_by_human"
    assert flow.path.read_bytes() == before and flow.submissions == []
    message = runner.human_stop_message(result, flow.root)
    assert "Print tuning was stopped" in message and "budget check" not in message
    assert "--resume-stopped-tuning" in message


@pytest.mark.parametrize("phase", ["initial", "proposal", "trial", "evaluation"])
def test_explicit_recovery_derives_progress_and_preserves_json_history(flow, phase):
    if phase in ("trial", "evaluation"):
        flow.path.write_bytes(flow.printable)
    elif phase == "proposal":
        flow.fixture.propose()
    if phase == "evaluation":
        record_scan_for_executor(flow.path, flow.read_scan(["phone.pdf"]), record_id="saved-evaluation")
    before = stopped_run(flow)
    originals = {p.relative_to(flow.root): p.read_bytes() for p in flow.root.rglob("*.json")
                 if "recovery_backups" not in p.relative_to(flow.root).parts}
    resumed, _ = runner.load_resume(flow.root, 10, resume_stopped_tuning=True)
    recovered = load_tuning_session(flow.path)
    expected = {"initial": "awaiting_gamma_proposal", "proposal": "ready_for_trial",
                "trial": "awaiting_scan", "evaluation": "awaiting_review"}[phase]
    assert recovered["status"] == "active" and recovered["phase"] == expected
    audit = recovered["recoveries"][-1]
    assert audit["source"] == "--resume-stopped-tuning" and audit["stop"] == before["outcomes"][-1]
    backup = Path(audit["backup_path"])
    assert all((backup / name).read_bytes() == content for name, content in originals.items())
    for key in before.keys() - {"status", "phase", "revision", "updated_at"}:
        assert recovered[key] == before[key]
    if phase == "trial":
        result = flow.build().invoke(resumed, flow.config)
        assert flow.kind(result) == "trial_print_approval"
        assert payload(result)["trial_id"] == before["active_trial_id"]
        flow.tools[0].func.assert_not_called()  # The existing trial is reused.
    assert flow.submissions == []


@pytest.mark.parametrize("missing_outcome", [False, True])
def test_recovery_never_duplicates_recorded_or_uncertain_submission(flow, missing_outcome):
    if missing_outcome:
        flow.path.write_bytes(flow.printable)
        session = load_tuning_session(flow.path)
        record_print_approval(flow.path, approval_id="approved", trial_id=session["active_trial_id"],
            pdf_sha256=flow.trial["pdf_sha256"], printer="Offline", print_settings={"copies": 1}, approved_by="test")
        begin_print_attempt(flow.path, approval_id="approved", attempt_id="intent-without-result")
    else:
        flow.printed()
        assert flow.kind(flow.resume("reprint")) == "trial_print_approval"
        assert flow.kind(flow.resume("approve")) == "tuning_scan"
        assert load_tuning_session(flow.path)["rounds"]["1"]["retries"]["reprints"] == 1
    before = stopped_run(flow)
    submissions = len(flow.submissions)
    resumed, _ = runner.load_resume(flow.root, 10, resume_stopped_tuning=True)
    result = flow.build().invoke(resumed, flow.config)
    assert flow.kind(result) == ("tuning_print_confirmation" if missing_outcome else "tuning_scan")
    assert len(flow.submissions) == submissions
    after = load_tuning_session(flow.path)
    assert after["rounds"]["1"]["print_attempts"] == before["rounds"]["1"]["print_attempts"]
    assert after["rounds"]["1"]["retries"] == before["rounds"]["1"]["retries"]


def test_recovery_rejects_changed_trial_without_reopening_or_resetting_limits(flow):
    flow.path.write_bytes(flow.printable)
    stopped_run(flow)
    before = flow.path.read_bytes()
    pdf = Path(flow.trial["pdf_path"])
    original = pdf.read_bytes()
    try:
        pdf.write_bytes(original + b"changed")
        with pytest.raises(ValueError, match="hash"):
            runner.load_resume(flow.root, 10, resume_stopped_tuning=True)
        assert flow.path.read_bytes() == before
    finally:
        pdf.write_bytes(original)


def test_explicit_recovery_rejects_non_stopped_sessions(flow):
    (flow.root / "state.json").write_text(json.dumps(flow.state))
    (flow.root / "config.json").write_text(json.dumps({"bench": {}, "request": "r"}))
    before = flow.path.read_bytes()
    with pytest.raises(ValueError, match="human-stopped"):
        runner.load_resume(flow.root, 10, resume_stopped_tuning=True)
    assert flow.path.read_bytes() == before


@pytest.mark.parametrize("round_id", [2, 3])
def test_recovery_keeps_later_round_winners_and_remaining_limits(flow, round_id):
    flow.printed()
    result = flow.resume("scan", scan_paths=["phone/round-1.pdf"])
    if round_id == 3:
        flow.resume("approve")
        result = flow.resume("scan", scan_paths=["phone/round-2.pdf"])
    assert payload(result)["round_id"] == round_id
    before = stopped_run(flow)
    resumed, _ = runner.load_resume(flow.root, 10, resume_stopped_tuning=True)
    recovered = load_tuning_session(flow.path)
    assert recovered["active_round"] == round_id and recovered["phase"] == "awaiting_scan"
    assert recovered["accepted_settings"] == before["accepted_settings"]
    assert recovered["rounds"] == before["rounds"] and recovered["retry_limits"] == before["retry_limits"]
    submissions = len(flow.submissions)
    result = flow.build().invoke(resumed, flow.config)
    assert flow.kind(result) == "trial_print_approval" and len(flow.submissions) == submissions


def test_budget_and_report_stops_have_distinct_messages(tmp_path):
    assert "budget check" in runner.human_stop_message({}, tmp_path)
    assert "Final-report printing" in runner.human_stop_message({"print_outcome": "stopped"}, tmp_path)
