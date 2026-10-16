"""The terminal runner and print tuning: tuning pauses as dicts, scan folders, tools and resume."""

import json
import os
from pathlib import Path
from uuid import uuid4

import pytest

from app.print_tuning.tuning_tools import create_tuning_tools
from app.workflow import runner
from app.workflow.tuning import load_tuning_session, record_scan_for_executor
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
                 "A4, grayscale, one-sided, Actual size / 100%"):
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


@pytest.mark.parametrize("choice", ["printed", "not_printed", "retry", "skip", "stop"])
def test_report_print_confirmation_returns_payload_choice_as_action(monkeypatch, choice):
    # Choices belong to the graph's payload, not a second hardcoded runner list.
    payload = {"kind": "report_print_confirmation", "question": "Confirm the report print outcome.",
               "choices": ["printed", "not_printed", "retry", "skip", "stop"]}
    answers(monkeypatch, "yes", choice)
    assert runner.ask_human(payload) == {"action": choice}


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
    assert runner.ask_human({"kind": "print_approval", "pdf_path": "r.pdf", "pages": 1, "printer": "P",
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
