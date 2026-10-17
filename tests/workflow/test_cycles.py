"""Versioned multi-cycle sessions, real offline trial generation, no hardware/models."""
from copy import deepcopy
from dataclasses import replace
import json
import numpy as np
import pytest
from app.workflow import tuning
from app.print_tuning.tuning_tools import _make_trial_sheet
from tests.workflow import test_tuning as fixtures


@pytest.fixture
def flow():
    f = fixtures.SessionTests(); f.setUp()
    try:
        f.init(max_cycles=2)
        yield f
    finally:
        f.doCleanups()


def scan(f, trial):
    result = {k: trial[k] for k in ("contract_version", "session_id", "trial_id", "round_id", "cycle_id", "step_id")}
    path = f.root / (trial["trial_id"] + "_evaluation.json")
    result.update(status="ok", scan_quality={"usable": True, "issues": []}, evaluation_path=str(path), pages=[], candidates=[])
    path.write_text(json.dumps(result))
    tuning.record_scan_result(f.path, result, record_id=trial["trial_id"])


def run_cycle(f, *, references=False, exercise_reprint=False, stop_after=None):
    session = tuning.load_tuning_session(f.path)
    proposal = fixtures.response("propose_gamma")
    # Fresh candidates around the CURRENT reference, avoiding the old winner.
    for experiment in proposal.experiments:
        used = {session["accepted_settings"][experiment.experiment_id]["gamma"]}
        options = [v for v in (.3, .4, .5, .7, .9, 1.3, 1.7, 2.5) if v not in used][:5]
        for test, value in zip(experiment.tests, options): test.gamma = value
    context = tuning.build_tuning_context(f.path, "propose_gamma")
    tuning.accept_specialist_response(f.path, proposal, context)
    for step, stage in enumerate(("review_gamma", "review_gray_levels", "review_steps"), 1):
        before = deepcopy(tuning.load_tuning_session(f.path)["accepted_settings"])
        trial = _make_trial_sheet(f.root, step)
        tuning.record_trial_result(f.path, trial)
        if exercise_reprint and step == 1:
            for attempt in (0, 1):
                if attempt: tuning.request_trial_reprint(f.path)
                approval_id = f"c{session['active_cycle']}-approval-{attempt}"
                tuning.record_print_approval(f.path, approval_id=approval_id, trial_id=trial["trial_id"],
                    pdf_sha256=trial["pdf_sha256"], printer="Offline", print_settings={"sides": "two-sided-long-edge"},
                    approved_by="offline human fixture", reprint=bool(attempt))
                tuning.begin_print_attempt(f.path, approval_id=approval_id, attempt_id=approval_id + ":intent")
                # Known local preflight failure fixture, not physical submission.
                tuning.record_print_result(f.path, attempt_id=approval_id + ":intent", result={"ok": False, "error": "offline preflight fixture"})
            with pytest.raises(tuning.TuningSessionError, match="One reprint"):
                tuning.request_trial_reprint(f.path)
        scan(f, trial)
        response = fixtures.response(stage)
        if step == 1:
            # Avoid current shared endpoints in subsequent-cycle test proposals.
            endpoints = {(m["dark_level"], m["light_level"]) for m in before.values()}
            options = [(d, 1.) for d in (.01, .03, .05, .07, .09, .11) if (d, 1.) not in endpoints][:5]
            for test, (dark, light) in zip(response.endpoint_tests, options):
                test.dark_level, test.light_level = dark, light
            if references:
                for win in response.gamma_winners: win.candidate_id = "reference"
        if step == 2 and references: response.endpoint_winner.candidate_id = "reference"
        context = tuning.build_tuning_context(f.path, stage)
        after = tuning.accept_specialist_response(f.path, response, context, evaluation_record_id=trial["trial_id"])
        allowed = {1: {"gamma"}, 2: {"dark_level", "light_level"}, 3: {"steps"}}[step]
        assert all(mapping[key] == before[eid][key] for eid, mapping in after["accepted_settings"].items()
                   for key in mapping if key not in allowed)
        if step == stop_after:
            return after
    return after


def test_cycle_two_carries_complete_mappings_history_and_stale_context(flow):
    first = run_cycle(flow, exercise_reprint=True)
    assert first["phase"] == "awaiting_cycle_decision" and first["final_settings"] is None
    frozen = deepcopy(first["rounds"])
    baseline = deepcopy(first["original_settings"])
    saved = tuning.decide_cycle(flow.path, "continue")
    assert saved["active_cycle"] == 2 and saved["active_step"] == 1
    assert saved["accepted_settings"] == first["accepted_settings"]
    assert saved["cycles"]["1"]["steps"] == frozen
    assert saved["cycles"]["1"]["steps"]["1"]["retries"]["reprints"] == 1
    assert saved["rounds"]["1"]["retries"]["reprints"] == 0
    context = tuning.build_tuning_context(flow.path, "propose_gamma")
    assert context.reference_mappings == first["accepted_settings"] and context.previous_cycle_evidence
    with pytest.raises(tuning.TuningSessionError, match="prior cycle"):
        tuning.accept_specialist_response(flow.path, fixtures.response("propose_gamma"), replace(context, cycle_id=1))
    old = first["rounds"]["1"]["trial_result"]
    with pytest.raises(tuning.TuningSessionError, match="cycle"):
        tuning.record_trial_result(flow.path, old)
    second = run_cycle(flow, references=True)
    assert second["rounds"]["1"]["reference_settings"] == first["accepted_settings"]
    assert second["cycles"]["1"]["steps"] == frozen and second["original_settings"] == baseline
    with pytest.raises(tuning.TuningSessionError, match="Maximum"):
        tuning.decide_cycle(flow.path, "continue")
    final = tuning.decide_cycle(flow.path, "finish")
    assert final["final_settings"] == second["accepted_settings"] and final["phase"] == "completed"
    assert final["metric_history"]["labels"] == ["Baseline", "C1-S1", "C1-S2", "C1-S3", "C2-S1", "C2-S2", "C2-S3"]
    # Deliberately sparse evaluation fixtures produce explicit missing points, never zeros.
    assert all(p["metrics"] is None and p["reason"] for e in final["metric_history"]["experiments"] for p in e["points"])


def test_cycle_stop_retains_decision_and_reopens_boundary(flow, tmp_path):
    run_cycle(flow, references=True)
    stopped = tuning.decide_cycle(flow.path, "stop")
    assert stopped["cycle_decisions"][-1]["choice"] == "stop"
    backup = tmp_path / "backup"; backup.mkdir()
    (backup / flow.path.name).write_bytes(flow.path.read_bytes())
    restored = tuning.recover_stopped_tuning(flow.path, backup_path=backup)
    assert restored["phase"] == "awaiting_cycle_decision"
    assert restored["rounds"] == stopped["rounds"]


def test_human_boundary_choices_maximum_and_no_implicit_approval(flow):
    from unittest.mock import Mock, patch
    import importlib
    graph = importlib.import_module("app.workflow.graph")
    run_cycle(flow, references=True)
    before = flow.path.read_bytes()
    builder = Mock()
    with patch.object(graph, "StateGraph", return_value=builder):
        graph.build_graph(Mock(), Mock(), Mock(), [], Mock(), flow.root, 5, None)
    nodes = {call.args[0]: call.args[1] for call in builder.add_node.call_args_list}
    state = {"tuning_session_path": str(flow.path)}
    with patch.object(graph, "interrupt", return_value={"action": "continue"}) as pause:
        result = nodes["tuning_cycle"](state)
    assert pause.call_args.args[0]["choices"] == ["continue", "finish", "stop"]
    assert result["pending_trial_print"] is None and tuning.load_tuning_session(flow.path)["active_cycle"] == 2
    saved = json.loads(before); saved["max_cycles"] = 1
    flow.path.write_text(json.dumps(saved))
    with patch.object(graph, "interrupt", return_value={"action": "finish"}) as pause:
        nodes["tuning_cycle"](state)
    assert pause.call_args.args[0]["choices"] == ["finish", "stop"]
    assert tuning.load_tuning_session(flow.path)["status"] == "completed"


def test_legacy_session_read_is_unchanged_and_globally_finished():
    f = fixtures.SessionTests(); f.setUp()
    try:
        f.init(); f.propose()
        for step, stage in enumerate(("review_gamma", "review_gray_levels", "review_steps"), 1):
            _, trial = f.trial(step); f.scan(trial); f.review(stage)
        before = f.path.read_bytes()
        session = tuning.load_tuning_session(f.path)
        assert session["contract_version"] == 1 and session["phase"] == "completed"
        assert f.path.read_bytes() == before
    finally:
        f.doCleanups()


def test_version_two_midcycle_stop_recovery_keeps_accepted_step(flow, tmp_path):
    accepted = run_cycle(flow, stop_after=1)
    tuning.record_stop(flow.path, "Human stopped print tuning")
    backup = tmp_path / "backup"; backup.mkdir()
    (backup / flow.path.name).write_bytes(flow.path.read_bytes())
    restored = tuning.recover_stopped_tuning(flow.path, backup_path=backup)
    assert restored["phase"] == "ready_for_trial" and restored["active_step"] == 2
    assert restored["rounds"] == accepted["rounds"] and restored["accepted_settings"] == accepted["accepted_settings"]
