"""Real selection/preview/sheet helpers, tiny arrays, and supplied evaluation fixtures."""

from copy import deepcopy
from dataclasses import replace
import importlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np

from app.print_tuning.final_figures import render_final_heatmaps
from app.print_tuning.grayscale_render import starting_mapping
from app.print_tuning.tuning_tools import _make_trial_sheet, _session_inputs, _load_selection, create_tuning_tools
from app.print_tuning.schemas import GammaProposalResponse, GammaReviewResponse, GrayLevelsReviewResponse, StepsReviewResponse
from app.workflow.tuning import (
    TuningSessionError, accept_specialist_response, begin_print_attempt, build_tuning_context,
    initialize_tuning_session, load_tuning_session, record_print_approval, record_print_result,
    record_scan_result, record_stop, record_trial_result,
)

IDS = ("exp/A", "exp-B", "exp-C", "exp-D")


def winner(experiment=None, candidate="reference"):
    result = dict(candidate_id=candidate, reason="Readable tested detail", confidence="medium", tie_explanation=None)
    return result | ({"experiment_id": experiment} if experiment else {})


def response(stage, assessment="usable"):
    if stage == "propose_gamma":
        return GammaProposalResponse(stage=stage, experiments=[dict(experiment_id=e, tests=[
            dict(candidate_id=f"test_{i}", gamma=value, reason="Explore tone curve")
            for i, value in enumerate((.4, .6, 1.5, 2.2, 3.), 1)]) for e in IDS])
    common = dict(stage=stage, assessment=assessment, issues=[] if assessment == "usable" else ["Unreliable image" if assessment == "needs_rescan" else "Low measured contrast"])
    usable = assessment == "usable"
    if stage == "review_gamma":
        return GammaReviewResponse(**common, gamma_winners=[winner(e, "test_2") for e in IDS] if usable else [], endpoint_tests=[
            dict(candidate_id=f"test_{i}", dark_level=d, light_level=l, reason="Compare endpoints")
            for i, (d, l) in enumerate(((.02, 1.), (.04, 1.), (.06, .95), (.08, .90), (.1, .85)), 1)] if usable else [])
    if stage == "review_gray_levels":
        return GrayLevelsReviewResponse(**common, endpoint_winner=winner(candidate="test_1") if usable else None, step_tests=[
            dict(experiment_id=e, tests=[dict(candidate_id=f"test_{i}", steps=value, reason="Compare bands")
                                         for i, value in enumerate((2, 4, 8, 16, 32), 1)]) for e in IDS] if usable else [])
    return StepsReviewResponse(**common, step_winners=[winner(e) for e in IDS] if usable else [])


class SessionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name) / "run"
        self.root.mkdir()
        self.path = self.root / "tuning_session.json"
        self.results, selections = [], []
        for index, experiment in enumerate(IDS):
            folder = self.root / f"source_{index}"
            folder.mkdir()
            initial, final = folder / "initial.npy", folder / "final.npy"
            np.save(initial, np.arange(16, dtype=float).reshape(4, 4) + index)
            np.save(final, np.arange(16, dtype=float).reshape(4, 4) * 2 + index)
            record = dict(ok=True, experiment_id=experiment, settings={"iterations": 3, "focus_size": 1},
                          initial_enhancement=1., final_enhancement=2., paths={"initial_intensity": str(initial),
                          "final_intensity": str(final), "initial_camera": str(folder / "initial.png"),
                          "final_camera": str(folder / "final.png"), "results_json": str(folder / "results.json")})
            (folder / "results.json").write_text(json.dumps(record), encoding="utf-8")
            self.results.append(record)
            selections.append(dict(experiment_id=experiment, validity="Valid", comparability="Comparable",
                                   relevance="Relevant", information_gained="Distinct evidence", paths=record["paths"]))
        self.selection = dict(selections=selections, group_reason="Four complementary experiments")
        (self.root / "config.json").write_text(json.dumps({"bench": {"target_position": [1, 1]}}))

    def init(self, **kwargs):
        return initialize_tuning_session(self.root, self.selection, self.results, **kwargs)

    def propose(self):
        context = build_tuning_context(self.path, "propose_gamma")
        proposal = response("propose_gamma")
        return accept_specialist_response(self.path, proposal, context)

    def trial(self, number):
        before = self.path.read_bytes()
        result = _make_trial_sheet(self.root, number)
        self.assertEqual(before, self.path.read_bytes(), "Existing tool must not write the session")
        return record_trial_result(self.path, result), result

    def scan(self, trial, *, name="scan", status="ok", usable=True):
        quality = {"usable": usable, "issues": [] if usable else ["Blurred"]}
        result = {k: trial[k] for k in ("contract_version", "session_id", "trial_id", "round_id")}
        result.update(status=status, scan_quality=quality)
        if status == "ok":
            evaluation_path = self.root / f"{name}_{trial['trial_id']}.json"
            evaluation = dict(result, schema_version="scan_eval/2", tested_parameter={1: "gamma", 2: "gray_levels", 3: "steps"}[trial["round_id"]],
                              manifest_path=trial["manifest_path"], pages=[], candidates=[], evaluation_path=str(evaluation_path))
            evaluation_path.write_text(json.dumps(evaluation))
            result.update(evaluation_path=str(evaluation_path), pages=[], candidates=[])
        else:
            result.update(error_code="low_dpi", message="Rescan required")
        session = record_scan_result(self.path, result, record_id=name)
        return session, result

    def review(self, stage, name="scan", assessment="usable"):
        return accept_specialist_response(self.path, response(stage, assessment), build_tuning_context(self.path, stage), evaluation_record_id=name)

    def round_one(self):
        self.init()
        self.propose()
        _, trial = self.trial(1)
        self.scan(trial)
        return trial

    def test_initialization_uses_existing_readers_and_previews(self):
        original = deepcopy(self.selection)
        session = self.init()
        self.assertEqual(self.selection, original)
        self.assertEqual(session["phase"], "awaiting_gamma_proposal")
        self.assertEqual([e["experiment_id"] for e in session["selected_experiments"]], list(IDS))
        self.assertEqual(len(session["previews"]["previews"]), 8)
        self.assertTrue(all(Path(p["path"]).is_file() for p in session["previews"]["previews"]))
        selection, association = _load_selection(_session_inputs(session, None))
        self.assertEqual(selection["selection_id"], session["selection_id"])
        self.assertEqual(selection["items"][0]["source"]["reasons"]["validity"], "Valid")
        self.assertEqual(association[IDS[0]], "exp_A")
        self.assertEqual(set(session["rounds"]), {"1", "2", "3"})
        self.assertIsNone(session["final_settings"])
        self.propose()
        self.assertEqual(_session_inputs(load_tuning_session(self.path), 1)["proposal"]["stage"], "propose_gamma")

    def test_colored_paper_presets_match_previews_trials_and_resume(self):
        paper = {"paper_session_id": "yellow-stock", "paper_color": "yellow"}
        original = deepcopy(paper)
        session = self.init(paper_session=paper)
        mapping, preset = starting_mapping(session["config_snapshot"], "yellow")
        self.assertEqual(paper, original)
        self.assertEqual(session["paper_color"], "yellow")
        self.assertEqual(session["previews"]["paper_color"], "yellow")
        self.assertEqual(session["previews"]["paper_preset"], preset)
        self.assertEqual(session["accepted_settings"], {e: mapping for e in IDS})
        self.assertEqual(session["previews"]["reference_mappings"], session["accepted_settings"])
        before = self.path.read_bytes()
        with patch("app.workflow.tuning.prepare_specialist_previews", side_effect=AssertionError("Do not recompute")):
            self.assertEqual(self.init(), session)
        self.assertEqual(self.path.read_bytes(), before)
        with self.assertRaisesRegex(TuningSessionError, "paper color"):
            self.init(paper_color="blue")
        self.assertEqual(self.path.read_bytes(), before)
        self.propose()
        _, trial = self.trial(1)
        manifest = json.loads(Path(trial["manifest_path"]).read_text())
        for entries in manifest["variations"].values():
            self.assertTrue(all(v["mapping"]["dark_level"] == mapping["dark_level"] for v in entries))

    def test_paper_color_argument_and_invalid_metadata(self):
        for value in (False, 12, "", " "):
            with self.subTest(value=value), self.assertRaises(TuningSessionError):
                self.init(paper_color=value)
        with self.assertRaisesRegex(TuningSessionError, "Conflicting paper colors"):
            self.init(paper_color="blue", paper_session={"paper_session_id": "stock", "paper_color": "yellow"})
        self.assertFalse(self.path.exists())
        session = self.init(paper_color="yellow")
        self.assertEqual(session["paper_color"], "yellow")
        self.assertEqual(session["previews"]["paper_color"], "yellow")
        self.assertEqual(session, self.init(paper_color="yellow"))

    def test_reload_and_repeated_initialization_never_recompute_limits(self):
        first = self.init()
        original = self.path.read_bytes()
        Path(self.results[0]["paths"]["initial_intensity"]).unlink()
        with patch("app.workflow.tuning.prepare_specialist_previews", side_effect=AssertionError("Do not recompute")):
            second = self.init()
            loaded = load_tuning_session(self.path)
        self.assertEqual(first, second)
        self.assertEqual(loaded["selected_experiments"], first["selected_experiments"])
        self.assertEqual(original, self.path.read_bytes())

    def test_read_scan_tool_uses_recorded_active_trial_without_session_writes(self):
        self.init()
        self.propose()
        session, trial = self.trial(1)
        before = self.path.read_bytes()
        supplied = {"trial_id": trial["trial_id"], "evaluation_path": str(self.root / "supplied.json"),
                    "scan_quality": {"usable": True, "issues": []}, "pages": [], "candidates": []}
        with patch("app.print_tuning.tuning_tools.evaluate_scan", return_value=supplied) as evaluate:
            result = create_tuning_tools(self.root)[1].invoke({"scan_paths": ["explicit.png"]})
        self.assertEqual(result["session_id"], session["session_id"])
        self.assertEqual(result["trial_id"], trial["trial_id"])
        self.assertEqual(evaluate.call_args.args[1], trial["manifest_path"])
        self.assertEqual(before, self.path.read_bytes())

    def test_incompatible_selection_and_config_do_not_overwrite(self):
        session = self.init()
        original = self.path.read_bytes()
        changed = deepcopy(self.selection)
        changed["selections"].reverse()
        with self.assertRaises(TuningSessionError):
            initialize_tuning_session(self.root, changed, self.results)
        config = deepcopy(session["config_snapshot"])
        config["parameters"]["gamma"]["max"] = 4.
        with self.assertRaises(TuningSessionError):
            self.init(config=config)
        self.assertEqual(original, self.path.read_bytes())

    def test_missing_arrays_and_deliberate_partial_retry(self):
        source = Path(self.results[0]["paths"]["initial_intensity"])
        saved = source.read_bytes()
        source.unlink()
        with self.assertRaisesRegex(TuningSessionError, "raw initial"):
            self.init()
        partial = load_tuning_session(self.path, allow_incomplete=True)
        self.assertEqual(partial["initialization"]["status"], "error")
        with self.assertRaisesRegex(TuningSessionError, "incomplete"):
            load_tuning_session(self.path)
        with self.assertRaisesRegex(TuningSessionError, "deliberate"):
            self.init()
        source.write_bytes(saved)
        session = self.init(retry_initialization=True)
        self.assertEqual(session["session_id"], partial["session_id"])
        self.assertEqual(len(session["initialization"]["attempts"]), 2)

    def test_bootstrap_precedes_previews_and_failed_preview_is_not_ready(self):
        from app.print_tuning.tuning_tools import prepare_specialist_previews
        def failed(path, out):
            bootstrap = json.loads(Path(path).read_text())
            _session_inputs(bootstrap, None)
            self.assertEqual(bootstrap["initialization"]["status"], "initializing")
            self.assertTrue(Path(bootstrap["selection_path"]).is_file())
            raise RuntimeError("interrupted")
        with patch("app.workflow.tuning.prepare_specialist_previews", side_effect=failed):
            with self.assertRaises(TuningSessionError):
                self.init()
        self.assertEqual(load_tuning_session(self.path, allow_incomplete=True)["status"], "error")
        self.init(retry_initialization=True)

    def test_all_four_decisions_progress_to_compatible_final_settings(self):
        self.init()
        self.propose()
        for number, stage in ((1, "review_gamma"), (2, "review_gray_levels"), (3, "review_steps")):
            _, trial = self.trial(number)
            _, result = self.scan(trial, name=f"scan{number}")
            context = build_tuning_context(self.path, stage)
            original = deepcopy(context)
            decision = response(stage)
            session = accept_specialist_response(self.path, decision, context, evaluation_record_id=f"scan{number}")
            self.assertEqual(context, original)
            if number < 3:
                self.assertIsNone(session["final_settings"])
                self.assertEqual(_session_inputs(session, number + 1)["proposal"]["stage"], stage)
                self.assertEqual(session["active_round"], number + 1)
            else:
                self.assertEqual(session["status"], "completed")
        self.assertEqual(set(session["rounds"]), {"1", "2", "3"})
        for mapping in session["final_settings"].values():
            self.assertEqual(mapping["gamma"], .6)
            self.assertEqual(mapping["dark_level"], .02)
            self.assertEqual(mapping["steps"], 0)
        figure = render_final_heatmaps(self.path, self.root / "final")
        self.assertTrue(Path(figure["figure_path"]).is_file())

    def test_invalid_response_leaves_entire_session_unchanged(self):
        self.round_one()
        before = self.path.read_bytes()
        decision = response("review_gamma")
        decision.endpoint_tests[0].dark_level = .9
        with self.assertRaises(TuningSessionError):
            accept_specialist_response(self.path, decision, build_tuning_context(self.path, "review_gamma"), evaluation_record_id="scan")
        self.assertEqual(before, self.path.read_bytes())

    def test_full_validation_precedes_resolution_and_atomic_write(self):
        self.round_one()
        import app.workflow.tuning as tuning
        events = []
        original_validate, original_resolve = tuning.validate_response, tuning.resolve_review_winners
        def validate(*args):
            events.append("validate")
            return original_validate(*args)
        def resolve(*args):
            events.append("resolve")
            return original_resolve(*args)
        with patch.object(tuning, "validate_response", side_effect=validate), patch.object(tuning, "resolve_review_winners", side_effect=resolve):
            self.review("review_gamma")
        self.assertEqual(events, ["validate", "resolve"])
        session = load_tuning_session(self.path)
        self.assertTrue(all(m["gamma"] == .6 for m in session["rounds"]["2"]["reference_settings"].values()))

    def test_wrong_context_stale_trial_and_out_of_order_review(self):
        self.init()
        with self.assertRaises(TuningSessionError):
            build_tuning_context(self.path, "review_gamma")
        self.propose()
        _, trial = self.trial(1)
        self.scan(trial)
        context = build_tuning_context(self.path, "review_gamma")
        before = self.path.read_bytes()
        for changed in (replace(context, selected_experiment_ids=list(reversed(IDS))),
                        replace(context, expected_trial_id="stale"), replace(context, expected_calibration_session_id="other")):
            with self.assertRaises(TuningSessionError):
                accept_specialist_response(self.path, response("review_gamma"), changed, evaluation_record_id="scan")
        with self.assertRaises(TuningSessionError):
            accept_specialist_response(self.path, response("review_gamma"), context, evaluation_record_id="stale")
        self.assertEqual(before, self.path.read_bytes())

    def test_rescan_and_contrast_preserve_accepted_settings(self):
        trial = self.round_one()
        before = load_tuning_session(self.path)["accepted_settings"]
        session = self.review("review_gamma", assessment="needs_rescan")
        self.assertEqual(session["phase"], "awaiting_scan")
        self.assertEqual(session["accepted_settings"], before)
        self.assertIsNone(session["rounds"]["2"]["proposal"])
        self.scan(trial, name="replacement")
        session = self.review("review_gamma", name="replacement", assessment="insufficient_contrast")
        self.assertEqual(session["status"], "blocked")
        self.assertIsNone(session["final_settings"])
        self.assertEqual(len(session["rounds"]["1"]["evaluations"]), 2)

    def test_review_requires_recorded_usable_evidence(self):
        self.init()
        self.propose()
        _, trial = self.trial(1)
        context = build_tuning_context(self.path, "review_gamma")
        with self.assertRaises(TuningSessionError):
            accept_specialist_response(self.path, response("review_gamma"), context, evaluation_record_id="none")
        self.scan(trial, status="needs_rescan", usable=False)
        with self.assertRaises(TuningSessionError):
            self.review("review_gamma")
        self.review("review_gamma", assessment="needs_rescan")

    def test_duplicate_acceptance_is_a_noop_and_conflicting_repeat_rejected(self):
        self.init()
        context = build_tuning_context(self.path, "propose_gamma")
        decision = response("propose_gamma")
        accept_specialist_response(self.path, decision, context)
        before = self.path.read_bytes()
        accept_specialist_response(self.path, decision, context)
        self.assertEqual(before, self.path.read_bytes())
        _, trial = self.trial(1)
        self.scan(trial)
        context = build_tuning_context(self.path, "review_gamma")
        decision = response("review_gamma")
        accept_specialist_response(self.path, decision, context, evaluation_record_id="scan")
        before = self.path.read_bytes()
        accept_specialist_response(self.path, decision, context, evaluation_record_id="scan")
        self.assertEqual(before, self.path.read_bytes())
        decision.gamma_winners[0].reason = "Different decision"
        with self.assertRaisesRegex(TuningSessionError, "Conflicting"):
            accept_specialist_response(self.path, decision, context, evaluation_record_id="scan")

    def test_regeneration_and_replacement_scans_retain_history(self):
        self.init()
        self.propose()
        _, first = self.trial(1)
        self.scan(first)
        _, second = self.trial(1)
        self.assertNotEqual(first["trial_id"], second["trial_id"])
        with self.assertRaises(TuningSessionError):
            self.scan(first, name="stale")
        session, result = self.scan(second, name="new")
        entry = session["rounds"]["1"]
        self.assertEqual(len(entry["trials"]), 2)
        self.assertEqual(len(entry["evaluations"]), 2)
        self.assertTrue(Path(first["pdf_path"]).is_file())
        before = self.path.read_bytes()
        record_scan_result(self.path, result, record_id="new")
        self.assertEqual(before, self.path.read_bytes())

    def test_trial_artifact_and_proposal_tampering_rejected(self):
        self.init()
        self.propose()
        trial = _make_trial_sheet(self.root, 1)
        before = self.path.read_bytes()
        with self.assertRaises(TuningSessionError):
            record_trial_result(self.path, trial | {"pdf_sha256": "wrong"})
        manifest = json.loads(Path(trial["manifest_path"]).read_text())
        manifest["items"][0]["limits"]["hi"] += 1
        Path(trial["manifest_path"]).write_text(json.dumps(manifest))
        with self.assertRaises(TuningSessionError):
            record_trial_result(self.path, trial)
        self.assertEqual(before, self.path.read_bytes())

    def test_print_intent_before_submission_and_one_approved_reprint(self):
        self.init()
        self.propose()
        _, trial = self.trial(1)
        def approve(key, reprint=False):
            return record_print_approval(self.path, approval_id=key, trial_id=trial["trial_id"],
                                         pdf_sha256=trial["pdf_sha256"], printer="mock-printer",
                                         print_settings={"paper": "A4", "scaling": "none", "copies": 1},
                                         approved_by="test user", reprint=reprint)
        approve("approval1")
        session = begin_print_attempt(self.path, approval_id="approval1", attempt_id="attempt1")
        self.assertEqual(session["rounds"]["1"]["print_attempts"][0]["submission_state"], "outcome_uncertain")
        with self.assertRaises(TuningSessionError):
            approve("approval2", True)
        raw = {"job_id": "opaque-job", "state": "vendor-specific", "extra": "preserve"}
        record_print_result(self.path, attempt_id="attempt1", result=raw)
        record_print_result(self.path, attempt_id="attempt1", result={"reading": "opaque"}, job_result=True)
        approve("approval2", True)
        begin_print_attempt(self.path, approval_id="approval2", attempt_id="attempt2")
        record_print_result(self.path, attempt_id="attempt2", result={"opaque": "result"})
        with self.assertRaises(TuningSessionError):
            approve("approval3", True)
        self.assertEqual(load_tuning_session(self.path)["rounds"]["1"]["print_attempts"][0]["result"], raw)

    def test_atomic_write_failure_preserves_prior_session(self):
        self.init()
        before = self.path.read_bytes()
        with patch("app.workflow.tuning.os.replace", side_effect=OSError("interrupted replacement")):
            with self.assertRaises(TuningSessionError):
                self.propose()
        self.assertEqual(before, self.path.read_bytes())
        self.assertEqual(list(self.root.glob("tuning_session.json.*.tmp")), [])

    def test_stop_and_error_only_record_outcomes(self):
        self.init()
        record_stop(self.path, "Human stopped")
        session = record_stop(self.path, "Operational issue", error=True, metadata={"code": "supplied"})
        self.assertEqual(session["status"], "error")
        self.assertEqual(len(session["outcomes"]), 2)
        self.assertIsNone(session["final_settings"])

    def test_graph_hook_for_new_and_existing_frozen_selection_with_mocks(self):
        graph = importlib.import_module("app.workflow.graph")
        state = {"dialogue": [], "paper_color": "yellow", "execution_results": [dict(tool_name="run_focusing_experiment", result=r) for r in self.results]}
        ready = dict(self.selection, status="ready", analysis="Evidence", proposed_experiments=[])
        tools = [Mock(name="unused") for _ in range(3)]
        with patch.object(graph, "StateGraph") as builder, patch.object(graph, "run_scientist", return_value={"scientist_response": ready, "dialogue": []}), \
                patch.object(graph, "_load_scientist_evidence", return_value={"experiments": [{"experiment_id": e} for e in IDS]}), \
                patch.object(graph, "initialize_tuning_session", return_value={"initialization": {"status": "complete"}, "session_id": "session"}) as initialize:
            graph.build_graph(Mock(), tools[0], tools[1], [], tools[2], self.root, 5, None)
            nodes = {call.args[0]: call.args[1] for call in builder.return_value.add_node.call_args_list}
            update = nodes["scientist"](state)
            self.assertEqual(update["tuning_initialization"]["status"], "complete")
            self.assertEqual([s["experiment_id"] for s in update["experiment_selection"]["selections"]], list(IDS))
            self.assertEqual(initialize.call_count, 1)
            self.assertEqual(initialize.call_args.kwargs["paper_color"], "yellow")
            frozen = state | {"experiment_selection": update["experiment_selection"], "decision_count": 10, "decision_budget": 10}
            budget = nodes["orchestrator"](frozen)
            self.assertEqual(budget["route"], "budget")
            self.assertEqual(initialize.call_count, 2)
            self.assertEqual(initialize.call_args.kwargs["paper_color"], "yellow")
            initialize.side_effect = TuningSessionError("missing array")
            failed = nodes["orchestrator"](frozen)
            self.assertEqual(failed["tuning_initialization"]["status"], "error")


if __name__ == "__main__":
    unittest.main()
