"""Real LangGraph interrupts/session/evidence, mocked models and offline tools."""

from copy import deepcopy
import importlib
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
from uuid import uuid4

from langchain_core.messages import AIMessage
from langchain_core.tools import StructuredTool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph
from langgraph.types import Command
from PIL import Image

from app.print_tuning.tuning_tools import create_tuning_tools
from app.workflow.state import State
from app.workflow.tuning import load_tuning_session, record_print_approval
from tests.workflow import test_tuning as fixtures
from tests.workflow import test_tuning_evidence as evidence_fixtures

graph_module = importlib.import_module("app.workflow.graph")


class TuningFlowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture = fixtures.SessionTests()
        cls.fixture.setUp()
        cls.addClassCleanup(cls.fixture.doCleanups)
        cls.fixture.init()
        cls.initial = cls.fixture.path.read_bytes()
        cls.fixture.propose()
        _, cls.trial = cls.fixture.trial(1)
        cls.printable = cls.fixture.path.read_bytes()

    def setUp(self):
        self.root, self.path = self.fixture.root, self.fixture.path
        self.path.write_bytes(self.initial)
        (self.root / "report_print.json").unlink(missing_ok=True)
        self.crop = self.root / "flow-crop.png"
        Image.new("RGB", (3, 2), (90, 80, 70)).save(self.crop)
        self.scans, self.stages, self.submissions = [], [], []
        self.job_states = []
        self.scan_status = "ok"
        self.assessments = []
        self.model_fail = None
        self.next_submission = None
        self.llm, self.specialist = Mock(), Mock()
        self.llm.profile = {"image_inputs": True}
        self.specialist.profile = {"image_inputs": True}
        self.llm.bind_tools.side_effect = self.bind_tools
        self.specialist.with_structured_output.side_effect = self.structured
        self.report = Mock(side_effect=self.report_result)
        self.addCleanup(patch.stopall)
        patch.object(graph_module, "run_orchestrator", side_effect=AssertionError("No tuning orchestrator calls")).start()
        patch.object(graph_module, "run_report_and_print", self.report).start()
        self.tools = create_tuning_tools(self.root)
        # Reuse the real generated round-1 artifact; later rounds use the real producer.
        original = self.tools[0].func
        def make(round_id):
            return deepcopy(self.trial) if round_id == 1 else original(round_id)
        self.tools[0] = self.tools[0].model_copy(update={"func": Mock(side_effect=make)})
        self.tools[1] = self.tools[1].model_copy(update={"func": Mock(side_effect=self.read_scan)})
        self.printers = []
        for name, fields, function in (
            ("printer_status", ("printer",), lambda **_: {"ok": True, "state": "idle"}),
            ("job_status", ("job_id",), self.job_status),
            ("submit_trial_print", ("pdf_path", "printer"), self.submit),
        ):
            self.printers.append(StructuredTool(name=name, description="Offline fake boundary",
                func=Mock(side_effect=function), args_schema={"type": "object", "required": list(fields),
                    "properties": {f: {"type": "string"} for f in fields}, "additionalProperties": False}))
        self.nodes = {}
        self.config = {"configurable": {"thread_id": uuid4().hex}, "recursion_limit": 200}
        self.state = {"dialogue": [], "execution_results": [
            {"tool_name": "run_focusing_experiment", "result": r} for r in self.fixture.results],
            "experiment_selection": deepcopy(self.fixture.selection), "tuning_session_path": str(self.path),
            "tuning_printer": "Offline", "decision_count": 99, "decision_budget": 1}

    def build(self, *, dependencies=True, separate=True):
        builder = StateGraph(State)
        original = builder.add_node
        def capture(name, node, *args, **kwargs):
            self.nodes[name] = node
            return original(name, node, *args, **kwargs)
        with patch.object(graph_module, "StateGraph", return_value=builder), patch.object(builder, "add_node", side_effect=capture):
            self.graph = graph_module.build_graph(self.llm, Mock(), Mock(), self.printers, Mock(), self.root, 5,
                InMemorySaver(), tuning_tools=self.tools if dependencies else None,
                specialist_llm=self.specialist if separate else None)
        return self.graph

    def bind_tools(self, tools, **options):
        def invoke(messages):
            if options.get("tool_choice") == "none":
                return AIMessage(content="Offline action completed")
            if self.model_fail:
                return self.model_fail()
            context = json.loads(messages[1].content.split("\n", 1)[1])
            name = context["requested_action"]
            args = {
                "make_trial_sheet": {"round_id": context["round_id"]},
                "read_scan": {"scan_paths": context["scan_paths"]},
                "printer_status": {"printer": context["printer"]},
                "job_status": {"job_id": context["job_id"]},
                "submit_trial_print": {"pdf_path": (context.get("trial") or {}).get("pdf_path"), "printer": context["printer"]},
            }[name]
            return AIMessage(content="", tool_calls=[{"id": "offline-call", "name": name, "args": args}])
        return SimpleNamespace(invoke=invoke)

    def structured(self, schema, **kwargs):
        stage = schema.model_fields["stage"].annotation.__args__[0]
        def invoke(messages):
            self.stages.append(stage)
            # The real specialist has loaded actual image parts from the adapter.
            self.assertTrue(any(p["type"] == "image_url" for p in messages[1].content))
            assessment = self.assessments.pop(0) if self.assessments and stage != "propose_gamma" else "usable"
            decision = fixtures.response(stage, assessment)
            if assessment == "usable":
                # Keep reference valid throughout, including steps=0 final winner.
                if stage == "review_gamma":
                    for winner in decision.gamma_winners:
                        winner.candidate_id = "reference"
            return {"parsed": decision, "raw": None, "parsing_error": None}
        return SimpleNamespace(invoke=invoke)

    def submit(self, **args):
        self.submissions.append(args)
        # The actual session intent MUST precede the fake external effect.
        attempt = self.session()["rounds"][str(self.session()["active_round"])]["print_attempts"][-1]
        self.assertIsNone(attempt["result"])
        if isinstance(self.next_submission, Exception):
            raise self.next_submission
        return self.next_submission or {"ok": True, "job_id": f"offline-{len(self.submissions)}", "tracked": True}

    def job_status(self, job_id):
        state = self.job_states.pop(0) if self.job_states else "done"
        return {"ok": True, "job_id": job_id, "state": state, "confirmed": state != "uncertain"}

    def read_scan(self, scan_paths):
        self.scans.append(scan_paths)
        session = self.session()
        trial = session["rounds"][str(session["active_round"])]["trial_result"]
        result = {k: trial[k] for k in ("contract_version", "session_id", "trial_id", "round_id")}
        if self.scan_status == "error":
            return {"contract_version": 1, "status": "error", "error_code": "unreadable", "message": "Cannot open supplied scan"}
        if self.scan_status == "needs_rescan":
            return {**result, "status": "needs_rescan", "scan_quality": {"usable": False, "issues": ["Blurred"]}}
        shell = SimpleNamespace(root=self.root, crop=self.crop)
        evaluation = evidence_fixtures.EvidenceTests.evaluation(shell, trial)
        path = self.root / f"flow-eval-{uuid4().hex}.json"
        evaluation["evaluation_path"] = str(path)
        path.write_text(json.dumps(evaluation), encoding="utf-8")
        return {**result, "status": "ok", **{k: evaluation[k] for k in
            ("evaluation_path", "scan_quality", "pages", "candidates", "printer_response", "measurement_docs", "scan_files")}}

    def report_result(self, *args):
        self.assertEqual(self.session()["status"], "completed")
        import pymupdf
        state = args[0]
        report = self.root / "report.pdf"
        with pymupdf.open() as document:
            page = document.new_page(width=595.276, height=841.89)
            page.insert_image(pymupdf.Rect(30, 70, 565, 700), filename=state["final_figure_path"])
            document.save(report)
        return {"outcome": "needs_print_approval", "pending_print": {"args": {
            "pdf_path": str(report), "printer": "Offline", "copies": 1}}, "dialogue": []}

    def session(self):
        return load_tuning_session(self.path)

    def start(self):
        self.build()
        return self.graph.invoke(self.state, self.config)

    def resume(self, action, **kwargs):
        return self.graph.invoke(Command(resume={"action": action, **kwargs}), self.config)

    def kind(self, result):
        return result["__interrupt__"][0].value["kind"]

    def printed(self):
        result = self.start()
        self.assertEqual(self.kind(result), "trial_print_approval")
        result = self.resume("approve")
        self.assertEqual(self.kind(result), "tuning_scan")
        return result

    def test_all_four_stages_three_rounds_and_report_gate(self):
        result = self.start()
        for number in range(1, 4):
            self.assertEqual(self.kind(result), "trial_print_approval", result.get("tuning_failure"))
            payload = result["__interrupt__"][0].value
            self.assertEqual(payload["round_id"], number)
            self.assertGreaterEqual(payload["page_count"], 2)
            self.assertEqual(len(self.submissions), number - 1)
            self.report.assert_not_called()
            result = self.resume("approve")
            self.assertEqual(self.kind(result), "tuning_scan")
            self.assertEqual(result["__interrupt__"][0].value["suggested_scan_directory"], str(self.root / "scans" / f"round_{number}"))
            result = self.resume("scan", scan_paths=[f"phone/round-{number}.pdf"])
        self.assertEqual(self.kind(result), "print_approval")
        self.assertTrue(result["report"]["figure_embedded"])
        self.assertEqual(self.stages, ["propose_gamma", "review_gamma", "review_gray_levels", "review_steps"])
        self.assertEqual(len(self.submissions), 3)
        self.assertEqual(self.session()["status"], "completed")
        self.assertTrue(all(m["steps"] == 0 for m in self.session()["final_settings"].values()))
        self.assertEqual(result["decision_count"], 99)
        for record in result["execution_results"]:
            if record["tool_name"] == "read_scan":
                self.assertNotIn("candidates", record["result"])
                self.assertNotIn("measurements", json.dumps(record["result"]))
                self.assertIn("candidate_columns", record["result"])
        self.assertNotIn("base64", json.dumps({k: v for k, v in result.items() if k != "__interrupt__"}))
        self.assertEqual(self.scans, [[str(self.root / "phone" / f"round-{n}.pdf")] for n in range(1, 4)])

    def test_invalid_approval_and_stop_never_print(self):
        result = self.start()
        self.assertEqual(self.kind(result), "trial_print_approval")
        result = self.resume("yes_probably")
        self.assertEqual(self.kind(result), "trial_print_approval")
        self.assertFalse(self.submissions)
        result = self.resume("stop")
        self.assertEqual(result["stop_reason"], "stopped_by_human")
        self.assertEqual(self.session()["status"], "stopped")
        self.report.assert_not_called()

    def test_changed_pdf_and_manifest_block_submission(self):
        for field in ("pdf_path", "manifest_path"):
            with self.subTest(field=field):
                self.path.write_bytes(self.initial)
                self.config["configurable"]["thread_id"] = uuid4().hex
                result = self.start()
                path = Path(self.trial[field])
                original = path.read_bytes()
                try:
                    path.write_bytes(original + b"changed")
                    result = self.resume("approve")
                    self.assertEqual(self.kind(result), "tuning_problem")
                    self.assertIn("fresh approval", result["tuning_failure"])
                    self.assertFalse(self.submissions)
                finally:
                    path.write_bytes(original)
        self.report.assert_not_called()

    def test_submission_node_replay_does_not_duplicate(self):
        self.printed()
        snapshot = self.graph.get_state(self.config).values
        approval = self.session()["rounds"]["1"]["approvals"][0]["metadata"]["approval_id"]
        replay = dict(snapshot, pending_trial_print={"approval_id": approval})
        result = self.nodes["tuning_submit"](replay)
        self.assertEqual(result["route"], "tuning_scan")
        self.assertEqual(len(self.submissions), 1)
        # A duplicate approval-shaped resume is now a scan answer, never a print.
        result = self.resume("approve")
        self.assertEqual(self.kind(result), "tuning_scan")
        self.assertEqual(len(self.submissions), 1)

    def test_unused_saved_approval_is_reconfirmed_and_reused(self):
        result = self.start()
        pending = result["pending_trial_print"]
        record_print_approval(self.path, approval_id=pending["approval_id"], trial_id=pending["trial_id"],
            pdf_sha256=pending["pdf_sha256"], printer="Offline", approved_by="graph human interrupt",
            print_settings={**pending["args"], "copies": 1, "paper": "A4", "color_mode": "grayscale",
                            "sides": "one-sided", "scale": "100%"})
        # Simulate losing the graph checkpoint after only recording approval.
        self.config["configurable"]["thread_id"] = uuid4().hex
        result = self.graph.invoke(self.state, self.config)
        self.assertEqual(self.kind(result), "trial_print_approval")
        self.assertEqual(result["pending_trial_print"]["approval_id"], pending["approval_id"])
        self.assertFalse(self.submissions)
        result = self.resume("approve")
        self.assertEqual(self.kind(result), "tuning_scan")
        self.assertEqual(len(self.submissions), 1)

    def test_main_model_is_default_specialist_and_report_gate_is_closed(self):
        self.llm.with_structured_output.side_effect = self.structured
        self.build(separate=False)
        gate = self.nodes["report_and_print"](self.state)
        self.assertEqual(gate["route"], "tuning_route")
        self.report.assert_not_called()
        result = self.graph.invoke(self.state, self.config)
        self.assertEqual(self.kind(result), "trial_print_approval")
        self.assertEqual(self.stages, ["propose_gamma"])
        self.specialist.with_structured_output.assert_not_called()

    def test_stale_trial_approval_cannot_submit(self):
        self.start()
        snapshot = dict(self.graph.get_state(self.config).values)
        snapshot["pending_trial_print"] = {**snapshot["pending_trial_print"], "trial_id": "old-trial"}
        result = self.nodes["tuning_submit"](snapshot)
        self.assertEqual(result["route"], "tuning_problem")
        self.assertFalse(self.submissions)

    def test_specialist_replay_after_committed_decision_does_not_call_model(self):
        self.start()
        state = self.graph.get_state(self.config).values
        calls = self.specialist.with_structured_output.call_count
        result = self.nodes["tuning_specialist"](state)
        self.assertEqual(result["route"], "tuning_route")
        self.assertEqual(self.specialist.with_structured_output.call_count, calls)

    def test_explicit_artifact_regeneration_requires_new_approval(self):
        self.start()
        path = Path(self.trial["pdf_path"])
        original = path.read_bytes()
        self.addCleanup(path.write_bytes, original)
        path.write_bytes(original + b"changed")
        result = self.resume("approve")
        self.assertEqual(self.kind(result), "tuning_problem")
        # Exercise the actual producer for the explicitly requested replacement.
        self.tools[0].func.side_effect = create_tuning_tools(self.root)[0].func
        result = self.resume("regenerate")
        self.assertEqual(self.kind(result), "trial_print_approval")
        self.assertNotEqual(result["pending_trial_print"]["trial_id"], self.trial["trial_id"])
        self.assertEqual(len(self.session()["rounds"]["1"]["trials"]), 2)
        self.assertFalse(self.submissions)

    def test_valid_artifacts_are_not_regenerated_for_a_model_error(self):
        self.printed()
        self.specialist.with_structured_output.side_effect = RuntimeError("Provider unavailable")
        result = self.resume("scan", scan_paths=["valid.pdf"])
        self.assertEqual(self.kind(result), "tuning_problem")
        self.assertEqual(self.session()["phase"], "awaiting_review")
        result = self.resume("regenerate")
        self.assertEqual(self.kind(result), "tuning_problem")
        self.assertIn("artifacts are valid", result["tuning_failure"])
        self.assertEqual(len(self.session()["rounds"]["1"]["trials"]), 1)
        self.assertEqual(len(self.submissions), 1)

    def test_printer_name_is_explicit_human_input(self):
        self.state.pop("tuning_printer")
        result = self.start()
        self.assertEqual(self.kind(result), "tuning_printer")
        self.assertFalse(self.printers[0].func.called)
        result = self.resume("printer", printer="Offline")
        self.assertEqual(self.kind(result), "trial_print_approval")
        self.assertEqual(result["pending_trial_print"]["args"]["printer"], "Offline")
        self.assertFalse(self.submissions)

    def test_crash_after_intent_requires_human_and_reprint_is_resolvable(self):
        self.next_submission = RuntimeError("connection lost after submission")
        result = self.start()
        result = self.resume("approve")
        self.assertEqual(self.kind(result), "tuning_print_confirmation")
        attempt = self.session()["rounds"]["1"]["print_attempts"][0]
        self.assertIsNone(attempt["result"])
        snapshot = dict(self.graph.get_state(self.config).values, pending_trial_print={"approval_id": attempt["approval_id"]})
        self.assertEqual(self.nodes["tuning_submit"](snapshot)["route"], "tuning_confirmation")
        self.assertEqual(len(self.submissions), 1)
        self.next_submission = None
        result = self.resume("not_printed")
        self.assertEqual(self.kind(result), "trial_print_approval")
        self.assertEqual(len(self.submissions), 1)
        result = self.resume("approve")
        self.assertEqual(self.kind(result), "tuning_scan")
        self.assertEqual(len(self.submissions), 2)
        self.assertEqual(self.session()["rounds"]["1"]["retries"]["reprints"], 1)

    def test_uncertain_job_requires_human_and_never_resubmits(self):
        self.job_states = ["uncertain"]
        self.start()
        result = self.resume("approve")
        self.assertEqual(self.kind(result), "tuning_print_confirmation")
        result = self.resume("maybe")
        self.assertEqual(self.kind(result), "tuning_print_confirmation")
        result = self.resume("printed")
        self.assertEqual(self.kind(result), "tuning_scan")
        self.assertEqual(len(self.submissions), 1)
        self.assertEqual(self.session()["rounds"]["1"]["print_attempts"][0]["human_resolution"]["outcome"], "printed")

    def test_untracked_submission_needs_confirmation(self):
        self.next_submission = {"ok": True, "job_id": "offline-untracked-1", "tracked": False}
        self.start()
        result = self.resume("approve")
        self.assertEqual(self.kind(result), "tuning_print_confirmation")
        self.assertFalse(self.printers[1].func.called)
        self.assertEqual(len(self.submissions), 1)
        self.resume("stop")
        self.report.assert_not_called()

    def test_reprint_fresh_approval_and_limit(self):
        self.printed()
        result = self.resume("reprint")
        self.assertEqual(self.kind(result), "trial_print_approval")
        self.assertEqual(len(self.submissions), 1)
        first = self.session()["rounds"]["1"]["approvals"][0]["metadata"]
        result = self.resume("approve")
        self.assertEqual(self.kind(result), "tuning_scan")
        last = self.session()["rounds"]["1"]["approvals"][-1]["metadata"]
        self.assertNotEqual(first["approval_id"], last["approval_id"])
        self.assertEqual(first["trial_id"], last["trial_id"])
        result = self.resume("reprint")
        self.assertEqual(result["stop_reason"], "tuning_blocked")
        self.assertEqual(len(self.submissions), 2)

    def test_needs_rescan_does_not_advance_or_consume_reprint(self):
        self.printed()
        self.scan_status = "needs_rescan"
        self.assessments = ["needs_rescan"]
        result = self.resume("scan", scan_paths=["blurred-1.png", "blurred-2.png"])
        self.assertEqual(self.kind(result), "tuning_scan")
        self.assertEqual(self.session()["active_round"], 1)
        self.assertEqual(self.session()["rounds"]["1"]["retries"]["reprints"], 0)
        self.assertIsNone(self.session()["rounds"]["1"]["winners"])
        self.scan_status = "ok"
        result = self.resume("scan", scan_paths=["replacement.pdf"])
        self.assertEqual(self.kind(result), "trial_print_approval")
        self.assertEqual(self.session()["active_round"], 2)

    def test_insufficient_contrast_blocks_without_report(self):
        self.printed()
        self.assessments = ["insufficient_contrast"]
        result = self.resume("scan", scan_paths=["contrast.pdf"])
        self.assertEqual(result["stop_reason"], "tuning_blocked")
        self.assertEqual(self.session()["status"], "blocked")
        self.assertIsNone(self.session()["final_settings"])
        self.report.assert_not_called()

    def test_operational_scan_error_never_reaches_specialist(self):
        self.printed()
        self.scan_status = "error"
        result = self.resume("scan", scan_paths=["bad.pdf"])
        self.assertEqual(self.kind(result), "tuning_problem")
        self.assertEqual(self.stages, ["propose_gamma"])
        self.assertEqual(self.session()["active_round"], 1)

    def test_polling_is_bounded_by_human_pause(self):
        self.job_states = ["waiting", "printing", "waiting", "done"]
        self.start()
        result = self.resume("approve")
        self.assertEqual(self.kind(result), "tuning_poll")
        self.assertEqual(self.printers[1].func.call_count, 3)
        result = self.resume("check")
        self.assertEqual(self.kind(result), "tuning_scan")
        self.assertEqual(len(self.submissions), 1)

    def test_failed_job_offers_controlled_reprint(self):
        self.job_states = ["failed"]
        self.start()
        result = self.resume("approve")
        self.assertEqual(self.kind(result), "tuning_problem")
        self.assertIn("reprint", result["__interrupt__"][0].value["choices"])
        result = self.resume("reprint")
        self.assertEqual(self.kind(result), "trial_print_approval")
        self.assertEqual(len(self.submissions), 1)

    def test_missing_dependencies_cannot_bypass_tuning(self):
        self.build(dependencies=False)
        result = self.graph.invoke(self.state, self.config)
        self.assertEqual(result["stop_reason"], "tuning_blocked")
        self.assertIn("make_trial_sheet", result["tuning_failure"])
        self.report.assert_not_called()
        self.assertFalse(self.submissions)

    def test_no_tool_failure_is_bounded_and_never_spends_budget(self):
        self.model_fail = lambda: AIMessage(content="No action")
        result = self.start()
        self.assertEqual(self.kind(result), "tuning_problem")
        result = self.resume("retry")
        self.assertEqual(self.kind(result), "tuning_problem")
        result = self.resume("retry")
        self.assertEqual(result["stop_reason"], "tuning_blocked")
        self.assertEqual(result["decision_count"], 99)
        self.assertFalse(self.submissions)

    def test_scan_input_requires_explicit_nonblank_paths(self):
        self.printed()
        for paths in ([], "guessed.png", [""]):
            result = self.resume("scan", scan_paths=paths)
            self.assertEqual(self.kind(result), "tuning_scan")
        self.assertFalse(self.scans)
        result = self.resume("stop")
        self.assertEqual(result["stop_reason"], "stopped_by_human")

    def test_frozen_selection_and_ordinary_experiment_routing_preserved(self):
        self.build()
        state = dict(self.state)
        state.pop("experiment_selection")
        state["decision_count"] = 0
        with patch.object(graph_module, "run_orchestrator", return_value={"decision": {"next_role": "executor"}, "dialogue": []}):
            result = self.nodes["orchestrator"](state)
        self.assertEqual(result["route"], "executor")
        frozen = self.nodes["orchestrator"](self.state)
        self.assertEqual(frozen["route"], "tuning_route")
        self.assertEqual(self.session()["accepted_selection"]["selections"], self.fixture.selection["selections"])


if __name__ == "__main__":
    unittest.main()
