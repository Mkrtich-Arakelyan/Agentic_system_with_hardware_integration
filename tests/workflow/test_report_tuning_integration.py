"""Completed real sessions, real PDF pixels, mocked model/tools; no hardware/API."""

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
import pymupdf
from PIL import Image

from app.agents.report_and_print import report_and_print
from app.workflow.state import State
from app.workflow import reporting
from app.workflow.tuning import load_tuning_session
from tests.workflow import test_tuning as fixtures

graph_module = importlib.import_module("app.workflow.graph")


class ReportIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture = fixtures.SessionTests()
        cls.fixture.setUp()
        cls.addClassCleanup(cls.fixture.doCleanups)
        cls.fixture.init()
        cls.fixture.propose()
        for number, stage in enumerate(("review_gamma", "review_gray_levels", "review_steps"), 1):
            _, trial = cls.fixture.trial(number)
            cls.fixture.scan(trial)
            cls.fixture.review(stage)
        cls.completed = cls.fixture.path.read_bytes()

    def setUp(self):
        self.root, self.path = self.fixture.root, self.fixture.path
        self.path.write_bytes(self.completed)
        for name in ("report.pdf", "report_print.json"):
            (self.root / name).unlink(missing_ok=True)
        self.model = Mock()
        self.model.bind_tools.side_effect = self.bind
        self.messages = []
        self.requests = []
        self.submitted = []
        self.submission = {"ok": True, "job_id": "Report-7", "tracked": True}
        self.reading = {"ok": True, "job_id": "Report-7", "state": "done", "confirmed": True}
        self.nodes = {}
        self.config = {"configurable": {"thread_id": uuid4().hex}, "recursion_limit": 100}
        self.state = {"dialogue": [], "execution_results": [
            {"tool_name": "run_focusing_experiment", "result": r} for r in self.fixture.results],
            "experiment_selection": deepcopy(self.fixture.selection), "tuning_session_path": str(self.path)}
        self.code = StructuredTool.from_function(self.write_pdf, name="build_final_report", description="Fake report boundary")
        self.printers = [StructuredTool.from_function(self.submit, name="submit_print", description="Fake print"),
                         StructuredTool.from_function(self.job, name="job_status", description="Fake reading")]
        self.information = StructuredTool.from_function(lambda missing, human_request: {},
            name="request_information", description="Information handoff")
        self.build()

    def build(self):
        builder = StateGraph(State)
        add = builder.add_node
        def capture(name, node, *args, **kwargs):
            self.nodes[name] = node
            return add(name, node, *args, **kwargs)
        with patch.object(graph_module, "StateGraph", return_value=builder), patch.object(builder, "add_node", side_effect=capture), \
                patch("app.workflow.report_document.create_report_tool", return_value=self.code):
            self.graph = graph_module.build_graph(self.model, Mock(), self.code, self.printers,
                self.information, self.root, 5, InMemorySaver())

    def bind(self, tools, **options):
        def invoke(messages):
            if options.get("tool_choice") == "none":
                return AIMessage(content="Recorded offline result")
            self.messages.append(messages)
            if self.requests:
                name, args = self.requests.pop(0)
            elif not (self.root / "report.pdf").exists():
                name, args = "build_final_report", {}
            elif reporting.print_state(reporting.load_record(self.root)) == "pending":
                name, args = "job_status", {"job_id": "Report-7"}
            else:
                name, args = "submit_print", {"pdf_path": str(self.root / "report.pdf"), "printer": "Offline", "copies": 1}
            return AIMessage(content="", tool_calls=[{"id": "call", "name": name, "args": args}])
        return SimpleNamespace(invoke=invoke)

    def write_pdf(self, findings: dict | None = None) -> dict:
        figure = reporting.load_record(self.root)["figure"]
        self.make_pdf(figure["figure_path"])
        return {"exit_code": 0, "stdout": "Created report", "stderr": ""}

    def make_pdf(self, image, *, pages=2, tiny=False):
        with pymupdf.open() as document:
            page = document.new_page(width=595.276, height=841.89)
            page.insert_text((30, 35), "Scientific comparison - tuned presentation")
            for _ in range(pages - 1):
                document.new_page(width=595.276, height=841.89)
            document[min(1, pages - 1)].insert_image(pymupdf.Rect(30, 60, 50, 80) if tiny else pymupdf.Rect(30, 70, 565, 700), filename=str(image))
            document.save(self.root / "report.pdf")

    def submit(self, pdf_path: str, printer: str, copies: int = 1) -> dict:
        attempt = reporting.load_record(self.root)["attempts"][-1]
        self.assertIsNone(attempt["result"])
        self.submitted.append({"pdf_path": pdf_path, "printer": printer, "copies": copies})
        if isinstance(self.submission, Exception):
            raise self.submission
        return self.submission

    def job(self, job_id: str) -> dict:
        return self.reading

    def start(self):
        return self.graph.invoke(self.state, self.config)

    def resume(self, reply):
        return self.graph.invoke(Command(resume=reply), self.config)

    def kind(self, result):
        return result["__interrupt__"][0].value["kind"]

    def approve(self):
        count = len(self.submitted)
        result = self.start()
        self.assertEqual(self.kind(result), "print_approval")
        self.assertEqual(len(self.submitted), count)
        return self.resume("yes")

    def test_completed_session_embeds_final_figure_and_prints_only_approved_bytes(self):
        original = self.path.read_bytes()
        with patch.object(reporting, "render_final_heatmaps", wraps=reporting.render_final_heatmaps) as render:
            result = self.approve()
            self.assertEqual(render.call_count, 1)
            self.assertEqual(render.call_args.args[0], str(self.path))
        self.assertEqual(result["stop_reason"], "finished")
        self.assertEqual(result["report"]["pages"], 2)
        self.assertTrue(result["report"]["figure_embedded"])
        self.assertEqual(self.path.read_bytes(), original)
        meta = result["final_figure_metadata"]
        session = load_tuning_session(self.path)
        self.assertEqual([r["experiment_id"] for r in meta["experiments"]], list(fixtures.IDS))
        self.assertEqual([r["mapping"] for r in meta["experiments"]], [session["final_settings"][e] for e in fixtures.IDS])
        self.assertEqual([r["limits"] for r in meta["experiments"]], [e["limits"] for e in session["selected_experiments"]])
        context = self.messages[0][1].content
        self.assertIn(result["final_figure_path"], context.replace("\\\\", "\\"))
        self.assertIn("Four complementary experiments", context)
        self.assertIn("final_enhancement", context)
        self.assertEqual(len(self.submitted), 1)
        self.assertEqual(reporting.load_record(self.root)["attempts"][-1]["resolution"], {})
        count = len(self.messages)
        self.build()
        self.config["configurable"]["thread_id"] = uuid4().hex
        resumed = self.start()
        self.assertEqual(resumed["stop_reason"], "finished")
        self.assertEqual(len(self.submitted), 1)
        self.assertEqual(len(self.messages), count)

    def test_incomplete_stopped_and_blocked_sessions_never_report(self):
        for status in ("active", "stopped", "blocked"):
            session = json.loads(self.completed)
            session.update(status=status, phase="awaiting_scan", final_settings=None)
            self.path.write_text(json.dumps(session))
            result = self.nodes["report_and_print"](self.state)
            self.assertNotIn("final_figure_path", result)
        self.assertFalse(self.messages)
        self.assertFalse(self.submitted)

    def test_legacy_print_history_without_a_journal_cannot_submit_again(self):
        result = self.nodes["report_and_print"]({**self.state, "active_job_id": "Legacy-4"})
        self.assertEqual(result["stop_reason"], "report_blocked")
        self.assertIn("reconcile", result["dialogue"][0]["text"])
        self.assertFalse(self.messages)
        self.assertFalse(self.submitted)

    def test_repeated_turns_reuse_figure_and_damaged_artifacts_regenerate(self):
        result = self.start()
        figure = result["final_figure_metadata"]
        with patch.object(reporting, "render_final_heatmaps", wraps=reporting.render_final_heatmaps) as render:
            for _ in range(2):
                current, changed = reporting.prepare_figure(self.path, load_tuning_session(self.path), self.root)
                self.assertFalse(changed)
                self.assertEqual(current, figure)
            render.assert_not_called()
            Path(figure["figure_path"]).unlink()
            fresh, changed = reporting.prepare_figure(self.path, load_tuning_session(self.path), self.root)
            self.assertTrue(changed)
            Path(fresh["figure_path"]).write_bytes(b"changed PNG")
            reporting.prepare_figure(self.path, load_tuning_session(self.path), self.root)
            self.assertEqual(render.call_count, 2)
        self.assertIsNone(reporting.load_record(self.root)["pending"])

    def test_stale_fingerprint_regenerates_without_changing_science(self):
        self.start()
        record = reporting.load_record(self.root)
        record["figure"]["inputs_sha256"] = "stale"
        reporting.save_record(self.root, record)
        before = self.path.read_bytes()
        figure, changed = reporting.prepare_figure(self.path, load_tuning_session(self.path), self.root)
        self.assertTrue(changed)
        self.assertNotEqual(figure["inputs_sha256"], "stale")
        self.assertEqual(self.path.read_bytes(), before)

    def test_old_two_column_cache_regenerates_and_cannot_use_previous_approval(self):
        self.start()
        snapshot = dict(self.graph.get_state(self.config).values)
        from app.print_tuning.final_figures import render_final_heatmaps
        legacy = render_final_heatmaps(self.path, self.root / "legacy_figure")
        record = reporting.load_record(self.root)
        legacy.update(inputs_sha256=record["figure"]["inputs_sha256"],
                      session_id=record["figure"]["session_id"],
                      comparison_version="scientific-pairs/1", columns=["Initial", "Final"])
        record["figure"] = legacy
        record["figure_history"].append(deepcopy(legacy))
        reporting.save_record(self.root, record)
        self.make_pdf(legacy["figure_path"])
        old_pdf, scientific_record = (self.root / "report.pdf").read_bytes(), self.path.read_bytes()
        figure, changed = reporting.prepare_figure(self.path, load_tuning_session(self.path), self.root)
        self.assertTrue(changed)
        self.assertNotEqual(figure["figure_path"], legacy["figure_path"])
        self.assertEqual(figure["columns"], ["Initial", "Final", "Initial", "Final"])
        self.assertIn(legacy, reporting.load_record(self.root)["figure_history"])
        self.assertIsNone(reporting.load_record(self.root)["pending"])
        self.assertEqual((self.root / "report.pdf").read_bytes(), old_pdf)
        self.assertEqual(self.path.read_bytes(), scientific_record)
        self.assertTrue(reporting.inspect_tuned_report(self.root / "report.pdf", legacy).get("error"))
        update = self.nodes["report_submit"](snapshot)
        self.assertEqual(update["route"], "report_and_print")
        self.assertFalse(self.submitted)

    def test_pdf_must_contain_actual_unchanged_figure_at_useful_size(self):
        result = self.start()
        figure = result["final_figure_metadata"]
        other = self.root / "untuned.png"
        with Image.open(figure["figure_path"]) as source:
            size = source.size
        # Same dimensions/placement, different pixels: a resource-name or size
        # check alone would incorrectly accept this untuned replacement.
        Image.new("L", size, 128).save(other)
        for image, pages, tiny in ((other, 2, False), (figure["figure_path"], 1, False), (figure["figure_path"], 2, True)):
            self.make_pdf(image, pages=pages, tiny=tiny)
            report = reporting.inspect_tuned_report(self.root / "report.pdf", figure)
            self.assertTrue(report.get("error"))
            with self.assertRaises(ValueError):
                reporting.require_report(report)

    def test_uncertain_untracked_and_exception_require_exact_human_interrupt(self):
        cases = [({"ok": True, "job_id": "Report-7", "tracked": False}, None),
                 ({"ok": True, "job_id": "Report-7", "tracked": True},
                  {"ok": True, "job_id": "Report-7", "state": "uncertain", "confirmed": False}),
                 ({"ok": True, "job_id": "Report-7", "tracked": True},
                  {"ok": True, "job_id": "Report-7", "state": "done", "confirmed": False}),
                 ({"ok": False, "error": "Connector exception: response lost"}, None),
                 ({"ok": True, "job_id": "Report-7", "tracked": True},
                  {"ok": False, "job_id": "Report-7", "error": "Status unavailable"}),
                 ({"ok": True, "job_id": "Report-7", "tracked": True},
                  {"ok": False, "error": "Status response has no identity"}),
                 (RuntimeError("lost reply"), None)]
        for submission, reading in cases:
            with self.subTest(submission=submission):
                (self.root / "report_print.json").unlink(missing_ok=True)
                (self.root / "report.pdf").unlink(missing_ok=True)
                self.config["configurable"]["thread_id"] = uuid4().hex
                self.submission = submission
                if reading:
                    self.reading = reading
                result = self.approve()
                payload = result["__interrupt__"][0].value
                self.assertEqual(payload["kind"], "report_print_confirmation")
                self.assertEqual(payload["choices"], ["printed", "not_printed", "stop"])
                self.assertEqual(set(payload), {"kind", "job_id", "reading", "question", "choices"})
                count = len(self.submitted)
                self.assertEqual(self.kind(self.resume({})), "report_print_confirmation")
                self.assertEqual(self.kind(self.resume({"action": "yes"})), "report_print_confirmation")
                result = self.resume({"action": "printed"})
                self.assertEqual(result["stop_reason"], "finished")
                self.assertEqual(len(self.submitted), count)
                self.assertEqual(reporting.load_record(self.root)["attempts"][-1]["resolution"]["source"], "human")

    def test_not_printed_requires_new_approval_new_attempt_and_stop_is_not_printed(self):
        self.submission = {"ok": True, "job_id": "Report-7", "tracked": False}
        result = self.approve()
        old = reporting.load_record(self.root)["attempts"][-1]["attempt_id"]
        result = self.resume({"action": "not_printed"})
        self.assertEqual(self.kind(result), "print_approval")
        self.assertEqual(len(self.submitted), 1)
        self.assertNotEqual(result["pending_print"]["attempt_id"], old)
        result = self.resume("yes")
        self.assertEqual(len(self.submitted), 2)
        self.assertEqual(self.kind(result), "report_print_confirmation")
        result = self.resume({"action": "stop"})
        self.assertEqual(result["stop_reason"], "stopped_by_human")
        self.assertNotEqual(result["print_outcome"], "printed")

    def test_changed_pdf_printer_or_args_never_uses_old_approval(self):
        result = self.start()
        state = dict(self.graph.get_state(self.config).values)
        for field, value in (("printer", "Other"), ("copies", 2)):
            pending = deepcopy(state["pending_print"])
            pending["args"][field] = value
            update = self.nodes["report_submit"]({**state, "pending_print": pending})
            self.assertEqual(update["route"], "report_and_print")
        self.make_pdf(state["final_figure_path"])
        with (self.root / "report.pdf").open("ab") as file:
            file.write(b"changed")
        update = self.nodes["report_submit"](state)
        self.assertEqual(update["route"], "report_and_print")
        self.assertFalse(self.submitted)

    def test_submission_replay_and_process_restart_do_not_duplicate(self):
        result = self.start()
        snapshot = dict(self.graph.get_state(self.config).values)
        self.submission = RuntimeError("submission reply lost")
        self.resume("yes")
        replay = self.nodes["report_submit"](snapshot)
        self.assertEqual(replay["route"], "report_print_confirmation")
        self.assertEqual(len(self.submitted), 1)
        # Fresh checkpointer/state: persisted intent still forces confirmation.
        self.build()
        self.config["configurable"]["thread_id"] = uuid4().hex
        result = self.start()
        self.assertEqual(self.kind(result), "report_print_confirmation")
        self.assertEqual(len(self.submitted), 1)

    def test_wrong_job_reading_cannot_finish(self):
        self.start()
        self.reading = {"ok": True, "job_id": "Wrong-9", "state": "done", "confirmed": True}
        result = self.resume("yes")
        self.assertEqual(self.kind(result), "report_workflow_problem")
        self.assertEqual(reporting.print_state(reporting.load_record(self.root)), "pending")
        self.assertNotEqual(result.get("print_outcome"), "printed")

    def ready_state(self):
        self.start()
        state = dict(self.graph.get_state(self.config).values)
        state["execution_results"] = [*state["execution_results"], {
            "tool_name": "printer_status", "tool_args": {"printer": "Offline"},
            "result": {"ok": True, "printer": "Offline", "state": "idle"}}]
        return state

    def test_valid_report_idle_printer_requests_approval_without_conversational_yes(self):
        state = self.ready_state()
        original = (self.root / "report.pdf").read_bytes()
        self.model.bind_tools.reset_mock()
        self.build()
        self.config["configurable"]["thread_id"] = uuid4().hex
        result = self.graph.invoke({**state, "dialogue": [{"role": "human", "text": "retry"}]}, self.config)
        self.assertEqual(self.kind(result), "print_approval")
        call = self.model.bind_tools.call_args
        self.assertEqual([t.name for t in call.args[0]], ["submit_print"])
        self.assertEqual(call.kwargs, {"tool_choice": "submit_print", "parallel_tool_calls": False})
        self.assertFalse(self.submitted)
        self.assertEqual((self.root / "report.pdf").read_bytes(), original)
        pending = result["pending_print"]
        payload = result["__interrupt__"][0].value
        self.assertEqual(payload["pdf_sha256"], pending["report_sha256"])
        self.assertEqual(payload["printer"], pending["args"]["printer"])
        self.assertEqual(self.kind(self.resume("retry")), "print_approval")
        self.assertFalse(self.submitted)
        self.assertFalse(reporting.load_record(self.root)["attempts"])

    def test_noncompliant_status_loop_has_bounded_workflow_recovery(self):
        state = self.ready_state()
        calls = []
        def status(printer: str) -> dict:
            calls.append(printer)
            return {"ok": True, "printer": printer, "state": "idle"}
        self.printers.append(StructuredTool.from_function(status, name="printer_status", description="Offline status"))
        self.requests = [("printer_status", {"printer": "Offline"})] * 12
        self.build()
        self.config["configurable"]["thread_id"] = uuid4().hex
        result = self.graph.invoke(state, self.config)
        self.assertEqual(self.kind(result), "report_workflow_problem")
        self.assertEqual(result["report_request_passes"], 2)
        self.assertFalse(calls)  # Reject disallowed calls even if the provider ignores tool_choice.
        self.assertFalse(self.submitted)
        self.assertIn("submit_print", result["__interrupt__"][0].value["message"])
        self.requests = []
        self.assertEqual(self.kind(self.resume({"action": "retry"})), "print_approval")
        self.assertFalse(self.submitted)

    def test_repeated_unready_status_reads_pause_as_workflow_failure(self):
        state = self.ready_state()
        state["execution_results"][-1]["result"]["state"] = "printing"
        def busy(printer: str) -> dict:
            return {"ok": True, "printer": printer, "state": "printing"}
        self.printers.append(StructuredTool.from_function(busy, name="printer_status", description="Offline busy"))
        self.requests = [("printer_status", {"printer": "Offline"})] * 10
        self.build()
        self.config["configurable"]["thread_id"] = uuid4().hex
        result = self.graph.invoke(state, self.config)
        self.assertEqual(self.kind(result), "report_workflow_problem")
        self.assertLess(result["hardware_passes"], 12)
        self.assertFalse(self.submitted)

    def test_report_pass_exhaustion_is_not_a_printer_fault(self):
        state = self.ready_state()
        update = self.nodes["report_and_print"]({**state, "hardware_passes": 12})
        self.assertEqual(update["route"], "report_workflow_problem")
        self.assertNotIn("last_printer_reading", update)
        self.assertIn("No hardware fault", update["report_workflow_failure"])
        self.assertFalse(self.submitted)

    def test_busy_to_idle_after_repeated_readings_advances_to_approval(self):
        state = self.ready_state()
        state["execution_results"][-1]["result"]["state"] = "printing"
        state["execution_results"].append(deepcopy(state["execution_results"][-1]))
        def idle(printer: str) -> dict:
            return {"ok": True, "printer": printer, "state": "idle"}
        self.printers.append(StructuredTool.from_function(idle, name="printer_status", description="Offline idle"))
        self.requests = [("printer_status", {"printer": "Offline"})]
        self.build()
        self.config["configurable"]["thread_id"] = uuid4().hex
        result = self.graph.invoke(state, self.config)
        self.assertEqual(self.kind(result), "print_approval")
        self.assertFalse(self.submitted)

    def test_report_change_can_revise_before_request_phase(self):
        state = self.ready_state()
        self.requests = [("request_information", {"missing": "scientific review", "human_request": "change claim"})]
        update = report_and_print({**state, "report_revision_pending": True}, self.model,
                                  [self.code, *self.printers, self.information], self.root)
        self.assertEqual(update["outcome"], "no_tool")
        self.assertEqual(self.model.bind_tools.call_args.kwargs["tool_choice"], "build_final_report")
        self.assertFalse(self.submitted)

    def test_revision_reaches_real_builder_and_requires_fresh_approval(self):
        self.start()
        old = reporting.sha(self.root / "report.pdf")
        from app.workflow.report_document import create_report_tool
        self.code = create_report_tool(self.root)
        self.build()
        self.config["configurable"]["thread_id"] = uuid4().hex
        self.start()
        instructions = "  Add a sentence at the end asking for the reader opinion  "
        self.requests = [("build_final_report", {"findings": {"closing_note": "What is your opinion of these results?"}})]
        result = self.resume(instructions)
        self.assertEqual(self.kind(result), "print_approval")
        self.assertFalse(self.submitted)
        self.assertNotEqual(reporting.sha(self.root / "report.pdf"), old)
        journal = reporting.load_record(self.root)
        self.assertIsNone(journal.get("revision"))
        self.assertEqual(journal["revisions"][-1]["instructions"], instructions)
        self.assertEqual(journal["attempts"], [])
        self.assertTrue((self.root / "report_versions" / (old + ".pdf")).exists())
        with pymupdf.open(self.root / "report.pdf") as pdf:
            self.assertIn("What is your opinion", " ".join(p.get_text() for p in pdf))
        self.assertEqual(result["pending_print"]["report_sha256"], reporting.sha(self.root / "report.pdf"))

    def test_pending_revision_blocks_old_pdf_submission_and_resume(self):
        self.start()
        snapshot = dict(self.graph.get_state(self.config).values)
        revision = reporting.request_revision(self.root, "Make the title clearer")
        self.assertEqual(self.nodes["report_submit"](snapshot)["route"], "report_and_print")
        self.assertFalse(self.submitted)
        self.requests = [("submit_print", snapshot["pending_print"]["args"])]
        update = self.nodes["report_and_print"](snapshot)
        self.assertTrue(update["report_revision_pending"])
        self.assertEqual(update["report_revision"], revision)
        self.assertIsNone(reporting.load_record(self.root)["pending"])

    def test_failed_revision_stays_pending_and_pauses_for_correction(self):
        self.start()
        from app.workflow.report_document import create_report_tool
        self.code = create_report_tool(self.root)
        self.build()
        self.config["configurable"]["thread_id"] = uuid4().hex
        self.start()
        self.requests = [("build_final_report", {})]
        result = self.resume("Make it clearer")
        self.assertEqual(self.kind(result), "report_workflow_problem")
        self.assertTrue(result["report_revision_pending"])
        self.assertIn("No content change", result["report_workflow_failure"])
        self.assertEqual(reporting.load_record(self.root)["revision"]["instructions"], "Make it clearer")
        self.assertFalse(self.submitted)

    def test_revision_waits_for_uncertain_submission_resolution(self):
        self.submission = RuntimeError("lost outcome")
        self.approve()
        before = deepcopy(reporting.load_record(self.root)["attempts"])
        revision = reporting.request_revision(self.root, "Clarify the title")
        snapshot = dict(self.graph.get_state(self.config).values)
        self.assertEqual(self.nodes["report_and_print"](snapshot)["route"], "report_print_confirmation")
        from app.workflow.report_document import create_report_tool
        result = create_report_tool(self.root).invoke({"revision_id": revision["id"],
            "revision_instructions": revision["instructions"], "findings": {"title": "Clarified comparison"}})
        self.assertFalse(result["ok"])
        self.assertEqual(reporting.load_record(self.root)["attempts"], before)
        self.assertEqual(len(self.submitted), 1)
        reporting.resolve_attempt(self.root, "printed")
        # Resolving the old copy must not discard the new pending revision.
        self.assertEqual(self.nodes["finish_gate"](snapshot)["route"], "report_and_print")
        self.assertEqual(reporting.load_record(self.root)["revision"], revision)

    def test_revision_of_printed_report_keeps_old_attempt_and_requires_new_approval(self):
        self.approve()
        before = deepcopy(reporting.load_record(self.root)["attempts"])
        from app.workflow.report_document import create_report_tool
        self.code = create_report_tool(self.root)
        self.build()
        reporting.request_revision(self.root, "Add a reader question")
        self.requests = [("build_final_report", {"findings": {"closing_note": "What should the next comparison test?"}})]
        self.config["configurable"]["thread_id"] = uuid4().hex
        result = self.graph.invoke(self.state, self.config)
        self.assertEqual(self.kind(result), "print_approval")
        self.assertEqual(reporting.load_record(self.root)["attempts"], before)
        self.assertEqual(len(self.submitted), 1)

    def test_executor_never_executes_submit_even_when_state_claims_approval(self):
        result = self.start()
        state = {**dict(self.graph.get_state(self.config).values), "approval": {"approved": True}}
        update = report_and_print(state, self.model, [self.code, *self.printers], self.root)
        self.assertEqual(update["outcome"], "needs_print_approval")
        self.assertFalse(self.submitted)

    def test_changed_figure_invalidates_pending_report_approval(self):
        self.start()
        snapshot = dict(self.graph.get_state(self.config).values)
        Path(snapshot["final_figure_path"]).write_bytes(b"modified")
        update = self.nodes["report_submit"](snapshot)
        self.assertEqual(update["route"], "report_and_print")
        self.assertFalse(self.submitted)
        self.assertIsNone(reporting.load_record(self.root)["pending"])

    def test_original_arrays_cannot_be_modified_to_regenerate_a_figure(self):
        self.start()
        source = Path(self.fixture.results[0]["paths"]["initial_intensity"])
        original = source.read_bytes()
        try:
            source.write_bytes(original + b"modified")
            with self.assertRaisesRegex(ValueError, "Original scientific array changed"):
                reporting.prepare_figure(self.path, load_tuning_session(self.path), self.root)
        finally:
            source.write_bytes(original)

    def test_finish_cannot_trust_state_claims_without_a_recorded_print(self):
        self.start()
        snapshot = dict(self.graph.get_state(self.config).values)
        update = self.nodes["finish_gate"]({**snapshot, "print_outcome": "printed"})
        self.assertEqual(update["stop_reason"], "report_blocked")

    def test_failed_print_allows_authorized_skip(self):
        self.reading = {"ok": True, "job_id": "Report-7", "state": "failed", "confirmed": True}
        result = self.approve()
        self.assertEqual(self.kind(result), "printer_problem")
        result = self.resume("skip")
        self.assertEqual(result["stop_reason"], "finished")
        self.assertEqual(result["print_outcome"], "skipped")

    def test_failed_print_retry_still_needs_fresh_approval(self):
        self.reading = {"ok": True, "job_id": "Report-7", "state": "failed", "confirmed": True}
        self.approve()
        result = self.resume("retry")
        self.assertEqual(self.kind(result), "print_approval")
        self.assertEqual(len(self.submitted), 1)

    def test_failed_intent_write_never_submits_and_failed_outcome_write_is_uncertain(self):
        self.start()
        snapshot = dict(self.graph.get_state(self.config).values)
        with patch.object(reporting.os, "replace", side_effect=OSError("disk full")):
            result = self.nodes["report_submit"](snapshot)
        self.assertEqual(result["route"], "report_and_print")
        self.assertFalse(self.submitted)
        self.assertFalse(reporting.load_record(self.root)["attempts"])
        save = reporting.save_record
        def fail_outcome(folder, record):
            if record["attempts"] and record["attempts"][-1]["result"] is not None:
                raise OSError("outcome could not be saved")
            return save(folder, record)
        with patch.object(reporting, "save_record", side_effect=fail_outcome):
            result = self.nodes["report_submit"](snapshot)
        self.assertEqual(result["route"], "report_print_confirmation")
        self.assertEqual(len(self.submitted), 1)
        self.assertIsNone(reporting.load_record(self.root)["attempts"][-1]["result"])


if __name__ == "__main__":
    unittest.main()
