import json
import math
from pathlib import Path
from uuid import uuid4

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.tools import BaseTool
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from app.workflow.state import State
import app.workflow.reporting as report_records
from app.workflow.tuning import (
    TuningSessionError, initialize_tuning_session, load_tuning_session,
    accept_specialist_response, record_print_approval, begin_print_attempt,
    record_print_result, resolve_print_outcome, request_trial_reprint, record_stop,
    _bound_trial, _trial_manifest,
)
from app.workflow.tuning_evidence import prepare_tuning_inputs
from app.agents.photo_quality_specialist import photo_quality_specialist

from app.agents.common import DEFAULT_INITIAL_DECISIONS, MIN_EXPERIMENTS
from app.agents.orchestrator import RoutingDecision, orchestrator as run_orchestrator, summarize_progress
from app.agents.scientist import ExperimentSelection, scientist as run_scientist
from app.agents.executor import executor as run_executor
from app.agents.general import general as run_general
from app.agents.report_and_print import (
    MAX_HARDWARE_PASSES, _parse_tool_output,
    report_and_print as run_report_and_print,
)


# Keys of a report_and_print result that belong in the graph state.
REPORT_STATE_KEYS = ("dialogue", "execution_results", "report", "active_job_id", "pending_print")
# The printer tools the report-and-print turn may call.
REPORT_PRINTER_TOOLS = ("list_printers", "printer_status", "submit_print", "job_status")
PROBLEM_JOB_STATES = {"held", "error", "cancelled", "canceled", "failed", "aborted"}
PROBLEM_PRINTER_STATES = {"offline", "stopped", "error", "out_of_paper", "paper_jam"}



def _successful_focusing_results(state: dict) -> list[dict]:
    """Check recorded tool payloads, not scores mentioned in dialogue text."""
    successful = []
    for record in state.get("execution_results", []):
        if (
            not isinstance(record, dict)
            or record.get("tool_name") != "run_focusing_experiment"
            or record.get("validation_error")
        ):
            continue
        result = record.get("result")
        if not isinstance(result, dict) or result.get("ok") is not True:
            continue
        if not isinstance(result.get("experiment_id"), str) or not result["experiment_id"]:
            continue
        if not isinstance(result.get("settings"), dict) or not result["settings"]:
            continue
        scores = (result.get("initial_enhancement"), result.get("final_enhancement"))
        try:
            if not all(
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(value)
                for value in scores
            ):
                continue
        except OverflowError:
            continue
        paths = result.get("paths")
        if isinstance(paths, dict) and all(
            isinstance(paths.get(name), str) and paths[name]
            for name in ("initial_camera", "final_camera", "results_json")
        ):
            successful.append(result)
    return successful


def _count_successful_focusing_results(state: dict) -> int:
    """Count valid focusing-tool evidence using the shared validation checks."""
    return len({result["experiment_id"] for result in _successful_focusing_results(state)})


def _selection_error(selection: dict, state: dict) -> str:
    """Validate IDs against recorded evidence without making scientific choices."""
    items = selection.get("selections", [])
    if not isinstance(items, list) or len(items) != MIN_EXPERIMENTS:
        return f"Select exactly {MIN_EXPERIMENTS} experiments."
    try:
        parsed = [ExperimentSelection.model_validate(item) for item in items]
    except (TypeError, ValueError):
        return "Each selection needs an ID and all four scientific reasons."
    ids = [item.experiment_id for item in parsed]
    if len(set(ids)) != MIN_EXPERIMENTS:
        return "Selected experiment IDs must be unique."
    available = {item["experiment_id"] for item in _successful_focusing_results(state)}
    if not set(ids) <= available:
        return "Selected IDs must refer to validated successful focusing results."
    if not isinstance(selection.get("group_reason"), str) or not selection["group_reason"].strip():
        return "Explain how the selected experiments jointly answer the scientific question."
    return ""


def _load_scientist_evidence(state: dict, run_dir: Path) -> dict:
    """Read recorded experiment JSON as application context, never an agent tool."""
    evidence = {"experiments": [], "errors": []}
    results = {item["experiment_id"]: item for item in _successful_focusing_results(state)}
    root = run_dir.resolve()
    for experiment_id, result in results.items():
        try:
            path = Path(result["paths"]["results_json"])
            if not path.is_absolute():
                path = root / path
            path = path.resolve()
            if not path.is_relative_to(root):
                raise ValueError("results_json must be inside this run directory.")
            record = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(record, dict):
                raise ValueError("results_json must contain an object.")
            # Also reject non-finite numbers anywhere in the supplied JSON.
            json.dumps(record, allow_nan=False)
            for name in ("experiment_id", "settings", "initial_enhancement", "final_enhancement"):
                expected = json.loads(json.dumps(result[name], allow_nan=False))
                if record.get(name) != expected:
                    raise ValueError(f"results_json {name} does not match recorded tool evidence.")
            evidence["experiments"].append({
                "experiment_id": experiment_id, "results_json": str(path), "record": record,
            })
        except (OSError, ValueError, TypeError, RecursionError) as error:
            evidence["errors"].append({"experiment_id": experiment_id, "error": str(error)})
    return evidence


def _has_problem_field(result: dict) -> bool:
    """True if a problem/problems/alerts field reports anything other than none."""
    for key in ("problem", "problems", "alerts"):
        value = result.get(key)
        items = value if isinstance(value, list) else [value]
        if any(item and str(item).strip().lower() != "none" for item in items):
            return True
    return False


def _reading_route(tool_name: str, result: dict) -> str:
    """Choose the next node from the last printer or report tool result."""
    if not isinstance(result, dict):
        return "report_and_print"
    state = str(result.get("state", "")).lower()
    if tool_name == "job_status":
        if state == "uncertain" or result.get("confirmed") is False:
            return "report_print_confirmation"
        if state == "done":
            return "finish_gate" if result.get("confirmed") is True else "report_print_confirmation"
        if state in PROBLEM_JOB_STATES or _has_problem_field(result):
            return "printer_question"
    elif tool_name == "printer_status":
        if state in PROBLEM_PRINTER_STATES or _has_problem_field(result):
            return "printer_question"
    elif tool_name == "submit_print":
        if result.get("tracked") is False:
            return "report_print_confirmation"
        if result.get("ok") is False and "offline" in str(result.get("error", "")).lower():
            return "printer_question"
    return "report_and_print"


def _route_update(tool_name: str, result: dict) -> dict:
    """Turn a reading into route, print_outcome and last_printer_reading updates."""
    route = _reading_route(tool_name, result)
    update = {"route": route}
    if route == "finish_gate":
        update["print_outcome"] = "printed"
    elif route == "printer_question":
        update["last_printer_reading"] = {"tool_name": tool_name, "result": result}
    return update


def _first_word(reply) -> str:
    text = str(reply).strip().lower()
    return text.split()[0].strip(".,!:;") if text else ""


def build_graph(
    llm: BaseChatModel,
    focusing_tool: BaseTool,
    coding_tool: BaseTool,
    printer_tools: list[BaseTool],
    info_tool: BaseTool,
    run_dir: str | Path,
    extra_decisions: int,
    checkpointer,
    *,
    tuning_tools: list[BaseTool] | None = None,
    specialist_llm: BaseChatModel | None = None,
):
    """Build the experiment, tuning, then report workflow.

    tuning_tools supplies make_trial_sheet/read_scan; printer readings and
    submit_trial_print may be supplied there or in printer_tools. No tools or
    models are constructed. specialist_llm defaults to llm. The terminal runner
    is not adapted here. Resume interrupts with these explicit dictionaries:

    tuning_printer: {action: printer, printer: <exact name>} or {action: stop}
    trial_print_approval: {action: approve} or {action: stop}
    tuning_scan: {action: scan, scan_paths: [<path>, ...]}, reprint, or stop
    tuning_print_confirmation: {action: printed}, not_printed, or stop
    tuning_poll: {action: check} or stop (at most three readings per pause)
    tuning_problem: {action: retry}, regenerate (artifact errors), reprint, stop
    report_print_confirmation: {action: printed}, {action: not_printed}, or
    {action: stop}. Report submission intent/outcomes persist in report_print.json.

    Bare words above abbreviate {action: <word>}. Invalid answers re-interrupt.
    Session operations must be serialized. InMemorySaver is not durable storage.
    """
    # The executor imports session helpers through app.workflow, whose public
    # initializer exports build_graph. Import here to keep both entry points
    # independently importable without a partially initialized module cycle.
    from app.agents.tuning_executor import TUNING_TOOLS, TuningTask, tuning_executor, _pending_trial

    run_dir = Path(run_dir).resolve()
    printer_map = {item.name: item for item in printer_tools}
    supplied = [*printer_tools, *(tuning_tools or [])]
    tuning_map = {item.name: item for item in supplied if item.name in TUNING_TOOLS}
    tuning_dependencies = list(tuning_map.values())
    # The report turn checks human approval only for submit_print, so it gets only that action
    # and the readings. cancel_job stays with the graph's own skip path (via printer_map), and
    # submit_trial_print is kept for the trial-sheet turn, which has its own approval.
    report_tools = [coding_tool, *(item for item in printer_tools if item.name in REPORT_PRINTER_TOOLS),
                    info_tool]

    def _invoke_printer(name: str, args: dict) -> dict:
        try:
            return _parse_tool_output(printer_map[name].invoke(args))
        except Exception as error:
            return {"ok": False, "error": f"{type(error).__name__}: {error}"}

    def initialize_selected_tuning(state: State, selection: dict) -> dict:
        """Freeze/reuse session before entering the direct tuning route."""
        if _selection_error(selection, state):
            return {}
        path = str(run_dir / "tuning_session.json")
        try:
            session = initialize_tuning_session(run_dir, selection, _successful_focusing_results(state),
                                                paper_color=state.get("paper_color"))
            metadata = {"status": session["initialization"]["status"], "session_id": session["session_id"]}
        except TuningSessionError as error:
            metadata = {"status": "error", "message": str(error)}
        return {"tuning_session_path": path, "tuning_initialization": metadata}

    def orchestrator_node(state: State) -> dict:
        # Also covers pre-existing frozen selections on resume. The session is
        # authoritative; do not mirror its accepted settings into graph state.
        tuning = initialize_selected_tuning(state, state["experiment_selection"]) if state.get("experiment_selection") else {}
        if state.get("experiment_selection") and not _selection_error(state["experiment_selection"], state):
            return {**tuning, "route": "tuning_route"}
        # The budget check comes before any LLM call and spends no decision.
        budget = state.get("decision_budget", DEFAULT_INITIAL_DECISIONS)
        if state.get("decision_count", 0) >= budget:
            return {"route": "budget", **tuning}

        update = run_orchestrator(state, llm)
        update.update(tuning)
        next_role = update["decision"]["next_role"]
        update["stop_reason"] = "running"

        if next_role == "report_and_print":
            count = _count_successful_focusing_results(state)
            if count < MIN_EXPERIMENTS:
                has_plan = any(turn["role"] == "scientist" for turn in state["dialogue"])
                next_role = "executor" if has_plan else "scientist"
                update["decision"]["next_role"] = next_role
                update["decision"]["instruction"] = (
                    "Run another planned focusing experiment with settings not tried yet; "
                    "follow the scientist's shared comparison settings; report its enhancement."
                    if has_plan else
                    f"Plan at least {MIN_EXPERIMENTS} focusing experiments with different "
                    "settings before the executor calls the tool."
                )
                update["dialogue"][0]["text"] = RoutingDecision.model_validate(
                    update["decision"]
                ).model_dump_json()
                update["dialogue"].append({
                    "role": "system",
                    "text": (
                        f"Report refused: only {count} successful focusing experiments "
                        f"are recorded; at least {MIN_EXPERIMENTS} are required. "
                        f"This decision was redirected to {next_role}."
                    ),
                })
        if next_role == "report_and_print":
            update["hardware_passes"] = 0
            selection_error = _selection_error(state.get("experiment_selection") or {}, state)
            if selection_error:
                next_role = "scientist"
                update["decision"]["next_role"] = next_role
                update["decision"]["instruction"] = (
                    "Review all successful results and return a ready four-experiment "
                    "selection, or request additional experiments with exact settings "
                    "and evidence gaps. " + selection_error
                )
                update["dialogue"][0]["text"] = RoutingDecision.model_validate(
                    update["decision"]
                ).model_dump_json()
                update["dialogue"].append({
                    "role": "system", "text": "Report refused: " + selection_error,
                })
        update["route"] = next_role
        return update

    def scientist_node(state: State) -> dict:
        evidence = _load_scientist_evidence(state, run_dir)
        update = run_scientist(state, llm, evidence=evidence)
        response = update["scientist_response"]
        # An accepted selection survives later analysis, print rounds and resume.
        if state.get("experiment_selection"):
            update.update(initialize_selected_tuning(state, state["experiment_selection"]))
            update["route"] = "tuning_route"
            return update
        error = ""
        if response["status"] == "ready":
            error = _selection_error(response, state)
            loaded_ids = {item["experiment_id"] for item in evidence["experiments"]}
            if any(item["experiment_id"] not in loaded_ids for item in response["selections"]):
                error = "Selected experiments must have readable, consistent results.json evidence."
            if response["proposed_experiments"]:
                error = "A ready selection must not also propose more experiments."
            if not error:
                results = {item["experiment_id"]: item
                           for item in _successful_focusing_results(state)}
                update["experiment_selection"] = {
                    "selections": [
                        {**item, "paths": dict(results[item["experiment_id"]]["paths"])}
                        for item in response["selections"]
                    ],
                    "group_reason": response["group_reason"],
                }
                update.update(initialize_selected_tuning(state, update["experiment_selection"]))
                update["route"] = "tuning_route"
        elif response["selections"] or not response["proposed_experiments"]:
            error = "Request more experiments with exact settings and reasons, without selections."
        if error:
            update["dialogue"].append({"role": "system", "text": "Selection not accepted: " + error})
        return update

    def budget_summary_node(state: State) -> dict:
        summary = summarize_progress(state, llm)
        return {
            "budget_summary": summary,
            "dialogue": [{"role": "orchestrator", "text": "Progress summary for the human: " + summary}],
        }

    def budget_review_node(state: State) -> dict:
        used = state.get("decision_count", 0)
        reply = interrupt({
            "kind": "budget",
            "summary": state.get("budget_summary", ""),
            "decisions_used": used,
            "extra": extra_decisions,
            "question": (
                f"Continue for {extra_decisions} more orchestrator decisions? "
                "Answer yes or no; you can add guidance after yes."
            ),
        })
        update = {"dialogue": [{"role": "human", "text": str(reply)}], "hardware_passes": 0}
        if _first_word(reply) in ("yes", "y"):
            budget = state.get("decision_budget", DEFAULT_INITIAL_DECISIONS)
            update.update(decision_budget=budget + extra_decisions, route="orchestrator")
        else:
            update.update(stop_reason="stopped_by_human", route="end")
        return update

    def report_and_print_node(state: State) -> dict:
        # Every entry, including direct/replayed report routing, checks the gate.
        try:
            session = load_tuning_session(state.get("tuning_session_path", run_dir / "tuning_session.json"))
            if session["status"] != "completed":
                return {"route": "tuning_route"}
            if not tuning_completed(session):
                return blocked("Session lacks an accepted final round-3 review and complete final mappings")
        except TuningSessionError as error:
            return blocked(str(error))
        try:
            journal = report_records.load_record(run_dir)
            if state.get("experiment_selection") != session["accepted_selection"]:
                raise ValueError("Graph scientific selection differs from the session's frozen selection/reasons")
            if not journal["attempts"] and (state.get("active_job_id") or any(
                    r.get("tool_name") == "submit_print" for r in state.get("execution_results", []))):
                raise ValueError("Saved report print history has no durable report-print journal; reconcile the prior job before printing")
            status = report_records.print_state(journal)
            if status == "stopped":
                return {"route": "end", "stop_reason": "stopped_by_human", "print_outcome": "stopped"}
            if status == "uncertain" and not journal.get("skip"):
                return {"route": "report_print_confirmation"}
            figure, changed = report_records.prepare_figure(
                state.get("tuning_session_path", run_dir / "tuning_session.json"), session, run_dir)
            state = {**state, "final_figure_path": figure["figure_path"], "final_figure_metadata": figure}
            artifact_update = {"final_figure_path": figure["figure_path"], "final_figure_metadata": figure}
            if changed:
                state.update(pending_print={}, approval={}, print_outcome="", report={})
                artifact_update.update(pending_print={}, approval={}, print_outcome="", report={})
            current = report_records.inspect_tuned_report(run_dir / "report.pdf", figure)
            state["report"] = current
            artifact_update["report"] = current
            if journal.get("skip") == {"report_sha256": current.get("sha256"), "figure_sha256": figure["figure_sha256"],
                                      "inputs_sha256": figure["inputs_sha256"]} and current.get("figure_embedded"):
                return {**artifact_update, "route": "finish_gate", "report": current, "print_outcome": "skipped"}
            if status == "uncertain":
                return {**artifact_update, "route": "report_print_confirmation"}
            if status == "printed":
                report = current
                attempt = journal["attempts"][-1]
                if report_records.attempt_matches(attempt, report, figure):
                    return {**artifact_update, "route": "finish_gate", "report": report, "print_outcome": "printed"}
            if status == "pending":
                job = journal["attempts"][-1]["result"]["job_id"]
                state["active_job_id"] = job
                artifact_update["active_job_id"] = job
            elif status == "failed":
                return {**artifact_update, "route": "printer_question", "last_printer_reading": {
                    "tool_name": "job_status", "result": journal["attempts"][-1]["readings"][-1]
                    if journal["attempts"][-1]["readings"] else journal["attempts"][-1]["result"]}}
            else:
                state["active_job_id"] = None
                artifact_update["active_job_id"] = None
        except (OSError, ValueError, KeyError) as error:
            return report_blocked(str(error))
        error = _selection_error(state.get("experiment_selection") or {}, state)
        if error:
            return {"route": "orchestrator", "dialogue": [{
                "role": "system", "text": "Report refused: " + error,
            }]}
        passes = state.get("hardware_passes", 0) + 1
        if passes > MAX_HARDWARE_PASSES:
            return {
                "hardware_passes": passes,
                "route": "printer_question",
                "dialogue": [{
                    "role": "system",
                    "text": f"The {MAX_HARDWARE_PASSES}-pass report limit was reached; "
                            "human recovery is required.",
                }],
            }

        result = run_report_and_print(state, llm, report_tools, run_dir)
        update = {**artifact_update, **{key: result[key] for key in REPORT_STATE_KEYS if key in result}}
        update["hardware_passes"] = passes
        outcome = result["outcome"]
        if outcome == "tool_ran":
            reading = result["result"]
            active_id = state.get("active_job_id")
            read_id = None
            if result["tool_name"] == "job_status":
                tool_args = result["execution_results"][0].get("tool_args") or {}
                read_id = (reading.get("job_id") if isinstance(reading, dict) else None) \
                    or tool_args.get("job_id")
            # Only a matching current-job reading may affect completion. A job
            # leaving the queue is uncertain, never evidence of a printed report.
            if result["tool_name"] == "job_status" and active_id and tool_args.get("job_id") == active_id \
                    and reading.get("ok") is False and reading.get("job_id") in (None, active_id):
                report_records.record_status_error(run_dir, active_id, reading)
                update.update(report_print_route())
            elif result["tool_name"] == "job_status" and (not active_id or read_id != active_id
                    or tool_args.get("job_id") != active_id or reading.get("job_id") != active_id):
                update["route"] = "report_and_print"
                update["dialogue"] = update.get("dialogue", []) + [{
                    "role": "system",
                    "text": f"job_status was read for {read_id}, but the current print job is "
                            f"{active_id or 'none'}; check job_status for the current job.",
                }]
            elif result["tool_name"] == "job_status":
                report_records.record_outcome(run_dir, journal["attempts"][-1]["attempt_id"], reading, reading=True)
                update.update(report_print_route())
            elif result["tool_name"] == "submit_print":
                return report_blocked("Report executor unexpectedly submitted outside the approval/intent node")
            else:
                update.update(_route_update(result["tool_name"], reading))
        elif outcome == "needs_print_approval":
            if status == "pending":
                update.update(route="report_and_print", pending_print={}, dialogue=[{
                    "role": "system", "text": "Report already submitted; check the active job instead."}])
            else:
                try:
                    report = report_records.inspect_tuned_report(run_dir / "report.pdf", figure)
                    pending = report_records.pending_request(run_dir, result["pending_print"]["args"], report, figure)
                    update.update(route="print_approval", pending_print=pending, report=report)
                except (OSError, ValueError, KeyError) as error:
                    update.update(route="report_and_print", pending_print={}, dialogue=[{
                        "role": "system", "text": "Report approval refused: " + str(error)}])
        elif outcome == "needs_information":
            update["route"] = "orchestrator"
        else:
            update["route"] = "report_and_print"
        return update

    def print_approval_node(state: State) -> dict:
        pending = state.get("pending_print") or {}
        args = dict(pending.get("args", {}))
        report = state.get("report") or {}
        reply = interrupt({
            "kind": "print_approval",
            "pdf_path": args.get("pdf_path"),
            "pages": report.get("pages"),
            "printer": args.get("printer"),
            "copies": args.get("copies", 1),
            "pdf_sha256": pending.get("report_sha256"),
            "question": "Print this report? Type yes, or describe what to change.",
        })
        text = str(reply).strip()

        if _first_word(text) in ("yes", "y", "approve"):
            # No non-idempotent action is performed inside an interrupt node.
            return {"route": "report_submit", "hardware_passes": 0,
                    "dialogue": [{"role": "human", "text": text}]}

        # No limit on change rounds: the human decides when to approve.
        return {
            "pending_print": {},
            "hardware_passes": 0,
            "route": "report_and_print",
            "dialogue": [{"role": "human", "text": "Change request for the report: " + text}],
        }

    def report_blocked(message):
        return {"route": "end", "stop_reason": "report_blocked", "dialogue": [{
            "role": "system", "text": "Report blocked: " + message}]}

    def report_print_route():
        journal = report_records.load_record(run_dir)
        status = report_records.print_state(journal)
        attempt = journal["attempts"][-1]
        result = attempt["result"] or {}
        update = {"active_job_id": result.get("job_id"), "pending_print": {}, "approval": {}}
        if status == "printed":
            return {**update, "route": "finish_gate", "print_outcome": "printed"}
        if status == "uncertain":
            return {**update, "route": "report_print_confirmation", "print_outcome": "uncertain"}
        if status == "failed":
            return {**update, "route": "printer_question", "last_printer_reading": {
                "tool_name": "job_status", "result": (attempt["readings"] or [result])[-1]}}
        if status == "stopped":
            return {**update, "route": "end", "stop_reason": "stopped_by_human", "print_outcome": "stopped"}
        return {**update, "route": "report_and_print"}

    def report_submit_node(state):
        pending = state.get("pending_print") or {}
        try:
            journal = report_records.load_record(run_dir)
            existing = next((a for a in journal["attempts"] if a["attempt_id"] == pending.get("attempt_id")), None)
            if existing:
                if any(existing[k] != pending.get(k) for k in ("args", "report_sha256", "figure_sha256", "inputs_sha256")):
                    return report_blocked("Conflicting replay of a report submission attempt")
                return report_print_route()
            session = load_tuning_session(state["tuning_session_path"])
            if not tuning_completed(session) or state.get("experiment_selection") != session["accepted_selection"]:
                return report_blocked("Completed tuning is required before report submission")
            figure, changed = report_records.prepare_figure(state["tuning_session_path"], session, run_dir)
            current = report_records.inspect_tuned_report(run_dir / "report.pdf", figure)
            if changed:
                raise ValueError("Final figure changed; rebuild the report and request fresh approval")
            attempt, new = report_records.start_submission(run_dir, pending, current, figure)
        except (OSError, ValueError, KeyError) as error:
            return {"route": "report_and_print", "pending_print": {}, "approval": {},
                    "dialogue": [{"role": "system", "text": "Print not submitted: " + str(error)}]}
        if not new:
            return report_print_route()
        try:
            # Invoke ONLY the model's saved, human-approved exact arguments.
            result = _parse_tool_output(printer_map["submit_print"].invoke(attempt["args"]))
            report_records.record_outcome(run_dir, attempt["attempt_id"], result)
        except Exception as error:
            # Persisted intent survives lost responses, exceptions, or a failed
            # outcome write. Never send another copy to find out what happened.
            try:
                record = report_records.load_record(run_dir)
                record["attempts"][-1]["submission_error"] = {"type": type(error).__name__, "message": str(error)}
                report_records.save_record(run_dir, record)
            except (OSError, ValueError):
                pass  # The already-persisted intent remains the recovery boundary.
            return {"route": "report_print_confirmation", "print_outcome": "uncertain", "pending_print": {}}
        return {**report_print_route(), "execution_results": [{"tool_name": "submit_print",
                "tool_args": attempt["args"], "result": result}]}

    def report_print_confirmation_node(state):
        journal = report_records.load_record(run_dir)
        if not journal["attempts"]:
            return report_blocked("No report submission attempt to confirm")
        attempt = journal["attempts"][-1]
        if attempt["resolution"]:
            return report_print_route()
        result = attempt["result"] or {}
        reply = interrupt({"kind": "report_print_confirmation", "job_id": result.get("job_id"),
            "reading": {"submission": attempt["result"], "latest_job_reading": (attempt["readings"] or [None])[-1],
                        "attempt_id": attempt["attempt_id"], "submission_intent_recorded": True,
                        "pdf_path": attempt["args"]["pdf_path"], "pdf_sha256": attempt["report_sha256"],
                        "printer": attempt["args"]["printer"], "copies": attempt["args"]["copies"],
                        "submission_error": attempt.get("submission_error"), "status_error": attempt.get("status_error")},
            "question": "Check the printer tray. Did this exact report print? Choose printed, not_printed, or stop.",
            "choices": ["printed", "not_printed", "stop"]})
        choice = reply.get("action") if isinstance(reply, dict) else None
        if choice not in ("printed", "not_printed", "stop"):
            return {"route": "report_print_confirmation"}
        report_records.resolve_attempt(run_dir, choice)
        if choice == "not_printed":
            return {"route": "report_and_print", "approval": {}, "pending_print": {}, "active_job_id": None,
                    "print_outcome": "", "hardware_passes": 0}
        return report_print_route()

    def printer_question_node(state: State) -> dict:
        reply = interrupt({
            "kind": "printer_problem",
            "reading": state.get("last_printer_reading"),
            "question": "The printer reports a problem. Fix it and type retry, "
                        "or type skip to finish without printing.",
            "choices": ["retry", "skip"],
        })
        update = {"dialogue": [{"role": "human", "text": str(reply)}], "hardware_passes": 0}
        if _first_word(reply) != "skip":
            journal = report_records.load_record(run_dir)
            if report_records.print_state(journal) == "failed":
                report_records.resolve_attempt(run_dir, "not_printed", source="human_retry_after_failure")
                update.update(approval={}, pending_print={}, active_job_id=None, print_outcome="")
            update["route"] = "report_and_print"
            return update

        job_id = state.get("active_job_id")
        job_done = any(
            record.get("tool_name") == "job_status"
            and isinstance(record.get("result"), dict)
            and record["result"].get("job_id") == job_id
            and record["result"].get("state") == "done"
            for record in state.get("execution_results", [])
        )
        if job_id and not job_done and "cancel_job" in printer_map:
            args = {"job_id": job_id}
            result = _invoke_printer("cancel_job", args)
            update["execution_results"] = [
                {"tool_name": "cancel_job", "tool_args": args, "result": result}
            ]
            update["dialogue"].append({
                "role": "system",
                "text": f"Printing skipped; cancel_job {job_id}: "
                        + json.dumps(result, ensure_ascii=False),
            })
        try:
            journal = report_records.load_record(run_dir)
            figure = journal["figure"]
            report = report_records.inspect_tuned_report(run_dir / "report.pdf", figure)
            report_records.require_report(report)
            journal["skip"] = {"report_sha256": report["sha256"], "figure_sha256": figure["figure_sha256"],
                               "inputs_sha256": figure["inputs_sha256"]}
            report_records.save_record(run_dir, journal)
        except (OSError, ValueError, KeyError, TypeError) as error:
            return report_blocked(str(error))
        update.update(print_outcome="skipped", route="finish_gate")
        return update

    def finish_gate_node(state: State) -> dict:
        missing = []
        try:
            session = load_tuning_session(state["tuning_session_path"])
            if not tuning_completed(session) or state.get("experiment_selection") != session["accepted_selection"]:
                return report_blocked("Tuning is not successfully completed")
            figure, changed = report_records.prepare_figure(state["tuning_session_path"], session, run_dir)
            report = report_records.inspect_tuned_report(run_dir / "report.pdf", figure)
            report_records.require_report(report)
            journal = report_records.load_record(run_dir)
            skip = {"report_sha256": report["sha256"], "figure_sha256": figure["figure_sha256"],
                    "inputs_sha256": figure["inputs_sha256"]}
            valid_print = (report_records.print_state(journal) == "printed" and journal["attempts"]
                           and report_records.attempt_matches(journal["attempts"][-1], report, figure))
            if changed or not (valid_print or journal.get("skip") == skip):
                return report_blocked("Report printing is unresolved or belongs to different report/figure bytes")
        except (OSError, ValueError, KeyError, TypeError) as error:
            return report_blocked(str(error))
        selection_error = _selection_error(state.get("experiment_selection") or {}, state)
        if selection_error:
            missing.append("a valid frozen scientist selection: " + selection_error)
        count = _count_successful_focusing_results(state)
        if count < MIN_EXPERIMENTS:
            missing.append(f"{MIN_EXPERIMENTS} successful focusing experiments (found {count})")
        if not report.get("exists") or report.get("pages") != 1:
            missing.append("a one-page report.pdf")
        if not missing:
            return {"stop_reason": "finished", "route": "end", "report": report,
                    "final_figure_path": figure["figure_path"], "final_figure_metadata": figure,
                    "print_outcome": "skipped" if journal.get("skip") == skip else "printed"}
        return {
            "route": "orchestrator",
            "dialogue": [{"role": "system", "text": "Finish refused; missing: " + "; ".join(missing) + "."}],
        }

    # Tuning uses direct, bounded application routing, never orchestrator decisions.
    def blocked(message: str) -> dict:
        return {"route": "end", "stop_reason": "tuning_blocked", "tuning_failure": message,
                "dialogue": [{"role": "system", "text": "Tuning blocked: " + message}]}

    def session_for(state):
        return load_tuning_session(state["tuning_session_path"])

    def tuning_completed(session):
        final = session["rounds"]["3"]
        review = final.get("review") or {}
        return (session["status"] == "completed" and session["phase"] == "completed"
                and session["active_round"] == 3 and bool(session.get("final_settings"))
                and session["final_settings"] == session["accepted_settings"] == final["winners"]
                and review.get("stage") == "review_steps" and review.get("assessment") == "usable")

    def action(name: str, **kwargs) -> dict:
        return {"route": "tuning_executor", "tuning_action": {"action": name, **kwargs}}

    def problem(state, message: str) -> dict:
        failures = state.get("tuning_failures", 0) + 1
        if failures >= 3:
            return blocked("Three tuning failures require application recovery. " + message)
        return {"route": "tuning_problem", "tuning_failure": message, "tuning_failures": failures}

    def latest_attempt(session):
        entry = session["rounds"][str(session["active_round"])]
        attempts = [a for a in entry["print_attempts"] if a["trial_id"] == session["active_trial_id"]]
        return attempts[-1] if attempts else None

    def after_print(state, session) -> dict:
        attempt = latest_attempt(session)
        if not attempt:
            return {"route": "tuning_printer"}
        resolution = (attempt.get("human_resolution") or {}).get("outcome")
        if resolution == "printed":
            return {"route": "tuning_scan"}
        if resolution == "not_printed":
            return {"route": "tuning_recovery"}
        result = attempt["result"]
        reading = (attempt["job_results"] or [result])[-1] or {}
        if (result is None or result.get("tracked") is False or reading.get("state") == "uncertain"
                or reading.get("confirmed") is False):
            return {"route": "tuning_confirmation"}
        if reading.get("ok") is False or reading.get("state") == "failed":
            return {"route": "tuning_recovery"}
        if reading.get("state") == "done" and reading.get("confirmed") is True:
            return {"route": "tuning_scan"}
        if _has_problem_field(reading):
            return problem(state, "Printer job problem: " + json.dumps(reading))
        if not isinstance(result.get("job_id"), str) or not result["job_id"]:
            return {"route": "tuning_confirmation"}
        if state.get("tuning_polls", 0) >= 3:
            return {"route": "tuning_poll"}
        return action("job_status", job_id=result["job_id"])

    def tuning_route_node(state):
        try:
            session = session_for(state)
            if session["status"] == "completed":
                if not tuning_completed(session):
                    return blocked("Completed session lacks round-3 final settings")
                return {"route": "report_and_print", "hardware_passes": 0}
            if session["status"] == "stopped":
                return {"route": "end", "stop_reason": "stopped_by_human"}
            if session["status"] != "active":
                review = session["rounds"][str(session["active_round"])].get("review") or {}
                explanation = review.get("issues") or session.get("outcomes", [])[-1:] or session["phase"]
                return blocked(f"Session {session['status']}: {explanation}")
            missing = TUNING_TOOLS - tuning_map.keys()
            if missing:
                return blocked("Missing injected tuning tools: " + ", ".join(sorted(missing)))
            phase = session["phase"]
            if phase == "awaiting_gamma_proposal":
                return {"route": "tuning_specialist"}
            if phase == "ready_for_trial":
                return action("make_trial_sheet")
            entry = session["rounds"][str(session["active_round"])]
            if entry.get("reprint_requested"):
                return {"route": "tuning_printer"}
            # A recorded rescan is reviewable once; an accepted rescan review is
            # followed by human scan input, not another call on the same evidence.
            evaluation = entry["active_evaluation_id"]
            reviewed = any(r["evaluation_record_id"] == evaluation for r in entry["reviews"])
            if evaluation and not reviewed:
                return {"route": "tuning_specialist"}
            if evaluation and reviewed:
                return {"route": "tuning_scan"}
            return after_print(state, session)
        except (TuningSessionError, KeyError, OSError) as error:
            return blocked(str(error))

    def tuning_specialist_node(state):
        try:
            session = session_for(state)
            if session["status"] != "active" or session["phase"] not in (
                    "awaiting_gamma_proposal", "awaiting_review", "awaiting_scan"):
                return {"route": "tuning_route"}
            if session["phase"] != "awaiting_gamma_proposal":
                entry = session["rounds"][str(session["active_round"])]
                evaluation = entry["active_evaluation_id"]
                if not evaluation or any(r["evaluation_record_id"] == evaluation for r in entry["reviews"]):
                    # Acceptance may have committed before this node checkpoint.
                    return {"route": "tuning_route"}
            stage = ("propose_gamma" if session["phase"] == "awaiting_gamma_proposal" else
                     {1: "review_gamma", 2: "review_gray_levels", 3: "review_steps"}[session["active_round"]])
            inputs = prepare_tuning_inputs(state["tuning_session_path"], stage)
            response = photo_quality_specialist(
                specialist_llm if specialist_llm is not None else llm, inputs.context, inputs.evidence,
                prior_decisions=inputs.prior_decisions, paper_color=inputs.paper_color)
            accept_specialist_response(state["tuning_session_path"], response, inputs.context,
                                       evaluation_record_id=inputs.evaluation_record_id)
            return {"route": "tuning_route", "tuning_polls": 0,
                    "dialogue": [{"role": "photo_quality_specialist", "text": response.model_dump_json()}]}
        except Exception as error:
            # Provider and bounded specialist errors surface to human/application
            # recovery, never as scan evidence or automatic model retries here.
            return problem(state, f"Specialist input/decision failed: {type(error).__name__}: {error}")

    def tuning_executor_node(state):
        task = TuningTask(**state["tuning_action"])
        result = tuning_executor(llm, tuning_dependencies, state["tuning_session_path"], task)
        update = {key: result[key] for key in ("dialogue", "execution_results") if key in result}
        outcome = result["outcome"]
        if outcome == "needs_trial_print_approval":
            session = session_for(state)
            entry = session["rounds"][str(session["active_round"])]
            pending = result["pending_trial_print"]
            consumed = {a["approval_id"] for a in entry["print_attempts"]}
            # Recovery after approval was saved but before intent was recorded:
            # ask again, then reuse the unconsumed identity (never a used one).
            reusable = [a["metadata"]["approval_id"] for a in entry["approvals"]
                        if a["metadata"]["approval_id"] not in consumed
                        and a["metadata"]["trial_id"] == pending["trial_id"]
                        and a["metadata"]["pdf_sha256"] == pending["pdf_sha256"]
                        and a["metadata"]["printer"] == pending["args"]["printer"]]
            update.update(route="tuning_approval", pending_trial_print={
                **pending, "approval_id": reusable[-1] if reusable else uuid4().hex})
        elif result.get("tool_name") == "job_status" and result.get("tool_executed") and result.get("result"):
            session = session_for(state)
            attempt = latest_attempt(session)
            record_print_result(state["tuning_session_path"], attempt_id=attempt["attempt_id"],
                                result=result["result"], job_result=True)
            update.update(after_print({**state, "tuning_polls": state.get("tuning_polls", 0) + 1}, session_for(state)))
            update["tuning_polls"] = state.get("tuning_polls", 0) + 1
        elif outcome in ("tool_ran", "needs_information") and task.action == "read_scan" and result.get("tool_executed"):
            if result.get("result", {}).get("status") in ("ok", "needs_rescan"):
                update["route"] = "tuning_route"
            else:
                update.update(problem(state, result["message"]))
        elif outcome == "tool_ran" and task.action == "make_trial_sheet":
            update.update(route="tuning_printer", tuning_polls=0)
        elif outcome == "tool_ran" and task.action == "printer_status":
            reading = result["result"]
            if reading.get("state") != "idle" or _has_problem_field(reading):
                update.update(problem(state, "Printer is not ready: " + json.dumps(reading)))
            else:
                update.update(action("submit_trial_print", printer=state["tuning_printer"]))
        elif outcome == "needs_information" and result.get("information_required", {}).get("kind") == "print_confirmation":
            update["route"] = "tuning_confirmation"
        else:
            update.update(problem(state, result["message"]))
        return update

    def stop_tuning(state):
        record_stop(state["tuning_session_path"], "Human stopped print tuning")
        return {"route": "end", "stop_reason": "stopped_by_human", "pending_trial_print": None}

    def tuning_printer_node(state):
        printer = state.get("tuning_printer")
        if not printer:
            reply = interrupt({"kind": "tuning_printer", "question": "Supply the exact printer name, or stop.",
                               "choices": ["printer", "stop"]})
            if isinstance(reply, dict) and reply.get("action") == "stop":
                return stop_tuning(state)
            if not (isinstance(reply, dict) and reply.get("action") == "printer"
                    and isinstance(reply.get("printer"), str) and reply["printer"].strip()):
                return {"route": "tuning_printer"}
            printer = reply["printer"]
        return {**action("printer_status", printer=printer), "tuning_printer": printer}

    def tuning_approval_node(state):
        pending = state.get("pending_trial_print") or {}
        reply = interrupt({"kind": "trial_print_approval", "pdf_path": pending.get("args", {}).get("pdf_path"),
                           "page_count": pending.get("trial_result", {}).get("page_count"),
                           "printer": pending.get("args", {}).get("printer"),
                           "session_id": pending.get("session_id"), "round_id": pending.get("round_id"),
                           "trial_id": pending.get("trial_id"), "pdf_sha256": pending.get("pdf_sha256"),
                           "question": "Approve printing this exact trial PDF? Choose approve or stop.",
                           "choices": ["approve", "stop"]})
        if isinstance(reply, dict) and reply.get("action") == "stop":
            return stop_tuning(state)
        if not isinstance(reply, dict) or reply.get("action") != "approve":
            return {"route": "tuning_approval"}
        return {"route": "tuning_submit"}

    def tuning_submit_node(state):
        pending = state["pending_trial_print"]
        approval_id = pending["approval_id"]
        attempt_id = approval_id + ":submission"
        try:
            session = session_for(state)
            entry = session["rounds"][str(session["active_round"])]
            # Replay after side effects: NEVER invoke the printer again.
            if any(a["attempt_id"] == attempt_id for a in entry["print_attempts"]):
                return {**after_print(state, session), "pending_trial_print": None}
            fresh = _pending_trial(session, Path(state["tuning_session_path"]).resolve().parent, pending["args"])
            if fresh != {k: v for k, v in pending.items() if k != "approval_id"}:
                raise TuningSessionError("Pending approval no longer matches the active trial/artifacts")
            record_print_approval(state["tuning_session_path"], approval_id=approval_id,
                trial_id=pending["trial_id"], pdf_sha256=pending["pdf_sha256"], printer=pending["args"]["printer"],
                print_settings={**pending["args"], "copies": 1, "paper": "A4", "color_mode": "grayscale",
                                "sides": "one-sided", "scale": "100%"},
                approved_by="graph human interrupt", reprint=bool(entry["print_attempts"]))
            begin_print_attempt(state["tuning_session_path"], approval_id=approval_id, attempt_id=attempt_id)
        except (TuningSessionError, KeyError, ValueError, OSError) as error:
            return {**problem(state, "Approval invalid; fresh approval is required: " + str(error)),
                    "pending_trial_print": None}
        try:
            # The only graph submission path uses the executor's exact request.
            result = _parse_tool_output(tuning_map["submit_trial_print"].invoke(pending["args"]))
        except Exception:
            # Intent is persisted, outcome is absent: ask the human, never retry.
            return {"route": "tuning_confirmation", "pending_trial_print": None}
        try:
            record_print_result(state["tuning_session_path"], attempt_id=attempt_id, result=result)
        except TuningSessionError as error:
            return {"route": "tuning_confirmation", "pending_trial_print": None,
                    "tuning_failure": "Submission outcome could not be recorded: " + str(error)}
        return {**after_print(state, session_for(state)), "pending_trial_print": None, "tuning_polls": 0,
                "execution_results": [{"tool_name": "submit_trial_print", "tool_args": pending["args"], "result": result}]}

    def tuning_scan_node(state):
        session = session_for(state)
        reply = interrupt({"kind": "tuning_scan", "round_id": session["active_round"],
            "trial_id": session["active_trial_id"],
            "suggested_scan_directory": str(run_dir / "scans" / f"round_{session['active_round']}"),
            "question": "Scan the printed pages with your phone, save them, and supply explicit paths; or reprint/stop.",
            "choices": ["scan", "reprint", "stop"]})
        if not isinstance(reply, dict):
            return {"route": "tuning_scan"}
        choice = reply.get("action")
        if choice == "stop":
            return stop_tuning(state)
        if choice == "reprint":
            return reprint(state)
        paths = reply.get("scan_paths")
        if choice == "scan" and isinstance(paths, list) and paths and all(isinstance(p, str) and p.strip() for p in paths):
            return action("read_scan", scan_paths=paths)
        return {"route": "tuning_scan"}

    def reprint(state):
        try:
            request_trial_reprint(state["tuning_session_path"])
            return {"route": "tuning_printer", "pending_trial_print": None, "tuning_polls": 0}
        except TuningSessionError as error:
            return blocked(str(error))

    def tuning_confirmation_node(state):
        attempt = latest_attempt(session_for(state))
        if not attempt:
            return blocked("No recorded print attempt to resolve")
        reply = interrupt({"kind": "tuning_print_confirmation", "attempt_id": attempt["attempt_id"],
                           "question": "Check the printer tray. Did this trial print?",
                           "choices": ["printed", "not_printed", "stop"]})
        choice = reply.get("action") if isinstance(reply, dict) else None
        if choice == "stop":
            return stop_tuning(state)
        if choice not in ("printed", "not_printed"):
            return {"route": "tuning_confirmation"}
        resolve_print_outcome(state["tuning_session_path"], attempt_id=attempt["attempt_id"], outcome=choice)
        return {"route": "tuning_scan"} if choice == "printed" else reprint(state)

    def tuning_poll_node(state):
        reply = interrupt({"kind": "tuning_poll", "question": "Job is still waiting/printing. Check again or stop?",
                           "choices": ["check", "stop"]})
        choice = reply.get("action") if isinstance(reply, dict) else None
        if choice == "stop":
            return stop_tuning(state)
        return {"route": "tuning_route", "tuning_polls": 0} if choice == "check" else {"route": "tuning_poll"}

    def tuning_problem_node(state):
        reply = interrupt({"kind": "tuning_problem", "message": state.get("tuning_failure"),
                           "question": "Resolve the problem, then retry; regenerate only damaged trial artifacts, or stop.",
                           "choices": ["retry", "regenerate", "stop"]})
        choice = reply.get("action") if isinstance(reply, dict) else None
        if choice == "stop":
            return stop_tuning(state)
        if choice == "regenerate":
            session = session_for(state)
            entry = session["rounds"][str(session["active_round"])]
            if entry["trial_result"]:
                try:
                    _bound_trial(session, session["active_trial_id"])
                    _trial_manifest(session, entry["trial_result"], entry)
                except (TuningSessionError, OSError, ValueError):
                    if entry["print_attempts"]:
                        try:
                            request_trial_reprint(state["tuning_session_path"])
                        except TuningSessionError as error:
                            return blocked(str(error))
                    return {**action("make_trial_sheet"), "pending_trial_print": None}
                return problem(state, "Existing trial artifacts are valid; use the normal reprint path")
            return action("make_trial_sheet")
        if choice == "retry":
            # Re-read the authoritative phase; never blindly repeat submission.
            return {"route": "tuning_route"}
        return {"route": "tuning_problem"}

    def tuning_recovery_node(state):
        reply = interrupt({"kind": "tuning_problem", "question": "The trial did not print successfully. Reprint with fresh approval, or stop?",
                           "choices": ["reprint", "stop"]})
        choice = reply.get("action") if isinstance(reply, dict) else None
        if choice == "stop":
            return stop_tuning(state)
        return reprint(state) if choice == "reprint" else {"route": "tuning_recovery"}

    def route(state: State) -> str:
        return state.get("route", "orchestrator")

    builder = StateGraph(State)
    builder.add_node("orchestrator", orchestrator_node)
    builder.add_node("scientist", scientist_node)
    builder.add_node("general", lambda state: run_general(state, llm))
    builder.add_node("executor", lambda state: run_executor(state, llm, focusing_tool))
    builder.add_node("report_and_print", report_and_print_node)
    builder.add_node("print_approval", print_approval_node)
    builder.add_node("report_submit", report_submit_node)
    builder.add_node("report_print_confirmation", report_print_confirmation_node)
    builder.add_node("printer_question", printer_question_node)
    builder.add_node("budget_summary", budget_summary_node)
    builder.add_node("budget_review", budget_review_node)
    builder.add_node("finish_gate", finish_gate_node)
    tuning_nodes = {
        "tuning_route": tuning_route_node, "tuning_specialist": tuning_specialist_node,
        "tuning_executor": tuning_executor_node, "tuning_printer": tuning_printer_node,
        "tuning_approval": tuning_approval_node, "tuning_submit": tuning_submit_node,
        "tuning_scan": tuning_scan_node, "tuning_confirmation": tuning_confirmation_node,
        "tuning_poll": tuning_poll_node, "tuning_problem": tuning_problem_node,
        "tuning_recovery": tuning_recovery_node,
    }
    for name, node in tuning_nodes.items():
        builder.add_node(name, node)
        builder.add_conditional_edges(name, route, {
            **{target: target for target in tuning_nodes},
            "report_and_print": "report_and_print", "end": END,
        })

    builder.add_edge(START, "orchestrator")
    builder.add_conditional_edges("orchestrator", route, {
        "scientist": "scientist", "general": "general", "executor": "executor",
        "report_and_print": "report_and_print", "budget": "budget_summary",
        "tuning_route": "tuning_route",
    })
    builder.add_conditional_edges("scientist", lambda state: "tuning_route" if state.get("experiment_selection") else "orchestrator",
                                  {"tuning_route": "tuning_route", "orchestrator": "orchestrator"})
    for role in ("general", "executor"):
        builder.add_edge(role, "orchestrator")
    builder.add_edge("budget_summary", "budget_review")
    builder.add_conditional_edges("budget_review", route, {
        "orchestrator": "orchestrator", "end": END,
    })
    builder.add_conditional_edges("report_and_print", route, {
        "report_and_print": "report_and_print", "print_approval": "print_approval",
        "printer_question": "printer_question", "finish_gate": "finish_gate",
        "orchestrator": "orchestrator",
        "tuning_route": "tuning_route", "end": END,
        "report_print_confirmation": "report_print_confirmation",
    })
    builder.add_conditional_edges("print_approval", route, {
        "report_and_print": "report_and_print", "printer_question": "printer_question",
        "finish_gate": "finish_gate", "orchestrator": "orchestrator",
        "report_submit": "report_submit",
    })
    for name in ("report_submit", "report_print_confirmation"):
        builder.add_conditional_edges(name, route, {
            "report_and_print": "report_and_print", "report_print_confirmation": "report_print_confirmation",
            "printer_question": "printer_question", "finish_gate": "finish_gate", "end": END,
        })
    builder.add_conditional_edges("printer_question", route, {
        "report_and_print": "report_and_print", "finish_gate": "finish_gate",
        "end": END,
    })
    builder.add_conditional_edges("finish_gate", route, {
        "orchestrator": "orchestrator", "end": END,
    })

    return builder.compile(checkpointer=checkpointer)
