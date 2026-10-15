import json
import math
from pathlib import Path

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.tools import BaseTool
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from app.workflow.state import State

from app.agents.common import DEFAULT_INITIAL_DECISIONS, MIN_EXPERIMENTS
from app.agents.orchestrator import RoutingDecision, orchestrator as run_orchestrator, summarize_progress
from app.agents.scientist import ExperimentSelection, scientist as run_scientist
from app.agents.executor import executor as run_executor
from app.agents.general import general as run_general
from app.agents.report_and_print import (
    MAX_HARDWARE_PASSES, _approval_key, _parse_tool_output, inspect_report,
    report_and_print as run_report_and_print,
)


# Keys of a report_and_print result that belong in the graph state.
REPORT_STATE_KEYS = ("dialogue", "execution_results", "report", "active_job_id", "pending_print")
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
        if state == "done":
            return "finish_gate"
        if state in PROBLEM_JOB_STATES or _has_problem_field(result):
            return "printer_question"
    elif tool_name == "printer_status":
        if state in PROBLEM_PRINTER_STATES or _has_problem_field(result):
            return "printer_question"
    elif tool_name == "submit_print" and result.get("ok") is False:
        if "offline" in str(result.get("error", "")).lower():
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
):
    """Route the four roles; only the executor's report_and_print turn gets the printer."""
    run_dir = Path(run_dir).resolve()
    printer_map = {item.name: item for item in printer_tools}
    report_tools = [coding_tool, *printer_tools, info_tool]

    def _invoke_printer(name: str, args: dict) -> dict:
        try:
            return _parse_tool_output(printer_map[name].invoke(args))
        except Exception as error:
            return {"ok": False, "error": f"{type(error).__name__}: {error}"}

    def orchestrator_node(state: State) -> dict:
        # The budget check comes before any LLM call and spends no decision.
        budget = state.get("decision_budget", DEFAULT_INITIAL_DECISIONS)
        if state.get("decision_count", 0) >= budget:
            return {"route": "budget"}

        update = run_orchestrator(state, llm)
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
        error = _selection_error(state.get("experiment_selection") or {}, state)
        if error:
            return {"route": "orchestrator", "dialogue": [{
                "role": "system", "text": "Report refused: " + error,
            }]}
        passes = state.get("hardware_passes", 0) + 1
        if passes > MAX_HARDWARE_PASSES:
            return {
                "hardware_passes": passes,
                "route": "orchestrator",
                "dialogue": [{
                    "role": "system",
                    "text": f"The {MAX_HARDWARE_PASSES}-pass hardware limit was reached; "
                            "back to the orchestrator.",
                }],
            }

        result = run_report_and_print(state, llm, report_tools, run_dir)
        update = {key: result[key] for key in REPORT_STATE_KEYS if key in result}
        update["hardware_passes"] = passes
        outcome = result["outcome"]
        if outcome == "tool_ran":
            reading = result["result"]
            active_id = update.get("active_job_id", state.get("active_job_id"))
            read_id = None
            if result["tool_name"] == "job_status":
                tool_args = result["execution_results"][0].get("tool_args") or {}
                read_id = (reading.get("job_id") if isinstance(reading, dict) else None) \
                    or tool_args.get("job_id")
            # Only the current print job may finish the run: on Windows a job that
            # left the queue reads as done, so a wrong id would look printed.
            if result["tool_name"] == "job_status" and read_id != active_id:
                update["route"] = "report_and_print"
                update["dialogue"] = update.get("dialogue", []) + [{
                    "role": "system",
                    "text": f"job_status was read for {read_id}, but the current print job is "
                            f"{active_id or 'none'}; check job_status for the current job.",
                }]
            else:
                update.update(_route_update(result["tool_name"], reading))
        elif outcome == "needs_print_approval":
            update["route"] = "print_approval"
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
            "question": "Print this report? Type yes, or describe what to change.",
        })
        text = str(reply).strip()

        if _first_word(text) in ("yes", "y", "approve"):
            # The file must still be the version the human was shown.
            current = inspect_report(args.get("pdf_path", ""))
            if not pending or current.get("sha256") != pending.get("report_sha256"):
                return {
                    "pending_print": {},
                    "hardware_passes": 0,
                    "report": current,
                    "route": "report_and_print",
                    "dialogue": [
                        {"role": "human", "text": text},
                        {"role": "system", "text": "Print not submitted: report.pdf changed "
                                                   "after approval was requested."},
                    ],
                }
            result = _invoke_printer("submit_print", args)
            update = {
                "pending_print": {},
                "hardware_passes": 0,
                "approval": _approval_key(pending["report_sha256"], args),
                "execution_results": [
                    {"tool_name": "submit_print", "tool_args": args, "result": result}
                ],
                "dialogue": [
                    {"role": "human", "text": text},
                    {"role": "executor", "text": "Human approved. submit_print result: "
                                                 + json.dumps(result, ensure_ascii=False)},
                ],
            }
            if result.get("ok") is not False and result.get("job_id"):
                update["active_job_id"] = result["job_id"]
            update.update(_route_update("submit_print", result))
            return update

        # No limit on change rounds: the human decides when to approve.
        return {
            "pending_print": {},
            "hardware_passes": 0,
            "route": "report_and_print",
            "dialogue": [{"role": "human", "text": "Change request for the report: " + text}],
        }

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
        update.update(print_outcome="skipped", route="finish_gate")
        return update

    def finish_gate_node(state: State) -> dict:
        missing = []
        selection_error = _selection_error(state.get("experiment_selection") or {}, state)
        if selection_error:
            missing.append("a valid frozen scientist selection: " + selection_error)
        count = _count_successful_focusing_results(state)
        if count < MIN_EXPERIMENTS:
            missing.append(f"{MIN_EXPERIMENTS} successful focusing experiments (found {count})")
        report = state.get("report") or {}
        if not report.get("exists") or report.get("pages") != 1:
            missing.append("a one-page report.pdf")
        if state.get("print_outcome") not in ("printed", "skipped"):
            missing.append("a printed or human-skipped print job")
        if not missing:
            return {"stop_reason": "finished", "route": "end"}
        return {
            "route": "orchestrator",
            "dialogue": [{"role": "system", "text": "Finish refused; missing: " + "; ".join(missing) + "."}],
        }

    def route(state: State) -> str:
        return state.get("route", "orchestrator")

    builder = StateGraph(State)
    builder.add_node("orchestrator", orchestrator_node)
    builder.add_node("scientist", scientist_node)
    builder.add_node("general", lambda state: run_general(state, llm))
    builder.add_node("executor", lambda state: run_executor(state, llm, focusing_tool))
    builder.add_node("report_and_print", report_and_print_node)
    builder.add_node("print_approval", print_approval_node)
    builder.add_node("printer_question", printer_question_node)
    builder.add_node("budget_summary", budget_summary_node)
    builder.add_node("budget_review", budget_review_node)
    builder.add_node("finish_gate", finish_gate_node)

    builder.add_edge(START, "orchestrator")
    builder.add_conditional_edges("orchestrator", route, {
        "scientist": "scientist", "general": "general", "executor": "executor",
        "report_and_print": "report_and_print", "budget": "budget_summary",
    })
    for role in ("scientist", "general", "executor"):
        builder.add_edge(role, "orchestrator")
    builder.add_edge("budget_summary", "budget_review")
    builder.add_conditional_edges("budget_review", route, {
        "orchestrator": "orchestrator", "end": END,
    })
    builder.add_conditional_edges("report_and_print", route, {
        "report_and_print": "report_and_print", "print_approval": "print_approval",
        "printer_question": "printer_question", "finish_gate": "finish_gate",
        "orchestrator": "orchestrator",
    })
    builder.add_conditional_edges("print_approval", route, {
        "report_and_print": "report_and_print", "printer_question": "printer_question",
        "finish_gate": "finish_gate", "orchestrator": "orchestrator",
    })
    builder.add_conditional_edges("printer_question", route, {
        "report_and_print": "report_and_print", "finish_gate": "finish_gate",
    })
    builder.add_conditional_edges("finish_gate", route, {
        "orchestrator": "orchestrator", "end": END,
    })

    return builder.compile(checkpointer=checkpointer)
