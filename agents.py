import hashlib
import json
import math
import traceback
from pathlib import Path
from typing import Literal

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool, tool
from pydantic import BaseModel, Field


MIN_EXPERIMENTS = 2
# Counts report_and_print passes since the last human answer;
# graph.py resets the counter to 0 after every human answer.
MAX_HARDWARE_PASSES = 12
DEFAULT_INITIAL_DECISIONS = 10
DEFAULT_EXTRA_DECISIONS = 5
# A pre-approved report may be resubmitted only if its last job was lost.
LOST_JOB_STATES = {"failed", "cancelled", "canceled", "error", "aborted", "unknown"}


ORCHESTRATOR_PROMPT = """
You coordinate autonomous focusing at the fixed camera target and the printed
one-page report of the results. Read the full dialogue and tool evidence. Return
next_role, a short instruction, and a one-sentence reason. Choose the role
needed now, not a fixed sequence:
- scientist: optical theory, experiment planning, analysis and the report spec;
- general: plain explanations, including the explanation of the report graph;
- executor: runs focusing experiments;
- report_and_print: the executor builds the report PDF and prints it after
  human approval.

The typical path is: scientist plan, at least the supplied minimum of successful
focusing experiments, scientist analysis with a REPORT SPEC, general explanation
of the chosen graph, then report_and_print. Get a scientist plan before
experiments and reuse it unless revision is needed. Only the executor runs
tools; you write no theory or code.

For fair comparisons, change one setting and hold the others fixed, including
search_seed, focus_size, background_size and iterations when varying macropixels.
A successful experiment has ok=true, finite enhancements, saved paths and no
validation_error; proposed experiments and supporting code do not count.

There is no finish choice. The run ends inside report_and_print, after the
report is printed or the human skips printing. If report_and_print came back
with "Needs information", read what is missing and the human's request, route
to the role that can supply it (scientist for analysis or verdict, executor for
a new experiment), then route back to report_and_print. If a system turn says
finishing was refused, route to the role that can supply what it names, then
back to report_and_print.

Human approval of printing is handled by the graph. Do not ask the human for
settings or approval yourself. Each orchestrator choice consumes one decision
from the current block; specialist replies and tool calls do not. Use the
remaining decisions wisely and get the report printed rather than counting on
extra decisions. Base decisions and conclusions on recorded results, never
invented measurements.
"""

SCIENTIST_PROMPT = """
You own optical theory, experiment design and the analysis of results. Follow the
latest instruction and use the dialogue's bench dimensions, target, noise and
tool constraints. Explain transmission, binary-amplitude DMD patterns, macropixels
and noise as needed.
Enhancement = focus_mean / max((TL_mean + TR_mean + BL_mean + BR_mean) / 4, 1e-12).
The four background means have equal weight; measurements are noisy.

Planning: give a short numbered plan that meets the supplied experiment minimum
and fits the current decision block. State shared settings once, then each
experiment's single changed setting and reason. Use macropixel_size, focus_size,
background_size, iterations and search_seed. Correct proposals that change
several settings at once. For macropixel comparisons keep iterations,
search_seed and measurement regions fixed across experiments. Budget about 20
iterations per controllable macropixel or more, capped at 1,000; use a common
budget sufficient for the finer grid where possible and explain any shortfall
caused by the cap.

The tool averages three shots per measurement and validates with five fresh shots.
Use its final_enhancement; no extra validation experiment or configurable shot counts.
Reuse existing plans and actual evidence; describe only needed revisions.

Analysis: when asked for the analysis after the experiments,
- judge whether the comparison is fair: one setting changed, same seed and
  regions, enough iterations for both;
- compare performance (final_enhancement) with computational cost
  (wall_time_seconds and camera_shots when the recorded results have them,
  otherwise the iterations and shots per measurement from the settings);
- suggest one next experiment and why;
- end with a block whose first line is exactly REPORT SPEC, containing:
  graph: exactly one of bar_final_enhancement (bars of final enhancement per
    experiment), enhancement_vs_iteration (current enhancement against
    iteration for each experiment, from the results.json history) or
    before_after_cameras (initial and final camera images of the best experiment);
  layout: title, where the graph and the text go, A4 portrait, one page;
  fairness, performance_vs_cost and next_experiment: one sentence each;
  verdict: one or two sentences.
Use only recorded numbers and name the experiment IDs.

You have no tools and do not invent results. If asked to execute, give ready-to-run
settings and direct the orchestrator to the executor. Do not ask the human for settings.
"""

GENERAL_PROMPT = """
Follow the orchestrator's instruction. Explain supporting optical or programming
concepts briefly and clearly. Leave experiment design to the scientist and code
and execution to the executor. Use recorded evidence; never invent measurements.
When asked for the report explanation, write 2-3 plain sentences that explain
what the graph in the latest REPORT SPEC shows and how to read it, for a reader
who is not an optics expert. Add no new numbers.
"""

EXECUTOR_PROMPT = """
You run the focusing experiments. Follow the latest instruction and existing
scientist plan; if no plan exists, request one without running an experiment.
Make exactly one run_focusing_experiment call per turn and use the planned
settings exactly, including search_seed.

The focusing tool averages three shots and validates with five fresh shots.
Its final_enhancement is ready to report; another call starts a new search,
not an extra validation. Follow the metric supplied in the bench context.
After the tool call, briefly report measured enhancements and saved paths
(and wall_time_seconds and camera_shots when the result has them), or its
actual error. A validation_error means invalid evidence.
Do not invent results, repeat the plan or request a second tool call.
"""

REPORT_AND_PRINT_PROMPT = """
You are the executor in the report-and-print turn. Make exactly one tool call per
pass. After each call its result is added to the dialogue and you are called again.

Steps:
1. If there is no up-to-date one-page report.pdf, call write_and_run with a
   complete script that reads experiments.csv and the results.json files named in
   the recorded results (never type numbers into the script), draws the graph
   named in the latest REPORT SPEC, adds the title, the explanation from general,
   the fairness, performance-vs-cost and next-experiment sentences and the
   verdict, and saves report.pdf in the current folder as exactly one A4 portrait
   page (matplotlib savefig or PdfPages is enough). Use a fresh script filename.
2. Call printer_status for the office printer named in the context.
3. Call submit_print with pdf_path set to the report path given below. The graph
   asks the human for approval before it really prints.
4. Call job_status with the returned job id until the job is done.

After a human turn starting with "Change request for the report:":
- if it only changes presentation (layout, titles, wording that does not change
  a claim, colours, sizes, which recorded results are plotted), write a revised
  script with a new filename, rebuild report.pdf and call submit_print again;
- if it changes a number, a claim or the verdict, or needs data or analysis you
  do not have, call request_information with missing (what is needed and from
  whom) and human_request (the human's words). Never change a claim just because
  the human asked; claims must come from recorded results.

After a printer problem and a human "retry", check job_status (or printer_status)
again; resubmit only if the job failed, was cancelled or was lost. If a tool returns an
error, read it and fix the cause on the next pass, for example a report with
two pages. Never invent results.
"""

SUMMARY_PROMPT = """
You are the orchestrator. The current block of orchestrator decisions is used up
and the human decides whether to grant more. Write at most 8 short lines for the
human: what has been done (experiments with their final enhancements, report
state, print state), what you plan next, and why you need more decisions.
Use only recorded results; never invent measurements.
"""


class RoutingDecision(BaseModel):
    """Required output of each orchestrator call."""

    next_role: Literal["scientist", "general", "executor", "report_and_print"]
    instruction: str = Field(min_length=1, description="Instruction for the chosen role")
    reason: str = Field(min_length=1, description="One-sentence reason for this choice")


def _messages(
    state: dict, prompt: str, *, include_budget: bool = False, extra: str = ""
) -> list:
    """Keep agent names intact without pretending they are API message roles."""
    dialogue = json.dumps(state["dialogue"], ensure_ascii=False)
    evidence = json.dumps(state.get("execution_results", []), ensure_ascii=False)
    content = "Dialogue:\n" + dialogue + "\n\nRecorded tool results:\n" + evidence
    if include_budget:
        used = state.get("decision_count", 0)
        budget_limit = state.get("decision_budget", DEFAULT_INITIAL_DECISIONS)
        budget = {
            "orchestrator_decisions_used": used,
            "current_decision_budget": budget_limit,
            "decisions_remaining_in_this_block": max(0, budget_limit - used),
            "note": (
                "When this block is used up the human is shown a summary and may "
                "grant more decisions; more are not guaranteed."
            ),
            "minimum_successful_focusing_experiments": MIN_EXPERIMENTS,
        }
        content = "Decision budget:\n" + json.dumps(budget) + "\n\n" + content
    if extra:
        content += "\n\n" + extra
    return [SystemMessage(content=prompt), HumanMessage(content=content)]


def _summarize(llm: BaseChatModel, tools: list, messages: list) -> str:
    """Ask for a short summary with tools declared but disabled."""
    try:
        # Keep the tool declarations, but disable further calls.
        return llm.bind_tools(tools, tool_choice="none").invoke(messages).text
    except Exception as error:
        # A summary failure must not discard a completed tool result.
        return f"Model summary unavailable ({type(error).__name__}); see the raw tool result."


def orchestrator(state: dict, llm: BaseChatModel) -> dict:
    decision = llm.with_structured_output(
        RoutingDecision, method="json_schema"
    ).invoke(_messages(state, ORCHESTRATOR_PROMPT, include_budget=True))
    return {
        "dialogue": [{"role": "orchestrator", "text": decision.model_dump_json()}],
        "decision": decision.model_dump(),
        "decision_count": state.get("decision_count", 0) + 1,
    }


def scientist(state: dict, llm: BaseChatModel) -> dict:
    response = llm.invoke(_messages(state, SCIENTIST_PROMPT, include_budget=True))
    return {"dialogue": [{"role": "scientist", "text": response.text}]}


def general(state: dict, llm: BaseChatModel) -> dict:
    response = llm.invoke(_messages(state, GENERAL_PROMPT))
    return {"dialogue": [{"role": "general", "text": response.text}]}


def summarize_progress(state: dict, llm: BaseChatModel) -> str:
    """Progress summary for the budget review; does not spend a decision."""
    return llm.invoke(_messages(state, SUMMARY_PROMPT, include_budget=True)).text


def executor(state: dict, llm: BaseChatModel, focusing_tool: BaseTool) -> dict:
    """Run one focusing experiment and retain its result as evidence."""
    messages = _messages(state, EXECUTOR_PROMPT)
    request = llm.bind_tools([focusing_tool], tool_choice="any").invoke(messages)
    if len(request.tool_calls) != 1:
        return {"dialogue": [{
            "role": "executor",
            "text": "No tool ran: expected exactly one tool call. Please retry."
        }]}

    call = request.tool_calls[0]
    if call["name"] != focusing_tool.name:
        return {"dialogue": [{
            "role": "executor", "text": "No tool ran: the requested tool is unavailable."
        }]}
    if not any(turn["role"] == "scientist" for turn in state["dialogue"]):
        return {"dialogue": [{
            "role": "executor", "text": "No tool ran: I need a scientist plan before focusing."
        }]}

    try:
        result = focusing_tool.invoke(call["args"])
    except Exception as error:
        result = {"ok": False, "error": f"{type(error).__name__}: {error}"}

    validation_error = None
    try:
        if not isinstance(result, dict):
            raise ValueError("The tool must return a dictionary.")
        if not isinstance(result.get("ok"), bool):
            raise ValueError("Focusing tool must return a boolean ok.")
        if result["ok"]:
            required = {
                "experiment_id", "settings", "initial_enhancement",
                "final_enhancement", "paths",
            }
            if not required <= result.keys():
                raise ValueError("Successful focusing result is missing required fields.")
            if not isinstance(result["experiment_id"], str) or not result["experiment_id"]:
                raise ValueError("A successful experiment needs an experiment ID.")
            if not isinstance(result["settings"], dict) or not result["settings"]:
                raise ValueError("A successful experiment needs its settings.")
            paths = result["paths"]
            if (
                not isinstance(paths, dict)
                or not {"initial_camera", "final_camera", "final_pattern",
                        "results_json", "experiments_csv"} <= paths.keys()
                or not all(isinstance(path, str) and path for path in paths.values())
            ):
                raise ValueError("A successful experiment needs saved output paths.")
            for name in ("initial_enhancement", "final_enhancement"):
                value = result[name]
                if (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(value)
                ):
                    raise ValueError(f"{name} must be a finite number.")
        elif not isinstance(result.get("error"), str) or not result["error"]:
            raise ValueError("A failed focusing result needs an error message.")
    except (ValueError, OverflowError) as error:
        validation_error = str(error)

    recorded = {"tool_name": call["name"], "tool_args": call["args"], "result": result}
    if validation_error:
        recorded["validation_error"] = validation_error
    evidence = json.dumps(result, ensure_ascii=False)
    # Preserve the original AIMessage, including its encrypted reasoning content.
    tool_message = evidence
    if validation_error:
        tool_message += "\nResult validation error: " + validation_error
    messages.extend([request, ToolMessage(content=tool_message, tool_call_id=call["id"])])
    summary = _summarize(llm, [focusing_tool], messages)
    if validation_error:
        summary += "\nResult validation error: " + validation_error
    return {
        "dialogue": [{
            "role": "executor",
            "text": summary + f"\n\nTool: {call['name']}\nRaw tool result:\n" + evidence,
        }],
        "execution_results": [recorded],
    }


def create_information_tool() -> BaseTool:
    """Create the tool that hands the report-and-print turn back to the orchestrator."""

    @tool
    def request_information(missing: str, human_request: str) -> dict:
        """Call this in the report-and-print turn when a human change request needs
        information or analysis you do not have, or changes a number, claim or the
        verdict. The turn goes back to the orchestrator, which will gather it.

        missing: What information is needed and from whom, for example
            "a third experiment with macropixel_size 8" or "the scientist's revised verdict".
        human_request: The human's change request, quoted.
        """
        return {"ok": True}

    return request_information


def inspect_report(path: str | Path) -> dict:
    """Page count and SHA-256 of the report; the hash ties an approval to one version."""
    path = Path(path)
    if not path.is_file():
        return {"path": str(path), "exists": False, "pages": 0, "sha256": None}
    data = path.read_bytes()
    report = {
        "path": str(path), "exists": True, "pages": 0,
        "sha256": hashlib.sha256(data).hexdigest(),
    }
    try:
        from pypdf import PdfReader

        report["pages"] = len(PdfReader(path).pages)
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
    return report


def _parse_tool_output(raw) -> dict:
    """Printer tools return a JSON string; keep unreadable output as an error."""
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, list):
        # MCP adapters may return content blocks instead of a plain string.
        raw = "".join(
            block.get("text", "") if isinstance(block, dict) else str(block) for block in raw
        )
    try:
        parsed = json.loads(raw)
    except (TypeError, ValueError):
        return {"ok": False, "error": str(raw)}
    return parsed if isinstance(parsed, dict) else {"ok": False, "error": str(raw)}


def _approval_key(report_sha256: str, args: dict) -> dict:
    """Normalised approval record; graph.py stores state["approval"] with this too."""
    return {
        "report_sha256": report_sha256,
        "printer": str(args.get("printer") or ""),
        "copies": int(args.get("copies") or 1),
    }


def _latest_job_state(state: dict, job_id: str | None) -> str | None:
    """State from the most recent recorded job_status for this job, if any."""
    if not job_id:
        return None
    for record in reversed(state.get("execution_results", [])):
        if record.get("tool_name") != "job_status":
            continue
        result = record.get("result")
        record_id = (record.get("tool_args") or {}).get("job_id")
        if isinstance(result, dict):
            record_id = result.get("job_id") or record_id
        if record_id == job_id:
            return str(result.get("state", "")).lower() if isinstance(result, dict) else None
    return None


def report_and_print(
    state: dict, llm: BaseChatModel, tools: list[BaseTool], run_dir: str | Path
) -> dict:
    """Make one report or printer tool call; the graph routes on the returned outcome."""
    report_path = (Path(run_dir) / "report.pdf").resolve()
    tool_map = {item.name: item for item in tools}
    context = {
        "report_path": str(report_path),
        "report": state.get("report"),
        "approval": state.get("approval"),
        "active_job_id": state.get("active_job_id"),
    }
    messages = _messages(
        state, REPORT_AND_PRINT_PROMPT,
        extra="Report-and-print context:\n" + json.dumps(context, ensure_ascii=False),
    )
    request = llm.bind_tools(tools, tool_choice="any").invoke(messages)
    if len(request.tool_calls) != 1:
        return {"outcome": "no_tool", "dialogue": [{
            "role": "executor",
            "text": "No tool ran: expected exactly one tool call. Please retry."
        }]}

    call = request.tool_calls[0]
    name, args = call["name"], dict(call["args"])
    if name not in tool_map:
        return {"outcome": "no_tool", "dialogue": [{
            "role": "executor", "text": f"No tool ran: the tool {name!r} is unavailable."
        }]}

    if name == "request_information":
        return {"outcome": "needs_information", "dialogue": [{
            "role": "executor",
            "text": f"Needs information: {args.get('missing', '')}. "
                    f"Human request: {args.get('human_request', '')}",
        }]}

    report = None
    if name == "submit_print":
        pdf_path = Path(str(args.get("pdf_path", "")))
        if not pdf_path.is_absolute():
            pdf_path = Path(run_dir) / pdf_path
        if pdf_path.resolve() != report_path:
            return {"outcome": "no_tool", "dialogue": [{
                "role": "executor",
                "text": f"No tool ran: submit_print must use the run's report {report_path}.",
            }]}
        # Inspect the file itself, so an approval always matches the bytes that print.
        report = inspect_report(report_path)
        if not report["exists"] or report["pages"] != 1:
            return {"outcome": "no_tool", "report": report, "dialogue": [{
                "role": "executor",
                "text": "No tool ran: report.pdf must exist and have exactly one page "
                        f"(found {report['pages']}). Rebuild it first.",
            }]}
        try:
            approved = _approval_key(report["sha256"], args)
        except (TypeError, ValueError):
            return {"outcome": "no_tool", "dialogue": [{
                "role": "executor",
                "text": f"No tool ran: copies must be a whole number, not {args.get('copies')!r}.",
            }]}
        args.update(pdf_path=str(report_path), printer=approved["printer"], copies=approved["copies"])
        if state.get("approval") != approved:
            printer = approved["printer"] or "the default printer"
            return {
                "outcome": "needs_print_approval",
                "report": report,
                "pending_print": {"args": args, "report_sha256": approved["report_sha256"]},
                "dialogue": [{
                    "role": "executor",
                    "text": f"Requesting human approval to print {report_path} on "
                            f"{printer}, {approved['copies']} copies.",
                }],
            }
        job_id = state.get("active_job_id")
        job_state = _latest_job_state(state, job_id)
        if job_id and job_state not in LOST_JOB_STATES:
            return {"outcome": "no_tool", "dialogue": [{
                "role": "executor",
                "text": f"No tool ran: this report was already submitted as job {job_id} "
                        f"(last job_status: {job_state or 'not checked yet'}). "
                        "Call job_status for that job instead of printing again.",
            }]}

    try:
        raw = tool_map[name].invoke(args)
    except Exception as error:
        if name == "write_and_run":
            raw = {"exit_code": None, "stdout": "", "stderr": traceback.format_exc()}
        else:
            raw = {"ok": False, "error": f"{type(error).__name__}: {error}"}
    result = raw if name == "write_and_run" and isinstance(raw, dict) else _parse_tool_output(raw)

    evidence = json.dumps(result, ensure_ascii=False)
    tool_message = evidence
    if name == "write_and_run":
        report = inspect_report(report_path)
        if not report["exists"]:
            tool_message += f"\nReport check: {report_path} was not created."
        elif report["pages"] != 1:
            tool_message += f"\nReport check: report.pdf has {report['pages']} pages; it must have exactly one."

    recorded = {"tool_name": name, "tool_args": args, "result": result}
    # Preserve the original AIMessage, including its encrypted reasoning content.
    messages.extend([request, ToolMessage(content=tool_message, tool_call_id=call["id"])])
    summary = _summarize(llm, tools, messages)
    update = {
        "outcome": "tool_ran",
        "tool_name": name,
        "result": result,
        "dialogue": [{
            "role": "executor",
            "text": summary + f"\n\nTool: {name}\nRaw tool result:\n" + tool_message,
        }],
        "execution_results": [recorded],
    }
    if report is not None:
        update["report"] = report
    if name == "submit_print" and result.get("ok") is not False and result.get("job_id"):
        update["active_job_id"] = result["job_id"]
    return update
