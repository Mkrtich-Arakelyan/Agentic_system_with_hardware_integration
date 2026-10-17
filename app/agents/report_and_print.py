"""Existing report-and-print executor mode and its report/approval helpers."""

import hashlib
import json
from pathlib import Path
import traceback

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import ToolMessage
from langchain_core.tools import BaseTool, tool

from app.agents.common import _messages, _summarize
from app.report_policy import REPORT_ALLOWED_PAGES


# Counts report_and_print passes since the last human answer;
# app/workflow/graph.py resets the counter to 0 after every human answer.
MAX_HARDWARE_PASSES = 12
MAX_SUBMISSION_REQUEST_PASSES = 2


# Legacy state labels retained for existing public exports. The graph, never
# these labels alone, authorizes another copy through fresh human approval.
LOST_JOB_STATES = {"failed", "cancelled", "canceled", "error", "aborted", "unknown"}


REPORT_AND_PRINT_PROMPT = """
You are the executor in the report-and-print turn. Make exactly one tool call.
Use build_final_report to generate or revise the report through the reusable Python
renderer. Default: three A4 pages; two pages are permitted only if readable.
Page 1: scientific question, controls, comparison chart/table, evidence-supported
interpretation and next experiment. Page 2: supplied scientific Initial/Final image
using final accepted tuning mappings, frozen data and normalization, and settings.
Page 3: detail preservation and CNR progress from existing accepted scan evidence.
No new experiments, tuning rounds, scans, or validation prints are required.
Do not change scientific numbers, frozen selection or mappings during revisions.
When report_revision is pending, translate its exact instructions into concrete
findings/presentation edits through build_final_report. Fields include title,
question, setup, interpretation, limitation_next, closing_note. Omitted text retains
prior edits. Do not request submission until generation has applied the revision.
Unsupported edits require clarification, never acceptance of the old PDF.
After validation, read printer_status then emit submit_print to REQUEST the human
approval interrupt. No conversational approval is needed to request that interrupt.
The graph alone submits after explicit approval bound to the complete PDF hash,
printer and arguments. Retry and executor prose are never approval. Use job_status
only for the active job. Uncertain outcomes require human confirmation; never retry
a submission automatically. Never claim semantic instruction compliance is proven;
the human must review the revised PDF. Do not invent improvements or measurements.
"""


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
    """Normalised approval record; app/workflow/graph.py stores state["approval"] with this too."""
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


def submission_request_ready(state: dict) -> bool:
    """Use validated artifact and structured readings, never dialogue as approval."""
    report = state.get("report") or {}
    if (state.get("active_job_id") or state.get("report_revision_pending")
            or not report.get("exists") or report.get("pages") not in REPORT_ALLOWED_PAGES
            or not report.get("figure_embedded") or report.get("error")):
        return False
    for record in reversed(state.get("execution_results", [])):
        name = record.get("tool_name")
        if name in ("write_and_run", "build_final_report", "submit_print"):
            return False  # Check readiness again after changing a PDF or submitting.
        if name == "printer_status":
            result = record.get("result") or {}
            printer = (record.get("tool_args") or {}).get("printer")
            return (result.get("ok") is True and result.get("state") == "idle"
                    and bool(printer) and result.get("printer", printer) == printer
                    and (not state.get("tuning_printer") or printer == state["tuning_printer"])
                    and not any(result.get(k) and str(result[k]).lower() != "none"
                                for k in ("problem", "problems", "alerts")))
    return False


def report_and_print(
    state: dict, llm: BaseChatModel, tools: list[BaseTool], run_dir: str | Path
) -> dict:
    """Make one report or printer tool call; the graph routes on the returned outcome."""
    report_path = (Path(run_dir) / "report.pdf").resolve()
    tool_map = {item.name: item for item in tools}
    context = {
        "report_path": str(report_path),
        "experiment_selection": state.get("experiment_selection"),
        "report": state.get("report"),
        "approval": state.get("approval"),
        "active_job_id": state.get("active_job_id"),
        "final_figure_path": state.get("final_figure_path"),
        "final_figure_metadata": state.get("final_figure_metadata"),
        "phase": "request_submission" if submission_request_ready(state) else "prepare_or_track",
        "report_revision_pending": state.get("report_revision_pending", False),
        "report_revision": state.get("report_revision"),
    }
    try:
        from app.workflow.report_document import current_report_inputs
        previous = current_report_inputs(run_dir)
        context.update(current_report_text=previous.get("findings"), current_presentation=previous.get("presentation"))
    except (OSError, ValueError):
        context["current_report_text"] = "Prior saved text unavailable; retain evidence and request clarification if needed."
    messages = _messages(
        state, REPORT_AND_PRINT_PROMPT,
        extra="Report-and-print context:\n" + json.dumps(context, ensure_ascii=False),
    )
    requesting = submission_request_ready(state)
    revising = bool(state.get("report_revision_pending") or state.get("report_revision"))
    available = [tool_map["build_final_report"]] if revising else [tool_map["submit_print"]] if requesting else tools
    # ChatOpenAI accepts a tool name and translates it for the Responses API.
    # The model still supplies the action and arguments; we never fabricate a call.
    options = {"tool_choice": "build_final_report", "parallel_tool_calls": False} if revising else {"tool_choice": "submit_print", "parallel_tool_calls": False} if requesting else {"tool_choice": "any"}
    request = llm.bind_tools(available, **options).invoke(messages)
    if len(request.tool_calls) != 1:
        return {"outcome": "no_tool", "dialogue": [{
            "role": "executor",
            "text": "No tool ran: expected exactly one tool call. Please retry."
        }]}

    call = request.tool_calls[0]
    name, args = call["name"], dict(call["args"])
    if revising:
        if name != "build_final_report":
            return {"outcome": "no_tool", "dialogue": [{"role": "system", "text": "Revision pending; request build_final_report. Printing is blocked."}]}
        revision = state.get("report_revision") or {}
        args.update(revision_id=revision.get("id", ""), revision_instructions=revision.get("instructions", ""))
    if requesting and name != "submit_print":
        return {"outcome": "no_tool", "dialogue": [{"role": "system", "text":
            "Submission-request phase stalled: emit submit_print to open human approval. "
            "No printer call ran; conversational approval is not required to request the interrupt."}]}
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

    if name == "job_status" and (not state.get("active_job_id") or args != {"job_id": state["active_job_id"]}):
        return {"outcome": "no_tool", "dialogue": [{"role": "executor",
                "text": "Read only the current report job ID supplied by the workflow."}]}

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
        from app.workflow.reporting import inspect_tuned_report
        figure = state.get("final_figure_metadata")
        if not figure:
            return {"outcome": "no_tool", "dialogue": [{"role": "executor", "text": "Validated final tuned figure is required."}]}
        report = inspect_tuned_report(report_path, figure)
        if not report["exists"] or report["pages"] not in REPORT_ALLOWED_PAGES or not report.get("figure_embedded") or report.get("error"):
            return {"outcome": "no_tool", "report": report, "dialogue": [{
                "role": "executor",
                "text": "No tool ran: report.pdf must exist and have 2 or 3 pages "
                        f"(found {report['pages']}) and embed the supplied tuned figure. {report.get('error', '')} Rebuild it first.",
            }]}
        try:
            if (not set(args) <= {"pdf_path", "printer", "copies"} or not isinstance(args.get("printer"), str)
                    or not args["printer"].strip() or type(args.get("copies", 1)) is not int or args.get("copies", 1) != 1):
                raise ValueError("An exact printer, one copy, and no extra submission arguments are required")
            approved = _approval_key(report["sha256"], args)
        except (TypeError, ValueError):
            return {"outcome": "no_tool", "dialogue": [{
                "role": "executor",
                "text": "No tool ran: supply an exact printer, one integer copy, and only the supported arguments.",
            }]}
        args.update(pdf_path=str(report_path), printer=approved["printer"], copies=approved["copies"])
        # Every submission is handled by the graph's persisted intent node.
        # Even a supplied approval never lets this model turn execute printing.
        return {
            "outcome": "needs_print_approval",
            "report": report,
            "pending_print": {"args": args, "report_sha256": approved["report_sha256"]},
            "dialogue": [{
                "role": "executor",
                "text": f"Requesting human approval to print {report_path} on "
                        f"{approved['printer']}, {approved['copies']} copies.",
            }],
        }

    try:
        raw = tool_map[name].invoke(args)
    except Exception as error:
        if name in ("write_and_run", "build_final_report"):
            raw = {"exit_code": None, "stdout": "", "stderr": traceback.format_exc()}
        else:
            raw = {"ok": False, "error": f"{type(error).__name__}: {error}"}
    result = raw if name == "write_and_run" and isinstance(raw, dict) else _parse_tool_output(raw)

    evidence = json.dumps(result, ensure_ascii=False)
    tool_message = evidence
    if name in ("write_and_run", "build_final_report"):
        from app.workflow.reporting import inspect_tuned_report
        figure = state.get("final_figure_metadata")
        report = inspect_tuned_report(report_path, figure) if figure else inspect_report(report_path)
        if not report["exists"]:
            tool_message += f"\nReport check: {report_path} was not created."
        elif report["pages"] not in REPORT_ALLOWED_PAGES:
            tool_message += f"\nReport check: report.pdf has {report['pages']} pages; it must have 2 or 3 pages."
        elif not report.get("figure_embedded"):
            tool_message += "\nReport check: " + str(report.get("error", "Final tuned PNG must be embedded unchanged."))

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
