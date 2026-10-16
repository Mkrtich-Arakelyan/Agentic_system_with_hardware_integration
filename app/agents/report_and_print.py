"""Existing report-and-print executor mode and its report/approval helpers."""

import hashlib
import json
from pathlib import Path
import traceback

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import ToolMessage
from langchain_core.tools import BaseTool, tool

from app.agents.common import _messages, _summarize


# Counts report_and_print passes since the last human answer;
# app/workflow/graph.py resets the counter to 0 after every human answer.
MAX_HARDWARE_PASSES = 12


# Legacy state labels retained for existing public exports. The graph, never
# these labels alone, authorizes another copy through fresh human approval.
LOST_JOB_STATES = {"failed", "cancelled", "canceled", "error", "aborted", "unknown"}


REPORT_AND_PRINT_PROMPT = """
You are the executor in the report-and-print turn. Make exactly one tool call per
pass. After each call its result is added to the dialogue and you are called again.

Steps:
1. If there is no up-to-date one-page report.pdf, call write_and_run with a
   complete script that reads experiments.csv and the results.json files named in
   the recorded results (never type numbers into the script), embeds the supplied
   final tuned figure, adds the title, the explanation from general,
   the fairness, performance-vs-cost and next-experiment sentences and the
   verdict, and saves report.pdf in the current folder as exactly one A4 portrait
   page (matplotlib savefig or PdfPages is enough). Use a fresh script filename.
2. Call printer_status for the office printer named in the context.
3. Call submit_print with pdf_path set to the report path given below. The graph
   asks the human for approval before it really prints.
4. Call job_status with the workflow's current job id. The graph handles done,
   failed, uncertain, and unconfirmed readings and pauses for human recovery.

After a human turn starting with "Change request for the report:":
- if it only changes presentation (layout, titles, wording that does not change
  a claim, colours, sizes), write a revised
  script with a new filename, rebuild report.pdf and call submit_print again;
- if it changes a number, a claim or the verdict, or needs data or analysis you
  do not have, call request_information with missing (what is needed and from
  whom) and human_request (the human's words). Never change a claim just because
  the human asked; claims must come from recorded results.

After a printer problem and a human "retry", check the tracked current job when
one is supplied. Only request another print after the graph resolves the previous
attempt and clears its job ID; every new copy needs fresh approval. If a tool returns an
error, read it and fix the cause on the next pass, for example a report with
two pages. Never invent results.

The report-and-print context supplies the frozen experiment_selection, including
the four selected IDs, reasons and saved paths. Read those results_json paths
and validate that their experiment_id values match the selection. Base the
report on those four experiments; any single-experiment graph must use one of
   them. Do not independently select from the CSV or change the selected IDs during
layout revisions or printer retries. A request to change the selection requires
request_information for scientific review, rather than silently substituting IDs.

The context supplies final_figure_path and final_figure_metadata, rendered by
the application from the completed calibration, original Initial/Final arrays,
fixed normalization limits, and final mappings. Embed that EXACT PNG as the
report's figure. Do not redraw it, replace it with camera previews, recompute
grayscale mappings, enhance its contrast, or change its pixels. Insert the PNG
directly with PyMuPDF page.insert_image, or use matplotlib imshow with
interpolation='none' to preserve its original pixels in the PDF. Keep its aspect
ratio, fully inside the one A4 portrait page, at least 25% of page width and
height. The application checks the actual embedded pixels before approval.
Scientific findings and selection reasons come from the scientific evidence;
gamma/endpoints/steps are presentation and print adjustments, not new findings.
submit_print is always an approval REQUEST here, never an immediate submission.
An uncertain/untracked job, unconfirmed reading, or submission exception needs
human resolution. Never resubmit an uncertain print or claim it printed.
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
        if not report["exists"] or report["pages"] != 1 or not report.get("figure_embedded") or report.get("error"):
            return {"outcome": "no_tool", "report": report, "dialogue": [{
                "role": "executor",
                "text": "No tool ran: report.pdf must exist and have exactly one page "
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
        if name == "write_and_run":
            raw = {"exit_code": None, "stdout": "", "stderr": traceback.format_exc()}
        else:
            raw = {"ok": False, "error": f"{type(error).__name__}: {error}"}
    result = raw if name == "write_and_run" and isinstance(raw, dict) else _parse_tool_output(raw)

    evidence = json.dumps(result, ensure_ascii=False)
    tool_message = evidence
    if name == "write_and_run":
        from app.workflow.reporting import inspect_tuned_report
        figure = state.get("final_figure_metadata")
        report = inspect_tuned_report(report_path, figure) if figure else inspect_report(report_path)
        if not report["exists"]:
            tool_message += f"\nReport check: {report_path} was not created."
        elif report["pages"] != 1:
            tool_message += f"\nReport check: report.pdf has {report['pages']} pages; it must have exactly one."
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
