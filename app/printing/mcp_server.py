"""MCP server for the office printer: exposes the connector's controls as tools.

Run over stdio as `python -m app.printing.mcp_server`; app.printing.tools starts it and
gives its tools to the executor. Every tool calls app.printing.connector, which is
the only code that talks to the printer, and returns the connector's result with
an "ok" field added, or {"ok": false, "error": ...}.

Tools: list_printers, printer_status, submit_print (the one-page report),
submit_trial_print (the trial sheets from trial_sheet.py, printed at 100% size in
grayscale), cancel_job (only when the human skips printing) and job_status.

Environment:
  PRINTER_MODE=fake      use the connector's fake printer Office_Printer.
  PRINTER_ALLOWED=A,B    only these printers may be read or printed on (empty: all).

Human approval is not checked here: the graph asks the human before the
executor's print is carried out.
"""
import hashlib
import json
import os
import re
from pathlib import Path

from mcp.server.fastmcp import FastMCP
from pypdf import PdfReader

from app.printing import connector

mcp = FastMCP("office-printer")
TRIAL_PDF_NAME = re.compile(r"^trial_r\d+_[A-Za-z0-9]{8}\.pdf$")   # as written by trial_sheet.py


def _call(function, *args):
    """Call one connector function and turn its answer into a tool result."""
    try:
        result = function(*args)
    except connector.PrinterError as error:
        return {"ok": False, "error": str(error)}
    if isinstance(result, list):
        return {"ok": True, "printers": result}
    return {"ok": True, **result}


def _allowed(printer):
    """True if PRINTER_ALLOWED is empty or names this printer."""
    allowed = [name.strip() for name in os.environ.get("PRINTER_ALLOWED", "").split(",") if name.strip()]
    return not allowed or printer in allowed


def _not_allowed(printer):
    return {"ok": False, "error": f"printer {printer!r} is not allowed; allowed: "
                                  f"{os.environ.get('PRINTER_ALLOWED')}"}


def _pdf_pages(pdf):
    """(page count, None) for a readable PDF file, or (None, error result)."""
    if not pdf.is_file():
        return None, {"ok": False, "error": f"file not found: {pdf}"}
    with pdf.open("rb") as file:
        if file.read(5) != b"%PDF-":
            return None, {"ok": False, "error": f"not a PDF file: {pdf}"}
    try:
        return len(PdfReader(pdf).pages), None
    except Exception as error:
        return None, {"ok": False, "error": f"cannot read the PDF: {type(error).__name__}: {error}"}


def _printer_ready(printer):
    """None if the printer may be used, else the error result (not allowed, unreadable, offline)."""
    if not _allowed(printer):
        return _not_allowed(printer)
    status = _call(connector.printer_status, printer)
    if not status["ok"]:
        return status
    if status["state"] == "offline":
        return {"ok": False, "error": f"printer {printer} is offline: {status['message']}"}
    return None


@mcp.tool()
def list_printers() -> dict:
    """Reading: the names of the printers this computer knows.

    Returns {"ok": true, "printers": [...]}. Use it to find the office printer's
    exact name when the context does not give it.
    """
    return _call(connector.list_printers)


@mcp.tool()
def printer_status(printer: str) -> dict:
    """Reading: the current state of one printer.

    printer: the exact printer name.
    Returns {"ok": true, "printer", "state", "message"}, where state is one of
    idle, printing, stopped, offline, out_of_paper, paper_jam, error, unknown.
    Check it before submit_print.
    """
    if not _allowed(printer):
        return _not_allowed(printer)
    return _call(connector.printer_status, printer)


@mcp.tool()
def submit_print(pdf_path: str, printer: str, copies: int = 1) -> dict:
    """Action: print a one-page PDF report and return the job id.

    Only call this after the human has approved printing; then read job_status
    with the returned job_id until the job is done.
    pdf_path: path of the one-page PDF report.
    printer: the exact printer name.
    copies: must be 1.
    Returns {"ok": true, "job_id", "printer", "file", "tracked"} or {"ok": false, "error"}.
    tracked false (with a job_id containing "-untracked-" and a message) means the job was
    never seen in the queue: do NOT print again; the human must check the printer tray.
    """
    if copies != 1:
        return {"ok": False, "error": f"only 1 copy can be printed, not {copies}"}
    pdf = Path(pdf_path)
    pages, error = _pdf_pages(pdf)
    if error:
        return error
    if pages != 1:
        return {"ok": False, "error": f"the PDF must have exactly 1 page, it has {pages}"}
    error = _printer_ready(printer)
    if error:
        return error
    return _call(connector.submit_print, str(pdf), printer)


@mcp.tool()
def submit_trial_print(pdf_path: str, printer: str) -> dict:
    """Action: print a grayscale trial sheet made by trial_sheet.py and return the job id.

    Only for trial sheets (trial_r<round>_<id>.pdf with its trial_r<round>_<id>_manifest.json
    next to it), and only call it after the human has approved printing. The report is
    never printed with this tool: the report must use submit_print.
    Prints 1 copy, A4, grayscale, one-sided, at 100% size (no fit-to-page), because the
    sheet is measured in mm on the scan. Then read job_status with the returned job_id.
    pdf_path: path of the trial sheet PDF (2 pages for 4 experiments).
    printer: the exact printer name.
    Returns {"ok": true, "job_id", "printer", "file", "tracked"} or {"ok": false, "error"}.
    tracked false (with a job_id containing "-untracked-" and a message) means the job was
    never seen in the queue: do NOT print again; the human must check the printer tray.
    """
    pdf = Path(pdf_path)
    if not TRIAL_PDF_NAME.match(pdf.name):
        return {"ok": False, "error": f"not a trial sheet: {pdf.name} must be named trial_r<round>_<8 chars>.pdf"}
    pages, error = _pdf_pages(pdf)
    if error:
        return error
    manifest_path = pdf.with_name(pdf.stem + "_manifest.json")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected_pages, expected_sha256 = manifest["pdf"]["pages"], manifest["pdf"]["sha256"]
    except (OSError, ValueError, KeyError, TypeError) as error:
        return {"ok": False, "error": f"no usable trial manifest {manifest_path.name} next to the PDF: {error}"}
    if pages < 2:
        return {"ok": False, "error": f"a trial sheet has at least 2 pages, this PDF has {pages}; "
                                      "a one-page report must use submit_print"}
    if pages != expected_pages:
        return {"ok": False, "error": f"the PDF has {pages} pages but its manifest says {expected_pages}"}
    if hashlib.sha256(pdf.read_bytes()).hexdigest() != expected_sha256:
        return {"ok": False, "error": "the PDF is not the one its manifest describes (sha256 differs)"}
    error = _printer_ready(printer)
    if error:
        return error
    return _call(connector.submit_trial_print, str(pdf), printer)


@mcp.tool()
def cancel_job(job_id: str) -> dict:
    """Action: cancel a print job that has not printed yet.

    Only used when the human skips printing; never cancel a job the human wants printed.
    job_id: the job_id returned by submit_print or submit_trial_print.
    Returns {"ok": true, "job_id", "state", "message"}: state is failed with message
    "cancelled", or done if the job was already printed, or uncertain (with confirmed
    false and reason) if the job cannot be tracked and so could not be cancelled.
    {"ok": false, "error"} for a job id that was never issued.
    """
    return _call(connector.cancel_job, job_id)


@mcp.tool()
def job_status(job_id: str) -> dict:
    """Reading: the state of one print job.

    job_id: the job_id returned by submit_print or submit_trial_print.
    Returns {"ok": true, "job_id", "state", "confirmed", "message"}, where state is one
    of waiting, printing, done, failed, uncertain. done always means a confirmed print.
    uncertain (confirmed false) means nobody can tell whether the page printed; its
    "reason" is untracked_submission, left_queue or not_found. Never treat uncertain as
    printed and never print again on your own: the human must check the printer tray.
    A waiting job that is stuck on a printer problem also has a "problem" field, such
    as out_of_paper.
    """
    return _call(connector.job_status, job_id)


if __name__ == "__main__":
    mcp.run()
