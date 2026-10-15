"""MCP server for the office printer: exposes the connector's controls as tools.

Run over stdio as `python -m app.printing.mcp_server`; app.printing.tools starts it and
gives its tools to the executor. Every tool calls app.printing.connector, which is
the only code that talks to the printer, and returns the connector's result with
an "ok" field added, or {"ok": false, "error": ...}.

Environment:
  PRINTER_MODE=fake      use the connector's fake printer Office_Printer.
  PRINTER_ALLOWED=A,B    only these printers may be read or printed on (empty: all).

Human approval is not checked here: the graph asks the human before the
executor's print is carried out.
"""
import os
from pathlib import Path

from mcp.server.fastmcp import FastMCP
from pypdf import PdfReader

from app.printing import connector

mcp = FastMCP("office-printer")


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
    Returns {"ok": true, "job_id", "printer", "file"} or {"ok": false, "error"}.
    """
    if copies != 1:
        return {"ok": False, "error": f"only 1 copy can be printed, not {copies}"}
    pdf = Path(pdf_path)
    if not pdf.is_file():
        return {"ok": False, "error": f"file not found: {pdf}"}
    with pdf.open("rb") as file:
        if file.read(5) != b"%PDF-":
            return {"ok": False, "error": f"not a PDF file: {pdf}"}
    try:
        pages = len(PdfReader(pdf).pages)
    except Exception as error:
        return {"ok": False, "error": f"cannot read the PDF: {type(error).__name__}: {error}"}
    if pages != 1:
        return {"ok": False, "error": f"the PDF must have exactly 1 page, it has {pages}"}
    if not _allowed(printer):
        return _not_allowed(printer)
    status = _call(connector.printer_status, printer)
    if not status["ok"]:
        return status
    if status["state"] == "offline":
        return {"ok": False, "error": f"printer {printer} is offline: {status['message']}"}
    return _call(connector.submit_print, str(pdf), printer)


@mcp.tool()
def job_status(job_id: str) -> dict:
    """Reading: the state of one print job.

    job_id: the job_id returned by submit_print.
    Returns {"ok": true, "job_id", "state", "message"}, where state is one of
    waiting, printing, done, failed, unknown. A waiting job that is stuck on a
    printer problem also has a "problem" field, such as out_of_paper.
    """
    return _call(connector.job_status, job_id)


if __name__ == "__main__":
    mcp.run()
