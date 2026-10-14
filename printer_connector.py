"""Connector for the office printer: the only code that talks to the instrument.

Interface: the operating system's own print commands, run through subprocess.
  - Windows:       PowerShell Get-Printer / Get-PrintJob, and SumatraPDF -print-to
                   (if SUMATRA_PATH is set) or Start-Process -Verb PrintTo.
  - macOS / Linux: CUPS commands lp and lpstat.

Set PRINTER_MODE=fake to use a pretend printer called Office_Printer instead of real
commands. Set FAKE_PRINTER_PROBLEM=out_of_paper (or offline, paper_jam) to make the
fake printer report that problem.

Printer states: idle, printing, stopped, offline, out_of_paper, paper_jam, error, unknown.
Job states:     waiting, printing, done, failed, unknown.
Job ids look like "<printer>-<number>", e.g. Office_Printer-1.
"""
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

FAKE_PRINTER = "Office_Printer"
FAKE_JOBS_FILE = Path(os.environ.get("FAKE_PRINTER_FILE", Path(__file__).with_name("fake_printer_jobs.json")))
FAKE_PROBLEMS = ("out_of_paper", "offline", "paper_jam")
FAKE_QUEUE_SECONDS = 1.0   # a fake job waits this long, then prints
FAKE_PRINT_SECONDS = 3.0   # a fake job is done this long after submission


class PrinterError(Exception):
    """A print command failed or the printer gave an answer we cannot use."""


def _fake():
    return os.environ.get("PRINTER_MODE", "").lower() == "fake"


def _windows():
    return sys.platform.startswith("win")


def _run(cmd):
    """Run one OS command and return its output, or raise PrinterError."""
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    except FileNotFoundError:
        raise PrinterError(f"command not found: {cmd[0]}")
    except subprocess.TimeoutExpired:
        raise PrinterError(f"command timed out: {cmd[0]}")
    if result.returncode != 0:
        raise PrinterError(f"{cmd[0]} failed: {(result.stderr or result.stdout).strip()}")
    return result.stdout.strip()


def _ps(script):
    return _run(["powershell", "-NoProfile", "-Command", script])


def _quote(text):
    return "'" + text.replace("'", "''") + "'"   # PowerShell single-quoted string


def _printer_state(text):
    """Turn the OS's description of a printer into one of our printer states."""
    t = text.lower()
    for state, words in [("paper_jam", ("jam",)),
                         ("out_of_paper", ("paperout", "paper out", "media-empty", "media empty", "out of paper")),
                         ("offline", ("offline", "not connected", "unreachable", "notavailable")),
                         ("error", ("error",)),
                         ("stopped", ("paused", "disabled", "stopped")),
                         ("printing", ("printing", "processing", "busy")),
                         ("idle", ("idle", "normal"))]:
        if any(w in t for w in words):
            return state
    return "unknown"


def _job_state(text):
    """Turn a Windows job status into one of our job states."""
    t = text.lower()
    for state, words in [("failed", ("error", "deleted", "aborted", "cancel")),
                         ("done", ("printed", "complete")),
                         ("printing", ("printing", "spooling", "processing"))]:
        if any(w in t for w in words):
            return state
    return "waiting"


# ---------- fake printer ----------

def _fake_load():
    if FAKE_JOBS_FILE.exists():
        return json.loads(FAKE_JOBS_FILE.read_text())
    return {"next_id": 1, "jobs": {}}


def _fake_job_state(job, problem):
    """A job sent during a problem never prints; a problem that starts later stops unfinished jobs."""
    age = time.time() - job["submitted"]
    if job.get("problem"):
        return "failed"
    if age >= FAKE_PRINT_SECONDS:
        return "done"
    if problem:
        return "failed"
    return "waiting" if age < FAKE_QUEUE_SECONDS else "printing"


def _fake_problem():
    problem = os.environ.get("FAKE_PRINTER_PROBLEM", "").lower()
    return problem if problem in FAKE_PROBLEMS else None


def _fake_check(printer):
    if printer != FAKE_PRINTER:
        raise PrinterError(f"unknown printer: {printer} (the fake printer is {FAKE_PRINTER})")


# ---------- the four connector functions ----------

def list_printers():
    """Return the names of the printers this computer knows."""
    if _fake():
        return [FAKE_PRINTER]
    if _windows():
        out = _ps("Get-Printer | Select-Object -ExpandProperty Name")
    else:
        out = "\n".join(line.split()[1] for line in _run(["lpstat", "-p"]).splitlines()
                        if line.startswith("printer "))
    return [name.strip() for name in out.splitlines() if name.strip()]


def printer_status(printer):
    """Return {"printer", "state", "message"} for one printer."""
    if _fake():
        _fake_check(printer)
        problem = _fake_problem()
        busy = any(_fake_job_state(j, problem) == "printing" for j in _fake_load()["jobs"].values())
        state = problem or ("printing" if busy else "idle")
        return {"printer": printer, "state": state, "message": f"fake printer is {state}"}
    if _windows():
        out = _ps(f"Get-Printer -Name {_quote(printer)} | Select-Object -ExpandProperty PrinterStatus")
    else:
        out = _run(["lpstat", "-p", printer])
    return {"printer": printer, "state": _printer_state(out.replace(printer, "")), "message": out}


def submit_print(pdf_path, printer):
    """Send a PDF to the printer. Return {"job_id", "printer", "file"}."""
    pdf = Path(pdf_path).resolve()
    if not pdf.is_file():
        raise PrinterError(f"file not found: {pdf}")
    if _fake():
        _fake_check(printer)
        data = _fake_load()
        job_id = f"{printer}-{data['next_id']}"
        data["next_id"] += 1
        data["jobs"][job_id] = {"file": str(pdf), "submitted": time.time(), "problem": _fake_problem()}
        FAKE_JOBS_FILE.write_text(json.dumps(data, indent=2))
    elif _windows():
        ids = f"Get-PrintJob -PrinterName {_quote(printer)} | Select-Object -ExpandProperty Id"
        before = set(_ps(ids).split())
        if os.environ.get("SUMATRA_PATH"):
            _run([os.environ["SUMATRA_PATH"], "-print-to", printer, "-silent", str(pdf)])
        else:
            _ps(f"Start-Process -FilePath {_quote(str(pdf))} -Verb PrintTo "
                f"-ArgumentList {_quote(chr(34) + printer + chr(34))} -WindowStyle Hidden")
        for _ in range(30):   # Windows gives no job id back, so wait for a new job in the queue
            new = set(_ps(ids).split()) - before
            if new:
                break
            time.sleep(0.5)
        else:
            raise PrinterError(f"print command ran, but no new job appeared on {printer}")
        job_id = f"{printer}-{max(new, key=int)}"
    else:
        match = re.search(r"request id is (\S+)", _run(["lp", "-d", printer, str(pdf)]))
        if not match:
            raise PrinterError("lp did not return a job id")
        job_id = match.group(1)
    return {"job_id": job_id, "printer": printer, "file": str(pdf)}


def job_status(job_id):
    """Return {"job_id", "state", "message"}: state is waiting, printing, done, failed or unknown."""
    printer, _, number = job_id.rpartition("-")
    if _fake():
        job = _fake_load()["jobs"].get(job_id)
        if job is None:
            raise PrinterError(f"unknown job: {job_id}")
        problem = job.get("problem") or _fake_problem()
        state = _fake_job_state(job, problem)
        message = f"printer reports {problem}" if state == "failed" else f"fake job is {state}"
        return {"job_id": job_id, "state": state, "message": message}
    if _windows():
        out = _ps(f"$j = Get-PrintJob -PrinterName {_quote(printer)} | Where-Object Id -eq {int(number)}; "
                  "if ($j) { 'found: ' + $j.JobStatus } else { 'gone' }")
        if out == "gone":
            return {"job_id": job_id, "state": "done", "message": "job left the queue (assumed printed)"}
        return {"job_id": job_id, "state": _job_state(out), "message": out}
    if any(line.split()[0] == job_id for line in _run(["lpstat", "-o", printer]).splitlines()):
        status = printer_status(printer)
        state = "printing" if f"now printing {job_id}" in status["message"] else "waiting"
        return {"job_id": job_id, "state": state, "message": f"printer is {status['state']}"}
    if any(line.split()[0] == job_id for line in _run(["lpstat", "-W", "completed", "-o", printer]).splitlines()):
        return {"job_id": job_id, "state": "done", "message": "job is in the completed list"}
    return {"job_id": job_id, "state": "unknown", "message": "job not found in the queue or completed list"}


# ---------- command line ----------

USAGE = """usage:
  python printer_connector.py list
  python printer_connector.py status PRINTER
  python printer_connector.py print PDF_FILE PRINTER
  python printer_connector.py job JOB_ID"""


def main(argv):
    commands = {"list": (list_printers, 0), "status": (printer_status, 1),
                "print": (submit_print, 2), "job": (job_status, 1)}
    if not argv or argv[0] not in commands or len(argv) - 1 != commands[argv[0]][1]:
        print(USAGE)
        return 2
    function = commands[argv[0]][0]
    try:
        print(json.dumps(function(*argv[1:]), indent=2))
    except PrinterError as error:
        print(f"PrinterError: {error}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
