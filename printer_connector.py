"""Connector for the office printer: the only code that talks to the instrument.

Interface: the operating system's own print commands, run through subprocess.
  - Windows:       PowerShell Get-Printer / Get-PrintJob / Remove-PrintJob, and SumatraPDF
                   -print-to (if SUMATRA_PATH is set) or Start-Process -Verb PrintTo.
  - macOS / Linux: CUPS commands lp, lpstat and cancel.

Two ways to print:
  - submit_print:       the one-page report, with the system's default print settings.
  - submit_trial_print: the grayscale trial sheets from trial_sheet.py: one copy, A4,
                        grayscale, one-sided and at 100% size (no fit-to-page), because the
                        sheet's positions are measured in mm on the scan. On Windows this
                        needs SumatraPDF (SUMATRA_PATH), since the default path may scale.
cancel_job removes a job that has not printed yet.

Set PRINTER_MODE=fake to use a pretend printer called Office_Printer instead of real
commands. Set FAKE_PRINTER_PROBLEM=out_of_paper (or offline, paper_jam) to make the
fake printer report that problem.

Printer states: idle, printing, stopped, offline, out_of_paper, paper_jam, error, unknown.
Job states:     waiting, printing, done, failed, unknown.
                A waiting job that is stuck on a printer problem also has a "problem" field
                (for example out_of_paper). A cancelled job is failed, with message "cancelled".
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

# Trial sheets: measured in mm, so 1 copy, A4, gray, one-sided and never scaled.
TRIAL_LP_OPTIONS = ["-n", "1", "-o", "media=A4", "-o", "ColorModel=Gray", "-o", "print-color-mode=monochrome",
                    "-o", "sides=one-sided", "-o", "print-scaling=none", "-o", "nofit-to-page"]
TRIAL_SUMATRA_SETTINGS = "noscale,monochrome,simplex,paper=A4,1x"


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
                         ("offline", ("offline", "not connected", "unreachable", "notavailable",
                                      "unable to connect", "unable to locate")),
                         ("error", ("error", "dooropen", "door open", "notoner", "no toner",
                                    "userintervention", "user intervention",
                                    "paperproblem", "paper problem")),
                         ("stopped", ("paused", "disabled", "stopped")),
                         ("printing", ("printing", "processing", "busy")),
                         ("idle", ("idle", "normal"))]:
        if any(w in t for w in words):
            return state
    return "unknown"


def _job_state(text):
    """Turn a Windows job status into (job state, printer problem or None).

    A job still in the queue has not failed: a printer problem leaves it waiting,
    and Windows resumes it once the problem is fixed.
    """
    t = text.lower()
    if any(w in t for w in ("deleted", "deleting", "aborted", "cancel")):
        return "failed", None
    if any(w in t for w in ("printed", "complete")):
        return "done", None
    for problem, words in [("out_of_paper", ("paperout", "paper out")),
                           ("paper_jam", ("jam",)),
                           ("offline", ("offline",)),
                           ("stopped", ("paused", "blocked", "userintervention", "user intervention")),
                           ("error", ("error",))]:
        if any(w in t for w in words):
            return "waiting", problem
    if any(w in t for w in ("printing", "spooling", "processing")):
        return "printing", None
    return "waiting", None


# ---------- fake printer ----------

def _fake_load():
    if FAKE_JOBS_FILE.exists():
        return json.loads(FAKE_JOBS_FILE.read_text())
    return {"next_id": 1, "jobs": {}}


def _fake_job_state(job, problem):
    """A job sent during a problem never prints; a problem that starts later stops unfinished jobs."""
    age = time.time() - job["submitted"]
    if job.get("cancelled"):
        return "failed"
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


def _fake_submit(pdf, printer, **extra):
    """Add a job to the fake jobs file and return its id."""
    _fake_check(printer)
    data = _fake_load()
    job_id = f"{printer}-{data['next_id']}"
    data["next_id"] += 1
    data["jobs"][job_id] = {"file": str(pdf), "submitted": time.time(), "problem": _fake_problem(), **extra}
    FAKE_JOBS_FILE.write_text(json.dumps(data, indent=2))
    return job_id


# ---------- shared by the two print functions ----------

def _existing_pdf(pdf_path):
    pdf = Path(pdf_path).resolve()
    if not pdf.is_file():
        raise PrinterError(f"file not found: {pdf}")
    return pdf


def _windows_submit(printer, pdf, print_command):
    """Run print_command, then find the new job in the printer's queue (Windows gives no id back)."""
    ids = f"Get-PrintJob -PrinterName {_quote(printer)} | Select-Object -ExpandProperty Id"
    before = set(_ps(ids).split())
    print_command()
    for _ in range(30):
        new = set(_ps(ids).split()) - before
        if new:
            break
        time.sleep(0.5)
    else:
        # Raising would make the agent print again; job_status reads "-0" as done.
        return {"job_id": f"{printer}-0", "printer": printer, "file": str(pdf),
                "message": "print command ran, but the job left the queue before it was seen"}
    return {"job_id": f"{printer}-{max(new, key=int)}", "printer": printer, "file": str(pdf)}


def _lp_submit(printer, pdf, options=()):
    match = re.search(r"request id is (\S+)", _run(["lp", "-d", printer, *options, str(pdf)]))
    if not match:
        raise PrinterError("lp did not return a job id")
    return {"job_id": match.group(1), "printer": printer, "file": str(pdf)}


# ---------- the connector functions ----------

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
    pdf = _existing_pdf(pdf_path)
    if _fake():
        return {"job_id": _fake_submit(pdf, printer), "printer": printer, "file": str(pdf)}
    if _windows():
        def print_command():
            if os.environ.get("SUMATRA_PATH"):
                _run([os.environ["SUMATRA_PATH"], "-print-to", printer, "-silent", str(pdf)])
            else:
                _ps(f"Start-Process -FilePath {_quote(str(pdf))} -Verb PrintTo "
                    f"-ArgumentList {_quote(chr(34) + printer + chr(34))} -WindowStyle Hidden")
        return _windows_submit(printer, pdf, print_command)
    return _lp_submit(printer, pdf)


def submit_trial_print(pdf_path, printer):
    """Send a trial sheet PDF: 1 copy, A4, grayscale, one-sided, 100% size. Same result as submit_print."""
    pdf = _existing_pdf(pdf_path)
    if _fake():
        return {"job_id": _fake_submit(pdf, printer, kind="trial"), "printer": printer, "file": str(pdf)}
    if _windows():
        sumatra = os.environ.get("SUMATRA_PATH")
        if not sumatra:
            raise PrinterError("trial sheets need SumatraPDF (set SUMATRA_PATH): the default Windows "
                               "print path may scale the page, and the sheet is measured in mm")
        return _windows_submit(printer, pdf, lambda: _run(
            [sumatra, "-print-to", printer, "-print-settings", TRIAL_SUMATRA_SETTINGS, "-silent", str(pdf)]))
    return _lp_submit(printer, pdf, TRIAL_LP_OPTIONS)


def job_status(job_id):
    """Return {"job_id", "state", "message"}: state is waiting, printing, done, failed or unknown."""
    printer, _, number = job_id.rpartition("-")
    if _fake():
        job = _fake_load()["jobs"].get(job_id)
        if job is None:
            raise PrinterError(f"unknown job: {job_id}")
        problem = job.get("problem") or _fake_problem()
        state = _fake_job_state(job, problem)
        if job.get("cancelled"):
            message = "cancelled"
        else:
            message = f"printer reports {problem}" if state == "failed" else f"fake job is {state}"
        return {"job_id": job_id, "state": state, "message": message}
    if _windows():
        out = _ps(f"$j = Get-PrintJob -PrinterName {_quote(printer)} | Where-Object Id -eq {int(number)}; "
                  "if ($j) { 'found: ' + $j.JobStatus } else { 'gone' }")
        if out == "gone":
            return {"job_id": job_id, "state": "done", "message": "job left the queue (assumed printed)"}
        state, problem = _job_state(out)
        result = {"job_id": job_id, "state": state, "message": out}
        if problem:
            result["problem"] = problem
        return result
    if any(line.split()[0] == job_id for line in _run(["lpstat", "-o", printer]).splitlines()):
        status = printer_status(printer)
        state = "printing" if f"now printing {job_id}" in status["message"] else "waiting"
        result = {"job_id": job_id, "state": state, "message": f"printer is {status['state']}"}
        if status["state"] in ("stopped", "offline", "out_of_paper", "paper_jam", "error"):
            result["problem"] = status["state"]
        return result
    if any(line.split()[0] == job_id for line in _run(["lpstat", "-W", "completed", "-o", printer]).splitlines()):
        return {"job_id": job_id, "state": "done", "message": "job is in the completed list"}
    return {"job_id": job_id, "state": "unknown", "message": "job not found in the queue or completed list"}


def cancel_job(job_id):
    """Cancel a job that has not printed yet. Return {"job_id", "state", "message"}.

    A job that is already done stays done. An unknown job raises PrinterError.
    """
    if _fake():
        data = _fake_load()
        job = data["jobs"].get(job_id)
        if job is None:
            raise PrinterError(f"unknown job: {job_id}")
        if _fake_job_state(job, job.get("problem") or _fake_problem()) == "done":
            return {"job_id": job_id, "state": "done", "message": "job was already printed; nothing to cancel"}
        job["cancelled"] = True
        FAKE_JOBS_FILE.write_text(json.dumps(data, indent=2))
        return {"job_id": job_id, "state": "failed", "message": "cancelled"}

    status = job_status(job_id)
    if status["state"] == "done":
        return {"job_id": job_id, "state": "done", "message": "job was already printed; nothing to cancel"}
    if status["state"] == "unknown":
        raise PrinterError(f"unknown job: {job_id}")
    if _windows():
        printer, _, number = job_id.rpartition("-")
        if not number.isdigit():
            raise PrinterError(f"unknown job: {job_id}")
        _ps(f"Remove-PrintJob -PrinterName {_quote(printer)} -ID {int(number)}")
    else:
        _run(["cancel", job_id])
    return {"job_id": job_id, "state": "failed", "message": "cancelled"}


# ---------- command line ----------

USAGE = """usage:
  python printer_connector.py list
  python printer_connector.py status PRINTER
  python printer_connector.py print PDF_FILE PRINTER
  python printer_connector.py trial PDF_FILE PRINTER
  python printer_connector.py job JOB_ID
  python printer_connector.py cancel JOB_ID"""


def main(argv):
    commands = {"list": (list_printers, 0), "status": (printer_status, 1),
                "print": (submit_print, 2), "trial": (submit_trial_print, 2),
                "job": (job_status, 1), "cancel": (cancel_job, 1)}
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
