"""Connector for the office printer: the only code that talks to the instrument.

Interface: the operating system's own print commands, run through subprocess.
  - Windows:       PowerShell Get-Printer / Get-PrintJob / Remove-PrintJob, and SumatraPDF
                   -print-to (if SUMATRA_PATH is set) or Start-Process -Verb PrintTo.
  - macOS / Linux: CUPS commands lp, lpstat and cancel.

Two ways to print:
  - submit_print:       the two-page report, with the system's default print settings.
  - submit_trial_print: the grayscale trial sheets from trial_sheet.py: one copy, A4,
                        grayscale, two-sided (long-edge) and at 100% size (no fit-to-page), because the
                        sheet's positions are measured in mm on the scan. On Windows this
                        needs SumatraPDF (SUMATRA_PATH), since the default path may scale.
cancel_job removes a job that has not printed yet.

Set PRINTER_MODE=fake to use a pretend printer called Office_Printer instead of real
commands. Set FAKE_PRINTER_PROBLEM=out_of_paper (or offline, paper_jam) to make the
fake printer report that problem, =untracked to make a submission that is never seen in
the queue, or =lost to make jobs leave the queue without a confirmed print.

Printer states: idle, printing, stopped, offline, out_of_paper, paper_jam, error, unknown.
Job states:     waiting, printing, done, failed, uncertain.
                Every job_status result has "confirmed": true when the state was read from the
                printer system, false when it is uncertain. done is only returned when the print
                is confirmed (the lp completed list, a Windows job still listed as printed or
                complete, or the fake printer's done); it is never guessed.
                uncertain means nobody can tell whether the page printed. It has a "reason":
                  untracked_submission  the job was never seen in the queue after submitting,
                  left_queue            the job was seen, then left the queue unconfirmed,
                  not_found             the job is in neither the queue nor the completed list,
                and a message asking the human to check the printer tray.
                A waiting job that is stuck on a printer problem also has a "problem" field
                (for example out_of_paper). A cancelled job is failed, with message "cancelled".
Submissions return "tracked": true, or false with an untracked job id when the job was never
seen in the queue: then do not print again automatically, ask the human instead.
Job ids look like "<printer>-<number>", e.g. Office_Printer-1, and untracked ones like
"<printer>-untracked-<timestamp>".
"""
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from app.printing.settings import TRIAL_LP_OPTIONS, TRIAL_SUMATRA_SETTINGS, TRIAL_SIDES

FAKE_PRINTER = "Office_Printer"
FAKE_JOBS_FILE = Path(os.environ.get("FAKE_PRINTER_FILE", Path(__file__).resolve().parents[2] / "fake_printer_jobs.json"))
FAKE_PROBLEMS = ("out_of_paper", "offline", "paper_jam")     # printer problems
FAKE_JOB_PROBLEMS = ("untracked", "lost")                    # submissions we cannot confirm
UNTRACKED = "-untracked-"          # in the job id of a submission that was never seen in the queue
CHECK_TRAY = "Could not confirm that the page printed. Please check the printer tray."
UNCERTAIN_DETAILS = {
    "untracked_submission": "The print command ran, but the job was never seen in the queue.",
    "left_queue": "The job left the queue without a printed status (printed, deleted or failed).",
    "not_found": "The job is in neither the queue nor the completed list.",
}
FAKE_QUEUE_SECONDS = 1.0   # a fake job waits this long, then prints
FAKE_PRINT_SECONDS = 3.0   # a fake job is done this long after submission

# Effective trial options are shared with the approval and journal paths.


class PrinterError(Exception):
    """A print command failed or the printer gave an answer we cannot use."""


def _fake():
    return os.environ.get("PRINTER_MODE", "").lower() == "fake"


def _windows():
    return sys.platform.startswith("win")


def trial_print_preflight(*, fake_printer=None):
    """Validate local printing configuration without commands, printer I/O or writes.

    SUMATRA_PATH remains authoritative; never silently replace a bad configured
    path with another installation. A bare executable name may resolve via PATH.
    """
    if (fake_printer if fake_printer is not None else _fake()) or not _windows():
        return None
    configured = os.environ.get("SUMATRA_PATH", "").strip()
    resolved = shutil.which(configured) if configured else None
    if not resolved or not Path(resolved).is_file() or Path(resolved).suffix.lower() != ".exe":
        raise PrinterError(
            f"Trial-print configuration: SUMATRA_PATH={configured!r} does not resolve to an existing .exe. "
            "Set SUMATRA_PATH to the installed SumatraPDF.exe (also check "
            r"%LOCALAPPDATA%\SumatraPDF\SumatraPDF.exe), then restart. "
            "If absent, install SumatraPDF. No print was requested by this check.")
    return str(Path(resolved).resolve())


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


def _uncertain(job_id, reason):
    """A job_status result for a job whose print nobody can confirm."""
    return {"job_id": job_id, "state": "uncertain", "confirmed": False, "reason": reason,
            "message": f"{CHECK_TRAY} {UNCERTAIN_DETAILS[reason]}"}


def _untracked_id(printer):
    return f"{printer}{UNTRACKED}{int(time.time() * 1000)}"


def _is_untracked(job_id):
    return re.search(re.escape(UNTRACKED) + r"\d+$", job_id) is not None


def _untracked_submission(printer, pdf):
    return {"job_id": _untracked_id(printer), "printer": printer, "file": str(pdf), "tracked": False,
            "message": f"{CHECK_TRAY} {UNCERTAIN_DETAILS['untracked_submission']} Do not print again "
                       "automatically."}


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
    """A job sent during a problem never prints; a problem that starts later stops unfinished jobs.

    A "lost" job leaves the queue when it would be done, without a confirmed print: uncertain.
    """
    age = time.time() - job["submitted"]
    if job.get("cancelled"):
        return "failed"
    if job.get("problem"):
        return "failed"
    if age >= FAKE_PRINT_SECONDS:
        return "uncertain" if job.get("lost") else "done"
    if problem:
        return "failed"
    return "waiting" if age < FAKE_QUEUE_SECONDS else "printing"


def _fake_problem():
    problem = os.environ.get("FAKE_PRINTER_PROBLEM", "").lower()
    return problem if problem in FAKE_PROBLEMS else None


def _fake_check(printer):
    if printer != FAKE_PRINTER:
        raise PrinterError(f"unknown printer: {printer} (the fake printer is {FAKE_PRINTER})")


def _fake_job_problem():
    problem = os.environ.get("FAKE_PRINTER_PROBLEM", "").lower()
    return problem if problem in FAKE_JOB_PROBLEMS else None


def _fake_submit(pdf, printer, **extra):
    """Add a job to the fake jobs file and return the submission result."""
    _fake_check(printer)
    if _fake_job_problem() == "untracked":       # the job is never seen, so nothing is recorded
        return _untracked_submission(printer, pdf)
    data = _fake_load()
    job_id = f"{printer}-{data['next_id']}"
    data["next_id"] += 1
    data["jobs"][job_id] = {"file": str(pdf), "submitted": time.time(), "problem": _fake_problem(),
                            "lost": _fake_job_problem() == "lost", **extra}
    FAKE_JOBS_FILE.write_text(json.dumps(data, indent=2))
    return {"job_id": job_id, "printer": printer, "file": str(pdf), "tracked": True}


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
        # Raising would make the agent print again, and nothing confirms it printed: untracked.
        return _untracked_submission(printer, pdf)
    return {"job_id": f"{printer}-{max(new, key=int)}", "printer": printer, "file": str(pdf), "tracked": True}


def _lp_submit(printer, pdf, options=()):
    match = re.search(r"request id is (\S+)", _run(["lp", "-d", printer, *options, str(pdf)]))
    if not match:
        raise PrinterError("lp did not return a job id")
    return {"job_id": match.group(1), "printer": printer, "file": str(pdf), "tracked": True}


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
    """Send a PDF to the printer. Return {"job_id", "printer", "file", "tracked"} (and "message" if untracked)."""
    pdf = _existing_pdf(pdf_path)
    if _fake():
        return _fake_submit(pdf, printer)
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
    """Send a trial sheet PDF: 1 copy, A4, gray, long-edge duplex, 100% size."""
    pdf = _existing_pdf(pdf_path)
    if _fake():
        return _fake_submit(pdf, printer, kind="trial", sides=TRIAL_SIDES)
    if _windows():
        sumatra = trial_print_preflight()
        return _windows_submit(printer, pdf, lambda: _run(
            [sumatra, "-print-to", printer, "-print-settings", TRIAL_SUMATRA_SETTINGS, "-silent", str(pdf)]))
    return _lp_submit(printer, pdf, TRIAL_LP_OPTIONS)


def job_status(job_id):
    """Return {"job_id", "state", "confirmed", "message"}: state is waiting, printing, done, failed or
    uncertain. done only when the print is confirmed; uncertain (confirmed false) also has a "reason".
    """
    if _is_untracked(job_id):          # never seen in the queue: there is nothing to ask the printer
        return _uncertain(job_id, "untracked_submission")
    printer, _, number = job_id.rpartition("-")
    if _fake():
        job = _fake_load()["jobs"].get(job_id)
        if job is None:
            raise PrinterError(f"unknown job: {job_id}")
        problem = job.get("problem") or _fake_problem()
        state = _fake_job_state(job, problem)
        if state == "uncertain":
            return _uncertain(job_id, "left_queue")
        if job.get("cancelled"):
            message = "cancelled"
        else:
            message = f"printer reports {problem}" if state == "failed" else f"fake job is {state}"
        return {"job_id": job_id, "state": state, "confirmed": True, "message": message}
    if _windows():
        out = _ps(f"$j = Get-PrintJob -PrinterName {_quote(printer)} | Where-Object Id -eq {int(number)}; "
                  "if ($j) { 'found: ' + $j.JobStatus } else { 'gone' }")
        if out == "gone":              # deleted, failed or printed: Windows does not say which
            return _uncertain(job_id, "left_queue")
        state, problem = _job_state(out)
        result = {"job_id": job_id, "state": state, "confirmed": True, "message": out}
        if problem:
            result["problem"] = problem
        return result
    if any(line.split()[0] == job_id for line in _run(["lpstat", "-o", printer]).splitlines()):
        status = printer_status(printer)
        state = "printing" if f"now printing {job_id}" in status["message"] else "waiting"
        result = {"job_id": job_id, "state": state, "confirmed": True, "message": f"printer is {status['state']}"}
        if status["state"] in ("stopped", "offline", "out_of_paper", "paper_jam", "error"):
            result["problem"] = status["state"]
        return result
    if any(line.split()[0] == job_id for line in _run(["lpstat", "-W", "completed", "-o", printer]).splitlines()):
        return {"job_id": job_id, "state": "done", "confirmed": True, "message": "job is in the completed list"}
    return _uncertain(job_id, "not_found")


def _cannot_cancel(status):
    """cancel_job's answer for an uncertain job: nothing was cancelled, and that is said plainly."""
    return status | {"message": "Could not cancel the job because it cannot be tracked. " + status["message"]}


def cancel_job(job_id):
    """Cancel a job that has not printed yet. Return {"job_id", "state", "message"}.

    A job that is already done stays done. An uncertain or untracked job cannot be cancelled: it
    returns state uncertain (with confirmed false and reason) instead of raising. A fake job id that
    was never issued raises PrinterError.
    """
    if _is_untracked(job_id):
        return _cannot_cancel(_uncertain(job_id, "untracked_submission"))
    if _fake():
        data = _fake_load()
        job = data["jobs"].get(job_id)
        if job is None:
            raise PrinterError(f"unknown job: {job_id}")
        state = _fake_job_state(job, job.get("problem") or _fake_problem())
        if state == "done":
            return {"job_id": job_id, "state": "done", "message": "job was already printed; nothing to cancel"}
        if state == "uncertain":
            return _cannot_cancel(_uncertain(job_id, "left_queue"))
        job["cancelled"] = True
        FAKE_JOBS_FILE.write_text(json.dumps(data, indent=2))
        return {"job_id": job_id, "state": "failed", "message": "cancelled"}

    status = job_status(job_id)
    if status["state"] == "done":
        return {"job_id": job_id, "state": "done", "message": "job was already printed; nothing to cancel"}
    if status["state"] == "uncertain":
        return _cannot_cancel(status)
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
  python -m app.printing.connector list
  python -m app.printing.connector status PRINTER
  python -m app.printing.connector print PDF_FILE PRINTER
  python -m app.printing.connector trial PDF_FILE PRINTER
  python -m app.printing.connector job JOB_ID
  python -m app.printing.connector cancel JOB_ID"""


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
