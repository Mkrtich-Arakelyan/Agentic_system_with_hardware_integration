"""Job tracking: done only when a print is confirmed, uncertain otherwise. No real printer is used."""

import time

import pymupdf
import pytest

from app.printing import connector, mcp_server

PRINTER = connector.FAKE_PRINTER


@pytest.fixture
def pdf(tmp_path):
    path = tmp_path / "report.pdf"
    with pymupdf.open() as document:
        document.new_page()
        document.save(path)
    return path


@pytest.fixture
def fake(monkeypatch, tmp_path):
    """The fake printer, with its jobs file in tmp_path and short print times."""
    monkeypatch.setenv("PRINTER_MODE", "fake")
    monkeypatch.delenv("FAKE_PRINTER_PROBLEM", raising=False)
    monkeypatch.setattr(connector, "FAKE_JOBS_FILE", tmp_path / "fake_printer_jobs.json")
    monkeypatch.setattr(connector, "FAKE_QUEUE_SECONDS", 0.05)
    monkeypatch.setattr(connector, "FAKE_PRINT_SECONDS", 0.15)
    return monkeypatch


def _wait_until_finished(job_id):
    for _ in range(40):
        status = connector.job_status(job_id)
        if status["state"] not in ("waiting", "printing"):
            return status
        time.sleep(0.02)
    raise AssertionError(f"job {job_id} never finished: {status}")


def test_fake_printer_without_problem_ends_confirmed_done(fake, pdf):
    submitted = connector.submit_print(pdf, PRINTER)
    assert submitted["tracked"] is True and submitted["job_id"] == f"{PRINTER}-1"
    first = connector.job_status(submitted["job_id"])
    assert first["state"] in ("waiting", "printing") and first["confirmed"] is True
    final = _wait_until_finished(submitted["job_id"])
    assert final == {"job_id": submitted["job_id"], "state": "done", "confirmed": True, "message": "fake job is done"}
    assert connector.cancel_job(submitted["job_id"])["state"] == "done"


def test_fake_cancel_before_printing_still_works(fake, pdf):
    job_id = connector.submit_trial_print(pdf, PRINTER)["job_id"]
    assert connector.cancel_job(job_id) == {"job_id": job_id, "state": "failed", "message": "cancelled"}
    assert connector.job_status(job_id)["state"] == "failed"


def test_fake_untracked_submission(fake, pdf):
    fake.setenv("FAKE_PRINTER_PROBLEM", "untracked")
    submitted = connector.submit_print(pdf, PRINTER)
    assert submitted["tracked"] is False
    assert f"{PRINTER}-untracked-" in submitted["job_id"] and "printer tray" in submitted["message"]
    status = connector.job_status(submitted["job_id"])
    assert status["state"] == "uncertain" and status["confirmed"] is False
    assert status["reason"] == "untracked_submission"
    cancelled = connector.cancel_job(submitted["job_id"])
    assert cancelled["state"] == "uncertain" and cancelled["message"].startswith("Could not cancel")
    assert connector.printer_status(PRINTER)["state"] == "idle"       # not a printer problem


def test_fake_lost_job_ends_uncertain_left_queue(fake, pdf):
    fake.setenv("FAKE_PRINTER_PROBLEM", "lost")
    submitted = connector.submit_trial_print(pdf, PRINTER)
    assert submitted["tracked"] is True
    final = _wait_until_finished(submitted["job_id"])
    assert final["state"] == "uncertain" and final["reason"] == "left_queue" and final["confirmed"] is False
    assert connector.cancel_job(submitted["job_id"])["state"] == "uncertain"


def test_mcp_server_passes_the_new_fields_through(fake, pdf):
    fake.setenv("FAKE_PRINTER_PROBLEM", "untracked")
    submitted = mcp_server.submit_print(str(pdf), PRINTER)
    assert submitted["ok"] is True and submitted["tracked"] is False
    status = mcp_server.job_status(submitted["job_id"])
    assert status["ok"] is True and status["state"] == "uncertain"
    assert status["confirmed"] is False and status["reason"] == "untracked_submission"
    assert mcp_server.cancel_job(submitted["job_id"])["state"] == "uncertain"


# ---------- Windows code, with a mocked PowerShell ----------

@pytest.fixture
def windows(monkeypatch):
    monkeypatch.delenv("PRINTER_MODE", raising=False)
    monkeypatch.delenv("SUMATRA_PATH", raising=False)
    monkeypatch.setattr(connector, "_windows", lambda: True)
    monkeypatch.setattr(connector.time, "sleep", lambda seconds: None)
    calls = []

    def install(queue_ids, job_line="gone"):
        """queue_ids: the job ids Get-PrintJob lists, call after call (the last one repeats)."""
        answers = iter(queue_ids)
        last = [""]

        def fake_ps(script):
            calls.append(script)
            if "ExpandProperty Id" in script:
                last[0] = next(answers, last[0])
                return last[0]
            if "Where-Object Id -eq" in script:
                return job_line
            return ""                                                    # Start-Process ... PrintTo
        monkeypatch.setattr(connector, "_ps", fake_ps)
        return calls
    return install


def test_windows_job_never_seen_is_untracked(windows, pdf):
    calls = windows(["3 4"])                         # the queue never shows a new job
    submitted = connector.submit_print(pdf, "HP")
    assert submitted["tracked"] is False and submitted["job_id"].startswith("HP-untracked-")
    before = len(calls)
    status = connector.job_status(submitted["job_id"])
    assert status["state"] == "uncertain" and status["reason"] == "untracked_submission"
    assert len(calls) == before                      # the printer was not asked
    assert connector.cancel_job(submitted["job_id"])["state"] == "uncertain"


def test_windows_job_gone_from_queue_is_never_done(windows, pdf):
    windows(["3", "3 7"], job_line="gone")
    submitted = connector.submit_print(pdf, "HP")
    assert submitted == {"job_id": "HP-7", "printer": "HP", "file": str(pdf.resolve()), "tracked": True}
    status = connector.job_status("HP-7")
    assert status["state"] == "uncertain" and status["reason"] == "left_queue" and status["confirmed"] is False
    assert connector.cancel_job("HP-7")["state"] == "uncertain"


def test_windows_job_listed_as_printed_is_confirmed_done(windows):
    windows([], job_line="found: Printed")
    assert connector.job_status("HP-7") == {"job_id": "HP-7", "state": "done", "confirmed": True,
                                             "message": "found: Printed"}


def test_windows_waiting_job_keeps_its_problem(windows):
    windows([], job_line="found: PaperOut")
    status = connector.job_status("HP-7")
    assert status["state"] == "waiting" and status["confirmed"] is True and status["problem"] == "out_of_paper"


# ---------- lp (macOS / Linux), with a mocked _run ----------

@pytest.fixture
def lp(monkeypatch):
    monkeypatch.delenv("PRINTER_MODE", raising=False)
    monkeypatch.setattr(connector, "_windows", lambda: False)

    def install(queue="", completed=""):
        def fake_run(cmd):
            if cmd[0] == "lp":
                return "request id is Office-12 (1 file(s))"
            if cmd[:2] == ["lpstat", "-W"]:
                return completed
            if cmd[:2] == ["lpstat", "-o"]:
                return queue
            if cmd[:2] == ["lpstat", "-p"]:
                return "printer Office is idle."
            if cmd[0] == "cancel":
                raise AssertionError("an uncertain job must not be cancelled")
            raise AssertionError(f"unexpected command {cmd}")
        monkeypatch.setattr(connector, "_run", fake_run)
    return install


def test_lp_completed_list_is_confirmed_done(lp, pdf):
    lp(completed="Office-12  razmik  1024  Fri 16 Oct 2026")
    assert connector.submit_trial_print(pdf, "Office")["tracked"] is True
    assert connector.job_status("Office-12") == {"job_id": "Office-12", "state": "done", "confirmed": True,
                                                  "message": "job is in the completed list"}


def test_lp_job_not_found_is_uncertain(lp):
    lp()
    status = connector.job_status("Office-12")
    assert status["state"] == "uncertain" and status["reason"] == "not_found" and status["confirmed"] is False
    assert "printer tray" in status["message"]
    assert connector.cancel_job("Office-12")["state"] == "uncertain"


def test_lp_job_in_queue_is_waiting(lp):
    lp(queue="Office-12  razmik  1024  Fri 16 Oct 2026")
    status = connector.job_status("Office-12")
    assert status["state"] == "waiting" and status["confirmed"] is True


def test_no_job_result_says_unknown(fake, pdf):
    job_id = connector.submit_print(pdf, PRINTER)["job_id"]
    assert connector.job_status(job_id)["state"] != "unknown"
