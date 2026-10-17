"""In-memory mock of the print-tuning web API, for building and testing the frontend.

Run with:  uvicorn frontend.mock_api:app --port 8000

It implements the same contract the real backend will (see frontend/README.md) and walks every
session through one realistic run that reaches every pause kind of app/workflow/graph.py:

  experiments -> budget -> tuning_printer -> trial_print_approval (round 1)
  -> tuning_print_confirmation -> tuning_scan (round 1) -> trial_print_approval (round 2)
  -> tuning_poll -> tuning_scan (round 2) -> tuning_problem -> trial_print_approval (round 3)
  -> tuning_scan (round 3) -> report_workflow_problem -> print_approval -> printer_problem
  -> report_print_confirmation -> finished

Answers branch like the real graph: stop ends the run, budget "no" stops it, not_printed and
reprint print the trial again, a change request at print_approval rebuilds the report (and offers
report_workflow_problem with "correct"), report_print_confirmation not_printed asks for approval
again, and printer_problem skip finishes without printing. A request containing "[fail]" makes
the run fail after the experiments, to try the failed state.

Nothing is kept after the process ends. Files are written under MOCK_RUNS_DIR (default: a new
temporary folder). MOCK_STEP_SECONDS (default 1.5) is how long the fake agents "work" between
pauses; the session shows status "running" meanwhile.
"""
from collections import deque
from datetime import datetime, timezone
from functools import lru_cache
import io
import json
import os
from pathlib import Path
import re
import tempfile
import threading
import time
from uuid import uuid4

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

PRINTERS = ["Office_Printer", "Lab_Printer"]
FAKE_PRINTER = "Office_Printer"
SCAN_FORMATS = ("png", "jpg", "jpeg", "pdf")
EXPERIMENTS = ("macropixel 1", "macropixel 2", "macropixel 4", "macropixel 8")
ROUND_GAMMAS = {1: (1.0, 0.18, 0.3, 0.5, 0.84, 1.4), 2: (0.3, 0.16, 0.22, 0.27, 0.34, 0.43),
                3: (0.34, 0.22, 0.27, 0.31, 0.38, 0.46)}
TRIAL_SETTINGS = {"copies": 1, "paper": "A4", "color_mode": "grayscale", "sides": "two-sided-long-edge",
                  "scale": "100%"}


class SessionRequest(BaseModel):
    request: str
    model: str | None = None
    specialist_model: str | None = None
    printer: str | None = None
    fake_printer: bool = True
    paper_color: str | None = None
    initial_decisions: int | None = None
    extra_decisions: int | None = None


class AnswerRequest(BaseModel):
    answer: str | dict
    interrupt_token: str | None = None


# ---------- sample files (small, generated once per process) ----------

def _figure_bytes(draw, pages: int, fmt: str) -> bytes:
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.backends.backend_pdf import PdfPages
    import matplotlib.pyplot as plt

    buffer = io.BytesIO()
    if fmt == "pdf":
        with PdfPages(buffer) as pdf:
            for page in range(pages):
                fig = plt.figure(figsize=(8.27, 11.69))
                draw(fig, page)
                pdf.savefig(fig)
                plt.close(fig)
    else:
        fig = plt.figure(figsize=(6, 4), dpi=80)
        draw(fig, 0)
        fig.savefig(buffer, format=fmt)
        plt.close(fig)
    return buffer.getvalue()


def _heatmap(gamma: float):
    import numpy as np
    y, x = np.mgrid[-1:1:48j, -1:1:48j]
    data = np.exp(-(x ** 2 + y ** 2) * 6) + 0.15 * np.exp(-((x - .5) ** 2 + (y + .4) ** 2) * 20)
    return (data / data.max()) ** gamma


@lru_cache(maxsize=None)
def trial_pdf(round_id: int) -> bytes:
    gammas = ROUND_GAMMAS[round_id]

    def draw(fig, page):
        fig.suptitle(f"MOCK trial sheet - round {round_id} - page {page + 1}/2 - gamma", fontsize=12)
        for row in range(2):
            for col, gamma in enumerate(gammas):
                ax = fig.add_subplot(4, 6, row * 12 + col + 1 + (6 if row else 0))
                ax.imshow(_heatmap(gamma), cmap="gray", vmin=0, vmax=1)
                ax.set_title(("ref " if col == 0 else "") + f"g {gamma:g}", fontsize=7)
                ax.axis("off")
            fig.text(0.05, 0.9 - row * 0.45, EXPERIMENTS[page * 2 + row], fontsize=9)
    return _figure_bytes(draw, 2, "pdf")


@lru_cache(maxsize=None)
def preview_png(gamma: float = 1.0) -> bytes:
    def draw(fig, _):
        for index, panel in enumerate(("Initial", "Final")):
            ax = fig.add_subplot(1, 2, index + 1)
            ax.imshow(_heatmap(gamma if panel == "Final" else gamma * 2), cmap="gray", vmin=0, vmax=1)
            ax.set_title(f"{panel} (gamma {gamma:g})", fontsize=9)
            ax.axis("off")
    return _figure_bytes(draw, 1, "png")


@lru_cache(maxsize=None)
def report_pdf(revision: int = 0) -> bytes:
    def draw(fig, page):
        if page == 0:
            fig.text(0.1, 0.92, "MOCK report: macropixel size vs focusing enhancement", fontsize=14)
            ax = fig.add_axes([0.12, 0.55, 0.76, 0.3])
            ax.bar([str(m) for m in (1, 2, 4, 8)], [41.2, 38.5, 22.1, 9.8], color="0.3")
            ax.set_xlabel("macropixel size")
            ax.set_ylabel("final enhancement")
            fig.text(0.1, 0.45, "Verdict: smaller macropixels focus better at a higher cost (mock text)."
                     + (f"  [revision {revision}]" if revision else ""), fontsize=9)
        else:
            fig.text(0.1, 0.92, "Tuned print comparison (mock)", fontsize=14)
            for index, gamma in enumerate((0.3, 0.34, 0.44, 0.49)):
                ax = fig.add_subplot(2, 2, index + 1)
                ax.imshow(_heatmap(gamma), cmap="gray", vmin=0, vmax=1)
                ax.set_title(f"{EXPERIMENTS[index]}: gamma {gamma:g}", fontsize=8)
                ax.axis("off")
    return _figure_bytes(draw, 2, "pdf")


# ---------- the scripted run ----------

def _say(role, text):
    return ("say", role, text)


def _tool(name, args, result):
    return ("tool", name, args, result)


def _pause(payload):
    return ("pause", payload)


def _write(run_dir: Path, relative: str, content: bytes) -> str:
    path = run_dir / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return str(path)


class MockRun:
    """One session: its public record plus the queue of scripted events still to come."""

    def __init__(self, session_id: str, run_dir: Path, options: SessionRequest, step_seconds: float):
        self.session_id, self.run_dir, self.options = session_id, run_dir, options
        self.step_seconds = step_seconds
        self.created_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        self.status, self.stop_reason = "running", "running"
        self.dialogue = [{"role": "user", "text": options.request}]
        self.execution_results: list[dict] = []
        self.pending: dict | None = None
        self.active_round: int | None = None
        self.printer: str | None = FAKE_PRINTER if options.fake_printer else options.printer
        self.tuning_printer: str | None = None      # asked once at the first trial, like the real graph
        self.revision = 0
        self.resume_at = time.monotonic() + step_seconds
        self.queue: deque = deque(self._experiments())
        if "[fail]" in options.request.lower():
            self.queue.append(("fail", "Execution/API failure (mock): the model API timed out; see error.txt."))
        else:
            self.queue.append(_pause({
                "kind": "budget",
                "summary": "Four focusing experiments finished (macropixel 1, 2, 4, 8). Next: tune how the "
                           "heatmaps print, then write and print the report.",
                "decisions_used": options.initial_decisions or 10, "extra": options.extra_decisions or 5,
                "question": f"Continue for {options.extra_decisions or 5} more orchestrator decisions? "
                            "Answer yes or no; you can add guidance after yes."}))

    # ----- script pieces -----

    def _experiments(self):
        events = [_say("system", "Homework 3: hardware in the loop (mock backend, no models are called)."),
                  _say("orchestrator", "Ask the scientist which four experiments answer the question.")]
        events.append(_say("scientist", "Vary only macropixel_size (1, 2, 4, 8); keep focus 3, background 9, "
                                        "200 iterations and seed 7."))
        for number, size in enumerate((1, 2, 4, 8), 1):
            events.append(_tool("run_focusing_experiment",
                                {"macropixel_size": size, "focus_size": 3, "background_size": 9,
                                 "iterations": 200, "search_seed": 7},
                                {"status": "ok", "experiment_id": f"experiment_mock{number}",
                                 "initial_enhancement": round(1.1 + number * .1, 3),
                                 "final_enhancement": [41.2, 38.5, 22.1, 9.8][number - 1],
                                 "paths": {"final_image": _write(self.run_dir, f"experiment_mock{number}/"
                                                                 "final_camera.png", preview_png(1.0))}}))
        events.append(_say("executor", "All four experiments ran; results are in experiments.csv."))
        return events

    def _tuning_start(self):
        preview = _write(self.run_dir, "print_tuning/previews/preview_gamma_1.png", preview_png(1.0))
        return [_say("orchestrator", "Experiments are chosen. Start print tuning: only gamma is tuned, in 3 rounds."),
                _say("general", "Selected the four experiments for the trial sheets."),
                _tool("render_previews", {"experiments": list(EXPERIMENTS)},
                      {"status": "ok", "preview_paths": [preview]}),
                _say("photo_quality_specialist", "propose_gamma: low values are dark, so test gammas below 1: "
                                                 "0.18, 0.3, 0.5, 0.84, 1.4."),
                *self._make_trial(1),
                ("call", "_after_trial_made")]

    def _make_trial(self, round_id):
        def make(run):
            run.active_round = round_id
            pdf = _write(run.run_dir, f"print_tuning/trial_r{round_id}.pdf", trial_pdf(round_id))
            run._record("make_trial_sheet", {"round_id": round_id},
                        {"status": "ok", "pdf_path": pdf, "page_count": 2, "round_id": round_id,
                         "tested_parameter": "gamma", "variations_per_item": 6})
        return [("call", make)]

    def _approval(self, round_id):
        pdf = str(self.run_dir / f"print_tuning/trial_r{round_id}.pdf")
        printer = self.tuning_printer or self.printer or FAKE_PRINTER
        return [_tool("printer_status", {"printer": printer}, {"printer": printer, "state": "idle"}),
                _pause({"kind": "trial_print_approval", "pdf_path": pdf, "page_count": 2, "printer": printer,
                        "session_id": f"tuning-{self.session_id[:8]}", "round_id": round_id,
                        "trial_id": f"trial-{round_id}-{uuid4().hex[:8]}",
                        "pdf_sha256": f"{round_id:02d}" + uuid4().hex + uuid4().hex[:30],
                        "print_settings": {**TRIAL_SETTINGS, "printer": printer},
                        "question": "Approve printing this exact trial PDF? Choose approve or stop.",
                        "choices": ["approve", "stop"]})]

    def _after_trial_made(self):
        """Ask for the printer only once; afterwards go straight to the approval."""
        if self.tuning_printer:
            return self._approval(self.active_round)
        return [_pause({"kind": "tuning_printer", "question": "Supply the exact printer name, or stop.",
                        "choices": ["printer", "stop"]})]

    def _submitted(self, round_id):
        printer = self.tuning_printer or self.printer or FAKE_PRINTER
        job = f"job-{round_id}-{uuid4().hex[:6]}"
        submit = _tool("submit_trial_print", {"pdf_path": str(self.run_dir / f"print_tuning/trial_r{round_id}.pdf"),
                                              "printer": printer}, {"status": "submitted", "job_id": job})
        if round_id == 1:       # the printer never reports this job: ask the human
            return [submit, _tool("job_status", {"job_id": job}, {"job_id": job, "state": "uncertain"}),
                    _pause({"kind": "tuning_print_confirmation", "attempt_id": f"attempt-{job}",
                            "question": "Check the printer tray. Did this trial print?",
                            "choices": ["printed", "not_printed", "stop"]})]
        if round_id == 2:       # still printing when checked
            return [submit, _tool("job_status", {"job_id": job}, {"job_id": job, "state": "printing"}),
                    _pause({"kind": "tuning_poll", "question": "Job is still waiting/printing. Check again or stop?",
                            "choices": ["check", "stop"]}),
                    _tool("job_status", {"job_id": job}, {"job_id": job, "state": "done"}), *self._scan(2)]
        return [submit, _tool("job_status", {"job_id": job}, {"job_id": job, "state": "done"}), *self._scan(3)]

    def _scan(self, round_id):
        return [_say("system", f"Round {round_id} printed. Scan the pages."),
                _pause({"kind": "tuning_scan", "round_id": round_id, "trial_id": f"trial-{round_id}",
                        "suggested_scan_directory": str(self.run_dir / "scans" / f"round_{round_id}"),
                        "question": "Scan BOTH faces of each printed sheet as separate pages, keeping all markers "
                                    "visible. Save and supply explicit paths; or reprint/stop.",
                        "choices": ["scan", "reprint", "stop"]})]

    def _after_scan(self, round_id, paths):
        events = [_tool("read_scan", {"scan_paths": paths},
                        {"status": "ok", "round_id": round_id, "candidates": 24, "pages": len(paths)})]
        if round_id < 3:
            winner = ROUND_GAMMAS[round_id + 1][0]
            preview = _write(self.run_dir, f"print_tuning/previews/preview_gamma_r{round_id}.png", preview_png(winner))
            events += [_say("photo_quality_specialist", f"review_gamma (round {round_id}): best gamma about "
                                                         f"{winner:g}; next tests refine around it."),
                       _tool("render_previews", {"round_id": round_id}, {"status": "ok", "preview_paths": [preview]})]
            if round_id == 2:   # rendering round 3 fails once
                events += [_tool("make_trial_sheet", {"round_id": 3},
                                 {"status": "error", "error": "trial PDF could not be written (disk busy)"}),
                           _pause({"kind": "tuning_problem", "message": "make_trial_sheet failed: trial PDF could not "
                                                                        "be written (disk busy)",
                                   "question": "Resolve the problem, then retry; regenerate only damaged trial "
                                               "artifacts, or stop.",
                                   "choices": ["retry", "regenerate", "stop"]})]
            else:
                events += [*self._make_trial(round_id + 1), ("call", "_after_trial_made")]
            return events
        figure = _write(self.run_dir, "print_tuning/final_heatmaps.png", preview_png(0.34))
        return events + [
            _say("photo_quality_specialist", "review_gamma_final: final gammas 0.30, 0.34, 0.44, 0.49."),
            _tool("render_final_heatmaps", {}, {"status": "ok", "figure_path": figure}),
            _say("orchestrator", "Tuning is complete. Write the report and ask before printing."),
            _tool("write_and_run", {"script": "build_report.py"},
                  {"exit_code": 1, "stderr": "Text clipped on page 1 (mock)"}),
            _pause({"kind": "report_workflow_problem", "message": "The report script failed twice: text clipped.",
                    "question": "Report workflow stalled. Retry the approval-request workflow, or stop for "
                                "application recovery.", "choices": ["retry", "stop"]})]

    def _report_ready(self):
        pdf = _write(self.run_dir, "report.pdf", report_pdf(self.revision))
        printer = self.printer or FAKE_PRINTER
        return [_tool("write_and_run", {"script": "build_report.py"}, {"exit_code": 0, "report_path": pdf}),
                _say("executor", "report.pdf is ready (2 pages). Asking the human before printing."),
                _pause({"kind": "print_approval", "pdf_path": pdf, "pages": 2, "printer": printer, "copies": 1,
                        "pdf_sha256": uuid4().hex + uuid4().hex,
                        "question": "Print this report? Type yes, or describe what to change."})]

    def _report_submitted(self):
        printer = self.printer or FAKE_PRINTER
        job = f"job-report-{uuid4().hex[:6]}"
        reading = {"printer": printer, "state": "paper_jam", "message": "Paper jam in tray 2 (mock)"}
        if not getattr(self, "_jam_shown", False):
            self._jam_shown = True
            return [_tool("printer_status", {"printer": printer}, reading),
                    _pause({"kind": "printer_problem", "reading": {"tool_name": "printer_status", "result": reading},
                            "question": "The printer reports a problem. Fix it and type retry, or type skip to "
                                        "finish without printing.", "choices": ["retry", "skip"]})]
        pdf = str(self.run_dir / "report.pdf")
        return [_tool("submit_print", {"pdf_path": pdf, "printer": printer, "copies": 1},
                      {"status": "submitted", "job_id": job}),
                _tool("job_status", {"job_id": job}, {"job_id": job, "state": "uncertain"}),
                _pause({"kind": "report_print_confirmation", "job_id": job,
                        "reading": {"submission": {"job_id": job}, "latest_job_reading": {"state": "uncertain"},
                                    "pdf_path": pdf, "printer": printer, "copies": 1},
                        "question": "Check the printer tray. Did this exact report print? Choose printed, "
                                    "not_printed, or stop.", "choices": ["printed", "not_printed", "stop"]})]

    # ----- engine -----

    def _record(self, name, args, result):
        self.execution_results.append({"tool_name": name, "tool_args": args, "result": result})

    def advance(self):
        """Run queued events until the next pause or the end, once the fake work time has passed."""
        if self.status != "running" or time.monotonic() < self.resume_at:
            return
        while self.queue:
            event = self.queue.popleft()
            kind = event[0]
            if kind == "say":
                self.dialogue.append({"role": event[1], "text": event[2]})
            elif kind == "tool":
                self._record(*event[1:])
            elif kind == "call":
                more = getattr(self, event[1])() if isinstance(event[1], str) else event[1](self)
                if more:
                    self.queue.extendleft(reversed(more))
            elif kind == "pause":
                self.pending, self.status = event[1], "waiting_for_human"
                self.pending = {**self.pending, "interrupt_token": uuid4().hex}
                return
            elif kind == "fail":
                self.dialogue.append({"role": "system", "text": event[1]})
                self.status = self.stop_reason = "failed"
                return
            elif kind == "end":
                self._end(event[1])
                return
        self._end("finished")

    def _end(self, reason):
        self.stop_reason = reason
        self.status = {"finished": "finished", "failed": "failed"}.get(reason, "stopped")
        self.pending = None
        self.queue.clear()

    def _continue(self, *events):
        self.queue.extendleft(reversed(events))
        self.pending, self.status = None, "running"
        self.resume_at = time.monotonic() + self.step_seconds

    def answer(self, answer):
        """Check the answer against the pending pause (ValueError = invalid), then continue the run."""
        payload = self.pending
        kind = payload["kind"]
        self.dialogue.append({"role": "human", "text": answer if isinstance(answer, str)
                              else json.dumps(answer, ensure_ascii=False)})
        action = validate_answer(payload, answer, self.run_dir)
        stop = ("end", "stopped_by_human")
        if action == "stop":
            return self._continue(stop)
        if kind == "budget":
            return self._continue(*(self._tuning_start() if action == "yes" else [stop]))
        if kind == "tuning_printer":
            self.tuning_printer = answer["printer"].strip()
            return self._continue(*self._approval(self.active_round))
        if kind == "trial_print_approval":
            return self._continue(*self._submitted(payload["round_id"]))
        if kind == "tuning_print_confirmation":
            if action == "not_printed":
                return self._continue(_say("system", "Not printed: the trial needs a fresh approval."),
                                      *self._approval(self.active_round))
            return self._continue(*self._scan(self.active_round))
        if kind == "tuning_poll":
            return self._continue()
        if kind == "tuning_scan":
            if action == "reprint":
                return self._continue(*self._approval(payload["round_id"]))
            return self._continue(*self._after_scan(payload["round_id"], answer["scan_paths"]))
        if kind == "tuning_problem":
            if action == "reprint":
                return self._continue(*self._approval(self.active_round))
            return self._continue(*self._make_trial(3), ("call", "_after_trial_made"))
        if kind == "report_workflow_problem":
            if action == "correct":
                self.revision += 1
                self.dialogue.append({"role": "system", "text": "Revision instructions corrected: "
                                      + answer["instructions"].strip()})
            return self._continue(("call", "_report_ready"))
        if kind == "print_approval":
            if action == "approve":
                return self._continue(("call", "_report_submitted"))
            if action == "retry":
                return self._continue(_say("system", "Retry is not approval. Review the exact PDF, printer and "
                                                     "hash, then explicitly approve."), ("call", "_report_ready"))
            self.revision += 1
            return self._continue(
                _say("executor", f"Applying the change request (revision {self.revision})."),
                _tool("write_and_run", {"script": f"build_report_v{self.revision + 1}.py"},
                      {"exit_code": 1, "stderr": "Footer exceeds bottom margin (mock)"}),
                _pause({"kind": "report_workflow_problem", "message": "The revised report failed its layout check.",
                        "question": "Report workflow stalled. Retry generation, correct revision instructions, "
                                    "or stop.", "choices": ["retry", "correct", "stop"]}))
        if kind == "printer_problem":
            if action == "skip":
                return self._continue(_say("system", "Printing skipped by the human."), ("end", "finished"))
            return self._continue(("call", "_report_submitted"))
        if kind == "report_print_confirmation":
            if action == "not_printed":
                return self._continue(_say("system", "Not printed: printing needs a fresh approval."),
                                      ("call", "_report_ready"))
            return self._continue(_say("system", "The report was printed."), ("end", "finished"))
        return self._continue()

    def public(self) -> dict:
        return {"session_id": self.session_id, "created_at": self.created_at, "status": self.status,
                "request": self.options.request, "run_dir": str(self.run_dir), "dialogue": list(self.dialogue),
                "execution_results": list(self.execution_results), "pending": self.pending,
                "stop_reason": self.stop_reason}


# ---------- answer checking (mirrors app/workflow/graph.py) ----------

def _first_word(text: str) -> str:
    return (text.strip().split() or [""])[0].lower().strip(".,!:;")


def validate_answer(payload: dict, answer, run_dir: Path) -> str:
    """Return the chosen action for a valid answer; raise ValueError with a message otherwise."""
    kind = payload.get("kind")
    if kind == "budget":
        if not isinstance(answer, str) or _first_word(answer) not in ("yes", "y", "no", "n"):
            raise ValueError('budget takes a text answer starting with "yes" (guidance may follow) or "no"')
        return "yes" if _first_word(answer) in ("yes", "y") else "no"
    if kind == "print_approval":
        if not isinstance(answer, str) or not answer.strip():
            raise ValueError('print_approval takes "yes" or a text describing what to change')
        word = answer.strip().lower()
        return "approve" if word in ("yes", "y", "approve") else "retry" if word == "retry" else "change"
    if kind == "printer_problem":
        if not isinstance(answer, str) or _first_word(answer) not in ("retry", "skip"):
            raise ValueError('printer_problem takes the text "retry" or "skip"')
        return _first_word(answer)
    choices = payload.get("choices") or []
    if not isinstance(answer, dict) or not isinstance(answer.get("action"), str):
        raise ValueError(f'{kind} takes an object {{"action": ...}} with action one of: {", ".join(choices)}')
    action = answer["action"]
    if action not in choices:
        raise ValueError(f'{kind}: action must be one of {", ".join(choices)}, got "{action}"')
    if kind == "tuning_printer" and action == "printer":
        if not isinstance(answer.get("printer"), str) or not answer["printer"].strip():
            raise ValueError('tuning_printer: give the exact printer name in "printer"')
    if kind == "report_workflow_problem" and action == "correct":
        if not isinstance(answer.get("instructions"), str) or not answer["instructions"].strip():
            raise ValueError('report_workflow_problem: "correct" needs the corrected "instructions" text')
    if kind == "tuning_scan" and action == "scan":
        paths = answer.get("scan_paths")
        if not isinstance(paths, list) or not paths or not all(isinstance(p, str) and p.strip() for p in paths):
            raise ValueError('tuning_scan: "scan" needs "scan_paths", a non-empty list of uploaded file paths')
        folder = (run_dir / "scans" / f"round_{payload.get('round_id')}").resolve()
        for text in paths:
            path = Path(text).resolve()
            if not path.is_file() or not path.is_relative_to(folder):
                raise ValueError(f"tuning_scan: {text} is not an uploaded scan of round {payload.get('round_id')}")
            if path.suffix.lower().lstrip(".") not in SCAN_FORMATS:
                raise ValueError(f"tuning_scan: {path.name} is not a scan file ({', '.join(SCAN_FORMATS)})")
    return action


# ---------- the app ----------

def _safe_name(name: str) -> str:
    base = re.sub(r"[^A-Za-z0-9._ -]", "_", Path(name.replace("\\", "/")).name).strip(" .")
    return base or "scan"


def create_app(runs_dir=None, step_seconds: float | None = None) -> FastAPI:
    runs_root = Path(runs_dir or os.environ.get("MOCK_RUNS_DIR") or tempfile.mkdtemp(prefix="mock_runs_"))
    runs_root.mkdir(parents=True, exist_ok=True)
    step = float(os.environ.get("MOCK_STEP_SECONDS", 1.5)) if step_seconds is None else step_seconds
    sessions: dict[str, MockRun] = {}
    lock = threading.Lock()

    app = FastAPI(title="Print-tuning mock API")
    origins = os.environ.get("CORS_ORIGINS", "http://localhost:8501,http://localhost:5173").split(",")
    app.add_middleware(CORSMiddleware, allow_origins=[o.strip() for o in origins if o.strip()],
                       allow_methods=["*"], allow_headers=["*"])

    def get(session_id: str) -> MockRun:
        run = sessions.get(session_id)
        if run is None:
            raise HTTPException(404, f"no session {session_id}")
        run.advance()
        return run

    @app.post("/api/sessions")
    def start(body: SessionRequest):
        if not body.request.strip():
            raise HTTPException(422, "request must not be blank")
        if body.printer and body.printer not in PRINTERS:
            raise HTTPException(422, f"unknown printer {body.printer}; choose one of {', '.join(PRINTERS)}")
        session_id = uuid4().hex
        run_dir = runs_root / (datetime.now().strftime("%Y%m%d_%H%M%S_") + session_id[:6])
        run_dir.mkdir(parents=True)
        with lock:
            sessions[session_id] = MockRun(session_id, run_dir, body, step)
        return {"session_id": session_id}

    @app.get("/api/sessions")
    def list_sessions():
        with lock:
            rows = []
            for run in sessions.values():
                run.advance()
                rows.append({"session_id": run.session_id, "created_at": run.created_at, "status": run.status,
                             "request": run.options.request})
        return sorted(rows, key=lambda row: row["created_at"], reverse=True)

    @app.get("/api/sessions/{session_id}")
    def session(session_id: str):
        with lock:
            return get(session_id).public()

    @app.post("/api/sessions/{session_id}/answer")
    def answer(session_id: str, body: AnswerRequest):
        with lock:
            run = get(session_id)
            if run.status != "waiting_for_human" or run.pending is None:
                raise HTTPException(409, f"nothing is waiting for an answer (session is {run.status})")
            if body.interrupt_token != run.pending["interrupt_token"]:
                raise HTTPException(409, "Stale or missing interrupt token")
            try:
                validate_answer(run.pending, body.answer, run.run_dir)
            except ValueError as error:
                raise HTTPException(422, str(error)) from None
            run.answer(body.answer)
        return {"ok": True}

    @app.post("/api/sessions/{session_id}/scans")
    def scans(session_id: str, files: list[UploadFile] = File(...), interrupt_token: str = Form(...)):
        with lock:
            run = get(session_id)
            if not run.pending or run.pending.get("kind") != "tuning_scan" or interrupt_token != run.pending.get("interrupt_token"):
                raise HTTPException(409, "Stale or missing interrupt token")
            if run.active_round is None or run.status in ("finished", "stopped", "failed"):
                raise HTTPException(409, "no print-tuning round is active, so there is nowhere to save scans")
            folder = run.run_dir / "scans" / f"round_{run.active_round}"
        bad = [f.filename for f in files if (f.filename or "").rsplit(".", 1)[-1].lower() not in SCAN_FORMATS]
        if bad:
            raise HTTPException(422, f"not a scan file ({', '.join(SCAN_FORMATS)}): {', '.join(map(str, bad))}")
        folder.mkdir(parents=True, exist_ok=True)
        saved = []
        for upload in files:
            name = _safe_name(upload.filename or "scan")
            target, number = folder / name, 2
            while target.exists():
                target = folder / f"{Path(name).stem}_{number}{Path(name).suffix}"
                number += 1
            target.write_bytes(upload.file.read())
            saved.append(str(target))
        return {"paths": saved}

    @app.get("/api/sessions/{session_id}/files/{path:path}")
    def files(session_id: str, path: str):
        with lock:
            root = get(session_id).run_dir.resolve()
        target = (root / path).resolve()
        if not target.is_relative_to(root):
            raise HTTPException(403, "that path is outside the run folder")
        if not target.is_file():
            raise HTTPException(404, f"no file {path} in the run folder")
        return FileResponse(target)

    @app.get("/api/sessions/{session_id}/events")
    def events(session_id: str):
        with lock:
            get(session_id)

        def stream():
            seen = {"dialogue": 0, "tool": 0, "pending": None}
            while True:
                with lock:
                    run = get(session_id)
                    record = run.public()
                for turn in record["dialogue"][seen["dialogue"]:]:
                    yield f"event: dialogue\ndata: {json.dumps(turn)}\n\n"
                for result in record["execution_results"][seen["tool"]:]:
                    yield f"event: tool\ndata: {json.dumps(result)}\n\n"
                seen["dialogue"], seen["tool"] = len(record["dialogue"]), len(record["execution_results"])
                if record["pending"] and record["pending"] != seen["pending"]:
                    yield f"event: interrupt\ndata: {json.dumps(record['pending'])}\n\n"
                seen["pending"] = record["pending"]
                if record["status"] in ("finished", "stopped", "failed"):
                    yield f"event: end\ndata: {json.dumps({'status': record['status'], 'stop_reason': record['stop_reason']})}\n\n"
                    return
                time.sleep(0.5)

        return StreamingResponse(stream(), media_type="text/event-stream")

    @app.get("/api/printers")
    def printers():
        return {"printers": list(PRINTERS)}

    app.state.sessions = sessions
    app.state.runs_root = runs_root
    return app


app = create_app()
