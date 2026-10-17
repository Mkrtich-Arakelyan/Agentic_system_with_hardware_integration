"""Single-process workers; disk state/journals survive, graph checkpoints do not.

Each claim is persisted before starting a worker. A restart invalidates old tokens
and requires explicit resume via runner.load_resume; it never replays an answer.
"""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import threading
from uuid import uuid4

from app.workflow import runner


class SessionError(Exception):
    def __init__(self, status, message):
        self.status, self.message = status, message
        super().__init__(message)


def public_value(value):
    """Defense in depth for tool/model errors; never serialize credential fields."""
    if isinstance(value, dict):
        return {k: public_value(v) for k, v in value.items()
                if not any(word in k.lower() for word in ("api_key", "authorization", "secret", "password"))}
    if isinstance(value, (list, tuple)):
        return [public_value(v) for v in value]
    if isinstance(value, str):
        key = os.getenv("OPENAI_API_KEY", "")
        if key:
            value = value.replace(key, "[redacted]")
        return re.sub(r"sk-[A-Za-z0-9_-]+", "[redacted]", value)
    return value


def atomic_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def inside(root, path):
    try:
        root = Path(root).resolve()
        target = (root / path).resolve()
    except (ValueError, OSError):
        raise SessionError(403, "Invalid artifact path") from None
    if not target.is_relative_to(root):
        raise SessionError(403, "Path is outside this session")
    return target


def validate_answer(payload, answer):
    kind = payload["kind"]
    if kind in ("budget", "printer_problem", "print_approval"):
        if not isinstance(answer, str) or not answer.strip():
            raise SessionError(422, f"{kind} requires a text answer")
        word = answer.strip().split()[0].lower().strip(".,!:;")
        allowed = {"budget": ("yes", "y", "no", "n"), "printer_problem": ("retry", "skip")}
        if kind in allowed and word not in allowed[kind]:
            raise SessionError(422, f"Choose one of: {', '.join(allowed[kind])}")
        return answer
    choices = payload.get("choices", [])
    if not isinstance(answer, dict) or answer.get("action") not in choices:
        raise SessionError(422, f"{kind} requires an action from: {', '.join(choices)}")
    if kind == "tuning_printer" and answer["action"] == "printer":
        if not isinstance(answer.get("printer"), str) or not answer["printer"].strip():
            raise SessionError(422, "An exact printer name is required")
    if kind == "report_workflow_problem" and answer["action"] == "correct":
        if not isinstance(answer.get("instructions"), str) or not answer["instructions"].strip():
            raise SessionError(422, "Corrected instructions are required")
    return answer


def prepare(run_dir, state, options, *, resume=False, recovery=False, revision=None):
    """Use the same setup and saved-run recovery as the terminal application."""
    from dotenv import load_dotenv
    load_dotenv(runner.BASE_DIR / ".env")
    from app.printing.connector import trial_print_preflight
    saved = json.loads((run_dir / "config.json").read_text(encoding="utf-8")) if resume else None
    opts = saved or options
    fake = opts.get("fake_printer", True)
    trial_print_preflight(fake_printer=fake)
    model = opts.get("model") or os.getenv("OPENAI_MODEL", "gpt-5-nano")
    specialist = opts.get("specialist_model") or os.getenv("OPENAI_SPECIALIST_MODEL") or None
    if specialist == model:
        specialist = None
    key = os.getenv("OPENAI_API_KEY", "").strip()
    if not key or key == "YOUR_OPENAI_API_KEY_HERE":
        raise ValueError("Set OPENAI_API_KEY in the server environment or .env")
    if resume:
        state, saved = runner.load_resume(run_dir, opts.get("initial_decisions", 10),
                                          resume_stopped_tuning=recovery)
        if revision:
            state = runner.request_saved_report_revision(run_dir, state, revision)
    workflow = runner.prepare_workflow(
        run_dir, state, request=opts["request"],
        model=model, specialist_model=specialist,
        initial_decisions=opts.get("initial_decisions") or 10, extra_decisions=opts.get("extra_decisions") or 5,
        fake_printer=fake, printer_name=opts.get("printer_name") or opts.get("printer") or
        (runner.FAKE_PRINTER_NAME if fake else None),
        cycle_limit=opts.get("max_tuning_cycles") or 3, saved_config=saved, api_key=key,
        fake_jobs_file=run_dir / "fake_printer_jobs.json")
    return workflow, state


class Session:
    def __init__(self, root, record, state):
        self.root, self.record, self.state = root, record, state
        self.lock = threading.RLock()
        self.workflow = None
        self.busy = False
        self.stop_requested = False

    def persist(self):
        atomic_json(self.root / "api_session.json", self.record)

    def public(self):
        return public_value({**{k: self.record.get(k) for k in
            ("session_id", "created_at", "status", "request", "pending", "error", "recovery_required")},
            "run_dir": str(self.root), "dialogue": self.state.get("dialogue", []),
            "execution_results": self.state.get("execution_results", []),
            "stop_reason": self.record.get("stop_reason") or self.state.get("stop_reason"),
            "fake_printer": self.record.get("options", {}).get("fake_printer", False)})


class SessionManager:
    def __init__(self, runs_root, prepare_fn=prepare, max_workers=1):
        self.root = Path(runs_root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.prepare = prepare_fn
        self.pool = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="workflow")
        self.lock = threading.RLock()
        self.sessions = {}
        self.closed = False
        self.discover()

    def discover(self):
        for folder in self.root.iterdir():
            if not folder.is_dir() or folder.is_symlink() or not folder.resolve().is_relative_to(self.root):
                continue
            try:
                state = json.loads((folder / "state.json").read_text(encoding="utf-8"))
                meta = folder / "api_session.json"
                if meta.exists():
                    record = json.loads(meta.read_text(encoding="utf-8"))
                else:
                    options = json.loads((folder / "config.json").read_text(encoding="utf-8"))
                    reason = state.get("stop_reason")
                    record = {"session_id": folder.name, "created_at": folder.name, "options": options,
                              "request": options["request"],
                              "status": {"finished": "finished", "execution_failure": "failed"}.get(reason, "stopped"),
                              "stop_reason": "execution_interrupted" if reason == "running" else reason,
                              "pending": None, "recovery_required": reason != "finished"}
                if record["status"] in ("running", "waiting_for_human"):
                    record.update(status="stopped", stop_reason="backend_interrupted", pending=None,
                                  recovery_required=True, error="Backend restarted. Explicit resume is required; in-flight work may restart.")
                session = Session(folder.resolve(), record, state)
                session.persist()
                self.sessions[record["session_id"]] = session
            except (OSError, ValueError, KeyError):
                continue

    def get(self, sid):
        with self.lock:
            if sid not in self.sessions:
                raise SessionError(404, "Session not found")
            return self.sessions[sid]

    def create(self, options):
        with self.lock:
            if self.closed:
                raise SessionError(503, "Backend is shutting down")
            root, state = runner.create_run(self.root, options["request"], options.get("initial_decisions") or 10,
                                             options.get("paper_color"))
            runner.save_run(state, root)
            record = {"session_id": root.name, "created_at": datetime.now(timezone.utc).isoformat(),
                      "request": options["request"], "options": options, "status": "running", "pending": None}
            session = Session(root, record, state)
            self.sessions[root.name] = session
            with session.lock:
                self._launch(session)
            return root.name

    def _launch(self, session, graph_input=None, *, resume=False, recovery=False, revision=None):
        if session.busy or self.closed:
            raise SessionError(409, "An execution is already active or the backend is closing")
        session.busy = True
        session.stop_requested = False
        session.record.update(status="running", pending=None, error=None, recovery_required=False, stop_reason=None)
        session.persist()  # consume the token durably BEFORE queueing execution
        self.pool.submit(self._work, session, graph_input, resume, recovery, revision)

    def _work(self, session, graph_input, resume, recovery, revision):
        from langgraph.types import Command
        try:
            if session.workflow is None or resume:
                workflow, state = self.prepare(session.root, deepcopy(session.state), session.record["options"],
                                                resume=resume, recovery=recovery, revision=revision)
                with session.lock:
                    session.workflow, session.state = workflow, state
                graph_input = state
            elif graph_input is not None:
                graph_input = Command(resume=graph_input)
            config = {"configurable": {"thread_id": session.root.name}, "recursion_limit": 200}
            for snapshot in session.workflow.stream(graph_input, config, stream_mode="values"):
                state = {k: v for k, v in snapshot.items() if k != "__interrupt__"}
                with session.lock:
                    session.state = deepcopy(state)
                    runner.save_run(state, session.root)
                    if session.stop_requested:
                        session.record.update(status="stopped", stop_reason="operator_paused", recovery_required=True)
                        session.workflow = None
                        session.persist()
                        return
            pending = session.workflow.get_state(config)
            with session.lock:
                if pending.next:
                    payload = deepcopy(pending.tasks[0].interrupts[0].value)
                    payload["interrupt_token"] = uuid4().hex
                    session.record.update(status="waiting_for_human", pending=payload, uploads=[])
                else:
                    reason = session.state.get("stop_reason", "unknown")
                    session.record.update(status="finished" if reason == "finished" else "stopped",
                                          pending=None, stop_reason=reason)
                session.persist()
        except Exception as error:
            # Provider exception strings can contain credentials/request bodies. Expose safe diagnostics only.
            message = f"{type(error).__name__}: workflow execution failed. Check server configuration and resume explicitly."
            from app.printing.connector import PrinterError
            if isinstance(error, (ValueError, PrinterError)):
                message = public_value(str(error))
            with session.lock:
                session.record.update(status="failed", pending=None, error=message, recovery_required=True,
                                      stop_reason="execution_failure")
                session.workflow = None
                session.persist()
        finally:
            with session.lock:
                session.busy = False

    def pending(self, session, token):
        payload = session.record.get("pending")
        if session.busy or session.record["status"] != "waiting_for_human" or not payload:
            raise SessionError(409, "No interrupt is currently waiting")
        if payload["interrupt_token"] != token:
            raise SessionError(409, "Stale interrupt token; refresh the session")
        return payload

    def answer(self, sid, token, answer):
        session = self.get(sid)
        with session.lock:
            payload = self.pending(session, token)
            answer = validate_answer(payload, answer)
            if payload["kind"] == "tuning_scan" and answer["action"] == "scan":
                paths = answer.get("scan_paths")
                if not isinstance(paths, list) or not paths or not all(isinstance(p, str) for p in paths):
                    raise SessionError(422, "Select uploaded scans explicitly")
                uploaded = session.record.get("uploads", [])
                if any(p not in uploaded or not inside(session.root, p).is_file() for p in paths):
                    raise SessionError(422, "Select files uploaded for this interrupt only")
            self._launch(session, deepcopy(answer))

    def resume(self, sid, recovery=False, revision=None):
        session = self.get(sid)
        with session.lock:
            if session.busy or session.record["status"] in ("running", "waiting_for_human"):
                raise SessionError(409, "Session is active; answer its current interrupt")
            if session.record["status"] == "finished" and not revision:
                raise SessionError(409, "Session already finished; provide an explicit report revision")
            path = session.root / "tuning_session.json"
            if path.exists():
                from app.workflow.tuning import load_tuning_session
                if load_tuning_session(path)["status"] == "stopped" and not recovery:
                    raise SessionError(409, "Human-stopped tuning requires explicit recover_stopped_tuning")
            session.workflow = None
            self._launch(session, resume=(session.root / "config.json").exists(), recovery=recovery, revision=revision)

    def close(self):
        with self.lock:
            self.closed = True
            for session in self.sessions.values():
                with session.lock:
                    session.stop_requested = True
        self.pool.shutdown(wait=True, cancel_futures=True)
