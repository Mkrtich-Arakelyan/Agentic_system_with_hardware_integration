"""Streamlit UI for the agentic print-tuning workflow.

Run with:  streamlit run frontend/streamlit_app.py
The backend (real or frontend/mock_api.py) is at API_URL (default http://localhost:8000).
While the agents work the page polls GET /api/sessions/{id} every 2 seconds; when the run
pauses, the pending question is shown as a card with the controls its kind needs.
"""
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

import streamlit as st  # noqa: E402

from api_client import SCAN_FORMATS, ApiClient, ApiError, find_preview_files, relative_run_path  # noqa: E402

DEFAULT_REQUEST = (
    "Compare at least four focusing experiments, changing one setting at a time while "
    "holding other settings fixed. Have the scientist select four experiments that "
    "answer the scientific question, requesting more runs if needed. "
    "Judge whether the comparison is fair, compare "
    "performance and computational cost, and suggest the next experiment. Then "
    "produce a one-page report with one graph, its explanation and a verdict, and "
    "print it on the office printer after my approval."
)
FAKE_PRINTER_OPTION = "Fake printer (Office_Printer)"
POLL_SECONDS = 2
STATUS_BADGES = {"running": ("🔄", "blue"), "waiting_for_human": ("✋", "orange"), "finished": ("✅", "green"),
                 "stopped": ("⏹️", "gray"), "failed": ("❌", "red")}
ROLE_AVATARS = {"orchestrator": "🧭", "scientist": "🔬", "executor": "🛠️", "general": "📋",
                "photo_quality_specialist": "📷", "human": "🙋", "user": "🙋", "system": "⚙️"}
ROLE_NAMES = {"photo_quality_specialist": "photo quality specialist", "user": "human (request)"}
CHOICE_LABELS = {"approve": "Approve", "stop": "Stop", "printed": "Printed", "not_printed": "Not printed",
                 "check": "Check again", "retry": "Retry", "regenerate": "Regenerate", "reprint": "Reprint",
                 "skip": "Skip printing", "correct": "Correct instructions", "scan": "Use scans",
                 "printer": "Use printer"}
KIND_TITLES = {"budget": "Continue the run?", "print_approval": "Approve printing the report",
               "report_workflow_problem": "The report workflow stalled",
               "report_print_confirmation": "Did the report print?", "printer_problem": "Printer problem",
               "tuning_printer": "Which printer prints the trial sheets?",
               "trial_print_approval": "Approve printing this trial sheet",
               "tuning_scan": "Upload the scans", "tuning_print_confirmation": "Did the trial sheet print?",
               "tuning_poll": "The trial is still printing", "tuning_problem": "Print-tuning problem"}

st.set_page_config(page_title="Print tuning", page_icon="🖨️", layout="wide")


@st.cache_resource
def get_client() -> ApiClient:
    return ApiClient()


client = get_client()
state = st.session_state
state.setdefault("session_id", None)
state.setdefault("outbox", None)          # the answer or upload being sent in this run
state.setdefault("answer_error", None)
state.setdefault("uploads", {})            # (session, round, trial) -> uploaded scan paths


@st.cache_data(ttl=30, show_spinner=False)
def load_printers(base_url: str) -> tuple[list[str], str | None]:
    try:
        return get_client().printers(), None
    except ApiError as error:
        return [], str(error)


def signature(session: dict) -> tuple:
    """What the page shows; polling reruns the page only when this changes."""
    return (session.get("status"), len(session.get("dialogue") or []), len(session.get("execution_results") or []),
            json.dumps(session.get("pending"), sort_keys=True, default=str))


# ---------- sending answers ----------

def pending_key(pending) -> str:
    return json.dumps(pending, sort_keys=True, default=str)


def queue_answer(answer):
    # Bound to the question on screen, so it is never sent to a different question that arrived meanwhile.
    state.outbox = {"type": "answer", "answer": answer, "for": state.get("shown_pending")}
    state.answer_error = None


def queue_text(key: str, prefix: str = "", required_message: str | None = None, empty_answer=None):
    """Queue the text of a widget (optionally after prefix); refuse it locally if it is required but empty."""
    text = (state.get(key) or "").strip()
    if not text and required_message:
        state.answer_error = required_message
        return
    queue_answer(f"{prefix} {text}".strip() if text else (empty_answer if empty_answer is not None else prefix))


def queue_upload(uploader_key: str, upload_key: tuple):
    files = state.get(uploader_key) or []
    if not files:
        state.answer_error = "Choose at least one scan file first."
        return
    state.outbox = {"type": "upload", "key": upload_key,
                    "files": [(f.name, f.getvalue(), f.type) for f in files]}
    state.answer_error = None


def queue_scans(upload_key: tuple):
    queue_answer({"action": "scan", "scan_paths": state.uploads.get(upload_key, [])})


def process_outbox(session_id: str, session: dict):
    job = state.outbox
    if not job:
        return
    if job["type"] == "answer" and job.get("for") != pending_key(session.get("pending")):
        state.outbox = None
        state.answer_error = "The question changed before your answer was sent, so nothing was sent."
        st.rerun()
    with st.spinner("Sending your answer..." if job["type"] == "answer" else "Uploading scans..."):
        try:
            if job["type"] == "answer":
                client.answer(session_id, job["answer"])
            else:
                state.uploads[job["key"]] = client.upload_scans(session_id, job["files"])
        except ApiError as error:
            state.answer_error = str(error)
    state.outbox = None
    st.rerun()


# ---------- files ----------

def show_file(session_id: str, relative: str, key: str):
    name = relative.rsplit("/", 1)[-1]
    try:
        content = client.download_file(session_id, relative)
    except ApiError as error:
        st.warning(f"{relative}: {error}")
        return
    if name.lower().endswith((".png", ".jpg", ".jpeg")):
        st.image(content, caption=relative, width="stretch")
    else:
        st.markdown(f"📄 **{relative}** ({len(content) / 1024:.0f} KB)")
        if hasattr(st, "pdf"):
            try:
                st.pdf(content, height=520, key=f"pdf-{key}")
            except Exception:      # st.pdf needs the optional streamlit[pdf] extra
                pass
    left, right = st.columns(2)
    left.download_button("Download", content, file_name=name, key=f"dl-{key}", width="stretch")
    right.link_button("Open in browser", client.file_url(session_id, relative), width="stretch")


def show_files(session_id: str, paths: list[str], prefix: str):
    if not paths:
        return
    columns = st.columns(min(len(paths), 3))
    for index, relative in enumerate(paths):
        with columns[index % len(columns)]:
            show_file(session_id, relative, f"{prefix}-{index}-{relative}")


# ---------- the pending question ----------

def choice_buttons(choices, key: str, busy: bool, answer_for=lambda choice: {"action": choice}, skip=()):
    shown = [choice for choice in choices if choice not in skip]
    if not shown:
        return
    for column, choice in zip(st.columns(len(shown)), shown):
        column.button(CHOICE_LABELS.get(choice, choice), key=f"{key}-{choice}", disabled=busy,
                      type="secondary" if choice in ("stop", "skip", "not_printed") else "primary",
                      on_click=queue_answer, args=(answer_for(choice),), width="stretch")


def facts(rows: dict):
    st.markdown("\n".join(f"- **{label}:** `{value}`" for label, value in rows.items()
                          if value not in (None, "")))


def pending_card(session: dict, busy: bool):
    payload = session["pending"]
    session_id, run_dir = session["session_id"], session.get("run_dir")
    kind = payload.get("kind", "unknown")
    key = f"{session_id}-{len(session.get('dialogue') or [])}-{kind}"
    choices = payload.get("choices") or []
    state.shown_pending = pending_key(payload)
    with st.container(border=True):
        st.markdown(f"### ✋ {KIND_TITLES.get(kind, kind.replace('_', ' ').capitalize())}")
        st.info(payload.get("question") or "The workflow is waiting for your answer.")
        if payload.get("message"):
            st.warning(payload["message"])

        if kind == "budget":
            st.markdown(payload.get("summary") or "")
            st.caption(f"Orchestrator decisions used: {payload.get('decisions_used')} · "
                       f"continuing adds {payload.get('extra')}")
            st.text_input("Guidance for the agents (optional)", key=f"{key}-guidance", disabled=busy)
            left, right = st.columns(2)
            left.button("Continue", key=f"{key}-yes", type="primary", disabled=busy, width="stretch",
                        on_click=queue_text, args=(f"{key}-guidance", "yes"))
            right.button("Stop", key=f"{key}-no", disabled=busy, width="stretch",
                         on_click=queue_answer, args=("no",))

        elif kind == "print_approval":
            facts({"Report": payload.get("pdf_path"), "Pages": payload.get("pages"),
                   "Printer": payload.get("printer") or "the default printer", "Copies": payload.get("copies"),
                   "PDF sha256": payload.get("pdf_sha256")})
            show_files(session_id, find_preview_files([payload.get("pdf_path")], run_dir), f"{key}-pdf")
            st.button("Approve and print", key=f"{key}-yes", type="primary", disabled=busy,
                      on_click=queue_answer, args=("yes",))
            st.text_area("Or describe what to change in the report", key=f"{key}-change", disabled=busy)
            st.button("Request this change", key=f"{key}-send-change", disabled=busy, on_click=queue_text,
                      args=(f"{key}-change",), kwargs={"required_message": "Describe the change first."})

        elif kind == "report_workflow_problem":
            choice_buttons(choices, key, busy, skip=("correct",))
            if "correct" in choices:
                st.text_area("Corrected revision instructions", key=f"{key}-instructions", disabled=busy)
                st.button("Correct instructions", key=f"{key}-correct", disabled=busy, on_click=queue_corrected,
                          args=(f"{key}-instructions",))

        elif kind == "printer_problem":
            with st.expander("Printer reading", expanded=True):
                st.json(payload.get("reading") or {})
            choice_buttons(choices or ["retry", "skip"], key, busy, answer_for=lambda choice: choice)

        elif kind == "report_print_confirmation":
            facts({"Job": payload.get("job_id")})
            with st.expander("Printer reading"):
                st.json(payload.get("reading") or {})
            choice_buttons(choices, key, busy)

        elif kind == "tuning_printer":
            printers, error = load_printers(client.base_url)
            if error:
                st.warning(f"Could not load the printer list: {error}")
            options = printers + ["Other (type the name)"]
            picked = st.selectbox("Printer", options, key=f"{key}-printer", disabled=busy)
            name = st.text_input("Exact printer name", key=f"{key}-other", disabled=busy) \
                if picked == options[-1] else picked
            left, right = st.columns(2)
            left.button("Use this printer", key=f"{key}-use", type="primary", width="stretch",
                        disabled=busy or not (name or "").strip(), on_click=queue_answer,
                        args=({"action": "printer", "printer": (name or "").strip()},))
            right.button("Stop", key=f"{key}-stop", disabled=busy, width="stretch",
                         on_click=queue_answer, args=({"action": "stop"},))

        elif kind == "trial_print_approval":
            facts({"Round": payload.get("round_id"), "Trial": payload.get("trial_id"),
                   "PDF": payload.get("pdf_path"), "Pages": payload.get("page_count"),
                   "Printer": payload.get("printer"), "PDF sha256": payload.get("pdf_sha256")})
            if payload.get("print_settings"):
                st.markdown("**Print settings**")
                st.table({"setting": list(payload["print_settings"]),
                          "value": [str(v) for v in payload["print_settings"].values()]})
            show_files(session_id, find_preview_files([payload.get("pdf_path")], run_dir), f"{key}-pdf")
            choice_buttons(choices, key, busy)

        elif kind == "tuning_scan":
            round_id = payload.get("round_id")
            facts({"Round": round_id, "Trial": payload.get("trial_id"),
                   "Scan folder": payload.get("suggested_scan_directory")})
            upload_key = (session_id, round_id, payload.get("trial_id"))
            uploaded = state.uploads.get(upload_key)
            uploader_key = f"{key}-files"
            st.file_uploader("Scan images or PDFs (both faces of every sheet)", type=list(SCAN_FORMATS),
                             accept_multiple_files=True, key=uploader_key, disabled=busy)
            st.button("Upload", key=f"{key}-upload", disabled=busy, on_click=queue_upload,
                      args=(uploader_key, upload_key))
            if uploaded:
                st.success(f"Uploaded {len(uploaded)} file(s):")
                st.code("\n".join(relative_run_path(p, run_dir) or p for p in uploaded), language=None)
                st.button(f"Use these {len(uploaded)} scans", key=f"{key}-scan", type="primary", disabled=busy,
                          on_click=queue_scans, args=(upload_key,))
            choice_buttons(choices, key, busy, skip=("scan",))

        else:   # tuning_print_confirmation, tuning_poll, tuning_problem and any unknown kind
            if kind not in KIND_TITLES:
                st.json(payload)
            if choices:
                choice_buttons(choices, key, busy)
            else:
                st.text_input("Your answer", key=f"{key}-text", disabled=busy)
                st.button("Send", key=f"{key}-send", disabled=busy, on_click=queue_text, args=(f"{key}-text",),
                          kwargs={"required_message": "Type an answer first."})

        if state.answer_error:
            st.error(state.answer_error)


def queue_corrected(key: str):
    text = (state.get(key) or "").strip()
    if not text:
        state.answer_error = "Write the corrected instructions first."
        return
    queue_answer({"action": "correct", "instructions": text})


# ---------- sidebar ----------

def sidebar():
    with st.sidebar:
        st.header("🖨️ Print tuning")
        st.caption(f"Backend: {client.base_url}")
        with st.expander("Start a new run", expanded=state.session_id is None):
            printers, error = load_printers(client.base_url)
            if error:
                st.caption(f"Printer list unavailable: {error}")
            with st.form("new-run"):
                request = st.text_area("Request", DEFAULT_REQUEST, height=180)
                model = st.text_input("Model", placeholder="backend default")
                specialist = st.text_input("Specialist model", placeholder="same as the model")
                printer = st.selectbox("Printer", [FAKE_PRINTER_OPTION, *printers])
                paper = st.text_input("Paper color", placeholder="plain white")
                left, right = st.columns(2)
                initial = left.number_input("Initial decisions", min_value=1, value=10, step=1)
                extra = right.number_input("Extra decisions", min_value=1, value=5, step=1)
                if st.form_submit_button("Start run", type="primary", width="stretch"):
                    fake = printer == FAKE_PRINTER_OPTION
                    try:
                        state.session_id = client.start_session(
                            request=request, model=model.strip() or None, specialist_model=specialist.strip() or None,
                            printer=None if fake else printer, fake_printer=fake, paper_color=paper.strip() or None,
                            initial_decisions=int(initial), extra_decisions=int(extra))
                        state.answer_error = None
                        st.rerun()
                    except ApiError as start_error:
                        st.error(str(start_error))

        st.subheader("Sessions")
        if st.button("Refresh list", width="stretch"):
            st.rerun()
        try:
            sessions = client.list_sessions()
        except ApiError as list_error:
            st.error(str(list_error))
            return
        if not sessions:
            st.caption("No sessions yet.")
        for row in sessions:
            icon = STATUS_BADGES.get(row.get("status"), ("❔", "gray"))[0]
            label = f"{icon} {row.get('created_at', '')[:16].replace('T', ' ')} · {(row.get('request') or '')[:38]}"
            current = row["session_id"] == state.session_id
            if st.button(label, key=f"open-{row['session_id']}", width="stretch",
                         type="primary" if current else "secondary", help=row.get("status")):
                state.session_id, state.answer_error = row["session_id"], None
                st.rerun()


# ---------- main ----------

@st.fragment(run_every=POLL_SECONDS)
def watch(session_id: str, shown: tuple):
    """Poll the session; rerun the whole page only when what it shows has changed.

    Typed text and chosen files survive: without a change only this (empty) fragment reruns.
    """
    try:
        latest = client.get_session(session_id)
    except ApiError:
        return
    if signature(latest) != shown:
        st.rerun(scope="app")


def main():
    sidebar()
    session_id = state.session_id
    if not session_id:
        st.title("Agentic print tuning")
        st.markdown("Start a new run or open a session in the sidebar.")
        return
    try:
        session = client.get_session(session_id)
    except ApiError as error:
        st.error(str(error))
        return

    status = session.get("status", "unknown")
    icon, color = STATUS_BADGES.get(status, ("❔", "gray"))
    st.title("Agentic print tuning")
    st.markdown(f"**Session** `{session_id}` &nbsp; :{color}-background[{icon} {status.replace('_', ' ')}]"
                + (f" &nbsp; stop reason: `{session.get('stop_reason')}`"
                   if status in ("finished", "stopped", "failed") else ""))
    st.caption(f"Run folder: {session.get('run_dir')}")

    busy = state.outbox is not None
    if status == "waiting_for_human" and session.get("pending"):
        pending_card(session, busy)
    elif state.answer_error:          # the answer failed and the question has gone meanwhile
        st.error(state.answer_error)
        state.answer_error = None
    if status == "running":
        st.info(f"🔄 The agents are working. This page refreshes every {POLL_SECONDS} s.")
    if status in ("running", "waiting_for_human") and not busy:
        # Also while waiting: the question may be answered in another tab, or the backend may move on.
        watch(session_id, signature(session))
    process_outbox(session_id, session)

    run_dir = session.get("run_dir")
    results = session.get("execution_results") or []
    pending_files = find_preview_files(session.get("pending") or {}, run_dir)
    recent = [path for path in find_preview_files(list(reversed(results[-6:])), run_dir, limit=4)
              if path not in pending_files]
    if recent:
        with st.expander(f"Latest files ({len(recent)})", expanded=status != "waiting_for_human"):
            show_files(session_id, recent, f"{session_id}-recent-{len(results)}")

    chat, tools = st.columns([3, 2])
    with chat:
        st.subheader("Dialogue")
        for turn in session.get("dialogue") or []:
            role = turn.get("role", "system")
            with st.chat_message(ROLE_NAMES.get(role, role), avatar=ROLE_AVATARS.get(role, "💬")):
                st.markdown(f"**{ROLE_NAMES.get(role, role)}**")
                st.text(turn.get("text", ""))
    with tools:
        st.subheader(f"Tool executions ({len(results)})")
        for number, record in reversed(list(enumerate(results, 1))):
            result = record.get("result")
            failed = isinstance(result, dict) and (result.get("status") == "error"
                                                   or result.get("exit_code") not in (None, 0))
            with st.expander(f"{'⚠️' if failed else '🔧'} {number}. {record.get('tool_name', 'tool')}"):
                st.json(record, expanded=2)


main()
