# Print-tuning frontend

A Streamlit UI for the agentic print-tuning workflow, plus an in-memory mock of the backend API so
the UI can be built and tested offline. The real backend is `app.api.main:app`;
see [project setup and limitations](../README.md). Nothing in the UI imports `app/`.

| File | What it is |
|---|---|
| `streamlit_app.py` | The UI: start runs, open sessions, follow the dialogue and tool executions, answer pauses. |
| `api_client.py` | Small `requests` client for the API contract below; every failure becomes an `ApiError` with a readable message. |
| `mock_api.py` | FastAPI mock of the contract. Each session walks through the scripted pauses below with fake dialogue, tool results and sample files. |
| `tests/` | `api_client` against the mock (FastAPI `TestClient`), and page smoke tests (Streamlit `AppTest`). |

## Run

From the repository root, with the project venv active:

```bash
python -m pip install -r requirements.txt -r frontend/requirements.txt

# terminal 1: real backend (single process; no auto-reload during runs)
python -m uvicorn app.api.main:app --host 127.0.0.1 --port 8000

# terminal 2: the UI on http://localhost:8501
python -m streamlit run frontend/streamlit_app.py
```

For offline UI development use `python -m uvicorn frontend.mock_api:app --port 8000`
instead. The mock is scripted and in-memory; it is not a workflow persistence test.

To use another backend, set `API_URL`, for example
`API_URL=http://localhost:9000 streamlit run frontend/streamlit_app.py`.

Mock settings (environment variables):

- `MOCK_STEP_SECONDS` (default `1.5`): how long the fake agents "work" between pauses. The session
  shows `running` meanwhile, and the UI polls every 2 s.
- `MOCK_RUNS_DIR` (default: a new temporary folder): where the mock writes the run folders.
- `CORS_ORIGINS` (default `http://localhost:8501,http://localhost:5173`).
- A request text containing `[fail]` makes the run fail after the experiments, so you can see the failed state.

PDFs are offered as **Download** and **Open in browser**. For an inline viewer, also install
`pip install "streamlit[pdf]"`; the page uses `st.pdf` automatically when it is available.

The project `.streamlit/config.toml` uses Streamlit's supported
`client.toolbarMode="minimal"` setting to hide
developer/deploy/menu options. Narrow CSS targets only the framework's
`stStatusWidget` (walking/swimming run indicator) and `stAppDeployButton`; application
status cards, buttons, images and headers remain visible. Status is static text,
and polling still refreshes the application only when session content changes.
Initial/additional decision-budget inputs are not shown. Session creation omits
those optional fields, retaining the backend's existing defaults (10 initially,
5 additional) and human budget-review prompts. Model, printer and paper settings
remain available.

Framework chrome is not rendered by Streamlit's AppTest. After a Streamlit upgrade,
verify in a browser that Deploy/menu controls and the run animation stay hidden,
that unsubmitted form text survives polling, and that sidebar controls, scan upload,
approvals and report downloads remain visible.
Restart only Streamlit after changing project configuration; leave an active backend
running. The page also sets toolbar mode for launches from another working directory.
Project server configuration also enables headless mode and binds to `127.0.0.1`,
avoiding the first-run email prompt when launched from the repository root.

Tests:

```bash
python -m pytest frontend/tests
```

## The mock run

Every mock session goes through these pauses, in this order:

budget → tuning_printer → trial_print_approval (round 1) → tuning_print_confirmation →
tuning_scan (round 1) → trial_print_approval (round 2) → tuning_poll → tuning_scan (round 2) →
tuning_problem → trial_print_approval (round 3) → tuning_scan (round 3) → report_workflow_problem →
print_approval → printer_problem → report_print_confirmation → finished.

Your answers branch the run the same way the real graph does:

- `stop` (or `no` at budget) stops the run.
- `not_printed` or `reprint` asks for a fresh trial approval.
- A change request at print_approval leads to a `report_workflow_problem` that offers `correct`.
- `skip` at printer_problem finishes without printing.

## API contract

Both adapters use these shapes. Error bodies are FastAPI's `{"detail": "..."}`.
New real sessions default to `fake_printer=true`. Model keys belong only in the
server environment, never in a session request. The real adapter also accepts
`max_tuning_cycles` (1–20, default 3).

| Method and path | Body | Response |
|---|---|---|
| `POST /api/sessions` | `{request, model, specialist_model, printer, fake_printer, paper_color, initial_decisions, extra_decisions}` | `{session_id}` |
| `GET /api/sessions` | | `[{session_id, created_at, status, request}]` |
| `GET /api/sessions/{id}` | | `{session_id, status, run_dir, dialogue: [{role, text}], execution_results: [{tool_name, tool_args, result}], pending, stop_reason}` |
| `POST /api/sessions/{id}/answer` | `{interrupt_token, answer}` (text or object) | `{ok}`; 409 for busy/stale/duplicate answers; 422 for invalid answers |
| `POST /api/sessions/{id}/scans` | multipart `interrupt_token` and `files` (png, jpg, jpeg, pdf) | `{paths}` saved under the active `scans/cycle_N/step_N` (legacy `round_N` supported) |
| `POST /api/sessions/{id}/resume` | `{recover_stopped_tuning: false, report_revision: null}` | `{ok}`; explicit restart recovery, or completed-report revision (real backend) |
| `POST /api/sessions/{id}/stop` | | `{ok, message}`; cooperative pause after the active node (real backend) |
| `GET /api/sessions/{id}/files/{path}` | | the file; `path` is relative to `run_dir`, and anything outside it gets a 403 |
| `GET /api/printers` | query `fake_printer=true` (default); false for hardware discovery | `{printers: [...]}` |

`status` is one of `running`, `waiting_for_human`, `finished`, `stopped`, `failed`.
The real API uses polling only; `/events` is not implemented (the mock retains its
optional scripted SSE demo). Session responses also include `error`,
`recovery_required`, and `fake_printer`. After backend restart, unfinished sessions
become stopped/recoverable; no printing resumes automatically.

Every `pending` object includes a stable `interrupt_token`. Bind the answer and
upload to the token **shown to the human**, never fetch a fresh token when retrying
an old action. Token consumption is atomic and durable. The UI passes it explicitly;
the client's omitted-token convenience is for synchronous scripting only. Uploading
never answers an interrupt. Only paths uploaded against that token can be selected
for `tuning_scan`; old cycle/step files cannot be attached automatically.

### Pauses (`pending.kind`) and their answers

| kind | payload fields | answer |
|---|---|---|
| `budget` | summary, decisions_used, extra, question | text: `"yes"` (optionally followed by guidance) or `"no"` |
| `print_approval` | pdf_path, pages, printer, copies, pdf_sha256, question | text: `"yes"`, or a description of the change |
| `report_workflow_problem` | message, question, choices `[retry, stop]` or `[retry, correct, stop]` | `{action}`; `correct` also needs `{instructions}` |
| `report_print_confirmation` | job_id, reading, question, choices `[printed, not_printed, stop]` | `{action}` |
| `printer_problem` | reading, question, choices `[retry, skip]` | text: `"retry"` or `"skip"` |
| `tuning_printer` | question, choices `[printer, stop]` | `{action: "printer", printer: <name>}` or `{action: "stop"}` |
| `trial_print_approval` | pdf_path, page_count, printer, session_id, round_id, trial_id, pdf_sha256, print_settings, question, choices `[approve, stop]` | `{action}` |
| `tuning_scan` | round_id, trial_id, suggested_scan_directory, question, choices `[scan, reprint, stop]` | `{action: "scan", scan_paths: <paths from /scans>}`, `{action: "reprint"}` or `{action: "stop"}` |
| `tuning_print_confirmation` | attempt_id, question, choices `[printed, not_printed, stop]` | `{action}` |
| `tuning_poll` | question, choices `[check, stop]` | `{action}` |
| `tuning_problem` | message, question, choices `[retry, regenerate, stop]` | `{action}` |
| `tuning_cycle` | cycle_id, metric_summary, choices `[continue, finish, stop]` (continue omitted at limit) | `{action}`; continue is not print approval |
| any other kind | shown as JSON | `{action}` with one of its `choices` |

Two variants found in `app/workflow/graph.py` that the UI also handles:

- `tuning_problem` without a message, with choices `[reprint, stop]`. This is the "trial did not print" recovery.
- `print_approval` also accepts the text `"retry"`. It is not an approval: the graph asks again.
