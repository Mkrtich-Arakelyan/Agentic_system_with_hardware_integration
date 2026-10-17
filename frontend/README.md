# Print-tuning frontend

A Streamlit UI for the agentic print-tuning workflow, plus an in-memory mock of the backend API so
the UI can be built and tested before the real FastAPI backend exists. Nothing here imports `app/`.

| File | What it is |
|---|---|
| `streamlit_app.py` | The UI: start runs, open sessions, follow the dialogue and tool executions, answer pauses. |
| `api_client.py` | Small `requests` client for the API contract below; every failure becomes an `ApiError` with a readable message. |
| `mock_api.py` | FastAPI mock of the same contract. Each session walks through every pause kind with fake dialogue, tool results and sample files. |
| `tests/` | `api_client` against the mock (FastAPI `TestClient`), and page smoke tests (Streamlit `AppTest`). |

## Run

From the repository root, with the project venv active:

```bash
pip install -r frontend/requirements.txt

# terminal 1: the mock backend on http://localhost:8000
uvicorn frontend.mock_api:app --port 8000

# terminal 2: the UI on http://localhost:8501
streamlit run frontend/streamlit_app.py
```

To use another backend (later, the real one), set `API_URL`, for example
`API_URL=http://localhost:9000 streamlit run frontend/streamlit_app.py`.

Mock settings (environment variables):

- `MOCK_STEP_SECONDS` (default `1.5`): how long the fake agents "work" between pauses. The session
  shows `running` meanwhile, and the UI polls every 2 s.
- `MOCK_RUNS_DIR` (default: a new temporary folder): where the mock writes the run folders.
- `CORS_ORIGINS` (default `http://localhost:8501,http://localhost:5173`).
- A request text containing `[fail]` makes the run fail after the experiments, so you can see the failed state.

PDFs are offered as **Download** and **Open in browser**. For an inline viewer, also install
`pip install "streamlit[pdf]"`; the page uses `st.pdf` automatically when it is available.

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

The real backend will implement exactly this. Error bodies are FastAPI's `{"detail": "..."}`.

| Method and path | Body | Response |
|---|---|---|
| `POST /api/sessions` | `{request, model, specialist_model, printer, fake_printer, paper_color, initial_decisions, extra_decisions}` | `{session_id}` |
| `GET /api/sessions` | | `[{session_id, created_at, status, request}]` |
| `GET /api/sessions/{id}` | | `{session_id, status, run_dir, dialogue: [{role, text}], execution_results: [{tool_name, tool_args, result}], pending, stop_reason}` |
| `GET /api/sessions/{id}/events` | | server-sent events `dialogue`, `tool`, `interrupt`, `end` (the UI polls instead) |
| `POST /api/sessions/{id}/answer` | `{answer}` (text or object) | `{ok}`; 409 if nothing is pending, 422 with a message if the answer is invalid |
| `POST /api/sessions/{id}/scans` | multipart `files` (png, jpg, jpeg, pdf) | `{paths}` saved in `run_dir/scans/round_N` of the active round |
| `GET /api/sessions/{id}/files/{path}` | | the file; `path` is relative to `run_dir`, and anything outside it gets a 403 |
| `GET /api/printers` | | `{printers: [...]}` |

`status` is one of `running`, `waiting_for_human`, `finished`, `stopped`, `failed`.

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
| any other kind | shown as JSON | `{action}` with one of its `choices` |

Two variants found in `app/workflow/graph.py` that the UI also handles:

- `tuning_problem` without a message, with choices `[reprint, stop]`. This is the "trial did not print" recovery.
- `print_approval` also accepts the text `"retry"`. It is not an approval: the graph asks again.
