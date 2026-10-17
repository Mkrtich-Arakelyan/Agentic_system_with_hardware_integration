# Agent workflow with Streamlit and FastAPI

The terminal and web adapters run the same LangGraph workflow. FastAPI does not
implement experiments, tuning, reports, or print decisions. `runner.prepare_workflow`
constructs the existing models, tools, bench, and graph for both adapters.

Tested with Python 3.14.4. For a fresh checkout, first create the environment with
your installed Python (`python -m venv .venv`). From the repository root, activate
it and install dependencies:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt -r requirements-dev.txt -r frontend/requirements.txt
```

For a fresh checkout, copy `.env.example` to `.env` if `.env` does not already
exist, then set `OPENAI_API_KEY` there. Never commit `.env` or overwrite an existing
credential file. Restart the backend after changing its environment configuration.

Terminal 1 (one process, **no `--reload` or additional workers** during active runs):

```powershell
python -m uvicorn app.api.main:app --host 127.0.0.1 --port 8000
```

Terminal 2 (activate the same environment):

```powershell
python -m streamlit run frontend/streamlit_app.py
```

Open http://localhost:8501. The UI defaults to `API_URL=http://localhost:8000`.
Project Streamlit configuration binds the frontend to localhost and disables the
first-run email prompt, so the plain command above starts without interactive setup.
For a different address, set `$env:API_URL = "http://127.0.0.1:8000"` in terminal 2
before launching Streamlit. Polling is the supported transport; the real API
does not provide an `/events` endpoint.

The backend loads the repository `.env` using the terminal's conventions:
`OPENAI_API_KEY`, `OPENAI_MODEL`, `OPENAI_SPECIALIST_MODEL`, and, for real Windows
trial printing, `SUMATRA_PATH`. Credentials remain on the server. Normal web runs
use paid models, including fake-printer runs. Offline tests never use paid models.

Optional server settings:

| Variable | Default / meaning |
| --- | --- |
| `API_RUNS_DIR` | Repository `runs`; persistent run and API records |
| `API_UPLOAD_MAX_BYTES` | 52428800; aggregate scan content limit per request (multipart envelope allows 1 MiB extra) |
| `CORS_ORIGINS` | `http://localhost:8501,http://127.0.0.1:8501` |

New sessions default to **fake printing**. To use hardware, explicitly enable
real printer selection in the sidebar and select its exact name. Printer discovery
only reads status. Existing exact-PDF/hash approval, effective trial settings,
intent-before-submission journals, uncertain-outcome questions and reprint limits
remain in the graph. Starting, uploading, continuing a cycle, retrying or viewing
a report is never print approval. Server configuration determines which printers
are available; this is a trusted localhost prototype, not an authenticated service.

At a scan pause, upload PDF/PNG/JPG/JPEG files containing both faces of the active
trial. Upload does **not** resume the graph. Review the displayed uploaded files,
then explicitly choose **Use these scans**. The API validates contents and stores
unique originals under that run's `scans/cycle_N/step_N` (legacy `round_N` paths
remain supported). The existing scan evaluator performs alignment and measurement.
Continue/Finish/Stop at a cycle boundary retain the accepted mappings.

For a fake-printer run, synthetic scans can be created in a second terminal using
the active trial PDF displayed in the UI (choose a fresh output folder):

```powershell
python -c "from app.print_tuning.fake_scan import fake_scan; print(*fake_scan(r'FULL_ACTIVE_TRIAL_PDF_PATH', r'output/synthetic_scan_attempt_1', noise=0.004, seed=42), sep='\n')"
```

Upload those generated PNGs through the UI. They carry synthetic provenance;
they do not measure physical print quality. Synthetic scans are never generated
automatically or substituted for a real scan.

Trial/report PDFs and image artifacts are downloaded through session-scoped API
file URLs with `Cache-Control: no-store`. Report change requests at the approval
card enter the existing revision journal and report builder. Old report versions
are archived, previous approval is invalidated, and the new PDF requires review
and approval. A completed session can also be reopened with explicit report
revision instructions in **Resume / recover / revise report**.

API state is persisted in `api_session.json`; the existing `state.json`, config,
tuning sessions and report/print journals remain authoritative. A lock and durable
token claim prevent simultaneous answers/resumes for a session. A single managed
worker executes blocking graph calls outside HTTP/event-loop threads; other runs
queue behind it, also avoiding concurrent plotting and fake-queue writes. The
frontend binds actions and uploads to the displayed token and never automatically
resends answers on polling.

After backend restart, unfinished runs are listed as `stopped` with
`backend_interrupted` and require **explicit resume**. Old interrupt tokens are
invalid. No worker or print starts during discovery. Resume calls the terminal's
existing `load_resume` and reconstructs the graph from saved evidence. Pending or
uncertain print intents stay in their journals and are reconciled by the graph;
the backend never replays an approval. Ordinary resume refuses human-stopped
tuning; explicitly checking recovery invokes the existing JSON backup/recovery
mechanism. Do not run the CLI against the same run while the backend owns it.

In-memory graph checkpoints are **not** restart durability: in-flight model or
numerical work may restart, and an unrecorded experiment can be repeated. The
backend saves after each graph snapshot. Cooperative stop (`POST .../stop`) and
shutdown wait for the current node; an already executing operation can finish.
No multi-process locking, authentication, scanner automation or exactly-once
numerical execution is provided.

Offline validation:

```powershell
python -m pytest tests/api frontend/tests tests/workflow tests/printing tests/print_tuning -q
```

`tests/api/test_real_workflow.py` drives the real graph and agents with scripted
model replies, numerical experiments, real trial PDFs, synthetic uploads, the
existing scan evaluator, two tuning cycles, report revision and fake printing.
All test outputs use temporary directories; no physical jobs or paid calls.

The terminal is still available: `python main.py --fake-printer`.
For offline UI development alone, run `python -m uvicorn frontend.mock_api:app
--host 127.0.0.1 --port 8000` instead of the real backend. See
[the frontend contract](frontend/README.md).
