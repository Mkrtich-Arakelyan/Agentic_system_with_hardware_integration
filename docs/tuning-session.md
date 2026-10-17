# Tuning session ownership and lifecycle

`app/workflow/tuning.py` owns `RUN_DIR/tuning_session.json` as plain application
code. Agent tools read this file and return artifacts/results; they never update
it. The scientist and photo quality specialist have no session-writing tools.
The tuning loop is not wired into the graph yet. Current report behavior, page
limits, roles, tool lists, and printer approval flow are unchanged.

## API

```python
from app.workflow.tuning import (
    initialize_tuning_session, load_tuning_session, build_tuning_context,
    accept_specialist_response, record_trial_result, record_scan_result,
    record_print_approval, begin_print_attempt, record_print_result, record_stop,
)

session = initialize_tuning_session(run_dir, accepted_selection, recorded_results)
session_path = session["session_path"]
context = build_tuning_context(session_path, "propose_gamma")
# Obtain a parsed decision separately; no model is invoked by this module.
accept_specialist_response(session_path, gamma_proposal, context)

# A future executor turn calls make_trial_sheet(1). Application code then records:
record_trial_result(session_path, trial_result)
# Printing/approval is separate; these recorders do not perform any action.
record_scan_result(session_path, supplied_scan_result, record_id="scan-attempt-1")
context = build_tuning_context(session_path, "review_gamma")
accept_specialist_response(
    session_path, gamma_review, context, evaluation_record_id="scan-attempt-1",
)
```

Repeat the trial/scan/review sequence for `review_gray_levels` (round 2) and
`review_steps` (round 3). Only the final usable review sets `final_settings` and
`status="completed"`. There is no round 4. Full `validate_response()` runs before
`resolve_review_winners()` and before one atomic write, so valid winners alone
cannot be accepted with invalid next proposals. The validator internally resolves
those winners before checking next proposals against their complete mappings.

`load_tuning_session(path)` reads without regenerating previews or limits.
An identical repeated decision for the same stage/trial/evaluation returns the
existing session without writing or advancing. A conflicting decision for that
identity fails. Review callers must supply the explicit recorded evaluation ID:
trial identity alone cannot distinguish a replacement scan. A stale trial or
evaluation cannot advance the session.

## Initialization and recovery

The initializer requires exactly four distinct IDs and the accepted scientific
selection reasons. It uses `selection_from_scientist()` and its existing array
checks/selection identity algorithm, preserving order, sources, and reasons.
Relative source paths resolve from the supplied run directory. The run ID defaults
to the run directory name; the target position, when not explicitly provided, is
read from that run's existing `config.json` bench settings if available.

The tuning configuration is copied into `config_snapshot`. Mappings start with
the renderer's `starting_mapping(config_snapshot, paper_color)`, including its
configured paper preset when one matches. The graph forwards the runner's
`state["paper_color"]`. Direct callers may pass `paper_color="yellow"` or include
`paper_color` in an optional `paper_session` dict; conflicting colors are rejected.
That dict must identify `paper_session_id`; omission means the `default` paper
session. With no color on a new session, configuration defaults apply. Both the
session and previews store the same color and starting mappings. Omitting both
paper inputs on resume preserves the existing paper metadata/color. Explicit
color changes are rejected, including for old sessions created without a color;
existing previews and accepted settings are never silently rewritten. To explicitly
select plain white on resume, supply `paper_session` with `paper_color=None`.
Each calibration receives a new UUID `session_id`.
Selection identity/version come from the existing selection packager, not a new
identity algorithm.

A bootstrap record is written before `prepare_specialist_previews()` is called.
The completed selection path is saved before that helper reads the session. Its
returned `normalization_limits`, labelled preview paths, and default mappings are
retained; fixed limits are also stored in `selected_experiments[].limits` as the
tool readers require. The final atomic write marks initialization complete.

Failures retain `initializing` or explicit initialization-error state. Ordinary
load rejects partial sessions. Inspect with `allow_incomplete=True`; deliberately
retry with `retry_initialization=True` and the same selection/configuration.
There are at most three initialization attempts. Preview artifacts use separate
attempt directories, so earlier partial artifacts are retained. A successfully
initialized compatible session is reused without original-array or preview reads.
Changed selection, run, paper metadata, or explicitly/currently supplied config
cannot overwrite an existing session. Original arrays still must remain available
for later rendering; successful reload does not certify that those files exist.

## Existing tool and artifact contracts

The record preserves `contract_version=1`, `session_id`, `run_id`, `selection_path`,
`config_snapshot`, ordered `selected_experiments`, `accepted_settings`,
`rounds["1"/"2"/"3"].proposal/trial_result`, and `active_trial_id` required by
`tuning_tools._session_inputs()`. `prepare_specialist_previews()` and
`render_final_heatmaps()` consume it unchanged. No renderer or scan module was
modified.

Each round retains trial attempts, scan references/hashes, reviews and reasons,
resolved winner mappings, approval metadata, print intents/results, and bounded
retry counters. The proposal for round 2 is the complete `GammaReviewResponse`;
the proposal for round 3 is the complete `GrayLevelsReviewResponse`, matching
the existing tools. Accepted settings carry forward as each next reference.

`record_trial_result()` checks the existing `trial/1` manifest, run/calibration/
selection identity and order, config, original panel hashes, fixed limits,
translated candidates and expected mappings, PDF path/hash and actual page count.
It reuses the validator's candidate correspondence and the tools' proposal-to-
variation helpers instead of duplicating rendering logic. Those narrowly scoped
private helpers are an explicit compatibility dependency; changes to their
signatures should be coordinated. The same recorders work with legacy internal
variation IDs through the shared label correspondence.

Regenerating before a review is accepted retains previous trials/evaluations and
invalidates the active evaluation. An unchanged physical reprint should reuse the
recorded PDF/trial, not call the generation tool again: that existing helper
generates a new trial UUID on each call. Scan replacement never creates a new
physical round. Current retry limits are three replacement scan results, two
trial regenerations, and one approved reprint per round. Limit errors leave the
session unchanged; later graph/human policy must decide the next action.

`record_scan_result()` accepts the current tool's `ok`, `needs_rescan`, or `error`
contract, verifies the active session/round/trial and evaluation-file identity,
and saves the evaluation path/hash. Large page/candidate measurement payloads
remain in that evaluation JSON. Reusing the same record ID with different content
fails; supplying a new ID distinguishes independent identical failure results.
Scans alone never accept settings or advance the round.

Usable reviews and insufficient-contrast judgments require a recorded usable
evaluation. `needs_rescan` records its issues and returns to `awaiting_scan` with
no winners/next proposal accepted. `insufficient_contrast` sets `status="blocked"`
and `phase="user_action_required"`; it never fabricates final settings. Blocked,
stopped, and error sessions retain progress for explicit future user-action policy;
ordinary initialization/load does not reset them.

## Printing integration boundary

`awaiting_scan` means a trial exists and print coordination or manual printing
may precede its scan; it is not proof a job was physically submitted or completed.

1. `record_print_approval()` binds an approval ID to active trial/PDF SHA, printer,
   approver, effective print settings, and explicit reprint flag.
2. `begin_print_attempt(approval_id=..., attempt_id=...)` atomically records a
   submission intent BEFORE future external submission. A missing result is
   labelled `outcome_uncertain`, never "not submitted". On recovery, inspect an
   existing attempt instead of automatically repeating submission. Callers must
   serialize these operations and distinguish a newly recorded intent from one
   already present before calling the external printer.
3. `record_print_result(attempt_id=..., result=...)` stores the supplied result
   verbatim; use `job_result=True` to append distinct job readings. It does not
   reinterpret vendor statuses, infer completion, resubmit, or cancel anything.

One initial print approval plus ONE explicit approved reprint is supported per
round. An approval cannot be consumed twice, and uncertain submission outcomes
must be recorded/reconciled before approving a reprint. Razmik's evolving printer
return semantics remain external; a later graph must decide which supplied job
states permit retry or completion. Metadata recording does not certify that a
printer/driver honored the recorded effective settings.

`record_stop(reason, error=False, metadata=None)` records stopped/error outcomes
without deleting artifacts or cancelling jobs. A completed session cannot be reset
through that API.

## Graph hook and persistence limits

In `scientist_node`, after all existing selection/evidence checks pass and the
selection is frozen, the graph calls initialization. The existing-frozen-selection
branch also initializes/reuses it. `orchestrator_node` repeats the idempotent hook
when encountering a frozen selection, including resume and budget-check paths.
Only `tuning_session_path` and a small `tuning_initialization` projection are added
to state; accepted/final settings are not independently copied into graph state.
Initialization failure is visible in that projection and in the partial session;
it does not add an orchestrator role, model call, pause, or change report routing.

JSON writes use a temporary file in the same directory, `allow_nan=False`, flush/
fsync and atomic replacement. Callers MUST serialize operations per session;
there is no database or distributed lock. This file preserves tuning recovery
data but does not turn the existing `InMemorySaver` into a persistent LangGraph
checkpointer. The evidence adapter below is available; full tuning routing/approval
integration is the next step. No tuning loop is active.

## Specialist evidence adapter

`prepare_tuning_inputs(session_path, stage)` in `app/workflow/tuning_evidence.py`
returns `TuningInputs(context, evidence, prior_decisions, paper_color,
evaluation_record_id)`. It is also exported from `app.workflow`. It reads only
the session-selected trial and evaluation; no folder search, model invocation,
image transformation, or session write occurs.

```python
from app.workflow import prepare_tuning_inputs
from app.agents.photo_quality_specialist import photo_quality_specialist

inputs = prepare_tuning_inputs(session_path, "review_gamma")
response = photo_quality_specialist(
    llm, inputs.context, inputs.evidence,
    prior_decisions=inputs.prior_decisions, paper_color=inputs.paper_color,
)
accept_specialist_response(
    session_path, response, inputs.context,
    evaluation_record_id=inputs.evaluation_record_id,
)
```

`load_tuning_evaluation()` centralizes active trial, file hash and evaluation
identity checks for the adapter and decision acceptance. `scan_eval/2` candidate
`experiment_id` is a manifest item ID and `candidate_id` is an internal variation
ID. The adapter uses `get_candidate_correspondence()` to recover original
experiment IDs and specialist reference/test labels. Panel `png` paths and whole
`tile_path` crops become labelled attachments; a scanned-candidate label with
`panel=None` denotes the whole tile (both panels, label and scale bar).
Measurements, null-value notes, page reference readings, combined printer response,
`measurement_docs` (units and meanings), `scan_files` (original path/SHA provenance),
the original structured `scan_quality`, and contextual issue strings remain evidence.
The specialist's existing message preparation attaches the original image bytes.

Initial proposals require all eight saved previews and have no scan measurements.
Reviews use the references printed BEFORE that round and earlier usable accepted
decisions, excluding the active stage. Missing scan panels/candidates become
`MissingImage` declarations. An explained recorded `needs_rescan` without an
evaluation artifact yields all scan crops missing and no fabricated measurements.
Operational failures and missing/corrupt/stale evaluations are application errors.
Absolute producer paths are retained; relative preview paths resolve from the run,
and relative panel PNGs resolve from the evaluation JSON's directory.

New initializations record hashes of saved preview PNGs. The adapter verifies
these when present; legacy sessions without preview hashes remain readable but
cannot certify historical preview pixel integrity. Scan-evaluation files and
manifests retain their existing recorded hash checks. The current scan producer
does not record crop image hashes, so individual crop pixel integrity is not
certified beyond their bound identities and the specialist's decoding checks.

## Full result versus executor summary

The latest tool contract is documented in [tuning_tool_results.md](tuning_tool_results.md).
Use the plain application function `record_scan_for_executor()` at the future
executor tool boundary:

```python
from app.workflow import record_scan_for_executor

# A future executor turn obtains the full read_scan tool result separately.
# Feed only the returned compact dict into its ToolMessage/dialogue/history.
executor_result = record_scan_for_executor(session_path, full_scan_result,
                                         record_id="scan-attempt-1")
# The specialist adapter reloads the full, unrounded evaluation from disk.
inputs = prepare_tuning_inputs(session_path, "review_gamma")
```

This reuses Razmik's `summarize_scan_result()` rather than implementing another
summary format. It checks that full result fields agree with the evaluation file,
records the full scan outcome, and returns only the compact model-facing summary.
Do not record the already summarized result: rescan summaries omit scan_quality,
and success summaries omit panel measurements and images. Original numerical
precision and null-value reasons remain available to the specialist. Full raw
results must not be copied into executor dialogue or execution_results; store the
summary there and keep the session/evaluation as the authoritative full evidence.

Errors with trial identity are recorded normally. Errors returned before an active
trial was known are returned unchanged in the documented short format, without
guessing a session/trial association or writing a session. Callers may separately
record a stop/error outcome when their application policy requires one. The
handoff itself never submits a scan, prints, calls a model, or advances a round.
It is ready for the next graph task; no new executor mode or graph node is active.

New trial sheets print on both sides with long-edge binding, A4, grayscale,
one copy, and Actual size / 100% (no scaling). The approval prompt and persistent
approval metadata bind these effective settings to the exact PDF and printer.
An unused legacy single-sided approval requires fresh approval for duplex;
existing submissions, scan originals, and history are retained. Submitted or
uncertain jobs are never automatically repeated when this policy changes.

Use a duplex-capable printer/driver. Scan BOTH faces as separate page images,
including the page markers; do not mirror the back. The reader matches page
codes against the manifest rather than relying on file order. Keep print and
scan acquisition settings consistent across rounds: duplex show-through may
affect measurements. Restart the application to load the new print policy;
do not reprint already completed trials just to change their sides setting.

For development, install application dependencies from requirements.txt and the
test runner with `python -m pip install -r requirements-dev.txt`. Offline
verification: `python -m pytest tests -q` (also discovers the new pytest-based fake
scan tests). Fixtures use real preview/trial/final-render helpers with tiny arrays,
supplied scan metadata, and generated fake scans through the real scan reader.
Models/graph hooks are mocked; no paid model, physical printing, live experiment,
or real scan is used.
