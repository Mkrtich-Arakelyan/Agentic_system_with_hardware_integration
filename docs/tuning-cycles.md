# Repeated tuning and metric progress

New terminal runs use at most three cycles by default. Set `--max-tuning-cycles N`
on a new run (1–20); the saved session retains this limit on resume. After every
accepted third step the application pauses: `continue`, `finish`, or `stop`.
At the maximum only `finish` and `stop` are available. Continue is not print
approval. Finish freezes the latest accepted step-3 complete mappings for reporting.

Each cycle retains gamma → shared gray endpoints → gray steps. A step changes
only its permitted keys; inversion, sources, selection and normalization stay
fixed. Gamma/steps remain per experiment, endpoints shared. All six candidates
(reference plus five distinct tests) retain their original validation rules;
the reference may win. A new cycle starts from the previous cycle's complete
winners. Specialist context identifies the cycle/step and includes the previous
cycle evidence and current references; original previews remain explicitly original.

## Session compatibility and recovery

`tuning_session.json` is authoritative. Version 2 uses `active_cycle`,
`active_step`, `cycles[cycle].steps[step]`, original mappings, boundary decisions,
metric snapshots and coverage revisions. Root `rounds` is the active cycle's
compatibility projection, verified on load and synchronized on atomic writes;
`round_id` always means the local step (1, 2, 3), never a global counter.
Completed cycle steps are not overwritten. Every trial/evaluation records
cycle/step explicitly; specialist acceptance binds cycle, trial and evaluation.
Each step has its own persistent one-approved-reprint budget. Rescans do not
consume it; resume does not reset it. Old cycles' trial/approval/attempt IDs
cannot be reused in a new cycle.

Version-1 sessions remain readable and unchanged, with their historical single
cycle and completion semantics. They are not silently migrated/reopened to add
cycles. The terminal enables v2 for new sessions; library callers opt in through
`initialize_tuning_session(..., max_cycles=3)`. Existing scans and paths remain
valid. New scan folders are `scans/cycle_N/step_N/`; legacy runs retain
`scans/round_N/`. Scan both faces of duplex trial sheets as separate pages.

New sheets carry a page code with a 12-bit trial-derived tag and page bits, plus
the printed cycle/step/trial header. Tags can collide: they are a stale-sheet
check, not proof of unique physical trial identity. Recorded trial IDs, manifest
hashes, source hashes, cycle/step and evaluation identities remain authoritative.
Legacy 6-bit codes remain readable, with their historical identity limitations.

Exact-PDF approval, persistent intent before I/O, duplicate prevention and uncertain
outcome confirmation remain mandatory. Before restarting, exit the paused process
and wait for it to stop. Never run concurrent instances against one run or edit
its records while active. Ordinary resume uses durable progress. An explicitly
human-stopped tuning run requires `--resume-stopped-tuning`, which backs up JSON
and records recovery; a stopped cycle boundary reopens that boundary. No print
history or retry allowance is cleared.

## Measurements and charts

The baseline is the original cycle-1 step-1 reference from its accepted usable
scan. Every later point is that step's actual accepted winning candidate,
including reference winners and the shared endpoint winner resolved per experiment.
Only scientific Final camera panels are measured. No display PNG is measured.

`cell-pairs/1` is unchanged: frozen source limits/masks; unique horizontal/vertical
neighbors with normalized source difference at least 0.05; central half-cell
interiors; 0.3 mm smoothing at 10 samples/mm. Noise is within-cell standard
deviation. CNR is absolute mean separation divided by the larger of pooled
within-cell deviation and the common finite floor. The floor is the maximum
measured gray-patch/quantization floor across available captures. Preservation
requires correct brightness ordering (including inversion) and CNR >=3.
Median CNR includes every jointly valid eligible pair, including quantization ties.
These are project-specific engineering criteria, not validated human-vision scores.

Scanner resolution must meet the manifest minimum; aligned captures are sampled
at the same physical scale. Smoothing reduces sensitivity to halftone dots but
cannot recover tones lost to quantization, bilevel thresholding or scanner
processing. Bilevel mode, calibration response, original DPI and the measured
noise floor remain recorded diagnostics rather than an assumed correction.

All plotted measurable snapshots for an experiment share the SAME valid source
pair set, mask, method and floor. Adding evidence recomputes old summaries from
immutable cell measurements and records the changed coverage; inadequate coverage
(default <80%) produces null. Missing/rejected/unusable evidence produces a missing
point with a reason, never zero/interpolation. Per-step snapshots retain the original
accepted measurements; `metric_history_revisions` records hashes and denominator
changes. Hash-verified cell measurements can be reused without re-aligning scans;
source, evaluation, manifest, scan and method dependencies must still match.

Saved history includes source hashes, candidate/mapping, geometry, calibration,
scan provenance, cell means/deviations/validity, common pair indices, CNR/order,
clipping and coverage. Synthetic capture provenance is explicit. Paper/black
normalization does not undo nonlinear scanner processing. Clear processing or
calibration mismatches yield missing points; unknown acquisition settings remain
unknown. Detail and CNR summarize the same pair evidence, not independent proof.

Charts use E1–E4 with distinct grayscale line styles/markers, missing-point gaps,
cycle boundaries, detail 0–100%, and finite CNR axes. Long histories sample tick
labels to retain legibility; the saved chart metadata retains every point label.
Descriptions report baseline/final deltas and last-cycle gain, plateau, regression
or trade-off without statistical confidence claims.

## Report and revision

Policy `report/4` accepts 2 or 3 A4 portrait pages and defaults to 3. Page 1 contains
scientific results; page 2 contains four panels per experiment, in frozen E1-E4
order: scientific Initial/Final before printer tuning, then Initial/Final after.
Before uses the original cycle-1 step-1 reference (not the current cycle's reference).
After uses the last completed cycle's accepted step-3 mapping. Each pair reuses the
same numerical arrays, frozen limits, annotations and native aspect ratios. Only
the print-rendering parameters differ. All panel boxes are equal; the gap between
pairs is wider than the gap within a pair. The figure occupies the full 180 mm
usable width with 10 pt labels. The final-settings table remains below it.
Page 3 contains the two progress charts, baseline/final table,
quantitative interpretation and acquisition/method limitations. The exact comparison
PNG is validated on page 2. Serif body text remains 11 pt, captions/tables >=10 pt,
approximately 15 mm margins, page numbers only. A two-page presentation request
fails if content does not fit; chart labels/fonts are not shrunk to force it.
The default three-page policy is unchanged; the four-panel figure is never shrunk
to force a two-page request. Figure contract `scientific-print-pairs/2` records
both mappings and their cycle/step sources. Old two-column artifacts are
regenerated, retained in figure history, and cannot satisfy PDF validation or
reuse a previous print approval. Regeneration never submits a job.

Report revision remains explicit/durable: instructions reach `build_final_report`,
old PDFs are archived, old approval is invalidated, failures/no-ops stay pending,
and changed output requires fresh exact-PDF approval and human semantic review.
No experiment or tuning cycle is rerun to revise a report.

## Commands

Fully offline synthetic two-cycle demonstration (new output directory required):

```powershell
.\.venv\Scripts\python.exe -m scripts.demo_cycles --out output/cycle_demo
```

It generates numerical fixtures and synthetic scans, not scientific experiments
or physical jobs. The resulting report is `output/cycle_demo/report/report.pdf`.

Terminal fake-printer run (uses configured paid models; not an offline model test):

```powershell
.\.venv\Scripts\python.exe main.py --fake-printer --printer Office_Printer --paper-color yellow --max-tuning-cycles 3
```

Supply explicit scan paths at each scan prompt. The Python `fake_scan(pdf, out_dir)`
helper can produce labeled synthetic scans for that active trial; use a new folder
each time. Continue/finish decisions are entered at the cycle boundary, not at the
trial print-approval prompt.

Resume an existing real-printer run, AFTER the old process exits:

```powershell
.\.venv\Scripts\python.exe main.py --resume runs/20261017_134707_564060 --no-fake-printer --printer ir2425
```

This may use paid models/read hardware; it was not executed during implementation.
Resolve any unknown print outcome before authorizing another job. Do not add
`--max-tuning-cycles` on resume or delete history to enable continuation.
