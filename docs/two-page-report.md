# Final report (report/4; three-page default)

`app/report_policy.py` centralizes the final-report page policy (three by default;
two permitted only when readable) and zero-based comparison page (1).
Trial-sheet page counts remain manifest-owned. The MCP report action rejects
other counts before printer access. The graph validates A4 portrait pages and
the exact comparison image on page 2;
SHA-256 approval covers all PDF bytes. Existing submission intent, replay,
uncertainty, reprint and journal checks remain in `workflow/reporting.py`.

## Rendering and regeneration

`workflow/report_document.py` uses PyMuPDF and built-in Times Roman: 21 pt titles,
14 pt headings, 11 pt body, and 10 pt tables/captions, with 15 mm margins.
Text is wrapped by measured font widths. Tables and text reject overflow rather
than silently shrinking. The scientific chart uses separate categorical bars
for E1-E4, not a continuous trend through incompatible conditions. Full IDs,
source results, reasons and paths live in `report_inputs_v2.json`.

Page 2 embeds `render_report_comparison`'s exact grayscale PNG at the full 180 mm
usable width. Frozen experiment order is labeled E1-E4. Each row has four equal
panel boxes: scientific Initial/Final under **Before printer tuning**, then
Initial/Final under **After printer tuning**, separated by a larger gap.
Before uses each experiment's original cycle-1 step-1 reference mapping; After
uses its final accepted step-3 mapping from the last completed cycle. The SAME
original Initial/Final numerical arrays are reused across both groups, with one
frozen normalization range per experiment, native aspect ratios, preserved
annotations, and nearest-neighbor rendering. No scan crops or panel-specific
auto-normalization are used. Group headings, column headings and compact E1-E4
labels are part of the figure, at 10 pt when printed. The caption explains that
only print-rendering parameters differ. The legacy two-panel
`render_final_heatmaps` function remains available for tuning clients.

Figure contract `scientific-print-pairs/2` records both mappings and provenance.
The application fingerprint includes the renderer hash, version, original
baseline, final mappings, frozen data/order/limits and configuration. Old layouts
are regenerated without overwriting history and invalidate pending approval.
PDF validation rejects stale contracts, changed image pixels, incorrect placement,
aspect distortion and shrinking below full usable width (apart from raster rounding).
The default remains three pages. A two-page request that cannot fit fails rather
than shrinking the figure or its labels.

Only the existing executor receives `build_final_report`. It supplies optional
structured scientific findings, not page coordinates. No independent report
agent is introduced. The report tool archives a prior PDF by hash before
installing new bytes; prior print records are retained, and new bytes require
fresh approval. It refuses rebuilding with unresolved or stopped submissions.
The graph checks completed tuning without resetting it. Normal resume discards
transient approvals; old one-page PDFs do not pass the new report policy.

Generate an independent offline bundle for a completed run from the repo root:

```powershell
& .\.venv\Scripts\python.exe -m app.workflow.report_document .\runs\20261016_183451_221483 --out .\output\pdf\regenerated_report
```

Use a new output directory. This command does **not** load a model, execute an
experiment, tune, print, install the report into the run, or modify any journal.
It reuses accepted scans; no new print or scan is requested. It writes the PDF,
comparison PNG and versioned report-input JSON. Repeating identical inputs uses
the intact cache. Changed dependencies/settings require a new output directory;
existing bundles are never silently overwritten. The cache hashes the session,
recorded source/evaluation/manifest/scan files, scientific results, method,
findings and implementation files. Missing dependencies are also fingerprinted.
Agents' structured wording is saved, so rebuilding a bundle needs no model call.
Older completed runs with missing metric evidence receive specific null reasons.

## Project-specific print-and-scan measurements: cell-pairs/1

These are engineering readability summaries, **not standardized perceptual
quality scores**. Detail preservation and median CNR summarize the **same pair
evidence**, and are not independent proof of improvement.

Evidence selection uses the accepted usable review's explicit evaluation record
ID in rounds 1 and 3. It checks stored evaluation and manifest hashes, trial PDF
hashes, session/round/trial identities, frozen experiment order, source array
hashes, Final panel identity, normalization limits, annotation masks, candidate
location, role and complete mapping. Before is each round-1 reference. After is
the accepted steps winner, including reference if it won. Rejected scans and
superseded trials cannot be chosen through filename or modification-time guesses.
Original evaluations are never overwritten or upgraded in place.

1. Re-read the original recorded scans and reuse existing marker alignment,
   page codes and manifest geometry. Convert RGB using Pillow's L conversion.
   Resample both captures onto the same 10 pixels/mm aligned grid and apply a
   fixed 3x3 mean filter (0.3 mm square). This reduces sensitivity to halftone
   pixels but does not remove halftoning, recover missing resolution or undo
   nonlinear scanner processing. PDF scan reading retains its existing 300 dpi
   rendering policy and checks the recorded acquisition resolution.
2. Use the central half of each cell in each dimension. Exclude cell/grid
   boundaries and the existing conservative box-outline masks, clipped plot
   regions, nonfinite pixels and interiors too small to measure. Calculate
   cell mean and population standard deviation **within its pixel region**.
   Variability between different-valued cells is never used as noise.
3. Normalize source arrays with frozen `lo,hi`. Select each horizontal/vertical
   neighbor once when both source values are finite, not clipped, unmasked and
   differ by at least **0.05** of that range. This configurable threshold is an
   engineering default, not a validated human-vision cutoff. The CLI exposes
   `--source-difference`; the saved method also includes CNR/coverage thresholds.
4. Evaluate only eligible pairs measurable in **both** captures. Save total
   eligible pairs, joint valid pairs, coverage and actual pair indices. Coverage
   below **80%**, zero eligible pairs or zero jointly valid pairs yields null
   with a reason, not a fabricated zero score.
5. Calibrate cell means/stds with that page's paper-minus-black range. Reject
   range below **0.2**, absent gray patches or a gray response reversing by more
   than **0.05** normalized brightness between increasing patch levels.
   The per-page noise floor is the larger of the median within-gray-patch
   standard deviation divided by range and `1/(255*range)`. Use the maximum of
   the two pages' floors and `1/255` for **both** versions' pairs.
6. For each pair, `pooled_std=sqrt((std_A^2+std_B^2)/2)` and
   `CNR=abs(mean_A-mean_B)/max(pooled_std, noise_floor)`. A finite positive floor
   prevents infinite or NaN CNR even for zero-variance images.
7. Correct brightness ordering follows source-value direction and the mapping's
   inversion. Detail preservation is `100*preserved/joint_valid`, where preserved
   requires correct order **and CNR >= 3**. CNR 3 is an operational criterion,
   not a proven human visibility threshold. Median CNR includes **all** joint
   eligible pairs, including merged/failed/wrong-order pairs. Save wrong-order
   fraction separately. Flat/quantized pairs are not dropped to inflate scores.
8. Save each experiment's before/after values, detail change in percentage
   points, absolute CNR change, and relative CNR change only for a positive,
   valid baseline. No pooled overall pair-count score is reported.

Outputs also contain cell statistics, masks, saturated-pixel share, clipped and
nonfinite source counts, missing-cell counts, calibration patches, alignment
errors, DPI, source hashes, candidate/mapping/scan provenance and method settings.
Sampling/calibration changes require a new implemented method version.

## Scanner comparability and provenance

Separate captures have separate calibrations. Paper/black normalization addresses
offset/range differences; it cannot undo nonlinear automatic scanner correction.
Explicit matching acquisition metadata is required to say settings are confirmed
consistent. Missing metadata means **unknown**. Explicitly different settings,
bilevel-versus-continuous processing, or gray-patch normalized response differences
above **0.2** return unavailable primary comparisons with a specific mismatch
reason; diagnostic computations remain saved separately. This is a conservative
engineering flag, not a validated scanner-comparability test.

Bilevel scans remain flagged even when both captures use them. Unknown settings
limit comparisons to descriptive scan measurements and prevent causal printer
gain claims. Scan origin is explicit when recorded and otherwise unknown; a
configured physical printer is not evidence that a particular scan is real.
New fake-scan PNG fixtures carry explicit synthetic metadata, which scan evaluation
preserves. Legacy files are not relabeled from their filenames.

## Demonstration evidence

The completed `20261016_183451_221483` session's accepted round-1 record is
`scan_r1_7410f4c5_06ed73ea.json` (scans `round_1/0781_001.pdf` and
`round_1/0779_001.pdf`). The accepted round-3 record is
`scan_r3_fea8b76b_71764968.json` (scans `round_3/0784_001.pdf` and
`round_3/0785_001.pdf`). Its final winners are `test_5` for all experiments.
Round 2 remains part of tuning provenance; it is not the before/after endpoint.
All four pairs of measurements have 100% joint coverage. Detail preservation
changes by +10.9, +4.5, -5.7 and -5.3 percentage points for E1-E4 respectively;
median CNR changes by about +0.51, +0.51, -0.35 and -0.64. These mixed descriptive
results do not establish net hardware improvement. Both captures are bilevel;
acquisition settings and scan origin are not recorded.
