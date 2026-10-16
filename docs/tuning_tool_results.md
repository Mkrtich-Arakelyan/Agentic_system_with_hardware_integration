# Tuning tool results

What `make_trial_sheet`, `read_scan` and `summarize_scan_result` return, for the workflow adapter.
All three live in `app/print_tuning/tuning_tools.py`. The examples are real values from a test
run, cut to one page and one candidate. `<run>` stands for the run folder.

Every result has `contract_version` (1) and `status`. Tools never raise: problems come back as
`status: "error"` or `"needs_rescan"` with an `error_code` and a `message`.

## make_trial_sheet(round_id)

### ok

| Key | Meaning |
|---|---|
| `contract_version` | 1 |
| `status` | `"ok"` |
| `session_id` | the session's `session_id` (the trial's `calibration_session_id`) |
| `trial_id` | new trial id (UUID) |
| `round_id` | 1, 2 or 3 |
| `pdf_path` | the trial sheet PDF to print |
| `pdf_sha256` | sha256 of that PDF; tie the print approval to it |
| `manifest_path` | the trial manifest JSON next to the PDF |
| `page_count` | pages in the PDF (2 for 4 experiments) |

```json
{
  "contract_version": 1, "status": "ok", "session_id": "cal_x",
  "trial_id": "a9848236-c778-471e-bdcd-0b6967ca611f", "round_id": 1,
  "pdf_path": "<run>/print_tuning/trial_r1_a9848236.pdf",
  "pdf_sha256": "8627fa93ddd757ac4140d743be4b5268edb68186c56c32dc2aec0e0409bceb51",
  "manifest_path": "<run>/print_tuning/trial_r1_a9848236_manifest.json",
  "page_count": 2
}
```

Store this whole result as `rounds["<round_id>"]["trial_result"]` in the session, and its
`trial_id` as `active_trial_id`; `read_scan` finds the manifest through them.

### error

Keys: `contract_version`, `status: "error"`, `round_id`, `error_code`, `message`.

```json
{
  "contract_version": 1, "status": "error", "round_id": 2, "error_code": "session_invalid",
  "message": "tuning session: accepted_settings has no mapping for ['experiment_53a4…', …, 'exp/A'] before round 2"
}
```

## read_scan(scan_paths)

`scan_paths`: one PDF with every page, or one PNG/JPG per page. Relative paths are taken from
the run folder. The trial is always the session's active trial.

### ok: top-level keys

| Key | Meaning |
|---|---|
| `contract_version`, `status` | 1, `"ok"` |
| `session_id`, `trial_id`, `round_id` | which session, trial and round was measured |
| `evaluation_path` | the saved evaluation JSON (the full scan_eval/2 result) |
| `scan_quality` | `{usable: true, issues: [{page_number, issue}]}` |
| `pages` | one record per page (below) |
| `candidates` | one record per experiment and candidate, manifest order, no ranking (24 for 4 × 6) |
| `printer_response` | paper and black `{gray, rgb}`, `gray_patches` `[{level, measured}]`, `notes`, averaged over pages |
| `measurement_docs` | `{name: {units, meaning}}` for every number (34 names, e.g. `contrast`, `focus.difference`) |
| `scan_files` | `[{path, sha256}]`, one per scan file |

Grays run from 0 (black) to 1 (white). A null measurement always has its reason in the same
record's `notes`.

### One page record

```json
{
  "page_number": 1, "scan_index": 0,
  "page_code_read": [0, 0, 1, 0, 0, 1], "page_code_expected": [0, 0, 1, 0, 0, 1], "code_matches": true,
  "aligned": true, "fit_error_mm": 0.001, "dpi": 300.0, "orientation": 0,
  "paper": {"gray": 1.0, "rgb": [1.0, 1.0, 1.0]}, "black": {"gray": 0.0, "rgb": [0.0, 0.0, 0.0]},
  "paper_to_black_contrast": 1.0,
  "gray_patches": [{"level": 0.0, "measured": 0.0, "std": 0.0}, {"level": 0.1, "measured": 0.098, "std": 0.0}],
  "issues": [], "notes": {}
}
```

### One candidate record

| Key | Meaning |
|---|---|
| `experiment_id` | the trial sheet's **item_id** (see below) |
| `candidate_id` | `reference`, `test_1` … `test_5` |
| `role` | `reference` or `test` |
| `mapping` | the full mapping printed: `gamma`, `dark_level`, `light_level`, `steps`, `invert` |
| `page_number` | page the tile is on |
| `tile_path` | colour PNG crop of the whole tile (both panels, scale bar, label) |
| `panels` | `[{panel, png, measurements, notes}]`, one per panel (`Initial`, `Final`) |
| `measurements` | the candidate summary: `paper_white_share`, `full_black_share`, `collapsed_levels`, `correlation`, `contrast`, `focus_difference` |
| `warnings` | short texts, e.g. `"many cells print as full black"` |
| `notes` | reasons for null measurements |

There is **no `variation_id` key**. `candidate_id` *is* the manifest's `variation_id`
(`make_trial_sheet` writes `variation_id` = `reference`, `test_1` … `test_5`), so use
`candidate_id` wherever a `variation_id` is needed.

```json
{
  "experiment_id": "experiment_53a404f7d2444d81beaa960fb11d4c8a",
  "candidate_id": "test_1", "role": "test",
  "mapping": {"gamma": 0.4, "dark_level": 0.0, "light_level": 1.0, "steps": 0, "invert": false},
  "page_number": 1,
  "tile_path": "<run>/print_tuning/scan_r1_a9848236_db799bba_tiles/p1__experiment_53a4…__test_1__tile.png",
  "panels": [{
    "panel": "Initial",
    "png": "<run>/print_tuning/scan_r1_a9848236_db799bba_tiles/p1_experiment_53a4…__test_1__Initial.png",
    "measurements": {
      "cells_used": 1004, "paper_white_share": 0.0, "full_black_share": 0.01,
      "intended_levels": 14, "printed_levels": 14, "collapsed_levels": 0,
      "correlation": 0.9999, "mean_abs_difference": 0.0014, "contrast": 0.4308, "intended_contrast": 0.4286,
      "focus": {"inside_mean": 0.6353, "rest_mean": 0.4278, "difference": 0.2075, "intended_difference": 0.2062}
    },
    "notes": {}
  }],
  "measurements": {
    "paper_white_share": 0.001, "full_black_share": 0.0075, "collapsed_levels": 0,
    "correlation": 0.9999, "contrast": 0.4347, "focus_difference": {"Initial": 0.2075, "Final": 0.5716}
  },
  "warnings": [], "notes": {}
}
```

### printer_response, measurement_docs, scan_files

```json
{
  "printer_response": {
    "paper": {"gray": 1.0, "rgb": [1.0, 1.0, 1.0]}, "black": {"gray": 0.0, "rgb": [0.0, 0.0, 0.0]},
    "gray_patches": [{"level": 0.0, "measured": 0.0}, {"level": 0.1, "measured": 0.098}],
    "notes": {}
  },
  "measurement_docs": {
    "contrast": {"units": "gray (0 black to 1 white)",
                 "meaning": "95th minus 5th percentile of the printed cell grays (mean over panels for a candidate)"},
    "collapsed_levels": {"units": "levels",
                         "meaning": "intended levels minus printed levels: gray levels that merge on paper (summed over panels for a candidate)"}
  },
  "scan_files": [
    {"path": "<run>/scan/scan_page_1.png", "sha256": "cb38eaacd55f70beb47a8125e7e301481393e39209b7bceba05d4b9e2bf96c25"},
    {"path": "<run>/scan/scan_page_2.png", "sha256": "0dc2a4ed0689ef86468ab9e0c1eabd7c8c4bb36e2aacb7e34432dcbbaa3a2a09"}
  ]
}
```

### needs_rescan

A new scan would fix it. Keys: `contract_version`, `status: "needs_rescan"`, `session_id`,
`trial_id`, `round_id`, `error_code`, `message`, `scan_quality: {usable: false, issues: [message]}`.

```json
{
  "contract_version": 1, "status": "needs_rescan", "session_id": "cal_x",
  "trial_id": "a9848236-c778-471e-bdcd-0b6967ca611f", "round_id": 1,
  "error_code": "page_missing", "message": "page(s) [2] of this trial are missing from the scan",
  "scan_quality": {"usable": false, "issues": ["page(s) [2] of this trial are missing from the scan"]}
}
```

### error

Keys: `contract_version`, `status: "error"`, `error_code`, `message`, and `session_id`,
`trial_id`, `round_id` when the active trial was already known.

```json
{
  "contract_version": 1, "status": "error", "error_code": "session_invalid",
  "message": "cannot read <run>/tuning_session.json: [Errno 2] No such file or directory: …"
}
```

## experiment_id: original id or item_id?

`candidates[].experiment_id` holds the trial sheet's **item_id**, not always the original
experiment id. `scan_evaluation.py` fills it from the manifest's `items[].item_id`.
`build_selection` makes the item_id by replacing every character other than letters, digits,
`_` and `-` with `_`. Checked in a test run: the original id `exp/A` came back as `exp_A`.
Ids from the focusing tool (`experiment_<32 hex>`) have no such characters, so for them both
are the same.

To map back, read the selection file (the session's `selection_path`):
`items[].item_id` → `items[].source.experiment_id`. The session itself (`selected_experiments`,
`accepted_settings`, proposals) always uses the original experiment ids.

## Where the files are saved

All in `<run>/print_tuning/`:

| File | Written by |
|---|---|
| `trial_r<round>_<trial8>.pdf`, `…_manifest.json`, `…_cells.npz` | `make_trial_sheet` |
| `scan_r<round>_<trial8>_<random8>.json` | `read_scan` (the evaluation, = `evaluation_path`) |
| `scan_r<round>_<trial8>_<random8>_tiles/` | `read_scan`: tile crops (`…__tile.png`) and panel PNGs (`…__Initial.png`, `…__Final.png`) |

Every `read_scan` picks a new random name that does not exist yet, so a rescan of the same trial
gets new files and never overwrites an earlier evaluation.

## summarize_scan_result(result)

A plain function, not a tool. The full `read_scan` result is about 46 KB; the summary is about
3.5 KB for 4 experiments × 6 candidates. **The executor model gets the summary; the photo quality
specialist gets the full result** (images, panels, measurement_docs, printer_response).

- `needs_rescan` / `error`: only `contract_version`, `status`, `error_code`, `message`,
  `session_id`, `trial_id`, `round_id` (whichever are present).
- `ok`: `contract_version`, `status`, `session_id`, `trial_id`, `round_id`, `evaluation_path`,
  `scan_quality` (`usable`, `issues`), `pages` (`page_number`, `code_matches`, `issues`),
  `candidate_columns` and `experiments`. Each experiment has its `experiment_id` (the item_id)
  and one row per candidate, in the order of `candidate_columns`:
  `candidate_id`, `role`, `mapping` (the mapping label), `warnings`, `collapsed_levels`,
  `full_black_share`, `paper_white_share`. Numbers are rounded to 3 decimals.
- No image paths, no per-panel numbers, no `measurement_docs`, no `printer_response`.
- Anything unexpected (a missing key, not a dict): `status: "error"`, `error_code: "summary_failed"`.

```json
{
  "contract_version": 1, "status": "ok", "session_id": "cal_x",
  "trial_id": "a9848236-c778-471e-bdcd-0b6967ca611f", "round_id": 1,
  "evaluation_path": "<run>/print_tuning/scan_r1_a9848236_db799bba.json",
  "scan_quality": {"usable": true, "issues": []},
  "pages": [{"page_number": 1, "code_matches": true, "issues": []},
            {"page_number": 2, "code_matches": true, "issues": []}],
  "candidate_columns": ["candidate_id", "role", "mapping", "warnings",
                        "collapsed_levels", "full_black_share", "paper_white_share"],
  "experiments": [{
    "experiment_id": "experiment_53a404f7d2444d81beaa960fb11d4c8a",
    "candidates": [
      ["reference", "reference", "γ 1.00 · 0.00–1.00 · smooth", [], 0, 0.138, 0.001],
      ["test_1", "test", "γ 0.40 · 0.00–1.00 · smooth", [], 0, 0.007, 0.001]
    ]
  }]
}
```

## Error codes

| Code | Status | From | Meaning |
|---|---|---|---|
| `session_invalid` | error | both tools | `tuning_session.json` is missing or unreadable, a key is wrong, the selection does not match, the previous round's manifest is missing, or the active manifest belongs to another trial |
| `proposal_missing` | error | make_trial_sheet | the round has no `proposal` in the session yet |
| `proposal_invalid` | error | make_trial_sheet | the proposal does not fit the round's schema, is not `usable`, lacks a test, gives an out-of-range mapping, or `round_id` is not 1, 2 or 3 |
| `trial_sheet_failed` | error | make_trial_sheet | `create_trial_sheet` refused (bad panel data, duplicate test, limits changed between rounds, …) |
| `no_active_trial` | error | read_scan | the session has no `active_trial_id` yet |
| `internal_error` | error | both tools | an unexpected exception; logged with its traceback |
| `summary_failed` | error | summarize_scan_result | the input was not a read_scan result |
| `manifest_missing` | error | read_scan | the active trial's manifest cannot be read |
| `cells_missing` | error | read_scan | the trial's `_cells.npz` is missing |
| `scan_unreadable` | needs_rescan | read_scan | a scan file is missing, of the wrong type, cannot be opened, or is landscape |
| `low_dpi` | needs_rescan | read_scan | a page is below the config's `min_dpi` (150; 300 recommended) |
| `markers_not_found` | needs_rescan | read_scan | a corner marker is missing, too faint or cut off, or the markers do not fit the page |
| `page_mismatch` | needs_rescan | read_scan | a page code matches no page of this trial (another round's sheet?) or a page was scanned twice |
| `page_missing` | needs_rescan | read_scan | a page of the trial is not in the scan |

Note: a mistyped scan path also gives `scan_unreadable` (needs_rescan), because the file is not found.

## Fake scans for tests

`app/print_tuning/fake_scan.py`, only for tests and fake-printer dry runs:

```python
from app.print_tuning.fake_scan import fake_scan
paths = fake_scan(trial["pdf_path"], out_dir, dpi=300, rotate_deg=0.0, noise=0.0, darken=0.0, seed=0)
result = read_scan.invoke({"scan_paths": paths})
```

| Argument | Meaning |
|---|---|
| `dpi` | render resolution (> 0), also written into the PNGs; 72 gives `low_dpi` |
| `rotate_deg` | small rotation, at most ±5°, white fill |
| `noise` | Gaussian noise standard deviation on the 0..1 gray scale (>= 0) |
| `darken` | darkens the whole page, paper included, by this fraction (0 <= darken < 1), like darker paper or a dim scan |
| `seed` | makes the noise repeatable |

It writes `scan_page_<n>.png` and returns the paths in page order. It never overwrites: if a
PNG of that name exists it raises ValueError, as it does for bad arguments.
