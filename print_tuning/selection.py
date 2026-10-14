"""Package experiments that the experiment side already chose into a trial sheet selection.

This module does not choose experiments: no ranking, no top-N and no random picks. The
choice belongs to the experiment code; here results are only looked up, checked and packaged
into the `selection` dict that print_tuning/trial_sheet.py create_trial_sheet() expects.

Command line, to print real experiments by hand:
    python print_tuning/selection.py RUN_FOLDER --ids ID1 ID2 ID3 ID4 [--out OUT]
"""
import argparse
import json
import math
import re
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # repo root, for running as a script

from print_tuning.grayscale_render import default_mapping, load_config
from print_tuning.trial_sheet import create_trial_sheet

SCHEMA_VERSION = "selection/1"
PANELS = ("Initial", "Final")
CLI_GAMMA_TESTS = (0.4, 0.6, 1.5, 2.2, 3.0)


class SelectionError(ValueError):
    """The chosen results cannot be packaged into a selection."""


def _check_ok(result):
    if not isinstance(result, dict) or result.get("ok") is not True:
        experiment_id = result.get("experiment_id") if isinstance(result, dict) else None
        raise SelectionError(f"experiment {experiment_id!r} is not a successful result (ok is not True)")


def pick_by_ids(results, experiment_ids):
    """Return the results with these experiment_ids, in the order given. Decides nothing."""
    by_id = {result.get("experiment_id"): result for result in results if isinstance(result, dict)}
    picked, seen = [], set()
    for experiment_id in experiment_ids:
        if experiment_id in seen:
            raise SelectionError(f"experiment {experiment_id!r} is listed more than once")
        seen.add(experiment_id)
        if experiment_id not in by_id:
            raise SelectionError(f"experiment {experiment_id!r} is not among the results")
        _check_ok(by_id[experiment_id])
        picked.append(by_id[experiment_id])
    return picked


def _raw_camera_path(result, stage):
    """The raw camera .npy of stage 'initial' or 'final'."""
    paths = result.get("paths") or {}
    candidates = [paths.get(f"{stage}_camera_npy"),
                  paths.get(f"{stage}_intensity")]           # what tools.py saves today
    if paths.get(f"{stage}_camera"):
        candidates.append(str(Path(paths[f"{stage}_camera"]).with_suffix(".npy")))
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return Path(candidate)
    raise SelectionError(f"experiment {result['experiment_id']}: no raw {stage} camera .npy found "
                         f"(tried {[c for c in candidates if c]}); the experiment code must save the raw "
                         "32 x 32 camera arrays as .npy")


def _check_raw(path, experiment_id):
    try:
        data = np.load(path, allow_pickle=False)
    except Exception as error:
        raise SelectionError(f"experiment {experiment_id}: cannot read {path}: {error}") from error
    if not isinstance(data, np.ndarray) or data.ndim != 2 or data.size == 0:
        raise SelectionError(f"experiment {experiment_id}: {path} must hold a non-empty 2D array")
    if not (np.issubdtype(data.dtype, np.integer) or np.issubdtype(data.dtype, np.floating)) \
            or not np.all(np.isfinite(data)):
        raise SelectionError(f"experiment {experiment_id}: {path} must hold finite numbers only")


def _box_mark(result, target_position):
    settings = result.get("settings") or {}
    target = target_position or result.get("target_position") or settings.get("target_position")
    size = settings.get("focus_size")
    if target is None or size is None:
        return []
    row, col = target
    return [{"type": "box", "row": row, "col": col, "size": size}]


def _number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def build_selection(chosen_results, run_id, out_dir, chosen_by, target_position=None, count=4):
    """Package the already chosen results, keeping their order; write out_dir/selection.json."""
    if len(chosen_results) != count:
        raise SelectionError(f"expected {count} chosen experiments, got {len(chosen_results)}")
    items, seen = [], set()
    for result in chosen_results:
        _check_ok(result)
        experiment_id = result.get("experiment_id")
        if not isinstance(experiment_id, str) or not experiment_id:
            raise SelectionError(f"a chosen result has no experiment_id: {experiment_id!r}")
        if experiment_id in seen:
            raise SelectionError(f"experiment {experiment_id!r} is chosen more than once")
        seen.add(experiment_id)
        initial, final = result.get("initial_enhancement"), result.get("final_enhancement")
        if not (_number(initial) and _number(final)):
            raise SelectionError(f"experiment {experiment_id}: enhancements must be finite numbers")

        item_id = re.sub(r"[^A-Za-z0-9_-]", "_", experiment_id)
        marks = _box_mark(result, target_position)
        panels = []
        for name in PANELS:
            path = _raw_camera_path(result, name.lower())
            _check_raw(path, experiment_id)
            panels.append({"name": name, "kind": "heatmap", "data_path": str(path),
                           "marks": [dict(mark) for mark in marks]})
        items.append({
            "item_id": item_id,
            "label": f"{experiment_id} · enhancement {initial:.1f} → {final:.1f}",
            "panels": panels,
            "source": {"experiment_id": experiment_id, "settings": result.get("settings"),
                       "initial_enhancement": initial, "final_enhancement": final,
                       "results_json": (result.get("paths") or {}).get("results_json")},
        })
    if len({item["item_id"] for item in items}) != len(items):
        raise SelectionError("two experiment_ids give the same item_id after replacing special characters")

    selection = {
        "schema_version": SCHEMA_VERSION,
        "selection_id": str(uuid.uuid4()),
        "selection_version": 1,
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "count": count,
        "chosen_by": chosen_by,
        "items": items,
    }
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "selection.json").write_text(json.dumps(selection, indent=2, ensure_ascii=False),
                                            encoding="utf-8")
    return selection


def load_run_results(run_folder):
    """Every experiment saved in a run folder by tools.py, read from its results.json."""
    results = []
    for path in sorted(Path(run_folder).glob("*/results.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        results.append({"ok": True, **record})     # tools.py only writes results.json on success
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(description="Print round-1 trial pages for experiments chosen by id.")
    parser.add_argument("run_folder", type=Path)
    parser.add_argument("--ids", nargs="+", required=True, help="the experiments chosen on the experiment side")
    parser.add_argument("--out", type=Path, help="output folder (default RUN_FOLDER/print_tuning)")
    args = parser.parse_args(argv)
    out = args.out or args.run_folder / "print_tuning"

    try:
        chosen = pick_by_ids(load_run_results(args.run_folder), args.ids)
        selection = build_selection(chosen, args.run_folder.name, out, "manual_ids")
    except SelectionError as error:
        sys.exit(f"error: {error}")

    # In the real loop the scientist agent chooses the gamma values; this command line is
    # only for trying the trial sheet by hand with fixed values.
    config = load_config()
    base = default_mapping(config)
    variations = {item["item_id"]:
                  [{"variation_id": "ref", "role": "reference", "mapping": base, "note": "best so far"}] +
                  [{"variation_id": f"g{k}", "role": "test", "mapping": base | {"gamma": gamma}, "note": ""}
                   for k, gamma in enumerate(CLI_GAMMA_TESTS)]
                  for item in selection["items"]}
    record = create_trial_sheet(selection, variations, 1, out, "cal_" + args.run_folder.name)

    for item in selection["items"]:
        print("Experiment:", item["label"])
    print("PDF:", record["pdf"]["path"], "pages:", record["pdf"]["pages"])


if __name__ == "__main__":
    main()
