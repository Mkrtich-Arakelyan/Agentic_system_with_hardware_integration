"""Package experiments that the experiment side already chose into a trial sheet selection.

This module does not choose experiments: no ranking, no top-N and no random picks. The
scientist agent chooses them (app/workflow/graph.py freezes its choice in state["experiment_selection"]);
here results are only looked up, checked and packaged into the `selection` dict that
app/print_tuning/trial_sheet.py create_trial_sheet() expects.

Command line, to print the round-1 sheet of a run:
    python -m app.experiments.selection RUN_FOLDER                              # the scientist's frozen choice
    python -m app.experiments.selection RUN_FOLDER --ids ID1 ID2 ID3 ID4        # or four ids chosen by hand
    [--out OUT]                                                 # default RUN_FOLDER/print_tuning
"""
import argparse
import hashlib
import json
import math
import re
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path, PureWindowsPath

from app.print_tuning.grayscale_render import _is_number, default_mapping, load_config
from app.print_tuning.trial_sheet import TrialSheetError, create_trial_sheet, load_panel_data

SCHEMA_VERSION = "selection/1"
PANELS = ("Initial", "Final")
REASON_KEYS = ("validity", "comparability", "relevance", "information_gained")
CLI_GAMMA_TESTS = (0.4, 0.6, 1.5, 2.2, 3.0)


class SelectionError(ValueError):
    """The chosen results cannot be packaged into a selection."""


def _finite(value) -> bool:
    """A real, finite number (not a bool, NaN or infinity)."""
    try:
        return _is_number(value) and math.isfinite(value)
    except OverflowError:                     # an int too large for a float
        return False


def _file_sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def successful_results(state: dict) -> list[dict]:
    """The successful focusing results recorded in a graph state.

    A copy of app/workflow/graph.py _successful_focusing_results (not imported, so this file does not depend
    on the graph): only tool payloads count, not a results.json lying in the run folder.
    """
    successful = []
    for record in state.get("execution_results", []):
        if (not isinstance(record, dict) or record.get("tool_name") != "run_focusing_experiment"
                or record.get("validation_error")):
            continue
        result = record.get("result")
        if not isinstance(result, dict) or result.get("ok") is not True:
            continue
        if not isinstance(result.get("experiment_id"), str) or not result["experiment_id"]:
            continue
        if not isinstance(result.get("settings"), dict) or not result["settings"]:
            continue
        if not all(_finite(result.get(name)) for name in ("initial_enhancement", "final_enhancement")):
            continue
        paths = result.get("paths")
        if isinstance(paths, dict) and all(isinstance(paths.get(name), str) and paths[name]
                                           for name in ("initial_camera", "final_camera", "results_json")):
            successful.append(result)
    return successful


def _check_ok(result) -> None:
    if not isinstance(result, dict) or result.get("ok") is not True:
        experiment_id = result.get("experiment_id") if isinstance(result, dict) else None
        raise SelectionError(f"experiment {experiment_id!r} is not a successful result (ok is not True)")


def pick_by_ids(results: list[dict], experiment_ids: list[str]) -> list[dict]:
    """Return the results with these experiment_ids, in the order given. Decides nothing."""
    by_id = {result.get("experiment_id"): result for result in results if isinstance(result, dict)}
    if len(set(experiment_ids)) != len(experiment_ids):
        raise SelectionError(f"an experiment is listed more than once in {experiment_ids}")
    unknown = [experiment_id for experiment_id in experiment_ids if experiment_id not in by_id]
    if unknown:
        raise SelectionError(f"experiments {unknown} are not among the results")
    for experiment_id in experiment_ids:
        _check_ok(by_id[experiment_id])
    return [by_id[experiment_id] for experiment_id in experiment_ids]


def _raw_camera_path(result: dict, stage: str) -> str:
    """The raw camera .npy of stage 'initial' or 'final', checked to be a finite 2D array."""
    paths = result.get("paths") or {}
    candidates = [paths.get(f"{stage}_camera_npy"), paths.get(f"{stage}_intensity")]   # app/experiments/tools.py: _intensity
    if paths.get(f"{stage}_camera"):
        candidates.append(str(Path(paths[f"{stage}_camera"]).with_suffix(".npy")))
    candidates = [candidate for candidate in candidates if candidate]
    found = next((candidate for candidate in candidates if Path(candidate).is_file()), None)
    if found is None:
        raise SelectionError(f"experiment {result['experiment_id']}: no raw {stage} camera .npy found "
                             f"(tried {candidates}); the experiment code must save the raw 32 x 32 "
                             "camera arrays as .npy")
    try:
        load_panel_data(found, f"experiment {result['experiment_id']} {stage} camera")
    except TrialSheetError as error:
        raise SelectionError(str(error)) from error
    return found


def _box_marks(result: dict, target_position) -> list[dict]:
    """A focus box at the target, sized focus_size; none if either is unknown."""
    settings = result.get("settings") or {}
    target = target_position or result.get("target_position") or settings.get("target_position")
    size = settings.get("focus_size")
    if target is None or size is None:
        return []
    return [{"type": "box", "row": target[0], "col": target[1], "size": size}]


def build_selection(chosen_results: list[dict], run_id: str, out_dir, chosen_by: str,
                    target_position=None, count: int = 4, reasons: dict | None = None,
                    group_reason: str | None = None) -> dict:
    """Package the already chosen results, keeping their order; write out_dir/selection_<id>.json.

    The selection_id is derived from run_id and the ordered experiment ids, so the same choice
    always has the same identity: if its file already exists it is read back, not overwritten,
    and every round prints the same selection. reasons, if given, maps experiment_id to why it
    was chosen and is kept in each item's source.
    """
    if len(chosen_results) != count:
        raise SelectionError(f"expected {count} chosen experiments, got {len(chosen_results)}")
    for result in chosen_results:
        _check_ok(result)
    ids = [result.get("experiment_id") for result in chosen_results]
    if not all(isinstance(experiment_id, str) and experiment_id for experiment_id in ids):
        raise SelectionError(f"every chosen result needs an experiment_id, got {ids}")
    if len(set(ids)) != len(ids):
        raise SelectionError(f"an experiment is chosen more than once in {ids}")
    item_ids = [re.sub(r"[^A-Za-z0-9_-]", "_", experiment_id) for experiment_id in ids]
    if len(set(item_ids)) != len(item_ids):
        raise SelectionError("two experiment_ids give the same item_id after replacing special characters")

    selection_id = str(uuid.uuid5(uuid.NAMESPACE_URL, "selection/" + json.dumps([run_id, ids])))
    items = []
    for result, item_id in zip(chosen_results, item_ids):
        experiment_id = result["experiment_id"]
        initial, final = result.get("initial_enhancement"), result.get("final_enhancement")
        if not (_finite(initial) and _finite(final)):
            raise SelectionError(f"experiment {experiment_id}: enhancements must be finite numbers, "
                                 f"got {initial!r} and {final!r}")
        marks = _box_marks(result, target_position)
        panels = []
        for name in PANELS:
            data_path = _raw_camera_path(result, name.lower())
            panels.append({"name": name, "kind": "heatmap", "data_path": data_path,
                           "sha256": _file_sha256(data_path), "marks": [dict(mark) for mark in marks]})
        items.append({
            "item_id": item_id,
            "label": f"enhancement {initial:.1f} → {final:.1f}",     # the sheet prints item_id next to it
            "panels": panels,
            "source": {"experiment_id": experiment_id, "settings": result.get("settings"),
                       "initial_enhancement": initial, "final_enhancement": final,
                       "results_json": (result.get("paths") or {}).get("results_json"),
                       "reasons": (reasons or {}).get(experiment_id)},
        })

    # every check is done: only now is anything read back or written
    path = Path(out_dir) / f"selection_{selection_id[:8]}.json"
    if path.exists():                  # an earlier round packaged the same choice: reuse it unchanged
        saved = json.loads(path.read_text(encoding="utf-8"))
        hashes = [[panel["sha256"] for panel in item["panels"]] for item in items]
        saved_hashes = [[panel.get("sha256") for panel in item.get("panels", [])] for item in saved.get("items", [])]
        if saved_hashes != hashes:
            raise SelectionError(f"the camera data of these experiments changed since {path} was written")
        same = {"selection_id": selection_id, "chosen_by": chosen_by, "group_reason": group_reason,
                "items": json.loads(json.dumps(items))}
        if any(saved.get(key) != value for key, value in same.items()):
            raise SelectionError(f"{path} already holds these experiments with different content; "
                                 "use another out_dir for a different selection")
        return saved

    selection = {
        "schema_version": SCHEMA_VERSION,
        "selection_id": selection_id,
        "selection_version": 1,
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "count": count,
        "chosen_by": chosen_by,
        "group_reason": group_reason,
        "items": items,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(selection, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    return selection


def selection_from_scientist(experiment_selection: dict, results: list[dict], run_id: str, out_dir,
                             target_position=None, state: dict | None = None) -> dict:
    """Package the scientist's frozen choice (app/workflow/graph.py state["experiment_selection"]).

    results hold the experiments' data paths, e.g. load_run_results(run_folder). If state is
    given, only experiments that succeeded in state["execution_results"] are accepted.
    The order and the reasons come from the scientist; nothing is chosen here.
    """
    chosen = experiment_selection.get("selections") if isinstance(experiment_selection, dict) else None
    if not isinstance(chosen, list) or not all(isinstance(entry, dict) for entry in chosen):
        raise SelectionError("experiment_selection needs a list of selections")
    ids = [entry.get("experiment_id") for entry in chosen]
    if not all(isinstance(experiment_id, str) and experiment_id for experiment_id in ids):
        raise SelectionError(f"every scientist selection needs a non-empty experiment_id, got {ids}")
    if state is not None:
        succeeded = {result["experiment_id"] for result in successful_results(state)}
        failed = [experiment_id for experiment_id in ids if experiment_id not in succeeded]
        if failed:
            raise SelectionError(f"experiments {failed} are not successful focusing results in the state")
    reasons = {entry["experiment_id"]: {key: entry[key] for key in REASON_KEYS if key in entry}
               for entry in chosen}
    return build_selection(pick_by_ids(results, ids), run_id, out_dir, "scientist", target_position,
                           reasons=reasons, group_reason=experiment_selection.get("group_reason"))


def _local_path(saved: str, experiment_dir: Path) -> str:
    """The saved path if it exists here, else the same file name in this experiment or run folder.

    results.json stores absolute paths, which break when a run folder is copied to another
    machine (for example from Windows to macOS).
    """
    if Path(saved).exists():
        return saved
    name = PureWindowsPath(saved).name           # understands both \\ and / separators
    for folder in (experiment_dir, experiment_dir.parent):
        if (folder / name).exists():
            return str(folder / name)
    return saved


def load_run_results(run_folder) -> list[dict]:
    """Every experiment saved in a run folder by app/experiments/tools.py, read from its results.json."""
    results = []
    for path in sorted(Path(run_folder).glob("*/results.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        record["paths"] = {key: _local_path(value, path.parent) for key, value in record.get("paths", {}).items()}
        results.append({"ok": True, **record})     # app/experiments/tools.py only writes results.json on success
    return results


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Print the round-1 trial sheet for the scientist's four experiments of a run.")
    parser.add_argument("run_folder", type=Path, help="a run folder written by main.py")
    parser.add_argument("--ids", nargs="+", help="choose the experiments by hand instead of the scientist")
    parser.add_argument("--out", type=Path, help="output folder (default RUN_FOLDER/print_tuning)")
    args = parser.parse_args(argv)
    run = args.run_folder
    out = args.out or run / "print_tuning"
    target = _read_json(run / "config.json").get("bench", {}).get("target_position")

    try:
        results = load_run_results(run)
        if args.ids:
            selection = build_selection(pick_by_ids(results, args.ids), run.name, out, "manual_ids", target)
        else:
            state = _read_json(run / "state.json")
            frozen = state.get("experiment_selection")
            if not frozen:
                sys.exit("error: the scientist has not chosen 4 experiments yet; use --ids to choose by hand")
            selection = selection_from_scientist(frozen, results, run.name, out, target, state=state)
    except SelectionError as error:
        sys.exit(f"error: {error}")

    # In the real loop the scientist agent chooses the gamma values; this command line is
    # only for trying the trial sheet by hand with fixed values.
    base = default_mapping(load_config())
    variations = {item["item_id"]:
                  [{"variation_id": "ref", "role": "reference", "mapping": base, "note": "best so far"}] +
                  [{"variation_id": f"g{k}", "role": "test", "mapping": base | {"gamma": gamma}, "note": ""}
                   for k, gamma in enumerate(CLI_GAMMA_TESTS)]
                  for item in selection["items"]}
    record = create_trial_sheet(selection, variations, 1, out, "cal_" + run.name)

    print("Chosen by:", selection["chosen_by"])
    for item in selection["items"]:
        source = item["source"]
        print(f"Experiment: {source['experiment_id']}  enhancement "
              f"{source['initial_enhancement']:.2f} -> {source['final_enhancement']:.2f}")
    print("PDF:", record["pdf"]["path"], "pages:", record["pdf"]["pages"])


if __name__ == "__main__":
    main()
