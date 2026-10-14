"""Tests for print_tuning/selection.py.

Run with `python -m pytest print_tuning/tests`, or without pytest as
`python print_tuning/tests/test_selection.py`.
"""
import json
from pathlib import Path
import sys
import tempfile

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # repo root, for running as a script

from print_tuning.grayscale_render import default_mapping, load_config
from print_tuning.selection import SelectionError, build_selection, pick_by_ids
from print_tuning.trial_sheet import create_trial_sheet

TARGET = (16, 16)


class raises:
    """Like pytest.raises, so the tests also run without pytest."""

    def __init__(self, error):
        self.error = error
        self.message = ""

    def __enter__(self):
        return self

    def __exit__(self, kind, value, traceback):
        if kind is None:
            raise AssertionError(f"{self.error.__name__} was not raised")
        self.message = str(value)
        return issubclass(kind, self.error)


def make_results(folder, n=6):
    """n fake successful results laid out like tools.py writes them, with 32 x 32 camera .npy files."""
    rng = np.random.default_rng(0)
    results = []
    for i in range(n):
        experiment_id = f"experiment_{i:02d}"
        directory = Path(folder) / experiment_id
        directory.mkdir()
        paths = {"initial_camera": str(directory / "initial_camera.png"),
                 "final_camera": str(directory / "final_camera.png"),
                 "initial_intensity": str(directory / "initial_intensity.npy"),
                 "final_intensity": str(directory / "final_intensity.npy"),
                 "final_pattern": str(directory / "final_pattern.npy"),
                 "results_json": str(directory / "results.json"),
                 "experiments_csv": str(Path(folder) / "experiments.csv")}
        np.save(paths["initial_intensity"], rng.random((32, 32)))
        np.save(paths["final_intensity"], rng.random((32, 32)))
        results.append({"ok": True, "experiment_id": experiment_id,
                        "settings": {"focus_size": 5, "target_position": list(TARGET), "iterations": 100},
                        "initial_enhancement": 1.0 + i, "final_enhancement": 10.0 * (i + 1), "paths": paths})
    return results


def test_chosen_order_is_kept():
    with tempfile.TemporaryDirectory() as folder:
        results = make_results(folder)
        chosen = [results[4], results[1], results[5], results[0]]
        selection = build_selection(chosen, "run_1", folder, "mkrtich_selection")
        assert [item["source"]["experiment_id"] for item in selection["items"]] == \
            ["experiment_04", "experiment_01", "experiment_05", "experiment_00"]
        assert selection["chosen_by"] == "mkrtich_selection" and selection["count"] == 4
        assert selection["schema_version"] == "selection/1" and selection["selection_version"] == 1
        item = selection["items"][0]
        assert item["label"] == "experiment_04 · enhancement 5.0 → 50.0"
        assert [p["name"] for p in item["panels"]] == ["Initial", "Final"]
        assert item["panels"][0]["data_path"] == results[4]["paths"]["initial_intensity"]


def test_pick_by_ids_keeps_given_order_and_rejects_bad_ids():
    with tempfile.TemporaryDirectory() as folder:
        results = make_results(folder)
        ids = ["experiment_03", "experiment_00", "experiment_05"]
        assert [r["experiment_id"] for r in pick_by_ids(results, ids)] == ids
        with raises(SelectionError):
            pick_by_ids(results, ["experiment_00", "experiment_99"])
        with raises(SelectionError):
            pick_by_ids(results, ["experiment_00", "experiment_00"])
        results[2] = {"ok": False, "experiment_id": "experiment_02", "error": "failed"}
        with raises(SelectionError):
            pick_by_ids(results, ["experiment_02"])


def test_wrong_count_repeats_and_not_ok_raise():
    with tempfile.TemporaryDirectory() as folder:
        results = make_results(folder)
        for chosen in (results[:3], results[:5], [results[0], results[1], results[2], results[0]],
                       results[:3] + [dict(results[3], ok=False)]):
            with raises(SelectionError):
                build_selection(chosen, "run_1", folder, "orchestrator")


def test_missing_npy_asks_for_raw_arrays():
    with tempfile.TemporaryDirectory() as folder:
        results = make_results(folder)
        Path(results[2]["paths"]["final_intensity"]).unlink()
        with raises(SelectionError) as caught:
            build_selection(results[:4], "run_1", folder, "orchestrator")
        assert "must save the raw 32 x 32 camera arrays as .npy" in caught.message


def test_camera_named_npy_is_used_when_no_intensity_path():
    with tempfile.TemporaryDirectory() as folder:
        results = make_results(folder)
        for result in results[:4]:
            for stage in ("initial", "final"):
                old = Path(result["paths"].pop(f"{stage}_intensity"))
                old.rename(Path(result["paths"][f"{stage}_camera"]).with_suffix(".npy"))
        selection = build_selection(results[:4], "run_1", folder, "orchestrator")
        assert selection["items"][0]["panels"][1]["data_path"].endswith("final_camera.npy")


def test_marks_from_target_and_focus_size():
    with tempfile.TemporaryDirectory() as folder:
        results = make_results(folder)
        selection = build_selection(results[:4], "run_1", folder, "orchestrator", target_position=(10, 12))
        for item in selection["items"]:
            for panel in item["panels"]:
                assert panel["marks"] == [{"type": "box", "row": 10, "col": 12, "size": 5}]
        selection = build_selection(results[:4], "run_1", folder, "orchestrator")
        assert selection["items"][0]["panels"][0]["marks"][0] == \
            {"type": "box", "row": TARGET[0], "col": TARGET[1], "size": 5}

        for result in results[:4]:
            del result["settings"]["focus_size"]
        selection = build_selection(results[:4], "run_1", folder, "orchestrator", target_position=(10, 12))
        assert all(panel["marks"] == [] for item in selection["items"] for panel in item["panels"])


def test_selection_json_matches_return_value():
    with tempfile.TemporaryDirectory() as folder:
        selection = build_selection(make_results(folder)[:4], "run_1", Path(folder) / "out", "orchestrator")
        saved = json.loads((Path(folder) / "out" / "selection.json").read_text(encoding="utf-8"))
        assert saved == json.loads(json.dumps(selection))


def test_selection_goes_into_create_trial_sheet():
    config = load_config()
    with tempfile.TemporaryDirectory() as folder:
        results = make_results(folder)
        chosen = pick_by_ids(results, ["experiment_05", "experiment_02", "experiment_03", "experiment_01"])
        selection = build_selection(chosen, "run_1", folder, "manual_ids")
        base = default_mapping(config)
        variations = {item["item_id"]:
                      [{"variation_id": "ref", "role": "reference", "mapping": base, "note": "best so far"}] +
                      [{"variation_id": f"g{k}", "role": "test", "mapping": base | {"gamma": g}, "note": ""}
                       for k, g in enumerate((0.4, 0.6, 1.5, 2.2, 3.0))]
                      for item in selection["items"]}
        record = create_trial_sheet(selection, variations, 1, folder, "cal_run_1")
        assert record["pdf"]["pages"] == 2
        assert [item["item_id"] for item in record["items"]] == [item["item_id"] for item in selection["items"]]


if __name__ == "__main__":
    tests = [(name, test) for name, test in list(globals().items()) if name.startswith("test_")]
    failed = 0
    for name, test in tests:
        try:
            test()
            print(f"PASS {name}")
        except Exception as error:
            failed += 1
            print(f"FAIL {name}: {type(error).__name__}: {error}")
    print(f"{len(tests) - failed} passed, {failed} failed")
    sys.exit(1 if failed else 0)
