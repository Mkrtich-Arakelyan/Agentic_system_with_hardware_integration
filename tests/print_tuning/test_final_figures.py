"""render_final_heatmaps on a session completed through the real tuning flow, with different final settings."""

import hashlib
import json
from pathlib import Path

import matplotlib.axes
import numpy as np
import pytest
from PIL import Image

import app.print_tuning.final_figures as final_figures
from app.print_tuning.fake_scan import fake_scan
from app.print_tuning.schemas import (GammaProposalResponse, GammaReviewResponse, GrayLevelsReviewResponse,
                                      StepsReviewResponse)
from app.print_tuning.tuning_tools import create_tuning_tools
from app.workflow.tuning import (accept_specialist_response, build_tuning_context, initialize_tuning_session,
                                 load_tuning_session, record_scan_for_executor, record_trial_result)

IDS = ("exp/A", "exp-B", "exp-C", "exp-D")       # exp/A becomes item exp_A on the trial sheet
SHAPE, TARGET = (40, 40), (20, 20)
GAMMAS = (0.4, 0.6, 1.5, 2.2, 3.0)
PAIRS = ((0.02, 1.0), (0.04, 1.0), (0.06, 0.95), (0.08, 0.9), (0.1, 0.85))
STEPS = (2, 4, 8, 16, 32)
# Different winners per experiment, so the final settings differ (reference: gamma 1.0, steps 0).
GAMMA_WINNERS = {"exp/A": "test_2", "exp-B": "reference", "exp-C": "test_4", "exp-D": "test_1"}
ENDPOINT_WINNER = "test_3"                         # shared: dark 0.06, light 0.95
STEP_WINNERS = {"exp/A": "reference", "exp-B": "test_2", "exp-C": "test_3", "exp-D": "test_1"}
EXPECTED_STEPS = {"exp/A": 0, "exp-B": 4, "exp-C": 8, "exp-D": 2}


def _winner(candidate, experiment=None):
    value = dict(candidate_id=candidate, reason="Most readable tested candidate", confidence="medium",
                 tie_explanation="No visible difference" if candidate == "reference" else None)
    return value | ({"experiment_id": experiment} if experiment else {})


def _response(stage):
    if stage == "propose_gamma":
        return GammaProposalResponse(stage=stage, experiments=[dict(experiment_id=e, tests=[
            dict(candidate_id=f"test_{i}", gamma=g, reason="Explore tone curve") for i, g in enumerate(GAMMAS, 1)])
            for e in IDS])
    common = dict(stage=stage, assessment="usable", issues=[])
    if stage == "review_gamma":
        return GammaReviewResponse(**common, gamma_winners=[_winner(GAMMA_WINNERS[e], e) for e in IDS],
                                   endpoint_tests=[dict(candidate_id=f"test_{i}", dark_level=d, light_level=l,
                                                        reason="Compare endpoints") for i, (d, l) in enumerate(PAIRS, 1)])
    if stage == "review_gray_levels":
        return GrayLevelsReviewResponse(**common, endpoint_winner=_winner(ENDPOINT_WINNER), step_tests=[
            dict(experiment_id=e, tests=[dict(candidate_id=f"test_{i}", steps=s, reason="Compare bands")
                                         for i, s in enumerate(STEPS, 1)]) for e in IDS])
    return StepsReviewResponse(**common, step_winners=[_winner(STEP_WINNERS[e], e) for e in IDS])


def _camera(index: int, focused: bool) -> np.ndarray:
    """A smooth background with one bright focus spot at the target, different per experiment."""
    rows, cols = np.indices(SHAPE)
    background = 0.2 + 0.05 * index + 0.3 * np.sin(rows / (5 + index)) * np.cos(cols / 7)
    spot = (2.0 + index if focused else 0.6) * np.exp(-((rows - TARGET[0]) ** 2 + (cols - TARGET[1]) ** 2) / 6)
    return np.clip(background, 0, None) + spot


def build_completed_session(root: Path) -> Path:
    """Run rounds 1 to 3 through the real session, tool, fake scan and acceptance functions."""
    root.mkdir(parents=True, exist_ok=True)
    results, selections = [], []
    for index, experiment in enumerate(IDS):
        folder = root / f"source_{index}"
        folder.mkdir()
        paths = {"initial_camera": str(folder / "initial.png"), "final_camera": str(folder / "final.png"),
                 "results_json": str(folder / "results.json")}
        for stage, focused in (("initial", False), ("final", True)):
            np.save(folder / f"{stage}.npy", _camera(index, focused))
            paths[f"{stage}_intensity"] = str(folder / f"{stage}.npy")
        record = dict(ok=True, experiment_id=experiment, settings={"iterations": 3, "focus_size": 3},
                      initial_enhancement=1.0, final_enhancement=2.0 + index, paths=paths)
        (folder / "results.json").write_text(json.dumps(record), encoding="utf-8")
        results.append(record)
        selections.append(dict(experiment_id=experiment, validity="Valid", comparability="Comparable",
                               relevance="Relevant", information_gained="Distinct evidence", paths=paths))
    (root / "config.json").write_text(json.dumps({"bench": {"target_position": list(TARGET)}}))
    initialize_tuning_session(root, dict(selections=selections, group_reason="Four complementary experiments"), results)
    path = root / "tuning_session.json"
    accept_specialist_response(path, _response("propose_gamma"), build_tuning_context(path, "propose_gamma"))
    make_trial_sheet, read_scan = create_tuning_tools(root)
    for number, stage in ((1, "review_gamma"), (2, "review_gray_levels"), (3, "review_steps")):
        trial = make_trial_sheet.invoke({"round_id": number})
        assert trial["status"] == "ok", trial
        record_trial_result(path, trial)
        scan = read_scan.invoke({"scan_paths": fake_scan(trial["pdf_path"], root / f"scan_{number}")})
        assert scan["status"] == "ok", scan
        record_scan_for_executor(path, scan, record_id=f"scan_{number}")
        accept_specialist_response(path, _response(stage), build_tuning_context(path, stage),
                                   evaluation_record_id=f"scan_{number}")
    return path


@pytest.fixture(scope="module")
def completed(tmp_path_factory):
    path = build_completed_session(tmp_path_factory.mktemp("completed") / "run")
    session = load_tuning_session(path)
    assert session["status"] == "completed" and session["phase"] == "completed"
    return path, session


@pytest.fixture
def spies(monkeypatch):
    """Record every apply_mapping call made by final_figures, and every 2D image it draws."""
    mapped, drawn = [], []
    real_apply, real_imshow = final_figures.apply_mapping, matplotlib.axes.Axes.imshow

    def apply_spy(data, limits, mapping):
        grays = real_apply(data, limits, mapping)
        mapped.append({"data": np.array(data), "limits": dict(limits), "mapping": dict(mapping), "grays": grays})
        return grays

    def imshow_spy(self, values, *args, **kwargs):
        drawn.append(np.array(values))
        return real_imshow(self, values, *args, **kwargs)

    monkeypatch.setattr("app.print_tuning.final_figures.apply_mapping", apply_spy)
    monkeypatch.setattr(matplotlib.axes.Axes, "imshow", imshow_spy)
    return mapped, drawn


def test_session_has_different_final_settings(completed):
    _, session = completed
    final = session["final_settings"]
    assert {e: final[e]["steps"] for e in IDS} == EXPECTED_STEPS
    assert len({final[e]["gamma"] for e in IDS}) == 4                         # 0.6, 1.0, 2.2, 0.4
    assert {(final[e]["dark_level"], final[e]["light_level"]) for e in IDS} == {(0.06, 0.95)}


def test_report_pairs_use_both_arrays_with_original_and_final_mappings(completed, spies, tmp_path):
    path, session = completed
    mapped, drawn = spies
    result = final_figures.render_report_comparison(path, tmp_path)
    assert result["columns"] == ["Initial", "Final", "Initial", "Final"]
    assert result["groups"] == ["Before printer tuning", "After printer tuning"]
    assert [row["label"] for row in result["experiments"]] == ["E1", "E2", "E3", "E4"]
    assert len(mapped) == 16
    for i, eid in enumerate(IDS):
        for group, mapping in enumerate((session["rounds"]["1"]["reference_settings"][eid], session["final_settings"][eid])):
            for stage in (0, 1):
                call = mapped[4*i+2*group+stage]
                assert np.array_equal(call["data"], _camera(i, bool(stage)))
                assert call["mapping"] == mapping
                assert call["limits"] == session["selected_experiments"][i]["limits"]
    panels = [values for values in drawn if values.shape == SHAPE]
    assert len(panels) == 16 and all(np.array_equal(v, call["grays"]) for v, call in zip(panels, mapped))


def test_report_uses_first_cycle_baseline_and_last_cycle_accepted_settings(completed, spies, tmp_path):
    from copy import deepcopy
    path, original = completed
    session = deepcopy(original)
    first = deepcopy(session["rounds"])
    # Distinct second-cycle reference: it MUST NOT become the Before mapping.
    latest = deepcopy(first)
    latest["1"]["reference_settings"] = deepcopy(session["final_settings"])
    final = deepcopy(session["final_settings"])
    for i, eid in enumerate(IDS):
        final[eid]["gamma"] = .9 + .1 * i
    latest["3"]["winners"] = deepcopy(final)
    session.update(contract_version=2, active_cycle=2, rounds=latest,
                   final_settings=final,
                   original_settings=deepcopy(first["1"]["reference_settings"]),
                   cycles={"1": {"steps": first}, "2": {"steps": latest}})
    copy = tmp_path / "tuning_session.json"
    copy.write_text(json.dumps(session), encoding="utf-8")
    result = final_figures.render_report_comparison(copy, tmp_path)
    assert result["mapping_sources"]["before"]["cycle_id"] == 1
    assert result["mapping_sources"]["after"]["cycle_id"] == 2
    for row in result["experiments"]:
        eid = row["experiment_id"]
        assert row["before_mapping"] == first["1"]["reference_settings"][eid]
        assert row["before_mapping"] != latest["1"]["reference_settings"][eid]
        assert row["after_mapping"] == latest["3"]["winners"][eid]
    session["final_settings"][IDS[0]]["gamma"] = .7
    copy.write_text(json.dumps(session), encoding="utf-8")
    with pytest.raises(ValueError, match="last completed cycle"):
        final_figures.render_report_comparison(copy, tmp_path)


@pytest.mark.parametrize("rectangular", [False, True])
def test_comparison_layout_keeps_labels_panels_aspects_and_annotations(completed, monkeypatch, tmp_path, rectangular):
    path, _ = completed
    marks = []
    if rectangular:
        reader = final_figures.load_panel_data
        monkeypatch.setattr(final_figures, "load_panel_data", lambda *args: np.repeat(reader(*args), 2, axis=1))
    real = final_figures._draw_box_mark
    def mark_spy(ax, mark, shape, box):
        marks.append((dict(mark), shape, box))
        return real(ax, mark, shape, box)
    monkeypatch.setattr(final_figures, "_draw_box_mark", mark_spy)
    # Capture canvas at draw time, before the renderer frees it.
    real_draw = final_figures.FigureCanvasAgg.draw
    captured = {}
    def draw(canvas):
        real_draw(canvas)
        captured["texts"] = [(t.get_text(), t.get_fontsize()) for ax in canvas.figure.axes for t in ax.texts]
        axes = [ax for ax in canvas.figure.axes if ax.images]
        captured["sizes"] = [(ax.get_position().width, ax.get_position().height) for ax in axes]
        captured["aspects"] = [ax.get_aspect() for ax in axes]
        captured["frames"] = [patch.get_bbox().bounds for patch in canvas.figure.axes[0].patches]
    monkeypatch.setattr(final_figures.FigureCanvasAgg, "draw", draw)
    final_figures.render_report_comparison(path, tmp_path)
    selection = json.loads(Path(json.loads(path.read_text())["selection_path"]).read_text())
    expected_marks = 2 * sum(len(item.get("marks", [])) + len(panel.get("marks", []))
                             for item in selection["items"] for panel in item["panels"])
    assert len(marks) == expected_marks > 0
    assert len(captured["sizes"]) == 16
    assert np.allclose(captured["sizes"], captured["sizes"][0])
    w, h = captured["sizes"][0]
    assert w * 180 / (h * 184) == pytest.approx(2 if rectangular else 1)
    assert captured["aspects"] == [1.] * 16
    assert len(captured["frames"]) == 16
    assert all((w, h) == (37.5, 37.5) for x, y, w, h in captured["frames"])
    frames = captured["frames"]
    assert frames[2][0] - frames[1][0] - 37.5 > frames[1][0] - frames[0][0] - 37.5
    assert all(size >= 10 for _, size in captured["texts"])
    assert [t for t, _ in captured["texts"] if t.startswith("E")] == ["E1", "E2", "E3", "E4"]


def test_figure_uses_exactly_the_saved_settings_and_frozen_limits(completed, spies, tmp_path):
    path, session = completed
    mapped, drawn = spies
    result = final_figures.render_final_heatmaps(path, tmp_path)
    frozen = {e["experiment_id"]: e["limits"] for e in session["selected_experiments"]}
    assert [row["experiment_id"] for row in result["experiments"]] == list(IDS)
    for row in result["experiments"]:
        assert row["mapping"] == session["final_settings"][row["experiment_id"]]
        assert row["limits"] == frozen[row["experiment_id"]]

    # apply_mapping: Initial then Final of every experiment, in order, with its limits and final mapping
    assert len(mapped) == 2 * len(IDS)
    for index, experiment in enumerate(IDS):
        for panel, stage in enumerate(("initial", "final")):
            call = mapped[2 * index + panel]
            assert call["limits"] == frozen[experiment] and call["mapping"] == session["final_settings"][experiment]
            assert np.array_equal(call["data"], _camera(index, stage == "final"))
    # what is drawn in the panels is exactly what apply_mapping returned
    panels = [image for image in drawn if image.shape == SHAPE]
    assert len(panels) == len(mapped)
    assert all(np.array_equal(image, call["grays"]) for image, call in zip(panels, mapped))

    for index, experiment in enumerate(IDS):
        levels = {len(np.unique(mapped[2 * index + panel]["grays"])) for panel in (0, 1)}
        if EXPECTED_STEPS[experiment] == 4:
            assert max(levels) <= 4, (experiment, levels)
        if EXPECTED_STEPS[experiment] == 0:
            assert min(levels) > 4, (experiment, levels)


def test_png_is_gray_with_its_size_dpi_and_hash(completed, tmp_path):
    path, _ = completed
    result = final_figures.render_final_heatmaps(path, tmp_path)
    figure = Path(result["figure_path"])
    with Image.open(figure) as image:
        assert image.mode == "L"
        assert image.info["dpi"] == pytest.approx((final_figures.DPI, final_figures.DPI), abs=0.01)
        expected = [mm / final_figures.MM_PER_INCH * final_figures.DPI for mm in (result["width_mm"], result["height_mm"])]
        assert all(abs(px - want) < 1 for px, want in zip(image.size, expected)), (image.size, expected)
    assert result["figure_sha256"] == hashlib.sha256(figure.read_bytes()).hexdigest()


def test_second_call_never_overwrites(completed, tmp_path):
    path, _ = completed
    first = final_figures.render_final_heatmaps(path, tmp_path)
    before = Path(first["figure_path"]).read_bytes()
    second = final_figures.render_final_heatmaps(path, tmp_path)
    assert second["figure_path"] != first["figure_path"]
    assert Path(first["figure_path"]).read_bytes() == before and Path(second["figure_path"]).is_file()


def _broken_session(path: Path, folder: Path, change) -> Path:
    session = json.loads(path.read_text(encoding="utf-8"))
    change(session, folder)
    broken = folder / "tuning_session.json"
    broken.write_text(json.dumps(session), encoding="utf-8")
    return broken


def _drop_final_settings(session, folder):
    session["final_settings"] = None


def _drop_one_experiment(session, folder):
    del session["final_settings"]["exp-C"]


def _other_dark_level(session, folder):
    session["final_settings"]["exp-B"]["dark_level"] = 0.02


def _gamma_out_of_range(session, folder):
    session["final_settings"]["exp-D"]["gamma"] = 9.0


def _flat_limits(session, folder):
    limits = session["selected_experiments"][1]["limits"]
    limits["lo"] = limits["hi"]


def _missing_panel_file(session, folder):
    selection = json.loads(Path(session["selection_path"]).read_text(encoding="utf-8"))
    selection["items"][2]["panels"][1]["data_path"] = str(folder / "gone.npy")
    copy = folder / "selection_copy.json"
    copy.write_text(json.dumps(selection), encoding="utf-8")
    session["selection_path"] = str(copy)


@pytest.mark.parametrize("change, message", [
    (_drop_final_settings, "final_settings .* is missing"),
    (_drop_one_experiment, r"final_settings has no mapping for \['exp-C'\]"),
    (_other_dark_level, "dark_level and light_level are shared"),
    (_gamma_out_of_range, r"final_settings\['exp-D'\] is not a valid mapping: gamma 9.0 is outside"),
    (_flat_limits, r"selected_experiments\['exp-B'\].limits needs finite lo < hi"),
    (_missing_panel_file, "gone.npy does not exist"),
])
def test_bad_sessions_raise_and_write_no_png(completed, tmp_path, change, message):
    path, _ = completed
    broken = _broken_session(path, tmp_path, change)
    out_dir = tmp_path / "figures"
    with pytest.raises(ValueError, match=message):
        final_figures.render_final_heatmaps(broken, out_dir)
    assert not out_dir.exists() or not any(out_dir.iterdir())
