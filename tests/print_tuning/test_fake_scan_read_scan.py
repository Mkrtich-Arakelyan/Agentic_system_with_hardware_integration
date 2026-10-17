"""Offline round-1 loop: make_trial_sheet, fake_scan and read_scan, with good and bad fake scans."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from app.experiments.selection import build_selection
from app.print_tuning.fake_scan import fake_scan
from app.print_tuning.grayscale_render import load_config
from app.print_tuning.tuning_tools import create_tuning_tools, summarize_scan_result

IDS = ("exp_A", "exp_B", "exp_C", "exp_D")
GAMMAS = (0.4, 0.6, 1.5, 2.2, 3.0)          # the reference keeps the default gamma 1.0
TARGET, SHAPE = (16, 16), (32, 32)


def _camera(rng: np.random.Generator, peak: float) -> np.ndarray:
    image = rng.gamma(2.0, 0.2, SHAPE)
    image[TARGET] += peak
    return image


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    """A run folder with 4 experiments, a selection, a tuning session and round 1's trial sheet."""
    run_dir = tmp_path_factory.mktemp("run")
    rng = np.random.default_rng(1)
    results = []
    for number, experiment_id in enumerate(IDS):
        folder = run_dir / experiment_id
        folder.mkdir()
        paths = {"results_json": str(folder / "results.json")}
        for stage, peak in (("initial", 0.5), ("final", 2.0 + number)):
            np.save(folder / f"{stage}_camera.npy", _camera(rng, peak))
            paths[f"{stage}_camera_npy"] = str(folder / f"{stage}_camera.npy")
        results.append({"ok": True, "experiment_id": experiment_id, "settings": {"focus_size": 3},
                        "initial_enhancement": 1.0, "final_enhancement": 2.0 + number, "paths": paths})
    build_selection(results, "run", run_dir / "print_tuning", "scientist", TARGET)
    proposal = {"stage": "propose_gamma", "experiments": [
        {"experiment_id": e, "tests": [{"candidate_id": f"test_{k}", "gamma": g, "reason": "compare tone curve"}
                                       for k, g in enumerate(GAMMAS, 1)]} for e in IDS]}
    session = {"contract_version": 1, "session_id": "cal_test", "run_id": "run",
               "selection_path": str(next((run_dir / "print_tuning").glob("selection_*.json"))),
               "config_snapshot": load_config(),
               "selected_experiments": [{"experiment_id": e, "limits": None} for e in IDS],
               "rounds": {"1": {"proposal": proposal}}}
    session_path = run_dir / "tuning_session.json"
    session_path.write_text(json.dumps(session), encoding="utf-8")
    make_trial_sheet, read_scan = create_tuning_tools(run_dir)
    trial = make_trial_sheet.invoke({"round_id": 1})
    assert trial["status"] == "ok", trial
    session["rounds"]["1"]["trial_result"] = trial
    session["active_trial_id"] = trial["trial_id"]
    session_path.write_text(json.dumps(session), encoding="utf-8")
    return {"dir": run_dir, "trial": trial, "read_scan": read_scan}


def _hashes(paths) -> dict:
    return {str(p): hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths}


def _evaluation_files(result: dict) -> list[Path]:
    json_path = Path(result["evaluation_path"])
    return [json_path, *sorted(json_path.with_name(json_path.stem + "_tiles").iterdir())]


def test_duplex_faces_match_page_codes_in_reverse_file_order(run, tmp_path):
    scans = fake_scan(run["trial"]["pdf_path"], tmp_path / "faces")
    originals = _hashes(scans)
    result = run["read_scan"].invoke({"scan_paths": list(reversed(scans))})
    assert result["status"] == "ok", result
    assert [page["page_number"] for page in result["pages"]] == [1, 2]
    assert [page["scan_index"] for page in result["pages"]] == [1, 0]
    assert _hashes(scans) == originals


def test_milestone_fake_scan_then_read_scan(run, tmp_path):
    scans = fake_scan(run["trial"]["pdf_path"], tmp_path / "scan")
    assert [Path(p).name for p in scans] == ["scan_page_1.png", "scan_page_2.png"]
    assert Image.open(scans[0]).info["dpi"] == pytest.approx((300, 300), abs=0.01)   # PNG stores pixels per metre
    result = run["read_scan"].invoke({"scan_paths": scans})
    assert result["status"] == "ok", result
    assert result["trial_id"] == run["trial"]["trial_id"] and result["round_id"] == 1
    assert len(result["candidates"]) == 24
    assert {"paper", "black", "gray_patches"} <= set(result["printer_response"])
    assert "contrast" in result["measurement_docs"]
    assert [f["path"] for f in result["scan_files"]] == [str(Path(p).resolve()) for p in scans]
    assert all(len(f["sha256"]) == 64 for f in result["scan_files"])
    evaluation = Path(result["evaluation_path"])
    assert evaluation.is_file() and evaluation.parent == run["dir"].resolve() / "print_tuning"
    for candidate in result["candidates"]:
        assert Path(candidate["tile_path"]).is_file()
        assert all(Path(panel["png"]).is_file() for panel in candidate["panels"])


@pytest.mark.parametrize("options", [{"rotate_deg": 1.0, "noise": 0.02}, {"darken": 0.2}])
def test_imperfect_scans_are_still_read(run, tmp_path, options):
    scans = fake_scan(run["trial"]["pdf_path"], tmp_path / "scan", seed=3, **options)
    result = run["read_scan"].invoke({"scan_paths": scans})
    assert result["status"] == "ok", result
    assert len(result["candidates"]) == 24


def test_bad_scans_need_a_rescan(run, tmp_path):
    pdf = run["trial"]["pdf_path"]
    good = fake_scan(pdf, tmp_path / "good")
    low = fake_scan(pdf, tmp_path / "low", dpi=72)
    blank = tmp_path / "blank.png"
    Image.new("L", Image.open(good[0]).size, 255).save(blank, dpi=(300, 300))
    for scans, code in (([good[0]], "page_missing"), (low, "low_dpi"), ([str(blank), good[1]], "markers_not_found")):
        result = run["read_scan"].invoke({"scan_paths": scans})
        assert result["status"] == "needs_rescan", result
        assert result["error_code"] == code, result
        assert result["scan_quality"]["usable"] is False and result["scan_quality"]["issues"]


def test_rescan_keeps_the_first_evaluation(run, tmp_path):
    first = run["read_scan"].invoke({"scan_paths": fake_scan(run["trial"]["pdf_path"], tmp_path / "first")})
    files = _evaluation_files(first)
    before = _hashes(files)
    second = run["read_scan"].invoke({"scan_paths": fake_scan(run["trial"]["pdf_path"], tmp_path / "second",
                                                              noise=0.01, seed=7)})
    assert second["status"] == "ok"
    assert second["evaluation_path"] != first["evaluation_path"]
    assert _hashes(files) == before
    assert not set(map(str, _evaluation_files(second))) & set(map(str, files))


def test_bilevel_scan_preserves_pixels_and_reports_tonal_evidence_limit(run, tmp_path):
    scans = fake_scan(run["trial"]["pdf_path"], tmp_path / "bilevel")
    for path in scans:
        with Image.open(path) as source:
            binary = source.convert("1")
        binary.save(path, dpi=(300, 300))
    before = _hashes(scans)
    result = run["read_scan"].invoke({"scan_paths": scans})
    assert result["status"] == "ok", result
    assert all(any("bilevel (1-bit)" in issue for issue in page["issues"]) for page in result["pages"])
    assert all("bilevel_scan_indices" in file for file in result["scan_files"])
    assert _hashes(scans) == before


def test_fake_scan_refuses_overwrite_and_bad_arguments(run, tmp_path):
    pdf = run["trial"]["pdf_path"]
    fake_scan(pdf, tmp_path)
    with pytest.raises(ValueError, match="never overwrites"):
        fake_scan(pdf, tmp_path)
    for bad in ({"dpi": 0}, {"dpi": -300}, {"noise": -0.1}, {"darken": 1.0}, {"darken": -0.1},
                {"rotate_deg": 12.0}, {"rotate_deg": "1"}, {"rotate_deg": True}, {"noise": "0.1"},
                {"noise": float("nan")}, {"darken": None}, {"darken": float("inf")}):
        with pytest.raises(ValueError):
            fake_scan(pdf, tmp_path / "bad", **bad)
    with pytest.raises(ValueError, match="cannot open"):
        fake_scan(tmp_path / "missing.pdf", tmp_path / "missing")


def test_summary_of_an_ok_scan_is_short(run, tmp_path):
    result = run["read_scan"].invoke({"scan_paths": fake_scan(run["trial"]["pdf_path"], tmp_path / "scan")})
    summary = summarize_scan_result(result)
    text = json.dumps(summary)
    print(f"summary size: {len(text.encode())} bytes (full result {len(json.dumps(result).encode())} bytes)")
    assert summary["status"] == "ok" and summary["trial_id"] == result["trial_id"]
    assert len(text.encode()) < 4096
    columns = summary["candidate_columns"]
    rows = [(e["experiment_id"], dict(zip(columns, row))) for e in summary["experiments"] for row in e["candidates"]]
    named = [(experiment, row["candidate_id"]) for experiment, row in rows]
    assert sorted(named) == sorted((c["experiment_id"], c["candidate_id"]) for c in result["candidates"])
    assert len(set(named)) == 24
    assert all(row["mapping"].startswith("γ ") for _, row in rows)
    assert "measurement_docs" not in text and "printer_response" not in text
    assert ".png" not in text and "tile_path" not in text and "panels" not in text
    assert [p["page_number"] for p in summary["pages"]] == [1, 2]


def test_summary_of_a_rescan_and_of_broken_input(run, tmp_path):
    scans = fake_scan(run["trial"]["pdf_path"], tmp_path / "scan")
    summary = summarize_scan_result(run["read_scan"].invoke({"scan_paths": scans[:1]}))
    assert summary["status"] == "needs_rescan" and summary["error_code"] == "page_missing"
    assert set(summary) == {"contract_version", "status", "error_code", "message", "session_id", "trial_id",
                            "round_id"}
    for broken in ({}, "not a result", {"status": "ok"}, None):
        summary = summarize_scan_result(broken)
        assert summary["status"] == "error" and summary["error_code"] == "summary_failed", summary
