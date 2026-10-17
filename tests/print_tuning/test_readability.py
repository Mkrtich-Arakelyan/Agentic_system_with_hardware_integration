"""Deterministic pair evidence and accepted-record selection, entirely offline."""

from copy import deepcopy
import hashlib
import json

import numpy as np
import pytest

from app.print_tuning.readability import (METHOD, accepted_capture, candidate_evidence, compare_pairs,
    measure_session, cell_statistics, comparability_reason)
from tests.print_tuning.test_final_figures import completed


def stats(means, noise=.02, floor=1 / 255):
    values = np.asarray(means, float)
    return {"means": values, "stds": np.full(values.shape, noise),
            "valid": np.ones(values.shape, bool), "noise_floor": floor}


MAPPING = {"gamma": 1., "dark_level": 0., "light_level": 1., "steps": 0, "invert": False}
SOURCE = np.array([[.1, .3, .6], [.2, .5, .9]])
LIMITS = {"lo": 0., "hi": 1.}


def compare(a, b, **kwargs):
    return compare_pairs(SOURCE, LIMITS, a, b, [MAPPING, MAPPING], **kwargs)


def test_identical_scans_have_identical_scores_and_no_mutation():
    a = stats(SOURCE)
    before = SOURCE.copy()
    result = compare(a, a)
    assert result["before"] == result["after"]
    assert result["detail_change_pp"] == 0 and result["cnr_change"] == 0
    assert result["coverage"] == 1 and result["eligible_pairs"] == 7
    np.testing.assert_array_equal(SOURCE, before)


def test_separation_increases_cnr_and_noise_decreases_it():
    a, b, noisy = stats(SOURCE * .25), stats(SOURCE * .7), stats(SOURCE * .7, .5)
    result = compare(a, b)
    assert result["cnr_change"] > 0
    assert compare(b, noisy)["cnr_change"] < 0


def test_quantization_includes_merged_pairs_in_median_and_denominator():
    result = compare(stats(SOURCE), stats(np.zeros_like(SOURCE)))
    assert result["after"]["detail_percent"] == 0
    assert result["after"]["median_cnr"] == 0
    assert len(result["after"]["pair_cnr"]) == result["jointly_valid_pairs"]
    assert result["detail_change_pp"] < 0


def test_wrong_order_and_inversion():
    result = compare(stats(SOURCE), stats(1 - SOURCE))
    assert result["after"]["detail_percent"] == 0 and result["after"]["wrong_order_fraction"] == 1
    result = compare_pairs(SOURCE, LIMITS, stats(SOURCE), stats(1 - SOURCE),
                           [MAPPING, MAPPING | {"invert": True}])
    assert result["after"]["detail_percent"] == 100


def test_zero_variance_is_finite_and_zero_baseline_has_no_relative_gain():
    result = compare(stats(SOURCE * 0, 0), stats(SOURCE, 0))
    assert np.isfinite(result["after"]["median_cnr"])
    assert result["cnr_relative_change_percent"] is None
    assert result["noise_floor"] >= 1 / 255


def test_noise_is_measured_within_cells_not_between_different_cells():
    image = np.empty((100, 200))
    image[:, :100], image[:, 100:] = .2, .8
    result = cell_statistics({"image": image, "black": 0, "span": 1, "noise_floor": 1 / 255},
                            {"data": np.array([[.2, .8]]), "plot": {"box": [0, 0, 20, 10]}, "marks": []})
    np.testing.assert_allclose(result["means"], [[.2, .8]])
    np.testing.assert_allclose(result["stds"], 0, atol=1e-15)


def test_material_processing_mismatch_is_flagged_and_unknown_is_not_confirmed():
    a = {"bilevel": False, "curve": np.array([0., .5, 1.])}
    assert comparability_reason([a, a], "unknown") is None
    assert comparability_reason([a, a | {"bilevel": True}], "unknown")
    assert comparability_reason([a, a | {"curve": np.array([0., .1, 1.])}], "unknown")
    assert comparability_reason([a, a], "different")


def test_no_pairs_insufficient_coverage_and_clipping_are_safe():
    result = compare_pairs(np.ones((2, 3)) * .5, LIMITS, stats(SOURCE), stats(SOURCE), [MAPPING] * 2)
    assert result["before"] is None and result["reason"] == "No eligible source pairs"
    bad = stats(SOURCE)
    bad["valid"][:] = False
    result = compare(stats(SOURCE), bad)
    assert result["before"] is None and result["coverage"] == 0
    assert "coverage" in result["reason"]
    source = SOURCE.copy()
    source[0, 0] = -1
    result = compare_pairs(source, LIMITS, stats(SOURCE), stats(SOURCE), [MAPPING] * 2)
    assert result["clipped_source_cells"] == 1
    with pytest.raises(ValueError):
        compare(stats(SOURCE), stats(SOURCE), method={"source_difference": 0})


def test_recorded_reference_and_final_winner_including_reference(completed):
    path, session = completed
    before, after = accepted_capture(session, 1), accepted_capture(session, 3)
    for experiment in session["selected_experiments"]:
        a = candidate_evidence(session, before, experiment, 1)
        b = candidate_evidence(session, after, experiment, 3)
        assert a["candidate_id"] == "reference"
        assert a["mapping"] == session["rounds"]["1"]["reference_settings"][experiment["experiment_id"]]
        assert b["mapping"] == session["final_settings"][experiment["experiment_id"]]
    assert candidate_evidence(session, after, session["selected_experiments"][0], 3)["candidate_id"] == "reference"
    assert candidate_evidence(session, after, session["selected_experiments"][1], 3)["candidate_id"] == "test_2"


def test_rejected_superseded_and_changed_evidence_is_not_selected(completed):
    _, session = completed
    copy = deepcopy(session)
    copy["rounds"]["1"]["evaluations"].append({"record_id": "rejected-newer", "result": {"status": "needs_rescan"}})
    assert accepted_capture(copy, 1)["evaluation"]["trial_id"] == session["rounds"]["1"]["trial_result"]["trial_id"]
    copy["rounds"]["1"]["trial_result"]["trial_id"] = "superseded"
    with pytest.raises(ValueError, match="superseded"):
        accepted_capture(copy, 1)
    capture = accepted_capture(session, 3)
    capture["manifest"]["variations"][session["selected_experiments"][1]["item_id"]][2]["mapping"]["gamma"] = 8
    with pytest.raises(ValueError):
        candidate_evidence(session, capture, session["selected_experiments"][1], 3)


def test_missing_scan_and_legacy_evidence_return_specific_null_reason(completed):
    _, session = completed
    legacy = deepcopy(session)
    legacy["rounds"]["1"]["reviews"] = []
    result = measure_session(legacy)
    assert all(r["before"] is None and "linkage" in r["reason"] for r in result["experiments"])
    capture = accepted_capture(session, 1)
    scan = capture["evaluation"]["scan_files"][0]
    path = __import__("pathlib").Path(scan["path"])
    original = path.read_bytes()
    try:
        path.unlink()
        result = measure_session(session)
        assert result["experiments"][0]["before"] is None
        assert result["experiments"][0]["reason"]
    finally:
        path.write_bytes(original)


def test_reprocesses_original_scans_with_provenance_and_preserves_all_inputs(completed):
    path, session = completed
    from pathlib import Path
    from app.workflow.report_document import digest, safe_json
    before = {str(p): digest(p) for p in path.parent.rglob("*") if p.is_file()}
    result = measure_session(session)
    assert result["acquisition_consistency"] == "unknown"
    assert all(r.get("before") and r.get("after") and r["coverage"] >= METHOD["minimum_coverage"] for r in result["experiments"]), [r.get("reason") for r in result["experiments"]]
    assert "synthetic" in str(result["captures"]["before"]["scan_provenance"])
    assert result["experiments"][0]["after_evidence"]["candidate_id"] == "reference"
    assert result["experiments"][1]["after_evidence"]["candidate_id"] == "test_2"
    json.dumps(safe_json(result), allow_nan=False)
    assert before == {str(p): digest(p) for p in path.parent.rglob("*") if p.is_file()}


def test_missing_legacy_final_candidate_is_null_instead_of_crashing(monkeypatch):
    from app.print_tuning import readability
    session = {"selected_experiments": [{"experiment_id": "E1"}]}
    monkeypatch.setattr(readability, "accepted_capture", lambda *args: {"provenance": {"acquisition_settings": None}})
    monkeypatch.setattr(readability, "aligned_capture", lambda capture: {})
    def missing(*args):
        raise StopIteration
    monkeypatch.setattr(readability, "candidate_evidence", missing)
    result = measure_session(session)
    assert all(r["before"] is None and "Final panel" in r["reason"] for r in result["experiments"])


def test_unusable_calibration_returns_null_from_real_patch_pixels(monkeypatch):
    from app.print_tuning import readability
    regions = {"paper_refs": [{"box": [0, 0, 5, 5]}], "black_patch": {"box": [5, 0, 5, 5]},
               "gray_patches": [{"box": [0, 5, 5, 5], "level": .5}]}
    manifest_page = {"page_number": 1, "regions": regions}
    capture = {"provenance": {"acquisition_settings": None}, "evaluation": {"scan_files": []},
               "manifest": {"pages": [manifest_page], "config": {"scan": {"accepted_formats": ["png"], "min_dpi": 200}}}}
    gray = np.full((100, 100), .5)
    rgb = np.full((100, 100, 3), 128, np.uint8)
    monkeypatch.setattr(readability, "accepted_capture", lambda *args: capture)
    monkeypatch.setattr(readability.scan, "_read_scan", lambda *args: ([(gray, rgb, 300)], []))
    monkeypatch.setattr(readability.scan, "_align_page", lambda *args: {"image": gray, "rgb": rgb, "manifest_page": manifest_page})
    result = measure_session({"selected_experiments": [{"experiment_id": "E1"}]})
    assert result["experiments"][0]["before"] is None
    assert "calibration range" in result["experiments"][0]["reason"]
