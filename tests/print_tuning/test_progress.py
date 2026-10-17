from copy import deepcopy
import numpy as np
import pytest
from app.print_tuning.progress import summarize_cells, render_progress, describe_history
from tests.print_tuning.test_final_figures import completed


def stats(values, valid=None, noise=.01):
    data = np.array(values, dtype=float)
    return {"means": data, "stds": np.full(data.shape, noise), "valid": np.ones(data.shape, bool) if valid is None else valid,
            "noise_floor": 1 / 255}


def test_common_denominator_recomputes_all_points_and_retains_missing():
    data = np.array([[.1, .3, .6], [.2, .5, .8]])
    a = stats(data); b = stats(data * .8); c = stats(data, np.array([[1, 1, 0], [1, 1, 0]], dtype=bool))
    original = deepcopy(a)
    summaries, coverage = summarize_cells(data, {"lo": 0, "hi": 1}, [a, b, None, c], [{}, {}, {}, {}],
                                          np.ones(data.shape, bool), {"minimum_coverage": .3})
    assert summaries[2] is None
    assert len(summaries[0]["pair_cnr"]) == len(summaries[1]["pair_cnr"]) == len(summaries[3]["pair_cnr"])
    assert coverage["jointly_valid_pairs"] < coverage["eligible_pairs"] and coverage["pair_indices"]
    assert np.array_equal(original["valid"], a["valid"])


def test_identical_zero_variance_and_missing_are_finite():
    data = np.array([[.1, .5]])
    zero = stats(data, noise=0)
    summaries, coverage = summarize_cells(data, {"lo": 0, "hi": 1}, [zero, zero, None], [{}, {}, {}], np.ones(data.shape, bool))
    assert summaries[0] == summaries[1] and np.isfinite(summaries[0]["median_cnr"])
    assert coverage["noise_floor"] >= 1 / 255
    values, diagnostics = summarize_cells(data * 0, {"lo": 0, "hi": 1}, [zero], [{}], np.ones(data.shape, bool))
    assert values == [None] and "No eligible" in diagnostics["reason"]


def test_charts_keep_labels_and_missing_breaks(tmp_path):
    labels = ["Baseline", "C1-S1", "C1-S2", "C1-S3", "C2-S1", "C2-S2", "C2-S3"]
    history = {"labels": labels, "experiments": [{"label": f"E{i+1}", "points": [
        {"label": label, "metrics": None if index == 2 else {"detail_percent": 50 + index, "median_cnr": 2 + index / 4}}
        for index, label in enumerate(labels)]} for i in range(4)]}
    figures = render_progress(history, tmp_path)
    assert figures["detail_percent"]["labels"] == labels
    assert figures["median_cnr"]["missing_points"].startswith("line breaks")
    import pymupdf
    with pymupdf.open(figures["median_cnr"]["path"]) as pdf:
        text = pdf[0].get_text()
        assert all(label in text for label in labels) and all(f"E{i}" in text for i in range(1, 5))
    assert "Latest cycle gains" in describe_history(history)


def test_history_uses_accepted_winners_shared_endpoints_and_hash_bound_cache(completed, monkeypatch):
    from app.print_tuning import progress
    path, session = completed
    before = path.read_bytes()
    history = progress.measure_history(session)
    assert history["labels"] == ["Baseline", "C1-S1", "C1-S2", "C1-S3"]
    for experiment, row in zip(session["selected_experiments"], history["experiments"]):
        eid = experiment["experiment_id"]
        assert row["points"][0]["evidence"]["candidate_id"] == "reference"
        for step in (1, 2, 3):
            evidence = row["points"][step]["evidence"]
            assert evidence["mapping"] == session["rounds"][str(step)]["winners"][eid]
            if step == 2:
                assert evidence["candidate_id"] == session["rounds"]["2"]["review"]["endpoint_winner"]["candidate_id"]
        measurable = [p["metrics"] for p in row["points"] if p["metrics"]]
        assert all(len(p["pair_cnr"]) == row["coverage"]["jointly_valid_pairs"] for p in measurable)
    assert "synthetic" in str(history["captures"]).lower()
    monkeypatch.setattr(progress.metric, "aligned_capture", lambda *a: pytest.fail("Hash-bound cell cache should be reused"))
    cached = progress.measure_history({**session, "metric_history": history})
    assert cached == history and path.read_bytes() == before
