"""Two-page artifact rendering, exact-page validation and reproducible read-only inputs."""

from copy import deepcopy
import json
from pathlib import Path
from unittest.mock import patch

import pymupdf
import pytest

from app.print_tuning.final_figures import render_report_comparison
from app.workflow.report_document import build_report, dependency_fingerprint, digest
from app.workflow.reporting import inspect_tuned_report, require_report
from tests.print_tuning.test_final_figures import completed


@pytest.fixture
def revision_run(completed, tmp_path, monkeypatch):
    from app.workflow import reporting
    from app.workflow.report_document import create_report_tool
    path, session = completed
    # Revision/layout tests exercise actual PDFs; metric measurement has separate
    # real-scan tests and need not re-align all large scans for every text edit.
    labels = ["Baseline", "C1-S1", "C1-S2", "C1-S3"]
    history = {"labels": labels, "captures": {}, "experiments": [
        {"experiment_id": e["experiment_id"], "label": f"E{i+1}", "coverage": {"reason": "Synthetic unavailable evidence fixture"},
         "points": [{"label": label, "metrics": None, "reason": "Synthetic unavailable evidence fixture"} for label in labels]}
        for i, e in enumerate(session["selected_experiments"])]}
    monkeypatch.setattr("app.workflow.report_document.measure_history", lambda *a, **k: deepcopy(history))
    root = tmp_path / "runs" / "revision"
    root.mkdir(parents=True)
    (root / "tuning_session.json").write_bytes(path.read_bytes())
    figure = render_report_comparison(root / "tuning_session.json", tmp_path / "figure")
    figure["inputs_sha256"] = "revision-fixture"
    journal = reporting.load_record(root)
    journal["figure"] = figure
    reporting.save_record(root, journal)
    tool = create_report_tool(root)
    result = tool.invoke({})
    assert result["ok"], result
    return root, tool, result


def test_revision_text_layout_cache_provenance_and_previous_pdf(revision_run):
    from app.workflow import reporting
    root, tool, initial = revision_run
    revision = reporting.request_revision(root, 'Use title "Measured comparison" and a smaller chart')
    result = tool.invoke({"revision_id": revision["id"], "revision_instructions": revision["instructions"],
                         "findings": {"title": "Measured comparison"}, "presentation": {"chart_height_pt": 190}})
    assert result["ok"], result
    saved = json.loads(Path(result["report_inputs_path"]).read_text(encoding="utf-8"))
    assert saved["revision"] == revision
    assert saved["presentation"]["chart_height_pt"] == 190
    assert saved["pdf_sha256"] != initial["pdf_sha256"]
    assert (root / "report_versions" / (initial["pdf_sha256"] + ".pdf")).is_file()
    with pymupdf.open(root / "report_versions" / (initial["pdf_sha256"] + ".pdf")) as old, \
            pymupdf.open(root / "report.pdf") as revised:
        # The layout request reaches measured placement, not just saved metadata.
        assert revised[0].search_for("Varied setting")[0].y0 == pytest.approx(
            old[0].search_for("Varied setting")[0].y0 - 25, abs=.1)
    assert reporting.load_record(root)["attempts"] == []
    assert reporting.load_record(root)["pending"] is None
    # Reusing the same bundle for a different instruction is a cache miss/error.
    with pytest.raises(ValueError, match="different inputs"):
        build_report(root, Path(result["report_inputs_path"]).parent, findings=saved["findings"],
                     figure=saved["figure"], presentation=saved["presentation"], revision={**revision, "instructions": "other"})


@pytest.mark.parametrize("failure", ["no_edits", "overflow", "same_hash", "same_text"])
def test_failed_or_unchanged_revision_remains_pending(revision_run, failure):
    from app.workflow import reporting
    root, tool, initial = revision_run
    before = (root / "report.pdf").read_bytes()
    revision = reporting.request_revision(root, "Change the report title")
    args = {"revision_id": revision["id"], "revision_instructions": revision["instructions"]}
    if failure == "overflow":
        args["findings"] = {"title": "word " * 2000}
    elif failure in ("same_hash", "same_text"):
        args["findings"] = {"title": "Changed title"}
    if failure in ("same_hash", "same_text"):
        bundle = json.loads(Path(initial["report_inputs_path"]).read_text(encoding="utf-8"))
        bundle["report_path"] = str(root / "report.pdf")
        if failure == "same_text":
            # A different hash alone cannot validate absent edited text.
            bundle["pdf_sha256"] = "different"
        with patch("app.workflow.report_document.build_report", return_value=bundle):
            result = tool.invoke(args)
    else:
        result = tool.invoke(args)
    assert not result["ok"]
    if failure == "same_hash":
        assert "hash is unchanged" in result["error"]
    assert reporting.load_record(root)["revision"] == revision
    assert (root / "report.pdf").read_bytes() == before
    assert reporting.load_record(root)["attempts"] == []


def test_revision_survives_resume_and_exact_instructions_required(revision_run):
    from app.workflow import reporting, runner
    root, tool, _ = revision_run
    revision = reporting.request_revision(root, "  Add a reader question.  ")
    state = {"dialogue": [], "execution_results": [], "report_revision_pending": True, "report_revision": revision}
    (root / "state.json").write_text(json.dumps(state), encoding="utf-8")
    (root / "config.json").write_text(json.dumps({"bench": {}, "request": "original"}), encoding="utf-8")
    original = (root / "report_print.json").read_bytes()
    resumed, _ = runner.load_resume(root, 10)
    assert resumed["report_revision"] == revision and resumed["report_revision_pending"]
    assert not tool.invoke({"findings": {"closing_note": "What do you think?"}})["ok"]
    assert (root / "report_print.json").read_bytes() == original


def test_explicit_cli_revision_for_completed_run_preserves_evidence(revision_run):
    from app.workflow import reporting, runner
    root, _, _ = revision_run
    before = (root / "tuning_session.json").read_bytes()
    state = {"report_revision_pending": False, "pending_print": {"old": True}, "print_outcome": "printed"}
    updated = runner.request_saved_report_revision(root, state, "Add a reader question")
    assert updated["report_revision_pending"] and not updated["pending_print"]
    assert updated["report_revision"]["instructions"] == "Add a reader question"
    assert (root / "tuning_session.json").read_bytes() == before
    assert reporting.load_record(root)["attempts"] == []
    assert not state["report_revision_pending"]


def test_rebuild_invalidates_unused_pending_approval(revision_run):
    from app.workflow import reporting
    root, tool, _ = revision_run
    journal = reporting.load_record(root)
    journal["pending"] = {"old": "approval"}
    reporting.save_record(root, journal)
    result = tool.invoke({"findings": {"title": "Updated experimental comparison"}})
    assert result["ok"], result
    assert reporting.load_record(root)["pending"] is None
    assert reporting.load_record(root)["attempts"] == []


def test_two_page_bundle_legacy_evidence_cache_and_unchanged_source(completed, tmp_path):
    path, session = completed
    # Explicitly test an older completed run with no usable metric linkage.
    original = path.read_bytes()
    legacy = deepcopy(session)
    legacy["rounds"]["1"]["reviews"] = []
    originals = {str(p): digest(p) for p in path.parent.rglob("*") if p.is_file()}
    try:
        path.write_text(json.dumps(legacy), encoding="utf-8")
        before = {str(p): digest(p) for p in path.parent.rglob("*") if p.is_file()}
        result = build_report(path.parent, tmp_path / "bundle")
        assert result["schema_version"] == "report/4" and result["layout"]["pages"] == 3
        assert "linkage" in result["metrics"]["experiments"][0]["reason"]
        require_report(inspect_tuned_report(result["report_path"], result["figure"]))
        with pymupdf.open(result["report_path"]) as doc:
            assert len(doc) == 3 and "Hardware loop results" in doc[2].get_text()
            assert "arrays are unchanged" in " ".join(doc[1].get_text().split())
            image = doc[1].get_image_info()[0]
            assert image["bbox"][2] - image["bbox"][0] >= 180 * 72 / 25.4 - .5
            for page in doc:
                for block in page.get_text("dict")["blocks"]:
                    for line in block.get("lines", []):
                        for span in line["spans"]:
                            assert span["size"] >= 10
                            assert page.rect.contains(pymupdf.Rect(span["bbox"]))
        with patch("app.workflow.report_document.measure_session", side_effect=AssertionError("cache missed")):
            assert build_report(path.parent, tmp_path / "bundle")["pdf_sha256"] == result["pdf_sha256"]
        assert before == {str(p): digest(p) for p in path.parent.rglob("*") if p.is_file()}
        with pytest.raises(ValueError, match="outside"):
            build_report(path.parent, path.parent / "overwrite")
        with pytest.raises(ValueError, match="different inputs"):
            build_report(path.parent, tmp_path / "bundle", method={"source_difference": .1})
    finally:
        path.write_bytes(original)
    assert originals == {str(p): digest(p) for p in path.parent.rglob("*") if p.is_file()}


def test_page_counts_and_comparison_page_are_enforced(completed, tmp_path):
    path, _ = completed
    figure = render_report_comparison(path, tmp_path)
    figure["inputs_sha256"] = "fixture"
    for count, location in ((1, 0), (4, 1), (3, 1), (2, 0), (2, 1)):
        target = tmp_path / f"{count}_{location}.pdf"
        with pymupdf.open() as doc:
            for _ in range(count):
                doc.new_page(width=595.276, height=841.89)
            doc[location].insert_image(pymupdf.Rect(42, 90, 553, 650), filename=figure["figure_path"])
            doc.save(target)
        result = inspect_tuned_report(target, figure)
        assert bool(result.get("figure_embedded") and not result.get("error")) == (count in (2, 3) and location == 1)


def test_four_panel_figure_cannot_be_shrunk_to_fit_and_old_metadata_is_rejected(completed, tmp_path):
    path, _ = completed
    figure = render_report_comparison(path, tmp_path)
    figure["inputs_sha256"] = "fixture"
    target = tmp_path / "shrunk.pdf"
    with pymupdf.open() as doc:
        for _ in range(3):
            doc.new_page(width=595.276, height=841.89)
        doc[1].insert_image(pymupdf.Rect(80, 90, 500, 650), filename=figure["figure_path"])
        doc.save(target)
    assert inspect_tuned_report(target, figure).get("error")
    stale = {**figure, "comparison_version": "scientific-pairs/1"}
    assert "Stale comparison" in inspect_tuned_report(target, stale)["error"]
    with pytest.raises(ValueError, match="Stale comparison"):
        build_report(path.parent, tmp_path / "stale_bundle", figure=stale)


def test_executor_report_tool_installs_valid_report_without_print_or_journal_changes(completed, tmp_path):
    path, session = completed
    from app.workflow import reporting
    from app.workflow.report_document import create_report_tool
    root = tmp_path / "runs" / "copy"
    root.mkdir(parents=True)
    (root / "tuning_session.json").write_bytes(path.read_bytes())
    figure = render_report_comparison(root / "tuning_session.json", tmp_path / "figure")
    figure["inputs_sha256"] = "bound-fixture"
    with pymupdf.open() as document:
        document.new_page()
        document.save(root / "report.pdf")
    old = digest(root / "report.pdf")
    journal = reporting.load_record(root)
    journal["figure"] = figure
    reporting.save_record(root, journal)
    before = (root / "report_print.json").read_bytes()
    unavailable = {"experiments": [{"experiment_id": e["experiment_id"], "before": None, "after": None,
                    "reason": "Accepted scan evidence not available"} for e in session["selected_experiments"]], "captures": {}}
    with patch("app.workflow.report_document.measure_session", return_value=unavailable), \
            patch("app.printing.connector.submit_print", side_effect=AssertionError("Must not print")):
        result = create_report_tool(root).invoke({})
    assert result["ok"] and result["pages"] == 3
    assert (root / "report_versions" / (old + ".pdf")).is_file()
    assert digest(root / "report.pdf") != old
    reporting.require_report(reporting.inspect_tuned_report(root / "report.pdf", figure))
    assert (root / "report_print.json").read_bytes() == before
