"""Real session/trial producers, scan_eval/2-shaped metadata and tiny PNG crops."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import os
import tempfile
import unittest
from unittest.mock import Mock

from PIL import Image

from app.agents.photo_quality_specialist import (
    ImageLabel, photo_quality_specialist, prepare_specialist_messages,
)
from app.print_tuning.schemas import GammaProposalResponse, GammaReviewResponse, GrayLevelsReviewResponse
from app.print_tuning.tuning_tools import _make_trial_sheet, create_tuning_tools, summarize_scan_result
from app.print_tuning.fake_scan import fake_scan
from app.print_tuning.validation import get_candidate_correspondence
from app.workflow import prepare_tuning_inputs
from app.workflow.tuning import (
    accept_specialist_response, build_tuning_context, load_tuning_session,
    record_scan_result, record_trial_result,
    record_scan_for_executor, TuningSessionError,
)
from app.workflow.tuning_evidence import TuningEvidenceError
from tests.workflow import test_tuning as fixtures


class EvidenceTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.SessionTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.root, self.path = self.fixture.root, self.fixture.path
        self.crop = self.root / "crop.png"
        Image.new("RGB", (3, 2), (100, 90, 80)).save(self.crop)
        self.fixture.init(paper_color="yellow")

    def trial(self, round_id, *, internal_ids=False):
        result = _make_trial_sheet(self.root, round_id)
        if internal_ids:
            path = Path(result["manifest_path"])
            manifest = json.loads(path.read_text())
            rename = {"reference": "ref", **{f"test_{i}": f"v-{i * 7}" for i in range(1, 6)}}
            def translate(value):
                if isinstance(value, dict):
                    return {k: rename[v] if k == "variation_id" else translate(v) for k, v in value.items()}
                return [translate(v) for v in value] if isinstance(value, list) else value
            path.write_text(json.dumps(translate(manifest)))
        record_trial_result(self.path, result)
        return result

    def evaluation(self, trial):
        manifest = json.loads(Path(trial["manifest_path"]).read_text())
        pages = []
        for page in manifest["pages"]:
            pages.append(dict(page_number=page["page_number"], scan_index=page["page_number"] - 1,
                              page_code_read=page["page_code"], page_code_expected=page["page_code"], code_matches=True,
                              aligned=True, fit_error_mm=.2, dpi=300, orientation=0,
                              paper={"gray": .72, "rgb": [.8, .75, .6]},
                              black={"gray": .12, "rgb": [.14, .12, .1]}, paper_to_black_contrast=.6,
                              gray_patches=[{"level": 0., "measured": .12}, {"level": 1., "measured": .72}],
                              issues=[], notes={}))
        candidates = []
        for item in manifest["items"]:
            for index, variation in enumerate(manifest["variations"][item["item_id"]]):
                plots = [(p["page_number"], v) for p in manifest["pages"] for v in p["regions"]["plots"]
                         if v["item_id"] == item["item_id"] and v["variation_id"] == variation["variation_id"]]
                candidates.append(dict(experiment_id=item["item_id"], candidate_id=variation["variation_id"],
                                       role=variation["role"], mapping=variation["mapping"], page_number=plots[0][0],
                                       tile_path=str(self.crop), panels=[dict(panel=v["panel"], png=str(self.crop),
                                           measurements={"contrast": .2 + index / 100, "noise": None},
                                           notes={"noise": "Not available"}) for _, v in plots],
                                       measurements={"contrast": .3 + index / 100, "correlation": None,
                                                     "collapsed_levels": 0, "full_black_share": .00754321,
                                                     "paper_white_share": .00123456},
                                       warnings=["Low local contrast"] if index == 0 else [],
                                       notes={"correlation": "Insufficient distinct levels"}))
        return dict(contract_version=1, schema_version="scan_eval/2", evaluation_id="eval",
                    session_id=trial["session_id"], trial_id=trial["trial_id"], round_id=trial["round_id"],
                    tested_parameter=manifest["tested_parameter"], manifest_path=trial["manifest_path"],
                    scan_files=[], scan_quality={"usable": True, "issues": [{"page_number": 1, "issue": "Lighting warning"}]},
                    pages=pages, candidates=candidates,
                    printer_response={"paper": pages[0]["paper"], "black": pages[0]["black"],
                                      "gray_patches": pages[0]["gray_patches"], "notes": {}},
                    measurement_docs={"contrast": {"units": "gray", "meaning": "p95 minus p5"}})

    def record(self, trial, evaluation, name="active", *, record_result=True):
        path = self.root / f"{name}.json"
        evaluation.update(evaluation_path=str(path), result_path=str(path))
        path.write_text(json.dumps(evaluation, allow_nan=False))
        result = {k: trial[k] for k in ("contract_version", "session_id", "trial_id", "round_id")}
        result.update(status="ok", evaluation_path=str(path), scan_quality=evaluation["scan_quality"],
                      pages=evaluation["pages"], candidates=evaluation["candidates"])
        if record_result:
            record_scan_result(self.path, result, record_id=name)
        return path

    def round_one(self, *, internal_ids=False):
        self.fixture.propose()
        trial = self.trial(1, internal_ids=internal_ids)
        evaluation = self.evaluation(trial)
        self.record(trial, evaluation)
        return trial, evaluation

    def messages(self, stage):
        before = self.path.read_bytes()
        inputs = prepare_tuning_inputs(self.path, stage)
        messages = prepare_specialist_messages(inputs.context, inputs.evidence,
                                               prior_decisions=inputs.prior_decisions, paper_color=inputs.paper_color)
        self.assertEqual(before, self.path.read_bytes())
        self.assertTrue(any(p["type"] == "image_url" for p in messages[1].content))
        return inputs, messages

    def test_initial_inputs_have_all_eight_previews_and_no_physical_evidence(self):
        inputs, messages = self.messages("propose_gamma")
        self.assertEqual({a.label for a in inputs.evidence.images},
                         {ImageLabel("source_preview", e, panel) for e in fixtures.IDS for panel in ("Initial", "Final")})
        self.assertEqual(len(inputs.evidence.images), 8)
        self.assertEqual(inputs.evidence.prior_settings, inputs.context.reference_mappings)
        self.assertFalse(inputs.evidence.candidate_measurements)
        self.assertIsNone(inputs.evidence.printer_response)
        self.assertEqual(inputs.prior_decisions, ())
        self.assertEqual(inputs.paper_color, "yellow")
        self.assertEqual(sum(p["type"] == "image_url" for p in messages[1].content), 8)
        model = Mock()
        model.profile = {"image_inputs": True}
        decision = fixtures.response("propose_gamma")
        model.with_structured_output.return_value.invoke.return_value = {"parsed": decision, "raw": None}
        self.assertIs(photo_quality_specialist(model, inputs.context, inputs.evidence,
                      prior_decisions=inputs.prior_decisions, paper_color=inputs.paper_color), decision)
        model.bind_tools.assert_not_called()

    def test_all_review_stages_keep_prior_references_and_accepted_history(self):
        self.fixture.propose()
        for number, stage in enumerate(("review_gamma", "review_gray_levels", "review_steps"), 1):
            with self.subTest(stage=stage):
                trial = self.trial(number)
                self.record(trial, self.evaluation(trial), f"scan-{number}")
                inputs, _ = self.messages(stage)
                self.assertEqual(len(inputs.evidence.candidate_measurements), 72)
                self.assertEqual(len(inputs.evidence.images), 80)
                session = load_tuning_session(self.path)
                self.assertEqual(inputs.evidence.prior_settings, session["rounds"][str(number)]["reference_settings"])
                self.assertEqual(inputs.evidence.prior_settings, session["accepted_settings"])
                self.assertEqual([type(r) for r in inputs.prior_decisions],
                                 [GammaProposalResponse, GammaReviewResponse, GrayLevelsReviewResponse][:number])
                self.assertEqual([r.stage for r in inputs.prior_decisions],
                                 ["propose_gamma", "review_gamma", "review_gray_levels"][:number])
                accepted = accept_specialist_response(self.path, fixtures.response(stage), inputs.context,
                                                      evaluation_record_id=inputs.evaluation_record_id)
                if number < 3:
                    self.assertNotEqual(inputs.evidence.prior_settings, accepted["accepted_settings"])

    def test_translation_keeps_item_variation_panel_page_and_measurements_aligned(self):
        _, evaluation = self.round_one(internal_ids=True)
        inputs, _ = self.messages("review_gamma")
        identities = get_candidate_correspondence(inputs.context)
        reference = identities[0]
        self.assertEqual(reference.experiment_id, "exp/A")
        self.assertEqual(reference.item_id, "exp_A")
        self.assertEqual(reference.variation_id, "ref")
        self.assertEqual(identities[1].variation_id, "v-7")
        measurements = inputs.evidence.candidate_measurements
        self.assertEqual(measurements[0].values["contrast"], .3)
        self.assertIsNone(measurements[0].values["correlation"])
        self.assertIn("correlation", measurements[0].values["notes"])
        self.assertEqual(measurements[1].panel, "Initial")
        self.assertEqual(measurements[1].values["contrast"], .2)
        self.assertEqual(measurements[1].values["page_number"], 1)
        first = next(a.label for a in inputs.evidence.images if a.label.purpose == "scanned_candidate" and a.label.panel == "Initial")
        self.assertEqual(first, ImageLabel("scanned_candidate", "exp/A", "Initial", "reference", "ref", 1))
        self.assertEqual(inputs.evidence.printer_response, evaluation["printer_response"])
        self.assertEqual(inputs.evidence.page_measurements[0]["gray_patches"], evaluation["pages"][0]["gray_patches"])
        self.assertEqual(inputs.evidence.page_measurements[0]["paper"], evaluation["pages"][0]["paper"])
        self.assertTrue(any('"page_number": 1' in issue and "Lighting warning" in issue for issue in inputs.evidence.issues))

    def test_full_scan_metadata_and_tile_images_reach_specialist_unchanged(self):
        self.fixture.propose()
        trial = self.trial(1)
        evaluation = self.evaluation(trial)
        evaluation["scan_files"] = [{"path": str(self.root / "original-scan.png"), "sha256": "a" * 64}]
        self.record(trial, evaluation)
        inputs, messages = self.messages("review_gamma")
        metadata = json.loads(messages[1].content[0]["text"])
        self.assertEqual(metadata["measurement_docs"], evaluation["measurement_docs"])
        self.assertEqual(metadata["scan_files"], evaluation["scan_files"])
        self.assertEqual(metadata["scan_quality"], evaluation["scan_quality"])
        self.assertEqual(metadata["printer_response"], evaluation["printer_response"])
        tiles = [a for a in inputs.evidence.images if a.label.purpose == "scanned_candidate" and a.label.panel is None]
        self.assertEqual(len(tiles), 24)
        self.assertEqual(tiles[0].label, ImageLabel("scanned_candidate", "exp/A", None, "reference", "reference", 1))
        self.assertEqual(inputs.evidence.candidate_measurements[0].values["full_black_share"], .00754321)

    def test_record_full_result_and_return_only_executor_summary(self):
        self.fixture.propose()
        trial = self.trial(1)
        evaluation = self.evaluation(trial)
        evaluation_path = self.record(trial, evaluation, record_result=False)
        # Start from the full tool result, not the reduced session record.
        result = {k: trial[k] for k in ("contract_version", "session_id", "trial_id", "round_id")}
        result.update(status="ok", evaluation_path=str(evaluation_path), **{k: evaluation[k]
                      for k in ("pages", "candidates", "scan_quality", "printer_response", "measurement_docs", "scan_files")})
        original = deepcopy(result)
        before = self.path.read_bytes()
        summary = record_scan_for_executor(self.path, result, record_id="same-full-result")
        self.assertEqual(summary, summarize_scan_result(result))
        self.assertEqual(result, original)
        self.assertEqual(summary["experiments"][0]["candidates"][0][-2], .008)
        self.assertFalse({"candidates", "measurement_docs", "printer_response", "scan_files"} & summary.keys())
        self.assertNotIn(".png", json.dumps(summary))
        self.assertNotEqual(before, self.path.read_bytes())
        accepted = self.path.read_bytes()
        self.assertEqual(record_scan_for_executor(self.path, result, record_id="same-full-result"), summary)
        self.assertEqual(accepted, self.path.read_bytes())
        with self.assertRaisesRegex(TuningSessionError, "full read_scan"):
            record_scan_for_executor(self.path, summary)
        changed = deepcopy(result)
        changed["candidates"][0]["measurements"]["contrast"] = .9
        with self.assertRaisesRegex(TuningSessionError, "differs from saved"):
            record_scan_for_executor(self.path, changed, record_id="bad")
        self.assertEqual(accepted, self.path.read_bytes())
        inputs, _ = self.messages("review_gamma")
        self.assertEqual(inputs.evidence.candidate_measurements[0].values["full_black_share"], .00754321)

    def test_handoff_records_full_rescan_and_handles_errors_without_guessing_identity(self):
        self.fixture.propose()
        trial = self.trial(1)
        result = {k: trial[k] for k in ("contract_version", "session_id", "trial_id", "round_id")}
        result.update(status="needs_rescan", error_code="page_missing", message="Page 2 missing",
                      scan_quality={"usable": False, "issues": ["Page 2 missing"]})
        short = record_scan_for_executor(self.path, result, record_id="rescan")
        self.assertEqual(short, summarize_scan_result(result))
        self.assertNotIn("scan_quality", short)
        inputs, _ = self.messages("review_gamma")
        self.assertEqual(inputs.evidence.scan_quality, result["scan_quality"])
        self.assertIsNone(inputs.evidence.measurement_docs)
        self.assertEqual(inputs.evidence.scan_files, ())
        error = {k: result[k] for k in ("contract_version", "session_id", "trial_id", "round_id")}
        Path(trial["manifest_path"]).unlink()
        error.update(status="error", error_code="manifest_missing", message="Manifest file missing")
        self.assertEqual(record_scan_for_executor(self.path, error, record_id="error"), error)
        with self.assertRaises(TuningEvidenceError):
            prepare_tuning_inputs(self.path, "review_gamma")
        before = self.path.read_bytes()
        unbound = dict(contract_version=1, status="error", error_code="session_invalid", message="No session")
        self.assertEqual(record_scan_for_executor(self.path, unbound), unbound)
        self.assertEqual(before, self.path.read_bytes())

    def test_real_updated_tools_feed_summary_and_full_specialist_inputs(self):
        self.fixture.propose()
        trial = self.trial(1)
        scan_paths = fake_scan(trial["pdf_path"], self.root / "fake-scan", dpi=300)
        before_tool = self.path.read_bytes()
        full = create_tuning_tools(self.root)[1].invoke({"scan_paths": scan_paths})
        self.assertEqual(full["status"], "ok", full)
        self.assertEqual(before_tool, self.path.read_bytes())
        self.assertTrue({"measurement_docs", "printer_response", "scan_files"} <= full.keys())
        summary = record_scan_for_executor(self.path, full, record_id="fake-1")
        self.assertEqual(summary, summarize_scan_result(full))
        inputs, messages = self.messages("review_gamma")
        metadata = json.loads(messages[1].content[0]["text"])
        self.assertEqual(metadata["measurement_docs"], full["measurement_docs"])
        self.assertEqual(metadata["scan_files"], full["scan_files"])
        self.assertEqual(inputs.evidence.printer_response, full["printer_response"])
        self.assertEqual(inputs.evidence.candidate_measurements[0].values["contrast"], full["candidates"][0]["measurements"]["contrast"])
        self.assertEqual(sum(a.label.purpose == "scanned_candidate" for a in inputs.evidence.images), 72)
        self.assertNotIn("measurement_docs", summary)
        self.assertNotIn(".png", json.dumps(summary))

    def test_missing_crops_panels_and_candidates_are_explicit(self):
        self.fixture.propose()
        trial = self.trial(1)
        evaluation = self.evaluation(trial)
        evaluation["candidates"][0]["panels"][0]["png"] = str(self.root / "absent.png")
        evaluation["candidates"][1]["panels"].pop()
        evaluation["candidates"].pop()
        self.record(trial, evaluation)
        inputs, _ = self.messages("review_gamma")
        self.assertEqual(len(inputs.evidence.missing_images), 4)
        self.assertTrue(all(m.reason for m in inputs.evidence.missing_images))
        self.assertTrue(inputs.evidence.incomplete_evidence_reason)

    def test_rescan_without_evaluation_has_issues_and_all_missing_crops(self):
        self.fixture.propose()
        trial = self.trial(1)
        self.fixture.scan(trial, status="needs_rescan", usable=False, name="rescan")
        inputs, _ = self.messages("review_gamma")
        self.assertEqual(inputs.evaluation_record_id, "rescan")
        self.assertEqual(len(inputs.evidence.missing_images), 48)
        self.assertEqual(inputs.evidence.candidate_measurements, ())
        self.assertIsNone(inputs.evidence.printer_response)
        self.assertTrue(any("Blurred" in v for v in inputs.evidence.issues))

    def test_uses_active_evaluation_not_filesystem_recency(self):
        trial, evaluation = self.round_one()
        later = deepcopy(evaluation)
        later["candidates"][0]["measurements"]["contrast"] = .456
        self.record(trial, later, "replacement")
        (self.root / "newest-unrecorded.json").write_text("unrelated")
        inputs, _ = self.messages("review_gamma")
        self.assertEqual(inputs.evaluation_record_id, "replacement")
        self.assertEqual(inputs.evidence.candidate_measurements[0].values["contrast"], .456)

    def test_operational_scan_error_is_not_visual_evidence(self):
        self.fixture.propose()
        trial = self.trial(1)
        result = {k: trial[k] for k in ("contract_version", "session_id", "trial_id", "round_id")}
        result.update(status="error", error_code="cells_missing", message="Trial cells artifact is missing")
        record_scan_result(self.path, result, record_id="failure")
        before = self.path.read_bytes()
        with self.assertRaisesRegex(TuningEvidenceError, "Operational scan error"):
            prepare_tuning_inputs(self.path, "review_gamma")
        self.assertEqual(before, self.path.read_bytes())

    def test_current_round_rescan_decision_is_not_prior_history(self):
        self.round_one()
        self.fixture.review("review_gamma", name="active", assessment="needs_rescan")
        inputs, _ = self.messages("review_gamma")
        self.assertEqual([r.stage for r in inputs.prior_decisions], ["propose_gamma"])
        self.assertEqual(inputs.evaluation_record_id, "active")

    def test_modified_or_missing_artifacts_fail_without_session_writes(self):
        trial, evaluation = self.round_one()
        before = self.path.read_bytes()
        for path in (Path(trial["manifest_path"]), Path(evaluation["evaluation_path"])):
            original = path.read_bytes()
            path.write_bytes(original + b" ")
            with self.assertRaisesRegex(TuningEvidenceError, "changed"):
                prepare_tuning_inputs(self.path, "review_gamma")
            path.write_bytes(original)
        Path(evaluation["evaluation_path"]).unlink()
        with self.assertRaises(TuningEvidenceError):
            prepare_tuning_inputs(self.path, "review_gamma")
        self.assertEqual(before, self.path.read_bytes())

    def test_missing_or_changed_initial_preview_fails(self):
        session = load_tuning_session(self.path)
        path = Path(session["previews"]["previews"][0]["path"])
        before = self.path.read_bytes()
        original = path.read_bytes()
        path.unlink()
        with self.assertRaisesRegex(TuningEvidenceError, "Required source preview"):
            prepare_tuning_inputs(self.path, "propose_gamma")
        path.write_bytes(original + b" ")
        with self.assertRaisesRegex(TuningEvidenceError, "Source preview changed"):
            prepare_tuning_inputs(self.path, "propose_gamma")
        self.assertEqual(before, self.path.read_bytes())

    def test_wrong_stage_phase_and_unrecorded_evidence_fail(self):
        for stage in ("review_gamma", "unknown"):
            with self.assertRaises(TuningEvidenceError):
                prepare_tuning_inputs(self.path, stage)
        self.fixture.propose()
        with self.assertRaises(TuningEvidenceError):
            prepare_tuning_inputs(self.path, "propose_gamma")
        self.trial(1)
        with self.assertRaisesRegex(TuningEvidenceError, "active evaluation"):
            prepare_tuning_inputs(self.path, "review_gamma")
        with self.assertRaises(TuningEvidenceError):
            prepare_tuning_inputs(self.path, "review_steps")

    def test_unknown_duplicate_or_conflicting_scan_records_rejected(self):
        self.fixture.propose()
        trial = self.trial(1)
        base = self.evaluation(trial)
        def duplicate_candidate(e): e["candidates"].append(deepcopy(e["candidates"][0]))
        def duplicate_panel(e): e["candidates"][0]["panels"].append(deepcopy(e["candidates"][0]["panels"][0]))
        mutations = [duplicate_candidate, duplicate_panel,
                     lambda e: e["candidates"][0].update(experiment_id="exp/A"),
                     lambda e: e["candidates"][0].update(candidate_id="unknown"),
                     lambda e: e["candidates"][0].update(page_number=2),
                     lambda e: e["candidates"][0]["mapping"].update(gamma=4.),
                     lambda e: e["pages"].append(deepcopy(e["pages"][0]))]
        # Each malformed file is recorded separately from a pristine session snapshot.
        snapshot = self.path.read_bytes()
        for index, mutate in enumerate(mutations):
            with self.subTest(index=index):
                self.path.write_bytes(snapshot)
                evaluation = deepcopy(base)
                mutate(evaluation)
                self.record(trial, evaluation, f"bad-{index}")
                before = self.path.read_bytes()
                with self.assertRaises(TuningEvidenceError):
                    prepare_tuning_inputs(self.path, "review_gamma")
                self.assertEqual(before, self.path.read_bytes())

    def test_stale_identity_is_rejected_even_with_updated_file_hash(self):
        _, evaluation = self.round_one()
        path = Path(evaluation["evaluation_path"])
        evaluation["trial_id"] = "stale"
        path.write_text(json.dumps(evaluation))
        session = load_tuning_session(self.path)
        session["rounds"]["1"]["evaluations"][-1]["evaluation_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        self.path.write_text(json.dumps(session))
        with self.assertRaisesRegex(TuningEvidenceError, "stale trial"):
            prepare_tuning_inputs(self.path, "review_gamma")

    def test_relative_panel_paths_do_not_depend_on_cwd(self):
        self.fixture.propose()
        trial = self.trial(1)
        evaluation = self.evaluation(trial)
        for candidate in evaluation["candidates"]:
            for panel in candidate["panels"]:
                panel["png"] = self.crop.name
        self.record(trial, evaluation)
        with tempfile.TemporaryDirectory() as elsewhere:
            original = Path.cwd()
            try:
                os.chdir(elsewhere)
                inputs, _ = self.messages("review_gamma")
                self.assertTrue(all(Path(a.path).is_absolute() for a in inputs.evidence.images))
            finally:
                os.chdir(original)


if __name__ == "__main__":
    unittest.main()
