"""All three physical rounds tune only gamma: real trial sheets, fake scans, read_scan and the specialist.

No provider calls: the specialist's model is a mock that returns the fixture decisions.
"""

import json
from pathlib import Path
import unittest
from unittest.mock import Mock

from app.agents.photo_quality_specialist import photo_quality_specialist
from app.print_tuning.fake_scan import fake_scan
from app.print_tuning.grayscale_render import changed_parameters
from app.print_tuning.tuning_tools import create_tuning_tools
from app.workflow.tuning import accept_specialist_response, load_tuning_session, record_scan_for_executor, \
    record_trial_result
from app.workflow.tuning_evidence import prepare_tuning_inputs
from tests.workflow import test_tuning as fixtures

FIXED = {"dark_level": 0., "light_level": 1., "steps": 0}
STAGES = {1: "review_gamma", 2: "review_gamma", 3: "review_gamma_final"}


class GammaOnlyRoundsTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.SessionTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.root, self.path = self.fixture.root, self.fixture.path

    def decide(self, stage, round_id=None):
        """Run the real specialist input preparation and acceptance with a mocked model."""
        inputs = prepare_tuning_inputs(self.path, stage)
        decision = fixtures.response(stage, round_id=round_id or 1)
        model = Mock()
        model.profile = {"image_inputs": True}
        model.with_structured_output.return_value.invoke.return_value = {"parsed": decision, "raw": None}
        self.assertIs(photo_quality_specialist(model, inputs.context, inputs.evidence,
                                               prior_decisions=inputs.prior_decisions,
                                               paper_color=inputs.paper_color), decision)
        schema = model.with_structured_output.call_args.args[0]
        self.assertNotIn("dark_level", json.dumps(schema.model_json_schema()))
        self.assertNotIn("steps", json.dumps(schema.model_json_schema()))
        return accept_specialist_response(self.path, decision, inputs.context,
                                          evaluation_record_id=inputs.evaluation_record_id)

    def test_three_gamma_rounds_with_fake_scans(self):
        self.fixture.init(paper_color="light yellow")
        self.decide("propose_gamma")
        make_trial_sheet, read_scan = create_tuning_tools(self.root)
        references = []
        for number in (1, 2, 3):
            trial = make_trial_sheet.invoke({"round_id": number})
            self.assertEqual(trial["status"], "ok", trial)
            record_trial_result(self.path, trial)
            manifest = json.loads(Path(trial["manifest_path"]).read_text(encoding="utf-8"))
            self.assertEqual(manifest["tested_parameter"], "gamma")
            round_references = {}
            for item_id, entries in manifest["variations"].items():
                self.assertEqual(len(entries), 6)
                reference = next(v["mapping"] for v in entries if v["role"] == "reference")
                round_references[item_id] = reference
                tested = sorted(v["mapping"]["gamma"] for v in entries if v["role"] == "test")
                self.assertEqual(tested, sorted(fixtures.ROUND_GAMMAS[number]))
                for variation in entries:
                    self.assertEqual({k: variation["mapping"][k] for k in FIXED}, FIXED)
                    if variation["role"] == "test":
                        self.assertEqual(changed_parameters(reference, variation["mapping"]), {"gamma"})
            references.append(round_references)

            scans = fake_scan(trial["pdf_path"], self.root / "scans" / f"round_{number}")
            result = read_scan.invoke({"scan_paths": scans})
            self.assertEqual(result["status"], "ok", result)
            self.assertEqual(len(result["candidates"]), 24)
            record_scan_for_executor(self.path, result, record_id=f"scan-{number}")
            session = self.decide(STAGES[number], number)
            if number < 3:
                self.assertEqual(session["active_round"], number + 1)
                self.assertEqual(session["phase"], "ready_for_trial")

        # Between rounds only the reference gamma changes: 1.0, then 0.6, then 0.5.
        self.assertEqual([{m["gamma"] for m in r.values()} for r in references], [{1.}, {.6}, {.5}])
        for earlier, later in zip(references, references[1:]):
            for key in earlier:
                self.assertEqual(changed_parameters(earlier[key], later[key]), {"gamma"})

        session = load_tuning_session(self.path)
        self.assertEqual(session["status"], "completed")
        self.assertEqual([d["stage"] for d in session["decisions"]],
                         ["propose_gamma", "review_gamma", "review_gamma", "review_gamma_final"])
        self.assertEqual(session["final_settings"], session["accepted_settings"])
        for mapping in session["final_settings"].values():
            self.assertEqual(mapping, {"gamma": fixtures.FINAL_GAMMA, **FIXED, "invert": False})


if __name__ == "__main__":
    unittest.main()
