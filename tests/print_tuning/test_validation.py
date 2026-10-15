"""Focused offline validation tests; run with standard-library unittest discovery."""

from copy import deepcopy
from dataclasses import replace
import json
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np
from pydantic import ValidationError

from app.experiments.selection import build_selection
from app.print_tuning.grayscale_render import default_mapping, load_config
from app.print_tuning.schemas import (
    GammaProposalResponse, GammaReviewResponse, GrayLevelsReviewResponse, StepsReviewResponse,
)
from app.print_tuning.trial_sheet import create_trial_sheet
from app.print_tuning.validation import (
    CandidateIdentity, TuningValidationError, ValidationContext, get_candidate_correspondence,
    resolve_review_winners, validate_response,
)


IDS = ("exp/A", "exp-B", "exp-C", "exp-D")
GAMMAS = (0.4, 0.6, 1.5, 2.2, 3.0)
PAIRS = ((0.02, 1.0), (0.04, 1.0), (0.06, 0.95), (0.08, 0.90), (0.1, 0.85))
STEPS = (0, 2, 4, 8, 16)


def winner(experiment=None, candidate="reference"):
    value = dict(candidate_id=candidate, reason="Visible detail retained", confidence="medium", tie_explanation=None)
    if experiment is not None:
        value["experiment_id"] = experiment
    return value


def payload(stage):
    gamma = [dict(candidate_id=f"test_{i}", gamma=g, reason="Compare tone curve") for i, g in enumerate(GAMMAS, 1)]
    endpoints = [dict(candidate_id=f"test_{i}", dark_level=d, light_level=l, reason="Compare contrast")
                 for i, (d, l) in enumerate(PAIRS, 1)]
    steps = [dict(candidate_id=f"test_{i}", steps=s, reason="Compare gray bands") for i, s in enumerate(STEPS, 1)]
    if stage == "propose_gamma":
        return dict(stage=stage, experiments=[dict(experiment_id=e, tests=deepcopy(gamma)) for e in IDS])
    data = dict(stage=stage, assessment="usable", issues=[])
    if stage == "review_gamma":
        data.update(gamma_winners=[winner(e) for e in IDS], endpoint_tests=endpoints)
    elif stage == "review_gray_levels":
        data.update(endpoint_winner=winner(), step_tests=[dict(experiment_id=e, tests=deepcopy(steps)) for e in IDS])
    else:
        data.update(step_winners=[winner(e) for e in IDS])
    return data


MODELS = {"propose_gamma": GammaProposalResponse, "review_gamma": GammaReviewResponse,
          "review_gray_levels": GrayLevelsReviewResponse, "review_steps": StepsReviewResponse}


class ValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.temp.cleanup)
        folder = Path(cls.temp.name)
        cls.config = load_config()
        # A valid hand-set initial band count keeps 0 available as a new test.
        base = default_mapping(cls.config) | {"steps": 6}
        cls.base = base
        # Real selection packaging preserves the original IDs but sanitizes exp/A.
        camera = folder / "camera.npy"
        np.save(camera, np.array([[0., 1.], [2., 4.]]))
        results = [dict(ok=True, experiment_id=e, settings={"iterations": 2},
                        initial_enhancement=1., final_enhancement=2.,
                        paths={"initial_intensity": str(camera), "final_intensity": str(camera),
                               "results_json": str(folder / "results.json")}) for e in IDS]
        selection = build_selection(results, "run", folder, "scientist")
        cls.association = {item["source"]["experiment_id"]: item["item_id"] for item in selection["items"]}
        cls.manifests = {}
        for round_id in (1, 2, 3):
            reference = dict(base)
            variants = {}
            for item in selection["items"]:
                values = ([{"gamma": g} for g in GAMMAS] if round_id == 1 else
                          [{"dark_level": d, "light_level": l} for d, l in PAIRS] if round_id == 2 else
                          [{"steps": s} for s in STEPS])
                variants[item["item_id"]] = [dict(variation_id="ref", role="reference", mapping=reference, note="")]
                variants[item["item_id"]] += [dict(variation_id=f"v{i}", role="test", mapping=reference | value, note="")
                                               for i, value in enumerate(values)]
            record = create_trial_sheet(selection, variants, round_id, folder, "calibration", config=cls.config)
            # Use the actual on-disk JSON representation, not a parallel fixture shape.
            cls.manifests[round_id] = json.loads(Path(record["manifest_path"]).read_text(encoding="utf-8"))

    def context(self, stage):
        if stage == "propose_gamma":
            return ValidationContext(stage, IDS, deepcopy(self.config), {e: dict(self.base) for e in IDS})
        round_id = {"review_gamma": 1, "review_gray_levels": 2, "review_steps": 3}[stage]
        manifest = deepcopy(self.manifests[round_id])
        return ValidationContext(stage, IDS, deepcopy(self.config), reviewed_manifest=manifest,
                                 expected_trial_id=manifest["trial_id"], expected_calibration_session_id="calibration",
                                 experiment_item_ids=self.association, expected_run_id="run",
                                 expected_selection_id=manifest["selection_id"], expected_selection_version=1)

    def check(self, data, context=None):
        response = MODELS[data["stage"]].model_validate(data)
        context = context or self.context(data["stage"])
        self.assertIs(validate_response(response, context), response)
        return response

    def rejects(self, data, message, context=None):
        response = MODELS[data["stage"]].model_validate(data)
        with self.assertRaisesRegex(TuningValidationError, message):
            validate_response(response, context or self.context(data["stage"]))

    def test_valid_all_stages_and_reference_winners(self):
        for stage in MODELS:
            with self.subTest(stage=stage):
                self.check(payload(stage))

    def test_nonreference_winners_and_ties(self):
        for stage, field in (("review_gamma", "gamma_winners"), ("review_steps", "step_winners")):
            data = payload(stage)
            data[field][0].update(candidate_id="test_2", tie_explanation="Effectively tied; retain detail")
            self.check(data)
        data = payload("review_gray_levels")
        data["endpoint_winner"].update(candidate_id="test_3", tie_explanation="Equivalent contrast")
        self.check(data)

    def test_blank_tie_explanation(self):
        for stage, field in (("review_gamma", "gamma_winners"), ("review_steps", "step_winners")):
            data = payload(stage)
            data[field][0]["tie_explanation"] = "  "
            self.rejects(data, "tie_explanation")
        data = payload("review_gray_levels")
        data["endpoint_winner"]["tie_explanation"] = "\n"
        self.rejects(data, "tie_explanation")

    def test_unusable_assessments_need_no_references(self):
        for stage in list(MODELS)[1:]:
            for assessment in ("needs_rescan", "insufficient_contrast"):
                data = payload(stage)
                data.update(assessment=assessment, issues=["Page alignment unreliable"])
                for field in ("gamma_winners", "endpoint_tests", "step_tests", "step_winners"):
                    if field in data:
                        data[field] = []
                if "endpoint_winner" in data:
                    data["endpoint_winner"] = None
                self.check(data)
                self.assertEqual(resolve_review_winners(MODELS[stage](**data), self.context(stage)), {})
                bad = deepcopy(data)
                bad["issues"] = []
                self.rejects(bad, "issues")
                bad = deepcopy(data)
                field = "step_tests" if stage == "review_gray_levels" else "gamma_winners" if stage == "review_gamma" else "step_winners"
                bad[field] = payload(stage)[field]
                self.rejects(bad, "must be empty")
                wrong = self.context(stage)
                self.rejects(data, "trial_id", replace(wrong, expected_trial_id="another-trial"))

    def test_missing_usable_decisions(self):
        for stage, field in (("review_gamma", "gamma_winners"), ("review_gamma", "endpoint_tests"),
                             ("review_gray_levels", "step_tests"), ("review_steps", "step_winners")):
            data = payload(stage)
            data[field] = []
            self.rejects(data, "missing")
        data = payload("review_gray_levels")
        data["endpoint_winner"] = None
        self.rejects(data, "endpoint_winner")

    def test_selected_ids_invalid(self):
        context = self.context("propose_gamma")
        for ids in (IDS[:3], (*IDS, "extra"), (IDS[0], IDS[0], IDS[2], IDS[3]), (" ", *IDS[1:])):
            self.rejects(payload("propose_gamma"), "four distinct nonblank", replace(context, selected_experiment_ids=ids))

    def test_experiment_coverage(self):
        for stage, field in (("propose_gamma", "experiments"), ("review_gray_levels", "step_tests"),
                             ("review_gamma", "gamma_winners"), ("review_steps", "step_winners")):
            for mode in ("missing", "extra", "duplicate"):
                data = payload(stage)
                if mode == "missing":
                    data[field].pop()
                elif mode == "extra":
                    data[field].append(deepcopy(data[field][0]) | {"experiment_id": "other"})
                else:
                    data[field][-1] = deepcopy(data[field][0])
                self.rejects(data, "missing=|duplicates=")

    def test_stage_mismatch(self):
        self.rejects(payload("propose_gamma"), "stage", replace(self.context("propose_gamma"), expected_stage="review_gamma"))
        response = GammaProposalResponse(**payload("propose_gamma"))
        response.stage = "review_steps"
        with self.assertRaisesRegex(TuningValidationError, "stage"):
            validate_response(response, self.context("propose_gamma"))

    def test_manifest_identity_and_round(self):
        for stage in list(MODELS)[1:]:
            context = self.context(stage)
            for field, value in (("round_id", 99), ("trial_id", "other"), ("calibration_session_id", "other"),
                                 ("run_id", "other"), ("selection_id", "other"), ("selection_version", 2),
                                 ("tested_parameter", "wrong"), ("schema_version", "trial/0")):
                broken = deepcopy(context.reviewed_manifest)
                broken[field] = value
                self.rejects(payload(stage), field, replace(context, reviewed_manifest=broken))
            for field in ("expected_trial_id", "expected_calibration_session_id"):
                self.rejects(payload(stage), field, replace(context, **{field: None}))
            self.rejects(payload(stage), "reviewed_manifest", replace(context, reviewed_manifest=None))

    def test_explicit_item_translation_required_for_sanitized_ids(self):
        context = self.context("review_gamma")
        self.rejects(payload("review_gamma"), "experiment_item_ids", replace(context, experiment_item_ids=None))
        for association in ({}, dict(self.association) | {IDS[1]: self.association[IDS[0]]}):
            self.rejects(payload("review_gamma"), "experiment_item_ids", replace(context, experiment_item_ids=association))

    def test_candidate_ids_counts(self):
        for stage, field in (("propose_gamma", "experiments"), ("review_gray_levels", "step_tests"),
                             ("review_gamma", "endpoint_tests")):
            for mode in ("missing", "duplicate", "extra"):
                data = payload(stage)
                tests = data[field] if field == "endpoint_tests" else data[field][0]["tests"]
                if mode == "missing":
                    tests.pop()
                elif mode == "duplicate":
                    tests[1]["candidate_id"] = "test_1"
                else:
                    tests.append(deepcopy(tests[0]))
                self.rejects(data, "candidate_id")

    def test_gamma_values_and_exact_comparison(self):
        for value in (-1., 5.1, 1., GAMMAS[1]):
            data = payload("propose_gamma")
            data["experiments"][0]["tests"][0]["gamma"] = value
            self.rejects(data, "outside|duplicate")
        data = payload("propose_gamma")
        data["experiments"][0]["tests"][0]["gamma"] = math.nextafter(GAMMAS[1], math.inf)
        self.check(data)  # Adjacent float64 values must not be merged by rounding.

    def test_nonfinite_and_schema_strict_types(self):
        for stage, field, key in (("propose_gamma", "experiments", "gamma"),
                                  ("review_gamma", "endpoint_tests", "dark_level")):
            for value in (float("nan"), float("inf"), float("-inf"), True):
                data = payload(stage)
                tests = data[field] if field == "endpoint_tests" else data[field][0]["tests"]
                tests[0][key] = value
                with self.assertRaises(ValidationError):
                    MODELS[stage](**data)
        for value in (True, 2.5, 2.0):
            data = payload("review_gray_levels")
            data["step_tests"][0]["tests"][0]["steps"] = value
            with self.assertRaises(ValidationError):
                GrayLevelsReviewResponse(**data)
        response = GammaProposalResponse(**payload("propose_gamma"))
        response.experiments[0].tests[0].gamma = float("nan")
        with self.assertRaisesRegex(TuningValidationError, "finite"):
            validate_response(response, self.context("propose_gamma"))

    def test_endpoints_bounds_gap_duplicates_and_reference(self):
        for dark, light in ((-.1, 1.), (.1, 1.1), (.4, .6), (.4, .2), (0., 1.), PAIRS[1]):
            data = payload("review_gamma")
            data["endpoint_tests"][0].update(dark_level=dark, light_level=light)
            self.rejects(data, "outside|smaller|duplicate")
        self.check(payload("review_gamma"))  # Several valid pairs share light=1.

    def test_steps_smooth_bounds_duplicates_reference(self):
        self.check(payload("review_gray_levels"))  # Includes smooth 0.
        context = self.context("review_gray_levels")
        for value in (1, -1, 33, 2):
            data = payload("review_gray_levels")
            data["step_tests"][0]["tests"][0]["steps"] = value
            self.rejects(data, "not in special_values|duplicate", context)
        # A caller may start from smooth 0; it must not then be proposed again.
        bad_context = self.context("review_gray_levels")
        for entries in bad_context.reviewed_manifest["variations"].values():
            for entry in entries:
                entry["mapping"]["steps"] = 0
        self.rejects(payload("review_gray_levels"), "duplicates the applicable reference", bad_context)

    def test_config_count_and_missing_context(self):
        context = self.context("propose_gamma")
        for count, refs in ((5, 1), (7, 1), (7, 2), (6, True)):
            config = deepcopy(self.config)
            config["page"]["variations_per_item"] = count
            config["variation_rules"]["reference_count"] = refs
            self.rejects(payload("propose_gamma"), "schema", replace(context, config=config))
        self.rejects(payload("propose_gamma"), "config", replace(context, config={}))
        self.rejects(payload("propose_gamma"), "reference_mappings", replace(context, reference_mappings=None))

    def test_winner_manifest_membership_and_incomplete_mappings(self):
        data = payload("review_gamma")
        data["gamma_winners"][0]["candidate_id"] = "test_5"
        context = self.context("review_gamma")
        item = self.association[IDS[0]]
        context.reviewed_manifest["variations"][item].pop()
        self.rejects(data, "need one reference|absent", context)
        for key, value in (("gamma", float("inf")), ("steps", True), ("steps", 1), ("invert", "false")):
            context = self.context("review_gamma")
            context.reviewed_manifest["variations"][item][0]["mapping"][key] = value
            self.rejects(payload("review_gamma"), "mapping", context)
        context = self.context("review_gamma")
        del context.reviewed_manifest["variations"][item][0]["mapping"]["invert"]
        self.rejects(payload("review_gamma"), "missing", context)

    def test_ambiguous_or_incomplete_manifest_translation(self):
        for mode in ("duplicate_id", "duplicate_reference", "canonical_conflict", "missing_item"):
            context = self.context("review_gamma")
            entries = context.reviewed_manifest["variations"][self.association[IDS[0]]]
            if mode == "duplicate_id":
                entries[1]["variation_id"] = entries[0]["variation_id"]
            elif mode == "duplicate_reference":
                entries[1]["role"] = "reference"
            elif mode == "canonical_conflict":
                entries[1]["variation_id"] = "test_2"
            else:
                context.reviewed_manifest["items"].pop()
            self.rejects(payload("review_gamma"), "variation|items", context)

    def test_shared_winner_allows_different_gamma_steps_but_not_endpoints(self):
        data = payload("review_gray_levels")
        data["endpoint_winner"]["candidate_id"] = "test_1"
        context = self.context("review_gray_levels")
        for index, entries in enumerate(context.reviewed_manifest["variations"].values()):
            for entry in entries:
                entry["mapping"].update(gamma=1. + index / 10, steps=20 + index)
        self.check(data, context)
        broken = deepcopy(context.reviewed_manifest)
        broken["variations"][self.association[IDS[0]]][1]["mapping"]["dark_level"] = .03
        self.rejects(data, "identical endpoints|endpoints must be identical", replace(context, reviewed_manifest=broken))

    def test_updated_references_and_pure_resolution(self):
        data = payload("review_gamma")
        data["gamma_winners"][0]["candidate_id"] = "test_1"
        response = GammaReviewResponse(**data)
        context = self.context("review_gamma")
        original = deepcopy(context.reviewed_manifest)
        resolved = resolve_review_winners(response, context)
        self.assertEqual(resolved[IDS[0]]["gamma"], GAMMAS[0])
        self.check(data, replace(context, reference_mappings=resolved))
        self.rejects(data, "stale", replace(context, reference_mappings={e: dict(self.base) for e in IDS}))
        resolved[IDS[0]]["gamma"] = 99
        self.assertEqual(context.reviewed_manifest, original)
        # Round-2 endpoint winner has steps inherited as 6 in this fixture.
        data = payload("review_gray_levels")
        self.check(data)
        data["step_tests"][0]["tests"][0]["steps"] = 6
        self.rejects(data, "duplicates the applicable reference")

    def test_unshared_reference_endpoints(self):
        context = self.context("propose_gamma")
        refs = deepcopy(context.reference_mappings)
        refs[IDS[0]]["dark_level"] = .02
        self.rejects(payload("propose_gamma"), "shared consistently", replace(context, reference_mappings=refs))

    def test_manifest_config_mismatch(self):
        context = self.context("review_gamma")
        context.reviewed_manifest["config"]["parameters"]["gamma"]["max"] = 4.
        self.rejects(payload("review_gamma"), "manifest.config", context)

    def test_final_stage_extra_proposals_rejected(self):
        data = payload("review_steps") | {"step_tests": []}
        with self.assertRaises(ValidationError):
            StepsReviewResponse(**data)

    def test_resolved_endpoint_winner_rejects_stale_full_references(self):
        data = payload("review_gray_levels")
        data["endpoint_winner"]["candidate_id"] = "test_1"
        context = self.context("review_gray_levels")
        response = GrayLevelsReviewResponse(**data)
        resolved = resolve_review_winners(response, context)
        self.assertEqual(resolved[IDS[0]]["dark_level"], PAIRS[0][0])
        self.check(data, replace(context, reference_mappings=resolved))
        stale = {e: dict(self.base, steps=6) for e in IDS}
        self.rejects(data, "stale", replace(context, reference_mappings=stale))

    def test_invalid_reference_mappings(self):
        context = self.context("propose_gamma")
        for key, value in (("gamma", float("nan")), ("gamma", 6), ("steps", 1),
                           ("steps", 2.5), ("steps", True), ("light_level", .1)):
            refs = deepcopy(context.reference_mappings)
            refs[IDS[0]][key] = value
            self.rejects(payload("propose_gamma"), "reference_mappings", replace(context, reference_mappings=refs))

    def test_unusable_endpoint_winner_forbidden(self):
        data = payload("review_gray_levels")
        data.update(assessment="needs_rescan", issues=["Blurred"], step_tests=[])
        self.rejects(data, "endpoint_winner: must be null")

    def test_collects_independent_errors(self):
        data = payload("propose_gamma")
        data["experiments"][0]["tests"][0]["gamma"] = 8.
        data["experiments"][1]["tests"][1]["gamma"] = 9.
        with self.assertRaises(TuningValidationError) as caught:
            validate_response(GammaProposalResponse(**data), self.context("propose_gamma"))
        self.assertTrue(any(IDS[0] in e for e in caught.exception.errors))
        self.assertTrue(any(IDS[1] in e for e in caught.exception.errors))

    def assert_review_rejected_by_both(self, stage, context, message):
        response = MODELS[stage](**payload(stage))
        for validate in (validate_response, resolve_review_winners):
            with self.subTest(api=validate.__name__), self.assertRaisesRegex(TuningValidationError, message):
                validate(response, context)

    def test_nonwinning_printed_endpoint_must_be_shared(self):
        context = self.context("review_gray_levels")
        # Winner is reference; test_5 is still required to be shared.
        entries = context.reviewed_manifest["variations"][self.association[IDS[0]]]
        entries[5]["mapping"]["dark_level"] = .11
        self.assert_review_rejected_by_both("review_gray_levels", context, "shared|identical")

    def test_printed_reference_endpoints_must_be_shared_in_every_round(self):
        for stage in list(MODELS)[1:]:
            context = self.context(stage)
            entries = context.reviewed_manifest["variations"][self.association[IDS[0]]]
            # Shift every candidate in gamma/steps rounds so each still changes
            # only its own parameter. This isolates cross-experiment sharing.
            for entry in entries if stage != "review_gray_levels" else entries[:1]:
                entry["mapping"]["dark_level"] = .01
            self.assert_review_rejected_by_both(stage, context, "shared|identical")

    def test_duplicate_printed_tests_with_distinct_ids(self):
        for stage in list(MODELS)[1:]:
            context = self.context(stage)
            # Duplicate tests in every item to preserve endpoint sharing.
            for entries in context.reviewed_manifest["variations"].values():
                entries[2]["mapping"] = dict(entries[1]["mapping"])
                self.assertNotEqual(entries[1]["variation_id"], entries[2]["variation_id"])
            self.assert_review_rejected_by_both(stage, context, "duplicate")

    def test_printed_tests_cannot_duplicate_reference(self):
        for stage in list(MODELS)[1:]:
            context = self.context(stage)
            for entries in context.reviewed_manifest["variations"].values():
                entries[1]["mapping"] = dict(entries[0]["mapping"])
            self.assert_review_rejected_by_both(stage, context, "reference|change only")

    def test_malformed_context_types(self):
        context = self.context("review_gamma")
        malformed = [replace(context, experiment_item_ids=value) for value in (42, "bad", [42])]
        malformed += [replace(context, experiment_item_ids=dict(self.association) | {IDS[0]: value})
                      for value in ([], {}, ["item"])]
        malformed += [replace(context, selected_experiment_ids=value)
                      for value in (None, 42, [[], *IDS[1:]], [{}, *IDS[1:]], [None, *IDS[1:]])]
        malformed += [replace(context, expected_stage=value) for value in ([], {}, 42)]
        for index, bad in enumerate(malformed):
            with self.subTest(case=index):
                response = GammaReviewResponse(**payload("review_gamma"))
                for validate in (validate_response, resolve_review_winners):
                    with self.subTest(api=validate.__name__), self.assertRaises(TuningValidationError):
                        validate(response, bad)

    def test_string_selection_is_not_a_sequence_of_four_ids(self):
        data = payload("propose_gamma")
        for group, experiment in zip(data["experiments"], "abcd"):
            group["experiment_id"] = experiment
        context = replace(self.context("propose_gamma"), selected_experiment_ids="abcd",
                          reference_mappings={e: dict(self.base) for e in "abcd"})
        self.rejects(data, "selected_experiment_ids", context)

    def test_malformed_reference_and_manifest_types(self):
        initial = self.context("propose_gamma")
        for refs in (42, [], {**initial.reference_mappings, "extra": self.base, 99: self.base},
                     dict(initial.reference_mappings) | {IDS[0]: []}):
            with self.subTest(references=refs):
                self.rejects(payload("propose_gamma"), "reference_mappings", replace(initial, reference_mappings=refs))
        for field, value in (("items", None), ("items", [None]), ("items", [{"item_id": []}]),
                             ("variations", []), ("variations", {"extra": [], 99: []})):
            context = self.context("review_gamma")
            context.reviewed_manifest[field] = value
            with self.subTest(field=field, value=value):
                self.assert_review_rejected_by_both("review_gamma", context, "reviewed_manifest")

    def test_candidate_correspondence_is_deterministic_and_pure(self):
        context = self.context("review_gamma")
        original = deepcopy(context.reviewed_manifest)
        correspondence = get_candidate_correspondence(context)
        self.assertEqual(correspondence, get_candidate_correspondence(context))
        self.assertEqual(len(correspondence), 24)
        self.assertEqual(correspondence[:2], (
            CandidateIdentity(IDS[0], self.association[IDS[0]], "reference", "ref"),
            CandidateIdentity(IDS[0], self.association[IDS[0]], "test_1", "v0"),
        ))
        self.assertEqual(context.reviewed_manifest, original)
        # Item-array/dict order is irrelevant; frozen selection order determines output.
        reordered = deepcopy(original)
        reordered["items"].reverse()
        reordered["variations"] = dict(reversed(list(reordered["variations"].items())))
        self.assertEqual(correspondence, get_candidate_correspondence(replace(context, reviewed_manifest=reordered)))
        # Reference may occur between tests; test order still determines labels.
        for entries in context.reviewed_manifest["variations"].values():
            entries.insert(3, entries.pop(0))
        self.assertEqual(correspondence, get_candidate_correspondence(context))

    def test_candidate_correspondence_rejects_conflicts_and_malformed_context(self):
        for mode in ("canonical_conflict", "duplicate_id", "duplicate_reference", "missing_test"):
            context = self.context("review_gamma")
            entries = context.reviewed_manifest["variations"][self.association[IDS[0]]]
            if mode == "canonical_conflict":
                entries[1]["variation_id"] = "test_2"
            elif mode == "duplicate_id":
                entries[1]["variation_id"] = "ref"
            elif mode == "duplicate_reference":
                entries[1]["role"] = "reference"
            else:
                entries.pop()
            with self.subTest(mode=mode), self.assertRaisesRegex(TuningValidationError, "variation|reference"):
                get_candidate_correspondence(context)
        for context in (None, self.context("propose_gamma"),
                        replace(self.context("review_gamma"), experiment_item_ids={IDS[0]: []})):
            with self.assertRaises(TuningValidationError):
                get_candidate_correspondence(context)

    def test_printed_duplicates_use_exact_binary64_equality(self):
        context = self.context("review_gamma")
        # Adjacent representable values remain distinct despite printed label rounding.
        for entries in context.reviewed_manifest["variations"].values():
            entries[2]["mapping"]["gamma"] = math.nextafter(entries[1]["mapping"]["gamma"], math.inf)
        self.check(payload("review_gamma"), context)
        get_candidate_correspondence(context)

    def test_inherited_values_may_differ_between_experiments(self):
        for stage in ("review_gamma", "review_gray_levels", "review_steps"):
            context = self.context(stage)
            for index, entries in enumerate(context.reviewed_manifest["variations"].values()):
                for entry in entries:
                    if stage != "review_gamma":
                        entry["mapping"]["gamma"] = 1. + index / 10
                    if stage != "review_steps":
                        entry["mapping"]["steps"] = 20 + index
            self.check(payload(stage), context)
            self.assertEqual(len(resolve_review_winners(MODELS[stage](**payload(stage)), context)), 4)

    def test_rescan_still_validates_manifest_and_skips_proposal_references(self):
        data = payload("review_gamma")
        data.update(assessment="needs_rescan", issues=["Blurred"], gamma_winners=[], endpoint_tests=[])
        response = GammaReviewResponse(**data)
        context = replace(self.context("review_gamma"), reference_mappings=None)
        self.assertIs(validate_response(response, context), response)
        self.assertEqual(resolve_review_winners(response, context), {})
        for validate in (validate_response, resolve_review_winners):
            with self.assertRaisesRegex(TuningValidationError, "trial_id"):
                validate(response, replace(context, expected_trial_id=None))
        # Even a nonusable response may not accept an internally inconsistent sheet.
        entries = context.reviewed_manifest["variations"][self.association[IDS[0]]]
        for entry in entries:
            entry["mapping"]["dark_level"] = .01
        for validate in (validate_response, resolve_review_winners):
            with self.assertRaisesRegex(TuningValidationError, "shared"):
                validate(response, context)

    def test_smooth_printed_candidate_can_win(self):
        data = payload("review_steps")
        data["step_winners"][0]["candidate_id"] = "test_1"
        context = self.context("review_steps")
        response = self.check(data, context)
        mappings = resolve_review_winners(response, context)
        self.assertEqual(mappings[IDS[0]]["steps"], 0)
        self.assertEqual(mappings[IDS[1]]["steps"], self.base["steps"])


if __name__ == "__main__":
    unittest.main()
