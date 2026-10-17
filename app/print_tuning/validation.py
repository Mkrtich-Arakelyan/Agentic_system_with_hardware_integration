"""Pure contextual validation of parsed print-tuning decisions.

No file access, model calls, retries, or persistence. Numeric equality uses exact
Python float (binary64) equality, matching the renderer's float64 inputs; no
rounding or tolerance merges distinct renderable parameter values.

The caller supplies loaded config/manifest and frozen IDs. For initial proposals
it supplies complete current reference mappings. For usable reviews this module
resolves the printed winners first, then checks proposals against those complete
mappings. Optional caller-supplied references must equal these resolved winners;
stale references are rejected. The future graph owns accepting/persisting them.

trial/1 drops selection.items[].source.experiment_id. If item IDs differ from
experiment IDs, callers must supply the original explicit experiment_item_ids
association; we never reverse or guess the lossy sanitization. Session identity
is calibration_session_id, not a new session_id field. Candidate translation uses
role=reference and the stored order of role=test entries, not variation_id text.
The application attaching images must use this same label translation.
"""

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
from typing import Any, Literal, TypeVar, get_args

from pydantic import ValidationError

from app.print_tuning.grayscale_render import (
    MAPPING_PARAMETERS, changed_parameters, validate_config, validate_mapping,
)
from app.print_tuning.schemas import (
    GammaProposalResponse, GammaReviewResponse, GrayLevelsReviewResponse,
    StepsReviewResponse, TestCandidateId, WinnerCandidateId,
)

Stage = Literal["propose_gamma", "review_gamma", "review_gray_levels", "review_steps"]
Response = GammaProposalResponse | GammaReviewResponse | GrayLevelsReviewResponse | StepsReviewResponse
ResponseT = TypeVar("ResponseT", bound=Response)
EffectiveMapping = dict[str, Any]
TEST_IDS = get_args(TestCandidateId)
_MODELS = dict(zip(get_args(Stage), (
    GammaProposalResponse, GammaReviewResponse, GrayLevelsReviewResponse, StepsReviewResponse,
)))
_ROUNDS = {"review_gamma": (1, "gamma"), "review_gray_levels": (2, "gray_levels"),
           "review_steps": (3, "steps")}


class TuningValidationError(ValueError):
    """Actionable contextual errors, available individually through ``errors``."""

    def __init__(self, errors: Sequence[str]):
        self.errors = tuple(errors)
        super().__init__("Invalid tuning decision:\n- " + "\n- ".join(self.errors))


@dataclass(frozen=True)
class ValidationContext:
    """Loaded evidence for one decision; objects are never mutated.

    Reviews require expected_trial_id and expected_calibration_session_id from
    the caller's active trial, even for unusable assessments. Optional run and
    selection identities are checked when supplied. experiment_item_ids maps
    original experiment IDs to trial/1 item IDs; omission permits exact identity
    only. reference_mappings is required for propose_gamma, optional for reviews
    (where any supplied mappings must already include the reviewed winners).
    """

    expected_stage: Stage
    selected_experiment_ids: Sequence[str]
    config: dict[str, Any]
    reference_mappings: Mapping[str, Mapping[str, Any]] | None = None
    reviewed_manifest: dict[str, Any] | None = None
    expected_trial_id: str | None = None
    expected_calibration_session_id: str | None = None
    experiment_item_ids: Mapping[str, str] | None = None
    expected_run_id: str | None = None
    expected_selection_id: str | None = None
    expected_selection_version: int | None = None
    cycle_id: int = 1
    step_id: int = 1
    previous_cycle_evidence: dict | None = None


@dataclass(frozen=True)
class CandidateIdentity:
    """One specialist label's correspondence to an existing trial variation."""

    experiment_id: str
    item_id: str
    candidate_id: WinnerCandidateId
    variation_id: str


def _nonblank(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _finite(value: Any) -> bool:
    try:
        return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
    except OverflowError:
        return False


def _coverage(actual: Sequence[str], wanted: Sequence[str], where: str, errors: list[str]) -> None:
    invalid = [i for i, value in enumerate(actual) if not _nonblank(value)]
    if invalid:
        errors.append(f"{where}: require nonblank string IDs; invalid entries at indices {invalid}")
        return
    counts = Counter(actual)
    missing, extra = set(wanted) - counts.keys(), counts.keys() - set(wanted)
    duplicates = [key for key, count in counts.items() if count > 1]
    if missing or extra or duplicates:
        errors.append(f"{where}: missing={sorted(missing)}, extra={sorted(extra)}, duplicates={duplicates}")


def _config(config: dict[str, Any], errors: list[str]) -> None:
    try:
        validate_config(config)
        count = config["page"]["variations_per_item"]
        refs = config["variation_rules"]["reference_count"]
        if type(refs) is not int or refs != 1 or count - refs != len(TEST_IDS):
            errors.append(f"config: variations_per_item ({count}) minus reference_count ({refs}) "
                          f"must match schema's {len(TEST_IDS)} test IDs and one reference")
        parameters = config["parameters"]
        for name, kind, scope in (("gamma", "float", "per_item"), ("gray_levels", "range", "shared"),
                                  ("steps", "int", "per_item"), ("invert", "bool", "per_item")):
            if parameters[name]["type"] != kind or parameters[name]["scope"] != scope:
                errors.append(f"config.parameters.{name}: expected type={kind}, scope={scope}")
        for name in ("gamma", "steps"):
            for key in ("min", "max", "default"):
                if not _finite(parameters[name][key]):
                    errors.append(f"config.parameters.{name}.{key}: must be finite")
        gray = parameters["gray_levels"]
        for end in ("dark", "light"):
            for key in ("min", "max", "default"):
                if not _finite(gray[end][key]):
                    errors.append(f"config.parameters.gray_levels.{end}.{key}: must be finite")
        if not _finite(gray["min_gap"]):
            errors.append("config.parameters.gray_levels.min_gap: must be finite")
        special = parameters["steps"].get("special_values", {})
        if not isinstance(special, dict) or any(str(int(key)) != key for key in special):
            errors.append("config.parameters.steps.special_values: keys must be canonical integer strings")
        expected = [(1, "gamma"), (2, "gray_levels"), (3, "steps")]
        if [(r["round_id"], r["parameter"]) for r in config["rounds"]] != expected:
            errors.append("config.rounds: expected gamma, gray_levels, steps as rounds 1, 2, 3")
    except (ValueError, KeyError, TypeError, AttributeError, OverflowError) as error:
        errors.append(f"config: incomplete or invalid tuning configuration: {error}")


def _mapping(mapping: Any, config: dict[str, Any], where: str, errors: list[str]) -> bool:
    before = len(errors)
    if not isinstance(mapping, Mapping):
        errors.append(f"{where}: complete effective mapping required")
        return False
    if not all(isinstance(key, str) for key in mapping):
        errors.append(f"{where}: effective mapping keys must be strings")
        return False
    missing = set(MAPPING_PARAMETERS) - set(mapping)
    if missing:
        errors.append(f"{where}: missing effective mapping keys {sorted(missing)}")
    for key in ("gamma", "dark_level", "light_level"):
        if not _finite(mapping.get(key)):
            errors.append(f"{where}.{key}: finite numeric value required")
    if not isinstance(mapping.get("steps"), int) or isinstance(mapping.get("steps"), bool):
        errors.append(f"{where}.steps: integer required (not a boolean)")
    if not isinstance(mapping.get("invert"), bool):
        errors.append(f"{where}.invert: boolean required")
    if len(errors) != before:
        return False
    try:
        validate_mapping(dict(mapping), config)
    except (ValueError, KeyError, TypeError, OverflowError) as error:
        errors.append(f"{where}: {error}")
    return len(errors) == before


def _prepare_context(context: ValidationContext) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    if not isinstance(context, ValidationContext):
        raise TuningValidationError(["context: a ValidationContext instance is required"])
    if type(context.cycle_id) is not int or context.cycle_id < 1 or type(context.step_id) is not int or context.step_id not in (1, 2, 3):
        errors.append("cycle_id/step_id: explicit positive cycle and step 1..3 required")
    selected = context.selected_experiment_ids
    if not isinstance(selected, Sequence) or isinstance(selected, (str, bytes, bytearray)):
        errors.append("selected_experiment_ids: require a sequence of four IDs, not a string or scalar")
        ids = []
    else:
        ids = list(selected)
    if len(ids) != 4 or not all(_nonblank(e) for e in ids) or len(set(ids)) != 4:
        errors.append("selected_experiment_ids: require exactly four distinct nonblank experiment IDs")
    if not isinstance(context.expected_stage, str) or context.expected_stage not in _MODELS:
        errors.append("expected_stage: require a supported decision-stage string")
    for field in ("expected_trial_id", "expected_calibration_session_id", "expected_run_id", "expected_selection_id"):
        value = getattr(context, field)
        if value is not None and not _nonblank(value):
            errors.append(f"{field}: must be a nonblank string when supplied")
    version = context.expected_selection_version
    if version is not None and (type(version) is not int or version < 1):
        errors.append("expected_selection_version: positive integer required when supplied")
    _config(context.config, errors)
    if errors:
        raise TuningValidationError(errors)
    return ids, errors


def _prepare(response: Response, context: ValidationContext) -> tuple[list[str], list[str]]:
    ids, errors = _prepare_context(context)
    model = _MODELS[context.expected_stage]
    if type(response) is not model or getattr(response, "stage", None) != context.expected_stage:
        errors.append(f"stage: expected {context.expected_stage!r} and its matching response model")
    # Reparse to guard against model_copy(update=...), mutation, or model_construct
    # bypassing declarative schema constraints (including extra final proposals).
    if model is not None and isinstance(response, tuple(_MODELS.values())):
        try:
            model.model_validate(response.model_dump())
        except ValidationError as error:
            errors.extend(f"response.{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in error.errors())
    if errors:
        raise TuningValidationError(errors)
    return ids, errors


def _translate_candidates(entries: Any, where: str, errors: list[str]) -> dict[str, dict]:
    """Single role/test-order translation used by validation and the public helper."""
    if not isinstance(entries, list) or not all(isinstance(entry, dict) for entry in entries):
        errors.append(f"{where}: ordered variation list required")
        return {}
    refs = [entry for entry in entries if entry.get("role") == "reference"]
    tests = [entry for entry in entries if entry.get("role") == "test"]
    internal = [entry.get("variation_id") for entry in entries]
    if (len(refs) != 1 or len(tests) != len(TEST_IDS) or len(entries) != len(TEST_IDS) + 1
            or not all(_nonblank(i) for i in internal) or len(set(internal)) != len(internal)):
        errors.append(f"{where}: need one reference and {len(TEST_IDS)} tests with unique variation_id values")
        return {}
    candidates = dict(zip(("reference", *TEST_IDS), (*refs, *tests)))
    for label, entry in candidates.items():
        if entry["variation_id"] in ("reference", *TEST_IDS) and entry["variation_id"] != label:
            errors.append(f"{where}.{label}: variation_id conflicts with role/test-order label translation")
    return candidates


def _parameter_values(mapping: Mapping[str, Any], parameter: str) -> tuple:
    """Exact binary64 equality for tones; exact integer equality for steps."""
    keys = ("dark_level", "light_level") if parameter == "gray_levels" else (parameter,)
    return tuple(mapping[key] if parameter == "steps" else float(mapping[key]) for key in keys)


def _manifest_candidates(context: ValidationContext, ids: list[str], errors: list[str]) -> dict[str, dict[str, dict]]:
    manifest = context.reviewed_manifest
    if not isinstance(manifest, dict):
        errors.append("reviewed_manifest: loaded trial/1 manifest required for a review")
        return {}
    round_id, parameter = _ROUNDS[context.expected_stage]
    checks = {"schema_version": "trial/1", "round_id": round_id, "tested_parameter": parameter}
    if "cycle_id" in manifest:
        checks.update(cycle_id=context.cycle_id, step_id=context.step_id)
    for field, expected in (("trial_id", context.expected_trial_id),
                            ("calibration_session_id", context.expected_calibration_session_id)):
        if not _nonblank(expected):
            errors.append(f"expected_{field}: required to bind the reviewed trial")
        checks[field] = expected
    for field, expected in (("run_id", context.expected_run_id), ("selection_id", context.expected_selection_id),
                            ("selection_version", context.expected_selection_version)):
        if expected is not None:
            checks[field] = expected
    for field, expected in checks.items():
        if field not in manifest or type(manifest[field]) is not type(expected) or manifest[field] != expected:
            errors.append(f"reviewed_manifest.{field}: expected {expected!r}, got {manifest.get(field)!r}")
    for field in ("run_id", "selection_id"):
        if not _nonblank(manifest.get(field)):
            errors.append(f"reviewed_manifest.{field}: nonblank identity required")
    version = manifest.get("selection_version")
    if type(version) is not int or version < 1:
        errors.append("reviewed_manifest.selection_version: positive integer required")
    if manifest.get("config") != context.config:
        errors.append("reviewed_manifest.config: must agree with the supplied frozen tuning configuration")
    if context.experiment_item_ids is not None and not isinstance(context.experiment_item_ids, Mapping):
        errors.append("experiment_item_ids: require an experiment-ID to item-ID mapping")
        return {}
    association = dict(context.experiment_item_ids) if context.experiment_item_ids is not None else dict(zip(ids, ids))
    _coverage(list(association), ids, "experiment_item_ids", errors)
    if not all(_nonblank(item) for item in association.values()) or len(set(association.values())) != len(ids):
        errors.append("experiment_item_ids: require distinct nonblank item IDs")
        return {}
    items, variations = manifest.get("items"), manifest.get("variations")
    if not isinstance(items, list) or not all(isinstance(item, dict) and _nonblank(item.get("item_id")) for item in items):
        errors.append("reviewed_manifest.items: require items with nonblank item_id")
        return {}
    _coverage([item["item_id"] for item in items], list(association.values()), "reviewed_manifest.items (supply experiment_item_ids if sanitized)", errors)
    if not isinstance(variations, dict):
        errors.append("reviewed_manifest.variations: require item_id -> ordered variations")
        return {}
    _coverage(list(variations), list(association.values()), "reviewed_manifest.variations", errors)
    result: dict[str, dict[str, dict]] = {}
    for experiment in ids:
        item_id = association.get(experiment)
        entries = variations.get(item_id)
        where = f"reviewed_manifest.variations[{experiment!r}]"
        candidates = _translate_candidates(entries, where, errors)
        valid_candidates = {}
        for label, entry in candidates.items():
            if _mapping(entry.get("mapping"), context.config, f"{where}.{label}.mapping", errors):
                valid_candidates[label] = entry
        if "reference" in valid_candidates:
            reference = valid_candidates["reference"]["mapping"]
            original = _parameter_values(reference, parameter)
            seen: dict[tuple, str] = {}
            for label in TEST_IDS:
                if label not in valid_candidates:
                    continue
                mapping = valid_candidates[label]["mapping"]
                if changed_parameters(reference, mapping) != {parameter}:
                    errors.append(f"{where}.{label}: must change only reviewed parameter {parameter!r}")
                values = _parameter_values(mapping, parameter)
                if values == original:
                    errors.append(f"{where}.{label}: duplicates printed reference {parameter}")
                if values in seen:
                    errors.append(f"{where}.{label}: duplicate printed {parameter} test (same as {seen[values]})")
                seen[values] = label
        result[experiment] = valid_candidates
    # Validate every printed label, including candidates not selected as winners.
    # In gamma/steps rounds, the per-item change-only check also freezes endpoints
    # within that item. Gamma and steps may vary between experiments.
    for label in ("reference", *TEST_IDS):
        shared: tuple | None = None
        first_experiment = None
        for experiment in ids:
            entry = result.get(experiment, {}).get(label)
            if entry is None:
                continue
            pair = _parameter_values(entry["mapping"], "gray_levels")
            if shared is None:
                shared, first_experiment = pair, experiment
            elif pair != shared:
                errors.append(f"reviewed_manifest.variations[{experiment!r}].{label}: shared endpoints "
                              f"must be identical to {first_experiment!r} for this candidate")
    return result


def get_candidate_correspondence(context: ValidationContext) -> tuple[CandidateIdentity, ...]:
    """Validate a bound reviewed manifest and return its specialist-label identities.

    Order is frozen experiment order, then reference/test_1/.../test_5. The sole
    role=reference entry maps to reference; role=test entries map in stored list
    order. Image-message construction and scan-tool integration must reuse this
    correspondence, not infer labels from variation IDs. No integration or state
    mutation is performed here. Proposal-stage contexts have no reviewed trial.
    """
    ids, errors = _prepare_context(context)
    if context.expected_stage not in _ROUNDS:
        raise TuningValidationError(["stage: candidate correspondence requires a review stage"])
    candidates = _manifest_candidates(context, ids, errors)
    if errors:
        raise TuningValidationError(errors)
    return tuple(
        CandidateIdentity(
            experiment_id=experiment,
            item_id=context.experiment_item_ids[experiment] if context.experiment_item_ids is not None else experiment,
            candidate_id=label,
            variation_id=entry["variation_id"],
        )
        for experiment in ids
        for label, entry in candidates[experiment].items()
    )


def _winners(response: Response, ids: list[str], candidates: dict[str, dict[str, dict]],
             errors: list[str]) -> dict[str, EffectiveMapping]:
    if isinstance(response, GrayLevelsReviewResponse):
        winner = response.endpoint_winner
        if winner is None:
            errors.append("endpoint_winner: required for a usable review")
            return {}
        winners = [(experiment, winner) for experiment in ids]
    else:
        field = "gamma_winners" if isinstance(response, GammaReviewResponse) else "step_winners"
        entries = getattr(response, field)
        _coverage([e.experiment_id for e in entries], ids, field, errors)
        winners = [(entry.experiment_id, entry) for entry in entries]
    resolved: dict[str, EffectiveMapping] = {}
    for experiment, winner in winners:
        where = f"winner[{experiment!r}].{winner.candidate_id}"
        if winner.tie_explanation is not None and not _nonblank(winner.tie_explanation):
            errors.append(f"{where}.tie_explanation: must be nonblank or null")
        candidate = candidates.get(experiment, {}).get(winner.candidate_id)
        if candidate is None:
            errors.append(f"{where}: absent from this experiment in the reviewed trial/round")
        elif isinstance(candidate.get("mapping"), dict):
            resolved[experiment] = dict(candidate["mapping"])
    if isinstance(response, GrayLevelsReviewResponse) and len(resolved) == len(ids):
        pairs = {(m.get("dark_level"), m.get("light_level")) for m in resolved.values()}
        if len(pairs) != 1:
            errors.append("endpoint_winner: candidate endpoints must be identical across all selected experiments")
    return resolved


def _assessment(response: Response, errors: list[str]) -> bool:
    if isinstance(response, GammaProposalResponse) or response.assessment == "usable":
        return True
    if not response.issues or not all(_nonblank(issue) for issue in response.issues):
        errors.append("issues: an unusable assessment requires at least one nonblank explanation")
    for field in ("gamma_winners", "endpoint_tests", "step_tests", "step_winners"):
        if getattr(response, field, []):
            errors.append(f"{field}: must be empty for assessment={response.assessment!r}")
    if getattr(response, "endpoint_winner", None) is not None:
        errors.append("endpoint_winner: must be null for an unusable assessment")
    return False


def resolve_review_winners(response: Response, context: ValidationContext) -> dict[str, EffectiveMapping]:
    """Validate review identity/assessment/winners and return copied full mappings.

    This phase deliberately does not check next proposals or supplied references.
    A future graph may use it before full validation; nothing is accepted or
    persisted by this function. Unusable reviews return an empty dict.
    """
    ids, errors = _prepare(response, context)
    if context.expected_stage not in _ROUNDS:
        raise TuningValidationError(["stage: winner resolution requires a review stage"])
    candidates = _manifest_candidates(context, ids, errors)
    resolved = _winners(response, ids, candidates, errors) if _assessment(response, errors) else {}
    if errors:
        raise TuningValidationError(errors)
    return resolved


def _references(references: Mapping[str, Mapping[str, Any]] | None, ids: list[str],
                config: dict, errors: list[str]) -> dict[str, EffectiveMapping]:
    if not isinstance(references, Mapping):
        errors.append("reference_mappings: complete mappings for all selected experiments required")
        return {}
    _coverage(list(references), ids, "reference_mappings", errors)
    valid = {}
    for experiment in ids:
        mapping = references.get(experiment)
        if _mapping(mapping, config, f"reference_mappings[{experiment!r}]", errors):
            valid[experiment] = dict(mapping)
    if len(valid) == len(ids) and len({(m["dark_level"], m["light_level"]) for m in valid.values()}) != 1:
        errors.append("reference_mappings: endpoint pairs must be shared consistently across all experiments")
    return valid


def validate_context(context: ValidationContext) -> ValidationContext:
    """Check non-model context before requesting a specialist decision.

    Initial proposals require references. Reviews require a valid bound manifest;
    optional supplied references are checked for shape/ranges/sharing only here.
    Their agreement with the eventual winners is checked by validate_response.
    No proposal or winner is fabricated to perform this preflight.
    """
    ids, errors = _prepare_context(context)
    if context.expected_stage in _ROUNDS:
        _manifest_candidates(context, ids, errors)
    if context.expected_stage == "propose_gamma" or context.reference_mappings is not None:
        _references(context.reference_mappings, ids, context.config, errors)
    if errors:
        raise TuningValidationError(errors)
    return context


def _tests(tests: Sequence[Any], reference: dict, parameter: str, config: dict,
           where: str, errors: list[str]) -> None:
    _coverage([t.candidate_id for t in tests], TEST_IDS, where + ".candidate_id", errors)
    seen: set[Any] = set()
    keys = ("dark_level", "light_level") if parameter == "gray_levels" else (parameter,)
    original = _parameter_values(reference, parameter)
    for test in tests:
        location = f"{where}.{test.candidate_id}"
        values = tuple(getattr(test, key) for key in keys)
        proposed = reference | dict(zip(keys, values))
        if not _mapping(proposed, config, location, errors):
            continue
        # Exact float64 comparison: no label-format rounding or arbitrary epsilon.
        comparable = _parameter_values(proposed, parameter)
        if comparable == original:
            errors.append(f"{location}: duplicates the applicable reference {parameter}")
        if comparable in seen:
            errors.append(f"{location}: duplicate {parameter} test")
        seen.add(comparable)


def validate_response(response: ResponseT, context: ValidationContext) -> ResponseT:
    """Return the same parsed response on success, or collect actionable errors.

    For usable reviews, validate/resolve winners from the bound manifest before
    validating next proposals. If reference_mappings is supplied, it must equal
    those resolved complete mappings. For unusable reviews no references are
    required, but reviewed trial identity and selection remain mandatory.
    """
    ids, errors = _prepare(response, context)
    usable = _assessment(response, errors)
    if context.expected_stage in _ROUNDS:
        candidates = _manifest_candidates(context, ids, errors)
        resolved = _winners(response, ids, candidates, errors) if usable else {}
        if usable and not errors:
            references = _references(resolved, ids, context.config, errors)
            if context.reference_mappings is not None:
                supplied = _references(context.reference_mappings, ids, context.config, errors)
                if supplied != references:
                    errors.append("reference_mappings: stale or inconsistent; supply complete mappings resolved from the reviewed winners")
        else:
            references = {}
    else:
        references = _references(context.reference_mappings, ids, context.config, errors)
    if usable:
        if isinstance(response, GammaReviewResponse):
            if references:
                _tests(response.endpoint_tests, next(iter(references.values())), "gray_levels", context.config, "endpoint_tests", errors)
            else:
                _coverage([t.candidate_id for t in response.endpoint_tests], TEST_IDS, "endpoint_tests.candidate_id", errors)
        elif isinstance(response, (GammaProposalResponse, GrayLevelsReviewResponse)):
            field, parameter = ("experiments", "gamma") if isinstance(response, GammaProposalResponse) else ("step_tests", "steps")
            groups = getattr(response, field)
            _coverage([group.experiment_id for group in groups], ids, field, errors)
            for group in groups:
                where = f"{field}[{group.experiment_id!r}]"
                if group.experiment_id in references:
                    _tests(group.tests, references[group.experiment_id], parameter, context.config, where, errors)
                else:
                    _coverage([t.candidate_id for t in group.tests], TEST_IDS, where + ".candidate_id", errors)
    if errors:
        raise TuningValidationError(errors)
    return response
