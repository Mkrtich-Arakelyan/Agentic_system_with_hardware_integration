"""Read-only adaptation of recorded tuning artifacts into specialist inputs.

Example (the caller owns the model and subsequent acceptance)::

    inputs = prepare_tuning_inputs(session_path, "review_gamma")
    response = photo_quality_specialist(
        llm, inputs.context, inputs.evidence,
        prior_decisions=inputs.prior_decisions, paper_color=inputs.paper_color,
    )
    accept_specialist_response(session_path, response, inputs.context,
                               evaluation_record_id=inputs.evaluation_record_id)

No models, tools, discovery, image processing, or session writes occur here.
The specialist's prepare_specialist_messages loads original image bytes. Producers
normally save absolute paths; relative preview paths are relative to the run,
and relative scan-panel paths are relative to the recorded evaluation JSON.
No basename searches or alternate files are used. Callers serialize session
operations, including preparation/model invocation/acceptance, or recheck the
returned evaluation_record_id at acceptance. Legacy previews without recorded
hashes can be loaded, but their historical pixel integrity cannot be certified.
"""

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from app.agents.photo_quality_specialist import (
    CandidateMeasurement, ImageAttachment, ImageLabel, MissingImage,
    SpecialistEvidence, SpecialistInputError,
)
from app.print_tuning.schemas import (
    GammaProposalResponse, GammaReviewResponse, GrayLevelsReviewResponse, StepsReviewResponse,
)
from app.print_tuning.grayscale_render import validate_mapping
from app.print_tuning.validation import (
    Response, Stage, TuningValidationError, ValidationContext,
    get_candidate_correspondence, validate_context,
)
from app.workflow.tuning import (
    TuningSessionError, build_tuning_context, load_tuning_evaluation, load_tuning_session,
)

_MODELS = {"propose_gamma": GammaProposalResponse, "review_gamma": GammaReviewResponse,
           "review_gray_levels": GrayLevelsReviewResponse, "review_steps": StepsReviewResponse}
_STAGES = tuple(_MODELS)


class TuningEvidenceError(SpecialistInputError):
    """Invalid recorded application evidence; do not retry this as a model error."""


@dataclass(frozen=True)
class TuningInputs:
    """Direct specialist arguments plus the evaluation token needed for acceptance."""

    context: ValidationContext
    evidence: SpecialistEvidence
    prior_decisions: tuple[Response, ...]
    paper_color: str | None
    evaluation_record_id: str | None = None


def _need(condition: bool, message: str) -> None:
    if not condition:
        raise TuningEvidenceError(message)


def _object(value: Any, where: str) -> dict:
    _need(isinstance(value, dict), f"{where}: object required")
    return value


def _records(value: Any, where: str) -> list[dict]:
    _need(isinstance(value, list) and all(isinstance(v, dict) for v in value), f"{where}: list of objects required")
    return value


def _path(value: Any, folder: Path, where: str) -> Path:
    _need(isinstance(value, str) and bool(value.strip()), f"{where}: nonblank artifact path required")
    path = Path(value)
    return (path if path.is_absolute() else folder / path).resolve()


def _issues(value: Any, where: str, page_numbers: set[int]) -> list[str]:
    """Keep structured issue context, including null reasons, as readable JSON."""
    _need(isinstance(value, list), f"{where}: issue/warning list required")
    result = []
    for issue in value:
        if isinstance(issue, str):
            _need(bool(issue.strip()), f"{where}: blank issue")
            result.append(f"{where}: {issue}")
        else:
            _object(issue, where)
            _need(bool(issue), f"{where}: empty issue record")
            if "page_number" in issue:
                _need(type(issue["page_number"]) is int and issue["page_number"] in page_numbers,
                      f"{where}: issue refers to an unknown page")
            result.append(f"{where}: {json.dumps(issue, ensure_ascii=False, sort_keys=True, allow_nan=False)}")
    return result


def _history(session: dict, stage: Stage) -> tuple[Response, ...]:
    found = {}
    for decision in _records(session.get("decisions"), "session.decisions"):
        previous = decision.get("stage")
        _need(isinstance(previous, str) and previous in _MODELS, "History contains an unknown decision stage")
        if _STAGES.index(previous) >= _STAGES.index(stage):
            continue
        data = _object(decision.get("response"), "historical response")
        if data.get("assessment", "usable") != "usable":
            continue
        _need(previous not in found and data.get("stage") == previous, "Conflicting accepted decision history")
        try:
            found[previous] = _MODELS[previous].model_validate(data)
        except ValidationError as error:
            raise TuningEvidenceError(f"Invalid accepted {previous} history: {error}") from None
    expected = _STAGES[:_STAGES.index(stage)]
    _need(set(found) == set(expected), "Earlier accepted specialist decisions are missing")
    return tuple(found[s] for s in expected)


def _previews(session: dict, root: Path, review: bool) -> tuple[list[ImageAttachment], list[MissingImage]]:
    metadata = _object(session.get("previews"), "session.previews")
    entries = _records(metadata.get("previews"), "session.previews.previews")
    expected = {(e["experiment_id"], panel) for e in session["selected_experiments"] for panel in ("Initial", "Final")}
    seen, images, missing = set(), [], []
    for entry in entries:
        experiment, panel = entry.get("experiment_id"), entry.get("panel")
        _need(isinstance(experiment, str) and isinstance(panel, str), "Preview requires experiment_id and panel strings")
        key = (experiment, panel)
        _need(key in expected and key not in seen, f"Unknown or duplicate source preview: {key}")
        seen.add(key)
        label = ImageLabel("source_preview", experiment, panel)
        path = _path(entry.get("path"), root, f"preview {key}")
        if not path.is_file():
            _need(review, f"Required source preview is missing: {experiment}/{panel}")
            missing.append(MissingImage(label, "Saved source preview is missing"))
            continue
        if "sha256" in entry:
            try:
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
            except OSError as error:
                raise TuningEvidenceError(f"Cannot read source preview {experiment}/{panel}: {error}") from None
            _need(digest == entry["sha256"], f"Source preview changed: {experiment}/{panel}")
        images.append(ImageAttachment(label, path))
    _need(seen == expected, "Saved preview metadata must identify all eight Initial/Final source previews")
    return images, missing


def _review_evidence(session: dict, context: ValidationContext, result: dict, evaluation: dict | None,
                     root: Path, images: list[ImageAttachment], missing: list[MissingImage]) -> SpecialistEvidence:
    manifest = context.reviewed_manifest
    correspondence = get_candidate_correspondence(context)
    identities = {(c.item_id, c.variation_id): c for c in correspondence}
    items = {i["item_id"]: i for i in manifest["items"]}
    mappings = {(item, v["variation_id"]): v for item, variations in manifest["variations"].items() for v in variations}
    locations = {}
    page_numbers = set()
    for page in _records(manifest.get("pages"), "manifest.pages"):
        number = page.get("page_number")
        _need(type(number) is int and number not in page_numbers, "Duplicate/invalid manifest page_number")
        page_numbers.add(number)
        for plot in _records(_object(page.get("regions"), "manifest.regions").get("plots"), "manifest.plots"):
            key = (plot.get("item_id"), plot.get("variation_id"), plot.get("panel"))
            _need(all(isinstance(v, str) for v in key), "Invalid manifest plot identity")
            _need(key not in locations and key[:2] in identities, "Ambiguous/unknown manifest plot identity")
            locations[key] = number
    expected = {}
    for identity in correspondence:
        for panel in _records(items[identity.item_id].get("panels"), "manifest.panels"):
            name = panel.get("name")
            _need(isinstance(name, str), "Manifest panel name required")
            key = (identity.item_id, identity.variation_id, name)
            _need(key in locations and key not in expected, "Missing/duplicate candidate panel in manifest plots")
            expected[key] = ImageLabel("scanned_candidate", identity.experiment_id, name,
                                       identity.candidate_id, identity.variation_id, locations[key])
    quality = _object(result.get("scan_quality"), "scan_quality")
    issues = _issues(quality.get("issues"), "scan_quality", page_numbers)
    measurements, pages, printer, supplied, seen = [], [], None, set(), set()
    measurement_docs, scan_files = None, ()
    if evaluation is not None:
        _need(evaluation.get("schema_version") == "scan_eval/2", "Expected scan_eval/2 evaluation")
        _need(evaluation.get("tested_parameter") == manifest["tested_parameter"], "Evaluation tested_parameter differs from manifest")
        folder = _path(result.get("evaluation_path"), root, "evaluation_path").parent
        for page in _records(evaluation.get("pages"), "evaluation.pages"):
            number = page.get("page_number")
            _need(type(number) is int and number in page_numbers and number not in seen, "Duplicate/unknown evaluation page")
            seen.add(number)
            manifest_page = next(p for p in manifest["pages"] if p["page_number"] == number)
            if "page_code_expected" in page:
                _need(page["page_code_expected"] == manifest_page["page_code"], "Evaluation page code conflicts with manifest")
            pages.append(deepcopy(page))
            issues.extend(_issues(page.get("issues", []), f"page {number}", page_numbers))
        printer = deepcopy(_object(evaluation.get("printer_response"), "evaluation.printer_response"))
        measurement_docs = deepcopy(_object(evaluation.get("measurement_docs"), "evaluation.measurement_docs"))
        scan_files = tuple(deepcopy(_records(evaluation.get("scan_files"), "evaluation.scan_files")))
        candidates = set()
        for candidate in _records(evaluation.get("candidates"), "evaluation.candidates"):
            item, variation = candidate.get("experiment_id"), candidate.get("candidate_id")
            _need(isinstance(item, str) and isinstance(variation, str), "Scan candidate requires item/variation ID strings")
            key = (item, variation)
            _need(key in identities and key not in candidates, f"Duplicate/unknown scan candidate: {key}")
            candidates.add(key)
            identity = identities[key]
            reference = mappings[key]
            try:
                validate_mapping(candidate.get("mapping"), context.config)
            except ValueError as error:
                raise TuningEvidenceError(f"Invalid scanned mapping for {key}: {error}") from None
            _need(candidate.get("mapping") == reference["mapping"] and candidate.get("role") == reference["role"],
                  f"Scan candidate mapping/role conflicts with manifest: {key}")
            number = candidate.get("page_number")
            _need(type(number) is int and number in seen, f"Candidate {key} has no corresponding evaluation page")
            _need(all(locations[k] == number for k in expected if k[:2] == key), f"Candidate {key} is on a different manifest page")
            where = f"{identity.experiment_id}/{identity.candidate_id} page {number}"
            warnings = _issues(candidate.get("warnings", []), where, page_numbers)
            values = {**deepcopy(_object(candidate.get("measurements"), where + ".measurements")),
                      "mapping": deepcopy(candidate["mapping"]), "page_number": number,
                      "notes": deepcopy(_object(candidate.get("notes", {}), where + ".notes"))}
            measurements.append(CandidateMeasurement(identity.experiment_id, identity.candidate_id, variation,
                                                     values, warnings=tuple(warnings)))
            issues.extend(warnings)
            tile_label = ImageLabel("scanned_candidate", identity.experiment_id, None,
                                    identity.candidate_id, variation, number)
            if candidate.get("tile_path") is None:
                missing.append(MissingImage(tile_label, "Evaluation supplies no whole-tile PNG"))
            else:
                tile_path = _path(candidate["tile_path"], folder, where + ".tile_path")
                if tile_path.is_file():
                    images.append(ImageAttachment(tile_label, tile_path))
                else:
                    missing.append(MissingImage(tile_label, "Recorded whole-tile PNG is missing"))
            for panel in _records(candidate.get("panels"), where + ".panels"):
                name = panel.get("panel")
                _need(isinstance(name, str), where + ": panel name required")
                panel_key = (*key, name)
                _need(panel_key in expected and panel_key not in supplied, f"Duplicate/unknown scan panel: {panel_key}")
                supplied.add(panel_key)
                _need(type(panel.get("page_number", number)) is int and panel.get("page_number", number) == number,
                      f"Panel {panel_key} page conflicts with candidate")
                measurements.append(CandidateMeasurement(
                    identity.experiment_id, identity.candidate_id, variation,
                    {**deepcopy(_object(panel.get("measurements"), where + f"/{name}.measurements")),
                     "page_number": number,
                     "notes": deepcopy(_object(panel.get("notes", {}), where + f"/{name}.notes"))}, panel=name))
                if panel.get("png") is None:
                    missing.append(MissingImage(expected[panel_key], "Evaluation supplies no panel PNG"))
                else:
                    path = _path(panel["png"], folder, where + f"/{name}.png")
                    if path.is_file():
                        images.append(ImageAttachment(expected[panel_key], path))
                    else:
                        missing.append(MissingImage(expected[panel_key], "Recorded panel PNG is missing"))
    for key, label in expected.items():
        if key not in supplied:
            missing.append(MissingImage(label, "Recorded rescan has no crop" if evaluation is None else "Evaluation omits this candidate/panel"))
    incomplete = bool(missing) or evaluation is None or len(pages) != len(page_numbers) or not quality.get("usable")
    reason = "Recorded scan evidence is incomplete or unusable; missing crops and quality issues are listed." if incomplete else None
    if evaluation is None:
        _need(result.get("status") == "needs_rescan" and quality.get("usable") is False and bool(issues),
              "Review needs an evaluation file or an explained needs_rescan outcome")
    entry = session["rounds"][str(session["active_round"])]
    prior = _object(entry.get("reference_settings"), "round.reference_settings")
    expected_references = {c.experiment_id: mappings[c.item_id, c.variation_id]["mapping"]
                           for c in correspondence if c.candidate_id == "reference"}
    _need(prior == expected_references, "Prior settings differ from the references printed before this round")
    return SpecialistEvidence(
        normalization_limits={e["experiment_id"]: deepcopy(e["limits"]) for e in session["selected_experiments"]},
        images=tuple(images), missing_images=tuple(missing), candidate_measurements=tuple(measurements),
        page_measurements=tuple(pages), printer_response=printer, incomplete_evidence_reason=reason,
        issues=tuple(issues), trial_id=manifest["trial_id"], round_id=manifest["round_id"],
        calibration_session_id=session["session_id"], prior_settings=deepcopy(prior),
        measurement_docs=measurement_docs, scan_files=scan_files, scan_quality=deepcopy(quality),
    )


def prepare_tuning_inputs(session_path: str | Path, stage: Stage) -> TuningInputs:
    """Prepare one active decision from authoritative artifacts, without writes.

    An awaiting_scan phase is reviewable only when a recorded evaluation/rescan
    exists. Operational scan failures, stale trials, and corrupt JSON are input
    errors; individual absent scan crops are explicit MissingImage entries.
    """
    _need(isinstance(stage, str) and stage in _MODELS, "Unknown specialist stage")
    path = Path(session_path).resolve()
    try:
        session = load_tuning_session(path)
        _need(session["status"] == "active", "Only an active tuning session can prepare specialist inputs")
        phase = session["phase"]
        _need(phase == "awaiting_gamma_proposal" if stage == "propose_gamma" else phase in ("awaiting_review", "awaiting_scan"),
              f"Stage {stage} is incompatible with phase {phase}")
        context = build_tuning_context(path, stage)
        validate_context(context)
        history = _history(session, stage)
        images, missing = _previews(session, path.parent, stage != "propose_gamma")
        evaluation_id = None
        if stage == "propose_gamma":
            _need(session["active_trial_id"] is None, "Initial proposal cannot review an active physical trial")
            _need(session["previews"].get("reference_mappings") == context.reference_mappings,
                  "Preview reference mappings differ from initial references")
            evidence = SpecialistEvidence(
                {e["experiment_id"]: deepcopy(e["limits"]) for e in session["selected_experiments"]},
                tuple(images), prior_settings=deepcopy(context.reference_mappings))
        else:
            evaluation_id, result, evaluation = load_tuning_evaluation(path)
            _need(result.get("status") in ("ok", "needs_rescan"), "Operational scan error must be resolved before specialist review")
            evidence = _review_evidence(session, context, result, evaluation, path.parent, images, missing)
    except (TuningSessionError, TuningValidationError) as error:
        raise TuningEvidenceError(str(error)) from None
    paper = session.get("paper_color")
    _need(paper is None or isinstance(paper, str) and bool(paper.strip()), "Invalid saved paper_color")
    return TuningInputs(context, evidence, history, paper, evaluation_id)
