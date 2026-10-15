"""Executor tools that connect rendering, trial sheets and scan reading to the tuning graph.

The executor is the only agent with tools. These tools read run_dir/tuning_session.json (the
graph writes it) and only return results: they never write the session file, never choose
winners, never print and never move to the next round. The executor never passes the session
or manifest paths; both come from the run folder the tools were made for.

    create_tuning_tools(run_dir)        -> [make_trial_sheet, read_scan]
    make_trial_sheet(round_id)          draws the round's trial sheet from the session's proposal
    read_scan(scan_paths)               measures the scan of the session's active trial
    prepare_specialist_previews(...)    plain function, before round 1: previews and limits

Session keys read (all in _session_inputs, the only place to change if the graph's names differ):
    contract_version        1
    session_id              becomes the trial sheet's calibration_session_id
    run_id
    selection_path          the selection JSON written by app.experiments.selection.build_selection
    config_snapshot         the tuning config dict, frozen for the session
    selected_experiments    [{experiment_id, limits}] in the frozen order
    accepted_settings       experiment_id -> complete mapping accepted BEFORE the round; missing or
                            empty for round 1, which then uses default_mapping(config) for every experiment
    rounds                  {"1", "2", "3"} -> {proposal, trial_result}: proposal is the validated
                            specialist response that proposed THIS round's tests (as a dict), and
                            trial_result is what make_trial_sheet returned once the round was drawn
    active_trial_id         the trial whose scan read_scan measures

Candidates are listed reference first, then test_1 ... test_5, with variation_id equal to the
candidate id: app/print_tuning/validation.py maps candidates by role and by test order.
"""
import json
import logging
import math
from pathlib import Path
from typing import Any, get_args

import numpy as np
from langchain_core.tools import BaseTool, tool
from PIL import Image, ImageDraw, ImageFont
from pydantic import BaseModel, ValidationError

from app.print_tuning.grayscale_render import (MappingError, apply_mapping, default_mapping, mapping_label,
                                               validate_config, validate_mapping)
from app.print_tuning.scan_evaluation import ScanError, evaluate_scan
from app.print_tuning.schemas import (GammaProposalResponse, GammaReviewResponse, GrayLevelsReviewResponse,
                                      TestCandidateId)
from app.print_tuning.trial_sheet import (TrialSheetError, _check_marks, _new_limits, create_trial_sheet,
                                          load_panel_data)

CONTRACT_VERSION = 1
SESSION_FILE = "tuning_session.json"
OUTPUT_FOLDER = "print_tuning"
TEST_IDS = get_args(TestCandidateId)
PREVIEW_MIN_PX = 256                  # previews are enlarged (nearest neighbour) to at least this
PREVIEW_FONT_PX = 13
ENDPOINT_ALIASES = {"gray_min": "dark_level", "gray_max": "light_level"}

logger = logging.getLogger(__name__)


class _ToolError(Exception):
    """A problem reported to the executor as {status: "error", error_code, message}."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


# ---------------------------------------------------------------- session

def _load_session(run_dir: Path) -> dict:
    path = run_dir / SESSION_FILE
    try:
        session = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise _ToolError("session_invalid", f"cannot read {path}: {error}") from error
    if not isinstance(session, dict):
        raise _ToolError("session_invalid", f"{path} must hold a JSON object")
    return session


def _session_inputs(session: dict, round_id: int | None) -> dict:
    """Every value the tools take from the session, checked; the only place that knows its key names.

    round_id selects the round's proposal, accepted settings and previous manifest; None (for
    previews and scans) skips them. Raises _ToolError("session_invalid" or "proposal_missing").
    """
    def need(condition, message):
        if not condition:
            raise _ToolError("session_invalid", f"tuning session: {message}")

    need(session.get("contract_version") == CONTRACT_VERSION,
         f"contract_version must be {CONTRACT_VERSION}, got {session.get('contract_version')!r}")
    for key in ("session_id", "run_id", "selection_path"):
        need(isinstance(session.get(key), str) and session[key].strip(), f"{key} must be a non-empty string")
    config = session.get("config_snapshot")
    try:
        validate_config(config)
    except (ValueError, KeyError, TypeError, AttributeError) as error:
        raise _ToolError("session_invalid", f"tuning session: config_snapshot is not a valid tuning config: "
                                            f"{error}") from error

    selected = session.get("selected_experiments")
    need(isinstance(selected, list) and selected and all(isinstance(e, dict) for e in selected),
         "selected_experiments must be a non-empty list of {experiment_id, limits}")
    experiment_ids = [entry.get("experiment_id") for entry in selected]
    need(all(isinstance(e, str) and e for e in experiment_ids) and len(set(experiment_ids)) == len(experiment_ids),
         f"selected_experiments need distinct experiment_ids, got {experiment_ids}")
    limits = {entry["experiment_id"]: entry.get("limits") for entry in selected}

    rounds = session.get("rounds") or {}
    need(isinstance(rounds, dict), "rounds must be a dict keyed by round number")
    round_of = lambda number: rounds.get(str(number)) or rounds.get(number) or {}

    # the active trial: the round whose trial_result has active_trial_id
    active, active_id = None, session.get("active_trial_id")
    if active_id:
        for key, entry in rounds.items():
            result = entry.get("trial_result") if isinstance(entry, dict) else None
            if isinstance(result, dict) and result.get("trial_id") == active_id:
                need(str(key).isdigit(), f"round key {key!r} must be a round number")
                need(isinstance(result.get("manifest_path"), str), f"round {key}: trial_result has no manifest_path")
                active = {"trial_id": active_id, "round_id": int(key), "manifest_path": result["manifest_path"]}
        need(active is not None, f"active_trial_id {active_id} is not the trial_result of any round")

    inputs = {"session_id": session["session_id"], "run_id": session["run_id"],
              "selection_path": session["selection_path"], "config": config, "experiment_ids": experiment_ids,
              "limits": limits, "active_trial": active}
    if round_id is None:
        return inputs

    accepted = session.get("accepted_settings") or {}
    need(isinstance(accepted, dict), "accepted_settings must be experiment_id -> mapping")
    if not accepted and round_id == 1:
        accepted = {experiment_id: default_mapping(config) for experiment_id in experiment_ids}
    missing = [e for e in experiment_ids if not isinstance(accepted.get(e), dict)]
    need(not missing, f"accepted_settings has no mapping for {missing} before round {round_id}")

    entry = round_of(round_id)
    need(isinstance(entry, dict), f"rounds[{round_id}] must be a dict")
    proposal = entry.get("proposal")
    if isinstance(proposal, BaseModel):
        proposal = proposal.model_dump()
    if not proposal:
        raise _ToolError("proposal_missing", f"round {round_id} has no proposal in the tuning session yet")

    previous_manifest = None
    if round_id > 1:
        previous = round_of(round_id - 1).get("trial_result")
        need(isinstance(previous, dict) and isinstance(previous.get("manifest_path"), str),
             f"round {round_id - 1} has no trial_result with a manifest_path; draw that round first")
        previous_manifest = previous["manifest_path"]
    return inputs | {"accepted": {e: dict(accepted[e]) for e in experiment_ids}, "proposal": proposal,
                     "previous_manifest_path": previous_manifest}


def _load_selection(inputs: dict) -> tuple[dict, dict]:
    """(selection, experiment_id -> item_id); the selection's items must be the session's experiments in order."""
    path = Path(inputs["selection_path"])
    try:
        selection = json.loads(path.read_text(encoding="utf-8"))
        items = selection["items"]
        experiment_ids = [(item.get("source") or {}).get("experiment_id") or item["item_id"] for item in items]
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as error:
        raise _ToolError("session_invalid", f"cannot read the selection {path}: {error}") from error
    if experiment_ids != inputs["experiment_ids"]:
        raise _ToolError("session_invalid", f"the selection {path.name} holds {experiment_ids}, but the session "
                                            f"selected {inputs['experiment_ids']} (same order needed)")
    return selection, {experiment_id: item["item_id"] for experiment_id, item in zip(experiment_ids, items)}


# ---------------------------------------------------------------- proposals

def _parse(model: type[BaseModel], proposal: dict, round_id: int) -> BaseModel:
    try:
        response = model.model_validate(proposal)
    except ValidationError as error:
        problems = "; ".join(f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in error.errors())
        raise _ToolError("proposal_invalid", f"round {round_id} proposal is not a valid "
                                             f"{model.__name__}: {problems}") from error
    if getattr(response, "assessment", "usable") != "usable":
        raise _ToolError("proposal_invalid", f"round {round_id} proposal has assessment "
                                             f"{response.assessment!r}, so it proposes no tests")
    return response


def _by_candidate(tests: list, where: str) -> dict:
    """candidate_id -> test, checked to be exactly test_1 ... test_5."""
    found = [test.candidate_id for test in tests]
    if sorted(found) != sorted(TEST_IDS):
        raise _ToolError("proposal_invalid", f"{where}: needs each of {list(TEST_IDS)} once, got {found}")
    return {test.candidate_id: test for test in tests}


def _per_experiment(groups: list, experiment_ids: list[str], where: str) -> dict:
    found = [group.experiment_id for group in groups]
    if sorted(found) != sorted(experiment_ids):
        raise _ToolError("proposal_invalid", f"{where}: needs each of {experiment_ids} once, got {found}")
    return {group.experiment_id: _by_candidate(group.tests, f"{where}[{group.experiment_id}]") for group in groups}


def _endpoint_aliases(proposal: dict) -> dict:
    """The proposal with gray_min/gray_max in endpoint_tests renamed to dark_level/light_level."""
    tests = proposal.get("endpoint_tests")
    if not isinstance(tests, list):
        return proposal
    renamed = []
    for test in tests:
        if isinstance(test, dict):
            test = dict(test)
            for alias, key in ENDPOINT_ALIASES.items():
                if alias in test:
                    value = test.pop(alias)
                    if key in test and test[key] != value:
                        raise _ToolError("proposal_invalid", f"endpoint test {test.get('candidate_id')}: {alias} "
                                                             f"{value!r} differs from {key} {test[key]!r}")
                    test[key] = value
        renamed.append(test)
    return proposal | {"endpoint_tests": renamed}


def _round_tests(proposal: dict, round_id: int, experiment_ids: list[str]) -> dict:
    """experiment_id -> candidate_id -> (changed mapping values, reason), from this round's proposal."""
    if not isinstance(proposal, dict):
        raise _ToolError("proposal_invalid", f"round {round_id} proposal must be a dict")
    if round_id == 1:
        response = _parse(GammaProposalResponse, proposal, round_id)
        tests = _per_experiment(response.experiments, experiment_ids, "experiments")
        return {e: {c: ({"gamma": t.gamma}, t.reason) for c, t in by_id.items()} for e, by_id in tests.items()}
    if round_id == 2:
        response = _parse(GammaReviewResponse, _endpoint_aliases(proposal), round_id)
        shared = _by_candidate(response.endpoint_tests, "endpoint_tests")
        return {e: {c: ({"dark_level": t.dark_level, "light_level": t.light_level}, t.reason)
                    for c, t in shared.items()} for e in experiment_ids}
    if round_id == 3:
        response = _parse(GrayLevelsReviewResponse, proposal, round_id)
        tests = _per_experiment(response.step_tests, experiment_ids, "step_tests")
        return {e: {c: ({"steps": t.steps}, t.reason) for c, t in by_id.items()} for e, by_id in tests.items()}
    raise _ToolError("proposal_invalid", f"round_id must be 1, 2 or 3, got {round_id!r}")


def _variations(tests: dict, accepted: dict, item_ids: dict, config: dict) -> dict:
    """item_id -> [reference, test_1, ..., test_5]: the accepted mapping, then one changed value per test."""
    variations = {}
    for experiment_id, by_candidate in tests.items():
        reference = dict(accepted[experiment_id])
        entries = [{"variation_id": "reference", "role": "reference", "mapping": reference,
                    "note": "accepted settings so far"}]
        for candidate_id in TEST_IDS:
            changes, reason = by_candidate[candidate_id]
            mapping = reference | changes
            try:
                validate_mapping(mapping, config)
            except MappingError as error:
                raise _ToolError("proposal_invalid", f"experiment {experiment_id} {candidate_id}: {error}") from error
            entries.append({"variation_id": candidate_id, "role": "test", "mapping": mapping, "note": reason})
        variations[item_ids[experiment_id]] = entries
    return variations


# ---------------------------------------------------------------- tools

def _error(code: str, message: str, **fields) -> dict:
    return {"contract_version": CONTRACT_VERSION, "status": "error", **fields, "error_code": code, "message": message}


def _make_trial_sheet(run_dir: Path, round_id: int) -> dict:
    if isinstance(round_id, bool) or round_id not in (1, 2, 3):
        raise _ToolError("proposal_invalid", f"round_id must be 1, 2 or 3, got {round_id!r}")
    inputs = _session_inputs(_load_session(run_dir), round_id)
    selection, item_ids = _load_selection(inputs)
    tests = _round_tests(inputs["proposal"], round_id, inputs["experiment_ids"])
    variations = _variations(tests, inputs["accepted"], item_ids, inputs["config"])
    limits = {item_ids[e]: value for e, value in inputs["limits"].items() if value is not None}
    previous_record = None
    if inputs["previous_manifest_path"]:
        try:
            previous_record = json.loads(Path(inputs["previous_manifest_path"]).read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            raise _ToolError("session_invalid", f"cannot read round {round_id - 1}'s manifest "
                                                f"{inputs['previous_manifest_path']}: {error}") from error
    try:
        record = create_trial_sheet(selection, variations, round_id, out_dir=run_dir / OUTPUT_FOLDER,
                                    calibration_session_id=inputs["session_id"], config=inputs["config"],
                                    limits=limits, previous_record=previous_record)
    except (TrialSheetError, MappingError) as error:
        raise _ToolError("trial_sheet_failed", str(error)) from error
    return {"contract_version": CONTRACT_VERSION, "status": "ok", "session_id": inputs["session_id"],
            "trial_id": record["trial_id"], "round_id": record["round_id"], "pdf_path": record["pdf"]["path"],
            "pdf_sha256": record["pdf"]["sha256"], "manifest_path": record["manifest_path"],
            "page_count": record["pdf"]["pages"]}


def _read_scan(run_dir: Path, scan_paths: list[str]) -> dict:
    inputs = _session_inputs(_load_session(run_dir), None)
    active = inputs["active_trial"]
    if active is None:
        raise _ToolError("no_active_trial", "the tuning session has no active trial; draw a trial sheet first")
    ids = {"session_id": inputs["session_id"], "trial_id": active["trial_id"], "round_id": active["round_id"]}
    paths = [str(path if path.is_absolute() else run_dir / path) for path in map(Path, scan_paths)]
    try:
        result = evaluate_scan(paths, active["manifest_path"], out_dir=run_dir / OUTPUT_FOLDER)
    except ScanError as error:
        if error.needs_rescan:
            return {"contract_version": CONTRACT_VERSION, "status": "needs_rescan", **ids,
                    "error_code": error.code, "message": str(error),
                    "scan_quality": {"usable": False, "issues": [str(error)]}}
        return _error(error.code, str(error), **ids)
    if result["trial_id"] != active["trial_id"]:
        return _error("session_invalid", f"the active trial's manifest is for trial {result['trial_id']}, "
                                         f"not {active['trial_id']}", **ids)
    return {"contract_version": CONTRACT_VERSION, "status": "ok", **ids,
            "evaluation_path": result["evaluation_path"], "scan_quality": result["scan_quality"],
            "pages": result["pages"], "candidates": result["candidates"]}


def create_tuning_tools(run_dir: str | Path) -> list[BaseTool]:
    """Create the make_trial_sheet and read_scan tools tied to this run's folder and tuning session."""
    directory = Path(run_dir).resolve()

    @tool
    def make_trial_sheet(round_id: int) -> dict:
        """Draw the printable trial sheet (PDF) for tuning round 1, 2 or 3.

        The tests come from the round's validated proposal in the tuning session: round 1 tests
        gamma, round 2 the shared dark/light gray levels, round 3 the number of gray steps. Each
        of the 4 selected experiments gets 6 candidates: reference (the accepted settings) and
        test_1 to test_5. Returns the trial_id, the PDF path and its sha256, the manifest path
        and the page count. It does not print the sheet.
        """
        try:
            return _make_trial_sheet(directory, round_id)
        except _ToolError as error:
            return _error(error.code, str(error), round_id=round_id)
        except Exception as error:
            logger.exception("make_trial_sheet(%r) failed", round_id)
            return _error("internal_error", f"{type(error).__name__}: {error}", round_id=round_id)

    @tool
    def read_scan(scan_paths: list[str]) -> dict:
        """Measure the scan of the active trial sheet.

        scan_paths is one PDF holding every page, or one image (PNG or JPG) per page; relative
        paths are taken from the run folder. The trial is the session's active trial: no
        manifest is chosen here. Returns status "ok" with the scan quality, page records and
        one measured record per candidate (no ranking), "needs_rescan" with the problem when a
        new scan would fix it, or "error".
        """
        try:
            return _read_scan(directory, scan_paths)
        except _ToolError as error:
            return _error(error.code, str(error))
        except Exception as error:
            logger.exception("read_scan(%r) failed", scan_paths)
            return _error("internal_error", f"{type(error).__name__}: {error}")

    return [make_trial_sheet, read_scan]


# ---------------------------------------------------------------- previews (not a tool)

def _font() -> ImageFont.ImageFont:
    """DejaVu Sans from matplotlib (it has γ, · and –), else Pillow's default font."""
    try:
        from matplotlib import font_manager
        return ImageFont.truetype(font_manager.findfont("DejaVu Sans"), PREVIEW_FONT_PX)
    except Exception:
        return ImageFont.load_default()


def _save_preview(grays: np.ndarray, lines: list[str], path: Path) -> None:
    """Grays 0..1 enlarged (nearest neighbour) to at least PREVIEW_MIN_PX, with text lines under it."""
    scale = max(1, math.ceil(PREVIEW_MIN_PX / min(grays.shape)))
    picture = Image.fromarray(np.clip(np.round(grays * 255), 0, 255).astype(np.uint8), "L")
    picture = picture.resize((grays.shape[1] * scale, grays.shape[0] * scale), Image.NEAREST)
    font, pad = _font(), 4
    widths = [math.ceil(font.getlength(line)) for line in lines]
    line_h = PREVIEW_FONT_PX + 4
    canvas = Image.new("L", (max(picture.width, max(widths) + 2 * pad), picture.height + pad + line_h * len(lines)),
                       255)
    canvas.paste(picture, (0, 0))
    draw = ImageDraw.Draw(canvas)
    for k, line in enumerate(lines):
        draw.text((pad, picture.height + pad + k * line_h), line, fill=0, font=font)
    canvas.save(path)


def prepare_specialist_previews(session_path, out_dir) -> dict[str, Any]:
    """Before round 1: one grayscale preview PNG per selected experiment and panel, and the limits.

    The limits are computed exactly as create_trial_sheet computes them (percentiles, with hi raised
    to the brightest cell inside a box mark), so they equal the limits later saved in the manifest.
    Each panel is drawn with the config's default mapping. Raises ValueError on a bad session or
    TrialSheetError on bad panel data; the caller saves the returned limits into the session.
    """
    session_path = Path(session_path)
    try:
        session = json.loads(session_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise ValueError(f"cannot read the tuning session {session_path}: {error}") from error
    try:
        inputs = _session_inputs(session, None)
        selection, item_ids = _load_selection(inputs)
    except _ToolError as error:
        raise ValueError(str(error)) from error
    config, mapping = inputs["config"], default_mapping(inputs["config"])
    items = {item["item_id"]: item for item in selection["items"]}
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    previews, limits = [], {}
    for experiment_id in inputs["experiment_ids"]:
        item = items[item_ids[experiment_id]]
        arrays, panels = [], []
        for panel in item["panels"]:          # the same loading and mark checks as trial_sheet._load_items
            where = f"item {item['item_id']} panel {panel['name']}"
            data = load_panel_data(panel["data_path"], where)
            marks = _check_marks(item.get("marks"), where, data.shape) + \
                _check_marks(panel.get("marks"), where, data.shape)
            arrays.append(data)
            panels.append({"name": panel["name"], "marks": marks})
        limits[experiment_id] = _new_limits(arrays, panels, config)
        for panel, data in zip(panels, arrays):
            path = out_dir / f"preview_{item['item_id']}_{panel['name']}.png"
            _save_preview(apply_mapping(data, limits[experiment_id], mapping),
                          [experiment_id, panel["name"], mapping_label(mapping)], path)
            previews.append({"experiment_id": experiment_id, "panel": panel["name"], "path": str(path)})
    return {"contract_version": CONTRACT_VERSION, "previews": previews, "normalization_limits": limits,
            "reference_mappings": {experiment_id: dict(mapping) for experiment_id in inputs["experiment_ids"]}}
