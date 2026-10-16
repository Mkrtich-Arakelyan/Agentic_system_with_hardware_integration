"""Application-owned tuning sessions, using the existing tuning_tools contract.

Callers MUST serialize operations for each session. Atomic replacement prevents
partial JSON, not concurrent lost updates. This is not a LangGraph checkpointer.
Only initialization calls the existing preview helper; recorders execute no
models, scans, printing, or cancellation. Paths refer to artifacts, not pixels.
"""

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any
from uuid import uuid4

from app.agents.scientist import ExperimentSelection
from app.experiments.selection import selection_from_scientist
from app.print_tuning.grayscale_render import load_config, starting_mapping
from app.print_tuning.tuning_tools import (
    OUTPUT_FOLDER, SESSION_FILE, _ToolError, _load_selection, _round_tests,
    _session_inputs, _variations, prepare_specialist_previews,
    summarize_scan_result,
)
from app.print_tuning.validation import (
    Response, Stage, TuningValidationError, ValidationContext,
    get_candidate_correspondence, resolve_review_winners, validate_context, validate_response,
)


class TuningSessionError(ValueError):
    """Invalid session input, incompatible artifacts, or out-of-order operation."""


_REVIEW_STAGES = {1: "review_gamma", 2: "review_gray_levels", 3: "review_steps"}
_RETRY_LIMITS = {"rescans": 3, "reprints": 1, "trial_regenerations": 2}


def _need(condition: bool, message: str) -> None:
    if not condition:
        raise TuningSessionError(message)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _clone(value: Any) -> Any:
    try:
        return json.loads(json.dumps(value, allow_nan=False))
    except (TypeError, ValueError, OverflowError) as error:
        raise TuningSessionError(f"Session metadata must be finite JSON: {error}") from None


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(_clone(value), sort_keys=True).encode()).hexdigest()


def _read(path: str | Path) -> dict:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
        _need(isinstance(value, dict), f"{path}: JSON object required")
        return _clone(value)
    except (OSError, ValueError) as error:
        raise TuningSessionError(f"Cannot load {path}: {error}") from None


def _sha(path: str | Path) -> str:
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except OSError as error:
        raise TuningSessionError(f"Cannot read artifact {path}: {error}") from None


def _write(path: Path, session: dict) -> dict:
    """Serialize first, then fsync a same-directory temporary file and replace."""
    session = _clone(session)
    session["revision"] += 1
    session["updated_at"] = _now()
    encoded = json.dumps(session, indent=2, ensure_ascii=False, allow_nan=False)
    temporary = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=path.name + ".", suffix=".tmp", delete=False) as file:
            temporary = Path(file.name)
            file.write(encoded)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, path)
    except OSError as error:
        raise TuningSessionError(f"Atomic session write failed: {error}") from None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return session


def _ids(session: dict) -> list[str]:
    return [e["experiment_id"] for e in session["selected_experiments"]]


def _selection(session: dict) -> tuple[dict, dict]:
    try:
        _need(_sha(session["selection_path"]) == session["selection_sha256"], "Packaged selection changed after freezing")
        selection, association = _load_selection(_session_inputs(session, None))
        _need(all(selection.get(k) == session[k] for k in ("run_id", "selection_id", "selection_version")), "Packaged selection identity differs")
        return selection, association
    except (_ToolError, KeyError, TypeError) as error:
        raise TuningSessionError(f"Invalid packaged selection/session: {error}") from None


def load_tuning_session(session_path: str | Path, *, allow_incomplete: bool = False) -> dict:
    """Load the authoritative record; never regenerate limits or reset progress."""
    session = _read(session_path)
    try:
        _need(session["contract_version"] == 1, "Unsupported tuning session contract")
        ready = session["initialization"]["status"] == "complete"
        _need(ready or allow_incomplete, "Tuning initialization is incomplete; retry_initialization=True is required deliberately")
        if ready:
            _selection(session)
            validate_context(ValidationContext("propose_gamma", _ids(session), session["config_snapshot"],
                                               reference_mappings=session["accepted_settings"]))
            _need(set(session["rounds"]) == {"1", "2", "3"}, "Session must retain exactly three rounds")
            _need(session["status"] in ("active", "blocked", "completed", "stopped", "error"), "Unknown session status")
            _need({e["experiment_id"]: e["limits"] for e in session["selected_experiments"]} == session["previews"]["normalization_limits"], "Frozen normalization limits changed")
            if session["status"] == "completed":
                _need(session["phase"] == "completed" and session["active_round"] == 3
                      and session["final_settings"] == session["accepted_settings"], "Incomplete final settings cannot be marked completed")
            else:
                _need(session["final_settings"] is None, "Final settings may appear only on completion")
            for entry in session["rounds"].values():
                for name, count in entry["retries"].items():
                    _need(type(count) is int and 0 <= count <= session["retry_limits"][name], f"Invalid retry counter {name}")
            for experiment in session["selected_experiments"]:
                limits = experiment["limits"]
                _need(isinstance(limits, dict) and limits.get("hi", 0) > limits.get("lo", 0), "Fixed limits missing")
    except (KeyError, TypeError, TuningValidationError, _ToolError) as error:
        raise TuningSessionError(f"Invalid tuning session: {error}") from None
    return session


def initialize_tuning_session(
    run_dir: str | Path, accepted_selection: dict, results: list[dict], *,
    config: dict | None = None, run_id: str | None = None,
    paper_session: dict | None = None, target_position: tuple[int, int] | None = None,
    retry_initialization: bool = False, paper_color: str | None = None,
) -> dict:
    """Package a frozen scientist selection, bootstrap, then freeze real previews.

    Existing compatible ready sessions are reused without preview/array reads.
    Incomplete initialization needs an explicit retry (maximum three attempts).
    An omitted paper session uses {paper_session_id: default, description:
    unspecified paper; no paper-color assumption}. A supplied paper record must
    contain a nonblank paper_session_id. Caller-supplied objects are not mutated.
    Paper color uses the renderer's existing starting_mapping presets. It may be
    supplied here or in paper_session; conflicting values are rejected. Omitting
    both paper inputs on resume preserves the frozen paper metadata and color.
    """
    root = Path(run_dir).resolve()
    path = root / SESSION_FILE
    paper = _clone(paper_session) if paper_session is not None else {
        "paper_session_id": "default", "description": "Unspecified paper; no paper-color assumption"}
    _need(isinstance(paper, dict) and isinstance(paper.get("paper_session_id"), str)
          and bool(paper["paper_session_id"].strip()), "paper_session.paper_session_id required")
    if "paper_color" in paper:
        _need(paper_color is None or paper_color == paper["paper_color"], "Conflicting paper colors supplied")
        paper_color = paper["paper_color"]
    _need(paper_color is None or isinstance(paper_color, str) and bool(paper_color.strip()),
          "paper_color must be a nonblank string or None")
    selection = _clone(accepted_selection)
    try:
        entries = selection["selections"]
        _need(selection.get("status", "ready") == "ready" and not selection.get("proposed_experiments"), "Only an accepted ready scientific selection may initialize tuning")
        parsed = [ExperimentSelection.model_validate(e) for e in entries]
        ids = [e.experiment_id for e in parsed]
        _need(all(e.strip() for e in ids), "Scientific IDs must be nonblank")
        _need(all(getattr(e, field).strip() for e in parsed for field in
                  ("validity", "comparability", "relevance", "information_gained")), "Scientific selection reasons must be nonblank")
        _need(isinstance(selection["group_reason"], str) and bool(selection["group_reason"].strip()), "Scientific group_reason required")
        frozen_config = _clone(config if config is not None else load_config())
        defaults = {e: dict(starting_mapping(frozen_config, paper_color)[0]) for e in ids}
        validate_context(ValidationContext("propose_gamma", ids, frozen_config, reference_mappings=defaults))
    except (ValueError, KeyError, TypeError, AttributeError) as error:
        raise TuningSessionError(f"Cannot initialize from scientific selection/configuration: {error}") from None
    run_id = run_id or root.name
    _need(isinstance(run_id, str) and bool(run_id.strip()), "Nonblank run_id required")
    if path.exists():
        session = load_tuning_session(path, allow_incomplete=True)
        _need(session["accepted_selection"] == selection and session["config_snapshot"] == frozen_config
              and session["run_id"] == run_id, "Existing session has incompatible frozen selection, run, or configuration")
        if paper_session is not None:
            _need(session["paper_session"] == paper, "Existing session uses a different paper session")
        if paper_session is None and paper_color is None:
            paper_color = session.get("paper_color")
        _need(session.get("paper_color") == paper_color, "Existing session uses a different paper color; frozen previews cannot be changed")
        defaults = {e: dict(starting_mapping(frozen_config, paper_color)[0]) for e in ids}
        if session["initialization"]["status"] == "complete":
            return load_tuning_session(path)
        _need(retry_initialization, "Initialization incomplete; deliberate retry_initialization=True required")
        _need(len(session["initialization"]["attempts"]) < 3, "Initialization retry limit reached")
    else:
        session = {
            "contract_version": 1, "session_id": str(uuid4()), "run_id": run_id,
            "session_path": str(path), "selection_path": "", "selection_sha256": None,
            "selection_id": None, "selection_version": None, "accepted_selection": selection,
            "config_snapshot": frozen_config, "paper_session_id": paper["paper_session_id"], "paper_session": paper,
            "paper_color": paper_color,
            "selected_experiments": [{"experiment_id": e, "limits": None} for e in ids],
            "accepted_settings": defaults, "final_settings": None, "previews": None,
            "status": "initializing", "phase": "initializing", "active_round": 1, "active_trial_id": None,
            "retry_limits": dict(_RETRY_LIMITS), "initialization": {"status": "initializing", "attempts": []},
            "rounds": {str(n): {"proposal": None, "trial_result": None, "trials": [], "evaluations": [],
                                "active_evaluation_id": None, "reviews": [], "review": None, "winners": None,
                                "reference_settings": None, "approvals": [], "print_attempts": [],
                                "retries": {key: 0 for key in _RETRY_LIMITS}} for n in (1, 2, 3)},
            "decisions": [], "outcomes": [], "revision": 0, "created_at": _now(),
        }
    attempt = {"started_at": _now(), "status": "initializing"}
    session["initialization"]["attempts"].append(attempt)
    session["initialization"]["status"] = session["status"] = session["phase"] = "initializing"
    session = _write(path, session)
    try:
        normalized_results = deepcopy(results)
        for result in normalized_results:
            result["paths"] = {k: str(p if p.is_absolute() else root / p) for k, v in result["paths"].items() for p in [Path(v)]}
        if target_position is None and (root / "config.json").exists():
            target_position = _read(root / "config.json").get("bench", {}).get("target_position")
        packaged = selection_from_scientist(selection, normalized_results, run_id, root / OUTPUT_FOLDER,
                                            target_position=target_position)
        for key in ("selection_id", "selection_version"):
            if key in selection:
                _need(selection[key] == packaged[key], f"Accepted scientific {key} disagrees with packaged identity")
        session.update(selection_path=str(root / OUTPUT_FOLDER / f"selection_{packaged['selection_id'][:8]}.json"),
                       selection_id=packaged["selection_id"], selection_version=packaged["selection_version"])
        session["selection_sha256"] = _sha(session["selection_path"])
        session["selected_experiments"] = [{"experiment_id": e, "item_id": item["item_id"], "limits": None,
                                            "source": item["source"], "panels": item["panels"]}
                                           for e, item in zip(ids, packaged["items"])]
        session = _write(path, session)
        preview = prepare_specialist_previews(path, root / OUTPUT_FOLDER / "previews" / session["session_id"] /
                                               str(len(session["initialization"]["attempts"])))
        for attachment in preview["previews"]:
            attachment["sha256"] = _sha(attachment["path"])
        _need(preview["reference_mappings"] == defaults and set(preview["normalization_limits"]) == set(ids), "Preview metadata disagrees with frozen references/selection")
        for experiment in session["selected_experiments"]:
            experiment["limits"] = preview["normalization_limits"][experiment["experiment_id"]]
        session["previews"] = preview
        session["initialization"]["status"] = "complete"
        session["initialization"]["attempts"][-1].update(status="complete", finished_at=_now())
        session.update(status="active", phase="awaiting_gamma_proposal")
        return _write(path, session)
    except Exception as error:
        session["initialization"]["status"] = "error"
        session["initialization"]["attempts"][-1].update(status="error", error=str(error), finished_at=_now())
        session.update(status="error", phase="initialization_error")
        _write(path, session)
        raise TuningSessionError(f"Tuning initialization failed; session retained for deliberate retry: {error}") from None


def _active(session: dict) -> dict:
    _need(session["status"] == "active", f"Session is {session['status']}; user action required")
    return session["rounds"][str(session["active_round"])]


def _increment(session: dict, entry: dict, counter: str) -> None:
    _need(entry["retries"][counter] < session["retry_limits"][counter], f"Round {counter} limit reached; user action required")
    entry["retries"][counter] += 1


def _identity(session: dict, result: dict) -> int:
    _need(isinstance(result, dict), "Tool result must be an object")
    _need(type(result.get("contract_version")) is int and result["contract_version"] == 1
          and result.get("session_id") == session["session_id"], "Result contract/session mismatch")
    if "run_id" in result:
        _need(result["run_id"] == session["run_id"], "Result run mismatch")
    number = result.get("round_id")
    _need(type(number) is int and number == session["active_round"], "Result round is stale/out of order")
    return number


def _context(session: dict, stage: Stage, manifest: dict | None = None) -> ValidationContext:
    _, association = _selection(session)
    return ValidationContext(stage, _ids(session), session["config_snapshot"],
                             reference_mappings=session["accepted_settings"] if stage == "propose_gamma" else None,
                             reviewed_manifest=manifest, expected_trial_id=manifest["trial_id"] if manifest else None,
                             expected_calibration_session_id=session["session_id"] if manifest else None,
                             experiment_item_ids=association, expected_run_id=session["run_id"],
                             expected_selection_id=session["selection_id"], expected_selection_version=session["selection_version"])


def build_tuning_context(session_path: str | Path, stage: Stage) -> ValidationContext:
    """Build an existing ValidationContext from the authoritative current session."""
    session = load_tuning_session(session_path)
    entry = _active(session)
    if stage == "propose_gamma":
        _need(session["active_round"] == 1, "Initial proposal stage has passed")
        return _context(session, stage)
    _need(_REVIEW_STAGES[session["active_round"]] == stage, "Review stage differs from active round")
    _, artifact = _bound_trial(session, session["active_trial_id"])
    return _context(session, stage, _read(artifact["result"]["manifest_path"]))


def _trial_manifest(session: dict, result: dict, entry: dict) -> dict:
    """Use existing validation and proposal translation; do not reimplement rendering."""
    manifest = _read(result["manifest_path"])
    round_id = result["round_id"]
    _need(manifest.get("run_id") == session["run_id"] and manifest.get("trial_id") == result["trial_id"], "Trial manifest run/trial mismatch")
    context = _context(session, _REVIEW_STAGES[round_id], manifest)
    correspondence = get_candidate_correspondence(context)
    selection, association = _selection(session)
    _need([item["item_id"] for item in manifest["items"]] == [association[e] for e in _ids(session)], "Manifest selection order differs")
    items = {item["item_id"]: item for item in manifest["items"]}
    for experiment, source in zip(session["selected_experiments"], selection["items"]):
        item = items[source["item_id"]]
        _need(item["limits"] == experiment["limits"], f"Manifest fixed limits changed for {experiment['experiment_id']}")
        _need([(p["name"], p["sha256"]) for p in item["panels"]] == [(p["name"], p["sha256"]) for p in source["panels"]], "Manifest original arrays differ")
    tests = _round_tests(entry["proposal"], round_id, _ids(session))
    expected = _variations(tests, entry["reference_settings"], association, session["config_snapshot"])
    for candidate in correspondence:
        actual = next(v for v in manifest["variations"][candidate.item_id] if v["variation_id"] == candidate.variation_id)
        wanted = next(v for v in expected[candidate.item_id] if v["variation_id"] == candidate.candidate_id)
        _need(actual["mapping"] == wanted["mapping"], f"Manifest mapping differs from proposal/reference for {candidate.experiment_id}/{candidate.candidate_id}")
    _need(Path(result["pdf_path"]).resolve() == Path(manifest["pdf"]["path"]).resolve(), "Trial PDF path differs from manifest")
    _need(result["pdf_sha256"] == manifest["pdf"]["sha256"] == _sha(result["pdf_path"]), "Trial PDF hash mismatch")
    _need(result["page_count"] == manifest["pdf"]["pages"], "Trial page count differs from manifest")
    from pypdf import PdfReader
    _need(len(PdfReader(result["pdf_path"]).pages) == result["page_count"], "Actual PDF page count differs from manifest")
    return manifest


def record_trial_result(session_path: str | Path, result: dict) -> dict:
    """Record a generated sheet; regenerated trials retain earlier artifacts/history."""
    path = Path(session_path)
    session = load_tuning_session(path)
    entry = _active(session)
    _identity(session, result)
    _need(result.get("status") == "ok", "Only successful trial artifacts can become active")
    for old in entry["trials"]:
        if old["result"]["trial_id"] == result.get("trial_id"):
            _need(old["result"] == result, "Conflicting duplicate trial result")
            _need(session["active_trial_id"] == result["trial_id"], "Cannot reactivate a stale trial")
            return session
    _need(not any(t["result"]["trial_id"] == result.get("trial_id") for r in session["rounds"].values() for t in r["trials"]), "Trial IDs must be unique within the session")
    _need(session["phase"] in ("ready_for_trial", "awaiting_scan", "awaiting_review") and entry["proposal"] is not None, "Trial generation is out of order")
    _need(not any(a["result"] is None and not a.get("human_resolution") for a in entry["print_attempts"]), "Submission outcome uncertain; resolve it before regenerating")
    try:
        _trial_manifest(session, result, entry)
    except (KeyError, TypeError, ValueError, _ToolError) as error:
        raise TuningSessionError(f"Invalid trial artifacts: {error}") from None
    if entry["trials"]:
        _increment(session, entry, "trial_regenerations")
    entry["trials"].append({"result": _clone(result), "manifest_sha256": _sha(result["manifest_path"]), "recorded_at": _now()})
    entry["trial_result"] = _clone(result)
    entry["active_evaluation_id"] = None
    session.update(active_trial_id=result["trial_id"], phase="awaiting_scan")
    return _write(path, session)


def _bound_trial_identity(session: dict, trial_id: str) -> tuple[dict, dict]:
    """Bind even an operational failure to a recorded trial, without reading its files."""
    entry = _active(session)
    _need(session["active_trial_id"] == trial_id and entry["trial_result"] is not None
          and entry["trial_result"]["trial_id"] == trial_id, "Stale or missing active trial")
    matches = [t for t in entry["trials"] if t["result"]["trial_id"] == trial_id]
    _need(len(matches) == 1, "Missing/ambiguous recorded trial artifact")
    return entry, matches[0]


def _bound_trial(session: dict, trial_id: str) -> tuple[dict, dict]:
    entry, artifact = _bound_trial_identity(session, trial_id)
    _need(_sha(artifact["result"]["manifest_path"]) == artifact["manifest_sha256"], "Recorded manifest changed")
    return entry, artifact


def record_scan_result(session_path: str | Path, result: dict, *, record_id: str | None = None) -> dict:
    """Bind a scan result to the active trial; no round is advanced.

    Full numerical evaluation lives at evaluation_path; only status/path/hash and
    small metadata are stored here. Explicit record_id distinguishes separate
    attempts producing identical failure messages. Exact duplicates are no-ops.
    """
    path = Path(session_path)
    session = load_tuning_session(path)
    _identity(session, result)
    # An error such as manifest_missing must itself be recordable. It is never
    # accepted as review evidence; successes/rescans still verify the manifest.
    bind = _bound_trial_identity if result.get("status") == "error" else _bound_trial
    entry, artifact = bind(session, result.get("trial_id"))
    _need(session["phase"] in ("awaiting_scan", "awaiting_review"), "Scan result out of order")
    key = record_id or result.get("evaluation_path") or _digest(result)
    _need(isinstance(key, str) and bool(key.strip()), "Evaluation record ID required")
    digest = _digest(result)
    for old in entry["evaluations"]:
        if old["record_id"] == key:
            _need(old["result_sha256"] == digest, "Conflicting duplicate evaluation")
            return session
    status = result.get("status")
    _need(status in ("ok", "needs_rescan", "error"), "Unknown scan status")
    file_hash = None
    if status == "ok":
        _need(isinstance(result.get("evaluation_path"), str) and isinstance(result.get("scan_quality"), dict), "Scan success needs evaluation_path and scan_quality")
        evaluation = _read(result["evaluation_path"])
        _need(all(evaluation.get(k) == result[k] for k in ("session_id", "trial_id", "round_id")), "Evaluation file identity differs")
        _need(evaluation.get("scan_quality") == result.get("scan_quality") and isinstance(result.get("scan_quality", {}).get("usable"), bool), "Evaluation quality differs/missing")
        if "manifest_path" in evaluation:
            _need(Path(evaluation["manifest_path"]).resolve() == Path(artifact["result"]["manifest_path"]).resolve(), "Evaluation refers to a different manifest")
        _need(not any(e["result"].get("evaluation_path") == result["evaluation_path"] for e in entry["evaluations"]), "Evaluation artifact already recorded under another ID")
        file_hash = _sha(result["evaluation_path"])
    elif status == "needs_rescan":
        _need(isinstance(result.get("scan_quality"), dict) and result["scan_quality"].get("usable") is False and bool(result["scan_quality"].get("issues")), "Rescan result must explain unusable evidence")
    else:
        _need(bool(result.get("error_code")) and bool(result.get("message")), "Operational scan error needs code/message")
    if entry["evaluations"]:
        _increment(session, entry, "rescans")
    summary = {k: v for k, v in result.items() if k not in ("pages", "candidates", "printer_response", "measurement_docs")}
    entry["evaluations"].append({"record_id": key, "result": _clone(summary), "result_sha256": digest,
                                  "evaluation_sha256": file_hash, "recorded_at": _now()})
    entry["active_evaluation_id"] = key
    session["phase"] = "awaiting_review" if status == "ok" and result["scan_quality"]["usable"] else "awaiting_scan"
    return _write(path, session)


def _load_active_evaluation(session: dict) -> tuple[str, dict, dict | None]:
    entry, artifact = _bound_trial(session, session["active_trial_id"])
    key = entry["active_evaluation_id"]
    matches = [e for e in entry["evaluations"] if e["record_id"] == key]
    _need(isinstance(key, str) and len(matches) == 1, "No unique active evaluation is recorded")
    record = matches[0]
    result = record["result"]
    _identity(session, result)
    _need(result.get("trial_id") == session["active_trial_id"], "Active evaluation belongs to a stale trial")
    evaluation = None
    if result.get("status") == "ok":
        _need(bool(record["evaluation_sha256"]), "Active evaluation has no recorded artifact hash")
    if record["evaluation_sha256"]:
        _need(_sha(result["evaluation_path"]) == record["evaluation_sha256"], "Recorded evaluation changed")
        evaluation = _read(result["evaluation_path"])
        _identity(session, evaluation)
        _need(evaluation.get("trial_id") == session["active_trial_id"], "Evaluation file belongs to a stale trial")
        _need(evaluation.get("scan_quality") == result.get("scan_quality"), "Evaluation quality differs from recorded result")
        if "manifest_path" in evaluation:
            _need(Path(evaluation["manifest_path"]).resolve() == Path(artifact["result"]["manifest_path"]).resolve(),
                  "Evaluation refers to a different manifest")
    return key, deepcopy(result), evaluation


def record_scan_for_executor(session_path: str | Path, result: dict, *, record_id: str | None = None) -> dict:
    """Application handoff: record the FULL read_scan result, return its model summary.

    Future executor code uses this return value in ToolMessage/dialogue, never the
    raw result. The specialist reloads the hash-bound full evaluation separately.
    This plain function executes no tools/models and activates no graph routing.
    Do not pass summarize_scan_result output as input: needs_rescan summaries have
    lost scan_quality, and success summaries lack the full numerical evidence.

    A tool error before a trial was known may lack identities. Return its exact
    documented short error without guessing an association or changing a session.
    Errors with full identity are recorded by the ordinary scan-result recorder.
    """
    _need(isinstance(result, dict) and type(result.get("contract_version")) is int
          and result["contract_version"] == 1, "Expected a full read_scan contract-version-1 result")
    _need(result.get("status") in ("ok", "needs_rescan", "error"), "Unknown read_scan status")
    if result["status"] == "ok":
        _need(isinstance(result.get("candidates"), list) and isinstance(result.get("pages"), list),
              "Pass the full read_scan result, not its executor summary")
        _need(isinstance(result.get("evaluation_path"), str), "Full scan result needs evaluation_path")
        evaluation = _read(result["evaluation_path"])
        for name in ("pages", "candidates", "printer_response", "measurement_docs", "scan_files"):
            if name in result:
                _need(result[name] == evaluation.get(name), f"Full scan {name} differs from saved evaluation")
    summary = summarize_scan_result(result)
    _need(summary.get("error_code") != "summary_failed", f"Cannot prepare executor scan summary: {summary.get('message')}")
    if result["status"] == "error":
        _need(isinstance(result.get("error_code"), str) and bool(result["error_code"].strip())
              and isinstance(result.get("message"), str) and bool(result["message"].strip()),
              "Scan error requires error_code and message")
        if not all(k in result for k in ("session_id", "trial_id", "round_id")):
            return _clone(summary)
    record_scan_result(session_path, result, record_id=record_id)
    return _clone(summary)


def load_tuning_evaluation(session_path: str | Path) -> tuple[str, dict, dict | None]:
    """Read the active evaluation after trial/hash/identity checks; never write.

    Returns (record_id, recorded tool-result summary, loaded scan JSON). A recorded
    needs_rescan/error may have no JSON artifact, so the third element is None.
    Callers must serialize session access, as with the other session functions.
    """
    return _load_active_evaluation(load_tuning_session(session_path))


def accept_specialist_response(
    session_path: str | Path, response: Response, context: ValidationContext, *, evaluation_record_id: str | None = None,
) -> dict:
    """Fully validate before resolving winners or performing one atomic write.

    Reviews require the explicit recorded evaluation ID (not part of the model
    schema). Exact repeat decisions for the same stage/trial/evaluation are no-ops;
    conflicting repeats fail. Nothing from an invalid decision is accepted.
    """
    path = Path(session_path)
    session = load_tuning_session(path)
    stage = context.expected_stage
    number = {"propose_gamma": 1, "review_gamma": 1, "review_gray_levels": 2, "review_steps": 3}.get(stage)
    _need(number is not None and response.stage == stage, "Decision stage/model mismatch")
    entry = session["rounds"][str(number)]
    _need(list(context.selected_experiment_ids) == _ids(session) and context.config == session["config_snapshot"], "Decision selection/order/config differs from session")
    _need(context.expected_calibration_session_id in (None, session["session_id"]), "Context calibration session differs")
    if stage == "propose_gamma":
        expected = _context(session, stage)
        _need(context.reference_mappings == expected.reference_mappings or entry["proposal"] == response.model_dump(), "Initial references differ from accepted settings")
        trial_id = None
        _need(context.expected_trial_id is None and context.reviewed_manifest is None, "Initial proposal must not review a physical trial")
        _need(evaluation_record_id is None, "Initial proposal has no evaluation")
    else:
        _need(entry["trial_result"] is not None, "Review has no recorded trial")
        result = entry["trial_result"]
        trial_id = result["trial_id"]
        artifact = next(t for t in entry["trials"] if t["result"]["trial_id"] == trial_id)
        _need(_sha(result["manifest_path"]) == artifact["manifest_sha256"], "Reviewed manifest changed")
        manifest = _read(result["manifest_path"])
        expected = _context(session, stage, manifest)
        _need(context.reviewed_manifest == manifest and context.expected_trial_id == trial_id
              and context.expected_calibration_session_id == session["session_id"], "Context is not bound to this session's reviewed trial")
        _need(isinstance(evaluation_record_id, str) and evaluation_record_id == entry["active_evaluation_id"], "Review needs the active evaluation record ID")
    _need(context.experiment_item_ids == expected.experiment_item_ids, "Context experiment/item association differs")
    for field in ("expected_run_id", "expected_selection_id", "expected_selection_version"):
        _need(getattr(context, field) in (None, getattr(expected, field)), f"Context {field} differs")
    try:
        validate_response(response, context)
    except TuningValidationError as error:
        raise TuningSessionError(str(error)) from None
    key = (stage, trial_id, evaluation_record_id)
    data = response.model_dump()
    for old in session["decisions"]:
        if (old["stage"], old["trial_id"], old["evaluation_record_id"]) == key:
            _need(old["response"] == data, "Conflicting decision for an already accepted stage/evaluation")
            return session
    _active(session)
    _need(session["active_round"] == number, "Decision round is out of order")
    if stage == "propose_gamma":
        _need(session["phase"] == "awaiting_gamma_proposal", "Gamma proposal out of order")
        entry["proposal"] = data
        entry["reference_settings"] = deepcopy(session["accepted_settings"])
        session["phase"] = "ready_for_trial"
    else:
        _need(session["active_trial_id"] == trial_id, "Stale reviewed trial")
        _need(session["phase"] in ("awaiting_review", "awaiting_scan"), "Review out of order")
        _, recorded_result, _ = _load_active_evaluation(session)
        if response.assessment != "needs_rescan":
            _need(recorded_result["status"] == "ok" and recorded_result["scan_quality"]["usable"], "Usable evidence required before accepting winners/contrast judgment")
        # Full response validation above includes next proposals against resolved
        # winners. Resolve only now, never accepting a valid winner alone.
        winners = resolve_review_winners(response, context)
        entry["review"] = data
        entry["reviews"].append({"response": data, "trial_id": trial_id, "evaluation_record_id": evaluation_record_id})
        if response.assessment == "needs_rescan":
            session["phase"] = "awaiting_scan"
        elif response.assessment == "insufficient_contrast":
            session.update(status="blocked", phase="user_action_required")
        else:
            entry["winners"] = deepcopy(winners)
            session["accepted_settings"] = winners
            if number == 3:
                session.update(final_settings=deepcopy(winners), status="completed", phase="completed")
            else:
                next_round = session["rounds"][str(number + 1)]
                next_round["proposal"] = data
                next_round["reference_settings"] = deepcopy(winners)
                session.update(active_round=number + 1, active_trial_id=None, phase="ready_for_trial")
    session["decisions"].append({"stage": stage, "trial_id": trial_id, "evaluation_record_id": evaluation_record_id,
                                 "response": data, "accepted_at": _now()})
    return _write(path, session)


def record_print_approval(
    session_path: str | Path, *, approval_id: str, trial_id: str, pdf_sha256: str,
    printer: str, print_settings: dict, approved_by: str, reprint: bool = False,
) -> dict:
    """Record explicit approval of this exact sheet/settings; allow ONE approved reprint per round."""
    path = Path(session_path)
    session = load_tuning_session(path)
    entry, artifact = _bound_trial(session, trial_id)
    _need(all(isinstance(v, str) and v.strip() for v in (approval_id, printer, approved_by)), "Approval ID/printer/approver required")
    _need(pdf_sha256 == artifact["result"]["pdf_sha256"] == _sha(artifact["result"]["pdf_path"]), "Approval PDF hash mismatch")
    _need(isinstance(print_settings, dict) and bool(print_settings), "Effective print settings required")
    _need(type(reprint) is bool and print_settings.get("printer", printer) == printer, "Reprint flag/printer settings mismatch")
    data = _clone(dict(approval_id=approval_id, trial_id=trial_id, pdf_sha256=pdf_sha256,
                       printer=printer, print_settings=print_settings, approved_by=approved_by, reprint=reprint))
    for old in entry["approvals"]:
        if old["metadata"]["approval_id"] == approval_id:
            _need(old["metadata"] == data, "Conflicting duplicate approval")
            return session
    _need(not any(a["metadata"]["approval_id"] == approval_id for r in session["rounds"].values() for a in r["approvals"]), "Approval IDs must be unique within the session")
    _need(session["phase"] == "awaiting_scan", "Approval is out of order")
    if reprint:
        _need(bool(entry["print_attempts"]) and all(a["result"] is not None or a.get("human_resolution") for a in entry["print_attempts"]), "Prior submission outcome must be recorded before approving a reprint")
        _increment(session, entry, "reprints")
    else:
        _need(not any(a["metadata"]["trial_id"] == trial_id for a in entry["approvals"])
              and not entry["print_attempts"], "Another print requires explicit reprint approval")
    entry["approvals"].append({"metadata": data, "recorded_at": _now()})
    return _write(path, session)


def begin_print_attempt(session_path: str | Path, *, approval_id: str, attempt_id: str) -> dict:
    """Persist BEFORE external submission. A missing result means outcome uncertain.

    The caller later invokes the approved printer interface; this function never
    submits or infers a printer/job status. Do not blindly retry uncertain work.
    """
    path = Path(session_path)
    session = load_tuning_session(path)
    entry = _active(session)
    _bound_trial(session, session["active_trial_id"])
    for old in entry["print_attempts"]:
        if old["attempt_id"] == attempt_id:
            _need(old["approval_id"] == approval_id, "Conflicting duplicate print attempt")
            return session
    _need(not any(a["attempt_id"] == attempt_id for r in session["rounds"].values() for a in r["print_attempts"]),
          "Print attempt IDs must be unique within the session")
    _need(isinstance(attempt_id, str) and bool(attempt_id.strip()), "Print attempt ID required")
    approvals = [a for a in entry["approvals"] if a["metadata"]["approval_id"] == approval_id]
    _need(len(approvals) == 1 and approvals[0]["metadata"]["trial_id"] == session["active_trial_id"], "Current-trial print approval required")
    _need(not any(a["approval_id"] == approval_id or (a["result"] is None and not a.get("human_resolution")) for a in entry["print_attempts"]), "Approval already consumed or submission outcome uncertain")
    _need(session["phase"] == "awaiting_scan", "Print submission is out of order")
    approval = approvals[0]["metadata"]
    _need(_sha(entry["trial_result"]["pdf_path"]) == approval["pdf_sha256"], "Approved PDF changed before submission")
    entry["print_attempts"].append({"attempt_id": attempt_id, "approval_id": approval_id, "trial_id": session["active_trial_id"],
                                    "submission_state": "outcome_uncertain", "result": None, "job_results": [], "recorded_at": _now()})
    entry["reprint_requested"] = False
    return _write(path, session)


def record_print_result(session_path: str | Path, *, attempt_id: str, result: dict, job_result: bool = False) -> dict:
    """Preserve supplied printer results verbatim; do not interpret success/job semantics."""
    path = Path(session_path)
    session = load_tuning_session(path)
    _need(isinstance(result, dict), "Printer result metadata must be an object")
    attempts = [a for r in session["rounds"].values() for a in r["print_attempts"] if a["attempt_id"] == attempt_id]
    _need(len(attempts) == 1, "Unknown/ambiguous print attempt ID")
    attempt = attempts[0]
    data = _clone(result)
    if job_result:
        _need(attempt["result"] is not None, "Record submission outcome before job readings")
        if data in attempt["job_results"]:
            return session
        attempt["job_results"].append(data)
    else:
        if attempt["result"] is not None:
            _need(attempt["result"] == data, "Conflicting duplicate submission outcome")
            return session
        attempt.update(result=data, submission_state="result_recorded")
    return _write(path, session)


def resolve_print_outcome(session_path: str | Path, *, attempt_id: str, outcome: str) -> dict:
    """Record a human tray check, separately from the verbatim printer results.

    Also resolves a crash between submission intent and recording its outcome.
    Callers serialize access; a conflicting second resolution is rejected.
    """
    path = Path(session_path)
    session = load_tuning_session(path)
    entry = _active(session)
    _need(session["phase"] == "awaiting_scan", "Print resolution is out of order")
    matches = [a for a in entry["print_attempts"] if a["attempt_id"] == attempt_id
               and a["trial_id"] == session["active_trial_id"]]
    _need(len(matches) == 1 and matches[0] is entry["print_attempts"][-1], "Resolve only the latest active-trial attempt")
    _need(outcome in ("printed", "not_printed"), "Human outcome must be printed or not_printed")
    attempt = matches[0]
    if attempt.get("human_resolution"):
        _need(attempt["human_resolution"]["outcome"] == outcome, "Conflicting human print resolution")
        return session
    attempt["human_resolution"] = {"outcome": outcome, "recorded_at": _now()}
    return _write(path, session)


def request_trial_reprint(session_path: str | Path) -> dict:
    """Record human recovery intent; fresh approval consumes the one-reprint limit.

    This does not print, approve, or regenerate a sheet. Scans remain historical
    artifacts, but their active pointer is cleared before replacement scanning.
    """
    path = Path(session_path)
    session = load_tuning_session(path)
    entry = _active(session)
    _need(session["phase"] in ("awaiting_scan", "awaiting_review"), "Reprint is out of order")
    _need(entry["retries"]["reprints"] < session["retry_limits"]["reprints"], "One reprint per round is permitted")
    _need(bool(entry["print_attempts"]), "There is no previous submission to reprint")
    latest = entry["print_attempts"][-1]
    reading = (latest["job_results"] or [latest["result"]])[-1] or {}
    resolved = latest.get("human_resolution")
    _need(bool(resolved) or reading.get("confirmed") is True and reading.get("state") in ("done", "failed")
          or latest["result"] is not None and latest["result"].get("ok") is False,
          "Resolve the previous submission before requesting a reprint")
    if entry.get("reprint_requested"):
        return session
    entry.update(reprint_requested=True, active_evaluation_id=None)
    session["phase"] = "awaiting_scan"
    return _write(path, session)


def record_stop(session_path: str | Path, reason: str, *, error: bool = False, metadata: dict | None = None) -> dict:
    """Record stopped/error status only; never cancel a physical job or erase progress."""
    path = Path(session_path)
    session = load_tuning_session(path)
    _need(isinstance(reason, str) and bool(reason.strip()), "Stop/error reason required")
    _need(session["status"] != "completed", "Completed calibration cannot be reset by a stop/error")
    outcome = _clone({"status": "error" if error else "stopped", "reason": reason, "metadata": metadata})
    if session["outcomes"] and session["outcomes"][-1]["outcome"] == outcome:
        return session
    session["outcomes"].append({"outcome": outcome, "recorded_at": _now()})
    session.update(status=outcome["status"], phase=outcome["status"])
    return _write(path, session)
