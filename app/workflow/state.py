"""Existing graph state and additive message/evidence reducers."""

from operator import add
from typing import Annotated, TypedDict


class State(TypedDict, total=False):
    dialogue: Annotated[list[dict], add]
    execution_results: Annotated[list[dict], add]
    decision: dict
    decision_count: int
    decision_budget: int
    stop_reason: str
    route: str
    pending_print: dict
    approval: dict
    report: dict
    print_outcome: str
    active_job_id: str | None
    hardware_passes: int
    report_request_passes: int
    report_workflow_failure: str
    report_revision_pending: bool
    report_revision: dict | None
    budget_summary: str
    last_printer_reading: dict
    scientist_response: dict
    experiment_selection: dict
    # Authoritative progress is in this JSON; this projection is initialization only.
    tuning_session_path: str
    tuning_initialization: dict
    # Routing/pending inputs only; progress and print history stay in the session.
    tuning_action: dict
    tuning_printer: str
    tuning_failure: str
    tuning_failures: int
    tuning_polls: int
    # Print tuning (optional; nodes that do not tune never set them).
    tuning_round: int | None
    active_trial_id: str | None
    pending_trial_print: dict | None
    trial_approval: dict | None
    last_scan_result: dict | None
    accepted_settings: dict | None
    final_settings: dict | None
    final_figure_path: str | None
    final_figure_metadata: dict
    # Paper the user prints on (None: plain white). Kept on resume, so it is asked only once.
    paper_color: str | None
